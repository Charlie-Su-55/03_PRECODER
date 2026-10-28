"""F evaluator/report contracts using only synthetic metadata and scalar CSV data."""
from __future__ import annotations

import ast
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tests"):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
import evaluate_grouped_codec_fdma as entry
from cli_contracts import assert_lightweight_cli
from test_evaluate_grouped_codec import fixtures as old_fixture


def synthetic_samples(count=16, dataset="new_replication", training_seed=20261030):
    _, samples = old_fixture(count, training_seed)
    for row in samples: row.update(dataset=dataset, seed=entry.DATASETS[dataset])
    for row in list(samples):
        if row["method_key"] != "legacy_group2": continue
        f = deepcopy(row); f.update(method_key="legacy_fdma", active_layout="fdma",
            active_symbols_per_ue=64, active_users_per_RE=1, receiver_regularization=row["noise_variance_N0"]+1e-6)
        scale={0:2.,5:-1.,10:5.,15:0.,20:3.,25:-2.}[row["gamma_db"]]
        f["sum_rate"] -= .75 + scale * (row["sample_index"]-(count-1)/2)/count
        samples.append(f)
    return samples


def paired_fixture(count=16,dataset="new_replication",training_seed=20261030):
    samples=synthetic_samples(count,dataset,training_seed)
    summary,_=entry.summarize_samples(samples,count,entry.DATASETS[dataset],training_seed,dataset)
    paired,raw=entry.paired_rows(summary,samples,count,entry.DATASETS[dataset],training_seed,dataset)
    return summary,samples,paired,raw


class ProtocolTests(unittest.TestCase):
    def test_explicit_two_sets_six_methods_and_onlyF_old_completion_forward(self):
        for seed in entry.SEEDS:
            for dataset in entry.DATASETS:
                p=entry.protocol(seed,dataset); plan=entry.checkpoint_plan(seed,dataset)
                self.assertEqual(p["seed"],entry.DATASETS[dataset]);self.assertEqual(p["num_samples"],1600)
                self.assertEqual(p["batch_size"],8);self.assertEqual(len(plan),6)
                active=[r["method_key"] for r in plan if r["forward_required"]]
                self.assertEqual(active,["legacy_fdma"] if dataset=="old_completion" else list(entry.METHODS))
        self.assertEqual(entry.protocol(20261030,"new_replication")["channel_data_protocol"],
                         entry.protocol(20261031,"new_replication")["channel_data_protocol"])
        self.assertNotEqual(entry.protocol(20261030,"old_completion")["channel_data_protocol"],
                            entry.protocol(20261030,"new_replication")["channel_data_protocol"])

    def test_stdlib_dryruns_no_weights_models_outputs_or_old_evidence_claim(self):
        assert_lightweight_cli(self,"evaluate_grouped_codec_fdma",
            [["--seed",str(seed),"--dataset",dataset,"--dry-run"] for seed in entry.SEEDS for dataset in entry.DATASETS],
            "assert all(p['models_not_imported'] and p['checkpoint_bytes_not_read'] and p['outputs_not_created'] for p in plans)\n"
            "assert all(not p['old_batch_verified'] for p in plans)\n"
            "assert all('batch_b314d7520c844f259ffa5db18ab0b149' in p['fixed_old_batch'] for p in plans)")

    def test_strict_loading_and_no_weight_migration(self):
        text=(ROOT/"evaluate_grouped_codec_fdma.py").read_text(encoding="utf-8")
        tree=ast.parse(text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=="load_state_dict"]
        self.assertEqual(len(calls),1)
        self.assertTrue(any(k.arg=="strict" and isinstance(k.value,ast.Constant) and k.value.value is True for k in calls[0].keywords))
        self.assertNotIn("strict=False",text);self.assertNotIn("migrate_",text)
        loads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="load_payload"]
        self.assertTrue(all("expected" in {k.arg for k in n.keywords} for n in loads))


class EvidenceTests(unittest.TestCase):
    def test_fixed_old_batch_missing_fails_no_alternate_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(FileNotFoundError):entry.verify_old_batch(root=root)
            alternative=root/"runs_as_screen/grouped_codec_v1/last_batch.json"
            alternative.parent.mkdir(parents=True)
            alternative.write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):entry.verify_old_batch(alternative,root=root)
            self.assertEqual(alternative.read_text(encoding="utf-8"),"{}")

    def test_all_old_stage_hashes_checked_and_no_missing_status_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); file=root/"proof.json";file.write_text("{}",encoding="utf-8")
            digest=hashlib.sha256(file.read_bytes()).hexdigest()
            names={"environment","new_cuda_preflight","organize_results"}
            names|={f"train_{arm}_{seed}" for seed in entry.SEEDS for arm in entry.old.METHODS}
            names|={f"evaluate_{seed}" for seed in entry.SEEDS}
            stages={name:dict(status="completed",evidence_sha256={str(file):digest},result={}) for name in names}
            entry.verify_stage_evidence(stages,entry._Verifier(root))
            bad=deepcopy(stages);bad.pop("organize_results")
            with self.assertRaises(ValueError):entry.verify_stage_evidence(bad,entry._Verifier(root))
            bad=deepcopy(stages);bad["environment"]["status"]="failed"
            with self.assertRaises(ValueError):entry.verify_stage_evidence(bad,entry._Verifier(root))
            file.write_text('{"changed":true}',encoding="utf-8")
            with self.assertRaises(ValueError):entry.verify_stage_evidence(stages,entry._Verifier(root))

    def test_old_completion_requires_exact_actual_allthree_hashes(self):
        row=dict(batch_index=0,sample_index_start=0,sample_index_stop=8,
                 H_dl_sha256="a"*64,H_ul_sha256="b"*64,standard_noise_sha256="c"*64)
        manifest=dict(shared_batch_inputs=[row])
        entry.match_old_batch_inputs(deepcopy(row),manifest)
        for field in row:
            other=deepcopy(row);other[field]="changed"
            with self.subTest(field=field),self.assertRaises(ValueError):entry.match_old_batch_inputs(other,manifest)

    def test_raw_csv_readonly_roundtrip_preserves_complete_key_and_numbers(self):
        samples=synthetic_samples(8,"old_completion")[:8]
        with tempfile.TemporaryDirectory() as temporary:
            file=Path(temporary)/"raw.csv.gz"
            with gzip.open(file,"wt",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
            before=file.read_bytes()
            parsed=entry.typed_rows(entry.csv_rows(file),raw=True)
            self.assertEqual(parsed,samples);self.assertEqual(before,file.read_bytes())


class MetadataTests(unittest.TestCase):
    def fixture(self):
        from test_train_grouped_codec import resume_metadata
        from train_grouped_codec_fdma import fingerprint
        cfg=entry.fdma_config(20261030);latest=resume_metadata(cfg,20000)
        best={name:deepcopy(latest[name]) for name in ("state","config","config_fingerprint","layout_spec","execution_protocol","step","best_metric","val_table")}
        best["step"]=19500
        sidecars=dict(config=dict(cfg.to_dict(),config_fingerprint=fingerprint(cfg)),
            status=dict(state="completed",step=20000,experiment=cfg.EXP_NAME,best_metric_name="mean_all",best_metric=best["best_metric"]))
        return best,latest,sidecars

    def test_F_complete20k_best_earlier_is_valid(self):
        record=entry.verify_training_metadata(20261030,*self.fixture())
        self.assertEqual((record["selected_best_step"],record["latest_step"]),(19500,20000))

    def test_F_incomplete_cross_arm_zero_best_wrong_fingerprint_rejected(self):
        for defect in ("zero","incomplete","arm","fingerprint","status","selection"):
            best,latest,sidecars=self.fixture()
            if defect=="zero":best["step"]=0
            elif defect=="incomplete":latest["step"]=500
            elif defect=="arm":best["config"]["arm"]="ordered_fdma"
            elif defect=="fingerprint":latest["config_fingerprint"]="wrong"
            elif defect=="status":sidecars["status"]["state"]="paused"
            else:best["val_table"]["layout0_gamma0"]+=1
            with self.subTest(defect=defect),self.assertRaises((ValueError,RuntimeError)):
                entry.verify_training_metadata(20261030,best,latest,sidecars)


class StatisticsTests(unittest.TestCase):
    def test36conditions96rows_all12directions_and_primary_dataset(self):
        self.assertEqual(entry.CONTRASTS["C_minus_F"],{"legacy_group2":1,"legacy_fdma":-1})
        self.assertEqual(entry.CONTRASTS["F_minus_E"],{"legacy_fdma":1,"ordered_fdma":-1})
        self.assertEqual(entry.CONTRASTS["interaction_C_minus_F_minus_D_plus_E"],
            {"legacy_group2":1,"legacy_fdma":-1,"ordered_group2":-1,"ordered_fdma":1})
        for dataset in entry.DATASETS:
            summary,samples,paired,raw=paired_fixture(dataset=dataset)
            self.assertEqual((len(summary),len(paired),len(raw)),(36,96,96*16))
            low=next(r for r in paired if r["contrast_id"]=="C_minus_F" and r["estimand"]=="low_gamma_mean")
            self.assertAlmostEqual(low["delta_mean"],.75)
            self.assertEqual(low["role"],"primary_confirmatory" if dataset=="new_replication" else "paired_completion")
            negative=next(r for r in paired if r["contrast_id"]=="F_minus_E" and r["estimand"]=="low_gamma_mean")
            self.assertLess(negative["delta_mean"],0)

    def test_old_seven_estimates_exactly_preserved_from_old_raw(self):
        summary,samples,paired,_=paired_fixture(dataset="old_completion")
        originals=[r for r in samples if r["method_key"] in entry.old.METHODS]
        old_summary,_=entry.old.summarize_samples(originals,16,20261210,20261030)
        old_pairs,_=entry.old.paired_rows(old_summary,originals,16,20261210,20261030)
        lookup={(r["contrast_id"],r["estimand"],r["gamma_points"]):r for r in paired}
        for old_row in old_pairs:
            current=lookup[old_row["contrast_id"],old_row["estimand"],old_row["gamma_points"]]
            for key in ("delta_mean","delta_se","delta_ci95_low","delta_ci95_high",
                        "batch_delta_mean","batch_delta_se","batch_delta_ci95_low","batch_delta_ci95_high"):
                self.assertEqual(current[key],old_row[key])

    def test_interaction_and_crossgamma_from_within_channel_differences_notSE_arithmetic(self):
        _,_,paired,raw=paired_fixture()
        low=next(r for r in paired if r["contrast_id"]=="C_minus_F" and r["estimand"]=="low_gamma_mean")
        values=[.75+2*(i-7.5)/16 for i in range(16)]
        for key,value in entry.sample_statistics(values).items():self.assertAlmostEqual(low["delta_"+key],value)
        batches=[statistics.fmean(values[:8]),statistics.fmean(values[8:])]
        for key,value in entry.sample_statistics(batches).items():self.assertAlmostEqual(low["batch_delta_"+key],value)
        name="interaction_C_minus_F_minus_D_plus_E"
        interaction=next(r for r in paired if r["contrast_id"]==name and r["estimand"]=="low_gamma_mean")
        # C-F = .75+2x; D-E=-1+4x, hence interaction=1.75-2x.
        expected=[1.75-2*(i-7.5)/16 for i in range(16)]
        self.assertAlmostEqual(interaction["delta_se"],statistics.stdev(expected)/4)
        self.assertAlmostEqual(interaction["delta_mean"],1.75)

    def test_onebatch_interval_unavailable_and_sets_seeds_not_pooled(self):
        for dataset in entry.DATASETS:
            for seed in entry.SEEDS:
                _,_,paired,_=paired_fixture(8,dataset,seed)
                self.assertTrue(all(row["batch_delta_se"] is None for row in paired))
                self.assertTrue(all(row["dataset"]==dataset and row["training_seed"]==seed and not row["datasets_pooled"] and not row["training_seeds_pooled"] for row in paired))

    def test_missing_duplicate_mixedset_nonfinite_wrongphysics_refused(self):
        for defect in ("missing","duplicate","dataset","seed","trainseed","nan","noise","energy","N0","power"):
            samples=synthetic_samples()
            if defect=="missing":samples.pop()
            elif defect=="duplicate":samples.append(deepcopy(samples[-1]))
            elif defect=="dataset":samples[-1]["dataset"]="old_completion"
            elif defect=="seed":samples[-1]["seed"]=20261210
            elif defect=="trainseed":samples[-1]["training_seed"]=20261031
            elif defect=="nan":samples[-1]["sum_rate"]=math.nan
            elif defect=="noise":samples[-1]["empirical_noise_power"]*=2
            elif defect=="energy":samples[-1]["energy_ue2"]=63
            elif defect=="N0":samples[-1]["noise_variance_N0"]*=2
            else:samples[-1]["precoder_power"]=1
            with self.subTest(defect=defect),self.assertRaises((ValueError,RuntimeError)):
                entry.summarize_samples(samples,16,20261220,20261030,"new_replication")

    def test_summary_mismatch_and_permutation_pairing(self):
        summary,samples,expected,raw=paired_fixture()
        actual,reordered=entry.paired_rows(list(reversed(summary)),list(reversed(samples)),16,20261220,20261030,"new_replication")
        self.assertEqual(expected,actual);self.assertEqual(raw,reordered)
        summary[0]["sum_rate_mean"]+=1
        with self.assertRaises(ValueError):entry.paired_rows(summary,samples,16,20261220,20261030,"new_replication")


def report_fixture(dataset="new_replication", seed=20261030):
    """Synthetic table-schema fixture, never formal evaluation or sampled data."""
    summary, _, paired, _ = paired_fixture(dataset=dataset, training_seed=seed)
    for row in summary + paired: row.update(num_samples=1600, num_batches=200, batch_size=8, preflight=False)
    return summary, paired


def write_csv_fixture(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def training_fixture(folder):
    """Twelve temporary synthetic logs; no checkpoint bytes created or read."""
    records = {}
    for seed in entry.SEEDS:
        records[str(seed)] = {}
        for index, method in enumerate(entry.METHODS):
            path = folder / f"synthetic_{seed}_{method}.jsonl"
            points = [dict(event="train_gamma", step=1, gamma_db=7.5)]
            for step in range(500, 20001, 500):
                mean = 10 + index + step * .0002
                points.append(dict(event="validation", step=step,
                    selection_metric_name="mean_all", selection_metric=mean,
                    gamma_db=list(entry.GAMMAS), downlink_snr_db=25,
                    table={f"layout0_gamma{gamma}": mean + (gamma-12.5)*.1 for gamma in entry.GAMMAS}))
            path.write_text("\n".join(json.dumps(p) for p in points)+"\n", encoding="utf-8")
            records[str(seed)][method] = dict(completed_updates=20000, best_step=20000,
                metrics_path=str(path), metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                best_sha256=hashlib.sha256(f"best/{seed}/{method}".encode()).hexdigest(),
                latest_sha256=hashlib.sha256(f"latest/{seed}/{method}".encode()).hexdigest())
    return records


def input_hash_fixture(dataset):
    return [dict(batch_index=index, sample_index_start=index*8, sample_index_stop=(index+1)*8,
        **{name: hashlib.sha256(f"synthetic/{dataset}/{index}/{name}".encode()).hexdigest()
           for name in ("H_dl_sha256", "H_ul_sha256", "standard_noise_sha256")}) for index in range(200)]


def bundle_fixture(folder):
    """Small synthetic audit artifacts; hashes do not represent real channels."""
    records = training_fixture(folder)
    old_summary_path = folder / "synthetic_old_batch_summary.json"
    old_summary_path.write_text("{}", encoding="utf-8")
    evidence = dict(summary_path=str(old_summary_path),
        summary_sha256=hashlib.sha256(old_summary_path.read_bytes()).hexdigest(),
        evaluations={}, training={str(seed):{key:deepcopy(records[str(seed)][key])
                    for key in entry.old.METHODS} for seed in entry.SEEDS}, frozen_files={})
    evaluations = {}
    for dataset in entry.DATASETS:
        evaluations[dataset] = {}
        for seed in entry.SEEDS:
            summary, paired = report_fixture(dataset, seed)
            directory = folder / f"synthetic_{dataset}_{seed}"; directory.mkdir()
            evaluations[dataset][str(seed)] = str(directory)
            write_csv_fixture(directory/f"summary_{entry.TAG}.csv", summary)
            write_csv_fixture(directory/f"paired_deltas_{entry.TAG}.csv", paired)
            parameters=[dict(method_key=key, training_seed=seed, full_parameters=123,
                ue_parameters=23, bs_parameters=100, dataset=dataset) for key in entry.METHODS]
            write_csv_fixture(directory/f"parameters_{entry.TAG}.csv",parameters)
            for filename in entry.OUTPUT_ARTIFACTS:
                path = directory/filename
                if path.exists():continue
                if filename.endswith(".gz"):
                    with gzip.open(path,"wt",encoding="utf-8") as stream:stream.write("synthetic_fixture_only\n")
                else:path.write_text("{}",encoding="utf-8")
            loaded = ["legacy_fdma"] if dataset=="old_completion" else list(entry.METHODS)
            manifest = dict(status="completed", results_verified=True,
                freeze_checks=dict(passed=True, loaded_methods=loaded),
                shared_channel_noise_inputs_unchanged=True, samples_hashed_pairing=True,
                protocol=entry.protocol(seed,dataset), output_sha256=entry.artifact_hashes(directory),
                shared_batch_inputs=input_hash_fixture(dataset), frozen_old_rows_preserved=dataset=="old_completion",
                methods={key:dict(loaded_in_this_run=key in loaded,
                    selected_best_step=records[str(seed)][key]["best_step"],
                    best_sha256=records[str(seed)][key]["best_sha256"],
                    latest_sha256=records[str(seed)][key]["latest_sha256"]) for key in entry.METHODS})
            (directory/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            if dataset=="old_completion":
                old_directory=folder/f"synthetic_old_{seed}"; old_directory.mkdir()
                old_summary=old_directory/f"summary_{entry.old.TAG}.csv"
                old_paired=old_directory/f"paired_deltas_{entry.old.TAG}.csv"
                old_parameters=old_directory/f"parameters_{entry.old.TAG}.csv"
                write_csv_fixture(old_summary,[r for r in summary if r["method_key"] in entry.old.METHODS])
                write_csv_fixture(old_paired,[r for r in paired if r["contrast_id"] in entry.old.CONTRASTS])
                write_csv_fixture(old_parameters,[r for r in parameters if r["method_key"] in entry.old.METHODS])
                evidence["evaluations"][str(seed)]=dict(directory=str(old_directory),summary_path=str(old_summary),
                    parameters_path=str(old_parameters),manifest=dict(shared_batch_inputs=input_hash_fixture(dataset)))
                for path in (old_summary,old_paired,old_parameters):
                    evidence["frozen_files"][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return evaluations, records, evidence


class ReportTests(unittest.TestCase):
    def test_report_same_signed_contrasts_and_dataset_primary_roles(self):
        import grouped_codec_fdma_results as report
        self.assertEqual(report.CONTRASTS,entry.CONTRASTS);self.assertEqual(report.DATASETS,entry.DATASETS)
        for dataset in entry.DATASETS:
            summary,paired=report_fixture(dataset)
            interpreted=report.interpret(summary,paired,20261030,dataset)
            self.assertEqual(len(interpreted["comparisons"]),12)
            self.assertFalse(interpreted["engineering_criterion_D_minus_E"])
            self.assertLess(interpreted["comparisons"]["F_minus_E"]["low_gamma"]["delta_mean"],0)
            self.assertEqual(interpreted["primary_role"],"primary_confirmatory" if dataset=="new_replication" else "paired_completion")
            self.assertFalse(interpreted["training_seeds_pooled"]);self.assertFalse(interpreted["datasets_pooled"])

    def test_report_rejects_missing_duplicate_wrong_units_signs_CI_and_pooling(self):
        import grouped_codec_fdma_results as report
        for defect in ("missing","duplicate","dataset","seed","n","noise","coefficients","ci","se",
                       "pooled_seed","pooled_set","user_unit","batch_unit","nonfinite","old_snr"):
            summary,paired=report_fixture()
            if defect=="missing":paired.pop()
            elif defect=="duplicate":paired.append(deepcopy(paired[0]))
            elif defect=="dataset":summary[0]["dataset"]="old_completion"
            elif defect=="seed":paired[0]["training_seed"]=20261031
            elif defect=="n":paired[0]["num_samples"]=6400
            elif defect=="noise":summary[0]["noise_variance_N0"]*=2
            elif defect=="coefficients":paired[0]["coefficients"]='{"legacy_all4":1,"ordered_all4":-1}'
            elif defect=="ci":paired[0]["batch_delta_ci95_low"]+=.1
            elif defect=="se":paired[0]["delta_se"]=-1
            elif defect=="pooled_seed":paired[0]["training_seeds_pooled"]=True
            elif defect=="pooled_set":paired[0]["datasets_pooled"]=True
            elif defect=="user_unit":paired[0]["sample_ci_unit"]="individual_user"
            elif defect=="batch_unit":paired[0]["batch_ci_unit"]="sample_index"
            elif defect=="nonfinite":summary[0]["sum_rate_mean"]=math.inf
            else:summary[0]["feedback_snr_db"]=0
            with self.subTest(defect=defect),self.assertRaises(ValueError):report.interpret(summary,paired,20261030,"new_replication")

    def test_engineering_thresholds_use_signed_batch_intervals_and_keep_old_rule(self):
        import grouped_codec_fdma_results as report
        summary,paired=report_fixture()
        low=next(r for r in paired if r["contrast_id"]=="C_minus_F" and r["estimand"]=="low_gamma_mean")
        high=next(r for r in paired if r["contrast_id"]=="C_minus_F" and r["gamma_points"]=="25")
        def stats(row,mean,se):
            for prefix in ("delta_","batch_delta_"):
                row.update({prefix+"mean":mean,prefix+"se":se,prefix+"ci95_low":mean-1.96*se,prefix+"ci95_high":mean+1.96*se})
        stats(low,.5,0);stats(high,-.1,0)
        value=report.interpret(summary,paired,20261030,"new_replication")
        self.assertTrue(value["engineering_criterion_C_minus_F"]);self.assertFalse(value["engineering_criterion_D_minus_E"])
        stats(low,.5,.5/1.96)
        self.assertFalse(report.interpret(summary,paired,20261030,"new_replication")["engineering_criterion_C_minus_F"])
        stats(low,.5,0);stats(high,-.1000001,0)
        self.assertFalse(report.interpret(summary,paired,20261030,"new_replication")["engineering_criterion_C_minus_F"])

    def test_twelve_curves_last_five_and_inclusive_last5k_not_convergence(self):
        import grouped_codec_fdma_results as report
        with tempfile.TemporaryDirectory() as temporary:
            records=training_fixture(Path(temporary));curves=report.training_curves(records)
            self.assertEqual(len(curves["source_files"]),12);self.assertFalse(curves["convergence_established"])
            for seed in entry.SEEDS:
                for method in entry.METHODS:
                    curve=curves["per_seed"][str(seed)][method]
                    self.assertEqual(len(curve["points"]),40)
                    self.assertEqual([r["step"] for r in curve["last_five_validation_points"]],[18000,18500,19000,19500,20000])
                    tail=curve["last_5000_updates"]
                    self.assertEqual(tail["num_points"],11);self.assertAlmostEqual(tail["end_minus_start"],1)
                    self.assertAlmostEqual(tail["ols_slope_per_1000_updates"],.2)

    def test_curves_reject_corrupt_hash_best_step_missing_duplicate_or_changed_protocol(self):
        import grouped_codec_fdma_results as report
        with tempfile.TemporaryDirectory() as temporary:
            records=training_fixture(Path(temporary));record=records[str(entry.SEEDS[0])][entry.METHODS[0]]
            path=Path(record["metrics_path"]);original=path.read_text(encoding="utf-8")
            for defect in ("hash","best","missing","duplicate","protocol","selection","mean","event"):
                path.write_text(original,encoding="utf-8");bad=deepcopy(records)
                target=bad[str(entry.SEEDS[0])][entry.METHODS[0]]
                if defect=="hash":target["metrics_sha256"]="0"*64
                elif defect=="best":target["best_step"]=19500
                else:
                    items=[json.loads(line) for line in original.splitlines()]
                    if defect=="missing":items.pop()
                    elif defect=="duplicate":items.append(deepcopy(items[-1]))
                    elif defect=="protocol":items[1]["gamma_db"]=[0,10,20,25]
                    elif defect=="selection":items[1]["selection_metric_name"]="gamma25"
                    elif defect=="mean":items[1]["selection_metric"]+=1
                    else:items[0]["event"]="unsupported"
                    path.write_text("\n".join(json.dumps(r) for r in items)+"\n",encoding="utf-8")
                    target["metrics_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(defect=defect),self.assertRaises(ValueError):report.training_curves(bad)

    def test_csv_only_bundle_roundtrip_preserves_old_values_and_excludes_raw_weights(self):
        import grouped_codec_fdma_results as report
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);evaluations,records,evidence=bundle_fixture(folder)
            frozen={name:Path(name).read_bytes() for name in evidence["frozen_files"]}
            result=report.organize_results(folder,evaluations,records,evidence,make_plots=False)
            self.assertEqual(set(result["per_dataset"]),set(entry.DATASETS));self.assertEqual(result["primary_dataset"],"new_replication")
            self.assertFalse(result["historical"]["old_files_modified"])
            for path,content in frozen.items():self.assertEqual(Path(path).read_bytes(),content)
            output=Path(result["export_directory"])
            with zipfile.ZipFile(output/"small_results_bundle.zip") as archive:
                names=archive.namelist()
                self.assertTrue(all(not n.endswith((".pth",".pt",".gz")) for n in names))
                self.assertFalse(any(n.endswith((".pdf",".png")) for n in names))
            zh=(output/"results_draft_zh.md").read_text(encoding="utf-8")
            self.assertIn("不证明收敛",zh);self.assertNotIn("\ufffd",zh)
            english=(output/"results_draft_en.md").read_text(encoding="utf-8")
            self.assertIn("not convergence",english);self.assertIn("not met",english)
            self.assertEqual(len(list(output.glob("*summary_*.csv"))),4)

    def test_bundle_rejects_bad_freeze_before_export_and_old_value_changes(self):
        import grouped_codec_fdma_results as report
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);evaluations,records,evidence=bundle_fixture(folder)
            path=Path(evaluations["new_replication"][str(entry.SEEDS[0])])/"manifest.json"
            original=path.read_text(encoding="utf-8");manifest=json.loads(original)
            manifest["freeze_checks"]["loaded_methods"]=["legacy_fdma"]
            path.write_text(json.dumps(manifest),encoding="utf-8")
            with self.assertRaises(ValueError):report.organize_results(folder,evaluations,records,evidence,make_plots=False)
            self.assertFalse(list(folder.glob("export_*")))
            path.write_text(original,encoding="utf-8")
            interpretations={"old_completion":{}}
            for seed in entry.SEEDS:
                summary,paired=report_fixture("old_completion",seed)
                interpretations["old_completion"][str(seed)]=report.interpret(summary,paired,seed,"old_completion")
            report.verify_old_completion(evidence,interpretations)
            interpretations["old_completion"][str(entry.SEEDS[0])]["comparisons"]["D_minus_E"]["low_gamma"]["delta_mean"]+=1e-12
            with self.assertRaises(ValueError):report.verify_old_completion(evidence,interpretations)


if __name__=="__main__":unittest.main()
