"""CPU/stdlib evaluator contracts only; never load a model, checkpoint or channel."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import evaluate_proposal_as_receiver as entry
from configs.as_receiver_train_config import as_receiver_config
from train_proposal_as_receiver import expected_cycle


def stats(values):
    mean = sum(values)/len(values)
    se = math.sqrt(sum((value-mean)**2 for value in values)/(len(values)-1)/len(values)) if len(values)>1 else None
    return dict(mean=mean, se=se, ci95_low=None if se is None else mean-1.96*se,
                ci95_high=None if se is None else mean+1.96*se)


def synthetic_plan():
    return [dict(method_key=key, model_key="swin" if key=="swin" else "proposed",
                 allocations=[[64,0]] if key=="swin" else [[0,256]]) for key in entry.METHODS]


def results(count=16, training_seed=20261030):
    summary, samples = [], []
    coeff = {0: 2.0, 5: -1.0, 10: 5.0, 15: 3.0, 20: -4.0, 25: 1.0}
    for method in synthetic_plan():
        key = method["method_key"]
        for fb in entry.FB_SNRS:
            condition = dict(method_key=key, model_key=method["model_key"], scenario_id=entry.SCENARIO_ID,
                             k_users=4, subcarriers=256, feedback_budget=256,
                             allocation_df=method["allocations"][0][0], allocation_da=method["allocations"][0][1],
                             feedback_snr_db=fb, downlink_snr_db=25, seed=entry.TEST_SEED, training_seed=training_seed)
            rows = []
            for i in range(count):
                x = (i-(count-1)/2)/count
                offset = {"full_control": 0.0, "as_receiver": -2+coeff[fb]*x,
                          "robust_frozen": -1+0.5*x, "swin": 1+0.25*x,
                          "as_receiver_off": -2.5+0.75*x}[key]
                rows.append(dict(condition, sample_index=i, sum_rate=100+2*i+fb/10+offset,
                                 nmse_db=-10+(i%3)/5, precoder_power=256.0))
            samples.extend(rows)
            row = dict(condition, num_samples=count, antennas=32, rzf_lambda=0.001)
            for metric in ("sum_rate", "nmse_db", "precoder_power"):
                row.update({metric+"_"+suffix: value for suffix,value in stats([r[metric] for r in rows]).items()})
            summary.append(row)
    return summary, samples


def metadata(arm="as_receiver", seed=20261030, best_step=19500):
    cfg = as_receiver_config(arm, seed)
    lr, beta = expected_cycle(cfg, 20000)
    state = {"running_pwr": None}
    if arm == "as_receiver":
        state["as_receiver.out_proj.weight"] = None
    latest = dict(state=state, config=cfg.to_dict(), config_fingerprint=entry.fingerprint(cfg),
                  allocation_grid=[(0,256)], step=20000, best_metric=1.25,
                  rng_state={"python": "schema", "numpy": "schema", "torch": "schema", "cuda": ["schema"]},
                  optimizer={"state": {0: {"step": "schema"}}, "param_groups": [dict(
                      params=[0], lr=lr, initial_lr=cfg.LR/25, max_lr=cfg.LR, min_lr=cfg.LR/25/10000,
                      weight_decay=cfg.WEIGHT_DECAY, eps=1e-8, base_momentum=0.85, max_momentum=0.95,
                      betas=(beta,0.999), amsgrad=False, maximize=False, capturable=False,
                      differentiable=False, foreach=None, fused=None)]},
                  scheduler=dict(total_steps=20000, last_epoch=20000, _step_count=20001,
                      _schedule_phases=[
                          dict(end_step=1999.0,start_lr="initial_lr",end_lr="max_lr",start_momentum="max_momentum",end_momentum="base_momentum"),
                          dict(end_step=19999,start_lr="max_lr",end_lr="min_lr",start_momentum="base_momentum",end_momentum="max_momentum")],
                      _anneal_func_type="cos",cycle_momentum=True,use_beta1=True,base_lrs=[cfg.LR/25],_last_lr=[lr]))
    best = deepcopy(latest)
    best["step"] = best_step
    for name in ("optimizer","scheduler","rng_state"):
        best.pop(name)
    sidecars = dict(config=dict(cfg.to_dict(),config_fingerprint=entry.fingerprint(cfg)),
                    status=dict(state="completed",step=20000,experiment=cfg.EXP_NAME,
                                best_metric_name="mean_all",best_metric=1.25))
    return best,latest,sidecars


class ProtocolTests(unittest.TestCase):
    def test_fixed_separate_seeds_and_common_test_data(self):
        a,b = entry.protocol(20261030),entry.protocol(20261031)
        self.assertEqual(a["channel_data_protocol"],b["channel_data_protocol"])
        self.assertEqual(a["seed"],20261101)
        self.assertNotEqual(a["training_seed"],b["training_seed"])
        self.assertEqual(a["num_samples"],1600)
        self.assertEqual(a["num_batches"],200)

    def test_preflight_separate_seed_and_eight_samples(self):
        p=entry.protocol(20261030,True)
        self.assertNotEqual(p["seed"],entry.TEST_SEED)
        self.assertEqual(p["num_samples"],8)
        self.assertEqual(p["num_batches"],1)

    def test_fixed_five_methods_and_same_off_checkpoint(self):
        plan=entry.checkpoint_plan(20261030)
        self.assertEqual(tuple(p["method_key"] for p in plan),entry.METHODS)
        self.assertEqual(plan[1]["path"],plan[-1]["path"])
        self.assertTrue(plan[-1]["inference_intervention"])
        self.assertEqual(len({p["path"] for p in plan}),4)

    def test_training_seed_paths_differ(self):
        self.assertNotEqual(entry.checkpoint_plan(20261030)[0]["path"],entry.checkpoint_plan(20261031)[0]["path"])

    def test_invalid_seed_rejected(self):
        for seed in (True,20261025,20261101,None):
            with self.subTest(seed=seed),self.assertRaises(ValueError):
                entry.protocol(seed)

    def test_noargs_help_and_dry_run_stdlib_only(self):
        help_run=subprocess.run([sys.executable,"-B","-S",str(ROOT/"evaluate_proposal_as_receiver.py")],capture_output=True,text=True)
        self.assertEqual(help_run.returncode,0,help_run.stderr)
        self.assertIn("usage:",help_run.stdout)
        run=subprocess.run([sys.executable,"-B","-S",str(ROOT/"evaluate_proposal_as_receiver.py"),"--seed","20261030","--dry-run"],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        data=json.loads(run.stdout)
        self.assertTrue(data["models_not_imported"])
        self.assertFalse(data["training_seeds_pooled"])

    def test_modes_exclusive(self):
        with self.assertRaises(SystemExit):
            entry.build_parser().parse_args(["--evaluate","--preflight"])


class CheckpointTests(unittest.TestCase):
    def test_best_may_precede_latest(self):
        for arm in ("full_control","as_receiver"):
            best,latest,sidecars=metadata(arm)
            record=entry.verify_arm_metadata(arm,20261030,best,latest,sidecars)
            self.assertEqual(record["selected_best_step"],19500)
            self.assertEqual(record["latest_step"],20000)

    def test_5000_not_completed20k(self):
        best,latest,sidecars=metadata()
        latest["step"]=5000
        with self.assertRaises(ValueError):
            entry.verify_arm_metadata("as_receiver",20261030,best,latest,sidecars)

    def test_config_fingerprint_missing_rejected(self):
        best,latest,sidecars=metadata()
        latest.pop("config_fingerprint")
        with self.assertRaises(ValueError):
            entry.verify_arm_metadata("as_receiver",20261030,best,latest,sidecars)

    def test_cross_arm_or_seed_rejected(self):
        best,latest,sidecars=metadata()
        for arm,seed in (("full_control",20261030),("as_receiver",20261031)):
            with self.subTest(arm=arm,seed=seed),self.assertRaises(ValueError):
                entry.verify_arm_metadata(arm,seed,best,latest,sidecars)

    def test_source_identity_change_rejected(self):
        best,latest,sidecars=metadata()
        latest["config"]["source_identity"]["checkpoint_sha256"]="0"*64
        with self.assertRaises(ValueError):
            entry.verify_arm_metadata("as_receiver",20261030,best,latest,sidecars)

    def test_completion_status_and_selection_metric_rejected(self):
        for alteration in ("status","metric"):
            best,latest,sidecars=metadata()
            if alteration=="status":
                sidecars["status"]["state"]="paused"
            else:
                best["best_metric"]=1.5
            with self.subTest(alteration=alteration),self.assertRaises((ValueError,RuntimeError)):
                entry.verify_arm_metadata("as_receiver",20261030,best,latest,sidecars)

    def test_pinned_hash_mismatch_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"opaque.data"
            path.write_bytes(b"not checkpoint bytes")
            with self.assertRaises(ValueError):
                entry.load_payload(None,path,{},"0"*64)


class StatisticsTests(unittest.TestCase):
    def evaluate(self,count=16,training_seed=20261030):
        summary,samples=results(count,training_seed)
        return entry.paired_rows(summary,samples,synthetic_plan(),count,entry.TEST_SEED,training_seed)

    def test_all40_pairs_retained(self):
        rows,raw=self.evaluate()
        self.assertEqual(len(rows),40)
        self.assertEqual(len(raw),40*16)
        self.assertEqual({row["comparison"] for row in rows},{f"{a}-{b}" for a,b in entry.COMPARISONS})

    def test_negative_mean_not_absolute(self):
        rows,_=self.evaluate()
        row=next(r for r in rows if r["comparison"]=="as_receiver-full_control" and r["estimand"]=="low_snr_mean")
        self.assertAlmostEqual(row["delta_mean"],-2.0)
        self.assertEqual(row["role"],"primary")

    def test_within_sample_snr_average_and_batch_ci(self):
        rows,_=self.evaluate()
        row=next(r for r in rows if r["comparison"]=="as_receiver-full_control" and r["estimand"]=="low_snr_mean")
        differences=[-2+2*(i-7.5)/16 for i in range(16)]
        expected=stats(differences)
        for suffix,value in expected.items():
            self.assertAlmostEqual(row["delta_"+suffix],value)
        batch=stats([sum(differences[:8])/8,sum(differences[8:])/8])
        for suffix,value in batch.items():
            self.assertAlmostEqual(row["batch_delta_"+suffix],value)
        self.assertEqual(row["num_samples"],16)
        self.assertEqual(row["num_batches"],2)
        self.assertNotEqual(row["delta_se"],row["batch_delta_se"])

    def test_one_batch_ci_unavailable_not_zero(self):
        rows,_=self.evaluate(8)
        for row in rows:
            self.assertFalse(row["batch_ci_estimable"])
            for suffix in ("se","ci95_low","ci95_high"):
                self.assertIsNone(row["batch_delta_"+suffix])

    def test_no_pooling_training_seeds(self):
        a,_=self.evaluate(training_seed=20261030)
        b,_=self.evaluate(training_seed=20261031)
        self.assertEqual(len(a),len(b))
        self.assertTrue(all(not r["training_seeds_pooled"] and r["num_samples"]==16 for r in a+b))
        self.assertEqual({r["training_seed"] for r in a+b},{20261030,20261031})

    def test_row_order_not_pairing_rule(self):
        summary,samples=results()
        expected=entry.paired_rows(summary,samples,synthetic_plan(),16,entry.TEST_SEED,20261030)
        self.assertEqual(expected,entry.paired_rows(list(reversed(summary)),list(reversed(samples)),synthetic_plan(),16,entry.TEST_SEED,20261030))

    def test_missing_duplicate_nonfinite_or_wrong_seed_rejected(self):
        for mode in ("missing","duplicate","nonfinite","seed","summary","oracle"):
            summary,samples=results()
            if mode=="missing": samples.pop()
            elif mode=="duplicate": samples.append(deepcopy(samples[0]))
            elif mode=="nonfinite": samples[0]["sum_rate"]=float("nan")
            elif mode=="seed": samples[0]["training_seed"]=20261031
            elif mode=="summary": summary[0]["sum_rate_mean"]+=1
            else: summary[0]["method_key"]="proposed_best"
            with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):
                entry.verify_results(summary,samples,synthetic_plan(),16,entry.TEST_SEED,20261030)

    def test_off_comparison_is_inference_intervention(self):
        rows,_=self.evaluate()
        off=[r for r in rows if r["reference_method_key"]=="as_receiver_off"]
        self.assertEqual(len(off),8)
        self.assertTrue(all(r["role"]=="inference_intervention" for r in off))


class IsolationTests(unittest.TestCase):
    def test_artifact_hashes_require_complete_safe_basename_set(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in entry.OUTPUT_ARTIFACTS:
                (folder/name).write_bytes(name.encode("utf-8"))
            expected={name:hashlib.sha256(name.encode("utf-8")).hexdigest() for name in entry.OUTPUT_ARTIFACTS}
            self.assertEqual(entry.artifact_hashes(folder),expected)
            (folder/"manifest.json").write_text("{}",encoding="utf-8")
            self.assertEqual(entry.artifact_hashes(folder),expected)
            (folder/entry.OUTPUT_ARTIFACTS[0]).unlink()
            with self.assertRaises(ValueError):
                entry.artifact_hashes(folder)

    def test_unexpected_artifact_rejected_not_silently_bundled(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in entry.OUTPUT_ARTIFACTS:
                (folder/name).write_bytes(b"synthetic metadata")
            (folder/"unexpected.csv").write_text("unknown",encoding="utf-8")
            with self.assertRaises(ValueError):
                entry.artifact_hashes(folder)

    def test_output_nested_allowed_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"evaluation"
            path=root/"batch"/"seed20261030"/"attempt001"
            self.assertEqual(entry.create_output(20261030,root=root,output_dir=path),path)
            with self.assertRaises((FileExistsError,ValueError)):
                entry.create_output(20261030,root=root,output_dir=path)
            with self.assertRaises(ValueError):
                entry.create_output(20261030,root=root,output_dir=Path(temp)/"outside")

    def test_strict_trained_receiver_load_no_migration(self):
        source=(ROOT/"evaluate_proposal_as_receiver.py").read_text(encoding="utf-8")
        tree=ast.parse(source)
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="build_method")
        calls=[n for n in ast.walk(method) if isinstance(n,ast.Call)]
        strict=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="load_state_dict"]
        self.assertEqual(len(strict),1)
        self.assertTrue(any(k.arg=="strict" and isinstance(k.value,ast.Constant) and k.value.value is True for k in strict[0].keywords))
        self.assertNotIn("migrate_original_state_dict",source)
        self.assertNotIn("strict=False",source)
        self.assertIn("model.receiver_enabled = key !=",source)

    def test_baseline_extension_preserves_default_branch(self):
        tree=ast.parse((ROOT/"baseline_evaluate.py").read_text(encoding="utf-8"))
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="evaluate_scenario")
        index=next(i for i,a in enumerate(func.args.kwonlyargs) if a.arg=="channel_data")
        self.assertIsNone(func.args.kw_defaults[index].value)
        first=func.body[0]
        self.assertIsInstance(first,ast.If)
        self.assertIn("build_channel_generator",ast.unparse(first.body))
        self.assertNotIn("build_channel_generator",ast.unparse(first.orelse))
        text=ast.unparse(func)
        self.assertIn("channel_data.training(batch_index + 1)",text)
        self.assertIn("generator.generate_batch_data(args.batch_size)",text)
        self.assertIn("set_seed(feedback_seed)",text)


if __name__=="__main__":
    unittest.main()
