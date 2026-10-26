"""Stdlib/synthetic AS-spatial evaluation checks; no models or channels run."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
if str(ROOT/"tests") not in sys.path:
    sys.path.insert(0,str(ROOT/"tests"))
import evaluate_proposal_as_spatial as entry


def statistics_(values):
    mean=sum(values)/len(values)
    se=math.sqrt(sum((value-mean)**2 for value in values)/(len(values)-1)/len(values)) if len(values)>1 else None
    return dict(mean=mean,se=se,ci95_low=None if se is None else mean-1.96*se,
                ci95_high=None if se is None else mean+1.96*se)


def plan():
    return [dict(method_key=key,model_key="swin" if key=="swin" else "proposed",
                 allocations=[[64,0]] if key=="swin" else [[0,256]]) for key in entry.METHODS]


def results(count=16,training_seed=20261030):
    summary,samples=[],[]
    for method in plan():
        for fb in entry.FB_SNRS:
            base=dict(method_key=method["method_key"],model_key=method["model_key"],scenario_id="K4_Nsc256_Dfb256",
                      k_users=4,subcarriers=256,feedback_budget=256,allocation_df=method["allocations"][0][0],
                      allocation_da=method["allocations"][0][1],feedback_snr_db=fb,downlink_snr_db=25,
                      seed=entry.TEST_SEED,training_seed=training_seed)
            rows=[]
            for i in range(count):
                x=(i-(count-1)/2)/count
                coefficient={0:2,5:-1,10:5,15:3,20:-4,25:1}[fb]
                offset={"frozen_ue":-1+coefficient*x,"joint":-2+2*coefficient*x,
                        "full_control":0,"as_receiver":0.5*x,"swin":2+x}[method["method_key"]]
                rows.append(dict(base,sample_index=i,sum_rate=20+i+fb/3+offset,nmse_db=-10+i/20,precoder_power=256.0))
            row=dict(base,num_samples=count,antennas=32,rzf_lambda=0.001)
            for metric in ("sum_rate","nmse_db","precoder_power"):
                row.update({metric+"_"+suffix:value for suffix,value in statistics_([r[metric] for r in rows]).items()})
            summary.append(row); samples.extend(rows)
    return summary,samples


def soft_rows(count=16,training_seed=20261030):
    return [dict(method_key=arm,feedback_snr_db=fb,downlink_snr_db=25,allocation_df=0,allocation_da=256,
                 sample_index=i,batch_index=i//8,seed=entry.TEST_SEED,training_seed=training_seed,
                 soft_mse=0.01+i/10000,soft_nmse_linear=0.02+i/10000,
                 soft_nmse_db=10*math.log10(0.02+i/10000),reliability_mean=0.5+i/(2*count),
                 noise_est=0.1+0.01*(i//8),regularization=0.1+0.01*(i//8))
            for arm in entry.ARMS for fb in entry.FB_SNRS for i in range(count)]


class ProtocolTests(unittest.TestCase):
    def test_fixed_five_methods_and_existing_control_paths(self):
        actual=entry.checkpoint_plan(20261030)
        self.assertEqual(tuple(r["method_key"] for r in actual),entry.METHODS)
        controls={r["method_key"]:r for r in entry.shared.checkpoint_plan(20261030)}
        for row in actual[2:]:
            self.assertEqual(row["path"],controls[row["method_key"]]["path"])
        self.assertEqual(len({r["path"] for r in actual}),5)

    def test_new_test_seed_shared_across_training_seeds(self):
        a,b=entry.protocol(20261030),entry.protocol(20261031)
        self.assertEqual(a["seed"],20261103)
        self.assertEqual(a["channel_data_protocol"],b["channel_data_protocol"])
        self.assertNotEqual(a["training_seed"],b["training_seed"])
        self.assertEqual((a["num_samples"],a["batch_size"],a["num_batches"]),(1600,8,200))
        self.assertEqual(entry.protocol(20261030,True)["seed"],20261104)
        self.assertEqual(entry.protocol(20261030,True)["num_samples"],8)

    def test_stdlib_dryrun_and_help_no_models(self):
        for seed in (20261030,20261031):
            run=subprocess.run([sys.executable,"-B","-S",str(ROOT/"evaluate_proposal_as_spatial.py"),"--seed",str(seed),"--dry-run"],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            output=json.loads(run.stdout)
            self.assertTrue(output["models_not_imported"])
            self.assertTrue(output["offline_truth_not_decoder_input"])
            self.assertFalse(output["training_seeds_pooled"])
        run=subprocess.run([sys.executable,"-B","-S",str(ROOT/"evaluate_proposal_as_spatial.py")],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn("usage:",run.stdout)

    def test_unapproved_training_seed_rejected(self):
        with self.assertRaises(ValueError): entry.protocol(20261025)
        with self.assertRaises(ValueError): entry.checkpoint_plan(True)


class CheckpointMetadataTests(unittest.TestCase):
    def fixture(self, arm="frozen_ue", best_step=19500):
        # These are metadata-only sentinels, not fake model dependencies or weights.
        from test_train_proposal_as_spatial import resume_metadata
        cfg=entry.as_spatial_config(arm,20261030)
        latest=resume_metadata(cfg,20000)
        best={key:deepcopy(latest[key]) for key in
              ("state","config","config_fingerprint","allocation_grid","step","best_metric")}
        best["step"]=best_step
        sidecars=dict(config=dict(cfg.to_dict(),config_fingerprint=entry.shared.fingerprint(cfg)),
                      status=dict(state="completed",step=20000,experiment=cfg.EXP_NAME,
                                  best_metric_name="mean_all",best_metric=best["best_metric"]))
        return cfg,best,latest,sidecars

    def test_both_20k_arms_accept_best_earlier_than_latest(self):
        for arm in entry.ARMS:
            cfg,best,latest,sidecars=self.fixture(arm)
            row=entry.verify_arm_metadata(arm,cfg.SEED,best,latest,sidecars)
            self.assertEqual(row["selected_best_step"],19500)
            self.assertEqual(row["latest_step"],20000)

    def test_incomplete_cross_arm_seed_fingerprint_source_and_status_rejected(self):
        for defect in ("latest_step","cross_arm","seed","fingerprint","source","status","metric"):
            cfg,best,latest,sidecars=self.fixture()
            if defect=="latest_step": latest["step"]=5000
            elif defect=="cross_arm": latest["config"]["arm"]="joint"
            elif defect=="seed": latest["config"]["experiment"]["seed"]=20261031
            elif defect=="fingerprint": latest.pop("config_fingerprint")
            elif defect=="source": latest["config"]["source_identity"]={}
            elif defect=="status": sidecars["status"]["state"]="paused"
            else: best["best_metric"]=latest["best_metric"]+1
            with self.subTest(defect=defect),self.assertRaises((ValueError,RuntimeError)):
                entry.verify_arm_metadata(cfg.arm,cfg.SEED,best,latest,sidecars)

    def test_constructed_decoder_and_receiver_metadata_must_match_in_full(self):
        saved=entry.as_spatial_config("joint",20261030).architecture_metadata
        entry.validate_loaded_architecture(saved,deepcopy(saved))
        for section,field,value in (("decoder","shifted_windows",True),
                                    ("decoder","spatial_expansion",[769,8191]),
                                    ("receiver","lambda_floor",1e-5)):
            actual=deepcopy(saved); actual[section][field]=value
            with self.subTest(section=section,field=field),self.assertRaises((ValueError,RuntimeError)):
                entry.validate_loaded_architecture(saved,actual)
        actual=deepcopy(saved); actual["decoder"].pop("initialization")
        with self.assertRaises((ValueError,RuntimeError)):
            entry.validate_loaded_architecture(saved,actual)
        self.assertIn("utils/metrics.py",entry.SOURCE_FILES)


class PairingTests(unittest.TestCase):
    def compute(self,count=16,training_seed=20261030):
        summary,samples=results(count,training_seed)
        return entry.paired_rows(summary,samples,plan(),count,entry.TEST_SEED,training_seed)

    def test_exact56_pairs_all_negative_results_kept(self):
        rows,raw=self.compute()
        self.assertEqual(len(rows),56)
        self.assertEqual(len(raw),56*16)
        self.assertEqual({r["comparison"] for r in rows},{f"{a}-{b}" for a,b in entry.COMPARISONS})
        for arm,expected in (("frozen_ue",-1),("joint",-2)):
            row=next(r for r in rows if r["comparison"]==f"{arm}-full_control" and r["estimand"]=="low_snr_mean")
            self.assertAlmostEqual(row["delta_mean"],expected)
            self.assertEqual(row["role"],"primary")

    def test_low_snr_averages_within_sample_then_batch(self):
        rows,_=self.compute()
        row=next(r for r in rows if r["comparison"]=="joint-full_control" and r["estimand"]=="low_snr_mean")
        values=[-2+4*(i-7.5)/16 for i in range(16)]
        expected=statistics_(values)
        for suffix,value in expected.items(): self.assertAlmostEqual(row["delta_"+suffix],value)
        batches=[sum(values[:8])/8,sum(values[8:])/8]
        for suffix,value in statistics_(batches).items(): self.assertAlmostEqual(row["batch_delta_"+suffix],value)
        self.assertEqual(row["num_samples"],16)
        self.assertNotEqual(row["delta_se"],row["batch_delta_se"])

    def test_one_batch_uncertainty_unavailable(self):
        rows,_=self.compute(8)
        for row in rows:
            self.assertFalse(row["batch_ci_estimable"])
            self.assertIsNone(row["batch_delta_se"])
            self.assertIsNone(row["batch_delta_ci95_low"])

    def test_training_seeds_not_pooled(self):
        a,_=self.compute(training_seed=20261030); b,_=self.compute(training_seed=20261031)
        self.assertTrue(all(r["num_samples"]==16 and not r["training_seeds_pooled"] for r in a+b))
        self.assertEqual({r["training_seed"] for r in a+b},{20261030,20261031})

    def test_sample_order_does_not_change_pairing(self):
        summary,samples=results()
        a=entry.paired_rows(summary,samples,plan(),16,entry.TEST_SEED,20261030)
        b=entry.paired_rows(summary[::-1],samples[::-1],plan(),16,entry.TEST_SEED,20261030)
        self.assertEqual(a,b)

    def test_missing_duplicate_nonfinite_or_wrong_seed_rejected(self):
        for kind in ("missing","duplicate","nan","seed","oracle"):
            summary,samples=results()
            if kind=="missing": samples.pop()
            elif kind=="duplicate": samples.append(deepcopy(samples[0]))
            elif kind=="nan": samples[0]["sum_rate"]=float("nan")
            elif kind=="seed": samples[0]["training_seed"]=20261031
            else: summary[0]["method_key"]="proposed_best"
            with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):
                entry.paired_rows(summary,samples,plan(),16,entry.TEST_SEED,20261030)


class SoftDiagnosticTests(unittest.TestCase):
    def test_all12_conditions_and_complete_channel_unit(self):
        output=entry.summarize_soft_rows(soft_rows(),16,entry.TEST_SEED,20261030)
        self.assertEqual(len(output),12)
        self.assertEqual({r["method_key"] for r in output},set(entry.ARMS))
        self.assertTrue(all(r["num_samples"]==16 and r["num_batches"]==2 for r in output))
        self.assertAlmostEqual(output[0]["soft_mse_mean"],statistics.mean([0.01+i/10000 for i in range(16)]))

    def test_preflight_batch_se_not_zero(self):
        output=entry.summarize_soft_rows(soft_rows(8),8,entry.TEST_SEED,20261030)
        for row in output:
            self.assertFalse(row["batch_ci_estimable"])
            self.assertIsNone(row["batch_soft_nmse_db_se"])

    def test_missing_duplicate_nonfinite_or_batch_scalar_change(self):
        for kind in ("missing","duplicate","nan","scalar","range","seed"):
            rows=soft_rows()
            if kind=="missing": rows.pop()
            elif kind=="duplicate": rows.append(deepcopy(rows[0]))
            elif kind=="nan": rows[0]["soft_mse"]=float("nan")
            elif kind=="scalar": rows[1]["noise_est"]=999
            elif kind=="range": rows[0]["reliability_mean"]=2
            else: rows[0]["training_seed"]=20261031
            with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):
                entry.summarize_soft_rows(rows,16,entry.TEST_SEED,20261030)

    def test_soft_summary_order_independent(self):
        rows=soft_rows()
        self.assertEqual(entry.summarize_soft_rows(rows,16,entry.TEST_SEED,20261030),
                         entry.summarize_soft_rows(rows[::-1],16,entry.TEST_SEED,20261030))


class IsolationTests(unittest.TestCase):
    def test_artifacts_exact_set_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            for name in entry.OUTPUT_ARTIFACTS: (path/name).write_bytes(name.encode())
            expected={name:hashlib.sha256(name.encode()).hexdigest() for name in entry.OUTPUT_ARTIFACTS}
            self.assertEqual(entry.artifact_hashes(path),expected)
            self.assertEqual(len(expected),10)
            (path/entry.OUTPUT_ARTIFACTS[-1]).unlink()
            with self.assertRaises(ValueError): entry.artifact_hashes(path)

    def test_output_protection_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"root"; output=root/"nested"/"new"
            entry.shared.create_output(20261030,root=root,output_dir=output)
            with self.assertRaises((ValueError,FileExistsError)):
                entry.shared.create_output(20261030,root=root,output_dir=output)

    def test_loads_complete_trained_state_never_initializes_ue(self):
        source=(ROOT/"evaluate_proposal_as_spatial.py").read_text(encoding="utf-8")
        self.assertNotIn("initialize_ue_from_full",source)
        self.assertNotIn("migrate_original_state_dict",source)
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="build_method")
        loads=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="load_state_dict"]
        self.assertEqual(len(loads),1)
        self.assertTrue(any(k.arg=="strict" and k.value.value is True for k in loads[0].keywords))

    def test_explicit_helper_dispatch_precedes_old_hook(self):
        tree=ast.parse((ROOT/"baseline_evaluate.py").read_text(encoding="utf-8"))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="forward_proposed_with_surrogate")
        source=ast.unparse(fn)
        self.assertLess(source.index("forward_with_diagnostics"),source.index("register_forward_hook"))
        self.assertLess(source.index("forward_with_representation"),source.index("register_forward_hook"))
        diagnostic=fn.body[1]
        calls=[n for n in ast.walk(diagnostic) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=="forward_with_diagnostics"]
        self.assertEqual(len(calls),1)
        self.assertNotIn("register_forward_hook",ast.unparse(diagnostic))
        index=next(i for i,a in enumerate(fn.args.kwonlyargs) if a.arg=="diagnostic_sink")
        self.assertIsNone(fn.args.kw_defaults[index].value)

    def test_no_runtime_model_replay_for_diagnostics(self):
        tree=ast.parse((ROOT/"evaluate_proposal_as_spatial.py").read_text(encoding="utf-8"))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_evaluation")
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=="evaluate_scenario"]
        self.assertEqual(len(calls),1)
        self.assertTrue(any(k.arg=="diagnostic_sink" for k in calls[0].keywords))


if __name__=="__main__": unittest.main()
