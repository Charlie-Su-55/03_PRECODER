"""Stdlib/synthetic contracts only: no models, checkpoint bytes or channels."""
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

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import evaluate_swin_adaptation as entry


def plan():
    return [dict(method_key=key, model_key="swin" if key.startswith("swin") else "proposed",
                 allocations=[[64, 0]] if key.startswith("swin") else [[0, 256]]) for key in entry.METHODS]


def stats(values):
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return dict(mean=mean, se=se, ci95_low=None if se is None else mean - 1.96 * se,
                ci95_high=None if se is None else mean + 1.96 * se)


def results(count=16, training_seed=20261030):
    summaries, samples = [], []
    for item in plan():
        for snr in entry.FB_SNRS:
            base = dict(method_key=item["method_key"], model_key=item["model_key"], scenario_id="K4_Nsc256_Dfb256",
                        k_users=4, subcarriers=256, feedback_budget=256, allocation_df=item["allocations"][0][0],
                        allocation_da=item["allocations"][0][1], feedback_snr_db=snr, downlink_snr_db=25,
                        seed=entry.TEST_SEED, training_seed=training_seed)
            rows = []
            for index in range(count):
                x = (index - (count - 1) / 2) / count
                slope = {0: 2, 5: -1, 10: 5, 15: 0, 20: 3, 25: -2}[snr]
                offset = {"frozen_ue": -2, "joint": -1 + slope * x, "full_control": 0,
                          "as_receiver": 1, "swin": 2, "swin_adapted": 3 + x}[item["method_key"]]
                rows.append(dict(base, sample_index=index, sum_rate=20 + index + snr / 5 + offset,
                                 nmse_db=-10 + index / 10, precoder_power=256.0))
            summary = dict(base, num_samples=count, antennas=32, rzf_lambda=0.001)
            for metric in ("sum_rate", "nmse_db", "precoder_power"):
                summary.update({metric + "_" + suffix: value for suffix, value in stats([r[metric] for r in rows]).items()})
            summaries.append(summary)
            samples.extend(rows)
    return summaries, samples


class ProtocolTests(unittest.TestCase):
    def test_six_fixed_methods_preserve_existing_control_identity(self):
        actual = entry.checkpoint_plan(20261030)
        old = entry.spatial.checkpoint_plan(20261030)
        self.assertEqual(actual[:-1], old)
        self.assertEqual(tuple(r["method_key"] for r in actual), entry.METHODS)
        self.assertEqual(len({r["path"] for r in actual}), 6)
        self.assertEqual(actual[-1]["allocations"], [[64, 0]])

    def test_fresh_shared_data_protocol_and_separate_training_seeds(self):
        a, b = entry.protocol(20261030), entry.protocol(20261031)
        self.assertEqual(a["seed"], 20261110)
        self.assertEqual(a["channel_data_protocol"], b["channel_data_protocol"])
        self.assertEqual((a["batch_size"], a["num_batches"], a["num_samples"]), (8, 200, 1600))
        self.assertNotEqual(a["training_seed"], b["training_seed"])
        self.assertEqual(entry.protocol(20261030, True)["seed"], 20261111)
        self.assertEqual(entry.protocol(20261030, True)["num_samples"], 8)
        with self.assertRaises(ValueError):
            entry.protocol(20261025)

    def test_stdlib_help_dryruns_and_explicit_action(self):
        for seed in entry.SEEDS:
            command = [sys.executable, "-B", "-S", str(ROOT / "evaluate_swin_adaptation.py"),
                       "--seed", str(seed), "--dry-run"]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertTrue(output["models_not_imported"])
            self.assertTrue(output["checkpoint_bytes_not_read"])
            self.assertTrue(output["best_step0_allowed"])
            self.assertFalse(output["training_seeds_pooled"])
        result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "evaluate_swin_adaptation.py")],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)


class CheckpointMetadataTests(unittest.TestCase):
    def fixture(self, selected_step=0):
        from test_train_swin_adaptation import resume_metadata, best_metadata
        cfg = entry.swin_adapt_config(20261030)
        latest = resume_metadata(cfg, 20000, selected_step=selected_step)
        best = best_metadata(cfg, selected_step)
        sidecars = dict(config=cfg.to_dict(), status=dict(state="completed", step=20000,
            total_steps=20000, experiment=cfg.EXP_NAME, best_score=best["best_score"],
            selected_best_step=selected_step, best_selection_name="mean_all"))
        return cfg, best, latest, sidecars

    def test_source_step0_and_earlier_best_are_valid_with_completed20k(self):
        for step in (0, 500, 19500, 20000):
            cfg, best, latest, sidecars = self.fixture(step)
            result = entry.verify_adapted_metadata(cfg.SEED, best, latest, sidecars)
            self.assertEqual(result["selected_best_step"], step)
            self.assertEqual(result["latest_step"], 20000)
            self.assertEqual(result["selected_source_step0"], step == 0)
            self.assertEqual(result["selection_metric"], "mean_all")

    def test_incomplete_or_wrong_seed_source_fingerprint_selection_rejected(self):
        for defect in ("incomplete", "seed", "source", "fingerprint", "selected_step", "status", "metric", "sidecar"):
            cfg, best, latest, sidecars = self.fixture()
            if defect == "incomplete": latest["step"] = 5000
            elif defect == "seed": latest["config"]["SEED"] = 20261031
            elif defect == "source": latest["config"]["source_identity"]["checkpoint_sha256"] = "0" * 64
            elif defect == "fingerprint": latest.pop("config_fingerprint")
            elif defect == "selected_step": sidecars["status"]["selected_best_step"] = 500
            elif defect == "status": sidecars["status"]["state"] = "paused"
            elif defect == "metric": best["validation"]["mean_all"] += 1
            else: sidecars["config"]["D_F"] = 32
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.verify_adapted_metadata(cfg.SEED, best, latest, sidecars)


class PairedStatisticsTests(unittest.TestCase):
    def compute(self, count=16, training_seed=20261030):
        summary, samples = results(count, training_seed)
        return entry.paired_rows(summary, samples, plan(), count, entry.TEST_SEED, training_seed)

    def test_all36_conditions_and48_paired_estimates_negative_retained(self):
        summary, samples = results()
        self.assertEqual(len(summary), 36)
        paired, raw = self.compute()
        self.assertEqual((len(paired), len(raw)), (48, 48 * 16))
        row = next(r for r in paired if r["comparison"] == "joint-swin_adapted" and r["estimand"] == "low_snr_mean")
        self.assertAlmostEqual(row["delta_mean"], -4)
        self.assertEqual(row["role"], "primary")
        self.assertTrue(all(r["num_samples"] == 16 for r in paired))

    def test_within_channel_snr_aggregation_precedes_sample_and_batch_se(self):
        paired, _ = self.compute()
        row = next(r for r in paired if r["comparison"] == "joint-swin_adapted" and r["estimand"] == "low_snr_mean")
        differences = [-4 + (i - 7.5) / 16 for i in range(16)]
        for key, value in stats(differences).items():
            self.assertAlmostEqual(row["delta_" + key], value)
        for key, value in stats([statistics.fmean(differences[:8]), statistics.fmean(differences[8:])]).items():
            self.assertAlmostEqual(row["batch_delta_" + key], value)
        self.assertNotEqual(row["delta_se"], row["batch_delta_se"])

    def test_one_batch_ci_unavailable_and_training_seeds_not_pooled(self):
        for seed in entry.SEEDS:
            paired, _ = self.compute(8, seed)
            self.assertTrue(all(r["training_seed"] == seed and not r["training_seeds_pooled"] for r in paired))
            self.assertTrue(all(not r["batch_ci_estimable"] and r["batch_delta_se"] is None for r in paired))

    def test_input_order_independent(self):
        summary, samples = results()
        a = entry.paired_rows(summary, samples, plan(), 16, entry.TEST_SEED, 20261030)
        b = entry.paired_rows(summary[::-1], samples[::-1], plan(), 16, entry.TEST_SEED, 20261030)
        self.assertEqual(a, b)

    def test_missing_duplicate_nonfinite_wrong_seed_allocation_or_oracle_rejected(self):
        for defect in ("missing", "duplicate", "nonfinite", "seed", "allocation", "oracle", "summary"):
            summary, samples = results()
            if defect == "missing": samples.pop()
            elif defect == "duplicate": samples.append(deepcopy(samples[0]))
            elif defect == "nonfinite": samples[0]["sum_rate"] = float("nan")
            elif defect == "seed": samples[0]["training_seed"] = 20261031
            elif defect == "allocation": samples[-1]["allocation_df"] = 0
            elif defect == "oracle": summary[0]["method_key"] = "proposed_best"
            else: summary[0]["sum_rate_mean"] += 1
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.paired_rows(summary, samples, plan(), 16, entry.TEST_SEED, 20261030)


class IsolationTests(unittest.TestCase):
    def test_output_directory_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            path = root / "seed_20261030" / "attempt_001"
            entry.shared.create_output(20261030, root=root, output_dir=path)
            with self.assertRaises((ValueError, FileExistsError)):
                entry.shared.create_output(20261030, root=root, output_dir=path)

    def test_artifact_set_exact_and_bound_to_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            expected = {}
            for name in entry.OUTPUT_ARTIFACTS:
                (path / name).write_bytes(name.encode())
                expected[name] = hashlib.sha256(name.encode()).hexdigest()
            self.assertEqual(len(expected), 7)
            self.assertEqual(entry.artifact_hashes(path), expected)
            (path / "unrelated.csv").write_text("not part of this run", encoding="utf-8")
            with self.assertRaises(ValueError): entry.artifact_hashes(path)

    def test_strict_trained_swin_load_no_reinitialization(self):
        tree = ast.parse((ROOT / "evaluate_swin_adaptation.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_method")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "load_state_dict"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(k.arg == "strict" and k.value.value is True for k in calls[0].keywords))
        self.assertNotIn("initialize", ast.unparse(fn))

    def test_one_shared_evaluator_loop_no_old_result_files(self):
        tree = ast.parse((ROOT / "evaluate_swin_adaptation.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_evaluation")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "evaluate_scenario"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(k.arg == "channel_data" for k in calls[0].keywords))
        self.assertNotIn("read_csv", ast.unparse(fn))
        self.assertIn("utils/metrics.py", entry.SOURCE_FILES)


if __name__ == "__main__":
    unittest.main()
