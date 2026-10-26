"""Joint+Swin256 evaluation protocol/statistics tests, without model execution."""
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
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import evaluate_proposal_joint_swin as entry


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
                offset = {"plain_swin256": 0, "joint_swin256": -1 + slope * x,
                          "spatial_joint": 1, "full_control": 2, "swin": 3, "swin_adapted": 4}[item["method_key"]]
                rows.append(dict(base, sample_index=index, sum_rate=20 + index + snr / 5 + offset,
                                 nmse_db=-10 + index / 10, precoder_power=256.0))
            summary = dict(base, num_samples=count, antennas=32, rzf_lambda=0.001)
            for metric in ("sum_rate", "nmse_db", "precoder_power"):
                summary.update({metric + "_" + suffix: value for suffix, value in stats([r[metric] for r in rows]).items()})
            summaries.append(summary); samples.extend(rows)
    return summaries, samples


def diagnostic_rows(count=16, training_seed=20261030):
    output = []
    for arm in entry.ARMS:
        for fb in entry.FB_SNRS:
            for index in range(count):
                mse = .01 + index / 10000
                row = dict(method_key=arm, feedback_snr_db=fb, downlink_snr_db=25,
                           allocation_df=0, allocation_da=256, sample_index=index, batch_index=index // 8,
                           seed=entry.TEST_SEED, training_seed=training_seed, reliability_mean=.5,
                           correction_mse=0.0 if arm == "plain_swin256" else .005,
                           noise_est=.1 + .01 * (index // 8), regularization=.1 + .01 * (index // 8))
                for label in ("recovered", "corrected"):
                    row.update({label + "_mse": mse, label + "_nmse_linear": 2 * mse,
                                label + "_nmse_db": 10 * math.log10(2 * mse)})
                output.append(row)
    return output


class ProtocolTests(unittest.TestCase):
    def test_old_checkpoint_identity_preserved_and_spatial_joint_renamed(self):
        actual = entry.checkpoint_plan(20261030)
        old = {item["method_key"]: item for item in entry.adapted.checkpoint_plan(20261030)}
        self.assertEqual(tuple(item["method_key"] for item in actual), entry.METHODS)
        self.assertEqual(len({item["path"] for item in actual}), 6)
        self.assertEqual(actual[2]["path"], old["joint"]["path"])
        self.assertEqual(actual[2]["arm"], "joint")
        for row in actual[3:]: self.assertEqual(row["path"], old[row["method_key"]]["path"])

    def test_renamed_spatial_joint_calls_original_spatial_loader(self):
        item = dict(method_key="spatial_joint", label="existing joint")
        sentinel = SimpleNamespace(method_key="joint", display_name="old")
        with mock.patch.object(entry.adapted, "build_method", return_value=sentinel) as call:
            result = entry.build_method(None, None, None, item, {}, None)
        self.assertEqual(call.call_args.args[3]["method_key"], "joint")
        self.assertEqual(result.method_key, "spatial_joint")
        self.assertEqual(item["method_key"], "spatial_joint")

    def test_fixed_fresh_data_seed_same_across_training_seeds(self):
        a, b = entry.protocol(20261030), entry.protocol(20261031)
        self.assertEqual(a["seed"], 20261120)
        self.assertEqual(a["channel_data_protocol"], b["channel_data_protocol"])
        self.assertEqual((a["batch_size"], a["num_batches"], a["num_samples"]), (8, 200, 1600))
        self.assertEqual(entry.protocol(20261030, True)["seed"], 20261121)
        self.assertEqual(entry.protocol(20261030, True)["num_samples"], 8)

    def test_standard_library_dryruns_default_help(self):
        for seed in entry.SEEDS:
            result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "evaluate_proposal_joint_swin.py"),
                                     "--seed", str(seed), "--dry-run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertTrue(data["models_not_imported"])
            self.assertTrue(data["checkpoint_bytes_not_read"])
            self.assertTrue(data["diagnostic_truth_not_decoder_input"])
            self.assertFalse(data["training_seeds_pooled"])
        result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "evaluate_proposal_joint_swin.py")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)


class CheckpointMetadataTests(unittest.TestCase):
    def fixture(self, arm="plain_swin256", selected_step=19500):
        from test_train_proposal_joint_swin import resume_metadata
        cfg = entry.joint_swin_config(arm, 20261030)
        latest = resume_metadata(cfg, 20000)
        best = {name: deepcopy(latest[name]) for name in
                ("state", "config", "config_fingerprint", "allocation_grid", "step", "best_metric")}
        best["step"] = selected_step
        sidecars = dict(config=dict(cfg.to_dict(), config_fingerprint=entry.shared.fingerprint(cfg)),
                        status=dict(state="completed", step=20000, experiment=cfg.EXP_NAME,
                                    best_metric_name="mean_all", best_metric=best["best_metric"]))
        return cfg, best, latest, sidecars

    def test_both_new_arms_complete20k_best_can_be_earlier(self):
        for arm in entry.ARMS:
            cfg, best, latest, sidecars = self.fixture(arm)
            row = entry.verify_new_metadata(arm, cfg.SEED, best, latest, sidecars)
            self.assertEqual((row["selected_best_step"], row["latest_step"]), (19500, 20000))

    def test_new_best_zero_cross_arm_source_fingerprint_or_incomplete_rejected(self):
        for defect in ("zero", "incomplete", "arm", "source", "fingerprint", "status", "sidecar"):
            cfg, best, latest, sidecars = self.fixture()
            if defect == "zero": best["step"] = 0
            elif defect == "incomplete": latest["step"] = 5000
            elif defect == "arm": latest["config"]["arm"] = "joint_swin256"
            elif defect == "source": latest["config"]["source_identity"] = {}
            elif defect == "fingerprint": latest.pop("config_fingerprint")
            elif defect == "status": sidecars["status"]["state"] = "paused"
            else: sidecars["config"]["architecture_metadata"]["arm"] = "joint_swin256"
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.verify_new_metadata(cfg.arm, cfg.SEED, best, latest, sidecars)


class PairedStatisticsTests(unittest.TestCase):
    def compute(self, count=16, training_seed=20261030):
        summary, samples = results(count, training_seed)
        return entry.paired_rows(summary, samples, plan(), count, entry.TEST_SEED, training_seed)

    def test_all36_conditions40_pairs_and_negative_primary_retained(self):
        self.assertEqual(len(results()[0]), 36)
        rows, raw = self.compute()
        self.assertEqual((len(rows), len(raw)), (40, 40 * 16))
        row = next(r for r in rows if r["comparison"] == "joint_swin256-plain_swin256" and r["estimand"] == "low_snr_mean")
        self.assertEqual(row["role"], "primary")
        self.assertAlmostEqual(row["delta_mean"], -1)

    def test_cross_snr_within_sample_first_then_batch_ci(self):
        rows, _ = self.compute()
        row = next(r for r in rows if r["comparison"] == "joint_swin256-plain_swin256" and r["estimand"] == "low_snr_mean")
        differences = [-1 + 2 * (i - 7.5) / 16 for i in range(16)]
        for suffix, value in stats(differences).items(): self.assertAlmostEqual(row["delta_" + suffix], value)
        for suffix, value in stats([statistics.fmean(differences[:8]), statistics.fmean(differences[8:])]).items():
            self.assertAlmostEqual(row["batch_delta_" + suffix], value)

    def test_preflight_one_batch_ci_unavailable_seeds_separate(self):
        for seed in entry.SEEDS:
            rows, _ = self.compute(8, seed)
            self.assertTrue(all(r["batch_delta_se"] is None and not r["batch_ci_estimable"] for r in rows))
            self.assertTrue(all(r["training_seed"] == seed and not r["training_seeds_pooled"] for r in rows))

    def test_missing_duplicate_nonfinite_seed_allocation_or_oracle_rejected(self):
        for defect in ("missing", "duplicate", "nan", "seed", "allocation", "oracle"):
            summary, samples = results()
            if defect == "missing": samples.pop()
            elif defect == "duplicate": samples.append(deepcopy(samples[0]))
            elif defect == "nan": samples[0]["sum_rate"] = float("nan")
            elif defect == "seed": samples[0]["training_seed"] = 20261031
            elif defect == "allocation": samples[-1]["allocation_df"] = 0
            else: summary[0]["method_key"] = "proposed_best"
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.paired_rows(summary, samples, plan(), 16, entry.TEST_SEED, 20261030)


class DiagnosticTests(unittest.TestCase):
    def test_actual_runtime_shape_record_distinguishes_plain_from_joint(self):
        plain = dict(recovered_shape=[8, 256, 4], joint_context_shape=None,
                     spatial_shape=[8, 4, 512, 128], angular_delay_shape=[8, 4, 2, 256, 32])
        joint = dict(plain, joint_context_shape=[8, 1024, 128])
        self.assertEqual(entry.validate_execution_record("plain_swin256", plain), plain)
        self.assertEqual(entry.validate_execution_record("joint_swin256", joint), joint)
        with self.assertRaises((ValueError, RuntimeError)):
            entry.validate_execution_record("joint_swin256", plain)
        with self.assertRaises((ValueError, RuntimeError)):
            entry.validate_execution_record("plain_swin256", joint)

    def test_complete_two_arms_and_six_points_with_both_error_scopes(self):
        rows = entry.summarize_diagnostics(diagnostic_rows(), 16, entry.TEST_SEED, 20261030)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["num_samples"] == 16 and r["num_batches"] == 2 for r in rows))
        self.assertIn("batch_corrected_nmse_db_ci95_low", rows[0])
        self.assertIn("recovered_mse_mean", rows[0])
        self.assertEqual(next(r for r in rows if r["method_key"] == "plain_swin256")["correction_mse_mean"], 0)

    def test_preflight_batch_uncertainty_unavailable(self):
        rows = entry.summarize_diagnostics(diagnostic_rows(8), 8, entry.TEST_SEED, 20261030)
        self.assertTrue(all(r["batch_corrected_mse_se"] is None for r in rows))

    def test_diagnostic_missing_duplicate_nonfinite_or_batch_scale_mismatch(self):
        for defect in ("missing", "duplicate", "nan", "batch", "seed", "range", "plain_changed"):
            rows = diagnostic_rows()
            if defect == "missing": rows.pop()
            elif defect == "duplicate": rows.append(deepcopy(rows[0]))
            elif defect == "nan": rows[0]["corrected_mse"] = float("nan")
            elif defect == "batch": rows[1]["noise_est"] = 999
            elif defect == "seed": rows[0]["seed"] = 20261110
            elif defect == "range": rows[0]["reliability_mean"] = 2
            else: rows[0]["correction_mse"] = .01
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.summarize_diagnostics(rows, 16, entry.TEST_SEED, 20261030)


class IsolationTests(unittest.TestCase):
    def test_output_and_artifacts_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            output = root / "seed" / "attempt_001"
            entry.shared.create_output(20261030, root=root, output_dir=output)
            with self.assertRaises((ValueError, FileExistsError)):
                entry.shared.create_output(20261030, root=root, output_dir=output)
            hashes = {}
            for name in entry.OUTPUT_ARTIFACTS:
                (output / name).write_bytes(name.encode())
                hashes[name] = hashlib.sha256(name.encode()).hexdigest()
            self.assertEqual(entry.artifact_hashes(output), hashes)
            self.assertEqual(len(hashes), 10)
            (output / entry.OUTPUT_ARTIFACTS[-1]).unlink()
            with self.assertRaises(ValueError): entry.artifact_hashes(output)

    def test_strict_new_state_no_source_ue_initialization(self):
        tree = ast.parse((ROOT / "evaluate_proposal_joint_swin.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_method")
        loads = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "load_state_dict"]
        self.assertEqual(len(loads), 1)
        self.assertTrue(any(k.arg == "strict" and k.value.value is True for k in loads[0].keywords))
        self.assertNotIn("initialize_ue", ast.unparse(fn))

    def test_diagnostics_share_single_evaluation_and_no_forward_replay(self):
        tree = ast.parse((ROOT / "evaluate_proposal_joint_swin.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_evaluation")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "evaluate_scenario"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(k.arg == "diagnostic_sink" for k in calls[0].keywords))
        self.assertIn("utils/metrics.py", entry.SOURCE_FILES)


if __name__ == "__main__": unittest.main()
