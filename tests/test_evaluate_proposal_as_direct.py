"""Synthetic screening-evaluator contracts; no model or evaluation execution.

Run from 03_PRECODER:
    python -X utf8 -B -m unittest discover -s tests -p test_evaluate_proposal_as_direct.py -v

Statistics and metadata tests use stdlib fixtures. Optional tensor-state checks
use tiny CPU tensors only, with explicit skips if PyTorch is unavailable.
No real weights, model construction, channel generation, or CUDA are used.
"""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for directory in (ROOT, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from configs.as_direct_train_config import as_direct_config
from test_train_proposal_as_direct import resume_metadata
import evaluate_proposal_as_direct as entry


METHODS = ("full_control", "as_direct", "robust_frozen", "swin")
FB_SNRS = (0, 5, 10, 15, 20, 25)
SEED = 20261026
COEFFICIENTS = {0: 2, 5: -1, 10: 5, 15: 3, 20: -4, 25: 1}
PINNED_LATEST = {
    "full_control": "3362abf4812f1fd61c2dc66795766bbfcf26845f59161be739328e15258492cc",
    "as_direct": "211b55c28a1501b34e4197c69098d402a6d63828b3c2a79cf4154987b5f77981",
}


def mean_se_ci(values):
    """Independent ddof=1 reference; never call the evaluator under test."""
    mean = sum(values) / len(values)
    se = (math.sqrt(sum((value - mean) ** 2 for value in values) /
                    (len(values) - 1) / len(values)) if len(values) > 1 else None)
    return dict(mean=mean, se=se, ci95_low=None if se is None else mean - 1.96 * se,
                ci95_high=None if se is None else mean + 1.96 * se)


def synthetic_plan():
    return [dict(method_key=method, model_key="swin" if method == "swin" else "proposed",
                 allocations=[[64, 0]] if method == "swin" else [[0, 256]])
            for method in METHODS]


def synthetic_results(count=16, seed=SEED):
    """Shared channel variation is much larger than the paired method effects."""
    summary, samples = [], []
    for method in synthetic_plan():
        key = method["method_key"]
        for fb in FB_SNRS:
            condition = dict(method_key=key, model_key=method["model_key"],
                             scenario_id="K4_Nsc256_Dfb256", k_users=4, subcarriers=256,
                             feedback_budget=256, allocation_df=method["allocations"][0][0],
                             allocation_da=method["allocations"][0][1], feedback_snr_db=fb,
                             downlink_snr_db=25, seed=seed)
            rows = []
            for index in range(count):
                x = (index - (count - 1) / 2) / count
                common = 100 + 2 * index + fb / 10
                offset = {"full_control": 0, "as_direct": -2 + COEFFICIENTS[fb] * x,
                          "robust_frozen": -1 + 0.5 * x, "swin": 1 + 0.25 * x}[key]
                rows.append(dict(condition, sample_index=index, sum_rate=common + offset,
                                 nmse_db=-10 + (index % 3) / 5, precoder_power=256.0))
            samples.extend(rows)
            row = dict(condition, num_samples=count, antennas=32, rzf_lambda=0.001)
            for metric in ("sum_rate", "nmse_db", "precoder_power"):
                for suffix, value in mean_se_ci([sample[metric] for sample in rows]).items():
                    row[metric + "_" + suffix] = value
            summary.append(row)
    return summary, samples


def arm_metadata(arm, *, step=5000):
    cfg = as_direct_config(arm)
    latest = resume_metadata(cfg, step=step)
    best = deepcopy(latest)
    for field in ("optimizer", "scheduler", "rng_state"):
        del best[field]
    sidecars = {
        "config": dict(cfg.to_dict(), config_fingerprint=entry.fingerprint(cfg)),
        "status": dict(state="completed", step=step, total_steps=5000,
                       experiment=cfg.EXP_NAME, best_metric_name="mean_all",
                       best_metric=best["best_metric"]),
    }
    return cfg, best, latest, sidecars


def archived_payload(method):
    saved = deepcopy(method["archive"]["loaded_checkpoint_config"])
    fingerprint_data = deepcopy(saved)
    if method["model_key"] != "proposed":
        for field in ("OUTPUT_ROOT", "SAVE_DIR", "BEST_PATH", "SAVE_PATH", "LOG_DIR",
                      "METRICS_PATH", "STATUS_PATH"):
            fingerprint_data.pop(field, None)
    digest = hashlib.sha256(json.dumps(
        fingerprint_data, sort_keys=True, ensure_ascii=method["model_key"] != "proposed",
        default=str).encode("utf-8")).hexdigest()[:16]
    return dict(config=saved, config_fingerprint=digest,
                step=method["archive"]["checkpoint_step"],
                allocation_grid=deepcopy(method["allocations"]))


def functions():
    tree = ast.parse((ROOT / "evaluate_proposal_as_direct.py").read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class SampleStatisticsTests(unittest.TestCase):
    def test_unbiased_standard_error_and_normal_interval(self):
        result = entry.sample_statistics([1, 2, 3])
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["se"], 1 / math.sqrt(3))
        self.assertAlmostEqual(result["ci95_low"], 2 - 1.96 / math.sqrt(3))
        self.assertAlmostEqual(result["ci95_high"], 2 + 1.96 / math.sqrt(3))

    def test_single_observation_has_unavailable_uncertainty_not_zero(self):
        self.assertEqual(entry.sample_statistics([-2]),
                         dict(mean=-2, se=None, ci95_low=None, ci95_high=None))

    def test_empty_or_nonfinite_observations_are_rejected(self):
        for values in ([], [float("nan")], [1, float("inf")], [-float("inf"), 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                entry.sample_statistics(values)


class ResultValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan = synthetic_plan()
        self.summary, self.samples = synthetic_results()

    def test_four_methods_six_points_and_complete_sample_ids_are_accepted(self):
        indexed = entry.verify_results(self.summary, self.samples, self.plan, 16, SEED)
        self.assertEqual(len(indexed), 24)
        self.assertEqual(len(self.samples), 4 * 6 * 16)
        for rows in indexed.values():
            self.assertEqual(set(rows), set(range(16)))
        shuffled = entry.verify_results(list(reversed(self.summary)), list(reversed(self.samples)),
                                        self.plan, 16, SEED)
        self.assertEqual(indexed, shuffled)

    def test_missing_or_duplicate_samples_and_summary_conditions_are_rejected(self):
        cases = ((self.summary, self.samples[:-1]),
                 (self.summary, self.samples + [self.samples[0]]),
                 (self.summary[:-1], self.samples),
                 (self.summary + [self.summary[0]], self.samples))
        for summary, samples in cases:
            with self.subTest(summary=len(summary), samples=len(samples)), self.assertRaises(ValueError):
                entry.verify_results(summary, samples, self.plan, 16, SEED)

    def test_raw_nonfinite_metrics_and_missing_metrics_are_rejected(self):
        for field in ("sum_rate", "nmse_db", "precoder_power"):
            for value in (float("nan"), float("inf"), -float("inf"), "missing"):
                with self.subTest(field=field, value=value):
                    rows = deepcopy(self.samples)
                    if value == "missing":
                        del rows[0][field]
                    else:
                        rows[0][field] = value
                    with self.assertRaises(ValueError):
                        entry.verify_results(self.summary, rows, self.plan, 16, SEED)

    def test_every_summary_statistic_must_match_raw_rows(self):
        for metric in ("sum_rate", "nmse_db", "precoder_power"):
            for suffix in ("mean", "se", "ci95_low", "ci95_high"):
                field = metric + "_" + suffix
                with self.subTest(field=field):
                    rows = deepcopy(self.summary)
                    rows[0][field] += 0.01
                    with self.assertRaisesRegex(ValueError, "Summary/raw mismatch"):
                        entry.verify_results(rows, self.samples, self.plan, 16, SEED)
        for value in (None, float("nan"), float("inf")):
            rows = deepcopy(self.summary)
            rows[0]["sum_rate_mean"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                entry.verify_results(rows, self.samples, self.plan, 16, SEED)

    def test_wrong_method_snr_scenario_seed_or_allocation_is_rejected(self):
        changes = {"method_key": "proposed_best", "model_key": "csinet", "feedback_snr_db": 7,
                   "downlink_snr_db": 10, "scenario_id": "other", "k_users": 2,
                   "subcarriers": 128, "feedback_budget": 128, "seed": SEED + 1,
                   "allocation_df": 64, "allocation_da": 128}
        for field, value in changes.items():
            with self.subTest(field=field):
                samples = deepcopy(self.samples)
                samples[0][field] = value
                with self.assertRaises(ValueError):
                    entry.verify_results(self.summary, samples, self.plan, 16, SEED)

    def test_invalid_sample_ids_counts_and_method_plans_are_rejected(self):
        for index in (True, 0.0, -1, 16):
            samples = deepcopy(self.samples)
            samples[0]["sample_index"] = index
            with self.subTest(index=index), self.assertRaises(ValueError):
                entry.verify_results(self.summary, samples, self.plan, 16, SEED)
        for count in (True, 0, 7, 9, 16.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                entry.verify_results(self.summary, self.samples, self.plan, count, SEED)
        for plan in (self.plan[:-1], self.plan + [self.plan[0]], list(reversed(self.plan))):
            with self.subTest(methods=[m["method_key"] for m in plan]), self.assertRaises(ValueError):
                entry.verify_results(self.summary, self.samples, plan, 16, SEED)


class PairedStatisticsTests(unittest.TestCase):
    def paired(self, count=16, *, seed=SEED):
        summary, samples = synthetic_results(count, seed)
        return entry.paired_rows(summary, list(reversed(samples)), synthetic_plan(), count, seed)

    def selected(self, rows, estimand, *, feedback=None):
        return next(row for row in rows if row["comparison"] == "as_direct-full_control"
                    and row["estimand"] == estimand and row["feedback_snr_db"] == feedback)

    def assert_statistics(self, row, values, prefix="delta_"):
        for field, expected in mean_se_ci(values).items():
            with self.subTest(statistic=prefix + field):
                if expected is None:
                    self.assertIsNone(row[prefix + field])
                else:
                    self.assertAlmostEqual(row[prefix + field], expected, places=11)

    def test_fixed_paired_comparisons_preserve_negative_differences(self):
        rows, raw = self.paired()
        self.assertEqual(len(rows), 4 * 8)
        self.assertEqual(len(raw), 4 * 8 * 16)
        self.assertEqual({row["comparison"] for row in rows}, {
            "as_direct-full_control", "full_control-robust_frozen",
            "as_direct-robust_frozen", "as_direct-swin"})
        self.assertEqual(sum(row["role"] == "primary" for row in rows), 1)
        self.assertEqual(sum(row["role"] == "secondary_25db_degradation_check" for row in rows), 1)
        row = self.selected(rows, "per_snr", feedback=0)
        deltas = [-2 + 2 * (index - 7.5) / 16 for index in range(16)]
        self.assert_statistics(row, deltas)
        self.assertLess(row["delta_ci95_high"], 0)
        self.assertEqual(row["sample_ci_unit"], "complete_multi_user_channel")

    def test_low_snr_averages_within_channel_before_ddof_one_uncertainty(self):
        rows, raw = self.paired()
        row = self.selected(rows, "low_snr_mean")
        expected = [-2 + 2 * (index - 7.5) / 16 for index in range(16)]
        self.assert_statistics(row, expected)
        self.assertEqual(row["feedback_snr_points"], "0,5,10")
        self.assertEqual(row["role"], "primary")
        values = sorted((item["sample_index"], item["delta"]) for item in raw
                        if item["comparison"] == row["comparison"] and item["estimand"] == "low_snr_mean")
        for (index, delta), expected_delta in zip(values, expected):
            self.assertAlmostEqual(delta, expected_delta)
        incorrectly_flattened = [-2 + COEFFICIENTS[fb] * (index - 7.5) / 16
                                 for fb in (0, 5, 10) for index in range(16)]
        self.assertFalse(math.isclose(row["delta_se"], mean_se_ci(incorrectly_flattened)["se"], rel_tol=1e-3))

    def test_all_snr_average_remains_one_observation_per_channel(self):
        rows, _ = self.paired()
        row = self.selected(rows, "all_snr_mean")
        expected = [-2 + (index - 7.5) / 16 for index in range(16)]
        self.assert_statistics(row, expected)
        self.assertEqual(row["num_samples"], 16)
        self.assertEqual(row["feedback_snr_points"], "0,5,10,15,20,25")

    def test_batch_uncertainty_groups_sample_index_divided_by_eight(self):
        rows, raw = self.paired(128)
        row = self.selected(rows, "low_snr_mean")
        differences = [-2 + 2 * (index - 63.5) / 128 for index in range(128)]
        batch_means = [sum(differences[start:start + 8]) / 8 for start in range(0, 128, 8)]
        self.assertEqual(row["num_batches"], 16)
        self.assertEqual(row["batch_size"], 8)
        self.assertTrue(row["batch_ci_estimable"])
        self.assertEqual(row["batch_ci_unit"], "sample_index//8")
        self.assert_statistics(row, differences)
        self.assert_statistics(row, batch_means, prefix="batch_delta_")
        selected = [item for item in raw if item["comparison"] == row["comparison"]
                    and item["estimand"] == row["estimand"]]
        self.assertEqual({item["batch_index"] for item in selected}, set(range(16)))
        for item in selected:
            self.assertEqual(item["batch_index"], item["sample_index"] // 8)
        self.assertGreater(row["batch_delta_se"], row["delta_se"])

    def test_preflight_single_batch_has_none_batch_interval_and_real_sample_interval(self):
        rows, raw = self.paired(8, seed=20261027)
        for row in rows:
            self.assertEqual((row["num_samples"], row["num_batches"]), (8, 1))
            self.assertFalse(row["batch_ci_estimable"])
            for field in ("batch_delta_se", "batch_delta_ci95_low", "batch_delta_ci95_high"):
                self.assertIsNone(row[field])
            self.assertIsNotNone(row["delta_se"])
        self.assertEqual({item["batch_index"] for item in raw}, {0})


class CheckpointMetadataTests(unittest.TestCase):
    def test_protocols_have_fixed_independent_seeds_counts_and_physics(self):
        for preflight, seed, count, batches in ((False, SEED, 1600, 200), (True, 20261027, 8, 1)):
            p = entry.protocol(preflight)
            self.assertEqual((p["seed"], p["num_samples"], p["num_batches"], p["batch_size"]),
                             (seed, count, batches, 8))
            self.assertEqual(p["feedback_snr_db"], list(FB_SNRS))
            self.assertEqual(p["downlink_snr_db"], 25)
            self.assertEqual(p["scenario"], dict(K=4, Nt=32, Nsc=256, Dtot=256, Nsb=4))
            self.assertEqual(p["channel_profile"], "CDL-C")
        self.assertNotEqual(entry.protocol(True)["dataset_role"], entry.protocol(False)["dataset_role"])

    def test_plan_keeps_pinned_latest_identity_separate_from_selected_best_files(self):
        plan = entry.checkpoint_plan()
        self.assertEqual(tuple(item["method_key"] for item in plan), METHODS)
        self.assertEqual(len({item["path"] for item in plan}), 4)
        self.assertEqual(entry.LATEST_SHA256, PINNED_LATEST)
        for item in plan[:2]:
            self.assertEqual(Path(item["path"]).name, "best.pth")
            self.assertEqual(Path(item["latest_path"]).name, "latest.pth")
            self.assertNotEqual(item["path"], item["latest_path"])
            self.assertEqual(item["expected_latest_sha256"], PINNED_LATEST[item["method_key"]])
            self.assertNotIn("expected_best_sha256", item)
            self.assertEqual(item["allocations"], [[0, 256]])
        self.assertEqual(plan[2]["expected_best_sha256"],
                         "e21ea1feae595e0898b6be724f7062b05c09b2d1ef03fe9ffec7ac91869064bf")
        self.assertEqual(plan[3]["expected_best_sha256"],
                         "c913f3477f6af74665d7ba1b11a7a8e06bdea8c051822205ace1a8a071489276")
        self.assertEqual(plan[3]["allocations"], [[64, 0]])

    def test_completed_arm_metadata_accepts_both_correct_identities(self):
        for arm in METHODS[:2]:
            with self.subTest(arm=arm):
                cfg, best, latest, sidecars = arm_metadata(arm)
                record = entry.verify_arm_metadata(arm, best, latest, sidecars, cfg)
                self.assertEqual((record["best_step"], record["latest_step"], record["completed_updates"]),
                                 (5000, 5000, 5000))
                self.assertEqual(record["config"], cfg.to_dict())

    def test_arm_metadata_rejects_wrong_arm_steps_configs_and_status(self):
        for change in ("wrong_arm", "best_step", "unfinished", "saved_config", "sidecar_config",
                       "sidecar_state", "sidecar_metric", "source_hash", "missing_gate"):
            with self.subTest(change=change):
                cfg, best, latest, sidecars = arm_metadata("as_direct")
                selected_arm = "as_direct"
                if change == "wrong_arm":
                    selected_arm = "full_control"
                elif change == "best_step":
                    best["step"] = 4500
                elif change == "unfinished":
                    cfg, best, latest, sidecars = arm_metadata("as_direct", step=4500)
                elif change == "saved_config":
                    latest["config"]["architecture_metadata"]["ue_ffn_dim"] = 1024
                elif change == "sidecar_config":
                    sidecars["config"]["experiment"]["seed"] = 42
                elif change == "sidecar_state":
                    sidecars["status"]["state"] = "paused"
                elif change == "sidecar_metric":
                    sidecars["status"]["best_metric"] += 1
                elif change == "source_hash":
                    best["config"]["source_identity"]["checkpoint_sha256"] = "a" * 64
                else:
                    del latest["state"]["layers.0.alpha"]
                with self.assertRaises(ValueError):
                    entry.verify_arm_metadata(selected_arm, best, latest, sidecars, cfg)

    def test_archived_metadata_checks_saved_config_step_and_fingerprint(self):
        for method in entry.checkpoint_plan()[2:]:
            with self.subTest(method=method["method_key"]):
                payload = archived_payload(method)
                result = entry.verify_archived_metadata(method, payload)
                self.assertEqual(result["best_step"], method["archive"]["checkpoint_step"])
                for change in ("config", "step", "fingerprint", "allocation"):
                    bad = deepcopy(payload)
                    if change == "config":
                        bad["config"]["unexpected"] = True
                    elif change == "step":
                        bad["step"] -= 500
                    elif change == "fingerprint":
                        bad["config_fingerprint"] = "0" * 16
                    else:
                        bad["allocation_grid"] = [[32, 128]]
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        entry.verify_archived_metadata(method, bad)

    def test_swin_manifest_rejects_missing_duplicate_and_changed_pinned_hash(self):
        source = entry.archived_swin()
        changed = deepcopy(source)
        changed["checkpoint_sha256"] = "a" * 64
        for methods in ([], [source, source], [changed]):
            with self.subTest(methods=len(methods)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "manifest.json"
                path.write_text(json.dumps({"methods": methods}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    entry.archived_swin(path)

    def test_pinned_hash_mismatch_stops_before_deserialization(self):
        # None is intentional: a mismatch must fail before any loader is used.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arbitrary-bytes.dat"
            content = b"test bytes only; not a serialized checkpoint"
            path.write_bytes(content)
            hashes = {}
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                entry.load_payload(None, path, hashes, expected="0" * 64)
            self.assertEqual(hashes[str(path)], hashlib.sha256(content).hexdigest())


class OutputAndCliTests(unittest.TestCase):
    def test_output_directories_are_new_exclusive_and_separate_by_mode(self):
        self.assertEqual(entry.OUTPUT_ROOT, ROOT / "runs_as_screen/evaluation/k4_as_direct_v1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "screening"
            first = entry.create_output(root, run_id="fixture")
            preflight = entry.create_output(root, preflight=True, run_id="fixture")
            self.assertEqual(first.name, "screening_fixture")
            self.assertEqual(preflight.name, "preflight_fixture")
            marker = first / "keep.txt"
            marker.write_text("existing result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                entry.create_output(root, run_id="fixture")
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing result")
            self.assertNotEqual(entry.create_output(root), entry.create_output(root))

    def test_output_run_id_cannot_escape_dedicated_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "screening"
            for run_id in ("../escape", "a/b", "a\\b", "/absolute", "C:\\absolute", "unsafe name"):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    entry.create_output(root, run_id=run_id)
            self.assertFalse(root.exists())

    def test_manifest_never_overwrites_and_nonfinite_failure_keeps_valid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            entry.write_manifest(folder, {"status": "completed"})
            before = (folder / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                entry.write_manifest(folder, {"status": "changed"})
            self.assertEqual((folder / "manifest.json").read_bytes(), before)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                entry.write_manifest(folder, {"status": "completed", "metric": float("inf")})
            def reject_constant(token):
                self.fail("Nonstandard JSON number: " + token)
            saved = json.loads((folder / "manifest.json").read_text(encoding="utf-8"),
                               parse_constant=reject_constant)
            self.assertEqual(saved["status"], "failed")
            self.assertIsNone(saved["metric"])
            self.assertIn("manifest.metric", saved["nonfinite_report_fields"])

    def test_help_and_conflicting_or_protocol_override_arguments(self):
        with redirect_stdout(io.StringIO()), mock.patch.object(entry, "run_evaluation") as run:
            self.assertEqual(entry.main([]), 0)
            with self.assertRaises(SystemExit) as raised:
                entry.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        run.assert_not_called()
        for arguments in (["--dry-run", "--preflight"], ["--preflight", "--evaluate"],
                          ["--seed", "42"], ["--output-dir", "old"], ["--checkpoint", "latest"],
                          ["--num-samples", "8"], ["--models", "proposed"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    entry.build_parser().parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_lightweight_cli_contract(self):
        from cli_contracts import assert_lightweight_cli
        assert_lightweight_cli(self, 'evaluate_proposal_as_direct', [['--dry-run']],
            "assert [p['method_key'] for p in plans[0]['plan']] == ['full_control','as_direct','robust_frozen','swin']; assert plans[0]['evaluation']['num_samples']==1600; assert plans[0]['preflight']['num_samples']==8")


class EvaluationSourceTests(unittest.TestCase):
    def test_direct_identity_strict_load_and_trained_gates_are_preserved(self):
        function = functions()["build_method"]
        selected = next(node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "cls" for target in node.targets))
        self.assertIsInstance(selected, ast.IfExp)
        self.assertEqual(ast.unparse(selected.test), "key == 'as_direct'")
        self.assertEqual(ast.unparse(selected.body), "AdaptiveHybridPrecoderASDirect")
        self.assertEqual(ast.unparse(selected.orelse), "AdaptiveHybridPrecoder")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        load = next(node for node in calls if ast.unparse(node.func) == "model.load_state_dict")
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in load.keywords}, {"strict": True})
        names = {ast.unparse(node.func) for node in calls}
        self.assertIn("model.to(device).eval", names)
        self.assertIn("check_tensor_state", names)
        self.assertFalse(any("migrate" in name or name.endswith((".zero_", ".fill_")) for name in names))
        imports = [alias.name for node in ast.walk(function) if isinstance(node, ast.ImportFrom) for alias in node.names]
        self.assertNotIn("migrate_original_state_dict", imports)

    def test_latest_hash_is_pinned_best_is_separately_hashed_and_states_compared(self):
        function = functions()["run_evaluation"]
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        loads = [node for node in calls if ast.unparse(node.func) == "load_payload"]
        latest = next(node for node in loads if "latest_path" in ast.unparse(node.args[1]))
        self.assertEqual(ast.unparse(latest.args[3]), "entry['expected_latest_sha256']")
        self.assertTrue(any(len(node.args) == 3 and ast.unparse(node.args[1]) == "entry['path']" for node in loads))
        comparisons = [node for node in calls if ast.unparse(node.func) == "check_tensor_state"]
        self.assertTrue(any([ast.unparse(arg) for arg in node.args[1:]] == ["latest['state']", "best['state']"]
                            and any(kw.arg == "equal" and ast.literal_eval(kw.value) is True for kw in node.keywords)
                            for node in comparisons))

    def test_shared_evaluation_disables_oracle_and_preserves_sample_rows(self):
        function = functions()["evaluation_args"]
        strings = [node.value for node in ast.walk(function) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertIn("--no-include-proposed-best", strings)
        self.assertNotIn("--no-save-samples", strings)
        self.assertIn("--checkpoint", strings)
        self.assertIn("best", strings)
        source = ast.unparse(function)
        self.assertIn("args.include_proposed_best or args.no_save_samples", source)
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(function)))

    def test_evaluation_delegates_once_and_never_trains_or_changes_precision(self):
        function = functions()["run_evaluation"]
        names = [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]
        self.assertEqual(names.count("ev.evaluate_scenario"), 1)
        forbidden = {"train_proposal", "optimizer.step", "loss.backward", "model.train",
                     "torch.set_float32_matmul_precision", "torch.use_deterministic_algorithms"}
        self.assertTrue(forbidden.isdisjoint(names))
        self.assertFalse(any(name.endswith(".forward") for name in names))
        self.assertIn("torch.no_grad", names)
        self.assertGreaterEqual(names.count("validate_frozen"), 2)
        self.assertFalse(any(isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                             and ast.unparse(node).startswith("torch.backends.") for node in ast.walk(function)))


class TensorStateTests(unittest.TestCase):
    def torch_module(self):
        try:
            return importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            self.skipTest("PyTorch unavailable; tiny CPU tensor-state guard not run (no model/dependency bypass).")

    def state(self, torch):
        return {"running_pwr": torch.tensor([1.0], dtype=torch.float32),
                "layers.0.alpha": torch.tensor(0.25, dtype=torch.float32)}

    def test_equal_trained_gate_and_state_values_are_accepted(self):
        torch = self.torch_module()
        state = self.state(torch)
        entry.check_tensor_state(torch, state, {key: value.clone() for key, value in state.items()}, equal=True)
        self.assertEqual(state["layers.0.alpha"].item(), 0.25)

    def test_missing_gate_changed_shape_dtype_and_nonfinite_values_are_rejected(self):
        torch = self.torch_module()
        state = self.state(torch)
        for change in ("missing_gate", "shape", "dtype", "nonfinite"):
            altered = {key: value.clone() for key, value in state.items()}
            if change == "missing_gate":
                del altered["layers.0.alpha"]
            elif change == "shape":
                altered["layers.0.alpha"] = torch.tensor([0.25])
            elif change == "dtype":
                altered["layers.0.alpha"] = altered["layers.0.alpha"].to(torch.float64)
            else:
                altered["layers.0.alpha"] = torch.tensor(float("nan"))
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.check_tensor_state(torch, state, altered)

    def test_equal_mode_rejects_gate_reset_or_different_best_weights(self):
        torch = self.torch_module()
        state = self.state(torch)
        altered = {key: value.clone() for key, value in state.items()}
        altered["layers.0.alpha"] = torch.tensor(0.0)
        entry.check_tensor_state(torch, state, altered, equal=False)
        with self.assertRaisesRegex(ValueError, "State values differ"):
            entry.check_tensor_state(torch, state, altered, equal=True)


if __name__ == "__main__":
    unittest.main()
