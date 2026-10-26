"""Lightweight batch orchestration/report contracts, never execute a model."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import run_as_spatial_batch as batch
import as_spatial_results as results
from as_receiver_batch_support import save_json
from cli_contracts import assert_lightweight_cli


def fixtures(seed=20261030):
    summary = [dict(method_key=method, feedback_snr_db=fb, training_seed=seed, seed=results.TEST_SEED,
                    num_samples=1600, preflight=False, sum_rate_mean=20.0, sum_rate_se=0.1,
                    sum_rate_ci95_low=19.804, sum_rate_ci95_high=20.196)
               for method in results.METHODS for fb in results.FB]
    scopes = [("per_snr", str(fb)) for fb in results.FB] + [("low_snr_mean", "0,5,10"), ("all_snr_mean", "0,5,10,15,20,25")]
    pairs = []
    for method, ref in results.PAIRS:
        for scope, points in scopes:
            mean = -0.25 if method == "frozen_ue" else 0.7
            pairs.append(dict(method_key=method, reference_method_key=ref, estimand=scope,
                feedback_snr_points=points, training_seed=seed, seed=results.TEST_SEED,
                num_samples=1600, num_batches=200, training_seeds_pooled=False,
                delta_mean=mean, delta_se=.1, delta_ci95_low=mean-.196, delta_ci95_high=mean+.196,
                batch_delta_mean=mean, batch_delta_se=.2,
                batch_delta_ci95_low=mean-.392, batch_delta_ci95_high=mean+.392))
    return summary, pairs


class ReportTests(unittest.TestCase):
    def test_negative_arm_retained_and_batch_criterion(self):
        summary, pairs = fixtures()
        result = results.interpret(summary, pairs, 20261030)
        self.assertFalse(result["arms"]["frozen_ue"]["engineering_continuation_criterion_met"])
        self.assertTrue(result["arms"]["joint"]["engineering_continuation_criterion_met"])
        self.assertEqual(len(result["arms"]["frozen_ue"]["six_point_gap_to_swin"]), 6)
        self.assertLess(result["arms"]["frozen_ue"]["low_snr_minus_full"]["delta_mean"], 0)

    def test_sample_ci_does_not_override_failed_batch_ci(self):
        summary, pairs = fixtures()
        for row in pairs:
            if row["method_key"] == "joint" and row["reference_method_key"] == "full_control" and row["estimand"] == "low_snr_mean":
                row["batch_delta_ci95_low"] = -0.01
        self.assertFalse(results.interpret(summary, pairs, 20261030)["arms"]["joint"]["engineering_continuation_criterion_met"])

    def test_reject_missing_duplicate_nonfinite_wrong_protocol_and_pooling(self):
        summary, pairs = fixtures()
        cases = [(summary[:-1], pairs), (summary+[summary[0]], pairs), (summary, pairs[:-1]), (summary, pairs+[pairs[0]])]
        a, b = copy.deepcopy((summary, pairs)); a[0]["sum_rate_mean"] = float("nan"); cases.append((a,b))
        a, b = copy.deepcopy((summary, pairs)); b[0]["training_seeds_pooled"] = True; cases.append((a,b))
        a, b = copy.deepcopy((summary, pairs)); a[0]["seed"] = 20261101; cases.append((a,b))
        a, b = copy.deepcopy((summary, pairs)); a[0]["preflight"] = True; cases.append((a,b))
        for a, b in cases:
            with self.subTest(case=cases.index((a,b))), self.assertRaises(ValueError):
                results.interpret(a, b, 20261030)


class ControllerTests(unittest.TestCase):
    def test_dry_run_no_dependencies_weights_outputs_or_old_retraining(self):
        assert_lightweight_cli(self, "run_as_spatial_batch", [["--dry-run"]],
            "assert len(plans[0]['new_runs'])==4\nassert plans[0]['old_controls_retrained'] is False\nassert plans[0]['cleanup'] is False\nassert {p['seed'] for p in plans[0]['tests'].values()}=={20261103}")

    def test_partial_run_uses_included_500_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = SimpleNamespace(SAVE_DIR=str(root/"ckpt"), LOG_DIR=str(root/"logs"),
                SAVE_PATH=str(root/"ckpt/latest.pth"), BEST_PATH=str(root/"ckpt/best.pth"),
                CONFIG_PATH=str(root/"ckpt/config.json"), STATUS_PATH=str(root/"ckpt/status.json"))
            calls = []
            def child(folder, name, command):
                calls.append(command)
                if "--stop-after" in command:
                    Path(cfg.SAVE_DIR).mkdir()
                    save_json(cfg.STATUS_PATH, {"state": "paused", "step": 500})
                else:
                    save_json(cfg.STATUS_PATH, {"state": "completed", "step": 20000}, replace=True)
                return str(root/(name+".log"))
            with mock.patch.object(batch, "as_spatial_config", return_value=cfg), mock.patch.object(batch, "_child", side_effect=child), mock.patch.object(batch, "_stage_child", return_value={"report": {"completed_updates": 20000}, "_evidence": []}):
                batch._train_one(root, "frozen_ue", 20261030)
            self.assertEqual(calls[0][-2:], ["--stop-after", "500"])
            self.assertEqual(calls[1][-2:], ["--resume", "--train"])
            self.assertTrue(all(command[3] == "train_proposal_as_spatial.py" for command in calls))

    def test_completed_stage_reuses_verified_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root/"done.json"
            save_json(evidence, {"step": 20000})
            state = {"stages": {}}
            action = mock.Mock(return_value={"completed": True, "_evidence": [evidence]})
            batch._run_stage(state, root, "train", action)
            batch._run_stage(state, root, "train", action)
            action.assert_called_once()
            save_json(evidence, {"step": 500}, replace=True)
            with self.assertRaises(RuntimeError):
                batch._run_stage(state, root, "train", action)

    def test_unknown_output_without_latest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = SimpleNamespace(SAVE_PATH=str(root/"missing.pth"), SAVE_DIR=str(root), LOG_DIR=str(root/"logs"))
            with mock.patch.object(batch, "as_spatial_config", return_value=cfg), mock.patch.object(batch, "_child") as child:
                with self.assertRaises(RuntimeError):
                    batch._train_one(root, "joint", 20261031)
                child.assert_not_called()

    def test_reference_changes_fail_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"state.json"
            save_json(path, {"identity": 1})
            record = {"references": {"checkpoint_and_sidecar_sha256": {str(path): batch.sha256(path)}}}
            batch._assert_references(record)
            save_json(path, {"identity": 2}, replace=True)
            with self.assertRaises(RuntimeError):
                batch._assert_references(record)

    def test_bundle_is_verified_unique_and_contains_current_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stale = root / "results_bundle_interrupted.tar.gz.partial"
            stale.write_bytes(b"interrupted package")
            summary = {"status": "completed", "stages": {}}
            batch._package(root, summary)
            first = Path(summary["return_files"][1])
            original_hash = batch.sha256(first)
            summary["resumed"] = True
            batch._package(root, summary)
            second = Path(summary["return_files"][1])
            self.assertNotEqual(first, second)
            self.assertEqual(batch.sha256(first), original_hash)
            self.assertEqual(stale.read_bytes(), b"interrupted package")
            with tarfile.open(second, "r:gz") as archive:
                self.assertEqual(json.load(archive.extractfile("batch_summary.json")), summary)
            self.assertFalse(Path(str(second)+".partial").exists())


if __name__ == "__main__":
    unittest.main()
