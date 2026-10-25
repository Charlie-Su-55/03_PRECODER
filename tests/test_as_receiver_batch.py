"""Focused controller safety/statistics checks; no model, channel or GPU calls."""
import csv
import hashlib
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
import as_receiver_batch_support as io
import as_receiver_results as results
import run_as_receiver_batch as batch


class CleanupTests(unittest.TestCase):
    def fixture(self, root, relative=None):
        relative = relative or io.CLEAN_TARGETS[0]
        folder = root / relative
        folder.mkdir(parents=True)
        if "diagnostics" in relative:
            io.save_json(folder / "report.json", {"passed": True, "status": "passed"})
        elif "evaluation" in relative:
            io.save_json(folder / "manifest.json", {"status": "completed", "protocol": {
                "num_samples": 8, "dataset_role": "preflight"}})
        else:
            (folder / "test.log").write_text("completed synthetic checks\n", encoding="utf-8")
        return relative, folder

    def test_archive_manifest_and_hash_precede_exact_removal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            relative, folder = self.fixture(root)
            result = io.archive_exact_target(relative, root / "archives", root=root)
            self.assertEqual(result["status"], "archived_and_removed")
            self.assertFalse(folder.exists())
            self.assertEqual(len(result["deleted_files"]), 1)
            listing = json.loads(Path(result["listing"]).read_text())
            with tarfile.open(result["archive"]) as archive:
                for row in listing["files"]:
                    content = archive.extractfile(row["path"]).read()
                    self.assertEqual(hashlib.sha256(content).hexdigest(), row["sha256"])
                    self.assertEqual(len(content), row["size"])
            self.assertEqual(io.sha256(result["archive"]), result["archive_sha256"])

    def test_missing_skipped_and_unknown_target_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertEqual(io.archive_exact_target(io.CLEAN_TARGETS[0], root / "archives", root=root)["status"], "skipped")
            for relative in ("runs_as_screen", "../escape", "terminal_logs/unapproved", "runs_*"):
                with self.assertRaises(ValueError):
                    io.archive_exact_target(relative, root / "archives", root=root)

    def test_weights_or_unknown_files_preserve_whole_target(self):
        for filename in ("best.pth", "channels.npz", "script.py"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                relative, folder = self.fixture(root)
                (folder / filename).write_bytes(b"keep")
                result = io.archive_exact_target(relative, root / "archives", root=root)
                self.assertEqual(result["status"], "preserved")
                self.assertEqual((folder / filename).read_bytes(), b"keep")
                self.assertFalse((root / "archives").exists())

    def test_formal_sample_ids_in_preflight_directory_are_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            relative, folder = self.fixture(root, io.CLEAN_TARGETS[1])
            (folder / "samples.csv").write_text("sample_index,sum_rate\n1599,20\n", encoding="utf-8")
            self.assertEqual(io.archive_exact_target(relative, root / "archives", root=root)["status"], "preserved")
            self.assertTrue((folder / "samples.csv").exists())

    def test_changed_inventory_prevents_deletion(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            relative, folder = self.fixture(root)
            before = io.inventory(folder, root)
            with mock.patch.object(io, "inventory", side_effect=[before, ([], [])]):
                result = io.archive_exact_target(relative, root / "archives", root=root)
            self.assertEqual(result["status"], "preserved")
            self.assertTrue((folder / "report.json").is_file())
            self.assertTrue(Path(result["archive"]).is_file())

    def test_output_protection_and_text_logs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in io.CLEAN_TARGETS[2:]:
                self.fixture(root, relative)
                self.assertEqual(io.archive_exact_target(relative, root / "archives", root=root)["status"], "archived_and_removed")
            with self.assertRaises(ValueError):
                io.safe_inside(root, root)
            marker = root / "existing.json"
            io.save_json(marker, {"keep": True})
            with self.assertRaises(FileExistsError):
                io.save_json(marker, {"keep": False})
            self.assertEqual(json.loads(marker.read_text()), {"keep": True})


def synthetic_tables(seed=20261030, low=0.6, lower=0.1, high_lower=-0.05):
    summary = [dict(method_key=method, feedback_snr_db=snr, num_samples=1600,
                    training_seed=seed, seed=20261101, preflight=False,
                    sum_rate_mean=10, sum_rate_se=.1, sum_rate_ci95_low=9.8, sum_rate_ci95_high=10.2)
               for method in results.METHODS for snr in results.FB]
    paired = []
    scopes = [("low_snr_mean", "0,5,10", None), ("all_snr_mean", "0,5,10,15,20,25", None)] + [("per_snr", str(snr), snr) for snr in results.FB]
    for method, reference in results.COMPARISONS:
        for estimand, points, snr in scopes:
            mean, lo = (low, lower) if estimand == "low_snr_mean" else (0, high_lower)
            row = dict(method_key=method, reference_method_key=reference, estimand=estimand,
                       feedback_snr_db=snr, feedback_snr_points=points, num_samples=1600,
                       num_batches=200, training_seed=seed, seed=20261101, training_seeds_pooled=False)
            for prefix in ("delta_", "batch_delta_"):
                row.update({prefix+"mean": mean, prefix+"se": .1, prefix+"ci95_low": lo, prefix+"ci95_high": mean+.2})
            paired.append(row)
    return summary, paired


class ReportTests(unittest.TestCase):
    def test_predeclared_criterion_uses_batch_interval_and_keeps_negatives(self):
        summary, paired = synthetic_tables()
        self.assertTrue(results.interpret(summary, paired, 20261030)["engineering_continuation_criterion_met"])
        paired[0]["batch_delta_ci95_low"] = -0.1
        self.assertFalse(results.interpret(summary, paired, 20261030)["engineering_continuation_criterion_met"])
        paired[0]["delta_mean"] = paired[0]["batch_delta_mean"] = -0.6
        record = results.interpret(summary, paired, 20261030)
        self.assertEqual(record["low_snr_as_receiver_minus_full"]["delta_mean"], -0.6)
        self.assertFalse(record["seeds_pooled"])

    def test_low_threshold_and_25db_degradation_are_both_required(self):
        for kwargs in (dict(low=.49), dict(lower=0), dict(high_lower=-.1001)):
            summary, paired = synthetic_tables(**kwargs)
            self.assertFalse(results.interpret(summary, paired, 20261030)["engineering_continuation_criterion_met"])

    def test_wrong_seed_preflight_duplicate_missing_nonfinite_refused(self):
        for change in ("missing", "duplicate", "preflight", "seed", "nonfinite"):
            summary, paired = synthetic_tables()
            if change == "missing": summary.pop()
            elif change == "duplicate": summary.append(summary[0])
            elif change == "preflight": summary[0]["num_samples"] = 8
            elif change == "seed": summary[0]["training_seed"] = 20261031
            else: paired[0]["delta_mean"] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                results.interpret(summary, paired, 20261030)


class ControllerTests(unittest.TestCase):
    def test_plan_fixed_four_runs_and_no_long_work_in_dry_run(self):
        from cli_contracts import assert_lightweight_cli
        assert_lightweight_cli(self, "run_as_receiver_batch", [["--dry-run"]],
            "assert len(plans[0]['runs'])==4; assert all(r['total_updates']==20000 for r in plans[0]['runs']); assert plans[0]['evaluation']['seed']==20261101")

    def test_completed_stage_is_verified_not_reexecuted(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            artifact = folder / "evidence.json"
            io.save_json(artifact, {"passed": True})
            summary = dict(stages={"done": dict(status="completed", result={"ok": True},
                evidence_sha256={str(artifact): io.sha256(artifact)})})
            action = mock.Mock(side_effect=AssertionError("must not rerun"))
            self.assertEqual(batch._run_stage(summary, folder, "done", action), {"ok": True})
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                batch._run_stage(summary, folder, "done", lambda: None)

    def test_smoke_is_first500_then_resume_not_extra_updates(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            cfg = SimpleNamespace(SAVE_PATH=str(folder/"ckpt/latest.pth"), BEST_PATH="best", SAVE_DIR=str(folder/"ckpt"),
                                  LOG_DIR=str(folder/"logs"), STATUS_PATH=str(folder/"status.json"), CONFIG_PATH="config")
            io.save_json(cfg.STATUS_PATH, {"state": "paused", "step": 500})
            with mock.patch.object(batch, "as_receiver_config", return_value=cfg), mock.patch.object(batch, "_child", return_value="log") as child, mock.patch.object(batch, "_stage_child", return_value={"report":{}, "_evidence":[]}):
                batch._train_one(folder, "as_receiver", 20261030)
            commands = [call.args[2] for call in child.call_args_list]
            self.assertEqual(commands[0][-2:], ["--stop-after", "500"])
            self.assertEqual(commands[1][-2:], ["--resume", "--train"])

    def test_failed_stage_preserves_report_and_nonfinite_not_hidden(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            summary = dict(stages={})
            def fail():
                raise FloatingPointError("nonfinite fixture")
            with self.assertRaises(FloatingPointError):
                batch._run_stage(summary, folder, "numerical", fail)
            saved = json.loads((folder/"batch_summary.json").read_text())
            self.assertEqual(saved["stages"]["numerical"]["status"], "failed")
            self.assertIn("nonfinite fixture", saved["stages"]["numerical"]["error"])


if __name__ == "__main__":
    unittest.main()
