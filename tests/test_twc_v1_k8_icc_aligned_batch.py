"""Standard-library controller regression checks; no model/CUDA execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_twc_v1_k8_icc_aligned as controller
from configs import twc_v1_k8_icc_aligned_config as cfg
from physical_split_finish_io import write_union_csv


class ControllerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "owned_icc_root"
        self.root.mkdir()
        self.patchers = [mock.patch.object(controller, "OUTPUT_ROOT", self.root),
                         mock.patch.object(controller.runtime, "OUTPUT_ROOT", self.root)]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def test_completed_real_stage_reuses_hashed_evidence_and_rejects_tampering(self):
        summary = dict(stages={}, namespace=cfg.NAMESPACE)
        path = self.root / "batch_summary.json"
        evidence = self.root / "completed.json"
        invocations = []
        def action(record):
            invocations.append(record["attempt_id"])
            evidence.write_text('{"value": 3}\n', encoding="utf-8")
            return dict(status="passed"), [evidence]
        first = controller.stage(summary, path, "component_100", action)
        reloaded = controller.runtime.read_json(path)
        second = controller.stage(reloaded, path, "component_100", action)
        self.assertEqual(first, second)
        self.assertEqual(len(invocations), 1)
        evidence.write_text('{"value": 4}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            controller.stage(reloaded, path, "component_100", action)
        self.assertEqual(len(invocations), 1)
        with self.assertRaisesRegex(ValueError, "no evidence"):
            controller.verify_evidence(dict(status="completed", files={}))

    def test_failed_stage_preserves_own_exception_in_receipt_and_readout(self):
        path = self.root / "batch_summary.json"
        summary = dict(stages={}, namespace=cfg.NAMESPACE)
        def action(record):
            record["log_path"] = str(self.root / "own_failure.log")
            raise ValueError("calibration-own-field-mismatch")
        with self.assertRaisesRegex(ValueError, "calibration-own"):
            controller.stage(summary, path, "ce_calibration", action)
        saved = controller.runtime.read_json(path)
        self.assertEqual(saved["status"], "failed")
        self.assertIn("calibration-own-field-mismatch", saved["stages"]["ce_calibration"]["traceback"])
        irrelevant = self.root / "last_training.log"
        irrelevant.write_text("UNRELATED-LATEST-TRAINING-OUTPUT", encoding="utf-8")
        text = controller.compact_summary(saved)
        self.assertIn("FAILED STAGE: ce_calibration", text)
        self.assertIn("calibration-own-field-mismatch", text)
        self.assertNotIn("UNRELATED", text)

    def test_child_startup_failure_uses_exact_recorded_log(self):
        path = self.root / "batch_summary.json"
        summary = dict(stages={}, namespace=cfg.NAMESPACE, core={})
        (self.root / "last_training.log").write_text("wrong-last-training-tail", encoding="utf-8")
        def failed_child(command, **kwargs):
            kwargs["stdout"].write("child-import-error\nNo module named real_dependency\n")
            kwargs["stdout"].flush()
            return subprocess.CompletedProcess(command, 1)
        def action(record):
            controller.child(summary, path, record, "standalone_eval.py", ["--run"])
            self.fail("A failed child must not return")
        with mock.patch.object(controller.runtime, "guard_core"), mock.patch.object(
                controller.subprocess, "run", side_effect=failed_child):
            with self.assertRaisesRegex(RuntimeError, "child-import-error"):
                controller.stage(summary, path, "final_validation", action)
        saved = controller.runtime.read_json(path)
        self.assertTrue(Path(saved["stages"]["final_validation"]["log_path"]).is_file())
        text = controller.compact_summary(saved)
        self.assertIn("No module named real_dependency", text)
        self.assertNotIn("wrong-last-training-tail", text)

    def test_default_summary_keeps_all_six_points_soft_diagnostics_and_bounded_size(self):
        final = self.root / "evaluation"
        final.mkdir()
        points = []
        for arm in cfg.ARMS:
            for gamma in cfg.GAMMAS:
                row = dict(arm=arm, gamma_db=gamma, sum_rate_bps_hz_mean=20. + gamma / 10,
                           dl_nmse_db_from_linear_mean=-3. - gamma / 2)
                if arm == "hybrid_g4":
                    row.update(soft_key_nmse_linear_mean=.1, soft_shared_nmse_linear_mean=.2)
                points.append(row)
        write_union_csv(final / "summary.csv", points)
        write_union_csv(final / "paired_deltas.csv", [dict(metric="sum_rate_bps_hz", scope=scope,
            gamma_db=25 if scope == "point" else None, mean=-.1, batch_ci95_low=-.3,
            batch_ci95_high=.1) for scope in ("low_0_5_10", "all_six", "point")])
        for component in ("ce",) + cfg.ARMS:
            directory = self.root / "training" / component
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(json.dumps(dict(status="completed", updated="fixed-date",
                counters=cfg.counters(component, cfg.CE_UPDATES if component == "ce" else cfg.CODEC_UPDATES),
                ETA_seconds=0, seconds_per_update=1.3, memory=dict(peak_allocated_bytes=1_000_000))), encoding="utf-8")
        summary = dict(status="completed", updated="fixed-date", training_code=dict(commit="a"*40),
            stages=dict(final_validation=dict(status="completed", details=dict(output_dir=str(final))),
                        result_package=dict(status="completed", details=dict(path=str(final / "package.zip")))))
        text = controller.compact_summary(summary)
        self.assertLessEqual(len(text.splitlines()), 40)
        self.assertLessEqual(len(text.encode("utf-8")), 8000)
        self.assertEqual(text.count("0,5,10,15,20,25"), 2)
        self.assertIn("soft key/shared", text)
        self.assertIn("batchCI=[-0.30000,+0.10000]", text)
        self.assertIn(str(final / "package.zip"), text)
        self.assertIn("updates=20000", text)
        self.assertIn("micro=160000", text)
        self.assertIn("MU=640000", text)


class AdmissionAndCLITests(unittest.TestCase):
    def test_gpu_shared_policy_is_explicit_and_reads_only_metadata(self):
        responses = ['GPU-known,"GPU, comma name",18000,65\n', 'GPU-known, 123\nGPU-other, 999\n']
        with mock.patch.object(controller.subprocess, "check_output", side_effect=responses) as read:
            record = controller.gpu_admission(allow_shared_gpu=True, min_free_mib=16000, max_gpu_util=75)
        self.assertEqual(record["compute_pids"], ["123"])
        self.assertEqual(record["gpu"], "GPU, comma name")
        self.assertFalse(record["continuous_monitor"])
        self.assertFalse(record["hard_resource_limit"])
        self.assertEqual(read.call_count, 2)
        with mock.patch.object(controller.subprocess, "check_output", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "no processes terminated"):
                controller.gpu_admission(allow_shared_gpu=False, min_free_mib=16000, max_gpu_util=75)
        with mock.patch.object(controller.subprocess, "check_output", side_effect=responses):
            with self.assertRaises(RuntimeError):
                controller.gpu_admission(allow_shared_gpu=True, min_free_mib=19000, max_gpu_util=75)
        with mock.patch.object(controller.subprocess, "check_output", side_effect=responses):
            with self.assertRaises(RuntimeError):
                controller.gpu_admission(allow_shared_gpu=True, min_free_mib=16000, max_gpu_util=20)

    def test_default_help_and_stdlib_dry_run_never_execute_queue(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "run_twc_v1_k8_icc_aligned.py"
        help_result = subprocess.run([sys.executable, "-B", "-S", str(script)],
            cwd=root, text=True, capture_output=True, check=True)
        self.assertIn("--run", help_result.stdout)
        self.assertIn("--allow-shared-gpu", help_result.stdout)
        plan_result = subprocess.run([sys.executable, "-B", "-S", str(script), "--dry-run",
            "--allow-shared-gpu", "--max-gpu-util", "75", "--min-free-mib", "16000"],
            cwd=root, text=True, capture_output=True, check=True)
        plan = json.loads(plan_result.stdout)
        self.assertFalse(plan["writes"])
        self.assertFalse(plan["weights_read"])
        self.assertEqual(plan["startup_policy"], dict(allow_shared_gpu=True, max_gpu_util=75, min_free_mib=16000))
        self.assertEqual(plan["codec_updates_each"], 20000)
        self.assertEqual(plan["CE_updates"], 5000)
        self.assertEqual(plan["microbatch"] * plan["accumulation"], 32)


if __name__ == "__main__":
    unittest.main()
