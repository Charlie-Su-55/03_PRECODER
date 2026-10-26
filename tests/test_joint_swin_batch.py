"""Lightweight orchestration contracts, without models, channels or GPU work."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli_contracts import assert_lightweight_cli
import run_joint_swin_batch as batch
import preflight_joint_swin as preflight


class BatchContracts(unittest.TestCase):
    def test_standard_library_plans_and_help(self):
        assert_lightweight_cli(self, "run_joint_swin_batch", [["--dry-run"]],
            "assert len(plans[0]['training']) == 4\nassert not plans[0]['old_controls_retrained']\n"
            "assert plans[0]['criteria']['reference'] == 'same-seed plain_swin256'")
        assert_lightweight_cli(self, "preflight_joint_swin", [["--dry-run"]],
            "assert plans[0]['batch_size'] == 48\nassert plans[0]['no_microbatch_or_automatic_structure_change']")

    def test_new_required_execution_paths(self):
        full = set(preflight.required_modules("joint_swin256"))
        plain = set(preflight.required_modules("plain_swin256"))
        self.assertTrue(plain < full)
        for layer in range(2):
            for part in ("self_attention", "cross_attention", "mlp"):
                self.assertIn(f"decoder.joint_layers.{layer}.{part}", full)
        self.assertIn("decoder.spatial_attention", full)
        self.assertIn("decoder.residual_head", full)
        self.assertEqual(sum(name.startswith("decoder.codec.blocks.") for name in full), 4)

    def test_preflight_does_not_advertise_parameter_counts_as_measured(self):
        plan = preflight.plan()
        self.assertIn("static_parameter_expectation", plan)
        plain, joint = [plan["static_parameter_expectation"][a] for a in batch.ARMS]
        self.assertEqual(plain["full"], plain["ue_and_condition"] + plain["bs"])
        self.assertEqual(joint["full"] - plain["full"], 510094)
        self.assertEqual(plain["ue_and_condition"], joint["ue_and_condition"])

    def test_smoke_is_included_then_resumes_not_a_second_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            from types import SimpleNamespace
            cfg = SimpleNamespace(SAVE_PATH=str(root / "ckpt/latest.pth"), BEST_PATH=str(root / "ckpt/best.pth"),
                SAVE_DIR=str(root / "ckpt"), LOG_DIR=str(root / "logs"),
                STATUS_PATH=str(root / "status.json"), CONFIG_PATH=str(root / "config.json"))
            commands = []
            def child(folder, label, command):
                commands.append(command)
                return str(root / (label + ".log"))
            with patch.object(batch, "joint_swin_config", return_value=cfg), \
                    patch.object(batch, "_child", side_effect=child), \
                    patch.object(batch, "read_json", return_value={"state": "paused"}), \
                    patch.object(batch, "_stage_child", return_value={"report": {}, "_evidence": []}):
                batch._train_one(root, "joint_swin256", 20261030)
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][-2:], ["--stop-after", "500"])
            self.assertEqual(commands[1][-2:], ["--resume", "--train"])
            self.assertTrue(all("train_proposal_joint_swin.py" in command for command in commands))

    def test_output_and_unknown_directory_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            from types import SimpleNamespace
            cfg = SimpleNamespace(SAVE_PATH=str(root / "latest.pth"), SAVE_DIR=str(root), LOG_DIR=str(root / "logs"))
            with patch.object(batch, "joint_swin_config", return_value=cfg), patch.object(batch, "_child") as child:
                with self.assertRaises(RuntimeError):
                    batch._train_one(root, "joint_swin256", 20261030)
                child.assert_not_called()

    def test_frozen_inputs_reject_missing_or_modified_data(self):
        with self.assertRaises(ValueError):
            batch._assert_frozen({})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "sidecar.json"
            path.write_text("{}", encoding="utf-8")
            with patch.object(batch, "ROOT", root):
                batch._assert_frozen({str(path): batch.sha256(path)})
                with self.assertRaises(RuntimeError):
                    batch._assert_frozen({str(path): "0" * 64})

    def test_cli_requires_explicit_execution(self):
        for args in (["--resume"], ["--_stage", "training_record"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                batch.main(args)
            self.assertEqual(error.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            preflight.main(["--run"])

    def test_existing_completed_stage_is_hash_verified_without_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            evidence.write_text("{}", encoding="utf-8")
            summary = {"stages": {"completed": {"status": "completed", "result": {"ok": True},
                "evidence_sha256": {str(evidence): batch.sha256(evidence)}}}}
            with patch.object(batch, "_child") as child:
                value = batch._run_stage(summary, root, "completed", lambda: self.fail("Repeated work"))
                child.assert_not_called()
            self.assertTrue(value["ok"])
            evidence.write_text("changed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                batch._run_stage(summary, root, "completed", lambda: self.fail("Repeated work"))

    def test_failed_cuda_report_survives_child_exception_for_small_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def child(folder, label, command, **kwargs):
                if label == "new_structural_tests":
                    return str(root / "tests.log")
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir()
                (output / "report.json").write_text(json.dumps({"passed": False, "error": "synthetic OOM"}), encoding="utf-8")
                raise RuntimeError("child failed")
            with patch.object(batch, "_child", side_effect=child):
                with self.assertRaisesRegex(RuntimeError, "child failed"):
                    batch._preflight(root)
            reports = [p for p in root.iterdir() if p.is_file() and p.suffix == ".json"]
            self.assertEqual(len(reports), 1)
            self.assertEqual(json.loads(reports[0].read_text(encoding="utf-8")),
                             {"passed": False, "error": "synthetic OOM"})


if __name__ == "__main__":
    unittest.main()
