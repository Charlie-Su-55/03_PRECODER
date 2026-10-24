"""Standard-library preflight checks; no weights, model imports, or CUDA work.

Run from 03_PRECODER:
    python -X utf8 -B -m unittest discover -s tests -p test_preflight_proposal_as_direct.py -v

Only the archived JSON manifest and arbitrary temporary byte fixtures are read.
The actual preflight is never invoked. All test writes stay in temporary folders.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import preflight_proposal_as_direct as preflight


ARCHIVE = ROOT / (
    "paper/data/diagnostics/k4_low_feedback_snr/"
    "validation256_cudnn_deterministic_20260930T165938_038117Z/"
    "diagnostic_manifest.json"
)
SOURCE_SHA256 = "e21ea1feae595e0898b6be724f7062b05c09b2d1ef03fe9ffec7ac91869064bf"
FULL_GRID = [[64, 0], [48, 64], [32, 128], [16, 192], [0, 256]]


def archive_manifest():
    return json.loads(ARCHIVE.read_text(encoding="utf-8"))


def archive_source():
    return next(record for record in archive_manifest()["methods"]
                if record["method_key"] == "robust_specialist")


def archived_config():
    return archive_source()["loaded_checkpoint_config"]


def checkpoint_metadata():
    """Metadata only: no tensors, state dict, or real checkpoint deserialization."""
    return {
        "config": archived_config(),
        "step": 20000,
        "allocation_grid": [(0, 256)],
    }


class SourceSelectionTests(unittest.TestCase):
    def load_fixture(self, manifest):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return preflight.load_source_record(path)

    def test_default_archive_and_exact_robust_source(self):
        self.assertEqual(Path(preflight.DEFAULT_MANIFEST).resolve(), ARCHIVE.resolve())
        source = preflight.load_source_record()
        expected = archive_source()
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(source[key], value)
        self.assertEqual(source["checkpoint_sha256"], SOURCE_SHA256)
        self.assertEqual(source["checkpoint_step"], 20000)
        self.assertEqual(source["cfg_index"], 0)
        self.assertEqual(source["allocations"], [[0, 256]])

    def test_missing_or_ambiguous_robust_source_is_rejected(self):
        manifest = archive_manifest()
        without = [record for record in manifest["methods"]
                   if record["method_key"] != "robust_specialist"]
        for records in (without, manifest["methods"] + [archive_source()]):
            with self.subTest(count=len(records)):
                manifest["methods"] = records
                with self.assertRaises(ValueError):
                    self.load_fixture(manifest)

    def test_wrong_source_identity_status_allocation_and_step_are_rejected(self):
        changes = {
            "model_key": "swin",
            "method_type": "specialist",
            "training_status": "running",
            "training_mode": "fixed_allocation_base_specialist",
            "allocations": [[64, 0]],
            "cfg_index": 4,
            "checkpoint_step": 19999,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                source = archive_source()
                source[field] = value
                with self.assertRaises(ValueError):
                    self.load_fixture({"methods": [source]})

    def test_missing_and_malformed_source_hashes_are_rejected(self):
        for value in (None, "", "a" * 63, "a" * 65, "g" * 64):
            with self.subTest(hash=value):
                source = archive_source()
                source["checkpoint_sha256"] = value
                with self.assertRaises(ValueError):
                    self.load_fixture({"methods": [source]})
        source = archive_source()
        del source["checkpoint_sha256"]
        with self.assertRaises(ValueError):
            self.load_fixture({"methods": [source]})

    def test_missing_or_invalid_saved_config_is_rejected(self):
        for value in (None, {}, []):
            with self.subTest(config=value):
                source = archive_source()
                source["loaded_checkpoint_config"] = value
                with self.assertRaises(ValueError):
                    self.load_fixture({"methods": [source]})
        source = archive_source()
        del source["loaded_checkpoint_config"]
        with self.assertRaises(ValueError):
            self.load_fixture({"methods": [source]})

    def test_well_formed_but_different_source_hash_is_rejected(self):
        self.assertEqual(preflight.EXPECTED_SOURCE_SHA256, SOURCE_SHA256)
        source = archive_source()
        source["checkpoint_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "(?i)(sha|hash)"):
            self.load_fixture({"methods": [source]})


class ConfigurationTests(unittest.TestCase):
    def test_archived_full_config_is_preserved_and_tuple_lists_are_normalized(self):
        expected = archived_config()
        config = deepcopy(expected)
        config["allocations"] = ((0, 256),)
        config["experiment"]["allocation"] = (0, 256)
        config["full_allocation_grid"] = tuple(map(tuple, FULL_GRID))
        config["training_recipe"]["stage1_snr"] = (0.0, 25.0)
        normalized = preflight.validate_source_config(config)
        self.assertEqual(normalized, expected)
        self.assertEqual(config["allocations"], ((0, 256),))

    def test_runtime_config_keeps_singleton_and_full_architecture_widths(self):
        cfg = preflight.runtime_config(archived_config())
        self.assertEqual((cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS,
                          cfg.D_TOT, cfg.NUM_SUBBANDS), (4, 32, 256, 256, 4))
        self.assertEqual(cfg.GAMMA_GRID, [(0, 256)])
        self.assertEqual((cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED), (64, 256))
        model_fields = ("D_MODEL", "NUM_HEADS", "NUM_ENCODER_LAYERS", "DROPOUT",
                        "NUM_UNFOLD_LAYERS", "GNN_AGG_DIM")
        for field in model_fields:
            with self.subTest(field=field):
                self.assertEqual(getattr(cfg, field), archived_config()["proposal_model"][field])

    def test_wrong_scenario_dimensions_are_rejected(self):
        changes = (
            ("experiment", "k_users", 2),
            ("experiment", "subcarriers", 128),
            ("experiment", "feedback_budget", 128),
            ("physical", "ANTENNAS", 16),
            ("physical", "NUM_SUBBANDS", 8),
        )
        for section, field, value in changes:
            with self.subTest(field=field):
                config = archived_config()
                config[section][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    preflight.validate_source_config(config)

    def test_wrong_allocation_or_width_grid_is_rejected(self):
        changes = (
            ("allocations", FULL_GRID),
            ("allocations", [[0, 128]]),
            ("full_allocation_grid", [[0, 256]]),
            ("full_allocation_grid", FULL_GRID[:-1]),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value):
                config = archived_config()
                config[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    preflight.validate_source_config(config)
        for field, value in (("d_f_max_per_user", 0), ("d_a_max_shared", 128)):
            with self.subTest(field=field):
                config = archived_config()
                config["resolved"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    preflight.validate_source_config(config)

    def test_changed_full_model_capacity_is_rejected(self):
        for field, value in (("D_MODEL", 128), ("NUM_HEADS", 4),
                             ("NUM_ENCODER_LAYERS", 2), ("GNN_AGG_DIM", 24),
                             ("NUM_UNFOLD_LAYERS", 3)):
            with self.subTest(field=field):
                config = archived_config()
                config["proposal_model"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    preflight.validate_source_config(config)

    def test_non_full_model_identity_and_ffn_are_rejected(self):
        changes = (
            ("experiment", "model_name", "proposal_lite"),
            ("experiment", "variant", "ueffn1024"),
            ("proposal_model", "UE_FFN_DIM", 1024),
        )
        for section, field, value in changes:
            with self.subTest(field=field):
                config = archived_config()
                config[section][field] = value
                with self.assertRaises(ValueError):
                    preflight.validate_source_config(config)
        config = archived_config()
        config["architecture_metadata"] = {
            "model_class": "AdaptiveHybridPrecoderLite",
            "architecture": "ueffn1024", "ue_ffn_dim": 1024,
        }
        with self.assertRaises(ValueError):
            preflight.validate_source_config(config)


class CheckpointMetadataTests(unittest.TestCase):
    def test_sha256_verifies_arbitrary_bytes_without_deserialization(self):
        content = b"arbitrary test bytes; this is not a serialized checkpoint\x00\xff"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arbitrary-bytes.dat"
            path.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(preflight.verify_checkpoint_hash(path, expected), expected)
            with self.assertRaisesRegex(ValueError, "(?i)(sha|hash)"):
                preflight.verify_checkpoint_hash(path, "0" * 64)

    def test_metadata_accepts_json_equivalent_tuple_lists(self):
        payload = checkpoint_metadata()
        payload["config"]["allocations"] = ((0, 256),)
        payload["config"]["full_allocation_grid"] = tuple(map(tuple, FULL_GRID))
        payload["config"]["experiment"]["allocation"] = (0, 256)
        payload["config"]["training_recipe"]["stage2_snr"] = (0.0, 25.0)
        normalized = preflight.validate_checkpoint_metadata(payload, archive_source())
        self.assertEqual(normalized, archived_config())

    def test_sha256_consumes_existing_stream_without_reopening_or_closing_it(self):
        content = b"temporary arbitrary bytes for stream hashing\x00\xff"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arbitrary-bytes.dat"
            missing = Path(temporary) / "must-not-be-opened.dat"
            path.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            with path.open("rb") as stream:
                self.assertEqual(stream.tell(), 0)
                self.assertEqual(preflight.verify_checkpoint_hash(
                    missing, expected, stream=stream), expected)
                self.assertFalse(stream.closed)
                self.assertEqual(stream.tell(), len(content))
                stream.seek(0)
                self.assertEqual(stream.read(), content)
            self.assertFalse(missing.exists())

    def test_sha256_mismatch_does_not_close_callers_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arbitrary-bytes.dat"
            path.write_bytes(b"not a checkpoint")
            with path.open("rb") as stream:
                with self.assertRaisesRegex(ValueError, "(?i)(sha|hash)"):
                    preflight.verify_checkpoint_hash(
                        Path(temporary) / "does-not-exist.dat", "0" * 64, stream=stream)
                self.assertFalse(stream.closed)

    def test_all_saved_config_fields_must_match_and_name_the_mismatch(self):
        changes = (
            ("training_recipe", "lr", 1e-4),
            ("resolved", "save_dir", "/different/recorded/training/path"),
            ("baseline_model", "SWIN_NC", 64),
            ("experiment", "seed", 43),
        )
        for section, field, value in changes:
            with self.subTest(field=field):
                payload = checkpoint_metadata()
                payload["config"][section][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    preflight.validate_checkpoint_metadata(payload, archive_source())

    def test_missing_and_extra_saved_config_fields_are_rejected(self):
        for extra in (False, True):
            with self.subTest(extra=extra):
                payload = checkpoint_metadata()
                if extra:
                    payload["config"]["unexpected_field"] = "must not be ignored"
                else:
                    del payload["config"]["baseline_model"]
                with self.assertRaises(ValueError):
                    preflight.validate_checkpoint_metadata(payload, archive_source())

    def test_saved_step_and_singleton_allocation_grid_are_required(self):
        for field, value in (("step", 19999), ("step", None),
                             ("allocation_grid", FULL_GRID),
                             ("allocation_grid", [[64, 0]]),
                             ("allocation_grid", [])):
            with self.subTest(field=field, value=value):
                payload = checkpoint_metadata()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    preflight.validate_checkpoint_metadata(payload, archive_source())
        for field in ("config", "step", "allocation_grid"):
            with self.subTest(missing=field):
                payload = checkpoint_metadata()
                del payload[field]
                with self.assertRaises(ValueError):
                    preflight.validate_checkpoint_metadata(payload, archive_source())


class OutputSafetyTests(unittest.TestCase):
    def test_output_root_is_dedicated_and_each_default_run_is_new(self):
        expected = ROOT / "runs_as_screen/diagnostics/as_direct_v1"
        self.assertEqual(Path(preflight.OUTPUT_ROOT).resolve(), expected.resolve())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostics"
            first = preflight.create_run_directory(root)
            second = preflight.create_run_directory(root)
            self.assertNotEqual(first, second)
            for path in (first, second):
                self.assertTrue(path.is_dir())
                self.assertEqual(path.parent, root)

    def test_existing_run_is_never_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = preflight.create_run_directory(root, run_id="existing-run_01")
            marker = first / "keep.txt"
            marker.write_text("keep existing output", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                preflight.create_run_directory(root, run_id="existing-run_01")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep existing output")

    def test_run_id_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostics"
            for run_id in ("", ".", "..", "../escape", "a/b", "a\\b", "/absolute",
                           "C:\\absolute", "name with spaces"):
                with self.subTest(run_id=run_id):
                    with self.assertRaises(ValueError):
                        preflight.create_run_directory(root, run_id=run_id)
            self.assertFalse(root.exists())

    def test_report_is_json_and_cannot_overwrite_existing_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report = {"passed": False, "reason": "lightweight test", "checks": []}
            preflight.save_report(folder, report)
            path = folder / "report.json"
            before = path.read_bytes()
            self.assertEqual(json.loads(before), report)
            with self.assertRaises(FileExistsError):
                preflight.save_report(folder, {"passed": True})
            self.assertEqual(path.read_bytes(), before)

    def test_report_sanitizes_nonfinite_values_and_marks_failure(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                report = {"passed": True, "metric": value,
                          "nested": [{"metric": value}], "finite": 1.25}
                preflight.save_report(Path(temporary), report)
                serialized = (Path(temporary) / "report.json").read_text(encoding="utf-8")
                # JSON's default decoder also accepts NaN/Infinity; reject those
                # extensions explicitly to verify the report is portable JSON.
                def reject_constant(token):
                    self.fail("Nonstandard JSON number persisted: " + token)
                saved = json.loads(serialized, parse_constant=reject_constant)
                self.assertFalse(saved["passed"])
                self.assertEqual(saved["status"], "nonfinite_report_metrics")
                self.assertIsNone(saved["metric"])
                self.assertIsNone(saved["nested"][0]["metric"])
                self.assertEqual(saved["finite"], 1.25)
                fields = saved["nonfinite_report_fields"]
                self.assertTrue(any("metric" in field for field in fields))
                self.assertTrue(any("nested" in field for field in fields))
                self.assertFalse(report["passed"])
                self.assertEqual(report["status"], "nonfinite_report_metrics")


class CommandLineTests(unittest.TestCase):
    def test_plan_has_exact_fixed_protocol_and_complete_source_metadata(self):
        plan = preflight.build_plan()
        self.assertEqual(plan["source"]["loaded_checkpoint_config"], archived_config())
        self.assertEqual(plan["source"]["checkpoint_sha256"], SOURCE_SHA256)
        self.assertEqual(Path(plan["manifest_path"]).resolve(), ARCHIVE.resolve())
        self.assertEqual(plan["manifest_sha256"], hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())
        expected_protocol = {
            "seed": 20261024, "num_samples": 2, "batch_size": 2,
            "K": 4, "Nt": 32, "Nsc": 256, "Dtot": 256, "Nsb": 4,
            "allocation": [0, 256], "cfg_idx": 0,
            "feedback_snr_db": [0, 10, 25], "downlink_snr_db": 25,
            "rtol": 1e-6, "atol": 1e-6,
        }
        for field, value in expected_protocol.items():
            with self.subTest(field=field):
                self.assertEqual(plan["protocol"][field], value)
        self.assertEqual((preflight.SEED, preflight.RTOL, preflight.ATOL),
                         (20261024, 1e-6, 1e-6))
        self.assertEqual(plan["checkpoint_path"], archive_source()["path"])
        self.assertEqual(Path(plan["output_root"]), preflight.OUTPUT_ROOT)

    def test_checkpoint_override_resolves_from_repo_and_keeps_source_provenance(self):
        plan = preflight.build_plan("nonexistent-weights/best.pth")
        self.assertEqual(Path(plan["checkpoint_path"]), ROOT / "nonexistent-weights/best.pth")
        self.assertEqual(plan["source"]["path"], archive_source()["path"])
        self.assertEqual(plan["source"]["checkpoint_sha256"], SOURCE_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            absolute = Path(temporary) / "missing.pth"
            self.assertEqual(Path(preflight.build_plan(absolute)["checkpoint_path"]), absolute)

    def test_no_arguments_show_help_and_do_not_run(self):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(preflight, "run_preflight") as run:
            self.assertEqual(preflight.main([]), 0)
        run.assert_not_called()
        self.assertIn("--dry-run", output.getvalue())
        self.assertIn("--run", output.getvalue())

    def test_explicit_help_exits_successfully_without_running(self):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(preflight, "run_preflight") as run:
            with self.assertRaises(SystemExit) as raised:
                preflight.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        run.assert_not_called()
        self.assertIn("--checkpoint", output.getvalue())

    def test_parser_requires_mutually_exclusive_modes(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            preflight.build_parser().parse_args(["--run", "--dry-run"])
        self.assertEqual(raised.exception.code, 2)

    def test_protocol_cannot_be_overridden_from_cli(self):
        for option, value in (("--seed", "7"), ("--manifest", "other.json"),
                              ("--output-root", "elsewhere"), ("--batch-size", "4")):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    preflight.build_parser().parse_args(["--dry-run", option, value])
                self.assertEqual(raised.exception.code, 2)

    def test_import_help_and_dry_run_never_import_heavy_modules_read_weights_or_write(self):
        # A fresh -S process prevents installed packages/site hooks from hiding an
        # accidental heavy import. The audit hook denies all write opens/mkdirs
        # and every checkpoint read, independently of current repository state.
        script = r'''
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
real_import = builtins.__import__
forbidden = {"torch", "numpy", "sionna", "tensorflow", "models", "utils",
             "main_train", "baseline_evaluate", "train_proposal_snr_robust"}

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError("Heavy import attempted: " + name)
    return real_import(name, *args, **kwargs)

def audit(event, args):
    if event == "os.mkdir":
        raise AssertionError("Directory creation attempted: " + str(args[0]))
    if event == "open":
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError("File write attempted: " + str(path))
        if isinstance(path, (str, bytes)) and os.fsdecode(path).lower().endswith(
                (".pth", ".pt", ".ckpt", ".safetensors")):
            raise AssertionError("Checkpoint read attempted: " + os.fsdecode(path))

builtins.__import__ = guarded_import
sys.addaudithook(audit)
import preflight_proposal_as_direct as preflight

help_output = io.StringIO()
with contextlib.redirect_stdout(help_output):
    assert preflight.main([]) == 0
assert "--run" in help_output.getvalue()

dry_output = io.StringIO()
with contextlib.redirect_stdout(dry_output):
    assert preflight.main(["--dry-run", "--checkpoint", "missing/no-load.pth"]) == 0
plan = json.loads(dry_output.getvalue())
assert Path(plan["checkpoint_path"]) == root / "missing/no-load.pth"
assert not forbidden.intersection(sys.modules)
print("import/help/dry-run stayed read-only and lightweight")
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-B", "-S", "-c", script, str(ROOT)],
                cwd=temporary, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("stayed read-only and lightweight", result.stdout)


if __name__ == "__main__":
    unittest.main()
