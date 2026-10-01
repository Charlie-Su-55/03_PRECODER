"""Lightweight training-entry checks; no training, evaluation, or GPU preflight.

Only scalar CPU fixtures exercise checkpoint utilities. Full Lite numerical
coverage remains in test_adaptive_hybrid_lite.py, whose 16 tests are unchanged.
All filesystem fixtures live in TemporaryDirectory, never in existing runs.
"""

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.train_config import ExperimentSpec, TRAINING_RECIPES, resolve_experiment
from configs.lite_train_config import LiteTrainConfig, lite_config, validate_lite_recipe

RUNTIME_ERROR = None
try:
    import numpy as np
    import torch
    import main_train
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "torch", "tqdm", "tqdm.auto"}:
        raise
    RUNTIME_ERROR = f"Missing {exc.name!r}; scalar CPU trainer-utility checks were not run."


# Captured from unmodified commit 873cd6479b546f3293f5fc9936ca6ba96c207515.
OLD_CONFIG_LF_SHA256 = "066b489e7027ad0d7c0ebe32b94b64c8f7c7444bd904c415a3faf8109845f72c"
OLD_SNAPSHOT_SHA256 = {
    "nt": "bf6a5ddd9cd19e4a1bd27bb67c1a5f4a199377209da3f1407436839ee8f689f6",
    "posix": "e39650e75ad43b4ddc1eac4410abb1de91a72b9c2995a2eb330ff7ab8cb9c514",
}
OLD_TRAINER_AST_SHA256 = "49e3b4a1184e5ee3a8289beb90f3e0e58f405401363a8b9538cc82149ecfe165"
OLD_HELPER_AST_SHA256 = {
    "validate_proposal": "7fbf9b8a9de2a67fdbe5d676ee76dacfc9a63750cdb6af583abcd28521f8a5c1",
    "build_proposal_model": "80439ab93ebf12583c49bc3427c840dbeac105098b2c196132cc12390cfb473d",
    "generate_training_batch": "920f1cd36350f98285f1554ea58c2cd713ddc51d868a320e3f199d3e1b909803",
    "checkpoint_payload": "d346516286b8d53733d26941659c4df30be949f5a35d5142d9eab58c8e765e59",
    "load_latest_checkpoint": "a6937d1ac296d2aed65a8f19451f2cb32ef313bb94300150633761c78292933f",
}


def trainer_functions():
    tree = ast.parse((ROOT / "main_train.py").read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def ast_digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()


def old_config():
    return resolve_experiment(ExperimentSpec(
        "proposal", "specialist", 4, 256, 256, "k4_n256_standard", allocation=(0, 256),
    ))


class LiteConfigTests(unittest.TestCase):
    def test_legacy_recipe_suite_and_serialization_source_is_unchanged(self):
        source = (ROOT / "configs/train_config.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(source).hexdigest(), OLD_CONFIG_LF_SHA256)

    def test_legacy_snapshot_has_the_recorded_platform_fingerprint(self):
        snapshot = old_config().to_dict()
        self.assertNotIn("architecture_metadata", snapshot)
        serialized = json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")
        expected = OLD_SNAPSHOT_SHA256[os.name]
        self.assertEqual(hashlib.sha256(serialized).hexdigest(), expected)
        if RUNTIME_ERROR is None:
            self.assertEqual(main_train.config_fingerprint(old_config()), expected[:16])

    def test_exact_original_recipe_is_reused_with_all_required_settings(self):
        cfg = lite_config()
        self.assertIsInstance(cfg, LiteTrainConfig)
        self.assertIs(cfg.recipe, TRAINING_RECIPES["k4_n256_standard"])
        expected = {
            "TOTAL_STEPS": 50_000, "BATCH_SIZE": 48, "LR": 1e-4,
            "WEIGHT_DECAY": 1e-4, "WARMUP_PCT": 0.35, "GRAD_CLIP": 1.0,
            "VAL_INTERVAL": 500, "VAL_BATCH": 32, "VAL_SNR_LIST": [0, 10, 20, 25],
            "BEST_METRIC": "mean_all", "PHASE1_STEPS": 20_000,
            "PH1_W_DIR": 50.0, "PH1_W_MSE": 100.0, "PH2_W_DIR": 5.0, "PH2_W_MSE": 0.0,
            "STAGE1_STEPS": 15_000, "STAGE2_STEPS": 35_000,
            "STAGE1_SNR": (20.0, 25.0), "STAGE2_SNR": (10.0, 25.0),
            "STAGE3_SNR": (0.0, 25.0), "DOWNLINK_SNR_DB": None,
        }
        for key, value in expected.items():
            with self.subTest(setting=key):
                self.assertEqual(getattr(cfg, key), value)

    def test_singleton_keeps_full_width_and_only_running_power_index_zero(self):
        cfg = lite_config()
        self.assertEqual(cfg.GAMMA_GRID, [(0, 256)])
        self.assertEqual(cfg.ALLOCATION_GRID, [(0, 256)])
        self.assertEqual(cfg.full_allocation_grid,
                         ((64, 0), (48, 64), (32, 128), (16, 192), (0, 256)))
        self.assertEqual((cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED), (64, 256))
        self.assertEqual((cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS, cfg.NUM_SUBBANDS),
                         (4, 32, 256, 4))
        if RUNTIME_ERROR is None:
            for step in (1, 500, 15_000, 20_000, 35_000, 50_000):
                self.assertEqual(main_train.allocation_index_for_step(step, len(cfg.GAMMA_GRID), cfg.SEED), 0)

    def test_identity_metadata_and_snapshot_retain_complete_base_configuration(self):
        cfg = lite_config()
        expected = {"model_class": "AdaptiveHybridPrecoderLite", "architecture": "ueffn1024",
                    "ue_ffn_dim": 1024, "ue_layers": {"fdma": 4, "as": 4}, "D_MODEL": 256}
        self.assertEqual(cfg.architecture_metadata, expected)
        snapshot = cfg.to_dict()
        self.assertEqual(snapshot["architecture_metadata"], expected)
        self.assertEqual(snapshot["allocations"], [[0, 256]])
        self.assertEqual(snapshot["resolved"]["d_f_max_per_user"], 64)
        self.assertEqual(snapshot["resolved"]["d_a_max_shared"], 256)
        for token in ("K4", "Nsc256", "Dfb256", "df0_da256", "k4_n256_standard", "ueffn1024", "seed42"):
            self.assertIn(token, cfg.EXP_NAME)
        base = resolve_experiment(cfg.spec, output_root=cfg.output_root).to_dict()
        self.assertEqual({key: value for key, value in snapshot.items() if key != "architecture_metadata"}, base)

    def test_config_construction_does_not_create_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            requested = Path(temporary) / "proposal_lite"
            cfg = lite_config(requested)
            self.assertEqual(cfg.output_root, requested.resolve())
            self.assertFalse(requested.exists())

    def test_recipe_guard_rejects_width_class_recipe_and_capacity_changes(self):
        cfg = lite_config()
        bad_configs = (
            replace(cfg, ue_ffn_dim=2048),
            replace(cfg, ue_ffn_dim=1024.0),
            replace(cfg, ue_ffn_dim=True),
            replace(cfg, ue_ffn_dim="1024"),
            replace(cfg, model_class="AdaptiveHybridPrecoder"),
            replace(cfg, architecture="original"),
            replace(cfg, recipe=replace(cfg.recipe, val_batch=64)),
            replace(cfg, full_allocation_grid=((0, 256),)),
            replace(cfg, spec=replace(cfg.spec, seed=43)),
        )
        for candidate in bad_configs:
            with self.subTest(candidate=candidate.to_dict()):
                with self.assertRaises(ValueError):
                    validate_lite_recipe(candidate)


class TrainerFactorySourceTests(unittest.TestCase):
    def test_model_factory_is_keyword_only_and_defaults_to_none(self):
        function = trainer_functions()["train_proposal"]
        self.assertEqual([arg.arg for arg in function.args.kwonlyargs], ["model_factory"])
        self.assertEqual([ast.literal_eval(value) for value in function.args.kw_defaults], [None])

    def test_default_and_explicit_factories_are_selected_without_running_trainer(self):
        function = trainer_functions()["train_proposal"]
        statement = next(node for node in function.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "model"
                                 for target in node.targets))
        # Execute only the actual source's construction expression. This does not
        # patch a model import, enter the trainer, or construct an optimizer.
        code = compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(statement)], type_ignores=[])),
                       "<isolated trainer model-factory selection>", "exec")
        cfg, device, original_result, lite_result = object(), object(), object(), object()
        for explicit in (False, True):
            original_factory = mock.Mock(return_value=original_result)
            lite_factory = mock.Mock(return_value=lite_result)
            namespace = {"cfg": cfg, "device": device, "build_proposal_model": original_factory,
                         "model_factory": lite_factory if explicit else None}
            exec(code, namespace)
            chosen, unused = (lite_factory, original_factory) if explicit else (original_factory, lite_factory)
            chosen.assert_called_once_with(cfg, device)
            unused.assert_not_called()
            self.assertIs(namespace["model"], lite_result if explicit else original_result)

    def test_old_model_builder_checkpoint_and_data_helpers_are_unchanged(self):
        functions = trainer_functions()
        for name, expected in OLD_HELPER_AST_SHA256.items():
            with self.subTest(helper=name):
                self.assertEqual(ast_digest(functions[name]), expected)

    def test_training_algorithm_is_unchanged_beyond_the_optional_factory(self):
        function = deepcopy(trainer_functions()["train_proposal"])
        function.args.kwonlyargs = []
        function.args.kw_defaults = []
        indices = [index for index, node in enumerate(function.body) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "model" for target in node.targets)]
        self.assertEqual(len(indices), 1)
        function.body[indices[0]] = ast.parse("model = build_proposal_model(cfg, device)").body[0]
        self.assertEqual(ast_digest(function), OLD_TRAINER_AST_SHA256)

    def test_pause_keeps_full_scheduler_and_original_stage_boundaries(self):
        function = trainer_functions()["train_proposal"]
        schedulers = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "OneCycleLR"]
        self.assertEqual(len(schedulers), 1)
        kwargs = {item.arg: item.value for item in schedulers[0].keywords}
        self.assertEqual(ast.unparse(kwargs["total_steps"]), "cfg.TOTAL_STEPS")
        cfg = lite_config()
        self.assertEqual((cfg.TOTAL_STEPS, cfg.PHASE1_STEPS, cfg.STAGE1_STEPS, cfg.STAGE2_STEPS),
                         (50_000, 20_000, 15_000, 35_000))


def entry_module():
    import train_proposal_lite
    return train_proposal_lite


def make_run_files(cfg, *, state="paused"):
    """Only temporary-directory fixtures; placeholder .pth files are not loaded."""
    Path(cfg.SAVE_DIR).mkdir(parents=True)
    Path(cfg.LOG_DIR).mkdir(parents=True)
    snapshot = cfg.to_dict()
    snapshot["config_fingerprint"] = entry_module().lite_fingerprint(cfg)
    Path(cfg.CONFIG_PATH).write_text(json.dumps(snapshot), encoding="utf-8")
    Path(cfg.STATUS_PATH).write_text(json.dumps({
        "state": state, "step": 500, "experiment": cfg.EXP_NAME,
        "fingerprint": entry_module().lite_fingerprint(cfg),
    }), encoding="utf-8")
    Path(cfg.SAVE_PATH).write_bytes(b"temporary checkpoint placeholder")
    Path(cfg.BEST_PATH).write_bytes(b"temporary checkpoint placeholder")
    Path(cfg.METRICS_PATH).write_text("{}\n", encoding="utf-8")


class LitePreflightSourceTests(unittest.TestCase):
    """Source-only boundaries: these tests never invoke gpu_preflight."""

    def preflight_function(self):
        tree = ast.parse((ROOT / "train_proposal_lite.py").read_text(encoding="utf-8"))
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "gpu_preflight")

    def test_diagnostic_full_grid_is_a_separate_config_not_a_formal_mutation(self):
        function = self.preflight_function()
        assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "diagnostic_cfg"
                                  for target in node.targets))
        self.assertIsInstance(assignment.value, ast.Call)
        self.assertEqual(ast.unparse(assignment.value.func), "replace")
        self.assertEqual(ast.unparse(assignment.value.args[0]), "cfg")
        keywords = {keyword.arg: keyword.value for keyword in assignment.value.keywords}
        self.assertEqual(ast.unparse(keywords["allocations"]), "cfg.full_allocation_grid")
        # Execute only the dataclass replacement expression, never model/CUDA code.
        cfg = lite_config()
        before = deepcopy(cfg.to_dict())
        namespace = {"replace": replace, "cfg": cfg}
        code = compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(assignment)], type_ignores=[])),
                       "<isolated preflight diagnostic config>", "exec")
        exec(code, namespace)
        diagnostic = namespace["diagnostic_cfg"]
        self.assertIsNot(diagnostic, cfg)
        self.assertEqual(diagnostic.GAMMA_GRID, list(cfg.full_allocation_grid))
        self.assertEqual(diagnostic.TRAIN_MODE, "adaptive")
        self.assertIsNone(diagnostic.spec.allocation)
        self.assertEqual(cfg.GAMMA_GRID, [(0, 256)])
        self.assertEqual(cfg.to_dict(), before)
        self.assertIs(diagnostic.recipe, cfg.recipe)
        self.assertEqual((diagnostic.D_F_MAX_PER_USER, diagnostic.D_A_MAX_SHARED), (64, 256))
        self.assertFalse([
            node for node in ast.walk(function) if isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, ast.Store) and ast.unparse(node).startswith("cfg.")
        ])

    def test_preflight_has_no_optimizer_steps_training_or_checkpoint_io(self):
        function = self.preflight_function()
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        forbidden_names = {
            "train_proposal", "checkpoint_payload", "load_latest_checkpoint",
            "atomic_torch_save", "atomic_write_json", "append_jsonl",
            "build_channel_generator", "generate_training_batch", "generate_validation_set",
            "validate_proposal", "_write_provenance", "_run_lock", "open",
        }
        forbidden_attributes = {
            "step", "AdamW", "OneCycleLR", "save", "load", "load_state_dict",
            "mkdir", "write_text", "write_bytes", "open", "touch",
        }
        forbidden = [
            ast.unparse(node.func) for node in calls
            if (isinstance(node.func, ast.Name) and node.func.id in forbidden_names)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attributes)
        ]
        self.assertEqual(forbidden, [])
        # CUDA is required before the diagnostic model is constructed.
        guard = next(node for node in function.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "not torch.cuda.is_available()")
        self.assertTrue(any(isinstance(node, ast.Raise) for node in guard.body))
        self.assertTrue(any(ast.unparse(node.func) == "loss.backward" for node in calls))

    def test_preflight_task_loss_uses_rzf_hhat_and_per_subcarrier_power(self):
        function = self.preflight_function()
        loop = next(node for node in ast.walk(function) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "allocation")
        self.assertEqual(ast.literal_eval(loop.iter), ((64, 0), (32, 128), (0, 256)))
        assignments = [node for node in loop.body if isinstance(node, ast.Assign)]
        extracted = next(node for node in assignments
                         if ast.unparse(node.targets[0]) == "(legacy_w, h_hat)")
        self.assertEqual(ast.unparse(extracted.value.func), "extract_training_outputs")
        precoders = [node for node in assignments if ast.unparse(node.targets[0]) == "w"]
        self.assertEqual(len(precoders), 2)
        self.assertEqual(ast.unparse(precoders[0].value), "model.zf_solver(h_hat)")
        expected_normalization = ast.parse(
            "w * torch.sqrt(cfg.TOTAL_POWER / "
            "(w.abs().square().sum(dim=(1, 2), keepdim=True) + 1e-9))", mode="eval"
        ).body
        self.assertEqual(ast.dump(precoders[1].value, include_attributes=False),
                         ast.dump(expected_normalization, include_attributes=False))
        criterion_calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name) and node.func.id == "criterion"]
        self.assertEqual(len(criterion_calls), 1)
        self.assertEqual([ast.unparse(arg) for arg in criterion_calls[0].args[:2]], ["w", "h_dl"])
        # The legacy output is only a checked interface, not an optimization input.
        uses_of_legacy = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
                          and any(isinstance(arg, ast.Name) and arg.id == "legacy_w" for arg in node.args)]
        self.assertEqual([ast.unparse(node.func) for node in uses_of_legacy], ["check_precoder"])


class LiteEntrySafetyTests(unittest.TestCase):
    def test_entry_fingerprint_matches_legacy_algorithm_and_includes_architecture(self):
        entry = entry_module()
        cfg = lite_config()
        baseline = entry.lite_fingerprint(cfg)
        if RUNTIME_ERROR is None:
            self.assertEqual(baseline, main_train.config_fingerprint(cfg))
        for changed in (replace(cfg, ue_ffn_dim=2048),
                        replace(cfg, model_class="AdaptiveHybridPrecoder"),
                        replace(cfg, architecture="other")):
            with self.subTest(metadata=changed.architecture_metadata):
                self.assertNotEqual(entry.lite_fingerprint(changed), baseline)

    def test_cli_dry_run_and_no_arguments_do_not_import_heavy_modules_or_write_runs(self):
        # A fresh process makes the no-heavy-import requirement observable even
        # though other tests in this process legitimately import torch utilities.
        script = (
            "import contextlib,io,json,sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "import train_proposal_lite as entry\n"
            "cfg=entry.lite_config()\n"
            "from pathlib import Path\n"
            "paths=[Path(cfg.SAVE_DIR),Path(cfg.LOG_DIR),Path(cfg.output_root)]\n"
            "before=[p.exists() for p in paths]\n"
            "stream=io.StringIO()\n"
            "with contextlib.redirect_stdout(stream):\n"
            "    assert entry.main([])==0\n"
            "    assert entry.main(['--dry-run'])==0\n"
            "assert [p.exists() for p in paths]==before\n"
            "blocked=('torch','models','utils','sionna','tensorflow')\n"
            "assert not [m for m in sys.modules if m.split('.')[0] in blocked]\n"
            "output=stream.getvalue()\n"
            "assert 'AdaptiveHybridPrecoderLite' in output\n"
            "assert 'ueffn1024' in output\n"
            "assert '50000' in output\n"
            "print(json.dumps({'no_heavy_imports':True,'no_directory_creation':True}))\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script],
                                    cwd=temporary, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["no_heavy_imports"])

    def test_cli_rejects_conflicting_modes_and_resume_without_explicit_action(self):
        entry = entry_module()
        invalid = (["--dry-run", "--train"], ["--preflight", "--stop-after", "500"],
                   ["--resume"], ["--resume", "--dry-run"], ["--resume", "--preflight"],
                   ["--stop-after", "0"], ["--stop-after", "-1"])
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    entry.main(arguments)
                self.assertEqual(raised.exception.code, 2)
        parsed = entry.build_arg_parser().parse_args(["--stop-after", "500", "--resume"])
        self.assertEqual(parsed.stop_after, 500)
        self.assertTrue(parsed.resume)

    def test_output_containment_rejects_foreign_and_parent_traversal_paths(self):
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "proposal_lite"
            cfg = lite_config(allowed)
            entry.validate_output_paths(cfg, allowed_root=allowed)
            for outside in (Path(temporary) / "old_runs", allowed / ".." / "old_runs"):
                with self.subTest(path=outside):
                    with self.assertRaises((ValueError, RuntimeError)):
                        entry.validate_output_paths(replace(cfg, output_root=outside), allowed_root=allowed)
            self.assertFalse(allowed.exists())

    def test_symlinked_checkpoint_directory_cannot_escape_the_lite_root(self):
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "proposal_lite"
            outside = Path(temporary) / "old_runs"
            allowed.mkdir()
            outside.mkdir()
            try:
                (allowed / "checkpoints").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Directory symlink fixture unavailable on this platform: {exc}")
            with self.assertRaises((ValueError, RuntimeError)):
                entry.validate_output_paths(lite_config(allowed), allowed_root=allowed)
            self.assertEqual(list(outside.iterdir()), [])


    def test_mocked_symlink_guard_rejects_the_root_and_its_parent(self):
        # This verifies rejection logic only. It is NOT a real symlink test;
        # the separate platform-dependent symlink test remains unchanged.
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "runs_as_screen" / "proposal_lite"
            cfg = lite_config(allowed)
            original_is_symlink = Path.is_symlink
            for flagged in (allowed, allowed.parent):
                def is_symlink(path, target=flagged):
                    return path.absolute() == target.absolute() or original_is_symlink(path)
                with self.subTest(flagged=flagged), mock.patch.object(
                    Path, "is_symlink", autospec=True, side_effect=is_symlink
                ):
                    with self.assertRaisesRegex(ValueError, "(?i)symlink|junction"):
                        entry.validate_output_paths(cfg, allowed_root=allowed)
            self.assertFalse(allowed.exists())

    def test_mocked_symlink_guard_rejects_an_atomic_checkpoint_temporary_file(self):
        # Simulate just the inspection result; do not mock model dependencies
        # or claim the host allowed creating a real filesystem symlink.
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "proposal_lite"
            cfg = lite_config(allowed)
            latest = Path(cfg.SAVE_PATH)
            flagged = latest.with_suffix(latest.suffix + ".tmp")
            original_is_symlink = Path.is_symlink

            def is_symlink(path):
                return path.absolute() == flagged.absolute() or original_is_symlink(path)

            with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=is_symlink):
                with self.assertRaisesRegex(ValueError, "(?i)symlink|junction"):
                    entry.validate_output_paths(cfg, allowed_root=allowed)
            self.assertFalse(allowed.exists())

    def test_fresh_run_rejects_existing_checkpoint_or_log_directory(self):
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            for index, kind in enumerate(("SAVE_DIR", "LOG_DIR")):
                allowed = Path(temporary) / f"proposal_lite_{index}"
                cfg = lite_config(allowed)
                Path(getattr(cfg, kind)).mkdir(parents=True)
                with self.subTest(directory=kind):
                    with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
                        entry.check_run_state(cfg, resume=False, allowed_root=allowed)

    def test_resume_without_latest_is_an_error_not_a_fresh_run(self):
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "proposal_lite"
            cfg = lite_config(allowed)
            with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
                entry.check_run_state(cfg, resume=True, allowed_root=allowed)
            make_run_files(cfg)
            Path(cfg.SAVE_PATH).unlink()
            with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
                entry.check_run_state(cfg, resume=True, allowed_root=allowed)

    def test_completed_runs_and_mismatched_saved_config_are_rejected(self):
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "proposal_lite"
            cfg = lite_config(allowed)
            make_run_files(cfg, state="completed")
            with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
                entry.check_run_state(cfg, resume=True, allowed_root=allowed)
            Path(cfg.STATUS_PATH).write_text(json.dumps({"state": "paused", "step": 500}), encoding="utf-8")
            bad = json.loads(Path(cfg.CONFIG_PATH).read_text(encoding="utf-8"))
            bad["architecture_metadata"]["ue_ffn_dim"] = 2048
            Path(cfg.CONFIG_PATH).write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                entry.check_run_state(cfg, resume=True, allowed_root=allowed)

    def test_valid_fresh_and_paused_resume_checks_are_read_only(self):
        entry = entry_module()
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "proposal_lite"
            cfg = lite_config(allowed)
            entry.check_run_state(cfg, resume=False, allowed_root=allowed)
            self.assertFalse(allowed.exists())
            make_run_files(cfg)
            before = {p.relative_to(allowed): p.read_bytes() for p in allowed.rglob("*") if p.is_file()}
            entry.check_run_state(cfg, resume=True, allowed_root=allowed)
            after = {p.relative_to(allowed): p.read_bytes() for p in allowed.rglob("*") if p.is_file()}
            self.assertEqual(before, after)


@unittest.skipIf(RUNTIME_ERROR is not None, RUNTIME_ERROR or "")
class LiteScalarCheckpointTests(unittest.TestCase):
    """Exercise real serialization with scalar fixtures, NOT Lite performance."""

    def setUp(self):
        self.previous_rng = main_train.capture_rng_state()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cfg = lite_config(Path(self.temporary.name) / "proposal_lite")

    def tearDown(self):
        main_train.restore_rng_state(self.previous_rng)

    def fixture(self):
        class ScalarCheckpointFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([3.5], dtype=torch.float32))
                self.register_buffer("running_pwr", torch.tensor([2.5], dtype=torch.float32))

        model = ScalarCheckpointFixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.cfg.LR, total_steps=self.cfg.TOTAL_STEPS,
            pct_start=self.cfg.WARMUP_PCT, anneal_strategy="cos",
        )
        # Synthetic saved state at 500: no optimizer.step(), scheduler.step(),
        # model forward, training batch, or training iteration is executed.
        optimizer.state[model.weight] = {
            "step": torch.tensor(500.0), "exp_avg": torch.full_like(model.weight, 0.125),
            "exp_avg_sq": torch.full_like(model.weight, 0.25),
        }
        scheduler.last_epoch = 500
        scheduler._step_count = 501
        return model, optimizer, scheduler

    def payload(self):
        model, optimizer, scheduler = self.fixture()
        return main_train.checkpoint_payload(
            cfg=self.cfg, step=500, model=model, optimizer=optimizer, scheduler=scheduler,
            best_metric=1.25, val_table={(0, 0): 0.5, (0, 25): 2.0},
        )

    def test_metadata_and_fingerprint_are_embedded_in_checkpoint_config(self):
        payload = self.payload()
        self.assertEqual(payload["config"], self.cfg.to_dict())
        self.assertEqual(payload["config"]["architecture_metadata"], self.cfg.architecture_metadata)
        self.assertEqual(payload["config_fingerprint"], entry_module().lite_fingerprint(self.cfg))
        self.assertEqual(payload["allocation_grid"], [(0, 256)])
        self.assertEqual(payload["scheduler"]["total_steps"], 50_000)
        entry_module().validate_resume_payload(payload, self.cfg)

    def test_synthetic_checkpoint_restores_strict_state_optimizer_scheduler_rng_and_progress(self):
        payload = self.payload()
        entry_module().validate_resume_payload(payload, self.cfg)
        main_train.atomic_torch_save(payload, self.cfg.SAVE_PATH)
        restored, optimizer, scheduler = self.fixture()
        with torch.no_grad():
            restored.weight.fill_(-9.0)
            restored.running_pwr.zero_()
        scheduler.last_epoch = 0
        optimizer.state[restored.weight]["exp_avg"].zero_()
        random.random()
        np.random.rand()
        torch.rand(1)
        start, best = main_train.load_latest_checkpoint(self.cfg, restored, optimizer, scheduler, torch.device("cpu"))
        self.assertEqual((start, best), (501, 1.25))
        self.assertTrue(torch.equal(restored.weight, payload["state"]["weight"]))
        self.assertTrue(torch.equal(restored.running_pwr, payload["state"]["running_pwr"]))
        self.assertTrue(torch.equal(optimizer.state[restored.weight]["exp_avg"], torch.tensor([0.125])))
        self.assertEqual(scheduler.state_dict()["last_epoch"], 500)
        self.assertEqual(scheduler.state_dict()["total_steps"], 50_000)
        self.assertEqual(random.getstate(), payload["rng_state"]["python"])
        np.testing.assert_equal(np.random.get_state(), payload["rng_state"]["numpy"])
        self.assertTrue(torch.equal(torch.get_rng_state(), payload["rng_state"]["torch"]))

    def test_resume_payload_rejects_architecture_config_width_and_fingerprint_mismatches(self):
        entry = entry_module()
        payload = self.payload()
        variants = []
        for field, value in (("ue_ffn_dim", 2048), ("model_class", "AdaptiveHybridPrecoder"),
                             ("architecture", "original")):
            broken = deepcopy(payload)
            broken["config"]["architecture_metadata"][field] = value
            variants.append(broken)
        broken = deepcopy(payload)
        broken["config_fingerprint"] = "not-this-config"
        variants.append(broken)
        broken = deepcopy(payload)
        broken["allocation_grid"] = [(64, 0)]
        variants.append(broken)
        broken = deepcopy(payload)
        broken["state"]["running_pwr"] = torch.ones(5)
        variants.append(broken)
        for index, broken in enumerate(variants):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, RuntimeError)):
                    entry.validate_resume_payload(broken, self.cfg)

    def test_resume_requires_optimizer_scheduler_rng_and_incomplete_progress(self):
        entry = entry_module()
        payload = self.payload()
        for key in ("optimizer", "scheduler", "rng_state", "config", "config_fingerprint", "state", "step"):
            broken = deepcopy(payload)
            del broken[key]
            with self.subTest(missing=key):
                with self.assertRaises((ValueError, RuntimeError)):
                    entry.validate_resume_payload(broken, self.cfg)
        for step in (0, -1, 50_000, 50_001):
            broken = deepcopy(payload)
            broken["step"] = step
            with self.subTest(step=step):
                with self.assertRaises((ValueError, RuntimeError)):
                    entry.validate_resume_payload(broken, self.cfg)

    def test_strict_restore_rejects_missing_and_incompatible_tensor_shapes(self):
        for broken_state in ({}, {"weight": torch.ones(2), "running_pwr": torch.ones(1)}):
            payload = self.payload()
            payload["state"] = broken_state
            main_train.atomic_torch_save(payload, self.cfg.SAVE_PATH)
            model, optimizer, scheduler = self.fixture()
            with self.subTest(keys=tuple(broken_state)):
                with self.assertRaises(RuntimeError):
                    main_train.load_latest_checkpoint(self.cfg, model, optimizer, scheduler, torch.device("cpu"))

    def test_real_lite_factory_and_strict_synthetic_checkpoint_when_dependencies_exist(self):
        # This is a constructor/strict-state check only, not a forward, training
        # smoke test or GPU preflight. Existing 16 model tests remain separate.
        try:
            from models.adaptive_hybrid_lite import AdaptiveHybridPrecoderLite
        except ModuleNotFoundError as exc:
            if exc.name not in {"sionna", "sionna.phy"}:
                raise
            self.skipTest(f"Missing {exc.name!r}: actual Lite CPU factory/checkpoint check not run; no bypass.")
        with redirect_stdout(io.StringIO()):
            model = entry_module().lite_model_factory(self.cfg, torch.device("cpu"))
        self.assertIs(type(model), AdaptiveHybridPrecoderLite)
        self.assertEqual(tuple(model.running_pwr.shape), (1,))
        self.assertEqual(model.encoder.fdma_backbone.layers[0].linear1.out_features, 1024)
        self.assertEqual(model.encoder.aircomp_backbone.layers[0].linear1.out_features, 1024)
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        checkpoint = Path(self.temporary.name) / "synthetic_lite_state.pth"
        torch.save(state, checkpoint)
        loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
