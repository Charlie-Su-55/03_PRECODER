"""Standard-library contracts for the isolated two-arm training entry.

Run from 03_PRECODER:
    python -X utf8 -B -S -m unittest discover -s tests -p test_train_proposal_as_direct.py -v

No models, CUDA, training loops, real weights, or channel generators are loaded.
Filesystem fixtures and opaque checkpoint placeholders stay in temporary folders.
"""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import as_direct_train_config as config_module
from configs.as_direct_train_config import ARMS, OUTPUT_ROOT, RECIPE, as_direct_config, validate_config
from configs.train_config import TRAINING_RECIPES
import train_proposal_as_direct as entry


SOURCE_SHA256 = "e21ea1feae595e0898b6be724f7062b05c09b2d1ef03fe9ffec7ac91869064bf"
FULL_GRID = ((64, 0), (48, 64), (32, 128), (16, 192), (0, 256))


def source_tree(filename):
    return ast.parse((ROOT / filename).read_text(encoding="utf-8"))


def source_function(filename, name):
    return next(node for node in source_tree(filename).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def make_run_files(cfg, *, step=500, state="paused", omit=()):
    """Metadata and arbitrary bytes only; placeholder weights are never loaded."""
    Path(cfg.SAVE_DIR).mkdir(parents=True)
    Path(cfg.LOG_DIR).mkdir(parents=True)
    files = {
        "CONFIG_PATH": json.dumps(dict(cfg.to_dict(), config_fingerprint=entry.fingerprint(cfg))).encode(),
        "STATUS_PATH": json.dumps({"state": state, "step": step,
                                   "experiment": cfg.EXP_NAME,
                                   "fingerprint": entry.fingerprint(cfg)}).encode(),
        "SAVE_PATH": b"opaque latest checkpoint placeholder, not serialized weights",
        "BEST_PATH": b"opaque best checkpoint placeholder, not serialized weights",
        "METRICS_PATH": b"{}\n",
    }
    for field, content in files.items():
        if field not in omit:
            Path(getattr(cfg, field)).write_bytes(content)


def resume_metadata(cfg, *, step=500):
    """Schema-only fixture: tensor/RNG sentinel values are not runtime-valid."""
    lr, beta1 = entry.expected_cycle(cfg, step)
    state = {"running_pwr": None}
    if cfg.arm == "as_direct":
        state.update({f"layers.{index}.alpha": None for index in range(5)})
    return {
        "state": state, "config": cfg.to_dict(), "config_fingerprint": entry.fingerprint(cfg),
        "allocation_grid": [(0, 256)], "step": step, "best_metric": 1.25,
        "optimizer": {
            "state": {0: {"step": None, "exp_avg": None, "exp_avg_sq": None}},
            "param_groups": [{
                "params": [0], "lr": lr, "initial_lr": cfg.LR / 25,
                "max_lr": cfg.LR, "min_lr": cfg.LR / 25 / 10000,
                "weight_decay": cfg.WEIGHT_DECAY, "eps": 1e-8,
                "base_momentum": 0.85, "max_momentum": 0.95, "betas": (beta1, 0.999),
                "amsgrad": False, "maximize": False, "capturable": False,
                "differentiable": False, "foreach": None, "fused": None,
            }],
        },
        "scheduler": {
            "total_steps": 5000, "last_epoch": step, "_step_count": step + 1,
            "_schedule_phases": [
                {"end_step": 499.0, "start_lr": "initial_lr", "end_lr": "max_lr",
                 "start_momentum": "max_momentum", "end_momentum": "base_momentum"},
                {"end_step": 4999, "start_lr": "max_lr", "end_lr": "min_lr",
                 "start_momentum": "base_momentum", "end_momentum": "max_momentum"},
            ],
            "_anneal_func_type": "cos", "cycle_momentum": True, "use_beta1": True,
            "base_lrs": [cfg.LR / 25], "_last_lr": [lr],
        },
        "rng_state": {"python": "metadata sentinel", "numpy": "metadata sentinel",
                      "torch": "metadata sentinel", "cuda": ["metadata sentinel"]},
    }


class ConfigurationTests(unittest.TestCase):
    def test_two_arms_have_identical_fixed_screening_recipe(self):
        self.assertEqual(ARMS, ("full_control", "as_direct"))
        expected = {
            "name": "as_direct_screen5k", "total_steps": 5000, "batch_size": 48,
            "lr": 2e-5, "val_interval": 500, "val_batch": 64,
            "warmup_pct": 0.10, "grad_clip": 0.5, "phase1_pct": 0.4,
            "snr_stage1_pct": 0.3, "snr_stage2_pct": 0.7,
            "stage1_snr": (0.0, 25.0), "stage2_snr": (0.0, 25.0),
            "stage3_snr": (0.0, 25.0), "val_snr_list": (0, 5, 10, 15, 20, 25),
            "dir_mode": "real", "ph1_w_dir": 0.0, "ph1_w_mse": 0.0,
            "ph2_w_dir": 0.0, "ph2_w_mse": 0.0, "weight_decay": 1e-5,
            "data_cache_size": 1, "log_interval": 100, "best_metric": "mean_all",
            "downlink_snr_db": 25.0,
        }
        self.assertEqual(asdict(RECIPE), expected)
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = as_direct_config(arm)
                self.assertIs(cfg.recipe, RECIPE)
                self.assertEqual(cfg.SEED, 20261025)
                self.assertEqual(cfg.TOTAL_STEPS, 5000)
                self.assertEqual((cfg.PHASE1_STEPS, cfg.STAGE1_STEPS, cfg.STAGE2_STEPS),
                                 (2000, 1500, 3500))
                validate_config(cfg)

    def test_screening_recipe_does_not_mutate_global_registry_or_source_recipe(self):
        before = deepcopy(TRAINING_RECIPES)
        cfg = as_direct_config("as_direct")
        self.assertEqual(TRAINING_RECIPES, before)
        self.assertNotIn(RECIPE.name, TRAINING_RECIPES)
        source_recipe = cfg.source_identity["loaded_checkpoint_config"]["training_recipe"]
        current_recipe = json.loads(json.dumps(asdict(RECIPE)))
        for field in source_recipe:
            if field not in ("name", "total_steps"):
                with self.subTest(field=field):
                    self.assertEqual(current_recipe[field], source_recipe[field])
        self.assertEqual(source_recipe["total_steps"], 20000)
        self.assertEqual(source_recipe["name"], "k4_n256_fb_robust_ft")

    def test_singleton_allocation_preserves_full_architecture_widths(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = as_direct_config(arm)
                self.assertEqual(cfg.GAMMA_GRID, [(0, 256)])
                self.assertEqual(cfg.ALLOCATION_GRID, [(0, 256)])
                self.assertEqual(cfg.full_allocation_grid, FULL_GRID)
                self.assertEqual((cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED), (64, 256))
                self.assertEqual((cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS,
                                  cfg.D_TOT, cfg.NUM_SUBBANDS), (4, 32, 256, 256, 4))
                self.assertEqual((cfg.D_MODEL, cfg.NUM_HEADS, cfg.NUM_ENCODER_LAYERS,
                                  cfg.NUM_UNFOLD_LAYERS, cfg.GNN_AGG_DIM), (256, 8, 4, 5, 48))
                self.assertEqual(cfg.architecture_metadata["ue_ffn_dim"], 2048)
                self.assertEqual(cfg.architecture_metadata["running_pwr_shape"], [1])
                self.assertEqual(cfg.architecture_metadata["cfg_idx"], 0)

    def test_arm_identity_is_distinct_and_configuration_does_not_create_directories(self):
        self.assertEqual(OUTPUT_ROOT, ROOT / "runs_as_screen/proposal_as_direct_v1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "screening"
            configs = [as_direct_config(arm, root) for arm in ARMS]
            self.assertNotEqual(configs[0].EXP_NAME, configs[1].EXP_NAME)
            self.assertNotEqual(configs[0].SAVE_DIR, configs[1].SAVE_DIR)
            self.assertNotEqual(configs[0].LOG_DIR, configs[1].LOG_DIR)
            self.assertEqual([cfg.architecture_metadata["model_class"] for cfg in configs],
                             ["AdaptiveHybridPrecoder", "AdaptiveHybridPrecoderASDirect"])
            self.assertEqual([cfg.architecture_metadata["additional_parameters"] for cfg in configs], [0, 5])
            self.assertEqual([cfg.architecture_metadata["architecture"] for cfg in configs],
                             ["full", "as_direct_v1"])
            self.assertFalse(root.exists())

    def test_source_identity_is_pinned_and_validation_is_shared_with_preflight(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = as_direct_config(arm)
                source = cfg.source_identity
                self.assertEqual(source["method_key"], "robust_specialist")
                self.assertEqual(source["checkpoint_sha256"], SOURCE_SHA256)
                self.assertEqual(source["checkpoint_step"], 20000)
                self.assertEqual(source["cfg_index"], 0)
                self.assertEqual(source["allocations"], [[0, 256]])
                with mock.patch.object(config_module, "load_source_record",
                                       side_effect=ValueError("pinned source rejected")) as load:
                    with self.assertRaisesRegex(ValueError, "pinned source rejected"):
                        as_direct_config(arm)
                load.assert_called_once_with()
        imports = [node for node in source_tree("configs/as_direct_train_config.py").body
                   if isinstance(node, ast.ImportFrom)]
        self.assertTrue(any(node.module == "preflight_proposal_as_direct"
                            and any(alias.name == "load_source_record" for alias in node.names)
                            for node in imports))

    def test_fixed_config_rejects_recipe_source_capacity_and_seed_changes(self):
        cfg = as_direct_config("as_direct")
        wrong_source = deepcopy(cfg.source_identity)
        wrong_source["checkpoint_sha256"] = "a" * 64
        changes = (
            replace(cfg, recipe=replace(cfg.recipe, total_steps=1000)),
            replace(cfg, recipe=replace(cfg.recipe, lr=1e-4)),
            replace(cfg, recipe=replace(cfg.recipe, ph2_w_mse=1.0)),
            replace(cfg, spec=replace(cfg.spec, seed=42)),
            replace(cfg, allocations=((64, 0),)),
            replace(cfg, full_allocation_grid=((0, 256),)),
            replace(cfg, proposal=replace(cfg.proposal, D_MODEL=128)),
            replace(cfg, source_identity=wrong_source),
        )
        for changed in changes:
            with self.subTest(config=changed.to_dict()):
                with self.assertRaises(ValueError):
                    validate_config(changed)
        with self.assertRaises(ValueError):
            as_direct_config("unknown")

    def test_fingerprint_matches_shared_trainer_serialization_and_covers_arm(self):
        fingerprints = []
        for arm in ARMS:
            cfg = as_direct_config(arm)
            serialized = json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
            expected = hashlib.sha256(serialized).hexdigest()[:16]
            self.assertEqual(entry.fingerprint(cfg), expected)
            fingerprints.append(expected)
        self.assertNotEqual(*fingerprints)


class RunStateTests(unittest.TestCase):
    def test_fresh_run_checks_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "screening"
            for arm in ARMS:
                cfg = as_direct_config(arm, root)
                entry.validate_output_paths(cfg, allowed_root=root)
                entry.check_run_state(cfg, resume=False, allowed_root=root)
            self.assertFalse(root.exists())

    def test_output_root_cannot_escape_to_foreign_or_parent_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "screening"
            cfg = as_direct_config("as_direct", root)
            for outside in (Path(temporary) / "old_runs", root / ".." / "old_runs"):
                with self.subTest(path=outside), self.assertRaises(ValueError):
                    entry.validate_output_paths(replace(cfg, output_root=outside), allowed_root=root)
            self.assertFalse(root.exists())

    def test_fresh_runs_never_reuse_checkpoint_or_log_directories(self):
        for arm in ARMS:
            for field in ("SAVE_DIR", "LOG_DIR"):
                with self.subTest(arm=arm, field=field), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "screening"
                    cfg = as_direct_config(arm, root)
                    path = Path(getattr(cfg, field))
                    path.mkdir(parents=True)
                    marker = path / "existing.txt"
                    marker.write_text("preserve existing output", encoding="utf-8")
                    with self.assertRaises(FileExistsError):
                        entry.check_run_state(cfg, resume=False, allowed_root=root)
                    self.assertEqual(marker.read_text(encoding="utf-8"), "preserve existing output")

    def test_resume_requires_every_run_file_and_cannot_fall_back_to_fresh(self):
        for omitted in ("SAVE_PATH", "CONFIG_PATH", "STATUS_PATH", "BEST_PATH", "METRICS_PATH"):
            with self.subTest(missing=omitted), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "screening"
                cfg = as_direct_config("as_direct", root)
                make_run_files(cfg, omit=(omitted,))
                with self.assertRaises(FileNotFoundError):
                    entry.check_run_state(cfg, resume=True, allowed_root=root)

    def test_valid_resume_run_checks_do_not_modify_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "screening"
            cfg = as_direct_config("full_control", root)
            make_run_files(cfg)
            before = {path.relative_to(root): path.read_bytes()
                      for path in root.rglob("*") if path.is_file()}
            entry.check_run_state(cfg, resume=True, allowed_root=root)
            after = {path.relative_to(root): path.read_bytes()
                     for path in root.rglob("*") if path.is_file()}
            self.assertEqual(after, before)

    def test_completed_run_and_cross_arm_saved_config_cannot_resume(self):
        for invalid in ("completed", "cross_arm", "changed_config"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "screening"
                cfg = as_direct_config("as_direct", root)
                make_run_files(cfg, state="completed" if invalid == "completed" else "paused")
                if invalid == "cross_arm":
                    other = as_direct_config("full_control", root)
                    Path(cfg.CONFIG_PATH).write_text(json.dumps(dict(
                        other.to_dict(), config_fingerprint=entry.fingerprint(other))), encoding="utf-8")
                if invalid == "changed_config":
                    snapshot = json.loads(Path(cfg.CONFIG_PATH).read_text(encoding="utf-8"))
                    snapshot["training_recipe"]["lr"] = 1e-4
                    Path(cfg.CONFIG_PATH).write_text(json.dumps(snapshot), encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    entry.check_run_state(cfg, resume=True, allowed_root=root)


class ResumeMetadataTests(unittest.TestCase):
    def best_metadata(self, cfg, *, step=500):
        selected = resume_metadata(cfg, step=step)
        for field in ("optimizer", "scheduler", "rng_state"):
            del selected[field]
        return selected

    def test_retained_best_accepts_same_arm_earlier_or_current_selected_step(self):
        for arm in ARMS:
            cfg = as_direct_config(arm)
            latest = resume_metadata(cfg, step=1500)
            for step in (500, 1500):
                with self.subTest(arm=arm, selected_step=step):
                    selected = self.best_metadata(cfg, step=step)
                    before = deepcopy(selected)
                    entry.validate_best_metadata(selected, latest, cfg)
                    self.assertEqual(selected, before)

    def test_retained_best_rejects_cross_arm_config_grid_and_selection_metric(self):
        cfg = as_direct_config("as_direct")
        latest = resume_metadata(cfg, step=1500)
        foreign = self.best_metadata(as_direct_config("full_control"))
        with self.assertRaisesRegex(ValueError, "(?i)(arm|config)"):
            entry.validate_best_metadata(foreign, latest, cfg)
        for change in ("config", "fingerprint", "allocation", "metric", "nonfinite_metric"):
            with self.subTest(change=change):
                selected = self.best_metadata(cfg)
                if change == "config":
                    selected["config"]["source_identity"]["checkpoint_sha256"] = "a" * 64
                elif change == "fingerprint":
                    selected["config_fingerprint"] = "0" * 16
                elif change == "allocation":
                    selected["allocation_grid"] = list(FULL_GRID)
                else:
                    selected["best_metric"] = 2.0 if change == "metric" else float("nan")
                with self.assertRaises(ValueError):
                    entry.validate_best_metadata(selected, latest, cfg)

    def test_retained_best_requires_state_metadata_and_a_valid_nonfuture_step(self):
        cfg = as_direct_config("as_direct")
        latest = resume_metadata(cfg, step=1500)
        for field in ("state", "config", "config_fingerprint", "allocation_grid", "step", "best_metric"):
            with self.subTest(missing=field):
                selected = self.best_metadata(cfg)
                del selected[field]
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    entry.validate_best_metadata(selected, latest, cfg)
        for step in (0, -500, True, 500.0, 501, 2000):
            with self.subTest(step=step):
                selected = self.best_metadata(cfg)
                selected["step"] = step
                with self.assertRaisesRegex(ValueError, "step"):
                    entry.validate_best_metadata(selected, latest, cfg)

    def test_both_arms_accept_complete_metadata_at_saved_validation_boundaries(self):
        for arm in ARMS:
            cfg = as_direct_config(arm)
            for step in (500, 1000, 4500):
                with self.subTest(arm=arm, step=step):
                    payload = resume_metadata(cfg, step=step)
                    before = deepcopy(payload)
                    entry.validate_resume_payload(payload, cfg)
                    self.assertEqual(payload, before)

    def test_every_resume_field_is_required(self):
        cfg = as_direct_config("as_direct")
        original = resume_metadata(cfg)
        for field in original:
            with self.subTest(field=field):
                payload = deepcopy(original)
                del payload[field]
                with self.assertRaises(ValueError):
                    entry.validate_resume_payload(payload, cfg)

    def test_cross_arm_source_config_and_fingerprint_mismatches_are_rejected(self):
        cfg = as_direct_config("as_direct")
        other = as_direct_config("full_control")
        with self.assertRaisesRegex(ValueError, "(?i)(arm|config|fingerprint)"):
            entry.validate_resume_payload(resume_metadata(other), cfg)
        for change in ("fingerprint", "recipe", "source", "architecture"):
            with self.subTest(change=change):
                payload = resume_metadata(cfg)
                if change == "fingerprint":
                    payload["config_fingerprint"] = "0" * 16
                elif change == "recipe":
                    payload["config"]["training_recipe"]["total_steps"] = 20000
                elif change == "source":
                    payload["config"]["source_identity"]["checkpoint_sha256"] = "a" * 64
                else:
                    payload["config"]["architecture_metadata"]["ue_ffn_dim"] = 1024
                with self.assertRaises(ValueError):
                    entry.validate_resume_payload(payload, cfg)

    def test_saved_step_must_be_an_unfinished_integer_validation_boundary(self):
        cfg = as_direct_config("as_direct")
        for step in (0, -500, True, 500.0, 1, 499, 501, 4501, 5000, 5500):
            with self.subTest(step=step):
                payload = resume_metadata(cfg)
                payload["step"] = step
                with self.assertRaisesRegex(ValueError, "(?i)step"):
                    entry.validate_resume_payload(payload, cfg)
        entry.validate_resume_payload(resume_metadata(cfg, step=5000), cfg, allow_completed=True)

    def test_grid_and_gate_keys_cannot_change_arm(self):
        for arm in ARMS:
            cfg = as_direct_config(arm)
            for bad_grid in (list(FULL_GRID), [], [(64, 0)]):
                with self.subTest(arm=arm, grid=bad_grid):
                    payload = resume_metadata(cfg)
                    payload["allocation_grid"] = bad_grid
                    with self.assertRaisesRegex(ValueError, "allocation_grid"):
                        entry.validate_resume_payload(payload, cfg)
            payload = resume_metadata(cfg)
            if arm == "as_direct":
                del payload["state"]["layers.0.alpha"]
            else:
                payload["state"]["layers.0.alpha"] = None
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "gate"):
                entry.validate_resume_payload(payload, cfg)
            payload = resume_metadata(cfg)
            del payload["state"]["running_pwr"]
            with self.assertRaisesRegex(ValueError, "running_pwr"):
                entry.validate_resume_payload(payload, cfg)

    def test_nonfinite_best_metric_is_rejected(self):
        cfg = as_direct_config("as_direct")
        for metric in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(metric=metric):
                payload = resume_metadata(cfg)
                payload["best_metric"] = metric
                with self.assertRaisesRegex(ValueError, "best_metric"):
                    entry.validate_resume_payload(payload, cfg)

    def test_scheduler_recipe_position_and_rate_must_match(self):
        cfg = as_direct_config("as_direct")
        changes = {
            "total_steps": 20000, "last_epoch": 499, "_step_count": 500,
            "_schedule_phases": [], "_anneal_func_type": "linear",
            "cycle_momentum": False, "use_beta1": False, "base_lrs": [cfg.LR],
            "_last_lr": [1e-4],
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                payload = resume_metadata(cfg)
                payload["scheduler"][field] = value
                with self.assertRaisesRegex(ValueError, "(?i)scheduler"):
                    entry.validate_resume_payload(payload, cfg)
        for value in (None, {}, {"total_steps": 5000}, {"_last_lr": []}):
            with self.subTest(scheduler=value):
                payload = resume_metadata(cfg)
                payload["scheduler"] = value
                with self.assertRaises(ValueError):
                    entry.validate_resume_payload(payload, cfg)

    def test_optimizer_has_one_shared_rate_and_original_adamw_settings(self):
        cfg = as_direct_config("as_direct")
        changes = {"lr": 1e-4, "initial_lr": cfg.LR, "max_lr": 1e-4,
                   "min_lr": 0.0, "weight_decay": 0.0, "eps": 1e-6,
                   "base_momentum": 0.8, "max_momentum": 0.99,
                   "betas": (0.9, 0.999), "amsgrad": True, "maximize": True,
                   "capturable": True, "differentiable": True, "foreach": True, "fused": True}
        for field, value in changes.items():
            with self.subTest(field=field):
                payload = resume_metadata(cfg)
                payload["optimizer"]["param_groups"][0][field] = value
                with self.assertRaises(ValueError):
                    entry.validate_resume_payload(payload, cfg)
        for change in ("empty_state", "no_groups", "two_groups"):
            with self.subTest(change=change):
                payload = resume_metadata(cfg)
                if change == "empty_state":
                    payload["optimizer"]["state"] = {}
                elif change == "no_groups":
                    payload["optimizer"]["param_groups"] = []
                else:
                    payload["optimizer"]["param_groups"] *= 2
                with self.assertRaises(ValueError):
                    entry.validate_resume_payload(payload, cfg)

    def test_all_rng_families_and_cuda_state_list_are_required(self):
        cfg = as_direct_config("as_direct")
        for field in ("python", "numpy", "torch", "cuda"):
            with self.subTest(field=field):
                payload = resume_metadata(cfg)
                del payload["rng_state"][field]
                with self.assertRaisesRegex(ValueError, "RNG"):
                    entry.validate_resume_payload(payload, cfg)
        payload = resume_metadata(cfg)
        payload["rng_state"]["cuda"] = []
        with self.assertRaisesRegex(ValueError, "CUDA"):
            entry.validate_resume_payload(payload, cfg)

    def test_onecycle_reference_values_preserve_schedule_and_beta1(self):
        cfg = as_direct_config("as_direct")
        expected = {
            0: (8e-7, 0.95),
            499: (2e-5, 0.85),
            2749: ((2e-5 + 8e-11) / 2, 0.90),
            4999: (8e-11, 0.95),
        }
        for step, values in expected.items():
            with self.subTest(step=step):
                actual = entry.expected_cycle(cfg, step)
                for value, reference in zip(actual, values):
                    self.assertTrue(math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-15))

    def test_only_explicitly_inactive_parameters_may_lack_optimizer_moments(self):
        inactive = (
            "encoder.fdma_sb_query", "pad_f", "pad_a", "residual_scale",
            "encoder.fdma_input_proj.weight", "encoder.fdma_input_norm.bias",
            "encoder.fdma_backbone.layers.0.linear1.weight",
            "encoder.fdma_attn_pool.in_proj_weight", "encoder.fc_fdma.weight",
            "fdma_emb.weight", "residual_head.weight",
        )
        active = (
            "encoder.aircomp_backbone.layers.0.linear1.weight",
            "encoder.fc_aircomp.weight", "aircomp_emb.weight",
            "layers.0.ffn.0.weight", "layers.4.norm3.bias", "unknown_new_parameter",
            "encoder.fdma_backbone_extra.weight", "fdma_embedded.weight",
        ) + tuple(f"layers.{index}.alpha" for index in range(5))
        for name in inactive:
            with self.subTest(parameter=name):
                self.assertFalse(entry.optimizer_state_required(name))
        for name in active:
            with self.subTest(parameter=name):
                self.assertTrue(entry.optimizer_state_required(name))


class InitializationTests(unittest.TestCase):
    def mock_model(self, *, trainable=True):
        model = mock.Mock()
        model.parameters.return_value = [SimpleNamespace(requires_grad=trainable)]
        model.to.return_value = model
        return model

    def test_full_control_initializes_source_with_strict_loading(self):
        cfg = as_direct_config("full_control")
        model = self.mock_model()
        source_state = {"running_pwr": "source metadata sentinel"}
        self.assertIs(entry.initialize_model(cfg, model, source_state=source_state), model)
        model.load_state_dict.assert_called_once_with(source_state, strict=True)

    def test_both_resume_arms_load_saved_state_strictly_without_migration(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = as_direct_config(arm)
                model = self.mock_model()
                saved_state = {"running_pwr": "resume sentinel", "layers.0.alpha": 0.25}
                self.assertIs(entry.initialize_model(cfg, model, resume_state=saved_state), model)
                model.load_state_dict.assert_called_once_with(saved_state, strict=True)
                self.assertEqual(saved_state["layers.0.alpha"], 0.25)

    def test_initialization_requires_exactly_one_state_and_all_parameters_trainable(self):
        cfg = as_direct_config("full_control")
        for arguments in ({}, {"source_state": {}, "resume_state": {}}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                entry.initialize_model(cfg, self.mock_model(), **arguments)
        with self.assertRaisesRegex(ValueError, "trainable"):
            entry.initialize_model(cfg, self.mock_model(trainable=False), resume_state={})

    def test_resume_factory_initializes_once_and_returns_the_same_model(self):
        cfg = as_direct_config("as_direct")
        model = self.mock_model()
        state = {"running_pwr": "resume sentinel"}
        factory = entry.make_model_factory(cfg, model, resume_state=state)
        model.load_state_dict.assert_not_called()
        device = object()
        self.assertIs(factory(cfg, device), model)
        model.load_state_dict.assert_called_once_with(state, strict=True)
        model.to.assert_called_once_with(device)
        with self.assertRaisesRegex(RuntimeError, "(?i)(reused|different)"):
            factory(cfg, device)
        model.load_state_dict.assert_called_once()

    def test_factory_rejects_another_arm_before_loading_state(self):
        cfg = as_direct_config("as_direct")
        model = self.mock_model()
        factory = entry.make_model_factory(cfg, model, resume_state={})
        with self.assertRaisesRegex(RuntimeError, "different"):
            factory(as_direct_config("full_control"), object())
        model.load_state_dict.assert_not_called()
        model.to.assert_not_called()


class TrainerIntegrationSourceTests(unittest.TestCase):
    def test_shared_trainer_calls_factory_before_optimizer_creation(self):
        train = source_function("main_train.py", "train_proposal")
        calls = [node for node in ast.walk(train) if isinstance(node, ast.Call)]
        factory = next(node for node in calls
                       if isinstance(node.func, ast.Name) and node.func.id == "model_factory")
        optimizer = next(node for node in calls
                         if isinstance(node.func, ast.Attribute) and node.func.attr == "AdamW")
        self.assertLess(factory.lineno, optimizer.lineno)
        self.assertEqual([ast.unparse(arg) for arg in factory.args], ["cfg", "device"])

    def test_entry_reuses_shared_trainer_and_disables_its_late_initialization(self):
        tree = source_tree("train_proposal_as_direct.py")
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        trainer_calls = [node for node in calls if ast.unparse(node.func) == "train_proposal"]
        self.assertTrue(trainer_calls)
        for call in trainer_calls:
            self.assertIn("model_factory", {keyword.arg for keyword in call.keywords})
        options = [node for node in calls if ast.unparse(node.func) == "RunOptions"]
        self.assertTrue(options)
        for call in options:
            values = {keyword.arg: keyword.value for keyword in call.keywords}
            self.assertIn("init_checkpoint", values)
            self.assertIsNone(ast.literal_eval(values["init_checkpoint"]))
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.module == "main_train"
                            and any(alias.name == "train_proposal" for alias in node.names)
                            for node in ast.walk(tree)))

    def test_shared_scheduler_uses_full_five_thousand_step_recipe(self):
        train = source_function("main_train.py", "train_proposal")
        scheduler = next(node for node in ast.walk(train) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "OneCycleLR")
        arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in scheduler.keywords}
        self.assertEqual(arguments["total_steps"], "cfg.TOTAL_STEPS")
        self.assertEqual(arguments["max_lr"], "cfg.LR")
        self.assertEqual(arguments["pct_start"], "cfg.WARMUP_PCT")
        self.assertEqual(as_direct_config("as_direct").TOTAL_STEPS, 5000)

    def test_source_migration_import_is_lazy_and_only_in_fresh_direct_branch(self):
        function = source_function("train_proposal_as_direct.py", "initialize_model")
        branch = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "cfg.arm == 'as_direct'")
        self.assertTrue(any(isinstance(node, ast.ImportFrom)
                            and node.module == "models.adaptive_hybrid_as_direct"
                            and any(alias.name == "migrate_original_state_dict" for alias in node.names)
                            for node in branch.body))
        calls = [node for node in ast.walk(branch) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "migrate_original_state_dict"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([ast.unparse(arg) for arg in calls[0].args], ["model", "source_state"])


class CommandLineTests(unittest.TestCase):
    def test_no_arguments_and_explicit_help_never_run_training(self):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(entry, "run_training") as run:
            self.assertEqual(entry.main([]), 0)
            with self.assertRaises(SystemExit) as raised:
                entry.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        run.assert_not_called()
        self.assertIn("--arm", output.getvalue())
        self.assertIn("--stop-after", output.getvalue())

    def test_conflicting_modes_and_unsafe_resume_or_stop_arguments_are_rejected(self):
        invalid = (
            ["--dry-run", "--train"], ["--train", "--stop-after", "500"],
            ["--resume"], ["--checkpoint", "missing.pth"], ["--dry-run"],
            ["--train"], ["--arm", "unknown", "--dry-run"],
            ["--arm", "as_direct", "--dry-run", "--resume"],
            ["--arm", "as_direct", "--train", "--resume", "--checkpoint", "missing.pth"],
        )
        invalid += tuple(["--arm", "as_direct", "--stop-after", str(value)]
                         for value in (0, -500, 1, 499, 501, 5500))
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    entry.main(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_fixed_protocol_has_no_seed_recipe_or_output_cli_overrides(self):
        for option, value in (("--seed", "7"), ("--total-steps", "1000"),
                              ("--lr", "0.01"), ("--output-root", "elsewhere"),
                              ("--manifest", "other.json")):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    entry.build_parser().parse_args(["--arm", "as_direct", "--dry-run", option, value])
                self.assertEqual(raised.exception.code, 2)

    def test_pause_dispatch_preserves_full_schedule_and_explicit_resume_flag(self):
        # Only the entry's own execution helper is mocked; no trainer or model is imported.
        for arm in ARMS:
            for resume in (False, True):
                arguments = ["--arm", arm, "--stop-after", "500"]
                if resume:
                    arguments.append("--resume")
                with self.subTest(arm=arm, resume=resume), mock.patch.object(entry, "run_training") as run:
                    self.assertEqual(entry.main(arguments), 0)
                run.assert_called_once()
                cfg, parsed = run.call_args.args
                self.assertEqual((cfg.arm, cfg.TOTAL_STEPS), (arm, 5000))
                self.assertEqual(parsed.stop_after, 500)
                self.assertEqual(parsed.resume, resume)

    def test_import_help_and_both_dry_runs_need_no_heavy_modules_weights_or_writes(self):
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
forbidden = {"torch", "numpy", "sionna", "tensorflow", "models", "utils", "main_train"}

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
import train_proposal_as_direct as entry

help_output = io.StringIO()
with contextlib.redirect_stdout(help_output):
    assert entry.main([]) == 0
assert "--train" in help_output.getvalue()

plans = []
for arm in ("full_control", "as_direct"):
    dry_output = io.StringIO()
    with contextlib.redirect_stdout(dry_output):
        assert entry.main(["--arm", arm, "--dry-run", "--checkpoint", "missing/source.pth"]) == 0
    plan = json.loads(dry_output.getvalue())
    assert plan["config"]["arm"] == arm
    assert plan["config"]["experiment"]["seed"] == 20261025
    assert plan["config"]["allocations"] == [[0, 256]]
    assert plan["new_optimizer"] == "AdamW"
    assert plan["scheduler_total_steps"] == 5000
    assert plan["checkpoint_availability_not_checked"] is True
    assert plan["config"]["architecture_metadata"]["ue_ffn_dim"] == 2048
    plans.append(plan)
assert plans[0]["fingerprint"] != plans[1]["fingerprint"]
assert plans[0]["latest"] != plans[1]["latest"]
assert not forbidden.intersection(sys.modules)
print("both dry runs stayed lightweight and read-only")
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-B", "-S", "-c", script, str(ROOT)],
                cwd=temporary, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("both dry runs stayed lightweight and read-only", result.stdout)


if __name__ == "__main__":
    unittest.main()
