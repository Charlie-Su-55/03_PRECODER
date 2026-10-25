"""Stdlib-only ASReceiver recipe/initialization/resume contracts.

No actual model, weights, channel generator, CUDA context or trainer is run.
Temporary metadata fixtures contain opaque placeholders, not checkpoint data.
"""
import ast
from copy import deepcopy
from dataclasses import asdict, replace
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

from configs.as_receiver_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, RECIPE, as_receiver_config, validate_config,
    make_channel_data, channel_data_api,
)
from configs.train_config import TRAINING_RECIPES
import train_proposal_as_receiver as entry


def metadata(cfg, step=500):
    """Schema fixture; deliberately not a runtime-loadable tensor checkpoint."""
    lr, beta1 = entry.expected_cycle(cfg, step)
    state = {"running_pwr": None}
    if cfg.arm == "as_receiver":
        state["as_receiver.output_proj.weight"] = None
    return {
        "state": state, "config": cfg.to_dict(),
        "config_fingerprint": entry.fingerprint(cfg), "allocation_grid": [(0, 256)],
        "step": step, "best_metric": 1.25,
        "optimizer": {"state": {0: {"step": None, "exp_avg": None, "exp_avg_sq": None}},
            "param_groups": [{"params": [0], "lr": lr, "initial_lr": cfg.LR / 25,
                "max_lr": cfg.LR, "min_lr": cfg.LR / 25 / 10000,
                "weight_decay": cfg.WEIGHT_DECAY, "eps": 1e-8,
                "base_momentum": 0.85, "max_momentum": 0.95, "betas": (beta1, 0.999),
                "amsgrad": False, "maximize": False, "capturable": False,
                "differentiable": False, "foreach": None, "fused": None}]},
        "scheduler": {
            "total_steps": 20000, "last_epoch": step, "_step_count": step + 1,
            "_schedule_phases": [
                {"end_step": 1999.0, "start_lr": "initial_lr", "end_lr": "max_lr",
                 "start_momentum": "max_momentum", "end_momentum": "base_momentum"},
                {"end_step": 19999, "start_lr": "max_lr", "end_lr": "min_lr",
                 "start_momentum": "base_momentum", "end_momentum": "max_momentum"}],
            "_anneal_func_type": "cos", "cycle_momentum": True, "use_beta1": True,
            "base_lrs": [cfg.LR / 25], "_last_lr": [lr]},
        "rng_state": {"python": "opaque metadata", "numpy": "opaque metadata",
                      "torch": "opaque metadata", "cuda": ["opaque metadata"]},
    }


def run_files(cfg, omit=()):
    Path(cfg.SAVE_DIR).mkdir(parents=True)
    Path(cfg.LOG_DIR).mkdir(parents=True)
    values = {
        "CONFIG_PATH": json.dumps(dict(cfg.to_dict(), config_fingerprint=entry.fingerprint(cfg))),
        "STATUS_PATH": json.dumps(dict(state="paused", step=500, experiment=cfg.EXP_NAME,
                                      fingerprint=entry.fingerprint(cfg))),
        "SAVE_PATH": "opaque placeholder; never loaded",
        "BEST_PATH": "opaque placeholder; never loaded", "METRICS_PATH": "{}\n",
    }
    for field, value in values.items():
        if field not in omit:
            Path(getattr(cfg, field)).write_text(value, encoding="utf-8")


def function(name):
    tree = ast.parse((ROOT / "train_proposal_as_receiver.py").read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class ConfigTests(unittest.TestCase):
    def test_fixed_arms_seeds_and_recipe(self):
        self.assertEqual(ARMS, ("full_control", "as_receiver"))
        self.assertEqual(SEEDS, (20261030, 20261031))
        expected = dict(total_steps=20000, batch_size=48, lr=2e-5, weight_decay=1e-5,
                        warmup_pct=0.1, grad_clip=0.5, val_interval=500, val_batch=64,
                        stage1_snr=(0.0, 25.0), stage2_snr=(0.0, 25.0), stage3_snr=(0.0, 25.0),
                        downlink_snr_db=25.0, val_snr_list=(0, 5, 10, 15, 20, 25),
                        ph1_w_dir=0.0, ph1_w_mse=0.0, ph2_w_dir=0.0, ph2_w_mse=0.0,
                        best_metric="mean_all", data_cache_size=1)
        saved = asdict(RECIPE)
        for key, value in expected.items():
            self.assertEqual(saved[key], value, key)
        for arm in ARMS:
            for seed in SEEDS:
                cfg = as_receiver_config(arm, seed)
                validate_config(cfg)
                self.assertEqual(cfg.SEED, seed)
                self.assertIs(cfg.recipe, RECIPE)

    def test_source_and_model_dimensions(self):
        for arm in ARMS:
            cfg = as_receiver_config(arm, SEEDS[0])
            self.assertEqual(cfg.source_identity["checkpoint_sha256"],
                "e21ea1feae595e0898b6be724f7062b05c09b2d1ef03fe9ffec7ac91869064bf")
            self.assertEqual((cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS, cfg.D_TOT, cfg.NUM_SUBBANDS),
                             (4, 32, 256, 256, 4))
            self.assertEqual(cfg.ALLOCATION_GRID, [(0, 256)])
            self.assertEqual((cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED), (64, 256))
            self.assertEqual(cfg.architecture_metadata["ue_ffn_dim"], 2048)
            self.assertEqual(cfg.architecture_metadata["running_pwr_shape"], [1])
            self.assertEqual(cfg.architecture_metadata["cfg_idx"], 0)
            self.assertEqual(cfg.AS_RECEIVER_LAMBDA_FLOOR, 1e-6)
            self.assertEqual(cfg.AS_RECEIVER_SOLVE_DTYPE, "complex64")

    def test_data_protocol_identical_between_arms_and_distinct_between_seeds(self):
        protocols = []
        for seed in SEEDS:
            full, receiver = [as_receiver_config(arm, seed) for arm in ARMS]
            self.assertEqual(full.channel_data_protocol, receiver.channel_data_protocol)
            self.assertNotEqual(entry.fingerprint(full), entry.fingerprint(receiver))
            for cfg in (full, receiver):
                obj = make_channel_data(cfg)
                self.assertEqual(channel_data_api().validate_channel_data(cfg, obj), cfg.channel_data_protocol)
                self.assertEqual(obj.seed_for("training", 500), make_channel_data(full).seed_for("training", 500))
            protocols.append(full.channel_data_protocol)
        self.assertNotEqual(*protocols)

    def test_all_four_paths_are_distinct_and_no_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "screen"
            configs = [as_receiver_config(arm, seed, root) for seed in SEEDS for arm in ARMS]
            for field in ("EXP_NAME", "SAVE_DIR", "LOG_DIR"):
                self.assertEqual(len({getattr(cfg, field) for cfg in configs}), 4)
            self.assertFalse(root.exists())

    def test_legacy_recipe_registry_and_source_config_not_mutated(self):
        before = deepcopy(TRAINING_RECIPES)
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        self.assertEqual(before, TRAINING_RECIPES)
        self.assertNotIn(RECIPE.name, TRAINING_RECIPES)
        self.assertEqual(cfg.source_identity["loaded_checkpoint_config"]["training_recipe"]["name"], "k4_n256_fb_robust_ft")

    def test_unknown_arm_seed_and_mutated_recipe_are_rejected(self):
        for arm, seed in (("as_direct", SEEDS[0]), ("as_receiver", 42), ("full_control", True)):
            with self.assertRaises(ValueError):
                as_receiver_config(arm, seed)
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        with self.assertRaises(ValueError):
            validate_config(replace(cfg, recipe=replace(cfg.recipe, lr=1e-4)))

    def test_data_protocol_is_in_fingerprint_and_cannot_be_silently_changed(self):
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        original = entry.fingerprint(cfg)
        cfg.channel_data_protocol["base_seed"] += 1
        self.assertNotEqual(original, entry.fingerprint(cfg))
        with self.assertRaises(ValueError):
            validate_config(cfg)


class ResumeTests(unittest.TestCase):
    def test_complete_schema_at_500_preserves_twenty_thousand_scheduler(self):
        for arm in ARMS:
            cfg = as_receiver_config(arm, SEEDS[0])
            payload = metadata(cfg)
            entry.validate_resume_payload(payload, cfg)
            self.assertEqual(payload["scheduler"]["last_epoch"], 500)
            self.assertEqual(payload["scheduler"]["total_steps"], 20000)

    def test_completed_only_allowed_for_explicit_completion_audit(self):
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        payload = metadata(cfg, step=20000)
        entry.validate_resume_payload(payload, cfg, allow_completed=True)
        with self.assertRaises(ValueError):
            entry.validate_resume_payload(payload, cfg)

    def test_missing_resume_fields_rejected(self):
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        for key in ("config", "config_fingerprint", "state", "optimizer", "scheduler",
                    "rng_state", "step", "best_metric", "allocation_grid"):
            payload = metadata(cfg)
            del payload[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_resume_payload(payload, cfg)

    def test_cross_arm_seed_and_missing_receiver_are_rejected(self):
        receiver = as_receiver_config("as_receiver", SEEDS[0])
        for other in (as_receiver_config("full_control", SEEDS[0]), as_receiver_config("as_receiver", SEEDS[1])):
            with self.assertRaises(ValueError):
                entry.validate_resume_payload(metadata(other), receiver)
        payload = metadata(receiver)
        del payload["state"]["as_receiver.output_proj.weight"]
        with self.assertRaises(ValueError):
            entry.validate_resume_payload(payload, receiver)

    def test_source_identity_and_complete_config_mismatch_rejected(self):
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        for location in ("source", "protocol", "recipe"):
            payload = metadata(cfg)
            if location == "source":
                payload["config"]["source_identity"]["checkpoint_sha256"] = "0" * 64
            elif location == "protocol":
                payload["config"]["channel_data_protocol"]["base_seed"] += 1
            else:
                payload["config"]["training_recipe"]["total_steps"] = 5000
            with self.subTest(location=location), self.assertRaises(ValueError):
                entry.validate_resume_payload(payload, cfg)

    def test_scheduler_step_or_budget_reset_rejected(self):
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        for key, value in (("last_epoch", 0), ("total_steps", 500), ("_step_count", 1)):
            payload = metadata(cfg)
            payload["scheduler"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_resume_payload(payload, cfg)

    def test_optimizer_or_rng_incomplete_rejected(self):
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        for kind in ("lr", "groups", "state", "cuda", "torch"):
            payload = metadata(cfg)
            if kind == "lr":
                payload["optimizer"]["param_groups"][0]["lr"] *= 2
            elif kind == "groups":
                payload["optimizer"]["param_groups"] *= 2
            elif kind == "state":
                payload["optimizer"]["state"] = {}
            else:
                payload["rng_state"][kind] = None
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                entry.validate_resume_payload(payload, cfg)

    def test_best_can_be_earlier_but_metric_must_match_latest(self):
        cfg = as_receiver_config("as_receiver", SEEDS[0])
        latest = metadata(cfg, 1000)
        best = metadata(cfg, 500)
        entry.validate_best_metadata(best, latest, cfg)
        best["best_metric"] += 0.1
        with self.assertRaises(ValueError):
            entry.validate_best_metadata(best, latest, cfg)


class SafetyAndSourceTests(unittest.TestCase):
    def test_lock_is_exclusive_and_cleans_up_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "screen"
            cfg = as_receiver_config("as_receiver", SEEDS[0], root)
            path = root / 'locks' / (cfg.EXP_NAME + '.lock')
            with self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                with entry.run_lock(cfg, allowed_root=root):
                    self.assertEqual(json.loads(path.read_text())['arm'], 'as_receiver')
                    with self.assertRaises(FileExistsError):
                        with entry.run_lock(cfg, allowed_root=root):
                            self.fail('Concurrent lock unexpectedly admitted')
                    raise RuntimeError('synthetic stop')
            self.assertFalse(path.exists())

    def test_unknown_lock_is_not_removed_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'screen'
            cfg = as_receiver_config('as_receiver', SEEDS[0], root)
            path = root / 'locks' / (cfg.EXP_NAME + '.lock')
            path.parent.mkdir(parents=True)
            path.write_text('pid=123\n')
            with self.assertRaises((ValueError, RuntimeError)):
                with entry.run_lock(cfg, resume=True, allowed_root=root):
                    self.fail('Unknown stale lock unexpectedly recovered')
            self.assertEqual(path.read_text(), 'pid=123\n')

    def test_foreign_host_lock_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'screen'
            cfg = as_receiver_config('as_receiver', SEEDS[0], root)
            path = root / 'locks' / (cfg.EXP_NAME + '.lock')
            path.parent.mkdir(parents=True)
            saved = dict(schema='as_receiver_lock_v1', hostname='not-this-host',
                         experiment=cfg.EXP_NAME, arm=cfg.arm, seed=cfg.SEED,
                         pid=123, token='old')
            path.write_text(json.dumps(saved))
            with self.assertRaises(RuntimeError):
                entry._recover_stale_lock(path, cfg, {'token': 'new'})
            self.assertEqual(json.loads(path.read_text()), saved)

    def test_explicit_posix_dead_owner_recovery_preserves_old_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'screen'
            cfg = as_receiver_config('as_receiver', SEEDS[0], root)
            path = root / 'owner.lock'
            root.mkdir()
            saved = dict(schema='as_receiver_lock_v1', hostname=entry.platform.node(),
                         experiment=cfg.EXP_NAME, arm=cfg.arm, seed=cfg.SEED,
                         pid=123456789, token='old')
            path.write_text(json.dumps(saved))
            with mock.patch.object(entry.os, 'name', 'posix'), mock.patch.object(
                    entry.os, 'kill', side_effect=ProcessLookupError) as check:
                entry._recover_stale_lock(path, cfg, {'token': 'new'})
                check.assert_called_once_with(123456789, 0)
            self.assertEqual(json.loads(path.read_text()), {'token': 'new'})
            archives = list(root.glob('*.stale_*.json'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(json.loads(archives[0].read_text()), saved)
            self.assertFalse(path.with_suffix('.lock.recovery').exists())

    def test_live_owner_is_never_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'screen'
            cfg = as_receiver_config('as_receiver', SEEDS[0], root)
            path = root / 'owner.lock'
            root.mkdir()
            saved = dict(schema='as_receiver_lock_v1', hostname=entry.platform.node(),
                         experiment=cfg.EXP_NAME, arm=cfg.arm, seed=cfg.SEED,
                         pid=123456789, token='old')
            path.write_text(json.dumps(saved))
            with mock.patch.object(entry.os, 'name', 'posix'), mock.patch.object(entry.os, 'kill'):
                with self.assertRaisesRegex(RuntimeError, 'still alive'):
                    entry._recover_stale_lock(path, cfg, {'token': 'new'})
            self.assertEqual(json.loads(path.read_text()), saved)
            self.assertEqual(list(root.glob('*.stale_*.json')), [])

    def test_lock_cleanup_never_removes_another_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'screen'
            cfg = as_receiver_config('as_receiver', SEEDS[0], root)
            path = root / 'locks' / (cfg.EXP_NAME + '.lock')
            with self.assertRaisesRegex(RuntimeError, 'ownership changed'):
                with entry.run_lock(cfg, allowed_root=root):
                    path.write_text(json.dumps({'token': 'other-owner'}))
            self.assertEqual(json.loads(path.read_text())['token'], 'other-owner')

    def test_fresh_run_rejects_existing_directory_without_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "screen"
            cfg = as_receiver_config("as_receiver", SEEDS[0], root)
            entry.check_run_state(cfg, resume=False, allowed_root=root)
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                entry.check_run_state(cfg, resume=False, allowed_root=root)
            self.assertTrue(Path(cfg.SAVE_DIR).is_dir())

    def test_resume_requires_sidecars_and_complete_source_matching_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "screen"
            cfg = as_receiver_config("as_receiver", SEEDS[0], root)
            with self.assertRaises(FileNotFoundError):
                entry.check_run_state(cfg, resume=True, allowed_root=root)
            run_files(cfg)
            entry.check_run_state(cfg, resume=True, allowed_root=root)
            saved = json.loads(Path(cfg.CONFIG_PATH).read_text())
            saved["arm"] = "full_control"
            Path(cfg.CONFIG_PATH).write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                entry.check_run_state(cfg, resume=True, allowed_root=root)

    def test_wrong_dedicated_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = as_receiver_config("as_receiver", SEEDS[0], Path(directory) / "screen")
            with self.assertRaises(ValueError):
                entry.validate_output_paths(cfg)

    def test_factory_migration_and_rng_reset_precede_trainer_optimizer(self):
        factory = function("make_model_factory")
        calls = [node for node in ast.walk(factory) if isinstance(node, ast.Call)]
        initialize = next(node.lineno for node in calls if ast.unparse(node.func) == "initialize_model")
        seed = next(node.lineno for node in calls if ast.unparse(node.func) == "set_seed")
        self.assertLess(initialize, seed)
        self.assertNotIn("optim.AdamW", [ast.unparse(node.func) for node in calls])
        self.assertIn("if fresh:", ast.unparse(factory))
        trainer = function("run_training")
        call = next(node for node in ast.walk(trainer) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "train_proposal")
        self.assertEqual({kw.arg for kw in call.keywords}, {"model_factory", "channel_data"})
        options = call.args[1]
        self.assertIsNone(ast.literal_eval(next(kw.value for kw in options.keywords if kw.arg == "init_checkpoint")))

    def test_new_receiver_initialization_is_seeded_before_constructor(self):
        calls = [node for node in ast.walk(function("run_training")) if isinstance(node, ast.Call)]
        seeded = next(node.lineno for node in calls if ast.unparse(node.func) == "set_seed")
        construct = next(node.lineno for node in calls if ast.unparse(node.func) == "cls")
        self.assertLess(seeded, construct)

    def test_resume_path_does_not_migrate_or_reset_receiver(self):
        node = function("initialize_model")
        condition = next(n for n in node.body if isinstance(n, ast.If)
                         and ast.unparse(n.test) == "resume_state is not None")
        calls = [n for n in ast.walk(ast.Module(body=condition.body, type_ignores=[])) if isinstance(n, ast.Call)]
        self.assertEqual([ast.unparse(n.func) for n in calls], ["model.load_state_dict"])
        self.assertTrue(ast.literal_eval(next(kw.value for kw in calls[0].keywords if kw.arg == "strict")))

    def test_no_old_training_configuration_default_mutation(self):
        source = (ROOT / "train_proposal_as_receiver.py").read_text(encoding="utf-8")
        self.assertNotIn("strict=False", source)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("TRAINING_RECIPES[", source)
        self.assertNotIn("monkey", source)


class LightweightCliTests(unittest.TestCase):
    def cli(self, *args):
        code = """
import os,runpy,sys
script=sys.argv[1];sys.argv=[script]+sys.argv[2:];sys.path.insert(0,os.path.dirname(script))
def audit(event,args):
    if event=='os.mkdir': raise AssertionError('Unexpected directory write')
    if event=='open':
        path,mode,flags=args
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND): raise AssertionError('Unexpected file write')
        if isinstance(path,(str,bytes)) and os.fsdecode(path).endswith(('.pth','.pt','.ckpt')): raise AssertionError('Unexpected checkpoint read')
sys.addaudithook(audit)
try: runpy.run_path(script,run_name='__main__')
except SystemExit as e:
    if e.code not in (None,0): raise
assert not any(n.split('.')[0] in ('torch','numpy','sionna','models') for n in sys.modules)
assert 'utils' not in sys.modules
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-S", "-c", code,
                str(ROOT / "train_proposal_as_receiver.py"), *args],
                cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(list(Path(directory).iterdir()), [])
        return result

    def test_default_help_has_no_runtime_side_effect(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)

    def test_all_four_dry_runs_are_stdlib_only(self):
        for arm in ARMS:
            for seed in SEEDS:
                result = self.cli("--arm", arm, "--seed", str(seed), "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["scheduler_total_steps"], 20000)
                self.assertEqual(plan["config"]["experiment"]["seed"], seed)
                self.assertEqual(plan["config"]["arm"], arm)

    def test_invalid_explicit_execution_arguments_rejected_before_runtime(self):
        for args in (("--resume",), ("--dry-run",),
                     ("--arm", "as_receiver", "--seed", "20261030", "--stop-after", "501"),
                     ("--arm", "as_receiver", "--seed", "20261030", "--dry-run", "--train"),
                     ("--arm", "as_receiver", "--seed", "20261030", "--dry-run", "--resume")):
            with self.subTest(args=args):
                self.assertNotEqual(self.cli(*args).returncode, 0)


if __name__ == "__main__":
    unittest.main()
