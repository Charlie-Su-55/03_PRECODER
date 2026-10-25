"""Lightweight optional channel-data contracts; no models, training, or fakes.

The actual CUDA channel acceptance is diagnose_training_channel_data.py --run.
These tests import the utility directly so legacy utils/__init__.py does not
initialize GPU setup or import Sionna. Genuine-runtime availability is reported
separately and never replaced by a synthetic Sionna implementation.
"""

import ast
import copy
from dataclasses import replace
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "utils" / "replayable_channel_data.py"
DIAGNOSTIC = ROOT / "diagnose_training_channel_data.py"
TRAINER = ROOT / "main_train.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_api = load_file(UTILITY, "_replayable_channel_data_tests")


def trainer_tree():
    return ast.parse(TRAINER.read_text(encoding="utf-8-sig"))


def trainer_function(name):
    return next(node for node in trainer_tree().body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def extract_function(node, namespace=None):
    """Compile only inspected source, not trainer imports or model code."""
    node = copy.deepcopy(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    module = ast.fix_missing_locations(module)
    namespace = dict(namespace or {})
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace[node.name]


def old_config():
    from configs.train_config import ExperimentSpec, resolve_experiment
    cfg = resolve_experiment(ExperimentSpec(
        "proposal", "specialist", 4, 256, 256,
        "k4_n256_standard", 42, (0, 256)), output_root="runs")
    return replace(cfg, recipe=replace(cfg.recipe, batch_size=2, val_batch=2))


class DeclaredConfig:
    """A local test-only wrapper; the real TrainConfig is not changed."""

    def __init__(self, original=None, protocol=None):
        self.original = original if original is not None else old_config()
        self.declared = copy.deepcopy(protocol)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def to_dict(self):
        result = self.original.to_dict()
        result["channel_data_protocol"] = copy.deepcopy(self.declared)
        return result


def make_data(**overrides):
    kwargs = dict(base_seed=20261028, k_users=4, nt=32, nsc=256,
                  batch_size=2, validation_batch_size=2,
                  carrier_freq=3.5e9, speed=1.0, cdl_model="C", device="cuda:0")
    kwargs.update(overrides)
    return data_api.ReplayableChannelData(**kwargs)


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True,
        ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


def safe_checkpoint(cfg):
    # Metadata fixture only; no tensors/checkpoint files/models are involved.
    return dict(config=cfg.to_dict(), config_fingerprint=fingerprint(cfg),
                step=500, best_metric=1.0, state={"fixture": 1},
                optimizer={"fixture": 1}, scheduler={"fixture": 1},
                rng_state={"fixture": 1})


def trainer_metadata_namespace():
    """Run actual metadata helper source, without importing the trainer runtime."""
    namespace = dict(Path=Path, sys=sys, json=json, hashlib=hashlib,
                     __file__=str(TRAINER))
    namespace['json_default'] = extract_function(trainer_function('json_default'), namespace)
    namespace['config_fingerprint'] = extract_function(trainer_function('config_fingerprint'), namespace)
    namespace['_channel_data_api'] = extract_function(trainer_function('_channel_data_api'), namespace)
    namespace['_validate_channel_data_options'] = extract_function(
        trainer_function('_validate_channel_data_options'), namespace)
    return namespace


class SeedAndIdentityTests(unittest.TestCase):
    def test_protocol_is_json_serializable_and_copy_safe(self):
        obj = make_data()
        first = obj.protocol
        self.assertEqual(json.loads(json.dumps(first)), first)
        first["test_only_mutation"] = True
        self.assertNotIn("test_only_mutation", obj.protocol)

    def test_identical_configuration_produces_identical_protocol_and_seeds(self):
        a, b = make_data(), make_data()
        self.assertEqual(a.protocol, b.protocol)
        for split, index in (("constructor", 0), ("validation", 0),
                             ("training", 1), ("training", 5000)):
            self.assertEqual(a.seed_for(split, index), b.seed_for(split, index))

    def test_training_seed_is_order_independent(self):
        obj = make_data()
        expected = {step: obj.seed_for("training", step) for step in (1, 2, 7, 5000)}
        obj.seed_for("validation")
        for step in (5000, 2, 7, 1):
            self.assertEqual(obj.seed_for("training", step), expected[step])

    def test_namespaces_do_not_overlap(self):
        obj = make_data()
        constructor = obj.seed_for("constructor")
        validation = obj.seed_for("validation")
        training = [obj.seed_for("training", step)
                    for step in (1, 2, 5000, 2**30 - 1)]
        self.assertEqual(constructor >> 30, 0)
        self.assertEqual(validation >> 30, 1)
        self.assertEqual({seed >> 30 for seed in training}, {2})
        self.assertEqual(len(set([constructor, validation] + training)), 6)
        self.assertTrue(all(0 <= seed <= 2**32 - 1
                            for seed in [constructor, validation] + training))

    def test_training_steps_are_bijective_within_supported_namespace(self):
        obj = make_data()
        seeds = [obj.seed_for("training", step) for step in range(1, 5001)]
        self.assertEqual(len(set(seeds)), 5000)

    def test_invalid_split_and_step_are_rejected(self):
        obj = make_data()
        invalid = [("unknown", 0), ("training", 0), ("training", -1),
                   ("training", 2**30), ("training", True),
                   ("training", 1.0), ("training", "1"),
                   ("validation", 1), ("constructor", 1)]
        for split, step in invalid:
            with self.subTest(split=split, step=step):
                with self.assertRaises((ValueError, TypeError)):
                    obj.seed_for(split, step)

    def test_base_seed_bounds_and_types(self):
        for seed in (0, 2**32 - 1):
            self.assertIsInstance(make_data(base_seed=seed).protocol, dict)
        for seed in (-1, 2**32, True, False, None, 1.0, "42"):
            with self.subTest(seed=seed):
                with self.assertRaises((ValueError, TypeError)):
                    make_data(base_seed=seed)

    def test_physical_configuration_and_batch_sizes_are_in_identity(self):
        original = make_data().protocol
        self.assertEqual(original["physical"]["nt"], 32)
        for change in (dict(k_users=8), dict(nsc=128),
                       dict(batch_size=4), dict(validation_batch_size=4),
                       dict(carrier_freq=3.6e9), dict(speed=2.0),
                       dict(cdl_model="A"), dict(base_seed=7)):
            with self.subTest(change=change):
                self.assertNotEqual(original, make_data(**change).protocol)

    def test_unchanged_fixed_bs_array_rejects_other_nt(self):
        with self.assertRaisesRegex(ValueError, "Nt=32"):
            make_data(nt=16)

    def test_bad_training_step_is_rejected_before_runtime_import(self):
        before = set(sys.modules)
        with self.assertRaises((ValueError, TypeError)):
            make_data().training(0)
        imported = set(sys.modules) - before
        self.assertFalse(any(name.split('.')[0] in ('torch', 'numpy', 'sionna')
                             for name in imported))

    def test_no_python_hash_or_experiment_identity_in_seed_mapping(self):
        tree = ast.parse(UTILITY.read_text(encoding="utf-8-sig"))
        self.assertFalse(any(isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Name)
                             and node.func.id == "hash" for node in ast.walk(tree)))
        forbidden = {"arm", "model", "model_name", "checkpoint", "timestamp", "path"}
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(v) for v in value.values()), set())
            if isinstance(value, (tuple, list)):
                return set().union(*(keys(v) for v in value), set())
            return set()
        self.assertFalse(forbidden & keys(make_data().protocol))


class BindingAndResumeTests(unittest.TestCase):
    def test_legacy_none_binding_does_not_change_old_serialization(self):
        cfg = old_config()
        before = copy.deepcopy(cfg.to_dict())
        old_fp = fingerprint(cfg)
        self.assertIsNone(data_api.validate_channel_data(cfg, None))
        self.assertEqual(cfg.to_dict(), before)
        self.assertEqual(fingerprint(cfg), old_fp)
        self.assertNotIn("channel_data_protocol", cfg.to_dict())

    def test_protocol_binding_accepts_matching_declaration(self):
        obj = make_data()
        cfg = DeclaredConfig(protocol=obj.protocol)
        self.assertEqual(data_api.validate_channel_data(cfg, obj), obj.protocol)

    def test_data_without_declaration_is_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_channel_data(old_config(), make_data())

    def test_declaration_without_data_is_rejected(self):
        cfg = DeclaredConfig(protocol=make_data().protocol)
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_channel_data(cfg, None)

    def test_protocol_mismatch_is_rejected(self):
        obj = make_data()
        cfg = DeclaredConfig(protocol=make_data(base_seed=99).protocol)
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_channel_data(cfg, obj)

    def test_matching_declaration_cannot_hide_physical_or_batch_mismatch(self):
        for change in (dict(k_users=8), dict(nsc=128),
                       dict(batch_size=4), dict(validation_batch_size=4),
                       dict(carrier_freq=3.6e9), dict(speed=2.0)):
            with self.subTest(change=change):
                obj = make_data(**change)
                cfg = DeclaredConfig(protocol=obj.protocol)
                with self.assertRaises((ValueError, RuntimeError)):
                    data_api.validate_channel_data(cfg, obj)

    def test_mutated_namespace_registry_is_rejected(self):
        obj = make_data()
        protocol = obj.protocol
        protocol['seed_mapping']['namespaces']['validation'] = 2
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_channel_data(DeclaredConfig(protocol=protocol), obj)

    def test_data_identity_cannot_be_relabelled(self):
        obj = make_data()
        protocol = obj.protocol
        protocol['data_id'] = '0' * 64
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_channel_data(DeclaredConfig(protocol=protocol), obj)

    def test_protocol_is_in_existing_fingerprint_algorithm(self):
        cfg = old_config()
        native = extract_function(trainer_function("config_fingerprint"),
            dict(json=json, hashlib=hashlib, json_default=str))
        declared = DeclaredConfig(cfg, make_data().protocol)
        self.assertEqual(native(cfg), fingerprint(cfg))
        self.assertEqual(native(declared), fingerprint(declared))
        self.assertNotEqual(native(cfg), native(declared))
        changed = DeclaredConfig(cfg, make_data(base_seed=99).protocol)
        self.assertNotEqual(native(declared), native(changed))

    def test_matching_complete_resume_is_accepted_without_mutation(self):
        cfg = DeclaredConfig(protocol=make_data().protocol)
        checkpoint = safe_checkpoint(cfg)
        before = copy.deepcopy(checkpoint)
        data_api.validate_resume_config(checkpoint, cfg)
        self.assertEqual(checkpoint, before)

    def test_resume_requires_config_and_fingerprint(self):
        cfg = DeclaredConfig(protocol=make_data().protocol)
        for absent in ("config", "config_fingerprint", "state", "optimizer",
                       "scheduler", "rng_state", "step", "best_metric"):
            with self.subTest(absent=absent):
                checkpoint = safe_checkpoint(cfg)
                del checkpoint[absent]
                with self.assertRaises((ValueError, RuntimeError)):
                    data_api.validate_resume_config(checkpoint, cfg)

    def test_resume_rejects_matching_fingerprint_but_changed_full_config(self):
        cfg = DeclaredConfig(protocol=make_data().protocol)
        checkpoint = safe_checkpoint(cfg)
        checkpoint["config"]["training_recipe"]["lr"] *= 2
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_resume_config(checkpoint, cfg)

    def test_resume_rejects_matching_config_but_wrong_fingerprint(self):
        cfg = DeclaredConfig(protocol=make_data().protocol)
        checkpoint = safe_checkpoint(cfg)
        checkpoint["config_fingerprint"] = "not-the-current-fingerprint"
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_resume_config(checkpoint, cfg)

    def test_legacy_checkpoint_is_not_upgraded_to_protocol(self):
        cfg = DeclaredConfig(protocol=make_data().protocol)
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_resume_config(safe_checkpoint(old_config()), cfg)

    def test_checkpoint_protocol_mismatch_is_rejected(self):
        cfg = DeclaredConfig(protocol=make_data().protocol)
        other = DeclaredConfig(protocol=make_data(base_seed=99).protocol)
        with self.assertRaises((ValueError, RuntimeError)):
            data_api.validate_resume_config(safe_checkpoint(other), cfg)


class TrainerIntegrationTests(unittest.TestCase):
    def test_legacy_none_precheck_never_requires_paths_or_runtime(self):
        # Deliberately no path/device/optimizer attributes: legacy None exits
        # before touching them. No Torch/Sionna stand-ins are supplied.
        namespace = trainer_metadata_namespace()
        cfg = SimpleNamespace(to_dict=lambda: {})
        before = set(sys.modules)
        result = namespace['_validate_channel_data_options'](cfg, object(), None)
        self.assertIsNone(result)
        imported = set(sys.modules) - before
        self.assertFalse(any(name.split('.')[0] in ('torch', 'numpy', 'sionna')
                             for name in imported))

    def test_actual_entry_rejects_bad_binding_before_any_runtime_or_output(self):
        namespace = trainer_metadata_namespace()
        entry = extract_function(trainer_function('train_proposal'), namespace)
        # No torch, setup_gpu, model_factory, or optimizer is present in this
        # namespace. Reaching them would fail this test rather than run them.
        with tempfile.TemporaryDirectory() as directory:
            original = replace(old_config(), output_root=Path(directory) / 'run')
            obj = make_data()
            cfg = DeclaredConfig(original, obj.protocol)
            mismatch = DeclaredConfig(original, make_data(base_seed=99).protocol)
            options = SimpleNamespace(resume=False)
            for bound_cfg, data in ((original, obj), (cfg, None), (mismatch, obj)):
                with self.subTest(config=type(bound_cfg).__name__, data=data is not None):
                    with self.assertRaises((ValueError, RuntimeError)):
                        entry(bound_cfg, options, channel_data=data)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_valid_new_binding_precheck_is_read_only(self):
        check = trainer_metadata_namespace()['_validate_channel_data_options']
        with tempfile.TemporaryDirectory() as directory:
            original = replace(old_config(), output_root=Path(directory) / 'run')
            obj = make_data()
            cfg = DeclaredConfig(original, obj.protocol)
            self.assertIsNone(check(cfg, SimpleNamespace(resume=False), obj))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_new_protocol_resume_requires_existing_latest_before_output(self):
        check = trainer_metadata_namespace()['_validate_channel_data_options']
        with tempfile.TemporaryDirectory() as directory:
            original = replace(old_config(), output_root=Path(directory) / 'run')
            obj = make_data()
            cfg = DeclaredConfig(original, obj.protocol)
            with self.assertRaises(FileNotFoundError):
                check(cfg, SimpleNamespace(resume=True), obj)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_real_loader_rejects_unsafe_new_metadata_before_loading_state(self):
        load = extract_function(trainer_function('load_latest_checkpoint'),
                                trainer_metadata_namespace())
        cfg = DeclaredConfig(protocol=make_data().protocol)
        for absent in ('config_fingerprint', 'config', 'optimizer', 'rng_state'):
            checkpoint = safe_checkpoint(cfg)
            del checkpoint[absent]
            with self.subTest(absent=absent):
                # No Tensor/Model/Optimizer is created. Touching these opaque
                # objects would error if validation were incorrectly delayed.
                with self.assertRaises((ValueError, RuntimeError)):
                    load(cfg, object(), object(), object(), None,
                         validated_checkpoint=checkpoint)

    def test_optional_interface_defaults_to_none(self):
        node = trainer_function("train_proposal")
        arguments = dict(zip([a.arg for a in node.args.kwonlyargs], node.args.kw_defaults))
        self.assertIn("channel_data", arguments)
        self.assertIsNone(ast.literal_eval(arguments["channel_data"]))

    def test_legacy_data_calls_and_custom_calls_both_remain(self):
        node = trainer_function("train_proposal")
        call_names = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]
        for name in ("build_channel_generator", "generate_validation_set",
                     "generate_training_batch", "channel_data.prepare",
                     "channel_data.validation", "channel_data.training"):
            self.assertIn(name, call_names)
        conditional = [n for n in ast.walk(node)
                       if isinstance(n, (ast.If, ast.IfExp))
                       and ast.unparse(n.test) == "channel_data is None"]
        self.assertGreaterEqual(len(conditional), 2)





class LightweightCliTests(unittest.TestCase):
    def _cli(self, *arguments):
        code = """
import os, runpy, sys
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
def audit(event, args):
    if event == 'os.mkdir':
        raise AssertionError('Lightweight command attempted mkdir')
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('Lightweight command attempted a file write')
        if isinstance(path, (str, bytes)) and os.fsdecode(path).lower().endswith(('.pth','.pt','.ckpt')):
            raise AssertionError('Lightweight command attempted checkpoint read')
sys.addaudithook(audit)
try:
    runpy.run_path(script, run_name='__main__')
except SystemExit as error:
    if error.code not in (None, 0):
        raise
assert 'utils' not in sys.modules
assert not any(name.split('.')[0] in ('torch', 'numpy', 'sionna', 'models') for name in sys.modules)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-S", "-c", code,
                str(DIAGNOSTIC), *arguments], cwd=directory,
                capture_output=True, text=True, timeout=30)
            self.assertEqual(list(Path(directory).iterdir()), [])
        return result

    def test_default_help_has_no_dependencies_or_side_effects(self):
        result = self._cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--run", result.stdout)

    def test_dry_run_is_stdlib_only_and_no_write(self):
        result = self._cli("--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        self.assertIsInstance(plan, dict)

    def test_run_flags_are_mutually_exclusive(self):
        self.assertNotEqual(self._cli("--dry-run", "--run").returncode, 0)

    def test_utility_import_and_construction_are_stdlib_only(self):
        code = """
import importlib.util, pathlib, sys
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('_data_import_only', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
obj = module.ReplayableChannelData(base_seed=42, k_users=4, nt=32, nsc=256,
    batch_size=2, validation_batch_size=2, device='cuda:0')
obj.seed_for('training', 5000)
assert 'utils' not in sys.modules
assert not any(name.split('.')[0] in ('torch', 'numpy', 'sionna', 'models') for name in sys.modules)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-S", "-c", code,
                str(UTILITY)], cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_diagnostic_output_refuses_overwrite(self):
        entry = load_file(DIAGNOSTIC, "_training_data_diagnostic_tests")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            first = entry.create_output(root=root, run_id="unit_replay")
            self.assertTrue(first.is_dir())
            with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
                entry.create_output(root=root, run_id="unit_replay")


class GenuineRuntimeBoundaryTests(unittest.TestCase):
    def test_genuine_sionna_version_requirement(self):
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "2.0.1":
            self.skipTest(f"Real Sionna 2.0.1/CUDA acceptance not run: installed {version!r}; no dependency installed or faked.")
        self.assertEqual(version, "2.0.1")
        # This intentionally does not instantiate a generator or launch CUDA.
        # Actual channel replay is explicit diagnose_training_channel_data --run.


if __name__ == "__main__":
    unittest.main()
