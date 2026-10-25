"""Lightweight contracts and genuine dependency-gated RNG isolation checks.

Local command (no models, checkpoints, or channels):
    python -B -m unittest discover -s tests -p test_sionna_rng.py -v

The tiny state tests require the real Sionna 2.0.1 distribution. They never
substitute fake Sionna objects. The separate diagnostic --run command is the
explicit CUDA channel-replay acceptance test; this module does not call it.
"""

import ast
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "utils" / "sionna_rng.py"
DIAGNOSTIC = ROOT / "diagnose_sionna_rng.py"


def load_file(path, name):
    """Do not run utils/__init__.py, which initializes the legacy GPU setup."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rng = load_file(UTILITY, "_sionna_rng_test_utility")
entry = load_file(DIAGNOSTIC, "_sionna_rng_test_diagnostic")


def installed_sionna_version():
    try:
        return importlib.metadata.version("sionna")
    except importlib.metadata.PackageNotFoundError:
        return None


SIONNA_VERSION = installed_sionna_version()
REAL_DEPENDENCIES = SIONNA_VERSION == "2.0.1"
DEPENDENCY_REASON = (
    "Real RNG tests require Sionna 2.0.1; installed distribution is "
    + repr(SIONNA_VERSION)
    + ". No installation or fake dependency is used."
)


class LightweightContracts(unittest.TestCase):
    def test_seed_range_includes_boundaries(self):
        for seed in (0, 1, 20261028, 2**32 - 1):
            with self.subTest(seed=seed):
                self.assertEqual(rng.validate_seed(seed), seed)

    def test_seed_rejects_none_booleans_and_noninteger_values(self):
        for seed in (None, True, False, 1.0, "1", [], {}):
            with self.subTest(seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    rng.validate_seed(seed)

    def test_seed_rejects_out_of_range_integers(self):
        for seed in (-1, -(2**63), 2**32, 2**64):
            with self.subTest(seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    rng.validate_seed(seed)

    def test_imports_are_stdlib_only_and_do_not_execute_utils_package(self):
        # -S removes site packages. A hidden Torch/Sionna import would fail here.
        code = """
import importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1])
for relative, name in [('utils/sionna_rng.py', '_rng_light_import'),
                       ('diagnose_sionna_rng.py', '_diagnostic_light_import')]:
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
assert 'utils' not in sys.modules
assert not any(name.split('.')[0] in ('torch', 'numpy', 'sionna', 'models')
               for name in sys.modules)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-c", code, str(ROOT)],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def _cli(self, *arguments):
        code = """
import os, pathlib, runpy, sys
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
def audit(event, args):
    if event == 'os.mkdir':
        raise AssertionError('Lightweight CLI attempted a directory write')
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('Lightweight CLI attempted a file write')
        if isinstance(path, (str, bytes)) and os.fsdecode(path).lower().endswith(('.pth', '.pt', '.ckpt')):
            raise AssertionError('Lightweight CLI attempted to read weights')
sys.addaudithook(audit)
try:
    runpy.run_path(script, run_name='__main__')
except SystemExit as error:
    if error.code not in (None, 0):
        raise
assert not any(name.split('.')[0] in ('torch', 'numpy', 'sionna', 'models')
               for name in sys.modules)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-c", code, str(DIAGNOSTIC), *arguments],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
        return result

    def test_default_cli_displays_help_without_runtime(self):
        result = self._cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--run", result.stdout)

    def test_explicit_help_does_not_import_heavy_dependencies(self):
        result = self._cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout.lower())

    def test_dry_run_is_json_and_imports_no_runtime(self):
        result = self._cli("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertIsInstance(plan, dict)
        self.assertIn("20261028", json.dumps(plan))

    def test_dry_run_and_run_are_mutually_exclusive(self):
        result = self._cli("--dry-run", "--run")
        self.assertNotEqual(result.returncode, 0)

    def test_dry_run_rejects_invalid_seed_without_runtime(self):
        for seed in ("-1", str(2**32), "1.5"):
            with self.subTest(seed=seed):
                result = self._cli("--dry-run", "--seed", seed)
                self.assertNotEqual(result.returncode, 0)

    def test_output_is_exclusive_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            output = entry.create_output(root=root, run_id="unit_replay")
            self.assertTrue(output.is_dir())
            self.assertEqual(output.parent, root)
            marker = output / "existing.json"
            marker.write_text('{"preserve": true}', encoding="utf-8")
            with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
                entry.create_output(root=root, run_id="unit_replay")
            self.assertEqual(marker.read_text(encoding="utf-8"), '{"preserve": true}')

    def test_output_rejects_path_traversal(self):
        for run_id in (".", "..", "../escape", "nested/name", "nested\\name", "/absolute"):
            with self.subTest(run_id=run_id), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises((ValueError, RuntimeError)):
                    entry.create_output(root=Path(directory) / "outputs", run_id=run_id)

    def test_original_channel_pair_is_two_contiguous_snapshots(self):
        tree = ast.parse((ROOT / "utils" / "channel_generator.py").read_text(encoding="utf-8-sig"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "CDLChannelGenerator_Feedback")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "generate_batch_data")
        statements = [node for node in method.body
                      if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
        self.assertEqual(len(statements), 3)
        for node, target in zip(statements[:2], ("H_dl", "H_ul")):
            self.assertIsInstance(node, ast.Assign)
            self.assertEqual(node.targets[0].id, target)
            self.assertIsInstance(node.value, ast.Call)
            self.assertEqual(node.value.func.attr, "_generate_single_snapshot")
        self.assertIsInstance(statements[-1], ast.Return)

    def test_report_preserves_nonfinite_failure_as_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            report = dict(status='passed', passed=True, checks={'max_rel': float('inf')})
            saved = entry.write_report(Path(directory), report)
            self.assertFalse(saved['passed'])
            self.assertEqual(saved['status'], 'failed')
            self.assertIsNone(saved['checks']['max_rel'])
            self.assertEqual(json.loads((Path(directory) / 'report.json').read_text(encoding='utf-8')), saved)
            self.assertFalse(report['passed'])

    def test_report_refuses_to_overwrite_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            entry.write_report(output, dict(status='failed', passed=False, error='original failure'))
            before = (output / 'report.json').read_bytes()
            with self.assertRaises(FileExistsError):
                entry.write_report(output, dict(status='passed', passed=True))
            self.assertEqual(before, (output / 'report.json').read_bytes())

    @unittest.skipIf(REAL_DEPENDENCIES, "Current Sionna 2.0.1 is supported, not an unsupported-version fixture.")
    def test_actual_unsupported_or_missing_distribution_is_rejected(self):
        before = set(sys.modules)
        with self.assertRaisesRegex(RuntimeError, r"2\.0\.1"):
            rng.runtime()
        imported = set(sys.modules) - before
        self.assertFalse(any(name.split(".")[0] in ("torch", "numpy", "sionna")
                             for name in imported))


@unittest.skipUnless(REAL_DEPENDENCIES, DEPENDENCY_REASON)
class RealRngIsolation(unittest.TestCase):
    """Tiny genuine RNG draws, not channel/model/GPU profiling experiments."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.torch, cls.np, cls.config = rng.runtime()
        except RuntimeError as error:
            if isinstance(error.__cause__, ImportError):
                raise unittest.SkipTest("Real runtime dependency missing: " + str(error))
            raise

    def setUp(self):
        initial = rng.snapshot_rng_state()
        self.addCleanup(initial.restore)

    def draw(self):
        result = {
            "python": random.random(),
            "numpy": self.np.random.random(3).tolist(),
            "torch_cpu": self.torch.rand(3, device="cpu").tolist(),
            "private_python": self.config.py_rng.random(),
            "private_numpy": self.config.np_rng.random(3).tolist(),
            "private_torch_cpu": self.torch.rand(
                3, generator=self.config.torch_rng("cpu"), device="cpu").tolist(),
        }
        if self.torch.cuda.is_available():
            for index in range(self.torch.cuda.device_count()):
                device = "cuda:" + str(index)
                result["torch_" + device] = self.torch.rand(3, device=device).cpu().tolist()
                result["private_torch_" + device] = self.torch.rand(
                    3, device=device, generator=self.config.torch_rng(device)).cpu().tolist()
        return result

    def test_normal_exit_restores_all_streams_not_only_seed(self):
        self.config.seed = 175
        self.draw()
        before = rng.snapshot_rng_state()
        expected_next = self.draw()
        before.restore()
        with rng.isolated_sionna_seed(24):
            self.draw()
            self.draw()
        self.assertEqual(before.fingerprint(), rng.snapshot_rng_state().fingerprint())
        self.assertEqual(self.draw(), expected_next)

    def test_exception_exit_restores_all_streams(self):
        self.config.seed = 175
        self.draw()
        before = rng.snapshot_rng_state()
        with self.assertRaisesRegex(ValueError, "intentional"):
            with rng.isolated_sionna_seed(17):
                self.draw()
                raise ValueError("intentional")
        self.assertEqual(before.fingerprint(), rng.snapshot_rng_state().fingerprint())

    def test_nested_context_preserves_outer_progress(self):
        before = rng.snapshot_rng_state()
        with rng.isolated_sionna_seed(31):
            self.draw()
            middle = rng.snapshot_rng_state()
            expected_next = self.draw()
            middle.restore()
            with rng.isolated_sionna_seed(32):
                self.draw()
                self.draw()
            self.assertEqual(middle.fingerprint(), rng.snapshot_rng_state().fingerprint())
            self.assertEqual(self.draw(), expected_next)
        self.assertEqual(before.fingerprint(), rng.snapshot_rng_state().fingerprint())

    def test_seed_none_and_generator_object_identities_are_restored(self):
        self.config.seed = None
        self.draw()
        before = rng.snapshot_rng_state()
        private_python = self.config._py_rng
        private_numpy = self.config._np_rng
        registry = self.config._torch_rngs
        generators = dict(registry)
        with rng.isolated_sionna_seed(0):
            self.draw()
        self.assertIsNone(self.config.seed)
        self.assertIs(self.config._py_rng, private_python)
        self.assertIs(self.config._np_rng, private_numpy)
        self.assertIs(self.config._torch_rngs, registry)
        self.assertEqual(set(registry), set(generators))
        for device, generator in generators.items():
            self.assertIs(registry[device], generator)
        self.assertEqual(before.fingerprint(), rng.snapshot_rng_state().fingerprint())

    def test_seed_none_lazy_private_rng_state_is_preserved(self):
        self.config.seed = None
        before = rng.snapshot_rng_state()
        private_python, private_numpy = self.config._py_rng, self.config._np_rng
        with rng.isolated_sionna_seed(8):
            self.draw()
        self.assertIsNone(self.config.seed)
        self.assertIs(self.config._py_rng, private_python)
        self.assertIs(self.config._np_rng, private_numpy)
        self.assertEqual(before.fingerprint(), rng.snapshot_rng_state().fingerprint())

    def test_same_seed_replays_and_different_seed_differs(self):
        with rng.isolated_sionna_seed(44):
            first = self.draw()
        with rng.isolated_sionna_seed(45):
            second = self.draw()
        with rng.isolated_sionna_seed(44):
            replay = self.draw()
        self.assertEqual(first, replay)
        self.assertNotEqual(first, second)

    def test_fingerprint_is_json_serializable(self):
        self.draw()
        fingerprint = rng.snapshot_rng_state().fingerprint()
        encoded = json.dumps(fingerprint, allow_nan=False, sort_keys=True)
        self.assertEqual(json.loads(encoded), fingerprint)
        self.assertIn("seed", fingerprint)
        self.assertIn("private", fingerprint)
        self.assertIn("defaults", fingerprint)

    def test_first_sionna_import_preserves_existing_default_rng_states(self):
        code = r'''
import importlib.util, pathlib, pickle, random, sys
import numpy as np
import torch
assert 'sionna.phy' not in sys.modules
random.seed(85)
np.random.seed(86)
torch.manual_seed(87)
def defaults():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return (pickle.dumps(random.getstate(), protocol=4),
            pickle.dumps(np.random.get_state(), protocol=4),
            torch.get_rng_state().clone(), [value.clone() for value in cuda])
before = defaults()
spec = importlib.util.spec_from_file_location('_first_import_rng', pathlib.Path(sys.argv[1]))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.runtime()
after = defaults()
assert before[0] == after[0] and before[1] == after[1]
assert torch.equal(before[2], after[2])
assert len(before[3]) == len(after[3])
assert all(torch.equal(left, right) for left, right in zip(before[3], after[3]))
print('first Sionna import preserved existing default states')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(UTILITY)],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
