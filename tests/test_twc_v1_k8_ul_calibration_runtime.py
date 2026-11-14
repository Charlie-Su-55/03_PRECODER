"""Real CPU replay, ownership and pinned-source/atomic record protections."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_ul_calibration_config as cfg
import ul_calibration_runtime as runtime


class ULCalibrationRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'calibration'
        patch = mock.patch.object(runtime, 'OUTPUT_ROOT', self.output)
        patch.start()
        self.addCleanup(patch.stop)

    def test_real_legacy_gate_rejects_changed_pin_and_wrong_new_identity(self):
        # Other new files are owned by parallel implementation tasks. The gate
        # under test still hashes and verifies all46 actual old source files.
        local = ('configs/twc_v1_k8_ul_calibration_config.py', 'ul_calibration_runtime.py')
        with mock.patch.object(cfg, 'CORE_FILES', local):
            core = runtime.core_identity()
            runtime.guard_core(core)
            with self.assertRaises(ValueError):
                runtime.guard_core(dict(core, schema='wrong study'))
            pins = copy.deepcopy(cfg.PINNED_PROPAGATION_CORE)
            first = next(iter(pins['sha256']))
            pins['sha256'][first] = '0'*64
            with mock.patch.object(cfg, 'PINNED_PROPAGATION_CORE', pins):
                with self.assertRaises(ValueError):
                    runtime.guard_core(core)

    def test_real_atomic_exclusive_write_and_replace_failure_preserve_prior_record(self):
        path = self.output/'batch'/'model_lock.json'
        runtime.atomic_json(path, dict(model='original'), exclusive=True)
        old = path.read_bytes()
        with self.assertRaises(FileExistsError):
            runtime.atomic_json(path, dict(model='overwrite'), exclusive=True)
        self.assertEqual(path.read_bytes(), old)
        with mock.patch.object(runtime.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                runtime.atomic_json(path, dict(model='new'))
        self.assertEqual(path.read_bytes(), old)
        with mock.patch.object(runtime.os, 'fsync', side_effect=OSError('sync failed')):
            with self.assertRaises(OSError):
                runtime.atomic_json(path, dict(model='new'))
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_owned_paths_reject_history_traversal_links_and_nonfinite_json(self):
        old = self.root/'historical'
        old.mkdir()
        self.output.mkdir()
        (self.output/'link').symlink_to(old, target_is_directory=True)
        for path in (self.output, old/'record.json', self.output/'..'/'outside.json',
                     self.output/'link'/'record.json'):
            with self.assertRaises(ValueError):
                runtime.atomic_json(path, dict(value=1))
        self.assertEqual(list(old.iterdir()), [])
        with self.assertRaises(ValueError):
            runtime.atomic_json(self.output/'bad.json', dict(value=float('nan')))
        self.assertFalse((self.output/'bad.json').exists())

    def test_real_nonblocking_lock_preserves_lock_file_and_releases_after_error(self):
        path = self.output/'controller.lock'
        with runtime.process_lock(path):
            with self.assertRaises(BlockingIOError):
                with runtime.process_lock(path):
                    self.fail('Concurrent lock must not be acquired')
        self.assertTrue(path.is_file())
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with runtime.process_lock(path):
                raise RuntimeError('fixture failure')
        with runtime.process_lock(path):
            self.assertTrue(path.is_file())

    def test_real_cpu_streams_replay_without_global_rng_or_role_aliasing(self):
        import torch
        state = torch.random.get_rng_state().clone()
        first = runtime.batch_random('calibration', 1, 2, 'cpu')
        second = runtime.batch_random('calibration', 1, 2, 'cpu')
        validation = runtime.batch_random('validation', 1, 2, 'cpu')
        other_gamma = runtime.batch_random('calibration', 1, 3, 'cpu')
        self.assertEqual(first[0].shape, (4, 32, 256))
        self.assertEqual(first[0].dtype, torch.complex64)
        self.assertEqual(first[1].shape, (4, 8))
        self.assertEqual(set(first[1].unique().tolist()), {-1., 1.})
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(torch.equal(first[0], validation[0]))
        self.assertFalse(torch.equal(first[0], other_gamma[0]))
        for role, batch, gamma in (('test', 0, 0), ([], 0, 0), ('validation', -1, 0),
                ('calibration', 64, 0), ('calibration', 0, 6), ('validation', True, 0)):
            with self.assertRaises(ValueError):
                runtime.batch_random(role, batch, gamma, 'cpu')


if __name__ == '__main__':
    unittest.main()
