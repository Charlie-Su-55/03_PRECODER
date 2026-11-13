"""New ownership, source pins and independent paired streams."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_ul_propagation_config as cfg
import ul_propagation_runtime as runtime


class PropagationRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'propagation'
        patch = mock.patch.object(runtime, 'OUTPUT_ROOT', self.output)
        patch.start()
        self.addCleanup(patch.stop)

    def test_real_core_checks_keep_all_39_prior_sources_pinned(self):
        core = runtime.core_identity()
        runtime.guard_core(core)
        self.assertEqual(len(cfg.PINNED_UL_CORE['sha256']), 39)
        with self.assertRaises(ValueError):
            runtime.guard_core(dict(core, schema='wrong namespace'))

    def test_owned_records_reject_historical_roots_and_failed_replace_preserves_bytes(self):
        path = self.output/'batch'/'record.json'
        runtime.atomic_json(path, dict(value=1), exclusive=True)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            runtime.atomic_json(path, dict(value=2), exclusive=True)
        with mock.patch.object(runtime.os, 'replace', side_effect=OSError('replace failure')):
            with self.assertRaises(OSError):
                runtime.atomic_json(path, dict(value=3))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])
        old = self.root/'old_outputs'
        old.mkdir()
        link = self.output/'link'
        link.symlink_to(old, target_is_directory=True)
        for target in (self.output, old/'file.json', link/'file.json'):
            with self.assertRaises(ValueError):
                runtime.atomic_json(target, dict(value=4))
        self.assertEqual(list(old.iterdir()), [])

    def test_random_streams_replay_and_reject_bad_addresses(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Real PyTorch unavailable')
        first = runtime.batch_random(1, 2, 'cpu')
        second = runtime.batch_random(1, 2, 'cpu')
        other = runtime.batch_random(1, 3, 'cpu')
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(torch.equal(first[0], other[0]))
        for batch, gamma in ((-1, 0), (cfg.BATCHES, 0), (0, len(cfg.GAMMAS)), (True, 0)):
            with self.assertRaises(ValueError):
                runtime.batch_random(batch, gamma, 'cpu')


if __name__ == '__main__':
    unittest.main()
