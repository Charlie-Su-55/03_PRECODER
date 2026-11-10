"""Real CPU checkpoint IO; no CUDA, Sionna or full model training."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from configs import twc_v1_k8_icc_aligned_config as cfg
import train_twc_v1_k8_icc_aligned as runtime

try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


@unittest.skipIf(torch is None or np is None, 'Real PyTorch/NumPy unavailable; no fake dependency')
class CheckpointSaveTests(unittest.TestCase):
    def test_real_roundtrip_replaces_existing_checkpoint_for_all_components(self):
        external_rng = runtime.capture_rng()
        self.addCleanup(runtime.restore_rng, external_rng)
        core = runtime.core_identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(runtime, 'OUTPUT_ROOT', root):
                for component in ('ce',) + cfg.ARMS:
                    with self.subTest(component=component):
                        directory = root / 'training' / component
                        directory.mkdir(parents=True)
                        target = directory / 'checkpoint_latest.pt'
                        torch.save({'old': torch.tensor([-1.])}, target)
                        model = torch.nn.Linear(2, 1)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY)
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                        for _ in range(100):
                            optimizer.zero_grad()
                            model(torch.ones(1, 2)).square().mean().backward()
                            optimizer.step()
                            scheduler.step()
                        stats = None if component == 'ce' else torch.nn.Linear(1, 1)
                        record = runtime.save_checkpoint(directory, model, stats, optimizer,
                            scheduler, component, 100, SimpleNamespace(protocol={'cpu_test': True}),
                            core, None, 0., 1., .01)
                        payload = torch.load(target, map_location='cpu', weights_only=False)
                        self.assertEqual(payload['component'], component)
                        self.assertEqual(payload['step'], 100)
                        self.assertEqual(payload['format'], cfg.NAMESPACE)
                        self.assertEqual(payload['scheduler']['last_epoch'], 100)
                        for key, value in model.state_dict().items():
                            self.assertTrue(torch.equal(payload['model'][key], value))
                        self.assertEqual(record['checkpoint_sha256'], runtime.sha256(target))
                        self.assertEqual({p.name for p in directory.iterdir()},
                            {'checkpoint_latest.pt', 'manifest.json', 'boundary_00100.json'})

    def test_real_serialization_failure_preserves_existing_bytes_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'checkpoint_latest.pt'
            runtime.atomic_torch_save({'value': torch.tensor([1., 2.])}, target)
            before = target.read_bytes()
            # A local lambda cannot be pickled: real torch.save fails after opening
            # its temporary archive. No mock replaces the serializer.
            with self.assertRaises((AttributeError, TypeError)):
                runtime.atomic_torch_save({'value': torch.zeros(2), 'bad': lambda: None}, target)
            self.assertEqual(target.read_bytes(), before)
            self.assertTrue(torch.equal(torch.load(target, weights_only=True)['value'],
                                        torch.tensor([1., 2.])))
            self.assertEqual(list(Path(temporary).iterdir()), [target])

    def test_replace_failure_preserves_existing_bytes_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'checkpoint_latest.pt'
            runtime.atomic_torch_save({'value': torch.ones(2)}, target)
            before = target.read_bytes()
            def fail_replace(source, destination):
                source = Path(source)
                self.assertEqual(source.parent, target.parent)
                self.assertFalse(source.name.startswith('.'))
                self.assertEqual(source.suffix, '.pt')
                self.assertTrue(torch.equal(torch.load(source, weights_only=True)['value'],
                                            torch.zeros(2)))
                raise OSError('replace failed')
            with mock.patch.object(runtime.os, 'replace', side_effect=fail_replace):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    runtime.atomic_torch_save({'value': torch.zeros(2)}, target)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(list(Path(temporary).iterdir()), [target])


if __name__ == '__main__':
    unittest.main()
