"""Actual CPU observation/recovery boundaries and owned-file protection."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_ul_diagnostics_config as cfg
import ul_diagnostics_runtime as runtime

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


class OwnedFilesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'diagnostics'
        patch = mock.patch.object(runtime, 'OUTPUT_ROOT', self.output)
        patch.start()
        self.addCleanup(patch.stop)

    def test_atomic_identity_and_failed_replace_preserve_existing_bytes(self):
        path = self.output/'attempt'/'record.json'
        runtime.atomic_json(path, dict(value=1), exclusive=True)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            runtime.atomic_json(path, dict(value=2), exclusive=True)
        with mock.patch.object(runtime.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                runtime.atomic_json(path, dict(value=3))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_owned_io_rejects_historical_roots_and_links(self):
        historical = self.root/'historical'
        historical.mkdir()
        self.output.mkdir()
        link = self.output/'linked'
        link.symlink_to(historical, target_is_directory=True)
        for path in (self.output, historical/'record.json', link/'record.json'):
            with self.assertRaises(ValueError):
                runtime.atomic_json(path, dict(value=1))
        self.assertEqual(list(historical.iterdir()), [])

    def test_actual_legacy_pin_and_new_core_reject_changes(self):
        identity = runtime.core_identity()
        runtime.guard_core(identity)
        changed = dict(identity, schema='different namespace')
        with self.assertRaises(ValueError):
            runtime.guard_core(changed)


@unittest.skipIf(torch is None, 'Real PyTorch unavailable; no fake tensor runtime')
class ActualRecoveryBoundaryTests(unittest.TestCase):
    def test_same_observation_prior_and_noise_for_both_recovery_conditions(self):
        phy = runtime.physics()
        generator = torch.Generator().manual_seed(604)
        raw = torch.randn(4, 8, 216, dtype=torch.complex64, generator=generator)
        H = torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator)
        gamma = torch.full((4,), 5.)
        noise, polarity = runtime.batch_random(0, 1, 'cpu')
        for arm in cfg.ARMS:
            stats = phy.ReceiverStatistics(phy.make_layout(arm),
                ce_errors=torch.full((6, 8, 256), .01)).eval()
            with torch.no_grad():
                n = stats.power_ema.shape[-1]
                stats.power_ema.copy_(torch.linspace(.1, 2., n).expand(8, -1))
                stats.ema_updates.fill_(20000)
            nominal = stats.nominal_power()
            wave = phy.build_waveform(raw, stats.layout, polarity)
            Y = phy.transmit(H, wave.x_grid, noise, gamma)
            observation = dict(wave=wave, Y=Y, prior=nominal,
                estimated=H*.97, ce_variance=stats.ce_variance(gamma))
            snapshots = {name: runtime.state_digest({name: value}) for name, value in
                dict(raw=raw, H=H, noise=noise, Y=Y, prior=nominal).items()}
            before_stats = runtime.state_digest(stats)
            with mock.patch.object(phy, 'recover', wraps=phy.recover) as actual:
                recovered, inputs = runtime.recover_variants(observation, H, stats.layout, gamma)
            self.assertEqual(actual.call_count, 2)
            for call in actual.call_args_list:
                self.assertIs(call.args[0], Y)
                self.assertIs(call.args[3], nominal)
            self.assertIs(actual.call_args_list[0].args[1], observation['estimated'])
            self.assertIs(actual.call_args_list[1].args[1], H)
            self.assertTrue(bool((inputs['true_ul_diagnostic'][1] == 0).all()))
            self.assertFalse(torch.allclose(wave.active.abs().square(), nominal[None].expand_as(wave.active.real)))
            for variant in cfg.VARIANTS:
                self.assertTrue(bool(torch.isfinite(recovered[variant]['soft']).all()))
                self.assertTrue(bool(torch.isfinite(recovered[variant]['reliability']).all()))
            for name, value in dict(raw=raw, H=H, noise=noise, Y=Y, prior=nominal).items():
                self.assertEqual(runtime.state_digest({name: value}), snapshots[name])
            self.assertEqual(runtime.state_digest(stats), before_stats)

    def test_random_stream_replay_pairing_and_invalid_addresses(self):
        first = runtime.batch_random(3, 2, 'cpu')
        second = runtime.batch_random(3, 2, 'cpu')
        other = runtime.batch_random(3, 3, 'cpu')
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(torch.equal(first[0], other[0]))
        for batch, gamma in ((-1, 0), (cfg.BATCHES, 0), (0, len(cfg.GAMMAS)), (True, 0)):
            with self.assertRaises(ValueError):
                runtime.batch_random(batch, gamma, 'cpu')

    def test_frozen_signature_rejects_training_or_grad_flags(self):
        modules = [nn.Linear(2, 2).eval().requires_grad_(False) for _ in range(3)]
        first = runtime.frozen_signature(*modules)
        with torch.no_grad():
            modules[0].weight.add_(.25)
        self.assertNotEqual(first, runtime.frozen_signature(*modules))
        modules[0].train()
        with self.assertRaises(ValueError):
            runtime.frozen_signature(*modules)
        modules[0].eval().requires_grad_(True)
        with self.assertRaises(ValueError):
            runtime.frozen_signature(*modules)


if __name__ == '__main__':
    unittest.main()
