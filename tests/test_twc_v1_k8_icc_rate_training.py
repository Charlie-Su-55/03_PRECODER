"""Real CPU decoder checkpoints and boundary resume; no Sionna or full model."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from configs import twc_v1_k8_icc_rate_config as cfg
import train_twc_v1_k8_icc_rate as runtime

try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


@unittest.skipIf(torch is None or np is None, "Real PyTorch/NumPy unavailable; no fake dependency")
class TinyRateCheckpointTests(unittest.TestCase):
    def setUp(self):
        external = runtime.capture_rng()
        self.addCleanup(runtime.restore_rng, external)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'new_rate_outputs'
        for patcher in (mock.patch.object(runtime, 'OUTPUT_ROOT', self.output),
                        mock.patch.object(cfg, 'UPDATES', 200)):
            patcher.start()
            self.addCleanup(patcher.stop)
        torch.manual_seed(9286)
        codec = torch.nn.Module()
        codec.encoder = torch.nn.Linear(2, 3)
        codec.decoder = torch.nn.Linear(3, 2)
        ce = torch.nn.Linear(2, 2)
        stats = torch.nn.Module()
        stats.register_buffer('historical_power', torch.tensor([24., 24.]))
        stats.register_buffer('ema_updates', torch.tensor(20000))
        # A genuinely populated previous optimizer is source evidence only.
        # The rate optimizer below must start with no state from it.
        old_optimizer = torch.optim.AdamW(codec.decoder.parameters(), lr=1e-4)
        codec.decoder(torch.ones(2, 3)).square().mean().backward()
        old_optimizer.step()
        old_optimizer.zero_grad(set_to_none=True)
        self.assertTrue(old_optimizer.state)
        for module in (codec, ce, stats):
            module.eval().requires_grad_(False)
            for parameter in module.parameters():
                parameter.grad = None
        self.base = (codec, stats, ce)
        self.source_path = self.root / 'original_checkpoint.pt'
        torch.save(dict(model=codec.state_dict(), optimizer=old_optimizer.state_dict()), self.source_path)
        self.source_bytes = self.source_path.read_bytes()
        self.source = dict(arm='fdma', source_commit=cfg.SOURCE_COMMIT,
            checkpoint_sha256=runtime.sha256(self.source_path),
            frozen_files=[dict(path=str(self.source_path), sha256=runtime.sha256(self.source_path))])
        self.codec, self.stats, self.ce = copy.deepcopy(self.base)
        parameters = runtime.helpers().configure_bs_only(self.codec, self.ce, self.stats)
        self.frozen = runtime.helpers().frozen_state(self.codec, self.ce, self.stats)
        self.optimizer = torch.optim.AdamW(parameters, lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY, betas=(.9, .999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.assertFalse(self.optimizer.state)
        runtime.optimizer_check(self.optimizer, self.scheduler, 0)
        self.data = SimpleNamespace(protocol={'fixture': 'tiny_real_CPU_no_CDL'})
        self.core = runtime.core_identity()
        self.directory = self.output / 'training' / 'fdma'
        self.directory.mkdir(parents=True)
        runtime.atomic_json(self.directory / 'initialization.json', dict(
            arm='fdma', core=self.core, source_identity=self.source,
            frozen_identity=self.frozen, recipe=cfg.recipe('fdma'),
            fingerprint=cfg.fingerprint('fdma'), original_optimizer_continued=False), exclusive=True)

    def advance(self, first, last):
        for previous in range(first, last):
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.codec.decoder(self.codec.encoder(torch.ones(2, 2)))
            (prediction - torch.tensor([[.2, -.1]])).square().mean().backward()
            runtime.commit_update(self.codec, self.optimizer, self.scheduler,
                                  previous_updates=previous)
        runtime.helpers().assert_frozen_state(self.codec, self.ce, self.stats, self.frozen)

    def save(self, step, **overrides):
        kwargs = dict(arm='fdma', step=step, data=self.data, core=self.core,
            source=self.source, frozen=self.frozen, elapsed=1., measured=.01, final=True)
        kwargs.update(overrides)
        return runtime.save_checkpoint(self.directory, self.codec, self.stats, self.ce,
                                       self.optimizer, self.scheduler, **kwargs)

    def source_factory(self, arm, device):
        self.assertEqual(arm, 'fdma')
        self.assertEqual(str(device), 'cpu')
        codec, stats, ce = copy.deepcopy(self.base)
        return codec, stats, ce, copy.deepcopy(self.source)

    def load_terminal(self):
        with mock.patch.object(runtime, 'load_source', side_effect=self.source_factory), \
             mock.patch.object(runtime, 'make_data', return_value=self.data):
            return runtime.load_finetuned('fdma', 'cpu')

    def test_real_checkpoint_resume_roundtrip_and_read_only_original(self):
        self.advance(0, 100)
        first = self.save(100)
        self.assertEqual(first['status'], 'paused_boundary')
        target = self.directory / 'checkpoint_latest.pt'
        payload = torch.load(target, map_location='cpu', weights_only=False)
        self.assertEqual(set(payload['model']), set(self.codec.decoder.state_dict()))
        self.assertNotIn('encoder.weight', payload['model'])
        self.assertEqual(payload['counters']['EMA_commits'], 0)
        self.assertEqual(first['checkpoint_sha256'], runtime.sha256(target))
        self.assertEqual(runtime.verify_saved(self.directory, payload, first, arm='fdma',
            data=self.data, core=self.core, source=self.source, frozen=self.frozen), 100)
        with self.assertRaisesRegex(ValueError, 'Fixed rate terminal'):
            self.load_terminal()

        # Restore actual decoder/AdamW/scheduler state into new CPU objects.
        self.codec, self.stats, self.ce, _ = self.source_factory('fdma', 'cpu')
        parameters = runtime.helpers().configure_bs_only(self.codec, self.ce, self.stats)
        runtime.original.strict_load(self.codec.decoder, payload['model'])
        self.optimizer = torch.optim.AdamW(parameters, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.assertFalse(self.optimizer.state)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.scheduler.load_state_dict(payload['scheduler'])
        runtime.restore_rng(payload['rng'])
        runtime.optimizer_check(self.optimizer, self.scheduler, 100)
        self.advance(100, 200)
        terminal = self.save(200)
        self.assertEqual(terminal['status'], 'completed')
        self.assertEqual(terminal['counters'], cfg.counters(200))
        codec, stats, ce, record = self.load_terminal()
        self.assertEqual(runtime.state_digest(codec.decoder), runtime.state_digest(self.codec.decoder))
        self.assertEqual(runtime.helpers().frozen_state(codec, ce, stats), self.frozen)
        self.assertEqual(record['step'], 200)
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)
        runtime.verify_files(self.source)
        self.assertFalse(codec.training)
        self.assertTrue(all(not p.requires_grad for p in codec.parameters()))

    def test_real_serializer_failure_preserves_checkpoint_and_manifest(self):
        self.advance(0, 100)
        self.save(100)
        target = self.directory / 'checkpoint_latest.pt'
        before_checkpoint = target.read_bytes()
        before_manifest = (self.directory / 'manifest.json').read_bytes()
        before_names = {p.name for p in self.directory.iterdir()}
        # Real torch.save reaches an unpicklable local lambda. No serializer mock.
        invalid_source = dict(self.source, invalid=lambda: None)
        with self.assertRaises((AttributeError, TypeError)):
            self.save(100, source=invalid_source)
        self.assertEqual(target.read_bytes(), before_checkpoint)
        self.assertEqual((self.directory / 'manifest.json').read_bytes(), before_manifest)
        self.assertEqual({p.name for p in self.directory.iterdir()}, before_names)
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)

    def test_terminal_checkpoint_byte_tampering_is_rejected(self):
        self.advance(0, 200)
        self.save(200)
        target = self.directory / 'checkpoint_latest.pt'
        target.write_bytes(target.read_bytes() + b'extra-bytes-must-not-pass-hash-check')
        with self.assertRaisesRegex(ValueError, 'checkpoint hash changed'):
            self.load_terminal()
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)

    def test_source_file_and_original_core_mutation_are_rejected(self):
        runtime.verify_files(self.source)
        self.source_path.write_bytes(self.source_bytes + b'changed')
        with self.assertRaisesRegex(ValueError, 'source artifact changed'):
            runtime.verify_files(self.source)
        with mock.patch.object(runtime.original, 'core_identity', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'Pinned original training core changed'):
                runtime.guard_source()

    def test_frozen_state_mutation_stops_save_without_overwriting_checkpoint(self):
        self.advance(0, 100)
        self.save(100)
        target = self.directory / 'checkpoint_latest.pt'
        before = target.read_bytes()
        self.stats.ema_updates.add_(1)
        with self.assertRaisesRegex(ValueError, 'Frozen encoder/CE/historical-statistics'):
            self.save(100)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)

    def test_bad_optimizer_boundaries_and_gradients_stop_before_update(self):
        before = runtime.state_digest(self.codec.decoder)
        for parameter in self.codec.decoder.parameters():
            parameter.grad = torch.full_like(parameter, float('nan'))
        with self.assertRaises(FloatingPointError):
            runtime.commit_update(self.codec, self.optimizer, self.scheduler, previous_updates=0)
        self.assertEqual(runtime.state_digest(self.codec.decoder), before)
        self.assertFalse(self.optimizer.state)
        self.assertEqual(self.scheduler.last_epoch, 0)
        self.optimizer.param_groups[0]['lr'] = cfg.LR * 2
        with self.assertRaisesRegex(ValueError, 'optimizer recipe changed'):
            runtime.optimizer_check(self.optimizer, self.scheduler, 0)
        runtime.helpers().assert_frozen_state(self.codec, self.ce, self.stats, self.frozen)

    def test_verified_boundary_and_terminal_receipts_can_be_recreated_but_not_changed(self):
        target = self.directory / 'checkpoint_latest.pt'
        for start, step, name in ((0, 100, 'boundary_00100.json'),
                                  (100, 200, 'terminal_receipt.json')):
            with self.subTest(receipt=name):
                self.advance(start, step)
                manifest = self.save(step)
                payload = torch.load(target, map_location='cpu', weights_only=False)
                runtime.verify_saved(self.directory, payload, manifest, arm='fdma',
                    data=self.data, core=self.core, source=self.source, frozen=self.frozen)
                receipt = self.directory / name
                before_checkpoint = target.read_bytes()
                receipt.unlink()
                runtime.ensure_boundary_receipts(self.directory, manifest)
                self.assertEqual(runtime.read_json(receipt), manifest)
                self.assertEqual(target.read_bytes(), before_checkpoint)
                runtime.atomic_json(receipt, dict(manifest, changed=True))
                before_bad_receipt = receipt.read_bytes()
                with self.assertRaisesRegex(ValueError, 'boundary receipt changed'):
                    runtime.ensure_boundary_receipts(self.directory, manifest)
                self.assertEqual(receipt.read_bytes(), before_bad_receipt)
                self.assertEqual(target.read_bytes(), before_checkpoint)
                runtime.atomic_json(receipt, manifest)

    def test_changed_source_file_stops_save_before_checkpoint_replacement(self):
        self.advance(0, 100)
        self.save(100)
        target = self.directory / 'checkpoint_latest.pt'
        before_checkpoint = target.read_bytes()
        before_manifest = (self.directory / 'manifest.json').read_bytes()
        self.source_path.write_bytes(self.source_bytes + b'changed-on-server')
        with self.assertRaisesRegex(ValueError, 'source artifact changed'):
            self.save(100)
        self.assertEqual(target.read_bytes(), before_checkpoint)
        self.assertEqual((self.directory / 'manifest.json').read_bytes(), before_manifest)

    def test_real_cpu_train_loop_and_actual_monitors_keep_frozen_state(self):
        import evaluate_twc_v1_k8_icc_rate as evaluate

        class TinyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.coupling = torch.nn.Parameter(torch.tensor(.4))
                channel = torch.zeros(1, 8, 32, 256, dtype=torch.complex64)
                channel[:, torch.arange(8), torch.arange(8), :] = 1.
                mix = torch.zeros_like(channel)
                mix[:, 0, 1, :] = 1.
                mix[:, 1, 0, :] = 1.
                self.register_buffer('base', channel)
                self.register_buffer('mix', mix)

            def forward(self, batch):
                return (self.base + self.coupling * self.mix).expand(batch, -1, -1, -1)

        class TinyStatistics(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('historical_power', torch.full((8, 24), 2.))
                self.register_buffer('ema_updates', torch.tensor(20000))

            def nominal_power(self):
                return self.historical_power

        source_codec = torch.nn.Module()
        source_codec.encoder = torch.nn.Linear(2, 3)
        source_codec.decoder = TinyDecoder()
        source_stats, source_ce = TinyStatistics(), torch.nn.Linear(2, 2)
        for module in (source_codec, source_stats, source_ce):
            module.eval().requires_grad_(False)
        integration_source = self.root / 'integration_source.pt'
        torch.save(dict(model=source_codec.state_dict(), stats=source_stats.state_dict()), integration_source)
        original_source_bytes = integration_source.read_bytes()
        source_record = dict(arm='fdma', source_commit=cfg.SOURCE_COMMIT,
            checkpoint_sha256=runtime.sha256(integration_source),
            frozen_files=[dict(path=str(integration_source), sha256=runtime.sha256(integration_source))])
        loaded = []

        def load_fixture(arm, device):
            self.assertEqual(arm, 'fdma')
            self.assertEqual(str(device), 'cpu')
            self.assertFalse((runtime.OUTPUT_ROOT / 'training/fdma').exists())
            codec, stats, ce = copy.deepcopy((source_codec, source_stats, source_ce))
            loaded.append((codec, stats, ce))
            return codec, stats, ce, copy.deepcopy(source_record)

        class TinyData:
            protocol = {'fixture': 'deterministic_tiny_full_shape_CPU_no_CDL'}

            def __init__(self):
                self.requests = []

            def sample(self, role, index):
                if role not in ('codec_train', 'validation'):
                    raise AssertionError('Training must not sample test or unrelated roles')
                self.requests.append((role, index))
                channel = source_codec.decoder.base.expand(4, -1, -1, -1).clone()
                return dict(H_dl=channel, H_ul=channel.clone(),
                            sample_fingerprint=cfg.digest([role, index]))

        data = TinyData()
        mode_calls = []

        def fixture_forward(codec, stats, ce, sample, gamma_db, noise_key):
            role = noise_key[0]
            mode_calls.append((role, codec.decoder.training, torch.is_grad_enabled()))
            self.assertFalse(codec.encoder.training)
            self.assertFalse(ce.training)
            self.assertFalse(stats.training)
            self.assertTrue(all(not p.requires_grad for p in codec.encoder.parameters()))
            if role == 'codec_train':
                self.assertTrue(codec.decoder.training)
                self.assertTrue(torch.is_grad_enabled())
            else:
                self.assertFalse(codec.decoder.training)
                self.assertFalse(torch.is_grad_enabled())
            active = torch.full((4, 8, 24), 2 ** .5, dtype=torch.complex64)
            grid = torch.zeros(4, 8, 256, dtype=torch.complex64)
            grid[:, :, :32] = 2 ** .5
            noise = torch.zeros(4, 32, 256, dtype=torch.complex64)
            polarity = torch.ones(4, 8)
            return dict(H_hat=codec.decoder(4), waveform=SimpleNamespace(active=active, x_grid=grid),
                soft=active * (1. + .01j), H_ul_est=sample['H_ul'], input_hashes=dict(
                    standard_noise=runtime.state_digest({'noise': noise}),
                    pilot_polarity=runtime.state_digest({'polarity': polarity})))

        original_evaluate = evaluate.evaluate_models
        monitor_checkpoint_steps = []

        def evaluate_on_cpu(*args, **kwargs):
            # Exercise the real monitor writer/scorer/freezing context, only
            # disable its CUDA capability requirement for this tensor fixture.
            monitor_checkpoint_steps.append(runtime.read_json(
                runtime.OUTPUT_ROOT / 'training/fdma/manifest.json')['step'])
            return original_evaluate(*args, **kwargs, require_cuda=False)

        with mock.patch.object(runtime, 'OUTPUT_ROOT', self.root / 'integration_outputs'), \
             mock.patch.object(cfg, 'UPDATES', 100), \
             mock.patch.object(cfg, 'MONITOR_BATCHES', 1), \
             mock.patch.object(runtime, 'configure_cuda', return_value=torch.device('cpu')), \
             mock.patch.object(torch.cuda, 'reset_peak_memory_stats'), \
             mock.patch.object(torch.cuda, 'synchronize'), \
             mock.patch.object(runtime, 'load_source', side_effect=load_fixture), \
             mock.patch.object(runtime, 'make_data', return_value=data), \
             mock.patch.object(runtime, 'forward_codec', side_effect=fixture_forward), \
             mock.patch.object(evaluate, 'evaluate_models', side_effect=evaluate_on_cpu):
            result = runtime.train('fdma')
            directory = runtime.OUTPUT_ROOT / 'training/fdma'
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['step'], 100)
            self.assertEqual(result['counters']['EMA_commits'], 0)
            self.assertTrue((directory / 'boundary_00100.json').is_file())
            self.assertTrue((directory / 'terminal_receipt.json').is_file())
            for step in (0, 100):
                receipt = runtime.read_json(directory / 'monitor' / f'step_{step:05d}_receipt.json')
                self.assertTrue(receipt['files'])
                monitor_manifest = next(Path(name) for name in receipt['files'] if name.endswith('manifest.json'))
                self.assertEqual(runtime.read_json(monitor_manifest)['status'], 'passed')
            payload = torch.load(directory / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(payload['scheduler']['last_epoch'], 100)
            self.assertEqual(len(payload['model']), len(source_codec.decoder.state_dict()))
            self.assertNotEqual(float(payload['model']['coupling']), float(source_codec.decoder.coupling))
        self.assertEqual([index for role, index in data.requests if role == 'codec_train'],
                         list(range(1, 801)))
        self.assertEqual(sum(role == 'monitor' for role, _, _ in mode_calls), 12)
        self.assertEqual(sum(role == 'codec_train' for role, _, _ in mode_calls), 800)
        self.assertEqual(monitor_checkpoint_steps, [0, 100])
        codec, stats, ce = loaded[0]
        expected = runtime.helpers().frozen_state(source_codec, source_ce, source_stats)
        runtime.helpers().assert_frozen_state(codec, ce, stats, expected)
        truth = source_codec.decoder.base.expand(4, -1, -1, -1)
        initial_loss, _ = runtime.helpers().rate_objective(source_codec.decoder(4), truth)
        final_loss, _ = runtime.helpers().rate_objective(codec.decoder(4), truth)
        self.assertLess(float(final_loss.detach()), float(initial_loss.detach()))
        self.assertEqual(integration_source.read_bytes(), original_source_bytes)


if __name__ == '__main__':
    unittest.main()
