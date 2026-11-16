"""Actual CPU UE/MAC/recovery gradients and exact warmstart identity tests."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from configs import twc_v1_k8_low_snr_config as cfg
import low_snr_runtime as runtime
import twc_v1_k8_low_snr as math_tools


def fixture(arm='hybrid_g4'):
    generator = torch.Generator().manual_seed(128041)
    codes = torch.randn(8, 216, dtype=torch.complex64, generator=generator)
    template = torch.randn(3, 8, 32, 256, dtype=torch.complex64, generator=generator)
    class Encoder(nn.Module):
        def __init__(self): super().__init__(); self.codes = nn.Parameter(codes.clone())
        def forward(self, H): return self.codes[None].expand(H.shape[0], -1, -1)
    class CE(nn.Module):
        def __init__(self): super().__init__(); self.scale = nn.Parameter(torch.tensor(.96))
        def forward(self, Y, pilots, initial, *, gamma_db): return self.scale*initial
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__(); self.mix = nn.Parameter(torch.tensor([.1, .2]))
            self.register_buffer('template', template)
        def forward(self, *, Y_rx, H_ul_est, soft, reliability, pilots, gamma_db, layout, prior, ce_error):
            return dict(H_hat=self.template[0][None]+
                self.mix[0]*soft[..., 0, None, None]*self.template[1][None]+
                self.mix[1]*soft[..., 1, None, None]*self.template[2][None])
    codec = nn.Module(); codec.encoder, codec.decoder = Encoder(), Decoder()
    phy = runtime.physics()
    stats = phy.ReceiverStatistics(phy.make_layout(arm), torch.full((6, 8, 256), .02))
    stats.ema_updates.fill_(40000 if arm == 'hybrid_g4' else 30000)
    ce = CE()
    for module in (codec, stats, ce): module.eval().requires_grad_(False)
    sample = {k: torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator)
        for k in ('H_dl', 'H_ul')}
    return codec, stats, ce, sample


class LowSNRRuntimeTests(unittest.TestCase):
    def test_parent77_and_new_core_remain_strictly_guarded(self):
        parent = runtime.guard_parent()
        self.assertEqual(cfg.digest(parent), cfg.SOURCE_CORE_SHA256)
        core = runtime.core_identity(); runtime.guard_core(core)
        actual = runtime.sha256
        for name, message in (('twc_v1_k8_hybrid_direction.py', 'parent scientific core changed'),
                ('twc_v1_k8_low_snr.py', 'experiment scientific core changed')):
            with mock.patch.object(runtime, 'sha256', side_effect=lambda path, n=name:
                    '0'*64 if Path(path).name == n else actual(path)):
                with self.assertRaisesRegex(ValueError, message): runtime.guard_core(core)

    def test_matched_real_forward_gradients_and_private_scoring_tensors_are_separate(self):
        for trajectory in cfg.TRAJECTORIES:
            codec, stats, ce, sample = fixture()
            runtime.configure_joint(codec, ce, stats)
            frozen = runtime.frozen_state(codec, ce, stats)
            prior = copy.deepcopy(stats.state_dict())
            phy = runtime.physics()
            with mock.patch.object(phy, 'recover', wraps=phy.recover) as recover:
                out = runtime.forward_codec(codec, stats, ce, sample, None, ('codec_train', 1, 0))
            self.assertTrue(torch.equal(recover.call_args.args[1], out['H_ul_est']))
            self.assertFalse(torch.equal(recover.call_args.args[1], sample['H_ul']))
            self.assertFalse(out['H_ul_est'].requires_grad)
            self.assertTrue(out['soft'].requires_grad and out['waveform'].active.requires_grad)
            self.assertNotIn('H_ul_true', out['observation'])
            self.assertTrue(torch.equal(out['observation']['estimated'], out['H_ul_est']))
            self.assertTrue(torch.equal(out['observation']['Y'], out['recovered']['Y_res']))
            loss, metrics = math_tools.objective(out['H_hat'], sample['H_dl'], out['gamma_db'], trajectory)
            loss.backward()
            for module in (codec.encoder, codec.decoder):
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 1e-5)
            self.assertTrue(all(p.grad is None for p in ce.parameters()))
            self.assertTrue(all(torch.equal(value, stats.state_dict()[key]) for key, value in prior.items()))
            runtime.assert_frozen_state(codec, ce, stats, frozen)
            self.assertEqual(int(stats.ema_updates), 40000)
            self.assertEqual(out['gamma_db'].tolist(), list(cfg.gamma_schedule(1)))

    def test_forward_random_stream_does_not_depend_on_trajectory(self):
        codec, stats, ce, sample = fixture()
        with torch.inference_mode():
            first = runtime.forward_codec(codec, stats, ce, sample, 0., ('final', 0, 0))
            second = runtime.forward_codec(codec, stats, ce, sample, 0., ('final', 0, 0))
        self.assertEqual(first['input_hashes'], second['input_hashes'])
        self.assertTrue(torch.equal(first['H_hat'], second['H_hat']))
        old = runtime.previous.make_data('cpu')
        new = runtime.make_data('cpu')
        self.assertNotEqual(new.protocol['data_id'], old.protocol['data_id'])
        self.assertEqual(new.protocol['seed_namespace'], cfg.NAMESPACE)
        for trajectory in cfg.TRAJECTORIES:
            # There is intentionally no trajectory argument in the stream API.
            values = runtime.batch_random('codec_train', 8, 'cpu')
            self.assertTrue(torch.equal(values[0], torch.tensor(cfg.gamma_schedule(8)).float()))
        codec, stats, ce, sample = fixture('fdma')
        with torch.inference_mode():
            runtime.forward_codec(codec, stats, ce, sample, 25., ('final', 0, 5))
        self.assertEqual(int(stats.ema_updates), 30000)

    def test_fdma_training_wrong_stream_and_cpu_production_loading_are_refused(self):
        codec, stats, ce, sample = fixture('fdma')
        with self.assertRaises(ValueError): runtime.configure_joint(codec, ce, stats)
        with self.assertRaises(ValueError): runtime.forward_codec(codec, stats, ce, sample, None, ('codec_train', 1, 0))
        codec, stats, ce, sample = fixture()
        for key in (('test', 0, 0), ('final', cfg.FINAL_BATCHES, 0), ('codec_train', 1, 1)):
            with self.assertRaises(ValueError): runtime.forward_codec(codec, stats, ce, sample, None, key)
        with self.assertRaises(ValueError): runtime.batch_random('codec_train', 1, 'cpu', gamma=25.)
        with self.assertRaises(ValueError): runtime.load_source('hybrid_g4', 'cpu')

    def test_latest_hybrid_source_uses_40000_and_preserves_exact_parent_inventory(self):
        parent_core = runtime.guard_parent()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'parent.txt'; path.write_text('frozen source evidence')
            files = [dict(path=str(path), sha256=runtime.sha256(path))]
            parent = dict(cfg.SOURCE_PINS, core=parent_core, step=10000,
                format=cfg.SOURCE_NAMESPACE+'_manifest_v1', arm='hybrid_g4',
                method='direction_joint', status='completed',
                production_evidence=True, total_ema_updates=40000, frozen_files=files,
                base_ema_updates=30000)
            source = dict(format=cfg.NAMESPACE+'_source_v1', arm='hybrid_g4',
                source_namespace=cfg.SOURCE_NAMESPACE, source_kind='direction10000',
                production_evidence=True, parent_record=parent, frozen_files=files,
                base_ema_updates=40000, base_encoder_state_sha256=cfg.SOURCE_PINS['encoder_state_sha256'],
                base_decoder_state_sha256=cfg.SOURCE_PINS['decoder_state_sha256'],
                base_statistics_state_sha256=cfg.SOURCE_PINS['statistics_state_sha256'],
                **{k: cfg.SOURCE_PINS[k] for k in runtime.STATE_FIELDS})
            runtime.verify_source_record('hybrid_g4', source)
            for key, value in (('base_ema_updates', 30000), ('state_sha256', 'f'*64)):
                with self.assertRaises(ValueError): runtime.verify_source_record('hybrid_g4', dict(source, **{key: value}))
            changed = copy.deepcopy(source); changed['parent_record']['checkpoint_sha256'] = 'f'*64
            with self.assertRaises(ValueError): runtime.verify_source_record('hybrid_g4', changed)
            changed = copy.deepcopy(source); changed['parent_record']['total_ema_updates'] = 30000
            with self.assertRaises(ValueError): runtime.verify_source_record('hybrid_g4', changed)
            path.write_text('source changed')
            with self.assertRaises(ValueError): runtime.verify_source_record('hybrid_g4', source)

    def test_owned_atomic_json_failure_keeps_existing_bytes_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runtime, 'OUTPUT_ROOT', Path(temporary)):
            path = Path(temporary)/'record.json'
            runtime.atomic_json(path, dict(value=1), exclusive=True); before = path.read_bytes()
            with self.assertRaises(FileExistsError): runtime.atomic_json(path, dict(value=2), exclusive=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(temporary).iterdir()], ['record.json'])
            with self.assertRaises(ValueError): runtime.inside(Path(temporary).parent/'outside.json')


if __name__ == '__main__': unittest.main()
