"""Real CPU nominal-PHY gradients and strict frozen-parent/owned-IO checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from configs import twc_v1_k8_hybrid_direction_config as cfg
import hybrid_direction_runtime as runtime
import twc_v1_k8_hybrid_direction as math_tools


def fixture(arm='hybrid_g4'):
    generator = torch.Generator().manual_seed(127041)
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
    stats.ema_updates.fill_(30000)
    ce = CE()
    for module in (codec, stats, ce): module.eval().requires_grad_(False)
    sample = {k: torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator)
        for k in ('H_dl', 'H_ul')}
    return codec, stats, ce, sample


class HybridDirectionRuntimeTests(unittest.TestCase):
    def test_old_scientific_core_and_new_analysis_bytes_remain_guarded(self):
        import diagnose_twc_v1_k8_rate_headroom as headroom
        core = runtime.core_identity()
        runtime.guard_core(core)
        actual = runtime.sha256
        with mock.patch.object(runtime, 'sha256', side_effect=lambda path:
                '0'*64 if Path(path).name == 'twc_v1_k8_hybrid_direction.py' else actual(path)):
            with self.assertRaisesRegex(ValueError, 'scientific core changed'):
                runtime.guard_core(core)
        previous = headroom.sha
        with mock.patch.object(headroom, 'sha', side_effect=lambda path:
                '0'*64 if Path(path).name == 'train_twc_v1_k8_ue_bs_rate.py' else previous(path)):
            with self.assertRaisesRegex(ValueError, 'joint scientific core changed'):
                runtime.guard_core(core)

    def test_real_objective_gradient_reaches_ue_and_bs_without_ce_or_prior_update(self):
        codec, stats, ce, sample = fixture()
        runtime.configure_joint(codec, ce, stats)
        frozen = runtime.frozen_state(codec, ce, stats)
        prior = copy.deepcopy(stats.state_dict())
        phy = runtime.physics()
        with mock.patch.object(phy, 'recover', wraps=phy.recover) as recover:
            out = runtime.forward_codec(codec, stats, ce, sample, None, ('codec_train', 1, 0))
        self.assertFalse(torch.equal(recover.call_args.args[1], sample['H_ul']))
        self.assertTrue(torch.equal(recover.call_args.args[1], out['H_ul_est']))
        self.assertTrue(out['waveform'].active.requires_grad and out['soft'].requires_grad)
        self.assertFalse(out['H_ul_est'].requires_grad)
        loss, _ = math_tools.objective(out['H_hat'], sample['H_dl']); loss.backward()
        for module in (codec.encoder, codec.decoder):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 1e-5)
        self.assertTrue(all(p.grad is None for p in ce.parameters()))
        self.assertTrue(all(torch.equal(v, stats.state_dict()[k]) for k, v in prior.items()))
        runtime.assert_frozen_state(codec, ce, stats, frozen)
        self.assertEqual(int(stats.ema_updates), 30000)

    def test_training_fdma_and_wrong_roles_gamma_are_refused(self):
        codec, stats, ce, sample = fixture('fdma')
        with self.assertRaises(ValueError): runtime.configure_joint(codec, ce, stats)
        with self.assertRaises(ValueError): runtime.forward_codec(codec, stats, ce, sample, None, ('codec_train', 1, 0))
        codec, stats, ce, sample = fixture()
        for key in (('test', 0, 0), ('final', cfg.FINAL_BATCHES, 0), ('codec_train', 1, 1)):
            with self.assertRaises(ValueError): runtime.forward_codec(codec, stats, ce, sample, None, key)
        with self.assertRaises(ValueError): runtime.batch_random('codec_train', 1, 'cpu', gamma=25.)
        with self.assertRaises(ValueError): runtime.load_source('hybrid_g4', 'cpu')

    def test_fresh_noise_metadata_and_frozen_evaluation_do_not_commit(self):
        data = runtime.make_data('cpu')
        old = runtime.previous.make_data('cpu')
        self.assertNotEqual(data.protocol['data_id'], old.protocol['data_id'])
        self.assertEqual(data.protocol['seed_namespace'], cfg.NAMESPACE)
        codec, stats, ce, sample = fixture()
        before = runtime.state_digest(stats)
        with torch.inference_mode():
            out = runtime.forward_codec(codec, stats, ce, sample, 0., ('final', 0, 0))
            again = runtime.forward_codec(codec, stats, ce, sample, 0., ('final', 0, 0))
        self.assertEqual(out['input_hashes'], again['input_hashes'])
        self.assertEqual(before, runtime.state_digest(stats))
        with self.assertRaises(ValueError): runtime.forward_codec(codec, stats, ce, sample, 5., ('final', 0, 0))

    def test_exact_parent_binding_uses_actual_30000_not_old_20000(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'parent.txt'; path.write_text('immutable evidence')
            files = [dict(path=str(path), sha256=runtime.sha256(path))]
            arm = 'hybrid_g4'; pins = cfg.SOURCE_PINS[arm]
            parent = dict(pins, step=10000, production_evidence=True, frozen_files=files,
                base_ema_updates=20000)
            record = dict(format=cfg.NAMESPACE+'_source_v1', arm=arm,
                source_namespace=cfg.SOURCE_NAMESPACE, source_kind='joint10000',
                production_evidence=True, parent_record=parent, frozen_files=files,
                base_ema_updates=30000, base_encoder_state_sha256=pins['encoder_state_sha256'],
                base_decoder_state_sha256=pins['decoder_state_sha256'],
                base_statistics_state_sha256=pins['statistics_state_sha256'],
                **{k: pins[k] for k in ('encoder_state_sha256', 'decoder_state_sha256',
                    'state_sha256', 'statistics_state_sha256', 'CE_state_sha256')})
            runtime.verify_source_record(arm, record)
            for key, bad in (('base_ema_updates', 20000), ('state_sha256', 'f'*64)):
                changed = dict(record, **{key: bad})
                with self.assertRaises(ValueError): runtime.verify_source_record(arm, changed)
            changed = copy.deepcopy(record); changed['parent_record']['checkpoint_sha256'] = 'f'*64
            with self.assertRaises(ValueError): runtime.verify_source_record(arm, changed)
            path.write_text('changed')
            with self.assertRaises(ValueError): runtime.verify_source_record(arm, record)

    def test_owned_atomic_io_preserves_existing_record_on_exclusive_retry(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runtime, 'OUTPUT_ROOT', Path(temporary)):
            path = Path(temporary)/'record.json'
            runtime.atomic_json(path, dict(value=1), exclusive=True)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): runtime.atomic_json(path, dict(value=2), exclusive=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(temporary).iterdir()], ['record.json'])
            with self.assertRaises(ValueError): runtime.inside(Path(temporary).parent/'outside.json')


if __name__ == '__main__': unittest.main()
