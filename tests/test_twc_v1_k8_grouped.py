"""CPU tests for the group-size study (generic waveform/receiver, loader, trainer)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import final_runtime as fr
import grouped_runtime as gr
import train_twc_v1_k8_grouped as gt
from configs import twc_v1_k8_grouped_config as cfg
from models.twc_v1_k8_icc_aligned import support_geometry as original_geometry
from tests.test_twc_v1_k8_final import Data, IdentityCE


class LayoutAndReceiverTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.phy = fr.physics()

    def test_g4_and_fdma_match_original_code(self):
        for arm in ('fdma', 'hybrid_g4'):
            layout = gr.make_layout(arm)
            raw = torch.randn(2, 8, 216, dtype=torch.complex64)
            pol = torch.ones(2, 8)
            a, b = gr.build_waveform(raw, layout, pol), self.phy.build_waveform(raw, self.phy.make_layout(arm), polarity=pol)
            self.assertTrue(torch.allclose(a.x_grid, b.x_grid))
            for x, y in zip(gr.support_geometry(layout), original_geometry(self.phy.make_layout(arm))):
                self.assertTrue(torch.equal(x, y))
            n = gr.active_dimension(layout)
            prior = torch.rand(2, 8, n)+.5; prior = 48*prior/prior.sum(-1, keepdim=True)
            Y = torch.randn(2, 32, 256, dtype=torch.complex64); H = torch.randn(2, 8, 32, 256, dtype=torch.complex64)
            var, gamma = torch.rand(2, 8, 256)*.01, torch.tensor([0., 20.])
            ref = fr.recover_batched(Y, H, self.phy.make_layout(arm), prior, var, gamma)
            new = gr.recover(Y, H, layout, prior, var, gamma)
            self.assertTrue(torch.allclose(ref['soft'], new['soft'], atol=1e-5, rtol=1e-4))
            self.assertTrue(torch.allclose(ref['reliability'], new['reliability'], rtol=1e-4))

    def test_g2_g1_layouts_and_near_perfect_recovery(self):
        for arm, m, n in (('hybrid_g2', 4, 84), ('hybrid_g1', 8, 164)):
            layout = gr.make_layout(arm)
            self.assertEqual(gr.active_dimension(layout), n)
            self.assertTrue(all(len(g) == m for g in layout.groups))
            raw = torch.randn(1, 8, 216, dtype=torch.complex64)
            wave = gr.build_waveform(raw, layout, torch.ones(1, 8))
            self.assertTrue(torch.allclose(wave.active.abs().square().sum(-1), torch.full((1, 8), 48.)))
            occupancy = (wave.data.abs() > 0).sum(1)
            self.assertEqual(int(occupancy.max()), m)
            H = torch.randn(1, 8, 32, 256, dtype=torch.complex64)
            Y = self.phy.transmit(H, wave.x_grid, torch.zeros(1, 32, 256, dtype=torch.complex64), torch.tensor([25.]))
            prior = wave.active.abs().square().clamp_min(1e-3); prior = 48*prior/prior.sum(-1, keepdim=True)
            out = gr.recover(Y, H, layout, prior, torch.zeros(1, 8, 256), torch.tensor([25.]))
            err = (out['soft'][..., :n]-wave.active).abs().square().sum()/wave.active.abs().square().sum()
            self.assertLess(float(err), 1e-3)
            support, active, exclusive = gr.support_geometry(layout)
            self.assertEqual(int(active.sum()), 8*n)
            self.assertEqual(int(exclusive.sum()), 32)

    def test_shared_ce_loader_refuses_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in cfg.SHARED_CE_SHA256:
                (Path(tmp)/name).write_text('{}')
            with self.assertRaises(ValueError):
                gr.load_shared_ce('cpu', Path(tmp))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patch in (mock.patch.object(cfg, 'NMSE_UPDATES', 2), mock.patch.object(cfg, 'UPDATES', 4),
                      mock.patch.object(cfg, 'SAVE_EVERY', 2), mock.patch.object(cfg, 'ACCUMULATION', 1),
                      mock.patch.object(cfg, 'WARMUP', 1)):
            patch.start(); self.addCleanup(patch.stop)
        torch.set_num_threads(2)
        self.ce = (IdentityCE(), torch.full((6, 8, 256), .002), dict(sha256=dict(cfg.SHARED_CE_SHA256), step=65000))

    def _train(self, out, stop, resume=False):
        return gt.train('hybrid_g2', stop, resume, device=torch.device('cpu'), ce_bundle=self.ce, data=Data(),
            out_dir=out, core=dict(commit='f'*40, sha256={'fixture': 'a'}))

    def test_two_stages_resume_and_evaluate(self):
        out = self.root/'g2'
        self._train(out, 2)
        manifest = self._train(out, 4, resume=True)
        self.assertEqual(manifest['status'], 'completed')
        stages = [json.loads(l)['stage'] for l in (out/'progress.jsonl').read_text().splitlines()]
        self.assertEqual(stages, ['nmse', 'objective'])
        loaded = gt.load_terminal('hybrid_g2', torch.device('cpu'), ce_bundle=self.ce, directory=out)
        with mock.patch.object(cfg, 'output_root', lambda arm: self.root/arm):
            report = gt.evaluate('hybrid_g2', torch.device('cpu'), loaded=loaded, data=Data(), out_root=self.root/'rep',
                sizes=dict(final=1, midpoint=1))
        self.assertTrue((report/'summary.csv').is_file())
        self.assertIn('hybrid_g2', (report/'summary.csv').read_text())


if __name__ == '__main__':
    unittest.main()
