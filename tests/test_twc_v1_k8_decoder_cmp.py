"""CPU tests for the K8 DETR-style decoder and the decoder-comparison trainer wrapper."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import final_runtime as fr
import grouped_runtime as gr
import train_twc_v1_k8_grouped as gt
import train_twc_v1_k8_decoder_cmp as dc
from configs import twc_v1_k8_decoder_cmp_config as cfg
from configs import twc_v1_k8_grouped_config as gcfg
from models.twc_v1_k8_detr_decoder import K8DETRDecoder
from tests.test_twc_v1_k8_final import Data, IdentityCE


class DETRTests(unittest.TestCase):
    def test_shapes_gradients_and_inactive_slots(self):
        torch.manual_seed(0)
        layout = gr.make_layout('hybrid_g4')
        dec = K8DETRDecoder(layout, activation_checkpointing=False)
        self.assertGreater(dec.metadata()['parameters'], 1e7)
        B = 2
        Y = torch.randn(B, 32, 256, dtype=torch.complex64); H = torch.randn(B, 8, 32, 256, dtype=torch.complex64)
        soft = torch.randn(B, 8, 216, dtype=torch.complex64); rel = torch.rand(B, 8, 216)*dec._fixed_active
        pil = torch.randn(B, 8, 256, dtype=torch.complex64)
        out = dec(Y, H, soft, rel, pil, gamma_db=torch.tensor([0., 25.]))['H_hat']
        self.assertEqual(out.shape, (B, 8, 32, 256))
        out.abs().square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in dec.parameters() if p.requires_grad))
        # values in padded slots must not matter
        soft2 = soft.clone(); soft2[:, ~dec._fixed_active] = 99
        a = dec(Y, H, soft, rel, pil, gamma_db=torch.tensor([0., 25.]))['H_hat']
        b = dec(Y, H, soft2, rel, pil, gamma_db=torch.tensor([0., 25.]))['H_hat']
        self.assertTrue(torch.allclose(a, b))


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        saved = (gt.cfg, gr.cfg, gt.build)
        self.addCleanup(lambda: (setattr(gt, 'cfg', saved[0]), setattr(gr, 'cfg', saved[1]), setattr(gt, 'build', saved[2])))
        dc.install()
        for patch in (mock.patch.object(cfg, 'NMSE_UPDATES', 2), mock.patch.object(cfg, 'UPDATES', 4),
                      mock.patch.object(cfg, 'SAVE_EVERY', 2), mock.patch.object(cfg, 'ACCUMULATION', 1),
                      mock.patch.object(cfg, 'WARMUP', 1)):
            patch.start(); self.addCleanup(patch.stop)
        torch.set_num_threads(2)
        self.ce = (IdentityCE(), torch.full((6, 8, 256), .002), dict(sha256=dict(cfg.SHARED_CE_SHA256), step=65000))

    def test_both_arms_train_resume_evaluate(self):
        self.assertIs(gt.cfg, cfg)
        self.assertNotEqual(cfg.NAMESPACE, gcfg.NAMESPACE)
        self.assertNotEqual(cfg.fingerprint('g4_detr'), cfg.fingerprint('g4_joint'))
        for arm in ('g4_joint', 'g4_detr'):
            out = self.root/arm
            kw = dict(device=torch.device('cpu'), ce_bundle=self.ce, data=Data(), out_dir=out,
                      core=dict(commit='f'*40, sha256={'fixture': 'a'}))
            gt.train(arm, 2, False, **kw)
            manifest = gt.train(arm, 4, True, **kw)
            self.assertEqual((manifest['status'], manifest['FDMA_updates']), ('completed', 0))
            loaded = gt.load_terminal(arm, torch.device('cpu'), ce_bundle=self.ce, directory=out)
            kind = type(loaded[0]['codec'].decoder).__name__
            self.assertEqual(kind == 'K8DETRDecoder', arm == 'g4_detr')
            with mock.patch.object(cfg, 'output_root', lambda a: self.root/a):
                report = gt.evaluate(arm, torch.device('cpu'), loaded=loaded, data=Data(), out_root=self.root/'rep',
                    sizes=dict(final=1, midpoint=1))
            prov = json.loads((report/'provenance.json').read_text())
            self.assertEqual((prov['namespace'], prov['arm']), (cfg.NAMESPACE, arm))


if __name__ == '__main__':
    unittest.main()
