"""CPU tests for the 3GPP Type II codebook baseline."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train_twc_v1_k8_final as trainer
import evaluate_twc_v1_k8_typeii as t2
from configs import twc_v1_k8_final_config as cfg
from tests.test_twc_v1_k8_final import Data, fixture_sources

ROOT = Path(__file__).resolve().parents[1]
GRID = (torch.arange(32)//4, torch.arange(32) % 4, 8, 4)


class TypeIITests(unittest.TestCase):
    def test_payload_matches_hand_count(self):
        # L=4, 8PSK, subband amplitude: 4 (rotation) + 16 (C(32,4)) + 2 (strongest) + 9 (WB amp) + 6*(3*(3+1))
        self.assertEqual(t2.payload_bits(4, 8, True, 4, 6), 4+16+2+9+6*12)
        self.assertEqual(t2.payload_bits(2, 4, False, 2, 6), 4+math.ceil(math.log2(496))+1+3+6*2)
        self.assertEqual(len(t2.subbands()), 6)

    def test_beams_are_orthogonal_and_single_beam_channel_is_exact(self):
        beams = t2.dft_beams(GRID, torch.device('cpu'))
        gram = beams[5].conj()@beams[5].T
        self.assertTrue(torch.allclose(gram, torch.eye(32, dtype=gram.dtype), atol=1e-5))
        # channel = one DFT beam (rotation 5, beam 7) with a random phase per subcarrier
        phase = torch.polar(torch.ones(256), 2*math.pi*torch.rand(256))
        H = (beams[5, 7][:, None]*phase[None])[None, None].expand(1, 8, 32, 256).contiguous()*32**.5
        sbs = t2.subbands()
        W, nz = t2.typeii(t2.beam_power(H, beams), t2.subband_eigvec(H, sbs), beams, 4, 8, True)
        H_hat = t2.expand(W, H, sbs)
        cos = (H_hat.conj()*H).sum(2).abs()/(H_hat.norm(dim=2)*H.norm(dim=2))
        self.assertGreater(float(cos.min()), 1-1e-4)

    def test_quantization_error_shrinks_with_resolution(self):
        g = torch.Generator().manual_seed(3)
        H = torch.view_as_complex(torch.randn(2, 8, 32, 256, 2, generator=g))
        beams, sbs = t2.dft_beams(GRID, torch.device('cpu')), t2.subbands()
        power, vecs = t2.beam_power(H, beams), t2.subband_eigvec(H, sbs)
        gains = []
        for L, npsk, sbamp in ((2, 4, False), (4, 8, True)):
            W, _ = t2.typeii(power, vecs, beams, L, npsk, sbamp)
            gains.append(float((W.conj()*vecs).sum(-1).abs().square().mean()))
        self.assertGreater(gains[1], gains[0])

    def test_reports_and_sealed_test_guard(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        with mock.patch.object(cfg, 'UPDATES', 2), mock.patch.object(cfg, 'SAVE_EVERY', 2), \
             mock.patch.object(cfg, 'ACCUMULATION', 2), mock.patch.object(cfg, 'WARMUP', 1):
            torch.set_num_threads(2)
            trainer.train(2, False, sources=fixture_sources(), data=Data(), device=torch.device('cpu'),
                out_dir=root/'final', core=dict(commit='f'*40, sha256={'fixture': 'a'}))
            loaded = trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=root/'final')
            val = t2.evaluate('validate', torch.device('cpu'), loaded=loaded, data=Data(), grid=GRID,
                              out_root=root/'val', batches=1)
            test = t2.evaluate('test', torch.device('cpu'), loaded=loaded, data=Data(), grid=GRID,
                               out_root=root/'test', batches=1)
            with self.assertRaises(FileExistsError):
                t2.evaluate('test', torch.device('cpu'), loaded=loaded, data=Data(), grid=GRID,
                            out_root=root/'test', batches=1)
        spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
        rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
        for report in (val, test):
            audit = rd.build(report)[2]['audit_recomputed_from_samples']
            self.assertLess(audit['max_relative_mean_difference'], 1e-9)
        text = (test/'summary.csv').read_text()
        for key in ('typeii_fdma', 'subband_eigvec', 'feedback_bits', 'capacity_bits', 'typeii_L'):
            self.assertIn(key, text)
        self.assertTrue(json.loads((test/'provenance.json').read_text())['test_used'])
        self.assertIn('fdma_minus_typeii', (test/'comparisons.csv').read_text())


if __name__ == '__main__':
    unittest.main()
