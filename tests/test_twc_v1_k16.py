"""CPU tests for the K=16 user-load study (protocol, physics, CE, codecs, evaluation, digest)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import k16_runtime as r
import train_twc_v1_k16 as t16
from configs import twc_v1_k16_config as cfg

ROOT = Path(__file__).resolve().parents[1]


class Data:
    """Low-rank random K16 channels (CPU stand-in for Sionna)."""
    def sample(self, role, index):
        g = torch.Generator().manual_seed(hash((role, index)) % 2**31)
        def h():
            a = torch.randn(cfg.MICRO_BATCH, 16, 32, 4, generator=g, dtype=torch.complex64)
            b = torch.randn(cfg.MICRO_BATCH, 16, 4, 256, generator=g, dtype=torch.complex64)
            return (a@b)/4
        return dict(H_dl=h(), H_ul=h())


class K16Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        torch.set_num_threads(4)

    def test_protocol_pilots_and_ls(self):
        self.assertTrue(r.check_protocol())
        self.assertEqual(sorted(p for row in r.PILOT_POSITIONS_BY_USER for p in row), list(range(0, 256, 4)))
        pil = r.known_pilots(2)
        self.assertTrue(torch.allclose(pil.abs().square().sum(-1), torch.full((2, 16), 16.)))
        # noiseless channel inside the rank-4 delay model is recovered exactly by LS
        taps = torch.randn(2, 16, 32, 4, dtype=torch.complex64)
        H = torch.einsum('qr,bknr->bknq', r.fourier_basis(), taps)
        Y = r.transmit(H, pil, torch.zeros(2, 32, 256, dtype=torch.complex64), torch.tensor([25., 25.]))
        est = r.fourier_ls(Y, pil)
        self.assertLess(float((est-H).abs().square().sum()/H.abs().square().sum()), 1e-3)

    def test_greedy_pairing_valid(self):
        H = torch.randn(3, 16, 32, 256, dtype=torch.complex64)
        p = r.greedy_pairing(H)
        self.assertEqual([sorted(x) for x in p.tolist()], [list(range(16))]*3)
        self.assertLessEqual(float(r.slot_pair_coupling(r.permute_users(H, p)).mean()),
                             float(r.slot_pair_coupling(H).mean())+1e-6)

    def test_pipeline_ce_codecs_evaluate_digest(self):
        patches = dict(CE_UPDATES=2, CE_SAVE_EVERY=2, CE_WARMUP=1, CALIBRATION_BATCHES=1, NMSE_UPDATES=2, UPDATES=4,
                       SAVE_EVERY=2, WARMUP=1, ACCUMULATION=1, OUTPUT_ROOT=self.root/'k16')
        with mock.patch.multiple(cfg, **patches), mock.patch.object(t16, 'CE_DIR', self.root/'k16'/'ce'):
            ce = r.new_ce()
            t16.train_ce(device=torch.device('cpu'), data=Data(), init_state=ce.state_dict(), out_dir=self.root/'k16'/'ce')
            bundle = t16.load_ce(torch.device('cpu'), self.root/'k16'/'ce')
            self.assertEqual(tuple(bundle[1].shape), (6, 16, 256))
            systems = {}
            for arm in cfg.ARMS:
                t16.train_codec(arm, device=torch.device('cpu'), ce_bundle=bundle, data=Data(),
                    out_dir=self.root/'k16'/arm, stop_after=2)
                manifest = t16.train_codec(arm, True, device=torch.device('cpu'), ce_bundle=bundle, data=Data(),
                    out_dir=self.root/'k16'/arm)
                self.assertEqual(manifest['status'], 'completed')
                systems[arm], _ = t16.load_codec(arm, torch.device('cpu'), bundle, self.root/'k16'/arm)
            report = t16.evaluate(torch.device('cpu'), systems=systems, data=Data(), out_root=self.root/'rep',
                sizes=dict(selection=1, final=1, midpoint=1))
        prov = json.loads((report/'provenance.json').read_text())
        self.assertEqual((prov['K'], len(prov['switch_decision'])), (16, 6))
        spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
        rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
        audit = rd.build(report)[2]['audit_recomputed_from_samples']
        self.assertLess(audit['max_relative_mean_difference'], 1e-9)
        self.assertIn('paired_minus_fdma', (report/'comparisons.csv').read_text())


if __name__ == '__main__':
    unittest.main()
