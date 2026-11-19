"""CPU tests for the idealized SSCC baseline."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train_twc_v1_k8_final as trainer
import evaluate_twc_v1_k8_sscc as sc
from configs import twc_v1_k8_final_config as cfg
from tests.test_twc_v1_k8_final import Data, fixture_sources

ROOT = Path(__file__).resolve().parents[1]


class SSCCTests(unittest.TestCase):
    def test_water_filling_meets_rate_and_bounds(self):
        lam = torch.tensor([4., 2., 1., .5, .1], dtype=torch.float64)
        bits = torch.tensor([[3., 8.]], dtype=torch.float64)
        d = sc.water_fill(lam, bits)
        rate = torch.log2(lam/d).sum(-1)
        self.assertTrue(torch.allclose(rate, bits, atol=1e-6))
        self.assertTrue(bool((d <= lam+1e-12).all()))
        self.assertLess(float(d[0, 1].sum()), float(d[0, 0].sum()))

    def test_report_digest(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        with mock.patch.object(cfg, 'UPDATES', 2), mock.patch.object(cfg, 'SAVE_EVERY', 2), \
             mock.patch.object(cfg, 'ACCUMULATION', 2), mock.patch.object(cfg, 'WARMUP', 1):
            torch.set_num_threads(2)
            trainer.train(2, False, sources=fixture_sources(), data=Data(), device=torch.device('cpu'),
                out_dir=root/'final', core=dict(commit='f'*40, sha256={'fixture': 'a'}))
            loaded = trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=root/'final')
            report = sc.evaluate(torch.device('cpu'), loaded=loaded, data=Data(), out_root=root/'r',
                sizes=dict(final=1), rank_batches=2)
        spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
        rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
        audit = rd.build(report)[2]['audit_recomputed_from_samples']
        self.assertLess(audit['max_relative_mean_difference'], 1e-9)
        text = (report/'summary.csv').read_text()
        self.assertIn('sscc_fdma', text); self.assertIn('feedback_bits', text)


if __name__ == '__main__':
    unittest.main()
