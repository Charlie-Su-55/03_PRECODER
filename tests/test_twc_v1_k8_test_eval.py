"""CPU test for the one-time sealed test evaluation (structure only, fixture models)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train_twc_v1_k8_final as trainer
import train_twc_v1_k8_paired_ft as ft
import evaluate_twc_v1_k8_test as te
from configs import twc_v1_k8_final_config as cfg
from tests.test_twc_v1_k8_final import Data, fixture_sources

ROOT = Path(__file__).resolve().parents[1]


class TestEval(unittest.TestCase):
    def test_once_only_and_digest(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        decision = {float(g): ('fdma_baseline' if g == 0 else 'ft_paired') for g in cfg.GAMMAS}
        with mock.patch.object(cfg, 'UPDATES', 2), mock.patch.object(cfg, 'SAVE_EVERY', 2), \
             mock.patch.object(cfg, 'ACCUMULATION', 2), mock.patch.object(cfg, 'WARMUP', 1), \
             mock.patch.object(ft, 'UPDATES', 2), mock.patch.object(ft, 'SAVE_EVERY', 2), \
             mock.patch.object(ft, 'WARMUP', 1), mock.patch.object(ft, 'DATA_OFFSET', 4):
            torch.set_num_threads(2)
            trainer.train(2, False, sources=fixture_sources(), data=Data(), device=torch.device('cpu'),
                out_dir=root/'final', core=dict(commit='f'*40, sha256={'fixture': 'a'}))
            load = lambda: trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=root/'final')
            ft.train(2, False, loaded=load(), data=Data(), device=torch.device('cpu'), out_dir=root/'ft')
            kw = dict(ft_dir=root/'ft', train_data=Data(), data=Data(), out_root=root/'rep', batches=1, decision=decision)
            with mock.patch.object(te.bl, 'RANK_BATCHES', 2):
                report = te.evaluate(torch.device('cpu'), loaded=load(), **kw)
                with self.assertRaises(FileExistsError):
                    te.evaluate(torch.device('cpu'), loaded=load(), **kw)
        prov = json.loads((report/'provenance.json').read_text())
        self.assertTrue(prov['test_used']); self.assertFalse(prov['selection_on_test'])
        spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
        rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
        self.assertLess(rd.build(report)[2]['audit_recomputed_from_samples']['max_relative_mean_difference'], 1e-9)
        self.assertIn('proposed_minus_fdma', (report/'comparisons.csv').read_text())


if __name__ == '__main__':
    unittest.main()
