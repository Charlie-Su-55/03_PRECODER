"""CPU tests: pairing-aware Hybrid fine-tune and training-free analog baselines."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import final_runtime as rt
import train_twc_v1_k8_final as trainer
import train_twc_v1_k8_paired_ft as ft
import evaluate_twc_v1_k8_baselines as bl
from configs import twc_v1_k8_final_config as cfg
from tests.test_twc_v1_k8_final import Data, fixture_sources

ROOT = Path(__file__).resolve().parents[1]


def digest(report):
    spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
    rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
    return rd.build(report)[2]['audit_recomputed_from_samples']


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for p in (mock.patch.object(cfg, 'UPDATES', 2), mock.patch.object(cfg, 'SAVE_EVERY', 2),
                  mock.patch.object(cfg, 'ACCUMULATION', 2), mock.patch.object(cfg, 'WARMUP', 1)):
            p.start(); self.addCleanup(p.stop)
        torch.set_num_threads(2)
        trainer.train(2, False, sources=fixture_sources(), data=Data(), device=torch.device('cpu'),
            out_dir=self.root/'final', core=dict(commit='f'*40, sha256={'fixture': 'a'}))
        self.loaded = trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=self.root/'final')

    def test_analog_roundtrip_and_report(self):
        H = Data().sample('validation', 0)['H_dl']
        self.assertTrue(torch.allclose(bl.from_coefficients(bl.to_coefficients(H)), H, atol=1e-4))
        order, power = bl.rank_coefficients(Data(), 'cpu', batches=2)
        codec = bl.AnalogCodec(order, power, 8192)
        raw = H.new_zeros((4, 8, rt.DMAX))  # full-rank codec only checks the inverse placement
        coeff = bl.to_coefficients(H).reshape(4, 8, -1)[..., codec.index]
        back = codec.decode(coeff*codec.scale, H.shape)
        self.assertTrue(torch.allclose(back, H, atol=1e-4))
        report = bl.evaluate(torch.device('cpu'), loaded=self.loaded, data=Data(), out_root=self.root/'r',
            sizes=dict(final=1, midpoint=1))
        self.assertLess(digest(report)['max_relative_mean_difference'], 1e-9)
        self.assertIn('paired_minus_analog_paired', (report/'comparisons.csv').read_text())

    def test_paired_finetune_train_resume_evaluate(self):
        with mock.patch.object(ft, 'UPDATES', 4), mock.patch.object(ft, 'SAVE_EVERY', 2), \
             mock.patch.object(ft, 'WARMUP', 1), mock.patch.object(ft, 'DATA_OFFSET', 4):
            out = self.root/'ft'
            ft.train(2, False, loaded=self.loaded, data=Data(), device=torch.device('cpu'), out_dir=out)
            with self.assertRaises(FileExistsError):
                ft.train(2, False, loaded=self.loaded, data=Data(), device=torch.device('cpu'), out_dir=out)
            loaded = trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=self.root/'final')
            manifest = ft.train(4, True, loaded=loaded, data=Data(), device=torch.device('cpu'), out_dir=out)
            self.assertEqual((manifest['status'], manifest['FDMA_updates']), ('completed', 0))
            fresh = trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=self.root/'final')
            report = ft.evaluate(torch.device('cpu'), loaded=fresh, ft_dir=out, data=Data(), out_root=self.root/'r',
                sizes=dict(selection=1, final=1, midpoint=1))
        prov = json.loads((report/'provenance.json').read_text())
        self.assertEqual(len(prov['switch_decision']), 6)
        self.assertLess(digest(report)['max_relative_mean_difference'], 1e-9)


if __name__ == '__main__':
    unittest.main()
