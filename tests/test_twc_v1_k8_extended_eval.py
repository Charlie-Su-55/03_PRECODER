"""CPU tests for the extended (forward-only) evaluation of the final Hybrid."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import final_runtime as rt
import train_twc_v1_k8_final as trainer
import evaluate_twc_v1_k8_extended as ext
from configs import twc_v1_k8_final_config as cfg
from tests.test_twc_v1_k8_final import Data, fixture_sources

ROOT = Path(__file__).resolve().parents[1]


class ExtendedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patch in (mock.patch.object(cfg, 'UPDATES', 2), mock.patch.object(cfg, 'SAVE_EVERY', 2),
                      mock.patch.object(cfg, 'ACCUMULATION', 2), mock.patch.object(cfg, 'WARMUP', 1)):
            patch.start(); self.addCleanup(patch.stop)
        torch.set_num_threads(2)

    def test_pairing_modes(self):
        torch.manual_seed(3)
        H = torch.randn(3, 8, 32, 256, dtype=torch.complex64)
        least, most = ext.choose_pairing(H, 'least'), ext.choose_pairing(H, 'most')
        rand = ext.choose_pairing(H, 'random', torch.Generator().manual_seed(1))
        ref, _ = rt.best_pairing(H)
        self.assertTrue(torch.equal(least, ref))
        for p in (least, most, rand):
            self.assertEqual([sorted(r) for r in p.tolist()], [list(range(8))]*3)
        cl = rt.slot_pair_coupling(rt.permute_users(H, least))
        cm = rt.slot_pair_coupling(rt.permute_users(H, most))
        cr = rt.slot_pair_coupling(rt.permute_users(H, rand))
        self.assertTrue(bool((cl <= cr+1e-6).all() & (cr <= cm+1e-6).all()))

    def test_report_reproduces_base_variants_and_digests(self):
        out = self.root/'run'
        trainer.train(2, False, sources=fixture_sources(), data=Data(), device=torch.device('cpu'),
            out_dir=out, core=dict(commit='f'*40, sha256={'fixture': 'a'}))
        loaded = trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=out)
        table, _ = ext.variants(torch.device('cpu'), loaded=loaded)
        data = Data()
        rows = ext.run_role(table, ('hybrid_final', 'hybrid_paired_reuse_noise', 'fdma_baseline'), data,
            'final', cfg.FINAL_FIRST, 1, (0, 25), torch.device('cpu'))
        import evaluate_twc_v1_k8_final as base_eval
        systems, _, _ = base_eval.systems(torch.device('cpu'), loaded=loaded)
        renamed = dict(hybrid_final='hybrid_final', hybrid_paired_reuse_noise='hybrid_final_paired',
                       fdma_baseline='fdma_ce65000')
        ref = base_eval.run_role(systems, tuple(renamed.values()), data, 'final', cfg.FINAL_FIRST, 1, (0, 25),
            torch.device('cpu'))
        index = {(r['variant'], r['gamma_db'], r['sample_index']): r for r in ref}
        for row in rows:
            other = index[renamed[row['variant']], row['gamma_db'], row['sample_index']]
            self.assertAlmostEqual(row['sum_rate_bps_hz'], other['sum_rate_bps_hz'], places=5)
        with mock.patch.object(trainer, 'TRAIN_DIR', out):
            report = ext.evaluate(torch.device('cpu'), loaded=loaded, data=data, out_root=self.root/'reports',
                sizes=dict(selection=1, final=1, midpoint=1))
        provenance = json.loads((report/'provenance.json').read_text())
        self.assertEqual(len(provenance['switch_decision']), 6)
        self.assertFalse(provenance['test_used'])
        spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
        rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
        _, _, tier1, _ = rd.build(report)
        audit = tier1['audit_recomputed_from_samples']
        self.assertGreater(audit['matched_summary_points'], 0)
        self.assertLess(audit['max_relative_mean_difference'], 1e-9)
        comps = {l.split(',')[0] for l in (report/'comparisons.csv').read_text().splitlines()[1:]}
        self.assertIn('upper_minus_paired', comps)
        self.assertIn('adaptive_minus_fdma', comps)


if __name__ == '__main__':
    unittest.main()
