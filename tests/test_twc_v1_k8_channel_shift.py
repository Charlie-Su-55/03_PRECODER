"""CPU tests for the channel-shift evaluation (rotation ramp, grid parsing, report)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import final_runtime as rt
import train_twc_v1_k8_final as trainer
import evaluate_twc_v1_k8_channel_shift as cs
from configs import twc_v1_k8_final_config as cfg
from tests.test_twc_v1_k8_final import Data, fixture_sources

ROOT = Path(__file__).resolve().parents[1]
LAM = 3e8/3.5e9


def fake_positions():
    # 4 rows (z) x 8 columns (y), row-major, spacing lambda/2, centred
    return torch.tensor([[0., (c-3.5)*LAM/2, (r-1.5)*LAM/2] for r in range(4) for c in range(8)])


class ShiftTests(unittest.TestCase):
    def test_grid_and_ramp_shift_angle(self):
        h, v, nh, nv = cs.antenna_grid(fake_positions())
        self.assertEqual((nh, nv), (8, 4))
        self.assertEqual(h.tolist()[:9], [0, 1, 2, 3, 4, 5, 6, 7, 0])
        self.assertEqual(sorted(set(v.tolist())), [0, 1, 2, 3])
        ramp = cs.rotation_phase((h, v, nh, nv), 2, 'final', 0, 3, 'cpu')
        self.assertTrue(torch.allclose(ramp.abs(), torch.ones(3, 8, 32)))
        # a plane wave at broadside becomes a plane wave: constant phase step along each axis
        wave = ramp[0, 0].reshape(4, 8)
        step = wave[:, 1:]/wave[:, :-1]
        self.assertTrue(torch.allclose(step, step[:1, :1].expand_as(step), atol=1e-5))
        self.assertTrue(torch.equal(ramp, cs.rotation_phase((h, v, nh, nv), 2, 'final', 0, 3, 'cpu')))

    def test_rot0_is_identity_and_rotation_reduces_coupling(self):
        grid = cs.antenna_grid(fake_positions())
        data = {'C': Data()}
        base = data['C'].sample('validation', cfg.FINAL_FIRST)
        same = cs.scenario_sample('rot0', data, grid, 0, 'cpu')
        self.assertTrue(torch.equal(same['H_dl'], base['H_dl']))
        # common-geometry channel: all users identical up to gain -> coupling 1; rotation must lower it
        a = torch.randn(4, 1, 32, 256, dtype=torch.complex64).expand(4, 8, 32, 256).clone()
        ramp = cs.rotation_phase(grid, 8, 'final', 0, 4, 'cpu')
        self.assertGreater(float(cs.true_coupling(a).mean()), .99)
        self.assertLess(float(cs.true_coupling(a*ramp[..., None]).mean()), .5)

    def test_report_digests(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        with mock.patch.object(cfg, 'UPDATES', 2), mock.patch.object(cfg, 'SAVE_EVERY', 2), \
             mock.patch.object(cfg, 'ACCUMULATION', 2), mock.patch.object(cfg, 'WARMUP', 1), \
             mock.patch.object(cs, 'SCENARIOS', ('rot0', 'rot4', 'cdl_d')):
            torch.set_num_threads(2)
            trainer.train(2, False, sources=fixture_sources(), data=Data(), device=torch.device('cpu'),
                out_dir=root/'run', core=dict(commit='f'*40, sha256={'fixture': 'a'}))
            loaded = trainer.load_terminal_ema(torch.device('cpu'), sources=fixture_sources(), directory=root/'run')
            report = cs.evaluate(torch.device('cpu'), loaded=loaded, datasets={'C': Data(), 'D': Data()},
                grid=cs.antenna_grid(fake_positions()), out_root=root/'reports', batches=1)
        prov = json.loads((report/'provenance.json').read_text())
        self.assertEqual(prov['scenarios'], ['rot0', 'rot4', 'cdl_d'])
        spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
        rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
        _, _, tier1, _ = rd.build(report)
        self.assertLess(tier1['audit_recomputed_from_samples']['max_relative_mean_difference'], 1e-9)
        text = (report/'comparisons.csv').read_text()
        self.assertIn('rot4:paired_minus_fdma', text)
        self.assertIn('cdl_d:upper_minus_paired', text)


if __name__ == '__main__':
    unittest.main()
