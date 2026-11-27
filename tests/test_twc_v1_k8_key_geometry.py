"""CPU tests for the key-count ablation (keyed_runtime) and the geometry study."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import final_runtime as fr
import grouped_runtime as gr
import keyed_runtime as kr
import train_twc_v1_k8_grouped as gt
import train_twc_v1_k8_key_ablation as ka
import train_twc_v1_k8_geometry as geo
from configs import twc_v1_k8_key_ablation_config as kcfg
from configs import twc_v1_k8_geometry_config as gcfg
from configs import twc_v1_k8_decoder_cmp_config as dcfg
from tests.test_twc_v1_k8_final import Data, IdentityCE


def fake_grid():
    index = torch.arange(32)
    return index % 8, index // 8, 8, 4


class KeyedLayoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.phy = fr.physics()

    def test_q4_reproduces_default_paired_layout_and_runtime(self):
        new, old = kr.KeyLayout(4), gr.make_layout('hybrid_g4')
        self.assertEqual(new.key_positions, old.key_positions)
        self.assertEqual(new.shared_positions_by_group, old.shared_positions_by_group)
        self.assertEqual(new.groups, old.groups)
        self.assertEqual(kr.active_dimension(new), gr.active_dimension(old))
        raw, pol = torch.randn(2, 8, 216, dtype=torch.complex64), torch.ones(2, 8)
        a, b = kr.build_waveform(raw, new, pol), gr.build_waveform(raw, old, pol)
        self.assertTrue(torch.equal(a.x_grid, b.x_grid) and torch.equal(a.key, b.key))
        for x, y in zip(kr.support_geometry(new), gr.support_geometry(old)):
            self.assertTrue(torch.equal(x, y))
        prior = torch.rand(2, 8, 44)+.5; prior = 48*prior/prior.sum(-1, keepdim=True)
        Y = torch.randn(2, 32, 256, dtype=torch.complex64); H = torch.randn(2, 8, 32, 256, dtype=torch.complex64)
        var, gamma = torch.rand(2, 8, 256)*.01, torch.tensor([0., 20.])
        ref, out = gr.recover(Y, H, old, prior, var, gamma), kr.recover(Y, H, new, prior, var, gamma)
        self.assertTrue(torch.allclose(ref['soft'], out['soft'], atol=1e-6))
        self.assertTrue(torch.allclose(ref['reliability'], out['reliability'], rtol=1e-6))

    def test_q0_q2_q8_budget_energy_and_recovery(self):
        pilots = {p for row in gr.protocol().PILOT_POSITIONS_BY_USER for p in row}
        for q, n in ((0, 48), (2, 46), (8, 40)):
            layout = kr.KeyLayout(q)
            self.assertEqual(kr.active_dimension(layout), n)
            used = [p for row in layout.key_positions for p in row]+[p for blk in layout.shared_positions_by_group for p in blk]
            self.assertEqual(len(used), 192); self.assertEqual(len(set(used)), 192)
            self.assertFalse(set(used) & pilots)
            raw = torch.randn(1, 8, 216, dtype=torch.complex64)
            wave = kr.build_waveform(raw, layout, torch.ones(1, 8))
            self.assertTrue(torch.allclose(wave.active.abs().square().sum(-1), torch.full((1, 8), 48.)))
            self.assertEqual(int((wave.data.abs() > 0).sum(1).max()), 2)
            self.assertEqual(int((wave.data.abs() > 0).sum()), 8*q+2*(192-8*q))
            H = torch.randn(1, 8, 32, 256, dtype=torch.complex64)
            Y = self.phy.transmit(H, wave.x_grid, torch.zeros(1, 32, 256, dtype=torch.complex64), torch.tensor([25.]))
            prior = wave.active.abs().square().clamp_min(1e-3); prior = 48*prior/prior.sum(-1, keepdim=True)
            out = kr.recover(Y, H, layout, prior, torch.zeros(1, 8, 256), torch.tensor([25.]))
            err = (out['soft'][..., :n]-wave.active).abs().square().sum()/wave.active.abs().square().sum()
            self.assertLess(float(err), 1e-3)
            self.assertTrue(bool((out['reliability'][..., n:] == 0).all()))
            support, active, exclusive = kr.support_geometry(layout)
            self.assertEqual((int(active.sum()), int(exclusive.sum())), (8*n, 8*q))
        with self.assertRaises(ValueError):
            kr.KeyLayout(3)


class _Patched(unittest.TestCase):
    cfg = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patch in (mock.patch.object(self.cfg, 'NMSE_UPDATES', 2), mock.patch.object(self.cfg, 'UPDATES', 4),
                      mock.patch.object(self.cfg, 'SAVE_EVERY', 2), mock.patch.object(self.cfg, 'ACCUMULATION', 1),
                      mock.patch.object(self.cfg, 'WARMUP', 1)):
            patch.start(); self.addCleanup(patch.stop)
        torch.set_num_threads(2)
        self.ce = (IdentityCE(), torch.full((6, 8, 256), .002), dict(sha256=dict(self.cfg.SHARED_CE_SHA256), step=65000))
        self.kw = dict(device=torch.device('cpu'), ce_bundle=self.ce, core=dict(commit='f'*40, sha256={'fixture': 'a'}))


class KeyAblationTrainerTests(_Patched):
    cfg = kcfg

    def test_recipe_identity_and_data_stream_of_g4_joint(self):
        self.assertEqual((kcfg.SEED, kcfg.DATA_SEED, kcfg.DATA_NAMESPACE), (dcfg.SEED, dcfg.SEED, dcfg.NAMESPACE))
        self.assertEqual(kcfg.CODEC_INIT_SEED, dcfg.CODEC_INIT_SEED)
        self.assertEqual((kcfg.NMSE_UPDATES, kcfg.UPDATES), (2, 4))   # patched here; real values below
        self.assertEqual(len({kcfg.fingerprint(a) for a in kcfg.ARMS}), 3)
        self.assertNotIn(kcfg.NAMESPACE, (dcfg.NAMESPACE,))
        self.assertEqual({a: kcfg.recipe(a)['data_dimensions_per_ue'] for a in kcfg.ARMS}, dict(q0=48, q2=46, q8=40))

    def test_train_resume_evaluate_and_test_once(self):
        restore = ka.install(); self.addCleanup(restore)
        self.assertIs(gr.forward, kr.forward)
        out = self.root/'q0'
        gt.train('q0', 2, False, data=Data(), out_dir=out, **self.kw)
        manifest = gt.train('q0', 4, True, data=Data(), out_dir=out, **self.kw)
        self.assertEqual((manifest['status'], manifest['FDMA_updates']), ('completed', 0))
        loaded = gt.load_terminal('q0', torch.device('cpu'), ce_bundle=self.ce, directory=out)
        self.assertEqual(loaded[0]['layout'].keys, 0)
        with mock.patch.object(kcfg, 'output_root', lambda a: self.root/a):
            report = gt.evaluate('q0', torch.device('cpu'), loaded=loaded, data=Data(), out_root=self.root/'rep',
                                 sizes=dict(final=1, midpoint=1))
        self.assertIn('q0', (report/'summary.csv').read_text())
        test = ka.evaluate_once_test('q0', torch.device('cpu'), loaded=loaded, data=Data(), out_root=self.root/'test',
                                     batches=1)
        prov = json.loads((test/'provenance.json').read_text())
        self.assertEqual((prov['namespace'], prov['layout']['keys_per_ue'], prov['test_used']),
                         (kcfg.TEST_NAMESPACE, 0, True))
        self.assertTrue((test/'batch_means.csv').is_file())
        with self.assertRaises(FileExistsError):
            ka.evaluate_once_test('q0', torch.device('cpu'), loaded=loaded, data=Data(), out_root=self.root/'test',
                                  batches=1)
        restore()
        self.assertIsNot(gr.forward, kr.forward)


class GeometryTests(_Patched):
    cfg = gcfg

    def test_training_ramp(self):
        grid = fake_grid()
        a, b = geo.train_ramp(grid, 7, 64, 'cpu'), geo.train_ramp(grid, 7, 64, 'cpu')
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.allclose(a.abs(), torch.ones_like(a.abs())))
        zero = (a == 1).all(-1).all(-1)          # D = 0 draws leave the whole MU unrotated
        self.assertTrue(0 < int(zero.sum()) < 64)
        data = geo.RotatedTraining(Data(), grid)
        plain, rotated = Data().sample('codec_train', 3), data.sample('codec_train', 3)
        ramp = geo.train_ramp(grid, 3, 4, 'cpu')
        self.assertTrue(torch.allclose(rotated['H_ul'], plain['H_ul']*ramp[..., None]))
        self.assertTrue(torch.allclose(rotated['H_dl'].abs(), plain['H_dl'].abs()))
        self.assertTrue(torch.equal(data.sample('validation', 3)['H_dl'], Data().sample('validation', 3)['H_dl']))

    def test_train_both_arms_and_evaluate(self):
        restore = geo.install(); self.addCleanup(restore)
        data, systems, manifests = geo.RotatedTraining(Data(), fake_grid()), {}, {}
        for arm in gcfg.ARMS:
            out = self.root/arm
            gt.train(arm, 2, False, data=data, out_dir=out, **self.kw)
            manifests[arm] = gt.train(arm, 4, True, data=data, out_dir=out, **self.kw)
            systems[arm] = gt.load_terminal(arm, torch.device('cpu'), ce_bundle=self.ce, directory=out)[0]
        self.assertEqual(gr.active_dimension(systems['fdma_geo']['layout']), 24)
        self.assertEqual(gr.active_dimension(systems['hybrid_geo']['layout']), 44)
        self.assertEqual(gcfg.recipe('fdma_geo')['new_FDMA_model_updates'], gcfg.UPDATES)
        systems['g4_joint_cdlc'] = geo.build('hybrid_geo', torch.device('cpu'), self.ce[0], self.ce[1])
        for module in (systems['g4_joint_cdlc']['codec'], systems['g4_joint_cdlc']['stats']):
            module.eval()
        report = geo.evaluate(torch.device('cpu'), systems=systems, manifests=manifests, data=Data(), grid=fake_grid(),
                              out_root=self.root/'rep', batches=1, rotations=(0, 2))
        summary = (report/'summary.csv').read_text()
        for name in geo.VARIANTS:
            self.assertIn('rot2:'+name, summary)
        self.assertIn('ul_nmse_linear', summary)
        self.assertIn('rot2:paired_minus_fdma', (report/'comparisons.csv').read_text())
        prov = json.loads((report/'provenance.json').read_text())
        self.assertEqual((prov['test_used'], prov['frozen_K8_FDMA_loaded']), (False, False))


if __name__ == '__main__':
    unittest.main()
