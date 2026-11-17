"""CPU tests for the final Hybrid experiment (real encoder/receiver, tiny decoder)."""
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import final_runtime as rt
import train_twc_v1_k8_final as trainer
import evaluate_twc_v1_k8_final as evaluator
import twc_v1_k8_final_results as results
from configs import twc_v1_k8_final_config as cfg
from models.twc_v1_k8_icc_aligned import K8ICCAlignedEncoder

ROOT = Path(__file__).resolve().parents[1]


class TinyDecoder(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        base = torch.randn(8, 32, 256, 2, generator=g)
        self.register_buffer('base', torch.view_as_complex(base))
        self.weight = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, *, soft, **_):
        gain = 1+self.weight*soft.real.mean(-1)
        return dict(H_hat=(self.base[None]*gain[:, :, None, None]).to(torch.complex64))


class Data:
    def sample(self, role, index):
        g = torch.Generator().manual_seed(hash((role, index)) % 2**31)
        def channel():
            value = torch.view_as_complex(torch.randn(4, 8, 32, 256, 2, generator=g))
            return (value/value.abs().square().mean((2, 3), keepdim=True).sqrt()).to(torch.complex64)
        return dict(H_dl=channel(), H_ul=channel(), address=dict(role=role, index=index))


class IdentityCE(torch.nn.Module):
    def forward(self, Y, pilots, initial, *, gamma_db):
        return initial


def fixture_sources():
    phy = rt.physics()
    torch.manual_seed(7)
    table_old = torch.full((6, 8, 256), .004)
    table_new = torch.full((6, 8, 256), .002)
    hybrid = torch.nn.Module(); hybrid.encoder = K8ICCAlignedEncoder(); hybrid.decoder = TinyDecoder(1)
    fdma = torch.nn.Module(); fdma.encoder = K8ICCAlignedEncoder(); fdma.decoder = TinyDecoder(2)
    hstats = phy.ReceiverStatistics(phy.make_layout('hybrid_g4'), table_old)
    fstats = phy.ReceiverStatistics(phy.make_layout('fdma'), table_old)
    ce = IdentityCE()
    return dict(hybrid=(hybrid, hstats, ce, dict(sha256='h', state_sha256='hs', CE_state_sha256='c')),
        fdma=(fdma, fstats, ce, {}), ce65000=(IdentityCE(), dict(checkpoint_sha256='n', model_state_sha256='ns')),
        table65000=(table_new, dict(path='fixture', sha256='x')))


class RuntimeTests(unittest.TestCase):
    def test_zero_init_encoder_matches_source(self):
        torch.manual_seed(0)
        source = K8ICCAlignedEncoder().eval()
        wrapped = rt.GammaAwareEncoder(source).eval()
        H = torch.randn(1, 8, 32, 256, dtype=torch.complex64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(wrapped(H, torch.tensor([3.])), source(H), atol=1e-6))

    def test_batched_receiver_matches_original(self):
        phy = rt.physics()
        torch.manual_seed(1)
        for arm in ('fdma', 'hybrid_g4'):
            layout = phy.make_layout(arm)
            n = phy.active_dimension(layout)
            prior = torch.rand(8, n)+.5; prior = 48*prior/prior.sum(-1, keepdim=True)
            Y = torch.randn(2, 32, 256, dtype=torch.complex64)
            H = torch.randn(2, 8, 32, 256, dtype=torch.complex64)
            var = torch.rand(2, 8, 256)*.01
            gamma = torch.tensor([0., 15.])
            reference = phy.recover(Y, H, layout, prior, var, gamma)
            batched = rt.recover_batched(Y, H, layout, prior[None].expand(2, -1, -1), var, gamma)
            self.assertTrue(torch.allclose(reference['soft'], batched['soft'], atol=1e-5, rtol=1e-4))
            self.assertTrue(torch.allclose(reference['reliability'], batched['reliability'], rtol=1e-4))

    def test_node_statistics_interpolate_and_commit(self):
        phy = rt.physics()
        layout = phy.make_layout('hybrid_g4')
        power = torch.full((8, 44), 48/44)
        stats = rt.NodeStatistics(layout, torch.arange(6.)[:, None, None].expand(6, 8, 256), power).train()
        self.assertTrue(torch.allclose(stats.ce_variance(torch.tensor([2.5]))[0], torch.full((8, 256), .5)))
        node_power = torch.zeros(6, 8, 44); node_power[0, :, :4] = 10.; node_power[0, :, 4:] = 1.
        stats.commit_power(node_power*2, torch.tensor([2, 0, 0, 0, 0, 0]))
        prior = stats.nominal_power(torch.tensor([0., 25.]))
        self.assertTrue(torch.allclose(prior.sum(-1), torch.full((2, 8), 48.)))
        self.assertGreater(float(prior[0, 0, 0]), float(prior[1, 0, 0]))
        self.assertEqual(stats.node_updates.tolist(), [1, 0, 0, 0, 0, 0])

    def test_best_pairing_is_exhaustive_optimum(self):
        self.assertEqual(len(rt._pairings()), 105)
        torch.manual_seed(2)
        H = torch.randn(3, 8, 32, 256, dtype=torch.complex64)
        perm, cost = rt.best_pairing(H)
        c = rt.coupling(H)
        for b in range(3):
            brute = min(sum(float(c[b, i, j]) for i, j in p)/4 for p in rt._pairings())
            self.assertAlmostEqual(float(cost[b]), brute, places=5)
            self.assertEqual(sorted(perm[b].tolist()), list(range(8)))
            self.assertAlmostEqual(float(rt.slot_pair_coupling(rt.permute_users(H, perm))[b]), brute, places=5)

    def test_schedule_and_lr(self):
        counts = {}
        for micro in range(cfg.ACCUMULATION):
            for g in cfg.gamma_schedule(cfg.micro_index(1, micro)):
                counts[g] = counts.get(g, 0)+1
        self.assertEqual([counts[g] for g in (0, 5, 10)], [8, 8, 8])
        self.assertEqual(sum(counts[g] for g in (15, 20, 25)), 8)
        self.assertAlmostEqual(cfg.lr_factor(cfg.UPDATES), cfg.MIN_FACTOR)
        self.assertAlmostEqual(cfg.lr_factor(cfg.WARMUP), 1.)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patch in (mock.patch.object(cfg, 'UPDATES', 4), mock.patch.object(cfg, 'SAVE_EVERY', 2),
                      mock.patch.object(cfg, 'ACCUMULATION', 2), mock.patch.object(cfg, 'WARMUP', 1)):
            patch.start(); self.addCleanup(patch.stop)
        torch.set_num_threads(2)

    def _train(self, out, stop, resume=False):
        return trainer.train(stop, resume, sources=fixture_sources(), data=Data(), device=torch.device('cpu'),
            out_dir=out, core=dict(commit='f'*40, sha256={'fixture': 'a'}))

    def test_resume_is_exact_and_evaluation_report_digests(self):
        straight, split = self.root/'straight', self.root/'split'
        self._train(straight, 4)
        self._train(split, 2)
        manifest = self._train(split, 4, resume=True)
        self.assertEqual(manifest['status'], 'completed')
        a = torch.load(straight/'checkpoint_latest.pt', weights_only=False)
        b = torch.load(split/'checkpoint_latest.pt', weights_only=False)
        for key in a['ema']:
            self.assertTrue(torch.equal(a['ema'][key], b['ema'][key]), key)
        self.assertEqual(b['statistics']['node_updates'].tolist()[:3], [4, 4, 4])
        sources = fixture_sources()
        before = {k: v.clone() for k, v in sources['hybrid'][0].state_dict().items()}
        loaded = trainer.load_terminal_ema(torch.device('cpu'), sources=sources, directory=split)
        after = sources['hybrid'][0].state_dict()
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before), 'source Hybrid must stay unchanged')
        self.assertFalse(torch.equal(loaded[0]['codec'].decoder.weight, sources['hybrid'][0].decoder.weight))
        with mock.patch.object(trainer, 'TRAIN_DIR', split):
            report = evaluator.evaluate(torch.device('cpu'), loaded=loaded, data=Data(), out_root=self.root/'reports',
                sizes=dict(selection=1, final=2, midpoint=1))
        names = {p.name for p in report.iterdir()}
        self.assertTrue({'samples.csv.gz', 'summary.csv', 'comparisons.csv', 'provenance.json',
            'export_manifest.json', 'curve_summary.csv'} <= names)
        provenance = json.loads((report/'provenance.json').read_text())
        self.assertIn(provenance['fdma_baseline'], ('fdma_ce5000', 'fdma_ce65000'))
        self.assertEqual(len(provenance['switch_decision']), 6)
        spec = importlib.util.spec_from_file_location('rd', ROOT/'tools'/'results_digest.py')
        rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
        namespace, export_id, tier1, optional = rd.build(report)
        audit = tier1['audit_recomputed_from_samples']
        self.assertGreater(audit['matched_summary_points'], 0)
        self.assertLess(audit['max_relative_mean_difference'], 1e-9)

    def test_resume_refuses_changed_recipe(self):
        out = self.root/'run'
        self._train(out, 2)
        with mock.patch.object(cfg, 'LR_NEW', 2e-4), self.assertRaises(ValueError):
            self._train(out, 4, resume=True)


if __name__ == '__main__':
    unittest.main()
