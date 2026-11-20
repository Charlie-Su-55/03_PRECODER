"""CPU test for the one-time H100 sealed test evaluation and the batch-means export (fixture models)."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

import torch

import grouped_runtime as gr
import train_twc_v1_k8_grouped as gt
import train_twc_v1_k8_decoder_cmp as dc
import evaluate_twc_v1_h100_test as ht
import k16_runtime as r
import train_twc_v1_k16 as t16
from configs import twc_v1_k8_grouped_config as gcfg
from configs import twc_v1_k16_config as c16
from tests.test_twc_v1_k8_final import Data, IdentityCE
from tests.test_twc_v1_k16 import Data as Data16

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class H100TestEval(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        saved = (gt.cfg, gr.cfg, gt.build)
        self.addCleanup(lambda: (setattr(gt, 'cfg', saved[0]), setattr(gr, 'cfg', saved[1]), setattr(gt, 'build', saved[2])))
        torch.set_num_threads(2)

    def test_k8_arms_once_batch_means_and_digest(self):
        ce = (IdentityCE(), torch.full((6, 8, 256), .002), dict(sha256=dict(gcfg.SHARED_CE_SHA256), step=65000))
        for arm in ('hybrid_g1', 'g4_detr'):
            gt_, _ = ht.use_study(arm)
            c = gt_.cfg
            with mock.patch.multiple(c, NMSE_UPDATES=2, UPDATES=2, SAVE_EVERY=2, ACCUMULATION=1, WARMUP=1):
                kw = dict(device=torch.device('cpu'), ce_bundle=ce, data=Data(), out_dir=self.root/arm,
                          core=dict(commit='f'*40, sha256={'fixture': 'a'}))
                gt_.train(arm, 2, False, **kw)
                loaded = gt_.load_terminal(arm, torch.device('cpu'), ce_bundle=ce, directory=self.root/arm)
                self.assertEqual(type(loaded[0]['codec'].decoder).__name__ == 'K8DETRDecoder', arm == 'g4_detr')
                report = ht.evaluate_k8_arm(arm, torch.device('cpu'), loaded=loaded, data=Data(),
                                            out_root=self.root/'rep', batches=2)
                with self.assertRaises(FileExistsError):
                    ht.evaluate_k8_arm(arm, torch.device('cpu'), loaded=loaded, data=Data(),
                                       out_root=self.root/'rep', batches=2)
            prov = json.loads((report/'provenance.json').read_text())
            self.assertTrue(prov['test_used']); self.assertEqual(prov['arm'], arm)
            # batch means reproduce the sample means exactly
            with gzip.open(report/'samples.csv.gz', 'rt') as s:
                samples = list(csv.DictReader(s))
            bm = list(csv.DictReader(open(report/'batch_means.csv')))
            self.assertEqual(len(bm), 2*6*3)
            row = next(x for x in bm if x['metric'] == 'sum_rate_bps_hz' and float(x['gamma_db']) == 10 and x['batch_index'] == '1')
            ref = statistics.mean(float(x['sum_rate_bps_hz']) for x in samples
                                  if float(x['gamma_db']) == 10 and x['batch_index'] == '1')
            self.assertAlmostEqual(float(row['mean']), ref, places=9)
            digest = module('tools/results_digest.py', 'rd').build(report)[2]
            self.assertIn('batch_index', digest['batch_means']['columns'])
            self.assertLess(digest['audit_recomputed_from_samples']['max_relative_mean_difference'], 1e-9)

    def test_k16_once(self):
        patches = dict(CE_UPDATES=2, CE_SAVE_EVERY=2, CE_WARMUP=1, CALIBRATION_BATCHES=1, NMSE_UPDATES=2, UPDATES=2,
                       SAVE_EVERY=2, WARMUP=1, ACCUMULATION=1, OUTPUT_ROOT=self.root/'k16')
        with mock.patch.multiple(c16, **patches), mock.patch.object(t16, 'CE_DIR', self.root/'k16'/'ce'):
            t16.train_ce(device=torch.device('cpu'), data=Data16(), init_state=r.new_ce().state_dict(),
                         out_dir=self.root/'k16'/'ce')
            bundle = t16.load_ce(torch.device('cpu'), self.root/'k16'/'ce')
            systems = {}
            for arm in c16.ARMS:
                t16.train_codec(arm, device=torch.device('cpu'), ce_bundle=bundle, data=Data16(), out_dir=self.root/'k16'/arm)
                systems[arm], _ = t16.load_codec(arm, torch.device('cpu'), bundle, self.root/'k16'/arm)
            decision = {float(g): ('fdma_baseline' if g == 0 else 'hybrid_paired') for g in c16.GAMMAS}
            kw = dict(systems=systems, data=Data16(), out_root=self.root/'rep16', batches=1, decision=decision)
            report = ht.evaluate_k16(torch.device('cpu'), **kw)
            with self.assertRaises(FileExistsError):
                ht.evaluate_k16(torch.device('cpu'), **kw)
        self.assertIn('adaptive_minus_fdma', (report/'comparisons.csv').read_text())
        self.assertTrue(json.loads((report/'provenance.json').read_text())['test_used'])

    def test_export_batch_means_never_touches_source(self):
        src = self.root/'runs'/'ns'/'reports'/'attempt_abc'
        src.mkdir(parents=True)
        with gzip.open(src/'samples.csv.gz', 'wt', newline='') as s:
            w = csv.DictWriter(s, fieldnames=['variant', 'role', 'gamma_db', 'sample_index', 'batch_index', 'sum_rate_bps_hz'])
            w.writeheader()
            for v in ('fdma_baseline', 'other'):
                for i in range(8):
                    w.writerow(dict(variant=v, role='test', gamma_db=0.0, sample_index=i, batch_index=i//4, sum_rate_bps_hz=i))
        before = (src/'samples.csv.gz').read_bytes()
        eb = module('tools/export_batch_means.py', 'eb')
        with mock.patch.object(eb, 'ROOT', self.root):
            eb.main([str(src)])
            with self.assertRaises(SystemExit):
                eb.main([str(src)])
        out = self.root/'runs'/'ns_batch_means'/'reports'/'attempt_abc'
        rows = list(csv.DictReader(open(out/'batch_means.csv')))
        self.assertEqual([(x['variant'], x['batch_index'], float(x['mean'])) for x in rows],
                         [('fdma_baseline', '0', 1.5), ('fdma_baseline', '1', 5.5)])
        self.assertEqual((src/'samples.csv.gz').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
