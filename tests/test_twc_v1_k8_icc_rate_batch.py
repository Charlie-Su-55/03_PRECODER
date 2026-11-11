"""Owned queue/evidence/failure reporting tests; no server or training."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_twc_v1_k8_icc_rate as runner


class RateQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        def inside(path):
            p=Path(path).resolve()
            if p==self.root or not p.is_relative_to(self.root):raise ValueError('owned child path required')
            return p
        def atomic_json(path,value,exclusive=False):
            path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x' if exclusive else 'w') as f:json.dump(value,f)
        self.runtime=SimpleNamespace(inside=inside,atomic_json=atomic_json,
            sha256=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            read_json=lambda p:json.loads(Path(p).read_text()),
            guard_core=lambda x:self.assertEqual(x,{'core':'tiny'}))
        self.cfg=SimpleNamespace(ROOT=self.root,OUTPUT_ROOT=self.root/'rate',NAMESPACE='tiny_rate',
            ARMS=('fdma','hybrid_g4'),UPDATES=5000)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(runner,'_runtime',return_value=self.runtime).start()
        mock.patch.object(runner,'_config',return_value=self.cfg).start()

    def test_completed_hash_bound_stage_reuses_and_refuses_changed_evidence(self):
        path=self.root/'batch'/'summary.json';path.parent.mkdir()
        summary=dict(stages={},core={'core':'tiny'})
        called=[]
        def action(record):
            called.append(record['attempt_id']);p=path.parent/'result.csv';p.write_text('verified\n')
            return {'status':'passed'},[p]
        first=runner.stage(summary,path,'rate_final',action)
        loaded=json.loads(path.read_text())
        self.assertEqual(runner.stage(loaded,path,'rate_final',action),first)
        self.assertEqual(len(called),1)
        (path.parent/'result.csv').write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            runner.stage(loaded,path,'rate_final',action)
        with self.assertRaisesRegex(ValueError,'no evidence'):
            runner.verify_evidence({'status':'completed'})

    def test_actual_failed_child_own_log_and_exception_are_preserved(self):
        path=self.root/'batch'/'summary.json';path.parent.mkdir()
        script=self.root/'fail_child.py';script.write_text("raise RuntimeError('distinct-rate-child-error')\n")
        summary=dict(stages={},core={'core':'tiny'})
        def action(record):
            runner.child(summary,path,record,'fail_child.py',[])
            self.fail('a failed child was treated as completion')
        with self.assertRaisesRegex(RuntimeError,'distinct-rate-child-error'):
            runner.stage(summary,path,'fdma_100',action)
        record=json.loads(path.read_text())
        self.assertEqual(record['status'],'failed')
        self.assertIn('distinct-rate-child-error',record['stages']['fdma_100']['child_error'])
        self.assertIn('distinct-rate-child-error',record['stages']['fdma_100']['traceback'])
        self.assertTrue(Path(record['stages']['fdma_100']['log_path']).is_file())

    def test_fixed_budget_stage_targets_and_no_ce_training_stage(self):
        stages=runner.stage_names()
        self.assertEqual(stages,('source_baseline','fdma_100','fdma_5000','hybrid_g4_100',
            'hybrid_g4_5000','rate_final','paired_comparison','result_package'))
        self.assertFalse(any(name.startswith('ce_') for name in stages))

    def test_pointer_namespace_and_owned_path_are_strict(self):
        self.cfg.OUTPUT_ROOT.mkdir()
        pointer=self.cfg.OUTPUT_ROOT/'last_batch.json'
        pointer.write_text(json.dumps({'namespace':'old_namespace','summary':str(self.root/'old.json')}))
        with self.assertRaisesRegex(ValueError,'Own rate-experiment pointer'):
            runner.load_summary()
        with self.assertRaisesRegex(ValueError,'owned child'):
            runner.files_identity([Path('/tmp/not_owned_by_test')])

    def test_summary_distinguishes_saved_boundary_from_active_stage(self):
        folder=self.cfg.OUTPUT_ROOT/'training'/'fdma';folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps({'step':800,'status':'paused',
            'ETA_seconds':120.,'updated':'record-time'}))
        summary=dict(status='running',current_stage='fdma_5000',updated='batch-time',
            training_code={'commit':'a'*40},stages={})
        text=runner.compact_summary(summary)
        self.assertIn('running | stage=fdma_5000',text)
        self.assertIn('saved_boundary=800/5000',text)
        self.assertNotIn('fdma: paused',text)
        self.assertIn('no CE/UE updates',text)


if __name__=='__main__':unittest.main()
