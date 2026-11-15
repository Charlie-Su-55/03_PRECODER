"""Controller lifecycle fixtures, with no training, CUDA or GitHub access."""
from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_twc_v1_k8_ue_bs_rate as queue

REAL_PREFLIGHT = queue.preflight_action
REAL_VERIFY_PREFLIGHT = queue.verify_preflight
REAL_VERIFY_TERMINAL = queue.verify_terminal
REAL_TRAINING_ACTION = queue.training_action


class JointRateQueueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'run'
        self.core = dict(schema='fixture_core',sha256={'source.py':'a'*64})
        self.events = []
        self.fail = None
        self.cfg = SimpleNamespace(ROOT=self.root,OUTPUT_ROOT=self.output,NAMESPACE='fixture_joint',
            ARMS=('fdma','hybrid_g4'),TRAIN_METHODS=('joint',),METHODS=('source','bs_only','joint'),
            GAMMAS=(0,5,10,15,20,25),UPDATES=200,MONITOR_STEPS=(0,100,200),MICRO_BATCH=4,
            SOURCE_CONTROL_PINS={'fdma':{},'hybrid_g4':{}},TRAINING_TRACE_PIN={'fixture':True},
            key=lambda a,m:m+'_'+a,
            recipe=lambda a,m='joint':dict(arm=a,method=m,updates=200),
            fingerprint=lambda a,m='joint':a+'_'+m,
            counters=lambda s,base=0:dict(optimizer_updates=s,gamma_counts={'0':s},base_EMA_updates=base),
            study_recipe=lambda:dict(fixture=True,updates=200),study_fingerprint=lambda:'d'*64)

        def inside(value):
            p = Path(value)
            if p.is_symlink() or p.resolve()==self.output or not p.resolve().is_relative_to(self.output):
                raise ValueError('Owned fixture child required')
            return p.resolve()
        def atomic(path,value,exclusive=False):
            p=inside(path)
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('x' if exclusive else 'w',encoding='utf-8') as stream:
                json.dump(value,stream,sort_keys=True)
        @contextmanager
        def lock(path):
            yield
        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        def guard(value):
            if value != self.core:
                raise ValueError('Fixture source changed')
        self.rt = SimpleNamespace(inside=inside,atomic_json=atomic,sha256=digest,
            read_json=lambda p:json.loads(Path(p).read_text()),process_lock=lock,
            core_identity=lambda:copy.deepcopy(self.core),source_identity=lambda:dict(commit='e'*40,sha256={}),
            guard_core=guard,verify_files=lambda source:None)
        for patch in (mock.patch.object(queue,'_config',return_value=self.cfg),
                mock.patch.object(queue,'_runtime',return_value=self.rt)):
            patch.start()
            self.addCleanup(patch.stop)

        def verify_preflight(directory):
            return self.rt.read_json(Path(directory)/'manifest.json')
        def verify_terminal(a,m):
            row=self.rt.read_json(self.output/'training'/self.cfg.key(a,m)/'terminal_receipt.json')
            if row['step'] != self.cfg.UPDATES or row['status'] != 'completed':
                raise ValueError('Fixture terminal incomplete')
            return row
        def verify_manifest(directory):
            return self.rt.read_json(Path(directory)/'manifest.json')
        def binding(path,final,report):
            meta=self.rt.read_json(Path(report)/'export_manifest.json')
            if (meta['final_manifest_sha256']!=digest(Path(final)/'manifest.json')
                    or meta['preregistration_sha256']!=digest(Path(path).parent/'preregistration.json')):
                raise ValueError('Foreign export binding')
            return meta
        self.publisher=SimpleNamespace(verify_export_binding=binding)
        def boundary(directory,arm,method='joint',core=None):
            row=self.rt.read_json(Path(directory)/'boundary_00100.json')
            if row['production_evidence'] is not True:raise ValueError('CPU fixture')
            if row['step']!=100 or row['arm']!=arm or row['method']!=method:
                raise ValueError('Boundary changed')
            return row

        patches=(mock.patch.object(queue,'gpu_admission',side_effect=lambda **kw:self.events.append('admission') or {'accepted':True}),
            mock.patch.object(queue,'verify_boundary',side_effect=boundary),
            mock.patch.object(queue,'preflight_action',side_effect=self.preflight),
            mock.patch.object(queue,'training_action',side_effect=self.training),
            mock.patch.object(queue,'evaluation_action',side_effect=self.evaluation),
            mock.patch.object(queue,'export_action',side_effect=self.export),
            mock.patch.object(queue,'verify_preflight',side_effect=verify_preflight),
            mock.patch.object(queue,'verify_terminal',side_effect=verify_terminal),
            mock.patch.object(queue,'verify_manifest',side_effect=verify_manifest),
            mock.patch.object(queue,'_publisher',return_value=self.publisher))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def args(self,resume=False):
        return SimpleNamespace(resume=resume,allow_shared_gpu=True,min_free_mib=12000,max_gpu_util=80)

    def preflight(self,summary,path,record):
        self.events.append('preflight')
        directory=self.output/'preflight'/('attempt_'+record['attempt_id'])
        self.rt.atomic_json(directory/'manifest.json',{'status':'passed'})
        return dict(output_dir=str(directory)),[directory/'manifest.json']

    def training(self,summary,path,arm,method,target,record):
        key=self.cfg.key(arm,method)
        self.events.append((key,target))
        if self.fail==(key,target):
            raise RuntimeError('Deliberate controller fixture failure')
        directory=self.output/'training'/key
        initial=dict(arm=arm,method=method,core=summary['core'],source_identity={'fixture':True},
            frozen_identity={'encoder':'f'*64},recipe=self.cfg.recipe(arm,method),
            fingerprint=self.cfg.fingerprint(arm,method),data_protocol={'fixture':True})
        if not (directory/'initialization.json').exists():
            self.rt.atomic_json(directory/'initialization.json',initial,exclusive=True)
        row=dict(initial,step=target,counters=self.cfg.counters(target),production_evidence=True,
            status='completed' if target==self.cfg.UPDATES else 'paused_boundary')
        self.rt.atomic_json(directory/'manifest.json',row)
        if target==100:
            self.rt.atomic_json(directory/'boundary_00100.json',row,exclusive=True)
            return row,[directory/'initialization.json',directory/'boundary_00100.json']
        self.rt.atomic_json(directory/'terminal_receipt.json',row,exclusive=True)
        return row,[directory/'initialization.json',directory/'terminal_receipt.json',directory/'manifest.json']

    def evaluation(self,summary,path,record):
        self.events.append('final')
        for a in self.cfg.ARMS:
            for m in self.cfg.TRAIN_METHODS:
                self.assertEqual(summary['stages'][self.cfg.key(a,m)+'_200']['status'],'completed')
        directory=self.output/'evaluation'/('attempt_'+record['attempt_id'])
        self.rt.atomic_json(directory/'manifest.json',{'status':'passed','fixture':True})
        (directory/'summary.csv').write_text('arm,method,scope,gamma_db,sum_rate_bps_hz_mean\n')
        (directory/'comparisons.csv').write_text('comparison,metric,scope,mean,batch_ci95_low,batch_ci95_high\n')
        return dict(output_dir=str(directory)),list(directory.iterdir())

    def export(self,path,final,record):
        self.events.append('export')
        directory=self.output/'reports'/('attempt_'+record['attempt_id'])
        meta=dict(final_manifest_sha256=self.rt.sha256(Path(final)/'manifest.json'),
            preregistration_sha256=self.rt.sha256(Path(path).parent/'preregistration.json'))
        self.rt.atomic_json(directory/'export_manifest.json',meta,exclusive=True)
        return dict(path=str(directory)),[directory/'export_manifest.json']

    def run_queue(self,resume=False):
        with redirect_stdout(io.StringIO()):
            queue.run(self.args(resume))
        return queue.load_summary()

    def test_complete_fixed_order_and_completed_resume_do_no_more_work(self):
        summary,path=self.run_queue()
        expected=['admission','preflight']+[(self.cfg.key(a,m),t) for a in self.cfg.ARMS
            for m in self.cfg.TRAIN_METHODS for t in (100,200)]+['final','export']
        self.assertEqual(self.events,expected)
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(summary['total_optimizer_updates'],400)
        self.assertFalse(summary['test_used'])
        self.assertTrue((path.parent/'preregistration.json').is_file())
        count=len(self.events)
        self.run_queue(resume=True)
        self.assertEqual(len(self.events),count)

    def test_failed_terminal_resumes_without_repeating_100_or_preflight(self):
        self.fail=('joint_fdma',200)
        with self.assertRaisesRegex(RuntimeError,'Deliberate'):
            self.run_queue()
        first,path=queue.load_summary()
        self.assertEqual(first['status'],'failed')
        self.assertEqual(first['stages']['joint_fdma_100']['status'],'completed')
        self.fail=None
        resumed,_=self.run_queue(resume=True)
        self.assertEqual(resumed['status'],'completed')
        self.assertEqual(self.events.count('preflight'),1)
        self.assertEqual(self.events.count('admission'),2)
        self.assertEqual(self.events.count(('joint_fdma',100)),1)
        self.assertEqual(len(resumed['stages']['joint_fdma_200']['attempts']),1)

    def test_immutable_100_boundary_survives_latest_terminal_overwrite(self):
        self.run_queue()
        directory=self.output/'training/joint_fdma'
        self.assertEqual(self.rt.read_json(directory/'manifest.json')['step'],200)
        self.assertEqual(queue.verify_boundary(directory,'fdma','joint')['step'],100)

    def test_preregistration_byte_tamper_stops_before_gpu(self):
        _,path=self.run_queue()
        file=path.parent/'preregistration.json'
        file.write_bytes(file.read_bytes()+b'\n')
        with mock.patch.object(queue,'gpu_admission',side_effect=AssertionError('must not admit')):
            with self.assertRaisesRegex(ValueError,'preregistration bytes'):
                self.run_queue(resume=True)

    def test_changed_recipe_stops_resume_before_gpu(self):
        self.run_queue()
        self.cfg.study_recipe=lambda:dict(fixture=True,updates=201)
        with mock.patch.object(queue,'gpu_admission',side_effect=AssertionError('must not admit')):
            with self.assertRaisesRegex(ValueError,'recipe changed'):
                self.run_queue(resume=True)

    def test_completed_boundary_byte_tamper_is_rejected(self):
        self.run_queue()
        p=self.output/'training/joint_fdma/boundary_00100.json'
        p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            self.run_queue(resume=True)

    def test_cpu_boundary_manifest_is_refused(self):
        self.run_queue()
        directory=self.output/'training/joint_fdma'
        data=self.rt.read_json(directory/'boundary_00100.json')
        data['production_evidence']=False
        self.rt.atomic_json(directory/'boundary_00100.json',data)
        with self.assertRaisesRegex(ValueError,'CPU fixture'):
            queue.verify_boundary(directory,'fdma','joint')

    def test_new_run_does_not_overwrite_existing_output(self):
        self.output.mkdir()
        p=self.output/'preserve.txt'
        p.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_queue()
        self.assertEqual(p.read_text(),'keep')

    def test_missing_completed_dependency_is_not_skipped(self):
        summary,path=self.run_queue()
        summary['stages']['source_control_preflight']['status']='failed'
        self.rt.atomic_json(path,summary)
        with self.assertRaisesRegex(ValueError,'dependency order'):
            self.run_queue(resume=True)

    def test_self_consistent_foreign_export_is_still_rejected(self):
        summary,path=self.run_queue()
        row=summary['stages']['directory_export']
        p=Path(row['details']['path'])/'export_manifest.json'
        value=self.rt.read_json(p)
        value['preregistration_sha256']='0'*64
        self.rt.atomic_json(p,value)
        row['files'][str(p)]=self.rt.sha256(p)
        self.rt.atomic_json(path,summary)
        with self.assertRaisesRegex(ValueError,'Foreign export binding'):
            self.run_queue(resume=True)

    def test_actual_failed_child_log_is_preserved(self):
        self.output.mkdir()
        path=self.output/'batches/batch_fixture/batch_summary.json'
        path.parent.mkdir(parents=True)
        script=self.root/'fail.py'
        script.write_text('print("explicit fixture child failure")\nraise SystemExit(3)\n')
        summary=dict(core=self.core,current_stage='fixture')
        record=dict(attempt_id='abc')
        with self.assertRaisesRegex(RuntimeError,'exit=3'):
            queue.child(summary,path,record,str(script),[])
        self.assertIn('explicit fixture child failure',Path(record['log_path']).read_text())

    def test_stdlib_dry_run_and_not_started_summary(self):
        script=Path(queue.__file__)
        result=subprocess.run([sys.executable,'-B','-S',str(script),'--dry-run'],
            cwd=script.parent,capture_output=True,text=True,check=True)
        doc=json.loads(result.stdout)
        self.assertFalse(doc['writes'])
        self.assertFalse(doc['weights_read'])
        self.assertEqual(doc['namespace'],'twc_v1_k8_ue_bs_rate_v1')
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(queue.main(['--summary']),0)
        self.assertIn('not_started',output.getvalue())
        self.assertFalse(self.output.exists())

    def test_preflight_loads_sources_and_controls_on_actual_device(self):
        import torch
        calls=[]
        class Part:
            def __init__(self,digest):self.digest=digest
        def load(a,m,device):
            calls.append(('source',a,str(device)))
            return (SimpleNamespace(encoder=Part(a+'encoder'),decoder=Part(a+'decoder')),
                Part(a+'stats'),Part('commonCE'),dict(base_ema_updates=20000))
        def control(a,device):
            calls.append(('control',a,str(device)))
            return (SimpleNamespace(encoder=Part(a+'encoder'),decoder=Part(a+'newdecoder')),
                Part(a+'stats'),Part('commonCE'),dict(base_ema_updates=20000))
        self.rt.configure_cuda=lambda:torch.device('cuda:0')
        self.rt.load_initial=load;self.rt.load_control=control
        self.rt.verify_control_files=lambda arm:None
        self.rt.state_digest=lambda value:value.digest
        self.rt.make_data=lambda device,role='final':SimpleNamespace(protocol={'device':device,'role':role})
        with mock.patch.object(queue,'verify_preflight',side_effect=REAL_VERIFY_PREFLIGHT), \
                mock.patch.object(torch.cuda,'empty_cache'):
            detail,files=REAL_PREFLIGHT(dict(core=self.core),self.output/'batches/x/batch_summary.json',dict(attempt_id='abc'))
        self.assertEqual(len(calls),4)
        self.assertTrue(all(device=='cuda:0' for _,_,device in calls))
        self.assertEqual(len(files),2)
        self.assertTrue(detail['manifest']['controls_verified'])

    def test_terminal_inspection_is_metadata_only(self):
        record=dict(step=200)
        trainer=SimpleNamespace(verify_terminal_record=mock.Mock(return_value=record),
            verify_terminal=mock.Mock(side_effect=AssertionError('GPU allocation forbidden')))
        with mock.patch.object(queue,'_trainer',return_value=trainer):
            self.assertEqual(REAL_VERIFY_TERMINAL('fdma','joint'),record)
        trainer.verify_terminal_record.assert_called_once_with('fdma')
        trainer.verify_terminal.assert_not_called()

    def test_source_ema_base_comes_from_accepted_preflight_bytes(self):
        self.output.mkdir()
        directory=self.output/'preflight/attempt_abc'
        self.rt.atomic_json(directory/'source_checks.json',dict(sources={'fdma':dict(base_ema_updates=20000)}))
        record=dict(status='completed',details=dict(output_dir=str(directory)),
            files={str(directory/'source_checks.json'):self.rt.sha256(directory/'source_checks.json')})
        summary=dict(stages={'source_control_preflight':record})
        self.assertEqual(queue._source_ema_base(summary,'fdma'),20000)
        self.rt.atomic_json(directory/'source_checks.json',dict(sources={'fdma':dict(base_ema_updates=0)}))
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            queue._source_ema_base(summary,'fdma')

    def test_training_boundary_preserves_nonzero_inherited_ema_counter(self):
        arm,method='fdma','joint'
        directory=self.output/'training/joint_fdma'
        self.rt.atomic_json(directory/'initialization.json',{})
        (directory/'checkpoint_latest.pt').write_bytes(b'fixture checkpoint')
        row=dict(arm=arm,method=method,step=100,base_ema_updates=20000,
            counters=self.cfg.counters(100,20000),core=self.core,recipe=self.cfg.recipe(arm,method),
            fingerprint=self.cfg.fingerprint(arm,method),production_evidence=True,
            checkpoint_sha256=self.rt.sha256(directory/'checkpoint_latest.pt'))
        self.rt.atomic_json(directory/'manifest.json',row)
        self.rt.atomic_json(directory/'boundary_00100.json',row)
        for step in (0,100):
            p=directory/'monitor'/f'step_{step:05d}_receipt.json'
            target=directory/'monitor'/f'step_{step:05d}_summary.csv'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture')
            self.rt.atomic_json(p,dict(files={str(target):self.rt.sha256(target)}))
        log=self.output/'child.log';log.write_text('fixture')
        with mock.patch.object(queue,'child',return_value=log), \
                mock.patch.object(queue,'_source_ema_base',return_value=20000):
            out,files=REAL_TRAINING_ACTION(dict(core=self.core),self.output/'batch.json',arm,method,100,{})
        self.assertEqual(out['counters']['base_EMA_updates'],20000)
        self.assertIn(directory/'boundary_00100.json',files)


if __name__=='__main__':
    unittest.main()
