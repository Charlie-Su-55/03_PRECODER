"""CPU-only queue integrity checks; no model, CUDA or training is invoked."""
from contextlib import nullcontext
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import run_twc_v1_k8_w_residual as runner


class ResidualQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'residual'
        self.output.mkdir()
        self.core = {'schema': 'tiny_residual_core', 'sha256': {'test': 'a'*64}}
        arms = ('fdma','hybrid_g4')
        policies = ('rate_source','rzf_continue','w_residual')
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='tiny_residual', ARMS=arms, TRAIN_METHODS=policies[1:], POLICIES=policies,
            METHODS=tuple(f'{policy}_{arm}' for policy in policies for arm in arms),
            GAMMAS=(0,5,10,15,20,25), MICRO_BATCH=4, FINAL_BATCHES=256,
            UPDATES=5000, MONITOR_STEPS=(0,100,1000,2500,5000),
            key=lambda arm,method: method+'_'+arm,
            method_name=lambda policy,arm: policy+'_'+arm,
            counters=lambda step: {'optimizer_updates':step, 'microbatches':step*8,
                                   'MU_frames':step*32,'EMA_commits':0},
            study_fingerprint=lambda: 'd'*64,
            fingerprint=lambda arm,method: hashlib.sha256((arm+method).encode()).hexdigest(),
            study_recipe=lambda: {'namespace':'tiny_residual','updates_per_arm_method':5000})
        def inside(path):
            path = Path(path).resolve()
            if path == self.output or not path.is_relative_to(self.output):
                raise ValueError('Owned residual-W child required')
            return path
        def atomic_json(path,value,exclusive=False):
            path = inside(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x' if exclusive else 'w') as stream:
                json.dump(value,stream,sort_keys=True,indent=2)
        self.runtime = SimpleNamespace(inside=inside, atomic_json=atomic_json,
            read_json=lambda path: json.loads(Path(path).read_text()),
            sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            core_identity=lambda: self.core,
            guard_core=lambda core: self.assertEqual(core,self.core),
            source_identity=lambda: {'commit':'tiny-test-only'},
            process_lock=lambda path: nullcontext())
        for patcher in (mock.patch.object(runner,'_runtime',return_value=self.runtime),
                        mock.patch.object(runner,'_config',return_value=self.cfg)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def phase(self,phase):
        directory = self.output/('preflight' if phase=='preflight' else 'evaluation/final')/'attempt_tiny'
        directory.mkdir(parents=True)
        contents = {'source_checks.json':b'{"source_hashes_verified":true,"warmstart_zero":true}\n'}
        if phase=='final':
            contents = {'samples.csv.gz':gzip.compress(b'arm,gamma_db,value\nrate_source_fdma,0,1\n',mtime=0),
                'summary.csv':b'arm,gamma_db,sum_rate_bps_hz_mean\nrate_source_fdma,0,1\n',
                'comparisons.csv':b'comparison,metric,scope,mean\nfdma_w_residual_minus_rzf_continue,sum_rate_bps_hz,all_six,0\n',
                'input_fingerprints.json':b'[]\n',
                'training_pairing.json':b'{"status":"passed","actual_training_inputs_paired":true}\n'}
        for name,data in contents.items():
            (directory/name).write_bytes(data)
        manifest = dict(status='passed',phase=phase,core=self.core,
            recipe_fingerprint=self.cfg.study_fingerprint(),
            files={name:self.runtime.sha256(directory/name) for name in contents})
        if phase=='final':
            manifest.update(batch_size=4,n_batches=256,samples_per_condition=1024,
                gamma_db=list(self.cfg.GAMMAS),methods=list(self.cfg.METHODS),test_used=False,
                no_optimizer=True,frozen_models_verified=True,transmitted_precoder_scored=True,
                training_pairing_verified=True)
        else:
            manifest.update(optimizer_updates=0,test_used=False)
        self.runtime.atomic_json(directory/'manifest.json',manifest,exclusive=True)
        return directory

    def training_boundary(self,arm,method,target):
        directory = self.output/'training'/self.cfg.key(arm,method)
        directory.mkdir(parents=True,exist_ok=True)
        checkpoint = directory/'checkpoint_latest.pt'
        checkpoint.write_bytes(b'tiny queue fixture, not a serialized model '+str(target).encode())
        manifest = dict(arm=arm,method=method,step=target,counters=self.cfg.counters(target),
            core=self.core,recipe_fingerprint=self.cfg.fingerprint(arm,method),
            checkpoint_sha256=self.runtime.sha256(checkpoint),
            status='completed' if target==self.cfg.UPDATES else 'paused')
        self.runtime.atomic_json(directory/'manifest.json',manifest)
        (directory/'progress.jsonl').write_text(json.dumps({
            'step':target,'paired_input_hashes':[{'H_dl':'1'*64,'H_ul':'2'*64,
                'standard_noise':'3'*64,'pilot_polarity':'4'*64,'gamma':'5'*64}]*8})+'\n')
        if not (directory/'initialization.json').exists():
            self.runtime.atomic_json(directory/'initialization.json',
                {'arm':arm,'method':method,'source':'pinned-rate5000','fresh_optimizer':True},exclusive=True)
        receipt = directory/('boundary_00100.json' if target==100 else 'terminal_receipt.json')
        self.runtime.atomic_json(receipt,manifest)
        for step in self.cfg.MONITOR_STEPS:
            if step<=target:
                output = directory/'monitor'/f'step_{step:05d}.csv'
                output.parent.mkdir(exist_ok=True)
                output.write_text('fixture-only\n')
                self.runtime.atomic_json(directory/'monitor'/f'step_{step:05d}_receipt.json',
                    {'step':step,'files':{str(output):self.runtime.sha256(output)}})
        return manifest

    def test_stage_plan_trains_both_methods_and_uses_declared_budget(self):
        self.cfg.UPDATES = 7000
        names = runner.stage_names()
        self.assertEqual(names[0],'source_preflight')
        self.assertEqual(names[-2:],('final_validation','result_package'))
        self.assertEqual(len(names),11)
        for arm in self.cfg.ARMS:
            for method in self.cfg.TRAIN_METHODS:
                self.assertIn(self.cfg.key(arm,method)+'_100',names)
                self.assertIn(self.cfg.key(arm,method)+'_7000',names)
        self.assertFalse(any('rate_source' in name for name in names))

    def test_completed_stage_reuse_requires_unchanged_evidence(self):
        path = self.output/'batch'/'summary.json'
        path.parent.mkdir()
        summary = {'stages':{},'core':self.core}
        calls = []
        def action(record):
            calls.append(record['attempt_id'])
            file = path.parent/'boundary.json'
            file.write_text('{"step":100}\n')
            return {'step':100},[file]
        first = runner.stage(summary,path,'rzf_continue_fdma_100',action)
        loaded = self.runtime.read_json(path)
        self.assertEqual(runner.stage(loaded,path,'rzf_continue_fdma_100',action),first)
        self.assertEqual(len(calls),1)
        (path.parent/'boundary.json').write_text('{"step":200}\n')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            runner.stage(loaded,path,'rzf_continue_fdma_100',action)
        with self.assertRaisesRegex(ValueError,'no evidence'):
            runner.verify_evidence({'status':'completed'})

    def test_actual_failed_child_preserves_exact_log_and_tail(self):
        path = self.output/'batch'/'summary.json'
        path.parent.mkdir()
        (self.root/'failed_child.py').write_text("raise RuntimeError('unique-residual-child-failure')\n")
        summary = {'stages':{},'core':self.core}
        def action(record):
            runner.child(summary,path,record,'failed_child.py',[])
            self.fail('failed child treated as success')
        with self.assertRaisesRegex(RuntimeError,'unique-residual-child-failure'):
            runner.stage(summary,path,'w_residual_hybrid_g4_100',action)
        saved = self.runtime.read_json(path)
        record = saved['stages']['w_residual_hybrid_g4_100']
        self.assertEqual(saved['status'],'failed')
        self.assertIn('unique-residual-child-failure',record['child_error'])
        self.assertIn('unique-residual-child-failure',record['traceback'])
        self.assertTrue(Path(record['log_path']).is_file())
        self.assertIn('Exact log: '+record['log_path'],runner.compact_summary(saved))

    def test_failed_source_preflight_prevents_all_training_and_final_evaluation(self):
        args = SimpleNamespace(resume=False,allow_shared_gpu=True,min_free_mib=12000,max_gpu_util=80)
        def evaluation(summary,path,phase,record):
            self.assertEqual(phase,'preflight')
            raise ValueError('source checkpoint digest mismatch')
        with mock.patch.object(runner,'gpu_admission',return_value={'allowed':True}), \
                mock.patch.object(runner,'evaluation_action',side_effect=evaluation) as evaluate, \
                mock.patch.object(runner,'training_action') as train:
            with self.assertRaisesRegex(ValueError,'source checkpoint digest mismatch'):
                runner.run(args)
            train.assert_not_called()
            self.assertEqual(evaluate.call_count,1)
        saved,_ = runner.load_summary()
        self.assertEqual(saved['current_stage'],'source_preflight')
        self.assertEqual(saved['status'],'failed')

    def test_core_guard_runs_before_creating_output(self):
        self.output.rmdir()
        self.runtime.guard_core = mock.Mock(side_effect=ValueError('protected core changed'))
        args = SimpleNamespace(resume=False)
        with self.assertRaisesRegex(ValueError,'protected core changed'):
            runner.run(args)
        self.assertFalse(self.output.exists())

    def test_train_resumes_only_own_checkpoint_and_keeps_100_receipt_reusable(self):
        path = self.output/'batch'/'summary.json'
        path.parent.mkdir()
        summary = {'stages':{},'core':self.core,'current_stage':'rzf_continue_fdma_100'}
        seen = []
        def fake_child(summary,path,record,script,args):
            seen.append(list(args))
            target = int(args[args.index('--stop-after')+1])
            self.training_boundary('fdma','rzf_continue',target)
            log = path.parent/f'train_{target}.log'
            log.write_text('queue fixture child\n')
            return log
        with mock.patch.object(runner,'child',side_effect=fake_child):
            first = runner.stage(summary,path,'rzf_continue_fdma_100',
                lambda record:runner.training_action(summary,path,'fdma','rzf_continue',100,record))
            runner.training_action(summary,path,'fdma','rzf_continue',5000,{'attempt_id':'full'})
        self.assertNotIn('--resume',seen[0])
        self.assertIn('--resume',seen[1])
        self.assertIn('--method',seen[0])
        self.assertIn('rzf_continue',seen[0])
        self.assertFalse(any(name.endswith('checkpoint_latest.pt') for name in first['files']))
        runner.verify_evidence(first)
        with mock.patch.object(runner,'child') as child:
            with self.assertRaisesRegex(ValueError,'Cannot rewind'):
                runner.training_action(summary,path,'fdma','rzf_continue',100,{'attempt_id':'rewind'})
            child.assert_not_called()

    def test_training_boundary_rejects_replaced_checkpoint_and_receipt(self):
        path = self.output/'batch'/'summary.json'
        path.parent.mkdir()
        summary = {'core':self.core}
        log = path.parent/'training.log'
        log.write_text('fixture\n')
        def bad_checkpoint(*args):
            self.training_boundary('hybrid_g4','w_residual',100)
            (self.output/'training'/'w_residual_hybrid_g4'/'checkpoint_latest.pt').write_bytes(b'tampered')
            return log
        with mock.patch.object(runner,'child',side_effect=bad_checkpoint):
            with self.assertRaisesRegex(ValueError,'checkpoint bytes/receipt changed'):
                runner.training_action(summary,path,'hybrid_g4','w_residual',100,{'attempt_id':'bad'})
        directory = self.output/'training'/'w_residual_hybrid_g4'
        self.training_boundary('hybrid_g4','w_residual',100)
        self.runtime.atomic_json(directory/'boundary_00100.json',{'step':99})
        with mock.patch.object(runner,'child',return_value=log):
            with self.assertRaisesRegex(ValueError,'immutable boundary receipt changed'):
                runner.training_action(summary,path,'hybrid_g4','w_residual',100,{'attempt_id':'bad'})

    def test_final_manifest_rejects_changed_budget_recipe_and_artifacts(self):
        final = self.phase('final')
        runner.verify_manifest(final,'final')
        path = final/'manifest.json'
        manifest = self.runtime.read_json(path)
        manifest['n_batches'] = 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'budget/policy changed'):
            runner.verify_manifest(final,'final')
        manifest['n_batches'] = 256
        manifest['recipe_fingerprint'] = '0'*64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'identity changed'):
            runner.verify_manifest(final,'final')
        manifest['recipe_fingerprint'] = self.cfg.study_fingerprint()
        path.write_text(json.dumps(manifest))
        (final/'summary.csv').write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            runner.verify_manifest(final,'final')

    def test_final_manifest_cannot_treat_cpu_pairing_fixture_as_production(self):
        final = self.phase('final')
        path = final/'manifest.json'
        manifest = self.runtime.read_json(path)
        manifest['training_pairing_verified'] = False
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'training_pairing_verified'):
            runner.verify_manifest(final,'final')
        manifest['training_pairing_verified'] = True
        self.runtime.atomic_json(final/'training_pairing.json',{'status':'test_fixture'})
        manifest['files']['training_pairing.json'] = self.runtime.sha256(final/'training_pairing.json')
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'Production residual-W training pairing'):
            runner.verify_manifest(final,'final')

    def test_archive_has_explicit_scalar_allowlist_and_hashes_never_weights(self):
        preflight,final = self.phase('preflight'),self.phase('final')
        batch = self.output/'batches'/'batch_tiny'
        batch.mkdir(parents=True)
        self.runtime.atomic_json(batch/'preregistration.json',{'test_used':False},exclusive=True)
        for arm in self.cfg.ARMS:
            for method in self.cfg.TRAIN_METHODS:
                self.training_boundary(arm,method,self.cfg.UPDATES)
        (final/'extra_weights.pt').write_bytes(b'do not include')
        before = {path:path.read_bytes() for folder in (preflight,final)
                  for path in folder.iterdir() if path.is_file()}
        result,evidence = runner.package_action(batch/'batch_summary.json',preflight,final,
                                              {'attempt_id':'archive_test'})
        self.assertEqual(result['sha256'],self.runtime.sha256(result['path']))
        with zipfile.ZipFile(result['path']) as archive:
            names = set(archive.namelist())
            self.assertIn('samples.csv.gz',names)
            self.assertIn('preflight_manifest.json',names)
            self.assertIn('source_checks.json',names)
            self.assertIn('training_pairing.json',names)
            for arm in self.cfg.ARMS:
                for method in self.cfg.TRAIN_METHODS:
                    self.assertIn(self.cfg.key(arm,method)+'_initialization.json',names)
                    self.assertIn(self.cfg.key(arm,method)+'_terminal_receipt.json',names)
                    self.assertIn(self.cfg.key(arm,method)+'_progress.jsonl',names)
            self.assertFalse(any(name.endswith(('.pt','.pth','.ckpt')) for name in names))
            whitelist = json.loads(archive.read('package_whitelist.json'))
            self.assertEqual(set(whitelist['files'])|{'package_whitelist.json'},names)
            for name,digest in whitelist['files'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),digest)
            self.assertIn(b'rate_source_fdma',gzip.decompress(archive.read('samples.csv.gz')))
        self.assertEqual(len(evidence),2)
        self.assertTrue(all(path.is_file() for path in evidence))
        self.assertTrue(all(path.read_bytes()==value for path,value in before.items()))

    def test_summary_and_dry_run_are_readonly_without_torch(self):
        before = set(self.output.rglob('*'))
        with mock.patch('sys.stdout',new_callable=io.StringIO) as stdout:
            runner.main(['--dry-run'])
            report = json.loads(stdout.getvalue())
        self.assertFalse(report['writes'])
        self.assertFalse(report['weights_read'])
        self.assertEqual(len(report['stages']),11)
        with mock.patch('sys.stdout',new_callable=io.StringIO) as stdout:
            runner.main(['--summary'])
            self.assertIn('not_started',stdout.getvalue())
        self.assertEqual(before,set(self.output.rglob('*')))

    def test_completed_summary_keeps_all_six_curves_and_matched_control_comparisons(self):
        final = self.phase('final')
        with (final/'summary.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=('arm','gamma_db','sum_rate_bps_hz_mean'))
            writer.writeheader()
            for method in self.cfg.METHODS:
                for gamma in self.cfg.GAMMAS:
                    writer.writerow(dict(arm=method,gamma_db=gamma,sum_rate_bps_hz_mean=20+gamma/10))
        labels = [arm+'_w_residual_minus_rzf_continue' for arm in self.cfg.ARMS]
        labels += ['interaction_w_residual_minus_rzf_continue']
        labels += [policy+'_hybrid_minus_fdma' for policy in self.cfg.POLICIES]
        with (final/'comparisons.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=('comparison','metric','scope','mean',
                                                       'batch_ci95_low','batch_ci95_high'))
            writer.writeheader()
            for label in labels:
                for scope in ('all_six','low_0_5_10'):
                    writer.writerow(dict(comparison=label,metric='sum_rate_bps_hz',scope=scope,
                        mean=-.1,batch_ci95_low=-.2,batch_ci95_high=0.))
        summary = dict(status='completed',stages={
            'final_validation':dict(status='completed',details={'output_dir':str(final)}),
            'result_package':dict(status='completed',details={'path':'/own/results.zip'})})
        text = runner.compact_summary(summary)
        self.assertEqual(text.count(' rate: '),6)
        for label in labels:
            self.assertIn(label+' all_six: -0.10000',text)
            self.assertIn(label+' low_0_5_10: -0.10000',text)
        self.assertIn('Small package: /own/results.zip',text)
        self.assertIn('relative gains do not imply Hybrid wins',text)

    def test_pointer_and_evidence_cannot_cross_experiment_namespace(self):
        self.runtime.atomic_json(self.output/'last_batch.json',
            {'namespace':'old_experiment','summary':str(self.root/'old.json')},exclusive=True)
        with self.assertRaisesRegex(ValueError,'Own residual-W pointer'):
            runner.load_summary()
        with self.assertRaisesRegex(ValueError,'Owned residual-W child'):
            runner.files_identity([self.root/'outside.txt'])


if __name__=='__main__':
    unittest.main()
