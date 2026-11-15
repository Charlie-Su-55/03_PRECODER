"""Real local Git publication with pinned archived control and joint evidence."""
from contextlib import contextmanager,redirect_stdout
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import publish_twc_v1_k8_ue_bs_rate as pub
import twc_v1_k8_ue_bs_rate_results as results


class JointPublicationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)/'source';self.root.mkdir()
        self.remote=self.root.parent/'remote.git'
        self.git('init','--initial-branch=main');self.git('init','--bare',str(self.remote))
        (self.root/'.gitignore').write_text('runs/\n')
        (self.root/'source.py').write_text('baseline\n')
        self.git('add','.gitignore','source.py')
        self.git('-c','user.name=Fixture','-c','user.email=test@example.invalid','commit','-m','baseline')
        self.git('remote','add','origin',str(self.remote));self.head=self.git('rev-parse','HEAD')
        (self.root/'source.py').write_text('preserve dirty source\n')
        (self.root/'staged.txt').write_text('preserve staged work\n');self.git('add','staged.txt')
        self.index=(self.root/'.git/index').read_bytes()
        self.output=self.root/'runs/fixture_joint';self.output.mkdir(parents=True)
        recipe=dict(namespace='fixture_joint',new_joint_updates=10000)
        trace=dict(format='fixture_trace_v1',initial_sha256='d'*64,sha256='e'*64,microbatches=80000)
        self.cfg=SimpleNamespace(ROOT=self.root,OUTPUT_ROOT=self.output,NAMESPACE='fixture_joint',
            ARMS=('fdma','hybrid_g4'),METHODS=('source','bs_only','joint'),TRAIN_METHODS=('joint',),
            GAMMAS=(0,5,10,15,20,25),MICRO_BATCH=4,ACCUMULATION=8,UPDATES=10000,FINAL_BATCHES=1,
            TRAINING_TRACE_PIN=trace,SOURCE_CONTROL_PINS={a:dict(checkpoint_sha256='7'*64) for a in ('fdma','hybrid_g4')},
            key=lambda a,m='joint':m+'_'+a,fingerprint=lambda a,m='joint':'d'*64,
            study_recipe=lambda:recipe,study_fingerprint=lambda:'d'*64,
            counters=lambda step:dict(gamma_counts={str(g):step for g in (0,5,10,15,20,25)}),
            recipe=lambda a,m='joint':dict(physical=dict(K=8,Ep=16,Ed=48,total_RE=256)),
            control_identity=lambda a:dict(arm=a,checkpoint_sha256='7'*64))
        def inside(value):
            path=Path(value)
            if path.is_symlink() or path.resolve()==self.output or not path.resolve().is_relative_to(self.output):
                raise ValueError('Owned child required')
            return path.resolve()
        def atomic(path,value,exclusive=False):
            path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x' if exclusive else 'w') as stream:json.dump(value,stream,sort_keys=True)
        @contextmanager
        def lock(path):yield
        self.runtime=SimpleNamespace(inside=inside,atomic_json=atomic,sha256=pub._sha,
            read_json=lambda p:json.loads(Path(p).read_text()),process_lock=lock,
            guard_core=lambda value:None,verify_control_files=lambda arm:None)
        self.core=dict(schema='fixture_core',sha256={'source.py':'a'*64})
        self.final=self.output/'evaluation/attempt_bbbbbbbbbbbb';self.final.mkdir(parents=True)
        rows=[dict(arm=arm,method=method,role='final',gamma_db=g,sample_index=i,
            batch_index=0,within_batch=i,**{name:.1 for name in results.REQUIRED_METRICS})
            for arm in self.cfg.ARMS for method in self.cfg.METHODS for g in self.cfg.GAMMAS for i in range(4)]
        self.csv(self.final/'samples.csv.gz',rows)
        self.csv(self.final/'summary.csv',[dict(arm='fdma',method='joint',scope='all_six',sum_rate_bps_hz_mean=.1)])
        self.csv(self.final/'comparisons.csv',[dict(comparison='joint_minus_bs_only',metric='sum_rate_bps_hz',mean=-.01)])
        atomic(self.final/'input_fingerprints.json',[dict(content_sha256='f'*64)])
        records,signatures={},{}
        for method in self.cfg.METHODS:
            records[method],signatures[method]={},{}
            for ai,arm in enumerate(self.cfg.ARMS):
                state={'source':'a','bs_only':'b','joint':'f'}[method]*64
                encoder=('1' if method=='joint' else 'c')*64
                statistics=(('6' if ai==0 else '7') if method=='joint' else ('4' if ai==0 else '5'))*64
                checkpoint={'source':'6','bs_only':'7','joint':'8'}[method]*64
                records[method][arm]=dict(kind='rate',arm=arm,method=method,
                    step=5000 if method=='source' else self.cfg.UPDATES,
                    checkpoint_sha256=checkpoint,state_sha256=state,CE_state_sha256='3'*64,
                    encoder_state_sha256=encoder,statistics_state_sha256=statistics,
                    path='/home/private/weights.pt',frozen_files=[dict(path='/home/private/weights.pt',sha256='8'*64)])
                before=dict(codec=state,encoder=encoder,decoder=('d' if method=='source' else 'e')*64,
                    statistics=statistics,CE='3'*64,CE_table='9'*64)
                signatures[method][arm]=dict(before=dict(before),after=dict(before))
        self.manifest=dict(status='passed',core=self.core,production_evidence=True,
            data_protocol=dict(seed_namespace='fixture',execution=dict(device='cuda:0')),
            source_records=records,frozen_signatures=signatures,terminal_records=dict(records['joint']),
            control_records=dict(records['bs_only']),all_terminals_verified_before_data=True,
            paired_inputs_verified=True,frozen_models_verified=True,source_initialization_matched=True,
            common_CE_verified=True)
        proof=dict(format='fixture_training_pairing_v1',status='passed',production_evidence=True,
            new_snapshot_pairing_verified=True,common_trace_matches_archived_controls=True,
            archived_raw_progress_replayed=False,common_training_trace=trace,
            hashed_microbatches=80000,hashed_MU_frames=320000,
            gamma_counts=self.cfg.counters(10000)['gamma_counts'],
            trajectories={a:dict(progress_sha256='c'*64,training_trace=trace) for a in self.cfg.ARMS},
            archived_controls={a:self.cfg.control_identity(a) for a in self.cfg.ARMS})
        atomic(self.final/'training_pairing.json',proof);self.manifest['training_pairing']=proof
        atomic(self.final/'manifest.json',self.manifest)
        self.summary_path=self.output/'batches/batch_fixture/batch_summary.json'
        self.summary_path.parent.mkdir(parents=True)
        prereg=self.summary_path.parent/'preregistration.json';prereg.write_text('{"fixture":true}\n')
        self.summary=dict(namespace=self.cfg.NAMESPACE,core=self.core,code=dict(commit=self.head),
            recipe=recipe,recipe_fingerprint='d'*64,preregistration_sha256=pub._sha(prereg),
            status='completed',stages={'final_validation':dict(status='completed',details=dict(output_dir=str(self.final)),
                files={str(self.final/'manifest.json'):pub._sha(self.final/'manifest.json')})})
        atomic(self.summary_path,self.summary)
        def preregistration(summary,path):
            if pub._sha(Path(path).parent/'preregistration.json')!=summary['preregistration_sha256']:
                raise ValueError('Preregistration changed')
        def evidence(record):
            if not record.get('files') or any(pub._sha(Path(p))!=s for p,s in record['files'].items()):
                raise ValueError('Evidence changed')
        self.queue=SimpleNamespace(verify_manifest=lambda p:self.runtime.read_json(Path(p)/'manifest.json'),
            verify_preregistration=preregistration,verify_evidence=evidence,
            load_summary=lambda:(self.runtime.read_json(self.summary_path),self.summary_path),
            stage_names=lambda:('final_validation','directory_export'))
        for patch in (mock.patch.object(pub,'_config',return_value=self.cfg),
                mock.patch.object(pub,'_runtime',return_value=self.runtime),
                mock.patch.object(pub,'_queue',return_value=self.queue),
                mock.patch.object(pub,'REPOSITORY',str(self.remote))):
            patch.start();self.addCleanup(patch.stop)
        self.report=self.output/'reports/attempt_0123456789ab'

    def git(self,*args):
        return subprocess.run(['git',*args],cwd=self.root,capture_output=True,text=True,check=True).stdout.strip()

    @staticmethod
    def csv(path,rows):
        text=io.StringIO()
        writer=csv.DictWriter(text,fieldnames=sorted(set().union(*(set(r) for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
        raw=text.getvalue().encode()
        path.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)

    def export(self):
        exported=pub.export_directory(self.summary_path,self.final,self.report)
        self.summary['stages']['directory_export']=dict(status='completed',details=exported,
            files={str(self.report/'export_manifest.json'):pub._sha(self.report/'export_manifest.json')})
        self.runtime.atomic_json(self.summary_path,self.summary)
        return exported

    def preserved(self):
        self.assertEqual(self.git('rev-parse','HEAD'),self.head)
        self.assertEqual((self.root/'.git/index').read_bytes(),self.index)
        self.assertEqual((self.root/'source.py').read_text(),'preserve dirty source\n')
        self.assertEqual((self.root/'staged.txt').read_text(),'preserve staged work\n')

    def test_exact_public_inventory_no_zip_or_private_sources(self):
        exported=self.export()
        self.assertEqual(set(exported['files']),set(pub.PUBLIC_FILES))
        text=(self.report/'provenance.json').read_text()
        for private in ('/home/','private','frozen_files','weights.pt'):
            self.assertNotIn(private,text)
        self.assertEqual(self.runtime.read_json(self.report/'provenance.json')['total_new_BS_updates'],20000)
        pub.verify_export_directory(self.report)
        self.assertFalse(list(self.output.rglob('*.zip')))
        self.preserved()

    def test_self_consistent_foreign_provenance_and_payload_cannot_be_grafted(self):
        self.export()
        pp=self.report/'provenance.json'
        mp=self.report/'export_manifest.json'
        provenance=self.runtime.read_json(pp)
        original=pp.read_bytes()
        m=self.runtime.read_json(mp)
        provenance['preregistration_sha256']='f'*64
        self.runtime.atomic_json(pp,provenance)
        m['files']['provenance.json']=pub._sha(pp)
        self.runtime.atomic_json(mp,m)
        pub.verify_export_directory(self.report)
        with self.assertRaisesRegex(ValueError,'provenance'):
            pub.verify_export_binding(self.summary_path,self.final,self.report)
        pp.write_bytes(original)
        m['files']['provenance.json']=pub._sha(pp)
        (self.report/'summary.csv').write_text('arm,method,value\nfdma,joint,9\n')
        m['files']['summary.csv']=pub._sha(self.report/'summary.csv')
        self.runtime.atomic_json(mp,m)
        with self.assertRaisesRegex(ValueError,'payload'):
            pub.verify_export_binding(self.summary_path,self.final,self.report)

    def test_all_fixed_controls_terminals_and_frozen_CE_required(self):
        for field in ('all_terminals_verified_before_data','common_CE_verified','production_evidence'):
            self.manifest[field]=False
            self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
            with self.assertRaises(ValueError):self.export()
            self.manifest[field]=True
        self.manifest['terminal_records']['fdma']['step']=9999
        self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
        with self.assertRaises(ValueError):self.export()
        self.assertFalse(self.report.exists())

    def test_joint_may_change_encoder_and_prior_but_not_common_CE_table(self):
        before=self.manifest['frozen_signatures']['joint']['fdma']['before']
        source=self.manifest['frozen_signatures']['source']['fdma']['before']
        self.assertNotEqual(before['encoder'],source['encoder'])
        self.assertNotEqual(before['statistics'],source['statistics'])
        state=self.manifest['frozen_signatures']['joint']['fdma']
        state['before']['CE_table']=state['after']['CE_table']='0'*64
        self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
        with self.assertRaisesRegex(ValueError,'CE-error table'):
            self.export()
        self.assertFalse(self.report.exists())

    def test_archived_control_checkpoint_pin_is_enforced(self):
        self.cfg.SOURCE_CONTROL_PINS['fdma']['checkpoint_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'control identity'):
            self.export()
        self.assertFalse(self.report.exists())

    def test_cannot_overwrite_or_append_neural_weights_or_modify_payload(self):
        self.export()
        with self.assertRaises(FileExistsError):pub.export_directory(self.summary_path,self.final,self.report)
        extra=self.report/'weights.pt'
        extra.write_bytes(b'never publish')
        with self.assertRaises(ValueError):pub.verify_export_directory(self.report)
        extra.unlink()
        (self.report/'summary.csv').write_text('changed')
        with self.assertRaises(ValueError):pub.verify_export_directory(self.report)

    def test_training_pairing_is_public_but_fixture_or_incomplete_proof_is_rejected(self):
        file=self.final/'training_pairing.json'
        original=file.read_bytes()
        proof=self.runtime.read_json(file)
        for key,value in (('production_evidence',False),('hashed_microbatches',1),
                ('common_trace_matches_archived_controls',False)):
            changed=dict(proof,**{key:value})
            self.runtime.atomic_json(file,changed)
            self.manifest['training_pairing']=changed
            self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
            with self.assertRaisesRegex(ValueError,'training-input'):
                self.export()
            self.assertFalse(self.report.exists())
        file.write_bytes(original)
        self.manifest['training_pairing']=proof
        self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
        self.export()
        self.assertEqual((self.report/'training_pairing.json').read_bytes(),original)
        self.assertEqual(self.runtime.read_json(self.report/'provenance.json')['training_pairing'],proof)

    def test_private_json_nonscalar_csv_and_incomplete_final_are_rejected(self):
        file=self.final/'input_fingerprints.json'
        original=file.read_bytes()
        file.write_text('{"token":"not-a-real-token"}')
        with self.assertRaises(ValueError):self.export()
        file.write_bytes(original)
        (self.final/'summary.csv').write_text('method,value\njoint,"[1,2]"\n')
        with self.assertRaises(ValueError):self.export()
        (self.final/'summary.csv').write_text('method,value\njoint,1\n')
        sample=self.final/'samples.csv.gz'
        lines=gzip.decompress(sample.read_bytes()).decode().splitlines()
        sample.write_bytes(gzip.compress(('\n'.join(lines[:-1])+'\n').encode(),mtime=0))
        with self.assertRaises(ValueError):self.export()
        self.assertFalse(self.report.exists())

    def test_duplicate_keys_columns_and_nonfinite_values_are_rejected(self):
        f=self.final/'input_fingerprints.json'
        for raw in ('{"field":"PRIVATE KEY","field":"public"}','{"value":NaN}'):
            f.write_text(raw)
            with self.assertRaises(ValueError):self.export()
        f.write_text('[]')
        (self.final/'summary.csv').write_text('method,value,value\njoint,1,2\n')
        with self.assertRaises(ValueError):self.export()

    def test_readme_cannot_carry_arbitrary_text_even_with_a_new_content_hash(self):
        self.export()
        path=self.report/'README.md'
        path.write_text('PRIVATE KEY accidental local material\n')
        mp=self.report/'export_manifest.json'
        m=self.runtime.read_json(mp)
        m['files']['README.md']=pub._sha(path)
        self.runtime.atomic_json(mp,m)
        with self.assertRaisesRegex(ValueError,'readme'):
            pub.verify_export_directory(self.report)

    def test_real_orphan_push_retry_preserves_source_and_report(self):
        self.export()
        before={p.name:p.read_bytes() for p in self.report.iterdir()}
        with redirect_stdout(io.StringIO()):receipt=pub.publish(self.report)
        self.assertEqual(receipt['status'],'uploaded')
        self.assertEqual(self.git('show','-s','--format=%P',receipt['commit']),'')
        self.assertEqual(self.git('ls-remote','origin','refs/heads/'+receipt['branch']).split()[0],receipt['commit'])
        names=self.git('--git-dir='+str(self.remote),'ls-tree','--name-only',receipt['commit']).splitlines()
        self.assertEqual(set(names),set(pub.PUBLIC_FILES)|{'export_manifest.json'})
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.report.iterdir()})
        with redirect_stdout(io.StringIO()):retry=pub.publish(self.report)
        self.assertEqual(receipt['commit'],retry['commit'])
        self.preserved()

    def test_failed_push_retry_and_remote_conflict_preserve_prepared_commit(self):
        self.export()
        missing=str(self.root.parent/'missing.git')
        self.git('remote','set-url','origin',missing)
        with mock.patch.object(pub,'REPOSITORY',missing):
            with self.assertRaisesRegex(RuntimeError,'local result export'):pub.publish(self.report)
        prepared=self.runtime.read_json(self.output/'uploads/0123456789ab.json')
        self.git('remote','set-url','origin',str(self.remote))
        with redirect_stdout(io.StringIO()):retry=pub.publish(self.report)
        self.assertEqual(retry['commit'],prepared['commit'])
        ref='refs/heads/'+retry['branch']
        self.git('push','origin',self.head+':refs/heads/main')
        self.git('--git-dir='+str(self.remote),'update-ref',ref,self.head,retry['commit'])
        with self.assertRaisesRegex(RuntimeError,'Git upload failed'):pub.publish(self.report)
        self.assertEqual(self.git('ls-remote','origin',ref).split()[0],self.head)
        self.preserved()

    def test_exact_lease_delete_preserves_main_and_rejects_race(self):
        self.export()
        self.git('push','origin',self.head+':refs/heads/main')
        with redirect_stdout(io.StringIO()):receipt=pub.publish(self.report)
        with self.assertRaises(ValueError):pub.delete_remote('main',receipt['commit'])
        with self.assertRaises(ValueError):pub.delete_remote(receipt['branch'],self.head)
        ref='refs/heads/'+receipt['branch']
        original=subprocess.run
        def race(args,*a,**kw):
            if args[:2]==['git','push'] and any(x.startswith('--force-with-lease=') for x in args):
                original(['git','--git-dir='+str(self.remote),'update-ref',ref,self.head,receipt['commit']],check=True,capture_output=True)
            return original(args,*a,**kw)
        with mock.patch.object(pub.subprocess,'run',side_effect=race):
            with self.assertRaises(RuntimeError):pub.delete_remote(receipt['branch'],receipt['commit'])
        self.assertEqual(self.git('ls-remote','origin',ref).split()[0],self.head)
        self.git('--git-dir='+str(self.remote),'update-ref',ref,receipt['commit'],self.head)
        with redirect_stdout(io.StringIO()):pub.delete_remote(receipt['branch'],receipt['commit'])
        self.assertEqual(self.git('ls-remote','origin',ref),'')
        self.assertEqual(self.git('ls-remote','origin','refs/heads/main').split()[0],self.head)
        self.preserved()


if __name__=='__main__':
    unittest.main()
