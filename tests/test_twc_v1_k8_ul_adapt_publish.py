"""Real local Git publication of fixed-budget multi-SNR adaptation evidence."""
from contextlib import contextmanager, redirect_stdout
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

import publish_twc_v1_k8_ul_adapt as pub
from twc_v1_k8_ul_calibration import fit_model
import twc_v1_k8_ul_adapt_results as results


class AdaptationPublicationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)/'source'
        self.root.mkdir()
        self.remote = self.root.parent/'remote.git'
        self.git('init', '--initial-branch=main')
        self.git('init', '--bare', str(self.remote))
        (self.root/'.gitignore').write_text('runs/\n')
        (self.root/'source.py').write_text('baseline\n')
        self.git('add', '.gitignore', 'source.py')
        self.git('-c','user.name=Fixture','-c','user.email=test@example.invalid','commit','-m','baseline')
        self.git('remote','add','origin',str(self.remote))
        self.head = self.git('rev-parse','HEAD')
        (self.root/'source.py').write_text('preserve dirty source\n')
        (self.root/'staged.txt').write_text('preserve staged work\n')
        self.git('add','staged.txt')
        self.index = (self.root/'.git/index').read_bytes()
        self.output = self.root/'runs/fixture_adapt'
        self.output.mkdir(parents=True)
        self.cal = self.root/'runs/previous_calibration/calibration/attempt_aaaaaaaaaaaa'
        self.cal.mkdir(parents=True)
        gammas = (0,5,10,15,20,25)
        calrows = [dict(role='calibration',gamma_db=g,feature=float(i),n_coordinates=1,
            h2_sum=1.,he_real_sum=0.,he_imag_sum=0.,parallel_scaled_sum=1.,
            he_scaled_real_sum=0.,he_scaled_imag_sum=0.,h2_scaled_sum=1.,perp_scaled_sum=1.)
            for g in gammas for i in range(4)]
        (self.cal/'model.json').write_text(json.dumps(fit_model(calrows)))
        (self.cal/'manifest.json').write_text('{"prior_calibration":true}\n')
        recipe = dict(namespace='fixture_adapt',new_BS_updates=10000)
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='fixture_adapt', ARMS=('fdma','hybrid_g4'),
            TRAIN_METHODS=('original','moment_receiver'), PHASES=('source','final'),
            GAMMAS=gammas, MICRO_BATCH=4, ACCUMULATION=8, UPDATES=10000,
            FINAL_BATCHES=1, CALIBRATION_DIR=self.cal,
            CALIBRATION_MODEL_SHA256=pub._sha(self.cal/'model.json'),
            CALIBRATION_MANIFEST_SHA256=pub._sha(self.cal/'manifest.json'),
            key=lambda a,m:m+'_'+a, fingerprint=lambda a,m:'d'*64,
            study_recipe=lambda:recipe, study_fingerprint=lambda:'d'*64,
            counters=lambda step:dict(gamma_counts={str(g):step for g in gammas}),
            recipe=lambda a,m:dict(physical=dict(K=8,Ep=16,Ed=48,total_RE=256)))

        def inside(value):
            path = Path(value)
            if path.is_symlink() or path.resolve()==self.output or not path.resolve().is_relative_to(self.output):
                raise ValueError('Owned child required')
            return path.resolve()
        def atomic(path,value,exclusive=False):
            path=inside(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x' if exclusive else 'w') as stream:
                json.dump(value,stream,sort_keys=True)
        @contextmanager
        def lock(path):
            yield
        self.runtime = SimpleNamespace(inside=inside,atomic_json=atomic,sha256=pub._sha,
            read_json=lambda p:json.loads(Path(p).read_text()),process_lock=lock,guard_core=lambda value:None)
        self.core = dict(schema='fixture_core',sha256={'source.py':'a'*64})
        self.final = self.output/'evaluation/attempt_bbbbbbbbbbbb'
        self.final.mkdir(parents=True)
        rows = [dict(phase=phase,arm=arm,method=method,role='final',gamma_db=g,
            sample_index=i,batch_index=0,within_batch=i,
            **{name:.1 for name in results.REQUIRED_METRICS})
            for phase in self.cfg.PHASES for arm in self.cfg.ARMS
            for method in self.cfg.TRAIN_METHODS for g in gammas for i in range(4)]
        self.csv(self.final/'samples.csv.gz',rows)
        self.csv(self.final/'summary.csv',[dict(phase='final',arm='fdma',method='original',
            scope='all_six',sum_rate_bps_hz_mean=.1)])
        self.csv(self.final/'comparisons.csv',[dict(comparison='moment_minus_original',
            metric='sum_rate_bps_hz',mean=-.01)])
        atomic(self.final/'input_fingerprints.json',[dict(content_sha256='f'*64)])
        records, signatures = {}, {}
        for phase in self.cfg.PHASES:
            records[phase],signatures[phase]={},{}
            for ai,arm in enumerate(self.cfg.ARMS):
                for mi,method in enumerate(self.cfg.TRAIN_METHODS):
                    key=self.cfg.key(arm,method)
                    state=('a' if phase=='source' else 'b')*64
                    statistics=('4' if ai==0 else '5')*64
                    records[phase][key]=dict(kind='rate',arm=arm,method=method,phase=phase,
                        step=5000 if phase=='source' else self.cfg.UPDATES,
                        checkpoint_sha256=('6' if phase=='source' else '7')*64,
                        state_sha256=state,CE_state_sha256='3'*64,statistics_state_sha256=statistics,
                        path='/home/private/weights.pt',
                        frozen_files=[dict(path='/home/private/weights.pt',sha256='8'*64)])
                    states=dict(codec=state,statistics=statistics,CE='3'*64)
                    signatures[phase][key]=dict(before=dict(states),after=dict(states))
        self.manifest=dict(status='passed',phase='final',core=self.core,
            data_protocol=dict(seed_namespace='fixture',execution=dict(device='cuda:0')),
            source_records=records,frozen_signatures=signatures,
            terminal_records={k:dict(v) for k,v in records['final'].items()},
            calibration_model_sha256=self.cfg.CALIBRATION_MODEL_SHA256,
            calibration_manifest_sha256=self.cfg.CALIBRATION_MANIFEST_SHA256,
            all_terminals_verified_before_data=True)
        pairing=dict(format='fixture_training_pairing_v1',status='passed',production_evidence=True,
            full_micro_trace_chain_verified=True,snapshot_pairing_verified=True,
            hashed_microbatches=self.cfg.UPDATES*self.cfg.ACCUMULATION,
            hashed_MU_frames=self.cfg.UPDATES*self.cfg.ACCUMULATION*self.cfg.MICRO_BATCH,
            gamma_counts=self.cfg.counters(self.cfg.UPDATES)['gamma_counts'],
            trajectories={key:dict(progress_sha256='c'*64,training_trace='d'*64)
                for key in self.manifest['terminal_records']})
        atomic(self.final/'training_pairing.json',pairing)
        self.manifest['training_pairing']=pairing
        atomic(self.final/'manifest.json',self.manifest)
        self.summary_path=self.output/'batches/batch_fixture/batch_summary.json'
        self.summary_path.parent.mkdir(parents=True)
        prereg=self.summary_path.parent/'preregistration.json'
        prereg.write_text('{"fixture":true}\n')
        self.summary=dict(namespace=self.cfg.NAMESPACE,core=self.core,code=dict(commit=self.head),
            recipe=recipe,recipe_fingerprint='d'*64,preregistration_sha256=pub._sha(prereg),
            status='completed',stages={'final_validation':dict(status='completed',
                details=dict(output_dir=str(self.final)),
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
            patch.start()
            self.addCleanup(patch.stop)
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
        self.assertEqual(self.runtime.read_json(self.report/'provenance.json')['total_new_BS_updates'],40000)
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
        (self.report/'summary.csv').write_text('phase,arm,method,value\nfinal,fdma,original,9\n')
        m['files']['summary.csv']=pub._sha(self.report/'summary.csv')
        self.runtime.atomic_json(mp,m)
        with self.assertRaisesRegex(ValueError,'payload'):
            pub.verify_export_binding(self.summary_path,self.final,self.report)

    def test_fixed_calibration_all_terminals_and_frozen_states_required(self):
        for field,value in (('calibration_model_sha256','0'*64),
                ('calibration_manifest_sha256','0'*64),('all_terminals_verified_before_data',False)):
            before=self.manifest[field]
            self.manifest[field]=value
            self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
            with self.assertRaises(ValueError):self.export()
            self.manifest[field]=before
        self.manifest['terminal_records']['original_fdma']['step']=9999
        self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
        with self.assertRaises(ValueError):self.export()
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
                ('full_micro_trace_chain_verified',False)):
            changed=dict(proof,**{key:value})
            self.runtime.atomic_json(file,changed)
            self.manifest['training_pairing']=changed
            self.runtime.atomic_json(self.final/'manifest.json',self.manifest)
            with self.assertRaisesRegex(ValueError,'pairing'):
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
        (self.final/'summary.csv').write_text('phase,value\nfinal,"[1,2]"\n')
        with self.assertRaises(ValueError):self.export()
        (self.final/'summary.csv').write_text('phase,value\nfinal,1\n')
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
        (self.final/'summary.csv').write_text('phase,value,value\nfinal,1,2\n')
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
