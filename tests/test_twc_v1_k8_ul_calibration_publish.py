"""Real local Git publication and bounded public calibration artifacts."""
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

import publish_twc_v1_k8_ul_calibration as pub
from twc_v1_k8_ul_calibration import fit_model
import twc_v1_k8_ul_calibration_results as results


class CalibrationPublicationTests(unittest.TestCase):
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
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'baseline')
        self.git('remote', 'add', 'origin', str(self.remote))
        self.head = self.git('rev-parse', 'HEAD')
        (self.root/'source.py').write_text('preserve dirty source\n')
        (self.root/'staged.txt').write_text('preserve staged work\n')
        self.git('add', 'staged.txt')
        self.index = (self.root/'.git/index').read_bytes()
        self.output = self.root/'runs/fixture_calibration'
        self.output.mkdir(parents=True)
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='fixture_calibration', ARMS=('fdma', 'hybrid_g4'),
            VARIANTS=('original', 'reliability_only', 'moment_receiver'), GAMMAS=(0,5,10,15,20,25),
            MICRO_BATCH=4, CALIBRATION_BATCHES=1, VALIDATION_BATCHES=1,
            CALIBRATION_SAMPLES=4, VALIDATION_SAMPLES=4,
            CHANNEL_ROLES=dict(calibration='ce_calibration', validation='validation'),
            recipe=lambda:dict(physical=dict(K=8,Ep=16,Ed=48,total_RE=256)))

        def inside(value):
            p = Path(value)
            if p.is_symlink() or p.resolve() == self.output or not p.resolve().is_relative_to(self.output):
                raise ValueError('Owned child required')
            return p.resolve()

        def atomic(path, value, exclusive=False):
            p = inside(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('x' if exclusive else 'w') as stream:
                json.dump(value, stream, sort_keys=True)

        @contextmanager
        def lock(path):
            yield

        self.runtime = SimpleNamespace(inside=inside, atomic_json=atomic,
            sha256=pub._sha, read_json=lambda p:json.loads(Path(p).read_text()),
            process_lock=lock, guard_core=lambda value:None)
        self.core = dict(schema='fixture_core',sha256={'source.py':'a'*64})
        self.cal = self.output/'calibration/attempt_aaaaaaaaaaaa'
        self.val = self.output/'evaluation/attempt_bbbbbbbbbbbb'
        self.cal.mkdir(parents=True)
        self.val.mkdir(parents=True)
        calrows = [dict(role='calibration',gamma_db=g,feature=float(i),n_coordinates=1,
            h2_sum=1.,he_real_sum=0.,he_imag_sum=0.,parallel_scaled_sum=1.,
            he_scaled_real_sum=0.,he_scaled_imag_sum=0.,h2_scaled_sum=1.,perp_scaled_sum=1.)
            for g in self.cfg.GAMMAS for i in range(4)]
        atomic(self.cal/'model.json',fit_model(calrows))
        self.csv(self.cal/'cal_samples.csv.gz',calrows)
        atomic(self.cal/'input_fingerprints.json',[dict(role='calibration',content_sha256='b'*64)])
        rows = [dict(arm=arm,variant=variant,role='validation',gamma_db=g,
            sample_index=i,batch_index=0,within_batch=i,
            **{name:.1 for name in results.REQUIRED_METRICS})
            for arm in self.cfg.ARMS for variant in self.cfg.VARIANTS for g in self.cfg.GAMMAS for i in range(4)]
        self.csv(self.val/'samples.csv.gz',rows)
        self.csv(self.val/'summary.csv',[dict(arm='fdma',variant='original',scope='all_six',sum_rate_bps_hz_mean=.1)])
        self.csv(self.val/'comparisons.csv',[dict(comparison='moment_receiver_minus_original',metric='sum_rate_bps_hz',mean=0.)])
        atomic(self.val/'input_fingerprints.json',[dict(role='validation',content_sha256='c'*64)])
        states = {arm:dict(before=dict(codec='1'*64,statistics='2'*64,CE='3'*64),
            after=dict(codec='1'*64,statistics='2'*64,CE='3'*64)) for arm in self.cfg.ARMS}
        records = {arm:dict(kind='rate',arm=arm,model_commit='9'*40,checkpoint_sha256='e'*64,
            step=5000,state_sha256='1'*64,CE_state_sha256='3'*64,statistics_state_sha256='2'*64,
            frozen_files=[dict(path='/home/private/weights.pt',sha256='e'*64)]) for arm in self.cfg.ARMS}
        common = dict(status='passed',core=self.core,recipe_fingerprint='d'*64,
            data_protocol=dict(seed_namespace='fixture',execution=dict(device='cuda:0')))
        self.cm = dict(common,common_CE_signature='3'*64,source_records={'fdma':records['fdma']},
            frozen_signatures={'fdma':states['fdma']})
        atomic(self.cal/'manifest.json',self.cm)
        self.vm = dict(common,source_records=records,frozen_signatures=states,
            calibration_dir=str(self.cal),
            model_sha256=pub._sha(self.cal/'model.json'),
            calibration_manifest_sha256=pub._sha(self.cal/'manifest.json'))
        atomic(self.val/'manifest.json',self.vm)
        self.summary_path = self.output/'batches/batch_fixture/batch_summary.json'
        self.summary_path.parent.mkdir(parents=True)
        prereg = self.summary_path.parent/'preregistration.json'
        prereg.write_text('{"fixture":true}\n')
        self.summary = dict(namespace=self.cfg.NAMESPACE,core=self.core,code=dict(commit=self.head),
            recipe_fingerprint='d'*64,preregistration_sha256=pub._sha(prereg),status='completed',
            stages={name:dict(status='completed',details=dict(output_dir=str(path)),
                files={str(path/'manifest.json'):pub._sha(path/'manifest.json')})
                for name,path in (('calibration',self.cal),('locked_validation',self.val))})
        atomic(self.summary_path,self.summary)

        def preregistration(summary,path):
            if pub._sha(Path(path).parent/'preregistration.json') != summary['preregistration_sha256']:
                raise ValueError('Preregistration changed')

        def evidence(record):
            if not record.get('files') or any(pub._sha(Path(p))!=s for p,s in record['files'].items()):
                raise ValueError('Evidence changed')

        def binding(cal,val):
            manifest=self.runtime.read_json(Path(val)/'manifest.json')
            if inside(manifest.get('calibration_dir','')) != inside(cal):
                raise ValueError('Phase binding changed')
            return manifest

        self.queue = SimpleNamespace(verify_calibration=lambda p:self.runtime.read_json(Path(p)/'manifest.json'),
            verify_validation=lambda p:self.runtime.read_json(Path(p)/'manifest.json'),
            verify_phase_binding=binding,
            verify_preregistration=preregistration,verify_evidence=evidence,
            load_summary=lambda:(self.runtime.read_json(self.summary_path),self.summary_path),
            stage_names=lambda:('calibration','locked_validation','directory_export'))
        for patch in (mock.patch.object(pub,'_config',return_value=self.cfg),
                mock.patch.object(pub,'_runtime',return_value=self.runtime),
                mock.patch.object(pub,'_queue',return_value=self.queue),
                mock.patch.object(pub,'REPOSITORY',str(self.remote))):
            patch.start()
            self.addCleanup(patch.stop)
        self.report = self.output/'reports/attempt_0123456789ab'

    def git(self,*args):
        return subprocess.run(['git',*args],cwd=self.root,capture_output=True,text=True,check=True).stdout.strip()

    @staticmethod
    def csv(path,rows):
        stream=io.StringIO()
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(set(r) for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
        raw=stream.getvalue().encode()
        path.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)

    def export(self):
        exported=pub.export_directory(self.summary_path,self.cal,self.val,self.report)
        self.summary['stages']['directory_export']=dict(status='completed',details=exported,
            files={str(self.report/'export_manifest.json'):pub._sha(self.report/'export_manifest.json')})
        self.runtime.atomic_json(self.summary_path,self.summary)
        return exported

    def preserved(self):
        self.assertEqual(self.git('rev-parse','HEAD'),self.head)
        self.assertEqual((self.root/'.git/index').read_bytes(),self.index)
        self.assertEqual((self.root/'source.py').read_text(),'preserve dirty source\n')
        self.assertEqual((self.root/'staged.txt').read_text(),'preserve staged work\n')

    def test_export_exact_public_inventory_redacts_sources_and_has_no_zip(self):
        exported=self.export()
        self.assertEqual(set(exported['files']),set(pub.PUBLIC_FILES))
        public=(self.report/'provenance.json').read_text()
        for secret in ('/home/','private','frozen_files','weights.pt'):
            self.assertNotIn(secret,public)
        pub.verify_export_directory(self.report)
        self.assertFalse(list(self.output.rglob('*.zip')))
        self.preserved()

    def test_self_consistent_foreign_provenance_and_payload_are_not_this_batch(self):
        self.export()
        provenance_path = self.report/'provenance.json'
        manifest_path = self.report/'export_manifest.json'
        provenance = self.runtime.read_json(provenance_path)
        original = provenance_path.read_bytes()
        manifest = self.runtime.read_json(manifest_path)
        provenance['preregistration_sha256'] = 'f'*64
        self.runtime.atomic_json(provenance_path, provenance)
        manifest['files']['provenance.json'] = pub._sha(provenance_path)
        self.runtime.atomic_json(manifest_path, manifest)
        pub.verify_export_directory(self.report)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            pub.verify_export_binding(self.summary_path, self.cal, self.val, self.report)
        provenance_path.write_bytes(original)
        manifest['files']['provenance.json'] = pub._sha(provenance_path)
        (self.report/'summary.csv').write_text('arm,variant,scope,sum_rate_bps_hz_mean\nfdma,original,all_six,9\n')
        manifest['files']['summary.csv'] = pub._sha(self.report/'summary.csv')
        self.runtime.atomic_json(manifest_path, manifest)
        pub.verify_export_directory(self.report)
        with self.assertRaisesRegex(ValueError, 'payload'):
            pub.verify_export_binding(self.summary_path, self.cal, self.val, self.report)

    def test_report_cannot_be_overwritten_or_add_weights_or_change_bytes(self):
        self.export()
        with self.assertRaises(FileExistsError):
            pub.export_directory(self.summary_path,self.cal,self.val,self.report)
        extra=self.report/'weights.pt'
        extra.write_bytes(b'never publish')
        with self.assertRaises(ValueError): pub.verify_export_directory(self.report)
        extra.unlink()
        (self.report/'summary.csv').write_text('changed')
        with self.assertRaises(ValueError): pub.verify_export_directory(self.report)

    def test_model_lock_preregistration_and_frozen_ce_are_required(self):
        for field,value in (('model_sha256','0'*64),('calibration_manifest_sha256','0'*64)):
            before=self.vm[field]
            self.vm[field]=value
            self.runtime.atomic_json(self.val/'manifest.json',self.vm)
            with self.assertRaises(ValueError): self.export()
            self.vm[field]=before
        self.runtime.atomic_json(self.val/'manifest.json',self.vm)
        self.vm['frozen_signatures']['hybrid_g4']['after']['CE']='4'*64
        self.runtime.atomic_json(self.val/'manifest.json',self.vm)
        with self.assertRaises(ValueError): self.export()
        self.assertFalse(self.report.exists())

    def test_private_json_nonscalar_csv_and_incomplete_validation_stop_before_export(self):
        f=self.val/'input_fingerprints.json'
        original=f.read_bytes()
        f.write_text('{"token":"not-a-real-token"}')
        with self.assertRaises(ValueError): self.export()
        f.write_bytes(original)
        (self.val/'summary.csv').write_text('arm,metric\nfdma,"[1,2]"\n')
        with self.assertRaises(ValueError): self.export()
        (self.val/'summary.csv').write_text('arm,value\nfdma,1\n')
        sample=self.val/'samples.csv.gz'
        lines=gzip.decompress(sample.read_bytes()).decode().splitlines()
        sample.write_bytes(gzip.compress(('\n'.join(lines[:-1])+'\n').encode(),mtime=0))
        with self.assertRaises(ValueError): self.export()
        self.assertFalse(self.report.exists())

    def test_duplicate_json_keys_duplicate_csv_columns_and_nonfinite_json_are_rejected(self):
        path=self.val/'input_fingerprints.json'
        for raw in ('{"field":"PRIVATE KEY hidden","field":"public"}', '{"value":NaN}'):
            path.write_text(raw)
            with self.assertRaises(ValueError): self.export()
        path.write_text('[]')
        (self.val/'summary.csv').write_text('arm,value,value\nfdma,1,2\n')
        with self.assertRaises(ValueError): self.export()
        self.assertFalse(self.report.exists())

    def test_validation_cannot_bind_another_calibration_directory(self):
        self.vm['calibration_dir']=str(self.output/'calibration/attempt_cccccccccccc')
        self.runtime.atomic_json(self.val/'manifest.json',self.vm)
        with self.assertRaises(ValueError): self.export()
        self.assertFalse(self.report.exists())

    def test_real_orphan_push_retry_preserves_source_and_export(self):
        self.export()
        before={p.name:p.read_bytes() for p in self.report.iterdir()}
        with redirect_stdout(io.StringIO()): receipt=pub.publish(self.report)
        self.assertEqual(receipt['status'],'uploaded')
        self.assertEqual(self.git('show','-s','--format=%P',receipt['commit']),'')
        self.assertEqual(self.git('ls-remote','origin','refs/heads/'+receipt['branch']).split()[0],receipt['commit'])
        names=self.git('--git-dir='+str(self.remote),'ls-tree','--name-only',receipt['commit']).splitlines()
        self.assertEqual(set(names),set(pub.PUBLIC_FILES)|{'export_manifest.json'})
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.report.iterdir()})
        with redirect_stdout(io.StringIO()): retry=pub.publish(self.report)
        self.assertEqual(receipt['commit'],retry['commit'])
        self.preserved()

    def test_push_failure_and_conflict_preserve_prepared_export(self):
        self.export()
        missing=str(self.root.parent/'missing.git')
        self.git('remote','set-url','origin',missing)
        with mock.patch.object(pub,'REPOSITORY',missing):
            with self.assertRaisesRegex(RuntimeError,'local result export'):
                pub.publish(self.report)
        prepared=self.runtime.read_json(self.output/'uploads/0123456789ab.json')
        self.git('remote','set-url','origin',str(self.remote))
        with redirect_stdout(io.StringIO()): retry=pub.publish(self.report)
        self.assertEqual(retry['commit'],prepared['commit'])
        reference='refs/heads/'+retry['branch']
        self.git('push','origin',self.head+':refs/heads/main')
        self.git('--git-dir='+str(self.remote),'update-ref',reference,self.head,retry['commit'])
        with self.assertRaisesRegex(RuntimeError,'Git upload failed'): pub.publish(self.report)
        self.assertEqual(self.git('ls-remote','origin',reference).split()[0],self.head)
        self.preserved()

    def test_exact_delete_keeps_main_and_lease_rejects_race(self):
        self.export()
        self.git('push','origin',self.head+':refs/heads/main')
        with redirect_stdout(io.StringIO()): receipt=pub.publish(self.report)
        with self.assertRaises(ValueError): pub.delete_remote('main',receipt['commit'])
        with self.assertRaises(ValueError): pub.delete_remote(receipt['branch'],self.head)
        ref='refs/heads/'+receipt['branch']
        real_run=subprocess.run
        def race(args,*a,**kw):
            if args[:2]==['git','push'] and any(x.startswith('--force-with-lease=') for x in args):
                real_run(['git','--git-dir='+str(self.remote),'update-ref',ref,self.head,receipt['commit']],check=True,capture_output=True)
            return real_run(args,*a,**kw)
        with mock.patch.object(pub.subprocess,'run',side_effect=race):
            with self.assertRaises(RuntimeError): pub.delete_remote(receipt['branch'],receipt['commit'])
        self.assertEqual(self.git('ls-remote','origin',ref).split()[0],self.head)
        self.git('--git-dir='+str(self.remote),'update-ref',ref,receipt['commit'],self.head)
        with redirect_stdout(io.StringIO()): pub.delete_remote(receipt['branch'],receipt['commit'])
        self.assertEqual(self.git('ls-remote','origin',ref),'')
        self.assertEqual(self.git('ls-remote','origin','refs/heads/main').split()[0],self.head)
        self.preserved()


if __name__=='__main__':
    unittest.main()
