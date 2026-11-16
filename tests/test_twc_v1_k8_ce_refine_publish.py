"""Actual Git plumbing and bounded public CE results; no network or GPU."""
from contextlib import redirect_stdout
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_ce_refine_config as cfg
import ce_refine_runtime as rt
import evaluate_twc_v1_k8_ce_refine as evaluation
import publish_twc_v1_k8_ce_refine as pub
import twc_v1_k8_ce_refine_results as results


def csv_bytes(rows, compressed=False):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    raw = stream.getvalue().encode()
    return gzip.compress(raw, mtime=0) if compressed else raw


class CEPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'source'; self.root.mkdir()
        self.remote = self.root.parent/'remote.git'
        self.git('init', '--initial-branch=main')
        self.git('init', '--bare', str(self.remote))
        (self.root/'.gitignore').write_text('runs/\n')
        (self.root/'science.py').write_text('baseline\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'baseline')
        self.git('remote', 'add', 'origin', str(self.remote))
        self.head = self.git('rev-parse', 'HEAD')
        (self.root/'science.py').write_text('keep dirty science\n')
        (self.root/'staged.txt').write_text('keep staged data\n'); self.git('add', 'staged.txt')
        self.index = (self.root/'.git'/'index').read_bytes()
        for patch in (mock.patch.object(pub, 'ROOT', self.root),
                mock.patch.object(pub, 'REPOSITORY', str(self.remote)),
                mock.patch.object(cfg, 'ROOT', self.root),
                mock.patch.object(cfg, 'OUTPUT_ROOT', self.root/'runs'/pub.NAMESPACE),
                mock.patch.object(rt, 'ROOT', self.root),
                mock.patch.object(cfg, 'FINAL_BATCHES', 2)):
            patch.start(); self.addCleanup(patch.stop)
        self.run = cfg.OUTPUT_ROOT; self.run.mkdir(parents=True)
        self.evaluation = self.run/'evaluation'/'attempt_112233445566'; self.evaluation.mkdir(parents=True)
        self.report = self.run/'reports'/'attempt_0123456789ab'
        self.core = {'schema': 'fixture_core', 'sha256': {'fixture.py': 'b'*64}}
        registration = dict(namespace=pub.NAMESPACE, core=self.core, recipe_fingerprint=cfg.fingerprint())
        self.prereg = self.run/'preregistration.json'
        self.prereg.write_text(json.dumps(registration))
        self.summary = dict(registration, preregistration_sha256=pub._sha(self.prereg))
        self.summary_path = self.run/'summary.json'; self.summary_path.write_text(json.dumps(self.summary))
        self.make_payloads()

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True,
            capture_output=True, text=True).stdout.strip()

    def preserved(self):
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.head)
        self.assertEqual((self.root/'.git'/'index').read_bytes(), self.index)
        self.assertEqual((self.root/'science.py').read_text(), 'keep dirty science\n')
        self.assertEqual((self.root/'staged.txt').read_text(), 'keep staged data\n')

    def make_payloads(self, *, fixture=False):
        rows = []
        for vi, variant in enumerate(results.DEFAULT_VARIANTS):
            for role, gammas in (('final', results.GAMMAS), ('midpoint', results.MIDPOINTS)):
                for gamma in gammas:
                    for sample in range(8):
                        row = dict(variant=variant, role=role, gamma_db=gamma,
                            sample_index=sample, batch_index=sample//4, within_batch=sample % 4,
                            ul_nmse_linear=.05+.001*vi+.0001*sample,
                            direction_error=.02+.001*vi, gram_error=.01+.0001*sample,
                            energy_ratio=1.+.001*vi, ce_mse=.003+.0001*sample)
                        rows.append(row)
        summaries, comparisons = [], []
        for role in ('final', 'midpoint'):
            part = [r for r in rows if r['role'] == role]
            summaries += results.summarize_rows(part, samples=8, role=role)
            comparisons += results.paired_comparisons(part, samples=8, role=role)
        states = {v: str(i+1)*64 for i,v in enumerate(results.DEFAULT_VARIANTS)}
        states['source_ce5000'] = cfg.SOURCE_STATE_SHA256
        source_identity = dict(checkpoint_sha256=cfg.SOURCE_CHECKPOINT_SHA256,
            source_state_sha256=cfg.SOURCE_STATE_SHA256,source_commit=cfg.SOURCE_COMMIT,
            source_core=cfg.SOURCE_CORE,source_updates=cfg.SOURCE_UPDATES,
            frozen_files=[dict(path='/home/metadata_fixture/old_ce.pt',sha256='a'*64)])
        # Exercise the actual evaluator's redaction of real trainer-record shape.
        # This is metadata-only fixture evidence, not a production model evaluation.
        source_models = {'source_ce5000': evaluation._public_record(source_identity)}
        for i,trajectory in enumerate(cfg.TRAJECTORIES):
            variant=trajectory+'65000'
            training_record=dict(step=cfg.UPDATES,counters=cfg.counters(cfg.UPDATES),
                fingerprint=cfg.fingerprint(trajectory),core=self.core,production_evidence=not fixture,
                CE_total_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,
                source_checkpoint_sha256=cfg.SOURCE_CHECKPOINT_SHA256,
                source_state_sha256=cfg.SOURCE_STATE_SHA256,source_commit=cfg.SOURCE_COMMIT,
                model_state_sha256=states[variant],checkpoint_sha256=str(i+4)*64,
                source_identity=source_identity,fixture=fixture)
            for field in ('CE_total_updates','source_checkpoint_sha256','source_state_sha256','source_commit'):
                training_record.pop(field, None)
            source_models[variant] = evaluation._public_record(training_record)
        provenance = dict(namespace=pub.NAMESPACE, production_evidence=not fixture,
            UE_updates=0, BS_updates=0, FDMA_updates=0, EMA_updates=0,
            CE_training_updates=120000, all_models_verified_before_data=True,
            paired_inputs_verified=True, test_used=False, freeze_before=states, freeze_after=states,
            fixture=fixture, fixed_terminal_new_updates=cfg.UPDATES,
            CE_total_updates_per_new_model=cfg.SOURCE_UPDATES+cfg.UPDATES,
            calibration_locked_before_final=True, fingerprint=cfg.fingerprint(),
            core=self.core, source_models=source_models)
        calibration = dict(namespace=pub.NAMESPACE, production_evidence=not fixture,
            nodes=list(cfg.GAMMAS), model_states=states,
            per_RE_error={v:[[[.01]*256 for _ in range(8)] for _ in range(6)] for v in states},
            samples_per_node=256, definition='mean CE error power', inputs=[], locked_before_final=True)
        calibration['record_id'] = cfg.digest(calibration)
        calibration_raw=json.dumps(calibration).encode()
        provenance.update(calibration_sha256=hashlib.sha256(calibration_raw).hexdigest(),
                          calibration_record_id=calibration['record_id'])
        payloads = {'README.md':pub.README_TEXT.encode(), 'samples.csv.gz':csv_bytes(rows, True),
            'summary.csv':csv_bytes(summaries), 'comparisons.csv':csv_bytes(comparisons),
            'input_fingerprints.json':b'{"paired":true}',
            'provenance.json':json.dumps(provenance).encode(),
            'calibration.json':calibration_raw,
            'curve_summary.csv':csv_bytes([dict(trajectory='nmse', step=0, gamma_db=0, ul_nmse_linear=.1)])}
        for name, raw in payloads.items(): (self.evaluation/name).write_bytes(raw)
        manifest = dict(namespace=pub.NAMESPACE, core=self.core,
            production_evidence=not fixture, recipe_fingerprint=cfg.fingerprint(),
            files={n:hashlib.sha256(raw).hexdigest() for n,raw in payloads.items()})
        (self.evaluation/'manifest.json').write_text(json.dumps(manifest))

    def export(self):
        return pub.export_directory(self.summary_path, self.evaluation, self.report)

    def test_real_export_publish_retry_and_exact_delete_preserve_source(self):
        result = self.export()
        self.assertFalse(result['contains_model_weights'])
        self.assertEqual(set(result['files']), set(pub.PUBLIC_FILES))
        with mock.patch.object(pub, '_completed', return_value=self.report), redirect_stdout(io.StringIO()):
            receipt = pub.publish()
            repeated = pub.publish()
        self.assertEqual(receipt['commit'], repeated['commit'])
        self.assertEqual(self.git('ls-remote', '--heads', 'origin', receipt['branch']).split()[0], receipt['commit'])
        self.assertEqual(self.git('show', '-s', '--format=%P', receipt['commit']), '')
        self.assertEqual(set(self.git('ls-tree', '--name-only', receipt['tree']).splitlines()),
                         set(pub.PUBLIC_FILES)|{'export_manifest.json'})
        with self.assertRaises(ValueError): pub.delete_remote(receipt['branch'], 'a'*40)
        pub.delete_remote(receipt['branch'], receipt['commit'])
        self.assertEqual(self.git('ls-remote', '--heads', 'origin', receipt['branch']), '')
        self.preserved()

    def test_evaluator_trainer_metadata_contract_and_source_tamper_rejection(self):
        self.export()
        provenance=rt.read_json(self.report/'provenance.json')
        raw=json.dumps(provenance)
        self.assertNotIn('/home/metadata_fixture',raw)
        self.assertNotIn('frozen_files',raw)
        for trajectory in cfg.TRAJECTORIES:
            record=provenance['source_models'][trajectory+'65000']
            self.assertEqual(record['source_checkpoint_sha256'],cfg.SOURCE_CHECKPOINT_SHA256)
            self.assertEqual(record['source_state_sha256'],cfg.SOURCE_STATE_SHA256)
            self.assertEqual(record['CE_total_updates'],65000)
        provenance['source_models']['nmse65000']['source_checkpoint_sha256']='f'*64
        file=self.report/'provenance.json';file.write_text(json.dumps(provenance))
        manifest_path=self.report/'export_manifest.json'
        manifest=rt.read_json(manifest_path);manifest['files']['provenance.json']=pub._sha(file)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'warm start'):
            pub.verify_export_directory(self.report)
        self.preserved()

    def test_rejected_actual_push_keeps_prepared_commit_and_local_export(self):
        self.export()
        hook = self.remote/'hooks'/'pre-receive'; hook.write_text('#!/bin/sh\nexit 1\n'); hook.chmod(0o755)
        with mock.patch.object(pub, '_completed', return_value=self.report), self.assertRaises(RuntimeError):
            pub.publish()
        receipt = rt.read_json(self.run/'uploads'/'0123456789ab.json')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(self.git('cat-file', '-t', receipt['commit']), 'commit')
        self.assertEqual(pub.verify_export_directory(self.report)['export_id'], '0123456789ab')
        self.preserved()

    def test_tampered_samples_and_even_rehashed_aggregate_are_refused(self):
        self.export()
        path = self.report/'summary.csv'
        text = path.read_text(); path.write_text(text.replace('0.05035', '0.95035', 1))
        if path.read_text() == text: path.write_text(text.replace('0.02', '0.92', 1))
        with self.assertRaises(ValueError): pub.verify_export_directory(self.report)
        manifest_path = self.report/'export_manifest.json'
        manifest = rt.read_json(manifest_path); manifest['files']['summary.csv'] = pub._sha(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'statistic'):
            pub.verify_export_directory(self.report)
        self.preserved()

    def test_fixture_and_credential_payloads_are_refused(self):
        self.make_payloads(fixture=True)
        with self.assertRaisesRegex(ValueError, 'production'):
            self.export()
        for raw in (b'{"token":"must-not-publish"}', b'{"path":"/home/rong/private"}',
                    b'{"x":NaN}'):
            with self.assertRaises(ValueError): pub._validate_payload('provenance.json', raw)
        with self.assertRaises(ValueError): pub._validate_payload('summary.csv', b'variant,x\nnmse,NaN\n')
        with self.assertRaises(ValueError): pub._validate_payload('samples.csv.gz', gzip.compress(b'variant,x\nnmse,Inf\n'))
        self.preserved()

    def test_changed_registration_and_symlink_are_refused(self):
        self.prereg.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'registration'): self.export()
        self.report.parent.mkdir(parents=True)
        external = self.root.parent/'external'; external.mkdir()
        self.report.symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError): pub.verify_export_directory(self.report)
        self.preserved()


if __name__ == '__main__': unittest.main()
