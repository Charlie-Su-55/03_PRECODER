"""Owned controller evidence and actual subprocess tests, without GPU access."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import csv
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

import run_twc_v1_k8_ul_calibration as runner


class CalibrationQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'calibration'
        self.output.mkdir()
        self.source = self.root/'source.bin'
        self.source.write_bytes(b'frozen fixture, not real checkpoint')
        self.core = dict(schema='fixture_core', sha256={'fixture': 'a'*64})
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='fixture_calibration', ARMS=('fdma', 'hybrid_g4'),
            VARIANTS=('original', 'reliability_only', 'moment_receiver'),
            GAMMAS=(0, 5, 10, 15, 20, 25), CALIBRATION_SAMPLES=256,
            VALIDATION_SAMPLES=256, recipe=lambda: dict(optimizer_updates=0),
            fingerprint=lambda: 'b'*64)

        def inside(value):
            original, path = Path(value), Path(value).resolve()
            if original.is_symlink() or path == self.output or not path.is_relative_to(self.output):
                raise ValueError('Owned child required')
            return path

        def atomic_json(value, content, exclusive=False):
            path = inside(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x' if exclusive else 'w') as stream:
                json.dump(content, stream, indent=2, sort_keys=True)

        self.runtime = SimpleNamespace(inside=inside, atomic_json=atomic_json,
            sha256=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            read_json=lambda p: json.loads(Path(p).read_text()),
            core_identity=lambda: self.core,
            guard_core=lambda core: self.assertEqual(core, self.core),
            source_identity=lambda: dict(commit='fixture', sha256=self.core['sha256']),
            process_lock=lambda p: nullcontext())
        self.source_hash = self.runtime.sha256(self.source)
        self.evaluator = SimpleNamespace(
            CALIBRATION_FILES=('model.json', 'cal_samples.csv.gz', 'input_fingerprints.json', 'source_records.json'),
            VALIDATION_FILES=('samples.csv.gz', 'summary.csv', 'comparisons.csv', 'input_fingerprints.json', 'source_records.json'),
            verify_calibration=lambda p: self.verify(p, 'calibration'),
            verify_validation=lambda p: self.verify(p, 'validation'))
        self.publisher = SimpleNamespace(export_directory=mock.Mock(side_effect=self.export),
            verify_export_directory=mock.Mock(side_effect=self.verify_export),
            verify_export_binding=mock.Mock(side_effect=self.verify_export_binding),
            publish=mock.Mock(side_effect=AssertionError('Controller must not publish')))
        for patcher in (mock.patch.object(runner, '_config', return_value=self.cfg),
                        mock.patch.object(runner, '_runtime', return_value=self.runtime),
                        mock.patch.object(runner, '_evaluator', return_value=self.evaluator),
                        mock.patch.object(runner, '_publisher', return_value=self.publisher)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def args(self, resume=False):
        return SimpleNamespace(resume=resume, allow_shared_gpu=True,
            min_free_mib=12000, max_gpu_util=80)

    def phase(self, kind, attempt, calibration=None):
        path = self.output/kind/('attempt_'+attempt)
        path.mkdir(parents=True, exist_ok=False)
        names = self.evaluator.CALIBRATION_FILES if kind == 'calibration' else self.evaluator.VALIDATION_FILES
        for name in names:
            (path/name).write_text('{}\n' if name.endswith('.json') else 'fixture\n')
        if kind == 'validation':
            with (path/'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=('arm', 'variant', 'scope', 'gamma_db', 'sum_rate_bps_hz_mean'))
                writer.writeheader()
                for arm in self.cfg.ARMS:
                    for variant in self.cfg.VARIANTS:
                        for gamma in self.cfg.GAMMAS:
                            writer.writerow(dict(arm=arm, variant=variant, scope='point', gamma_db=gamma, sum_rate_bps_hz_mean=25.))
        manifest = dict(status='passed', phase=kind, core=self.core,
            source_sha256=self.source_hash,
            files={n: self.runtime.sha256(path/n) for n in names})
        if calibration is not None:
            manifest.update(calibration_dir=str(calibration),
                model_sha256=self.runtime.sha256(calibration/'model.json'),
                calibration_manifest_sha256=self.runtime.sha256(calibration/'manifest.json'))
        self.runtime.atomic_json(path/'manifest.json', manifest, exclusive=True)
        return path

    def verify(self, value, kind):
        path = self.runtime.inside(value)
        if not path.is_relative_to(self.output/kind):
            raise ValueError('Wrong phase subtree')
        manifest = self.runtime.read_json(path/'manifest.json')
        if manifest.get('status') != 'passed' or manifest.get('phase') != kind:
            raise ValueError('Invalid phase manifest')
        names = self.evaluator.CALIBRATION_FILES if kind == 'calibration' else self.evaluator.VALIDATION_FILES
        if set(manifest['files']) != set(names):
            raise ValueError('Invalid phase inventory')
        for name, digest in manifest['files'].items():
            if self.runtime.sha256(path/name) != digest:
                raise ValueError('Phase bytes changed')
        if manifest['source_sha256'] != self.runtime.sha256(self.source):
            raise ValueError('Frozen source changed')
        if kind == 'validation':
            cal = Path(manifest['calibration_dir'])
            self.verify(cal, 'calibration')
            if (manifest['model_sha256'] != self.runtime.sha256(cal/'model.json')
                    or manifest['calibration_manifest_sha256'] != self.runtime.sha256(cal/'manifest.json')):
                raise ValueError('Calibration lock changed')
        return manifest

    def export(self, summary_path, calibration, validation, output):
        output.mkdir(parents=True, exist_ok=False)
        for name, data in {'README.md': b'fixture', 'model.json': (calibration/'model.json').read_bytes(),
                'summary.csv': (validation/'summary.csv').read_bytes()}.items():
            (output/name).write_bytes(data)
        summary = self.runtime.read_json(summary_path)
        self.runtime.atomic_json(output/'provenance.json', dict(
            preregistration_sha256=summary['preregistration_sha256'],
            calibration_manifest_sha256=self.runtime.sha256(calibration/'manifest.json'),
            source_manifest_sha256=self.runtime.sha256(validation/'manifest.json')), exclusive=True)
        hashes = {p.name: self.runtime.sha256(p) for p in output.iterdir()}
        self.runtime.atomic_json(output/'export_manifest.json', dict(files=hashes), exclusive=True)
        return dict(path=str(output), manifest=str(output/'export_manifest.json'), files=hashes)

    def verify_export(self, output):
        output = Path(output)
        manifest = self.runtime.read_json(output/'export_manifest.json')
        if set(p.name for p in output.iterdir()) != set(manifest['files']) | {'export_manifest.json'}:
            raise ValueError('Report inventory changed')
        for name, digest in manifest['files'].items():
            if self.runtime.sha256(output/name) != digest:
                raise ValueError('Report bytes changed')
        return manifest

    def verify_export_binding(self, summary_path, calibration, validation, output):
        manifest = self.verify_export(output)
        provenance = self.runtime.read_json(Path(output)/'provenance.json')
        summary = self.runtime.read_json(summary_path)
        if provenance != dict(preregistration_sha256=summary['preregistration_sha256'],
                calibration_manifest_sha256=self.runtime.sha256(Path(calibration)/'manifest.json'),
                source_manifest_sha256=self.runtime.sha256(Path(validation)/'manifest.json')):
            raise ValueError('Export provenance/batch binding changed')
        return manifest

    def execute(self, calls, *, fail_validation=False):
        def child(summary, path, record, script, args):
            calls.append(args[0])
            output = Path(args[args.index('--output-dir')+1])
            kind = 'calibration' if args[0] == '--calibrate' else 'validation'
            calibration = Path(args[args.index('--calibration-dir')+1]) if kind == 'validation' else None
            self.phase(kind, output.name[8:], calibration)
            log = path.parent/(kind+'_'+record['attempt_id']+'.log')
            log.write_text('explicit subprocess fixture\n')
            if fail_validation and kind == 'validation':
                raise RuntimeError('validation failed after owned output creation')
            return log
        return child

    def test_fixed_plan_and_no_network_publication(self):
        calls = []
        self.assertEqual(runner.stage_names(), ('calibration', 'locked_validation', 'directory_export'))
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'child', side_effect=self.execute(calls)), redirect_stdout(io.StringIO()):
            runner.run(self.args())
        self.assertEqual(calls, ['--calibrate', '--evaluate'])
        summary, _ = runner.load_summary()
        self.assertEqual(summary['status'], 'completed')
        self.assertIn('Locked model SHA256:', runner.compact_summary(summary))
        self.assertIn('fdma/original sum_rate_bps_hz: 25,25,25,25,25,25', runner.compact_summary(summary))
        self.publisher.publish.assert_not_called()
        self.assertFalse(list(self.output.rglob('*.zip')))

    def test_failed_validation_resume_preserves_attempt_and_never_refits_calibration(self):
        calls = []
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'child', side_effect=self.execute(calls, fail_validation=True)):
            with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                runner.run(self.args())
        failed, _ = runner.load_summary()
        old_validation = Path(failed['stages']['locked_validation']['output_dir'])
        calibration = Path(failed['stages']['calibration']['details']['output_dir'])
        before = {p: p.read_bytes() for p in calibration.iterdir()}
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'child', side_effect=self.execute(calls)), redirect_stdout(io.StringIO()):
            runner.run(self.args(True))
        self.assertEqual(calls, ['--calibrate', '--evaluate', '--evaluate'])
        self.assertEqual(before, {p: p.read_bytes() for p in calibration.iterdir()})
        summary, _ = runner.load_summary()
        self.assertEqual(summary['status'], 'completed')
        self.assertTrue(old_validation.is_dir())
        self.assertNotEqual(summary['stages']['locked_validation']['details']['output_dir'], str(old_validation))

    def test_calibration_tamper_blocks_resume_before_gpu_or_validation(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'child', side_effect=self.execute([], fail_validation=True)):
            with self.assertRaises(RuntimeError):
                runner.run(self.args())
        summary, _ = runner.load_summary()
        (Path(summary['stages']['calibration']['details']['output_dir'])/'model.json').write_text('{"changed":true}')
        with mock.patch.object(runner, 'gpu_admission') as admission, mock.patch.object(runner, 'child') as child:
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                runner.run(self.args(True))
            admission.assert_not_called()
            child.assert_not_called()

    def test_calibration_identity_mismatch_stops_validation_completion(self):
        cal = self.phase('calibration', 'cal')
        summary = dict(stages={}, core=self.core)
        path = self.output/'batches'/'batch_fixture'/'batch_summary.json'
        path.parent.mkdir(parents=True)
        real = self.evaluator.verify_validation
        self.evaluator.verify_validation = lambda p: dict(real(p), model_sha256='0'*64)
        with mock.patch.object(runner, 'child', side_effect=self.execute([])):
            with self.assertRaisesRegex(ValueError, 'artifact binding|pre-locked'):
                runner.validation_action(summary, path, dict(attempt_id='new'), cal)

    def test_completed_validation_cannot_bind_another_valid_calibration(self):
        cal_a = self.phase('calibration', 'a')
        cal_b = self.phase('calibration', 'b')
        validation = self.phase('validation', 'val', cal_a)
        runner.verify_phase_binding(cal_a, validation)
        with self.assertRaisesRegex(ValueError, 'artifact binding'):
            runner.verify_phase_binding(cal_b, validation)

    def test_actual_child_failure_retains_log_and_failure_state(self):
        (self.root/'bad_child.py').write_text("raise RuntimeError('retained-child-error')\n")
        path = self.output/'batches'/'batch_fixture'/'batch_summary.json'
        path.parent.mkdir(parents=True)
        summary = dict(core=self.core, stages={})
        with self.assertRaisesRegex(RuntimeError, 'retained-child-error'):
            runner.stage(summary, path, 'calibration',
                lambda record: runner.child(summary, path, record, 'bad_child.py', []))
        saved = self.runtime.read_json(path)
        record = saved['stages']['calibration']
        self.assertEqual(record['status'], 'failed')
        self.assertIn('retained-child-error', Path(record['log_path']).read_text())
        self.assertIn('Exact log:', runner.compact_summary(saved))

    def test_preregistration_and_recipe_tamper_stop_before_admission(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'child', side_effect=self.execute([])), redirect_stdout(io.StringIO()):
            runner.run(self.args())
        summary, path = runner.load_summary()
        prereg = path.parent/'preregistration.json'
        registered = self.runtime.read_json(prereg)
        registered['validation_used_for_fitting'] = True
        self.runtime.atomic_json(prereg, registered)
        with mock.patch.object(runner, 'gpu_admission') as admission:
            with self.assertRaisesRegex(ValueError, 'preregistration bytes changed'):
                runner.run(self.args(True))
            admission.assert_not_called()
        self.assertNotEqual(summary['preregistration_sha256'], self.runtime.sha256(prereg))

    def test_completed_resume_is_readonly_and_rejects_export_tampering(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'child', side_effect=self.execute([])), redirect_stdout(io.StringIO()):
            runner.run(self.args())
        before = {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        with mock.patch.object(runner, 'gpu_admission') as admission, \
                mock.patch.object(runner, 'child') as child, redirect_stdout(io.StringIO()):
            runner.run(self.args(True))
            admission.assert_not_called()
            child.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()})
        summary, _ = runner.load_summary()
        report = Path(summary['stages']['directory_export']['details']['path'])
        (report/'weights.pt').write_bytes(b'forbidden')
        with self.assertRaisesRegex(ValueError, 'Report inventory'):
            runner.run(self.args(True))

    def test_completed_resume_rejects_self_consistent_foreign_export_provenance(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'child', side_effect=self.execute([])), redirect_stdout(io.StringIO()):
            runner.run(self.args())
        summary, path = runner.load_summary()
        stage = summary['stages']['directory_export']
        report = Path(stage['details']['path'])
        provenance = self.runtime.read_json(report/'provenance.json')
        provenance['preregistration_sha256'] = 'e'*64
        self.runtime.atomic_json(report/'provenance.json', provenance)
        export = self.runtime.read_json(report/'export_manifest.json')
        export['files']['provenance.json'] = self.runtime.sha256(report/'provenance.json')
        self.runtime.atomic_json(report/'export_manifest.json', export)
        stage['files'] = {str(p): self.runtime.sha256(p) for p in report.iterdir()}
        self.runtime.atomic_json(path, summary)
        self.verify_export(report)
        with mock.patch.object(runner, 'gpu_admission') as admission, \
                mock.patch.object(runner, 'child') as child:
            with self.assertRaisesRegex(ValueError, 'provenance/batch binding'):
                runner.run(self.args(True))
            admission.assert_not_called()
            child.assert_not_called()

    def test_existing_output_and_pointer_scope_are_preserved(self):
        marker = self.output/'old_failure.json'
        marker.write_text('{}')
        with self.assertRaises(FileExistsError):
            runner.run(self.args())
        self.assertEqual(marker.read_text(), '{}')
        bad = self.output/'validation'/'foreign.json'
        self.runtime.atomic_json(bad, dict(namespace=self.cfg.NAMESPACE))
        self.runtime.atomic_json(self.output/'last_batch.json', dict(namespace=self.cfg.NAMESPACE, summary=str(bad)))
        with self.assertRaisesRegex(ValueError, 'batch summary path'):
            runner.load_summary()

    def test_invalid_cli_combinations_fail_before_config(self):
        for args in (['--resume'], ['--details']):
            with mock.patch.object(runner, '_config') as cfg, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main(args)
                self.assertEqual(error.exception.code, 2)
                cfg.assert_not_called()

    def test_guard_failure_precedes_output_creation(self):
        self.output.rmdir()
        self.runtime.guard_core = mock.Mock(side_effect=ValueError('old source changed'))
        with self.assertRaisesRegex(ValueError, 'old source changed'):
            runner.run(self.args())
        self.assertFalse(self.output.exists())

    def test_stdlib_dry_run_and_summary_read_no_weights_or_output(self):
        script = Path(runner.__file__).resolve()
        output = script.parent/'runs'/'twc_v1_k8_ul_calibration_v1'
        before = output.exists()
        result = subprocess.run([sys.executable, '-B', '-S', str(script), '--dry-run'],
            check=True, capture_output=True, text=True)
        result = json.loads(result.stdout)
        self.assertFalse(result['weights_read'])
        self.assertFalse(result['writes'])
        self.assertEqual(result['stages'], ['calibration', 'locked_validation', 'directory_export'])
        subprocess.run([sys.executable, '-B', '-S', str(script), '--summary'],
            check=True, capture_output=True, text=True)
        self.assertEqual(output.exists(), before)


if __name__ == '__main__':
    unittest.main()
