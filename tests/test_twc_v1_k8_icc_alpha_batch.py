"""Owned no-training queue and selection-before-final checks; stdlib only."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import run_twc_v1_k8_icc_alpha as runner


class AlphaQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'alpha'
        self.output.mkdir()
        self.core = {'schema': 'tiny_alpha_core', 'sha256': {'test': 'a' * 64}}
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='tiny_alpha', ARMS=('fdma', 'hybrid_g4'), GAMMAS=(0,5,10,15,20,25),
            POLICIES=('source_fixed', 'source_calibrated', 'rate_fixed'),
            MICRO_BATCH=4, CALIBRATION_BATCHES=64, FINAL_BATCHES=256,
            fingerprint=lambda: 'd'*64,
            recipe=lambda: {'optimizer_updates': 0, 'data_namespace': 'tiny_alpha'},
            method_name=lambda policy, arm: policy + '_' + arm)
        def inside(path):
            path = Path(path).resolve()
            if path == self.output or not path.is_relative_to(self.output):
                raise ValueError('Owned alpha child required')
            return path
        def atomic_json(path, value, exclusive=False):
            path = inside(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x' if exclusive else 'w') as stream:
                json.dump(value, stream, sort_keys=True, indent=2)
        self.runtime = SimpleNamespace(inside=inside, atomic_json=atomic_json,
            read_json=lambda path: json.loads(Path(path).read_text()),
            sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            core_identity=lambda: self.core,
            guard_core=lambda core: self.assertEqual(core, self.core),
            load_selection=mock.Mock(side_effect=lambda path: json.loads(Path(path).read_text())),
            lock_selection=mock.Mock())
        for patcher in (mock.patch.object(runner, '_runtime', return_value=self.runtime),
                        mock.patch.object(runner, '_config', return_value=self.cfg)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def selection(self, calibration_dir=None):
        value = dict(record_id='b' * 64, selected={'fdma': .03, 'hybrid_g4': .05},
            calibration_dir=str(calibration_dir or self.output/'calibration'/'attempt_original'),
            scores=[dict(arm='fdma', alpha=.03, mean=1.),dict(arm='hybrid_g4',alpha=.05,mean=2.)])
        self.runtime.atomic_json(self.output / 'selection.json', value, exclusive=True)
        return value

    def phase(self, phase):
        folder = self.output / phase / 'attempt_tiny'
        folder.mkdir(parents=True)
        contents = {'samples.csv.gz': gzip.compress(b'arm,gamma_db,value\nfdma,0,1\n', mtime=0),
                    'input_fingerprints.json': b'[]\n'}
        if phase == 'calibration':
            contents.update({'scores.csv': b'arm,alpha,mean\nfdma,0.03,1\n',
                             'selection_candidate.json': b'{"selected":{"fdma":0.03,"hybrid_g4":0.05}}\n'})
        else:
            contents.update({'summary.csv': b'arm,gamma_db,sum_rate_bps_hz_mean\nsource_fixed_fdma,0,1\n',
                             'comparisons.csv': b'comparison,metric,scope,mean\nsource_fixed_hybrid_minus_fdma,sum_rate_bps_hz,all_six,0\n'})
        for name, data in contents.items():
            (folder / name).write_bytes(data)
        manifest = dict(status='passed', phase=phase, core=self.core,
                        recipe_fingerprint=self.cfg.fingerprint(), gamma_db=list(self.cfg.GAMMAS),
                        batch_size=4, n_batches=64 if phase=='calibration' else 256,
                        samples_per_condition=256 if phase=='calibration' else 1024,
                        no_optimizer=True, test_used=False,
                        files={name: self.runtime.sha256(folder / name) for name in contents})
        if phase == 'final':
            manifest['selection_record_id'] = 'b' * 64
        self.runtime.atomic_json(folder / 'manifest.json', manifest, exclusive=True)
        return folder

    def test_completed_stage_reuses_only_hash_bound_evidence(self):
        path = self.output / 'batch' / 'summary.json'
        path.parent.mkdir()
        summary = dict(stages={}, core=self.core)
        calls = []
        def action(record):
            calls.append(record['attempt_id'])
            file = path.parent / 'scores.csv'
            file.write_text('verified\n')
            return {'status': 'passed'}, [file]
        first = runner.stage(summary, path, 'calibration', action)
        loaded = self.runtime.read_json(path)
        self.assertEqual(runner.stage(loaded, path, 'calibration', action), first)
        self.assertEqual(len(calls), 1)
        (path.parent / 'scores.csv').write_text('tampered\n')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            runner.stage(loaded, path, 'calibration', action)
        with self.assertRaisesRegex(ValueError, 'no evidence'):
            runner.verify_evidence({'status': 'completed'})

    def test_actual_child_failure_preserves_exact_log(self):
        path = self.output / 'batch' / 'summary.json'
        path.parent.mkdir()
        (self.root / 'failed_child.py').write_text("raise RuntimeError('unique-alpha-child-failure')\n")
        summary = dict(stages={}, core=self.core)
        def action(record):
            runner.child(summary, path, record, 'failed_child.py', [])
            self.fail('failed child treated as success')
        with self.assertRaisesRegex(RuntimeError, 'unique-alpha-child-failure'):
            runner.stage(summary, path, 'calibration', action)
        saved = self.runtime.read_json(path)
        record = saved['stages']['calibration']
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('unique-alpha-child-failure', record['child_error'])
        self.assertIn('unique-alpha-child-failure', record['traceback'])
        self.assertTrue(Path(record['log_path']).is_file())

    def test_queue_has_no_training_or_optimizer_stage(self):
        self.assertEqual(runner.stage_names(),
            ('calibration', 'selection_lock', 'final_validation', 'result_package'))
        self.assertFalse(any('train' in name for name in runner.stage_names()))

    def test_final_cannot_start_without_a_valid_locked_selection(self):
        summary = dict(stages={}, core=self.core)
        with mock.patch.object(runner, 'child') as child:
            with self.assertRaisesRegex(ValueError, 'Locked alpha selection required'):
                runner.evaluation_action(summary, self.output/'summary.json', 'final', {'attempt_id': 'x'})
            child.assert_not_called()
        self.selection()
        self.runtime.load_selection.side_effect = ValueError('invalid selection identity')
        with self.assertRaisesRegex(ValueError, 'invalid selection identity'):
            runner.require_locked_selection()

    def test_partial_final_forbids_missing_lock_reselection(self):
        attempt = self.output / 'evaluation' / 'final' / 'attempt_partial'
        attempt.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'final attempt already exists'):
            runner.selection_action(self.output/'calibration'/'attempt_original')
        self.runtime.lock_selection.assert_not_called()
        self.assertTrue(attempt.exists())

    def test_existing_selection_is_validated_and_never_reselected(self):
        chosen = self.selection()
        (self.output/'evaluation'/'final'/'attempt_partial').mkdir(parents=True)
        value, files = runner.selection_action(self.output/'calibration'/'attempt_original')
        self.assertEqual(value, chosen)
        self.assertEqual(files, [self.output/'selection.json'])
        self.runtime.load_selection.assert_called()
        self.runtime.lock_selection.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'different calibration evidence'):
            runner.selection_action(self.output/'calibration'/'attempt_other')

    def test_manifest_checks_artifacts_and_final_selection_record(self):
        self.selection()
        final = self.phase('final')
        runner.verify_manifest(final, 'final')
        path = final/'manifest.json'
        manifest = self.runtime.read_json(path)
        manifest['selection_record_id'] = 'c' * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'different locked selection'):
            runner.verify_manifest(final, 'final')
        manifest['selection_record_id'] = 'b' * 64
        manifest['n_batches'] = 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'budget identity changed'):
            runner.verify_manifest(final, 'final')
        manifest['n_batches'] = 256
        path.write_text(json.dumps(manifest))
        (final/'summary.csv').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            runner.verify_manifest(final, 'final')

    def test_package_contains_scalar_raw_metrics_and_hash_whitelist_not_weights(self):
        calibration = self.phase('calibration')
        self.selection(calibration)
        final = self.phase('final')
        batch = self.output/'batches'/'batch_tiny'
        batch.mkdir(parents=True)
        summary_path = batch/'batch_summary.json'
        self.runtime.atomic_json(batch/'preregistration.json', {'optimizer_updates': 0}, exclusive=True)
        (final/'checkpoint.pt').write_bytes(b'never include this extra file')
        before = {p: p.read_bytes() for folder in (calibration, final)
                  for p in folder.iterdir() if p.is_file()}
        result, evidence = runner.package_action(summary_path, calibration, final, {'attempt_id': 'package_test'})
        archive = Path(result['path'])
        self.assertEqual(result['sha256'], self.runtime.sha256(archive))
        with zipfile.ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            self.assertIn('calib_samples.csv.gz', names)
            self.assertIn('final_samples.csv.gz', names)
            self.assertFalse(any(name.endswith(('.pt', '.pth', '.ckpt')) for name in names))
            whitelist = json.loads(zipped.read('package_whitelist.json'))
            self.assertEqual(set(whitelist['files']) | {'package_whitelist.json'}, names)
            for name, digest in whitelist['files'].items():
                self.assertEqual(hashlib.sha256(zipped.read(name)).hexdigest(), digest)
            self.assertEqual(gzip.decompress(zipped.read('final_samples.csv.gz')),
                             b'arm,gamma_db,value\nfdma,0,1\n')
        self.assertEqual(len(evidence), 2)
        self.assertTrue(all(path.is_file() for path in evidence))
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))

    def test_pointer_and_owned_evidence_paths_are_strict(self):
        self.runtime.atomic_json(self.output/'last_batch.json',
            {'namespace': 'old_experiment', 'summary': str(self.root/'old.json')}, exclusive=True)
        with self.assertRaisesRegex(ValueError, 'Own alpha-experiment pointer'):
            runner.load_summary()
        with self.assertRaisesRegex(ValueError, 'Owned alpha child'):
            runner.files_identity([self.root/'outside.txt'])


if __name__ == '__main__':
    unittest.main()
