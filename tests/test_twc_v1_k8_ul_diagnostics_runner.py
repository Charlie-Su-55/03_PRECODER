"""Real file/subprocess queue tests; no CUDA, scientific model or optimizer is run."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import csv
import gzip
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
import zipfile

import run_twc_v1_k8_ul_diagnostics as runner


class DiagnosticQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'diagnostic'
        self.output.mkdir()
        self.source = self.root/'frozen_source.pt'
        self.source.write_bytes(b'queue-only checkpoint identity fixture; not a scientific model')
        self.core = {'schema': 'tiny_diagnostic_core', 'sha256': {'test': 'a'*64}}
        self.protocol = {'seed_namespace': 'tiny_diagnostic', 'base_seed': 20261121,
                         'execution': {'device': 'cuda:0', 'sionna_precision': 'single'}}
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='tiny_diagnostic', DATA_NAMESPACE='tiny_diagnostic',
            ARMS=('fdma', 'hybrid_g4'), VARIANTS=('estimated_ul', 'true_ul_diagnostic'),
            GAMMAS=(0, 5, 10, 15, 20, 25), MICRO_BATCH=4, BATCHES=64, SAMPLES=256,
            recipe=lambda: {'namespace': 'tiny_diagnostic', 'optimizer_updates': 0},
            fingerprint=lambda: 'd'*64)

        def inside(value):
            original = Path(value)
            path = original.resolve()
            if original.is_symlink() or path == self.output or not path.is_relative_to(self.output):
                raise ValueError('Own diagnostic child required')
            return path

        def atomic_json(value, record, exclusive=False):
            path = inside(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x' if exclusive else 'w') as stream:
                json.dump(record, stream, sort_keys=True, indent=2)

        def sha256(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        def verify_source_records(records):
            for record in records.values():
                for entry in record['frozen_files']:
                    if sha256(entry['path']) != entry['sha256']:
                        raise ValueError('Pinned source changed')

        def make_data(device):
            self.assertEqual(device, 'cuda:0')
            return SimpleNamespace(protocol=self.protocol)

        self.runtime = SimpleNamespace(inside=inside, atomic_json=atomic_json,
            read_json=lambda path: json.loads(Path(path).read_text()), sha256=sha256,
            core_identity=lambda: self.core,
            guard_core=lambda value: self.assertEqual(value, self.core),
            source_identity=lambda: {'commit': 'tiny-test-only', 'sha256': self.core['sha256']},
            process_lock=lambda path: nullcontext(), make_data=make_data,
            verify_source_records=verify_source_records)
        for patcher in (mock.patch.object(runner, '_runtime', return_value=self.runtime),
                        mock.patch.object(runner, '_config', return_value=self.cfg)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def arguments(self, *, resume=False):
        return SimpleNamespace(resume=resume, allow_shared_gpu=True,
                               min_free_mib=12000, max_gpu_util=80)

    def phase(self, attempt='tiny'):
        directory = self.output/'evaluation'/('attempt_'+attempt)
        directory.mkdir(parents=True)
        summary = io.StringIO()
        writer = csv.DictWriter(summary, fieldnames=('arm', 'variant', 'gamma_db', 'soft_nmse_linear_mean'))
        writer.writeheader()
        for arm in self.cfg.ARMS:
            for variant in self.cfg.VARIANTS:
                for gamma in self.cfg.GAMMAS:
                    writer.writerow(dict(arm=arm, variant=variant, gamma_db=gamma,
                                         soft_nmse_linear_mean=.1 if variant=='estimated_ul' else .05))
        comparisons = io.StringIO()
        writer = csv.DictWriter(comparisons, fieldnames=(
            'comparison', 'scope', 'metric', 'mean', 'batch_ci95_low', 'batch_ci95_high'))
        writer.writeheader()
        for arm in self.cfg.ARMS:
            writer.writerow(dict(comparison=arm+'_true_ul_diagnostic_minus_estimated_ul',
                scope='all_six', metric='soft_nmse_linear', mean=-.05,
                batch_ci95_low=-.06, batch_ci95_high=-.04))
        records = {arm: {'frozen_files': [dict(path=str(self.source),
            sha256=self.runtime.sha256(self.source))]} for arm in self.cfg.ARMS}
        contents = {
            'samples.csv.gz': gzip.compress(b'arm,variant,gamma_db,soft_nmse_linear\nfdma,estimated_ul,0,0.1\n', mtime=0),
            'summary.csv': summary.getvalue().encode(),
            'comparisons.csv': comparisons.getvalue().encode(),
            'calibration_bins.csv.gz': gzip.compress(b'arm,variant,bin,count\nfdma,estimated_ul,0,1\n', mtime=0),
            'input_fingerprints.json': b'[{"H_dl":"1111","H_ul":"2222","standard_noise":"3333"}]\n',
            'source_records.json': json.dumps(records, sort_keys=True).encode(),
        }
        self.assertEqual(set(contents), set(runner.EVALUATION_FILES))
        for name, value in contents.items():
            (directory/name).write_bytes(value)
        manifest = dict(status='passed', phase='diagnostic', format=self.cfg.NAMESPACE+'_manifest_v1',
            core=self.core, recipe_fingerprint=self.cfg.fingerprint(),
            arms=list(self.cfg.ARMS), variants=list(self.cfg.VARIANTS), gamma_db=list(self.cfg.GAMMAS),
            batch_size=4, n_batches=64, samples_per_condition=256,
            optimizer_updates=0, UE_updates=0, CE_updates=0, EMA_updates=0, test_used=False,
            paired_inputs_verified=True, frozen_models_verified=True,
            privileged_truth_for_diagnostics_only=True, production_evidence=True,
            source_checkpoints_verified=True, no_downlink_forward=True, data_protocol=self.protocol,
            source_records=records,
            files={name: self.runtime.sha256(directory/name) for name in contents})
        self.runtime.atomic_json(directory/'manifest.json', manifest, exclusive=True)
        return directory

    def evaluation(self, calls):
        def action(summary, path, record):
            calls.append(record['attempt_id'])
            directory = self.phase(record['attempt_id'])
            log = path.parent/('fixture_'+record['attempt_id']+'.log')
            log.write_text('queue-only evaluator output\n')
            manifest = runner.verify_manifest(directory)
            return dict(output_dir=str(directory), manifest=manifest), [
                directory/'manifest.json', *[directory/name for name in runner.EVALUATION_FILES], log]
        return action

    def test_stage_plan_has_no_training_or_model_selection_stage(self):
        self.assertEqual(runner.stage_names(), ('mechanism_diagnostics', 'result_package'))

    def test_manifest_rejects_updates_truth_leak_protocol_and_artifact_changes(self):
        directory = self.phase()
        original = self.runtime.read_json(directory/'manifest.json')
        runner.verify_manifest(directory)
        for key, value in [('optimizer_updates', 1), ('UE_updates', 1), ('CE_updates', 1),
                           ('EMA_updates', 1), ('optimizer_updates', False),
                           ('privileged_truth_for_diagnostics_only', False),
                           ('production_evidence', False), ('source_checkpoints_verified', False),
                           ('no_downlink_forward', False),
                           ('paired_inputs_verified', False), ('n_batches', 1), ('test_used', True)]:
            with self.subTest(key=key, value=value):
                manifest = dict(original, **{key: value})
                self.runtime.atomic_json(directory/'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'identity/budget changed'):
                    runner.verify_manifest(directory)
        manifest = dict(original, data_protocol=dict(self.protocol, seed_namespace='old_namespace'))
        self.runtime.atomic_json(directory/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'data protocol changed'):
            runner.verify_manifest(directory)
        self.runtime.atomic_json(directory/'manifest.json', original)
        (directory/'summary.csv').write_text('altered\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            runner.verify_manifest(directory)

    def test_cpu_fixture_manifest_is_not_accepted_as_production_evidence(self):
        directory = self.phase()
        manifest = self.runtime.read_json(directory/'manifest.json')
        manifest['status'] = 'test_fixture'
        self.runtime.atomic_json(directory/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'identity/budget changed: status'):
            runner.verify_manifest(directory)
        manifest['status'] = 'passed'
        manifest['production_evidence'] = False
        self.runtime.atomic_json(directory/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'identity/budget changed: production_evidence'):
            runner.verify_manifest(directory)

    def test_manifest_checks_actual_source_and_manifest_source_record_identity(self):
        directory = self.phase()
        self.source.write_bytes(b'replaced frozen source')
        with self.assertRaisesRegex(ValueError, 'Pinned source changed'):
            runner.verify_manifest(directory)
        directory = self.phase('fresh')
        manifest = self.runtime.read_json(directory/'manifest.json')
        manifest['source_records']['fdma'] = {'frozen_files': []}
        self.runtime.atomic_json(directory/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'source records required'):
            runner.verify_manifest(directory)

    def test_manifest_rejects_missing_extra_artifacts_and_non_evaluation_path(self):
        directory = self.phase()
        manifest = self.runtime.read_json(directory/'manifest.json')
        manifest['files']['extra_weights.pt'] = '0'*64
        self.runtime.atomic_json(directory/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'artifact inventory required'):
            runner.verify_manifest(directory)
        with self.assertRaisesRegex(ValueError, 'evaluation attempt required'):
            runner.verify_manifest(self.output/'packages'/'attempt_x')

    def test_actual_failed_child_records_exact_log_and_retry_preserves_attempt(self):
        path = self.output/'batches'/'batch_tiny'/'batch_summary.json'
        path.parent.mkdir(parents=True)
        (self.root/'failed_child.py').write_text("raise RuntimeError('unique-ul-diagnostic-child-failure')\n")
        summary = {'stages': {}, 'core': self.core}
        with self.assertRaisesRegex(RuntimeError, 'unique-ul-diagnostic-child-failure'):
            runner.stage(summary, path, 'mechanism_diagnostics',
                lambda record: runner.child(summary, path, record, 'failed_child.py', []))
        saved = self.runtime.read_json(path)
        old = saved['stages']['mechanism_diagnostics']
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('unique-ul-diagnostic-child-failure', old['child_error'])
        self.assertIn('unique-ul-diagnostic-child-failure', Path(old['log_path']).read_text())
        self.assertIn('Exact log: '+old['log_path'], runner.compact_summary(saved))
        calls = []
        result = runner.stage(saved, path, 'mechanism_diagnostics',
                              lambda record: self.evaluation(calls)(saved, path, record))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['attempts'][0]['attempt_id'], old['attempt_id'])
        self.assertNotEqual(result['attempt_id'], old['attempt_id'])
        self.assertTrue(Path(old['log_path']).exists())

    def test_completed_stage_reuse_rejects_tampered_evidence(self):
        path = self.output/'batches'/'batch_tiny'/'batch_summary.json'
        path.parent.mkdir(parents=True)
        summary = {'stages': {}, 'core': self.core}
        calls = []
        action = lambda record: self.evaluation(calls)(summary, path, record)
        first = runner.stage(summary, path, 'mechanism_diagnostics', action)
        self.assertEqual(runner.stage(summary, path, 'mechanism_diagnostics', action), first)
        self.assertEqual(len(calls), 1)
        (Path(first['details']['output_dir'])/'samples.csv.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            runner.stage(summary, path, 'mechanism_diagnostics', action)
        with self.assertRaisesRegex(ValueError, 'no evidence'):
            runner.verify_evidence({'status': 'completed'})

    def test_package_is_explicit_scalar_archive_with_correct_hashes(self):
        directory = self.phase()
        (directory/'extra_weights.pt').write_bytes(b'not for export')
        (directory/'raw_true_channels.npy').write_bytes(b'not for export')
        summary_path = self.output/'batches'/'batch_tiny'/'batch_summary.json'
        summary_path.parent.mkdir(parents=True)
        registered = dict(namespace=self.cfg.NAMESPACE, core=self.core,
            code=self.runtime.source_identity(), recipe=self.cfg.recipe(),
            recipe_fingerprint=self.cfg.fingerprint(), optimizer_updates=0,
            UE_updates=0, CE_updates=0, EMA_updates=0, test_used=False,
            original_outputs_preserved=True)
        preregistration = summary_path.parent/'preregistration.json'
        self.runtime.atomic_json(preregistration, registered, exclusive=True)
        self.runtime.atomic_json(summary_path, dict(registered,
            preregistration_sha256=self.runtime.sha256(preregistration)))
        before = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
        result, files = runner.package_action(summary_path, directory, {'attempt_id': 'tiny'})
        self.assertEqual(result['sha256'], self.runtime.sha256(result['path']))
        with zipfile.ZipFile(result['path']) as archive:
            allowed = {*runner.EVALUATION_FILES, 'diagnostic_manifest.json', 'preregistration.json',
                       'experiment_recipe.json', 'package_whitelist.json'}
            self.assertEqual(set(archive.namelist()), allowed)
            whitelist = json.loads(archive.read('package_whitelist.json'))
            self.assertIs(whitelist['contains_model_weights'], False)
            self.assertIs(whitelist['contains_channel_tensors'], False)
            self.assertIs(whitelist['privileged_diagnostic_not_deployable'], True)
            for name, digest in whitelist['files'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
            self.assertEqual(archive.read('source_records.json'), (directory/'source_records.json').read_bytes())
        self.assertEqual(before, {path: path.read_bytes() for path in directory.iterdir() if path.is_file()})
        self.assertTrue(all(Path(path).is_file() for path in files))

    def test_guard_core_fails_before_output_creation(self):
        self.output.rmdir()
        self.runtime.guard_core = mock.Mock(side_effect=ValueError('protected core changed'))
        with self.assertRaisesRegex(ValueError, 'protected core changed'):
            runner.run(self.arguments())
        self.assertFalse(self.output.exists())

    def test_failed_diagnostic_preserves_failure_and_never_packages(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=ValueError('source byte mismatch')), \
                mock.patch.object(runner, 'package_action') as package:
            with self.assertRaisesRegex(ValueError, 'source byte mismatch'):
                runner.run(self.arguments())
            package.assert_not_called()
        summary, _ = runner.load_summary()
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['current_stage'], 'mechanism_diagnostics')
        self.assertTrue((self.output/'last_batch.json').is_file())

    def test_failed_package_resume_reuses_diagnostic_and_preserves_failed_attempt(self):
        calls, failed_paths = [], []
        real_package = runner.package_action
        def failed_package(summary_path, evaluation_directory, record):
            output = self.output/'packages'/('attempt_'+record['attempt_id'])
            output.mkdir(parents=True)
            path = output/'partial.json'
            path.write_text('{"partial":true}\n')
            failed_paths.append(path)
            raise OSError('simulated package filesystem failure')
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=self.evaluation(calls)), \
                mock.patch.object(runner, 'package_action', side_effect=failed_package):
            with self.assertRaisesRegex(OSError, 'filesystem failure'):
                runner.run(self.arguments())
        old, _ = runner.load_summary()
        old_attempt = old['stages']['result_package']['attempt_id']
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=AssertionError('must reuse diagnostic')), \
                mock.patch.object(runner, 'package_action', side_effect=real_package), redirect_stdout(io.StringIO()):
            runner.run(self.arguments(resume=True))
        summary, _ = runner.load_summary()
        self.assertEqual(summary['status'], 'completed')
        self.assertNotIn('error', summary)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary['stages']['result_package']['attempts'][0]['attempt_id'], old_attempt)
        self.assertNotEqual(summary['stages']['result_package']['attempt_id'], old_attempt)
        self.assertEqual(failed_paths[0].read_text(), '{"partial":true}\n')

    def test_completed_resume_is_read_only_and_rejects_changed_archive(self):
        calls = []
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=self.evaluation(calls)), \
                redirect_stdout(io.StringIO()):
            runner.run(self.arguments())
        before = {path: path.read_bytes() for path in self.output.rglob('*') if path.is_file()}
        with mock.patch.object(runner, 'gpu_admission') as admission, \
                mock.patch.object(runner, 'evaluation_action') as evaluate, redirect_stdout(io.StringIO()):
            runner.run(self.arguments(resume=True))
            admission.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in self.output.rglob('*') if path.is_file()})
        summary, _ = runner.load_summary()
        Path(summary['stages']['result_package']['details']['path']).write_bytes(b'tampered archive')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            runner.run(self.arguments(resume=True))

    def test_resume_rejects_changed_preregistration_before_gpu_or_diagnostic(self):
        calls = []
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=self.evaluation(calls)), \
                redirect_stdout(io.StringIO()):
            runner.run(self.arguments())
        summary, path = runner.load_summary()
        registered = self.runtime.read_json(path.parent/'preregistration.json')
        registered['recipe']['optimizer_updates'] = 1
        self.runtime.atomic_json(path.parent/'preregistration.json', registered)
        with mock.patch.object(runner, 'gpu_admission') as admission, \
                mock.patch.object(runner, 'evaluation_action') as evaluate:
            with self.assertRaisesRegex(ValueError, 'preregistration bytes changed'):
                runner.run(self.arguments(resume=True))
            admission.assert_not_called()
            evaluate.assert_not_called()
        self.assertNotIn('preregistration_sha256', registered)
        self.assertNotEqual(summary['preregistration_sha256'],
                            self.runtime.sha256(path.parent/'preregistration.json'))

    def test_existing_output_requires_explicit_resume_and_preserves_files(self):
        marker = self.output/'prior_failure.json'
        marker.write_text('{"preserve":true}\n')
        with self.assertRaisesRegex(FileExistsError, 'explicit --resume'):
            runner.run(self.arguments())
        self.assertEqual(marker.read_text(), '{"preserve":true}\n')

    def test_pointer_cannot_target_non_batch_owned_file(self):
        path = self.output/'evaluation'/'arbitrary.json'
        self.runtime.atomic_json(path, {'namespace': self.cfg.NAMESPACE})
        self.runtime.atomic_json(self.output/'last_batch.json',
            {'namespace': self.cfg.NAMESPACE, 'summary': str(path)})
        with self.assertRaisesRegex(ValueError, 'batch summary path required'):
            runner.load_summary()

    def test_cli_invalid_flags_fail_without_loading_config(self):
        for arguments in (['--resume'], ['--details']):
            with self.subTest(arguments=arguments), \
                    mock.patch.object(runner, '_config') as config, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(arguments)
                self.assertEqual(raised.exception.code, 2)
                config.assert_not_called()

    def test_actual_dry_run_works_without_site_packages_and_writes_nothing(self):
        script = Path(runner.__file__).resolve()
        output = script.parent/'runs'/'twc_v1_k8_ul_diagnostics_v1'
        existed = output.exists()
        result = subprocess.run([sys.executable, '-B', '-S', str(script), '--dry-run'],
            cwd=script.parent, check=True, capture_output=True, text=True)
        record = json.loads(result.stdout)
        self.assertIs(record['weights_read'], False)
        self.assertIs(record['writes'], False)
        self.assertEqual(record['recipe']['optimizer_updates'], 0)
        self.assertEqual(record['stages'], ['mechanism_diagnostics', 'result_package'])
        self.assertEqual(output.exists(), existed)


if __name__ == '__main__':
    unittest.main()
