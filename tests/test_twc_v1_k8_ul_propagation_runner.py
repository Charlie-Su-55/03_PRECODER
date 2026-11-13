"""Real owned IO and subprocess checks for the offline propagation queue."""
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

import run_twc_v1_k8_ul_propagation as runner


class PropagationQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'propagation'
        self.output.mkdir()
        self.source = self.root/'frozen_source.pt'
        self.source.write_bytes(b'queue identity fixture; not a scientific checkpoint')
        self.core = {'schema': 'tiny_propagation_core', 'sha256': {'test': 'a'*64}}
        self.protocol = dict(seed_namespace='tiny_propagation', base_seed=20261121,
                             execution=dict(device='cuda:0', sionna_precision='single'))
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='tiny_propagation', DATA_NAMESPACE='tiny_propagation',
            ARMS=('fdma', 'hybrid_g4'), GAMMAS=(0, 5, 10, 15, 20, 25),
            MICRO_BATCH=4, BATCHES=64, SAMPLES=256,
            recipe=lambda: dict(namespace='tiny_propagation', optimizer_updates=0),
            fingerprint=lambda: 'd'*64)

        def inside(value):
            original, path = Path(value), Path(value).resolve()
            if original.is_symlink() or path == self.output or not path.is_relative_to(self.output):
                raise ValueError('Own propagation child required')
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
            source_identity=lambda: dict(commit='fixture-only', sha256=self.core['sha256']),
            process_lock=lambda path: nullcontext(), make_data=make_data,
            verify_source_records=verify_source_records)

        def export_directory(summary_path, evaluation_directory, output):
            output.mkdir(parents=True, exist_ok=False)
            contents = {'summary.csv': (evaluation_directory/'summary.csv').read_bytes(),
                        'provenance.json': b'{"public_scalar_fixture":true}\n',
                        'README.md': b'Scalar fixture only, no weights or tensors.\n'}
            hashes = {}
            for name, content in contents.items():
                (output/name).write_bytes(content)
                hashes[name] = sha256(output/name)
            atomic_json(output/'export_manifest.json', dict(files=hashes), exclusive=True)
            return dict(path=str(output), manifest=str(output/'export_manifest.json'),
                        files=hashes, format=self.cfg.NAMESPACE+'_export_v1')

        def verify_export_directory(path):
            path = Path(path)
            manifest = self.runtime.read_json(path/'export_manifest.json')
            if set(p.name for p in path.iterdir()) != set(manifest['files']) | {'export_manifest.json'}:
                raise ValueError('Report inventory changed')
            for name, digest in manifest['files'].items():
                if sha256(path/name) != digest:
                    raise ValueError('Report bytes changed')
            return manifest

        self.publisher = SimpleNamespace(export_directory=mock.Mock(side_effect=export_directory),
            verify_export_directory=mock.Mock(side_effect=verify_export_directory),
            publish=mock.Mock(side_effect=AssertionError('queue must never publish online')))
        for patcher in (mock.patch.object(runner, '_runtime', return_value=self.runtime),
                        mock.patch.object(runner, '_config', return_value=self.cfg),
                        mock.patch.object(runner, '_publisher', return_value=self.publisher)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def arguments(self, *, resume=False):
        return SimpleNamespace(resume=resume, allow_shared_gpu=True,
                               min_free_mib=12000, max_gpu_util=80)

    def phase(self, attempt='tiny'):
        directory = self.output/'evaluation'/('attempt_'+attempt)
        directory.mkdir(parents=True)
        summary = io.StringIO()
        writer = csv.DictWriter(summary, fieldnames=('arm', 'scope', 'gamma_db',
            'soft_actual_to_reported_mse_ratio_mean', 'soft_actual_to_propagated_mse_ratio_mean'))
        writer.writeheader()
        for arm in self.cfg.ARMS:
            for gamma in self.cfg.GAMMAS:
                writer.writerow(dict(arm=arm, scope='point', gamma_db=gamma,
                    soft_actual_to_reported_mse_ratio_mean=3.,
                    soft_actual_to_propagated_mse_ratio_mean=1.))
            writer.writerow(dict(arm=arm, scope='all_six', gamma_db='',
                soft_actual_to_reported_mse_ratio_mean=3.,
                soft_actual_to_propagated_mse_ratio_mean=1.))
        records = {arm: dict(frozen_files=[dict(path=str(self.source),
            sha256=self.runtime.sha256(self.source))]) for arm in self.cfg.ARMS}
        contents = {
            'samples.csv.gz': gzip.compress(b'arm,gamma_db,soft_nmse_linear\nfdma,0,0.1\n', mtime=0),
            'summary.csv': summary.getvalue().encode(),
            'comparisons.csv': b'arm,comparison,scope,metric,mean\nfdma,propagation,all_six,mse,0.2\n',
            'calibration_bins.csv.gz': gzip.compress(b'arm,component,bin,count\nfdma,reported,0,1\n', mtime=0),
            'input_fingerprints.json': b'[{"normal_filter_sha256":"1111","H_ul":"2222"}]\n',
            'source_records.json': json.dumps(records, sort_keys=True).encode(),
        }
        self.assertEqual(set(contents), set(runner.EVALUATION_FILES))
        for name, value in contents.items():
            (directory/name).write_bytes(value)
        manifest = dict(status='passed', phase='propagation', format=self.cfg.NAMESPACE+'_manifest_v1',
            core=self.core, recipe_fingerprint=self.cfg.fingerprint(), arms=list(self.cfg.ARMS),
            gamma_db=list(self.cfg.GAMMAS), batch_size=4, n_batches=64, samples_per_condition=256,
            optimizer_updates=0, UE_updates=0, CE_updates=0, EMA_updates=0, test_used=False,
            original_filter_unchanged=True, true_UL_used_for_scoring_only=True,
            filter_reconstruction_verified=True, error_decomposition_verified=True,
            paired_inputs_verified=True, frozen_models_verified=True, production_evidence=True,
            source_checkpoints_verified=True, no_downlink_forward=True, data_protocol=self.protocol,
            source_records=records, files={name: self.runtime.sha256(directory/name) for name in contents})
        self.runtime.atomic_json(directory/'manifest.json', manifest, exclusive=True)
        return directory

    def evaluation(self, calls):
        def action(summary, path, record):
            calls.append(record['attempt_id'])
            directory = self.phase(record['attempt_id'])
            log = path.parent/('fixture_'+record['attempt_id']+'.log')
            log.write_text('offline evaluator fixture\n')
            manifest = runner.verify_manifest(directory)
            return dict(output_dir=str(directory), manifest=manifest), [
                directory/'manifest.json', *[directory/name for name in runner.EVALUATION_FILES], log]
        return action

    def register(self):
        path = self.output/'batches'/'batch_tiny'/'batch_summary.json'
        path.parent.mkdir(parents=True)
        record = dict(namespace=self.cfg.NAMESPACE, core=self.core,
            code=self.runtime.source_identity(), recipe=self.cfg.recipe(),
            recipe_fingerprint=self.cfg.fingerprint(), optimizer_updates=0,
            UE_updates=0, CE_updates=0, EMA_updates=0, test_used=False,
            original_outputs_preserved=True)
        preregistration = path.parent/'preregistration.json'
        self.runtime.atomic_json(preregistration, record, exclusive=True)
        self.runtime.atomic_json(path, dict(record,
            preregistration_sha256=self.runtime.sha256(preregistration)))
        return path

    def test_stage_plan_is_fixed_filter_and_directory_export_only(self):
        self.assertEqual(runner.stage_names(), ('mechanism_propagation', 'directory_export'))

    def test_manifest_rejects_updates_filter_change_truth_leak_and_nonproduction(self):
        directory = self.phase()
        original = self.runtime.read_json(directory/'manifest.json')
        runner.verify_manifest(directory)
        for key, value in [('optimizer_updates', 1), ('UE_updates', 1), ('CE_updates', 1),
                           ('EMA_updates', 1), ('optimizer_updates', False),
                           ('original_filter_unchanged', False), ('true_UL_used_for_scoring_only', False),
                           ('filter_reconstruction_verified', False), ('error_decomposition_verified', False),
                           ('production_evidence', False), ('source_checkpoints_verified', False),
                           ('no_downlink_forward', False), ('status', 'test_fixture'),
                           ('paired_inputs_verified', False), ('n_batches', 1), ('test_used', True)]:
            with self.subTest(key=key, value=value):
                self.runtime.atomic_json(directory/'manifest.json', dict(original, **{key: value}))
                with self.assertRaisesRegex(ValueError, 'identity/budget changed'):
                    runner.verify_manifest(directory)

    def test_manifest_rejects_changed_protocol_inventory_bytes_and_teacher(self):
        directory = self.phase()
        original = self.runtime.read_json(directory/'manifest.json')
        for key, value, message in (
            ('data_protocol', dict(self.protocol, seed_namespace='old_namespace'), 'data protocol changed'),
            ('files', dict(original['files'], weights_pt='0'*64), 'artifact inventory required'),
            ('source_records', dict(original['source_records'], fdma={}), 'source records required')):
            self.runtime.atomic_json(directory/'manifest.json', dict(original, **{key: value}))
            with self.assertRaisesRegex(ValueError, message):
                runner.verify_manifest(directory)
        self.runtime.atomic_json(directory/'manifest.json', original)
        (directory/'summary.csv').write_text('altered\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            runner.verify_manifest(directory)
        fresh = self.phase('fresh')
        self.source.write_bytes(b'changed source identity')
        with self.assertRaisesRegex(ValueError, 'Pinned source changed'):
            runner.verify_manifest(fresh)
        with self.assertRaisesRegex(ValueError, 'evaluation attempt required'):
            runner.verify_manifest(self.output/'reports'/'attempt_x')

    def test_actual_failed_child_log_and_retry_attempt_remain_available(self):
        path = self.output/'batches'/'batch_child'/'batch_summary.json'
        path.parent.mkdir(parents=True)
        (self.root/'failed_child.py').write_text("raise RuntimeError('propagation-child-failure')\n")
        summary = dict(stages={}, core=self.core)
        with self.assertRaisesRegex(RuntimeError, 'propagation-child-failure'):
            runner.stage(summary, path, 'mechanism_propagation',
                lambda record: runner.child(summary, path, record, 'failed_child.py', []))
        saved = self.runtime.read_json(path)
        old = saved['stages']['mechanism_propagation']
        self.assertIn('propagation-child-failure', Path(old['log_path']).read_text())
        self.assertIn('Exact log: '+old['log_path'], runner.compact_summary(saved))
        calls = []
        result = runner.stage(saved, path, 'mechanism_propagation',
            lambda record: self.evaluation(calls)(saved, path, record))
        self.assertEqual(result['attempts'][0]['attempt_id'], old['attempt_id'])
        self.assertNotEqual(result['attempt_id'], old['attempt_id'])
        self.assertTrue(Path(old['log_path']).is_file())

    def test_completed_stage_reuses_only_unchanged_evidence(self):
        path = self.output/'batches'/'batch_stage'/'batch_summary.json'
        path.parent.mkdir(parents=True)
        summary = dict(stages={}, core=self.core)
        calls = []
        action = lambda record: self.evaluation(calls)(summary, path, record)
        first = runner.stage(summary, path, 'mechanism_propagation', action)
        self.assertEqual(runner.stage(summary, path, 'mechanism_propagation', action), first)
        self.assertEqual(len(calls), 1)
        (Path(first['details']['output_dir'])/'samples.csv.gz').write_bytes(b'tamper')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            runner.stage(summary, path, 'mechanism_propagation', action)

    def test_export_is_explicit_local_directory_and_never_publishes(self):
        directory = self.phase()
        (directory/'weights.pt').write_bytes(b'private weight bytes')
        (directory/'true_channels.npy').write_bytes(b'private channel bytes')
        summary_path = self.register()
        before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        result, files = runner.export_action(summary_path, directory, dict(attempt_id='tiny'))
        output = Path(result['path'])
        self.assertEqual(set(p.name for p in output.iterdir()),
            {'summary.csv', 'provenance.json', 'README.md', 'export_manifest.json'})
        self.assertTrue(all(path.is_file() for path in files))
        self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir() if p.is_file()})
        self.assertFalse(any(p.suffix == '.zip' for p in self.output.rglob('*')))
        self.publisher.publish.assert_not_called()

    def test_export_result_cannot_escape_owned_report_or_sneak_extra_paths(self):
        directory = self.phase()
        summary_path = self.register()
        for bad in (
            dict(path=str(self.root/'external'), manifest=str(self.root/'external'/'export_manifest.json'), files={'a.csv': '0'*64}),
            dict(path=str(self.output/'reports'/'attempt_tiny'), manifest=str(self.output/'reports'/'attempt_tiny'/'export_manifest.json'), files={'../weights.pt': '0'*64}),
        ):
            self.publisher.export_directory.return_value = bad
            self.publisher.export_directory.side_effect = None
            with self.assertRaises(ValueError):
                runner.export_action(summary_path, directory, dict(attempt_id='tiny'))

    def test_guard_fails_before_owned_output_creation(self):
        self.output.rmdir()
        self.runtime.guard_core = mock.Mock(side_effect=ValueError('old core changed'))
        with self.assertRaisesRegex(ValueError, 'old core changed'):
            runner.run(self.arguments())
        self.assertFalse(self.output.exists())

    def test_failed_diagnostic_does_not_export_or_publish(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=ValueError('source mismatch')), \
                mock.patch.object(runner, 'export_action') as exported:
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                runner.run(self.arguments())
            exported.assert_not_called()
        summary, _ = runner.load_summary()
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['current_stage'], 'mechanism_propagation')
        self.publisher.publish.assert_not_called()

    def test_failed_export_resume_reuses_diagnostic_and_keeps_partial_directory(self):
        calls, partials = [], []
        real_export = runner.export_action
        def failed_export(summary_path, evaluation_directory, record):
            directory = self.output/'reports'/('attempt_'+record['attempt_id'])
            directory.mkdir(parents=True)
            marker = directory/'partial.json'
            marker.write_text('{"partial":true}\n')
            partials.append(marker)
            raise OSError('export filesystem failure')
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=self.evaluation(calls)), \
                mock.patch.object(runner, 'export_action', side_effect=failed_export):
            with self.assertRaisesRegex(OSError, 'filesystem failure'):
                runner.run(self.arguments())
        old, _ = runner.load_summary()
        old_id = old['stages']['directory_export']['attempt_id']
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=AssertionError('must reuse')), \
                mock.patch.object(runner, 'export_action', side_effect=real_export), redirect_stdout(io.StringIO()):
            runner.run(self.arguments(resume=True))
        summary, _ = runner.load_summary()
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary['stages']['directory_export']['attempts'][0]['attempt_id'], old_id)
        self.assertNotEqual(summary['stages']['directory_export']['attempt_id'], old_id)
        self.assertEqual(partials[0].read_text(), '{"partial":true}\n')
        self.publisher.publish.assert_not_called()

    def test_completed_resume_is_readonly_and_rejects_report_tampering(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=self.evaluation([])), \
                redirect_stdout(io.StringIO()):
            runner.run(self.arguments())
        before = {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        with mock.patch.object(runner, 'gpu_admission') as admission, \
                mock.patch.object(runner, 'evaluation_action') as evaluate, redirect_stdout(io.StringIO()):
            runner.run(self.arguments(resume=True))
            admission.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()})
        summary, _ = runner.load_summary()
        report = Path(summary['stages']['directory_export']['details']['path'])
        (report/'unexpected_weights.pt').write_bytes(b'not allowed')
        with self.assertRaisesRegex(ValueError, 'Report inventory changed'):
            runner.run(self.arguments(resume=True))

    def test_preregistration_tamper_stops_before_gpu_or_diagnostic(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=self.evaluation([])), \
                redirect_stdout(io.StringIO()):
            runner.run(self.arguments())
        summary, path = runner.load_summary()
        registered = self.runtime.read_json(path.parent/'preregistration.json')
        registered['recipe']['optimizer_updates'] = 1
        self.runtime.atomic_json(path.parent/'preregistration.json', registered)
        with mock.patch.object(runner, 'gpu_admission') as admission:
            with self.assertRaisesRegex(ValueError, 'preregistration bytes changed'):
                runner.run(self.arguments(resume=True))
            admission.assert_not_called()
        self.assertNotEqual(summary['preregistration_sha256'],
                            self.runtime.sha256(path.parent/'preregistration.json'))

    def test_existing_output_and_pointer_scope_preserve_prior_files(self):
        marker = self.output/'prior_failure.json'
        marker.write_text('{"preserve":true}\n')
        with self.assertRaisesRegex(FileExistsError, 'explicit --resume'):
            runner.run(self.arguments())
        self.assertEqual(marker.read_text(), '{"preserve":true}\n')
        path = self.output/'evaluation'/'not_a_batch.json'
        self.runtime.atomic_json(path, dict(namespace=self.cfg.NAMESPACE))
        self.runtime.atomic_json(self.output/'last_batch.json',
            dict(namespace=self.cfg.NAMESPACE, summary=str(path)), exclusive=True)
        with self.assertRaisesRegex(ValueError, 'batch summary path required'):
            runner.load_summary()

    def test_summary_handles_pooled_scopes_and_prints_only_explicit_upload_command(self):
        with mock.patch.object(runner, 'gpu_admission', return_value={'allowed': True}), \
                mock.patch.object(runner, 'evaluation_action', side_effect=self.evaluation([])), \
                redirect_stdout(io.StringIO()):
            runner.run(self.arguments())
        summary, _ = runner.load_summary()
        text = runner.compact_summary(summary)
        self.assertIn('soft_actual_to_reported_mse_ratio: 3,3,3,3,3,3', text)
        self.assertIn('Report directory:', text)
        self.assertIn('--publish --export-dir', text)
        self.publisher.publish.assert_not_called()

    def test_cli_invalid_combinations_fail_before_config(self):
        for flags in (['--resume'], ['--details']):
            with mock.patch.object(runner, '_config') as config, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    runner.main(flags)
                self.assertEqual(caught.exception.code, 2)
                config.assert_not_called()

    def test_actual_stdlib_dry_run_and_summary_do_not_create_output(self):
        script = Path(runner.__file__).resolve()
        output = script.parent/'runs'/'twc_v1_k8_ul_propagation_v1'
        before = output.exists()
        result = subprocess.run([sys.executable, '-B', '-S', str(script), '--dry-run'],
            cwd=script.parent, check=True, capture_output=True, text=True)
        record = json.loads(result.stdout)
        self.assertFalse(record['weights_read'])
        self.assertFalse(record['writes'])
        self.assertEqual(record['recipe']['optimizer_updates'], 0)
        self.assertEqual(record['stages'], ['mechanism_propagation', 'directory_export'])
        subprocess.run([sys.executable, '-B', '-S', str(script), '--summary'],
            cwd=script.parent, check=True, capture_output=True, text=True)
        self.assertEqual(output.exists(), before)


if __name__ == '__main__':
    unittest.main()
