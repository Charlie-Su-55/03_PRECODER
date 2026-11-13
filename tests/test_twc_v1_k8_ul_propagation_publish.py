"""Public-directory safety and actual Git plumbing against a local bare remote."""
from contextlib import contextmanager, redirect_stdout
import gzip
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import publish_twc_v1_k8_ul_propagation as publication


class ResultPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'source'
        self.root.mkdir()
        self.remote = self.root.parent/'remote.git'
        self.git('init', '--initial-branch=main')
        self.git('init', '--bare', str(self.remote))
        (self.root/'.gitignore').write_text('runs/\n')
        (self.root/'source.py').write_text('unchanged scientific source\n')
        self.git('add', '.gitignore', 'source.py')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-m', 'scientific baseline')
        self.git('remote', 'add', 'origin', str(self.remote))
        self.head = self.git('rev-parse', 'HEAD')
        (self.root/'source.py').write_text('preserve this unrelated dirty source\n')
        (self.root/'staged.txt').write_text('preserve original user index\n')
        self.git('add', 'staged.txt')
        self.index_before = (self.root/'.git'/'index').read_bytes()
        self.output = self.root/'runs'/'fixture_propagation'
        self.output.mkdir(parents=True)
        self.cfg = SimpleNamespace(ROOT=self.root, OUTPUT_ROOT=self.output,
            NAMESPACE='fixture_propagation', ARMS=('fdma', 'hybrid_g4'),
            GAMMAS=(0, 5, 10, 15, 20, 25), MICRO_BATCH=4, BATCHES=1, SAMPLES=4,
            recipe=lambda: dict(physical=dict(K=8, total_RE=256, Ed=48., Ep=16.)))

        def inside(value):
            original, path = Path(value), Path(value).resolve()
            if original.is_symlink() or path == self.output or not path.is_relative_to(self.output):
                raise ValueError('Owned propagation child required')
            return path

        def atomic_json(path, value, exclusive=False):
            path = inside(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x' if exclusive else 'w') as stream:
                json.dump(value, stream, sort_keys=True)

        @contextmanager
        def process_lock(path):
            yield

        self.runtime = SimpleNamespace(inside=inside, atomic_json=atomic_json,
            read_json=lambda p: json.loads(Path(p).read_text()), sha256=publication._sha,
            process_lock=process_lock, guard_core=lambda core: None)
        self.core = dict(schema='fixture', sha256={'relative/source.py': 'a'*64})
        self.evaluation = self.output/'evaluation'/'attempt_aaaaaaaaaaaa'
        self.evaluation.mkdir(parents=True)
        for name in publication.SCALAR_FILES:
            if name.endswith('.json'):
                content = b'[{"arm":"fdma","channel_hash":"aaaaaaaa"}]\n'
            else:
                content = b'arm,gamma_db,metric\nfdma,0,0.25\n'
                if name == 'samples.csv.gz':
                    stream = io.StringIO()
                    fields = ('arm', 'gamma_db', 'sample_index', 'batch_index', 'within_batch',
                        'soft_nmse_linear', 'soft_nominal_scaled_actual_mse',
                        'soft_nominal_scaled_reported_variance', 'soft_nominal_scaled_propagated_variance')
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for arm in self.cfg.ARMS:
                        for gamma in self.cfg.GAMMAS:
                            for index in range(self.cfg.SAMPLES):
                                writer.writerow(dict(arm=arm, gamma_db=gamma, sample_index=index,
                                    batch_index=index//4, within_batch=index % 4,
                                    **{field: .1 for field in fields[5:]}))
                    content = stream.getvalue().encode()
                if name.endswith('.gz'):
                    content = gzip.compress(content, mtime=0)
            (self.evaluation/name).write_bytes(content)
        frozen = {arm: dict(before={'codec': '1'*64, 'statistics': '2'*64, 'CE': '3'*64},
                            after={'codec': '1'*64, 'statistics': '2'*64, 'CE': '3'*64})
                  for arm in self.cfg.ARMS}
        self.manifest = dict(status='passed', core=self.core, recipe_fingerprint='d'*64,
            data_protocol=dict(seed_namespace='fixture', execution=dict(device='cuda:0')),
            frozen_signatures=frozen,
            source_records={arm: dict(kind='rate', arm=arm, model_commit='9'*40,
                checkpoint_sha256='e'*64, step=5000, state_sha256='1'*64,
                CE_state_sha256='3'*64, statistics_state_sha256='2'*64,
                frozen_files=[dict(path='/home/private-user/weights.pt', sha256='e'*64)])
                for arm in self.cfg.ARMS})
        self.runtime.atomic_json(self.evaluation/'manifest.json', self.manifest)
        self.summary_path = self.output/'batches'/'batch_fixture'/'batch_summary.json'
        self.summary_path.parent.mkdir(parents=True)
        prereg = self.summary_path.parent/'preregistration.json'
        prereg.write_text('{"fixture":true}\n')
        self.summary = dict(namespace=self.cfg.NAMESPACE, core=self.core,
            recipe_fingerprint='d'*64, code=dict(commit=self.head),
            preregistration_sha256=publication._sha(prereg), stages={}, status='completed')
        self.runtime.atomic_json(self.summary_path, self.summary)
        self.queue = SimpleNamespace(verify_manifest=lambda path: self.manifest,
            load_summary=lambda: (self.runtime.read_json(self.summary_path), self.summary_path),
            verify_evidence=lambda record: None)
        for patch in (mock.patch.object(publication, '_config', return_value=self.cfg),
                      mock.patch.object(publication, '_runtime', return_value=self.runtime),
                      mock.patch.object(publication, '_queue', return_value=self.queue),
                      mock.patch.object(publication, 'REPOSITORY', str(self.remote))):
            patch.start()
            self.addCleanup(patch.stop)
        self.report = self.output/'reports'/'attempt_0123456789ab'

    def git(self, *args, cwd=None):
        return subprocess.run(['git', *args], cwd=cwd or self.root, check=True,
            text=True, capture_output=True).stdout.strip()

    def export(self):
        result = publication.export_directory(self.summary_path, self.evaluation, self.report)
        self.summary['stages']['directory_export'] = dict(status='completed', details=result)
        self.runtime.atomic_json(self.summary_path, self.summary)
        return result

    def assert_source_preserved(self):
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.head)
        self.assertEqual((self.root/'.git'/'index').read_bytes(), self.index_before)
        self.assertEqual((self.root/'source.py').read_text(), 'preserve this unrelated dirty source\n')
        self.assertEqual((self.root/'staged.txt').read_text(), 'preserve original user index\n')

    def test_public_export_exact_inventory_redacts_paths_and_never_creates_zip(self):
        result = self.export()
        self.assertEqual(set(result['files']), set(publication.PUBLIC_FILES))
        self.assertEqual(result['manifest_sha256'], publication._sha(self.report/'export_manifest.json'))
        provenance = (self.report/'provenance.json').read_text()
        self.assertNotIn('/home/', provenance)
        self.assertNotIn('private-user', provenance)
        self.assertNotIn('frozen_files', provenance)
        self.assertFalse(any(p.suffix == '.zip' for p in self.output.rglob('*')))
        publication.verify_export_directory(self.report)
        self.assert_source_preserved()

    def test_existing_report_is_not_overwritten(self):
        self.export()
        before = (self.report/'summary.csv').read_bytes()
        with self.assertRaises(FileExistsError):
            publication.export_directory(self.summary_path, self.evaluation, self.report)
        self.assertEqual((self.report/'summary.csv').read_bytes(), before)

    def test_extra_weights_changed_bytes_links_and_manifest_fields_are_rejected(self):
        self.export()
        extra = self.report/'checkpoint.pt'
        extra.write_bytes(b'never publish weights')
        with self.assertRaises(ValueError):
            publication.verify_export_directory(self.report)
        extra.unlink()
        target = self.report/'summary.csv'
        original = target.read_bytes()
        target.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            publication.verify_export_directory(self.report)
        target.unlink()
        target.symlink_to(self.root/'source.py')
        with self.assertRaises(ValueError):
            publication.verify_export_directory(self.report)
        target.unlink()
        target.write_bytes(original)
        manifest = self.runtime.read_json(self.report/'export_manifest.json')
        manifest['password'] = 'not-a-real-secret'
        self.runtime.atomic_json(self.report/'export_manifest.json', manifest)
        with self.assertRaises(ValueError):
            publication.verify_export_directory(self.report)

    def test_machine_paths_in_fingerprints_and_changed_preregistration_stop_export(self):
        (self.evaluation/'input_fingerprints.json').write_text('{"path":"/home/user/private"}')
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.report.exists())
        (self.evaluation/'input_fingerprints.json').write_text('[]')
        (self.summary_path.parent/'preregistration.json').write_text('changed')
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.report.exists())

    def test_nonscalar_csv_payload_and_incomplete_sample_inventory_are_rejected(self):
        path = self.evaluation/'summary.csv'
        path.write_text('arm,gamma_db,soft_mean\nfdma,0,"[1,2,3]"\n')
        with self.assertRaises(ValueError):
            self.export()
        path.write_text('arm,gamma_db,soft_mean\nfdma,0,0.1\n')
        samples = self.evaluation/'samples.csv.gz'
        rows = gzip.decompress(samples.read_bytes()).decode().splitlines()
        samples.write_bytes(gzip.compress(('\n'.join(rows[:-1])+'\n').encode(), mtime=0))
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.report.exists())

    def test_actual_orphan_push_keeps_head_index_worktree_and_export_immutable(self):
        result = self.export()
        before = {p.name: p.read_bytes() for p in self.report.iterdir()}
        with redirect_stdout(io.StringIO()):
            receipt = publication.publish(self.report)
        self.assertEqual(receipt['status'], 'uploaded')
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/'+receipt['branch']).split()[0], receipt['commit'])
        self.assertEqual(self.git('show', '-s', '--format=%P', receipt['commit']), '')
        names = self.git('--git-dir='+str(self.remote), 'ls-tree', '--name-only', receipt['commit']).splitlines()
        self.assertEqual(set(names), set(publication.PUBLIC_FILES) | {'export_manifest.json'})
        for name in names:
            data = subprocess.run(['git', '--git-dir='+str(self.remote), 'show', receipt['commit']+':'+name],
                check=True, capture_output=True).stdout
            self.assertEqual(data, before[name])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.report.iterdir()})
        self.assert_source_preserved()
        with redirect_stdout(io.StringIO()):
            again = publication.publish(self.report)
        self.assertEqual(receipt['commit'], again['commit'])
        self.assert_source_preserved()

    def test_distinct_existing_result_branch_is_not_overwritten(self):
        self.export()
        branch = publication._branch('0123456789ab')
        self.git('push', 'origin', self.head+':refs/heads/'+branch)
        with self.assertRaisesRegex(RuntimeError, 'upload failed'):
            publication.publish(self.report)
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/'+branch).split()[0], self.head)
        self.assertTrue(self.report.is_dir())
        self.assert_source_preserved()

    def test_push_failure_preserves_export_and_prepared_commit_for_retry(self):
        self.export()
        saved = (self.report/'summary.csv').read_bytes()
        self.git('remote', 'set-url', 'origin', str(self.root.parent/'missing-remote.git'))
        with mock.patch.object(publication, 'REPOSITORY', str(self.root.parent/'missing-remote.git')):
            with self.assertRaisesRegex(RuntimeError, 'local result export is preserved'):
                publication.publish(self.report)
        receipt = self.runtime.read_json(self.output/'uploads'/'0123456789ab.json')
        self.assertEqual(receipt['status'], 'failed')
        self.git('remote', 'set-url', 'origin', str(self.remote))
        with redirect_stdout(io.StringIO()):
            retry = publication.publish(self.report)
        self.assertEqual(receipt['commit'], retry['commit'])
        self.assertEqual((self.report/'summary.csv').read_bytes(), saved)
        self.assert_source_preserved()

    def test_delete_requires_exact_result_ref_commit_and_preserves_main(self):
        self.export()
        self.git('push', 'origin', self.head+':refs/heads/main')
        with redirect_stdout(io.StringIO()):
            receipt = publication.publish(self.report)
        for branch, expected in [('main', receipt['commit']),
                                 (receipt['branch'], self.head),
                                 (receipt['branch']+'/other', receipt['commit'])]:
            with self.assertRaises(ValueError):
                publication.delete_remote(branch, expected)
        with redirect_stdout(io.StringIO()):
            publication.delete_remote(receipt['branch'], receipt['commit'])
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/'+receipt['branch']), '')
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/main').split()[0], self.head)
        self.assertTrue(self.report.exists())
        self.assert_source_preserved()

    def test_delete_lease_rejects_actual_ref_change_after_precheck(self):
        self.export()
        self.git('push', 'origin', self.head+':refs/heads/main')
        with redirect_stdout(io.StringIO()):
            receipt = publication.publish(self.report)
        other_file = self.root.parent/'other-scalar.txt'
        other_file.write_text('another result arrived after the precheck\n')
        other_commit, _ = publication._orphan_commit(self.root, {'other.txt': other_file})
        self.git('push', 'origin', other_commit+':refs/heads/race-staging')
        reference = 'refs/heads/'+receipt['branch']
        real_run = subprocess.run
        changed = []

        def race_then_execute(command, *args, **kwargs):
            if (command[:2] == ['git', 'push']
                    and any(value.startswith('--force-with-lease=') for value in command)):
                real_run(['git', '--git-dir='+str(self.remote), 'update-ref', reference,
                          other_commit, receipt['commit']], check=True, capture_output=True)
                changed.append(True)
            return real_run(command, *args, **kwargs)

        with mock.patch.object(publication.subprocess, 'run', side_effect=race_then_execute):
            with self.assertRaisesRegex(RuntimeError, 'deletion failed'):
                publication.delete_remote(receipt['branch'], receipt['commit'])
        self.assertEqual(changed, [True])
        self.assertEqual(self.git('ls-remote', 'origin', reference).split()[0], other_commit)
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/main').split()[0], self.head)
        self.assertTrue(self.report.exists())
        self.assert_source_preserved()

    def test_network_debug_traces_are_not_inherited_and_prompt_policy_is_not_forced(self):
        with mock.patch.dict('os.environ', {'GIT_TRACE': '1', 'GIT_TRACE_CURL': '1',
                                           'GIT_CURL_VERBOSE': '1', 'https_proxy': 'http://localhost:3129'}):
            environment = publication._git_environment()
        self.assertNotIn('GIT_TRACE', environment)
        self.assertNotIn('GIT_TRACE_CURL', environment)
        self.assertNotIn('GIT_CURL_VERBOSE', environment)
        self.assertEqual(environment['https_proxy'], 'http://localhost:3129')


if __name__ == '__main__':
    unittest.main()
