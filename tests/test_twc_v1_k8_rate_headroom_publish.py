"""Real local Git checks for explicit, scalar-only result publication."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import publish_twc_v1_k8_rate_headroom as pub


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'source'
        self.root.mkdir()
        self.remote = self.root.parent/'remote.git'
        self.git('init', '--initial-branch=main')
        self.git('init', '--bare', str(self.remote))
        (self.root/'.gitignore').write_text('runs/\n')
        (self.root/'science.py').write_text('baseline\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-m', 'baseline')
        self.git('remote', 'add', 'origin', str(self.remote))
        self.head = self.git('rev-parse', 'HEAD')
        (self.root/'science.py').write_text('preserve dirty scientific file\n')
        (self.root/'staged.txt').write_text('preserve staged change\n')
        self.git('add', 'staged.txt')
        self.index = (self.root/'.git'/'index').read_bytes()
        for patch in (mock.patch.object(pub, 'ROOT', self.root),
                      mock.patch.object(pub, 'REPOSITORY', str(self.remote)),
                      mock.patch.object(pub, '_completed', return_value=None)):
            patch.start()
            self.addCleanup(patch.stop)
        self.report = self.root/'runs'/pub.NAMESPACE/'reports'/'attempt_0123456789ab'
        states = {name: 'd'*64 for name in ('codec', 'encoder', 'decoder', 'statistics', 'CE')}
        signature = hashlib.sha256(json.dumps(states, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        provenance = dict(namespace=pub.NAMESPACE, optimizer_updates=0, UE_updates=0,
            BS_updates=0, CE_updates=0, EMA_updates=0, test_used=False,
            contains_model_weights=False, contains_channel_tensors=False,
            production_evidence=True,
            source_core_sha256='a'*64, preregistration_sha256='b'*64,
            source_records={arm: {'checkpoint_sha256': 'c'*64, 'step': 10000,
                                 'production_evidence': True}
                for arm in ('fdma', 'hybrid_g4')},
            frozen_signatures={arm: dict(before=signature, after=signature, states=states)
                for arm in ('fdma', 'hybrid_g4')})
        self.payloads = {name: b'arm,gamma_db,rate\nfdma,0,30.5\n'
            for name in ('samples.csv', 'summary.csv', 'comparisons.csv')}
        self.payloads.update({'README.md': b'# Fixed-model scalar report\n',
            'input_fingerprints.json': b'{"H_dl_sha256":"eeeeeeee"}',
            'provenance.json': json.dumps(provenance).encode()})

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True,
            capture_output=True, text=True).stdout.strip()

    def preserved(self):
        self.assertEqual(self.head, self.git('rev-parse', 'HEAD'))
        self.assertEqual(self.index, (self.root/'.git'/'index').read_bytes())
        self.assertEqual('preserve dirty scientific file\n', (self.root/'science.py').read_text())
        self.assertEqual('preserve staged change\n', (self.root/'staged.txt').read_text())

    def test_directory_has_only_verified_scalars_and_existing_export_is_preserved(self):
        manifest = pub.create_export(self.report, self.payloads)
        self.assertEqual(set(pub.PUBLIC_FILES), set(manifest['files']))
        original = {p.name: p.read_bytes() for p in self.report.iterdir()}
        with self.assertRaises(FileExistsError):
            pub.create_export(self.report, self.payloads)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.report.iterdir()})
        self.assertFalse(any(self.report.parent.rglob('*.zip')))
        self.preserved()

    def test_private_metadata_nonfinite_values_arrays_and_changed_sources_are_rejected(self):
        cases = [('input_fingerprints.json', b'{"path":"/home/private/channel.pt"}'),
                 ('input_fingerprints.json', b'{"token":"do-not-publish"}'),
                 ('input_fingerprints.json', b'{"number":NaN}'),
                 ('samples.csv', b'arm,rate\nfdma,nan\n'),
                 ('samples.csv', b'arm,rate\nfdma,"[1,2]"\n')]
        for name, raw in cases:
            with self.subTest(name=name, raw=raw):
                payloads = dict(self.payloads, **{name: raw})
                with self.assertRaises(ValueError):
                    pub.create_export(self.report, payloads)
                self.assertFalse(self.report.exists())
        changed = json.loads(self.payloads['provenance.json'])
        changed['frozen_signatures']['fdma']['after'] = 'f'*64
        with self.assertRaises(ValueError):
            pub.create_export(self.report, dict(self.payloads,
                **{'provenance.json': json.dumps(changed).encode()}))
        changed['frozen_signatures']['fdma']['after'] = changed['frozen_signatures']['fdma']['before']
        changed['optimizer_updates'] = 1
        with self.assertRaises(ValueError):
            pub.create_export(self.report, dict(self.payloads,
                **{'provenance.json': json.dumps(changed).encode()}))

    def test_modified_inventory_bytes_and_symlinks_stop_before_publication(self):
        pub.create_export(self.report, self.payloads)
        extra = self.report/'checkpoint.pt'
        extra.write_bytes(b'forbidden')
        with self.assertRaises(ValueError):
            pub.verify_export(self.report)
        extra.unlink()
        item = self.report/'summary.csv'
        original = item.read_bytes()
        item.write_bytes(b'arm,rate\nfdma,99\n')
        with self.assertRaises(ValueError):
            pub.verify_export(self.report)
        item.unlink()
        target = self.root/'external.csv'
        target.write_bytes(original)
        item.symlink_to(target)
        with self.assertRaises(ValueError):
            pub.verify_export(self.report)

    def test_frozen_state_signature_must_bind_canonical_states_and_integer_zero_updates(self):
        changed = json.loads(self.payloads['provenance.json'])
        changed['frozen_signatures']['fdma']['states']['encoder'] = 'f'*64
        with self.assertRaises(ValueError):
            pub.create_export(self.report, dict(self.payloads,
                **{'provenance.json': json.dumps(changed).encode()}))
        changed = json.loads(self.payloads['provenance.json'])
        changed['UE_updates'] = False
        with self.assertRaises(ValueError):
            pub.create_export(self.report, dict(self.payloads,
                **{'provenance.json': json.dumps(changed).encode()}))
        self.assertFalse(self.report.exists())

    def test_actual_orphan_push_retry_and_exact_deletion_preserve_user_checkout(self):
        pub.create_export(self.report, self.payloads)
        original = {p.name: p.read_bytes() for p in self.report.iterdir()}
        with redirect_stdout(io.StringIO()):
            receipt = pub.publish(self.report)
            again = pub.publish(self.report)
        self.assertEqual(receipt['commit'], again['commit'])
        self.assertEqual('', self.git('show', '-s', '--format=%P', receipt['commit']))
        listing = self.git('--git-dir='+str(self.remote), 'ls-tree', '--name-only', receipt['commit'])
        self.assertEqual(set(original), set(listing.splitlines()))
        for name, expected in original.items():
            raw = subprocess.run(['git', '--git-dir='+str(self.remote), 'show',
                receipt['commit']+':'+name], check=True, capture_output=True).stdout
            self.assertEqual(expected, raw)
        self.preserved()
        with self.assertRaises(ValueError):
            pub.delete_remote(receipt['branch'], 'f'*40)
        with self.assertRaises(ValueError):
            pub.delete_remote('main', receipt['commit'])
        with redirect_stdout(io.StringIO()):
            pub.delete_remote(receipt['branch'], receipt['commit'])
        self.assertEqual('', self.git('ls-remote', '--heads', 'origin',
            'refs/heads/'+receipt['branch']))
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.report.iterdir()})
        self.preserved()

    def test_failed_git_upload_keeps_local_report_and_prepared_commit(self):
        pub.create_export(self.report, self.payloads)
        hook = self.remote/'hooks'/'pre-receive'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o700)
        with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
            pub.publish(self.report)
        receipt = json.loads((self.report.parent.parent/'uploads'/'0123456789ab.json').read_text())
        self.assertEqual('failed', receipt['status'])
        self.assertEqual('commit', self.git('cat-file', '-t', receipt['commit']))
        self.assertEqual(pub.NAMESPACE, pub.verify_export(self.report)['namespace'])
        self.preserved()

    def test_completed_run_binding_is_checked_before_network(self):
        pub.create_export(self.report, self.payloads)
        with mock.patch.object(pub, '_completed', side_effect=ValueError('incomplete batch')):
            with self.assertRaises(ValueError):
                pub.publish(self.report)
        self.assertEqual('', self.git('ls-remote', '--heads', 'origin'))
        self.preserved()


if __name__ == '__main__':
    unittest.main()
