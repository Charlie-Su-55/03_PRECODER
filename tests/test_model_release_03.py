"""Real tar I/O verifies model byte preservation; fixtures are not torch models."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import model_release_03 as archive

CE = 'twc_v1_k8_ce_refine_v1'
SOURCE = 'twc_v1_k8_icc_aligned_savefix_v1'
LOW = 'twc_v1_k8_low_snr_interference_v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def init_repo(root):
    root.mkdir()
    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init', '--initial-branch=main')
    (root / '.gitignore').write_text('runs/\n')
    git('add', '.gitignore')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-m', 'fixture source identity')
    return git('rev-parse', 'HEAD')


def terminal(root, namespace, name, step, *, source_files=(), production=True):
    directory = root / 'runs' / namespace / 'training' / name
    directory.mkdir(parents=True)
    # Deliberately not a pickle: packaging must never deserialize checkpoint bytes.
    raw = b'not-a-pickle\x00\xff' + name.encode() + bytes(range(256)) * 4
    (directory / 'checkpoint_latest.pt').write_bytes(raw)
    record = dict(status='completed', step=step, production_evidence=production,
                  checkpoint_sha256=digest(raw), source_identity=dict(frozen_files=list(source_files)))
    if namespace == LOW:
        progress = b'{\"step\":10000}\n'
        (directory / 'progress.jsonl').write_bytes(progress)
        record['progress_sha256'] = digest(progress)
    write_json(directory / 'manifest.json', record)
    write_json(directory / 'terminal_receipt.json', record)
    write_json(directory / 'initialization.json', {'source': 'fixture'})
    if namespace == CE:
        ce_history(directory)
    return directory


def ce_history(directory):
    for boundary in range(0, 60001, 5000):
        raw = b'immutable source snapshot\x00' + str(boundary).encode()
        (directory / f'checkpoint_step_{boundary:05d}.pt').write_bytes(raw)
        write_json(directory / f'checkpoint_step_{boundary:05d}.json',
                   {'step': boundary, 'checkpoint_sha256': digest(raw)})
        monitor = directory / 'monitor' / f'step_{boundary:05d}'
        monitor.mkdir(parents=True)
        summary = b'gamma_db,nmse\n0,0.1\n'
        inputs = b'{\"paired\":true}\n'
        (monitor / 'summary.csv').write_bytes(summary)
        (monitor / 'input_fingerprints.json').write_bytes(inputs)
        write_json(monitor / 'manifest.json', dict(status='completed', step=boundary,
            files={'summary.csv': digest(summary), 'input_fingerprints.json': digest(inputs)}))


class ModelArchiveTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'source'
        self.commit = init_repo(self.root)
        self.bundle = self.base / 'bundle'
        self.source = terminal(self.root, SOURCE, 'ce', 5000)
        source_ref = dict(path=str(self.source / 'checkpoint_latest.pt'),
                          sha256=digest((self.source / 'checkpoint_latest.pt').read_bytes()))
        write_json(self.root / 'runs' / CE / 'summary.json', {'status': 'completed', 'namespace': CE})
        self.new = {}
        for name in ('nmse', 'geometry'):
            directory = terminal(self.root, CE, name, 60000, source_files=[source_ref])
            self.new[name] = directory
        (self.root / 'overleaf').mkdir()
        (self.root / 'overleaf' / 'private.tex').write_text('preserve local manuscript')
        (self.root / '.credentials').write_text('do not archive')

    def snapshot(self, root=None):
        root = root or self.root
        return {str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob('*') if path.is_file() and '.git' not in path.parts}

    def prepare(self, profile='new', part_bytes=777):
        return archive.prepare(self.root, profile, self.bundle, part_bytes=part_bytes)

    def manifest(self):
        return json.loads((self.bundle / 'manifest.json').read_text())

    def test_real_split_archive_roundtrip_preserves_source_and_complete_snapshots(self):
        before = self.snapshot()
        self.prepare()
        manifest = self.manifest()
        archive.verify(self.bundle)
        self.assertEqual(manifest['source_commit'], self.commit)
        self.assertGreater(len(manifest['parts']), 1)
        self.assertTrue(all(part['size'] <= 777 for part in manifest['parts']))
        names = {item['path'] for item in manifest['members']}
        self.assertIn(f'runs/{CE}/training/nmse/checkpoint_step_00000.pt', names)
        self.assertIn(f'runs/{CE}/training/geometry/monitor/step_00000/summary.csv', names)
        self.assertIn(f'runs/{SOURCE}/training/ce/checkpoint_latest.pt', names)
        self.assertNotIn('overleaf/private.tex', names)
        self.assertNotIn('.credentials', names)
        destination = self.base / 'restored'
        destination.mkdir()
        archive.restore(self.bundle, destination)
        restored = self.snapshot(destination)
        self.assertEqual(set(restored), names)
        for name, data in restored.items():
            self.assertEqual(data, before[name])
        self.assertEqual(self.snapshot(), before)
        # Matching pre-existing files make retry safe and keep original data intact.
        archive.restore(self.bundle, destination)
        self.assertEqual(self.snapshot(destination), restored)
        self.assertFalse(any('.tmp' in path.name for path in destination.rglob('*')))

    def test_corrupt_part_is_rejected_before_restore_writes(self):
        self.prepare()
        part = self.bundle / self.manifest()['parts'][0]['name']
        raw = bytearray(part.read_bytes()); raw[-1] ^= 1; part.write_bytes(raw)
        destination = self.base / 'restored'; destination.mkdir()
        with self.assertRaises((ValueError, OSError)):
            archive.restore(self.bundle, destination)
        self.assertEqual(list(destination.iterdir()), [])

    def test_manifest_member_hash_tamper_is_rejected(self):
        self.prepare()
        manifest = self.manifest()
        manifest['members'][0]['sha256'] = '0' * 64
        write_json(self.bundle / 'manifest.json', manifest)
        with self.assertRaises((ValueError, OSError)):
            archive.verify(self.bundle)

    def test_conflicting_destination_preflight_preserves_every_file(self):
        self.prepare()
        name = self.manifest()['members'][-1]['path']
        destination = self.base / 'restored'; destination.mkdir()
        conflict = destination / name
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b'valuable existing checkpoint')
        before = self.snapshot(destination)
        with self.assertRaises((ValueError, OSError)):
            archive.restore(self.bundle, destination)
        self.assertEqual(self.snapshot(destination), before)

    def test_missing_source_dependency_refuses_completed_archive(self):
        record_path = self.new['nmse'] / 'manifest.json'
        record = json.loads(record_path.read_text())
        record['source_identity']['frozen_files'][0]['sha256'] = '0' * 64
        write_json(record_path, record)
        write_json(self.new['nmse'] / 'terminal_receipt.json', record)
        with self.assertRaises((ValueError, OSError)):
            self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_incomplete_study_and_missing_terminal_are_rejected(self):
        for mode in ('running', 'missing_terminal'):
            with self.subTest(mode=mode):
                if mode == 'running':
                    write_json(self.root / 'runs' / CE / 'summary.json', {'status': 'running', 'namespace': CE})
                else:
                    write_json(self.root / 'runs' / CE / 'summary.json', {'status': 'completed', 'namespace': CE})
                    (self.new['geometry'] / 'terminal_receipt.json').unlink()
                with self.assertRaises((ValueError, OSError)):
                    self.prepare()
                self.assertFalse(self.bundle.exists())

    def test_missing_ce_curve_checkpoint_and_changed_monitor_refuse_backup(self):
        snapshot = self.new['geometry'] / 'checkpoint_step_35000.pt'
        original = snapshot.read_bytes(); snapshot.unlink()
        with self.assertRaises((ValueError, OSError)):
            self.prepare()
        self.assertFalse(self.bundle.exists())
        snapshot.write_bytes(original)
        monitor = self.new['nmse'] / 'monitor' / 'step_20000' / 'summary.csv'
        monitor.write_bytes(b'changed curve diagnostics')
        with self.assertRaises((ValueError, OSError)):
            self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_old_terminal_requires_unchanged_progress_evidence(self):
        write_json(self.root / 'runs' / LOW / 'summary.json',
                   {'status': 'completed', 'namespace': LOW})
        for name in ('low_snr_control', 'low_snr_interference'):
            terminal(self.root, LOW, name, 10000)
        progress = self.root / 'runs' / LOW / 'training' / 'low_snr_control' / 'progress.jsonl'
        progress.write_bytes(b'changed optimizer-step trace')
        with self.assertRaises((ValueError, OSError)):
            self.prepare('old')
        self.assertFalse(self.bundle.exists())

    def test_source_symlink_is_rejected_and_existing_output_not_overwritten(self):
        link = self.new['nmse'] / 'unexpected.pt'
        link.symlink_to(self.root / '.credentials')
        with self.assertRaises((ValueError, OSError)):
            self.prepare()
        self.assertFalse(self.bundle.exists())
        link.unlink()
        self.bundle.mkdir()
        sentinel = self.bundle / 'preserve.txt'; sentinel.write_bytes(b'keep')
        with self.assertRaises((ValueError, OSError)):
            self.prepare()
        self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_symlink_destination_parent_cannot_redirect_restore(self):
        self.prepare()
        destination = self.base / 'restored'; destination.mkdir()
        outside = self.base / 'outside'; outside.mkdir()
        (destination / 'runs').symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            archive.restore(self.bundle, destination)
        self.assertEqual(list(outside.iterdir()), [])

    def rewrite_tar(self, mode):
        manifest = self.manifest()
        raw = b''.join((self.bundle / part['name']).read_bytes() for part in manifest['parts'])
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as stream:
            files = [(item.name, stream.extractfile(item).read()) for item in stream]
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:') as stream:
            for index, (name, data) in enumerate(files):
                item = tarfile.TarInfo(name)
                item.size = len(data)
                if index == 0 and mode == 'traversal':
                    item.name = 'runs/../../outside.pt'
                    manifest['members'][0]['path'] = item.name
                if index == 0 and mode in ('symlink', 'hardlink'):
                    item.type = tarfile.SYMTYPE if mode == 'symlink' else tarfile.LNKTYPE
                    item.linkname = '../../outside.pt'; item.size = 0
                    manifest['members'][0].update(size=0, sha256=digest(b''))
                    stream.addfile(item)
                else:
                    stream.addfile(item, io.BytesIO(data))
                if index == 0 and mode == 'duplicate':
                    stream.addfile(item, io.BytesIO(data))
        for part in manifest['parts']:
            (self.bundle / part['name']).unlink()
        raw = output.getvalue()
        name = 'models.tar.part0000'
        (self.bundle / name).write_bytes(raw)
        manifest['parts'] = [dict(name=name, size=len(raw), sha256=digest(raw))]
        manifest['total_bytes'] = sum(item['size'] for item in manifest['members'])
        write_json(self.bundle / 'manifest.json', manifest)

    def test_tar_traversal_duplicate_and_link_members_are_rejected(self):
        for mode in ('traversal', 'duplicate', 'symlink', 'hardlink'):
            with self.subTest(mode=mode):
                self.prepare()
                self.rewrite_tar(mode)
                destination = self.base / ('restore-' + mode); destination.mkdir()
                with self.assertRaises((ValueError, OSError)):
                    archive.restore(self.bundle, destination)
                self.assertEqual(list(destination.iterdir()), [])
                import shutil
                shutil.rmtree(self.bundle)

    def test_old_profile_keeps_both_terminals_and_excludes_failed_historical_root(self):
        write_json(self.root / 'runs' / LOW / 'summary.json', {'status': 'completed', 'namespace': LOW})
        for name in ('low_snr_control', 'low_snr_interference'):
            directory = terminal(self.root, LOW, name, 10000)
        failed = self.root / 'runs' / 'twc_v1_k8_icc_aligned_v1'
        failed.mkdir(); (failed / 'failed.pt').write_bytes(b'keep old failure')
        self.prepare('old')
        names = {item['path'] for item in self.manifest()['members']}
        self.assertIn(f'runs/{LOW}/training/low_snr_control/checkpoint_latest.pt', names)
        self.assertIn(f'runs/{LOW}/training/low_snr_interference/progress.jsonl', names)
        self.assertFalse(any('/twc_v1_k8_icc_aligned_v1/' in name for name in names))
        self.assertEqual((failed / 'failed.pt').read_bytes(), b'keep old failure')


if __name__ == '__main__':
    unittest.main()
