"""Real local Git/filesystem checks for bounded retired-entrypoint cleanup."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('obsolete_cleanup',
    Path(__file__).resolve().parents[1]/'tools/cleanup_03_obsolete.py')
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)/'03_PRECODER'
        self.root.mkdir()
        self.git('init', '--initial-branch=main')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('remote', 'add', 'origin', cleanup.ORIGIN)
        (self.root/'retained.py').write_text('retained source\n')
        self.git('add', 'retained.py')
        self.git('commit', '-m', 'retained')
        self.payload = {}
        for name in cleanup.RETIRED:
            self.payload[name] = subprocess.run(['git', 'show', cleanup.SOURCE_COMMIT+':'+name],
                cwd=cleanup.ROOT, capture_output=True, check=True).stdout
        self.guard = mock.patch.object(cleanup, 'active_processes', return_value=[])
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, capture_output=True,
                              check=True).stdout.decode().strip()

    def put(self, name, raw=None):
        path = self.root/name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(self.payload[name] if raw is None else raw)
        return path

    def snapshot(self):
        return self.git('rev-parse', 'HEAD'), (self.root/'.git/index').read_bytes()

    def test_default_plan_is_read_only_and_manifest_matches_historical_blobs(self):
        for name, expected in cleanup.RETIRED.items():
            self.assertEqual(hashlib.sha256(self.payload[name]).hexdigest(), expected)
            self.put(name)
        before = self.snapshot()
        result = cleanup.cleanup(self.root)
        self.assertEqual(before, self.snapshot())
        self.assertFalse((self.root/'.git/precoder_cleanup').exists())
        self.assertTrue(all(r['status']=='archive_candidate' for r in result['records']))
        self.assertTrue(all((self.root/name).is_file() for name in cleanup.RETIRED))

    def test_apply_archives_exact_orphans_and_preserves_dirty_work_and_index(self):
        for name in cleanup.RETIRED:
            self.put(name)
        (self.root/'retained.py').write_text('preserve local source edit\n')
        (self.root/'staged.py').write_text('preserve staged source\n')
        self.git('add', 'staged.py')
        (self.root/'untracked.py').write_text('preserve unknown work\n')
        before = self.snapshot()
        result = cleanup.cleanup(self.root, apply=True)
        self.assertEqual(before, self.snapshot())
        self.assertEqual((self.root/'retained.py').read_text(), 'preserve local source edit\n')
        self.assertTrue((self.root/'untracked.py').is_file())
        archive = self.root/result['archive']
        self.assertEqual(archive.stat().st_mode & 0o777, 0o700)
        receipt = json.loads((archive/'receipt.json').read_text())
        self.assertEqual(receipt['git_before'], receipt['git_after'])
        for name, expected in cleanup.RETIRED.items():
            self.assertFalse((self.root/name).exists())
            saved = archive/'files'/name
            self.assertEqual(cleanup.sha256(saved), expected)
            self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertTrue(all(r['status']=='archived' for r in receipt['records']))

    def test_tracked_staged_modified_and_missing_sources_are_preserved(self):
        names = list(cleanup.RETIRED)
        self.put(names[0]);self.git('add', names[0]);self.git('commit', '-m', 'still tracked')
        self.put(names[1]);self.git('add', names[1])
        self.put(names[2], b'user edited obsolete source\n')
        before = self.snapshot()
        result = cleanup.cleanup(self.root, apply=True)
        states = {r['path']:r['status'] for r in result['records']}
        self.assertEqual(states[names[0]], 'tracked_preserved')
        self.assertEqual(states[names[1]], 'staged_preserved')
        self.assertEqual(states[names[2]], 'modified_or_unknown_preserved')
        self.assertEqual(states[names[3]], 'absent')
        self.assertEqual(before, self.snapshot())
        self.assertIsNone(result['archive'])

    def test_links_and_hardlinks_are_preserved(self):
        names = list(cleanup.RETIRED)
        outside = self.root.parent/'outside.py';outside.write_bytes(self.payload[names[0]])
        (self.root/names[0]).symlink_to(outside)
        path = self.put(names[1]);os.link(path, self.root/'other-link.py')
        result = cleanup.cleanup(self.root, apply=True)
        states = {r['path']:r['status'] for r in result['records']}
        self.assertEqual(states[names[0]], 'unsafe_or_linked_preserved')
        self.assertEqual(states[names[1]], 'nonregular_or_hardlinked_preserved')
        self.assertTrue(outside.is_file())
        self.assertTrue(path.is_file())

    def test_linked_archive_parent_is_refused_without_touching_original(self):
        name = next(iter(cleanup.RETIRED));source = self.put(name)
        outside = self.root.parent/'outside';outside.mkdir()
        (self.root/'.git/precoder_cleanup').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            cleanup.cleanup(self.root, apply=True)
        self.assertEqual(source.read_bytes(), self.payload[name])
        self.assertEqual(list(outside.iterdir()), [])

    def test_copy_failure_preserves_existing_source(self):
        name = next(iter(cleanup.RETIRED));source = self.put(name)
        original_open = Path.open
        def failed(path, *args, **kwargs):
            if args and args[0]=='xb':
                raise OSError('simulated full backup device')
            return original_open(path, *args, **kwargs)
        with mock.patch.object(Path, 'open', failed):
            result = cleanup.cleanup(self.root, apply=True)
        row = next(r for r in result['records'] if r['path']==name)
        self.assertEqual(row['status'], 'archive_failed_original_preserved')
        self.assertEqual(source.read_bytes(), self.payload[name])

    def test_changed_after_inventory_is_preserved(self):
        name = next(iter(cleanup.RETIRED));source = self.put(name)
        row = next(r for r in cleanup.inventory(self.root) if r['path']==name)
        source.write_bytes(b'new work after inventory\n')
        target = self.root/'.git/archive_fixture';target.mkdir()
        with self.assertRaisesRegex(ValueError, 'changed'):
            cleanup.archive_one(self.root, row, target)
        self.assertEqual(source.read_bytes(), b'new work after inventory\n')

    def test_active_or_uncertain_process_refuses_apply(self):
        name = next(iter(cleanup.RETIRED));source = self.put(name)
        with mock.patch.object(cleanup, 'active_processes', return_value=[dict(pid=123)]):
            with self.assertRaisesRegex(ValueError, 'Active'):
                cleanup.cleanup(self.root, apply=True)
        self.assertEqual(source.read_bytes(), self.payload[name])
        self.assertFalse((self.root/'.git/precoder_cleanup').exists())

    def test_real_same_repository_python_process_is_detected(self):
        self.guard.stop()
        child = subprocess.Popen([os.sys.executable, '-B', '-c', 'import time; time.sleep(30)'],
                                 cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.assertIn(child.pid, {r.get('pid') for r in cleanup.active_processes(self.root)})
        finally:
            child.terminate();child.wait(timeout=5)
        self.guard.start()

    def test_unreadable_executable_of_known_non_python_is_not_a_blocker(self):
        self.guard.stop()
        child = subprocess.Popen(['/bin/sleep', '30'], cwd=self.root)
        resolve = Path.resolve
        def restricted(path, *args, **kwargs):
            if path.name=='exe' and path.parent.name==str(child.pid):
                raise PermissionError('simulated nondumpable non-Python process')
            return resolve(path, *args, **kwargs)
        try:
            with mock.patch.object(Path, 'resolve', restricted):
                self.assertNotIn(child.pid, {r.get('pid') for r in cleanup.active_processes(self.root)})
        finally:
            child.terminate();child.wait(timeout=5)
        self.guard.start()

    def test_other_origin_or_non_main_apply_is_refused(self):
        self.git('remote', 'set-url', 'origin', 'https://example.invalid/another.git')
        with self.assertRaisesRegex(ValueError, 'Official'):
            cleanup.cleanup(self.root)
        self.git('remote', 'set-url', 'origin', cleanup.ORIGIN)
        self.git('switch', '-c', 'local-work')
        with self.assertRaisesRegex(ValueError, 'main'):
            cleanup.cleanup(self.root, apply=True)

    def test_no_arbitrary_paths_or_outputs_are_authorized(self):
        self.assertEqual(len(cleanup.RETIRED), 5)
        self.assertTrue(all(name.startswith('old/') or name=='diagnose_cross_cdl_condition.py'
                            for name in cleanup.RETIRED))
        for name in ('../outside.py', '/etc/passwd', 'old/../paper/file.py'):
            with self.assertRaises(ValueError):
                cleanup.safe_child(self.root, name)
        from configs import twc_v1_k8_ul_adapt_config as cfg
        self.assertFalse(set(cleanup.RETIRED).intersection(cfg.CORE_FILES))

    def test_historical_manifest_only_records_old_paths_as_worktree_changes(self):
        name = 'paper/data/k4_pure_as_lite1024/formal_20261002T124602_541062Z/manifest.json'
        pinned = subprocess.run(['git', 'show', cleanup.SOURCE_COMMIT+':'+name],
            cwd=cleanup.ROOT, capture_output=True, check=True).stdout
        self.assertEqual((cleanup.ROOT/name).read_bytes(), pinned)
        runtime = json.loads(pinned)['runtime']
        old_paths = {path for path in cleanup.RETIRED if path.startswith('old/')}
        self.assertTrue(old_paths.issubset(runtime['tracked_changes']))
        self.assertFalse(set(cleanup.RETIRED).intersection(runtime['source_sha256']))
        historical = self.root/name
        historical.parent.mkdir(parents=True)
        historical.write_bytes(pinned)
        for retired in cleanup.RETIRED:
            self.put(retired)
        cleanup.cleanup(self.root, apply=True)
        self.assertEqual(historical.read_bytes(), pinned)


if __name__ == '__main__':
    unittest.main()
