"""Small standard-library safety fixtures; no models, weights or experiments."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cleanup_twc_v1_k8 as cleanup


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=cleanup.ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'03_PRECODER';self.root.mkdir()
        for args in (('init','-q','-b','main'),('remote','add','origin',cleanup.ORIGIN)):
            subprocess.run(['git',*args],cwd=self.root,check=True,capture_output=True)
        (self.root/'.gitignore').write_text('__pycache__/\n*.pyc\n',encoding='utf-8')
        self.guard=patch.object(cleanup,'process_guard',return_value={'clear':True,'blockers':[]})
        self.guard.start();self.addCleanup(self.guard.stop)
        self.held=patch.object(cleanup,'held_on_windows',return_value=False)
        self.held.start();self.addCleanup(self.held.stop)

    def bytecode(self,relative='models/__pycache__/example.cpython-312.pyc'):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.parent.name=='__pycache__':source=Path(importlib.util.source_from_cache(str(path)))
        else:source=path.with_suffix('.py')
        source.write_text('# unexecuted source fixture\n',encoding='utf-8')
        path.write_bytes(importlib.util.MAGIC_NUMBER+b'\0'*12+b'cache fixture bytes')
        return path

    def test_default_preview_preserves_cache_and_writes_nonoverwriting_receipt(self):
        cache=self.bytecode();report=cleanup.run_cleanup(self.root)
        self.assertTrue(cache.is_file());self.assertEqual(report['deleted'],[])
        self.assertEqual(len(report['candidates']),1)
        self.assertEqual(json.loads(Path(report['report_path']).read_text())['mode'],'preview')
        with self.assertRaises(FileExistsError):
            cleanup.run_cleanup(self.root,apply=True,report_path=report['report_path'])
        self.assertTrue(cache.is_file())

    def test_apply_removes_only_untracked_regenerable_cache_and_retains_evidence(self):
        cache=self.bytecode();tracked=self.bytecode('tests/__pycache__/tracked.cpython-312.pyc')
        subprocess.run(['git','add','-f','tests/__pycache__/tracked.cpython-312.pyc'],cwd=self.root,check=True,capture_output=True)
        orphan=self.bytecode('configs/__pycache__/orphan.cpython-312.pyc')
        Path(importlib.util.source_from_cache(str(orphan))).unlink()
        protected={}
        for relative in ('runs/twc_v1_k8_ce/__pycache__/weights.pyc','models/__pycache__/manual.txt',
                         'models/.pytest_cache/checkpoint.pt','models/.pytest_cache/manifest.json',
                         'paper/figures/manual.png','cleanup_reports/earlier.json','locks/active.lock'):
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'protected evidence')
            protected[path]=path.read_bytes()
        report=cleanup.run_cleanup(self.root,apply=True)
        self.assertFalse(cache.exists());self.assertTrue(tracked.exists());self.assertTrue(orphan.exists())
        self.assertEqual(len(report['deleted']),1)
        self.assertTrue(report['protected_and_tracked_state_unchanged'])
        self.assertEqual(report['before']['tracked_status'],report['after']['tracked_status'])
        for path,value in protected.items():self.assertEqual(path.read_bytes(),value)

    def test_active_or_uncertain_process_conservatively_skips_without_failure(self):
        cache=self.bytecode()
        with patch.object(cleanup,'process_guard',return_value={'clear':False,'blockers':[{'reason':'unknown cwd'}]}):
            report=cleanup.run_cleanup(self.root,apply=True)
        self.assertTrue(cache.exists());self.assertEqual(report['logical_deleted_bytes'],0)
        self.assertEqual(report['status'],'completed');self.assertTrue(report['skipped'])

    def test_held_file_is_preserved(self):
        cache=self.bytecode()
        with patch.object(cleanup,'held_on_windows',return_value=True):report=cleanup.run_cleanup(self.root,apply=True)
        self.assertTrue(cache.exists());self.assertEqual(report['deleted'],[])
        self.assertIn('Open/held',report['skipped'][0]['reason'])

    def test_known_tool_cache_shape_is_required_and_manual_readme_is_retained(self):
        folder=self.root/'tests/.pytest_cache/v/cache';folder.mkdir(parents=True)
        nodeids=folder/'nodeids';nodeids.write_text('["test_example.py::case"]',encoding='utf-8')
        manual=folder.parent.parent/'README.md';manual.write_text('manual note: keep me',encoding='utf-8')
        invalid=folder/'lastfailed';invalid.write_text('not a generated JSON cache',encoding='utf-8')
        report=cleanup.run_cleanup(self.root,apply=True)
        self.assertFalse(nodeids.exists());self.assertTrue(manual.exists());self.assertTrue(invalid.exists())
        self.assertEqual(len(report['deleted']),1)

    def test_unknown_bytecode_and_tool_named_manual_files_are_preserved(self):
        invalid_flags=self.bytecode('models/__pycache__/badflags.cpython-312.pyc')
        invalid_flags.write_bytes(importlib.util.MAGIC_NUMBER+(2).to_bytes(4,'little')+b'\0'*12)
        wrong_magic=self.bytecode('models/__pycache__/badmagic.cpython-312.pyc')
        wrong_magic.write_bytes(b'XX\r\n'+b'\0'*20)
        files=[invalid_flags,wrong_magic]
        for relative,value in (
                ('models/.mypy_cache/example.data.json',b'[]'),
                ('models/.mypy_cache/example.meta.json',b'manual note'),
                ('models/.mypy_cache/cache.db',b'not an SQLite cache'),
                ('models/.mypy_cache/CACHEDIR.TAG',b'manual cache name'),
                ('models/.ruff_cache/deadbeef',b'manual hex-named file')):
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);files.append(path)
        contents={p:p.read_bytes() for p in files}
        report=cleanup.run_cleanup(self.root,apply=True)
        self.assertEqual(report['deleted'],[])
        for path,value in contents.items():self.assertEqual(path.read_bytes(),value)

    def test_protection_mismatch_writes_failed_receipt_before_raising(self):
        cache=self.bytecode();manual=self.root/'.gitignore';original=manual.read_bytes()
        before=cleanup.snapshot(self.root);after=json.loads(json.dumps(before))
        after['protected_sha256']['.gitignore']='0'*64
        report_path=self.root/'cleanup_reports/failure.json'
        with patch.object(cleanup,'snapshot',side_effect=[before,after]):
            with self.assertRaisesRegex(RuntimeError,'evidence:'):
                cleanup.run_cleanup(self.root,apply=True,report_path=report_path)
        report=json.loads(report_path.read_text(encoding='utf-8'))
        self.assertEqual(report['status'],'failed');self.assertFalse(report['protected_and_tracked_state_unchanged'])
        self.assertEqual(len(report['deleted']),1);self.assertFalse(cache.exists());self.assertEqual(manual.read_bytes(),original)

    def test_hard_link_with_retained_partner_claims_no_recovered_storage(self):
        cache=self.bytecode();partner=self.root/'retained-copy.bin'
        try:os.link(cache,partner)
        except (OSError,NotImplementedError) as error:self.skipTest('Real hard links unavailable: '+str(error))
        report=cleanup.run_cleanup(self.root,apply=True)
        self.assertFalse(cache.exists());self.assertTrue(partner.exists())
        self.assertEqual(report['reclaimable_allocation_estimate_bytes'],0)
        self.assertEqual(report['logical_deleted_bytes'],partner.stat().st_size)

    def test_symlink_source_root_is_skipped(self):
        outside=Path(self.temp.name)/'outside';outside.mkdir()
        (outside/'not_touched.pyc').write_bytes(importlib.util.MAGIC_NUMBER+b'\0'*20)
        try:(self.root/'utils').symlink_to(outside,target_is_directory=True)
        except (OSError,NotImplementedError) as error:self.skipTest('Real symlink creation unavailable: '+str(error))
        report=cleanup.run_cleanup(self.root,apply=True)
        self.assertTrue((outside/'not_touched.pyc').exists());self.assertEqual(report['deleted'],[])
        self.assertTrue(any('Link/reparse' in row['reason'] for row in report['skipped']))

    def test_outside_report_is_rejected_before_cache_deletion(self):
        cache=self.bytecode()
        with self.assertRaises(ValueError):cleanup.run_cleanup(self.root,apply=True,report_path=self.root/'wrong.json')
        self.assertTrue(cache.exists());self.assertFalse((self.root/'wrong.json').exists())

    def test_windows_process_inventory_does_not_assume_unknown_python_cwd_is_safe(self):
        # This is metadata parsing, not a fabricated numerical dependency.
        self.guard.stop()
        records=[dict(ProcessId=os.getpid(),Name='python.exe',CommandLine='self'),
                 dict(ProcessId=12345,Name='python.exe',CommandLine='python elsewhere.py')]
        with patch.object(cleanup.subprocess,'check_output',return_value=json.dumps(records)):
            result=cleanup.process_guard(self.root,platform_name='nt')
        self.assertFalse(result['clear']);self.assertEqual(len(result['blockers']),1)


if __name__=='__main__':unittest.main()
