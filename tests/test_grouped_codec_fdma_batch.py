"""Light controller/preflight contracts; no model, channel or training runs."""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from cli_contracts import assert_lightweight_cli
from configs.grouped_codec_fdma_train_config import fdma_config, validate_config
from as_receiver_batch_support import save_json, sha256
from run_as_receiver_batch import _run_stage


class PlanningTests(unittest.TestCase):
    def test_batch_and_preflight_plans_are_stdlib_readonly(self):
        assert_lightweight_cli(self,'run_grouped_codec_fdma_batch',[['--dry-run']],
            "assert plans[0]['total_new_official_updates']==40000")
        assert_lightweight_cli(self,'preflight_grouped_codec_fdma',[['--dry-run']],
            "assert plans[0]['counts_measured'] is False")

    def test_runtime_config_overrides_not_silently_accepted(self):
        for field,value in (('LR',2e-5),('K_USERS',8),('D_A_MAX_SHARED',64)):
            cfg=fdma_config(20261030)
            setattr(cfg,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_config(cfg)

    def test_resume_requires_explicit_run(self):
        import run_grouped_codec_fdma_batch as batch
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
            batch.main(['--resume'])
        self.assertEqual(error.exception.code,2)


class EvidenceTests(unittest.TestCase):
    def test_completed_stage_is_skipped_only_with_unchanged_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary); evidence=folder/'proof.json'
            save_json(evidence,{'synthetic':True})
            summary={'stages':{}}; calls=[]
            def action():
                calls.append('executed');return {'ok':True,'_evidence':[evidence]}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_run_stage(summary,folder,'F',action),{'ok':True})
                self.assertEqual(_run_stage(summary,folder,'F',action),{'ok':True})
            self.assertEqual(calls,['executed'])
            save_json(evidence,{'synthetic':False},replace=True)
            with self.assertRaises(RuntimeError):_run_stage(summary,folder,'F',action)
            self.assertEqual(calls,['executed'])

    def test_stage_failure_preserves_failure_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary); summary={'stages':{}}
            def failing():raise ValueError('inconsistent old evidence')
            with self.assertRaises(ValueError):_run_stage(summary,folder,'old_evidence',failing)
            saved=json.loads((folder/'batch_summary.json').read_text())
            self.assertEqual(saved['stages']['old_evidence']['status'],'failed')
            self.assertIn('inconsistent old evidence',saved['stages']['old_evidence']['error'])

    def test_missing_old_evidence_cannot_select_another_batch(self):
        from evaluate_grouped_codec_fdma import verify_old_batch,OLD_BATCH_RELATIVE
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(FileNotFoundError):verify_old_batch(root=root)
            alternate=root/'another_batch_summary.json'
            save_json(alternate,{'status':'completed'})
            with self.assertRaises(ValueError):verify_old_batch(alternate,root=root)
            self.assertFalse((root/OLD_BATCH_RELATIVE).exists())

    def test_small_package_excludes_weights_raw_and_logs(self):
        import run_grouped_codec_fdma_batch as batch
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            parent=Path(temporary); folder=parent/'batch_test'; folder.mkdir()
            output=folder/'export';output.mkdir()
            files={}
            for name in ('summary.csv','manifest.json','notes.md','preview.pdf','preview.png',
                         'samples.csv.gz','weights.pth','terminal.log','small_results_bundle.zip'):
                path=output/name;path.write_bytes(b'synthetic package fixture')
                files[str(path)]=sha256(path)
            summary={'status':'failed','error':'synthetic failure','stages':{},'results':{'files':files}}
            save_json(folder/'batch_summary.json',summary)
            with patch.object(batch,'BATCH_ROOT',parent):batch.package_small(folder,summary)
            package=Path(summary['result_package'])
            self.assertTrue(package.is_file())
            with zipfile.ZipFile(package) as archive:
                names=archive.namelist()
                self.assertTrue(all(Path(n).suffix in ('.csv','.json','.md','.pdf','.png') for n in names))
                self.assertIn('batch_summary.json',names)
                self.assertTrue(any(n.endswith('summary.csv') for n in names))
                self.assertFalse(any(n.endswith(('.pth','.gz','.zip','.log')) for n in names))
            self.assertEqual(summary['result_package_sha256'],sha256(package))
            self.assertTrue((output/'weights.pth').exists())


class OrchestrationTests(unittest.TestCase):
    def test_only_F_500_then_resume_20k_and_completed_run_not_retrained(self):
        import run_grouped_codec_fdma_batch as batch
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary); config=fdma_config(20261030,folder/'F')
            calls=[]
            def child(parent,label,command,**unused):
                calls.append(command)
                Path(config.SAVE_DIR).mkdir(parents=True,exist_ok=True)
                Path(config.LOG_DIR).mkdir(parents=True,exist_ok=True)
                Path(config.SAVE_PATH).write_bytes(b'not a real checkpoint')
                state='paused' if '--stop-after' in command else 'completed'
                save_json(config.STATUS_PATH,{'state':state},replace=Path(config.STATUS_PATH).exists())
                return 'synthetic.log'
            record={'report':{'completed_updates':20000},'_evidence':[]}
            with patch.object(batch,'fdma_config',return_value=config),patch.object(batch,'_child',side_effect=child),\
                    patch.object(batch,'_stage_child',side_effect=lambda *a,**k:{'report':dict(record['report']),'_evidence':[]}):
                batch._train_one(folder,20261030)
                batch._train_one(folder,20261030)
            self.assertEqual(len(calls),2)
            self.assertTrue(all('train_grouped_codec_fdma.py' in call for call in calls))
            self.assertEqual(calls[0][-2:],['--stop-after','500'])
            self.assertEqual(calls[1][-2:],['--resume','--train'])

    def test_existing_F_directory_without_latest_is_preserved(self):
        import run_grouped_codec_fdma_batch as batch
        with tempfile.TemporaryDirectory() as temporary:
            cfg=fdma_config(20261030,Path(temporary)/'F')
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with patch.object(batch,'fdma_config',return_value=cfg),patch.object(batch,'_child') as child:
                with self.assertRaises(RuntimeError):batch._train_one(Path(temporary),20261030)
                child.assert_not_called()
            self.assertTrue(Path(cfg.SAVE_DIR).is_dir())

    def test_progress_reads_small_records_not_checkpoint_contents(self):
        import run_grouped_codec_fdma_batch as batch
        with tempfile.TemporaryDirectory() as temporary:
            cfg=fdma_config(20261030,Path(temporary)/'F')
            Path(cfg.SAVE_DIR).mkdir(parents=True);Path(cfg.LOG_DIR).mkdir(parents=True)
            save_json(cfg.STATUS_PATH,{'state':'running','step':500,'experiment':cfg.EXP_NAME})
            Path(cfg.METRICS_PATH).write_text(json.dumps({'event':'validation','step':500,
                'selection_metric':12.,'selection_metric_name':'mean_all'})+'\n'+json.dumps({'event':'train_gamma','step':600})+'\n',encoding='utf-8')
            # Invalid bytes would fail any checkpoint deserialization; a status read must not care.
            Path(cfg.SAVE_PATH).write_bytes(b'not checkpoint data')
            result=batch.training_progress(20261030,cfg=cfg)
            self.assertFalse(result['checkpoint_verified'])
            self.assertIn('600',json.dumps(result))
            self.assertNotIn('torch',sys.modules if sys.flags.no_site else {})


if __name__=='__main__':unittest.main()
