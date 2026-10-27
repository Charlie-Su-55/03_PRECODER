"""Orchestration contracts without model imports, channels, or training."""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli_contracts import assert_lightweight_cli
import run_grouped_codec_batch as batch
import preflight_grouped_codec as preflight


class BatchContracts(unittest.TestCase):
    def test_stdlib_plans_and_no_writes(self):
        assert_lightweight_cli(self, "run_grouped_codec_batch", [["--dry-run"]],
            "assert len(plans[0]['training'])==10\n"
            "assert plans[0]['total_official_updates']==200000\n"
            "assert len(plans[0]['contrasts'])==7\n"
            "assert not plans[0]['cleanup'] and not plans[0]['historical_runs_retrained']")
        assert_lightweight_cli(self, "preflight_grouped_codec", [["--dry-run"]],
            "assert plans[0]['batch_size']==48\nassert plans[0]['counts_not_measured_locally']\n"
            "assert not plans[0]['historical_checkpoints_loaded']")

    def test_static_counts_and_complete_joint_execution_contract(self):
        for counts in preflight.STATIC_PARAMETERS.values():
            self.assertEqual(counts['full'], counts['ue'] + counts['bs'])
        paths = set(preflight.required_modules())
        for i in range(2):
            for part in ('self_attention', 'cross_attention', 'mlp'):
                self.assertIn(f'decoder.joint_layers.{i}.{part}', paths)
        self.assertIn('decoder.spatial_attention', paths)
        self.assertEqual(sum(p.startswith('decoder.codec.blocks.') for p in paths), 4)

    def test_smoke_included_and_completed_training_not_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = SimpleNamespace(SAVE_PATH=str(root/'ckpt/latest.pth'), BEST_PATH=str(root/'ckpt/best.pth'),
                SAVE_DIR=str(root/'ckpt'), LOG_DIR=str(root/'logs'),
                STATUS_PATH=str(root/'status.json'), CONFIG_PATH=str(root/'config.json'))
            commands = []
            with patch.object(batch, 'grouped_codec_config', return_value=cfg), \
                    patch.object(batch, '_child', side_effect=lambda folder, label, command: commands.append(command) or str(root/(label+'.log'))), \
                    patch.object(batch, 'read_json', return_value={'state':'paused'}), \
                    patch.object(batch, '_stage_child', return_value={'report':{}, '_evidence':[]}):
                batch._train_one(root, 'ordered_group2', 20261030)
            self.assertEqual(commands[0][-2:], ['--stop-after', '500'])
            self.assertEqual(commands[1][-2:], ['--resume', '--train'])
            Path(cfg.SAVE_DIR).mkdir()
            Path(cfg.SAVE_PATH).touch()
            with patch.object(batch, 'grouped_codec_config', return_value=cfg), \
                    patch.object(batch, '_child') as child, \
                    patch.object(batch, 'read_json', return_value={'state':'completed'}), \
                    patch.object(batch, '_stage_child', return_value={'report':{}, '_evidence':[]}):
                batch._train_one(root, 'ordered_group2', 20261030)
                child.assert_not_called()

    def test_unknown_output_and_changed_completed_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = SimpleNamespace(SAVE_PATH=str(root/'missing.pth'), SAVE_DIR=str(root), LOG_DIR=str(root/'logs'))
            with patch.object(batch, 'grouped_codec_config', return_value=cfg), patch.object(batch, '_child') as child:
                with self.assertRaises(RuntimeError):
                    batch._train_one(root, 'ordered_fdma', 20261030)
                child.assert_not_called()
            evidence = root/'evidence.json'
            evidence.write_text('{}', encoding='utf-8')
            summary = {'stages': {'done': {'status':'completed', 'result':{'ok':True},
                'evidence_sha256': {str(evidence):batch.sha256(evidence)}}}}
            result = batch._run_stage(summary, root, 'done', lambda: self.fail('Repeated completed work'))
            self.assertTrue(result['ok'])
            evidence.write_text('changed', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                batch._run_stage(summary, root, 'done', lambda: self.fail('Replaced evidence'))

    def test_failed_gpu_evidence_retained_for_failure_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def child(folder, label, command, **kwargs):
                if label.startswith('test_'):
                    return str(root/(label+'.log'))
                output = Path(command[command.index('--output-dir')+1])
                output.mkdir()
                (output/'report.json').write_text(json.dumps({'passed':False, 'error':'synthetic OOM'}), encoding='utf-8')
                raise RuntimeError('failed CUDA stage')
            with patch.object(batch, '_child', side_effect=child):
                with self.assertRaisesRegex(RuntimeError, 'failed CUDA stage'):
                    batch._preflight(root)
            files = [p for p in root.iterdir() if p.is_file() and p.suffix=='.json']
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text(encoding='utf-8'))['error'], 'synthetic OOM')

    def test_explicit_modes_only(self):
        for arguments in (['--resume'], ['--_stage','training_record']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                batch.main(arguments)
            self.assertEqual(e.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            preflight.main(['--run'])


if __name__ == '__main__':
    unittest.main()
