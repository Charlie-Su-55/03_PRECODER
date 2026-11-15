"""Controller checks: explicit actions, immutable evidence and stage budgets."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_twc_v1_k8_hybrid_direction as runner


class MetadataTests(unittest.TestCase):
    def test_dry_run_without_site_packages_does_not_load_weights_or_write_outputs(self):
        script = Path(runner.__file__).resolve()
        output = script.parent/'runs'/'twc_v1_k8_hybrid_direction_v1'
        before = list(output.rglob('*')) if output.exists() else []
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, '-B', '-S', str(script), '--dry-run'],
                cwd=temporary, capture_output=True, text=True, check=True)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        plan = json.loads(result.stdout)
        self.assertEqual(plan['Hybrid_new_updates'], 10000)
        self.assertEqual(plan['FDMA_new_updates'], 0)
        self.assertFalse(plan['weights_read']); self.assertFalse(plan['writes'])
        self.assertEqual(tuple(plan['stages']), runner.stage_names())
        self.assertEqual(list(output.rglob('*')) if output.exists() else [], before)

    def test_resume_without_run_is_rejected(self):
        with self.assertRaises(SystemExit) as context:
            runner.main(['--resume'])
        self.assertEqual(context.exception.code, 2)

    def test_no_fdma_training_stage_or_extra_100_updates(self):
        names = runner.stage_names()
        self.assertNotIn('fdma_10000', names)
        self.assertEqual([name for name in names if name.startswith('hybrid_')],
            ['hybrid_100', 'hybrid_10000'])

    def test_production_trace_cannot_relabel_a_different_data_protocol(self):
        import hybrid_direction_runtime as runtime
        import train_twc_v1_k8_hybrid_direction as trainer
        protocol = runtime.make_data('cuda:0').protocol
        trace = trainer.initial_training_trace(protocol)
        trainer._trace_check(trace, protocol, 0, True)
        other = dict(protocol, unrelated_namespace='not the registered data stream')
        with self.assertRaisesRegex(ValueError, 'data protocol changed'):
            trainer._trace_check(trainer.initial_training_trace(other), other, 0, True)


class StageEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root/'summary.json'
        self.file = self.root/'immutable.json'; self.file.write_text('source unchanged\n')
        import hashlib
        def inside(value):
            value = Path(value)
            if value.is_symlink() or not value.resolve().is_relative_to(self.root.resolve()):
                raise ValueError('owned unlinked path required')
            return value.resolve()
        def atomic(path, value):
            inside(path).write_text(json.dumps(value))
        self.runtime = SimpleNamespace(inside=inside, atomic_json=atomic,
            sha256=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
        for patch in (mock.patch.object(runner, '_runtime', return_value=self.runtime),
                mock.patch.object(runner, 'summary_path', return_value=self.path)):
            patch.start(); self.addCleanup(patch.stop)
        self.summary = dict(stages={})

    def test_reused_completed_stage_checks_bytes_and_does_not_repeat_action(self):
        action = mock.Mock(return_value=({'accepted': True}, [self.file]))
        record = runner.stage(self.summary, 'source_preflight', action)
        self.assertEqual(record['status'], 'completed')
        runner.stage(self.summary, 'source_preflight', action)
        self.assertEqual(action.call_count, 1)
        self.file.write_text('changed source evidence\n')
        with self.assertRaises(ValueError): runner.stage(self.summary, 'source_preflight', action)
        self.assertEqual(action.call_count, 1)

    def test_failed_attempt_is_retained_before_successful_explicit_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            runner.stage(self.summary, 'startup_policy', mock.Mock(side_effect=RuntimeError('busy')))
        self.assertEqual(self.summary['status'], 'failed')
        record = runner.stage(self.summary, 'startup_policy',
            lambda _: ({'admitted': True}, [self.file]), reuse=False)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(record['attempts']), 1)
        self.assertEqual(record['attempts'][0]['status'], 'failed')

    def test_unlinked_owned_nonempty_evidence_is_required(self):
        with self.assertRaises(ValueError): runner.evidence([])
        link = self.root/'link.json'; link.symlink_to(self.file)
        with self.assertRaises(ValueError): runner.evidence([link])
        with self.assertRaises(ValueError): runner.evidence([self.root.parent/'other.json'])


class CompletedReadoutTests(unittest.TestCase):
    def test_completed_readout_consumes_actual_long_result_and_comparison_schema(self):
        from configs import twc_v1_k8_hybrid_direction_config as cfg
        import twc_v1_k8_hybrid_direction_results as results
        from evaluate_twc_v1_k8_hybrid_direction import _csv
        rows = []
        for vi, variant in enumerate(cfg.EVAL_VARIANTS):
            for gamma in cfg.GAMMAS:
                for sample in range(8):
                    rows.append(dict(variant=variant, arm=results.physical_arm(variant), role='final',
                        gamma_db=float(gamma), sample_index=sample, batch_index=sample//4,
                        within_batch=sample % 4,
                        **{metric: 10.+vi+gamma/10+sample/100 for metric in results.REQUIRED_METRICS}))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); report = root/'report'; report.mkdir()
            _csv(report/'summary.csv', results.summarize_rows(rows, samples=8))
            _csv(report/'comparisons.csv', results.paired_comparisons(rows, samples=8))
            summary = dict(status='completed', stage='directory_export', updated='fixture',
                code={'commit': 'a'*40}, report_directory=str(report))
            with mock.patch.object(cfg, 'OUTPUT_ROOT', root):
                text = runner.compact_summary(summary)
            self.assertIn('hybrid_direction gamma=25.0 rate=', text)
            self.assertIn('hybrid_direction_minus_fdma_frozen all_six: +2.000000', text)
            self.assertIn('hybrid_direction_minus_hybrid_source low_0_5_10: +1.000000', text)
            self.assertIn('--publish', text)


if __name__ == '__main__':
    unittest.main()
