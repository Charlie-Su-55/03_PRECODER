"""Controller boundaries, accounting and side-effect-free inspection."""
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs import twc_v1_k8_ce_refine_config as cfg
import ce_refine_runtime as runtime
import run_twc_v1_k8_ce_refine as queue


class MetadataAndCLI(unittest.TestCase):
    def test_declared_stages_cover_only_ce_trajectories(self):
        names = queue.stage_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertIn('nmse_60000', names)
        self.assertIn('geometry_60000', names)
        self.assertNotIn('fdma_100', names)
        self.assertNotIn('codec_train', names)
        self.assertEqual(cfg.counters(cfg.UPDATES)['CE_total_updates'], 65000)
        self.assertEqual(cfg.counters(cfg.UPDATES)['UE_updates'], 0)
        self.assertEqual(cfg.counters(cfg.UPDATES)['BS_updates'], 0)

    def test_dry_run_works_without_site_packages_and_does_not_read_weights(self):
        result = subprocess.run([sys.executable, '-B', '-S',
            str(cfg.ROOT/'run_twc_v1_k8_ce_refine.py'), '--dry-run'],
            cwd=cfg.ROOT, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertIs(value['writes'], False)
        self.assertIs(value['weights_read'], False)
        self.assertEqual(value['total_new_CE_updates'], 120000)
        self.assertNotIn('torch', result.stderr)

    def test_resume_requires_explicit_run(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                queue.main(['--summary', '--resume'])

    def test_not_started_summary_does_not_make_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'never_created'
            with patch.object(cfg, 'OUTPUT_ROOT', path), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(queue.main(['--summary']), 0)
            self.assertIn('not started', out.getvalue())
            self.assertFalse(path.exists())

    def test_summary_reports_real_boundaries_and_future_ce_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'training'/'nmse'/'manifest.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(step=100, seconds_per_update=.14,
                ETA_seconds=8386., updated='latest-checkpoint-time')))
            with patch.object(cfg, 'OUTPUT_ROOT', root):
                text = queue.compact_summary(dict(status='running', stage='nmse_60000',
                    updated='old-stage-time', code=dict(commit='a'*40)))
            self.assertIn('100/60000', text)
            self.assertIn('total_CE=5100', text)
            self.assertIn('latest-checkpoint-time', text)
            self.assertIn('geometry: not started', text)
            self.assertIn('16786s', text)
            self.assertIn('evaluation excluded', text)


class OwnedController(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'study'
        self.patch = patch.object(cfg, 'OUTPUT_ROOT', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def summary(self):
        return dict(namespace=cfg.NAMESPACE, status='running', stage='first', stages={},
            core=dict(test='source-boundary'), code=dict(commit='a'*40), recipe=cfg.recipe(),
            recipe_fingerprint=cfg.fingerprint(), source_pins=queue.source_pins())

    def test_completed_stage_reuse_checks_bytes_before_skipping(self):
        summary = self.summary()
        output = self.root/'evidence.json'
        def work(_):
            runtime.atomic_json(output, dict(actual='value'), exclusive=True)
            return dict(done=True), [output]
        queue.stage(summary, 'first', work)
        queue.stage(summary, 'first', lambda _: self.fail('completed stage reran'))
        output.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            queue.stage(summary, 'first', lambda _: self.fail('tamper was accepted'))

    def test_failure_records_evidence_without_removing_inputs(self):
        summary = self.summary()
        sentinel = self.root/'keep.txt'
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text('previous checkpoint placeholder')
        def fail(_):
            raise RuntimeError('actual-child-error')
        with self.assertRaisesRegex(RuntimeError, 'actual-child-error'):
            queue.stage(summary, 'first', fail)
        saved = queue.load_summary()
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['stages']['first']['status'], 'failed')
        self.assertIn('actual-child-error', saved['error'])
        self.assertEqual(sentinel.read_text(), 'previous checkpoint placeholder')

    def test_registration_rejects_changed_source_and_budget(self):
        summary = self.summary()
        runtime.atomic_json(self.root/'preregistration.json', summary, exclusive=True)
        summary['preregistration_sha256'] = runtime.sha256(self.root/'preregistration.json')
        with patch.object(runtime, 'guard_core'):
            queue.verify_registration(summary)
            changed = copy.deepcopy(summary)
            changed['source_pins']['updates'] = 4000
            with self.assertRaisesRegex(ValueError, 'registered identity'):
                queue.verify_registration(changed)
        registered = self.root/'preregistration.json'
        registered.write_text(registered.read_text()+' ')
        with patch.object(runtime, 'guard_core'), self.assertRaisesRegex(ValueError, 'registration changed'):
            queue.verify_registration(summary)

    def test_evidence_rejects_symlink_even_inside_namespace(self):
        self.root.mkdir(parents=True)
        target = self.root/'real.txt'
        target.write_text('saved')
        link = self.root/'linked.txt'
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            queue.evidence([link])


class Pairing(unittest.TestCase):
    @staticmethod
    def records():
        base = dict(production_evidence=True, step=cfg.UPDATES, counters=cfg.counters(cfg.UPDATES),
            source_identity=dict(checkpoint='same immutable CE5000'), data_protocol=dict(data_id='same'),
            training_trace=dict(sha256='b'*64, microbatches=cfg.UPDATES*cfg.ACCUMULATION))
        return {name: copy.deepcopy(base) for name in cfg.TRAJECTORIES}

    def test_pairing_requires_actual_matching_input_trace_and_source(self):
        records = self.records()
        self.assertTrue(queue.check_training_pairing(records)['actual_input_trace_matched'])
        for field in ('source_identity', 'data_protocol', 'training_trace'):
            changed = self.records()
            changed['geometry'][field] = dict(changed=True)
            with self.assertRaisesRegex(ValueError, field):
                queue.check_training_pairing(changed)

    def test_fixture_and_partial_terminal_cannot_be_promoted(self):
        for field, value in (('production_evidence', False), ('step', 5000), ('counters', {})):
            changed = self.records()
            changed['geometry'][field] = value
            with self.assertRaisesRegex(ValueError, 'complete CE-only'):
                queue.check_training_pairing(changed)


if __name__ == '__main__':
    unittest.main()
