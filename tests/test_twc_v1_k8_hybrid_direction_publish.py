"""Real local Git checks for explicit, scalar-only result publication."""
from contextlib import redirect_stdout
import hashlib
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import publish_twc_v1_k8_hybrid_direction as pub


class PublicationTests(unittest.TestCase):
    @staticmethod
    def actual_result_payloads():
        """Use the real three-view result producer and all six gamma nodes."""
        import twc_v1_k8_hybrid_direction_results as results
        from evaluate_twc_v1_k8_hybrid_direction import _csv
        rows = []
        cfg = results._config()
        for vi, variant in enumerate(cfg.EVAL_VARIANTS):
            for gamma in cfg.GAMMAS:
                for sample in range(8):
                    batch, within = divmod(sample, 4)
                    row = dict(variant=variant, arm=results.physical_arm(variant), role='final',
                        gamma_db=float(gamma), sample_index=sample, batch_index=batch,
                        within_batch=within, sum_rate_bps_hz=30+.03*gamma+.2*vi+.01*sample,
                        dl_nmse_linear=.005+.0001*sample, soft_nmse_linear=.01+.001*vi,
                        desired_signal_power_mean=.8+.01*within,
                        interference_power_mean=.01+.001*batch,
                        direction_error=.002+.0001*vi, signed_term=-.01*sample)
                    if variant != 'fdma_frozen': row['shared_soft_error'] = .03+.001*vi
                    rows.append(row)
        results.validate_rows(rows, samples=8)
        summaries = results.summarize_rows(rows, samples=8)
        comparisons = results.paired_comparisons(rows, samples=8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            for name, table in (('samples.csv.gz', rows), ('summary.csv', summaries),
                                ('comparisons.csv', comparisons)):
                _csv(path/name, table)
            return {name: (path/name).read_bytes() for name in
                ('samples.csv.gz', 'summary.csv', 'comparisons.csv')}

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
        states['CE_table'] = {'ce_errors': 'e'*64, 'ce_calibrated': 'c'*64}
        provenance = dict(namespace=pub.NAMESPACE,
            evaluation_updates=dict(optimizer=0, UE=0, BS=0, CE=0, EMA=0), test_used=False,
            experiment_code_commit='9'*40,
            contains_model_weights=False, contains_channel_tensors=False,
            production_evidence=True, FDMA_new_training_updates=0,
            Hybrid_new_training_updates=10000, budgets_equal=False,
            single_factor_loss_attribution=False,
            source_core_sha256='a'*64, preregistration_sha256='b'*64,
            recipe_fingerprint='c'*64, evaluation_manifest_sha256='e'*64,
            source_records={variant: {'checkpoint_sha256': 'c'*64}
                for variant in ('fdma_frozen', 'hybrid_source', 'hybrid_direction')},
            frozen_signatures={variant: dict(before=states, after=states)
                for variant in ('fdma_frozen', 'hybrid_source', 'hybrid_direction')})
        self.payloads = self.actual_result_payloads()
        self.payloads.update({'README.md': pub.README_TEXT.encode(),
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

    def test_actual_result_tables_with_text_contrast_endpoints_export_and_validate(self):
        import csv
        import twc_v1_k8_hybrid_direction_results as results
        comparisons = list(csv.DictReader(io.StringIO(self.payloads['comparisons.csv'].decode())))
        self.assertEqual({row['comparison'] for row in comparisons},
            {item[0] for item in results.CONTRASTS})
        self.assertTrue(all(row['positive'] in results._config().EVAL_VARIANTS and
            row['negative'] in results._config().EVAL_VARIANTS for row in comparisons))
        for name in ('samples.csv.gz', 'summary.csv', 'comparisons.csv'):
            pub._validate_payload(name, self.payloads[name])
        pub.create_export(self.report, self.payloads)
        pub.verify_export_directory(self.report)
        actual_rows = results.read_raw_csv(self.report/'samples.csv.gz')
        self.assertEqual(len(actual_rows), 3*6*8)
        results.validate_rows(actual_rows, samples=8)
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
        changed['frozen_signatures']['fdma_frozen']['after'] = {'codec': 'f'*64}
        with self.assertRaises(ValueError):
            pub.create_export(self.report, dict(self.payloads,
                **{'provenance.json': json.dumps(changed).encode()}))
        changed['frozen_signatures']['fdma_frozen']['after'] = changed['frozen_signatures']['fdma_frozen']['before']
        changed['evaluation_updates']['optimizer'] = 1
        with self.assertRaises(ValueError):
            pub.create_export(self.report, dict(self.payloads,
                **{'provenance.json': json.dumps(changed).encode()}))

    def test_modified_inventory_bytes_and_symlinks_stop_before_publication(self):
        pub.create_export(self.report, self.payloads)
        extra = self.report/'checkpoint.pt'
        extra.write_bytes(b'forbidden')
        with self.assertRaises(ValueError):
            pub.verify_export_directory(self.report)
        extra.unlink()
        item = self.report/'summary.csv'
        original = item.read_bytes()
        item.write_bytes(b'arm,rate\nfdma,99\n')
        with self.assertRaises(ValueError):
            pub.verify_export_directory(self.report)
        item.unlink()
        target = self.root/'external.csv'
        target.write_bytes(original)
        item.symlink_to(target)
        with self.assertRaises(ValueError):
            pub.verify_export_directory(self.report)

    def test_frozen_state_signature_and_integer_evaluation_updates_are_required(self):
        changed = json.loads(self.payloads['provenance.json'])
        changed['frozen_signatures']['fdma_frozen']['before'] = dict(changed['frozen_signatures']['fdma_frozen']['before'], encoder='f'*64)
        with self.assertRaises(ValueError):
            pub.create_export(self.report, dict(self.payloads,
                **{'provenance.json': json.dumps(changed).encode()}))
        changed = json.loads(self.payloads['provenance.json'])
        changed['evaluation_updates']['UE'] = False
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
        self.assertEqual(pub.NAMESPACE, pub.verify_export_directory(self.report)['namespace'])
        self.preserved()

    def test_completed_run_binding_is_checked_before_network(self):
        pub.create_export(self.report, self.payloads)
        with mock.patch.object(pub, '_completed', side_effect=ValueError('incomplete batch')):
            with self.assertRaises(ValueError):
                pub.publish(self.report)
        self.assertEqual('', self.git('ls-remote', '--heads', 'origin'))
        self.preserved()

    def test_scalar_export_redacts_parent_paths_and_default_publish_selects_completed_report(self):
        cfg = pub._config()
        # Keep the unit fixture's declared final budget consistent with its
        # two real result batches; no production metadata is relaxed.
        for patch in (mock.patch.object(cfg, 'FINAL_BATCHES', 2),
                      mock.patch.dict(cfg.ROLE_RANGES, {'final': [4096, 4097]})):
            patch.start()
            self.addCleanup(patch.stop)
        base = self.root/'runs'/pub.NAMESPACE
        evaluation = base/'evaluation'/'attempt_111111111111'
        evaluation.mkdir(parents=True)
        for name in ('samples.csv.gz', 'summary.csv', 'comparisons.csv', 'input_fingerprints.json'):
            (evaluation/name).write_bytes(self.payloads[name])
        source = {variant: dict(arm='fdma' if variant == 'fdma_frozen' else 'hybrid_g4',
            state_sha256='d'*64, base_ema_updates=30000, production_evidence=True,
            frozen_files=[dict(path='/home/private/checkpoint.pt', sha256='c'*64)],
            parent_record=dict(step=10000, checkpoint_sha256='c'*64,
                manifest_sha256='e'*64, path='/home/private/checkpoint.pt'))
            for variant in cfg.EVAL_VARIANTS}
        provenance = json.loads(self.payloads['provenance.json'])
        manifest = dict(status='passed', production_evidence=True,
            core=dict(schema='fixture', sha256={'science.py': 'f'*64}),
            recipe_fingerprint=cfg.fingerprint(), source_records=source,
            frozen_signatures=provenance['frozen_signatures'], data_protocol={'fixture': True},
            files={name: pub._sha(evaluation/name) for name in
                ('samples.csv.gz', 'summary.csv', 'comparisons.csv', 'input_fingerprints.json')},
            all_models_verified_before_data=True, paired_inputs_verified=True,
            samples_per_condition=8, batch_size=4, n_batches=2,
            gamma_db=list(cfg.GAMMAS), variants=list(cfg.EVAL_VARIANTS))
        (evaluation/'manifest.json').write_text(json.dumps(manifest))
        prereg = base/'preregistration.json'
        prereg.write_text(json.dumps({'private_output': str(base)}))
        summary = dict(namespace=pub.NAMESPACE, core=manifest['core'], recipe=cfg.recipe(),
            code={'commit': '9'*40},
            recipe_fingerprint=cfg.fingerprint(), preregistration_sha256=pub._sha(prereg))
        summary_path = base/'summary.json'
        summary_path.write_text(json.dumps(summary))
        evaluator = SimpleNamespace(verify_manifest=lambda path: manifest)
        with mock.patch.dict(sys.modules, {'evaluate_twc_v1_k8_hybrid_direction': evaluator}):
            report = pub.export_directory(evaluation, summary_path)
        exported = json.loads((report/'provenance.json').read_text())
        self.assertEqual(exported['Hybrid_new_training_updates'], 10000)
        self.assertEqual(exported['FDMA_new_training_updates'], 0)
        self.assertFalse(exported['budgets_equal'])
        self.assertNotIn('/home/', (report/'provenance.json').read_text())
        self.assertEqual(exported['source_records']['hybrid_source']['base_ema_updates'], 30000)
        self.assertEqual(exported['source_records']['hybrid_source']['parent_record']['checkpoint_sha256'], 'c'*64)
        summary.update(status='completed', report_directory=str(report))
        summary_path.write_text(json.dumps(summary))
        with redirect_stdout(io.StringIO()):
            receipt = pub.publish()
        self.assertEqual('uploaded', receipt['status'])
        self.preserved()


if __name__ == '__main__':
    unittest.main()
