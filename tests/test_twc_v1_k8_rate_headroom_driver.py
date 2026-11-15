"""Frozen CPU integration and complete-MU statistics, never real checkpoints."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

import diagnose_twc_v1_k8_rate_headroom as driver


def cpu_fixture(runtime):
    """Real original waveform/receiver math with small deterministic modules."""
    generator = torch.Generator().manual_seed(17126)
    hdl = torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator)
    hul = torch.randn(4, 8, 32, 1, dtype=torch.complex64, generator=generator).expand(-1, -1, -1, 256).clone()
    codes = torch.randn(8, 216, dtype=torch.complex64, generator=generator)
    template = torch.randn(8, 32, 256, dtype=torch.complex64, generator=generator)
    events = []

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('codes', codes.clone())
            self.scale = nn.Parameter(torch.tensor(1.))

        def forward(self, H_dl):
            self.assert_input(H_dl)
            return self.scale*self.codes[None].expand(H_dl.shape[0], -1, -1)

        @staticmethod
        def assert_input(value):
            if not torch.equal(value, hdl):
                raise AssertionError('UE must receive only its ordinary full-CSI source')

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('template', template.clone())
            self.scale = nn.Parameter(torch.tensor(.8))

        def forward(self, Y_rx, H_ul_est, soft, reliability, pilots, *, gamma_db, layout, prior, ce_error):
            events.append(('decoder', float(gamma_db[0])))
            gain = soft.real.mean(-1)+.01*torch.log1p(reliability).mean(-1)
            return dict(H_hat=self.scale*self.template[None]*(1+.01*gain[..., None, None]))

    class CE(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(.96))

        def forward(self, Y, pilots, initial, *, gamma_db):
            return initial*self.scale

    loaded = {}
    for arm in ('fdma', 'hybrid_g4'):
        codec = nn.Module()
        codec.encoder, codec.decoder = Encoder(), Decoder()
        phy = runtime.physics()
        stats = phy.ReceiverStatistics(phy.make_layout(arm), ce_errors=torch.full((6, 8, 256), .02))
        stats.ema_updates.fill_(30000)
        ce = CE()
        for module in (codec, stats, ce):
            module.eval().requires_grad_(False)
        loaded[arm] = (codec, stats, ce, dict(fixture=True))
    return hdl, hul, loaded, events


class MetadataContractTests(unittest.TestCase):
    def test_plan_runs_without_site_packages_or_result_writes(self):
        root = Path(driver.__file__).resolve().parent
        before = {p.relative_to(root).as_posix() for p in (root/'runs').glob('twc_v1_k8_rate_headroom_v1/**/*')}
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, '-B', '-S', str(Path(driver.__file__).resolve()), '--plan'],
                cwd=temporary, capture_output=True, text=True, check=True)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        plan = json.loads(result.stdout)
        self.assertEqual(plan['namespace'], 'twc_v1_k8_rate_headroom_v1')
        self.assertEqual(plan['optimizer_updates'], 0)
        self.assertFalse(plan['test_used'])
        after = {p.relative_to(root).as_posix() for p in (root/'runs').glob('twc_v1_k8_rate_headroom_v1/**/*')}
        self.assertEqual(before, after)

    def test_audited_scientific_core_and_terminal_pins_cannot_be_relaxed(self):
        self.assertEqual(driver.digest(driver.source_guard()), driver.SOURCE_CORE_SHA256)
        original = driver.sha
        def changed(path):
            return '0'*64 if Path(path).name == 'train_twc_v1_k8_ue_bs_rate.py' else original(path)
        with mock.patch.object(driver, 'sha', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'scientific core changed'):
                driver.source_guard()
        for arm in driver.ARMS:
            accepted = dict(driver.SOURCE_PINS[arm], step=10000, production_evidence=True)
            driver.check_source_record(arm, accepted)
            for field, wrong in (('checkpoint_sha256', '0'*64), ('CE_state_sha256', '0'*64),
                    ('statistics_state_sha256', '0'*64), ('progress_sha256', '0'*64),
                    ('step', 9999), ('production_evidence', False)):
                bad = dict(accepted, **{field: wrong})
                with self.subTest(arm=arm, field=field), self.assertRaises(ValueError):
                    driver.check_source_record(arm, bad)


class CompleteMUStatisticsTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for arm in driver.ARMS:
            for gamma in driver.GAMMAS:
                for sample in range(8):
                    batch, within = divmod(sample, 4)
                    rate = gamma*.25 + batch*.5 + within*.01
                    if arm == 'hybrid_g4':
                        rate += 1+2*batch
                    rows.append(dict(arm=arm, gamma_db=float(gamma), sample_index=sample,
                        batch_index=batch, within_batch=within,
                        **{metric: rate for metric in driver.RATE_METRICS},
                        signed_cross_term=-.01*(sample+1)))
        return rows

    def test_gamma_views_are_averaged_before_complete_batch_uncertainty(self):
        summaries, comparisons = driver.aggregate(self.rows())
        full = next(row for row in comparisons if row['scope'] == 'all_six' and row['metric'] == 'rate')
        self.assertEqual(full['n_samples'], 8)
        self.assertEqual(full['n_batches'], 2)
        self.assertEqual(full['n_gamma_points'], 6)
        self.assertAlmostEqual(full['mean'], 2.)
        self.assertAlmostEqual(full['batch_se'], 1.)
        self.assertAlmostEqual(full['ci_low'], .04)
        self.assertAlmostEqual(full['ci_high'], 3.96)
        points = [row for row in comparisons if row['scope'] == 'point' and row['metric'] == 'rate']
        self.assertEqual(len(points), 6)
        self.assertTrue(all(row['n_samples'] == 8 and row['n_batches'] == 2 for row in points))
        self.assertTrue(all(math.isclose(row['mean'], 2) and math.isclose(row['batch_se'], 1) for row in points))
        fdma = next(row for row in summaries if row['arm'] == 'fdma' and row['scope'] == 'all_six' and row['metric'] == 'rate')
        self.assertAlmostEqual(fdma['mean'], 3.39)
        self.assertAlmostEqual(fdma['batch_se'], .25)
        self.assertTrue(any(row['metric'] == 'signed_cross_term' and row['mean'] < 0 for row in summaries))

    def test_incomplete_duplicate_nonfinite_or_partial_metric_rows_are_refused(self):
        originals = self.rows()
        cases = [originals[:-1], originals+[originals[0]], copy.deepcopy(originals), copy.deepcopy(originals)]
        cases[2][0]['rate'] = float('nan')
        cases[3][0].pop('signed_cross_term')
        for rows in cases:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                driver.aggregate(rows)


class CompletedExportTests(unittest.TestCase):
    def test_actual_scalar_export_and_completed_binding_refuse_wrong_endpoint(self):
        import publish_twc_v1_k8_rate_headroom as publisher
        import ue_bs_rate_runtime as runtime
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'runs'/driver.NAMESPACE
            output.mkdir(parents=True)
            with mock.patch.object(driver, 'OUTPUT_ROOT', output), mock.patch.object(
                    publisher, 'ROOT', Path(temporary)):
                code = driver.code_identity()
                registered = dict(namespace=driver.NAMESPACE, plan=driver.plan(), code=code,
                    source_pins=driver.SOURCE_PINS)
                prereg = output/'preregistration.json'
                driver.atomic_json(prereg, registered)
                signatures = {}
                for arm in driver.ARMS:
                    pins = driver.SOURCE_PINS[arm]
                    states = {state: pins[field] for state, field in (
                        ('codec', 'state_sha256'), ('encoder', 'encoder_state_sha256'),
                        ('decoder', 'decoder_state_sha256'), ('statistics', 'statistics_state_sha256'),
                        ('CE', 'CE_state_sha256'))}
                    signatures[arm] = dict(before=driver.digest(states), after=driver.digest(states), states=states)
                provenance = dict(namespace=driver.NAMESPACE, production_evidence=True,
                    optimizer_updates=0, UE_updates=0, BS_updates=0, CE_updates=0, EMA_updates=0,
                    test_used=False, contains_model_weights=False, contains_channel_tensors=False,
                    source_core_sha256=driver.SOURCE_CORE_SHA256,
                    source_records={arm: dict(driver.SOURCE_PINS[arm], step=10000,
                        production_evidence=True) for arm in driver.ARMS},
                    frozen_signatures=signatures, preregistration_sha256=driver.sha(prereg),
                    plan={k: v for k, v in driver.plan().items() if k != 'output'},
                    code=code, data_protocol=driver.make_data(runtime, 'cuda:0').protocol)
                report = output/'reports'/'attempt_0123456789ab'
                rows = CompleteMUStatisticsTests.rows()
                summaries, comparisons = driver.aggregate(rows)
                payloads = {'README.md': b'Local scalar export contract fixture.\n',
                    'samples.csv': driver.csv_bytes(rows), 'summary.csv': driver.csv_bytes(summaries),
                    'comparisons.csv': driver.csv_bytes(comparisons),
                    'input_fingerprints.json': driver.canonical({'fixture': True}),
                    'provenance.json': driver.canonical(provenance)}
                publisher.create_export(report, payloads)
                driver.atomic_json(output/'summary.json', dict(status='completed',
                    report_directory=str(report), export_manifest_sha256=driver.sha(report/'export_manifest.json'),
                    preregistration_sha256=driver.sha(prereg)))
                driver.verify_completed_export(report)
                # Re-hashing a changed report cannot silently select a different
                # source terminal even when the scalar-file inventory is valid.
                changed = copy.deepcopy(provenance)
                changed['source_records']['fdma']['checkpoint_sha256'] = 'f'*64
                (report/'provenance.json').write_bytes(driver.canonical(changed))
                manifest = json.loads((report/'export_manifest.json').read_text())
                manifest['files']['provenance.json'] = driver.sha(report/'provenance.json')
                (report/'export_manifest.json').write_bytes(driver.canonical(manifest))
                summary = json.loads((output/'summary.json').read_text())
                summary['export_manifest_sha256'] = driver.sha(report/'export_manifest.json')
                driver.atomic_json(output/'summary.json', summary)
                with self.assertRaisesRegex(ValueError, 'terminal file/state identity changed'):
                    driver.verify_completed_export(report)


class RealFrozenPipelineTests(unittest.TestCase):
    def setUp(self):
        import ue_bs_rate_runtime as runtime
        self.runtime = runtime
        self.hdl, self.hul, self.loaded, self.events = cpu_fixture(runtime)
        self.metadata = driver.make_data(runtime, 'cpu')
        outer = self

        class Data:
            protocol = outer.metadata.protocol

            def address(self, role, index):
                return outer.metadata.address(role, index)

            def sample(self, role, index):
                outer.events.append(('sample', role, index))
                return dict(H_dl=outer.hdl.clone(), H_ul=outer.hul.clone(),
                    address=self.address(role, index))

        self.Data = Data

    def test_actual_cpu_physics_rzf_and_error_analysis_preserve_frozen_models(self):
        before = {arm: driver.frozen_signature(self.runtime, *modules[:3]) for arm, modules in self.loaded.items()}
        phy = self.runtime.physics()
        with mock.patch.object(phy, 'recover', wraps=phy.recover) as actual_recovery:
            rows, fingerprints, signatures, checks, protocol = driver.evaluate_models(self.runtime,
                self.loaded, lambda device: self.Data(), batches=2)
        self.assertEqual(len(rows), 2*2*6*4)
        self.assertEqual(len(fingerprints), 2*2*6)
        self.assertEqual(protocol, self.metadata.protocol)
        self.assertEqual(len([event for event in self.events if event[0] == 'decoder']), 24)
        self.assertEqual(actual_recovery.call_count, 24)
        from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
        for call, recorded in zip(actual_recovery.call_args_list, fingerprints):
            self.assertFalse(torch.equal(call.args[1], self.hul))
            self.assertEqual(_tensor_hash(call.args[1]), recorded['fingerprint']['estimated_UL'])
        self.assertEqual({event[1:] for event in self.events if event[0] == 'sample'},
            {('validation', 8192), ('validation', 8193)})
        for arm, modules in self.loaded.items():
            self.assertEqual(before[arm], driver.frozen_signature(self.runtime, *modules[:3]))
            self.assertEqual(signatures[arm]['before'], signatures[arm]['after'])
            self.assertEqual(int(modules[1].ema_updates), 30000)
            self.assertTrue(all(parameter.grad is None for module in modules[:3] for parameter in module.parameters()))
        self.assertTrue(checks)
        self.assertTrue(all(math.isfinite(value) for value in checks.values()))
        self.assertTrue(all(row['rate'] <= row['interference_free_upper_bound']+1e-3 for row in rows))
        self.assertTrue(all(row['ul_all_realized_cross_user_power'] == 0 for row in rows if row['arm'] == 'fdma'))
        self.assertTrue(any(row['ul_shared_realized_cross_user_power'] > 0 for row in rows if row['arm'] == 'hybrid_g4'))
        summaries, comparisons = driver.aggregate(rows)
        self.assertTrue(summaries and comparisons)
        # Pairing checks examine actual channels/noise/pilots, not only seeds.
        for mode in ('missing', 'noise', 'channel', 'address'):
            bad = copy.deepcopy(fingerprints)
            if mode == 'missing':
                bad.pop()
            elif mode == 'address':
                bad[-1]['address']['index'] += 1
            else:
                bad[-1]['fingerprint']['standard_noise' if mode == 'noise' else 'H_dl'] = 'f'*64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                driver.verify_pairing(bad, batches=2)

    def test_mutating_prior_buffer_is_rejected(self):
        codec, stats, ce, _ = self.loaded['fdma']
        original = codec.encoder.forward
        def changed(H_dl):
            stats.power_ema.add_(.001)
            return original(H_dl)
        codec.encoder.forward = changed
        with self.assertRaisesRegex(ValueError, 'Frozen model/CE/prior changed'):
            driver.evaluate_models(self.runtime, self.loaded, lambda device: self.Data(), batches=2)


if __name__ == '__main__':
    unittest.main()
