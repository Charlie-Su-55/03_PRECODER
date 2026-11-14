"""Actual CPU pilot CE, fitted moments, receivers and RZF integration.

Tiny deterministic neural modules exercise the real physical tensor path;
none of these fixtures represent production checkpoints or performance.
"""
import copy
import csv
import gzip
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from configs import twc_v1_k8_ul_calibration_config as cfg
import evaluate_twc_v1_k8_ul_calibration as evaluate
import ul_calibration_runtime as runtime
import twc_v1_k8_ul_calibration_results as results


class LockedCalibrationPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'calibration_run'
        self.source = self.root/'fixture_source.bin'
        self.source.write_bytes(b'Explicit CPU fixture; never a real scientific checkpoint')
        for patch in (mock.patch.object(cfg, 'OUTPUT_ROOT', self.output),
                mock.patch.object(runtime, 'OUTPUT_ROOT', self.output),
                mock.patch.object(cfg, 'CALIBRATION_BATCHES', 1),
                mock.patch.object(cfg, 'VALIDATION_BATCHES', 1),
                mock.patch.object(cfg, 'CALIBRATION_SAMPLES', 4),
                mock.patch.object(cfg, 'VALIDATION_SAMPLES', 4)):
            patch.start()
            self.addCleanup(patch.stop)
        self.calibration = self.output/'calibration'/'attempt_fixture'
        self.validation = self.output/'validation'/'attempt_fixture'
        generator = torch.Generator().manual_seed(8947)
        codes = torch.randn(8, 216, dtype=torch.complex64, generator=generator)
        template = torch.randn(8, 32, 256, dtype=torch.complex64, generator=generator)
        self.channels = {}
        for role in ('ce_calibration', 'validation'):
            self.channels[role] = dict(
                H_dl=torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator),
                H_ul=torch.randn(4, 8, 32, 1, dtype=torch.complex64, generator=generator)
                    .expand(-1, -1, -1, 256).clone())
        self.instances, self.decoder_calls, self.data_reads = [], [], []
        outer = self
        metadata = runtime.make_data('cpu')

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('codes', codes.clone())
                self.scale = nn.Parameter(torch.tensor(1.))
            def forward(self, H_dl):
                return self.codes[None].expand(H_dl.shape[0], -1, -1)*self.scale

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('template', template.clone())
                self.scale = nn.Parameter(torch.tensor(.9))
            def forward(self, Y, H_ul_est, soft, reliability, pilots, *, gamma_db, layout, prior, ce_error):
                outer.decoder_calls.append(dict(Y=Y.clone(), H=H_ul_est.clone(), soft=soft.clone(),
                    reliability=reliability.clone()))
                # Actual deterministic tensor reconstruction, using only
                # public receiver inputs. Truth is not an accepted argument.
                gain = soft.real.mean(-1)+.01*torch.log1p(reliability).mean(-1)
                return dict(H_hat=self.scale*self.template[None]*(1+.01*gain[..., None, None]))

        class CE(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.96))
            def forward(self, Y, pilots, initial, *, gamma_db):
                return initial*self.scale

        class Data:
            protocol = metadata.protocol
            def address(self, role, index):
                return metadata.address(role, index)
            def sample(self, role, index):
                outer.data_reads.append((role, index))
                return dict(**{k:v.clone() for k,v in outer.channels[role].items()},
                    address=self.address(role, index))
        self.Data = Data

        def loader(arm, device):
            codec = nn.Module()
            codec.encoder, codec.decoder = Encoder(), Decoder()
            phy = runtime.physics()
            stats = phy.ReceiverStatistics(phy.make_layout(arm),
                ce_errors=torch.full((6, 8, 256), .02))
            stats.ema_updates.fill_(20000)
            ce = CE()
            for module in (codec, stats, ce):
                module.eval().requires_grad_(False)
            self.instances.append((codec, stats, ce))
            return codec, stats, ce, dict(fixture=True,
                frozen_files=[dict(path=str(self.source), sha256=runtime.sha256(self.source))])
        self.loader = loader

    def calibrate(self):
        return evaluate.calibrate(self.calibration, require_cuda=False, loader=self.loader,
            data_factory=lambda device: self.Data())

    def validate(self, *, loader=None, data_factory=None):
        return evaluate.evaluate(self.validation, self.calibration, require_cuda=False,
            loader=loader or self.loader, data_factory=data_factory or (lambda device:self.Data()))

    def test_real_calibration_and_frozen_three_receiver_decoder_rzf_pipeline(self):
        before = self.source.read_bytes()
        cal = self.calibrate()
        locked = (self.calibration/'model.json').read_bytes()
        self.assertEqual(self.decoder_calls, [])
        self.assertEqual(set(self.data_reads), {('ce_calibration', 0)})
        math = evaluate._math()
        with mock.patch.object(math, 'recover_variants', wraps=math.recover_variants) as recover:
            manifest = self.validate()
        self.assertEqual(recover.call_count, 2*6)
        for call in recover.call_args_list:
            self.assertEqual(set(call.args[0]), {'Y','estimated','ce_variance','prior','pilots'})
        self.assertEqual(manifest['status'], 'test_fixture')
        self.assertFalse(manifest['production_evidence'])
        self.assertFalse(manifest['source_checkpoints_verified'])
        self.assertTrue(manifest['model_frozen'])
        self.assertEqual(manifest['model_sha256_before'], manifest['model_sha256_after'])
        self.assertEqual(manifest['calibration_manifest_sha256'], runtime.sha256(self.calibration/'manifest.json'))
        self.assertEqual((self.calibration/'model.json').read_bytes(), locked)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(len(self.decoder_calls), 2*6*3)
        self.assertEqual(set(self.data_reads), {('ce_calibration',0),('validation',0)})
        for offset in range(0, len(self.decoder_calls), 3):
            original, reliability, moment = self.decoder_calls[offset:offset+3]
            self.assertTrue(torch.equal(original['soft'], reliability['soft']))
            self.assertTrue(torch.equal(original['H'], reliability['H']))
            self.assertTrue(torch.equal(original['Y'], moment['Y']))
        rows = results.read_raw_csv(self.validation/'samples.csv.gz')
        self.assertEqual(len(rows), 2*3*6*4)
        self.assertTrue(all(row['role']=='validation' for row in rows))
        self.assertTrue(all('sum_rate_bps_hz' in row and row['sum_rate_bps_hz'] >= 0 for row in rows))
        self.assertTrue(all(row['rzf_power_max_error'] <= 1e-5 for row in rows))
        self.assertTrue(all('effective_ce_nmse_linear' in row for row in rows))
        indexed = {(r['arm'],r['variant'],r['gamma_db'],r['sample_index']):r for r in rows}
        for arm in cfg.ARMS:
            for gamma in cfg.GAMMAS:
                for i in range(cfg.VALIDATION_SAMPLES):
                    original = indexed[arm,'original',float(gamma),i]
                    reliability = indexed[arm,'reliability_only',float(gamma),i]
                    self.assertEqual(original['effective_ce_nmse_linear'], reliability['effective_ce_nmse_linear'])
                    for name in (name for name in original if name.startswith('source_')):
                        self.assertEqual(original[name], reliability[name])
        for arm, signatures in manifest['frozen_signatures'].items():
            self.assertEqual(signatures['before'], signatures['after'])
        with gzip.open(self.calibration/'cal_samples.csv.gz', 'rt') as stream:
            sufficient = list(csv.DictReader(stream))
        self.assertEqual(len(sufficient), 6*4)
        self.assertTrue(all(row['role']=='calibration' for row in sufficient))
        evaluate.verify_calibration(self.calibration, allow_fixture=True)
        evaluate.verify_validation(self.validation, allow_fixture=True)
        self.assertFalse(list(self.output.rglob('*.zip')))
        for path, verifier in ((self.calibration,evaluate.verify_calibration),
                (self.validation,evaluate.verify_validation)):
            with self.assertRaisesRegex(ValueError, 'not production'):
                verifier(path)

    def test_tampered_model_is_rejected_before_validation_data_is_created(self):
        self.calibrate()
        path = self.calibration/'model.json'
        path.write_bytes(path.read_bytes()+b'\n')
        factory = mock.Mock(side_effect=AssertionError('validation data must not be accessed'))
        with self.assertRaisesRegex(ValueError, 'artifact bytes'):
            self.validate(data_factory=factory)
        factory.assert_not_called()
        self.assertFalse(self.validation.exists())

    def test_tampered_calibration_sufficient_stats_are_also_locked(self):
        self.calibrate()
        path = self.calibration/'cal_samples.csv.gz'
        path.write_bytes(path.read_bytes()+b'changed')
        factory = mock.Mock(side_effect=AssertionError('validation must not start'))
        with self.assertRaisesRegex(ValueError, 'artifact bytes'):
            self.validate(data_factory=factory)
        factory.assert_not_called()

    def test_different_source_CE_is_rejected_before_validation_forward(self):
        self.calibrate()
        original = self.loader
        def wrong(arm, device):
            codec, stats, ce, record = original(arm, device)
            ce.scale.fill_(.8)
            return codec, stats, ce, record
        with self.assertRaisesRegex(ValueError, 'Common CE'):
            self.validate(loader=wrong)
        self.assertFalse(self.decoder_calls)
        self.assertEqual(runtime.read_json(self.validation/'manifest.json')['status'], 'failed')

    def test_wrong_validation_address_fails_before_observation(self):
        self.calibrate()
        parent = self.Data
        class WrongData(parent):
            def sample(self, role, index):
                sample = super().sample(role, index)
                sample['address'] = self.address('test', index)
                return sample
        with mock.patch.object(runtime, 'observe', side_effect=AssertionError('must not observe')):
            with self.assertRaisesRegex(ValueError, 'address'):
                self.validate(data_factory=lambda device: WrongData())

    def test_reusing_calibration_channels_is_rejected_despite_different_addresses(self):
        self.calibrate()
        self.channels['validation'] = self.channels['ce_calibration']
        with self.assertRaisesRegex(ValueError, 'channel payloads overlap'):
            self.validate()

    def test_changed_frozen_prior_is_not_accepted_as_calibration(self):
        original = runtime.observe
        def changed(raw, stats, ce, *args):
            out = original(raw, stats, ce, *args)
            stats.power_ema.add_(.001)
            return out
        with mock.patch.object(runtime, 'observe', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Frozen calibration'):
                self.calibrate()
        self.assertEqual(runtime.read_json(self.calibration/'manifest.json')['status'], 'failed')

    def test_fingerprint_checker_rejects_pairing_and_receiver_control_mutations(self):
        self.calibrate()
        self.validate()
        entries = runtime.read_json(self.validation/'input_fingerprints.json')
        for mutation in ('CE_pair', 'truth_input', 'soft_control', 'after', 'role', 'missing'):
            bad = copy.deepcopy(entries)
            if mutation == 'CE_pair':
                bad[-1]['fingerprint']['estimated_UL'] = 'f'*64
            elif mutation == 'truth_input':
                bad[0]['true_UL_receiver_input'] = True
            elif mutation == 'soft_control':
                bad[0]['recovery_hashes']['reliability_only']['soft'] = 'f'*64
                bad[0]['recovery_hashes_after'] = copy.deepcopy(bad[0]['recovery_hashes'])
            elif mutation == 'after':
                bad[0]['recovery_hashes_after']['original']['variance'] = 'f'*64
            elif mutation == 'role':
                bad[0]['role'] = 'calibration'
            else:
                bad.pop()
            with self.assertRaises(ValueError, msg=mutation):
                evaluate.fingerprint_map(bad, role='validation', batches=1)

    def test_mutating_the_in_memory_calibration_model_is_rejected(self):
        self.calibrate()
        math = evaluate._math()
        original = math.recover_variants
        def changed(observation, layout, gamma, model):
            recovered = original(observation, layout, gamma, model)
            model['nodes'][0]['strata'][0]['alpha_real'] += .001
            return recovered
        with mock.patch.object(math, 'recover_variants', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'in-memory'):
                self.validate()
        self.assertFalse(self.decoder_calls)
        self.assertEqual(runtime.read_json(self.validation/'manifest.json')['status'], 'failed')

    def test_mutating_filter_tensors_during_scoring_is_rejected(self):
        self.calibrate()
        math, metrics, holder = evaluate._math(), evaluate._metrics(), {}
        original_recover, original_diagnostics = math.recover_variants, metrics.diagnostic_vectors
        def recovered(*args):
            holder['value'] = original_recover(*args)
            return holder['value']
        def changed(*args):
            vectors = original_diagnostics(*args)
            holder['value']['original']['filters']['exclusive'].add_(.001)
            return vectors
        with mock.patch.object(math, 'recover_variants', side_effect=recovered), \
                mock.patch.object(metrics, 'diagnostic_vectors', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed receiver observations'):
                self.validate()
        self.assertEqual(runtime.read_json(self.validation/'manifest.json')['status'], 'failed')

    def test_real_pipeline_scalar_artifacts_pass_public_export_schema(self):
        self.calibrate()
        self.validate()
        import publish_twc_v1_k8_ul_calibration as publisher
        output = self.root/'scalar_export_only'
        output.mkdir()
        sources = {
            'model.json':self.calibration/'model.json',
            'cal_samples.csv.gz':self.calibration/'cal_samples.csv.gz',
            'cal_input_fingerprints.json':self.calibration/'input_fingerprints.json',
            **{name:self.validation/name for name in
                ('samples.csv.gz','summary.csv','comparisons.csv','input_fingerprints.json')},
        }
        self.assertEqual(set(sources), set(publisher.SCALAR_FILES))
        for name, source in sources.items():
            destination = output/name
            shutil.copyfile(source, destination)
            publisher._check_scalar_file(destination)
        with (output/'summary.csv').open() as stream:
            summary_columns = len(csv.DictReader(stream).fieldnames)
        total_bytes = sum(path.stat().st_size for path in output.iterdir())
        self.assertLess(summary_columns, 2048)
        self.assertLess(total_bytes, publisher.MAX_TOTAL_BYTES)
        print(f'Actual CPU integration export: summary_columns={summary_columns}; scalar_bytes={total_bytes}', flush=True)
        # This scalar validation is deliberately not production export: the
        # genuine CPU fixture remains inadmissible to the production verifier.
        with self.assertRaisesRegex(ValueError, 'not production'):
            evaluate.verify_validation(self.validation)


if __name__ == '__main__':
    unittest.main()
