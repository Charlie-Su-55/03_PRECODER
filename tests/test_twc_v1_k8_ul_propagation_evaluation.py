"""Actual CPU receiver/filter/conditional-scoring integration, not real weights."""
import copy
import csv
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_ul_propagation_config as cfg
import evaluate_twc_v1_k8_ul_propagation as evaluate
import ul_propagation_runtime as runtime
import twc_v1_k8_ul_propagation_results as results

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Real PyTorch unavailable; no fake tensor math')
class PropagationPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'propagation'
        self.source = self.root/'teacher_fixture.bin'
        self.source.write_bytes(b'CPU fixture only; not a scientific checkpoint')
        for patch in (mock.patch.object(runtime, 'OUTPUT_ROOT', self.output),
                      mock.patch.object(cfg, 'OUTPUT_ROOT', self.output),
                      mock.patch.object(cfg, 'BATCHES', 1), mock.patch.object(cfg, 'SAMPLES', 4)):
            patch.start()
            self.addCleanup(patch.stop)
        generator = torch.Generator().manual_seed(1291)
        self.H_dl = torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator)
        self.H_ul = torch.randn(4, 8, 32, 1, dtype=torch.complex64, generator=generator).expand(-1, -1, -1, 256).clone()
        codes = torch.randn(8, 216, dtype=torch.complex64, generator=generator)
        metadata = runtime.make_data('cpu')
        self.instances = {}
        outer = self

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('codes', codes.clone())
                self.scale = nn.Parameter(torch.tensor(1.))
            def forward(self, H_dl):
                return self.codes[None].expand(H_dl.shape[0], -1, -1)*self.scale

        class NeverDecoder(nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError('Propagation diagnostics must not execute a BS decoder')

        class CE(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.96))
            def forward(self, Y, pilot, initial, *, gamma_db):
                return initial*self.scale

        class Data:
            protocol = metadata.protocol
            def address(self, role, index):
                return metadata.address(role, index)
            def sample(self, role, index):
                return dict(H_dl=outer.H_dl.clone(), H_ul=outer.H_ul.clone(),
                    address=self.address(role, index))
        self.Data = Data

        def loader(arm, device):
            codec = nn.Module()
            codec.encoder, codec.decoder = Encoder(), NeverDecoder()
            phy = runtime.physics()
            stats = phy.ReceiverStatistics(phy.make_layout(arm),
                ce_errors=torch.full((6, 8, 256), .02))
            stats.ema_updates.fill_(20000)
            ce = CE()
            for module in (codec, stats, ce):
                module.eval().requires_grad_(False)
            self.instances[arm] = (codec, stats, ce)
            return codec, stats, ce, dict(fixture=True,
                frozen_files=[dict(path=str(self.source), sha256=runtime.sha256(self.source))])
        self.loader = loader

    def run_fixture(self, name='attempt_fixture', data_factory=None):
        return evaluate.evaluate(self.output/'evaluation'/name, require_cuda=False,
            loader=self.loader, data_factory=data_factory or (lambda device: self.Data()))

    def test_real_cpu_fixed_filter_pipeline_and_calibration_cardinality(self):
        source_before = self.source.read_bytes()
        phy = runtime.physics()
        with mock.patch.object(phy, 'recover', wraps=phy.recover) as real_recovery:
            manifest = self.run_fixture()
        self.assertEqual(real_recovery.call_count, len(cfg.ARMS)*len(cfg.GAMMAS))
        self.assertEqual(manifest['status'], 'test_fixture')
        self.assertFalse(manifest['production_evidence'])
        self.assertFalse(manifest['source_checkpoints_verified'])
        self.assertTrue(manifest['original_filter_unchanged'])
        self.assertTrue(manifest['filter_reconstruction_verified'])
        self.assertTrue(manifest['error_decomposition_verified'])
        self.assertEqual(manifest['optimizer_updates'], 0)
        directory = self.output/'evaluation'/'attempt_fixture'
        entries = runtime.read_json(directory/'input_fingerprints.json')
        evaluate.fingerprint_map(entries, batches=1)
        rows = results.read_raw_csv(directory/'samples.csv.gz')
        self.assertEqual(len(rows), 2*6*4)
        self.assertTrue(all('variant' not in row for row in rows))
        for arm, signature in manifest['frozen_signatures'].items():
            self.assertEqual(signature['before'], signature['after'])
            self.assertEqual(signature['after'], runtime.frozen_signature(*self.instances[arm]))
        self.assertEqual(self.source.read_bytes(), source_before)
        with gzip.open(directory/'calibration_bins.csv.gz', 'rt') as stream:
            bins = list(csv.DictReader(stream))
        counts = {}
        for row in bins:
            key = row['arm'], float(row['gamma_db']), row['component']
            counts[key] = counts.get(key, 0)+int(row['count'])
        for (arm, gamma, component), count in counts.items():
            dimension = 24 if arm == 'fdma' else 4 if component.endswith('key') else 40 if component.endswith('shared') else 44
            self.assertEqual(count, cfg.SAMPLES*8*dimension)
        self.assertNotIn('sum_rate_bps_hz', (directory/'summary.csv').read_text())
        self.assertFalse(list(directory.rglob('*.zip')))
        # Exercise actual pipeline tables, including signed terms and pooled
        # gamma scopes, against the public scalar schema without publishing.
        import publish_twc_v1_k8_ul_propagation as publisher
        for name in publisher.SCALAR_FILES:
            publisher._check_scalar_file(directory/name)
        import run_twc_v1_k8_ul_propagation as runner
        with self.assertRaises(ValueError):
            runner.verify_manifest(directory)

    def test_changed_normal_soft_is_rejected_instead_of_scored_as_new_receiver(self):
        metrics = evaluate._metrics()
        original = metrics.analyze
        def changed(observation, recovered, *args):
            result = original(observation, recovered, *args)
            recovered['soft'].add_(.1)
            return result
        with mock.patch.object(metrics, 'analyze', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'original filter or recovery'):
                self.run_fixture('attempt_changed_soft')
        manifest = runtime.read_json(self.output/'evaluation'/'attempt_changed_soft'/'manifest.json')
        self.assertEqual(manifest['status'], 'failed')

    def test_changed_frozen_prior_fails_and_leaves_historical_bytes_untouched(self):
        original = runtime.observe
        def changed(raw, stats, ce, *args):
            result = original(raw, stats, ce, *args)
            stats.power_ema.add_(.001)
            return result
        before = self.source.read_bytes()
        with mock.patch.object(runtime, 'observe', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                self.run_fixture('attempt_changed_prior')
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(runtime.read_json(self.output/'evaluation'/'attempt_changed_prior'/'manifest.json')['status'], 'failed')

    def test_wrong_role_never_reaches_the_public_receiver(self):
        parent = self.Data
        class WrongData(parent):
            def sample(self, role, index):
                sample = super().sample(role, index)
                sample['address'] = runtime.make_data('cpu').address('test', index)
                return sample
        with mock.patch.object(runtime, 'observe', side_effect=AssertionError('must not forward')):
            with self.assertRaisesRegex(ValueError, 'address'):
                self.run_fixture('attempt_wrong_role', data_factory=lambda device: WrongData())

    def test_fingerprint_checks_refuse_truth_filter_changed_noise_and_missing_rows(self):
        self.run_fixture()
        entries = runtime.read_json(self.output/'evaluation'/'attempt_fixture'/'input_fingerprints.json')
        for mode in ('truth_filter', 'filter_hash', 'soft_hash', 'noise', 'missing', 'duplicate'):
            bad = copy.deepcopy(entries)
            if mode == 'truth_filter':
                bad[0]['true_UL_used_to_design_filter'] = True
            elif mode == 'filter_hash':
                bad[0]['filter_hashes_after']['exclusive'] = 'f'*64
            elif mode == 'soft_hash':
                bad[0]['recovery_hashes_after']['soft'] = 'f'*64
            elif mode == 'noise':
                bad[-1]['fingerprint']['standard_noise'] = 'f'*64
            elif mode == 'missing':
                bad.pop()
            else:
                bad.append(copy.deepcopy(bad[0]))
            with self.assertRaises(ValueError, msg=mode):
                evaluate.fingerprint_map(bad, batches=1)


if __name__ == '__main__':
    unittest.main()
