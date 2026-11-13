"""Full real-CPU diagnostic pipeline with explicit, nonproduction tiny models."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_ul_diagnostics_config as cfg
import evaluate_twc_v1_k8_ul_diagnostics as evaluate
import ul_diagnostics_runtime as runtime
import twc_v1_k8_ul_diagnostics_results as results

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Real PyTorch unavailable; no fake serialization/math')
class DiagnosticPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'diagnostics'
        self.source = self.root/'source_fixture.bin'
        self.source.write_bytes(b'fixture bytes, not a trained checkpoint')
        for patch in (mock.patch.object(runtime, 'OUTPUT_ROOT', self.output),
                      mock.patch.object(cfg, 'OUTPUT_ROOT', self.output),
                      mock.patch.object(cfg, 'BATCHES', 1),
                      mock.patch.object(cfg, 'SAMPLES', 4)):
            patch.start()
            self.addCleanup(patch.stop)
        generator = torch.Generator().manual_seed(842)
        self.H_dl = torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator)
        self.H_ul = torch.randn(4, 8, 32, 1, dtype=torch.complex64, generator=generator).expand(-1, -1, -1, 256).clone()
        codes = torch.randn(8, 216, dtype=torch.complex64, generator=generator)
        self.instances = {}
        outer = self
        metadata = runtime.make_data('cpu')

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('codes', codes.clone())
                self.scale = nn.Parameter(torch.tensor(1.))
            def forward(self, H_dl):
                return self.codes[None].expand(H_dl.shape[0], -1, -1)*self.scale

        class NeverDecoder(nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError('UL diagnostics must never run a DL decoder')

        class CE(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.97))
                self.register_buffer('tracked', torch.tensor(0.))
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

    def test_real_cpu_pipeline_pairs_inputs_freezes_state_and_never_runs_decoder(self):
        original_source = self.source.read_bytes()
        result = self.run_fixture()
        self.assertEqual(result['status'], 'test_fixture')
        self.assertFalse(result['production_evidence'])
        self.assertFalse(result['source_checkpoints_verified'])
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertTrue(result['paired_inputs_verified'])
        self.assertTrue(result['frozen_models_verified'])
        path = self.output/'evaluation'/'attempt_fixture'
        entries = json.loads((path/'input_fingerprints.json').read_text())
        evaluate.fingerprint_map(entries, batches=1)
        rows = results.read_raw_csv(path/'samples.csv.gz')
        self.assertEqual(len(rows), 2*2*6*4)
        self.assertEqual(self.source.read_bytes(), original_source)
        for arm, signature in result['frozen_signatures'].items():
            self.assertEqual(signature['before'], signature['after'])
            self.assertEqual(signature['after'], runtime.frozen_signature(*self.instances[arm]))
        for row in rows:
            if row['variant'] == 'true_ul_diagnostic':
                self.assertEqual(row['ul_ce_nmse_linear'], 0.)
                self.assertEqual(row['ul_ce_predicted_variance'], 0.)
        summary = (path/'summary.csv').read_text()
        self.assertNotIn('-inf', summary)
        self.assertNotIn('sum_rate_bps_hz', summary)
        import run_twc_v1_k8_ul_diagnostics as queue
        with self.assertRaises(ValueError):
            queue.verify_manifest(path)

    def test_changed_frozen_buffer_is_rejected_and_failed_manifest_retained(self):
        original = runtime.observe
        def changed(raw, stats, ce, *args):
            observation = original(raw, stats, ce, *args)
            ce.tracked.add_(1.)
            return observation
        with mock.patch.object(runtime, 'observe', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'frozen'):
                self.run_fixture('attempt_mutation')
        manifest = runtime.read_json(self.output/'evaluation'/'attempt_mutation'/'manifest.json')
        self.assertEqual(manifest['status'], 'failed')
        self.assertFalse((self.output/'evaluation'/'attempt_mutation'/'summary.csv').exists())

    def test_wrong_role_address_stops_before_any_receiver_forward(self):
        parent = self.Data
        class WrongData(parent):
            def sample(self, role, index):
                value = super().sample(role, index)
                value['address'] = runtime.make_data('cpu').address('test', index)
                return value
        with mock.patch.object(runtime, 'observe', side_effect=AssertionError('must not forward')):
            with self.assertRaisesRegex(ValueError, 'address'):
                self.run_fixture('attempt_wrong_role', data_factory=lambda device: WrongData())
        self.assertEqual(runtime.read_json(self.output/'evaluation'/'attempt_wrong_role'/'manifest.json')['status'], 'failed')

    def test_fingerprint_validator_rejects_variant_truth_leak_and_unpaired_noise(self):
        self.run_fixture()
        path = self.output/'evaluation'/'attempt_fixture'/'input_fingerprints.json'
        entries = runtime.read_json(path)
        for mode in ('normal_truth', 'changed_Y', 'cross_arm_noise', 'missing', 'duplicate', 'oracle_variance'):
            bad = copy.deepcopy(entries)
            if mode == 'normal_truth':
                bad[0]['recovery_inputs']['channel'] = bad[0]['fingerprint']['H_ul']
            elif mode == 'changed_Y':
                bad[1]['fingerprint']['Y_rx'] = 'f'*64
                bad[1]['recovery_inputs']['Y_rx'] = 'f'*64
            elif mode == 'cross_arm_noise':
                for row in bad:
                    if row['arm'] == 'hybrid_g4':
                        row['fingerprint']['standard_noise'] = 'f'*64
            elif mode == 'missing':
                bad.pop()
            elif mode == 'duplicate':
                bad.append(copy.deepcopy(bad[0]))
            else:
                bad[1]['receiver_ce_variance_max'] = .1
            with self.assertRaises(ValueError, msg=mode):
                evaluate.fingerprint_map(bad, batches=1)


if __name__ == '__main__':
    unittest.main()
