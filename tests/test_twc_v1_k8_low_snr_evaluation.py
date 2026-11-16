"""Real CPU waveform/recovery/CSI/RZF replay; no scientific source weights."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import evaluate_twc_v1_k8_low_snr as evaluation
import twc_v1_k8_low_snr_results as results


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        from configs import twc_v1_k8_low_snr_config as cfg
        import low_snr_runtime as runtime
        from test_twc_v1_k8_rate_headroom_driver import cpu_fixture
        self.cfg, self.runtime = cfg, runtime
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)/'extension'
        self.directory = self.output/'evaluation/attempt_fixture'
        source_file = Path(temporary.name)/'explicit_cpu_fixture.bin'
        source_file.write_bytes(b'No scientific checkpoint, explicit CPU fixture only')
        core = dict(schema='explicit_cpu_fixture', sha256={'fixture.py': 'a'*64})
        for patch in (mock.patch.object(cfg, 'OUTPUT_ROOT', self.output),
                mock.patch.object(runtime, 'OUTPUT_ROOT', self.output),
                mock.patch.object(cfg, 'FINAL_BATCHES', 2),
                mock.patch.object(cfg, 'UPDATES', 4),
                mock.patch.dict(cfg.ROLE_RANGES, final=[4096, 4097]),
                mock.patch.object(runtime, 'core_identity', return_value=core),
                mock.patch.object(runtime, 'guard_core', side_effect=lambda value: self.assertEqual(value, core))):
            patch.start(); self.addCleanup(patch.stop)
        hdl, hul, models, _ = cpu_fixture(runtime)
        self.hdl, self.hul = hdl, hul
        models['hybrid_g4'][1].ema_updates.fill_(40000)
        self.models = dict(fdma_frozen=models['fdma'][:3], hybrid_source=models['hybrid_g4'][:3],
            low_snr_control=copy.deepcopy(models['hybrid_g4'][:3]),
            low_snr_interference=copy.deepcopy(models['hybrid_g4'][:3]))
        codec, stats, ce = self.models['low_snr_interference']
        with torch.no_grad():
            codec.encoder.codes[:, 0].add_(.1+.05j)
            stats.power_ema.mul_(torch.linspace(.9, 1.1, stats.power_ema.shape[-1]))
            stats.ema_updates.fill_(50000)
        self.models['low_snr_control'][1].ema_updates.fill_(50000)
        source_codec, source_stats, source_ce = self.models['hybrid_source']
        self.common_source = dict(base_ema_updates=40000,
            state_sha256=runtime.state_digest(source_codec),
            base_encoder_state_sha256=runtime.state_digest(source_codec.encoder),
            base_decoder_state_sha256=runtime.state_digest(source_codec.decoder),
            base_statistics_state_sha256=runtime.state_digest(source_stats))
        self.events = []
        self.metadata = runtime.make_data('cpu')
        outer = self
        class Data:
            protocol = outer.metadata.protocol
            def sample(self, role, index):
                outer.events.append(('sample', role, index))
                return dict(H_dl=outer.hdl.clone(), H_ul=outer.hul.clone(),
                    address=outer.metadata.address(role, index))
        def factory(device):
            outer.events.append(('factory', str(device)))
            return Data()
        self.factory = factory
        def loader(variant, device):
            outer.events.append(('load', variant))
            record = dict(production_evidence=False,
                frozen_files=[dict(path=str(source_file), sha256=runtime.sha256(source_file))])
            if variant in cfg.TRAJECTORIES:
                record.update(step=cfg.UPDATES,method=variant,source_identity=copy.deepcopy(outer.common_source),
                    base_ema_updates=40000,total_ema_updates=40000+cfg.UPDATES,
                    counters=cfg.counters(cfg.UPDATES,base_ema_updates=40000),
                    training_trace=dict(format='fixturetrace',initial_sha256='a'*64,sha256='b'*64,microbatches=cfg.UPDATES*cfg.ACCUMULATION))
            return (*outer.models[variant], record)
        self.loader = loader

    def run_evaluation(self):
        return evaluation.evaluate(self.directory, require_cuda=False,
            loader=self.loader, data_factory=self.factory)

    def test_actual_four_view_replay_and_ul_partitions_accept_models_and_pairing_before_data(self):
        before = {v: evaluation._signature(*modules) for v, modules in self.models.items()}
        manifest = self.run_evaluation()
        self.assertEqual(self.events[:5], [('load', 'fdma_frozen'), ('load', 'hybrid_source'),
            ('load', 'low_snr_control'), ('load', 'low_snr_interference'), ('factory', 'cpu')])
        self.assertEqual([e for e in self.events if e[0] == 'sample'],
            [('sample', 'validation', 4096), ('sample', 'validation', 4097)])
        self.assertEqual(manifest['status'], 'test_fixture')
        self.assertFalse(manifest['production_evidence'])
        self.assertEqual(manifest['FDMA_new_training_updates'], 0)
        self.assertFalse(manifest['budgets_equal'])
        self.assertFalse(manifest['single_factor_loss_attribution'])
        self.assertTrue(manifest['paired_trajectory_budgets_equal'])
        self.assertEqual(manifest['training_pairing']['status'],'passed')
        self.assertTrue(manifest['closure_checks'])
        rows = results.read_raw_csv(self.directory/'samples.csv.gz')
        self.assertEqual(len(rows), 4*2*6*4)
        self.assertEqual({r['variant'] for r in rows}, set(self.cfg.EVAL_VARIANTS))
        self.assertTrue(all('ul_all_nominal_scaled_actual_error_power' in r for r in rows))
        self.assertFalse(any('ul_shared_nominal_scaled_realized_ce_power' in r for r in rows if r['arm']=='fdma'))
        for row in rows:
            prefix='ul_all_nominal_scaled_'
            self.assertAlmostEqual(row[prefix+'actual_error_power'],sum(row[prefix+k] for k in ('realized_self_power','realized_cross_user_power','realized_noise_power','realized_self_cross_user_cross','realized_self_noise_cross','realized_cross_user_noise_cross')),places=5)
        self.assertEqual({r['gamma_db'] for r in rows}, set(self.cfg.GAMMAS))
        for variant, modules in self.models.items():
            self.assertEqual(before[variant], evaluation._signature(*modules))
        fingerprints = self.runtime.read_json(self.directory/'input_fingerprints.json')
        self.assertEqual(len(fingerprints), 4*2*6)
        evaluation.fingerprint_map(fingerprints, batches=2, metadata=self.metadata)
        bad = copy.deepcopy(fingerprints); bad[-1]['fingerprint']['standard_noise'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'identical physical input bytes'):
            evaluation.fingerprint_map(bad, batches=2, metadata=self.metadata)
        self.assertEqual(manifest, evaluation.verify_manifest(self.directory))
        (self.directory/'summary.csv').write_text('corrupt scalar evidence\n')
        with self.assertRaisesRegex(ValueError, 'scalar artifact changed'):
            evaluation.verify_manifest(self.directory)

    def test_ce_mismatch_or_failed_terminal_stops_before_data(self):
        ce = self.models['low_snr_interference'][2]
        with torch.no_grad(): ce.scale.add_(.01)
        with self.assertRaisesRegex(ValueError, 'CE weights and calibrated table'):
            self.run_evaluation()
        self.assertFalse(any(e[0] in ('sample', 'factory') for e in self.events))

    def test_pairing_mismatch_stops_before_final_data(self):
        original=self.loader
        def loader(variant,device):
            codec,stats,ce,record=original(variant,device)
            if variant=='low_snr_interference': record['training_trace']['sha256']='e'*64
            return codec,stats,ce,record
        self.loader=loader
        with self.assertRaisesRegex(ValueError,'complete actual input trace'):
            self.run_evaluation()
        self.assertFalse(any(e[0] in ('sample','factory') for e in self.events))

    def test_mutating_frozen_prior_cannot_be_exported(self):
        original = self.runtime.forward_codec
        def changed(codec, stats, ce, *args, **kwargs):
            output = original(codec, stats, ce, *args, **kwargs)
            stats.ema_updates.add_(1)
            return output
        with mock.patch.object(self.runtime, 'forward_codec', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed frozen model/prior/CE'):
                self.run_evaluation()
        self.assertEqual(self.runtime.read_json(self.directory/'manifest.json')['status'], 'failed')
        self.assertFalse((self.directory/'samples.csv.gz').exists())


if __name__ == '__main__': unittest.main()
