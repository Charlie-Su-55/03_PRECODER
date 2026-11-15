"""Actual CPU waveform/recovery/decoder/RZF checks for fresh final replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from configs import twc_v1_k8_ue_bs_rate_config as cfg
import evaluate_twc_v1_k8_ue_bs_rate as evaluation
import twc_v1_k8_ue_bs_rate_results as results
import ue_bs_rate_runtime as runtime


class JointFinalReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'joint'
        self.directory = self.output/'evaluation/attempt_fixture'
        self.file = self.root/'explicit_cpu_fixture.bin'
        self.file.write_bytes(b'CPU fixture, not a scientific checkpoint')
        self.core = dict(schema='explicit_fixture', sha256={'fixture.py': 'a'*64})
        for patch in (mock.patch.object(cfg, 'OUTPUT_ROOT', self.output),
                mock.patch.object(runtime, 'OUTPUT_ROOT', self.output),
                mock.patch.object(cfg, 'FINAL_BATCHES', 1),
                mock.patch.dict(cfg.ROLE_RANGES, final=[4096, 4096]),
                mock.patch.object(cfg, 'CALIBRATION_DIR', self.root/'no_prior_labels'),
                mock.patch.object(runtime, 'core_identity', return_value=self.core),
                mock.patch.object(runtime, 'guard_core', side_effect=lambda core: self.assertEqual(core, self.core))):
            patch.start()
            self.addCleanup(patch.stop)
        gen = torch.Generator().manual_seed(98163)
        self.hdl = torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=gen)
        self.hul = torch.randn(4, 8, 32, 1, dtype=torch.complex64, generator=gen).expand(-1, -1, -1, 256).clone()
        codes = torch.randn(8, 216, dtype=torch.complex64, generator=gen)
        template = torch.randn(8, 32, 256, dtype=torch.complex64, generator=gen)
        outer = self
        self.events, self.data_reads, self.instances, self.factories = [], [], [], []

        class Encoder(nn.Module):
            def __init__(self, method):
                super().__init__()
                change = torch.linspace(.8, 1.2, 216) if method == 'joint' else torch.ones(216)
                self.register_buffer('codes', codes.clone()*change)
                self.scale = nn.Parameter(torch.tensor(1.))

            def forward(self, H):
                return self.scale*self.codes[None].expand(H.shape[0], -1, -1)

        class Decoder(nn.Module):
            def __init__(self, method):
                super().__init__()
                self.register_buffer('template', template.clone())
                self.scale = nn.Parameter(torch.tensor({'source': .7, 'bs_only': .74, 'joint': .78}[method]))

            def forward(self, Y_rx, H_ul_est, soft, reliability, pilots, *, gamma_db, layout, prior, ce_error):
                gain = soft.real.mean(-1)+.01*torch.log1p(reliability).mean(-1)
                return dict(H_hat=self.scale*self.template[None]*(1+.01*gain[..., None, None]))

        class CE(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.96))

            def forward(self, Y, pilots, initial, *, gamma_db):
                return initial*self.scale

        self.metadata = runtime.make_data('cpu', role='final')

        class Data:
            protocol = outer.metadata.protocol

            def sample(self, role, index):
                outer.data_reads.append((role, index))
                return dict(H_dl=outer.hdl.clone(), H_ul=outer.hul.clone(),
                    address=outer.metadata.address(role, index))

        self.Data = Data

        def loader(method, arm, device):
            outer.events.append(('load', method, arm))
            codec = nn.Module()
            codec.encoder, codec.decoder = Encoder(method), Decoder(method)
            phy = runtime.physics()
            stats = phy.ReceiverStatistics(phy.make_layout(arm), ce_errors=torch.full((6, 8, 256), .02))
            stats.ema_updates.fill_(30000 if method == 'joint' else 20000)
            if method == 'joint':
                stats.power_ema.mul_(torch.linspace(.85, 1.15, stats.power_ema.shape[-1]))
            ce = CE()
            for module in (codec, stats, ce):
                module.eval().requires_grad_(False)
            outer.instances.append((codec, stats, ce))
            record = dict(arm=arm, method=method, step=5000 if method == 'source' else cfg.UPDATES,
                production_evidence=False, state_sha256=runtime.state_digest(codec),
                encoder_state_sha256=runtime.state_digest(codec.encoder),
                decoder_state_sha256=runtime.state_digest(codec.decoder),
                CE_state_sha256=runtime.state_digest(ce), statistics_state_sha256=runtime.state_digest(stats),
                frozen_files=[dict(path=str(outer.file), sha256=runtime.sha256(outer.file))])
            return codec, stats, ce, record

        self.loader = loader

        def factory(device):
            outer.events.append(('factory',))
            outer.factories.append(device)
            return Data()

        self.factory = factory

    def run_eval(self, **kwargs):
        return evaluation.evaluate(self.directory, require_cuda=False,
            source_loader=kwargs.pop('source_loader', lambda a, d: self.loader('source', a, d)),
            control_loader=kwargs.pop('control_loader', lambda a, d: self.loader('bs_only', a, d)),
            terminal_loader=kwargs.pop('terminal_loader', lambda a, d: self.loader('joint', a, d)),
            data_factory=kwargs.pop('data_factory', self.factory), **kwargs)

    def test_real_cpu_replay_accepts_six_models_before_data_and_allows_joint_changes(self):
        original = self.file.read_bytes()
        manifest = self.run_eval()
        self.assertEqual(manifest['status'], 'test_fixture')
        self.assertFalse(manifest['production_evidence'])
        first = next(i for i, e in enumerate(self.events) if e[0] == 'factory')
        self.assertEqual(self.events[:first], [('load', m, a) for m in cfg.METHODS for a in cfg.ARMS]
            +[('load', 'source', 'fdma')])
        self.assertEqual(set(self.data_reads), {('validation', 4096)})
        self.assertEqual(len(self.factories), 6)
        rows = results.read_raw_csv(self.directory/'samples.csv.gz')
        self.assertEqual(len(rows), 3*2*6*4)
        self.assertTrue(all(r['sum_rate_bps_hz'] >= 0 and r['rzf_power_max_error'] <= 1e-5 for r in rows))
        self.assertTrue(all('soft_actual_to_reported_mse_ratio' in r for r in rows))
        fingerprints = runtime.read_json(self.directory/'input_fingerprints.json')
        self.assertEqual(len(fingerprints), 12)
        for row in fingerprints:
            self.assertEqual(set(row['conditions']), set(cfg.METHODS))
            self.assertEqual(row['method_inputs']['source'], row['method_inputs']['bs_only'])
            self.assertNotEqual(row['method_inputs']['joint']['wave_active'], row['method_inputs']['source']['wave_active'])
            self.assertNotEqual(row['method_inputs']['joint']['prior'], row['method_inputs']['source']['prior'])
        self.assertEqual(self.file.read_bytes(), original)
        self.assertTrue(all(v['before'] == v['after'] for values in manifest['frozen_signatures'].values() for v in values.values()))
        evaluation.verify_manifest(self.directory, allow_fixture=True)
        with self.assertRaisesRegex(ValueError, 'fixture'):
            evaluation.verify_manifest(self.directory)
        import publish_twc_v1_k8_ue_bs_rate as publisher
        for name in publisher.SCALAR_FILES:
            publisher._check_scalar_file(self.directory/name)
        self.assertFalse(list(self.output.rglob('*.zip')))

    def test_missing_last_joint_terminal_stops_before_final_data(self):
        def wrong(arm, device):
            codec, stats, ce, record = self.loader('joint', arm, device)
            if arm == 'hybrid_g4':
                record['step'] -= 1
            return codec, stats, ce, record
        with self.assertRaisesRegex(ValueError, 'fixed joint terminal'):
            self.run_eval(terminal_loader=wrong)
        self.assertFalse(self.factories)
        self.assertFalse(self.data_reads)
        self.assertFalse(self.directory.exists())

    def test_wrong_control_budget_stops_before_final_data(self):
        def wrong(arm, device):
            codec, stats, ce, record = self.loader('bs_only', arm, device)
            record['step'] -= 100
            return codec, stats, ce, record
        with self.assertRaisesRegex(ValueError, 'identity and budget'):
            self.run_eval(control_loader=wrong)
        self.assertFalse(self.factories)

    def test_changed_source_file_is_rejected_before_final_data(self):
        def wrong(arm, device):
            values = self.loader('source', arm, device)
            self.file.write_bytes(self.file.read_bytes()+b'x')
            return values
        with self.assertRaisesRegex(ValueError, 'read-only|Read-only'):
            self.run_eval(source_loader=wrong)
        self.assertFalse(self.factories)

    def test_source_and_control_cannot_have_different_encoder(self):
        def wrong(arm, device):
            codec, stats, ce, record = self.loader('bs_only', arm, device)
            codec.encoder.codes[0, 0] += .01
            return codec, stats, ce, record
        with self.assertRaisesRegex(ValueError, 'retain the source UE/prior'):
            self.run_eval(control_loader=wrong)
        self.assertFalse(self.factories)

    def test_different_CE_table_or_network_is_rejected_before_final_data(self):
        for field in ('network', 'table'):
            with self.subTest(field=field):
                def wrong(arm, device):
                    codec, stats, ce, record = self.loader('joint', arm, device)
                    if field == 'network':
                        ce.scale.fill_(.8)
                    else:
                        stats.ce_errors.add_(.01)
                    return codec, stats, ce, record
                with self.assertRaisesRegex(ValueError, 'CE network or paid-pilot CE-error'):
                    self.run_eval(terminal_loader=wrong)
                self.assertFalse(self.factories)

    def test_wrong_final_address_cannot_enter_monitor_or_test_stream(self):
        parent = self.Data
        class Wrong(parent):
            def sample(inner, role, index):
                sample = super().sample(role, index)
                sample['address'] = self.metadata.address('validation', 0)
                return sample
        with self.assertRaisesRegex(ValueError, 'Final data address'):
            self.run_eval(data_factory=lambda device: Wrong())
        self.assertEqual(runtime.read_json(self.directory/'manifest.json')['status'], 'failed')

    def test_common_noise_change_is_rejected(self):
        original = runtime.forward_codec
        def wrong(*args, **kwargs):
            out = original(*args, **kwargs)
            if kwargs['method'] == 'joint':
                out['input_hashes']['standard_noise'] = 'f'*64
            return out
        with self.assertRaisesRegex(ValueError, 'identical newly addressed common inputs'):
            self.run_eval(forward=wrong)

    def test_scoring_cannot_change_actual_receiver_or_decoder_output(self):
        original = evaluation.score_out
        def wrong(out, truth):
            result = original(out, truth)
            out['H_hat'].add_(.1)
            return result
        with mock.patch.object(evaluation, 'score_out', side_effect=wrong):
            with self.assertRaisesRegex(ValueError, 'Offline scoring mutated'):
                self.run_eval()

    def test_eval_cannot_commit_prior(self):
        original = runtime.forward_codec
        def wrong(codec, stats, ce, *args, **kwargs):
            out = original(codec, stats, ce, *args, **kwargs)
            if kwargs['method'] == 'joint':
                stats.power_ema.add_(.01)
            return out
        with self.assertRaisesRegex(ValueError, 'Final inference changed frozen'):
            self.run_eval(forward=wrong)

    def test_fingerprint_guard_rejects_missing_view_and_wrong_gamma(self):
        self.run_eval()
        entries = runtime.read_json(self.directory/'input_fingerprints.json')
        for field in ('missing', 'gamma', 'wrong_source_receiver'):
            with self.subTest(field=field):
                changed = copy.deepcopy(entries)
                if field == 'missing':
                    changed[0]['conditions'].remove('joint')
                elif field == 'gamma':
                    changed[0]['input_hashes']['gamma'] = '0'*64
                else:
                    changed[0]['recovery_hashes']['bs_only']['soft'] = '1'*64
                with self.assertRaises(ValueError):
                    evaluation.fingerprint_map(changed, device='cpu')

    def test_self_consistent_wrong_manifest_budget_is_refused(self):
        self.run_eval()
        path = self.directory/'manifest.json'
        manifest = runtime.read_json(path)
        manifest['terminal_updates'] += 100
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'terminal_updates'):
            evaluation.verify_manifest(self.directory, allow_fixture=True)


class JointTrainingPairingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)/'joint'
        for patch in (mock.patch.object(cfg, 'OUTPUT_ROOT', self.output),
                mock.patch.object(runtime, 'OUTPUT_ROOT', self.output),
                mock.patch.object(cfg, 'UPDATES', 2), mock.patch.object(cfg, 'SAVE_EVERY', 1)):
            patch.start()
            self.addCleanup(patch.stop)
        self.metadata = runtime.make_training_data('cuda:0')
        self.protocol = self.metadata.protocol
        api = evaluation._trace()
        trace = api.initial_training_trace(self.protocol)
        rows = []
        for step in (1, 2):
            before = dict(trace)
            items = []
            for micro in range(cfg.ACCUMULATION):
                index = cfg.micro_index(step, micro)
                item = dict(micro_index=index, address=self.metadata.address('codec_train', index),
                    channel=f'{index:064x}', standard_noise=f'{index+20:064x}',
                    pilot_polarity=f'{index+40:064x}', gamma=evaluation._tensor_hash(
                        torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32)))
                trace = api.advance_training_trace(trace, item, self.protocol)
                items.append(item)
            rows.append(dict(arm='fdma', method='joint', step=step, counters=cfg.counters(step, 20000),
                input_fingerprints=items, training_trace_before_update=before, training_trace=dict(trace)))
        self.pin = dict(trace)
        patch = mock.patch.object(cfg, 'TRAINING_TRACE_PIN', self.pin)
        patch.start()
        self.addCleanup(patch.stop)
        self.terminals, self.controls, self.rows = {}, {}, rows
        for arm in cfg.ARMS:
            directory = self.output/'training'/cfg.key(arm, 'joint')
            directory.mkdir(parents=True)
            path = directory/'progress.jsonl'
            path.write_text(''.join(json.dumps(dict(row, arm=arm))+'\n' for row in rows))
            self.terminals[arm] = dict(data_protocol=self.protocol, base_ema_updates=20000,
                training_trace=dict(trace), progress_sha256=runtime.sha256(path))
            self.controls[arm] = dict(training_trace=dict(trace))

    def test_real_snapshot_chain_closes_and_matches_archived_trace(self):
        proof = evaluation.verify_training_pairing(self.terminals, self.controls)
        self.assertEqual(proof['status'], 'passed')
        self.assertEqual(proof['common_training_trace'], self.pin)
        self.assertTrue(proof['new_snapshot_pairing_verified'])
        self.assertFalse(proof['archived_raw_progress_replayed'])
        self.assertEqual(proof['hashed_microbatches'], 16)

    def test_wrong_archived_endpoint_is_refused(self):
        self.controls['fdma']['training_trace']['sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'Archived control common-input trace pin'):
            evaluation.verify_training_pairing(self.terminals, self.controls)

    def test_tampered_actual_snapshot_is_refused_even_when_file_hash_is_rebound(self):
        path = self.output/'training'/cfg.key('fdma', 'joint')/'progress.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['input_fingerprints'][0]['gamma'] = 'f'*64
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.terminals['fdma']['progress_sha256'] = runtime.sha256(path)
        with self.assertRaisesRegex(ValueError, 'Actual joint gamma'):
            evaluation.verify_training_pairing(self.terminals, self.controls)

    def test_failed_save_duplicate_boundary_retains_history_and_uses_last_row(self):
        for arm in cfg.ARMS:
            path = self.output/'training'/cfg.key(arm, 'joint')/'progress.jsonl'
            lines = path.read_text().splitlines(keepends=True)
            path.write_text(lines[0]+''.join(lines))
            self.terminals[arm]['progress_sha256'] = runtime.sha256(path)
        proof = evaluation.verify_training_pairing(self.terminals, self.controls)
        self.assertTrue(all(r['superseded_journal_rows'] == 1 for r in proof['trajectories'].values()))


if __name__ == '__main__':
    unittest.main()
