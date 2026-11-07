"""Task-loss screen checks using metadata and tiny CPU tensors only.

Schema fixtures below are deliberately not serialized model checkpoints.  No
full codec, real weights, CDL sampling, training or model evaluation runs here.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import twc_v1_k8_task_loss_screen_config as cfg
import evaluate_twc_v1_k8_task_loss_screen as evaluation
import train_twc_v1_k8_task_loss_screen as training

TORCH_PRESENT = importlib.util.find_spec('torch') is not None
if TORCH_PRESENT:
    import torch


def calibration_fixture(beta=1.25, *, evidence='metadata-only fixture'):
    """A canonical identity fixture; not a claim that calibration was run."""
    record = dict(format=cfg.CALIBRATION_FORMAT, status='passed',
                  beta_target=beta, fixture_evidence=evidence)
    record['record_id'] = cfg.json_sha256(record)
    return record


def resume_fixture(arm='g4_joint', step=108, calibration=None):
    """Exercise the real metadata gate without pretending to hold tensors."""
    calibration = calibration or calibration_fixture()
    return dict(format=cfg.CHECKPOINT_FORMAT, arm=arm, seed=cfg.SEED,
                step=step, cumulative_updates=cfg.SOURCE_UPDATES + step,
                recipe=cfg.recipe(arm, calibration),
                config_fingerprint=cfg.fingerprint(arm, calibration),
                source=cfg.source_identity(arm),
                calibration=cfg.calibration_identity(calibration),
                codec_model={'schema_fixture': 'not numerical weights'},
                optimizer={'schema_fixture': 'not optimizer state'},
                scheduler={'last_epoch': step},
                rng={key: 'schema only' for key in
                     ('python', 'numpy', 'torch_cpu', 'torch_cuda')},
                data_protocol={'schema_fixture': True},
                ce_identity={'schema_fixture': True},
                code_sha256={'new_task_source.py': 'new experiment identity'})


class ConfigurationTests(unittest.TestCase):
    def test_source_pairs_share_identity_but_keep_old_and_new_stages_separate(self):
        self.assertEqual(cfg.source_identity('fdma_nmse'), cfg.source_identity('fdma_joint'))
        self.assertEqual(cfg.source_identity('g4_nmse'), cfg.source_identity('g4_joint'))
        self.assertNotEqual(cfg.source_identity('fdma_joint')['sha256'],
                            cfg.source_identity('g4_joint')['sha256'])
        source = cfg.source_identity('g4_joint')
        self.assertEqual(source['step'], 3024)
        self.assertEqual(source['cumulative_updates'], 16048)
        self.assertEqual(source['implementation_commit'], cfg.SOURCE_COMMIT)
        self.assertEqual(source['internal_source_commit'], cfg.INTERNAL_SOURCE_COMMIT)
        self.assertNotEqual(source['implementation_commit'], source['internal_source_commit'])
        self.assertNotEqual(cfg.NAMESPACE, cfg.previous.NAMESPACE)

    def test_training_addresses_continue_without_replaying_source_updates(self):
        for arm in cfg.ARMS:
            self.assertEqual(cfg.schedule(arm, 1)['global_step'], 16049)
            self.assertEqual(cfg.schedule(arm, 108)['global_step'], 16156)
            self.assertEqual(cfg.schedule(arm, 6048)['global_step'], 22096)
            self.assertEqual(sum(row['count'] for row in cfg.schedule_counts(arm)), 6048)
            for invalid in (0, 6049, True, 108.0):
                with self.subTest(arm=arm, step=invalid), self.assertRaises(ValueError):
                    cfg.schedule(arm, invalid)
        for step in (1, 108, 756, 6048):
            conditions = [cfg.schedule(arm, step) for arm in cfg.ARMS]
            self.assertEqual(len({x['global_step'] for x in conditions}), 1)
            self.assertEqual(len({x['gamma_db'] for x in conditions}), 1)
            self.assertEqual(len({x['pilot_energy_per_ue'] for x in conditions}), 1)
            self.assertEqual(cfg.training_noise_seed(step),
                             cfg.SEED*10_000_000 + conditions[0]['global_step'])
            self.assertEqual(conditions[0]['group_count'], 1)
            self.assertIsNone(conditions[0]['energy_ratio'])
            self.assertEqual(conditions[2]['group_count'], 4)
            self.assertIn(tuple(conditions[2]['energy_ratio']), ((1, 3), (1, 1)))

    def test_single_calibration_identity_binds_beta_and_full_record_content(self):
        first = calibration_fixture()
        other_beta = calibration_fixture(2.5)
        other_evidence = calibration_fixture(evidence='different frozen-source evidence')
        for arm in cfg.ARMS:
            first_recipe = cfg.recipe(arm, first)
            self.assertEqual(first_recipe['calibration']['record_id'], first['record_id'])
            self.assertEqual(first_recipe['beta_target'], 1.25)
            self.assertNotEqual(cfg.fingerprint(arm, first), cfg.fingerprint(arm, other_beta))
            self.assertNotEqual(cfg.fingerprint(arm, first), cfg.fingerprint(arm, other_evidence))
            self.assertNotEqual(cfg.fingerprint(arm, first), cfg.fingerprint(arm))
        edited = deepcopy(first)
        edited['beta_target'] = 2.5
        with self.assertRaises(ValueError):
            cfg.calibration_identity(edited)
        for beta in (0., -1., float('nan'), float('inf'), True):
            invalid = deepcopy(first)
            invalid['beta_target'] = beta
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                cfg.calibration_identity(invalid)

    def test_loss_weight_uses_local_step_and_one_joint_warmup(self):
        calibration = calibration_fixture()
        for arm in cfg.ARMS:
            expected = 1.25 if cfg.is_joint(arm) else 0.
            self.assertEqual(cfg.loss_weight(arm, 378, calibration), expected/2)
            self.assertEqual(cfg.loss_weight(arm, 756, calibration), expected)
            self.assertEqual(cfg.loss_weight(arm, 6048, calibration), expected)
        self.assertEqual(cfg.WARMUP_UPDATES, 756)

    def test_resume_metadata_accepts_json_roundtrip_without_confusing_source_commit(self):
        calibration = calibration_fixture()
        payload = json.loads(json.dumps(resume_fixture(calibration=calibration)))
        self.assertEqual(cfg.validate_resume_metadata(payload, 'g4_joint', calibration,
                        code_sha256=payload['code_sha256']), 108)
        payload['step'] = 6048
        payload['cumulative_updates'] = 22096
        payload['scheduler']['last_epoch'] = 6048
        self.assertEqual(cfg.validate_resume_metadata(payload, 'g4_joint', calibration), 6048)

    def test_resume_rejects_changed_calibration_source_config_state_and_schedule(self):
        calibration = calibration_fixture()
        base = resume_fixture(calibration=calibration)
        mutations = {
            'missing calibration': lambda x: x.pop('calibration'),
            'other arm': lambda x: x.update(arm='fdma_joint'),
            'wrong source sha': lambda x: x['source'].update(sha256='0'*64),
            'wrong source implementation': lambda x: x['source'].update(implementation_commit='new head'),
            'wrong recipe': lambda x: x['recipe'].update(learning_rate=1e-3),
            'missing optimizer': lambda x: x.pop('optimizer'),
            'empty optimizer': lambda x: x.update(optimizer={}),
            'missing CUDA RNG': lambda x: x['rng'].pop('torch_cuda'),
            'wrong scheduler': lambda x: x['scheduler'].update(last_epoch=0),
            'wrong cumulative step': lambda x: x.update(cumulative_updates=108),
            'non-save boundary': lambda x: x.update(step=109, cumulative_updates=16157),
            'legacy calibration identity': lambda x: x['calibration'].update(record_id='0'*64),
        }
        for label, mutate in mutations.items():
            payload = deepcopy(base)
            mutate(payload)
            with self.subTest(change=label), self.assertRaises(ValueError):
                cfg.validate_resume_metadata(payload, 'g4_joint', calibration)
        with self.assertRaises(ValueError):
            cfg.validate_resume_metadata(base, 'g4_joint', calibration_fixture(2.))
        with self.assertRaises(ValueError):
            cfg.validate_resume_metadata(base, 'g4_joint', calibration,
                                         code_sha256={'other.py': 'changed'})

    def test_output_guards_refuse_overwrite_and_unowned_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'task-loss-only'
            child = root/'training'/'g4_joint'
            self.assertEqual(cfg.prepare_output(child, root=root), child.resolve())
            with self.assertRaises(FileExistsError):
                cfg.prepare_output(child, root=root)
            with self.assertRaises(FileNotFoundError):
                cfg.prepare_output(child, root=root, resume=True)
            marker = child/'checkpoint_latest.pt'
            marker.write_bytes(b'schema marker only, not weights')
            self.assertEqual(cfg.prepare_output(child, root=root, resume=True), child.resolve())
            self.assertEqual(marker.read_bytes(), b'schema marker only, not weights')
            for forbidden in (root, Path(temporary)/'outside'):
                with self.subTest(path=forbidden), self.assertRaises(ValueError):
                    cfg.prepare_output(forbidden, root=root)


def sample_fixture(count=8):
    """Correlated, signed synthetic rates with two complete four-sample batches."""
    rows = []
    interaction = (-4.,)*4 + (-2.,)*4
    for method, group, policy, gamma in evaluation.conditions():
        step, cumulative = evaluation.method_steps(method)
        factor = (1., 3., 5., 2., 4., 6.)[cfg.GAMMAS.index(gamma)]
        for i in range(count):
            common = 40. + .1*i + .01*gamma
            offsets = dict(fdma_nmse=0., fdma_joint=2., g4_nmse=1.,
                           g4_joint=3.+interaction[i]*factor,
                           source_fdma=-.5, source_g4=.5)
            rate = common + offsets[method]
            row = dict(arm=method, group_count=group, energy_policy=policy,
                       gamma_db=gamma, seed=cfg.SEED, checkpoint_step=step,
                       cumulative_updates=cumulative, sample_index=i,
                       batch_index=i//4, pilot_energy_per_ue=16., downlink_snr_db=25.,
                       channel_fingerprint=f'{i//4:064x}', dl_nmse=.4+.01*i,
                       sum_rate_bps_hz=rate, desired_signal_power=10.+i,
                       interference_power=.2+.001*i, data_energy_per_ue=48.,
                       frame_energy_per_ue=64., rzf_power_mean=1.,
                       energy_max_abs_error=0., rzf_power_max_abs_error=0.)
            row.update({key: rate/8 for key in evaluation.USER_RATE_METRICS})
            rows.append(row)
    return rows


class PairedStatisticsTests(unittest.TestCase):
    def test_conditions_cover_four_continuations_two_sources_without_duplicate_fdma(self):
        samples = sample_fixture()
        summary, indexed = evaluation.summarize_samples(samples, count=8)
        self.assertEqual(len(summary), 54)
        self.assertEqual(len(samples), 54*8)
        self.assertEqual(len(indexed), 54)
        for method in cfg.METHODS:
            rows = [row for row in summary if row['arm'] == method]
            self.assertEqual(len(rows), 6 if evaluation.method_is_fdma(method) else 12)
            self.assertTrue(all(row['num_samples'] == 8 and row['num_batches'] == 2 for row in rows))
            expected = (3024, 16048) if method.startswith('source_') else (6048, 22096)
            self.assertEqual({(row['checkpoint_step'], row['cumulative_updates']) for row in rows}, {expected})
        self.assertTrue(all(not row['test_used'] for row in summary))

    def test_interaction_computed_from_raw_samples_retains_negative_delta_and_batch_ci(self):
        paired = evaluation.paired_statistics(sample_fixture(), count=8)
        self.assertEqual(len(paired), 13*8*4)
        row = next(row for row in paired if row['contrast'] == 'gap_change_hybrid_1_3'
                   and row['metric'] == 'sum_rate_bps_hz' and row['estimand'] == 'low_0_5_10')
        expected = evaluation.statistics_for([-12.]*4 + [-6.]*4)
        expected_batch = evaluation.statistics_for([-12., -6.])
        for name, value in expected.items():
            self.assertAlmostEqual(row['sample_'+name], value, places=11)
        for name, value in expected_batch.items():
            self.assertAlmostEqual(row['batch_'+name], value, places=11)
        self.assertEqual(row['sample_mean'], -9.)
        self.assertEqual(row['batch_mean'], -9.)
        self.assertGreater(row['batch_se'], row['sample_se'])
        self.assertEqual((row['num_samples'], row['num_batches']), (8, 2))
        self.assertTrue(row['interaction_from_raw_samples'])
        all_six = next(item for item in paired if item['contrast'] == row['contrast']
                       and item['metric'] == row['metric'] and item['estimand'] == 'all_six')
        self.assertAlmostEqual(all_six['sample_mean'], -10.5)

    def test_pairing_is_by_complete_sample_identity_not_input_order(self):
        original = sample_fixture()
        shuffled = list(reversed(original))
        self.assertEqual(evaluation.paired_statistics(original, count=8),
                         evaluation.paired_statistics(shuffled, count=8))

    def test_rejects_duplicate_missing_nonfinite_unpaired_and_wrong_source_rows(self):
        samples = sample_fixture()
        scenarios = {
            'duplicate': samples+[deepcopy(samples[0])],
            'missing': samples[:-1],
        }
        for name, field, value in (
                ('wrong batch', 'batch_index', 1),
                ('changed actual channels', 'channel_fingerprint', 'f'*64),
                ('nonfinite metric', 'dl_nmse', float('nan')),
                ('changed per-user rate', evaluation.USER_RATE_METRICS[0], 0.)):
            changed = deepcopy(samples)
            changed[0][field] = value
            scenarios[name] = changed
        changed = deepcopy(samples)
        source_row = next(row for row in changed if row['arm'] == 'source_fdma')
        source_row['checkpoint_step'] = 6048
        scenarios['source mistaken for new checkpoint'] = changed
        for name, rows in scenarios.items():
            with self.subTest(change=name), self.assertRaises(ValueError):
                evaluation.paired_statistics(rows, count=8)


def calibration_rows():
    records = []
    for method, nmse_norm, rate_norm in (('source_fdma', 2., 1.), ('source_g4', 18., 3.)):
        for offset in range(12):
            records.append(dict(source_key=method, batch_index=500000+offset,
                gamma_db=cfg.GAMMAS[offset%6], nmse_grad_norm=nmse_norm,
                rate_grad_norm=rate_norm, sample_fingerprint=f'{offset:064x}',
                pilot_sha256=f'{offset+12:064x}', data_sha256=f'{offset+24:064x}'))
    return records


def complete_calibration_fixture():
    """Full schema-only calibration evidence, with no claimed real source file."""
    rows = calibration_rows()
    record = dict(format=cfg.CALIBRATION_FORMAT, status='passed', records=rows,
        **training.calibration_from_records(rows),
        implementation_commit=cfg.SOURCE_COMMIT, internal_source_commit=cfg.INTERNAL_SOURCE_COMMIT,
        sources={method:training.source_identity(method) for method in training.SOURCE_METHODS},
        code=dict(commit='new experiment identity', sha256={'source.py':'a'*64}),
        data_protocol={'schema_fixture': True}, frozen_files={'metadata-only fixture':'b'*64},
        state_freeze={method:{'passed':True} for method in training.SOURCE_METHODS})
    record['record_id'] = cfg.json_sha256(record)
    return record


def source_schema_fixture(method='source_g4'):
    """The real source metadata gate is tested without reading model weights."""
    identity = training.source_identity(method)
    source_cfg = training.source_cfg
    arm = identity['arm']
    archived_hashes = {'schema_fixture_source.py':'c'*64}
    source_code = dict(commit=cfg.SOURCE_COMMIT, sha256=archived_hashes)
    common = dict(recipe=source_cfg.recipe(arm), source=source_cfg.source_identity(arm),
                  ce_identity={'schema_fixture': True})
    payload = dict(**deepcopy(common), format=source_cfg.CHECKPOINT_FORMAT, arm=arm,
        seed=cfg.SEED, step=3024, cumulative_updates=16048,
        config_fingerprint=source_cfg.fingerprint(arm),
        codec_model={'schema_fixture': 'not weights'}, optimizer={'schema_fixture':True},
        scheduler={'last_epoch':3024}, code_sha256=archived_hashes,
        rng={key:'schema only' for key in ('python','numpy','torch_cpu','torch_cuda')},
        data_protocol={'schema_fixture':True})
    manifest = dict(**deepcopy(common), format=source_cfg.CHECKPOINT_FORMAT, status='completed',
        arm=arm, seed=cfg.SEED, step=3024, cumulative_updates=16048,
        config_fingerprint=source_cfg.fingerprint(arm), checkpoint_sha256=identity['sha256'],
        source_commit=cfg.INTERNAL_SOURCE_COMMIT, new_experiment_commit=cfg.SOURCE_COMMIT,
        code=source_code)
    initial = dict(**deepcopy(common), code=source_code, source_commit=cfg.INTERNAL_SOURCE_COMMIT,
                   new_experiment_commit=cfg.SOURCE_COMMIT, new_optimizer=True)
    return payload, manifest, initial, archived_hashes


class CalibrationAndSourceTests(unittest.TestCase):
    def test_beta_is_ratio_of_pooled_medians_not_average_of_layout_betas(self):
        record = training.calibration_from_records(calibration_rows())
        self.assertEqual(record['median_nmse_grad_norm'], 10.)
        self.assertEqual(record['median_rate_grad_norm'], 2.)
        self.assertEqual(record['beta_target'], 1.25)
        independent_layout_average = (.25*2/1 + .25*18/3)/2
        self.assertNotEqual(record['beta_target'], independent_layout_average)
        self.assertEqual(record, training.calibration_from_records(list(reversed(calibration_rows()))))

    def test_calibration_rejects_wrong_coverage_norms_gamma_and_unpaired_inputs(self):
        rows = calibration_rows()
        invalid = {'missing address':rows[:-1], 'duplicated address':rows[:-1]+[deepcopy(rows[0])]}
        for label, key, value in (
                ('wrong gamma', 'gamma_db', 25.), ('zero gradient', 'nmse_grad_norm', 0.),
                ('nonfinite gradient', 'rate_grad_norm', float('inf')),
                ('boolean gradient', 'nmse_grad_norm', True),
                ('different shared data', 'data_sha256', 'f'*64),
                ('invalid SHA characters', 'pilot_sha256', 'z'*64)):
            changed = deepcopy(rows)
            changed[0][key] = value
            invalid[label] = changed
        for label, values in invalid.items():
            with self.subTest(change=label), self.assertRaises(ValueError):
                training.calibration_from_records(values)

    def test_complete_record_checks_hash_and_semantics_even_after_rehashing(self):
        record = complete_calibration_fixture()
        self.assertEqual(training.validate_calibration_record(record), cfg.calibration_identity(record))
        tampered = deepcopy(record)
        tampered['records'][0]['nmse_grad_norm'] += 1
        with self.assertRaises(ValueError):
            training.validate_calibration_record(tampered)
        for label, mutate in (
                ('beta changed', lambda x: x.update(beta_target=99.)),
                ('source changed', lambda x: x['sources']['source_g4'].update(sha256='0'*64)),
                ('state changed', lambda x: x['state_freeze']['source_fdma'].update(passed=False)),
                ('missing code', lambda x: x['code'].update(sha256={})),
                ('missing frozen evidence', lambda x: x.update(frozen_files={})),
                ('unshared calibration inputs', lambda x: x['records'][0].update(data_sha256='f'*64))):
            changed = deepcopy(record)
            mutate(changed)
            changed.pop('record_id')
            changed['record_id'] = cfg.json_sha256(changed)
            with self.subTest(change=label), self.assertRaises(ValueError):
                training.validate_calibration_record(changed)

    def test_actual_source_gate_accepts_json_roundtrip_and_keeps_commits_distinct(self):
        for method in training.SOURCE_METHODS:
            payload, manifest, initial, hashes = source_schema_fixture(method)
            payload, manifest, initial = json.loads(json.dumps((payload, manifest, initial)))
            with patch.object(training, 'source_code_record', side_effect=AssertionError('new HEAD is not source identity')):
                result = training.validate_source_metadata(payload, manifest, initial, method, archived_hashes=hashes)
            self.assertTrue(result['new_HEAD_may_differ'])
            self.assertEqual(result['implementation_commit'], cfg.SOURCE_COMMIT)
            self.assertEqual(result['internal_source_commit'], cfg.INTERNAL_SOURCE_COMMIT)

    def test_actual_source_gate_rejects_real_source_changes_not_new_head(self):
        base = source_schema_fixture()
        changes = (
            ('wrong final local step', lambda p,m,i:p.update(step=2916)),
            ('wrong cumulative step', lambda p,m,i:p.update(cumulative_updates=13024)),
            ('wrong seed', lambda p,m,i:p.update(seed=0)),
            ('wrong source weight', lambda p,m,i:m.update(checkpoint_sha256='0'*64)),
            ('wrong implementation commit', lambda p,m,i:m.update(new_experiment_commit='other')),
            ('wrong internal chain', lambda p,m,i:i.update(source_commit='other')),
            ('wrong recipe', lambda p,m,i:p['recipe'].update(learning_rate=9.)),
            ('wrong source path', lambda p,m,i:p['source'].update(path='other.pt')),
            ('wrong source code', lambda p,m,i:p.update(code_sha256={'other.py':'d'*64})),
        )
        for label, mutate in changes:
            payload, manifest, initial, hashes = deepcopy(base)
            mutate(payload, manifest, initial)
            with self.subTest(change=label), self.assertRaises(ValueError):
                training.validate_source_metadata(payload, manifest, initial, 'source_g4', archived_hashes=hashes)

    def test_immutable_calibration_writer_cannot_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'calibration.json'
            record = complete_calibration_fixture()
            inside_root = cfg.inside_root
            with patch.object(training.cfg, 'inside_root', side_effect=lambda value:inside_root(value, root=root)):
                training.write_once_json(path, record)
                original = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    training.write_once_json(path, calibration_fixture(99.))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(original), record)


@unittest.skipUnless(TORCH_PRESENT, 'PyTorch unavailable; tiny CPU tensor tests not executed')
class TinyTensorTests(unittest.TestCase):
    def test_gradient_norm_includes_missing_as_zero_and_rejects_invalid_effective_vectors(self):
        gradients = [None, torch.tensor([3., 4.]), torch.zeros(2), None]
        self.assertEqual(training.gradient_l2_norm(torch, gradients), 5.)
        for gradients in ([None], [torch.zeros(3), None], [torch.tensor([float('nan')])],
                          [torch.tensor([float('inf')])], [True]):
            with self.subTest(gradients=gradients), self.assertRaises(ValueError):
                training.gradient_l2_norm(torch, gradients)

    def test_loss_preserves_nmse_scale_rate_sign_warmup_and_both_gradients(self):
        calibration = calibration_fixture()
        for arm in cfg.ARMS:
            for step in (1, 378, 756, 6048):
                nmse = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
                rate = torch.tensor(19., dtype=torch.float64, requires_grad=True)
                loss = training.task_loss(nmse, rate, arm, step, calibration)
                beta = cfg.loss_weight(arm, step, calibration)
                self.assertAlmostEqual(loss.item(), .7-beta*19., places=12)
                nmse_grad, rate_grad = torch.autograd.grad(loss, (nmse, rate), allow_unused=True)
                self.assertEqual(nmse_grad.item(), 1.)
                if cfg.is_joint(arm):
                    self.assertAlmostEqual(rate_grad.item(), -beta, places=12)
                    self.assertLess(rate_grad.item(), 0.)
                else:
                    self.assertIs(loss, nmse)
                    self.assertIsNone(rate_grad)

    def test_actual_new_factory_delegates_strict_resume_without_migration_or_residual_reset(self):
        from types import SimpleNamespace
        class TinyV2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.key_window_residual = torch.nn.Linear(3, 2, bias=False)
                self.encoder.shared_window_residual = torch.nn.Linear(3, 2, bias=False)
                self.register_buffer('running_example', torch.tensor([7.]))
        original = TinyV2()
        with torch.no_grad():
            original.encoder.key_window_residual.weight.fill_(.125)
            original.encoder.shared_window_residual.weight.fill_(-.25)
        payload = {'codec_model': deepcopy(original.state_dict())}
        def tiny_strict_boundary(torch_module, device, source_arm, saved, *, initializing_source):
            # Real tensor/strict loading at the expensive model-construction
            # boundary; no fake Torch/Sionna or full network is instantiated.
            self.assertIs(torch_module, torch)
            self.assertFalse(initializing_source)
            self.assertIn(source_arm, ('ordered_fdma', 'ordered_g4'))
            model = TinyV2()
            training.previous.old.check_tensor_state(torch, model, saved['codec_model'])
            model.load_state_dict(saved['codec_model'], strict=True)
            return SimpleNamespace(codec_model=model.to(device))
        with patch.object(training.previous, 'build_chain', side_effect=tiny_strict_boundary) as boundary:
            for arm in cfg.ARMS + tuple(training.SOURCE_METHODS):
                chain = training.build_chain(torch, torch.device('cpu'), arm, payload)
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(chain.codec_model.state_dict()[key], value), key)
                self.assertFalse(boundary.call_args.kwargs['initializing_source'])
                expected = training.SOURCE_METHODS.get(arm, cfg.SOURCE_ARMS.get(arm))
                self.assertEqual(boundary.call_args.args[2], expected)
            for change in ('missing residual', 'wrong shape', 'wrong dtype', 'unknown key'):
                bad = deepcopy(payload)
                key = 'encoder.key_window_residual.weight'
                if change == 'missing residual': bad['codec_model'].pop(key)
                elif change == 'wrong shape': bad['codec_model'][key] = torch.zeros(9)
                elif change == 'wrong dtype': bad['codec_model'][key] = bad['codec_model'][key].double()
                else: bad['codec_model']['other.weight'] = torch.ones(1)
                with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                    training.build_chain(torch, torch.device('cpu'), 'g4_joint', bad)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(payload['codec_model'][key], value))


class ControllerResumeTests(unittest.TestCase):
    def test_actual_summary_preserves_partial_calibration_and_reports_failure_without_models(self):
        import contextlib
        import io
        import run_twc_v1_k8_task_loss_screen as controller
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);batch=root/'batches/batch_fixture';batch.mkdir(parents=True)
            (root/'last_batch.json').write_text(json.dumps({'directory':str(batch)}),encoding='utf-8')
            (batch/'batch_summary.json').write_text(json.dumps(dict(status='failed',error='fixture interrupted calibration',
                code={'commit':'fixture'},stages={})),encoding='utf-8')
            calibration=root/'calibration.json';calibration.write_bytes(b'{"incomplete":')
            before=calibration.read_bytes();stream=io.StringIO();guard=controller.cfg.inside_root
            with patch.object(controller.cfg,'OUTPUT_ROOT',root),patch.object(controller.cfg,'CALIBRATION_PATH',calibration), \
                    patch.object(controller.cfg,'inside_root',side_effect=lambda p:guard(p,root=root)), \
                    contextlib.redirect_stdout(stream):
                self.assertEqual(controller.print_summary(),0)
            self.assertIn('fixture interrupted calibration',stream.getvalue())
            self.assertIn('calibration_invalid=',stream.getvalue())
            self.assertIn('g4_joint: not_started',stream.getvalue())
            self.assertEqual(calibration.read_bytes(),before)

    def _calibration(self):
        calibration = complete_calibration_fixture()
        for item in calibration['state_freeze'].values():
            item['codec_model_sha256'] = 'd'*64
        calibration.pop('record_id')
        calibration['record_id'] = cfg.json_sha256(calibration)
        return calibration

    def _write_receipt(self, controller, root, arm, step, calibration, code):
        """Controller evidence fixture: the .pt marker is never torch-loaded."""
        destination = root/'training'/arm
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = destination/'checkpoint_latest.pt'
        checkpoint.write_bytes(f'schema marker only, not weights; step={step}'.encode())
        status = 'completed' if step == 6048 else 'paused'
        record = dict(format=cfg.CHECKPOINT_FORMAT, arm=arm, seed=cfg.SEED,
            source=cfg.source_identity(arm), code=code, calibration=cfg.calibration_identity(calibration),
            recipe=cfg.recipe(arm, calibration), config_fingerprint=cfg.fingerprint(arm, calibration),
            step=step, cumulative_updates=16048+step, status=status,
            checkpoint_sha256=controller.sha256(checkpoint))
        (destination/'manifest.json').write_text(json.dumps(record), encoding='utf-8')
        (destination/'initialization.json').write_text(json.dumps(
            dict(initialized_state_sha256='d'*64, migration=False)), encoding='utf-8')
        events = [dict(step=i, cumulative_updates=16048+i) for i in range(108, step+1, 108)]
        (destination/'progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events), encoding='utf-8')
        return destination

    def test_actual_controller_pauses108_continues6048_and_skips_verified_completion(self):
        import run_twc_v1_k8_task_loss_screen as controller
        calibration = self._calibration()
        code = dict(commit='new task-loss experiment', sha256={'schema_source.py':'e'*64})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []
            def child(folder, label, command):
                calls.append(command)
                if len(calls) == 1:
                    self.assertEqual(command[-2:], ['--stop-after', '108'])
                    step = 108
                else:
                    self.assertEqual(command[-2:], ['--resume', '--train'])
                    step = 6048
                self._write_receipt(controller, root, 'g4_joint', step, calibration, code)
                log = root/(label+'.log')
                log.write_text('mocked subprocess boundary, no training', encoding='utf-8')
                return log
            with patch.object(controller.cfg, 'OUTPUT_ROOT', root), \
                    patch.object(controller, 'source_code_record', return_value=code), \
                    patch.object(controller, '_child', side_effect=child):
                result = controller.run_arm(root, 'g4_joint', calibration)
                self.assertEqual(result['report']['step'], 6048)
                self.assertEqual(result['report']['cumulative_updates'], 22096)
                self.assertEqual(len(calls), 2)
                controller.run_arm(root, 'g4_joint', calibration)
                self.assertEqual(len(calls), 2)

    def test_paused_controller_resumes_without_reinitializing_or_repeating108(self):
        import run_twc_v1_k8_task_loss_screen as controller
        calibration = self._calibration()
        code = dict(commit='new task-loss experiment', sha256={'schema_source.py':'e'*64})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(controller.cfg, 'OUTPUT_ROOT', root), \
                    patch.object(controller, 'source_code_record', return_value=code):
                destination = self._write_receipt(controller, root, 'fdma_joint', 108, calibration, code)
                def child(folder, label, command):
                    self.assertEqual(command[-2:], ['--resume', '--train'])
                    self._write_receipt(controller, root, 'fdma_joint', 6048, calibration, code)
                    return root/'resume.log'
                with patch.object(controller, '_child', side_effect=child) as child_mock:
                    controller.run_arm(root, 'fdma_joint', calibration)
                    child_mock.assert_called_once()
                events = [json.loads(line) for line in (destination/'progress.jsonl').read_text().splitlines()]
                self.assertEqual([row['step'] for row in events], list(range(108, 6049, 108)))
                self.assertEqual(events[0]['cumulative_updates'], 16156)
                self.assertEqual(events[-1]['cumulative_updates'], 22096)

    def test_failed_smoke_retains_evidence_and_cannot_automatically_start_over(self):
        import run_twc_v1_k8_task_loss_screen as controller
        calibration = self._calibration()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root/'training'/'g4_joint'
            def failed_child(folder, label, command):
                self.assertEqual(command[-2:], ['--stop-after', '108'])
                destination.mkdir(parents=True)
                (destination/'failure.json').write_bytes(b'{"fixture":"retained smoke failure"}')
                raise RuntimeError('mocked child failure; no training ran')
            with patch.object(controller.cfg, 'OUTPUT_ROOT', root), \
                    patch.object(controller, '_child', side_effect=failed_child) as child:
                with self.assertRaises(RuntimeError):
                    controller.run_arm(root, 'g4_joint', calibration)
                child.assert_called_once()
                before = (destination/'failure.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    controller.run_arm(root, 'g4_joint', calibration)
                child.assert_called_once()
                self.assertEqual((destination/'failure.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
