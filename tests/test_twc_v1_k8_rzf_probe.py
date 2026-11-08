"""RZF-probe regressions: metadata and small CPU tensors, never model runs."""
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

from configs import twc_v1_k8_rzf_probe_config as cfg
import evaluate_twc_v1_k8_rzf_probe as evaluation

TORCH_PRESENT = importlib.util.find_spec('torch') is not None
if TORCH_PRESENT:
    import torch
    # Load the actual leaf implementation without utils.__init__, whose legacy
    # GPU setup mutates process settings and prints console-dependent symbols.
    _spec = importlib.util.spec_from_file_location('rzf_probe_production_downlink',
                                                  ROOT/'utils/twc_v1_k8_downlink.py')
    _downlink = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_downlink)
    design_rzf, downlink_nmse = _downlink.design_rzf, _downlink.downlink_nmse

RZF_SCALE_RTOL, RZF_SCALE_ATOL = 2e-5, 2e-6


def selection_fixture():
    """Canonical metadata-only selection identity; not measured model results."""
    record = dict(format=cfg.SELECTION_FORMAT, status='passed', grid=list(cfg.ALPHAS),
        selected_alphas={arm:.01 for arm in cfg.ARMS}, evidence='metadata fixture only')
    record['record_id'] = cfg.json_sha256(record)
    return record


def score_fixture():
    """Condition means exposing global-vs-pointwise and exact-vs-near ties."""
    rows = []
    for arm in cfg.ARMS:
        policies = ('fdma',) if cfg.is_fdma(arm) else ('hybrid_1_3','hybrid_1_1')
        for policy in policies:
            for gamma in cfg.GAMMAS:
                for alpha in cfg.ALPHAS:
                    rate = 0.
                    if arm == 'fdma_nmse':
                        rate = 10. if alpha == .003 else (50. if alpha == .03 and gamma == 0. else 0.)
                    elif arm == 'fdma_joint':
                        rate = 2.  # Exact global tie must choose the smallest candidate.
                    elif arm == 'g4_nmse':
                        if alpha == .03: rate = 20. if policy == 'hybrid_1_3' else 0.
                        elif alpha == .1: rate = 11.
                    elif arm == 'g4_joint':
                        if alpha == .01: rate = 5.
                        elif alpha == .03: rate = 5.+1e-12  # A real advantage, not an isclose tie.
                    rows.append(dict(arm=arm, group_count=1 if cfg.is_fdma(arm) else 4,
                        energy_policy=policy, gamma_db=gamma, alpha=alpha,
                        num_samples=64, sum_rate_bps_hz_mean=rate))
    return rows


def complete_selection_fixture():
    rows = score_fixture()
    record = dict(format=cfg.SELECTION_FORMAT, status='passed', grid=list(cfg.ALPHAS),
        calibration_summary=rows, **evaluation.select_global_alphas(rows),
        source_implementation_commit=cfg.SOURCE_COMMIT,
        sources={arm:cfg.source_identity(arm) for arm in cfg.ARMS},
        validation_addresses=list(range(4096,4112)), shared_inputs=[{'schema_fixture':i} for i in range(16)],
        code=dict(commit='new diagnostic identity', sha256={'probe.py':'a'*64}),
        frozen_files={'metadata fixture only':'b'*64}, freeze_checks={'passed':True})
    record['record_id'] = cfg.json_sha256(record)
    return record


class GlobalSelectionTests(unittest.TestCase):
    def test_actual_selector_uses_all_conditions_and_exact_not_approximate_ties(self):
        rows = score_fixture()
        result = evaluation.select_global_alphas(rows)
        self.assertEqual(len(rows), 324)
        self.assertEqual(result['selected_alphas'],
            dict(fdma_nmse=.003, fdma_joint=.0001, g4_nmse=.1, g4_joint=.03))
        self.assertEqual(len(result['scores']), 4*9)
        for score in result['scores']:
            self.assertEqual(score['equal_weight_conditions'], 6 if cfg.is_fdma(score['arm']) else 12)
        self.assertTrue(result['boundary_optima']['fdma_joint'])
        self.assertFalse(result['boundary_optima']['g4_nmse'])
        self.assertEqual(result, evaluation.select_global_alphas(list(reversed(rows))))

    def test_selector_refuses_missing_duplicate_nonfinite_or_unequal_condition_protocol(self):
        rows = score_fixture()
        invalid = {'missing condition':rows[:-1], 'duplicate condition':rows+[deepcopy(rows[0])]}
        for label, field, value in (
            ('wrong candidate', 'alpha', .02), ('wrong sample count', 'num_samples', 8),
            ('wrong gamma', 'gamma_db', 12.), ('wrong group', 'group_count', 4),
            ('nonfinite rate', 'sum_rate_bps_hz_mean', float('nan'))):
            changed = deepcopy(rows)
            changed[0][field] = value
            invalid[label] = changed
        for label, values in invalid.items():
            with self.subTest(change=label), self.assertRaises(ValueError):
                evaluation.select_global_alphas(values)

    def test_record_recomputes_selection_not_just_self_hash(self):
        record = complete_selection_fixture()
        self.assertEqual(evaluation.validate_selection_record(record), cfg.selection_identity(record))
        changed = deepcopy(record)
        changed['calibration_summary'][0]['sum_rate_bps_hz_mean'] += 1.
        with self.assertRaises(ValueError):evaluation.validate_selection_record(changed)
        for label, mutate in (
            ('wrong selected alpha', lambda x:x['selected_alphas'].update(g4_joint=.01)),
            ('changed source', lambda x:x['sources']['g4_joint'].update(sha256='0'*64)),
            ('mixed calibration/validation addresses', lambda x:x.update(validation_addresses=list(range(5120,5136)))),
            ('failed state freeze', lambda x:x['freeze_checks'].update(passed=False))):
            changed = deepcopy(record)
            mutate(changed)
            changed.pop('record_id')
            changed['record_id'] = cfg.json_sha256(changed)
            with self.subTest(change=label), self.assertRaises(ValueError):
                evaluation.validate_selection_record(changed)

    def test_locked_selection_write_and_reload_leave_original_evidence_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _,output,code = ControllerResumeTests._completed_calibration(root)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            evidence = root/'frozen_source_receipt.json'
            record = complete_selection_fixture()
            record.update(code=code, frozen_files=manifest['frozen_files'],
                calibration_directory=str(output),
                calibration_manifest_sha256=evaluation.sha256(output/'manifest.json'))
            record.pop('record_id')
            record['record_id'] = cfg.json_sha256(record)
            path = root/'alpha_selection.json'
            inside = cfg.inside_root
            with patch.object(cfg, 'inside_root', side_effect=lambda value:inside(value, root=root)), \
                 patch.object(cfg, 'SELECTION_PATH', path), \
                 patch.object(evaluation, 'source_code_record', return_value=record['code']):
                evaluation.write_once_json(path, record)
                before = (path.read_bytes(), evidence.read_bytes(), (output/'manifest.json').read_bytes())
                self.assertEqual(evaluation.load_selection(), record)
                self.assertEqual(evaluation.load_selection(), record)
                with self.assertRaises(FileExistsError):evaluation.write_once_json(path, selection_fixture())
                self.assertEqual((path.read_bytes(), evidence.read_bytes(), (output/'manifest.json').read_bytes()), before)
                evidence.write_bytes(b'{"schema_fixture":"changed"}')
                with self.assertRaises(ValueError):evaluation.load_selection()
                evidence.write_bytes(before[1])
                artifact = output/'alpha_global_scores.csv'
                original_artifact = artifact.read_bytes()
                artifact.write_bytes(b'changed calibration artifact')
                with self.assertRaises(ValueError):evaluation.load_selection()
                artifact.write_bytes(original_artifact)
                (output/'manifest.json').write_bytes(before[2]+b'\n')
                with self.assertRaises(ValueError):evaluation.load_selection()


def raw_fixture(count=8):
    """Correlated complete-MU records with negative tuned-minus-fixed effects."""
    rows = []
    selected = evaluation.select_global_alphas(score_fixture())['selected_alphas']
    for arm, mode, group, policy, gamma in evaluation.conditions():
        factor = (1.,3.,5.,2.,4.,6.)[cfg.GAMMAS.index(gamma)]
        for i in range(count):
            baseline = 100.+i*.1+gamma*.01
            e = -4. if i<4 else -2.
            offsets = dict(fdma_nmse=0., fdma_joint=2., g4_nmse=1., g4_joint=3.+e*factor)
            tuning = dict(fdma_nmse=.5, fdma_joint=1., g4_nmse=2.,
                          g4_joint=(-1. if i<4 else -3.)*factor)
            rate = baseline+offsets[arm]+(tuning[arm] if mode=='tuned' else 0.)
            alpha = .01 if mode=='fixed' else selected[arm]
            row = dict(arm=arm, rzf_mode=mode, group_count=group, energy_policy=policy,
                gamma_db=gamma, alpha=alpha, sample_index=i, batch_index=i//4, seed=cfg.SEED,
                channel_fingerprint=f'{i//4:064x}', pilot_energy_per_ue=16., downlink_snr_db=25.,
                dl_nmse=.2+i*.01, sum_rate_bps_hz=rate, desired_signal_power=rate/4,
                interference_power=.4+i*.001, hhat_energy_total=8192., hhat_fro_norm=8192.**.5,
                hhat_to_true_norm_ratio=2., hhat_tone_user_energy_mean=4.,
                relative_alpha_mean=alpha/4, relative_alpha_min=alpha/4, relative_alpha_max=alpha/4,
                data_energy_per_ue=48., frame_energy_per_ue=64., rzf_power_mean=1.,
                energy_max_abs_error=0., rzf_power_max_abs_error=0.)
            row.update({key:rate/8 for key in evaluation.USER_RATES})
            rows.append(row)
    return rows


class PairedDiagnosticsTests(unittest.TestCase):
    def test_all_conditions_and_raw_invariants_share_one_alpha_per_checkpoint(self):
        summary, indexed = evaluation.summarize_samples(raw_fixture(), count=8)
        self.assertEqual(len(summary), 72)
        self.assertEqual(len(indexed), 72)
        for arm in cfg.ARMS:
            selected = [row for row in summary if row['arm']==arm and row['rzf_mode']=='tuned']
            self.assertEqual(len(selected), 6 if cfg.is_fdma(arm) else 12)
            self.assertEqual(len({row['alpha'] for row in selected}), 1)
            self.assertTrue(all(row['num_samples']==8 and row['num_batches']==2 for row in selected))

    def test_signed_tuning_and_interaction_use_sample_averages_then_batch4(self):
        paired = evaluation.paired_statistics(raw_fixture(), count=8)
        self.assertEqual(len(paired), 20*8*4)
        tuning = next(row for row in paired if row['contrast']=='g4_joint_tuned_minus_fixed_hybrid_1_3'
                      and row['estimand']=='low_0_5_10' and row['metric']=='sum_rate_bps_hz')
        expected = evaluation.statistics_for([-3.]*4+[-9.]*4)
        expected_batch = evaluation.statistics_for([-3.,-9.])
        for name, value in expected.items():self.assertAlmostEqual(tuning['sample_'+name], value, places=11)
        for name, value in expected_batch.items():self.assertAlmostEqual(tuning['batch_'+name], value, places=11)
        self.assertEqual(tuning['sample_mean'], -6.)
        self.assertGreater(tuning['batch_se'], tuning['sample_se'])
        interaction = next(row for row in paired if row['contrast']=='gap_interaction_tuned_hybrid_1_3'
                      and row['estimand']=='low_0_5_10' and row['metric']=='sum_rate_bps_hz')
        self.assertAlmostEqual(interaction['sample_mean'], -17.5)
        self.assertEqual(evaluation.paired_statistics(list(reversed(raw_fixture())),count=8), paired)

    def test_missing_duplicates_alpha_reselection_and_changed_reconstruction_are_rejected(self):
        rows = raw_fixture()
        invalid = {'missing':rows[:-1], 'duplicate':rows+[deepcopy(rows[0])]}
        for label, field, value in (
            ('NMSE changed by alpha', 'dl_nmse', .9),
            ('Hhat energy changed by alpha', 'hhat_energy_total', 8193.),
            ('unpaired channels', 'channel_fingerprint', 'f'*64),
            ('wrong batch', 'batch_index', 1),
            ('nonfinite diagnostic', 'hhat_fro_norm', float('nan'))):
            changed = deepcopy(rows)
            row = next(row for row in changed if row['arm']=='g4_joint' and row['rzf_mode']=='tuned')
            row[field] = value
            invalid[label] = changed
        changed = deepcopy(rows)
        next(row for row in changed if row['arm']=='g4_joint' and row['rzf_mode']=='tuned')['alpha'] = .1
        invalid['per-condition alpha reselection'] = changed
        for label, records in invalid.items():
            with self.subTest(change=label), self.assertRaises(ValueError):
                evaluation.paired_statistics(records, count=8)


def original_beta_fixture():
    # Reuse the existing pure metadata fixture rather than duplicating an old
    # calibration test framework.  It does not read source/model files.
    from tests.test_twc_v1_k8_task_loss_screen import complete_calibration_fixture
    record = complete_calibration_fixture()
    for row in record['state_freeze'].values():row['codec_model_sha256'] = 'c'*64
    record.pop('record_id')
    record['record_id'] = evaluation.source_cfg.json_sha256(record)
    return record


def source_fixture(arm='g4_joint'):
    from tests.test_twc_v1_k8_task_loss_screen import resume_fixture
    beta = original_beta_fixture()
    hashes = {'accepted_task_source.py':'d'*64}
    code = dict(commit=cfg.SOURCE_COMMIT, sha256=hashes)
    payload = resume_fixture(arm=arm, step=6048, calibration=beta)
    payload.update(code_sha256=deepcopy(hashes), calibration_record=beta, calibration_sha256='e'*64)
    common = {key:deepcopy(payload[key]) for key in ('arm','seed','step','cumulative_updates','recipe',
        'source','calibration','config_fingerprint','ce_identity')}
    manifest = dict(**deepcopy(common), format=evaluation.source_cfg.CHECKPOINT_FORMAT,
        status='completed', code=code, checkpoint_sha256=cfg.source_identity(arm)['sha256'])
    initial = {key:deepcopy(common[key]) for key in ('recipe','source','calibration','ce_identity')}
    initial.update(code=code, new_optimizer=True, strict_loading=True, migration=False,
                   initialized_state_sha256='c'*64)
    return payload, manifest, initial, beta, hashes


def progress_fixture(beta, *, monitors=True):
    progress = {}
    for arm in cfg.ARMS:
        events = []
        for step in range(108,6049,108):
            monitor = None
            if monitors and (step==108 or step%756==0):
                monitor = [dict(batch=b,group_count=1 if cfg.is_fdma(arm) else 4,
                    nmse=.4-step*.00001,rate=20.+step*.001+b*.1) for b in range(4)]
            events.append(dict(step=step,cumulative_updates=16048+step,
                condition=evaluation.source_cfg.schedule(arm,step),nmse=1.-step*.00001,
                rate=100.+step*.001,beta=evaluation.source_cfg.loss_weight(arm,step,beta),monitor=monitor))
        progress[arm] = events
    return progress


class SourceAndTrendTests(unittest.TestCase):
    def test_actual_source_gate_accepts_saved_json_and_three_commit_layers_without_new_head(self):
        for arm in cfg.ARMS:
            payload, manifest, initial, beta, hashes = source_fixture(arm)
            payload, manifest, initial, beta = json.loads(json.dumps((payload,manifest,initial,beta)))
            with patch.object(evaluation,'source_code_record',side_effect=AssertionError('new HEAD is not source provenance')):
                result = evaluation.validate_source_metadata(payload,manifest,initial,arm,beta,
                    archived_hashes=hashes,calibration_sha256='e'*64)
            self.assertTrue(result['new_diagnostic_HEAD_may_differ'])
            self.assertEqual((result['implementation_commit'],result['internal_source_commit'],result['ancestor_source_commit']),
                             (cfg.SOURCE_COMMIT,cfg.INTERNAL_SOURCE_COMMIT,cfg.ANCESTOR_SOURCE_COMMIT))

    def test_actual_source_gate_rejects_real_step_weight_config_and_migration_changes(self):
        fixture = source_fixture()
        for label, mutate in (
            ('wrong source step',lambda p,m,i:p.update(step=5940)),
            ('wrong source weight',lambda p,m,i:m.update(checkpoint_sha256='0'*64)),
            ('wrong source implementation',lambda p,m,i:m['code'].update(commit=cfg.INTERNAL_SOURCE_COMMIT)),
            ('wrong recipe',lambda p,m,i:p['recipe'].update(learning_rate=1.)),
            ('window-state reinitialization',lambda p,m,i:i.update(initialized_state_sha256='0'*64)),
            ('migration enabled',lambda p,m,i:i.update(migration=True)),
            ('changed beta record',lambda p,m,i:p.update(calibration_sha256='0'*64))):
            payload,manifest,initial,beta,hashes = deepcopy(fixture)
            mutate(payload,manifest,initial)
            with self.subTest(change=label),self.assertRaises(ValueError):
                evaluation.validate_source_metadata(payload,manifest,initial,'g4_joint',beta,
                    archived_hashes=hashes,calibration_sha256='e'*64)

    def test_read_only_trends_keep_saved_training_conditions_and_fixed_monitor_separate(self):
        beta = original_beta_fixture()
        progress = progress_fixture(beta)
        before = deepcopy(progress)
        result = evaluation.analyze_training_trends(progress,beta)
        self.assertEqual(progress,before)
        self.assertEqual(len(result['training_rows']),4*56)
        for arm in cfg.ARMS:
            value = result['arms'][arm]
            self.assertEqual(len(value['training_by_condition']),1)
            self.assertEqual([row['step'] for row in value['last_four_monitor']],[3780,4536,5292,6048])
            self.assertAlmostEqual(value['last_four_rate_change'],2.268)
            for row in value['monitor']:
                self.assertEqual((row['gamma_db'],row['pilot_energy_per_ue'],row['samples']), (10.,16.,16))
                self.assertEqual(row['energy_policy'],'fdma' if cfg.is_fdma(arm) else 'hybrid_1_3')
            self.assertIn('not a full-SNR convergence test',value['limitation'])

    def test_insufficient_monitors_are_unknown_and_wrong_progress_protocol_is_rejected(self):
        beta = original_beta_fixture()
        no_monitor = evaluation.analyze_training_trends(progress_fixture(beta,monitors=False),beta)
        for result in no_monitor['arms'].values():
            self.assertEqual(result['last_four_monitor'],[])
            self.assertIsNone(result['last_four_rate_change'])
            self.assertIsNone(result['last_four_nmse_change'])
        for label, mutate in (
            ('missing update',lambda x:x['fdma_nmse'].pop()),
            ('mixed condition',lambda x:x['g4_joint'][0]['condition'].update(gamma_db=0.)),
            ('wrong monitor layout',lambda x:x['g4_joint'][0]['monitor'][0].update(group_count=1)),
            ('wrong logged beta',lambda x:x['g4_joint'][0].update(beta=99.))):
            progress = progress_fixture(beta)
            mutate(progress)
            with self.subTest(change=label),self.assertRaises(ValueError):
                evaluation.analyze_training_trends(progress,beta)


class ControllerResumeTests(unittest.TestCase):
    @staticmethod
    def _completed_calibration(root):
        """Scalar receipt fixture, never inference or a numerical checkpoint."""
        folder = root/'batches'/'batch_fixture'
        output = folder/'calibration_fixture'
        output.mkdir(parents=True)
        source_receipt = root/'frozen_source_receipt.json'
        source_receipt.write_bytes(b'{"schema_fixture":true}')
        code = dict(commit='new diagnostic identity',sha256={'probe.py':'a'*64})
        for name in evaluation.ARTIFACTS['calibration']:
            (output/name).write_text('schema fixture; no measured results\n',encoding='utf-8')
        summary = score_fixture()
        manifest = dict(status='completed',stage='calibration',code=code,freeze_checks={'passed':True},
            test_used=False,optimizer_updates=0,source_implementation_commit=cfg.SOURCE_COMMIT,
            validation_addresses=list(range(4096,4112)),shared_inputs=[{'schema_fixture':i} for i in range(16)],
            network_forward_calls=16*36,methods={arm:{'schema_fixture':True} for arm in cfg.ARMS},
            rzf_solves=16*36*9,output_sha256={name:evaluation.sha256(output/name) for name in evaluation.ARTIFACTS['calibration']},
            frozen_files={str(source_receipt):evaluation.sha256(source_receipt)},data_protocol={'schema_fixture':True},
            original_beta_identity={'schema_fixture':True},calibration_summary=summary,
            **evaluation.select_global_alphas(summary))
        (output/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        (folder/'calibration_pointer.json').write_text(json.dumps({'directory':str(output)}),encoding='utf-8')
        (root/'selection_attempt.json').write_bytes(b'{"schema_fixture":"preserved attempt"}')
        return folder,output,code

    def test_actual_controller_seals_completed_evidence_then_reuses_lock_without_reselection(self):
        import run_twc_v1_k8_rzf_probe as controller
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder,output,code = self._completed_calibration(root)
            inside = cfg.inside_root
            before = {str(path):path.read_bytes() for path in root.rglob('*') if path.is_file()}
            with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(cfg,'SELECTION_PATH',root/'alpha_selection.json'), \
                 patch.object(cfg,'inside_root',side_effect=lambda value:inside(value,root=root)), \
                 patch.object(evaluation,'source_code_record',return_value=code), \
                 patch.object(controller,'_child',side_effect=AssertionError('completed calibration must not rerun inference')):
                first = controller.selection_stage(folder)
                self.assertEqual(first['selected_alphas'],evaluation.select_global_alphas(score_fixture())['selected_alphas'])
                locked_bytes = cfg.SELECTION_PATH.read_bytes()
                with patch.object(evaluation,'seal_selection',side_effect=AssertionError('already locked selection must not be chosen again')):
                    second = controller.selection_stage(folder)
                self.assertEqual(first['record_id'],second['record_id'])
                self.assertEqual(cfg.SELECTION_PATH.read_bytes(),locked_bytes)
                for path,contents in before.items():self.assertEqual(Path(path).read_bytes(),contents)
                (output/'alpha_global_scores.csv').write_text('tampered artifact',encoding='utf-8')
                with self.assertRaises(ValueError):controller.selection_stage(folder)
                self.assertEqual(cfg.SELECTION_PATH.read_bytes(),locked_bytes)

    def test_partial_attempt_remains_evidence_and_cannot_automatically_recalibrate(self):
        import run_twc_v1_k8_rzf_probe as controller
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder,output,code = self._completed_calibration(root)
            manifest = json.loads((output/'manifest.json').read_text())
            manifest.update(status='failed',error='schema fixture failure')
            (output/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            before = (output/'manifest.json').read_bytes()
            inside = cfg.inside_root
            with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(cfg,'SELECTION_PATH',root/'alpha_selection.json'), \
                 patch.object(cfg,'inside_root',side_effect=lambda value:inside(value,root=root)), \
                 patch.object(evaluation,'source_code_record',return_value=code), \
                 patch.object(controller,'_child',side_effect=AssertionError('failed evidence must stop without new inference')):
                with self.assertRaises(RuntimeError):controller.selection_stage(folder)
                with patch.object(evaluation,'run_stage',side_effect=AssertionError('attempt guard must run before inference')):
                    with self.assertRaises(RuntimeError):evaluation.calibrate(root/'new_calibration')
                self.assertEqual((output/'manifest.json').read_bytes(),before)
                self.assertFalse(cfg.SELECTION_PATH.exists())


class ProbeConfigurationTests(unittest.TestCase):
    def test_four_frozen_sources_keep_three_commit_layers_and_terminal_budget(self):
        identities = [cfg.source_identity(arm) for arm in cfg.ARMS]
        self.assertEqual(len({row['sha256'] for row in identities}), 4)
        for arm, row in zip(cfg.ARMS, identities):
            self.assertEqual(row['arm'], arm)
            self.assertEqual((row['step'], row['cumulative_updates']), (6048, 22096))
            commits = (row['implementation_commit'], row['internal_source_commit'], row['ancestor_source_commit'])
            self.assertEqual(commits, (cfg.SOURCE_COMMIT, cfg.INTERNAL_SOURCE_COMMIT, cfg.ANCESTOR_SOURCE_COMMIT))
            self.assertEqual(len(set(commits)), 3)
            self.assertIn('/twc_v1_k8_task_loss_screen_v1/training/'+arm+'/', row['path'].replace('\\', '/'))

    def test_selection_identity_locks_one_global_alpha_and_complete_record(self):
        selection = selection_fixture()
        identity = cfg.selection_identity(json.loads(json.dumps(selection)))
        self.assertEqual(identity['selected_alphas'], {arm:.01 for arm in cfg.ARMS})
        self.assertEqual(tuple(cfg.ALPHAS), (.0001,.0003,.001,.003,.01,.03,.1,.3,1.))
        changed = deepcopy(selection)
        changed['selected_alphas']['g4_joint'] = .03
        with self.assertRaises(ValueError):
            cfg.selection_identity(changed)
        changed.pop('record_id')
        changed['record_id'] = cfg.json_sha256(changed)
        self.assertNotEqual(cfg.fingerprint(selection), cfg.fingerprint(changed))
        self.assertNotEqual(cfg.fingerprint(selection), cfg.fingerprint())
        for invalid in (True, .02, float('nan'), {'0dB':.01,'25dB':.03}):
            bad = deepcopy(selection)
            bad['selected_alphas']['g4_joint'] = invalid
            with self.subTest(alpha=invalid), self.assertRaises(ValueError):
                cfg.selection_identity(bad)

    def test_calibration_and_final_validation_use_disjoint_declared_addresses_and_noise(self):
        calibration = set(range(cfg.CALIBRATION_START_INDEX, cfg.CALIBRATION_START_INDEX+cfg.CALIBRATION_BATCHES))
        validation = set(range(cfg.VALIDATION_START_INDEX, cfg.VALIDATION_START_INDEX+cfg.VALIDATION_BATCHES))
        self.assertEqual(calibration, set(range(4096, 4112)))
        self.assertEqual(validation, set(range(5120, 5184)))
        self.assertTrue(calibration.isdisjoint(validation))
        self.assertEqual(cfg.CALIBRATION_SAMPLES, 64)
        self.assertEqual(cfg.VALIDATION_SAMPLES, 256)
        noise = {}
        for stage, batches in (('calibration', 16), ('evaluation', 64)):
            noise[stage] = {cfg.noise_seed(stage,batch,gamma,kind)
                for batch in range(batches) for gamma in range(6) for kind in ('pilot','data')}
            self.assertEqual(len(noise[stage]), batches*6*2)
        self.assertTrue(noise['calibration'].isdisjoint(noise['evaluation']))
        recipe = cfg.recipe(selection_fixture())
        self.assertEqual(recipe['optimizer_updates'], 0)
        self.assertFalse(recipe['test_used'])
        self.assertFalse(recipe['validation']['reselection'])
        self.assertTrue(recipe['H_hat_reused_for_alpha_grid'])


@unittest.skipUnless(TORCH_PRESENT, 'PyTorch unavailable; synthetic RZF algebra not executed')
class RZFAlgebraTests(unittest.TestCase):
    def test_production_normalized_rzf_scale_identity_without_changing_reconstruction(self):
        generator = torch.Generator(device='cpu').manual_seed(20261108)
        real = torch.randn(1, 8, 32, 256, generator=generator)
        imaginary = torch.randn(1, 8, 32, 256, generator=generator)
        H = torch.complex(real, imaginary)
        original = H.clone()
        alpha = .01
        errors = []
        for scale in (.5, 1.7, 2.):
            scaled, scaled_power = design_rzf(scale*H, alpha)
            equivalent, equivalent_power = design_rzf(H, alpha/scale**2)
            max_abs = float((scaled-equivalent).abs().max())
            max_rel = float(((scaled-equivalent).abs()/equivalent.abs().clamp_min(1e-12)).max())
            evidence = dict(scale=scale, alpha=alpha, equivalent_alpha=alpha/scale**2,
                rtol=RZF_SCALE_RTOL, atol=RZF_SCALE_ATOL, max_abs=max_abs, max_rel=max_rel)
            self.assertTrue(torch.allclose(scaled, equivalent, rtol=RZF_SCALE_RTOL, atol=RZF_SCALE_ATOL), evidence)
            for power in (scaled_power, equivalent_power):
                self.assertLessEqual(float((power-1.).abs().max()), 1e-6)
            errors.append(evidence)
        self.assertTrue(torch.equal(H, original))
        # Alpha acts after CSI reconstruction: its scan cannot change this NMSE.
        reference = .9*H
        nmse_before = downlink_nmse(H, reference)
        for candidate in (1e-5, .01, 10.):
            design_rzf(H, candidate)
            self.assertTrue(torch.equal(downlink_nmse(H, reference), nmse_before))
        print('production CPU normalized-RZF scaling identity', errors)

    def test_cached_hhat_scoring_uses_original_solver_and_preserves_nmse_and_norm_diagnostics(self):
        generator = torch.Generator(device='cpu').manual_seed(17)
        Hhat = torch.complex(torch.randn(1,8,32,256,generator=generator),
                             torch.randn(1,8,32,256,generator=generator))
        Htrue = .5*Hhat
        before = Hhat.clone()
        with patch.object(_downlink,'design_rzf',wraps=design_rzf) as solver:
            results = evaluation.score_cached_hhat(torch,_downlink,Hhat,Htrue,(.01,.03,.01))
        self.assertEqual(set(results),{.01,.03})
        self.assertEqual(solver.call_count,2)
        for call in solver.call_args_list:self.assertIs(call.args[0],Hhat)
        self.assertTrue(torch.equal(Hhat,before))
        for field in evaluation.INVARIANT:
            self.assertTrue(torch.equal(results[.01][field],results[.03][field]),field)
        self.assertTrue(torch.allclose(results[.01]['hhat_to_true_norm_ratio'],torch.tensor([2.])))
        self.assertTrue(torch.allclose(results[.01]['hhat_energy_total'],Hhat.abs().square().sum((1,2,3))))
        self.assertTrue(torch.allclose(results[.03]['relative_alpha_mean'],3*results[.01]['relative_alpha_mean']))
        for row in results.values():
            self.assertLessEqual(float(row['rzf_power_max_abs_error'].max()),1e-5)
            self.assertTrue(torch.allclose(sum(row[key] for key in evaluation.USER_RATES),row['sum_rate_bps_hz']))

    def test_actual_pilot_only_observation_excludes_data_and_rejects_overlap(self):
        rx = torch.ones(1,32,256,dtype=torch.complex64)
        data = torch.ones_like(rx)
        indices = (0,8,16)
        data[...,list(indices)] = 0
        output = dict(Y_rx=rx,tx={'Y':data})
        actual = evaluation.pilot_only_observation(torch,output,indices)
        expected = torch.zeros_like(rx)
        expected[...,list(indices)] = rx[...,list(indices)]
        self.assertTrue(torch.equal(actual,expected))
        rx[...,25] = 900.+300.j
        self.assertTrue(torch.equal(evaluation.pilot_only_observation(torch,output,indices),expected))
        data[...,8] = 1
        with self.assertRaises(ValueError):evaluation.pilot_only_observation(torch,output,indices)


if __name__ == '__main__':
    unittest.main()
