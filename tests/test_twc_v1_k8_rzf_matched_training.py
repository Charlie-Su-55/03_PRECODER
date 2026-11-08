"""Matched-RZF batch checks using metadata and tiny CPU tensors only.

No full codec/CE construction, real checkpoint, CDL sampling or training runs.
Metadata markers are explicitly schema fixtures, never numerical checkpoints.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from configs import twc_v1_k8_rzf_matched_training_config as cfg
import evaluate_twc_v1_k8_rzf_matched_training as evaluation
import train_twc_v1_k8_rzf_matched_training as training

TORCH_PRESENT = importlib.util.find_spec('torch') is not None
NUMPY_PRESENT = importlib.util.find_spec('numpy') is not None
if TORCH_PRESENT:import torch
if NUMPY_PRESENT:import numpy as np


def beta_fixture(beta=1.25, *, provenance='new beta2 metadata fixture'):
    record = dict(format=cfg.CALIBRATION_FORMAT,status='passed',beta_target=beta,
                  training_alpha_provenance=provenance)
    record['record_id'] = cfg.json_sha256(record)
    return record


def resume_fixture(arm='g4_joint',step=108,calibration=None):
    calibration = calibration or beta_fixture()
    return dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=step,
        cumulative_updates=22096+step,recipe=cfg.recipe(arm,calibration),
        config_fingerprint=cfg.fingerprint(arm,calibration),source=cfg.source_identity(arm),
        calibration=cfg.calibration_identity(calibration),codec_model={'schema_fixture':'not weights'},
        optimizer={'schema_fixture':True},scheduler={'last_epoch':step},
        rng={key:'schema only' for key in ('python','numpy','torch_cpu','torch_cuda')},
        data_protocol={'schema_fixture':True},ce_identity={'schema_fixture':True},
        code_sha256={'new_matched_source.py':'new experiment identity'})


class MatchedConfigurationTests(unittest.TestCase):
    def test_both_objectives_start_same_old_nmse_state_and_keep_lineage_separate(self):
        for family in ('fdma','g4'):
            nmse = cfg.source_identity(family+'_nmse')
            joint = cfg.source_identity(family+'_joint')
            self.assertEqual(nmse,joint)
            self.assertEqual(nmse['arm'],family+'_nmse')
            self.assertNotIn(family+'_joint',nmse['path'])
            self.assertEqual(nmse,cfg.source_identity('source_'+family))
            self.assertEqual((nmse['step'],nmse['cumulative_updates']),(6048,22096))
            self.assertEqual((nmse['implementation_commit'],nmse['diagnostic_commit'],
                nmse['internal_source_commit'],nmse['ancestor_source_commit']),
                (cfg.SOURCE_COMMIT,cfg.DIAGNOSTIC_COMMIT,cfg.INTERNAL_SOURCE_COMMIT,cfg.ANCESTOR_SOURCE_COMMIT))
        self.assertEqual(len(set(cfg.SOURCE_SHA256)),2)
        self.assertEqual(cfg.training_alpha('fdma_joint'),.03)
        self.assertEqual(cfg.training_alpha('source_fdma'),.03)
        self.assertEqual(cfg.training_alpha('g4_joint'),.1)
        self.assertEqual(cfg.training_alpha('source_g4'),.1)

    def test_budget_continues_cumulative_schedule_with_distinct_data_and_noise_stages(self):
        for arm in cfg.ARMS:
            self.assertEqual(cfg.schedule(arm,1)['global_step'],22097)
            self.assertEqual(cfg.schedule(arm,108)['global_step'],22204)
            self.assertEqual(cfg.schedule(arm,12096)['global_step'],34192)
            self.assertEqual(sum(row['count'] for row in cfg.schedule_counts(arm)),12096)
            for step in (0,12097,True):
                with self.subTest(arm=arm,step=step),self.assertRaises(ValueError):cfg.schedule(arm,step)
        stages = [('beta_calibration',12),('monitor',16),('alpha_calibration',16),('evaluation',64)]
        used = set()
        for stage,batches in stages:
            values = {cfg.noise_seed(stage,batch,gamma,kind) for batch in range(batches)
                      for gamma in range(6) for kind in ('pilot','data')}
            self.assertEqual(len(values),batches*6*2)
            self.assertTrue(used.isdisjoint(values))
            used.update(values)
        self.assertEqual((cfg.CALIBRATION_START_INDEX,cfg.CALIBRATION_BATCHES),(500100,12))
        self.assertEqual((cfg.MONITOR_START_INDEX,cfg.MONITOR_BATCHES),(8960,16))
        self.assertEqual((cfg.ALPHA_CALIBRATION_START_INDEX,cfg.ALPHA_CALIBRATION_BATCHES),(6144,16))
        self.assertEqual((cfg.VALIDATION_START_INDEX,cfg.VALIDATION_BATCHES),(7168,64))
        self.assertEqual(tuple(cfg.PROGRESS_VALIDATION_STEPS),(3024,6048,9072,12096))

    def test_fresh_beta2_binds_full_provenance_and_ramps_within_budget(self):
        calibration = beta_fixture()
        for arm in cfg.ARMS:
            self.assertNotEqual(cfg.fingerprint(arm,calibration),cfg.fingerprint(arm,beta_fixture(2.)))
            self.assertNotEqual(cfg.fingerprint(arm,calibration),cfg.fingerprint(arm,beta_fixture(provenance='different old-alpha identity')))
            expected = 1.25 if cfg.is_joint(arm) else 0.
            self.assertEqual(cfg.loss_weight(arm,378,calibration),expected/2)
            self.assertEqual(cfg.loss_weight(arm,756,calibration),expected)
            self.assertEqual(cfg.loss_weight(arm,12096,calibration),expected)
            recipe = cfg.recipe(arm,calibration)
            self.assertEqual(recipe['training_rzf_alpha'],cfg.training_alpha(arm))
            self.assertFalse(recipe['monitor']['checkpoint_selection'])
        old_format = deepcopy(calibration)
        old_format['format'] = cfg.previous.CALIBRATION_FORMAT
        with self.assertRaises(ValueError):cfg.calibration_identity(old_format)

    def test_resume_requires_new_budget_beta_source_and_scheduler_identity(self):
        calibration = beta_fixture()
        payload = json.loads(json.dumps(resume_fixture(calibration=calibration)))
        self.assertEqual(cfg.validate_resume_metadata(payload,'g4_joint',calibration),108)
        for label,mutate in (
            ('old checkpoint format',lambda x:x.update(format=cfg.previous.CHECKPOINT_FORMAT)),
            ('old joint initialization',lambda x:x['source'].update(arm='g4_joint')),
            ('changed training alpha',lambda x:x['recipe'].update(training_rzf_alpha=.01)),
            ('wrong cumulative address',lambda x:x.update(cumulative_updates=22188)),
            ('missing RNG',lambda x:x['rng'].pop('torch_cpu')),
            ('reset scheduler',lambda x:x['scheduler'].update(last_epoch=0))):
            changed = deepcopy(payload)
            mutate(changed)
            with self.subTest(change=label),self.assertRaises(ValueError):cfg.validate_resume_metadata(changed,'g4_joint',calibration)
        with self.assertRaises(ValueError):cfg.validate_resume_metadata(payload,'g4_joint',beta_fixture(2.))
        payload.update(step=12096,cumulative_updates=34192)
        payload['scheduler']['last_epoch'] = 12096
        self.assertEqual(cfg.validate_resume_metadata(payload,'g4_joint',calibration),12096)


def final_raw_fixture():
    from tests.test_twc_v1_k8_rzf_probe import raw_fixture
    rows = raw_fixture()
    for row in list(rows):
        if row['arm'] in ('fdma_nmse','g4_nmse'):
            source = deepcopy(row)
            source['arm'] = 'source_fdma' if cfg.is_fdma(row['arm']) else 'source_g4'
            source['sum_rate_bps_hz'] -= .5
            rows.append(source)
    for row in rows:
        method = row['arm']
        step,cumulative = evaluation.method_steps(method)
        row.update(checkpoint_step=step,cumulative_updates=cumulative)
        if row['rzf_mode']=='fixed':row['alpha'] = cfg.training_alpha(method)
        elif method=='source_fdma':row['alpha'] = .3
        elif method=='source_g4':row['alpha'] = 1.
        for field in ('relative_alpha_mean','relative_alpha_min','relative_alpha_max'):
            row[field] = row['alpha']/row['hhat_tone_user_energy_mean']
        for key in evaluation.USER_RATES:row[key] = row['sum_rate_bps_hz']/8
    return rows


class FinalMatchedStatisticsTests(unittest.TestCase):
    def test_all_six_methods_have_layout_fixed_alpha_and_correct_terminal_source_steps(self):
        rows = final_raw_fixture()
        summary,indexed = evaluation.summarize_samples(rows,count=8)
        self.assertEqual((len(rows),len(summary),len(indexed)),(108*8,108,108))
        for method in cfg.METHODS:
            fixed = [row for row in summary if row['arm']==method and row['rzf_mode']=='fixed']
            self.assertEqual(len(fixed),6 if cfg.is_fdma(method) else 12)
            self.assertEqual({row['alpha'] for row in fixed},{cfg.training_alpha(method)})
            expected = (6048,22096) if method.startswith('source_') else (12096,34192)
            self.assertEqual({(row['checkpoint_step'],row['cumulative_updates']) for row in fixed},{expected})
            self.assertTrue(all(not row['test_used'] and row['num_samples']==8 and row['num_batches']==2 for row in fixed))

    def test_signed_raw_interactions_and_batch_ci_are_not_replaced_by_independent_se(self):
        rows = final_raw_fixture()
        paired = evaluation.paired_statistics(rows,count=8)
        self.assertEqual(len(paired),30*8*4)
        fixed = next(row for row in paired if row['contrast']=='gap_interaction_fixed_hybrid_1_3'
                     and row['metric']=='sum_rate_bps_hz' and row['estimand']=='low_0_5_10')
        for prefix,values in (('sample',[-12.]*4+[-6.]*4),('batch',[-12.,-6.])):
            for name,value in evaluation.statistics_for(values).items():
                self.assertAlmostEqual(fixed[prefix+'_'+name],value,places=11)
        self.assertEqual(fixed['sample_mean'],-9.)
        self.assertGreater(fixed['batch_se'],fixed['sample_se'])
        tuned = next(row for row in paired if row['contrast']=='gap_interaction_tuned_hybrid_1_3'
                     and row['metric']=='sum_rate_bps_hz' and row['estimand']=='low_0_5_10')
        self.assertAlmostEqual(tuned['sample_mean'],-17.5)
        own_source = next(row for row in paired if row['contrast']=='g4_joint_minus_source_g4_fixed_hybrid_1_3'
                     and row['metric']=='sum_rate_bps_hz' and row['estimand']=='low_0_5_10')
        self.assertLess(own_source['sample_mean'],0.)
        self.assertEqual(paired,evaluation.paired_statistics(list(reversed(rows)),count=8))

    def test_invariants_old_alpha_and_source_mixing_are_rejected(self):
        base = final_raw_fixture()
        invalid = {'duplicate':base+[deepcopy(base[0])],'missing':base[:-1]}
        for label,field,value in (
            ('old alpha0.01 used as fixed','alpha',.01),
            ('NMSE modified by alpha','dl_nmse',.8),
            ('unpaired actual channels','channel_fingerprint','f'*64),
            ('nonfinite diagnostic','hhat_energy_total',float('nan'))):
            rows = deepcopy(base)
            row = next(row for row in rows if row['arm']=='g4_joint' and row['rzf_mode']=='fixed')
            row[field] = value
            invalid[label] = rows
        rows = deepcopy(base)
        next(row for row in rows if row['arm']=='source_g4')['checkpoint_step'] = 12096
        invalid['source mislabeled as new training'] = rows
        for label,rows in invalid.items():
            with self.subTest(change=label),self.assertRaises(ValueError):evaluation.paired_statistics(rows,count=8)

    def test_final_global_alpha_selection_is_six_checkpoints_and_does_not_change_training_alpha(self):
        from tests.test_twc_v1_k8_rzf_probe import score_fixture
        rows = score_fixture()
        for row in list(rows):
            if row['arm'] in ('fdma_nmse','g4_nmse'):
                source = deepcopy(row)
                source['arm'] = 'source_fdma' if cfg.is_fdma(row['arm']) else 'source_g4'
                rows.append(source)
        before = {method:cfg.training_alpha(method) for method in cfg.METHODS}
        result = evaluation.select_global_alphas(rows)
        self.assertEqual(len(rows),486)
        self.assertEqual(set(result['selected_alphas']),set(cfg.METHODS))
        self.assertEqual(result['selected_alphas']['fdma_nmse'],.003)
        self.assertEqual(result['selected_alphas']['fdma_joint'],.0001)
        self.assertEqual(result['selected_alphas']['g4_joint'],.03)
        self.assertEqual(before,{method:cfg.training_alpha(method) for method in cfg.METHODS})
        self.assertEqual(before['fdma_nmse'],.03)
        self.assertEqual(before['g4_joint'],.1)
        with self.assertRaises(ValueError):evaluation.select_global_alphas(rows[:-1])


def old_alpha_fixture():
    from tests.test_twc_v1_k8_rzf_probe import complete_selection_fixture
    record = complete_selection_fixture()
    rows = record['calibration_summary']
    for row in rows:
        expected = .03 if row['arm'].startswith('fdma') else .1
        row['sum_rate_bps_hz_mean'] = 10. if row['alpha']==expected else 0.
    record.update(training.probe.select_global_alphas(rows))
    hashes = {'old_probe.py':'a'*64}
    record['code'] = dict(commit=cfg.DIAGNOSTIC_COMMIT,sha256=hashes)
    record.pop('record_id')
    record['record_id'] = training.probe.cfg.json_sha256(record)
    manifest = {key:deepcopy(record[key]) for key in ('code','selected_alphas','calibration_summary','scores',
        'boundary_optima','shared_inputs','frozen_files')}
    manifest.update(status='completed',stage='calibration',source_implementation_commit=cfg.SOURCE_COMMIT,
        freeze_checks={'passed':True},validation_addresses=list(range(4096,4112)),network_forward_calls=576,rzf_solves=5184)
    return record,manifest,hashes


def new_beta_rows():
    from tests.test_twc_v1_k8_task_loss_screen import calibration_rows
    rows = calibration_rows()
    for row in rows:
        row['batch_index'] += 100
        row['rzf_alpha'] = cfg.training_alpha(row['source_key'])
    return rows


def complete_beta_fixture():
    rows = new_beta_rows()
    old,_,_ = old_alpha_fixture()
    record = dict(format=cfg.CALIBRATION_FORMAT,status='passed',records=rows,
        **training.calibration_from_records(rows),sources={method:cfg.source_identity(method) for method in training.SOURCE_METHODS},
        original_training_alpha_identity=training.probe.cfg.selection_identity(old),
        training_alphas={method:cfg.training_alpha(method) for method in training.SOURCE_METHODS},
        code=dict(commit='new matched experiment',sha256={'matched.py':'b'*64}),
        frozen_files={'metadata fixture only':'c'*64},data_protocol={'schema_fixture':True},
        state_freeze={method:dict(passed=True,codec_model_sha256='d'*64) for method in training.SOURCE_METHODS})
    record['record_id'] = cfg.json_sha256(record)
    return record


class MatchedCalibrationTests(unittest.TestCase):
    def test_old_alpha_record_is_read_only_0e_identity_not_new_head_or_596(self):
        record,manifest,hashes = old_alpha_fixture()
        record,manifest = json.loads(json.dumps((record,manifest)))
        with patch.object(training,'source_code_record',side_effect=AssertionError('new HEAD must not substitute old provenance')):
            identity = training.validate_training_alpha_record(record,manifest,archived_hashes=hashes)
        self.assertEqual(identity['selected_alphas'],dict(fdma_nmse=.03,fdma_joint=.03,g4_nmse=.1,g4_joint=.1))
        for label,mutate in (
            ('wrong old implementation',lambda r,m:r['code'].update(commit=cfg.SOURCE_COMMIT)),
            ('wrong old source implementation',lambda r,m:m.update(source_implementation_commit=cfg.INTERNAL_SOURCE_COMMIT)),
            ('wrong decision',lambda r,m:r['selected_alphas'].update(g4_nmse=.03)),
            ('wrong source weight',lambda r,m:r['sources']['fdma_nmse'].update(sha256='0'*64)),
            ('receipt disagrees',lambda r,m:m.update(network_forward_calls=9))):
            changed,receipt = deepcopy(record),deepcopy(manifest)
            mutate(changed,receipt)
            changed.pop('record_id')
            changed['record_id'] = training.probe.cfg.json_sha256(changed)
            with self.subTest(change=label),self.assertRaises(ValueError):
                training.validate_training_alpha_record(changed,receipt,archived_hashes=hashes)

    def test_new_beta_pools_both_sources_and_enforces_new_alpha_and_addresses(self):
        rows = new_beta_rows()
        result = training.calibration_from_records(rows)
        self.assertEqual((result['beta_target'],result['median_nmse_grad_norm'],result['median_rate_grad_norm']),(1.25,10.,2.))
        self.assertNotEqual(result['beta_target'],(.25*2/1+.25*18/3)/2)
        invalid = {'missing':rows[:-1],'duplicate':rows[:-1]+[deepcopy(rows[0])]}
        for label,field,value in (
            ('old RZF alpha','rzf_alpha',.01),('old calibration address','batch_index',500000),
            ('zero gradient','nmse_grad_norm',0.),('nonfinite gradient','rate_grad_norm',float('nan')),
            ('unpaired data','data_sha256','f'*64)):
            changed = deepcopy(rows)
            changed[0][field] = value
            invalid[label] = changed
        for label,values in invalid.items():
            with self.subTest(change=label),self.assertRaises(ValueError):training.calibration_from_records(values)

    def test_new_beta_complete_identity_cannot_relabel_old_beta_or_ignore_source_changes(self):
        record = complete_beta_fixture()
        self.assertEqual(training.validate_calibration_record(record),cfg.calibration_identity(record))
        for label,mutate in (
            ('old beta format',lambda r:r.update(format=cfg.previous.CALIBRATION_FORMAT)),
            ('wrong source arm',lambda r:r['sources']['source_g4'].update(arm='g4_joint')),
            ('wrong training alpha',lambda r:r['training_alphas'].update(source_g4=.01)),
            ('missing old alpha proof',lambda r:r.update(original_training_alpha_identity={})),
            ('changed beta',lambda r:r.update(beta_target=99.)),
            ('source state changed',lambda r:r['state_freeze']['source_fdma'].update(passed=False))):
            changed = deepcopy(record)
            mutate(changed)
            changed.pop('record_id')
            changed['record_id'] = cfg.json_sha256(changed)
            with self.subTest(change=label),self.assertRaises(ValueError):training.validate_calibration_record(changed)

    def test_new_beta_write_once_reload_binds_unchanged_old_alpha_and_source_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_alpha = root/'old_alpha_selection.json'
            old_alpha.write_text(json.dumps(old_alpha_fixture()[0]),encoding='utf-8')
            source = root/'source_receipt.json'
            source.write_bytes(b'{"schema_fixture":"NMSE source receipt"}')
            record = complete_beta_fixture()
            record['frozen_files'] = {str(path):training.sha256(path) for path in (old_alpha,source)}
            record.pop('record_id')
            record['record_id'] = cfg.json_sha256(record)
            path = root/'calibration.json'
            inside = cfg.inside_root
            with patch.object(cfg,'CALIBRATION_PATH',path), \
                 patch.object(cfg,'inside_root',side_effect=lambda value:inside(value,root=root)), \
                 patch.object(training,'source_code_record',return_value=record['code']):
                training.write_once_json(path,record)
                before = (path.read_bytes(),old_alpha.read_bytes(),source.read_bytes())
                self.assertEqual(training.load_calibration(),record)
                self.assertEqual(training.load_calibration(),record)
                with self.assertRaises(FileExistsError):training.write_once_json(path,beta_fixture(99.))
                self.assertEqual((path.read_bytes(),old_alpha.read_bytes(),source.read_bytes()),before)
                old_alpha.write_bytes(b'{"changed":"old alpha evidence"}')
                with self.assertRaises(ValueError):training.load_calibration()
                old_alpha.write_bytes(before[1])
                source.write_bytes(b'{"changed":"source evidence"}')
                with self.assertRaises(ValueError):training.load_calibration()


@unittest.skipUnless(TORCH_PRESENT,'PyTorch unavailable; tiny strict-state/loss tests not executed')
class TinyStateAndLossTests(unittest.TestCase):
    def test_actual_factory_maps_both_losses_to_same_nmse_source_without_residual_reset(self):
        from types import SimpleNamespace
        class TinyV2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.key_window_residual = torch.nn.Linear(3,2,bias=False)
                self.encoder.shared_window_residual = torch.nn.Linear(3,2,bias=False)
                self.register_buffer('running_pwr',torch.tensor([3.]))
        original = TinyV2()
        with torch.no_grad():
            original.encoder.key_window_residual.weight.fill_(.125)
            original.encoder.shared_window_residual.weight.fill_(-.25)
        payload = {'codec_model':deepcopy(original.state_dict())}
        def tiny_boundary(torch_module,device,source_arm,saved):
            self.assertIs(torch_module,torch)
            self.assertIn(source_arm,('fdma_nmse','g4_nmse'))
            model = TinyV2()
            training.task.previous.old.check_tensor_state(torch,model,saved['codec_model'])
            model.load_state_dict(saved['codec_model'],strict=True)
            return SimpleNamespace(codec_model=model.to(device))
        with patch.object(training.task,'build_chain',side_effect=tiny_boundary) as boundary:
            states = {}
            for method in cfg.METHODS:
                chain = training.build_chain(torch,torch.device('cpu'),method,payload)
                states[method] = chain.codec_model.state_dict()
                expected = 'fdma_nmse' if cfg.is_fdma(method) else 'g4_nmse'
                self.assertEqual(boundary.call_args.args[2],expected)
                for key,value in original.state_dict().items():self.assertTrue(torch.equal(states[method][key],value),key)
            for family in ('fdma','g4'):
                for key in states[family+'_nmse']:
                    self.assertTrue(torch.equal(states[family+'_nmse'][key],states[family+'_joint'][key]))
            for label in ('missing','wrong dtype','wrong shape'):
                changed = deepcopy(payload)
                key = 'encoder.key_window_residual.weight'
                if label=='missing':changed['codec_model'].pop(key)
                elif label=='wrong dtype':changed['codec_model'][key] = changed['codec_model'][key].double()
                else:changed['codec_model'][key] = torch.zeros(7)
                with self.subTest(change=label),self.assertRaises((ValueError,RuntimeError)):
                    training.build_chain(torch,torch.device('cpu'),'g4_joint',changed)
        for key,value in original.state_dict().items():self.assertTrue(torch.equal(payload['codec_model'][key],value))

    def test_new_beta_loss_has_correct_sign_differentiability_and_no_nmse_rescaling(self):
        calibration = beta_fixture()
        for arm in cfg.ARMS:
            nmse = torch.tensor(.7,dtype=torch.float64,requires_grad=True)
            rate = torch.tensor(19.,dtype=torch.float64,requires_grad=True)
            loss = training.task_loss(nmse,rate,arm,378,calibration)
            ngrad,rgrad = torch.autograd.grad(loss,(nmse,rate),allow_unused=True)
            self.assertEqual(ngrad.item(),1.)
            if cfg.is_joint(arm):
                self.assertEqual(rgrad.item(),-.625)
                self.assertAlmostEqual(loss.item(),.7-.625*19.)
            else:
                self.assertIs(loss,nmse)
                self.assertIsNone(rgrad)


@unittest.skipUnless(TORCH_PRESENT and NUMPY_PRESENT,'Real PyTorch/NumPy required for tiny monitor RNG tests')
class MonitorRestorationTests(unittest.TestCase):
    def _fixture(self,*,fail=False,bad_power=None):
        from types import SimpleNamespace
        tester = self
        class TinyChain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.codec_model = torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Dropout(.1))
                self.codec_model.register_buffer('running_pwr',torch.tensor([1.]))
                self.ce_model = torch.nn.Linear(1,1)
                self.ce_model.register_buffer('frozen_marker',torch.tensor([2.]))
                self.ce_model.requires_grad_(False)
                self.train();self.ce_model.eval();self.codec_model[1].eval()
                self.alphas = []
            def metrics(self,output,truth,*,rzf_alpha,downlink_snr_db):
                tester.assertFalse(any(module.training for module in self.modules()))
                tester.assertFalse(any(p.requires_grad for p in self.ce_model.parameters()))
                tester.assertEqual(downlink_snr_db,25.)
                self.alphas.append(rzf_alpha)
                random.random();np.random.rand();torch.rand(1)
                gamma = output['condition']['gamma_db']
                power = torch.ones(4,2)
                if bad_power=='rzf':power.fill_(float('nan'))
                return dict(dl_nmse_per_user=torch.ones(4,8)*(.2+gamma*.001),
                    sum_rate_bps_hz=torch.ones(4)*(10.+gamma),rzf_per_subcarrier_power=power,
                    desired_signal_power_mean=torch.ones(4),interference_power_mean=torch.ones(4)*.1)
        chain = TinyChain();addresses=[];conditions=[]
        def sample(role,address):
            tester.assertEqual(role,'validation');addresses.append(address)
            random.random();np.random.rand();torch.rand(1)
            return dict(H_dl=torch.ones(4,1),H_ul=torch.ones(4,1),sample_fingerprint=f'{address:064x}')
        def run(actual,warmup,hdl,hul,condition,pilot,noise):
            tester.assertIs(actual,chain)
            tester.assertFalse(any(module.training for module in actual.modules()))
            conditions.append(deepcopy(condition))
            if fail:raise RuntimeError('synthetic forward-boundary failure')
            energy = torch.full((4,8),48.)
            if bad_power=='energy':energy.fill_(float('nan'))
            return dict(condition=condition,energy={'actual_data_per_ue':energy})
        warmup = SimpleNamespace(standard_noise=lambda batch,generator,device:torch.randn(batch,1,generator=generator,device=device))
        return chain,SimpleNamespace(sample=sample),warmup,run,addresses,conditions

    def _rng_equal(self,before,after):
        self.assertEqual(before['python'],after['python'])
        self.assertEqual(before['numpy'][0],after['numpy'][0])
        self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
        self.assertEqual(before['numpy'][2:],after['numpy'][2:])
        self.assertTrue(torch.equal(before['torch_cpu'],after['torch_cpu']))
        self.assertEqual(len(before['torch_cuda']),len(after['torch_cuda']))
        for a,b in zip(before['torch_cuda'],after['torch_cuda']):self.assertTrue(torch.equal(a,b))

    def test_full_grid_monitor_restores_modes_rng_and_frozen_ce_for_both_layouts(self):
        for arm in ('fdma_joint','g4_joint'):
            chain,data,warmup,run,addresses,conditions = self._fixture()
            modes = [module.training for module in chain.modules()]
            state = deepcopy(chain.state_dict())
            before = training.capture_rng(torch,np)
            with patch.object(training.task.previous.old,'_run',side_effect=run):
                result = training.monitor(torch,np,warmup,chain,data,arm,3024)
            self._rng_equal(before,training.capture_rng(torch,np))
            self.assertEqual([module.training for module in chain.modules()],modes)
            for key,value in state.items():self.assertTrue(torch.equal(chain.state_dict()[key],value),key)
            self.assertEqual(addresses,list(range(8960,8976)))
            self.assertEqual(len(result['summary']),6 if cfg.is_fdma(arm) else 12)
            self.assertEqual(len(result['all_six']),1 if cfg.is_fdma(arm) else 2)
            self.assertEqual(len(result['shared_inputs']),16)
            self.assertTrue(all(row['num_samples']==64 and row['alpha']==cfg.training_alpha(arm) for row in result['summary']))
            self.assertEqual(set(chain.alphas),{cfg.training_alpha(arm)})
            self.assertEqual(len(conditions),16*(6 if cfg.is_fdma(arm) else 12))
            self.assertFalse(result['selection']);self.assertFalse(result['test_used'])

    def test_monitor_exception_still_restores_exact_modes_and_rng(self):
        chain,data,warmup,run,_,_ = self._fixture(fail=True)
        modes = [module.training for module in chain.modules()]
        before = training.capture_rng(torch,np)
        with patch.object(training.task.previous.old,'_run',side_effect=run):
            with self.assertRaisesRegex(RuntimeError,'synthetic forward-boundary failure'):
                training.monitor(torch,np,warmup,chain,data,'g4_joint',6048)
        self._rng_equal(before,training.capture_rng(torch,np))
        self.assertEqual([module.training for module in chain.modules()],modes)
        self.assertFalse(chain.ce_model.training)

    def test_nonfinite_power_diagnostics_are_rejected_without_hiding_restoration(self):
        for bad in ('energy','rzf'):
            chain,data,warmup,run,_,_ = self._fixture(bad_power=bad)
            modes = [module.training for module in chain.modules()]
            before = training.capture_rng(torch,np)
            with patch.object(training.task.previous.old,'_run',side_effect=run):
                with self.subTest(power=bad),self.assertRaises((ValueError,FloatingPointError)):
                    training.monitor(torch,np,warmup,chain,data,'g4_nmse',3024)
            self._rng_equal(before,training.capture_rng(torch,np))
            self.assertEqual([module.training for module in chain.modules()],modes)


class ControllerMatchedResumeTests(unittest.TestCase):
    def _write_receipt(self,controller,root,arm,step,calibration,code):
        destination = root/'training'/arm
        destination.mkdir(parents=True,exist_ok=True)
        checkpoint = destination/'checkpoint_latest.pt'
        checkpoint.write_bytes(f'metadata fixture marker only, not weights; step={step}'.encode())
        initial = dict(initialized_state_sha256='d'*64,migration=False,new_optimizer=True)
        (destination/'initialization.json').write_text(json.dumps(initial),encoding='utf-8')
        events = [dict(step=i,cumulative_updates=22096+i,condition=cfg.schedule(arm,i),
            standard_noise_seed=cfg.training_noise_seed(i),sample_fingerprint=f'{i:064x}') for i in range(108,step+1,108)]
        (destination/'progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events),encoding='utf-8')
        monitors = {}
        policies = ('fdma',) if cfg.is_fdma(arm) else ('hybrid_1_3','hybrid_1_1')
        shared = [dict(address=i,channel_fingerprint=f'{i:064x}',noise={'schema_fixture':'shared'}) for i in range(8960,8976)]
        for monitor_step in cfg.PROGRESS_VALIDATION_STEPS:
            if monitor_step>step:continue
            report = dict(step=monitor_step,cumulative_updates=22096+monitor_step,arm=arm,alpha=cfg.training_alpha(arm),
                summary=[dict(energy_policy=p,gamma_db=g,num_samples=64,alpha=cfg.training_alpha(arm)) for p in policies for g in cfg.GAMMAS],
                all_six=[dict(energy_policy=p,num_samples=64,alpha=cfg.training_alpha(arm)) for p in policies],
                shared_inputs=shared,selection=False,test_used=False)
            path = destination/f'monitor_fixture_{monitor_step}.json'
            path.write_text(json.dumps(report),encoding='utf-8')
            monitors[str(monitor_step)] = dict(path=str(path),sha256=controller.sha256(path))
        record = dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,source=cfg.source_identity(arm),
            code=code,calibration=cfg.calibration_identity(calibration),training_alpha=cfg.training_alpha(arm),
            recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration),
            step=step,cumulative_updates=22096+step,status='completed' if step==12096 else 'paused',
            actual_condition_counts=training.expected_counts(arm,step),checkpoint_sha256=controller.sha256(checkpoint),
            monitor_records=monitors)
        (destination/'manifest.json').write_text(json.dumps(record),encoding='utf-8')
        return destination

    def test_actual_controller_108_is_in_budget_then12096_and_verified_completion_skips(self):
        import run_twc_v1_k8_rzf_matched_training as controller
        calibration = complete_beta_fixture()
        code = dict(commit='new matched experiment',sha256={'matched.py':'a'*64})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inside = cfg.inside_root
            calls = []
            def child(folder,label,command):
                calls.append(command)
                step = 108 if command[-2:]==['--stop-after','108'] else 12096
                if step==12096:self.assertEqual(command[-2:],['--resume','--train'])
                self._write_receipt(controller,root,'g4_joint',step,calibration,code)
                return root/(label+'.log')
            with patch.object(cfg,'OUTPUT_ROOT',root), \
                 patch.object(cfg,'inside_root',side_effect=lambda value:inside(value,root=root)), \
                 patch.object(controller,'source_code_record',return_value=code), \
                 patch.object(controller,'_child',side_effect=child):
                result = controller.run_arm(root,'g4_joint',calibration)
                self.assertEqual((result['report']['step'],result['report']['cumulative_updates']),(12096,34192))
                self.assertEqual(len(calls),2)
                self.assertEqual(calls[0][-2:],['--stop-after','108'])
                controller.run_arm(root,'g4_joint',calibration)
                self.assertEqual(len(calls),2)

    def test_paused108_resumes_once_and_actual_receipt_rejects_wrong_counts_or_alpha(self):
        import run_twc_v1_k8_rzf_matched_training as controller
        calibration = complete_beta_fixture()
        code = dict(commit='new matched experiment',sha256={'matched.py':'a'*64})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inside = cfg.inside_root
            with patch.object(cfg,'OUTPUT_ROOT',root), \
                 patch.object(cfg,'inside_root',side_effect=lambda value:inside(value,root=root)), \
                 patch.object(controller,'source_code_record',return_value=code):
                destination = self._write_receipt(controller,root,'fdma_joint',108,calibration,code)
                receipt_path = destination/'manifest.json'
                original = receipt_path.read_bytes()
                for label,field,value in (('wrong counts','actual_condition_counts',{}),('old alpha','training_alpha',.01)):
                    changed = json.loads(original)
                    changed[field] = value
                    receipt_path.write_text(json.dumps(changed),encoding='utf-8')
                    with self.subTest(change=label),self.assertRaises(ValueError):controller.inspect_training('fdma_joint',calibration)
                receipt_path.write_bytes(original)
                def child(folder,label,command):
                    self.assertEqual(command[-2:],['--resume','--train'])
                    self._write_receipt(controller,root,'fdma_joint',12096,calibration,code)
                    return root/'resume.log'
                with patch.object(controller,'_child',side_effect=child) as child_mock:
                    result = controller.run_arm(root,'fdma_joint',calibration)
                    child_mock.assert_called_once()
                self.assertEqual(result['report']['cumulative_updates'],34192)
                self.assertEqual(len((destination/'progress.jsonl').read_text().splitlines()),112)
                self.assertEqual(set(result['report']['monitor_records']),{'3024','6048','9072','12096'})

    def test_failed_smoke_preserves_unowned_directory_and_never_continues_or_restarts(self):
        import run_twc_v1_k8_rzf_matched_training as controller
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root/'training'/'g4_joint'
            def failed_child(folder,label,command):
                self.assertEqual(command[-2:],['--stop-after','108'])
                destination.mkdir(parents=True)
                (destination/'failure.json').write_bytes(b'{"fixture":"preserved failure"}')
                raise RuntimeError('mocked subprocess boundary failed; no training')
            with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(controller,'_child',side_effect=failed_child) as child:
                with self.assertRaises(RuntimeError):controller.run_arm(root,'g4_joint',complete_beta_fixture())
                before = (destination/'failure.json').read_bytes()
                with self.assertRaises(FileExistsError):controller.run_arm(root,'g4_joint',complete_beta_fixture())
                child.assert_called_once()
                self.assertEqual((destination/'failure.json').read_bytes(),before)


if __name__ == '__main__':unittest.main()
