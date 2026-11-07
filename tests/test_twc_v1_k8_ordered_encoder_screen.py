"""Tiny CPU algebra/state tests; no full codec construction, CDL or weights."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TORCH_PRESENT = importlib.util.find_spec('torch') is not None
if TORCH_PRESENT:
    import torch
    try:
        from models import twc_v1_k8_ordered_encoder as ordered
        MODEL_IMPORT_ERROR = None
    except ModuleNotFoundError as error:
        MODEL_IMPORT_ERROR = str(error)
else:
    MODEL_IMPORT_ERROR = 'PyTorch absent; no synthetic numerical checks executed'


@unittest.skipIf(MODEL_IMPORT_ERROR is not None, MODEL_IMPORT_ERROR or '')
class OrderedAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator(device='cpu').manual_seed(20261107)

    def test_window_order_matches_actual_row_major_grid(self):
        grid = torch.arange(64*8,dtype=torch.float64).reshape(1,64,8,1)
        tokens=grid.reshape(1,512,1)
        result=ordered.ordered_window_pool(tokens)
        expected=torch.stack([grid[0,r:r+4,c:c+4].mean()
                              for r in range(0,64,4) for c in range(0,8,4)]).reshape(1,32,1)
        self.assertTrue(torch.equal(result,expected))
        self.assertTrue(torch.equal(result.mean(1),tokens.mean(1)))
        with self.assertRaises(ValueError):ordered.ordered_window_pool(tokens[:,:511])

    def _heads(self,channels=5,dtype=torch.float32):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            key=torch.nn.Linear(channels,2,dtype=dtype)
            shared=torch.nn.Linear(channels,3,dtype=dtype)
            key_residual=torch.nn.Linear(31*channels,2,bias=False,dtype=dtype)
            shared_residual=torch.nn.Linear(31*channels,3,bias=False,dtype=dtype)
        torch.nn.init.zeros_(key_residual.weight)
        torch.nn.init.zeros_(shared_residual.weight)
        return key,shared,key_residual,shared_residual

    def _zero_algebra(self,dtype):
        tokens=torch.randn(6,512,5,generator=self.generator,dtype=dtype)
        condition=torch.randn(2,5,generator=self.generator,dtype=dtype)
        heads=self._heads(dtype=dtype)
        conditioned=tokens.mean(dim=1).reshape(2,3,5)+condition[:,None]
        old=(heads[0](conditioned),heads[1](conditioned))
        actual=ordered.readout_from_normalized_tokens(tokens,condition,*heads,users=3)
        for got,expected in zip(actual,old):
            evidence=ordered.comparison_error(got,expected)
            self.assertTrue(torch.equal(got,expected),evidence)
            self.assertEqual((evidence['max_abs'],evidence['max_rel']),(0.,0.))
            print('ordered v2 tiny zero residual',dtype,evidence)

    def test_zero_residual_preserves_original_mean_head_path_exactly_fp32(self):
        self._zero_algebra(torch.float32)

    def test_zero_residual_preserves_original_mean_head_path_exactly_fp64(self):
        self._zero_algebra(torch.float64)

    def test_arbitrary_ordered_head_has_equivalent_mean_contrast_coordinates_including_G(self):
        tokens=torch.randn(6,512,5,generator=self.generator,dtype=torch.float64)
        condition=torch.randn(2,5,generator=self.generator,dtype=torch.float64)
        heads=self._heads(dtype=torch.float64)
        windows=ordered.ordered_window_pool(tokens).reshape(2,3,32,5)
        ordered_features=(windows+condition[:,None,None]).flatten(-2)
        expected=[]
        with torch.no_grad():
            for main,residual in ((heads[0],heads[2]),(heads[1],heads[3])):
                W=torch.randn(main.out_features,160,generator=self.generator,dtype=torch.float64)
                mean_W,residual_W=ordered.ordered_head_coordinates(W)
                main.weight.copy_(mean_W);residual.weight.copy_(residual_W)
                expected.append(torch.nn.functional.linear(ordered_features,W,main.bias))
        actual=ordered.readout_from_normalized_tokens(tokens,condition,*heads,users=3)
        for got,wanted in zip(actual,expected):
            error=(got-wanted).abs().max().item()
            self.assertLess(error,1e-12)
            print('ordered v2 arbitrary head capacity identity fp64 max_abs',error)

    def test_ordered_features_retain_window_arrangement_and_free_weights_can_use_it(self):
        means=torch.arange(32,dtype=torch.float32).reshape(1,16,2,1)
        grid=means.repeat_interleave(4,1).repeat_interleave(4,2)
        original=grid.reshape(1,512,1)
        swapped=grid.roll(4,1).reshape(1,512,1)
        self.assertTrue(torch.equal(original.mean(1),swapped.mean(1)))
        a,b=ordered.ordered_window_pool(original),ordered.ordered_window_pool(swapped)
        self.assertFalse(torch.equal(a,b))
        readout=torch.arange(32,dtype=torch.float32)
        self.assertNotEqual((a.flatten()*readout).sum().item(),(b.flatten()*readout).sum().item())

    def test_zero_residual_weights_receive_nonzero_finite_gradients_without_a_gate_skip(self):
        tokens=torch.randn(6,512,5,generator=self.generator,requires_grad=True)
        condition=torch.randn(2,5,generator=self.generator)
        heads=self._heads()
        result=ordered.readout_from_normalized_tokens(tokens,condition,*heads,users=3)
        sum(value.square().mean() for value in result).backward()
        for residual in heads[2:]:
            self.assertIsNone(residual.bias)
            self.assertEqual(torch.count_nonzero(residual.weight).item(),0)
            self.assertIsNotNone(residual.weight.grad)
            self.assertTrue(torch.isfinite(residual.weight.grad).all())
            self.assertGreater(residual.weight.grad.norm().item(),0.)
            self.assertFalse(torch.equal(residual.weight.grad[:,:5],residual.weight.grad[:,5:10]))
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertEqual(ordered.EXPECTED_PARAMETER_DELTA,1_301_504)
        self.assertEqual(sum(p.numel() for residual in heads[2:] for p in residual.parameters()),31*5*(2+3))

    def _fixture(self):
        # A real, small module exercises the same strict loading wrapper; it is
        # explicitly not a K8 model or a substitute for missing dependencies.
        class TinyCodec(torch.nn.Module):
            def __init__(self,with_residual):
                super().__init__()
                self.encoder=torch.nn.Module()
                self.encoder.key_head=torch.nn.Linear(5,2)
                self.encoder.shared_head=torch.nn.Linear(5,3)
                self.encoder.group_embedding=torch.nn.Embedding(3,5)
                if with_residual:
                    self.encoder.key_window_residual=torch.nn.Linear(155,2,bias=False)
                    self.encoder.shared_window_residual=torch.nn.Linear(155,3,bias=False)
                    torch.nn.init.zeros_(self.encoder.key_window_residual.weight)
                    torch.nn.init.zeros_(self.encoder.shared_window_residual.weight)
                self.decoder=torch.nn.Linear(5,4)
                self.register_buffer('counter',torch.tensor(13,dtype=torch.int64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            mean=TinyCodec(False);target=TinyCodec(True)
        return mean,target

    def test_complete_state_migration_keeps_all_old_values_and_only_adds_zero_residuals(self):
        mean,target=self._fixture();source=deepcopy(mean.state_dict());before=deepcopy(source)
        report=ordered.migrate_mean_state(target,source)
        self.assertEqual(report['changed_keys'],[])
        self.assertEqual(set(report['new_parameter_keys']),set(ordered.RESIDUAL_WEIGHT_KEYS))
        self.assertEqual(set(target.state_dict())-set(source),set(ordered.RESIDUAL_WEIGHT_KEYS))
        for name,value in source.items():
            self.assertTrue(torch.equal(value,before[name]))
            self.assertTrue(torch.equal(target.state_dict()[name],value))
        for name in ordered.RESIDUAL_WEIGHT_KEYS:
            self.assertEqual(torch.count_nonzero(target.state_dict()[name]).item(),0)

    def test_new_state_strict_roundtrip_preserves_learned_nonzero_residual_without_remigration(self):
        mean,target=self._fixture();source=deepcopy(mean.state_dict())
        ordered.migrate_mean_state(target,source)
        with torch.no_grad():
            target.encoder.key_window_residual.weight.fill_(.125)
            target.encoder.shared_window_residual.weight.fill_(-.25)
        clone=self._fixture()[1]
        clone.load_state_dict(target.state_dict(),strict=True)
        self.assertTrue(torch.equal(clone.encoder.key_window_residual.weight,target.encoder.key_window_residual.weight))
        self.assertTrue(torch.equal(clone.encoder.shared_window_residual.weight,target.encoder.shared_window_residual.weight))
        with self.assertRaises(RuntimeError):clone.load_state_dict(source,strict=True)
        with self.assertRaises(ValueError):ordered.migrate_mean_state(clone,target.state_dict())

    def test_migration_rejects_missing_extra_dtype_shape_and_nonfinite_state(self):
        mean,target=self._fixture();original=mean.state_dict()
        for issue in ('missing','extra','dtype','head_shape','other_shape','nonfinite','v1_expanded_head'):
            state=deepcopy(original)
            if issue=='missing':state.pop('decoder.bias')
            elif issue=='extra':state['unexpected']=torch.zeros(1)
            elif issue=='dtype':state['encoder.key_head.weight']=state['encoder.key_head.weight'].double()
            elif issue=='head_shape':state['encoder.key_head.weight']=torch.zeros(2,6)
            elif issue=='other_shape':state['decoder.weight']=torch.zeros(3,5)
            elif issue=='v1_expanded_head':state['encoder.key_head.weight']=torch.zeros(2,160)
            else:state['encoder.key_head.bias'][0]=float('nan')
            with self.subTest(issue=issue),self.assertRaises(ValueError):
                ordered.migrate_mean_state(target,state)
        for issue in ('residual_shape','residual_dtype','extra_new_parameter'):
            template=deepcopy(target.state_dict())
            if issue=='residual_shape':template['encoder.key_window_residual.weight']=torch.zeros(2,154)
            elif issue=='residual_dtype':template['encoder.key_window_residual.weight']=template['encoder.key_window_residual.weight'].double()
            else:template['encoder.unreviewed.weight']=torch.zeros(1)
            with self.subTest(issue=issue),self.assertRaises(ValueError):
                ordered.migrate_state_dict(original,template)

    def test_error_report_uses_fixed_tolerance_and_handles_near_zero(self):
        expected=torch.tensor([0.,1.],dtype=torch.float32)
        actual=torch.tensor([5e-7,1.+5e-6],dtype=torch.float32)
        report=ordered.comparison_error(actual,expected)
        self.assertTrue(report['passed'])
        self.assertGreater(report['max_rel'],ordered.RTOL)
        self.assertEqual((report['rtol'],report['atol']),(1e-5,1e-6))
        bad=ordered.comparison_error(torch.tensor([1e-2,1.]),expected)
        self.assertFalse(bad['passed'])


class OrderedEvaluationContracts(unittest.TestCase):
    def fixture(self,count=8):
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        import evaluate_twc_v1_k8_ordered_encoder_screen as ev
        rows=[]
        for arm,g,policy,gamma in ev.conditions():
            for i in range(count):
                effect=1. if arm=='ordered_fdma' else .5 if arm=='ordered_g4' else 0.
                rows.append(dict(arm=arm,group_count=g,energy_policy=policy,gamma_db=gamma,
                    seed=cfg.SEED,checkpoint_step=cfg.UPDATES,cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,
                    sample_index=i,batch_index=i//4,pilot_energy_per_ue=16.,downlink_snr_db=25.,
                    channel_fingerprint=f'{i//4:064x}',dl_nmse=1.,sum_rate_bps_hz=10.+i*.1+effect,
                    desired_signal_power=1.,interference_power=.1,data_energy_per_ue=48.,frame_energy_per_ue=64.,
                    rzf_power_mean=1.,energy_max_abs_error=0.,rzf_power_max_abs_error=0.))
        return rows

    def test_condition_counts_sources_and_fdma_rows_are_not_duplicated(self):
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        import evaluate_twc_v1_k8_ordered_encoder_screen as ev
        self.assertEqual(len(ev.conditions()),36)
        self.assertEqual(sum(cfg.is_fdma(row[0]) for row in ev.conditions()),12)
        self.assertEqual(cfg.source_identity('mean_fdma_control'),cfg.source_identity('ordered_fdma'))
        self.assertEqual(cfg.source_identity('mean_g4_control'),cfg.source_identity('ordered_g4'))
        self.assertEqual(cfg.schedule_counts('mean_g4_control'),cfg.schedule_counts('ordered_g4'))
        self.assertEqual({r['count'] for r in cfg.schedule_counts('ordered_g4')},{84})
        self.assertEqual(cfg.schedule('ordered_g4',1)['global_step'],13025)
        self.assertEqual(cfg.schedule('ordered_g4',3024)['global_step'],16048)
        self.assertEqual(cfg.EXPECTED_PARAMETER_DELTA,1301504)
        self.assertEqual(sum(c['name']=='ordered_fdma_minus_mean_fdma' for c in ev.contrasts()),1)

    def test_interaction_is_signed_raw_pairing_and_detects_fdma_improving_more(self):
        import evaluate_twc_v1_k8_ordered_encoder_screen as ev
        rows=self.fixture();summary,_=ev.summarize_samples(rows,8)
        paired=ev.paired_statistics(rows,8)
        self.assertEqual((len(summary),len(paired)),(36,288))
        own=[r for r in paired if r['contrast'].startswith('ordered_g4_minus_mean_g4') and r['metric']=='sum_rate_bps_hz']
        interaction=[r for r in paired if r['contrast'].startswith('gap_change_') and r['metric']=='sum_rate_bps_hz']
        self.assertTrue(all(abs(r['sample_mean']-.5)<1e-12 for r in own))
        self.assertTrue(all(abs(r['sample_mean']+.5)<1e-12 for r in interaction))
        self.assertTrue(all(r['sample_se']<1e-12 and r['batch_se']<1e-12 for r in interaction))

    def test_cross_snr_covariance_and_batch_uncertainty_keep_correct_units(self):
        import evaluate_twc_v1_k8_ordered_encoder_screen as ev
        rows=self.fixture()
        for r in rows:
            r['sum_rate_bps_hz']=10.
            if r['arm']=='ordered_g4':
                delta=-2. if r['sample_index']<4 else 2.
                r['sum_rate_bps_hz'] += delta if r['gamma_db']==0 else -delta if r['gamma_db']==5 else 0.
        paired=ev.paired_statistics(rows,8)
        selected=[r for r in paired if r['contrast']=='ordered_g4_minus_mean_g4_hybrid_1_3' and r['metric']=='sum_rate_bps_hz']
        low=next(r for r in selected if r['estimand']=='low_0_5_10')
        zero=next(r for r in selected if r['estimand']=='0')
        self.assertEqual((low['sample_mean'],low['sample_se'],low['batch_se']),(0.,0.,0.))
        self.assertGreater(zero['sample_se'],0.)
        self.assertAlmostEqual(zero['batch_se'],2.)
        self.assertGreater(zero['batch_se'],zero['sample_se'])
        one=ev.paired_statistics(self.fixture(4),4)
        self.assertTrue(all(r['batch_se'] is None and r['batch_ci95_low'] is None and not r['batch_ci_estimable'] for r in one))

    def test_missing_duplicate_nonfinite_pairing_and_physical_mismatch_rejected(self):
        import evaluate_twc_v1_k8_ordered_encoder_screen as ev
        original=self.fixture()
        for issue in ('missing','duplicate','nonfinite','wrong_id','pairing','energy','rzf','drop_unfavorable'):
            rows=deepcopy(original)
            if issue=='missing':rows.pop()
            elif issue=='duplicate':rows.append(deepcopy(rows[0]))
            elif issue=='nonfinite':rows[0]['desired_signal_power']=float('nan')
            elif issue=='wrong_id':rows[0]['cumulative_updates']=13024
            elif issue=='pairing':rows[0]['channel_fingerprint']='f'*64
            elif issue=='energy':rows[0]['frame_energy_per_ue']=65.
            elif issue=='rzf':rows[0]['rzf_power_mean']=2.
            else:rows=[r for r in rows if r['arm']!='ordered_fdma']
            with self.subTest(issue=issue),self.assertRaises(ValueError):ev.paired_statistics(rows,8)


class OrderedSourceContracts(unittest.TestCase):
    def fixture(self,arm='ordered_g4'):
        import train_twc_v1_k8_ordered_encoder_screen as train
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        from configs import twc_v1_k8_schedule_screen_config as previous
        from tests.test_twc_v1_k8_schedule_screen import schema_resume
        identity=cfg.source_identity(arm);source_arm=identity['arm']
        archived={'fixture-old-source.py':'a'*64}
        payload=schema_resume(source_arm,3024)
        payload['code_sha256']=deepcopy(archived)
        code=dict(commit=cfg.SOURCE_COMMIT,sha256=deepcopy(archived))
        manifest=dict(status='completed',arm=source_arm,seed=cfg.SEED,step=3024,
            cumulative_updates=13024,config_fingerprint=previous.fingerprint(source_arm),
            checkpoint_sha256=identity['sha256'],code=code,
            recipe=previous.recipe(source_arm),source=previous.source_identity(source_arm),
            ce_identity=deepcopy(payload['ce_identity']))
        initial={key:deepcopy(manifest[key]) for key in ('code','recipe','source','ce_identity')}
        # These are declared metadata-schema fixtures, not numerical checkpoints.
        return train,cfg,payload,manifest,initial,archived

    def test_actual_source_guard_accepts_json_roundtrip_and_new_HEAD_without_weakening_old_identity(self):
        import json
        from unittest.mock import patch
        for arm in ('mean_fdma_control','ordered_fdma','mean_g4_control','ordered_g4'):
            train,cfg,payload,manifest,initial,archived=self.fixture(arm)
            original=json.dumps([payload,manifest,initial],sort_keys=True)
            args=json.loads(original)
            with patch.object(train,'source_code_record',side_effect=AssertionError('Historical source guard must not inspect new HEAD')) as current:
                result=train.validate_source_metadata(*args,arm,archived_hashes=archived)
            current.assert_not_called()
            self.assertTrue(result['new_HEAD_may_differ'])
            self.assertEqual(result['source_commit'],cfg.SOURCE_COMMIT)
            self.assertEqual((result['local_step'],result['cumulative_updates']),(3024,13024))
            self.assertEqual(original,json.dumps([payload,manifest,initial],sort_keys=True))

    def test_actual_source_guard_rejects_changed_commit_code_hash_weight_hash_config_path_seed_and_step(self):
        train,cfg,payload,manifest,initial,archived=self.fixture()
        for issue in ('commit','old_code_hash','weight_hash','config','path','seed','unfinished_step','source_hash','initial_identity'):
            p,m,i=deepcopy(payload),deepcopy(manifest),deepcopy(initial)
            if issue=='commit':m['code']['commit']='b'*40
            elif issue=='old_code_hash':p['code_sha256']['fixture-old-source.py']='b'*64
            elif issue=='weight_hash':m['checkpoint_sha256']='b'*64
            elif issue=='config':p['recipe']['learning_rate']=1e-3
            elif issue=='path':p['source']['path']='not-the-source.pt'
            elif issue=='seed':p['seed']+=1
            elif issue=='unfinished_step':
                p['step']=108;p['cumulative_updates']=10108;p['scheduler']['last_epoch']=108
            elif issue=='source_hash':p['source']['sha256']='b'*64
            else:i['source']['sha256']='b'*64
            with self.subTest(issue=issue),self.assertRaises(ValueError):
                train.validate_source_metadata(p,m,i,'ordered_g4',archived_hashes=archived)

    def test_repeated_interruption_retains_verified_checkpoint_and_distinct_source_new_commits(self):
        import json
        import train_twc_v1_k8_ordered_encoder_screen as train
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        code=dict(commit='b'*40,sha256={'new-code.py':'c'*64})
        report=train.training_manifest('ordered_g4',108,data_protocol={'schema':'only'},
            ce={'schema':'only'},code=code,resume_hash='a'*64)
        report.update(status='failed',error='Synthetic interruption before next checkpoint save')
        saved=json.loads(json.dumps(report))
        self.assertEqual(saved['checkpoint_sha256'],'a'*64)
        self.assertEqual((saved['step'],saved['cumulative_updates']),(108,13132))
        self.assertEqual(saved['source_commit'],cfg.SOURCE_COMMIT)
        self.assertEqual(saved['new_experiment_commit'],'b'*40)
        with self.assertRaises(ValueError):
            train.training_manifest('ordered_g4',108,data_protocol={},ce={},code=code)


class OrderedControllerContracts(unittest.TestCase):
    def test_failed_migration_retains_exact_report_pointer(self):
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import patch
        import run_twc_v1_k8_ordered_encoder_screen as controller
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            def failed_child(root,label,command):
                report=Path(command[command.index('--report')+1])
                report.write_text(json.dumps(dict(status='failed',error='Synthetic fixed-tolerance failure',
                    comparisons={'ordered_g4':{'raw':{'passed':False,'max_abs':.01,'max_rel':.02},
                        'energy':{'passed':True,'max_abs':0.}}})),encoding='utf-8')
                raise RuntimeError('Synthetic child failure')
            with patch.object(controller,'_child',side_effect=failed_child):
                with self.assertRaises(RuntimeError):controller.migration_preflight(folder)
            pointer=json.loads((folder/'migration_preflight_pointer.json').read_text())
            report=Path(pointer['report'])
            self.assertEqual(report.parent,folder)
            self.assertEqual(json.loads(report.read_text())['comparisons']['ordered_g4']['raw']['passed'],False)
            # Exercise the actual summary route, including failed leaves and
            # power diagnostics that have no relative-error field.
            root=folder/'v2';batch=root/'batches/batch_fixture';batch.mkdir(parents=True)
            (root/'last_batch.json').write_text(json.dumps({'directory':str(batch)}),encoding='utf-8')
            (batch/'batch_summary.json').write_text(json.dumps(dict(status='failed',code={'commit':'fixture'},stages={})),encoding='utf-8')
            copied_report=batch/'report.json';copied_report.write_bytes(report.read_bytes())
            (batch/'migration_preflight_pointer.json').write_text(json.dumps({'report':str(copied_report)}),encoding='utf-8')
            original_guard=controller.cfg.inside_root;stream=io.StringIO()
            with patch.object(controller.cfg,'OUTPUT_ROOT',root),patch.object(controller.cfg,'inside_root',
                    side_effect=lambda p:original_guard(p,root=root)),contextlib.redirect_stdout(stream):
                self.assertEqual(controller.print_summary(),0)
            self.assertIn('migration_failed_field=',stream.getvalue())
            self.assertIn('comparisons.ordered_g4.raw',stream.getvalue())
            self.assertIn('"all_passed": false',stream.getvalue())

    def test_valid_terminal_checkpoint_can_finalize_without_updates_and_preserves_old_receipt(self):
        import train_twc_v1_k8_ordered_encoder_screen as train
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        code=dict(commit='b'*40,sha256={'new.py':'c'*64});ce={'schema':'only'};arm='ordered_g4'
        payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=3024,cumulative_updates=16048,
            recipe=cfg.recipe(arm),config_fingerprint=cfg.fingerprint(arm),source=cfg.source_identity(arm),
            codec_model={'schema':'only'},optimizer={'schema':'only'},scheduler={'last_epoch':3024},
            rng={k:'schema only' for k in ('python','numpy','torch_cpu','torch_cuda')},data_protocol={'schema':'only'},
            ce_identity=ce,code_sha256=code['sha256'])
        previous=dict(status='failed',checkpoint_sha256='a'*64,error='Synthetic receipt-write interruption',failure_history=[])
        before=deepcopy(previous)
        recovered=train.completed_receipt(arm,payload,previous,checkpoint_sha='a'*64,ce=ce,code=code)
        self.assertEqual(previous,before)
        self.assertEqual((recovered['status'],recovered['step'],recovered['cumulative_updates']),('completed',3024,16048))
        self.assertEqual(recovered['additional_optimizer_updates'],0)
        with self.assertRaises(ValueError):
            train.completed_receipt(arm,payload,previous,checkpoint_sha='d'*64,ce=ce,code=code)

    def test_actual_run_arm_pauses108_then_resumes_to3024_and_never_repeats_completed(self):
        import json
        import tempfile
        from unittest.mock import patch
        import run_twc_v1_k8_ordered_encoder_screen as controller
        arm='ordered_g4'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);destination=root/'training'/arm;commands=[]
            def child(folder,label,command):
                commands.append(command)
                destination.mkdir(parents=True,exist_ok=True)
                if command[-2:]==['--stop-after','108']:
                    self.assertFalse((destination/'checkpoint_latest.pt').exists())
                    (destination/'checkpoint_latest.pt').write_bytes(b'Subprocess control fixture, not a numerical checkpoint')
                    state,step='paused',108
                else:
                    self.assertEqual(command[-2:],['--resume','--train'])
                    self.assertEqual(json.loads((destination/'manifest.json').read_text())['step'],108)
                    state,step='completed',3024
                (destination/'manifest.json').write_text(json.dumps(dict(status=state,step=step,cumulative_updates=13024+step)),encoding='utf-8')
                return str(root/(label+'.log'))
            def completed(*args):return dict(report=dict(status='completed',step=3024),_evidence=[])
            with patch.object(controller.cfg,'OUTPUT_ROOT',root),patch.object(controller,'_child',side_effect=child),patch.object(controller,'verify_training',side_effect=completed):
                controller.run_arm(root,arm)
                self.assertEqual([c[-2:] for c in commands],[['--stop-after','108'],['--resume','--train']])
                controller.run_arm(root,arm)
                self.assertEqual(len(commands),2)
                (destination/'manifest.json').write_text(json.dumps(dict(status='paused',step=108)),encoding='utf-8')
                controller.run_arm(root,arm)
                self.assertEqual(len(commands),3)
                self.assertEqual(commands[-1][-2:],['--resume','--train'])

    def test_failed_technical_pause_stops_before_continue_and_preserves_failure_evidence(self):
        import tempfile
        from unittest.mock import patch,Mock
        import run_twc_v1_k8_ordered_encoder_screen as controller
        arm='ordered_fdma'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);destination=root/'training'/arm
            def failed_child(*args):
                destination.mkdir(parents=True)
                (destination/'failure.json').write_text('{"schema_only":"failed technical pause"}',encoding='utf-8')
                raise RuntimeError('Synthetic technical-pause failure')
            verify=Mock()
            with patch.object(controller.cfg,'OUTPUT_ROOT',root),patch.object(controller,'_child',side_effect=failed_child) as child,patch.object(controller,'verify_training',verify):
                with self.assertRaises(RuntimeError):controller.run_arm(root,arm)
                self.assertEqual(child.call_count,1)
                before=(destination/'failure.json').read_bytes()
                with self.assertRaises(FileExistsError):controller.run_arm(root,arm)
                self.assertEqual(child.call_count,1)
                self.assertEqual(before,(destination/'failure.json').read_bytes())
            verify.assert_not_called()

    def test_unowned_training_directory_is_retained_without_launch(self):
        import tempfile
        from unittest.mock import patch,Mock
        import run_twc_v1_k8_ordered_encoder_screen as controller
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);arm='mean_fdma_control';destination=root/'training'/arm
            destination.mkdir(parents=True);marker=destination/'manual.txt';marker.write_text('preserve',encoding='utf-8')
            child=Mock();verify=Mock()
            with patch.object(controller.cfg,'OUTPUT_ROOT',root),patch.object(controller,'_child',child),patch.object(controller,'verify_training',verify):
                with self.assertRaises(FileExistsError):controller.run_arm(root,arm)
            child.assert_not_called();verify.assert_not_called()
            self.assertEqual(marker.read_text(encoding='utf-8'),'preserve')


@unittest.skipIf(MODEL_IMPORT_ERROR is not None, MODEL_IMPORT_ERROR or '')
class OrderedPreflightCollectionTests(unittest.TestCase):
    def test_old_optimizer_evidence_uses_original_parameter_order_before_two_new_residuals(self):
        import train_twc_v1_k8_ordered_encoder_screen as train
        mean,target=OrderedAlgebraTests()._fixture()
        source=deepcopy(mean.state_dict())
        ordered.migrate_mean_state(target,source)
        named=list(mean.named_parameters())
        # Explicit small tensor moment-schema fixture, not a trained checkpoint;
        # no optimizer step, full model or real channel is used.
        moments={index:dict(step=torch.tensor(3024.),exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.ones_like(parameter)) for index,(_,parameter) in enumerate(named)}
        payload=dict(codec_model=source,optimizer=dict(param_groups=[{'params':list(range(len(named)))}],state=moments))
        before=deepcopy(payload)
        control=train.check_source_optimizer_parameters(torch,mean,payload,ordered=False)
        actual=train.check_source_optimizer_parameters(torch,target,payload,ordered=True)
        self.assertEqual(actual['original_parameter_count'],control['original_parameter_count'])
        self.assertEqual(len(list(target.parameters()))-actual['original_parameter_count'],2)
        self.assertEqual(set(actual['new_residual_parameters_excluded']),set(ordered.RESIDUAL_WEIGHT_KEYS))
        self.assertFalse(actual['source_moments_reused'])
        for key in source:self.assertTrue(torch.equal(payload['codec_model'][key],before['codec_model'][key]))
        for index,state in moments.items():
            for key in state:self.assertTrue(torch.equal(state[key],before['optimizer']['state'][index][key]))
        for issue in ('new_count','wrong_shape','wrong_dtype','unknown_parameter'):
            bad=deepcopy(payload)
            if issue=='new_count':bad['optimizer']['param_groups'][0]['params']+=list(range(len(named),len(named)+2))
            elif issue=='wrong_shape':bad['optimizer']['state'][len(named)-1]['exp_avg']=torch.zeros(99)
            elif issue=='wrong_dtype':bad['optimizer']['state'][0]['exp_avg_sq']=bad['optimizer']['state'][0]['exp_avg_sq'].double()
            else:bad['optimizer']['state'][len(named)]=deepcopy(moments[0])
            with self.subTest(issue=issue),self.assertRaises(ValueError):
                train.check_source_optimizer_parameters(torch,target,bad,ordered=True)

    def fixture(self,*,repeat_shift=0.,metric_error=False):
        from types import SimpleNamespace
        outputs={name:dict(H_dl_hat=torch.tensor([2.]),tx={'X':torch.tensor([1.])},
            energy={'actual_data_per_ue':torch.tensor([48.])}) for name in ('mean','ordered')}
        calls={'mean':0,'ordered':0}
        def helper(name):
            def metrics(output,truth):
                calls[name]+=1
                if metric_error and name=='mean':raise RuntimeError('Explicit synthetic metric failure')
                shift=repeat_shift if name=='mean' and calls[name]%2==0 else 0.
                return dict(precoder=torch.tensor([1.+shift]),rzf_per_subcarrier_power=torch.ones(1),
                    dl_nmse_per_user=torch.tensor([.1]),sum_rate_bps_hz=torch.tensor([3.]))
            return SimpleNamespace(metrics=metrics)
        return outputs,{name:helper(name) for name in calls},calls

    def test_multiple_failed_fields_still_collect_nmse_rate_replays_and_serializable_evidence(self):
        import json
        import train_twc_v1_k8_ordered_encoder_screen as train
        outputs,chains,calls=self.fixture()
        outputs['ordered']['tx']['X']+=1.
        outputs['ordered']['H_dl_hat']+=1.
        result=train.collect_output_comparisons(torch,outputs,chains,torch.zeros(1))
        self.assertFalse(result['transmit']['passed']);self.assertFalse(result['H_hat']['passed'])
        for name in ('RZF','dl_nmse','rate'):self.assertTrue(result[name]['passed'])
        self.assertEqual(calls,{'mean':2,'ordered':2})
        failures=train.failed_preflight_checks({'condition':result})
        self.assertIn('condition.transmit',failures);self.assertIn('condition.H_hat',failures)
        self.assertIn('rate',json.loads(json.dumps(result,allow_nan=False)))

    def test_identical_H_hat_repeat_RZF_is_separate_and_stops_even_inside_cross_model_tolerance(self):
        import train_twc_v1_k8_ordered_encoder_screen as train
        outputs,chains,calls=self.fixture(repeat_shift=1e-6)
        result=train.collect_output_comparisons(torch,outputs,chains,torch.zeros(1))
        repeat=result['RZF_repeat']['mean']
        self.assertLess(repeat['max_abs'],repeat['atol'])
        self.assertFalse(repeat['exact_equal']);self.assertTrue(repeat['H_hat_unchanged'])
        self.assertFalse(repeat['passed']);self.assertTrue(result['RZF']['passed'])
        self.assertTrue(result['dl_nmse']['passed']);self.assertTrue(result['rate']['passed'])
        self.assertIn('RZF_repeat.mean',train.failed_preflight_checks(result))

    def test_metric_exception_and_nonfinite_diagnostics_do_not_hide_later_fields_or_break_JSON(self):
        import json
        import train_twc_v1_k8_ordered_encoder_screen as train
        outputs,chains,calls=self.fixture(metric_error=True)
        outputs['ordered']['energy']['actual_data_per_ue'].fill_(float('nan'))
        result=train.collect_output_comparisons(torch,outputs,chains,torch.zeros(1))
        self.assertTrue(result['H_hat']['passed']);self.assertTrue(result['transmit']['passed'])
        for name in ('RZF','dl_nmse','rate'):self.assertFalse(result[name]['passed'])
        self.assertFalse(result['energy_ordered']['passed']);self.assertIsNone(result['energy_ordered']['max_abs'])
        self.assertEqual(calls['ordered'],2)
        json.dumps(result,allow_nan=False)


class OrderedV2IsolationTests(unittest.TestCase):
    def test_v2_paths_and_format_reject_v1_without_reading_weights(self):
        import evaluate_twc_v1_k8_ordered_encoder_screen as ev
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        self.assertEqual(cfg.NAMESPACE,'twc_v1_k8_ordered_encoder_screen_v2')
        self.assertEqual(cfg.architecture('ordered_g4')['head_input_width'],128)
        self.assertEqual(cfg.architecture('ordered_g4')['ordered_residual_input_width'],3968)
        paths={arm:cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt' for arm in cfg.ARMS}
        self.assertEqual(set(ev.validate_checkpoint_paths(paths)),set(cfg.ARMS))
        bad=dict(paths);bad['ordered_g4']=cfg.OUTPUT_ROOT.parent/'twc_v1_k8_ordered_encoder_screen_v1/training/ordered_g4/checkpoint_latest.pt'
        with self.assertRaises(ValueError):ev.validate_checkpoint_paths(bad)
        payload=dict(format=cfg.CHECKPOINT_FORMAT,arm='ordered_g4',seed=cfg.SEED,step=108,cumulative_updates=13132,
            recipe=cfg.recipe('ordered_g4'),config_fingerprint=cfg.fingerprint('ordered_g4'),source=cfg.source_identity('ordered_g4'),
            codec_model={'schema':'only'},optimizer={'schema':'only'},scheduler={'last_epoch':108},
            rng={k:'schema only' for k in ('python','numpy','torch_cpu','torch_cuda')},data_protocol={'schema':'only'},
            ce_identity={'schema':'only'},code_sha256={'schema':'only'})
        cfg.validate_resume_metadata(payload,'ordered_g4')
        payload['format']='twc_v1_k8_ordered_encoder_screen_v1'
        with self.assertRaises(ValueError):cfg.validate_resume_metadata(payload,'ordered_g4')

    def test_new_output_creation_and_rejection_leave_old_failure_bytes_unchanged(self):
        import tempfile
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);v1=root/'twc_v1_k8_ordered_encoder_screen_v1';v1.mkdir()
            evidence=v1/'report.json';evidence.write_bytes(b'{"status":"failed","historical":true}')
            before=evidence.read_bytes();v2=root/cfg.NAMESPACE;output=v2/'training/ordered_g4'
            cfg.prepare_output(output,root=v2)
            with self.assertRaises(ValueError):cfg.prepare_output(v1/'training/ordered_g4',root=v2)
            with self.assertRaises(FileExistsError):cfg.prepare_output(output,root=v2)
            self.assertEqual(evidence.read_bytes(),before)

    def test_controller_requires_complete_v2_comparisons_replay_and_gradients(self):
        from unittest.mock import patch
        import run_twc_v1_k8_ordered_encoder_screen as controller
        from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
        code=dict(commit='b'*40,sha256={'fixture':'c'*64})
        good=dict(passed=True,max_abs=0.,max_rel=0.)
        report=dict(status='passed',namespace=cfg.NAMESPACE,code=code,comparisons_complete=True,
            tolerances=deepcopy(controller.EQUIVALENCE_TOLERANCES),failed_checks=[],comparisons={},
            gradient_checks={},parameter_checks={},buffer_checks={})
        for arm in ('ordered_fdma','ordered_g4'):
            comparisons={f'raw_G{g}_{branch}':deepcopy(good) for g in (1,2,4) for branch in ('key','shared')}
            ratios=(None,) if arm=='ordered_fdma' else cfg.ENERGY_RATIOS
            for gamma in cfg.GAMMAS:
                for ratio in ratios:
                    fields={key:deepcopy(good) for key in ('transmit','H_hat','RZF','dl_nmse','rate',
                        'energy_mean','energy_ordered','RZF_power_mean','RZF_power_ordered')}
                    fields['RZF_repeat']={name:dict(good,exact_equal=True,H_hat_unchanged=True) for name in ('mean','ordered')}
                    comparisons[f'gamma{gamma}_ratio{ratio}']=fields
            report['comparisons'][arm]=comparisons
            report['parameter_checks'][arm]={'passed':True};report['buffer_checks'][arm]={'passed':True}
            gradient={key:dict(passed=True,exists=True,finite=True,norm=1.) for key in cfg.architecture(arm)['ordered_residual_weight_keys']}
            gradient.update({key:{'passed':True} for key in ('state_unchanged','frozen_CE','active_gradients')})
            report['gradient_checks'][arm]=gradient
        with patch.object(controller,'source_code_record',return_value=code):
            controller.verify_preflight_report(report)
            first_condition=f'gamma{cfg.GAMMAS[0]}_ratioNone'
            for issue in ('v1','incomplete','missing_rate','missing_gradient','RZF_replay','tolerance'):
                bad=deepcopy(report)
                if issue=='v1':bad['namespace']='twc_v1_k8_ordered_encoder_screen_v1'
                elif issue=='incomplete':bad['comparisons_complete']=False
                elif issue=='missing_rate':bad['comparisons']['ordered_fdma'][first_condition].pop('rate')
                elif issue=='missing_gradient':bad['gradient_checks']['ordered_g4'].pop('encoder.key_window_residual.weight')
                elif issue=='RZF_replay':bad['comparisons']['ordered_fdma'][first_condition]['RZF_repeat']['mean']['exact_equal']=False
                else:bad['tolerances']['RZF']['atol']=1.
                with self.subTest(issue=issue),self.assertRaises(ValueError):controller.verify_preflight_report(bad)


if __name__=='__main__':unittest.main()
