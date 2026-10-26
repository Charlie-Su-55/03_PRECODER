"""Metadata/CLI/loop-hook contracts only; no neural forward or training."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,Path(__file__).resolve().parent):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from configs.swin_adapt_train_config import swin_adapt_config,SEEDS,optimizer_group_spec,validate_config
from configs.as_spatial_train_config import as_spatial_config
from configs.baseline_config import make_config
from cli_contracts import assert_lightweight_cli
import train_swin_adaptation as entry


def resume_metadata(cfg,step=500,selected_step=None):
    """Schema-only fixture. Tensor sentinels are never presented as real weights."""
    selected=step if selected_step is None else selected_step
    groups,last=[],[]
    for index,spec in enumerate(optimizer_group_spec(cfg)):
        peak=spec['peak_lr'];lr,beta=entry.cycle_values(cfg,peak,step)
        groups.append(dict(group_name=spec['name'],parameter_names=[spec['prefix']+'test'],params=[index],
            lr=lr,initial_lr=peak/25,max_lr=peak,min_lr=peak/25/10000,weight_decay=1e-5,
            eps=1e-8,base_momentum=.85,max_momentum=.95,betas=(beta,.999),amsgrad=False,
            maximize=False,capturable=False,differentiable=False,foreach=None,fused=None))
        last.append(lr)
    phases=[dict(end_step=1999.,start_lr='initial_lr',end_lr='max_lr',start_momentum='max_momentum',end_momentum='base_momentum'),
            dict(end_step=19999,start_lr='max_lr',end_lr='min_lr',start_momentum='base_momentum',end_momentum='max_momentum')]
    return dict(config=cfg.to_dict(),config_fingerprint=entry.fingerprint(cfg),step=step,
        state={'encoder.test':None,'decoder.test':None},best_score=20.,selected_best_step=selected,
        validation=dict(rates={str(n):20. for n in cfg.VAL_SNR_LIST},mean_rate=20.,mean_all=20.,
                        selection_score=20.,selection_name='mean_all'),
        optimizer=dict(param_groups=groups,state={0:{'synthetic':True},1:{'synthetic':True}}),
        scheduler=dict(total_steps=20000,last_epoch=step,_step_count=step+1,_schedule_phases=phases,
            _anneal_func_type='cos',cycle_momentum=True,use_beta1=True,
            base_lrs=[s['peak_lr']/25 for s in optimizer_group_spec(cfg)],_last_lr=last),
        rng_state=dict(python='schema-only',numpy='schema-only',torch='schema-only',cuda=['schema-only']))


def best_metadata(cfg,step=0,score=20.):
    result=resume_metadata(cfg,step)
    result['best_score']=score
    result['validation'].update(rates={str(n):score for n in cfg.VAL_SNR_LIST},
        mean_rate=score,mean_all=score,selection_score=score)
    if step==0:result['optimizer']['state']={}
    return result


class ConfigurationTests(unittest.TestCase):
    def test_fixed_recipe_groups_and_all_trainable_declaration(self):
        cfg=swin_adapt_config(SEEDS[0]);validate_config(cfg)
        self.assertEqual((cfg.TOTAL_STEPS,cfg.BATCH_SIZE,cfg.VAL_BATCH,cfg.VAL_INTERVAL),(20000,48,64,500))
        self.assertEqual((cfg.WARMUP_PCT,cfg.WEIGHT_DECAY,cfg.GRAD_CLIP),(.1,1e-5,.5))
        self.assertEqual(cfg.DOWNLINK_SNR_DB,25.)
        self.assertEqual(cfg.STAGE1_SNR,cfg.STAGE2_SNR)
        self.assertEqual(cfg.STAGE3_SNR,(0.,25.))
        self.assertEqual([g['peak_lr'] for g in optimizer_group_spec(cfg)],[2e-5,1e-4])
        self.assertTrue(cfg.to_dict()['adaptation_protocol']['source_step_zero_competes'])

    def test_same_channel_data_as_existing_spatial_control_for_each_seed(self):
        for seed in SEEDS:
            cfg=swin_adapt_config(seed)
            self.assertEqual(cfg.channel_data_protocol,as_spatial_config('joint',seed).channel_data_protocol)
        self.assertNotEqual(swin_adapt_config(SEEDS[0]).channel_data_protocol,
                            swin_adapt_config(SEEDS[1]).channel_data_protocol)

    def test_fixed_source_and_architecture_not_old_recipe_overridden_globally(self):
        old=make_config('swin','k4_n256_d256').to_dict()
        cfg=swin_adapt_config(SEEDS[0])
        self.assertEqual(old,make_config('swin','k4_n256_d256').to_dict())
        self.assertEqual(cfg.source_identity['checkpoint_step'],49000)
        self.assertEqual(cfg.source_identity['checkpoint_sha256'],
            'c913f3477f6af74665d7ba1b11a7a8e06bdea8c051822205ace1a8a071489276')
        self.assertEqual((cfg.D_F,cfg.RZF_REG,cfg.SWIN_EMBED_DIM),(64,.001,128))
        cfg.source_identity['checkpoint_sha256']='0'*64
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_seed_source_protocol_and_lr_changes_affect_fingerprint(self):
        cfg=swin_adapt_config(SEEDS[0]);initial=entry.fingerprint(cfg)
        self.assertNotEqual(initial,entry.fingerprint(swin_adapt_config(SEEDS[1])))
        for field in ('SEED','LR','WEIGHT_DECAY'):
            changed=deepcopy(cfg);setattr(changed,field,getattr(cfg,field)+1)
            self.assertNotEqual(initial,entry.fingerprint(changed))
        changed=deepcopy(cfg);changed.channel_data_protocol['base_seed']+=1
        self.assertNotEqual(initial,entry.fingerprint(changed))

    def test_fingerprint_matches_actual_legacy_function_without_importing_runtime(self):
        tree=ast.parse((ROOT/'baseline_train.py').read_text(encoding='utf-8'))
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='config_fingerprint')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),func],type_ignores=[])
        ast.fix_missing_locations(module)
        context={'json':json,'hashlib':hashlib}
        exec(compile(module,'baseline_fingerprint','exec'),context)
        cfg=swin_adapt_config(SEEDS[0])
        self.assertEqual(entry.fingerprint(cfg),context['config_fingerprint'](cfg))


class ResumeTests(unittest.TestCase):
    def test_500_pause_and_complete_20k_not_extra_budget(self):
        cfg=swin_adapt_config(SEEDS[0])
        entry.validate_resume_payload(resume_metadata(cfg,500),cfg)
        entry.validate_resume_payload(resume_metadata(cfg,20000),cfg,allow_completed=True)
        with self.assertRaises(ValueError):entry.validate_resume_payload(resume_metadata(cfg,20000),cfg)

    def test_source_step_zero_can_remain_selected_best(self):
        cfg=swin_adapt_config(SEEDS[0]);latest=resume_metadata(cfg,20000,0)
        entry.validate_resume_payload(latest,cfg,allow_completed=True)
        entry.validate_best_metadata(best_metadata(cfg,0),latest,cfg)
        entry.validate_best_metadata(best_metadata(cfg,500),resume_metadata(cfg,20000,500),cfg)

    def test_missing_resume_fields_and_wrong_source_seed_rejected(self):
        cfg=swin_adapt_config(SEEDS[0])
        for field in ('config','config_fingerprint','optimizer','scheduler','rng_state','selected_best_step'):
            payload=resume_metadata(cfg);payload.pop(field)
            with self.subTest(field=field),self.assertRaises((ValueError,TypeError)):
                entry.validate_resume_payload(payload,cfg)
        with self.assertRaises(ValueError):
            entry.validate_resume_payload(resume_metadata(cfg),swin_adapt_config(SEEDS[1]))

    def test_group_swaps_collapses_and_learning_rate_mismatch_rejected(self):
        cfg=swin_adapt_config(SEEDS[0])
        for kind in ('swap','collapse','lr','names','ids','moments'):
            payload=resume_metadata(cfg);groups=payload['optimizer']['param_groups']
            if kind=='swap':groups.reverse()
            elif kind=='collapse':groups.pop()
            elif kind=='lr':groups[0]['lr']=groups[1]['lr']
            elif kind=='names':groups[0]['parameter_names']=['decoder.test']
            elif kind=='ids':groups[1]['params']=[0]
            else:payload['optimizer']['state'].pop(1)
            with self.subTest(kind=kind),self.assertRaises(ValueError):entry.validate_resume_payload(payload,cfg)

    def test_scheduler_restart_incomplete_rng_and_stale_best_rejected(self):
        cfg=swin_adapt_config(SEEDS[0])
        for kind in ('budget','restart','rng','stale'):
            payload=resume_metadata(cfg)
            if kind=='budget':payload['scheduler']['total_steps']=500
            elif kind=='restart':payload['scheduler']['last_epoch']=0
            elif kind=='rng':payload['rng_state']['cuda']=[]
            else:payload['best_score']=19.
            with self.subTest(kind=kind),self.assertRaises(ValueError):entry.validate_resume_payload(payload,cfg)

    def test_mean_all_must_be_the_six_point_average(self):
        cfg=swin_adapt_config(SEEDS[0])
        for kind in ('name','missing','mean'):
            payload=resume_metadata(cfg)
            if kind=='name':payload['validation']['selection_name']='mean_rate'
            elif kind=='missing':payload['validation']['rates'].pop('0')
            else:payload['validation']['mean_all']=21.
            with self.subTest(kind=kind),self.assertRaises(ValueError):entry.validate_resume_payload(payload,cfg)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tree=ast.parse((ROOT/'baseline_train.py').read_text(encoding='utf-8'))
        self.train=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one')

    def test_default_optional_hooks_and_legacy_optimizer_unchanged(self):
        defaults={a.arg:ast.literal_eval(d) for a,d in zip(self.train.args.kwonlyargs,self.train.args.kw_defaults) if d is not None}
        for key in ('model_factory','optimizer_scheduler_factory','channel_data','downlink_snr_db','resume_validator'):
            self.assertIsNone(defaults[key])
        self.assertIs(defaults['initial_validation'],False)
        branch=next(n for n in self.train.body if isinstance(n,ast.If) and ast.unparse(n.test)=='optimizer_scheduler_factory is None')
        expected=ast.parse('''optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=cfg.LR,total_steps=cfg.TOTAL_STEPS,pct_start=cfg.WARMUP_PCT,anneal_strategy="cos")''').body
        self.assertEqual([ast.dump(n) for n in branch.body],[ast.dump(n) for n in expected])

    def test_fixed_dl_is_separate_from_feedback_snr_in_train_and_validation(self):
        validation=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
        for func in (self.train,validation):
            assign=next(n for n in ast.walk(func) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dl_snr' for t in n.targets))
            expression=compile(ast.Expression(body=assign.value),'dl-snr','eval')
            for fb in (0,5,10,15,20,25):
                self.assertEqual(eval(expression,{'snr_db':fb,'downlink_snr_db':25.}),25.)
                self.assertEqual(eval(expression,{'snr_db':fb,'downlink_snr_db':None}),fb)

    def test_initial_selection_precedes_updates_and_adapted_latest_score_not_stale(self):
        initial=next(n for n in self.train.body if isinstance(n,ast.If) and ast.unparse(n.test)=='initial_validation')
        loop=next(n for n in self.train.body if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
        self.assertLess(initial.lineno,loop.lineno)
        self.assertIn('checkpoint_payload(cfg, 0',ast.unparse(initial))
        adapted=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='initial_validation')
        text=ast.unparse(adapted)
        self.assertLess(text.index('best_score, best_step ='),text.index('checkpoint_payload('))
        self.assertIn('torch.save(payload, latest_path)',text)

    def test_fresh_output_protection_and_stdlib_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'control';cfg=swin_adapt_config(SEEDS[0],root)
            entry.check_run_state(cfg,resume=False,allowed_root=root)
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError):entry.check_run_state(cfg,resume=False,allowed_root=root)
            with self.assertRaises(FileNotFoundError):entry.check_run_state(cfg,resume=True,allowed_root=root)
        assert_lightweight_cli(self,'train_swin_adaptation',[['--seed',str(s),'--dry-run'] for s in SEEDS],
            "assert len(plans)==2; assert all(p['scheduler_total_steps']==20000 and p['source_step_zero_competes'] for p in plans)")


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Optional real CPU leaf-tensor optimizer test needs installed PyTorch; never installed/faked.')
class TinyOptimizerTests(unittest.TestCase):
    def test_two_actual_lr_groups_no_backward_update_or_model(self):
        import torch
        owner=SimpleNamespace(named_parameters=lambda:iter([('encoder.test',torch.tensor([1.],requires_grad=True)),
                                                            ('decoder.test',torch.tensor([1.],requires_grad=True))]))
        cfg=swin_adapt_config(SEEDS[0]);optimizer,scheduler=entry.optimizer_scheduler_factory(cfg,owner)
        self.assertEqual([g['max_lr'] for g in optimizer.param_groups],[2e-5,1e-4])
        self.assertEqual([g['group_name'] for g in optimizer.param_groups],['encoder','decoder'])
        self.assertEqual(scheduler.total_steps,20000);self.assertEqual(scheduler.last_epoch,0)
        self.assertEqual(optimizer.state,{})


if __name__=='__main__':unittest.main()
