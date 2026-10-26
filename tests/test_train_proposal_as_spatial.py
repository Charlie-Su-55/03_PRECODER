"""AS-spatial recipe/group/resume contracts; no model or training execution."""
import ast
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from configs.as_spatial_train_config import (
    ARMS, SEEDS, RECIPE, OUTPUT_ROOT, as_spatial_config, validate_config,
    make_channel_data, optimizer_group_spec,
)
from configs.as_receiver_train_config import as_receiver_config
from configs.train_config import TRAINING_RECIPES
import train_proposal_as_spatial as entry
from cli_contracts import assert_lightweight_cli


def resume_metadata(cfg, step=500):
    """Stdlib metadata fixture only; tensor sentinel values cannot be restored."""
    state = {'running_pwr': None, 'decoder.test': None,
             'encoder.aircomp.test': None, 'cond_embed.test': None}
    groups, rates, next_id = [], [], 0
    for spec in optimizer_group_spec(cfg):
        peak = spec['peak_lr']
        rate, beta1 = entry.cycle_values(cfg, peak, step)
        names = ['decoder.test'] if spec['name'] == 'bs' else ['encoder.aircomp.test', 'cond_embed.test']
        ids = list(range(next_id, next_id + len(names)))
        next_id += len(names)
        groups.append(dict(group_name=spec['name'], parameter_names=names, params=ids,
            lr=rate, initial_lr=peak/25, max_lr=peak, min_lr=peak/25/10000,
            weight_decay=cfg.WEIGHT_DECAY, eps=1e-8, base_momentum=0.85, max_momentum=0.95,
            betas=(beta1,0.999), amsgrad=False, maximize=False, capturable=False,
            differentiable=False, foreach=None, fused=None))
        rates.append(rate)
    return dict(state=state, config=cfg.to_dict(), config_fingerprint=entry.fingerprint(cfg),
        allocation_grid=[(0,256)], step=step, best_metric=1.25,
        optimizer=dict(state={i: dict(step=None,exp_avg=None,exp_avg_sq=None) for i in range(next_id)},
                       param_groups=groups),
        scheduler=dict(total_steps=20000, last_epoch=step, _step_count=step+1,
            _schedule_phases=[dict(end_step=1999.0,start_lr='initial_lr',end_lr='max_lr',
                start_momentum='max_momentum',end_momentum='base_momentum'),
                dict(end_step=19999,start_lr='max_lr',end_lr='min_lr',
                start_momentum='base_momentum',end_momentum='max_momentum')],
            _anneal_func_type='cos', cycle_momentum=True, use_beta1=True,
            base_lrs=[spec['peak_lr']/25 for spec in optimizer_group_spec(cfg)], _last_lr=rates),
        rng_state=dict(python='schema only',numpy='schema only',torch='schema only',cuda=['schema only']))


class ConfigTests(unittest.TestCase):
    def test_fixed_twenty_thousand_recipe_and_groups(self):
        self.assertEqual(ARMS, ('frozen_ue','joint'))
        self.assertEqual(SEEDS, (20261030,20261031))
        self.assertEqual(RECIPE.total_steps,20000)
        for field,value in dict(batch_size=48,lr=1e-4,weight_decay=1e-5,warmup_pct=.1,
            grad_clip=.5,val_interval=500,val_batch=64,downlink_snr_db=25.,best_metric='mean_all',
            ph1_w_dir=0.,ph1_w_mse=0.,ph2_w_dir=0.,ph2_w_mse=0.,
            stage1_snr=(0.,25.),stage2_snr=(0.,25.),stage3_snr=(0.,25.),
            val_snr_list=(0,5,10,15,20,25)).items():
            self.assertEqual(getattr(RECIPE,field),value,field)
        self.assertEqual(optimizer_group_spec(as_spatial_config('frozen_ue',SEEDS[0])),
                         [dict(name='bs',peak_lr=1e-4)])
        self.assertEqual(optimizer_group_spec(as_spatial_config('joint',SEEDS[0])),
                         [dict(name='bs',peak_lr=1e-4),dict(name='ue',peak_lr=2e-5)])

    def test_same_channels_as_existing_control_within_each_seed(self):
        for seed in SEEDS:
            reference=as_receiver_config('full_control',seed)
            for arm in ARMS:
                cfg=as_spatial_config(arm,seed)
                validate_config(cfg)
                self.assertEqual(cfg.channel_data_protocol,reference.channel_data_protocol)
                self.assertEqual(cfg.source_identity,reference.source_identity)
                self.assertEqual(make_channel_data(cfg).seed_for('training',500),
                                 make_channel_data(reference).seed_for('training',500))

    def test_independent_paths_and_explicit_structural_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new'
            configs=[as_spatial_config(arm,seed,root) for seed in SEEDS for arm in ARMS]
            self.assertEqual(len({cfg.SAVE_DIR for cfg in configs}),4)
            self.assertFalse(root.exists())
        cfg=configs[0]
        meta=cfg.to_dict()['architecture_metadata']
        self.assertEqual(meta['decoder']['spatial_expansion'],[769,8192])
        self.assertEqual(meta['decoder']['angular_delay_shape'],[32,32])
        self.assertEqual(meta['decoder']['swin_blocks'],4)
        self.assertFalse(meta['decoder']['shifted_windows'])
        self.assertEqual(meta['receiver']['solve_dtype'],'complex64')
        self.assertEqual(meta['running_power_ema_momentum'],.02)

    def test_legacy_registry_unchanged_and_invalid_recipe_rejected(self):
        before=deepcopy(TRAINING_RECIPES)
        cfg=as_spatial_config('joint',SEEDS[0])
        self.assertEqual(before,TRAINING_RECIPES)
        self.assertNotIn(RECIPE.name,TRAINING_RECIPES)
        with self.assertRaises(ValueError):
            validate_config(replace(cfg,recipe=replace(cfg.recipe,lr=2e-5)))
        for arm,seed in [('full_control',SEEDS[0]),('joint',42),('joint',True)]:
            with self.assertRaises(ValueError):
                as_spatial_config(arm,seed)


class ResumeTests(unittest.TestCase):
    def test_both_group_schemas_accept_smoke_and_explicit_completion(self):
        for arm in ARMS:
            cfg=as_spatial_config(arm,SEEDS[0])
            entry.validate_resume_payload(resume_metadata(cfg),cfg)
            entry.validate_resume_payload(resume_metadata(cfg,20000),cfg,allow_completed=True)
            with self.assertRaises(ValueError):
                entry.validate_resume_payload(resume_metadata(cfg,20000),cfg)

    def test_missing_resume_fields_and_cross_arm_seed_are_rejected(self):
        cfg=as_spatial_config('joint',SEEDS[0])
        for field in ('config','config_fingerprint','state','optimizer','scheduler','rng_state','step','allocation_grid'):
            payload=resume_metadata(cfg)
            del payload[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                entry.validate_resume_payload(payload,cfg)
        for other in (as_spatial_config('frozen_ue',SEEDS[0]),as_spatial_config('joint',SEEDS[1])):
            with self.assertRaises(ValueError):
                entry.validate_resume_payload(resume_metadata(other),cfg)

    def test_two_group_lr_values_cannot_collapse_or_swap(self):
        cfg=as_spatial_config('joint',SEEDS[0])
        for kind in ('count','names','peak','last_lr'):
            payload=resume_metadata(cfg)
            groups=payload['optimizer']['param_groups']
            if kind=='count': groups.pop()
            elif kind=='names': groups.reverse()
            elif kind=='peak': groups[1]['max_lr']=1e-4
            else: payload['scheduler']['_last_lr'][1]=payload['scheduler']['_last_lr'][0]
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                entry.validate_resume_payload(payload,cfg)

    def test_parameter_ids_names_and_ownership_are_strict(self):
        cfg=as_spatial_config('joint',SEEDS[0])
        for kind in ('duplicate_id','duplicate_name','wrong_owner','unknown_state'):
            payload=resume_metadata(cfg)
            ue=payload['optimizer']['param_groups'][1]
            if kind=='duplicate_id': ue['params'][0]=0
            elif kind=='duplicate_name': ue['parameter_names'][1]=ue['parameter_names'][0]
            elif kind=='wrong_owner': ue['parameter_names'][0]='decoder.test'
            else: ue['parameter_names'][0]='encoder.nonexistent'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                entry.validate_resume_payload(payload,cfg)

    def test_scheduler_budget_step_and_rng_missing_are_rejected(self):
        cfg=as_spatial_config('joint',SEEDS[0])
        for kind in ('budget','step','rng'):
            payload=resume_metadata(cfg)
            if kind=='budget':payload['scheduler']['total_steps']=500
            elif kind=='step':payload['scheduler']['last_epoch']=0
            else:payload['rng_state']['cuda']=[]
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                entry.validate_resume_payload(payload,cfg)

    def test_selected_best_can_be_earlier_and_must_match_latest_metric(self):
        cfg=as_spatial_config('joint',SEEDS[0])
        latest,best=resume_metadata(cfg,1000),resume_metadata(cfg,500)
        entry.validate_best_metadata(best,latest,cfg)
        best['best_metric']+=.1
        with self.assertRaises(ValueError):entry.validate_best_metadata(best,latest,cfg)


class InterfaceTests(unittest.TestCase):
    def test_old_optimizer_scheduler_ast_is_unchanged_under_none(self):
        tree=ast.parse((ROOT/'main_train.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_proposal')
        defaults=dict(zip([arg.arg for arg in node.args.kwonlyargs],node.args.kw_defaults))
        self.assertIsNone(ast.literal_eval(defaults['optimizer_scheduler_factory']))
        branch=next(n for n in node.body if isinstance(n,ast.If)
                    and ast.unparse(n.test)=='optimizer_scheduler_factory is None')
        expected=ast.parse('''
optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.LR,
    total_steps=cfg.TOTAL_STEPS, pct_start=cfg.WARMUP_PCT, anneal_strategy="cos")
''').body
        self.assertEqual([ast.dump(n,include_attributes=False) for n in branch.body],
                         [ast.dump(n,include_attributes=False) for n in expected])
        model_line=next(n.lineno for n in node.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='model' for t in n.targets))
        self.assertLess(model_line,branch.lineno)
        self.assertEqual(ast.unparse(branch.orelse[0].value),'optimizer_scheduler_factory(cfg, model)')

    def test_entry_passes_optin_factory_data_and_no_init_checkpoint(self):
        tree=ast.parse((ROOT/'train_proposal_as_spatial.py').read_text(encoding='utf-8'))
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='train_proposal')
        self.assertEqual({kw.arg for kw in call.keywords},
                         {'model_factory','channel_data','optimizer_scheduler_factory'})
        self.assertIsNone(ast.literal_eval(next(kw.value for kw in call.args[1].keywords if kw.arg=='init_checkpoint')))

    def test_fresh_and_resume_model_factory_paths_are_separate(self):
        source=(ROOT/'train_proposal_as_spatial.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_model_factory')
        calls=[ast.unparse(n.func) for n in ast.walk(factory) if isinstance(n,ast.Call)]
        self.assertIn('model.initialize_ue_from_full',calls)
        self.assertIn('model.load_state_dict',calls)
        self.assertNotIn('strict=False',source)
        self.assertIn('if fresh:',ast.unparse(factory))

    def test_fresh_run_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'screen'
            cfg=as_spatial_config('joint',SEEDS[0],root)
            entry.check_run_state(cfg,resume=False,allowed_root=root)
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError):entry.check_run_state(cfg,resume=False,allowed_root=root)
            with self.assertRaises(FileNotFoundError):entry.check_run_state(cfg,resume=True,allowed_root=root)

    def test_all_dry_runs_have_no_runtime_imports_weights_or_writes(self):
        calls=[['--arm',arm,'--seed',str(seed),'--dry-run'] for seed in SEEDS for arm in ARMS]
        assert_lightweight_cli(self,'train_proposal_as_spatial',calls,
            "assert len(plans)==4; assert all(p['scheduler_total_steps']==20000 for p in plans)")


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,
                     'Optional real CPU optimizer construction requires installed PyTorch; no install/fake used.')
class TinyCpuOptimizerTests(unittest.TestCase):
    def test_ue_digest_includes_condition_and_running_power_not_bs(self):
        import torch
        state = {'encoder.test': torch.tensor([1., 2.]),
                 'cond_embed.test': torch.tensor(3.),
                 'running_pwr': torch.tensor([4.]),
                 'decoder.test': torch.tensor([5.])}
        digest = entry.ue_state_digest(torch, state)
        state['decoder.test'] = torch.tensor([99.])
        self.assertEqual(digest, entry.ue_state_digest(torch, state))
        for key in ('encoder.test', 'cond_embed.test', 'running_pwr'):
            changed = dict(state)
            changed[key] = state[key] + 1
            self.assertNotEqual(digest, entry.ue_state_digest(torch, changed))

    def test_actual_optimizer_has_distinct_groups_without_training(self):
        import torch
        class TinyParameterOwner:
            # Only three synthetic leaf tensors; no neural model or forward.
            def __init__(self,frozen):
                self.ue_frozen=frozen
                self.named=[('decoder.test',torch.tensor([1.],requires_grad=True)),
                            ('encoder.aircomp.test',torch.tensor([1.],requires_grad=not frozen)),
                            ('cond_embed.test',torch.tensor([1.],requires_grad=not frozen))]
            def named_parameters(self):return iter(self.named)
            def ue_parameters(self):return (p for n,p in self.named if n.startswith(entry.UE_PREFIXES))
            def bs_parameters(self):return (p for n,p in self.named if not n.startswith(entry.UE_PREFIXES))
        for arm in ARMS:
            cfg=as_spatial_config(arm,SEEDS[0])
            model=TinyParameterOwner(arm=='frozen_ue')
            optimizer,scheduler=entry.optimizer_scheduler_factory(cfg,model)
            peaks=[spec['peak_lr'] for spec in optimizer_group_spec(cfg)]
            self.assertEqual([group['max_lr'] for group in optimizer.param_groups],peaks)
            # OneCycle's cosine interpolation introduces its ordinary ~1e-21
            # rounding at step zero; use the SAME scalar LR validator as resume.
            for group,peak in zip(optimizer.param_groups,peaks):
                entry._close(group['lr'],peak/25,'initial group LR')
            self.assertEqual(scheduler.total_steps,20000)
            self.assertEqual(scheduler.last_epoch,0)
            self.assertEqual(optimizer.state,{})  # No backward/update occurred.


if __name__=='__main__':unittest.main()
