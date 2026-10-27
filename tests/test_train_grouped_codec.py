"""Metadata/CLI tests only; no network, channel generation or large models."""
import ast
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,Path(__file__).resolve().parent):
    if str(path) not in sys.path: sys.path.insert(0,str(path))
from configs.grouped_codec_train_config import ARMS,SEEDS,grouped_codec_config,validate_config,protocol
from configs.train_config import TRAINING_RECIPES
from cli_contracts import assert_lightweight_cli
import train_grouped_codec as entry


def resume_metadata(cfg,step=500):
    """JSON-schema sentinels, deliberately not fake model tensors or checkpoints."""
    lr,beta1=entry.expected_cycle(cfg,step)
    return dict(config=cfg.to_dict(),config_fingerprint=entry.fingerprint(cfg),
        layout_spec=cfg.layout_spec.to_dict(),execution_protocol=cfg.to_dict()['execution_protocol'],
        state={'metadata_fixture_only':'not numerical weights'},step=step,best_metric=12.,
        val_table={f'layout0_gamma{x}':12. for x in cfg.VAL_SNR_LIST},
        optimizer=dict(state={0:{'fixture':True}},param_groups=[dict(params=[0],lr=lr,
            initial_lr=cfg.LR/25,max_lr=cfg.LR,min_lr=cfg.LR/25/10000,weight_decay=cfg.WEIGHT_DECAY,
            eps=1e-8,base_momentum=.85,max_momentum=.95,betas=(beta1,.999),amsgrad=False,
            maximize=False,capturable=False,differentiable=False,foreach=None,fused=None)]),
        scheduler=dict(total_steps=20000,last_epoch=step,_step_count=step+1,
            _schedule_phases=[dict(end_step=1999.,start_lr='initial_lr',end_lr='max_lr',
                start_momentum='max_momentum',end_momentum='base_momentum'),
                dict(end_step=19999,start_lr='max_lr',end_lr='min_lr',
                start_momentum='base_momentum',end_momentum='max_momentum')],
            _anneal_func_type='cos',cycle_momentum=True,use_beta1=True,base_lrs=[cfg.LR/25],_last_lr=[lr]),
        rng_state=dict(python='schema-only',numpy='schema-only',torch='schema-only',cuda=['schema-only']))


class ConfigTests(unittest.TestCase):
    def test_five_layouts_fixed_random_recipe_and_same_channel_protocol(self):
        for seed in SEEDS:
            data=[];execution=[]
            for arm in ARMS:
                cfg=grouped_codec_config(arm,seed);validate_config(cfg)
                self.assertFalse(hasattr(cfg,'ALLOCATION_GRID'))
                self.assertEqual((cfg.TOTAL_STEPS,cfg.BATCH_SIZE,cfg.VAL_BATCH,cfg.VAL_INTERVAL),(20000,48,64,500))
                self.assertEqual((cfg.LR,cfg.WEIGHT_DECAY,cfg.WARMUP_PCT,cfg.GRAD_CLIP),(1e-4,1e-5,.1,.5))
                self.assertEqual(cfg.VAL_SNR_LIST,[0,5,10,15,20,25]);self.assertEqual(cfg.DOWNLINK_SNR_DB,25.)
                self.assertEqual([cfg.PH1_W_DIR,cfg.PH1_W_MSE,cfg.PH2_W_DIR,cfg.PH2_W_MSE],[0.,0.,0.,0.])
                self.assertEqual(cfg.layout_spec.occupied_resources,256)
                self.assertEqual(cfg.to_dict()['initialization']['source'],'random')
                self.assertIsNone(cfg.to_dict()['initialization']['historical_checkpoint'])
                data.append(cfg.channel_data_protocol);execution.append(cfg.to_dict()['execution_protocol'])
            self.assertTrue(all(x==data[0] for x in data))
            self.assertTrue(all(x==execution[0] for x in execution))
        self.assertEqual([grouped_codec_config(a,SEEDS[0]).layout_spec.active_per_user for a in ARMS],[256,256,128,128,64])

    def test_no_old_registry_mutation_and_architecture_declared(self):
        registry=deepcopy(TRAINING_RECIPES)
        for arm in ARMS:
            cfg=grouped_codec_config(arm,SEEDS[0])
            self.assertEqual(cfg.architecture_metadata,protocol.architecture_metadata(arm))
            self.assertEqual(cfg.to_dict()['architecture_metadata'],cfg.architecture_metadata)
        self.assertEqual(registry,TRAINING_RECIPES)
        with self.assertRaises(ValueError): grouped_codec_config('joint_swin256',SEEDS[0])
        with self.assertRaises(ValueError): grouped_codec_config(ARMS[0],42)

    def test_changed_recipe_or_data_rejected_and_fingerprint_differs(self):
        cfg=grouped_codec_config(ARMS[0],SEEDS[0]);baseline=entry.fingerprint(cfg)
        changed=deepcopy(cfg);changed.recipe=replace(changed.recipe,lr=2e-5)
        self.assertNotEqual(entry.fingerprint(changed),baseline)
        with self.assertRaises(ValueError):validate_config(changed)
        changed=deepcopy(cfg);changed.channel_data_protocol['base_seed']+=1
        with self.assertRaises(ValueError):validate_config(changed)

    def test_gamma_addresses_ignore_default_rng_and_layout(self):
        state=random.getstate()
        try:
            values=[protocol.sample_gamma(SEEDS[0],step) for step in (1,500,501,20000)]
            random.seed(9);[random.random() for _ in range(100)]
            self.assertEqual(values,[protocol.sample_gamma(SEEDS[0],step) for step in (1,500,501,20000)])
            self.assertTrue(all(0<=x<=25 for x in values));self.assertEqual(len(set(values)),4)
            address=protocol.RandomAddress(SEEDS[0],'train',501,'standard_noise')
            self.assertNotEqual(address.seed_value,protocol.RandomAddress(SEEDS[0],'validation',0,'standard_noise').seed_value)
            self.assertNotEqual(address.seed_value,protocol.RandomAddress(SEEDS[0],'test',501,'standard_noise').seed_value)
        finally:random.setstate(state)


class ResumeTests(unittest.TestCase):
    def test_500_pause_and_20k_completion_same_schedule(self):
        for arm in ARMS:
            cfg=grouped_codec_config(arm,SEEDS[0])
            entry.validate_resume_payload(resume_metadata(cfg),cfg)
            entry.validate_resume_payload(resume_metadata(cfg,20000),cfg,allow_completed=True)
            with self.assertRaises(ValueError):entry.validate_resume_payload(resume_metadata(cfg,20000),cfg)

    def test_cross_arm_seed_protocol_and_legacy_checkpoint_rejected(self):
        cfg=grouped_codec_config(ARMS[0],SEEDS[0]);payload=resume_metadata(cfg)
        for other in (grouped_codec_config(ARMS[1],SEEDS[0]),grouped_codec_config(ARMS[0],SEEDS[1])):
            with self.assertRaises(ValueError):entry.validate_resume_payload(payload,other)
        for key in ('layout_spec','execution_protocol','config_fingerprint','rng_state','optimizer','scheduler'):
            changed=deepcopy(payload);changed.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):entry.validate_resume_payload(changed,cfg)
        changed=deepcopy(payload);changed['allocation_grid']=[(0,256)]
        with self.assertRaises(ValueError):entry.validate_resume_payload(changed,cfg)

    def test_resume_rejects_schedule_restart_wrong_lr_and_short_budget(self):
        cfg=grouped_codec_config(ARMS[3],SEEDS[0])
        for key in ('last_epoch','total_steps','max_lr','gamma_table','best'):
            p=resume_metadata(cfg)
            if key=='last_epoch':p['scheduler'][key]=0
            elif key=='total_steps':p['scheduler'][key]=500
            elif key=='max_lr':p['optimizer']['param_groups'][0][key]=2e-5
            elif key=='gamma_table':p['val_table'].pop('layout0_gamma0')
            else:p['best_metric']=1.
            with self.subTest(key=key),self.assertRaises(ValueError):entry.validate_resume_payload(p,cfg)

    def test_positive_best_selection_requires_exact_six_point_mean(self):
        cfg=grouped_codec_config(ARMS[4],SEEDS[0])
        latest,best=resume_metadata(cfg,20000),resume_metadata(cfg,500)
        entry.validate_best_metadata(best,latest,cfg)
        best['step']=0
        with self.assertRaises(ValueError):entry.validate_best_metadata(best,latest,cfg)
        best=resume_metadata(cfg);best['val_table']['layout0_gamma0']=0
        with self.assertRaises(ValueError):entry.validate_best_metadata(best,latest,cfg)

    def test_no_output_overwrite_and_incomplete_resume_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new';cfg=grouped_codec_config(ARMS[0],SEEDS[0],root)
            entry.check_run_state(cfg,resume=False,allowed_root=root)
            self.assertFalse(root.exists())
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError):entry.check_run_state(cfg,resume=False,allowed_root=root)
            with self.assertRaises(FileNotFoundError):entry.check_run_state(cfg,resume=True,allowed_root=root)

    def test_interrupted_final_update_can_finalize_status_without_more_updates(self):
        cfg=grouped_codec_config(ARMS[3],SEEDS[0])
        latest,best=resume_metadata(cfg,20000),resume_metadata(cfg,19500)
        previous=dict(state='running',experiment=cfg.EXP_NAME)
        status=entry.completed_resume_status(cfg,latest,best,previous)
        self.assertEqual(status['state'],'completed');self.assertEqual(status['step'],20000)
        self.assertEqual(status['best_metric_name'],'mean_all')
        self.assertEqual(status['best_metric'],best['best_metric'])
        self.assertEqual(status['additional_optimizer_updates'],0)
        self.assertEqual((status['best_path'],status['latest_path']),(cfg.BEST_PATH,cfg.SAVE_PATH))
        for changed in (dict(previous,state='completed'),dict(previous,experiment='different'),
                        dict(previous,state='paused',step=500)):
            with self.assertRaises(ValueError):entry.completed_resume_status(cfg,latest,best,changed)
        with self.assertRaises(ValueError):entry.completed_resume_status(cfg,resume_metadata(cfg,19500),best,previous)
        tree=ast.parse((ROOT/'train_grouped_codec.py').read_text(encoding='utf-8'))
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='finished')
        calls=[ast.unparse(n.func) for n in ast.walk(branch) if isinstance(n,ast.Call)]
        self.assertNotIn('train_proposal',calls);self.assertNotIn('optimizer_scheduler_factory',calls)
        self.assertNotIn('make_model_factory',calls)
        self.assertTrue(any(isinstance(n,ast.Return) for n in branch.body))


class MainIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree=ast.parse((ROOT/'main_train.py').read_text(encoding='utf-8-sig'))

    def test_opt_in_binding_is_early_and_legacy_none_is_noop(self):
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_execution_protocol')
        scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'binding','exec'),scope)
        check=scope[node.name]
        check(SimpleNamespace(to_dict=lambda:{}),None)
        cfg=grouped_codec_config(ARMS[0],SEEDS[0]);obj=entry.GroupedExecutionProtocol(cfg)
        check(cfg,obj)
        with self.assertRaises(ValueError):check(cfg,None)
        with self.assertRaises(ValueError):check(SimpleNamespace(to_dict=lambda:{}),obj)
        with self.assertRaises(ValueError):check(grouped_codec_config(ARMS[0],SEEDS[1]),obj)
        with self.assertRaises(ValueError):check(grouped_codec_config(ARMS[1],SEEDS[0]),obj)
        train=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_proposal')
        self.assertEqual(ast.unparse(train.body[0].body[0]),'_validate_execution_protocol(cfg, execution_protocol)')
        self.assertIn("'execution_protocol' in cfg.to_dict()",ast.unparse(train.body[0].test))
        defaults=dict(zip([x.arg for x in train.args.kwonlyargs],train.args.kw_defaults))
        self.assertIsNone(ast.literal_eval(defaults['execution_protocol']))

    def test_original_numeric_helpers_unchanged_from_baseline(self):
        old=ast.parse(subprocess.check_output(['git','show','1dac61b:main_train.py'],cwd=ROOT,text=True,encoding='utf-8'))
        index=lambda t:{n.name:n for n in t.body if hasattr(n,'name')}
        before,after=index(old),index(self.tree)
        for name in ('get_snr','get_downlink_snr','SumRateLoss','alignment_losses','validate_proposal',
                     'validation_metric','format_val_table','load_latest_checkpoint','build_proposal_model'):
            self.assertEqual(ast.dump(before[name],include_attributes=False),ast.dump(after[name],include_attributes=False),name)

    def test_backend_only_after_setup_and_no_duplicated_training_loop(self):
        train=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_proposal')
        setup=next(n.lineno for n in ast.walk(train) if isinstance(n,ast.Call) and ast.unparse(n.func)=='setup_gpu')
        backend=next(n for n in ast.walk(train) if isinstance(n,ast.If) and ast.unparse(n.test)=='execution_protocol is not None'
            and any('execution_protocol.apply_backend' in ast.unparse(s) for s in n.body))
        self.assertGreater(backend.lineno,setup)
        source=(ROOT/'train_grouped_codec.py').read_text(encoding='utf-8')
        for forbidden in ('loss.backward(', 'optimizer.step(', 'scheduler.step(', 'strict=False','initialize_ue_from_full'):
            self.assertNotIn(forbidden,source)
        self.assertNotIn('--checkpoint',[x for x in entry.build_parser()._option_string_actions])

    def test_default_checkpoint_layout_metadata_does_not_relabel_legacy(self):
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint_payload')
        for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
        node.returns=None
        scope={'config_fingerprint':lambda cfg:'id','capture_rng_state':lambda:{'state':'sentinel'}}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'checkpoint','exec'),scope)
        model=SimpleNamespace(state_dict=lambda:{'not_model_execution':True})
        old=SimpleNamespace(to_dict=lambda:{},ALLOCATION_GRID=[(0,256)])
        common=dict(step=500,model=model,optimizer=None,scheduler=None,best_metric=1.,val_table={(0,0):1.})
        a=scope[node.name](cfg=old,**common)
        b=scope[node.name](cfg=grouped_codec_config(ARMS[2],SEEDS[0]),**common)
        self.assertEqual(a['allocation_grid'],[(0,256)]);self.assertEqual(a['val_table'],{'alloc0_snr0':1.})
        self.assertNotIn('execution_protocol',a);self.assertNotIn('allocation_grid',b)
        self.assertEqual(b['val_table'],{'layout0_gamma0':1.});self.assertEqual(b['layout_spec']['occupied_resources'],256)

    def test_ten_dry_runs_are_stdlib_readonly_no_weights(self):
        calls=[['--arm',a,'--seed',str(s),'--dry-run'] for s in SEEDS for a in ARMS]
        assert_lightweight_cli(self,'train_grouped_codec',calls,
            "assert len(plans)==10; assert all(p['scheduler_total_steps']==20000 and p['historical_checkpoint'] is None for p in plans)")


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Installed PyTorch unavailable; no dependencies installed/faked.')
class SmallTensorTests(unittest.TestCase):
    def test_shared_private_noise_and_external_rng_unchanged(self):
        import torch
        before=torch.get_rng_state().clone()
        a=protocol.standard_noise(SEEDS[0],'train',501,2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.rand(13)
        b=protocol.standard_noise(SEEDS[0],'train',501,2)
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(2,32,256))
        self.assertFalse(torch.equal(a,protocol.standard_noise(SEEDS[0],'validation',0,2)))

    def test_single_lr_optimizer_construction_no_updates_or_model(self):
        import torch
        owner=SimpleNamespace(parameters=lambda:iter([torch.tensor([1.],requires_grad=True)]))
        cfg=grouped_codec_config(ARMS[4],SEEDS[0]);optimizer,scheduler=entry.optimizer_scheduler_factory(cfg,owner)
        self.assertEqual(len(optimizer.param_groups),1)
        self.assertEqual(optimizer.param_groups[0]['max_lr'],1e-4)
        self.assertEqual(scheduler.total_steps,20000);self.assertEqual(scheduler.last_epoch,0)
        self.assertEqual(optimizer.state,{})


if __name__=='__main__':unittest.main()
