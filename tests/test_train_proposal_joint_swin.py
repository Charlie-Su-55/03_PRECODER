"""Joint+Swin256 stdlib/metadata contracts; no model/channel/training execution."""
import ast
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,Path(__file__).resolve().parent):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from configs.joint_swin_train_config import ARMS,SEEDS,joint_swin_config,optimizer_group_spec,validate_config
from configs.as_spatial_train_config import as_spatial_config
from configs.train_config import TRAINING_RECIPES
from tests.test_train_proposal_as_spatial import resume_metadata as _shared_metadata
from cli_contracts import assert_lightweight_cli
import train_proposal_joint_swin as entry


def resume_metadata(cfg,step=500):
    """Only schema sentinels: shared fixture retains the actual serialized arm."""
    return _shared_metadata(entry._JointGroupPolicy(cfg),step)


class ConfigurationTests(unittest.TestCase):
    def test_fixed_recipe_and_two_lr_groups(self):
        for arm in ARMS:
            cfg=joint_swin_config(arm,SEEDS[0]);validate_config(cfg)
            self.assertEqual((cfg.TOTAL_STEPS,cfg.BATCH_SIZE,cfg.VAL_BATCH,cfg.VAL_INTERVAL),(20000,48,64,500))
            self.assertEqual((cfg.DOWNLINK_SNR_DB,cfg.WARMUP_PCT,cfg.WEIGHT_DECAY,cfg.GRAD_CLIP),(25.,.1,1e-5,.5))
            self.assertEqual(list(cfg.VAL_SNR_LIST),[0,5,10,15,20,25])
            self.assertEqual(cfg.STAGE1_SNR,cfg.STAGE2_SNR)
            self.assertEqual(cfg.STAGE3_SNR,(0.,25.))
            self.assertEqual([g['peak_lr'] for g in optimizer_group_spec(cfg)],[1e-4,2e-5])
            self.assertEqual([cfg.PH1_W_DIR,cfg.PH1_W_MSE,cfg.PH2_W_DIR,cfg.PH2_W_MSE],[0.,0.,0.,0.])

    def test_shared_source_and_data_with_existing_control_per_seed(self):
        for seed in SEEDS:
            plain,joint=[joint_swin_config(a,seed) for a in ARMS]
            old=as_spatial_config('joint',seed)
            self.assertEqual(plain.source_identity,joint.source_identity)
            self.assertEqual(plain.source_identity,old.source_identity)
            self.assertEqual(plain.channel_data_protocol,joint.channel_data_protocol)
            self.assertEqual(plain.channel_data_protocol,old.channel_data_protocol)
            self.assertNotEqual(plain.SAVE_DIR,joint.SAVE_DIR)

    def test_architecture_and_memory_policy_are_in_config_fingerprint(self):
        cfg=joint_swin_config(ARMS[1],SEEDS[0])
        self.assertEqual(cfg.JOINT_SWIN_QUERY_CHUNK,64)
        self.assertIs(cfg.JOINT_SWIN_ACTIVATION_CHECKPOINT,True)
        self.assertEqual(cfg.JOINT_SWIN_SOLVE_DTYPE,'complex64')
        self.assertEqual(cfg.architecture_metadata['architecture'],'joint_swin256_v1')
        self.assertFalse(cfg.architecture_metadata['ue_frozen'])
        self.assertEqual(cfg.to_dict()['architecture_metadata'],cfg.architecture_metadata)
        changed=deepcopy(cfg);changed.channel_data_protocol['base_seed']+=1
        self.assertNotEqual(entry.fingerprint(changed),entry.fingerprint(cfg))
        with self.assertRaises(ValueError):validate_config(changed)

    def test_no_old_recipe_registry_mutation_or_unknown_arm_seed(self):
        before=deepcopy(TRAINING_RECIPES)
        for arm in ARMS:
            joint_swin_config(arm,SEEDS[0])
        self.assertEqual(before,TRAINING_RECIPES)
        for arm,seed in [('joint',SEEDS[0]),(ARMS[0],42)]:
            with self.assertRaises(ValueError):joint_swin_config(arm,seed)

    def test_declared_settings_match_actual_model_source_without_import(self):
        tree=ast.parse((ROOT/'models/adaptive_hybrid_joint_swin.py').read_text(encoding='utf-8'))
        settings=next(n.value for n in tree.body if isinstance(n,ast.Assign) and
                      any(isinstance(t,ast.Name) and t.id=='SETTINGS' for t in n.targets))
        cfg=joint_swin_config(ARMS[0],SEEDS[0])
        self.assertEqual(ast.literal_eval(settings),cfg.architecture_metadata['settings'])


class CompositionTests(unittest.TestCase):
    def test_group_view_never_rewrites_checkpoint_identity_or_real_arm(self):
        fingerprints=[]
        for arm in ARMS:
            cfg=joint_swin_config(arm,SEEDS[0]);view=entry._JointGroupPolicy(cfg)
            self.assertEqual(view.arm,'joint')
            self.assertEqual(cfg.arm,arm)
            self.assertEqual(view.to_dict(),cfg.to_dict())
            self.assertEqual(view.to_dict()['arm'],arm)
            payload=resume_metadata(cfg)
            self.assertEqual(payload['config']['arm'],arm)
            self.assertEqual(payload['config_fingerprint'],entry.fingerprint(cfg))
            fingerprints.append(entry.fingerprint(cfg))
            with self.assertRaises(AttributeError):view.arm='frozen_ue'
        self.assertNotEqual(*fingerprints)

    def test_resume_500_and_completed20k(self):
        for arm in ARMS:
            cfg=joint_swin_config(arm,SEEDS[0])
            entry.validate_resume_payload(resume_metadata(cfg),cfg)
            entry.validate_resume_payload(resume_metadata(cfg,20000),cfg,allow_completed=True)
            with self.assertRaises(ValueError):entry.validate_resume_payload(resume_metadata(cfg,20000),cfg)

    def test_cross_arm_seed_and_wrong_source_rejected(self):
        cfg=joint_swin_config(ARMS[0],SEEDS[0]);payload=resume_metadata(cfg)
        for other in (joint_swin_config(ARMS[1],SEEDS[0]),joint_swin_config(ARMS[0],SEEDS[1])):
            with self.assertRaises(ValueError):entry.validate_resume_payload(payload,other)
        changed=deepcopy(payload);changed['config']['source_identity']['checkpoint_sha256']='0'*64
        with self.assertRaises(ValueError):entry.validate_resume_payload(changed,cfg)

    def test_incomplete_resume_or_lr_scheduler_restart_rejected(self):
        cfg=joint_swin_config(ARMS[1],SEEDS[0])
        for kind in ('fingerprint','optimizer','scheduler','rng_state','groups','restart','lr'):
            payload=resume_metadata(cfg)
            if kind=='fingerprint':payload.pop('config_fingerprint')
            elif kind in ('optimizer','scheduler','rng_state'):payload.pop(kind)
            elif kind=='groups':payload['optimizer']['param_groups'].pop()
            elif kind=='restart':payload['scheduler']['last_epoch']=0
            else:payload['optimizer']['param_groups'][1]['max_lr']=1e-4
            with self.subTest(kind=kind),self.assertRaises(ValueError):entry.validate_resume_payload(payload,cfg)

    def test_best_may_be_earlier_but_not_source_step_zero(self):
        cfg=joint_swin_config(ARMS[1],SEEDS[0])
        latest,best=resume_metadata(cfg,20000),resume_metadata(cfg,500)
        entry.validate_best_metadata(best,latest,cfg)
        best['step']=0
        with self.assertRaises(ValueError):entry.validate_best_metadata(best,latest,cfg)

    def test_declared_architecture_checked_independently_of_group_policy(self):
        cfg=joint_swin_config(ARMS[1],SEEDS[0])
        metadata=deepcopy(cfg.architecture_metadata)
        holder=SimpleNamespace(architecture_metadata=metadata,ue_frozen=False)
        entry.validate_model_identity(cfg,holder)
        metadata['arm']=ARMS[0]
        with self.assertRaises(ValueError):entry.validate_model_identity(cfg,holder)


class EntryTests(unittest.TestCase):
    def test_real_config_not_policy_view_passed_to_model_and_trainer(self):
        tree=ast.parse((ROOT/'train_proposal_joint_swin.py').read_text(encoding='utf-8'))
        model=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='AdaptiveHybridPrecoderJointSwin')
        self.assertEqual(ast.unparse(model.args[0]),'cfg')
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='train_proposal')
        self.assertEqual(ast.unparse(call.args[0]),'cfg')
        self.assertEqual({k.arg for k in call.keywords},{'model_factory','channel_data','optimizer_scheduler_factory'})
        self.assertIsNone(ast.literal_eval(next(k.value for k in call.args[1].keywords if k.arg=='init_checkpoint')))

    def test_no_copied_training_loop_or_global_monkeypatch(self):
        source=(ROOT/'train_proposal_joint_swin.py').read_text(encoding='utf-8')
        self.assertNotIn('loss.backward(',source)
        self.assertNotIn('optimizer.step(',source)
        self.assertNotIn('scheduler.step(',source)
        self.assertNotIn('strict=False',source)
        tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and
            ast.unparse(t).startswith('_group_tools.') for t in n.targets) for n in ast.walk(tree)))

    def test_fresh_and_resume_initialization_are_separate(self):
        tree=ast.parse((ROOT/'train_proposal_joint_swin.py').read_text(encoding='utf-8'))
        factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_model_factory')
        branch=next(n for n in ast.walk(factory) if isinstance(n,ast.If) and ast.unparse(n.test)=='fresh' and n.orelse)
        self.assertIn('model.initialize_ue_from_full(source_state)',ast.unparse(branch.body[0]))
        self.assertIn('model.load_state_dict(resume_state, strict=True)',ast.unparse(branch.orelse[0]))

    def test_output_protection_and_four_standard_library_dry_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'experiment';cfg=joint_swin_config(ARMS[0],SEEDS[0],root)
            entry.check_run_state(cfg,resume=False,allowed_root=root)
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError):entry.check_run_state(cfg,resume=False,allowed_root=root)
            with self.assertRaises(FileNotFoundError):entry.check_run_state(cfg,resume=True,allowed_root=root)
        calls=[['--arm',a,'--seed',str(s),'--dry-run'] for s in SEEDS for a in ARMS]
        assert_lightweight_cli(self,'train_proposal_joint_swin',calls,
            "assert len(plans)==4; assert all(p['scheduler_total_steps']==20000 for p in plans)")


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Optional real CPU optimizer construction requires installed PyTorch; no dependency installed/faked.')
class TinyOptimizerTests(unittest.TestCase):
    def test_same_two_lr_groups_for_both_arms_without_model_or_updates(self):
        import torch
        class Owner:
            ue_frozen=False
            def __init__(self):
                self.named=[('decoder.test',torch.tensor([1.],requires_grad=True)),
                            ('encoder.aircomp.test',torch.tensor([1.],requires_grad=True)),
                            ('cond_embed.test',torch.tensor([1.],requires_grad=True))]
            def named_parameters(self):return iter(self.named)
            def ue_parameters(self):return (p for n,p in self.named if n.startswith(('encoder.','cond_embed.')))
            def bs_parameters(self):return (p for n,p in self.named if n.startswith('decoder.'))
        for arm in ARMS:
            cfg=joint_swin_config(arm,SEEDS[0])
            optimizer,scheduler=entry.optimizer_scheduler_factory(cfg,Owner())
            self.assertEqual([g['max_lr'] for g in optimizer.param_groups],[1e-4,2e-5])
            self.assertEqual([g['group_name'] for g in optimizer.param_groups],['bs','ue'])
            self.assertEqual(scheduler.total_steps,20000)
            self.assertEqual(scheduler.last_epoch,0);self.assertEqual(optimizer.state,{})


if __name__=='__main__':unittest.main()
