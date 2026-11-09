"""Read-only finish acceptance: JSON schemas and tiny real tensors, no codec."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import physical_split_finish_validation as validation
from configs import twc_v1_k8_physical_split_config as cfg

HAS_TENSORS=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None
if HAS_TENSORS:
    import torch
    import numpy as np


def fixture(arm='g4_split'):
    """Synthetic metadata, deliberately not a serialized checkpoint or result."""
    code=dict(commit=validation.TRAINING_COMMIT,sha256={name:'b'*64 for name in validation.TRAINING_SOURCE_FILES})
    accepted=dict(format=cfg.SOURCE_ACCEPTANCE_FORMAT,status='passed',sources={},code=code,
        ce_identity={'schema_only':True},data_protocol={'schema_only':True})
    for method in cfg.SOURCE_METHODS:
        identity=cfg.planned_source_identity(method);identity['sha256']='a'*64
        accepted['sources'][method]=dict(identity=identity,verification=dict(passed=True,ue_sha256='e'*64),
            frozen_files={identity['path']:'a'*64})
    accepted['record_id']=cfg.json_sha256(accepted)
    counts=cfg.update_counts(6048)
    names=dict(head=[validation.training.HEAD_PREFIX+'0.weight'],bs=['joint_decoder.input_projection.weight'])
    monitors={str(s):dict(path='schema_only_monitor_'+str(s),sha256='c'*64) for s in cfg.MONITOR_STEPS}
    payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=6048,phase=cfg.phase(6048),update_counts=counts,
        source_codec_updates=34192,source_bs_updates=6048,source_data_address=40240,data_global_step=46288,ue_frozen=True,
        source=cfg.source_identity(arm,accepted),source_acceptance=cfg.acceptance_identity(accepted),
        recipe=cfg.recipe(arm,accepted),config_fingerprint=cfg.fingerprint(arm,accepted),codec_model={'schema_only':True},
        head_optimizer={'schema_only':True},bs_optimizer={'schema_only':True},head_scheduler={'last_epoch':6048},
        bs_scheduler={'last_epoch':5940},rng={k:'schema only' for k in ('python','numpy','torch_cpu','torch_cuda')},
        ce_identity=accepted['ce_identity'],data_protocol=accepted['data_protocol'],code_sha256=code['sha256'],
        source_acceptance_sha256='f'*64,optimizer_parameter_names=names,
        actual_condition_counts=validation.training.expected_counts(arm,6048),frozen_state=dict(ue='e'*64,ce='c'*64),
        training_input_history='1'*64,layout_transmit_history='2'*64,monitor_records=monitors,
        resume_records={'108':dict(path='schema_only_resume',sha256='c'*64)})
    manifest={key:deepcopy(value) for key,value in payload.items() if key not in
        ('codec_model','head_optimizer','bs_optimizer','head_scheduler','bs_scheduler','rng','code_sha256','source_acceptance_sha256')}
    manifest.update(status='completed',code=code,checkpoint_sha256='d'*64)
    initial={key:deepcopy(manifest[key]) for key in ('code','source','source_acceptance','recipe','ce_identity','optimizer_parameter_names','frozen_state')}
    initial.update(new_two_optimizers=True,migration=dict(strict=True,source_values_preserved=True))
    terminal=dict(path=str(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt'),sha256='d'*64,
        update_counts=counts,data_global_step=46288)
    events=[dict(step=s,phase=cfg.phase(s),update_counts=cfg.update_counts(s),data_global_step=40240+s,
        condition=cfg.schedule(arm,s),input_fingerprints={key:'a'*64 for key in ('channel','pilot','data','X','Y','Hest')})
        for s in range(108,6049,108)]
    return payload,manifest,initial,terminal,events,arm,accepted,code,'d'*64


def batch_fixture():
    args=fixture();code=args[-2]
    stages={name:dict(status='completed',evidence_sha256={'schema_only_evidence':'a'*64})
        for name in ('sources','cuda_preflight',*('train_'+arm for arm in cfg.ARMS))}
    stages['training_audit']=dict(status='failed',error="ValueError: dict contains fields not in fieldnames: 'hybrid_1_3'",traceback='recorded schema failure')
    return dict(status='failed',error=stages['training_audit']['error'],code=code,stages=stages,
        training={arm:dict(step=6048) for arm in cfg.ARMS}),code


class MetadataTests(unittest.TestCase):
    def test_terminal_actual_schema_all_arms_accepts_json_roundtrip(self):
        for arm in cfg.ARMS:
            args=json.loads(json.dumps(fixture(arm)))
            accepted=validation.verify_training_metadata(*args,source_acceptance_sha256='f'*64)
            self.assertEqual(accepted['step'],6048)
            self.assertEqual(accepted['update_counts']['bs_updates'],5940)
            self.assertEqual(accepted['checkpoint_path'],args[3]['path'])
        self.assertNotEqual(validation.TRAINING_COMMIT,'b6336de4c80588e8d1ddb57a0607f774182a0f67')

    def test_unfinished_source_changed_weight_or_training_identity_is_rejected(self):
        for label,index,mutate in (
            ('unfinished',1,lambda p:p.update(status='running')),
            ('new_HEAD_is_not_old_training',7,lambda p:p.update(commit='f'*40)),
            ('checkpoint_SHA',3,lambda p:p.update(sha256='0'*64)),
            ('source_configuration',0,lambda p:p['source'].update(seed=0)),
            ('BS_counter',0,lambda p:p['update_counts'].update(bs_updates=6048)),
            ('UE_was_trained',0,lambda p:p['update_counts'].update(ue_updates=1)),
            ('missing_progress',4,lambda p:p.pop()),
            ('changed_data_address',4,lambda p:p[-1].update(data_global_step=999)),
            ('actual_input_missing',4,lambda p:p[-1]['input_fingerprints'].pop('Y')),
            ('optimizer_cross_group',0,lambda p:p['optimizer_parameter_names']['bs'].append(p['optimizer_parameter_names']['head'][0])),
            ('initialization_changed',2,lambda p:p.update(new_two_optimizers=False)),
            ('monitor_missing',0,lambda p:p['monitor_records'].pop('6048'))):
            args=deepcopy(fixture());mutate(args[index])
            with self.subTest(issue=label),self.assertRaises(ValueError):validation.verify_training_metadata(*args)
        with self.assertRaises(ValueError):validation.verify_training_metadata(*fixture(),source_acceptance_sha256='0'*64)

    def test_only_the_failed_post_training_audit_is_accepted(self):
        batch,code=batch_fixture();failure=validation.validate_source_batch(batch,code)
        self.assertEqual(failure['training_audit']['traceback'],'recorded schema failure')
        for label,mutate in (
            ('earlier_failure',lambda b:b['stages']['cuda_preflight'].update(status='failed')),
            ('missing_training_stage',lambda b:b['stages'].pop('train_fdma_control')),
            ('missing_training_arm',lambda b:b['training'].pop('fdma_split')),
            ('audit_not_failed',lambda b:b['stages']['training_audit'].update(status='completed')),
            ('unrelated_audit_failure',lambda b:b['stages']['training_audit'].update(error='Nonfinite validation rate')),
            ('alpha_started',lambda b:b['stages'].update(terminal_alpha_selection=dict(status='running'))),
            ('alpha_results',lambda b:b.update(alpha_selection={'record_id':'a'*64})),
            ('completed_batch',lambda b:b.update(status='completed'))):
            bad=deepcopy(batch);mutate(bad)
            with self.subTest(issue=label),self.assertRaises(ValueError):validation.validate_source_batch(bad,code)
        bad=deepcopy(code);bad['sha256'].pop(validation.TRAINING_SOURCE_FILES[-1])
        with self.assertRaises(ValueError):validation.validate_source_batch(batch,bad)

    def test_old_sources_including_status_are_raw_byte_checked(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory);status=root/'docs/TWC_V1_STATUS.md';status.parent.mkdir();status.write_bytes(b'status\n')
            implementation=root/'implementation.py';implementation.write_bytes(b'old source\n')
            code=dict(commit=validation.TRAINING_COMMIT,sha256={str(path.relative_to(root)):validation.sha256(path)
                for path in (implementation,status)})
            with patch.object(validation,'ROOT',root),patch.object(validation,'training_code',return_value=code),patch.object(
                validation,'TRAINING_SOURCE_FILES',tuple(code['sha256'])):
                self.assertEqual(validation.verify_source_files(),code)
                status.write_bytes(b'status\r\n')
                with self.assertRaisesRegex(ValueError,'TWC_V1_STATUS'):validation.verify_source_files()
            self.assertEqual(implementation.read_bytes(),b'old source\n')

    def test_evidence_and_exclusive_finish_outputs_never_modify_original_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory);original=root/'historical.json';original.write_bytes(b'{"original": true}\n')
            before=original.read_bytes();files={str(original):validation.sha256(original)}
            validation.verify_files(files)
            with patch.object(validation,'FINISH_ROOT',root/'finish'):
                target=root/'finish/new.json';validation.write_once_json(target,{'passed':True})
                with self.assertRaises(FileExistsError):validation.write_once_json(target,{'passed':False})
                with self.assertRaises(ValueError):validation.write_once_json(original,{'changed':True})
            self.assertEqual(original.read_bytes(),before)
            original.write_bytes(before+b' ')
            with self.assertRaises(ValueError):validation.verify_files(files)

    def test_acceptance_lock_keeps_old_training_and_new_postprocessing_distinct(self):
        args=fixture();code=args[-2];old=args[-3];post=dict(commit='f'*40,sha256={'new_finish.py':'c'*64})
        record=dict(format=validation.ACCEPTANCE_FORMAT,status='passed',training_code=code,postprocessing_code=post,
            source_batch=str(validation.BATCH_PATH),training={arm:{} for arm in cfg.ARMS},additional_training_updates=0,
            frozen_files={'schema_evidence':'a'*64},old_source_acceptance=old)
        record['record_id']=validation.json_sha256(record)
        with patch.object(validation,'read_json',return_value=record),patch.object(validation,'training_code',return_value=code),patch.object(
            validation,'postprocessing_code_record',return_value=post),patch.object(validation,'verify_training_source_files') as source_check,patch.object(
            validation,'verify_files') as evidence_check:
            self.assertEqual(validation.load_acceptance()['training_code']['commit'],validation.TRAINING_COMMIT)
            source_check.assert_called_once_with(code);evidence_check.assert_called_once_with(record['frozen_files'])
            record['additional_training_updates']=1
            with self.assertRaises(ValueError):validation.load_acceptance()
            record['additional_training_updates']=0;post['commit']='e'*40
            with self.assertRaises(ValueError):validation.load_acceptance()


@unittest.skipUnless(HAS_TENSORS,'Real PyTorch/NumPy unavailable; no dependency installation or substitutes')
class TinyNumericalTests(unittest.TestCase):
    def optimizer_fixture(self):
        parameter=torch.nn.Parameter(torch.ones(2,dtype=torch.float32));updates=5940
        payload=dict(optimizer_parameter_names={'bs':['tiny.weight']},bs_optimizer=dict(
            param_groups=[dict(params=[0],lr=2e-4,initial_lr=2e-4,weight_decay=1e-5,eps=1e-8,betas=(.9,.999))],
            state={0:dict(step=torch.tensor(float(updates)),exp_avg=torch.zeros(2),exp_avg_sq=torch.ones(2))}),
            bs_scheduler=dict(last_epoch=updates,_step_count=updates+1,base_lrs=[2e-4],_last_lr=[2e-4]))
        return payload,[('tiny.weight',parameter)],updates

    def test_dual_optimizer_numerics_are_checked_without_constructing_optimizer(self):
        payload,parameters,updates=self.optimizer_fixture();validation._validate_optimizer(torch,payload,'bs',parameters,updates)
        for label,mutate in (
            ('schedule',lambda p:p['bs_scheduler'].update(last_epoch=6048)),
            ('moment_step',lambda p:p['bs_optimizer']['state'][0]['step'].fill_(6048)),
            ('moment_dtype',lambda p:p['bs_optimizer']['state'][0].update(exp_avg=torch.zeros(2,dtype=torch.float64))),
            ('nonfinite',lambda p:p['bs_optimizer']['state'][0]['exp_avg'].fill_(float('nan'))),
            ('negative_second_moment',lambda p:p['bs_optimizer']['state'][0]['exp_avg_sq'].fill_(-1)),
            ('unknown_parameter',lambda p:p['bs_optimizer']['param_groups'][0].update(params=[1]))):
            bad=deepcopy(payload);mutate(bad)
            with self.subTest(issue=label),self.assertRaises(ValueError):validation._validate_optimizer(torch,bad,'bs',parameters,updates)

    def test_rng_structural_evidence_does_not_restore_or_bind_current_cuda_count(self):
        before=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
        record=dict(python=before[0],numpy=before[1],torch_cpu=before[2],torch_cuda=[torch.arange(32,dtype=torch.uint8) for _ in range(2)])
        with patch.object(torch.cuda,'device_count',side_effect=AssertionError('Current GPU visibility is not training identity')):
            self.assertEqual(validation._validate_rng(torch,record),2)
        self.assertEqual(random.getstate(),before[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
        bad=deepcopy(record);bad['torch_cuda'][0]=torch.zeros(2,dtype=torch.float32)
        with self.assertRaises(ValueError):validation._validate_rng(torch,bad)
        bad=deepcopy(record);bad['torch_cuda'][0]=torch.empty(0,dtype=torch.uint8)
        with self.assertRaises(ValueError):validation._validate_rng(torch,bad)
        bad=deepcopy(record);bad['torch_cpu']=torch.zeros(7,dtype=torch.uint8)
        with self.assertRaises(RuntimeError):validation._validate_rng(torch,bad)

    def test_strict_loading_preserves_nonzero_trained_heads_and_shared_decoder_alias(self):
        class Codec(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(2,2);self.decoder=torch.nn.Linear(2,2)
                self.joint_decoder=torch.nn.Module();self.joint_decoder.base_decoder=self.decoder
                self.joint_decoder.physical_head=torch.nn.Linear(2,2)
        class Chain(torch.nn.Module):
            def __init__(self):super().__init__();self.codec_model=Codec();self.ce_model=torch.nn.Linear(2,2)
        source=Chain();source.codec_model.joint_decoder.physical_head.weight.data.fill_(.75)
        payload=dict(codec_model=deepcopy(source.codec_model.state_dict()))
        def factory(torch_arg,device,method,value,*,include_stages,initialize):
            self.assertFalse(initialize);self.assertTrue(include_stages)
            chain=Chain();chain.codec_model.load_state_dict(value['codec_model'],strict=True);return chain
        with patch.object(validation.training,'build_chain',side_effect=factory):
            restored=validation.build_chain(torch,torch.device('cpu'),'g4_split',payload)
            self.assertTrue(torch.equal(restored.codec_model.joint_decoder.physical_head.weight,
                source.codec_model.joint_decoder.physical_head.weight))
            self.assertIs(restored.codec_model.decoder,restored.codec_model.joint_decoder.base_decoder)
            self.assertTrue(all(not module.training for module in restored.modules()))
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            bad=deepcopy(payload);bad['codec_model'].pop('joint_decoder.physical_head.weight')
            with self.assertRaises(RuntimeError):validation.build_chain(torch,torch.device('cpu'),'g4_split',bad)
            bad=deepcopy(payload);bad['codec_model']['joint_decoder.base_decoder.weight']=bad['codec_model']['joint_decoder.base_decoder.weight'].clone()+1.
            with self.assertRaisesRegex(ValueError,'aliases'):validation.build_chain(torch,torch.device('cpu'),'g4_split',bad)


if __name__=='__main__':unittest.main()
