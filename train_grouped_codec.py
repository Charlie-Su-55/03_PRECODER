#!/usr/bin/env python3
"""Random-initialized grouped-codec training using the existing task trainer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback
import uuid

from configs.grouped_codec_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, REPO_ROOT, grouped_codec_config, validate_config,
    channel_data_api, make_channel_data, protocol, execution_descriptor)
from train_proposal_as_receiver import fingerprint, _json, run_lock
from train_proposal_as_receiver import validate_output_paths as _output_paths
from train_proposal_as_direct import _safe_path, _close, expected_cycle
from preflight_proposal_as_direct import sha256_file, _backend_info

SOURCE_FILES = (
    'train_grouped_codec.py','configs/grouped_codec_train_config.py',
    'models/grouped_codec_precoder.py','utils/grouped_feedback_protocol.py',
    'models/adaptive_hybrid.py','models/adaptive_hybrid_joint_swin.py',
    'models/swin_precoder.py','models/baseline_common.py',
    'main_train.py','configs/train_config.py','configs/as_receiver_train_config.py',
    'utils/replayable_channel_data.py','utils/sionna_rng.py','utils/channel_generator.py',
    'utils/layers.py','train_proposal_as_receiver.py','train_proposal_as_direct.py',
    'preflight_proposal_as_direct.py')


def apply_backend(torch):
    """Explicit new-protocol FP32 policy; never called by the legacy None path."""
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.get_float32_matmul_precision() != 'highest':
        raise RuntimeError('New protocol requires highest/FP32 matrix multiplication')


class GroupedExecutionProtocol:
    def __init__(self,cfg):
        validate_config(cfg)
        self.cfg = cfg
        self.config_snapshot = cfg.to_dict()

    @property
    def descriptor(self): return execution_descriptor(self.cfg.SEED)

    @staticmethod
    def apply_backend(torch): apply_backend(torch)

    def training(self,model,H_dl,H_ul,global_step,device):
        if type(global_step) is not int or not 1 <= global_step <= self.cfg.TOTAL_STEPS:
            raise ValueError('Training random address must be an absolute update within the fixed budget')
        gamma = protocol.sample_gamma(self.cfg.SEED,global_step)
        noise = protocol.standard_noise(self.cfg.SEED,'train',global_step,H_dl.shape[0],device=device)
        output = model.forward_with_representation(H_dl,H_ul,gamma,standard_noise=noise)
        return output,gamma

    def validate(self,model,H_dl,H_ul,criterion,cfg,device):
        import torch
        if cfg.to_dict()!=self.cfg.to_dict(): raise ValueError('Validation config changed')
        model.eval()
        noise = protocol.standard_noise(cfg.SEED,'validation',0,H_dl.shape[0],device=device)
        table = {}
        with torch.no_grad():
            for gamma in cfg.VAL_SNR_LIST:
                W,H_hat = model.forward_with_representation(H_dl,H_ul,float(gamma),standard_noise=noise)
                if not torch.isfinite(W).all() or not torch.isfinite(H_hat).all():
                    raise FloatingPointError(f'Nonfinite validation output at gamma={gamma}')
                _,rate = criterion(W,H_dl,10.**(-cfg.DOWNLINK_SNR_DB/10.))
                if not math.isfinite(rate): raise FloatingPointError('Nonfinite validation rate')
                table[(0,gamma)] = rate
        return table

    def format_validation(self,table,step,cfg):
        expected={(0,gamma) for gamma in cfg.VAL_SNR_LIST}
        if set(table)!=expected or any(not math.isfinite(value) for value in table.values()):
            raise ValueError('Incomplete/nonfinite layout validation table')
        metric=sum(table.values())/len(table)
        text=f'Step {step} | layout={cfg.layout_spec.layout}; gamma transmit reference; DL25\n'
        text+=' | '.join(f'{gamma}dB: {table[(0,gamma)]:.3f}' for gamma in cfg.VAL_SNR_LIST)
        return text+f' | mean_all: {metric:.6f}',metric


def validate_output_paths(cfg,*,allowed_root=OUTPUT_ROOT):
    return _output_paths(cfg,allowed_root=allowed_root)


def check_run_state(cfg,*,resume,allowed_root=OUTPUT_ROOT):
    validate_config(cfg)
    validate_output_paths(cfg,allowed_root=allowed_root)
    if not resume:
        for path in (cfg.SAVE_DIR,cfg.LOG_DIR):
            if Path(path).exists(): raise FileExistsError(f'Refusing to overwrite/reinitialize {path}')
        return
    for key in ('SAVE_PATH','BEST_PATH','CONFIG_PATH','STATUS_PATH','METRICS_PATH'):
        if not Path(getattr(cfg,key)).is_file(): raise FileNotFoundError(f'Resume requires {key}')
    snapshot=json.loads(Path(cfg.CONFIG_PATH).read_text(encoding='utf-8'))
    if _json(snapshot)!=_json(dict(cfg.to_dict(),config_fingerprint=fingerprint(cfg))):
        raise ValueError('Resume full config/layout/protocol/fingerprint mismatch')
    status=json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
    if status.get('experiment')!=cfg.EXP_NAME or status.get('state') not in ('paused','running'):
        raise ValueError('Resume requires this unfinished arm and seed')
    if status.get('fingerprint',fingerprint(cfg))!=fingerprint(cfg):
        raise ValueError('Resume status fingerprint mismatch')


def _validate_identity(payload,cfg):
    if not isinstance(payload,dict) or payload.get('config_fingerprint')!=fingerprint(cfg):
        raise ValueError('Missing or incorrect grouped-codec fingerprint')
    if _json(payload.get('config'))!=_json(cfg.to_dict()):
        raise ValueError('Complete grouped-codec configuration mismatch')
    for key in ('layout_spec','execution_protocol'):
        if _json(payload.get(key))!=_json(cfg.to_dict()[key]):
            raise ValueError(f'Checkpoint {key} mismatch')
    if 'allocation_grid' in payload or not isinstance(payload.get('state'),dict) or not payload['state']:
        raise ValueError('Independent grouped layout requires its own complete state, not old allocations')


def _validation_values(payload,cfg):
    table=payload.get('val_table')
    keys={f'layout0_gamma{gamma}' for gamma in cfg.VAL_SNR_LIST}
    if not isinstance(table,dict) or set(table)!=keys:
        raise ValueError('Checkpoint must retain all six gamma validation rates')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in table.values()):
        raise ValueError('Nonfinite/invalid retained validation rate')
    return sum(table.values())/len(table)


def validate_resume_payload(payload,cfg,*,allow_completed=False):
    validate_config(cfg)
    channel_data_api().validate_resume_config(payload,cfg)
    _validate_identity(payload,cfg)
    step=payload['step']; upper=cfg.TOTAL_STEPS if allow_completed else cfg.TOTAL_STEPS-1
    if type(step) is not int or not 0<step<=upper or step%cfg.VAL_INTERVAL:
        raise ValueError('Latest must be a positive 500-update boundary within the fixed 20k plan')
    current=_validation_values(payload,cfg)
    if current>payload['best_metric']+1e-10: raise ValueError('Latest validation exceeds retained best metric')
    scheduler=payload['scheduler']
    phases=[dict(end_step=cfg.WARMUP_PCT*cfg.TOTAL_STEPS-1,start_lr='initial_lr',end_lr='max_lr',
                 start_momentum='max_momentum',end_momentum='base_momentum'),
            dict(end_step=cfg.TOTAL_STEPS-1,start_lr='max_lr',end_lr='min_lr',
                 start_momentum='base_momentum',end_momentum='max_momentum')]
    expected={'total_steps':cfg.TOTAL_STEPS,'last_epoch':step,'_step_count':step+1,
              '_schedule_phases':phases,'_anneal_func_type':'cos','cycle_momentum':True,
              'use_beta1':True,'base_lrs':[cfg.LR/25]}
    for key,value in expected.items():
        if scheduler.get(key)!=value: raise ValueError(f'Scheduler {key} mismatch')
    lr,beta1=expected_cycle(cfg,step)
    if not isinstance(scheduler.get('_last_lr'),list) or len(scheduler['_last_lr'])!=1:
        raise ValueError('One shared UE/BS LR required')
    _close(scheduler['_last_lr'][0],lr,'scheduler LR')
    optimizer=payload['optimizer']
    if not isinstance(optimizer.get('state'),dict) or not optimizer['state'] or len(optimizer.get('param_groups',[]))!=1:
        raise ValueError('Complete AdamW history and one shared UE/BS group required')
    group=optimizer['param_groups'][0]
    for key,value in dict(lr=lr,initial_lr=cfg.LR/25,max_lr=cfg.LR,min_lr=cfg.LR/25/10000,
            weight_decay=cfg.WEIGHT_DECAY,eps=1e-8,base_momentum=.85,max_momentum=.95).items():
        _close(group.get(key),value,f'optimizer {key}')
    if len(group.get('betas',[]))!=2: raise ValueError('AdamW betas missing')
    _close(group['betas'][0],beta1,'AdamW beta1');_close(group['betas'][1],.999,'AdamW beta2')
    for key in ('amsgrad','maximize','capturable','differentiable'):
        if group.get(key) is not False: raise ValueError(f'Unexpected AdamW {key}')
    if group.get('foreach') is not None or group.get('fused') is not None:
        raise ValueError('No foreach/fused override')
    rng=payload['rng_state']
    if any(rng.get(k) is None for k in ('python','numpy','torch','cuda')):
        raise ValueError('Complete default RNG states required')
    if not isinstance(rng['cuda'],(tuple,list)) or not rng['cuda']: raise ValueError('CUDA RNG missing')
    return payload


def validate_best_metadata(best,latest,cfg):
    validate_config(cfg);_validate_identity(best,cfg)
    step=best.get('step')
    if type(step) is not int or not 0<step<=latest['step'] or step%cfg.VAL_INTERVAL:
        raise ValueError('Random-init best must be a positive validation boundary, never step0')
    _close(best.get('best_metric'),latest['best_metric'],'best/latest selected metric')
    _close(_validation_values(best,cfg),best['best_metric'],'best six-gamma mean_all')


def validate_model_identity(cfg,model):
    if _json(model.architecture_metadata)!=_json(cfg.architecture_metadata):
        raise ValueError('Actual model layout/UE/BS/protocol differs from saved architecture')
    if model.ue_frozen is not False or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('All effective UE/BS parameters must remain trainable')
    if any(name=='running_pwr' or name.endswith('.running_pwr') for name,_ in model.named_buffers()):
        raise ValueError('New fixed-energy protocol must not contain a running-power prior')
    return model.architecture_metadata


def validate_model_state(torch,model,state):
    expected=model.state_dict()
    if not isinstance(state,dict) or set(state)!=set(expected):
        raise ValueError('Complete state key mismatch; no filtering or partial restore')
    for name,value in state.items():
        reference=expected[name]
        if not torch.is_tensor(value) or value.shape!=reference.shape or value.dtype!=reference.dtype:
            raise ValueError(f'State shape/dtype mismatch: {name}')
        if not torch.isfinite(value).all(): raise ValueError(f'Nonfinite state: {name}')
    # Public layout mappings and masks are not trainable; a checkpoint may not alter them.
    for name,reference in model.named_buffers():
        if name in state and not torch.equal(state[name].cpu(),reference.detach().cpu()):
            raise ValueError(f'Fixed layout/position buffer differs: {name}')


def validate_runtime_resume(torch,model,payload,cfg,*,allow_completed=False):
    import numpy as np
    validate_resume_payload(payload,cfg,allow_completed=allow_completed)
    validate_model_identity(cfg,model);validate_model_state(torch,model,payload['state'])
    model.load_state_dict(payload['state'],strict=True)
    parameters=list(model.named_parameters());group=payload['optimizer']['param_groups'][0]
    ids=group.get('params');moments=payload['optimizer']['state']
    if ids!=list(range(len(parameters))) or not set(moments)<=set(ids):
        raise ValueError('AdamW parameter identity/order/count mismatch')
    inactive=set(model.inactive_parameter_names())
    if not inactive<=set(dict(parameters)): raise ValueError('Invalid inactive parameter declaration')
    for index,(name,p) in enumerate(parameters):
        if name in inactive:
            if index in moments: raise ValueError(f'Unexpected inactive FDMA optimizer state: {name}')
            continue
        saved=moments.get(index)
        if not isinstance(saved,dict) or set(saved)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError(f'Missing active AdamW history: {name}')
        counter=saved['step']
        if not torch.is_tensor(counter) or counter.numel()!=1 or not torch.isfinite(counter).all() or counter.item()!=payload['step']:
            raise ValueError(f'AdamW update count mismatch: {name}')
        for key in ('exp_avg','exp_avg_sq'):
            value=saved[key]
            if not torch.is_tensor(value) or value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all():
                raise ValueError(f'AdamW {key} mismatch: {name}')
        if not (saved['exp_avg_sq']>=0).all(): raise ValueError(f'Negative second moment: {name}')
    rng=payload['rng_state']
    if len(rng['cuda'])!=torch.cuda.device_count(): raise ValueError('Visible CUDA count differs from resume')
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    for device,value in [('cpu',rng['torch'])]+[(f'cuda:{i}',s) for i,s in enumerate(rng['cuda'])]:
        if not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1 or value.device.type!='cpu':
            raise ValueError(f'Malformed RNG tensor for {device}')
        torch.Generator(device=device).set_state(value)


def optimizer_scheduler_factory(cfg,model):
    import torch
    validate_config(cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=cfg.LR,total_steps=cfg.TOTAL_STEPS,
        pct_start=cfg.WARMUP_PCT,anneal_strategy='cos')
    return optimizer,scheduler


def make_model_factory(cfg,model,*,resume_state=None):
    used=False
    def factory(actual_cfg,device):
        nonlocal used,resume_state
        if used or _json(actual_cfg.to_dict())!=_json(cfg.to_dict()):
            raise RuntimeError('Factory reused or grouped experiment identity changed')
        used=True;fresh=resume_state is None
        if resume_state is not None:
            validate_model_state(__import__('torch'),model,resume_state)
            model.load_state_dict(resume_state,strict=True)
        model.to(device);validate_model_identity(cfg,model)
        resume_state=None
        if fresh:
            from main_train import set_seed
            set_seed(cfg.SEED)
        # The shared trainer restores resumed default RNG after optimizer construction.
        return model
    return factory


def _record(cfg,name,data):
    path=Path(cfg.LOG_DIR)/name;_safe_path(path,OUTPUT_ROOT)
    with path.open('x',encoding='utf-8') as stream: json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False)


def completed_resume_status(cfg,latest,best,previous_status):
    """Pure metadata gate for a verified final-update/status-write interruption."""
    validate_resume_payload(latest,cfg,allow_completed=True)
    validate_best_metadata(best,latest,cfg)
    if latest['step']!=cfg.TOTAL_STEPS:
        raise ValueError('Completion recovery requires all planned updates, not a short run')
    if (previous_status.get('experiment')!=cfg.EXP_NAME
            or previous_status.get('state') not in ('running','paused')):
        raise ValueError('Completion recovery requires this unfinished run status')
    if (previous_status.get('state')=='paused'
            and previous_status.get('step')!=latest['step']):
        raise ValueError('Paused/latest completion step mismatch')
    return dict(state='completed',experiment=cfg.EXP_NAME,step=cfg.TOTAL_STEPS,
        best_metric_name=cfg.BEST_METRIC,best_metric=best['best_metric'],
        best_path=cfg.BEST_PATH,latest_path=cfg.SAVE_PATH,
        completion_recovered=True,additional_optimizer_updates=0)


def run_training(cfg,args):
    check_run_state(cfg,resume=args.resume)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Grouped codec requires CUDA; no laptop fallback')
    from main_train import RunOptions,set_seed,train_proposal
    from models.grouped_codec_precoder import GroupedCodecPrecoder
    data=make_channel_data(cfg);execution=GroupedExecutionProtocol(cfg)
    with run_lock(cfg,resume=args.resume,allowed_root=OUTPUT_ROOT):
        check_run_state(cfg,resume=args.resume)
        set_seed(cfg.SEED);model=GroupedCodecPrecoder(cfg)
        actual=validate_model_identity(cfg,model)
        record=dict(arm=cfg.arm,seed=cfg.SEED,config=cfg.to_dict(),config_fingerprint=fingerprint(cfg),
            mode='resume' if args.resume else 'random_initialization',historical_weights_loaded=False,
            actual_model_metadata=actual,git_commit=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip(),
            source_sha256={name:sha256_file(REPO_ROOT/name) for name in SOURCE_FILES},
            backend_before=_backend_info(torch),python=sys.version,platform=platform.platform(),
            scheduler_total_steps=cfg.TOTAL_STEPS,screening_only=True)
        if args.resume:
            payload=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            finished=payload.get('step')==cfg.TOTAL_STEPS
            validate_runtime_resume(torch,model,payload,cfg,allow_completed=finished)
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
            validate_best_metadata(best,payload,cfg);validate_model_state(torch,model,best['state'])
            status=json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
            if status['state']=='paused' and status.get('step')!=payload['step']:
                raise ValueError('Paused/latest update count mismatch')
            start=payload['step']+1;record['resume_sha256']=sha256_file(cfg.SAVE_PATH)
            if finished:
                # No trainer/optimizer/scheduler construction, RNG restoration or
                # additional update: only finalize evidence after strict checks.
                if best['step']==payload['step']:
                    for name,value in payload['state'].items():
                        if not torch.equal(value,best['state'][name]):
                            raise ValueError(f'Same-step best/latest state mismatch: {name}')
                completed=completed_resume_status(cfg,payload,best,status)
                record.update(mode='finalize_interrupted_completion',start_step=start,
                    planned_end_step=cfg.TOTAL_STEPS,additional_optimizer_updates=0)
                token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')+uuid.uuid4().hex[:8]
                _record(cfg,f'invocation_{token}.json',record)
                result=dict(completed_updates=payload['step'],best_step=best['step'],
                    best_metric=best['best_metric'],latest_sha256=record['resume_sha256'],
                    best_sha256=sha256_file(cfg.BEST_PATH),completion_recovered=True,
                    additional_optimizer_updates=0,screening_only=True,
                    backend_after=_backend_info(torch))
                _record(cfg,f'completion_{token}.json',result)
                from as_receiver_batch_support import save_json
                save_json(cfg.STATUS_PATH,completed,replace=True)
                print(json.dumps(result,indent=2));return
            factory=make_model_factory(cfg,model,resume_state=payload['state'])
            del best,payload
        else:
            start=1;factory=make_model_factory(cfg,model)
            Path(cfg.SAVE_DIR).mkdir(parents=True,exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True,exist_ok=False)
        record.update(start_step=start,planned_end_step=min(cfg.TOTAL_STEPS,start+args.stop_after-1)
            if args.stop_after is not None else cfg.TOTAL_STEPS,
            random_address_resume='train index equals next global_step; fixed validation index0',
            full_training_bitwise_resume_verified=False)
        token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')+uuid.uuid4().hex[:8]
        _record(cfg,f'invocation_{token}.json',record)
        try:
            train_proposal(cfg,RunOptions(resume=args.resume,skip_existing=False,
                stop_after=args.stop_after,init_checkpoint=None),model_factory=factory,channel_data=data,
                optimizer_scheduler_factory=optimizer_scheduler_factory,execution_protocol=execution)
            latest=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            validate_resume_payload(latest,cfg,allow_completed=True)
            if latest['step']!=record['planned_end_step']: raise RuntimeError('Unexpected completed update count')
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
            validate_best_metadata(best,latest,cfg)
            result=dict(completed_updates=latest['step'],best_step=best['step'],best_metric=best['best_metric'],
                latest_sha256=sha256_file(cfg.SAVE_PATH),best_sha256=sha256_file(cfg.BEST_PATH),
                backend_after=_backend_info(torch),screening_only=True)
            _record(cfg,f'completion_{token}.json',result);print(json.dumps(result,indent=2))
        except BaseException as error:
            _record(cfg,f'failure_{token}.json',dict(error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(),automatic_fallback=False))
            raise


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=SEEDS)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--train',action='store_true')
    mode.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true')
    return parser


def main(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    if not(args.dry_run or args.train or args.stop_after is not None):
        if args.resume: parser.error('Choose an explicit run mode')
        parser.print_help();return 0
    if args.arm is None or args.seed is None: parser.error('Choose arm and seed')
    if args.resume and args.dry_run: parser.error('Dry-run does not resume checkpoints')
    if args.stop_after is not None and (not 0<args.stop_after<=20000 or args.stop_after%500):
        parser.error('stop-after must be a positive multiple of500 within the fixed20k plan')
    cfg=grouped_codec_config(args.arm,args.seed);validate_config(cfg);validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps(dict(action='plan only; no weights/model imports/output writes',config=cfg.to_dict(),
            fingerprint=fingerprint(cfg),latest=cfg.SAVE_PATH,best=cfg.BEST_PATH,
            scheduler_total_steps=20000,historical_checkpoint=None),indent=2));return 0
    run_training(cfg,args);return 0


if __name__=='__main__': raise SystemExit(main())
