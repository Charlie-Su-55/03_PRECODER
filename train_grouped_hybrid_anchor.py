#!/usr/bin/env python3
"""Independent three-arm hybrid continuation; explicitly invoked CUDA only."""
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

from configs.grouped_hybrid_anchor_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, REPO_ROOT, anchor_config, validate_config,
    channel_data_api, make_channel_data, protocol, lifecycle_descriptor,
    execution_descriptor, source_identity, SOURCE_BEST_SHA256,
)
from configs.grouped_codec_fdma_train_config import fdma_config
from train_grouped_codec import (apply_backend, fingerprint, _json, run_lock,
    _output_paths, _safe_path, _close, expected_cycle, sha256_file, _backend_info,
    _validate_identity, _validation_values, SOURCE_FILES as COMMON_SOURCE_FILES)

SOURCE_FILES = tuple(dict.fromkeys(COMMON_SOURCE_FILES + (
    'train_grouped_hybrid_anchor.py','configs/grouped_hybrid_anchor_train_config.py',
    'models/grouped_hybrid_anchor.py','utils/grouped_hybrid_feedback.py',
    'models/grouped_codec_fdma.py','configs/grouped_codec_fdma_train_config.py',
    'train_grouped_codec_fdma.py','evaluate_grouped_codec_fdma.py',
    'grouped_hybrid_resume_compat.py')))
PRIOR_KEYS = ('power_prior.pi_nodes','power_prior.update_count','power_prior.node_update_mass')


class AnchorExecutionProtocol:
    def __init__(self,cfg):
        validate_config(cfg); self.cfg=cfg; self.config_snapshot=cfg.to_dict()

    @property
    def descriptor(self): return execution_descriptor(self.cfg.SEED)

    @staticmethod
    def apply_backend(torch): apply_backend(torch)

    def training(self,model,H_dl,H_ul,global_step,device):
        if type(global_step) is not int or not 1<=global_step<=self.cfg.TOTAL_STEPS: raise ValueError('Invalid absolute training step')
        model.power_prior.validate_state(global_step-1)
        gamma=protocol.sample_gamma(self.cfg.SEED,global_step)
        noise=protocol.standard_noise(self.cfg.SEED,'train',global_step,H_dl.shape[0],device=device)
        output=model.training_with_statistics(H_dl,H_ul,gamma,global_step=global_step,standard_noise=noise)
        return output,gamma

    def validate(self,model,H_dl,H_ul,criterion,cfg,device):
        import torch
        if cfg.to_dict()!=self.config_snapshot or H_dl.shape[0]!=256 or H_ul.shape[0]!=256:
            raise ValueError('Fixed 256-channel validation identity changed')
        before={name:value.detach().clone() for name,value in model.named_buffers()}
        modes={name:module.training for name,module in model.named_modules()}
        # Independent noise generation has no default-RNG side effects; also
        # restore default streams so validation never changes training dropout.
        from main_train import capture_rng_state, restore_rng_state
        rng=capture_rng_state();table={(0,gamma):0. for gamma in cfg.VAL_SNR_LIST}
        try:
            model.eval()
            with torch.no_grad():
                for start in range(0,256,8):
                    noise=protocol.standard_noise(cfg.SEED,'validation',start//8,8,device=device)
                    for gamma in cfg.VAL_SNR_LIST:
                        W,Hhat=model.forward_with_representation(H_dl[start:start+8],H_ul[start:start+8],float(gamma),standard_noise=noise)
                        if not torch.isfinite(W).all() or not torch.isfinite(Hhat).all(): raise FloatingPointError('Nonfinite validation output')
                        _,rate=criterion(W,H_dl[start:start+8],10.**(-25/10))
                        if not math.isfinite(rate): raise FloatingPointError('Nonfinite validation rate')
                        table[(0,gamma)]+=rate/32
            for name,value in model.named_buffers():
                if not torch.equal(before[name],value): raise ValueError('Validation changed model buffer '+name)
        finally:
            restore_rng_state(rng)
            for name,module in model.named_modules(): module.training=modes[name]
        return table

    def format_validation(self,table,step,cfg):
        if set(table)!={(0,g) for g in cfg.VAL_SNR_LIST} or any(not math.isfinite(v) for v in table.values()):
            raise ValueError('Incomplete/nonfinite six-node validation')
        mean=sum(table.values())/6
        return f'Step {step} | {cfg.arm} | gamma; DL25 | '+ ' | '.join(f'{g}dB: {table[0,g]:.6f}' for g in cfg.VAL_SNR_LIST)+f' | mean_all: {mean:.6f}',mean


class AnchorLifecycle:
    def __init__(self,cfg):
        validate_config(cfg);self.cfg=cfg;self.config_snapshot=cfg.to_dict()
    @property
    def descriptor(self):return lifecycle_descriptor()
    def after_optimizer_step(self,model,output,step):
        if not isinstance(output,(tuple,list)) or len(output)!=3: raise ValueError('Expected W,Hhat,detached training statistics')
        model.post_optimizer_commit(output[2],step)
    def validate_model_step(self,model,step):model.power_prior.validate_state(step)
    def validate_resume_checkpoint(self,payload):
        return validate_resume_payload(payload,self.cfg,allow_completed=True)


def validate_output_paths(cfg,*,allowed_root=OUTPUT_ROOT):return _output_paths(cfg,allowed_root=allowed_root)


def check_run_state(cfg,*,resume,allowed_root=OUTPUT_ROOT):
    validate_config(cfg);validate_output_paths(cfg,allowed_root=allowed_root)
    if not resume:
        for path in (cfg.SAVE_DIR,cfg.LOG_DIR):
            if Path(path).exists():raise FileExistsError('Refusing to overwrite '+str(path))
        return
    for name in ('SAVE_PATH','BEST_PATH','CONFIG_PATH','STATUS_PATH','METRICS_PATH'):
        if not Path(getattr(cfg,name)).is_file():raise FileNotFoundError('Resume requires '+name)
    if not (Path(cfg.LOG_DIR)/'initialization.json').is_file():raise FileNotFoundError('Resume requires immutable initialization provenance')
    saved=json.loads(Path(cfg.CONFIG_PATH).read_text(encoding='utf-8'))
    if _json(saved)!=_json(dict(cfg.to_dict(),config_fingerprint=fingerprint(cfg))):raise ValueError('Resume complete config/fingerprint mismatch')
    status=json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
    if status.get('experiment')!=cfg.EXP_NAME or status.get('state') not in ('running','paused'):
        raise ValueError('Resume requires this unfinished arm/seed; completed runs are not retrained')


def verify_source_payloads(seed,best,latest,sidecars,hashes):
    from evaluate_grouped_codec_fdma import verify_training_metadata as verify_F
    cfg=fdma_config(seed)
    if hashes.get('best_sha256')!=SOURCE_BEST_SHA256[seed]:raise ValueError('Pinned Legacy-FDMA best SHA256 mismatch')
    checked=verify_F(seed,best,latest,sidecars)
    return dict(source_identity(seed),**hashes,best_step=best['step'],latest_step=latest['step'],
        best_metric=best['best_metric'],config=best['config'],verified_selection=checked,
        source_files_unchanged=True,source_checkpoint_not_modified=True)


def load_source(torch,seed):
    """Read only exact same-seed F paths, not the historical batch/source-code hash."""
    cfg=fdma_config(seed);paths={name:getattr(cfg,attr) for name,attr in (
        ('best','BEST_PATH'),('latest','SAVE_PATH'),('config','CONFIG_PATH'),('status','STATUS_PATH'))}
    hashes={name+'_sha256':sha256_file(path) for name,path in paths.items()}
    if hashes['best_sha256']!=SOURCE_BEST_SHA256[seed]:raise ValueError('Pinned source best SHA256 mismatch before deserialization')
    best=torch.load(paths['best'],map_location='cpu',weights_only=False)
    latest=torch.load(paths['latest'],map_location='cpu',weights_only=False)
    sidecars={name:json.loads(Path(paths[name]).read_text(encoding='utf-8')) for name in ('config','status')}
    record=verify_source_payloads(seed,best,latest,sidecars,hashes)
    from evaluate_proposal_as_direct import check_tensor_state
    check_tensor_state(torch,latest['state'],best['state'],equal=latest['step']==best['step'])
    for name,path in paths.items():
        if sha256_file(path)!=hashes[name+'_sha256']:raise ValueError('Source changed while reading '+path)
    return best,record


def _prior_count(payload):
    state=payload['state']
    if any(key not in state for key in PRIOR_KEYS):raise ValueError('Checkpoint missing prior state/count/mass')
    count=state['power_prior.update_count']
    if hasattr(count,'numel'):
        if count.numel()!=1:raise ValueError('Prior count is not scalar')
        count=count.item()
    if isinstance(count,bool) or not isinstance(count,(int,float)) or count!=payload['step']:
        raise ValueError('Prior committed count must equal checkpoint optimizer step')


def validate_resume_payload(payload,cfg,*,allow_completed=False):
    validate_config(cfg);_validate_identity(payload,cfg)
    # The historical data helper intentionally requires step>=1. This declared
    # lifecycle validates step0 directly without changing old rules or its step.
    required={'step','state','optimizer','scheduler','rng_state','best_metric'}
    if not required<=payload.keys() or not isinstance(payload['best_metric'],(int,float)) or not math.isfinite(payload['best_metric']):
        raise ValueError('Incomplete/nonfinite full resume state')
    step=payload['step'];upper=cfg.TOTAL_STEPS if allow_completed else cfg.TOTAL_STEPS-1
    if type(step) is not int or not 0<=step<=upper or step%500:raise ValueError('Resume needs step0 or 500-step boundary of fixed20k')
    _prior_count(payload)
    if _validation_values(payload,cfg)>payload['best_metric']+1e-10:raise ValueError('Latest metric exceeds selected best')
    phases=[dict(end_step=1999.,start_lr='initial_lr',end_lr='max_lr',start_momentum='max_momentum',end_momentum='base_momentum'),
        dict(end_step=19999,start_lr='max_lr',end_lr='min_lr',start_momentum='base_momentum',end_momentum='max_momentum')]
    scheduler=payload.get('scheduler',{})
    for key,value in dict(total_steps=20000,last_epoch=step,_step_count=step+1,_schedule_phases=phases,
            _anneal_func_type='cos',cycle_momentum=True,use_beta1=True,base_lrs=[cfg.LR/25]).items():
        if scheduler.get(key)!=value:raise ValueError('Scheduler '+key+' mismatch')
    lr,beta1=expected_cycle(cfg,step)
    if len(scheduler.get('_last_lr',[]))!=1:raise ValueError('One optimizer LR required')
    _close(scheduler['_last_lr'][0],lr,'scheduler LR')
    optimizer=payload.get('optimizer',{});moments=optimizer.get('state')
    if not isinstance(moments,dict) or bool(moments)!=(step>0) or len(optimizer.get('param_groups',[]))!=1:
        raise ValueError('Step0 empty optimizer; positive step complete history and one group required')
    group=optimizer['param_groups'][0]
    for key,value in dict(lr=lr,initial_lr=cfg.LR/25,max_lr=cfg.LR,min_lr=cfg.LR/25/10000,
        weight_decay=cfg.WEIGHT_DECAY,eps=1e-8,base_momentum=.85,max_momentum=.95).items():_close(group.get(key),value,key)
    if len(group.get('betas',[]))!=2:raise ValueError('AdamW betas missing')
    _close(group['betas'][0],beta1,'beta1');_close(group['betas'][1],.999,'beta2')
    for key in ('amsgrad','maximize','capturable','differentiable'):
        if group.get(key) is not False:raise ValueError('AdamW '+key+' changed')
    if group.get('foreach') is not None or group.get('fused') is not None:raise ValueError('No optimizer override')
    rng=payload.get('rng_state',{})
    if any(rng.get(k) is None for k in ('python','numpy','torch','cuda')) or not isinstance(rng['cuda'],(list,tuple)) or not rng['cuda']:
        raise ValueError('Complete Python/NumPy/CPU/CUDA RNG required')
    return payload


def validate_best_metadata(best,latest,cfg):
    validate_resume_payload(best,cfg,allow_completed=True)
    if best['step']>latest['step']:raise ValueError('Best follows latest')
    _close(best['best_metric'],latest['best_metric'],'best/latest metric')
    _close(_validation_values(best,cfg),best['best_metric'],'selected best mean_all')


def validate_model_identity(cfg,model):
    if _json(model.architecture_metadata)!=_json(cfg.architecture_metadata):raise ValueError('Hybrid architecture metadata differs')
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('All UE/BS parameters must remain trainable')
    return model.architecture_metadata


def validate_model_state(torch,model,state,*,step):
    expected=model.state_dict()
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('Strict complete state keys required')
    for name,value in state.items():
        if not torch.is_tensor(value) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError('State shape/dtype/finite mismatch '+name)
    _prior_count(dict(state=state,step=step))
    for name,value in model.named_buffers():
        if name in state and name not in PRIOR_KEYS and not torch.equal(value.detach().cpu(),state[name].cpu()):
            raise ValueError('Fixed geometry buffer changed '+name)
    # Validate a tiny independent prior, not a second model or a state mutation.
    prior=protocol.make_power_prior(model.power_prior.layout)
    prior.load_state_dict({key.split('.',1)[1]:state[key] for key in PRIOR_KEYS},strict=True)
    prior.validate_state(step)


def build_model(torch,arm,seed,payload,device):
    from models.grouped_hybrid_anchor import GroupedHybridAnchorPrecoder
    cfg=anchor_config(arm,seed);_validate_identity(payload,cfg)
    model=GroupedHybridAnchorPrecoder(cfg)
    validate_model_identity(cfg,model);validate_model_state(torch,model,payload['state'],step=payload['step'])
    model.load_state_dict(payload['state'],strict=True);model.power_prior.validate_state(payload['step'])
    return model.to(device).eval(),cfg


def verify_training_metadata(arm,seed,best,latest,sidecars):
    cfg=anchor_config(arm,seed);validate_resume_payload(latest,cfg,allow_completed=True);validate_best_metadata(best,latest,cfg)
    if latest['step']!=20000:raise ValueError('Training has not completed20k')
    if _json(sidecars['config'])!=_json(dict(cfg.to_dict(),config_fingerprint=fingerprint(cfg))):raise ValueError('Saved full config mismatch')
    for key,value in dict(state='completed',step=20000,experiment=cfg.EXP_NAME,best_metric_name='mean_all',best_metric=best['best_metric']).items():
        if sidecars['status'].get(key)!=value:raise ValueError('Training status mismatch '+key)
    return dict(config=best['config'],config_fingerprint=best['config_fingerprint'],selected_best_step=best['step'],
        latest_step=20000,completed_updates=20000,best_metric=best['best_metric'],training_status=sidecars['status'])


def validate_runtime_resume(torch,model,payload,cfg,*,allow_completed=False):
    import numpy as np
    validate_resume_payload(payload,cfg,allow_completed=allow_completed);validate_model_identity(cfg,model)
    validate_model_state(torch,model,payload['state'],step=payload['step'])
    model.load_state_dict(payload['state'],strict=True);model.power_prior.validate_state(payload['step'])
    params=list(model.named_parameters());group=payload['optimizer']['param_groups'][0];moments=payload['optimizer']['state']
    if group.get('params')!=list(range(len(params))) or not set(moments)<=set(group['params']):raise ValueError('Optimizer parameter order mismatch')
    inactive=set(model.inactive_parameter_names())
    for index,(name,p) in enumerate(params):
        if payload['step']==0 or name in inactive:
            if index in moments:raise ValueError('Unexpected initial/inactive optimizer state '+name)
            continue
        saved=moments.get(index)
        if not isinstance(saved,dict) or set(saved)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Incomplete active moments '+name)
        if not torch.is_tensor(saved['step']) or saved['step'].numel()!=1 or saved['step'].item()!=payload['step']:raise ValueError('AdamW step mismatch '+name)
        for key in ('exp_avg','exp_avg_sq'):
            v=saved[key]
            if not torch.is_tensor(v) or v.shape!=p.shape or v.dtype!=p.dtype or not torch.isfinite(v).all():raise ValueError('Moment mismatch '+name)
        if not (saved['exp_avg_sq']>=0).all():raise ValueError('Negative second moment')
    rng=payload['rng_state']
    if len(rng['cuda'])!=torch.cuda.device_count():raise ValueError('Visible CUDA count changed')
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    for device,value in [('cpu',rng['torch'])]+[(f'cuda:{i}',v) for i,v in enumerate(rng['cuda'])]:
        if not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1 or value.device.type!='cpu':raise ValueError('Invalid RNG state')
        torch.Generator(device=device).set_state(value)


def optimizer_scheduler_factory(cfg,model):
    import torch
    validate_config(cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=cfg.LR,total_steps=20000,pct_start=.1,anneal_strategy='cos')
    return optimizer,scheduler


def _record(cfg,name,value):
    path=Path(cfg.LOG_DIR)/name;_safe_path(path,OUTPUT_ROOT)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def validate_initialization_provenance(cfg,initial,record,compat_path=None):
    """Actual resume boundary: JSON semantics for metadata, exact bytes for code."""
    from grouped_hybrid_resume_compat import differences, validate_trainer_transition
    if _json(initial.get('source'))!=_json(record['source']):
        raise ValueError('Resume source checkpoint/config metadata changed: '+
            json.dumps(differences(initial.get('source'),record['source']),sort_keys=True))
    if initial.get('config_fingerprint')!=fingerprint(cfg) or _json(initial.get('config'))!=_json(cfg.to_dict()):
        raise ValueError('Resume initialization configuration changed: '+
            json.dumps(differences(dict(config=initial.get('config'),config_fingerprint=initial.get('config_fingerprint')),
                dict(config=cfg.to_dict(),config_fingerprint=fingerprint(cfg))),sort_keys=True))
    if compat_path is not None:
        return validate_trainer_transition(cfg,initial,record,compat_path)
    if initial.get('source_sha256')!=record['source_sha256']:
        raise ValueError('Resume source-code SHA256 changed (explicit registered repair required): '+
            json.dumps(differences(initial.get('source_sha256'),record['source_sha256']),sort_keys=True))
    return None


def run_training(cfg,args):
    check_run_state(cfg,resume=args.resume)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')
    from main_train import RunOptions,train_proposal,set_seed
    from models.grouped_hybrid_anchor import GroupedHybridAnchorPrecoder,migrate_legacy_fdma_state_dict
    with run_lock(cfg,resume=args.resume,allowed_root=OUTPUT_ROOT):
        check_run_state(cfg,resume=args.resume)
        source,source_record=load_source(torch,cfg.SEED)
        set_seed(protocol.namespace_seed(cfg.SEED,'initialization'))
        model=GroupedHybridAnchorPrecoder(cfg);validate_model_identity(cfg,model)
        record=dict(arm=cfg.arm,seed=cfg.SEED,config=cfg.to_dict(),config_fingerprint=fingerprint(cfg),
            source=source_record,mode='resume' if args.resume else 'explicit_source_migration',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip(),
            source_sha256={name:sha256_file(REPO_ROOT/name) for name in SOURCE_FILES},
            backend_before=_backend_info(torch),python=sys.version,platform=platform.platform(),
            full_training_bitwise_resume_verified=False,screening_only=True)
        if args.resume:
            initial=json.loads((Path(cfg.LOG_DIR)/'initialization.json').read_text(encoding='utf-8'))
            receipt=validate_initialization_provenance(cfg,initial,record,getattr(args,'resume_compat',None))
            if receipt is not None:record['resume_compatibility']=receipt
            payload=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            validate_runtime_resume(torch,model,payload,cfg,allow_completed=True)
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False);validate_best_metadata(best,payload,cfg)
            validate_model_state(torch,model,best['state'],step=best['step'])
            status=json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
            if status['state']=='paused' and status.get('step')!=payload['step']:raise ValueError('Paused/latest step mismatch')
            start=payload['step']+1;record['resume_sha256']=sha256_file(cfg.SAVE_PATH)
            if payload['step']==20000:
                from as_receiver_batch_support import save_json
                completion=dict(state='completed',experiment=cfg.EXP_NAME,step=20000,best_metric_name='mean_all',
                    best_metric=best['best_metric'],best_path=cfg.BEST_PATH,latest_path=cfg.SAVE_PATH,
                    completion_recovered=True,additional_optimizer_updates=0)
                save_json(cfg.STATUS_PATH,completion,replace=True);print(json.dumps(completion,indent=2));return
        else:
            record['migration']=migrate_legacy_fdma_state_dict(model,source['state'])
            model.power_prior.validate_state(0);start=1
            Path(cfg.SAVE_DIR).mkdir(parents=True,exist_ok=False);Path(cfg.LOG_DIR).mkdir(parents=True,exist_ok=False)
            _record(cfg,'initialization.json',record)
        del source
        record.update(start_step=start,planned_end_step=min(20000,start+args.stop_after-1) if args.stop_after else 20000)
        token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')+uuid.uuid4().hex[:8]
        _record(cfg,'invocation_'+token+'.json',record)
        def factory(actual,device):
            if actual.to_dict()!=cfg.to_dict():raise ValueError('Factory configuration changed')
            if args.resume and sha256_file(cfg.SAVE_PATH)!=record['resume_sha256']:
                raise ValueError('Latest changed between strict resume validation and optimizer construction')
            return model.to(device)
        try:
            train_proposal(cfg,RunOptions(resume=args.resume,skip_existing=False,stop_after=args.stop_after,init_checkpoint=None),
                model_factory=factory,channel_data=make_channel_data(cfg),optimizer_scheduler_factory=optimizer_scheduler_factory,
                execution_protocol=AnchorExecutionProtocol(cfg),lifecycle_protocol=AnchorLifecycle(cfg))
            latest=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            validate_resume_payload(latest,cfg,allow_completed=True)
            if latest['step']!=record['planned_end_step']:raise ValueError('Unexpected completed update count')
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False);validate_best_metadata(best,latest,cfg)
            _record(cfg,'completion_'+token+'.json',dict(completed_updates=latest['step'],best_step=best['step'],
                best_metric=best['best_metric'],best_sha256=sha256_file(cfg.BEST_PATH),latest_sha256=sha256_file(cfg.SAVE_PATH),
                committed_prior_updates=int(model.power_prior.update_count.item()),backend_after=_backend_info(torch)))
        except BaseException as error:
            _record(cfg,'failure_'+token+'.json',dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),automatic_fallback=False));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=SEEDS)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--train',action='store_true');mode.add_argument('--stop-after',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-compat',type=Path,help='Exact registered 266d9e5 repair record; not an ignore-hash switch')
    args=parser.parse_args(argv)
    if args.resume_compat is not None and (not args.resume or args.dry_run):parser.error('--resume-compat requires actual --resume')
    if not(args.dry_run or args.train or args.stop_after is not None):parser.print_help();return 0
    if args.arm is None or args.seed is None:parser.error('Explicit arm and seed required')
    if args.dry_run and args.resume:parser.error('Dry-run never resumes weights')
    if args.stop_after is not None and(not 0<args.stop_after<=20000 or args.stop_after%500):parser.error('Pause must be a 500-step multiple within20k')
    cfg=anchor_config(args.arm,args.seed);validate_config(cfg);validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps(dict(action='plan only; no model/weights/output writes',config=cfg.to_dict(),
            fingerprint=fingerprint(cfg),latest=cfg.SAVE_PATH,best=cfg.BEST_PATH),indent=2));return 0
    run_training(cfg,args);return 0


if __name__=='__main__':raise SystemExit(main())
