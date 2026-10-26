#!/usr/bin/env python3
"""Pinned Swin +20k control, reusing baseline_train's opt-in hooks and loop."""
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
from types import SimpleNamespace
import uuid

from configs.swin_adapt_train_config import (
    SEEDS, OUTPUT_ROOT, REPO_ROOT, swin_adapt_config, validate_config,
    optimizer_group_spec, make_channel_data, channel_data_api,
)
from evaluate_proposal_as_direct import check_tensor_state, verify_archived_metadata
from evaluate_proposal_lite import _saved_fingerprint
from preflight_proposal_as_direct import DEFAULT_MANIFEST, _backend_info, sha256_file, verify_checkpoint_hash
from train_proposal_as_direct import _safe_path, _close, expected_cycle
from train_proposal_as_receiver import run_lock

SOURCE_FILES = (
    'train_swin_adaptation.py', 'configs/swin_adapt_train_config.py', 'baseline_train.py',
    'models/swin_precoder.py', 'models/baseline_common.py', 'configs/baseline_config.py',
    'utils/replayable_channel_data.py', 'utils/sionna_rng.py', 'utils/channel_generator.py',
    'configs/as_receiver_train_config.py', 'train_proposal_as_receiver.py',
    'train_proposal_as_direct.py', 'evaluate_proposal_as_direct.py', 'evaluate_proposal_lite.py',
)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def fingerprint(cfg):
    return _saved_fingerprint(cfg.to_dict(), 'swin')


def validate_output_paths(cfg, *, allowed_root=OUTPUT_ROOT):
    root = Path(allowed_root).absolute()
    if Path(cfg.OUTPUT_ROOT).absolute() != root:
        raise ValueError('Wrong dedicated Swin adaptation output root')
    for field, directory, filename in (
        ('SAVE_DIR', 'checkpoints', None), ('LOG_DIR', 'logs', None),
        ('BEST_PATH', 'checkpoints', 'best.pth'), ('SAVE_PATH', 'checkpoints', 'latest.pth'),
        ('CONFIG_PATH', 'logs', 'config.json'), ('STATUS_PATH', 'logs', 'status.json'),
        ('METRICS_PATH', 'logs', 'metrics.jsonl'),
    ):
        expected = root / directory / cfg.EXP_NAME
        if filename is not None:
            expected /= filename
        actual = Path(getattr(cfg, field))
        _safe_path(actual, root)
        if actual.absolute() != expected:
            raise ValueError(f'Unexpected {field}')


def check_run_state(cfg, *, resume, allowed_root=OUTPUT_ROOT):
    validate_config(cfg)
    validate_output_paths(cfg, allowed_root=allowed_root)
    if not resume:
        for value in (cfg.SAVE_DIR, cfg.LOG_DIR):
            if Path(value).exists():
                raise FileExistsError(f'Refusing to overwrite/reinitialize {value}')
        return
    for field in ('SAVE_PATH','BEST_PATH','CONFIG_PATH','STATUS_PATH','METRICS_PATH'):
        if not Path(getattr(cfg,field)).is_file():
            raise FileNotFoundError(f'Resume requires {field}: {getattr(cfg,field)}')
    if _json(json.loads(Path(cfg.CONFIG_PATH).read_text(encoding='utf-8'))) != _json(cfg.to_dict()):
        raise ValueError('Resume full sidecar config/source/data mismatch')
    status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
    if status.get('experiment') != cfg.EXP_NAME or status.get('state') not in ('paused','running'):
        raise ValueError('Resume requires this unfinished seed/experiment')


def cycle_values(cfg, peak, step):
    return expected_cycle(SimpleNamespace(LR=peak,TOTAL_STEPS=cfg.TOTAL_STEPS,
                                         WARMUP_PCT=cfg.WARMUP_PCT), step)


def parameter_groups(cfg, model):
    named = list(model.named_parameters())
    if not named or len({id(p) for _,p in named}) != len(named) or any(not p.requires_grad for _,p in named):
        raise ValueError('Swin adaptation requires unique, all-trainable parameters')
    groups = []
    for spec in optimizer_group_spec(cfg):
        selected = [(name,p) for name,p in named if name.startswith(spec['prefix'])]
        if not selected:
            raise ValueError(f'Missing {spec["name"]} parameter group')
        groups.append(dict(params=[p for _,p in selected], group_name=spec['name'],
                           parameter_names=[n for n,_ in selected], lr=spec['peak_lr']))
    if sum(len(g['params']) for g in groups) != len(named):
        raise ValueError('Unexpected Swin parameter outside shared encoder/decoder')
    return groups


def optimizer_scheduler_factory(cfg, model):
    import torch
    optimizer = torch.optim.AdamW(parameter_groups(cfg,model), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[s['peak_lr'] for s in optimizer_group_spec(cfg)],
        total_steps=cfg.TOTAL_STEPS,pct_start=cfg.WARMUP_PCT,anneal_strategy='cos')
    return optimizer,scheduler


def validate_common(payload,cfg):
    if not isinstance(payload,dict) or _json(payload.get('config')) != _json(cfg.to_dict()):
        raise ValueError('Checkpoint full source/config/data protocol mismatch')
    if payload.get('config_fingerprint') != fingerprint(cfg):
        raise ValueError('Missing/mismatched Swin adaptation fingerprint')
    if not isinstance(payload.get('state'),dict) or not payload['state']:
        raise ValueError('Missing Swin model state')
    if any(not isinstance(k,str) or not k.startswith(('encoder.','decoder.')) for k in payload['state']):
        raise ValueError('Unexpected model identity/state keys')
    value=payload.get('best_score')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Nonfinite/missing selected validation score')
    val=payload.get('validation')
    if not isinstance(val,dict) or val.get('selection_name') != 'mean_all':
        raise ValueError('Expected six-point mean_all validation selection')
    rates=val.get('rates')
    if not isinstance(rates,dict) or set(rates) != {str(n) for n in cfg.VAL_SNR_LIST}:
        raise ValueError('Incomplete six-SNR validation')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in rates.values()):
        raise ValueError('Nonfinite validation rates')
    mean=sum(rates.values())/len(rates)
    _close(val.get('mean_rate'),mean,'six-SNR mean rate')
    _close(val.get('mean_all'),mean,'six-SNR mean_all')
    _close(val.get('selection_score'),mean,'six-SNR selection')


def validate_resume_payload(payload,cfg,*,allow_completed=False):
    validate_common(payload,cfg)
    step=payload.get('step')
    maximum=cfg.TOTAL_STEPS if allow_completed else cfg.TOTAL_STEPS-1
    if type(step) is not int or not 0<step<=maximum or step%cfg.VAL_INTERVAL:
        raise ValueError('Resume needs a 500-update boundary inside fixed 20k budget')
    selected=payload.get('selected_best_step')
    if type(selected) is not int or not 0<=selected<=step or selected%500:
        raise ValueError('Missing/invalid selected best step (source step0 allowed)')
    if payload['best_score']+1e-12 < payload['validation']['selection_score']:
        raise ValueError('Latest best_score is stale or below current validation')
    scheduler=payload.get('scheduler',{})
    specs=optimizer_group_spec(cfg)
    phases=[dict(end_step=cfg.WARMUP_PCT*cfg.TOTAL_STEPS-1,start_lr='initial_lr',end_lr='max_lr',
                 start_momentum='max_momentum',end_momentum='base_momentum'),
            dict(end_step=cfg.TOTAL_STEPS-1,start_lr='max_lr',end_lr='min_lr',
                 start_momentum='base_momentum',end_momentum='max_momentum')]
    for key,value in dict(total_steps=20000,last_epoch=step,_step_count=step+1,
        _schedule_phases=phases,_anneal_func_type='cos',cycle_momentum=True,use_beta1=True,
        base_lrs=[s['peak_lr']/25 for s in specs]).items():
        if scheduler.get(key)!=value:
            raise ValueError(f'Scheduler mismatch: {key}')
    optimizer=payload.get('optimizer',{})
    groups=optimizer.get('param_groups',[])
    last=scheduler.get('_last_lr',[])
    if len(groups)!=2 or len(last)!=2 or not isinstance(optimizer.get('state'),dict) or not optimizer['state']:
        raise ValueError('Two LR groups and nonempty optimizer history required')
    all_ids,all_names=[],[]
    for group,spec,saved_lr in zip(groups,specs,last):
        names,ids=group.get('parameter_names'),group.get('params')
        if group.get('group_name')!=spec['name'] or not isinstance(names,list) or not names or not isinstance(ids,list) or len(ids)!=len(names):
            raise ValueError('Optimizer group identity/name/parameter count mismatch')
        if any(not isinstance(n,str) or not n.startswith(spec['prefix']) or n not in payload['state'] for n in names):
            raise ValueError('Optimizer encoder/decoder ownership mismatch')
        if any(type(i) is not int or i<0 for i in ids):
            raise ValueError('Invalid optimizer parameter IDs')
        all_ids+=ids; all_names+=names
        peak=spec['peak_lr'];lr,beta=cycle_values(cfg,peak,step)
        for key,value in dict(lr=lr,initial_lr=peak/25,max_lr=peak,min_lr=peak/25/10000,
            weight_decay=1e-5,eps=1e-8,base_momentum=.85,max_momentum=.95).items():
            _close(group.get(key),value,f'{spec["name"]}.{key}')
        _close(saved_lr,lr,'scheduler LR')
        if len(group.get('betas',[]))!=2:
            raise ValueError('Missing AdamW betas')
        _close(group['betas'][0],beta,'beta1');_close(group['betas'][1],.999,'beta2')
        if any(group.get(k) is not False for k in ('amsgrad','maximize','capturable','differentiable')):
            raise ValueError('Unexpected AdamW mode')
        if group.get('foreach') is not None or group.get('fused') is not None:
            raise ValueError('Unexpected optimizer backend override')
    if all_ids!=list(range(len(all_ids))) or len(set(all_names))!=len(all_names) or set(optimizer['state'])!=set(all_ids):
        raise ValueError('Incomplete/reordered/duplicate optimizer states')
    rng=payload.get('rng_state',{})
    if any(rng.get(k) is None for k in ('python','numpy','torch','cuda')) or not isinstance(rng['cuda'],(list,tuple)) or not rng['cuda']:
        raise ValueError('Complete Python/NumPy/CPU/all-CUDA RNG required')


def validate_best_metadata(best,latest,cfg):
    validate_common(best,cfg)
    step=best.get('step')
    if type(step) is not int or not 0<=step<=latest['step'] or step%500:
        raise ValueError('Selected best step must be source0 or a completed validation step')
    if latest.get('selected_best_step')!=step or best.get('selected_best_step')!=step:
        raise ValueError('Selected step disagrees with saved best/latest')
    _close(best['best_score'],latest['best_score'],'best/latest selection score')
    _close(best['best_score'],best['validation']['selection_score'],'best validation score')


def validate_runtime_resume(torch,model,payload,cfg):
    import numpy as np
    validate_resume_payload(payload,cfg)
    check_tensor_state(torch,model.state_dict(),payload['state'])
    expected=parameter_groups(cfg,model)
    for current,saved in zip(expected,payload['optimizer']['param_groups']):
        if current['parameter_names']!=saved['parameter_names']:
            raise ValueError('Actual Swin optimizer parameter names/order differ')
        for name,param,index in zip(current['parameter_names'],current['params'],saved['params']):
            state=payload['optimizer']['state'][index]
            if not isinstance(state,dict) or set(state)!={'step','exp_avg','exp_avg_sq'}:
                raise ValueError(f'Incomplete optimizer moments: {name}')
            counter=state['step']
            if not torch.is_tensor(counter) or counter.numel()!=1 or not torch.isfinite(counter).all() or counter.item()!=payload['step']:
                raise ValueError(f'Optimizer update counter differs: {name}')
            for key in ('exp_avg','exp_avg_sq'):
                value=state[key]
                if not torch.is_tensor(value) or value.shape!=param.shape or value.dtype!=param.dtype or not torch.isfinite(value).all():
                    raise ValueError(f'Optimizer moment shape/dtype/finiteness: {name}')
            if not (state['exp_avg_sq']>=0).all():
                raise ValueError('Negative optimizer second moment')
    rng=payload['rng_state']
    if len(rng['cuda'])!=torch.cuda.device_count():
        raise ValueError('Changed visible CUDA RNG count')
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    for device,value in [('cpu',rng['torch'])]+[(f'cuda:{i}',v) for i,v in enumerate(rng['cuda'])]:
        if not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1 or value.device.type!='cpu':
            raise ValueError('Malformed RNG state')
        torch.Generator(device=device).set_state(value)


def make_model_factory(cfg,model,state):
    used=False
    def factory(actual_cfg,device):
        nonlocal used,state
        if used or fingerprint(actual_cfg)!=fingerprint(cfg):
            raise ValueError('Factory reused or called with changed config')
        used=True
        model.load_state_dict(state,strict=True)
        state=None
        model.to(device)
        parameter_groups(cfg,model)
        from baseline_train import set_seed
        set_seed(cfg.SEED)  # Resume loader restores saved RNG after construction.
        return model
    return factory


def write_record(cfg,name,record):
    path=Path(cfg.LOG_DIR)/name
    _safe_path(path,OUTPUT_ROOT)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,sort_keys=True,allow_nan=False)


def run_training(cfg,args):
    check_run_state(cfg,resume=args.resume)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Swin adaptation requires CUDA; no CPU fallback')
    from baseline_train import train_one,set_seed
    from models.swin_precoder import Swin_E2E_Precoder
    data=make_channel_data(cfg)
    channel_data_api().validate_channel_data(cfg,data)
    with run_lock(cfg,resume=args.resume,allowed_root=OUTPUT_ROOT):
        check_run_state(cfg,resume=args.resume)
        set_seed(cfg.SEED)
        model=Swin_E2E_Precoder(cfg)
        parameter_groups(cfg,model)
        record=dict(mode='resume' if args.resume else 'source_weights_new_optimizer',
            seed=cfg.SEED,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip(),
            code_sha256={p:sha256_file(REPO_ROOT/p) for p in SOURCE_FILES},
            manifest_sha256=sha256_file(DEFAULT_MANIFEST),config=cfg.to_dict(),config_fingerprint=fingerprint(cfg),
            actual_model_class=type(model).__name__,backend_before=_backend_info(torch),
            python=sys.version,platform=platform.platform(),screening_only=True,
            full_training_bitwise_resume_verified=False)
        if args.resume:
            payload=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            validate_runtime_resume(torch,model,payload,cfg)
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
            validate_best_metadata(best,payload,cfg)
            check_tensor_state(torch,model.state_dict(),best['state'])
            status=json.loads(Path(cfg.STATUS_PATH).read_text())
            if status['state']=='paused' and status.get('step')!=payload['step']:
                raise ValueError('Paused status/latest mismatch')
            record.update(start_step=payload['step']+1,resume_sha256=sha256_file(cfg.SAVE_PATH))
            state=payload['state']
            del best,payload
        else:
            path=cfg.source_identity['path'] if args.checkpoint is None else str((REPO_ROOT/args.checkpoint).resolve())
            with Path(path).open('rb') as stream:
                verify_checkpoint_hash(path,cfg.source_identity['checkpoint_sha256'],stream=stream)
                stream.seek(0)
                source=torch.load(stream,map_location='cpu',weights_only=False)
            verify_archived_metadata(dict(archive=cfg.source_identity,model_key='swin',method_key='swin'),source)
            check_tensor_state(torch,model.state_dict(),source['state'])
            state=source['state']
            record.update(start_step=1,source_path=path,source_sha256=cfg.source_identity['checkpoint_sha256'])
            del source
        factory=make_model_factory(cfg,model,state)
        del state
        if not args.resume:
            Path(cfg.SAVE_DIR).mkdir(parents=True,exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True,exist_ok=False)
        token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')+uuid.uuid4().hex[:8]
        write_record(cfg,f'invocation_{token}.json',record)
        try:
            train_one(cfg,stop_after=args.stop_after,no_resume=not args.resume,skip_existing=False,
                model_factory=factory,optimizer_scheduler_factory=optimizer_scheduler_factory,
                channel_data=data,downlink_snr_db=25.,initial_validation=True,
                resume_validator=lambda payload,cfg: validate_runtime_resume(torch,model,payload,cfg))
            latest=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            validate_resume_payload(latest,cfg,allow_completed=True)
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
            validate_best_metadata(best,latest,cfg)
            if latest['step']!=(args.stop_after or cfg.TOTAL_STEPS):
                raise RuntimeError('Unexpected completed update count')
            if not args.resume:
                verify_checkpoint_hash(record['source_path'],cfg.source_identity['checkpoint_sha256'])
            result=dict(completed_updates=latest['step'],best_step=best['step'],best_score=best['best_score'],
                latest_sha256=sha256_file(cfg.SAVE_PATH),best_sha256=sha256_file(cfg.BEST_PATH),
                backend_after=_backend_info(torch),screening_only=True)
            write_record(cfg,f'completion_{token}.json',result)
            print(json.dumps(result,indent=2))
        except Exception as error:
            write_record(cfg,f'failure_{token}.json',dict(error_type=type(error).__name__,error=str(error)))
            raise


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--train',action='store_true')
    mode.add_argument('--stop-after',type=int,metavar='ABSOLUTE_STEP')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path,help='Fresh only: explicitly relocated identical source bytes')
    return parser


def main(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    if not(args.dry_run or args.train or args.stop_after is not None):
        if args.resume or args.checkpoint is not None:parser.error('Choose explicit execution mode')
        parser.print_help();return 0
    if args.seed is None:parser.error('Choose --seed explicitly')
    if args.resume and (args.dry_run or args.checkpoint is not None):parser.error('Resume requires execution and forbids source override')
    if args.stop_after is not None and (not 0<args.stop_after<=20000 or args.stop_after%500):
        parser.error('stop-after is an absolute 500-update boundary in the fixed 20k plan')
    cfg=swin_adapt_config(args.seed);validate_config(cfg);validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps(dict(action='plan only; no weight reads/files/runtime imports',config=cfg.to_dict(),
            fingerprint=fingerprint(cfg),best=cfg.BEST_PATH,latest=cfg.SAVE_PATH,
            source_path=str(args.checkpoint) if args.checkpoint else cfg.source_identity['path'],
            scheduler_total_steps=20000,source_step_zero_competes=True),indent=2));return 0
    run_training(cfg,args);return 0


if __name__=='__main__':raise SystemExit(main())
