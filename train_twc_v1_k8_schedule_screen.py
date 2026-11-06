#!/usr/bin/env python3
"""Optional 3024-update K8 schedule screen; explicit CUDA execution only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import SimpleNamespace

from configs import twc_v1_k8_schedule_screen_config as cfg

ROOT = Path(__file__).resolve().parent
FORMAT = 'twc_v1_k8_schedule_screen_v1'
SOURCE_FILES = (
    'configs/twc_v1_k8_schedule_screen_config.py', 'train_twc_v1_k8_schedule_screen.py',
    'evaluate_twc_v1_k8_schedule_screen.py', 'run_twc_v1_k8_schedule_screen.py',
    'train_twc_v1_k8_codec_warmup.py', 'configs/twc_v1_k8_codec_warmup_config.py',
    'configs/twc_v1_k8_protocol_config.py', 'models/twc_v1_k8_neural.py',
    'models/twc_v1_k8_joint_decoder.py', 'models/swin_precoder.py', 'models/baseline_common.py',
    'utils/twc_v1_k8_system.py', 'utils/twc_v1_k8_protocol.py', 'utils/twc_v1_k8_link.py',
    'utils/twc_v1_k8_downlink.py', 'utils/twc_v1_k8_calibration.py',
    'utils/twc_v1_k8_replayable_data.py', 'utils/sionna_rng.py', 'utils/channel_generator.py',
    'run_as_receiver_batch.py', 'train_proposal_as_receiver.py',
)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def source_code_record():
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sha256={name: sha256(ROOT/name) for name in SOURCE_FILES})


def ce_identity():
    base = ROOT/'runs/twc_v1_k8_ce/seed_20261105'
    result={name: dict(path=str(base/filename), sha256=sha256(base/filename))
            for name, filename in (('checkpoint', 'checkpoint_latest.pt'), ('calibration', 'calibration.csv'))}
    if any(result[name]['sha256']!=cfg.CE_SHA256[name] for name in result):
        raise ValueError('Pinned CE checkpoint/calibration SHA256 mismatch')
    return result


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def runtime():
    import torch
    import numpy as np
    import train_twc_v1_k8_codec_warmup as warmup
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; no CPU fallback')
    return torch, np, warmup


def make_data(warmup, device):
    sizes = {role: cfg.BATCH_SIZE for role in ('ce_train','codec_train','ce_calibration','validation')}
    sizes['test'] = 1
    old = warmup.cfg
    return warmup.TWCv1K8ReplayableData(base_seed=old.BASE_DATA_SEED, batch_sizes=sizes,
        device=str(device), cdl_model=old.CDL_MODEL, speed=old.SPEED_MPS, seed_namespace=old.SEED_NAMESPACE)


def check_tensor_state(torch, model, state):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected): raise ValueError('Exact complete model keys required')
    for name, value in state.items():
        if (not torch.is_tensor(value) or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not bool(torch.isfinite(value).all())): raise ValueError('Model tensor mismatch: '+name)


def tensor_digest(torch, values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        item = tensor.detach().contiguous().cpu()
        digest.update(name.encode()); digest.update(str((tuple(item.shape),item.dtype)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def load_source(torch, warmup, arm, data_protocol):
    identity = cfg.source_identity(arm); path = Path(identity['path'])
    if sha256(path) != identity['sha256']: raise ValueError('Pinned source checkpoint SHA256 mismatch: '+str(path))
    payload = torch.load(path, map_location='cpu', weights_only=False)
    for key, expected in dict(format='twc_v1_k8_codec_warmup_v1', arm=identity['arm'],
                              seed=cfg.SEED, step=cfg.SOURCE_UPDATES, ce_checkpoint_step=5000).items():
        if payload.get(key) != expected: raise ValueError('Source checkpoint '+key+' mismatch')
    if canonical(payload.get('config')) != canonical(warmup.RECIPE): raise ValueError('Complete source recipe mismatch')
    if canonical(payload.get('data_protocol')) != canonical(data_protocol): raise ValueError('Source data protocol mismatch')
    if not all(k in payload for k in ('codec_model','optimizer','torch_cpu_rng','torch_cuda_rng')):
        raise ValueError('Source checkpoint is incomplete')
    manifest = json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
    for key, expected in dict(status='completed', arm=identity['arm'], seed=cfg.SEED,
                              optimizer_updates=cfg.SOURCE_UPDATES).items():
        if manifest.get(key) != expected: raise ValueError('Source manifest '+key+' mismatch')
    if canonical(manifest.get('config')) != canonical(warmup.RECIPE): raise ValueError('Source manifest recipe mismatch')
    if canonical(manifest.get('data_protocol')) != canonical(data_protocol): raise ValueError('Source manifest data identity mismatch')
    if sha256(path) != identity['sha256']: raise ValueError('Source changed while loading')
    return payload


def verify_completed_payload(torch, payload, arm):
    cfg.validate_resume_metadata(payload, arm, code_sha256=source_code_record()['sha256'])
    if payload['step'] != cfg.UPDATES: raise ValueError('Final3024 checkpoint required')
    _check_numerical_payload(torch, payload)


def _check_numerical_payload(torch, payload):
    import numpy as np
    step = payload['step']; group = payload['optimizer'].get('param_groups', [])
    if len(group) != 1: raise ValueError('Exactly one AdamW parameter group required')
    for name, value in dict(lr=cfg.LEARNING_RATE, initial_lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY, eps=1e-8, betas=(.9,.999)).items():
        if canonical(group[0].get(name)) != canonical(value): raise ValueError('Optimizer '+name+' mismatch')
    scheduler = payload['scheduler']
    if (scheduler.get('last_epoch') != step or scheduler.get('_step_count') != step+1
            or scheduler.get('base_lrs') != [cfg.LEARNING_RATE]
            or scheduler.get('_last_lr') != [cfg.LEARNING_RATE]): raise ValueError('Constant scheduler/local step mismatch')
    moments = payload['optimizer'].get('state', {})
    if not isinstance(moments,dict) or bool(moments) != (step>0): raise ValueError('AdamW state/local step mismatch')
    for record in moments.values():
        if set(record) != {'step','exp_avg','exp_avg_sq'}: raise ValueError('Incomplete optimizer moment state')
        if not torch.is_tensor(record['step']) or record['step'].numel()!=1 or record['step'].item()!=step:
            raise ValueError('AdamW update count mismatch')
        for name in ('exp_avg','exp_avg_sq'):
            if not torch.is_tensor(record[name]) or not bool(torch.isfinite(record[name]).all()):
                raise ValueError('Nonfinite optimizer moments')
        if not bool((record['exp_avg_sq']>=0).all()): raise ValueError('Negative second moment')
    for value in [payload['rng']['torch_cpu'],*payload['rng']['torch_cuda']]:
        if not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1 or value.device.type!='cpu':
            raise ValueError('CPU RNG byte tensors required')
    if len(payload['rng']['torch_cuda'])!=torch.cuda.device_count(): raise ValueError('Visible CUDA device count changed')
    random.Random().setstate(payload['rng']['python'])
    np.random.RandomState().set_state(payload['rng']['numpy'])


def validate_runtime_resume(torch, model, payload, arm):
    cfg.validate_resume_metadata(payload,arm,code_sha256=source_code_record()['sha256'])
    _check_numerical_payload(torch,payload); check_tensor_state(torch,model,payload['codec_model'])
    parameters=list(model.parameters()); group=payload['optimizer']['param_groups'][0]
    if group['params']!=list(range(len(parameters))): raise ValueError('Optimizer parameter order changed')
    if not set(payload['optimizer']['state'])<=set(group['params']): raise ValueError('Unknown optimizer parameter')
    for index, state in payload['optimizer']['state'].items():
        for name in ('exp_avg','exp_avg_sq'):
            if state[name].shape!=parameters[index].shape or state[name].dtype!=parameters[index].dtype:
                raise ValueError('Optimizer moment shape/dtype mismatch')
    model.load_state_dict(payload['codec_model'],strict=True)


def capture_rng(torch,np):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(torch,np,record):
    random.setstate(record['python']);np.random.set_state(record['numpy'])
    torch.set_rng_state(record['torch_cpu']);torch.cuda.set_rng_state_all(record['torch_cuda'])


def training_manifest(arm,start,*,data_protocol,ce,code,resume_hash=None):
    if type(start) is not int or not 0<=start<cfg.UPDATES or start%cfg.SAVE_EVERY:
        raise ValueError('A saved incomplete local checkpoint boundary is required')
    if start and (not isinstance(resume_hash,str) or len(resume_hash)!=64
            or any(c not in '0123456789abcdef' for c in resume_hash)):
        raise ValueError('Resume must retain the verified latest checkpoint hash')
    value=dict(format=FORMAT,arm=arm,seed=cfg.SEED,status='running',step=start,target=cfg.UPDATES,
        cumulative_updates=cfg.SOURCE_UPDATES+start,recipe=cfg.recipe(arm),config_fingerprint=cfg.fingerprint(arm),
        source=cfg.source_identity(arm),ce_identity=ce,code=code,data_protocol=data_protocol,
        updated_utc=datetime.now(timezone.utc).isoformat(),test_used=False,checkpoint_selection='final3024 only')
    if resume_hash is not None:value['checkpoint_sha256']=resume_hash
    return value


def source_checks():
    torch,np,warmup=runtime(); device=torch.device('cuda:0'); data=make_data(warmup,device)
    ce=ce_identity();data.prepare();chain=warmup.build_chain(device);checked={}
    for arm in cfg.ARMS:
        payload=load_source(torch,warmup,arm,data.protocol)
        check_tensor_state(torch,chain.codec_model,payload['codec_model'])
        chain.codec_model.load_state_dict(payload['codec_model'],strict=True)
        checked[arm]=cfg.source_identity(arm);del payload
    return dict(status='passed',sources=checked,ce_identity=ce,code=source_code_record(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device),
        data_protocol=data.protocol,source_optimizer_reused=False,
        backend=dict(matmul_precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))


def _run(chain, warmup, Hdl, Hul, condition, pilot_noise, data_noise):
    if condition['energy_ratio'] is None:
        layout,split=warmup.protocol.FDMALayout(),None
    else:
        layout=warmup.protocol.default_hybrid_layout(condition['group_count'])
        available=64-condition['pilot_energy_per_ue'];r=condition['energy_ratio']
        split=warmup.protocol.EnergySplit(condition['pilot_energy_per_ue'],available*r[0]/sum(r),available*r[1]/sum(r))
    return chain(Hdl,Hul,feedback_gamma_db=condition['gamma_db'],
        pilot_energy_per_ue=condition['pilot_energy_per_ue'],pilot_standard_noise=pilot_noise,
        data_standard_noise=data_noise,layout=layout,energy_split=split)


def progress_validation(torch,warmup,chain,data,arm):
    """Small G-specific gamma10 monitor, never checkpoint selection."""
    chain.codec_model.eval();records=[]
    groups=(1,) if arm=='fdma_control' else (4,) if arm=='specialist_g4' else (1,2,4)
    with torch.no_grad():
        for bi in range(4):
            sample=data.sample('validation',bi);Hdl,Hul=sample['H_dl'],sample['H_ul']
            gen=torch.Generator(device=Hdl.device).manual_seed(810000000+bi)
            pilot=warmup.standard_noise(4,gen,Hdl.device);noise=warmup.standard_noise(4,gen,Hdl.device)
            for g in groups:
                cond=dict(gamma_db=10.,pilot_energy_per_ue=16.,group_count=g,
                    energy_ratio=None if arm=='fdma_control' else [1,3])
                output=_run(chain,warmup,Hdl,Hul,cond,pilot,noise);metrics=chain.metrics(output,Hdl)
                records.append(dict(batch=bi,group_count=g,nmse=float(metrics['dl_nmse_per_user'].mean()),
                    rate=float(metrics['sum_rate_bps_hz'].mean())))
    chain.codec_model.train();return records


def train(arm, *, resume=False, stop_after=None):
    from train_proposal_as_receiver import run_lock
    torch,np,warmup=runtime();device=torch.device('cuda:0');folder=cfg.OUTPUT_ROOT/'training'/arm
    owner=SimpleNamespace(EXP_NAME='k8_schedule_'+arm,arm=arm,SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        if not resume and folder.exists():raise FileExistsError('Training output exists: '+str(folder))
        if resume and not (folder/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Explicit resume requires latest')
        random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED);torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark=False
        data=make_data(warmup,device);ce=ce_identity();code=source_code_record()
        source=load_source(torch,warmup,arm,data.protocol);chain=warmup.build_chain(device);model=chain.codec_model
        check_tensor_state(torch,model,source['codec_model']);model.load_state_dict(source['codec_model'],strict=True);del source
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,weight_decay=cfg.WEIGHT_DECAY)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        start=0
        resume_hash=None
        if resume:
            saved_manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if sha256(folder/'checkpoint_latest.pt')!=saved_manifest.get('checkpoint_sha256'):
                raise ValueError('Latest checkpoint bytes differ from saved receipt')
            resume_hash=saved_manifest['checkpoint_sha256']
            initial=json.loads((folder/'initialization.json').read_text(encoding='utf-8'))
            for key,value in dict(source=cfg.source_identity(arm),ce_identity=ce,code=code,recipe=cfg.recipe(arm)).items():
                if canonical(initial.get(key))!=canonical(value):raise ValueError('Initialization '+key+' changed')
            payload=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            validate_runtime_resume(torch,model,payload,arm)
            if payload['ce_identity']!=ce or canonical(payload['data_protocol'])!=canonical(data.protocol):
                raise ValueError('Resume CE/calibration/data identity changed')
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
            start=payload['step'];restore_rng(torch,np,payload['rng']);del payload
        folder.mkdir(parents=True,exist_ok=True)
        if not resume:
            save_json(folder/'initialization.json',dict(source=cfg.source_identity(arm),ce_identity=ce,code=code,
                recipe=cfg.recipe(arm),model_state_sha256=tensor_digest(torch,model.state_dict()),
                source_optimizer_reused=False,source_step=10000,local_step=0))
        end=min(cfg.UPDATES,start+stop_after) if stop_after else cfg.UPDATES
        if not start<end:raise ValueError('No new updates planned; completed arms are not retrained')
        manifest=training_manifest(arm,start,data_protocol=data.protocol,ce=ce,code=code,resume_hash=resume_hash)
        save_json(folder/'manifest.json',manifest)
        ce_before=tensor_digest(torch,chain.ce_model.state_dict());model.train();torch.cuda.reset_peak_memory_stats(device)
        try:
            for step in range(start+1,end+1):
                cond=cfg.schedule(arm,step);sample=data.sample('codec_train',cond['global_step'])
                Hdl,Hul=sample['H_dl'],sample['H_ul']
                gen=torch.Generator(device=device).manual_seed(cfg.SEED*10000000+cond['global_step'])
                output=_run(chain,warmup,Hdl,Hul,cond,warmup.standard_noise(4,gen,device),warmup.standard_noise(4,gen,device))
                metrics=chain.metrics(output,Hdl);loss=metrics['dl_nmse_per_user'].mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite NMSE loss')
                if bool((output['energy']['actual_data_per_ue']+cond['pilot_energy_per_ue']-64).abs().max()>1e-4):
                    raise ValueError('Per-UE frame energy changed')
                if bool((metrics['rzf_per_subcarrier_power']-1).abs().max()>1e-5):raise ValueError('RZF power changed')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.GRAD_CLIP_NORM,error_if_nonfinite=True)
                if any(p.grad is not None for p in chain.ce_model.parameters()) or chain.ce_model.training:
                    raise ValueError('CE frozen/eval contract changed')
                optimizer.step();scheduler.step()
                if step%108==0 or step==end:
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite parameter')
                    if tensor_digest(torch,chain.ce_model.state_dict())!=ce_before:raise ValueError('CE state changed')
                    rng_before_monitor=capture_rng(torch,np)
                    monitor=progress_validation(torch,warmup,chain,data,arm) if step%756==0 or step==end else None
                    restore_rng(torch,np,rng_before_monitor)
                    payload=dict(format=FORMAT,arm=arm,seed=cfg.SEED,step=step,cumulative_updates=cfg.SOURCE_UPDATES+step,
                        codec_model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        rng=capture_rng(torch,np),recipe=cfg.recipe(arm),config_fingerprint=cfg.fingerprint(arm),
                        source=cfg.source_identity(arm),code_sha256=code['sha256'],ce_identity=ce,data_protocol=data.protocol)
                    path=folder/'checkpoint_latest.pt';temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(path)
                    if ((step==end and sha256(Path(cfg.source_identity(arm)['path']))!=cfg.source_identity(arm)['sha256'])
                            or ce_identity()!=ce):
                        raise ValueError('Frozen input file changed')
                    manifest.update(step=step,cumulative_updates=cfg.SOURCE_UPDATES+step,
                        status='completed' if step==cfg.UPDATES else 'paused' if step==end else 'running',
                        checkpoint_sha256=sha256(path),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        updated_utc=datetime.now(timezone.utc).isoformat(),last_loss=float(loss.detach()),last_grad_norm=float(grad),
                        last_condition=cond,parameters=sum(p.numel() for p in model.parameters()))
                    save_json(folder/'manifest.json',manifest)
                    row=dict(step=step,cumulative_updates=cfg.SOURCE_UPDATES+step,loss=float(loss.detach()),
                        grad_norm=float(grad),condition=cond,monitor=monitor,utc=manifest['updated_utc'])
                    with (folder/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    print(json.dumps(row),flush=True)
        except BaseException as error:
            manifest.update(status='failed',error=f'{type(error).__name__}: {error}',updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(folder/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'),manifest)
            save_json(folder/'manifest.json',manifest);raise
        return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=cfg.ARMS)
    modes=parser.add_mutually_exclusive_group()
    for name in ('dry-run','train','check-sources'):modes.add_argument('--'+name,action='store_true')
    modes.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.report is not None and not args.check_sources:parser.error('--report requires --check-sources')
    if args.check_sources:
        report=cfg.inside_root(args.report) if args.report is not None else None
        if report is not None and report.exists():raise FileExistsError(report)
        try:
            result=source_checks()
        except BaseException as error:
            if report is not None:save_json(report,dict(status='failed',error=f'{type(error).__name__}: {error}'))
            raise
        if report is not None:save_json(report,result)
        print(json.dumps(result,indent=2));return 0
    if not(args.dry_run or args.train or args.stop_after is not None):parser.print_help();return 0
    if args.arm is None:parser.error('Explicit --arm required')
    if args.resume and args.dry_run:parser.error('Dry-run does not resume')
    if args.stop_after is not None and (args.stop_after<108 or args.stop_after%108 or args.stop_after>cfg.UPDATES):
        parser.error('Pause updates must be108-step multiples within3024')
    if args.dry_run:print(json.dumps(dict(recipe=cfg.recipe(args.arm),conditions=cfg.schedule_counts(args.arm)),indent=2));return 0
    train(args.arm,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':raise SystemExit(main())
