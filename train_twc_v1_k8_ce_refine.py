#!/usr/bin/env python3
"""CE-only continuation with strict source, resumable AdamW and immutable snapshots."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time
import traceback
import os
from configs import twc_v1_k8_ce_refine_config as cfg
import ce_refine_runtime as runtime

now, sha256, read_json = runtime.now, runtime.sha256, runtime.read_json
state_digest, capture_rng, restore_rng = runtime.state_digest, runtime.capture_rng, runtime.restore_rng


def schema(state):
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()}


def _directory(trajectory):
    return runtime.inside(cfg.OUTPUT_ROOT/'training'/cfg.key(trajectory))


def optimizer_check(model, optimizer, scheduler, step):
    import torch
    if (scheduler.last_epoch!=step or scheduler._step_count!=step+1
            or scheduler.T_max!=cfg.UPDATES or scheduler.eta_min!=cfg.MIN_LR
            or scheduler.base_lrs!=[cfg.LR]*len(optimizer.param_groups)):
        raise ValueError('CE cosine scheduler boundary changed')
    expected_lr=cfg.learning_rate(step)
    if (step==0 and optimizer.state or len(scheduler._last_lr)!=len(optimizer.param_groups)
            or any(not math.isclose(value,expected_lr,rel_tol=2e-10,abs_tol=1e-14)
                for value in scheduler._last_lr)):
        raise ValueError('CE optimizer initialization/cosine LR changed')
    for group in optimizer.param_groups:
        if (not math.isclose(group['lr'],expected_lr,rel_tol=2e-10,abs_tol=1e-14)
                or group['weight_decay']!=cfg.WEIGHT_DECAY or group['betas']!=(.9,.999)
                or group['eps']!=1e-8 or group.get('initial_lr')!=cfg.LR):
            raise ValueError('CE optimizer recipe changed')
        for parameter in group['params']:
            state=optimizer.state.get(parameter,{})
            if step and (not state or int(state['step'])!=step):
                raise ValueError('Incomplete CE AdamW boundary')
            for key in ('exp_avg','exp_avg_sq'):
                if step and (state[key].shape!=parameter.shape or state[key].dtype!=parameter.dtype
                        or not bool(torch.isfinite(state[key]).all())):
                    raise ValueError('Invalid CE AdamW moments')
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    if ids!=[id(p) for p in model.parameters()]:
        raise ValueError('Only the entire CE model may be optimized')


def commit_update(model,optimizer,scheduler,step):
    import torch
    optimizer_check(model,optimizer,scheduler,step)
    params=list(model.parameters())
    if not params or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise FloatingPointError('Missing/nonfinite CE gradient; update refused')
    norm=torch.nn.utils.clip_grad_norm_(params,cfg.GRAD_CLIP,error_if_nonfinite=True)
    if not bool(norm>0):raise FloatingPointError('Zero CE gradient; update refused')
    optimizer.step();scheduler.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise FloatingPointError('Nonfinite CE parameters; checkpoint refused')
    optimizer_check(model,optimizer,scheduler,step+1)
    return norm


def initial_trace(protocol):
    return dict(microbatches=0,first_micro_index=None,last_micro_index=0,
        sha256=cfg.digest(dict(namespace=cfg.NAMESPACE,seed=cfg.SEED,protocol=protocol)))


def advance_trace(trace,entry):
    if entry['micro_index']!=trace['last_micro_index']+1:
        raise ValueError('CE physical input trace is not contiguous')
    return dict(microbatches=trace['microbatches']+1,
        first_micro_index=1,last_micro_index=entry['micro_index'],
        sha256=cfg.digest(dict(previous=trace['sha256'],entry=entry)))


def _snapshot(directory,latest,record):
    step=record['step']
    if step and step%cfg.MONITOR_EVERY:return
    target=directory/f'checkpoint_step_{step:05d}.pt'
    receipt=directory/f'checkpoint_step_{step:05d}.json'
    if target.exists():
        if sha256(runtime.regular(target))!=record['checkpoint_sha256']:
            raise ValueError('Immutable CE learning-curve snapshot changed')
    else:
        fd,name=tempfile.mkstemp(prefix='checkpoint_',suffix='.pt',dir=directory)
        try:
            with os.fdopen(fd,'wb') as sink, latest.open('rb') as source:
                shutil.copyfileobj(source,sink);sink.flush();os.fsync(sink.fileno())
            os.link(name,target)
        finally:
            if os.path.exists(name):os.unlink(name)
    if receipt.exists():
        old=read_json(receipt)
        if any(old.get(k)!=record.get(k) for k in ('step','checkpoint_sha256','core','training_trace','source_identity')):
            raise ValueError('Immutable CE snapshot receipt changed')
    else:runtime.atomic_json(receipt,record,exclusive=True)


def save_checkpoint(directory,model,optimizer,scheduler,*,trajectory,step,data,core,source,
                    trace,fixture,measured=0.,elapsed=0.):
    import torch
    runtime.guard_core(core);runtime.verify_source(source)
    optimizer_check(model,optimizer,scheduler,step)
    state=model.state_dict()
    payload=dict(format=cfg.NAMESPACE,trajectory=trajectory,step=step,recipe=cfg.recipe(trajectory),
        fingerprint=cfg.fingerprint(trajectory),core=core,training_code=runtime.source_identity(),
        source_identity=source,model=state,model_schema=schema(state),
        parameter_schema={k:dict(shape=list(p.shape),dtype=str(p.dtype)) for k,p in model.named_parameters()},
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(),
        counters=cfg.counters(step),data_protocol=data.protocol,training_trace=trace,
        fixture=bool(fixture),production_evidence=not fixture)
    latest=directory/'checkpoint_latest.pt'
    runtime.original.atomic_torch_save(payload,latest)
    record=dict(status='completed' if step==cfg.UPDATES else 'paused',trajectory=trajectory,step=step,
        counters=payload['counters'],checkpoint_sha256=sha256(latest),model_state_sha256=state_digest(model),
        core=core,training_code=payload['training_code'],source_identity=source,
        fingerprint=payload['fingerprint'],recipe=payload['recipe'],training_trace=trace,
        data_protocol=data.protocol,fixture=bool(fixture),production_evidence=not fixture,updated=now(),
        seconds_per_update=measured,ETA_seconds=measured*(cfg.UPDATES-step),elapsed_seconds=elapsed)
    runtime.atomic_json(directory/'manifest.json',record)
    _snapshot(directory,latest,record)
    if step==cfg.SAVE_EVERY:
        path=directory/'boundary_00100.json'
        if not path.exists():runtime.atomic_json(path,record,exclusive=True)
    if step==cfg.UPDATES:
        path=directory/'terminal_receipt.json'
        if path.exists() and read_json(path)!=record:raise ValueError('Terminal CE receipt already differs')
        if not path.exists():runtime.atomic_json(path,record,exclusive=True)
    return record


def verify_saved(directory,payload,record,*,trajectory,data,core,source,allow_fixture=False):
    import torch
    cfg.key(trajectory)
    step=payload.get('step')
    if type(step)is not int or not 0<=step<=cfg.UPDATES or step%cfg.SAVE_EVERY:
        raise ValueError('Complete CE saved boundary required')
    expected=dict(format=cfg.NAMESPACE,trajectory=trajectory,recipe=cfg.recipe(trajectory),
        fingerprint=cfg.fingerprint(trajectory),core=core,source_identity=source,
        counters=cfg.counters(step),data_protocol=data.protocol)
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('CE resume identity changed')
    fixture=payload.get('fixture')
    if (type(fixture)is not bool or fixture and not allow_fixture
            or payload.get('production_evidence') is not (not fixture)
            or record.get('production_evidence') is not (not fixture)):
        raise ValueError('CPU fixture is not production CE evidence')
    if (record.get('checkpoint_sha256')!=sha256(runtime.regular(directory/'checkpoint_latest.pt'))
            or any(record.get(k)!=payload.get(k) for k in ('trajectory','step','core','source_identity',
                'fingerprint','recipe','counters','training_trace','data_protocol','fixture','production_evidence'))):
        raise ValueError('CE checkpoint/manifest identity changed')
    if set(payload.get('rng',{}))!={'python','numpy','torch_cpu','torch_cuda'}:
        raise ValueError('Complete CE RNG state required')
    state=payload.get('model')
    if (not isinstance(state,dict) or not state or schema(state)!=payload.get('model_schema')
            or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in state.values())
            or state_digest(state)!=record.get('model_state_sha256')):
        raise ValueError('Finite complete CE model state required')
    trace=payload.get('training_trace',{})
    if (trace.get('microbatches')!=step*cfg.ACCUMULATION or trace.get('last_micro_index')!=step*cfg.ACCUMULATION
            or trace.get('first_micro_index')!=(1 if step else None)
            or not isinstance(trace.get('sha256'),str) or len(trace['sha256'])!=64):
        raise ValueError('CE physical training trace boundary changed')
    initial=read_json(runtime.regular(directory/'initialization.json'))
    if any(initial.get(k)!=payload.get(k) for k in ('trajectory','core','source_identity','recipe',
            'fingerprint','data_protocol','fixture','production_evidence','model_schema','parameter_schema')):
        raise ValueError('CE initialization identity changed')
    if initial.get('source_state_sha256')!=source['source_state_sha256'] or initial.get('optimizer_continued') is not False:
        raise ValueError('CE source state/fresh optimizer provenance changed')
    return step


def _monitor(model,data,trajectory,step,directory,monitor,*,fixture=False):
    if step%cfg.MONITOR_EVERY:return
    target=directory/'monitor'/f'step_{step:05d}'
    receipt=target/'manifest.json'
    if receipt.exists():
        from evaluate_twc_v1_k8_ce_refine import verify_monitor
        value=verify_monitor(target,trajectory=trajectory,step=step,production_required=not fixture)
        if value['model_state_sha256']!=state_digest(model):
            raise ValueError('Completed CE monitor belongs to a different model')
        return
    if monitor is None:
        from evaluate_twc_v1_k8_ce_refine import monitor_model
        monitor=monitor_model
    rng=capture_rng();modes={k:m.training for k,m in model.named_modules()};before=state_digest(model)
    try:
        model.eval()
        monitor(model,data,trajectory,step,target)
    finally:
        for name,module in model.named_modules():module.training=modes[name]
        restore_rng(rng)
    if state_digest(model)!=before:raise ValueError('CE monitor changed model state')


def train(trajectory,*,resume=False,stop_after=None,require_cuda=True,loader=None,
          data_factory=None,forward=None,monitor=None):
    import torch
    import numpy as np
    import twc_v1_k8_ce_refine as objective_math
    cfg.key(trajectory)
    target=cfg.UPDATES if stop_after is None else stop_after
    if type(target)is not int or not cfg.SAVE_EVERY<=target<=cfg.UPDATES or target%cfg.SAVE_EVERY:
        raise ValueError('Complete CE save boundary target required')
    device=runtime.configure_cuda() if require_cuda else torch.device('cpu')
    core=runtime.core_identity();runtime.guard_core(core)
    random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED)
    if device.type=='cuda':torch.cuda.manual_seed_all(cfg.SEED)
    fixture=not require_cuda or any(x is not None for x in (loader,data_factory,forward,monitor))
    loader=runtime.load_source if loader is None else loader
    data_factory=runtime.make_data if data_factory is None else data_factory
    forward=runtime.pilot_forward if forward is None else forward
    model,source=loader(device);runtime.verify_source(source)
    if bool(source.get('fixture',False))!=fixture:raise ValueError('CE production/fixture source mismatch')
    if state_digest(model)!=source['source_state_sha256']:raise ValueError('Initial CE source state changed')
    model.train().requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY,
        betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.UPDATES,eta_min=cfg.MIN_LR)
    data=data_factory(device);directory=_directory(trajectory)
    if not resume:directory.mkdir(parents=True,exist_ok=False)
    elif not (directory/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Own CE checkpoint required for resume')
    start=0;trace=initial_trace(data.protocol)
    with runtime.process_lock(directory/'training.lock'):
        if resume:
            payload=torch.load(runtime.regular(directory/'checkpoint_latest.pt'),map_location='cpu',weights_only=False)
            record=read_json(runtime.regular(directory/'manifest.json'))
            start=verify_saved(directory,payload,record,trajectory=trajectory,data=data,core=core,
                source=source,allow_fixture=fixture)
            if target<start:raise ValueError('CE target cannot rewind saved progress')
            if (schema(model.state_dict())!=payload['model_schema']
                    or payload['parameter_schema']!={k:dict(shape=list(p.shape),dtype=str(p.dtype))
                        for k,p in model.named_parameters()}):
                raise ValueError('CE architecture changed on resume')
            runtime.strict_load(model,payload['model'])
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
            optimizer_check(model,optimizer,scheduler,start);restore_rng(payload['rng'])
            trace=payload['training_trace']
        else:
            runtime.atomic_json(directory/'initialization.json',dict(trajectory=trajectory,core=core,
                source_identity=source,recipe=cfg.recipe(trajectory),fingerprint=cfg.fingerprint(trajectory),
                data_protocol=data.protocol,fixture=fixture,production_evidence=not fixture,source_state_sha256=state_digest(model),
                optimizer_continued=False,model_schema=schema(model.state_dict()),
                parameter_schema={k:dict(shape=list(p.shape),dtype=str(p.dtype)) for k,p in model.named_parameters()},
                training_trace=trace,created=now()),exclusive=True)
            save_checkpoint(directory,model,optimizer,scheduler,trajectory=trajectory,step=0,data=data,
                core=core,source=source,trace=trace,fixture=fixture)
        try:
            _monitor(model,data,trajectory,start,directory,monitor,fixture=fixture)
            if start==target:return read_json(directory/'manifest.json')
            started=time.perf_counter();timings=[]
            for step in range(start+1,target+1):
                begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                logs=[];entries=[];before_trace=dict(trace)
                for micro in range(cfg.ACCUMULATION):
                    index=cfg.micro_index(step,micro);sample=data.sample('ce_train',index)
                    if sample.get('address')!=data.address('ce_train',index):
                        raise ValueError('CE physical data address changed')
                    gamma,noise,polarity=runtime.batch_random('ce_train',index,device)
                    pred,initial,Y,pilots=forward(model,sample,gamma,noise,polarity)
                    loss,metrics=objective_math.objective(pred,sample['H_ul'],trajectory)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite CE loss')
                    (loss/cfg.ACCUMULATION).backward()
                    logs.append(dict(loss=float(loss.detach()),**{
                        k:float(v.detach().mean()) for k,v in metrics.items()}))
                    entry=dict(micro_index=index,address=sample['address'],channel=sample['sample_fingerprint'],
                        gamma=state_digest({'gamma':gamma}),noise=state_digest({'noise':noise}),
                        polarity=state_digest({'polarity':polarity}),Y=state_digest({'Y':Y}),
                        pilots=state_digest({'pilots':pilots}))
                    trace=advance_trace(trace,entry);entries.append(entry)
                norm=commit_update(model,optimizer,scheduler,step-1)
                if device.type=='cuda':torch.cuda.synchronize()
                timings.append(time.perf_counter()-begin)
                if step%cfg.SAVE_EVERY==0:
                    measured=sum(timings[-cfg.SAVE_EVERY:])/len(timings[-cfg.SAVE_EVERY:])
                    row=dict(step=step,trajectory=trajectory,counters=cfg.counters(step),
                        **{k:sum(v[k] for v in logs)/len(logs) for k in logs[0]},
                        gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],
                        input_fingerprints=entries,training_trace_before_update=before_trace,
                        training_trace=trace,mean_update_seconds=measured,
                        ETA_seconds=measured*(cfg.UPDATES-step),updated=now())
                    with (directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                    save_checkpoint(directory,model,optimizer,scheduler,trajectory=trajectory,step=step,
                        data=data,core=core,source=source,trace=trace,fixture=fixture,
                        measured=measured,elapsed=time.perf_counter()-started)
                    print(f'{trajectory} CE +{step}/{cfg.UPDATES} NMSE={row["ul_nmse_linear"]:.6g} '
                        f'ETA={row["ETA_seconds"]:.0f}s',flush=True)
                _monitor(model,data,trajectory,step,directory,monitor,fixture=fixture)
            return read_json(directory/'manifest.json')
        except BaseException as exc:
            runtime.atomic_json(directory/'failure.json',dict(trajectory=trajectory,status='failed',
                attempted_step=locals().get('step',start),error=f'{type(exc).__name__}: {exc}',
                traceback=traceback.format_exc(),updated=now()))
            raise


def load_terminal(trajectory,device='cpu'):
    import torch
    directory=_directory(trajectory)
    model,source=runtime.load_source(device)
    payload=torch.load(runtime.regular(directory/'checkpoint_latest.pt'),map_location='cpu',weights_only=False)
    record=read_json(runtime.regular(directory/'manifest.json'))
    step=verify_saved(directory,payload,record,trajectory=trajectory,data=runtime.make_data('cuda:0'),
        core=runtime.core_identity(),source=source,allow_fixture=False)
    if (step!=cfg.UPDATES or record.get('status')!='completed'
            or read_json(runtime.regular(directory/'terminal_receipt.json'))!=record):
        raise ValueError('Fixed terminal CE evidence required')
    expected_parameters={k:dict(shape=list(p.shape),dtype=str(p.dtype)) for k,p in model.named_parameters()}
    if payload['parameter_schema']!=expected_parameters or payload['model_schema']!=schema(model.state_dict()):
        raise ValueError('Terminal CE parameter architecture changed')
    runtime.strict_load(model,payload['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY,
        betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.UPDATES,eta_min=cfg.MIN_LR)
    optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
    optimizer_check(model,optimizer,scheduler,step)
    if state_digest(model)!=record['model_state_sha256']:raise ValueError('Terminal CE state changed')
    from evaluate_twc_v1_k8_ce_refine import verify_monitor
    for boundary in range(0,cfg.UPDATES+1,cfg.MONITOR_EVERY):
        snapshot=directory/f'checkpoint_step_{boundary:05d}.pt'
        snapshot_record=read_json(runtime.regular(directory/f'checkpoint_step_{boundary:05d}.json'))
        if (sha256(runtime.regular(snapshot))!=snapshot_record.get('checkpoint_sha256')
                or snapshot_record.get('step')!=boundary or snapshot_record.get('core')!=record['core']
                or snapshot_record.get('recipe')!=cfg.recipe(trajectory)
                or snapshot_record.get('source_identity')!=source
                or snapshot_record.get('production_evidence') is not True):
            raise ValueError('Immutable CE curve snapshot identity changed')
        monitor=verify_monitor(directory/'monitor'/f'step_{boundary:05d}',
            trajectory=trajectory,step=boundary,production_required=True)
        if monitor['model_state_sha256']!=snapshot_record['model_state_sha256']:
            raise ValueError('CE learning curve is not bound to its snapshot')
    model.eval().requires_grad_(False)
    return model,dict(record,path=str(directory/'checkpoint_latest.pt'),frozen_files=[
        dict(path=str(directory/name),sha256=sha256(directory/name)) for name in
        ('checkpoint_latest.pt','manifest.json','terminal_receipt.json','initialization.json')])


def verify_terminal(trajectory):return load_terminal(trajectory,'cpu')[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    act=p.add_mutually_exclusive_group(required=True)
    act.add_argument('--train',action='store_true');act.add_argument('--verify-terminal',action='store_true')
    p.add_argument('--trajectory',choices=cfg.TRAJECTORIES,required=True)
    p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if args.verify_terminal:
        if args.stop_after is not None or args.resume:p.error('Terminal verification does not train or resume')
        print(json.dumps(verify_terminal(args.trajectory),sort_keys=True,indent=2));return 0
    train(args.trajectory,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':raise SystemExit(main())
