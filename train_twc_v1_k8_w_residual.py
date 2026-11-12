#!/usr/bin/env python3
"""Strict BS-only continuation versus constrained residual-W rate training.

The CLI requires CUDA. The Python function accepts explicit CPU fixtures for
checkpoint and optimizer checks without loading Sionna or the full model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
import traceback
import uuid

from configs import twc_v1_k8_w_residual_config as cfg
import w_residual_runtime as runtime

original = runtime.rate.original
now, sha256, read_json = runtime.now, runtime.sha256, runtime.read_json
state_digest = runtime.state_digest
capture_rng, restore_rng = runtime.capture_rng, runtime.restore_rng
helpers = runtime.helpers
inside, atomic_json = runtime.inside, runtime.atomic_json


def _regular(path):
    path=inside(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular own residual-W evidence file required: '+str(path))
    return path


def _schema(state):
    return {name:dict(shape=list(value.shape),dtype=str(value.dtype))
            for name,value in state.items()}


def _parameter_schema(decoder):
    return [dict(name=name,shape=list(value.shape),dtype=str(value.dtype))
            for name,value in decoder.named_parameters()]


def optimizer_check(optimizer,scheduler,step):
    """Require the constant fresh-AdamW recipe and a complete update boundary."""
    import torch
    if scheduler.last_epoch!=step or scheduler._step_count!=step+1:
        raise ValueError('Residual-W scheduler boundary changed')
    if len(optimizer.param_groups)!=1:
        raise ValueError('One complete BS AdamW parameter group required')
    for group in optimizer.param_groups:
        if (group['lr']!=cfg.LR or group['weight_decay']!=cfg.WEIGHT_DECAY
                or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8
                or group.get('amsgrad',False) or group.get('maximize',False)
                or group.get('capturable',False) or group.get('differentiable',False)):
            raise ValueError('Residual-W optimizer recipe changed')
        for parameter in group['params']:
            state=optimizer.state.get(parameter,{})
            if step==0 and state:
                raise ValueError('Fresh BS optimizer must have no source moments')
            if step:
                if (not state or set(state)!={'step','exp_avg','exp_avg_sq'}
                        or not torch.is_tensor(state['step']) or state['step'].numel()!=1
                        or not bool(torch.isfinite(state['step']).all())
                        or float(state['step'])!=step):
                    raise ValueError('Incomplete BS AdamW boundary')
                for name in ('exp_avg','exp_avg_sq'):
                    value=state[name]
                    if (not torch.is_tensor(value) or value.shape!=parameter.shape
                            or value.dtype!=parameter.dtype or not bool(torch.isfinite(value).all())):
                        raise ValueError('Invalid BS AdamW moments')
                if bool((state['exp_avg_sq']<0).any()):
                    raise ValueError('Negative BS AdamW squared moments')


def commit_update(codec,optimizer,scheduler,*,previous_updates):
    import torch
    optimizer_check(optimizer,scheduler,previous_updates)
    parameters=tuple(codec.decoder.parameters())
    actual=tuple(p for group in optimizer.param_groups for p in group['params'])
    if tuple(map(id,parameters))!=tuple(map(id,actual)):
        raise ValueError('Optimizer must contain exactly the BS decoder parameters')
    if not parameters or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError('Missing/nonfinite BS gradient; update refused')
    norm=torch.nn.utils.clip_grad_norm_(parameters,cfg.GRAD_CLIP,error_if_nonfinite=True)
    if not bool(norm>0):raise ValueError('Zero BS rate gradient; update refused')
    optimizer.step();scheduler.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError('Nonfinite BS parameter after AdamW')
    optimizer_check(optimizer,scheduler,previous_updates+1)
    return norm


def _validate_saved_optimizer(payload,step):
    import torch
    optimizer=payload['optimizer'];scheduler=payload['scheduler']
    if (not isinstance(optimizer,dict) or set(optimizer)!={'state','param_groups'}
            or not isinstance(optimizer['state'],dict)
            or not isinstance(optimizer['param_groups'],list) or len(optimizer['param_groups'])!=1):
        raise ValueError('Complete fresh BS optimizer state required')
    group=optimizer['param_groups'][0]
    if (not isinstance(group,dict) or group.get('lr')!=cfg.LR
            or group.get('initial_lr')!=cfg.LR or group.get('weight_decay')!=cfg.WEIGHT_DECAY
            or tuple(group.get('betas',()))!=(.9,.999) or group.get('eps')!=1e-8
            or group.get('amsgrad',False) or group.get('maximize',False)
            or group.get('capturable',False) or group.get('differentiable',False)):
        raise ValueError('Saved BS optimizer recipe changed')
    parameters=payload['parameter_schema'];ids=group.get('params')
    if (not isinstance(ids,list) or ids!=list(range(len(parameters))) or not ids):
        raise ValueError('Saved BS optimizer parameter layout changed')
    states=optimizer['state']
    if set(states)!=(set(ids) if step else set()):
        raise ValueError('Saved BS optimizer states do not match its update boundary')
    for identifier,parameter in zip(ids,parameters):
        if not step:continue
        state=states[identifier]
        if not isinstance(state,dict) or set(state)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Incomplete saved BS AdamW moments')
        count=state['step']
        if (not torch.is_tensor(count) or count.numel()!=1
                or not bool(torch.isfinite(count).all()) or float(count)!=step):
            raise ValueError('Saved BS AdamW update count changed')
        for name in ('exp_avg','exp_avg_sq'):
            value=state[name]
            if (not torch.is_tensor(value) or list(value.shape)!=parameter['shape']
                    or str(value.dtype)!=parameter['dtype'] or not bool(torch.isfinite(value).all())):
                raise ValueError('Saved BS AdamW moment shape/dtype/finiteness changed')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('Negative saved AdamW squared moments')
    if (not isinstance(scheduler,dict) or type(scheduler.get('last_epoch'))is not int
            or scheduler['last_epoch']!=step or scheduler.get('_step_count')!=step+1
            or scheduler.get('base_lrs')!=[cfg.LR] or scheduler.get('_last_lr')!=[cfg.LR]
            or scheduler.get('lr_lambdas')!=[None]):
        raise ValueError('Saved BS scheduler boundary or constant recipe changed')


def _validate_model_and_rng(payload):
    import torch
    model=payload['model'];schema=payload['model_schema'];parameters=payload['parameter_schema']
    if not isinstance(model,dict) or not model or not isinstance(schema,dict) or set(model)!=set(schema):
        raise ValueError('Complete BS model/schema required')
    for name,value in model.items():
        if (not isinstance(name,str) or not torch.is_tensor(value)
                or not bool(torch.isfinite(value).all())
                or schema[name]!=dict(shape=list(value.shape),dtype=str(value.dtype))):
            raise ValueError('Saved BS model schema/finiteness changed: '+str(name))
    if not isinstance(parameters,list) or not parameters:
        raise ValueError('Saved BS parameter schema required')
    names=[]
    for row in parameters:
        if (not isinstance(row,dict) or set(row)!={'name','shape','dtype'}
                or row['name'] not in schema
                or dict(shape=row['shape'],dtype=row['dtype'])!=schema[row['name']]):
            raise ValueError('Saved BS parameter schema changed')
        names.append(row['name'])
    if len(set(names))!=len(names):raise ValueError('Duplicate saved BS parameter names')
    rng=payload['rng']
    if not isinstance(rng,dict) or set(rng)!={'python','numpy','torch_cpu','torch_cuda'}:
        raise ValueError('Complete external RNG state required')
    current=capture_rng()
    try:
        restore_rng(rng)
    except Exception as exc:
        raise ValueError('Invalid external RNG state') from exc
    finally:
        restore_rng(current)


def ensure_boundary_receipts(directory,manifest):
    """Recreate only a current, hash-verified boundary receipt."""
    directory=inside(directory)
    if read_json(_regular(directory/'manifest.json'))!=manifest:
        raise ValueError('Cannot export a changed checkpoint manifest')
    if manifest.get('checkpoint_sha256')!=sha256(_regular(directory/'checkpoint_latest.pt')):
        raise ValueError('Cannot export an unverified checkpoint boundary')
    names=[]
    if manifest['step']==100:names.append('boundary_00100.json')
    if manifest['step']==cfg.UPDATES:names.append('terminal_receipt.json')
    for name in names:
        path=directory/name
        if path.exists() or path.is_symlink():
            if read_json(_regular(path))!=manifest:
                raise ValueError('Existing residual-W boundary receipt changed: '+name)
        else:atomic_json(path,manifest,exclusive=True)


def save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,*,arm,method,step,
                    data,core,source,frozen,elapsed,measured,final=False):
    import torch
    if type(step)is not int or step<0 or step>cfg.UPDATES or step%cfg.SAVE_EVERY:
        raise ValueError('Complete100-update own save boundaries required')
    runtime.guard_core(core);optimizer_check(optimizer,scheduler,step)
    runtime.verify_files(source);helpers().assert_frozen_state(codec,ce,stats,frozen)
    initial_hash=sha256(_regular(directory/'initialization.json'))
    payload=dict(format=cfg.NAMESPACE,arm=arm,method=method,step=step,
        recipe=cfg.recipe(arm,method),fingerprint=cfg.fingerprint(arm,method),
        core=core,data_protocol=data.protocol,source_identity=source,frozen_identity=frozen,
        model=codec.decoder.state_dict(),model_schema=_schema(codec.decoder.state_dict()),
        parameter_schema=_parameter_schema(codec.decoder),optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),rng=capture_rng(),counters=cfg.counters(step),
        training_code=runtime.source_identity(),initialization_sha256=initial_hash)
    target=inside(directory/'checkpoint_latest.pt')
    original.atomic_torch_save(payload,target)
    memory=(dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if next(codec.decoder.parameters()).device.type=='cuda'
            else dict(peak_allocated_bytes=0,peak_reserved_bytes=0))
    record=dict(format=cfg.NAMESPACE+'_manifest_v1',
        status='completed' if step==cfg.UPDATES else 'paused_boundary' if final else 'running',
        arm=arm,method=method,step=step,counters=payload['counters'],recipe=payload['recipe'],
        fingerprint=payload['fingerprint'],recipe_fingerprint=payload['fingerprint'],
        core=core,data_protocol=data.protocol,source_identity=source,frozen_identity=frozen,
        checkpoint_sha256=sha256(target),initialization_sha256=initial_hash,
        updated=now(),elapsed_seconds=elapsed,seconds_per_update=measured,
        ETA_seconds=measured*(cfg.UPDATES-step),memory=memory,
        trainable='BS base decoder and residual head' if method=='w_residual' else 'BS base decoder',
        additional_EMA_commits=0,training_code=payload['training_code'])
    atomic_json(directory/'manifest.json',record)
    ensure_boundary_receipts(directory,record)
    return record


def verify_saved(directory,payload,manifest,*,arm,method,data,core,source,frozen,require_final=False):
    """Accept only this study's exact recipe, source and complete boundaries."""
    directory=inside(directory)
    expected_dir=inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm,method))
    if directory!=expected_dir:raise ValueError('Residual-W method output directory changed')
    runtime.guard_core(core);runtime.verify_files(source)
    checkpoint=_regular(directory/'checkpoint_latest.pt')
    initial_path=_regular(directory/'initialization.json')
    actual_manifest=read_json(_regular(directory/'manifest.json'))
    if actual_manifest!=manifest:raise ValueError('Residual-W manifest bytes/content changed')
    required={'format','arm','method','step','recipe','fingerprint','core','data_protocol',
        'source_identity','frozen_identity','model','model_schema','parameter_schema','optimizer',
        'scheduler','rng','counters','training_code','initialization_sha256'}
    if not isinstance(payload,dict) or set(payload)!=required:
        raise ValueError('Exact complete residual-W checkpoint schema required')
    identity=dict(format=cfg.NAMESPACE,arm=arm,method=method,recipe=cfg.recipe(arm,method),
        fingerprint=cfg.fingerprint(arm,method),core=core,data_protocol=data.protocol,
        source_identity=source,frozen_identity=frozen,initialization_sha256=sha256(initial_path))
    for name,value in identity.items():
        if payload.get(name)!=value:raise ValueError('Residual-W resume identity changed: '+name)
    step=payload['step']
    if type(step)is not int or step<0 or step>cfg.UPDATES or step%cfg.SAVE_EVERY:
        raise ValueError('Complete100-update own boundaries required')
    if payload['counters']!=cfg.counters(step):raise ValueError('Residual-W counters/cursor changed')
    code=payload['training_code']
    if (not isinstance(code,dict) or set(code)!={'commit','sha256'} or code['sha256']!=core['sha256']
            or not isinstance(code['commit'],str) or len(code['commit'])!=40
            or any(c not in '0123456789abcdef' for c in code['commit'])):
        raise ValueError('Residual-W training code identity changed')
    _validate_model_and_rng(payload);_validate_saved_optimizer(payload,step)
    expected=dict(arm=arm,method=method,step=step,counters=cfg.counters(step),core=core,
        data_protocol=data.protocol,source_identity=source,frozen_identity=frozen,
        fingerprint=cfg.fingerprint(arm,method),recipe_fingerprint=cfg.fingerprint(arm,method),
        recipe=cfg.recipe(arm,method),initialization_sha256=sha256(initial_path),training_code=code,
        additional_EMA_commits=0,format=cfg.NAMESPACE+'_manifest_v1')
    for name,value in expected.items():
        if manifest.get(name)!=value:raise ValueError('Residual-W manifest identity changed: '+name)
    if manifest.get('checkpoint_sha256')!=sha256(checkpoint):
        raise ValueError('Residual-W checkpoint bytes/receipt changed')
    if manifest.get('status') not in (('completed',) if step==cfg.UPDATES else ('running','paused_boundary')):
        raise ValueError('Residual-W checkpoint boundary status changed')
    initial=read_json(initial_path)
    initial_expected=dict(arm=arm,method=method,core=core,data_protocol=data.protocol,
        source_identity=source,frozen_identity=frozen,recipe=cfg.recipe(arm,method),
        fingerprint=cfg.fingerprint(arm,method),original_optimizer_continued=False,
        source_decoder_sha256=source['base_decoder_state_sha256'])
    for name,value in initial_expected.items():
        if initial.get(name)!=value:raise ValueError('Residual-W initialization identity changed: '+name)
    if (initial.get('model_schema')!=payload['model_schema']
            or initial.get('parameter_schema')!=payload['parameter_schema']):
        raise ValueError('Residual-W initialization model/parameter schema changed')
    if step==0 and initial.get('initial_decoder_sha256')!=state_digest(payload['model']):
        raise ValueError('Fresh zero-update decoder differs from initialization')
    if require_final and (step!=cfg.UPDATES or manifest.get('status')!='completed'
            or read_json(_regular(directory/'terminal_receipt.json'))!=manifest):
        raise ValueError('Fixed residual-W terminal not complete')
    for name in ('boundary_00100.json',) if step==100 else ():
        path=directory/name
        if path.exists() or path.is_symlink():
            if read_json(_regular(path))!=manifest:
                raise ValueError('Residual-W boundary receipt changed')
    return step


def monitor_if_missing(arm,method,step,codec,stats,ce,data,directory,*,forward=None):
    """Freeze actual-W scoring on128MU/six gamma, without selecting a model."""
    if step not in cfg.MONITOR_STEPS:return
    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash, checked_input_hashes, frozen_evaluation
    import evaluate_twc_v1_k8_w_residual as evaluate
    import twc_v1_k8_icc_aligned_results as tables
    from physical_split_finish_io import write_union_csv
    forward=runtime.forward_codec if forward is None else forward
    directory=inside(directory);receipt=directory/'monitor'/f'step_{step:05d}_receipt.json'
    manifest=read_json(_regular(directory/'manifest.json'));core=manifest['core']
    expected=dict(arm=arm,method=method,step=step,core=core,
        recipe_fingerprint=cfg.fingerprint(arm,method),data_protocol=data.protocol,
        checkpoint_sha256=manifest['checkpoint_sha256'],selection=False)
    if receipt.exists() or receipt.is_symlink():
        record=read_json(_regular(receipt))
        if any(record.get(name)!=value for name,value in expected.items()) or not record.get('files'):
            raise ValueError('Residual-W monitor receipt identity changed')
        for name,digest in record['files'].items():
            if sha256(_regular(name))!=digest:raise ValueError('Residual-W monitor evidence changed')
        return
    runtime.guard_core(core);runtime.verify_files(manifest['source_identity'])
    frozen=helpers().frozen_state(codec,ce,stats)
    output=inside(directory/'monitor'/f'step_{step:05d}_{uuid.uuid4().hex[:10]}')
    output.mkdir(parents=True,exist_ok=False)
    rows=[];fingerprints=[];name=cfg.method_name(method,arm)
    first,last=cfg.ROLE_RANGES['monitor']
    if last-first+1!=cfg.MONITOR_BATCHES:raise ValueError('Monitor sample budget/address changed')
    with frozen_evaluation({name:(codec,stats)},ce):
        for bi in range(cfg.MONITOR_BATCHES):
            index=first+bi;sample=data.sample('validation',index)
            if sample.get('address')!=data.address('validation',index):
                raise ValueError('Actual monitor address changed')
            hdl,hul=_tensor_hash(sample['H_dl']),_tensor_hash(sample['H_ul'])
            for gi,gamma in enumerate(cfg.GAMMAS):
                out=forward(codec,stats,ce,sample,gamma_db=gamma,noise_key=('monitor',bi,gi))
                before={key:_tensor_hash(out[key]) for key in ('H_hat','W','W_base','delta_W')}
                vectors=evaluate.score_out(out,sample['H_dl'])
                evaluate._validate_vectors(vectors,cfg.MICRO_BATCH)
                for within in range(cfg.MICRO_BATCH):
                    rows.append(dict(arm=name,gamma_db=float(gamma),sample_index=bi*cfg.MICRO_BATCH+within,
                        batch_index=bi,within_batch=within,**{key:value[within] for key,value in vectors.items()}))
                fingerprints.append(dict(arm=arm,method=method,batch_index=bi,gamma_db=gamma,address=sample['address'],
                    fingerprint=checked_input_hashes(out.get('input_hashes'),H_dl=hdl,H_ul=hul)))
                if any(_tensor_hash(out[key])!=digest for key,digest in before.items()):
                    raise RuntimeError('Monitor scoring mutated cached BS output')
            if _tensor_hash(sample['H_dl'])!=hdl or _tensor_hash(sample['H_ul'])!=hul:
                raise RuntimeError('Monitor input channels mutated')
    helpers().assert_frozen_state(codec,ce,stats,frozen)
    summary=tables.summarize_rows(rows,samples=cfg.MONITOR_BATCHES*cfg.MICRO_BATCH,
                                 batch_size=cfg.MICRO_BATCH,arms=(name,))
    write_union_csv(output/'samples.csv.gz',rows);write_union_csv(output/'summary.csv',summary)
    atomic_json(output/'input_fingerprints.json',fingerprints,exclusive=True)
    files={str(path):sha256(path) for path in output.iterdir() if path.is_file()}
    atomic_json(output/'manifest.json',dict(status='passed',phase='monitor',**expected,
        n_batches=cfg.MONITOR_BATCHES,samples_per_condition=cfg.MONITOR_BATCHES*cfg.MICRO_BATCH,
        gamma_db=list(cfg.GAMMAS),test_used=False,files=files),exclusive=True)
    files[str(output/'manifest.json')]=sha256(output/'manifest.json')
    runtime.guard_core(core);runtime.verify_files(manifest['source_identity'])
    atomic_json(receipt,dict(status='passed',**expected,path=str(output),files=files,updated=now()),exclusive=True)


def train(arm,method,stop_after=None,resume=False,*,require_cuda=True,
          loader=None,data_factory=None,forward=None,monitor=None):
    torch,np=original.tensor_runtime()
    target=cfg.UPDATES if stop_after is None else stop_after
    if (arm not in cfg.ARMS or method not in cfg.TRAIN_METHODS or type(target)is not int
            or target<100 or target>cfg.UPDATES or target%100):
        raise ValueError('Declared arm/method and complete100 boundary within fixed budget required')
    device=runtime.configure_cuda() if require_cuda else torch.device('cpu')
    core=runtime.core_identity();runtime.guard_core(core)
    random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED)
    if device.type=='cuda':torch.cuda.manual_seed_all(cfg.SEED)
    loader=runtime.load_initial if loader is None else loader
    data_factory=runtime.make_data if data_factory is None else data_factory
    forward=runtime.forward_codec if forward is None else forward
    monitor=monitor_if_missing if monitor is None else monitor
    directory=inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm,method))
    # Source/hash acceptance precedes all training output creation.
    codec,stats,ce,source=loader(arm,method,device)
    parameters=helpers().configure_bs_only(codec,ce,stats)
    frozen=helpers().frozen_state(codec,ce,stats)
    optimizer=torch.optim.AdamW(parameters,lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    data=data_factory(device);start=0
    if not resume:directory.mkdir(parents=True,exist_ok=False)
    elif not (directory/'checkpoint_latest.pt').is_file():
        raise FileNotFoundError('Own latest residual-W checkpoint required for explicit resume')
    with runtime.process_lock(directory/'training.lock'):
        if resume:
            manifest=read_json(_regular(directory/'manifest.json'));path=_regular(directory/'checkpoint_latest.pt')
            if manifest.get('checkpoint_sha256')!=sha256(path):raise ValueError('Resume checkpoint hash changed')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            start=verify_saved(directory,payload,manifest,arm=arm,method=method,data=data,core=core,source=source,frozen=frozen)
            if target<start:raise ValueError('Cannot rewind an owned residual-W checkpoint')
            ensure_boundary_receipts(directory,manifest)
            original.strict_load(codec.decoder,payload['model'])
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
            optimizer_check(optimizer,scheduler,start);restore_rng(payload['rng'])
        else:
            atomic_json(directory/'initialization.json',dict(arm=arm,method=method,core=core,
                source_identity=source,frozen_identity=frozen,data_protocol=data.protocol,
                recipe=cfg.recipe(arm,method),fingerprint=cfg.fingerprint(arm,method),
                source_decoder_sha256=source['base_decoder_state_sha256'],
                initial_decoder_sha256=state_digest(codec.decoder),model_schema=_schema(codec.decoder.state_dict()),
                parameter_schema=_parameter_schema(codec.decoder),created=now(),
                original_optimizer_continued=False),exclusive=True)
            save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,arm=arm,method=method,
                step=0,data=data,core=core,source=source,frozen=frozen,elapsed=0.,measured=0.)
        started=time.perf_counter();update_seconds=[];attempt_id=uuid.uuid4().hex
        try:
            monitor(arm,method,start,codec,stats,ce,data,directory,forward=forward)
            if start==target:
                runtime.verify_files(source);return read_json(directory/'manifest.json')
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            for step in range(start+1,target+1):
                begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                hashes=[];loss_values=[];rates=[];interferences=[];predicted_energies=[]
                for micro in range(cfg.ACCUMULATION):
                    index=cfg.micro_index(step,micro);sample=data.sample('codec_train',index)
                    if sample.get('address')!=data.address('codec_train',index):
                        raise ValueError('Actual training data address changed')
                    out=forward(codec,stats,ce,sample,gamma_db=None,noise_key=('codec_train',index,0))
                    # Score the transmitted W. Recomputing RZF(H_hat) here would
                    # silently discard the head and is intentionally forbidden.
                    loss,metrics=runtime.model_module().precoding_objective(out['W'],sample['H_dl'],downlink_snr_db=cfg.DL_SNR)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite actual-W rate loss')
                    (loss/cfg.ACCUMULATION).backward()
                    loss_values.append(float(loss.detach()));rates.append(float(metrics['sum_rate_bps_hz'].detach().mean()))
                    interferences.append(float(metrics['interference_power_mean'].detach().mean()))
                    predicted_energies.append(float(out['H_hat'].detach().abs().square().mean()))
                    hashes.append(dict(micro_index=index,address=sample['address'],
                        channel=sample['sample_fingerprint'],**out['input_hashes']))
                norm=commit_update(codec,optimizer,scheduler,previous_updates=step-1)
                if device.type=='cuda':torch.cuda.synchronize()
                update_seconds.append(time.perf_counter()-begin)
                if step%cfg.SAVE_EVERY==0:
                    helpers().assert_frozen_state(codec,ce,stats,frozen)
                    measured=sum(update_seconds[-100:])/len(update_seconds[-100:])
                    row=dict(arm=arm,method=method,step=step,attempt_id=attempt_id,resumed_from=start,
                        counters=cfg.counters(step),loss=sum(loss_values)/cfg.ACCUMULATION,
                        sum_rate_bps_hz=sum(rates)/cfg.ACCUMULATION,
                        interference_power_mean=sum(interferences)/cfg.ACCUMULATION,
                        predicted_dl_energy=sum(predicted_energies)/cfg.ACCUMULATION,
                        gradient_norm=float(norm),input_fingerprints=hashes,updated=now(),
                        mean_update_seconds=measured,ETA_seconds=measured*(cfg.UPDATES-step))
                    with (directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                    save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,arm=arm,method=method,
                        step=step,data=data,core=core,source=source,frozen=frozen,
                        elapsed=time.perf_counter()-started,measured=measured,final=step==target)
                    print(f'{arm} {method} BS {step}/{cfg.UPDATES} rate={row["sum_rate_bps_hz"]:.6g} ETA={row["ETA_seconds"]:.0f}s',flush=True)
                monitor(arm,method,step,codec,stats,ce,data,directory,forward=forward)
            runtime.guard_core(core);runtime.verify_files(source)
            return read_json(directory/'manifest.json')
        except BaseException as exc:
            atomic_json(directory/'failure.json',dict(status='failed',arm=arm,method=method,
                step=locals().get('step',start),last_saved_step=read_json(directory/'manifest.json')['step'],
                error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),updated=now()))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--run',action='store_true');actions.add_argument('--dry-run',action='store_true')
    parser.add_argument('--arm',choices=cfg.ARMS);parser.add_argument('--method',choices=cfg.TRAIN_METHODS)
    parser.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.resume and not args.run:parser.error('--resume requires --run')
    if args.dry_run:
        print(json.dumps(dict(root=str(cfg.OUTPUT_ROOT),recipe=cfg.recipe(args.arm or 'fdma',args.method or 'rzf_continue'),
            weights_read=False,writes=False),indent=2));return 0
    if not args.run:parser.print_help();return 0
    if args.arm is None or args.method is None:parser.error('--run requires --arm and --method')
    train(args.arm,args.method,args.stop_after,args.resume);return 0


if __name__=='__main__':raise SystemExit(main())
