#!/usr/bin/env python3
"""Fresh, boundary-resumable CE and paired-codec training for ICC-aligned K8.

Imports are deliberately light. No historical trainer, source chain or mutable
global adapter participates. Runtime tensor dependencies are loaded explicitly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

from configs import twc_v1_k8_icc_aligned_config as cfg

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def load_local(relative, name):
    path = ROOT / relative
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError('Module identity collision: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_identity():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,
        text=True).strip(), sha256={name:sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE + '_core_v1', sha256=source_identity()['sha256'])


def guard_core(identity):
    if identity != core_identity():
        raise ValueError('Training/model/physical/data/optimizer core source changed')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(path):
    target, allowed = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned independent ICC-aligned child path required')
    # Do not follow links into the new tree, including same-root symlinks.
    for item in (Path(path), *Path(path).parents):
        if item.is_symlink():
            raise ValueError('Linked output paths are forbidden')
        if item.resolve() == ROOT:
            break
    return target


def atomic_json(path, value, *, exclusive=False):
    path = inside(path)
    data = json.dumps(cfg.plain(value), indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        return
    handle, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def process_lock(path):
    """Persistent lock file, kernel ownership; never delete an old lock."""
    import fcntl
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def tensor_runtime():
    import torch
    import numpy as np
    return torch, np


def configure_cuda():
    torch, _ = tensor_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required; no CPU fallback')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda:0')


def environment():
    torch, np = tensor_runtime()
    import platform
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, CUDA=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_TF32=torch.backends.cuda.matmul.allow_tf32, cudnn_TF32=torch.backends.cudnn.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


def capture_rng():
    torch, np = tensor_runtime()
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    torch, np = tensor_runtime()
    if set(state) != {'python','numpy','torch_cpu','torch_cuda'}:
        raise ValueError('Incomplete default RNG')
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if len(state['torch_cuda']) != torch.cuda.device_count():
        raise ValueError('Saved CUDA RNG device count changed')
    torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def state_digest(model):
    h = hashlib.sha256()
    state = model.state_dict() if hasattr(model, 'state_dict') else model
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def strict_load(model, state):
    torch, _ = tensor_runtime()
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError('Strict checkpoint key set changed')
    for key, value in expected.items():
        other = state[key]
        if other.shape != value.shape or other.dtype != value.dtype or not bool(torch.isfinite(other).all()):
            raise ValueError('Checkpoint shape/dtype/finiteness changed: ' + key)
    model.load_state_dict(state, strict=True)


def verify_calibrated_statistics(stats,record,step):
    torch,_=tensor_runtime()
    expected=torch.tensor(record['per_RE_error'],dtype=torch.float32,device=stats.ce_errors.device)
    if (not torch.equal(stats.ce_errors,expected) or not bool(stats.ce_calibrated)
            or int(stats.ema_updates)!=step):
        raise ValueError('Saved frozen CE calibration or causal EMA counter changed')
    stats.nominal_power()  # validate finite nonnegative historical state without mutating it


def verify_training_receipts(directory,payload,manifest):
    initial=read_json(directory/'initialization.json')
    component=payload['component']
    expected=dict(component=component,core=payload['core'],fingerprint=payload['fingerprint'])
    for key,value in expected.items():
        if manifest.get(key)!=value or initial.get(key)!=value:
            raise ValueError('Training initialization/manifest identity changed: '+key)
    if initial.get('recipe')!=payload['recipe'] or initial.get('CE_identity')!=payload['ce_identity']:
        raise ValueError('Initial complete recipe/CE identity changed')
    if manifest.get('step')!=payload['step'] or manifest.get('counters')!=payload['counters']:
        raise ValueError('Manifest optimizer boundary changed')


def make_data(device):
    source = load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
        device=str(device), cdl_model='C', speed=1., seed_namespace=cfg.NAMESPACE)


def physics():
    return load_local('utils/twc_v1_k8_icc_aligned_physics.py', '_icc_aligned_physics')


def new_model(component, device):
    torch, _ = tensor_runtime()
    from models.twc_v1_k8_icc_aligned import K8ICCAlignedCodec, K8ICCAlignedPilotUNet
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(cfg.INITIAL_CE_SEED if component == 'ce' else cfg.INITIAL_CODEC_SEED)
        model = K8ICCAlignedPilotUNet() if component == 'ce' else K8ICCAlignedCodec(component)
    return model.to(device)


def batch_random(role, index, device, *, gamma=None):
    torch, _ = tensor_runtime()
    if gamma is None:
        gen = torch.Generator(device='cpu').manual_seed(cfg.noise_seed(role,index,'gamma'))
        gamma = (25 * torch.rand(4, generator=gen)).to(device)
    else:
        gamma = torch.as_tensor(gamma, device=device, dtype=torch.float32)
        if gamma.ndim == 0:
            gamma = gamma.expand(4)
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'noise'))
    # Complex standard noise has E|xi|^2=1; transmit adds sqrt(N0) once.
    noise = torch.view_as_complex(torch.randn(4,32,256,2,device=device,generator=gen).contiguous()) / 2**.5
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'polarity'))
    polarity = (2 * torch.randint(2,(4,8),device=device,generator=gen)-1).float()
    return gamma, noise, polarity


def pilot_forward(ce, sample, gamma, noise, polarity):
    phy = physics()
    pilots = phy.known_pilots(4, device=sample['H_ul'].device, polarity=polarity)
    Y = phy.transmit(sample['H_ul'], pilots, noise, gamma)
    initial = phy.fourier_ls(Y, pilots)
    corrected = ce(Y, pilots, initial, gamma_db=gamma)
    return corrected, initial, Y, pilots


def forward_codec(model, stats, ce, sample, gamma_db, noise_key):
    torch, _ = tensor_runtime(); phy = physics()
    role, batch, gi = noise_key
    if role not in cfg.ROLE_RANGES:
        raise ValueError('Explicit new data/noise namespace required')
    gamma, noise, polarity = batch_random(role,batch*10+gi,sample['H_ul'].device,gamma=gamma_db)
    raw = model.encoder(sample['H_dl'])
    wave = phy.build_waveform(raw, stats.layout, polarity=polarity)
    Y = phy.transmit(sample['H_ul'], wave.x_grid, noise, gamma)
    with torch.no_grad():
        initial = phy.fourier_ls(Y, wave.pilot)
        Hest = ce(Y, wave.pilot, initial, gamma_db=gamma)
        cevar = stats.ce_variance(gamma)
    recovered = phy.recover(Y,Hest,stats.layout,stats.nominal_power(),cevar,gamma)
    result = model.decoder(Y_rx=Y,H_ul_est=Hest,soft=recovered['soft'],
        reliability=recovered['reliability'],pilots=wave.pilot,gamma_db=gamma,
        layout=stats.layout,prior=stats.nominal_power(),ce_error=cevar)
    if torch.is_tensor(result):
        result = dict(H_hat=result)
    output=dict(result, waveform=wave, soft=recovered['soft'],
        reliability=recovered['reliability'],H_ul_est=Hest, gamma_db=gamma,
        input_fingerprint=cfg.digest(dict(channel=sample['sample_fingerprint'],
            noise_seed=cfg.noise_seed(role,batch*10+gi,'noise'),gamma=gamma.detach().cpu().tolist())))
    if role in ('monitor','final'):
        output['input_hashes']=dict(standard_noise=state_digest({'noise':noise}),
            pilot_polarity=state_digest({'polarity':polarity}))
    return output


def load_checkpoint(component, device='cpu', *, require_final=True):
    torch, _ = tensor_runtime()
    directory = OUTPUT_ROOT/'training'/component
    path = directory/'checkpoint_latest.pt'; terminal=directory/'terminal_receipt.json'
    manifest = read_json(directory/'manifest.json')
    if require_final and (manifest.get('status') != 'completed' or not terminal.is_file()):
        raise ValueError('Fixed terminal training not complete: '+component)
    record = read_json(terminal) if terminal.is_file() else manifest
    if record.get('checkpoint_sha256') != sha256(path):
        raise ValueError('Checkpoint receipt/hash changed: '+component)
    payload = torch.load(path,map_location='cpu',weights_only=False)
    expected_ce = None if component == 'ce' else ce_identity()
    step = cfg.validate_resume(payload,component,core=core_identity(),
        data_protocol=make_data(device).protocol,ce_identity=expected_ce)
    verify_training_receipts(directory,payload,manifest)
    if require_final and record!=manifest:raise ValueError('Terminal receipt/manifest differ')
    if require_final and step != (cfg.CE_UPDATES if component == 'ce' else cfg.CODEC_UPDATES):
        raise ValueError('Fixed final update count required')
    model = new_model(component,device); strict_load(model,payload['model'])
    return model,payload,dict(path=str(path),sha256=sha256(path),step=step,
        state_sha256=state_digest(model),component=component,training_code=payload['training_code'],
        recipe=payload['recipe'],core=payload['core'],
        frozen_files=[dict(path=str(path),sha256=sha256(path)),
            dict(path=str(terminal),sha256=sha256(terminal))] if terminal.is_file() else [])


def calibration_record():
    path=OUTPUT_ROOT/'ce_calibration'/'calibration.json'
    record=read_json(path)
    if record.get('record_id') != cfg.digest({k:v for k,v in record.items() if k!='record_id'}):
        raise ValueError('CE calibration lock was modified')
    if record['core'] != core_identity() or record['CE_checkpoint_sha256'] != sha256(OUTPUT_ROOT/'training/ce/checkpoint_latest.pt'):
        raise ValueError('CE calibration core/checkpoint changed')
    return record


def ce_identity():
    record=calibration_record()
    return dict(checkpoint_sha256=record['CE_checkpoint_sha256'],
        calibration_record_id=record['record_id'])


def load_ce(device):
    model,payload,record=load_checkpoint('ce',device)
    path=OUTPUT_ROOT/'ce_calibration/calibration.json'
    if path.is_file():record['frozen_files'].append(dict(path=str(path),sha256=sha256(path)))
    return model.eval().requires_grad_(False),record


def load_codec(arm,device):
    model,payload,record=load_checkpoint(arm,device)
    stats=physics().ReceiverStatistics(physics().Layout(arm),calibration_record()['per_RE_error']).to(device)
    strict_load(stats,payload['statistics'])
    verify_calibrated_statistics(stats,calibration_record(),cfg.CODEC_UPDATES)
    return model.eval().requires_grad_(False),stats.eval(),record


def calibrate_ce(output):
    torch,_=tensor_runtime(); device=configure_cuda(); output=inside(output)
    output.mkdir(parents=True,exist_ok=False)
    ce,record=load_ce(device); before=state_digest(ce); data=make_data(device)
    node_errors=[]; inputs=[]
    with torch.no_grad():
        for gi,gamma in enumerate(cfg.GAMMAS):
            total=torch.zeros(8,256,device=device)
            for bi in range(cfg.CE_CALIBRATION_BATCHES):
                sample=data.sample('ce_calibration',bi)
                g,noise,polarity=batch_random('ce_calibration',bi*10+gi,device,gamma=gamma)
                corrected,_,_,_=pilot_forward(ce,sample,g,noise,polarity)
                total+=(corrected-sample['H_ul']).abs().square().mean(dim=(0,2))
                inputs.append(dict(gamma=gamma,batch=bi,fingerprint=sample['sample_fingerprint'],
                    noise_seed=cfg.noise_seed('ce_calibration',bi*10+gi,'noise')))
            node_errors.append((total/cfg.CE_CALIBRATION_BATCHES).cpu().tolist())
    if state_digest(ce)!=before or sha256(record['path'])!=record['sha256']:
        raise ValueError('Frozen CE changed during calibration')
    result=dict(status='completed',namespace=cfg.NAMESPACE,nodes=list(cfg.GAMMAS),
        per_RE_error=node_errors,definition='mean absolute complex channel error power per UE and RE, averaged batch/antenna',
        samples_per_node=256,interpolation='nonnegative piecewise-linear gamma;no extrapolation',
        CE_checkpoint_sha256=record['sha256'],CE_state_sha256=before,core=core_identity(),
        training_code=source_identity(),data_protocol=data.protocol,inputs=inputs,created=now())
    result['record_id']=cfg.digest(result)
    atomic_json(output/'calibration.json',result,exclusive=True)
    return result


def optimizer_check(optimizer, scheduler, step):
    torch,_=tensor_runtime()
    if scheduler.last_epoch!=step or scheduler._step_count!=step+1:
        raise ValueError('Constant scheduler resume boundary changed')
    for group in optimizer.param_groups:
        if group['lr']!=cfg.LEARNING_RATE or group['weight_decay']!=cfg.WEIGHT_DECAY:
            raise ValueError('AdamW recipe changed')
        for parameter in group['params']:
            state=optimizer.state.get(parameter,{})
            if step and (not state or int(state['step'])!=step):
                raise ValueError('Incomplete AdamW update state')
            for key in ('exp_avg','exp_avg_sq'):
                if step and (state[key].shape!=parameter.shape or state[key].dtype!=parameter.dtype
                             or not bool(torch.isfinite(state[key]).all())):
                    raise ValueError('Invalid AdamW moment: '+key)


def commit_update(model,optimizer,scheduler,*,stats=None,power_sum=None,
                  frames=32,previous_updates):
    """One successful effective-batch update followed by one causal EMA commit."""
    torch,_=tensor_runtime()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Invalid accumulated gradients')
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.GRAD_CLIP,error_if_nonfinite=True)
    if not bool(grad_norm>0):raise ValueError('Zero effective gradient norm')
    optimizer.step();scheduler.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite model after AdamW; no EMA commit/checkpoint')
    if stats is not None:
        stats.commit_power(power_sum,frames,expected_previous_updates=previous_updates)
    return grad_norm


def save_checkpoint(directory,model,stats,optimizer,scheduler,component,step,data,core,ce_record,started,elapsed,measured):
    torch,_=tensor_runtime()
    guard_core(core); optimizer_check(optimizer,scheduler,step)
    payload=dict(format=cfg.NAMESPACE,component=component,step=step,
        recipe=cfg.recipe(component),fingerprint=cfg.fingerprint(component),
        core=core,training_code=source_identity(),model=model.state_dict(),
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(),
        counters=cfg.counters(component,step),data_protocol=data.protocol,ce_identity=ce_record,
        statistics=None if stats is None else stats.state_dict())
    target=directory/'checkpoint_latest.pt'
    handle,name=tempfile.mkstemp(prefix='.checkpoint_',dir=directory); os.close(handle)
    try:
        torch.save(payload,name);os.replace(name,target)
    finally:
        if os.path.exists(name):os.unlink(name)
    total=cfg.CE_UPDATES if component=='ce' else cfg.CODEC_UPDATES
    record=dict(status='completed' if step==total else 'paused',component=component,step=step,
        counters=payload['counters'],checkpoint_sha256=sha256(target),core=core,
        training_code=payload['training_code'],fingerprint=payload['fingerprint'],updated=now(),
        elapsed_seconds=elapsed,ETA_seconds=measured*(total-step),seconds_per_update=measured,
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        measurements='actual microbatch4/accumulation8 with allocated AdamW state;monitor excluded',
        environment=environment(),ce_identity=ce_record)
    atomic_json(directory/'manifest.json',record)
    if step == cfg.SMOKE_UPDATES:
        atomic_json(directory/'boundary_00100.json',record,exclusive=True)
    if step==total:
        atomic_json(directory/'terminal_receipt.json',record,exclusive=True)
    return record


def monitor_if_missing(component,step,model,stats,ce,data,directory):
    """A report failure after a saved update can be retried without an update."""
    if component=='ce' or step not in cfg.MONITOR_STEPS:return
    receipt=directory/'monitor'/f'step_{step:05d}_receipt.json'
    if receipt.is_file():
        record=read_json(receipt)
        for path,digest in record['files'].items():
            if sha256(path)!=digest:raise ValueError('Completed monitor file changed')
        return
    import uuid
    import evaluate_twc_v1_k8_icc_aligned as evaluate
    saved_rng=capture_rng();saved_modes={n:m.training for n,m in model.named_modules()}
    prior_state=state_digest(stats)
    output=directory/'monitor'/f'step_{step:05d}_{uuid.uuid4().hex[:10]}'
    try:
        model.eval()
        evaluate.evaluate_models({component:(model,stats)},ce,data,
            mode='monitor',output_dir=output,checkpoint_records={},code_identity=source_identity())
    finally:
        for name,module in model.named_modules():module.training=saved_modes[name]
        restore_rng(saved_rng)
    if state_digest(stats)!=prior_state:raise ValueError('Monitor changed historical prior')
    files={str(path):sha256(path) for path in output.iterdir() if path.is_file()}
    atomic_json(receipt,dict(step=step,path=str(output),files=files,updated=now()),exclusive=True)
    atomic_json(directory/'monitor_latest.json',dict(step=step,path=str(output),updated=now()))


def train(component,*,resume=False,stop_after=None):
    torch,np=tensor_runtime();device=configure_cuda();core=core_identity()
    total=cfg.CE_UPDATES if component=='ce' else cfg.CODEC_UPDATES
    target=total if stop_after is None else stop_after
    if type(target)is not int or not 1<=target<=total or target%cfg.SAVE_EVERY:
        raise ValueError('Only complete100-update boundaries within fixed budget may stop')
    directory=inside(OUTPUT_ROOT/'training'/component)
    if resume:
        if not (directory/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Explicit latest required')
    else:
        directory.mkdir(parents=True,exist_ok=False)
    with process_lock(directory/'run.lock'):
        random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED);torch.cuda.manual_seed_all(cfg.SEED)
        data=make_data(device);model=new_model(component,device)
        ce_record=None;ce=None;stats=None
        if component!='ce':
            ce,ce_artifact=load_ce(device);ce_record=ce_identity()
            stats=physics().ReceiverStatistics(physics().Layout(component),calibration_record()['per_RE_error']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        start=0
        if resume:
            payload=torch.load(directory/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            manifest=read_json(directory/'manifest.json')
            if manifest.get('checkpoint_sha256')!=sha256(directory/'checkpoint_latest.pt'):
                raise ValueError('Resume checkpoint receipt mismatch')
            start=cfg.validate_resume(payload,component,core=core,data_protocol=data.protocol,ce_identity=ce_record)
            verify_training_receipts(directory,payload,manifest)
            strict_load(model,payload['model'])
            if stats is not None:
                strict_load(stats,payload['statistics'])
                verify_calibrated_statistics(stats,calibration_record(),start)
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
            optimizer_check(optimizer,scheduler,start);restore_rng(payload['rng'])
        else:
            atomic_json(directory/'initialization.json',dict(component=component,
                model_state_sha256=state_digest(model),core=core,training_code=source_identity(),
                recipe=cfg.recipe(component),fingerprint=cfg.fingerprint(component),
                random_fresh=True,old_weights_loaded=False,CE_identity=ce_record),exclusive=True)
        if start>target:raise ValueError('Cannot rewind training boundary')
        if start==target:
            monitor_if_missing(component,start,model,stats,ce,data,directory)
            return read_json(directory/'manifest.json')
        model.train();before_ce=None if ce is None else state_digest(ce)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter();update_seconds=[]
        try:
            for step in range(start+1,target+1):
                begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                power_sum=None;micro_losses=[];input_hashes=[]
                old_prior=None if stats is None else stats.nominal_power().detach().clone()
                for mi in range(cfg.ACCUMULATION):
                    index=cfg.micro_index(step,mi);role='ce_train' if component=='ce' else 'codec_train'
                    sample=data.sample(role,index)
                    gamma,noise,polarity=batch_random(role,index,device)
                    if component=='ce':
                        prediction,_,_,_=pilot_forward(model,sample,gamma,noise,polarity)
                        truth=sample['H_ul']
                    else:
                        result=forward_codec(model,stats,ce,sample,gamma,(role,index,0))
                        prediction=result['H_hat'];truth=sample['H_dl']
                        power=result['waveform'].active.detach().abs().square().sum(dim=0)
                        power_sum=power if power_sum is None else power_sum+power
                        if not torch.equal(stats.nominal_power(),old_prior):
                            raise ValueError('Prior changed inside effective batch')
                    loss=((prediction-truth).abs().square().sum(dim=(2,3))/
                        truth.abs().square().sum(dim=(2,3)).clamp_min(1e-20)).mean()
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite NMSE')
                    (loss/cfg.ACCUMULATION).backward();micro_losses.append(float(loss.detach()))
                    input_hashes.append(sample['sample_fingerprint'])
                grad_norm=commit_update(model,optimizer,scheduler,stats=stats,power_sum=power_sum,
                    frames=32,previous_updates=step-1)
                torch.cuda.synchronize();update_seconds.append(time.perf_counter()-begin)
                if step%cfg.SAVE_EVERY==0:
                    guard_core(core)
                    if ce is not None and state_digest(ce)!=before_ce:raise ValueError('Frozen CE changed')
                    measured=sum(update_seconds[-100:])/len(update_seconds[-100:])
                    row=dict(step=step,counters=cfg.counters(component,step),NMSE=sum(micro_losses)/8,
                        gradient_norm=float(grad_norm),input_fingerprints=input_hashes,updated=now(),
                        mean_update_seconds=measured,ETA_seconds=measured*(total-step))
                    with (directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                    save_checkpoint(directory,model,stats,optimizer,scheduler,component,step,data,core,
                        ce_record,started,time.perf_counter()-started,measured)
                    print(f'{component} {step}/{total} NMSE={row["NMSE"]:.6g} ETA={row["ETA_seconds"]:.0f}s',flush=True)
                monitor_if_missing(component,step,model,stats,ce,data,directory)
            return read_json(directory/'manifest.json')
        except BaseException as exc:
            atomic_json(directory/'failure.json',dict(status='failed',component=component,
                step=locals().get('step',start),last_completed_checkpoint_step=read_json(directory/'manifest.json').get('step')
                    if (directory/'manifest.json').exists() else None,
                error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),updated=now()))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('ce',)+cfg.ARMS)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--dry-run',action='store_true');actions.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--calibrate-ce',action='store_true')
    parser.add_argument('--calibration-output',type=Path)
    args=parser.parse_args(argv)
    if not args.run and not args.dry_run:parser.print_help();return 0
    if args.dry_run:
        print(json.dumps(dict(output=str(OUTPUT_ROOT),recipe=cfg.recipe(args.component or 'ce'),
            operation='CE calibration' if args.calibrate_ce else 'training',weights_read=False,files_written=False),indent=2))
        return 0
    if args.calibrate_ce:
        calibrate_ce(args.calibration_output or OUTPUT_ROOT/'ce_calibration');return 0
    if args.component is None:parser.error('--component required for explicit training')
    train(args.component,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':
    raise SystemExit(main())
