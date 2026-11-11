#!/usr/bin/env python3
"""Independent, decoder-only rate adaptation of pinned ICC-aligned terminals."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

from configs import twc_v1_k8_icc_rate_config as cfg
import train_twc_v1_k8_icc_aligned as original

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
now = original.now
sha256 = original.sha256
read_json = original.read_json
state_digest = original.state_digest
capture_rng = original.capture_rng
restore_rng = original.restore_rng
configure_cuda = original.configure_cuda
environment = original.environment


def helpers():
    return original.load_local('utils/twc_v1_k8_icc_rate.py', '_icc_rate_math')


def source_identity():
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), sha256={n:sha256(ROOT/n) for n in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1', sha256=source_identity()['sha256'])


def guard_source():
    if original.core_identity() != cfg.SOURCE_CORE:
        raise ValueError('Pinned original training core changed; no source migration allowed')


def guard_core(expected):
    guard_source()
    if expected != core_identity():
        raise ValueError('Rate experiment source changed')


def inside(path):
    path = Path(path)
    target, allowed = path.resolve(), OUTPUT_ROOT.resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned rate experiment child path required')
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Linked rate output paths are forbidden')
        if item.resolve() == ROOT:
            break
    return target


def atomic_json(path, value, *, exclusive=False):
    path = inside(path)
    data = json.dumps(cfg.plain(value), indent=2, sort_keys=True, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        return
    handle, name = tempfile.mkstemp(prefix='record_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


@contextmanager
def process_lock(path):
    import fcntl
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_files(record):
    files = record.get('frozen_files', [])
    if not files:
        raise ValueError('Read-only source evidence required')
    for entry in files:
        path = Path(entry['path'])
        if path.is_symlink() or not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError('Read-only source artifact changed: '+str(path))


def load_source(arm, device='cpu'):
    """Use unchanged strict original loaders; pin actual terminals before loading."""
    if arm not in cfg.ARMS:
        raise ValueError('Declared paired arm required')
    guard_source()
    files = []
    for component in ('ce', arm):
        directory = cfg.SOURCE_ROOT/'training'/component
        for path, digest in ((directory/'checkpoint_latest.pt', cfg.SOURCE_CHECKPOINT_SHA256[component]),
                             (directory/'terminal_receipt.json', cfg.SOURCE_TERMINAL_SHA256[component])):
            if path.is_symlink() or not path.is_file() or sha256(path) != digest:
                raise ValueError('Pinned source terminal missing/changed: '+str(path))
            files.append(dict(path=str(path), sha256=digest))
        for name in ('manifest.json', 'initialization.json'):
            path = directory/name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Original source receipt missing: '+str(path))
            files.append(dict(path=str(path), sha256=sha256(path)))
    calibration = cfg.SOURCE_ROOT/'ce_calibration/calibration.json'
    if calibration.is_symlink() or not calibration.is_file() or sha256(calibration) != cfg.SOURCE_CALIBRATION_SHA256:
        raise ValueError('Pinned CE calibration changed')
    files.append(dict(path=str(calibration), sha256=cfg.SOURCE_CALIBRATION_SHA256))
    ce, ce_record = original.load_ce(device)
    codec, statistics, codec_record = original.load_codec(arm, device)
    for record in (ce_record, codec_record):
        if record['training_code']['commit'] != cfg.SOURCE_COMMIT or record['core'] != cfg.SOURCE_CORE:
            raise ValueError('Pinned source training identity changed')
    codec.eval().requires_grad_(False)
    ce.eval().requires_grad_(False)
    statistics.eval().requires_grad_(False)
    record = dict(arm=arm, source_commit=cfg.SOURCE_COMMIT,
        checkpoint_sha256=codec_record['sha256'], CE_checkpoint_sha256=ce_record['sha256'],
        calibration_sha256=cfg.SOURCE_CALIBRATION_SHA256, core=cfg.SOURCE_CORE,
        source_state_sha256=state_digest(codec), frozen_files=files)
    verify_files(record)
    return codec, statistics, ce, record


def make_data(device):
    source = original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
        device=str(device), cdl_model='C', speed=1., seed_namespace=cfg.DATA_NAMESPACE)


def batch_random(role, index, device, *, gamma=None):
    import torch
    if gamma is None:
        gen = torch.Generator(device='cpu').manual_seed(cfg.noise_seed(role,index,'gamma'))
        gamma = (25*torch.rand(cfg.MICRO_BATCH, generator=gen)).to(device)
    else:
        gamma = torch.as_tensor(gamma, device=device, dtype=torch.float32)
        if gamma.ndim == 0: gamma = gamma.expand(cfg.MICRO_BATCH)
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'noise'))
    noise = torch.view_as_complex(torch.randn(4,32,256,2,device=device,generator=gen).contiguous())/2**.5
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'polarity'))
    polarity = (2*torch.randint(2,(4,8),device=device,generator=gen)-1).float()
    return gamma, noise, polarity


def forward_codec(codec, stats, ce, sample, gamma_db, noise_key):
    """Only BS decoder builds a gradient graph; truth is an UE input/loss label."""
    import torch
    role, batch, gi = noise_key
    if role not in cfg.ROLE_RANGES:
        raise ValueError('Declared rate data/noise role required')
    phy = original.physics()
    with torch.no_grad():
        gamma, noise, polarity = batch_random(role,batch*10+gi,sample['H_ul'].device,gamma=gamma_db)
        raw = codec.encoder(sample['H_dl'])
        wave = phy.build_waveform(raw,stats.layout,polarity=polarity)
        Y = phy.transmit(sample['H_ul'],wave.x_grid,noise,gamma)
        initial = phy.fourier_ls(Y,wave.pilot)
        Hest = ce(Y,wave.pilot,initial,gamma_db=gamma)
        cevar = stats.ce_variance(gamma)
        recovered = phy.recover(Y,Hest,stats.layout,stats.nominal_power(),cevar,gamma)
    result = codec.decoder(Y_rx=Y,H_ul_est=Hest,soft=recovered['soft'],
        reliability=recovered['reliability'],pilots=wave.pilot,gamma_db=gamma,
        layout=stats.layout,prior=stats.nominal_power(),ce_error=cevar)
    if torch.is_tensor(result): result = dict(H_hat=result)
    return dict(result,waveform=wave,soft=recovered['soft'],reliability=recovered['reliability'],
        H_ul_est=Hest,gamma_db=gamma,input_hashes=dict(
            standard_noise=state_digest({'noise':noise}),pilot_polarity=state_digest({'polarity':polarity})))


def optimizer_check(optimizer, scheduler, step):
    import torch
    if scheduler.last_epoch != step or scheduler._step_count != step+1:
        raise ValueError('Rate scheduler boundary changed')
    for group in optimizer.param_groups:
        if group['lr'] != cfg.LR or group['weight_decay'] != cfg.WEIGHT_DECAY:
            raise ValueError('Rate optimizer recipe changed')
        for parameter in group['params']:
            state = optimizer.state.get(parameter,{})
            if step and (not state or int(state['step']) != step):
                raise ValueError('Incomplete decoder AdamW boundary')
            for key in ('exp_avg','exp_avg_sq'):
                if step and (state[key].shape != parameter.shape or state[key].dtype != parameter.dtype
                             or not bool(torch.isfinite(state[key]).all())):
                    raise ValueError('Invalid decoder AdamW moments')


def commit_update(codec, optimizer, scheduler, *, previous_updates):
    import torch
    optimizer_check(optimizer,scheduler,previous_updates)
    parameters = tuple(codec.decoder.parameters())
    if not parameters or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError('Missing/nonfinite decoder gradient; update refused')
    norm = torch.nn.utils.clip_grad_norm_(parameters,cfg.GRAD_CLIP,error_if_nonfinite=True)
    if not bool(norm>0):
        raise ValueError('Zero rate gradient; optimizer update refused')
    optimizer.step(); scheduler.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError('Nonfinite decoder after AdamW; checkpoint refused')
    optimizer_check(optimizer,scheduler,previous_updates+1)
    return norm


def ensure_boundary_receipts(directory, manifest):
    """Finish a missing export only from a hash-verified current checkpoint."""
    names=[]
    if manifest['step']==100: names.append('boundary_00100.json')
    if manifest['step']==cfg.UPDATES: names.append('terminal_receipt.json')
    for name in names:
        path=directory/name
        if path.exists():
            if read_json(path)!=manifest:
                raise ValueError('Existing rate boundary receipt changed: '+name)
        else:
            atomic_json(path,manifest,exclusive=True)


def save_checkpoint(directory, codec, stats, ce, optimizer, scheduler, *, arm, step,
                    data, core, source, frozen, elapsed, measured, final=False):
    import torch
    guard_core(core); optimizer_check(optimizer,scheduler,step)
    verify_files(source)
    helpers().assert_frozen_state(codec,ce,stats,frozen)
    payload=dict(format=cfg.NAMESPACE,arm=arm,step=step,recipe=cfg.recipe(arm),
        fingerprint=cfg.fingerprint(arm),core=core,data_protocol=data.protocol,
        source_identity=source,frozen_identity=frozen,model=codec.decoder.state_dict(),
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(),
        counters=cfg.counters(step),training_code=source_identity())
    target=inside(directory/'checkpoint_latest.pt')
    original.atomic_torch_save(payload,target)
    record=dict(status='completed' if step==cfg.UPDATES else 'paused_boundary' if final else 'running',
        arm=arm,step=step,counters=payload['counters'],recipe=payload['recipe'],
        fingerprint=payload['fingerprint'],core=core,source_identity=source,frozen_identity=frozen,
        checkpoint_sha256=sha256(target),updated=now(),elapsed_seconds=elapsed,
        seconds_per_update=measured,ETA_seconds=measured*(cfg.UPDATES-step),
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        trainable='BS decoder only',additional_EMA_commits=0,training_code=payload['training_code'])
    atomic_json(directory/'manifest.json',record)
    ensure_boundary_receipts(directory,record)
    return record


def verify_saved(directory, payload, manifest, *, arm, data, core, source, frozen, require_final=False):
    step=cfg.validate_resume(payload,arm,core=core,data_protocol=data.protocol,
        source_identity=source,frozen_identity=frozen)
    for name,value in dict(arm=arm,step=step,counters=cfg.counters(step),core=core,
        source_identity=source,frozen_identity=frozen,fingerprint=cfg.fingerprint(arm),recipe=cfg.recipe(arm)).items():
        if manifest.get(name) != value:
            raise ValueError('Rate checkpoint manifest identity changed: '+name)
    if manifest.get('checkpoint_sha256') != sha256(directory/'checkpoint_latest.pt'):
        raise ValueError('Rate checkpoint bytes/receipt changed')
    initial=read_json(directory/'initialization.json')
    for name,value in dict(arm=arm,core=core,source_identity=source,frozen_identity=frozen,
                           recipe=cfg.recipe(arm),fingerprint=cfg.fingerprint(arm)).items():
        if initial.get(name) != value:
            raise ValueError('Rate initialization identity changed: '+name)
    if require_final and (step!=cfg.UPDATES or manifest.get('status')!='completed'
                         or read_json(directory/'terminal_receipt.json')!=manifest):
        raise ValueError('Fixed rate terminal not complete')
    return step


def load_finetuned(arm, device='cpu'):
    import torch
    codec,stats,ce,source=load_source(arm,device)
    frozen=helpers().frozen_state(codec,ce,stats)
    directory=inside(OUTPUT_ROOT/'training'/arm)
    manifest=read_json(directory/'manifest.json')
    path=directory/'checkpoint_latest.pt'
    if manifest.get('checkpoint_sha256')!=sha256(path):
        raise ValueError('Rate checkpoint hash changed')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    verify_saved(directory,payload,manifest,arm=arm,data=make_data(device),core=core_identity(),
                 source=source,frozen=frozen,require_final=True)
    original.strict_load(codec.decoder,payload['model'])
    helpers().assert_frozen_state(codec,ce,stats,frozen)
    codec.eval().requires_grad_(False)
    record=dict(arm=arm,path=str(path),sha256=sha256(path),step=payload['step'],
        core=payload['core'],source_identity=source,state_sha256=state_digest(codec),
        frozen_files=source['frozen_files']+[dict(path=str(path),sha256=sha256(path)),
            dict(path=str(directory/'terminal_receipt.json'),sha256=sha256(directory/'terminal_receipt.json'))])
    return codec,stats,ce,record


def monitor_if_missing(arm,step,codec,stats,ce,data,directory):
    if step not in cfg.MONITOR_STEPS: return
    receipt=directory/'monitor'/f'step_{step:05d}_receipt.json'
    if receipt.exists():
        for name,digest in read_json(receipt)['files'].items():
            if sha256(inside(name))!=digest: raise ValueError('Rate monitor evidence changed')
        return
    import evaluate_twc_v1_k8_icc_rate as evaluate
    saved=capture_rng(); frozen=helpers().frozen_state(codec,ce,stats)
    modes={n:m.training for n,m in codec.named_modules()}
    output=directory/'monitor'/f'step_{step:05d}_{uuid.uuid4().hex[:10]}'
    try:
        codec.eval()
        evaluate.evaluate_models({arm:(codec,stats)},ce,data,mode='monitor',output_dir=output,
            checkpoint_records={},code_identity=source_identity())
    finally:
        for name,module in codec.named_modules(): module.training=modes[name]
        restore_rng(saved)
    helpers().assert_frozen_state(codec,ce,stats,frozen)
    files={str(p):sha256(p) for p in output.iterdir() if p.is_file()}
    atomic_json(receipt,dict(step=step,path=str(output),files=files,updated=now()),exclusive=True)


def train(arm, stop_after=None, resume=False):
    torch,np=original.tensor_runtime()
    target=cfg.UPDATES if stop_after is None else stop_after
    if arm not in cfg.ARMS or type(target)is not int or target<100 or target>cfg.UPDATES or target%100:
        raise ValueError('Declared arm and complete100 boundary within fixed budget required')
    device=configure_cuda(); core=core_identity(); guard_core(core)
    random.seed(cfg.SEED); np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED); torch.cuda.manual_seed_all(cfg.SEED)
    directory=inside(OUTPUT_ROOT/'training'/arm)
    # Source acceptance/model allocation happen before creating training outputs.
    codec,stats,ce,source=load_source(arm,device)
    parameters=helpers().configure_bs_only(codec,ce,stats)
    frozen=helpers().frozen_state(codec,ce,stats)
    optimizer=torch.optim.AdamW(parameters,lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY,
                                betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    data=make_data(device); start=0
    if not resume:
        directory.mkdir(parents=True,exist_ok=False)
    elif not (directory/'checkpoint_latest.pt').is_file():
        raise FileNotFoundError('Own latest rate checkpoint required for explicit resume')
    with process_lock(directory/'training.lock'):
        if resume:
            manifest=read_json(directory/'manifest.json'); path=directory/'checkpoint_latest.pt'
            if manifest.get('checkpoint_sha256')!=sha256(path): raise ValueError('Resume checkpoint hash changed')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            start=verify_saved(directory,payload,manifest,arm=arm,data=data,core=core,source=source,frozen=frozen)
            if target<start: raise ValueError('Cannot rewind an owned rate checkpoint')
            ensure_boundary_receipts(directory,manifest)
            original.strict_load(codec.decoder,payload['model'])
            optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
            optimizer_check(optimizer,scheduler,start); restore_rng(payload['rng'])
        else:
            atomic_json(directory/'initialization.json',dict(arm=arm,core=core,source_identity=source,
                frozen_identity=frozen,recipe=cfg.recipe(arm),fingerprint=cfg.fingerprint(arm),
                source_decoder_sha256=state_digest(codec.decoder),created=now(),
                original_optimizer_continued=False),exclusive=True)
            # A real zero-update checkpoint makes monitor0 failures resumable.
            save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,arm=arm,step=0,
                data=data,core=core,source=source,frozen=frozen,elapsed=0.,measured=0.)
        started=time.perf_counter(); update_seconds=[]; attempt_id=uuid.uuid4().hex
        try:
            monitor_if_missing(arm,start,codec,stats,ce,data,directory)
            if start==target:
                verify_files(source)
                return read_json(directory/'manifest.json')
            torch.cuda.reset_peak_memory_stats()
            for step in range(start+1,target+1):
                begin=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                hashes=[]; loss_values=[]; rates=[]; interferences=[]; predicted_energies=[]
                for micro in range(cfg.ACCUMULATION):
                    index=cfg.micro_index(step,micro)
                    sample=data.sample('codec_train',index)
                    out=forward_codec(codec,stats,ce,sample,None,('codec_train',index,0))
                    loss,metrics=helpers().rate_objective(out['H_hat'],sample['H_dl'],
                        alpha=cfg.ALPHA,downlink_snr_db=cfg.DL_SNR)
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite negative-rate loss')
                    (loss/cfg.ACCUMULATION).backward()
                    loss_values.append(float(loss.detach()))
                    rates.append(float(metrics['sum_rate_bps_hz'].detach().mean()))
                    interferences.append(float(metrics['interference_power_mean'].detach().mean()))
                    predicted_energies.append(float(metrics['predicted_dl_energy'].detach().mean()))
                    hashes.append(dict(channel=sample['sample_fingerprint'],**out['input_hashes']))
                norm=commit_update(codec,optimizer,scheduler,previous_updates=step-1)
                torch.cuda.synchronize(); update_seconds.append(time.perf_counter()-begin)
                if step%cfg.SAVE_EVERY==0:
                    helpers().assert_frozen_state(codec,ce,stats,frozen)
                    measured=sum(update_seconds[-100:])/len(update_seconds[-100:])
                    row=dict(step=step,attempt_id=attempt_id,resumed_from=start,
                        counters=cfg.counters(step),loss=sum(loss_values)/cfg.ACCUMULATION,
                        sum_rate_bps_hz=sum(rates)/cfg.ACCUMULATION,
                        interference_power_mean=sum(interferences)/cfg.ACCUMULATION,
                        predicted_dl_energy=sum(predicted_energies)/cfg.ACCUMULATION,
                        gradient_norm=float(norm),input_fingerprints=hashes,updated=now(),
                        mean_update_seconds=measured,ETA_seconds=measured*(cfg.UPDATES-step))
                    with (directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                    save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,arm=arm,step=step,
                        data=data,core=core,source=source,frozen=frozen,
                        elapsed=time.perf_counter()-started,measured=measured,final=step==target)
                    print(f'{arm} BS {step}/{cfg.UPDATES} rate={row["sum_rate_bps_hz"]:.6g} ETA={row["ETA_seconds"]:.0f}s',flush=True)
                monitor_if_missing(arm,step,codec,stats,ce,data,directory)
            guard_core(core); verify_files(source)
            return read_json(directory/'manifest.json')
        except BaseException as exc:
            atomic_json(directory/'failure.json',dict(status='failed',arm=arm,step=locals().get('step',start),
                last_saved_step=read_json(directory/'manifest.json').get('step') if (directory/'manifest.json').exists() else None,
                error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),updated=now()))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--run',action='store_true'); action.add_argument('--dry-run',action='store_true')
    parser.add_argument('--arm',choices=cfg.ARMS); parser.add_argument('--stop-after',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.resume and not args.run: parser.error('--resume requires --run')
    if args.dry_run:
        print(json.dumps(dict(root=str(OUTPUT_ROOT),recipe=cfg.recipe(args.arm or 'fdma'),
            weights_read=False,writes=False),indent=2)); return 0
    if not args.run: parser.print_help(); return 0
    if args.arm is None: parser.error('--run requires --arm')
    train(args.arm,args.stop_after,args.resume); return 0


if __name__=='__main__': raise SystemExit(main())
