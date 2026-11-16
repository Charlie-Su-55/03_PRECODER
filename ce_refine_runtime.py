"""Strict CE-only source bridge and isolated physical/data runtime."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
from configs import twc_v1_k8_ce_refine_config as cfg
import train_twc_v1_k8_icc_aligned as original

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
now, sha256, read_json = original.now, original.sha256, original.read_json
state_digest, strict_load = original.state_digest, original.strict_load
capture_rng, restore_rng = original.capture_rng, original.restore_rng
configure_cuda, environment = original.configure_cuda, original.environment


def source_identity():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={p:sha256(ROOT/p) for p in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1',sha256=source_identity()['sha256'])


def guard_core(core):
    if original.core_identity() != cfg.SOURCE_CORE:
        raise ValueError('Original pinned physical/model/data core changed')
    if core_identity() != core: raise ValueError('CE refinement scientific core changed')


def inside(path):
    path=Path(path).absolute(); resolved=path.resolve(); allowed=cfg.OUTPUT_ROOT.resolve()
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise ValueError('Owned CE-refinement child path required')
    for item in (path,*path.parents):
        if item.is_symlink(): raise ValueError('Linked CE paths forbidden')
        if item.resolve() == ROOT: break
    return resolved


def atomic_json(path, value, *, exclusive=False):
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(cfg.plain(value),sort_keys=True,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8') as s:s.write(raw);s.flush();os.fsync(s.fileno())
        return
    fd,name=tempfile.mkstemp(prefix='record_',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as s:s.write(raw);s.flush();os.fsync(s.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@contextmanager
def process_lock(path):
    import fcntl
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as s:
        fcntl.flock(s.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(s.fileno(),fcntl.LOCK_UN)


def regular(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink(): raise ValueError('Regular unlinked source file required')
    for p in path.parents:
        if p.is_symlink(): raise ValueError('Linked source parent forbidden')
    return path


def verify_source(record):
    if not isinstance(record,dict) or not record.get('frozen_files'):
        raise ValueError('Complete immutable source record required')
    if not record.get('fixture',False):
        expected=dict(source_commit=cfg.SOURCE_COMMIT,source_core=cfg.SOURCE_CORE,
            checkpoint_sha256=cfg.SOURCE_CHECKPOINT_SHA256,terminal_sha256=cfg.SOURCE_TERMINAL_SHA256,
            source_state_sha256=cfg.SOURCE_STATE_SHA256,source_updates=cfg.SOURCE_UPDATES)
        if any(record.get(k)!=v for k,v in expected.items()):
            raise ValueError('Pinned original CE identity changed')
    for entry in record['frozen_files']:
        if sha256(regular(entry['path'])) != entry['sha256']:
            raise ValueError('Immutable CE source file changed')


def load_source(device='cpu'):
    import torch
    from models.twc_v1_k8_icc_aligned import K8ICCAlignedPilotUNet
    if original.core_identity() != cfg.SOURCE_CORE: raise ValueError('Pinned original core changed')
    root=cfg.SOURCE_ROOT
    checkpoint=regular(root/'checkpoint_latest.pt');terminal=regular(root/'terminal_receipt.json')
    manifest=regular(root/'manifest.json');initial=regular(root/'initialization.json')
    if sha256(checkpoint)!=cfg.SOURCE_CHECKPOINT_SHA256 or sha256(terminal)!=cfg.SOURCE_TERMINAL_SHA256:
        raise ValueError('Pinned source CE bytes changed')
    receipt=read_json(terminal);current=read_json(manifest)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (receipt!=current or receipt.get('status')!='completed' or receipt.get('step')!=cfg.SOURCE_UPDATES
            or receipt.get('checkpoint_sha256')!=cfg.SOURCE_CHECKPOINT_SHA256
            or payload.get('format')!=original.cfg.NAMESPACE or payload.get('component')!='ce'
            or payload.get('step')!=cfg.SOURCE_UPDATES or payload.get('core')!=cfg.SOURCE_CORE
            or payload.get('training_code',{}).get('commit')!=cfg.SOURCE_COMMIT
            or payload.get('recipe')!=original.cfg.recipe('ce')
            or payload.get('fingerprint')!=original.cfg.fingerprint('ce')
            or payload.get('counters')!=original.cfg.counters('ce',cfg.SOURCE_UPDATES)
            or payload.get('statistics') is not None or payload.get('ce_identity') is not None
            or payload.get('data_protocol',{}).get('seed_namespace')!=original.cfg.NAMESPACE):
        raise ValueError('Original CE source provenance/recipe changed')
    original.verify_training_receipts(root,payload,current)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(original.cfg.INITIAL_CE_SEED)
        model=K8ICCAlignedPilotUNet().to(device)
    strict_load(model,payload['model'])
    record=dict(source_commit=cfg.SOURCE_COMMIT,source_core=cfg.SOURCE_CORE,
        checkpoint_sha256=cfg.SOURCE_CHECKPOINT_SHA256,terminal_sha256=cfg.SOURCE_TERMINAL_SHA256,
        source_state_sha256=state_digest(model),source_updates=cfg.SOURCE_UPDATES,
        fixture=False,frozen_files=[dict(path=str(p),sha256=sha256(p)) for p in
            (checkpoint,terminal,manifest,initial)])
    verify_source(record)
    return model.eval().requires_grad_(False),record


def make_data(device):
    src=original.load_local('utils/twc_v1_k8_replayable_data.py','_icc_aligned_data')
    return src.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4,codec_train=4,ce_calibration=4,validation=4,test=1),
        device=str(device),cdl_model='C',speed=1.,seed_namespace=cfg.DATA_NAMESPACE)


def physics():return original.physics()


def batch_random(role,index,device,*,gamma=None):
    import torch
    if gamma is None:
        if role!='ce_train':raise ValueError('Evaluation gamma must be explicit')
        gamma=cfg.gamma_schedule(index)
    supplied=torch.as_tensor(gamma)
    if supplied.is_complex() or supplied.dtype==torch.bool or supplied.requires_grad:
        raise ValueError('Real nonlearned feedback gamma required')
    gamma=torch.as_tensor(gamma,device=device,dtype=torch.float32)
    if gamma.ndim==0:gamma=gamma.expand(cfg.MICRO_BATCH)
    if gamma.shape!=(cfg.MICRO_BATCH,) or not bool(torch.isfinite(gamma).all() & (gamma>=0).all() & (gamma<=25).all()):
        raise ValueError('Public gamma shape/range changed')
    if role=='ce_train' and not torch.equal(gamma, gamma.new_tensor(cfg.gamma_schedule(index))):
        raise ValueError('Balanced training gamma changed')
    gen=torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'noise'))
    noise=torch.view_as_complex(torch.randn(cfg.MICRO_BATCH,32,256,2,device=device,generator=gen).contiguous())/2**.5
    gen=torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'polarity'))
    polarity=(2*torch.randint(2,(cfg.MICRO_BATCH,8),device=device,generator=gen)-1).float()
    return gamma,noise,polarity


def pilot_forward(model,sample,gamma,noise,polarity):
    phy=physics()
    pilots=phy.known_pilots(cfg.MICRO_BATCH,device=sample['H_ul'].device,polarity=polarity)
    Y=phy.transmit(sample['H_ul'],pilots,noise,gamma)
    initial=phy.fourier_ls(Y,pilots)
    return model(Y,pilots,initial,gamma_db=gamma),initial,Y,pilots
