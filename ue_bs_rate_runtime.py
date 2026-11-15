"""Differentiable UE/MAC/recovery/BS graph with a frozen paid-pilot CE.

The forward pass never commits receiver statistics. A caller commits the
effective-batch transmitted-power observation only after an optimizer update.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from configs import twc_v1_k8_ue_bs_rate_config as cfg
import ul_adapt_runtime as previous
import train_twc_v1_k8_ul_adapt as benchmark

ROOT,OUTPUT_ROOT=cfg.ROOT,cfg.OUTPUT_ROOT
original=previous.original
rate=previous.rate
now,sha256,read_json=previous.now,previous.sha256,previous.read_json
state_digest,capture_rng,restore_rng=previous.state_digest,previous.capture_rng,previous.restore_rng
configure_cuda,environment=previous.configure_cuda,previous.environment
physics,helpers=previous.physics,previous.helpers
atomic_torch_save,strict_load=previous.atomic_torch_save,previous.strict_load


def source_identity():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={name:sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1',sha256=source_identity()['sha256'])


def guard_core(expected):
    previous.guard_core(cfg.CONTROL_CORE)
    if core_identity()!=expected:
        raise ValueError('UE-BS joint scientific core changed')


def inside(path):
    path=Path(path)
    target,allowed=path.resolve(),OUTPUT_ROOT.resolve()
    if target==allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned UE-BS joint child path required')
    for item in (path,*path.parents):
        if item.is_symlink():raise ValueError('Linked joint output paths forbidden')
        if item.resolve()==ROOT:break
    return target


def atomic_json(path,value,*,exclusive=False):
    path=inside(path)
    data=json.dumps(cfg.plain(value),indent=2,sort_keys=True,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='joint_record_',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        if exclusive:os.link(name,path)
        else:os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@contextmanager
def process_lock(path):
    import fcntl
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _file(path,digest):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or sha256(path)!=digest:
        raise ValueError('Pinned read-only joint source artifact changed: '+str(path))
    return dict(path=str(path),sha256=digest)


def verify_files(record):
    if not isinstance(record,dict) or not record.get('frozen_files'):
        raise ValueError('Explicit read-only source/benchmark evidence required')
    for entry in record['frozen_files']:
        if not isinstance(entry,dict) or set(entry)!={'path','sha256'}:
            raise ValueError('Exact read-only file identity required')
        _file(entry['path'],entry['sha256'])


def _cuda_source_device(device):
    # The historical source records pin execution as cuda:0. A synthetic CPU
    # caller must inject a fixture loader rather than reinterpret that record.
    if str(device)!='cuda:0':
        raise ValueError('Production source loader requires canonical cuda:0; inject an explicit CPU fixture loader')


def _record_states(codec,stats,ce,record,*,arm,method,step):
    return dict(record,arm=arm,method=method,step=step,
        production_evidence=True,evidence_kind='production',
        source_kind='rate5000' if method in ('source','joint') else 'archived_bs_only10000',
        state_sha256=state_digest(codec),encoder_state_sha256=state_digest(codec.encoder),
        decoder_state_sha256=state_digest(codec.decoder),
        statistics_state_sha256=state_digest(stats),CE_state_sha256=state_digest(ce),
        base_ema_updates=int(stats.ema_updates))


def load_initial(arm,method='joint',device='cpu'):
    cfg.key(arm,method);_cuda_source_device(device)
    previous.guard_core(cfg.CONTROL_CORE)
    codec,stats,ce,record=previous.load_initial(arm,'original',device)
    base=cfg.plain(record)
    record=_record_states(codec,stats,ce,record,arm=arm,method=method,step=5000)
    record.update(base_encoder_state_sha256=record['encoder_state_sha256'],
        base_decoder_state_sha256=record['decoder_state_sha256'],
        base_statistics_state_sha256=record['statistics_state_sha256'],
        source_identity=base,base_source_identity=base)
    verify_files(record)
    return codec,stats,ce,record


def verify_control_files(arm):
    cfg.key(arm,'bs_only')
    previous.guard_core(cfg.CONTROL_CORE)
    directory=cfg.CONTROL_ROOT/('training/original_'+arm)
    files=[_file(directory/name,digest) for name,digest in cfg.CONTROL_FILE_SHA256S[arm].items()]
    terminal=benchmark.verify_terminal_record(arm,'original')
    pins=cfg.SOURCE_CONTROL_PINS[arm]
    for key in ('checkpoint_sha256','counters','training_trace'):
        if terminal.get(key)!=pins[key]:
            raise ValueError('Archived BS-only terminal identity changed: '+key)
    return dict(arm=arm,method='bs_only',identity=cfg.control_identity(arm),
        frozen_files=files,step=10000,training_trace=pins['training_trace'])


def verify_control_record(record,arm):
    pins=cfg.SOURCE_CONTROL_PINS[arm]
    for key in ('checkpoint_sha256','manifest_sha256','terminal_receipt_sha256',
            'initialization_sha256','progress_sha256','encoder_state_sha256',
            'decoder_state_sha256','state_sha256','statistics_state_sha256',
            'CE_state_sha256','step','counters','training_trace'):
        if record.get(key)!=pins[key]:
            raise ValueError('Pinned loaded BS-only benchmark changed: '+key)
    if record.get('core')!=cfg.CONTROL_CORE:
        raise ValueError('Pinned BS-only benchmark scientific core changed')


def load_control(arm,device='cpu'):
    _cuda_source_device(device)
    accepted=verify_control_files(arm)
    codec,stats,ce,record=benchmark.load_finetuned(arm,'original',device)
    # Historical loaders already verify decoder/CE/prior bytes. Bind their
    # accepted loaded result to the exact independently audited report.
    record=_record_states(codec,stats,ce,record,arm=arm,method='bs_only',step=10000)
    verify_control_record(record,arm)
    record.update(pinned_control_identity=cfg.control_identity(arm),
        base_source_identity=cfg.plain(record['source_identity']),
        frozen_files=[*record['frozen_files'],*accepted['frozen_files']])
    verify_files(record)
    return codec,stats,ce,record


def load_variant(arm,method,device='cpu'):
    cfg.key(arm,method)
    if method=='source':return load_initial(arm,'source',device)
    if method=='bs_only':return load_control(arm,device)
    raise ValueError('Joint terminal loader belongs to the owned joint trainer')


def stable_source_identity(record):
    """Extract an explicit original source binding without a None fallback."""
    value=record
    for _ in range(4):
        if not isinstance(value,dict):break
        if isinstance(value.get('base_source_identity'),dict):
            return cfg.plain(value['base_source_identity'])
        value=value.get('source_identity')
    raise ValueError('Explicit pinned common source identity required')


def make_data(device,role='final'):
    if role=='codec_train':return previous.make_data(device)
    if role not in ('monitor','final'):
        raise ValueError('Declared training, monitor or final data role required')
    source=original.load_local('utils/twc_v1_k8_replayable_data.py','_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4,codec_train=cfg.MICRO_BATCH,ce_calibration=4,
            validation=cfg.MICRO_BATCH,test=1),device=str(device),cdl_model='C',
        speed=1.,seed_namespace=cfg.DATA_NAMESPACE)


def make_training_data(device):
    return make_data(device,role='codec_train')


def batch_random(role,index,device,*,gamma=None):
    import torch
    noise_seed=cfg.noise_seed(role,index,'noise')
    if role=='codec_train':
        scheduled=torch.tensor(cfg.gamma_schedule(index),dtype=torch.float32,device=device)
        if gamma is None:gamma=scheduled
    elif gamma is None:raise ValueError('Explicit declared evaluation gamma required')
    supplied=torch.as_tensor(gamma)
    if supplied.is_complex() or supplied.dtype==torch.bool or supplied.requires_grad:
        raise ValueError('Nonlearned real public evaluation gamma required')
    gamma=torch.as_tensor(gamma,dtype=torch.float32,device=device)
    if gamma.ndim==0:gamma=gamma.expand(cfg.MICRO_BATCH)
    if (gamma.shape!=(cfg.MICRO_BATCH,) or not bool(torch.isfinite(gamma).all())
            or not bool((gamma[:,None]==gamma.new_tensor(cfg.GAMMAS)[None]).any(-1).all())):
        raise ValueError('Exactly the prescribed public six gamma nodes required')
    if role=='codec_train' and not torch.equal(gamma,scheduled):
        raise ValueError('Joint training gamma differs from archived balanced schedule')
    gen=torch.Generator(device=device).manual_seed(noise_seed)
    noise=torch.view_as_complex(torch.randn(cfg.MICRO_BATCH,32,256,2,
        generator=gen,device=device).contiguous())/2**.5
    gen=torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'polarity'))
    polarity=(2*torch.randint(2,(cfg.MICRO_BATCH,8),generator=gen,device=device)-1).float()
    return gamma.detach(),noise,polarity


def _tensor_hash(value):
    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash as function
    return function(value)


def _ce_table_state(stats):
    return dict(ce_errors=_tensor_hash(stats.ce_errors),ce_calibrated=_tensor_hash(stats.ce_calibrated),
        layout=cfg.plain(stats.layout.to_dict()))


def _assert_ce_frozen(ce,stats):
    if (any(m.training for m in ce.modules())
            or any(p.requires_grad or p.grad is not None for p in ce.parameters())
            or any(p.requires_grad or p.grad is not None for p in stats.parameters())):
        raise ValueError('CE and statistics parameters must remain frozen')


def frozen_state(codec,ce,stats):
    _assert_ce_frozen(ce,stats)
    return dict(ce=state_digest(ce),**_ce_table_state(stats))


def assert_frozen_state(codec,ce,stats,expected):
    if frozen_state(codec,ce,stats)!=expected:
        raise ValueError('Frozen CE, calibrated CE table or physical layout changed')


def configure_joint(codec,ce,stats):
    import torch
    if not hasattr(codec,'encoder') or not hasattr(codec,'decoder'):
        raise ValueError('Explicit UE encoder and BS decoder required')
    codec.train().requires_grad_(True)
    ce.eval().requires_grad_(False)
    stats.train().requires_grad_(False)
    for module in (ce,stats):
        for p in module.parameters():p.grad=None
    parameters=tuple(codec.parameters())
    expected=tuple((*codec.encoder.parameters(),*codec.decoder.parameters()))
    if (not parameters or len(set(map(id,parameters)))!=len(parameters)
            or set(map(id,parameters))!=set(map(id,expected))
            or any(not p.requires_grad for p in parameters)):
        raise ValueError('Optimizer scope must be exactly independent UE and BS parameters')
    _assert_ce_frozen(ce,stats)
    return parameters


def forward_codec(codec,stats,ce,sample,gamma_db,noise_key,*,method='joint',calibration=None):
    """No prior update or truth input to receiver/BS; trainable data path intact."""
    import torch
    if method not in cfg.METHODS or not isinstance(noise_key,(tuple,list)) or len(noise_key)!=3:
        raise ValueError('Declared UE-BS method and noise address required')
    role,index,gi=noise_key
    if type(gi) is not int or not 0<=gi<len(cfg.GAMMAS):
        raise ValueError('Declared gamma index required')
    if role=='codec_train':
        if method!='joint' or gi!=0:
            raise ValueError('Only joint training uses one balanced vector per micro')
        random_index=index
    else:
        if (role not in ('monitor','final') or type(index) is not int
                or not 0<=index<(cfg.MONITOR_BATCHES if role=='monitor' else cfg.FINAL_BATCHES)):
            raise ValueError('Declared bounded monitor/final local batch index required')
        random_index=index*len(cfg.GAMMAS)+gi
    _assert_ce_frozen(ce,stats)
    if method!='joint' and (any(m.training for m in codec.modules())
            or any(p.requires_grad or p.grad is not None for p in codec.parameters()) or stats.training):
        raise ValueError('Source and archived benchmark must be frozen for evaluation')
    table_before=_ce_table_state(stats)
    prior_state=state_digest(stats)
    gamma,noise,polarity=batch_random(role,random_index,sample['H_ul'].device,gamma=gamma_db)
    if role!='codec_train' and not bool((gamma==cfg.GAMMAS[gi]).all()):
        raise ValueError('Evaluation gamma differs from its prescribed node index')
    phy=physics()
    # This is the essential graph change: data symbols, the physical MAC and
    # estimated-channel recovery must stay differentiable back to the UE.
    raw=codec.encoder(sample['H_dl'])
    wave=phy.build_waveform(raw,stats.layout,polarity=polarity)
    Y=phy.transmit(sample['H_ul'],wave.x_grid,noise,gamma)
    with torch.no_grad():
        initial=phy.fourier_ls(Y,wave.pilot)
        estimated=ce(Y,wave.pilot,initial,gamma_db=gamma)
        cevar,prior=stats.ce_variance(gamma),stats.nominal_power()
    recovered=phy.recover(Y,estimated,stats.layout,prior,cevar,gamma)
    result=codec.decoder(Y_rx=recovered['Y_res'],H_ul_est=recovered['H_ul_est'],
        soft=recovered['soft'],reliability=recovered['reliability'],pilots=wave.pilot,
        gamma_db=gamma,layout=stats.layout,prior=prior,ce_error=cevar)
    if torch.is_tensor(result):result=dict(H_hat=result)
    if not isinstance(result,dict) or 'H_hat' not in result:
        raise ValueError('Full CSI output required from unchanged BS decoder')
    if state_digest(stats)!=prior_state or _ce_table_state(stats)!=table_before:
        raise ValueError('Forward pass changed causal prior or frozen CE table')
    return dict(result,waveform=wave,soft=recovered['soft'],reliability=recovered['reliability'],
        H_ul_est=estimated,source_estimated_UL=estimated,recovered=recovered,
        layout=stats.layout,gamma_db=gamma,
        input_hashes=dict(standard_noise=_tensor_hash(noise),pilot_polarity=_tensor_hash(polarity),
            gamma=_tensor_hash(gamma),wave_active=_tensor_hash(wave.active),Y_rx=_tensor_hash(Y),
            estimated_UL=_tensor_hash(estimated),CE_variance=_tensor_hash(cevar),prior=_tensor_hash(prior)))


forward_joint=forward_codec
