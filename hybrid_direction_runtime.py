"""Owned Hybrid extension IO, exact parent loading and differentiable PHY."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile

from configs import twc_v1_k8_hybrid_direction_config as cfg
import ue_bs_rate_runtime as previous

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
original, helpers, physics = previous.original, previous.helpers, previous.physics
now, sha256, read_json = previous.now, previous.sha256, previous.read_json
state_digest, capture_rng, restore_rng = previous.state_digest, previous.capture_rng, previous.restore_rng
configure_cuda, environment = previous.configure_cuda, previous.environment
strict_load, atomic_torch_save = previous.strict_load, previous.atomic_torch_save
frozen_state, assert_frozen_state = previous.frozen_state, previous.assert_frozen_state


def source_identity():
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
        text=True).strip(), sha256={name: sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1', sha256=source_identity()['sha256'])


def guard_core(expected):
    from diagnose_twc_v1_k8_rate_headroom import source_guard
    source_guard()
    if core_identity() != expected:
        raise ValueError('Hybrid direction scientific core changed')


def inside(path):
    path = Path(path)
    target, allowed = path.resolve(), OUTPUT_ROOT.resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned Hybrid direction child path required')
    for item in (path, *path.parents):
        if item.is_symlink(): raise ValueError('Linked Hybrid output paths forbidden')
        if item.resolve() == ROOT: break
    return target


def atomic_json(path, value, *, exclusive=False):
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(cfg.plain(value), indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    fd, name = tempfile.mkstemp(prefix='hybrid_direction_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if exclusive: os.link(name, path)
        else: os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


@contextmanager
def process_lock(path):
    import fcntl
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_files(record):
    previous.verify_files(record)


def verify_source_record(arm, record):
    from diagnose_twc_v1_k8_rate_headroom import check_source_record
    if arm not in cfg.SOURCE_PINS or not isinstance(record, dict):
        raise ValueError('Exact frozen parent arm required')
    parent = record.get('parent_record', {})
    check_source_record(arm, parent)
    expected = dict(format=cfg.NAMESPACE+'_source_v1', arm=arm,
        source_namespace=cfg.SOURCE_NAMESPACE, source_kind='joint10000',
        production_evidence=True, base_ema_updates=cfg.BASE_EMA_UPDATES,
        base_encoder_state_sha256=cfg.SOURCE_PINS[arm]['encoder_state_sha256'],
        base_decoder_state_sha256=cfg.SOURCE_PINS[arm]['decoder_state_sha256'],
        base_statistics_state_sha256=cfg.SOURCE_PINS[arm]['statistics_state_sha256'],
        **{k: cfg.SOURCE_PINS[arm][k] for k in ('encoder_state_sha256',
            'decoder_state_sha256', 'state_sha256', 'statistics_state_sha256', 'CE_state_sha256')})
    if record.get('production_evidence') is not True or any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Inherited Hybrid source identity/counter changed')
    if record.get('frozen_files') != parent.get('frozen_files'):
        raise ValueError('Immutable parent file inventory changed')
    verify_files(record)


def load_source(arm, device):
    """Load exact joint10000 endpoints; never the older rate5000 initialization."""
    if arm not in cfg.SOURCE_PINS or str(device) != 'cuda:0':
        raise ValueError('Production parent loader requires declared arm and canonical cuda:0')
    from diagnose_twc_v1_k8_rate_headroom import source_guard, check_source_record
    import train_twc_v1_k8_ue_bs_rate as parent_trainer
    source_guard()
    codec, stats, ce, parent = parent_trainer.load_finetuned(arm, device)
    check_source_record(arm, parent)
    if int(stats.ema_updates) != cfg.BASE_EMA_UPDATES:
        raise ValueError('Warmstart must inherit actual EMA30000')
    actual = dict(encoder_state_sha256=state_digest(codec.encoder),
        decoder_state_sha256=state_digest(codec.decoder), state_sha256=state_digest(codec),
        statistics_state_sha256=state_digest(stats), CE_state_sha256=state_digest(ce))
    if any(actual[k] != cfg.SOURCE_PINS[arm][k] for k in actual):
        raise ValueError('Loaded parent model/statistics differ from audited joint terminal')
    record = dict(format=cfg.NAMESPACE+'_source_v1', arm=arm,
        source_namespace=cfg.SOURCE_NAMESPACE, source_kind='joint10000',
        production_evidence=True, parent_record=cfg.plain(parent),
        frozen_files=cfg.plain(parent['frozen_files']), base_ema_updates=int(stats.ema_updates),
        base_encoder_state_sha256=actual['encoder_state_sha256'],
        base_decoder_state_sha256=actual['decoder_state_sha256'],
        base_statistics_state_sha256=actual['statistics_state_sha256'], **actual)
    verify_source_record(arm, record)
    return codec, stats, ce, record


def make_data(device, role=None):
    if role not in (None, 'codec_train', 'final'):
        raise ValueError('Only fresh training and fixed final roles permitted')
    source = original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4,
            validation=cfg.MICRO_BATCH, test=1), device=str(device), cdl_model='C', speed=1.,
        seed_namespace=cfg.NAMESPACE)


def batch_random(role, index, device, *, gamma=None):
    import torch
    noise_seed = cfg.noise_seed(role, index, 'noise')
    if role == 'codec_train':
        scheduled = torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32, device=device)
        if gamma is None: gamma = scheduled
    elif gamma is None:
        raise ValueError('Explicit evaluation gamma required')
    supplied = torch.as_tensor(gamma)
    if supplied.is_complex() or supplied.dtype == torch.bool or supplied.requires_grad:
        raise ValueError('Nonlearned real public gamma required')
    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=device)
    if gamma.ndim == 0: gamma = gamma.expand(cfg.MICRO_BATCH)
    if (gamma.shape != (cfg.MICRO_BATCH,) or not bool(torch.isfinite(gamma).all()) or
            not bool((gamma[:, None] == gamma.new_tensor(cfg.GAMMAS)[None]).any(-1).all())):
        raise ValueError('Exactly the six prescribed gamma nodes required')
    if role == 'codec_train' and not torch.equal(gamma, scheduled):
        raise ValueError('Training gamma differs from balanced schedule')
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, 32, 256, 2,
        generator=gen, device=device).contiguous())/2**.5
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(role, index, 'polarity'))
    polarity = (2*torch.randint(2, (cfg.MICRO_BATCH, 8), generator=gen, device=device)-1).float()
    return gamma.detach(), noise, polarity


def configure_joint(codec, ce, stats):
    if physics().active_dimension(stats.layout) != 44:
        raise ValueError('Only Hybrid G4 may receive new optimizer updates')
    return previous.configure_joint(codec, ce, stats)


def forward_codec(codec, stats, ce, sample, gamma_db, noise_key):
    """Differentiable nominal PHY; true UL remains confined to simulation."""
    import torch
    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
    if not isinstance(noise_key, (tuple, list)) or len(noise_key) != 3:
        raise ValueError('Explicit role/index/gamma noise address required')
    role, index, gi = noise_key
    if type(gi) is not int or not 0 <= gi < len(cfg.GAMMAS):
        raise ValueError('Declared gamma index required')
    if role == 'codec_train':
        if gi != 0 or physics().active_dimension(stats.layout) != 44:
            raise ValueError('Only balanced Hybrid training is permitted')
        random_index = index
    elif role == 'final' and type(index) is int and 0 <= index < cfg.FINAL_BATCHES:
        random_index = index*len(cfg.GAMMAS)+gi
    else:
        raise ValueError('Bounded declared training or final role required')
    previous._assert_ce_frozen(ce, stats)
    if role == 'final' and (any(m.training for m in codec.modules()) or stats.training or
            any(p.requires_grad or p.grad is not None for p in codec.parameters())):
        raise ValueError('Every evaluation model and prior must be frozen')
    table, prior_state = previous._ce_table_state(stats), state_digest(stats)
    gamma, noise, polarity = batch_random(role, random_index, sample['H_ul'].device, gamma=gamma_db)
    if role == 'final' and not bool((gamma == cfg.GAMMAS[gi]).all()):
        raise ValueError('Final gamma differs from prescribed node')
    phy = physics()
    raw = codec.encoder(sample['H_dl'])
    wave = phy.build_waveform(raw, stats.layout, polarity=polarity)
    Y = phy.transmit(sample['H_ul'], wave.x_grid, noise, gamma)
    with torch.no_grad():
        initial = phy.fourier_ls(Y, wave.pilot)
        estimated = ce(Y, wave.pilot, initial, gamma_db=gamma)
        cevar, prior = stats.ce_variance(gamma), stats.nominal_power()
    recovered = phy.recover(Y, estimated, stats.layout, prior, cevar, gamma)
    out = codec.decoder(Y_rx=recovered['Y_res'], H_ul_est=recovered['H_ul_est'],
        soft=recovered['soft'], reliability=recovered['reliability'], pilots=wave.pilot,
        gamma_db=gamma, layout=stats.layout, prior=prior, ce_error=cevar)
    if torch.is_tensor(out): out = dict(H_hat=out)
    if not isinstance(out, dict) or 'H_hat' not in out:
        raise ValueError('Unchanged full-CSI decoder output required')
    if state_digest(stats) != prior_state or previous._ce_table_state(stats) != table:
        raise ValueError('Forward changed causal prior or frozen CE table')
    return dict(out, waveform=wave, soft=recovered['soft'], reliability=recovered['reliability'],
        H_ul_est=estimated, source_estimated_UL=estimated, recovered=recovered,
        layout=stats.layout, gamma_db=gamma,
        input_hashes=dict(standard_noise=_tensor_hash(noise), pilot_polarity=_tensor_hash(polarity),
            gamma=_tensor_hash(gamma), wave_active=_tensor_hash(wave.active), Y_rx=_tensor_hash(Y),
            estimated_UL=_tensor_hash(estimated), CE_variance=_tensor_hash(cevar), prior=_tensor_hash(prior)))
