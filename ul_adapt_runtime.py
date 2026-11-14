"""Pinned sources, balanced paired inputs and decoder-only forward graphs."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from configs import twc_v1_k8_ul_adapt_config as cfg
import ul_calibration_runtime as previous
import train_twc_v1_k8_icc_rate as rate

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
original = rate.original
now, sha256, read_json = previous.now, previous.sha256, previous.read_json
state_digest, capture_rng, restore_rng = rate.state_digest, rate.capture_rng, rate.restore_rng
configure_cuda, environment = previous.configure_cuda, previous.environment
physics, helpers = previous.physics, rate.helpers
atomic_torch_save, strict_load = original.atomic_torch_save, original.strict_load


def source_identity():
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), sha256={name:sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1', sha256=source_identity()['sha256'])


def guard_core(expected):
    previous.guard_core(cfg.PINNED_CALIBRATION_CORE)
    if core_identity() != expected:
        raise ValueError('UL adaptation source identity changed')


def inside(path):
    path = Path(path)
    target, allowed = path.resolve(), OUTPUT_ROOT.resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned UL adaptation child path required')
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Linked UL adaptation output paths are forbidden')
        if item.resolve() == ROOT:
            break
    return target


def atomic_json(path, value, *, exclusive=False):
    path = inside(path)
    data = json.dumps(cfg.plain(value), indent=2, sort_keys=True, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='record_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def process_lock(path):
    import fcntl
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _calibration_math():
    import twc_v1_k8_ul_calibration as calibrated
    return calibrated


def _checked_model(model):
    model = _calibration_math().validate_model(model)
    canonical = (json.dumps(model, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if hashlib.sha256(canonical).hexdigest() != cfg.CALIBRATION_MODEL_SHA256:
        raise ValueError('Cached receiver calibration is not the pinned completed model')
    return model


def _file(path, digest):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256(path) != digest:
        raise ValueError('Pinned read-only input artifact changed: '+str(path))
    return dict(path=str(path), sha256=digest)


def load_calibration():
    """Verify the completed calibration without refitting or changing it."""
    from evaluate_twc_v1_k8_ul_calibration import verify_calibration
    previous.guard_core(cfg.PINNED_CALIBRATION_CORE)
    directory = Path(cfg.CALIBRATION_DIR)
    manifest_file = _file(directory/'manifest.json', cfg.CALIBRATION_MANIFEST_SHA256)
    _file(directory/'model.json', cfg.CALIBRATION_MODEL_SHA256)
    manifest = verify_calibration(directory)
    if manifest.get('model_sha256') != cfg.CALIBRATION_MODEL_SHA256:
        raise ValueError('Completed CE calibration model identity changed')
    model = _checked_model(read_json(directory/'model.json'))
    files = [manifest_file]
    for name, digest in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Calibration artifact name escapes its directory')
        files.append(_file(directory/name, digest))
    return model, dict(directory=str(directory), model_sha256=cfg.CALIBRATION_MODEL_SHA256,
        manifest_sha256=cfg.CALIBRATION_MANIFEST_SHA256,
        common_CE_signature=manifest['common_CE_signature'],
        calibration_identity=cfg.calibration_identity(), frozen_files=files)


def verify_files(record):
    if not isinstance(record, dict) or not record.get('frozen_files'):
        raise ValueError('Read-only source and calibration evidence required')
    for entry in record['frozen_files']:
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
            raise ValueError('Exact frozen input file identity required')
        _file(entry['path'], entry['sha256'])
    _file(cfg.CALIBRATION_DIR/'manifest.json', cfg.CALIBRATION_MANIFEST_SHA256)
    _file(cfg.CALIBRATION_DIR/'model.json', cfg.CALIBRATION_MODEL_SHA256)
    if ('calibration_model_sha256' in record
            and record['calibration_model_sha256'] != cfg.CALIBRATION_MODEL_SHA256):
        raise ValueError('Source calibration model pin changed')
    if ('calibration_manifest_sha256' in record
            and record['calibration_manifest_sha256'] != cfg.CALIBRATION_MANIFEST_SHA256):
        raise ValueError('Source calibration manifest pin changed')


def load_initial(arm, method, device='cpu'):
    cfg.key(arm, method)
    _, calibration = load_calibration()
    codec, stats, ce, source = previous.load_source(arm, device)
    previous.verify_source_record(arm, source)
    if state_digest(ce) != calibration['common_CE_signature']:
        raise ValueError('Source CE differs from the locked calibration source')
    source = dict(source, base_decoder_state_sha256=state_digest(codec.decoder),
        calibration_model_sha256=cfg.CALIBRATION_MODEL_SHA256,
        calibration_manifest_sha256=cfg.CALIBRATION_MANIFEST_SHA256,
        calibration_identity=cfg.calibration_identity(), calibration_record=calibration,
        frozen_files=[*source['frozen_files'], *calibration['frozen_files']])
    for module in (codec, stats, ce):
        module.eval().requires_grad_(False)
    verify_files(source)
    return codec, stats, ce, source


def make_data(device):
    source = original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4,
            validation=cfg.MICRO_BATCH, test=1), device=str(device),
        cdl_model='C', speed=1., seed_namespace=cfg.DATA_NAMESPACE)


def batch_random(role, index, device, *, gamma=None):
    import torch
    # Validate both the role and its bounded stream index before allocating.
    noise_seed = cfg.noise_seed(role, index, 'noise')
    if role == 'codec_train':
        scheduled = torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32, device=device)
        if gamma is None:
            gamma = scheduled
    elif gamma is None:
        raise ValueError('Monitor/final gamma must be explicitly declared')
    supplied = torch.as_tensor(gamma)
    if supplied.is_complex() or supplied.dtype == torch.bool or supplied.requires_grad:
        raise ValueError('Public nonlearned real gamma required')
    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=device)
    if gamma.ndim == 0:
        gamma = gamma.expand(cfg.MICRO_BATCH)
    if (gamma.shape != (cfg.MICRO_BATCH,) or not bool(torch.isfinite(gamma).all())
            or not bool((gamma[:, None] == gamma.new_tensor(cfg.GAMMAS)[None]).any(-1).all())):
        raise ValueError('Exactly the prescribed six public gamma nodes are supported')
    if role == 'codec_train' and not torch.equal(gamma, scheduled):
        raise ValueError('Training gamma differs from the balanced fixed schedule')
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, 32, 256, 2,
        generator=generator, device=device).contiguous())/2**.5
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed(role, index, 'polarity'))
    polarity = (2*torch.randint(2, (cfg.MICRO_BATCH, 8),
        generator=generator, device=device)-1).float()
    return gamma.detach(), noise, polarity


def _assert_frozen_inputs(codec, stats, ce):
    for module in (codec.encoder, stats, ce):
        if (any(child.training for child in module.modules())
                or any(p.requires_grad or p.grad is not None for p in module.parameters())):
            raise ValueError('UE/CE/receiver history must remain frozen and in eval mode')


def forward_codec(codec, stats, ce, sample, gamma_db, noise_key, *, method, calibration=None):
    """Only the BS decoder builds a gradient graph; receiver inputs are public."""
    import torch
    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
    if method not in cfg.TRAIN_METHODS or not isinstance(noise_key, (tuple, list)) or len(noise_key) != 3:
        raise ValueError('Declared adaptation method and noise address required')
    role, index, gi = noise_key
    if type(gi) is not int or not 0 <= gi < len(cfg.GAMMAS):
        raise ValueError('Declared gamma index required')
    if role == 'codec_train':
        if gi != 0:
            raise ValueError('Training has one balanced gamma vector per micro index')
        random_index = index
    else:
        if (role not in ('monitor', 'final') or type(index) is not int
                or not 0 <= index < (cfg.MONITOR_BATCHES if role == 'monitor' else cfg.FINAL_BATCHES)):
            raise ValueError('Declared role-local monitor/final batch index required')
        random_index = index*len(cfg.GAMMAS)+gi
    _assert_frozen_inputs(codec, stats, ce)
    calibration = load_calibration()[0] if calibration is None else _checked_model(calibration)
    phy = physics()
    with torch.no_grad():
        gamma, noise, polarity = batch_random(role, random_index,
            sample['H_ul'].device, gamma=gamma_db)
        if role != 'codec_train' and not bool((gamma == cfg.GAMMAS[gi]).all()):
            raise ValueError('Monitor/final gamma does not match its declared node index')
        raw = codec.encoder(sample['H_dl'])
        wave = phy.build_waveform(raw, stats.layout, polarity=polarity)
        Y = phy.transmit(sample['H_ul'], wave.x_grid, noise, gamma)
        initial = phy.fourier_ls(Y, wave.pilot)
        estimated = ce(Y, wave.pilot, initial, gamma_db=gamma)
        cevar, prior = stats.ce_variance(gamma), stats.nominal_power()
        public = dict(Y=Y, estimated=estimated, ce_variance=cevar, prior=prior, pilots=wave.pilot)
        recovered = (phy.recover(Y, estimated, stats.layout, prior, cevar, gamma)
            if method == 'original' else _calibration_math().recover_variants(public,
                stats.layout, gamma, calibration)['moment_receiver'])
    result = codec.decoder(Y_rx=recovered['Y_res'], H_ul_est=recovered['H_ul_est'],
        soft=recovered['soft'], reliability=recovered['reliability'], pilots=wave.pilot,
        gamma_db=gamma, layout=stats.layout, prior=prior, ce_error=cevar)
    if torch.is_tensor(result):
        result = dict(H_hat=result)
    if not isinstance(result, dict) or 'H_hat' not in result:
        raise ValueError('Full CSI reconstruction required from the BS decoder')
    return dict(result, waveform=wave, soft=recovered['soft'], reliability=recovered['reliability'],
        H_ul_est=recovered['H_ul_est'], source_estimated_UL=estimated, recovered=recovered,
        layout=stats.layout,
        gamma_db=gamma, input_hashes=dict(standard_noise=_tensor_hash(noise),
            pilot_polarity=_tensor_hash(polarity), gamma=_tensor_hash(gamma),
            wave_active=_tensor_hash(wave.active), Y_rx=_tensor_hash(Y),
            estimated_UL=_tensor_hash(estimated), CE_variance=_tensor_hash(cevar), prior=_tensor_hash(prior)))
