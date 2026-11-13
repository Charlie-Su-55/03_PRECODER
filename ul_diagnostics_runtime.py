"""Read-only source bridge and owned IO for zero-update UL diagnostics."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile

from configs import twc_v1_k8_ul_diagnostics_config as cfg
import icc_alpha_runtime as pinned
import w_residual_runtime as legacy

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
now, sha256, read_json = legacy.now, legacy.sha256, legacy.read_json
state_digest, configure_cuda = legacy.state_digest, legacy.configure_cuda
environment = legacy.environment


def source_identity():
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), sha256={name: sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1', sha256=source_identity()['sha256'])


def guard_core(expected):
    legacy.guard_core(cfg.PINNED_W_CORE)
    if core_identity() != expected:
        raise ValueError('UL diagnostic source identity changed')


def inside(path):
    path = Path(path)
    target, allowed = path.resolve(), OUTPUT_ROOT.resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned UL diagnostic child path required')
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Linked UL diagnostic output paths are forbidden')
        if item.resolve() == ROOT:
            break
    return target


def atomic_json(path, value, *, exclusive=False):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(cfg.plain(value), indent=2, sort_keys=True, allow_nan=False)+'\n'
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


def verify_source_record(arm, record):
    if arm not in cfg.ARMS:
        raise ValueError('Declared diagnostic physical arm required')
    pinned.validate_model_record(record, cfg.SOURCE_KIND, arm)


def verify_source_records(records):
    if set(records) != set(cfg.ARMS):
        raise ValueError('Both pinned diagnostic source records required')
    for arm, record in records.items():
        verify_source_record(arm, record)


def load_source(arm, device):
    codec, stats, ce, record = pinned.load_model(cfg.SOURCE_KIND, arm, device)
    verify_source_record(arm, record)
    codec.eval().requires_grad_(False)
    stats.eval().requires_grad_(False)
    ce.eval().requires_grad_(False)
    return codec, stats, ce, record


def frozen_signature(codec, stats, ce):
    modules = dict(codec=codec, statistics=stats, CE=ce)
    for module in modules.values():
        if any(item.training for item in module.modules()):
            raise ValueError('Diagnostic modules must all remain in evaluation mode')
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError('Diagnostic parameters must all remain frozen')
    return {name: state_digest(module) for name, module in modules.items()}


def make_data(device):
    source = legacy.rate.original.load_local('utils/twc_v1_k8_replayable_data.py',
        '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
        device=str(device), cdl_model='C', speed=1., seed_namespace=cfg.DATA_NAMESPACE)


def batch_random(batch_index, gamma_index, device):
    import torch
    if (type(batch_index) is not int or not 0 <= batch_index < cfg.BATCHES
            or type(gamma_index) is not int or not 0 <= gamma_index < len(cfg.GAMMAS)):
        raise ValueError('Declared UL diagnostic batch/gamma address required')
    index = batch_index*len(cfg.GAMMAS)+gamma_index
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed('diagnostic', index, 'noise'))
    noise = torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, 32, 256, 2,
        device=device, generator=generator).contiguous())/2**.5
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed('diagnostic', index, 'polarity'))
    polarity = (2*torch.randint(2, (cfg.MICRO_BATCH, 8), device=device, generator=generator)-1).float()
    return noise, polarity


def physics():
    return legacy.rate.original.physics()


def observe(raw, stats, ce, sample, gamma, noise, polarity):
    """Build one paid waveform/Y; true UL is consumed only by the simulator."""
    phy = physics()
    wave = phy.build_waveform(raw, stats.layout, polarity=polarity)
    Y = phy.transmit(sample['H_ul'], wave.x_grid, noise, gamma)
    initial = phy.fourier_ls(Y, wave.pilot)
    estimated = ce(Y, wave.pilot, initial, gamma_db=gamma)
    variance = stats.ce_variance(gamma)
    prior = stats.nominal_power()
    ledger = phy.energy_ledger(wave, stats.layout)
    import torch
    for field, expected in (('pilot', 16.), ('data', 48.), ('total', 64.)):
        value = ledger[field]
        if not bool(torch.isfinite(value).all() &
                    torch.isclose(value, torch.full_like(value, expected), rtol=1e-5, atol=1e-5).all()):
            raise FloatingPointError('Diagnostic paid '+field+' energy changed')
    return dict(wave=wave, Y=Y, estimated=estimated, ce_variance=variance, prior=prior)


def recover_variants(observation, true_ul, layout, gamma):
    """The sole privileged branch substitutes true UL and zero CE uncertainty.

    Both branches retain the identical noisy Y, historical prior and receiver.
    Realized transmit power never replaces the historical receiver prior.
    """
    import torch
    phy = physics()
    Y, prior = observation['Y'], observation['prior']
    inputs = dict(
        estimated_ul=(observation['estimated'], observation['ce_variance']),
        true_ul_diagnostic=(true_ul, torch.zeros_like(observation['ce_variance'])))
    recovered = {}
    for variant in cfg.VARIANTS:
        channel, variance = inputs[variant]
        recovered[variant] = phy.recover(Y, channel, layout, prior, variance, gamma)
    return recovered, inputs
