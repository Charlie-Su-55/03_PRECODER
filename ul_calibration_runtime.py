"""Owned records and pinned source access for independent CE calibration.

No existing source/weight identity is relaxed. The two sampling roles have
separate replay addresses and noise streams; a production caller still needs
actual CUDA and verified original rate5000 checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile

from configs import twc_v1_k8_ul_calibration_config as cfg
import ul_propagation_runtime as previous

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
now, sha256, read_json = previous.now, previous.sha256, previous.read_json
state_digest, configure_cuda = previous.state_digest, previous.configure_cuda
frozen_signature = previous.frozen_signature
load_source, physics, observe = previous.load_source, previous.physics, previous.observe
verify_source_record, verify_source_records = previous.verify_source_record, previous.verify_source_records
environment = previous.previous.environment


def source_identity():
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), sha256={name: sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1', sha256=source_identity()['sha256'])


def guard_core(expected):
    previous.guard_core(cfg.PINNED_PROPAGATION_CORE)
    if core_identity() != expected:
        raise ValueError('Independent CE calibration source identity changed')


def inside(path):
    path = Path(path)
    target, allowed = path.resolve(), OUTPUT_ROOT.resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned CE calibration child path required')
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Linked CE calibration output paths are forbidden')
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


def make_data(device):
    source = previous.previous.legacy.rate.original.load_local(
        'utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=cfg.MICRO_BATCH,
            validation=cfg.MICRO_BATCH, test=1), device=str(device),
        cdl_model='C', speed=1., seed_namespace=cfg.DATA_NAMESPACE)


def batch_random(role, batch_index, gamma_index, device):
    import torch
    if (not isinstance(role, str) or role not in cfg.BATCHES_BY_ROLE or type(batch_index) is not int
            or not 0 <= batch_index < cfg.BATCHES_BY_ROLE[role]
            or type(gamma_index) is not int or not 0 <= gamma_index < len(cfg.GAMMAS)):
        raise ValueError('Declared calibration/validation batch/gamma address required')
    index = batch_index*len(cfg.GAMMAS)+gamma_index
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed(role, index, 'noise'))
    noise = torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, 32, 256, 2,
        device=device, generator=generator).contiguous())/2**.5
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed(role, index, 'polarity'))
    polarity = (2*torch.randint(2, (cfg.MICRO_BATCH, 8),
        device=device, generator=generator)-1).float()
    return noise, polarity
