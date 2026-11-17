#!/usr/bin/env python3
"""Train the final gamma-aware Hybrid-G4 model (FDMA is not touched).

Starts from the verified low-SNR interference Hybrid terminal, swaps in the
verified CE nmse65000 and its locked error table, adds the zero-initialized
gamma conditioner and a per-node power prior, then trains UE+BS with cosine LR
and a weight EMA. Saves every SAVE_EVERY updates; ``--resume`` continues from
the latest own checkpoint after checking recipe, code and source identity.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import time
import traceback

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt

TRAIN_DIR = cfg.OUTPUT_ROOT/'training'


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(cfg.plain(value), indent=2, sort_keys=True)+'\n').encode()
    fd, name = tempfile.mkstemp(prefix='.final_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if exclusive:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_save(payload, path):
    import torch
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='checkpoint_', suffix='.pt', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def ema_init(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def ema_update(ema, model, decay):
    import torch
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.is_floating_point() or v.is_complex():
                ema[k].mul_(decay).add_(v.detach(), alpha=1-decay)
            else:
                ema[k].copy_(v)


def make_optimizer(model):
    import torch
    groups = [dict(params=model.old_parameters(), lr=cfg.LR_BASE, name='source'),
              dict(params=model.new_parameters(), lr=cfg.LR_NEW, name='gamma_conditioner')]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY, betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cfg.lr_factor)
    return optimizer, scheduler


def source_summary(sources):
    hybrid = sources['hybrid'][3]
    ce = sources['ce65000'][1]
    return cfg.plain(dict(hybrid_trajectory=cfg.SOURCE_TRAJECTORY,
        hybrid_checkpoint_sha256=hybrid.get('sha256'), hybrid_state_sha256=hybrid.get('state_sha256'),
        hybrid_CE_state_sha256=hybrid.get('CE_state_sha256'),
        ce_trajectory=cfg.CE_TRAJECTORY, ce_checkpoint_sha256=ce.get('checkpoint_sha256'),
        ce_model_state_sha256=ce.get('model_state_sha256'), ce_calibration=sources['table65000'][1]))


def build_system(sources, device):
    hybrid, hybrid_stats, _, _ = sources['hybrid']
    ce65000, _ = sources['ce65000']
    table, _ = sources['table65000']
    model, stats = rt.build_final(hybrid, hybrid_stats, table)
    return dict(kind='final', codec=model.to(device), ce=ce65000, stats=stats.to(device))


def train(stop_after=None, resume=False, *, sources=None, data=None, device=None, out_dir=None,
          core=None, objective=None):
    """Production call uses no keyword fixtures; tests inject small CPU fixtures."""
    import torch
    import numpy as np
    target = cfg.UPDATES if stop_after is None else stop_after
    if not cfg.SAVE_EVERY <= target <= cfg.UPDATES or target % cfg.SAVE_EVERY:
        raise ValueError('Stop at a save boundary')
    fixture = sources is not None
    directory = Path(out_dir or TRAIN_DIR)
    if device is None:
        device = rt.base().configure_cuda()
    core = core or rt.source_identity()
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    sources = sources or rt.load_sources(device)
    data = data or rt.make_data(device)
    if objective is None:
        import twc_v1_k8_low_snr as low_snr_objective
        objective = lambda H_hat, H_dl, gamma: low_snr_objective.objective(H_hat, H_dl, gamma, cfg.TRAJECTORY_OBJECTIVE)
    system = build_system(sources, device)
    model, stats, ce = system['codec'], system['stats'], system['ce']
    model.train().requires_grad_(True)
    ce.eval().requires_grad_(False)
    stats.train()
    optimizer, scheduler = make_optimizer(model)
    identity = dict(recipe_fingerprint=cfg.fingerprint(), core_sha256=core['sha256'],
        source=source_summary(sources), production_evidence=not fixture)
    start, ema = 0, ema_init(model)
    if resume:
        manifest = json.loads((directory/'manifest.json').read_text())
        if {k: manifest.get(k) for k in identity} != identity:
            raise ValueError('Resume identity changed (recipe, code or source)')
        payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
        if rt.sha256(directory/'checkpoint_latest.pt') != manifest['checkpoint_sha256']:
            raise ValueError('Latest checkpoint bytes differ from manifest')
        model.load_state_dict(payload['model']); stats.load_state_dict(payload['statistics'])
        optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
        ema = {k: v.to(device) for k, v in payload['ema'].items()}
        rt.base().restore_rng(payload['rng'])
        start = payload['step']
        if target < start:
            raise ValueError('Cannot rewind')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        with torch.no_grad():
            probe = torch.zeros(2, 8, 32, 256, dtype=torch.complex64, device=device)
            probe[..., 0, :] = 1
            g = torch.tensor([0., 25.], device=device)
            same = torch.allclose(model.encoder(probe, g), model.encoder.base(probe), atol=1e-6, rtol=1e-5)
        if not same:
            raise ValueError('Zero-initialized gamma conditioner must reproduce the source encoder')
        atomic_json(directory/'initialization.json', dict(identity, recipe=cfg.recipe(), created=now(),
            zero_init_encoder_equivalence=True), exclusive=True)
    started, timings = time.perf_counter(), []
    log = directory/'progress.jsonl'
    try:
        for step in range(start+1, target+1):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            n = rt.physics().active_dimension(stats.layout)
            node_power = torch.zeros(6, 8, n, device=device)
            node_frames = torch.zeros(6, dtype=torch.int64, device=device)
            sums = dict(loss=0., sum_rate_bps_hz=0., direction_error=0., key_energy_fraction=0.)
            low_rate, low_count = 0., 0
            for micro in range(cfg.ACCUMULATION):
                index = cfg.micro_index(step, micro)
                sample = data.sample('codec_train', index)
                gamma = torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32, device=device)
                noise, polarity = rt.batch_random('codec_train', index, device)
                out = rt.forward(system, sample, gamma, noise, polarity)
                loss, metrics = objective(out['H_hat'], out['H_dl'], gamma)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                (loss/cfg.ACCUMULATION).backward()
                power = out['wave'].active.detach().abs().square()
                node = rt.node_index(gamma)
                node_power.index_add_(0, node, power)
                node_frames += torch.bincount(node, minlength=6)
                rate = metrics['sum_rate_bps_hz'].detach()
                sums['loss'] += float(loss.detach())/cfg.ACCUMULATION
                sums['sum_rate_bps_hz'] += float(rate.mean())/cfg.ACCUMULATION
                sums['direction_error'] += float(metrics['direction_error'].detach().mean())/cfg.ACCUMULATION
                key = out['wave'].key.abs().square().sum(-1).mean(-1)/out['wave'].active.abs().square().sum(-1).mean(-1)
                sums['key_energy_fraction'] += float(key.detach().mean())/cfg.ACCUMULATION
                low = gamma <= 10
                low_rate += float(rate[low].sum()); low_count += int(low.sum())
            params = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError('Nonfinite gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
            optimizer.step(); scheduler.step()
            stats.commit_power(node_power, node_frames)
            ema_update(ema, model, cfg.WEIGHT_EMA_DECAY)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.perf_counter()-begin)
            if step % cfg.SAVE_EVERY == 0 or step == target:
                measured = sum(timings[-100:])/len(timings[-100:])
                row = dict(step=step, **sums, low_gamma_sum_rate=low_rate/max(1, low_count),
                    gradient_norm=float(norm), lr=[g['lr'] for g in optimizer.param_groups],
                    seconds_per_update=measured, ETA_seconds=measured*(cfg.UPDATES-step), updated=now())
                with log.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, sort_keys=True)+'\n')
                payload = dict(step=step, model=model.state_dict(), ema={k: v.detach().cpu() for k, v in ema.items()},
                    statistics=stats.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=rt.base().capture_rng(), **identity)
                atomic_save(payload, directory/'checkpoint_latest.pt')
                manifest = dict(identity, step=step, status='completed' if step == cfg.UPDATES else 'running',
                    checkpoint_sha256=rt.sha256(directory/'checkpoint_latest.pt'),
                    progress_sha256=rt.sha256(log), FDMA_updates=0, CE_updates=0,
                    elapsed_seconds=time.perf_counter()-started, updated=now())
                atomic_json(directory/'manifest.json', manifest)
                if step == cfg.UPDATES:
                    atomic_json(directory/'terminal_receipt.json', manifest, exclusive=True)
                print(f'final {step}/{cfg.UPDATES} rate={row["sum_rate_bps_hz"]:.4f} '
                      f'low={row["low_gamma_sum_rate"]:.4f} key_frac={row["key_energy_fraction"]:.4f} '
                      f'ETA={row["ETA_seconds"]/3600:.2f}h', flush=True)
        return json.loads((directory/'manifest.json').read_text())
    except BaseException as exc:
        if directory.is_dir():
            atomic_json(directory/'failure.json', dict(status='failed', step=locals().get('step', start),
                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), updated=now()))
        raise


def load_terminal_ema(device, *, sources=None, directory=None):
    """Return the fixed terminal EMA system (evaluation entry point)."""
    import torch
    directory = Path(directory or TRAIN_DIR)
    manifest = json.loads((directory/'terminal_receipt.json').read_text())
    if manifest.get('status') != 'completed' or manifest.get('step') != cfg.UPDATES:
        raise ValueError('Completed fixed terminal required')
    if rt.sha256(directory/'checkpoint_latest.pt') != manifest['checkpoint_sha256']:
        raise ValueError('Terminal checkpoint bytes changed')
    if manifest['recipe_fingerprint'] != cfg.fingerprint():
        raise ValueError('Terminal recipe differs from current configuration')
    sources = sources or rt.load_sources(device)
    if manifest['source'] != source_summary(sources):
        raise ValueError('Terminal source identity differs from the verified sources')
    payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    system = build_system(sources, device)
    system['codec'].load_state_dict(payload['ema'])
    system['stats'].load_state_dict(payload['statistics'])
    for module in (system['codec'], system['stats'], system['ce']):
        module.eval().requires_grad_(False)
    return system, manifest, sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', required=True)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(train(args.stop_after, args.resume), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
