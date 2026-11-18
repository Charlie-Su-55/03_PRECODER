#!/usr/bin/env python3
"""Pairing-aware fine-tuning of the final Hybrid (Hybrid only; FDMA frozen, CE frozen).

The final Hybrid was trained with fixed slot pairs but is deployed with
least-coupled pairing. This run continues from its terminal EMA weights and
trains with least-coupled pairing in the loop, so the decoder sees the pairs
it will see at inference. Fresh codec_train frames (beyond those used by the
final run), same objective, same gamma schedule, single LR with warm-up and
cosine decay, weight EMA, fixed terminal (no checkpoint selection).

  --train [--resume]   train UPDATES updates, save every SAVE_EVERY
  --evaluate           forward-only evaluation on the final-role validation frames
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
import traceback
import uuid

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt
import train_twc_v1_k8_final as base
import evaluate_twc_v1_k8_final as base_eval
import evaluate_twc_v1_k8_extended as ext
import twc_v1_k8_final_results as results

NAMESPACE = 'twc_v1_k8_paired_ft_v1'
OUTPUT_ROOT = Path('runs')/NAMESPACE
TRAIN_DIR = OUTPUT_ROOT/'training'
UPDATES, SAVE_EVERY, WARMUP, LR, MIN_FACTOR = 10000, 250, 200, 1e-5, .1
DATA_OFFSET = cfg.UPDATES*cfg.ACCUMULATION   # first micro index after the final run's frames
SEED = 20261118


def recipe():
    return dict(namespace=NAMESPACE, init='final terminal EMA (twc_v1_k8_final_hybrid_v1)', updates=UPDATES,
        save_every=SAVE_EVERY, warmup=WARMUP, lr=LR, min_factor=MIN_FACTOR, data_offset=DATA_OFFSET,
        accumulation=cfg.ACCUMULATION, micro_batch=cfg.MICRO_BATCH, objective=cfg.TRAJECTORY_OBJECTIVE,
        pairing='least-coupled from pilot sounding (training reuses the feedback noise draw)',
        weight_ema=cfg.WEIGHT_EMA_DECAY, grad_clip=cfg.GRAD_CLIP, weight_decay=cfg.WEIGHT_DECAY,
        FDMA_updates=0, CE_updates=0, seed=SEED)


def fingerprint():
    import hashlib
    return hashlib.sha256(json.dumps(cfg.plain(recipe()), sort_keys=True).encode()).hexdigest()


def lr_factor(step):
    if step < WARMUP:
        return (step+1)/WARMUP
    progress = min(1., (step-WARMUP)/max(1, UPDATES-WARMUP))
    return MIN_FACTOR+(1-MIN_FACTOR)*.5*(1+math.cos(math.pi*progress))


def micro_index(step, micro):
    return (step-1)*cfg.ACCUMULATION+micro+1


def gamma_for(index):
    return cfg.gamma_schedule((index-1) % (cfg.UPDATES*cfg.ACCUMULATION)+1)


def train(stop_after=None, resume=False, *, loaded=None, data=None, device=None, out_dir=None):
    import numpy as np
    import torch
    device = device or rt.base().configure_cuda()
    target = UPDATES if stop_after is None else stop_after
    if not SAVE_EVERY <= target <= UPDATES or target % SAVE_EVERY:
        raise ValueError('Stop at a save boundary')
    directory = Path(out_dir or TRAIN_DIR)
    if not resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'{directory} exists; use --resume')
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    system, source_manifest, _ = loaded or base.load_terminal_ema(device)
    model, stats = system['codec'], system['stats']
    model.train().requires_grad_(True)
    stats.train()
    data = data or rt.make_data(device)
    import twc_v1_k8_low_snr as low_snr_objective
    objective = lambda H_hat, H_dl, gamma: low_snr_objective.objective(H_hat, H_dl, gamma, cfg.TRAJECTORY_OBJECTIVE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.WEIGHT_DECAY, betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    identity = dict(recipe_fingerprint=fingerprint(), source_checkpoint_sha256=source_manifest['checkpoint_sha256'],
        code=rt.source_identity())
    start, ema = 0, base.ema_init(model)
    if resume and (directory/'checkpoint_latest.pt').exists():
        payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
        if payload['recipe_fingerprint'] != identity['recipe_fingerprint'] or \
           payload['source_checkpoint_sha256'] != identity['source_checkpoint_sha256']:
            raise ValueError('Resume identity differs')
        model.load_state_dict(payload['model']); stats.load_state_dict(payload['statistics'])
        optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
        ema = {k: v.to(device) for k, v in payload['ema'].items()}
        rt.base().restore_rng(payload['rng']); start = payload['step']
    elif not resume:
        base.atomic_json(directory/'initialization.json', dict(identity, recipe=recipe(), created=base.now()),
            exclusive=True)
    log, started, timings = directory/'progress.jsonl', time.perf_counter(), []
    try:
        for step in range(start+1, target+1):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            n = rt.physics().active_dimension(stats.layout)
            node_power = torch.zeros(6, 8, n, device=device)
            node_frames = torch.zeros(6, dtype=torch.int64, device=device)
            sums = dict(loss=0., sum_rate_bps_hz=0., key_energy_fraction=0., pair_coupling=0.)
            low_rate, low_count = 0., 0
            for micro in range(cfg.ACCUMULATION):
                index = DATA_OFFSET+micro_index(step, micro)
                sample = data.sample('codec_train', index)
                gamma = torch.tensor(gamma_for(index), dtype=torch.float32, device=device)
                noise, polarity = rt.batch_random('codec_train', index, device)
                out = rt.forward(system, sample, gamma, noise, polarity, pairing=True)
                loss, metrics = objective(out['H_hat'], out['H_dl'], gamma)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                (loss/cfg.ACCUMULATION).backward()
                power = out['wave'].active.detach().abs().square()
                node = rt.node_index(gamma)
                node_power.index_add_(0, node, power)
                node_frames += torch.bincount(node, minlength=6)
                rate = metrics['sum_rate_bps_hz'].detach()
                key = out['wave'].key.abs().square().sum(-1).mean(-1)/out['wave'].active.abs().square().sum(-1).mean(-1)
                sums['loss'] += float(loss.detach())/cfg.ACCUMULATION
                sums['sum_rate_bps_hz'] += float(rate.mean())/cfg.ACCUMULATION
                sums['key_energy_fraction'] += float(key.detach().mean())/cfg.ACCUMULATION
                sums['pair_coupling'] += float(rt.slot_pair_coupling(out['estimated']).mean())/cfg.ACCUMULATION
                low = gamma <= 10
                low_rate += float(rate[low].sum()); low_count += int(low.sum())
            params = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError('Nonfinite gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
            optimizer.step(); scheduler.step()
            stats.commit_power(node_power, node_frames)
            base.ema_update(ema, model, cfg.WEIGHT_EMA_DECAY)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.perf_counter()-begin)
            if step % SAVE_EVERY == 0 or step == target:
                measured = sum(timings[-100:])/len(timings[-100:])
                row = dict(step=step, **sums, low_gamma_sum_rate=low_rate/max(1, low_count), gradient_norm=float(norm),
                    lr=optimizer.param_groups[0]['lr'], seconds_per_update=measured,
                    ETA_seconds=measured*(UPDATES-step), updated=base.now())
                with log.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, sort_keys=True)+'\n')
                payload = dict(step=step, model=model.state_dict(), ema={k: v.detach().cpu() for k, v in ema.items()},
                    statistics=stats.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=rt.base().capture_rng(), **identity)
                base.atomic_save(payload, directory/'checkpoint_latest.pt')
                manifest = dict(identity, step=step, status='completed' if step == UPDATES else 'running',
                    checkpoint_sha256=rt.sha256(directory/'checkpoint_latest.pt'), progress_sha256=rt.sha256(log),
                    FDMA_updates=0, CE_updates=0, elapsed_seconds=time.perf_counter()-started, updated=base.now())
                base.atomic_json(directory/'manifest.json', manifest)
                if step == UPDATES:
                    base.atomic_json(directory/'terminal_receipt.json', manifest, exclusive=True)
                print(f'paired_ft {step}/{UPDATES} rate={row["sum_rate_bps_hz"]:.4f} low={row["low_gamma_sum_rate"]:.4f} '
                      f'key_frac={row["key_energy_fraction"]:.4f} coupling={row["pair_coupling"]:.3f} '
                      f'ETA={row["ETA_seconds"]/3600:.2f}h', flush=True)
        return json.loads((directory/'manifest.json').read_text())
    except BaseException as exc:
        if directory.is_dir():
            base.atomic_json(directory/'failure.json', dict(status='failed', step=locals().get('step', start),
                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), updated=base.now()))
        raise


def load_terminal(device, *, loaded=None, directory=None):
    import torch
    directory = Path(directory or TRAIN_DIR)
    manifest = json.loads((directory/'terminal_receipt.json').read_text())
    if manifest.get('status') != 'completed' or manifest.get('step') != UPDATES:
        raise ValueError('Completed fixed terminal required')
    if rt.sha256(directory/'checkpoint_latest.pt') != manifest['checkpoint_sha256']:
        raise ValueError('Terminal checkpoint bytes changed')
    if manifest['recipe_fingerprint'] != fingerprint():
        raise ValueError('Terminal recipe differs from current configuration')
    final, final_manifest, sources = loaded or base.load_terminal_ema(device)
    if manifest['source_checkpoint_sha256'] != final_manifest['checkpoint_sha256']:
        raise ValueError('Fine-tune source differs from the final terminal')
    payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    system = base.build_system(sources, device)
    system['codec'].load_state_dict(payload['ema']); system['stats'].load_state_dict(payload['statistics'])
    for module in (system['codec'], system['stats'], system['ce']):
        module.eval().requires_grad_(False)
    return system, manifest


SWITCH = ('fdma_baseline', 'hybrid_paired', 'ft_paired')
CONTRASTS = (('ft_paired_minus_fdma', 'ft_paired', 'fdma_baseline'),
             ('ft_paired_minus_paired', 'ft_paired', 'hybrid_paired'),
             ('ft_fixed_minus_final', 'ft_fixed', 'hybrid_final'),
             ('adaptive_ft_minus_fdma', 'adaptive_ft', 'fdma_baseline'),
             ('paired_minus_fdma', 'hybrid_paired', 'fdma_baseline'))


def evaluate(device=None, *, loaded=None, ft_dir=None, data=None, out_root=None, sizes=None):
    import statistics as st
    import torch
    device = device or rt.base().configure_cuda()
    sizes = sizes or dict(selection=cfg.SELECTION_BATCHES, final=cfg.FINAL_BATCHES, midpoint=cfg.MID_BATCHES)
    loaded = loaded or base.load_terminal_ema(device)
    table, manifest = ext.variants(device, loaded=loaded)
    ft, ft_manifest = load_terminal(device, loaded=loaded, directory=ft_dir)
    table = dict(fdma_baseline=table['fdma_baseline'], hybrid_final=table['hybrid_final'],
        hybrid_paired=table['hybrid_paired'], ft_fixed=(ft, {}),
        ft_paired=(ft, dict(pairing='least', sounding=True)), perfect_csi_rzf=table['perfect_csi_rzf'])
    data = data or rt.make_data(device)
    started = time.perf_counter()
    selection = ext.run_role(table, SWITCH, data, 'selection', cfg.SELECTION_FIRST, sizes['selection'], cfg.GAMMAS, device)
    sums = {}
    for row in selection:
        sums.setdefault((row['gamma_db'], row['variant']), []).append(row['sum_rate_bps_hz'])
    decision = {g: max(SWITCH, key=lambda v: st.mean(sums[g, v])) for g in sorted({g for g, _ in sums})}
    final = ext.run_role(table, tuple(table), data, 'final', cfg.FINAL_FIRST, sizes['final'], cfg.GAMMAS, device)
    final += [dict(r, variant='adaptive_ft') for r in final if decision.get(r['gamma_db']) == r['variant']]
    mid = ext.run_role(table, ('fdma_baseline', 'hybrid_paired', 'ft_paired'), data, 'midpoint', cfg.MID_FIRST,
        sizes['midpoint'], cfg.MID_GAMMAS, device)
    path = Path(out_root or OUTPUT_ROOT/'reports')/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    metrics = ext.METRICS
    base_eval.write_csv(path/'samples.csv.gz', selection+final+mid, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(final, 'final', cfg.GAMMAS, metrics)
        + results.summarize(mid, 'midpoint', cfg.MID_GAMMAS, metrics))
    base_eval.write_csv(path/'comparisons.csv', results.compare(final, 'final', cfg.GAMMAS, metrics, CONTRASTS)
        + results.compare(mid, 'midpoint', cfg.MID_GAMMAS, metrics, CONTRASTS))
    provenance = dict(namespace=NAMESPACE, recipe=recipe(), finetune_manifest=ft_manifest, final_manifest=manifest,
        switch_candidates=list(SWITCH), switch_decision={str(k): v for k, v in decision.items()},
        code=rt.source_identity(), elapsed_seconds=time.perf_counter()-started, test_used=False, FDMA_updates=0,
        contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('Pairing-aware fine-tune of the final Hybrid (Hybrid only; validation role; '
        'test sealed). Hybrid has more cumulative updates than FDMA.\n')
    files = {p.name: rt.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=NAMESPACE, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.train:
        print(json.dumps(train(resume=args.resume), indent=2, sort_keys=True))
    else:
        evaluate()


if __name__ == '__main__':
    main()
