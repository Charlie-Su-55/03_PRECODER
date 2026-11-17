#!/usr/bin/env python3
"""Train and evaluate one group-size arm (hybrid_g2 or hybrid_g1) from scratch.

    python -B train_twc_v1_k8_grouped.py --arm hybrid_g2 --train [--resume]
    python -B train_twc_v1_k8_grouped.py --arm hybrid_g2 --evaluate

Stage 1 (NMSE_UPDATES): DL NMSE. Stage 2: the final model's low-SNR interference
objective. Same gamma schedule, gamma-aware UE and per-node prior as the final
Hybrid; CE nmse65000 frozen. The evaluated model is the fixed terminal EMA.
Evaluation reuses the final model's validation frames and noise streams.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
import traceback
import uuid

from configs import twc_v1_k8_grouped_config as cfg
from configs import twc_v1_k8_final_config as final_cfg
import final_runtime as fr
import grouped_runtime as gr
import train_twc_v1_k8_final as ft


def identity(arm, core, ce_info):
    return dict(arm=arm, recipe_fingerprint=cfg.fingerprint(arm), core_sha256=core['sha256'],
        ce=ce_info['sha256'])


def build(arm, device, ce, table):
    import torch
    from models.twc_v1_k8_icc_aligned import create_codec
    layout = gr.make_layout(arm)
    codec = create_codec('hybrid_g4', device=str(device), seed=cfg.CODEC_INIT_SEED)
    gr.set_geometry(codec.decoder, layout)
    model = fr.FinalCodec(codec).to(device)
    n = gr.active_dimension(layout)
    stats = gr.NodeStatistics(layout, table.detach().cpu(), torch.full((8, n), 48./n)).to(device)
    return dict(kind='grouped', codec=model, ce=ce, stats=stats, layout=layout)


def nmse_loss(H_hat, H_dl):
    err = (H_hat-H_dl).abs().square().sum((2, 3))/H_dl.abs().square().sum((2, 3))
    return err.mean(), dict(dl_nmse_linear=err.mean(1).detach())


def train(arm, stop_after=None, resume=False, *, device=None, ce_bundle=None, data=None, out_dir=None,
          core=None, objective=None):
    import torch
    import numpy as np
    target = cfg.UPDATES if stop_after is None else stop_after
    if not cfg.SAVE_EVERY <= target <= cfg.UPDATES or target % cfg.SAVE_EVERY:
        raise ValueError('Stop at a save boundary')
    directory = Path(out_dir or cfg.output_root(arm)/'training')
    device = device or fr.base().configure_cuda()
    core = core or fr.source_identity() | dict(sha256={n: fr.sha256(cfg.ROOT/n) for n in cfg.CORE_FILES+final_cfg.CORE_FILES[:2]})
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    ce, table, ce_info = ce_bundle or gr.load_shared_ce(device)
    data = data or gr.make_train_data(device)
    if objective is None:
        import twc_v1_k8_low_snr as low_snr_objective
        objective = lambda H_hat, H_dl, gamma: low_snr_objective.objective(H_hat, H_dl, gamma, final_cfg.TRAJECTORY_OBJECTIVE)
    system = build(arm, device, ce, table)
    model, stats = system['codec'], system['stats']
    model.train().requires_grad_(True); stats.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cfg.lr_factor)
    ident = identity(arm, core, ce_info)
    start, ema = 0, ft.ema_init(model)
    if resume:
        manifest = json.loads((directory/'manifest.json').read_text())
        if {k: manifest.get(k) for k in ident} != ident:
            raise ValueError('Resume identity changed (recipe, code or CE)')
        if fr.sha256(directory/'checkpoint_latest.pt') != manifest['checkpoint_sha256']:
            raise ValueError('Latest checkpoint bytes differ from manifest')
        payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model']); stats.load_state_dict(payload['statistics'])
        optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
        ema = {k: v.to(device) for k, v in payload['ema'].items()}
        fr.base().restore_rng(payload['rng'])
        start = payload['step']
    else:
        directory.mkdir(parents=True, exist_ok=False)
        ft.atomic_json(directory/'initialization.json', dict(ident, recipe=cfg.recipe(arm), created=ft.now()), exclusive=True)
    log, started, timings = directory/'progress.jsonl', time.perf_counter(), []
    n = gr.active_dimension(system['layout'])
    try:
        for step in range(start+1, target+1):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            node_power = torch.zeros(6, 8, n, device=device)
            node_frames = torch.zeros(6, dtype=torch.int64, device=device)
            sums = dict(loss=0., sum_rate_bps_hz=0., dl_nmse_linear=0., key_energy_fraction=0.)
            stage = 'nmse' if step <= cfg.NMSE_UPDATES else 'objective'
            for micro in range(cfg.ACCUMULATION):
                index = cfg.micro_index(step, micro)
                sample = data.sample('codec_train', index)
                gamma = torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32, device=device)
                noise, polarity = gr.train_random(arm, index, device)
                out = gr.forward(system, sample, gamma, noise, polarity)
                if stage == 'nmse':
                    loss, metrics = nmse_loss(out['H_hat'], out['H_dl'])
                else:
                    loss, metrics = objective(out['H_hat'], out['H_dl'], gamma)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                (loss/cfg.ACCUMULATION).backward()
                node = fr.node_index(gamma)
                node_power.index_add_(0, node, out['wave'].active.detach().abs().square())
                node_frames += torch.bincount(node, minlength=6)
                sums['loss'] += float(loss.detach())/cfg.ACCUMULATION
                for name in ('sum_rate_bps_hz', 'dl_nmse_linear'):
                    if name in metrics:
                        sums[name] += float(metrics[name].detach().float().mean())/cfg.ACCUMULATION
                w = out['wave']
                sums["key_energy_fraction"] += float((w.key.detach().abs().square().sum(-1)/w.active.detach().abs().square().sum(-1)).mean())/cfg.ACCUMULATION
            params = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError('Nonfinite gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
            optimizer.step(); scheduler.step()
            stats.commit_power(node_power, node_frames)
            ft.ema_update(ema, model, cfg.WEIGHT_EMA_DECAY)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.perf_counter()-begin)
            if step % cfg.SAVE_EVERY == 0 or step == target:
                measured = sum(timings[-100:])/len(timings[-100:])
                row = dict(step=step, stage=stage, **sums, gradient_norm=float(norm), lr=optimizer.param_groups[0]['lr'],
                    seconds_per_update=measured, ETA_seconds=measured*(cfg.UPDATES-step), updated=ft.now())
                with log.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, sort_keys=True)+'\n')
                payload = dict(step=step, model=model.state_dict(), ema={k: v.detach().cpu() for k, v in ema.items()},
                    statistics=stats.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng=fr.base().capture_rng(), **ident)
                ft.atomic_save(payload, directory/'checkpoint_latest.pt')
                manifest = dict(ident, step=step, status='completed' if step == cfg.UPDATES else 'running',
                    checkpoint_sha256=fr.sha256(directory/'checkpoint_latest.pt'), progress_sha256=fr.sha256(log),
                    FDMA_updates=0, CE_updates=0, elapsed_seconds=time.perf_counter()-started, updated=ft.now())
                ft.atomic_json(directory/'manifest.json', manifest)
                if step == cfg.UPDATES:
                    ft.atomic_json(directory/'terminal_receipt.json', manifest, exclusive=True)
                print(f'{arm} {step}/{cfg.UPDATES} [{stage}] loss={sums["loss"]:.4f} rate={sums["sum_rate_bps_hz"]:.3f} '
                      f'nmse={sums["dl_nmse_linear"]:.4f} key_frac={sums["key_energy_fraction"]:.3f} '
                      f'ETA={row["ETA_seconds"]/3600:.2f}h', flush=True)
        return json.loads((directory/'manifest.json').read_text())
    except BaseException as exc:
        if directory.is_dir():
            ft.atomic_json(directory/'failure.json', dict(status='failed', step=locals().get('step', start),
                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), updated=ft.now()))
        raise


def load_terminal(arm, device, *, ce_bundle=None, directory=None):
    import torch
    directory = Path(directory or cfg.output_root(arm)/'training')
    manifest = json.loads((directory/'terminal_receipt.json').read_text())
    if manifest.get('status') != 'completed' or manifest.get('step') != cfg.UPDATES:
        raise ValueError('Completed fixed terminal required')
    if fr.sha256(directory/'checkpoint_latest.pt') != manifest['checkpoint_sha256'] or manifest['recipe_fingerprint'] != cfg.fingerprint(arm):
        raise ValueError('Terminal bytes or recipe changed')
    ce, table, ce_info = ce_bundle or gr.load_shared_ce(device)
    if manifest['ce'] != ce_info['sha256']:
        raise ValueError('Terminal CE identity differs')
    payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    system = build(arm, device, ce, table)
    system['codec'].load_state_dict(payload['ema']); system['stats'].load_state_dict(payload['statistics'])
    for module in (system['codec'], system['stats']):
        module.eval().requires_grad_(False)
    return system, manifest


def evaluate(arm, device=None, *, loaded=None, data=None, out_root=None, sizes=None):
    """Score the terminal EMA on the final model's validation frames and noise."""
    import torch
    import evaluate_twc_v1_k8_final as fe
    import twc_v1_k8_final_results as results
    device = device or fr.base().configure_cuda()
    sizes = sizes or dict(final=final_cfg.FINAL_BATCHES, midpoint=final_cfg.MID_BATCHES)
    system, manifest = loaded or load_terminal(arm, device)
    data = data or fr.make_data(device)
    rows = []
    for role, first, batches, gammas in (('final', final_cfg.FINAL_FIRST, sizes['final'], final_cfg.GAMMAS),
                                         ('midpoint', final_cfg.MID_FIRST, sizes['midpoint'], final_cfg.MID_GAMMAS)):
        for b in range(batches):
            sample = data.sample('validation', first+b)
            for gi, gamma in enumerate(gammas):
                noise, polarity = fr.batch_random(role, b*len(gammas)+gi, device)
                g = torch.full((final_cfg.MICRO_BATCH,), float(gamma), device=device)
                with torch.inference_mode():
                    metrics = fr.sample_metrics(gr.forward(system, sample, g, noise, polarity))
                for j in range(final_cfg.MICRO_BATCH):
                    row = dict(variant=arm, role=role, gamma_db=float(gamma), sample_index=b*final_cfg.MICRO_BATCH+j,
                        batch_index=b, **{k: float(v[j]) for k, v in metrics.items()})
                    row.pop('pair_coupling', None)
                    rows.append(row)
            if (b+1) % 32 == 0:
                print(f'{arm} {role} {b+1}/{batches}', flush=True)
    path = Path(out_root or cfg.output_root(arm)/'reports')/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    final_rows = [r for r in rows if r['role'] == 'final']
    mid_rows = [r for r in rows if r['role'] == 'midpoint']
    fe.write_csv(path/'samples.csv.gz', rows, compress=True)
    fe.write_csv(path/'summary.csv', results.summarize(final_rows, 'final', final_cfg.GAMMAS, fe.METRICS)
        + results.summarize(mid_rows, 'midpoint', final_cfg.MID_GAMMAS, fe.METRICS))
    progress = cfg.output_root(arm)/'training'/'progress.jsonl'
    if progress.is_file():
        curve = []
        for line in progress.read_text().splitlines():
            r = json.loads(line)
            for metric in ('loss', 'sum_rate_bps_hz', 'dl_nmse_linear', 'key_energy_fraction'):
                curve.append(dict(variant=arm, role='training', scope='all', step=r['step'], metric=metric, mean=r[metric]))
        fe.write_csv(path/'curve_summary.csv', curve)
    (path/'provenance.json').write_text(json.dumps(final_cfg.plain(dict(namespace=cfg.NAMESPACE, arm=arm,
        recipe=cfg.recipe(arm), terminal=manifest, frames='final-model validation frames and noise',
        FDMA_updates=0, test_used=False, contains_model_weights=False)), indent=2, sort_keys=True)+'\n')
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=cfg.NAMESPACE, export_id=path.name[8:],
        files={p.name: fr.sha256(p) for p in sorted(path.iterdir())}), indent=2, sort_keys=True)+'\n')
    print('REPORT', path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(cfg.ARMS), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args(argv)
    if args.train:
        print(json.dumps(train(args.arm, args.stop_after, args.resume), indent=2, sort_keys=True))
    else:
        evaluate(args.arm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
