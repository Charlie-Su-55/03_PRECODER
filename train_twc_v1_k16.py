#!/usr/bin/env python3
"""K=16 user-load study (CE fine-tune -> matched FDMA/Hybrid from scratch -> joint evaluation).

    python -B train_twc_v1_k16.py --ce [--resume]          CE fine-tune + calibration table
    python -B train_twc_v1_k16.py --codec fdma [--resume]  (or hybrid)
    python -B train_twc_v1_k16.py --evaluate               both arms on identical frames
    python -B train_twc_v1_k16.py --pipeline               all of the above; both arms in parallel

Terminal weights only (no checkpoint selection). Validation role; test sealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics as st
import subprocess
import sys
import time
import traceback
import uuid

from configs import twc_v1_k16_config as cfg
import k16_runtime as r
import train_twc_v1_k8_final as ft          # atomic I/O and EMA helpers only

CE_DIR = cfg.OUTPUT_ROOT/'ce'


def device_default():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_data(device):
    return r.K16Data(cfg.NAMESPACE, cfg.SEED, cfg.MICRO_BATCH, device)


def streams(role, index, device):
    return r.random_streams(cfg.NAMESPACE, cfg.SEED, role, index, cfg.MICRO_BATCH, device)


# ----------------------------------------------------------------------- CE --

def load_k8_ce_state(directory=None):
    import torch
    directory = Path(directory or cfg.SHARED_CE_DIR)
    for name, expected in cfg.SHARED_CE_SHA256.items():
        if r.sha256(directory/name) != expected:
            raise ValueError('Shared K8 CE file differs from the pinned bytes: '+name)
    return torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)['model']


def train_ce(resume=False, *, device=None, data=None, init_state=None, out_dir=None, stop_after=None):
    import numpy as np
    import torch
    device = device or device_default()
    directory = Path(out_dir or CE_DIR)
    target = cfg.CE_UPDATES if stop_after is None else stop_after
    if not resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'{directory} exists; use --resume')
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    ce = r.new_ce()
    ce.load_state_dict(init_state if init_state is not None else load_k8_ce_state(), strict=True)
    ce = ce.to(device).train()
    optimizer = torch.optim.AdamW(ce.parameters(), lr=cfg.CE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cfg.cosine(s, cfg.CE_UPDATES, cfg.CE_WARMUP))
    data = data or make_data(device)
    ident = dict(recipe_fingerprint=cfg.digest(cfg.ce_recipe()))
    start = 0
    if resume and (directory/'checkpoint_latest.pt').exists():
        payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
        if payload['recipe_fingerprint'] != ident['recipe_fingerprint']:
            raise ValueError('CE recipe changed')
        ce.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler']); start = payload['step']
    log, started = directory/'progress.jsonl', time.perf_counter()
    for step in range(start+1, target+1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for micro in range(cfg.ACCUMULATION):
            index = cfg.micro_index(step, micro)
            sample = data.sample('ce_train', index)
            gamma = torch.tensor(cfg.ce_gamma(index), dtype=torch.float32, device=device)
            noise, _, polarity = streams('ce_train', index, device)
            est, _, _ = r.ce_forward(ce, sample, gamma, noise, polarity)
            H = sample['H_ul']
            loss = ((est-H).abs().square().sum((2, 3))/H.abs().square().sum((2, 3))).mean()
            (loss/cfg.ACCUMULATION).backward()
            total += float(loss.detach())/cfg.ACCUMULATION
        torch.nn.utils.clip_grad_norm_(ce.parameters(), 1.)
        optimizer.step(); scheduler.step()
        if step % cfg.CE_SAVE_EVERY == 0 or step == target:
            ft.atomic_save(dict(step=step, model=ce.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), **ident), directory/'checkpoint_latest.pt')
            row = dict(step=step, ul_nmse=total, lr=optimizer.param_groups[0]['lr'],
                elapsed_seconds=time.perf_counter()-started, updated=ft.now())
            with log.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(f'k16_ce {step}/{cfg.CE_UPDATES} ul_nmse={total:.4f} '
                  f'ETA={(time.perf_counter()-started)/(step-start)*(cfg.CE_UPDATES-step)/3600:.2f}h', flush=True)
    if target < cfg.CE_UPDATES:
        return None
    return calibrate(ce, data, device, directory, ident)


def calibrate(ce, data, device, directory, ident, batches=None):
    import torch
    batches = batches or cfg.CALIBRATION_BATCHES
    ce.eval()
    totals = torch.zeros(6, r.K, r.NSC, device=device)
    nmse = torch.zeros(6, device=device)
    with torch.inference_mode():
        for b in range(batches):
            sample = data.sample('ce_calibration', b)
            for gi, gamma in enumerate(cfg.GAMMAS):
                noise, _, polarity = streams('ce_calibration', b*6+gi, device)
                g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
                est, _, _ = r.ce_forward(ce, sample, g, noise, polarity)
                totals[gi] += (est-sample['H_ul']).abs().square().mean((0, 2))
                H = sample['H_ul']
                nmse[gi] += ((est-H).abs().square().sum((2, 3))/H.abs().square().sum((2, 3))).mean()
    table = (totals/batches).cpu().tolist()
    calibration = dict(per_RE_error=table, ul_nmse_by_gamma=(nmse/batches).cpu().tolist(), nodes=list(cfg.GAMMAS),
        batches=batches, definition='absolute complex error power per UE and RE, averaged over MU and antennas')
    ft.atomic_json(directory/'calibration.json', calibration)
    manifest = dict(ident, status='completed', step=cfg.CE_UPDATES, checkpoint_sha256=r.sha256(directory/'checkpoint_latest.pt'),
        calibration_sha256=r.sha256(directory/'calibration.json'), updated=ft.now())
    ft.atomic_json(directory/'terminal_receipt.json', manifest, exclusive=True)
    print('k16_ce calibrated ul_nmse_dB=', [round(10*__import__('math').log10(x), 2) for x in calibration['ul_nmse_by_gamma']], flush=True)
    return manifest


def load_ce(device, directory=None):
    import torch
    directory = Path(directory or CE_DIR)
    receipt = json.loads((directory/'terminal_receipt.json').read_text())
    if (receipt['checkpoint_sha256'] != r.sha256(directory/'checkpoint_latest.pt')
            or receipt['calibration_sha256'] != r.sha256(directory/'calibration.json')):
        raise ValueError('K16 CE terminal bytes changed')
    ce = r.new_ce()
    ce.load_state_dict(torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)['model'])
    ce = ce.to(device).eval().requires_grad_(False)
    table = torch.tensor(json.loads((directory/'calibration.json').read_text())['per_RE_error'], dtype=torch.float32)
    return ce, table.to(device), dict(checkpoint_sha256=receipt['checkpoint_sha256'],
        calibration_sha256=receipt['calibration_sha256'])


# -------------------------------------------------------------------- codec --

def build(arm, device, ce, table):
    import torch
    layout = r.make_layout(arm)
    n = r.active_dimension(layout)
    codec = r.Codec(arm, cfg.CODEC_INIT_SEED).to(device)
    stats = r.NodeStatistics(layout, table.detach().cpu(), torch.full((r.K, n), r.ED/n)).to(device)
    return dict(codec=codec, ce=ce, stats=stats, layout=layout, arm=arm)


def train_codec(arm, resume=False, *, device=None, ce_bundle=None, data=None, out_dir=None, stop_after=None):
    import numpy as np
    import torch
    device = device or device_default()
    directory = Path(out_dir or cfg.OUTPUT_ROOT/arm)
    target = cfg.UPDATES if stop_after is None else stop_after
    if not resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'{directory} exists; use --resume')
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    ce, table, ce_info = ce_bundle or load_ce(device)
    system = build(arm, device, ce, table)
    model, stats = system['codec'], system['stats']
    model.train(); stats.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cfg.cosine(s, cfg.UPDATES, cfg.WARMUP))
    ident = dict(arm=arm, recipe_fingerprint=cfg.digest(cfg.codec_recipe(arm)), ce=ce_info)
    data = data or make_data(device)
    start, ema = 0, ft.ema_init(model)
    if resume and (directory/'checkpoint_latest.pt').exists():
        payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
        if payload['recipe_fingerprint'] != ident['recipe_fingerprint'] or payload['ce'] != ce_info:
            raise ValueError('Resume identity changed')
        model.load_state_dict(payload['model']); stats.load_state_dict(payload['statistics'])
        optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
        ema = {k: v.to(device) for k, v in payload['ema'].items()}; start = payload['step']
    pairing = arm == 'hybrid' and cfg.HYBRID_TRAIN_PAIRING
    n = r.active_dimension(system['layout'])
    log, started, timings = directory/'progress.jsonl', time.perf_counter(), []
    try:
        for step in range(start+1, target+1):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            stage = 'nmse' if step <= cfg.NMSE_UPDATES else 'objective'
            node_power = torch.zeros(6, r.K, n, device=device)
            node_frames = torch.zeros(6, dtype=torch.int64, device=device)
            sums = dict(loss=0., sum_rate_bps_hz=0., dl_nmse_linear=0., key_energy_fraction=0.)
            for micro in range(cfg.ACCUMULATION):
                index = cfg.micro_index(step, micro)
                sample = data.sample('codec_train', index)
                gamma = torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32, device=device)
                noise, sounding, polarity = streams(f'codec_train_{arm}', index, device)
                out = r.forward(system, sample, gamma, noise, polarity, pairing=pairing, sounding=sounding)
                if stage == 'nmse':
                    loss, metrics = r.nmse_loss(out['H_hat'], out['H_dl'])
                else:
                    loss, metrics = r.objective(out['H_hat'], out['H_dl'], gamma)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                (loss/cfg.ACCUMULATION).backward()
                node = r.node_index(gamma)
                node_power.index_add_(0, node, out['wave'].active.detach().abs().square())
                node_frames += torch.bincount(node, minlength=6)
                sums['loss'] += float(loss.detach())/cfg.ACCUMULATION
                for name in ('sum_rate_bps_hz', 'dl_nmse_linear'):
                    if name in metrics:
                        sums[name] += float(metrics[name].detach().float().mean())/cfg.ACCUMULATION
                w = out['wave']
                sums['key_energy_fraction'] += float((w.key.detach().abs().square().sum(-1)
                    /w.active.detach().abs().square().sum(-1)).mean())/cfg.ACCUMULATION
            params = [p for p in model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite gradient')
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
                with log.open('a') as stream:
                    stream.write(json.dumps(row, sort_keys=True)+'\n')
                ft.atomic_save(dict(step=step, model=model.state_dict(), ema={k: v.detach().cpu() for k, v in ema.items()},
                    statistics=stats.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    **ident), directory/'checkpoint_latest.pt')
                manifest = dict(ident, step=step, status='completed' if step == cfg.UPDATES else 'running',
                    checkpoint_sha256=r.sha256(directory/'checkpoint_latest.pt'), FDMA_K8_updates=0, updated=ft.now())
                ft.atomic_json(directory/'manifest.json', manifest)
                if step == cfg.UPDATES:
                    ft.atomic_json(directory/'terminal_receipt.json', manifest, exclusive=True)
                print(f'k16_{arm} {step}/{cfg.UPDATES} [{stage}] loss={sums["loss"]:.4f} rate={sums["sum_rate_bps_hz"]:.3f} '
                      f'nmse={sums["dl_nmse_linear"]:.4f} key_frac={sums["key_energy_fraction"]:.3f} '
                      f'ETA={row["ETA_seconds"]/3600:.2f}h', flush=True)
        return json.loads((directory/'manifest.json').read_text())
    except BaseException as exc:
        ft.atomic_json(directory/'failure.json', dict(status='failed', step=locals().get('step', start),
            error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), updated=ft.now()))
        raise


def load_codec(arm, device, ce_bundle, directory=None):
    import torch
    directory = Path(directory or cfg.OUTPUT_ROOT/arm)
    receipt = json.loads((directory/'terminal_receipt.json').read_text())
    if (receipt['status'] != 'completed' or receipt['step'] != cfg.UPDATES
            or receipt['checkpoint_sha256'] != r.sha256(directory/'checkpoint_latest.pt')
            or receipt['recipe_fingerprint'] != cfg.digest(cfg.codec_recipe(arm)) or receipt['ce'] != ce_bundle[2]):
        raise ValueError(f'K16 {arm} terminal identity mismatch')
    payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    system = build(arm, device, ce_bundle[0], ce_bundle[1])
    system['codec'].load_state_dict(payload['ema']); system['stats'].load_state_dict(payload['statistics'])
    system['codec'].eval().requires_grad_(False); system['stats'].eval()
    return system, receipt


# --------------------------------------------------------------- evaluation --

VARIANTS = dict(fdma_baseline=('fdma', {}), fdma_genie_ul=('fdma', dict(genie_ul=True)),
    hybrid_fixed=('hybrid', {}), hybrid_paired=('hybrid', dict(pairing=True)),
    hybrid_paired_genie_ul=('hybrid', dict(pairing=True, genie_ul=True)), perfect_csi_rzf=(None, {}))
SWITCH = ('fdma_baseline', 'hybrid_paired')
CONTRASTS = (('paired_minus_fdma', 'hybrid_paired', 'fdma_baseline'),
             ('fixed_minus_fdma', 'hybrid_fixed', 'fdma_baseline'),
             ('adaptive_minus_fdma', 'adaptive', 'fdma_baseline'),
             ('paired_minus_fixed', 'hybrid_paired', 'hybrid_fixed'),
             ('genie_paired_minus_genie_fdma', 'hybrid_paired_genie_ul', 'fdma_genie_ul'),
             ('fdma_genie_minus_fdma', 'fdma_genie_ul', 'fdma_baseline'),
             ('genie_minus_paired', 'hybrid_paired_genie_ul', 'hybrid_paired'),
             ('upper_minus_paired', 'perfect_csi_rzf', 'hybrid_paired'),
             ('upper_minus_fdma', 'perfect_csi_rzf', 'fdma_baseline'))
METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'direction_error', 'interference_power_mean',
           'desired_signal_power_mean', 'min_user_rate_bps_hz', 'key_energy_fraction', 'pair_coupling')


def run_role(systems, names, data, role, first, batches, gammas, device):
    import torch
    rows = []
    for b in range(batches):
        sample = data.sample('validation', first+b)
        for gi, gamma in enumerate(gammas):
            noise, sounding, polarity = streams(role, b*len(gammas)+gi, device)
            g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
            for name in names:
                arm, options = VARIANTS[name]
                with torch.inference_mode():
                    if arm is None:
                        metrics = r.upper_bound_metrics(sample['H_dl'])
                    else:
                        metrics = r.sample_metrics(r.forward(systems[arm], sample, g, noise, polarity,
                            sounding=sounding, **options))
                for j in range(cfg.MICRO_BATCH):
                    row = dict(variant=name, role=role, gamma_db=float(gamma), sample_index=b*cfg.MICRO_BATCH+j,
                        batch_index=b, **{k: float(v[j]) for k, v in metrics.items()})
                    if name.startswith('fdma'):
                        row.pop('key_energy_fraction', None); row.pop('pair_coupling', None)
                    rows.append(row)
        if (b+1) % 16 == 0:
            print(f'k16 eval {role} {b+1}/{batches}', flush=True)
    return rows


def evaluate(device=None, *, systems=None, data=None, out_root=None, sizes=None):
    import torch
    import evaluate_twc_v1_k8_final as base_eval
    import twc_v1_k8_final_results as results
    device = device or device_default()
    sizes = sizes or dict(selection=cfg.SELECTION_BATCHES, final=cfg.FINAL_BATCHES, midpoint=cfg.MID_BATCHES)
    receipts = {}
    if systems is None:
        bundle = load_ce(device)
        systems = {}
        for arm in cfg.ARMS:
            systems[arm], receipts[arm] = load_codec(arm, device, bundle)
    data = data or make_data(device)
    started = time.perf_counter()
    selection = run_role(systems, SWITCH, data, 'selection', cfg.SELECTION_FIRST, sizes['selection'], cfg.GAMMAS, device)
    sums = {}
    for row in selection:
        sums.setdefault((row['gamma_db'], row['variant']), []).append(row['sum_rate_bps_hz'])
    decision = {g: max(SWITCH, key=lambda v: st.mean(sums[g, v])) for g in sorted({g for g, _ in sums})}
    final = run_role(systems, tuple(VARIANTS), data, 'final', cfg.FINAL_FIRST, sizes['final'], cfg.GAMMAS, device)
    final += [dict(x, variant='adaptive') for x in final if decision.get(x['gamma_db']) == x['variant']]
    mid = run_role(systems, ('fdma_baseline', 'hybrid_paired'), data, 'midpoint', cfg.MID_FIRST, sizes['midpoint'],
        cfg.MID_GAMMAS, device)
    path = Path(out_root or cfg.OUTPUT_ROOT/'reports')/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', selection+final+mid, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(final, 'final', cfg.GAMMAS, METRICS)
        + results.summarize(mid, 'midpoint', cfg.MID_GAMMAS, METRICS))
    base_eval.write_csv(path/'comparisons.csv', results.compare(final, 'final', cfg.GAMMAS, METRICS, CONTRASTS)
        + results.compare(mid, 'midpoint', cfg.MID_GAMMAS, METRICS, CONTRASTS[:1]))
    ce_cal = json.loads((CE_DIR/'calibration.json').read_text()) if (CE_DIR/'calibration.json').exists() else {}
    provenance = dict(namespace=cfg.NAMESPACE, K=r.K, codec_recipes={a: cfg.codec_recipe(a) for a in cfg.ARMS},
        ce_recipe=cfg.ce_recipe(), receipts=receipts, ul_nmse_by_gamma=ce_cal.get('ul_nmse_by_gamma'),
        switch_decision={str(k): v for k, v in decision.items()}, elapsed_seconds=time.perf_counter()-started,
        test_used=False, contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('K=16 user-load study: FDMA and Hybrid trained with the same recipe and budget '
        '(validation role; test sealed).\n')
    files = {p.name: r.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=cfg.NAMESPACE, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


# ----------------------------------------------------------------- pipeline --

def pipeline():
    py = sys.executable
    if not (CE_DIR/'terminal_receipt.json').exists():
        resume = ['--resume'] if (CE_DIR/'checkpoint_latest.pt').exists() else []
        subprocess.run([py, '-B', '-u', __file__, '--ce', *resume], check=True)
    procs = []
    for arm in cfg.ARMS:
        if (cfg.OUTPUT_ROOT/arm/'terminal_receipt.json').exists():
            continue
        resume = ['--resume'] if (cfg.OUTPUT_ROOT/arm/'checkpoint_latest.pt').exists() else []
        procs.append((arm, subprocess.Popen([py, '-B', '-u', __file__, '--codec', arm, *resume])))
    failed = [arm for arm, p in procs if p.wait() != 0]
    if failed:
        raise SystemExit(f'codec training failed: {failed}')
    subprocess.run([py, '-B', '-u', __file__, '--evaluate'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ce', action='store_true')
    mode.add_argument('--codec', choices=cfg.ARMS)
    mode.add_argument('--evaluate', action='store_true')
    mode.add_argument('--pipeline', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.ce:
        train_ce(args.resume)
    elif args.codec:
        train_codec(args.codec, args.resume)
    elif args.evaluate:
        evaluate()
    else:
        pipeline()


if __name__ == '__main__':
    main()
