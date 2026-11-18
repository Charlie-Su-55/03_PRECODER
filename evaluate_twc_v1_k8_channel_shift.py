#!/usr/bin/env python3
"""Channel-shift evaluation of the frozen final Hybrid and frozen FDMA (forward passes only).

All models were trained on CDL-C, in which every user shares the same cluster
delays and angles. Two pre-registered shift families are scored; every
scenario is reported, none is dropped:

  rot<D>   CDL-C with a per-user random angular offset at the BS array. A
           phase ramp exp(j*pi*(u*y + v*z)) over the antenna grid (y, z in
           units of lambda/2) moves every path of that user by (u, v) in
           sine space. The same ramp is applied to the user's DL and UL
           channel (angular reciprocity in FDD). Offsets are uniform in
           [-D, D] DFT bins per axis (one bin = 2/N_axis). rot0 is the
           unchanged training distribution and reproduces the first frames
           of the final evaluation exactly.
  cdl_<X>  CDL-A/B/D/E with the generator otherwise unchanged.

Variants per scenario: fdma_baseline, hybrid_final, hybrid_paired (sounding
with independent noise), perfect_csi_rzf. Validation frames, test sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt
import evaluate_twc_v1_k8_final as base_eval
import evaluate_twc_v1_k8_extended as ext
import twc_v1_k8_final_results as results

NAMESPACE = 'twc_v1_k8_channel_shift_v1'
OUTPUT_ROOT = Path('runs')/NAMESPACE
ROTATIONS = (0, 1, 2, 4, 8)
CDL_MODELS = ('A', 'B', 'D', 'E')
SCENARIOS = tuple(f'rot{d}' for d in ROTATIONS)+tuple(f'cdl_{m.lower()}' for m in CDL_MODELS)
VARIANTS = ('fdma_baseline', 'hybrid_final', 'hybrid_paired', 'perfect_csi_rzf')
BATCHES = 64          # 256 complete MUs per scenario, first frames of the final role
METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'direction_error', 'interference_power_mean',
           'min_user_rate_bps_hz', 'pair_coupling', 'true_ul_coupling')


def _seed(*parts):
    raw = ':'.join(str(p) for p in (NAMESPACE, cfg.SEED)+parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def antenna_grid(positions):
    """positions [Nt,3] in metres -> integer (horizontal, vertical) indices in units of lambda/2."""
    import torch
    # Sionna returns the positions on the GPU; the grid and the ramp are built on the CPU.
    pos = torch.as_tensor(positions).detach().to('cpu', torch.float64).reshape(-1, 3)
    axes = []
    for axis in range(3):
        values = torch.unique(torch.round(pos[:, axis]*1e6)/1e6)
        axes.append((len(values), axis))
    counts = sorted(axes, reverse=True)
    (nh, ah), (nv, av) = counts[0], counts[1]
    if nh*nv != pos.shape[0] or nh < 2:
        raise ValueError('Antenna positions are not a rectangular planar grid')
    def index(axis):
        values = pos[:, axis]
        step = (values.max()-values.min())/(len(torch.unique(torch.round(values*1e6)/1e6))-1)
        return torch.round((values-values.min())/step).long()
    return index(ah), index(av), nh, nv


def bs_positions(data):
    """BS antenna positions from the prepared Sionna generator (tries known attribute names)."""
    array = data.prepare()._generator.bs_array
    for name in ('ant_pos', 'positions', '_ant_pos', 'antenna_positions'):
        value = getattr(array, name, None)
        if callable(value):
            try:
                value = value(3e8/3.5e9)
            except TypeError:
                value = value()
        if value is not None:
            return value
    raise AttributeError('Cannot read BS antenna positions from the Sionna AntennaArray')


def rotation_phase(grid, bins, role, index, batch, device):
    """Per-user phase ramp [B,K,Nt] for offsets uniform in [-bins, bins] DFT bins per axis."""
    import torch
    h, v, nh, nv = grid
    gen = torch.Generator().manual_seed(_seed('rotation', role, index))
    offsets = (2*torch.rand(batch, rt.K, 2, generator=gen, dtype=torch.float64)-1)*bins
    u, w = offsets[..., 0]*2/nh, offsets[..., 1]*2/nv
    phase = math.pi*(u[..., None]*h.double()[None, None]+w[..., None]*v.double()[None, None])
    return torch.polar(torch.ones_like(phase), phase).to(torch.complex64).to(device)


def true_coupling(H_ul):
    c = rt.coupling(H_ul)
    off = ~__import__('torch').eye(rt.K, dtype=bool, device=c.device)
    return c[:, off].reshape(c.shape[0], -1).mean(-1).float().cpu()


def make_data(model, device):
    source = rt.base().original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4,
            validation=cfg.MICRO_BATCH, test=1), device=str(device), cdl_model=model, speed=1.,
        seed_namespace=cfg.NAMESPACE)


def scenario_sample(scenario, datasets, grid, b, device):
    first = cfg.FINAL_FIRST+b
    if scenario.startswith('rot'):
        sample = datasets['C'].sample('validation', first)
        bins = int(scenario[3:])
        if bins:
            ramp = rotation_phase(grid, bins, 'final', b, sample['H_dl'].shape[0], device)
            sample = dict(sample, H_dl=sample['H_dl']*ramp[..., None], H_ul=sample['H_ul']*ramp[..., None])
        return sample
    return datasets[scenario[-1].upper()].sample('validation', first)


def run(table, datasets, grid, batches, device, scenarios=None):
    import torch
    rows = []
    scenarios = SCENARIOS if scenarios is None else scenarios
    for scenario in scenarios:
        for b in range(batches):
            sample = scenario_sample(scenario, datasets, grid, b, device)
            coupling = true_coupling(sample['H_ul'])
            for gi, gamma in enumerate(cfg.GAMMAS):
                index = b*len(cfg.GAMMAS)+gi
                noise, polarity = rt.batch_random('final', index, device)
                sound = ext.sounding_noise('final', index, device)
                g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
                for name in VARIANTS:
                    system, options = table[name]
                    with torch.inference_mode():
                        if system is None:
                            metrics = ext.upper_bound_metrics(sample['H_dl'])
                            metrics = {k: v for k, v in metrics.items() if not k.startswith('sum_rate_dl')}
                        else:
                            options = dict(options)
                            if options.pop('sounding', False):
                                options['sounding'] = sound
                            metrics = rt.sample_metrics(ext.forward(system, sample, g, noise, polarity, **options))
                    metrics['true_ul_coupling'] = coupling
                    for j in range(cfg.MICRO_BATCH):
                        row = dict(variant=f'{scenario}:{name}', role='final', gamma_db=float(gamma),
                            sample_index=b*cfg.MICRO_BATCH+j, batch_index=b,
                            **{k: float(v[j]) for k, v in metrics.items() if k in METRICS})
                        if name.startswith('fdma'):
                            row.pop('pair_coupling', None)
                        rows.append(row)
            if (b+1) % 16 == 0:
                print(f'{scenario} {b+1}/{batches}', flush=True)
    return rows


def contrasts(scenarios=None):
    out = []
    scenarios = SCENARIOS if scenarios is None else scenarios
    for s in scenarios:
        out += [(f'{s}:final_minus_fdma', f'{s}:hybrid_final', f'{s}:fdma_baseline'),
                (f'{s}:paired_minus_fdma', f'{s}:hybrid_paired', f'{s}:fdma_baseline'),
                (f'{s}:paired_minus_final', f'{s}:hybrid_paired', f'{s}:hybrid_final'),
                (f'{s}:upper_minus_paired', f'{s}:perfect_csi_rzf', f'{s}:hybrid_paired')]
    return tuple(out)


def evaluate(device=None, *, loaded=None, datasets=None, grid=None, out_root=None, batches=BATCHES):
    import torch
    device = device or rt.base().configure_cuda()
    table, manifest = ext.variants(device, loaded=loaded)
    table = {k: table[k] for k in VARIANTS}
    needed = sorted({'C' if s.startswith('rot') else s[-1].upper() for s in SCENARIOS})
    datasets = datasets or {m: make_data(m, device) for m in needed}
    positions = None
    if grid is None:
        positions = bs_positions(datasets['C'])
        grid = antenna_grid(positions)
    started = time.perf_counter()
    rows = run(table, datasets, grid, batches, device)
    root = Path(out_root or OUTPUT_ROOT/'reports')
    path = root/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', rows, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(rows, 'final', cfg.GAMMAS, METRICS))
    base_eval.write_csv(path/'comparisons.csv', results.compare(rows, 'final', cfg.GAMMAS, METRICS, contrasts()))
    h, v, nh, nv = grid
    provenance = dict(namespace=NAMESPACE, final_recipe=cfg.recipe(), terminal_manifest=manifest,
        code=rt.source_identity(), scenarios=list(SCENARIOS), variants=list(VARIANTS),
        rotation_bins=list(ROTATIONS), antenna_grid=dict(horizontal=nh, vertical=nv,
            horizontal_index=h.tolist(), vertical_index=v.tolist()),
        frames=dict(role='validation', first=cfg.FINAL_FIRST, batches=batches, mus=batches*cfg.MICRO_BATCH),
        elapsed_seconds=time.perf_counter()-started, training_updates=0, FDMA_updates=0, test_used=False,
        contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('Channel-shift evaluation of frozen CDL-C models (validation frames; test sealed; '
        'no training). All pre-registered scenarios are reported.\n')
    files = {p.name: rt.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=NAMESPACE, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--probe', action='store_true', help='read-only: print the BS antenna grid and exit')
    mode.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.probe:
        device = rt.base().configure_cuda()
        h, v, nh, nv = antenna_grid(bs_positions(make_data('C', device)))
        print('grid', nh, 'x', nv)
        print('horizontal', h.tolist())
        print('vertical  ', v.tolist())
        return
    evaluate()


if __name__ == '__main__':
    main()
