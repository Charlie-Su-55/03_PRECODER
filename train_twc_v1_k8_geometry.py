#!/usr/bin/env python3
"""Geometry study: FDMA and paired Hybrid trained from scratch on CDL-C with per-user angular offsets.

    python -B train_twc_v1_k8_geometry.py --arm fdma_geo --train [--resume]
    python -B train_twc_v1_k8_geometry.py --arm hybrid_geo --train [--resume]
    python -B train_twc_v1_k8_geometry.py --evaluate        # both terminals + CDL-C-only g4_joint reference

Training reuses the group-size trainer unchanged; this process swaps in the geometry
configuration, the arm's layout and a training data stream whose codec-training MUs
get per-user angular offsets. Evaluation scores, on the channel-shift scenarios
rot0/1/2/4/8 (validation frames, offsets and noise identical to the channel-shift
report, unseen in training):
  fdma_geo            FDMA trained with offsets
  hybrid_geo          paired Hybrid trained with offsets, fixed consecutive pairs
  hybrid_geo_paired   the same model with least-coupled pairing from an independent pilot sounding
  g4_joint_cdlc       the CDL-C-only paired Hybrid at the same budget (reference, fixed pairs)
  perfect_csi_rzf     RZF with the true DL channel
No test frames are read; the frozen K8 FDMA is not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

from configs import twc_v1_k8_geometry_config as cfg
import final_runtime as fr
import grouped_runtime as gr
import train_twc_v1_k8_grouped as gt

VARIANTS = ('fdma_geo', 'hybrid_geo', 'hybrid_geo_paired', 'g4_joint_cdlc', 'perfect_csi_rzf')
METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'direction_error', 'interference_power_mean',
           'min_user_rate_bps_hz', 'ul_nmse_linear', 'pair_coupling', 'true_ul_coupling', 'key_energy_fraction')


def _seed(*parts):
    raw = ':'.join(str(p) for p in (cfg.NAMESPACE, cfg.SEED)+parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def train_ramp(grid, index, batch, device):
    """Per-user phase ramp [B,K,Nt]; per MU a range D from TRAIN_BINS, offsets uniform in [-D,D] bins per axis."""
    import torch
    h, v, nh, nv = grid
    gen = torch.Generator().manual_seed(_seed('train_rotation', index))
    choice = torch.randint(len(cfg.TRAIN_BINS), (batch,), generator=gen)
    bins = torch.tensor(cfg.TRAIN_BINS, dtype=torch.float64)[choice]
    offsets = (2*torch.rand(batch, fr.K, 2, generator=gen, dtype=torch.float64)-1)*bins[:, None, None]
    u, w = offsets[..., 0]*2/nh, offsets[..., 1]*2/nv
    phase = math.pi*(u[..., None]*h.double()[None, None]+w[..., None]*v.double()[None, None])
    return torch.polar(torch.ones_like(phase), phase).to(torch.complex64).to(device)


class RotatedTraining:
    """Codec-training MUs get fresh per-user angular offsets; every other role is unchanged."""
    def __init__(self, data, grid):
        self.data, self.grid = data, grid

    def __getattr__(self, name):
        return getattr(self.data, name)

    def sample(self, role, index):
        sample = self.data.sample(role, index)
        if role != 'codec_train':
            return sample
        ramp = train_ramp(self.grid, index, sample['H_dl'].shape[0], sample['H_dl'].device)
        return dict(sample, H_dl=sample['H_dl']*ramp[..., None], H_ul=sample['H_ul']*ramp[..., None])


def make_train_data(device):
    import evaluate_twc_v1_k8_channel_shift as cs
    source = fr.base().original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    data = source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4, validation=4, test=1),
        device=str(device), cdl_model='C', speed=1., seed_namespace=cfg.NAMESPACE)
    return RotatedTraining(data, cs.antenna_grid(cs.bs_positions(data)))


def build(arm, device, ce, table):
    import torch
    from models.twc_v1_k8_icc_aligned import create_codec
    layout = gr.make_layout(cfg.ARMS[arm])
    codec = create_codec('hybrid_g4', device=str(device), seed=cfg.CODEC_INIT_SEED)
    gr.set_geometry(codec.decoder, layout)
    model = fr.FinalCodec(codec).to(device)
    n = gr.active_dimension(layout)
    stats = gr.NodeStatistics(layout, table.detach().cpu(), torch.full((8, n), 48./n)).to(device)
    return dict(kind='grouped', codec=model, ce=ce, stats=stats, layout=layout)


_SAVED = {}


def install():
    """Point the group-size trainer/runtime at this study (this process only); returns a restore function."""
    if not _SAVED:
        _SAVED.update(gt_cfg=gt.cfg, gr_cfg=gr.cfg, build=gt.build, make_train_data=gr.make_train_data)
    gt.cfg, gr.cfg, gt.build, gr.make_train_data = cfg, cfg, build, make_train_data

    def restore():
        gt.cfg, gr.cfg, gt.build = _SAVED['gt_cfg'], _SAVED['gr_cfg'], _SAVED['build']
        gr.make_train_data = _SAVED['make_train_data']
    return restore


def load_reference(device, ce_bundle=None):
    """The CDL-C-only g4_joint terminal (decoder comparison), loaded with its own study's checks."""
    import train_twc_v1_k8_decoder_cmp as dc
    saved = (gt.cfg, gr.cfg, gt.build)
    dc.install()
    try:
        return gt.load_terminal('g4_joint', device, ce_bundle=ce_bundle)
    finally:
        gt.cfg, gr.cfg, gt.build = saved


def paired_forward(system, sample, gamma, noise, polarity, sounding):
    """Least-coupled pairs from a pilot-only sounding with independent noise, then the grouped forward."""
    import torch
    import evaluate_twc_v1_k8_extended as ext
    phy = fr.physics()
    H_ul = sample['H_ul']
    pilots = phy.known_pilots(H_ul.shape[0], H_ul.device, polarity)
    Y0 = phy.transmit(H_ul, pilots, sounding, gamma)
    with torch.no_grad():
        est0 = system['ce'](Y0, pilots, phy.fourier_ls(Y0, pilots), gamma_db=gamma)
        perm = ext.choose_pairing(est0, 'least')
    moved = dict(sample, H_dl=fr.permute_users(sample['H_dl'], perm), H_ul=fr.permute_users(H_ul, perm))
    return gr.forward(system, moved, gamma, noise, polarity)


def run(systems, data, grid, batches, device, rotations=None):
    import torch
    import evaluate_twc_v1_k8_channel_shift as cs
    import evaluate_twc_v1_k8_extended as ext
    from configs import twc_v1_k8_final_config as kcfg
    rows = []
    for d in (cfg.EVAL_ROTATIONS if rotations is None else rotations):
        scenario = f'rot{d}'
        for b in range(batches):
            sample = cs.scenario_sample(scenario, {'C': data}, grid, b, device)
            coupling = cs.true_coupling(sample['H_ul'])
            for gi, gamma in enumerate(kcfg.GAMMAS):
                index = b*len(kcfg.GAMMAS)+gi
                noise, polarity = fr.batch_random('final', index, device)
                sound = ext.sounding_noise('final', index, device)
                g = torch.full((kcfg.MICRO_BATCH,), float(gamma), device=device)
                for name in VARIANTS:
                    with torch.inference_mode():
                        if name == 'perfect_csi_rzf':
                            metrics = {k: v for k, v in ext.upper_bound_metrics(sample['H_dl']).items()
                                       if not k.startswith('sum_rate_dl')}
                        else:
                            system = systems['hybrid_geo' if name == 'hybrid_geo_paired' else name]
                            if name == 'hybrid_geo_paired':
                                out = paired_forward(system, sample, g, noise, polarity, sound)
                            else:
                                out = gr.forward(system, sample, g, noise, polarity)
                            metrics = fr.sample_metrics(out)
                            if name != 'hybrid_geo_paired':     # users are permuted in the paired variant
                                ul = sample['H_ul']
                                err = (out['estimated']-ul).abs().square().sum((2, 3))/ul.abs().square().sum((2, 3))
                                metrics['ul_nmse_linear'] = err.mean(1).float().cpu()
                    metrics['true_ul_coupling'] = coupling
                    for j in range(kcfg.MICRO_BATCH):
                        row = dict(variant=f'{scenario}:{name}', role='final', gamma_db=float(gamma),
                            sample_index=b*kcfg.MICRO_BATCH+j, batch_index=b,
                            **{k: float(v[j]) for k, v in metrics.items() if k in METRICS})
                        if name.startswith('fdma'):
                            row.pop('pair_coupling', None)
                            row.pop('key_energy_fraction', None)
                        rows.append(row)
            if (b+1) % 16 == 0:
                print(f'{scenario} {b+1}/{batches}', flush=True)
    return rows


def contrasts(rotations=None):
    out = []
    for d in (cfg.EVAL_ROTATIONS if rotations is None else rotations):
        s = f'rot{d}'
        out += [(f'{s}:hybrid_minus_fdma', f'{s}:hybrid_geo', f'{s}:fdma_geo'),
                (f'{s}:paired_minus_fdma', f'{s}:hybrid_geo_paired', f'{s}:fdma_geo'),
                (f'{s}:paired_minus_fixed', f'{s}:hybrid_geo_paired', f'{s}:hybrid_geo'),
                (f'{s}:geo_minus_cdlc', f'{s}:hybrid_geo', f'{s}:g4_joint_cdlc'),
                (f'{s}:upper_minus_paired', f'{s}:perfect_csi_rzf', f'{s}:hybrid_geo_paired')]
    return tuple(out)


def evaluate(device=None, *, systems=None, manifests=None, data=None, grid=None, out_root=None,
             batches=None, rotations=None):
    import torch
    import evaluate_twc_v1_k8_channel_shift as cs
    import evaluate_twc_v1_k8_final as base_eval
    import evaluate_twc_v1_h100_test as ht
    import twc_v1_k8_final_results as results
    from configs import twc_v1_k8_final_config as kcfg
    device = device or fr.base().configure_cuda()
    batches = batches or cfg.EVAL_BATCHES
    if systems is None:
        ce_bundle = gr.load_shared_ce(device)
        systems, manifests = {}, {}
        systems['g4_joint_cdlc'], manifests['g4_joint_cdlc'] = load_reference(device, ce_bundle)
        install()
        for arm in cfg.ARMS:
            systems[arm], manifests[arm] = gt.load_terminal(arm, device, ce_bundle=ce_bundle)
    data = data or cs.make_data('C', device)
    if grid is None:
        grid = cs.antenna_grid(cs.bs_positions(data))
    started = time.perf_counter()
    rows = run(systems, data, grid, batches, device, rotations)
    path = Path(out_root or cfg.ROOT/'runs'/cfg.NAMESPACE/'reports')/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', rows, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(rows, 'final', kcfg.GAMMAS, METRICS))
    base_eval.write_csv(path/'comparisons.csv', results.compare(rows, 'final', kcfg.GAMMAS, METRICS,
                                                                contrasts(rotations)))
    base_eval.write_csv(path/'batch_means.csv', ht.batch_means(rows, ('sum_rate_bps_hz', 'min_user_rate_bps_hz')))
    h, v, nh, nv = grid
    provenance = dict(namespace=cfg.NAMESPACE, recipes={a: cfg.recipe(a) for a in cfg.ARMS}, terminals=manifests,
        code=fr.source_identity(), variants=list(VARIANTS),
        rotations=list(cfg.EVAL_ROTATIONS if rotations is None else rotations),
        antenna_grid=dict(horizontal=nh, vertical=nv), frames=dict(role='validation', first=kcfg.FINAL_FIRST,
            batches=batches, mus=batches*kcfg.MICRO_BATCH, offsets='channel-shift report seeds (unseen in training)'),
        elapsed_seconds=time.perf_counter()-started, training_updates=0, frozen_K8_FDMA_loaded=False,
        test_used=False, contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(kcfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('Geometry study: FDMA and paired Hybrid trained with per-user angular offsets, '
        'scored on unseen offsets (validation frames; test sealed). Every scenario and variant is reported.\n')
    files = {p.name: fr.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=cfg.NAMESPACE, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', choices=tuple(cfg.ARMS))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args(argv)
    if args.train:
        if not args.arm:
            parser.error('--train needs --arm')
        install()
        print(json.dumps(gt.train(args.arm, args.stop_after, args.resume), indent=2, sort_keys=True))
    else:
        evaluate()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
