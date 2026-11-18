#!/usr/bin/env python3
"""Extended paper evaluation of the frozen final Hybrid and frozen FDMA (forward passes only).

No training, no new weights. Same validation frames, feedback noise and pilot
polarity as attempt_9c173fe908c6 of the final evaluation, so ``hybrid_final``
and ``fdma_baseline`` must reproduce that report exactly (consistency check).

Variants
  fdma_baseline                frozen FDMA with CE nmse65000 (chosen on the selection set in 9c173fe908c6)
  fdma_genie_ul                FDMA, BS knows the true UL channel (pilot-cost reference)
  hybrid_final                 terminal EMA Hybrid, fixed slot pairs (1,2),(3,4),...
  hybrid_paired                least-coupled pairing from a pilot sounding with INDEPENDENT noise
  hybrid_paired_reuse_noise    as in 9c173fe908c6 (sounding reuses the feedback noise draw)
  hybrid_most_coupled          ablation: most-coupled pairing
  hybrid_random_pairing        ablation: uniformly random pairing
  hybrid_paired_no_ce_error    ablation: receiver ignores the CE error variance
  hybrid_paired_genie_ul       paired Hybrid, BS knows the true UL channel
  perfect_csi_rzf              upper bound: RZF on the true DL channel
  adaptive_paired              per-gamma choice among fdma_baseline / hybrid_final / hybrid_paired,
                               fixed on the selection set before the final set is scored
Every variant also reports the sum rate at DL SNR 10/15/20/30 dB from the same
reconstruction (the precoder is unchanged; only scoring varies).
Test remains sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import uuid

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt
import train_twc_v1_k8_final as trainer
import evaluate_twc_v1_k8_final as base_eval
import twc_v1_k8_final_results as results

NAMESPACE = 'twc_v1_k8_extended_eval_v1'
OUTPUT_ROOT = Path('runs')/NAMESPACE
SOURCE_REPORT = 'twc_v1_k8_final_hybrid_v1/reports/attempt_9c173fe908c6'
DL_SNRS = (10, 15, 20, 30)
SWITCH_CANDIDATES = ('fdma_baseline', 'hybrid_final', 'hybrid_paired')
SELECTION_VARIANTS = SWITCH_CANDIDATES
MID_VARIANTS = ('fdma_baseline', 'hybrid_final', 'hybrid_paired', 'perfect_csi_rzf')
METRICS = base_eval.METRICS+tuple(f'sum_rate_dl{s}' for s in DL_SNRS)
CONTRASTS = (
    ('final_minus_fdma', 'hybrid_final', 'fdma_baseline'),
    ('paired_minus_fdma', 'hybrid_paired', 'fdma_baseline'),
    ('adaptive_minus_fdma', 'adaptive_paired', 'fdma_baseline'),
    ('paired_minus_final', 'hybrid_paired', 'hybrid_final'),
    ('paired_minus_reuse_noise', 'hybrid_paired', 'hybrid_paired_reuse_noise'),
    ('paired_minus_most_coupled', 'hybrid_paired', 'hybrid_most_coupled'),
    ('paired_minus_random', 'hybrid_paired', 'hybrid_random_pairing'),
    ('paired_minus_no_ce_error', 'hybrid_paired', 'hybrid_paired_no_ce_error'),
    ('genie_minus_paired', 'hybrid_paired_genie_ul', 'hybrid_paired'),
    ('fdma_genie_minus_fdma', 'fdma_genie_ul', 'fdma_baseline'),
    ('genie_paired_minus_genie_fdma', 'hybrid_paired_genie_ul', 'fdma_genie_ul'),
    ('upper_minus_paired', 'perfect_csi_rzf', 'hybrid_paired'),
    ('upper_minus_fdma', 'perfect_csi_rzf', 'fdma_baseline'),
)


def _seed(role, index, kind):
    raw = f'{NAMESPACE}:{cfg.SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def sounding_noise(role, index, device):
    import torch
    gen = torch.Generator(device=device).manual_seed(_seed(role, index, 'sounding'))
    return torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, rt.NT, rt.NSC, 2, generator=gen,
        device=device).contiguous())/2**.5


def choose_pairing(estimate, mode, generator=None):
    """Permutation [B,8] placing the chosen pairs on slots (0,1),(2,3),..."""
    import torch
    pairs = torch.tensor(rt._pairings(), device=estimate.device)
    cost = rt.coupling(estimate)[:, pairs[..., 0], pairs[..., 1]].sum(-1)
    if mode == 'least':
        choice = cost.argmin(-1)
    elif mode == 'most':
        choice = cost.argmax(-1)
    elif mode == 'random':
        choice = torch.randint(len(pairs), (estimate.shape[0],), generator=generator,
            device='cpu').to(estimate.device)
    else:
        raise ValueError('pairing mode')
    return pairs[choice].reshape(-1, rt.K)


def forward(system, sample, gamma, noise, polarity, *, pairing=None, sounding=None, random_gen=None,
            genie_ul=False, ce_error_aware=True):
    """rt.forward with explicit pairing mode, sounding noise, genie UL and CE-error switch."""
    import torch
    phy = rt.physics()
    codec, ce, stats, kind = system['codec'], system['ce'], system['stats'], system['kind']
    H_dl, H_ul = sample['H_dl'], sample['H_ul']
    perm = None
    if pairing is not None:
        pilots = phy.known_pilots(H_ul.shape[0], H_ul.device, polarity)
        Y0 = phy.transmit(H_ul, pilots, noise if sounding is None else sounding, gamma)
        with torch.no_grad():
            est0 = ce(Y0, pilots, phy.fourier_ls(Y0, pilots), gamma_db=gamma)
            perm = choose_pairing(est0, pairing, random_gen)
        H_dl, H_ul = rt.permute_users(H_dl, perm), rt.permute_users(H_ul, perm)
    raw = codec.encoder(H_dl, gamma) if kind == 'final' else codec.encoder(H_dl)
    wave = phy.build_waveform(raw, stats.layout, polarity=polarity)
    Y = phy.transmit(H_ul, wave.x_grid, noise, gamma)
    with torch.no_grad():
        estimated = H_ul if genie_ul else ce(Y, wave.pilot, phy.fourier_ls(Y, wave.pilot), gamma_db=gamma)
        cevar = stats.ce_variance(gamma)
        if genie_ul or not ce_error_aware:
            cevar = torch.zeros_like(cevar)
        prior = stats.nominal_power(gamma) if kind == 'final' else stats.nominal_power()
    recovered = rt.recover_batched(Y, estimated, stats.layout, prior, cevar, gamma)
    out = codec.decoder(Y_rx=recovered['Y_res'], H_ul_est=estimated, soft=recovered['soft'],
        reliability=recovered['reliability'], pilots=wave.pilot, gamma_db=gamma, layout=stats.layout)
    return dict(H_hat=out['H_hat'], H_dl=H_dl, wave=wave, estimated=estimated, gamma=gamma,
        perm=perm, pair_coupling=rt.slot_pair_coupling(estimated))


def extra_rates(H_hat, H_dl):
    helpers = rt.base().helpers()
    out = {}
    for snr in DL_SNRS:
        _, rates = helpers.rate_objective(H_hat, H_dl, alpha=cfg.ALPHA, downlink_snr_db=snr)
        out[f'sum_rate_dl{snr}'] = rates['sum_rate_bps_hz'].detach().float().cpu()
    return out


def upper_bound_metrics(H_dl):
    _, rates = rt.base().helpers().rate_objective(H_dl, H_dl, alpha=cfg.ALPHA, downlink_snr_db=cfg.DL_SNR)
    metrics = dict(sum_rate_bps_hz=rates['sum_rate_bps_hz'],
        interference_power_mean=rates['interference_power_mean'],
        desired_signal_power_mean=rates['desired_signal_power_mean'],
        min_user_rate_bps_hz=rates['user_rate_bps_hz'].min(-1).values)
    metrics = {k: v.detach().float().cpu() for k, v in metrics.items()}
    return dict(metrics, **extra_rates(H_dl, H_dl))


def variants(device, *, loaded=None):
    systems, manifest, sources = base_eval.systems(device, loaded=loaded)
    fdma, final = systems['fdma_ce65000'][0], systems['hybrid_final'][0]
    table = dict(
        fdma_baseline=(fdma, {}),
        fdma_genie_ul=(fdma, dict(genie_ul=True)),
        hybrid_final=(final, {}),
        hybrid_paired=(final, dict(pairing='least', sounding=True)),
        hybrid_paired_reuse_noise=(final, dict(pairing='least')),
        hybrid_most_coupled=(final, dict(pairing='most', sounding=True)),
        hybrid_random_pairing=(final, dict(pairing='random', sounding=True)),
        hybrid_paired_no_ce_error=(final, dict(pairing='least', sounding=True, ce_error_aware=False)),
        hybrid_paired_genie_ul=(final, dict(pairing='least', sounding=True, genie_ul=True)),
        perfect_csi_rzf=(None, {}))
    return table, manifest


def run_role(table, names, data, role, first, batches, gammas, device):
    import torch
    rows = []
    for b in range(batches):
        sample = data.sample('validation', first+b)
        for gi, gamma in enumerate(gammas):
            index = b*len(gammas)+gi
            noise, polarity = rt.batch_random(role, index, device)
            sound = sounding_noise(role, index, device)
            g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
            for name in names:
                system, options = table[name]
                with torch.inference_mode():
                    if system is None:
                        metrics = upper_bound_metrics(sample['H_dl'])
                    else:
                        options = dict(options)
                        if options.pop('sounding', False):
                            options['sounding'] = sound
                        if options.get('pairing') == 'random':
                            options['random_gen'] = torch.Generator().manual_seed(_seed(role, index, 'random_pairing'))
                        result = forward(system, sample, g, noise, polarity, **options)
                        metrics = dict(rt.sample_metrics(result), **extra_rates(result['H_hat'], result['H_dl']))
                for j in range(cfg.MICRO_BATCH):
                    row = dict(variant=name, role=role, gamma_db=float(gamma), sample_index=b*cfg.MICRO_BATCH+j,
                        batch_index=b, **{k: float(v[j]) for k, v in metrics.items()})
                    if name.startswith('fdma'):
                        row.pop('key_energy_fraction', None); row.pop('pair_coupling', None)
                    rows.append(row)
        if (b+1) % 16 == 0:
            print(f'{role} {b+1}/{batches}', flush=True)
    return rows


def choose_switch(selection_rows):
    """Per gamma node: the candidate with the highest selection-set mean sum rate."""
    sums = {}
    for row in selection_rows:
        if row['variant'] in SWITCH_CANDIDATES:
            sums.setdefault((float(row['gamma_db']), row['variant']), []).append(float(row['sum_rate_bps_hz']))
    decision = {}
    for gamma in sorted({g for g, _ in sums}):
        means = {v: statistics.mean(sums[gamma, v]) for v in SWITCH_CANDIDATES}
        decision[gamma] = dict(mode=max(SWITCH_CANDIDATES, key=means.get), means=means)
    return decision


def materialize(rows, role, decision):
    return [dict(row, variant='adaptive_paired') for row in rows
            if row['role'] == role and decision.get(float(row['gamma_db']), {}).get('mode') == row['variant']]


def evaluate(device=None, *, loaded=None, data=None, out_root=None, sizes=None):
    import torch
    device = device or rt.base().configure_cuda()
    sizes = sizes or dict(selection=cfg.SELECTION_BATCHES, final=cfg.FINAL_BATCHES, midpoint=cfg.MID_BATCHES)
    table, manifest = variants(device, loaded=loaded)
    data = data or rt.make_data(device)
    started = time.perf_counter()
    selection = run_role(table, SELECTION_VARIANTS, data, 'selection', cfg.SELECTION_FIRST,
        sizes['selection'], cfg.GAMMAS, device)
    decision = choose_switch(selection)
    final = run_role(table, tuple(table), data, 'final', cfg.FINAL_FIRST, sizes['final'], cfg.GAMMAS, device)
    final += materialize(final, 'final', decision)
    mid = run_role(table, MID_VARIANTS, data, 'midpoint', cfg.MID_FIRST, sizes['midpoint'], cfg.MID_GAMMAS, device)
    root = Path(out_root or OUTPUT_ROOT/'reports')
    path = root/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', selection+final+mid, compress=True)
    summary = (results.summarize(final, 'final', cfg.GAMMAS, METRICS)
               + results.summarize(mid, 'midpoint', cfg.MID_GAMMAS, METRICS))
    comparisons = (results.compare(final, 'final', cfg.GAMMAS, METRICS, CONTRASTS)
                   + results.compare(mid, 'midpoint', cfg.MID_GAMMAS, METRICS, CONTRASTS))
    base_eval.write_csv(path/'summary.csv', summary)
    base_eval.write_csv(path/'comparisons.csv', comparisons)
    provenance = dict(namespace=NAMESPACE, source_report=SOURCE_REPORT, final_recipe=cfg.recipe(),
        terminal_manifest=manifest, code=rt.source_identity(), fdma_baseline='fdma_ce65000',
        switch_candidates=list(SWITCH_CANDIDATES), switch_decision={str(k): v for k, v in decision.items()},
        variants=list(table)+['adaptive_paired'], dl_snrs_db=list(DL_SNRS),
        elapsed_seconds=time.perf_counter()-started,
        samples=dict(selection=sizes['selection']*cfg.MICRO_BATCH, final=sizes['final']*cfg.MICRO_BATCH,
            midpoint=sizes['midpoint']*cfg.MICRO_BATCH),
        training_updates=0, FDMA_updates=0, test_used=False, contains_model_weights=False,
        contains_channel_tensors=False, torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('Extended evaluation of the frozen final Hybrid and frozen FDMA '
        '(validation role; test sealed; no training). Switch fixed on the selection set. '
        'Hybrid received more cumulative training updates than FDMA.\n')
    files = sorted(p.name for p in path.iterdir())
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=NAMESPACE, export_id=path.name[8:],
        files={n: rt.sha256(path/n) for n in files}), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--evaluate', action='store_true', required=True)
    parser.parse_args()
    evaluate()


if __name__ == '__main__':
    main()
