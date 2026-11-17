#!/usr/bin/env python3
"""Paper evaluation for the final Hybrid model (forward passes only, FDMA frozen).

Variants (all share each MU's channel, noise and pilot polarity):
  fdma_ce5000 / fdma_ce65000      frozen FDMA with its own CE, or with CE nmse65000
  hybrid_source                    low-SNR interference Hybrid terminal (CE5000)
  hybrid_final                     terminal weight EMA of the final training
  hybrid_final_paired              + least-coupled user pairing from the pilot CE estimate
  hybrid_final_ce5000              ablation: original CE and its table
  hybrid_final_fixed_gamma         ablation: UE conditioner frozen at gamma=12.5 dB
  hybrid_final_pooled_prior        ablation: one power prior pooled over nodes
The FDMA baseline and the per-gamma adaptive switch are fixed on a separate
selection set before the final set is scored. Test remains sealed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import time
import uuid

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt
import train_twc_v1_k8_final as trainer
import twc_v1_k8_final_results as results

METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'direction_error', 'interference_power_mean',
           'desired_signal_power_mean', 'min_user_rate_bps_hz', 'key_energy_fraction', 'pair_coupling')


def systems(device, *, loaded=None):
    final, manifest, sources = loaded or trainer.load_terminal_ema(device)
    fdma, fdma_stats, fdma_ce, _ = sources['fdma']
    hybrid, hybrid_stats, ce5000, _ = sources['hybrid']
    ce65000, _ = sources['ce65000']
    table, _ = sources['table65000']
    old_table_final = rt.NodeStatistics(final['stats'].layout, hybrid_stats.ce_errors.detach().cpu(),
        hybrid_stats.power_ema.detach().cpu()).to(table.device)
    old_table_final.load_state_dict(dict(final['stats'].state_dict(), ce_errors=hybrid_stats.ce_errors))
    for module in (fdma, hybrid, fdma_stats, hybrid_stats, old_table_final):
        module.eval().requires_grad_(False)
    plain = lambda codec, ce, stats: dict(kind='plain', codec=codec, ce=ce, stats=stats)
    variants = dict(
        fdma_ce5000=(plain(fdma, fdma_ce, fdma_stats), {}),
        fdma_ce65000=(plain(fdma, ce65000, rt.swapped_statistics(fdma_stats, table)), {}),
        hybrid_source=(plain(hybrid, ce5000, hybrid_stats), {}),
        hybrid_final=(final, {}),
        hybrid_final_paired=(final, dict(pairing=True)),
        hybrid_final_ce5000=(dict(final, ce=ce5000, stats=old_table_final), {}),
        hybrid_final_fixed_gamma=(final, dict(encoder_gamma=12.5)),
        hybrid_final_pooled_prior=(final, dict(pooled_prior=True)))
    return variants, manifest, sources


def run_role(variants, names, data, role, first, batches, gammas, device):
    import torch
    rows = []
    for b in range(batches):
        sample = data.sample('validation', first+b)
        for gi, gamma in enumerate(gammas):
            noise, polarity = rt.batch_random(role, b*len(gammas)+gi, device)
            g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
            for name in names:
                system, options = variants[name]
                with torch.inference_mode():
                    metrics = rt.sample_metrics(rt.forward(system, sample, g, noise, polarity, **options))
                for j in range(cfg.MICRO_BATCH):
                    row = dict(variant=name, role=role, gamma_db=float(gamma), sample_index=b*cfg.MICRO_BATCH+j,
                        batch_index=b, **{k: float(v[j]) for k, v in metrics.items()})
                    if name.startswith('fdma'):
                        row.pop('key_energy_fraction', None); row.pop('pair_coupling', None)
                    rows.append(row)
        if (b+1) % 16 == 0:
            print(f'{role} {b+1}/{batches}', flush=True)
    return rows


def write_csv(path, rows, *, compress=False):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    path.write_bytes(gzip.compress(data, mtime=0) if compress else data)


def curve_rows(progress_path):
    rows = []
    for line in Path(progress_path).read_text().splitlines():
        record = json.loads(line)
        for metric in ('sum_rate_bps_hz', 'low_gamma_sum_rate', 'direction_error', 'key_energy_fraction', 'loss'):
            rows.append(dict(variant='hybrid_final_training', role='training', scope='all', step=record['step'],
                metric=metric, mean=record[metric]))
    return rows


def evaluate(device=None, *, loaded=None, data=None, out_root=None, sizes=None):
    import torch
    device = device or rt.base().configure_cuda()
    sizes = sizes or dict(selection=cfg.SELECTION_BATCHES, final=cfg.FINAL_BATCHES, midpoint=cfg.MID_BATCHES)
    variants, manifest, sources = systems(device, loaded=loaded)
    data = data or rt.make_data(device)
    started = time.perf_counter()
    selection = run_role(variants, ('fdma_ce5000', 'fdma_ce65000', 'hybrid_final'), data, 'selection',
        cfg.SELECTION_FIRST, sizes['selection'], cfg.GAMMAS, device)
    fdma_variant, fdma_means = results.choose_fdma_baseline(selection)
    decision = results.choose_switch(selection, fdma_variant)
    final = run_role(variants, tuple(variants), data, 'final', cfg.FINAL_FIRST, sizes['final'], cfg.GAMMAS, device)
    final += results.materialize(final, 'final', fdma_variant, decision)
    mid = run_role(variants, ('fdma_ce5000', 'fdma_ce65000', 'hybrid_final'), data, 'midpoint',
        cfg.MID_FIRST, sizes['midpoint'], cfg.MID_GAMMAS, device)
    mid += [dict(r, variant='fdma_baseline') for r in mid if r['variant'] == fdma_variant]
    root = Path(out_root or cfg.OUTPUT_ROOT/'reports')
    path = root/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    write_csv(path/'samples.csv.gz', selection+final+mid, compress=True)
    summary = results.summarize(final, 'final', cfg.GAMMAS, METRICS)+results.summarize(mid, 'midpoint', cfg.MID_GAMMAS, METRICS)
    comparisons = results.compare(final, 'final', cfg.GAMMAS, METRICS)+results.compare(mid, 'midpoint', cfg.MID_GAMMAS, METRICS)
    write_csv(path/'summary.csv', summary)
    write_csv(path/'comparisons.csv', comparisons)
    progress = Path(manifest.get('progress_path', trainer.TRAIN_DIR/'progress.jsonl'))
    if progress.is_file():
        write_csv(path/'curve_summary.csv', curve_rows(progress))
    provenance = dict(namespace=cfg.NAMESPACE, recipe=cfg.recipe(), fingerprint=cfg.fingerprint(),
        terminal_manifest=manifest, code=rt.source_identity(), fdma_baseline=fdma_variant,
        fdma_selection_means=fdma_means, switch_decision={str(k): v for k, v in decision.items()},
        variants=list(variants), elapsed_seconds=time.perf_counter()-started,
        samples=dict(selection=sizes['selection']*cfg.MICRO_BATCH, final=sizes['final']*cfg.MICRO_BATCH,
            midpoint=sizes['midpoint']*cfg.MICRO_BATCH),
        FDMA_updates=0, test_used=False, contains_model_weights=False, contains_channel_tensors=False,
        torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('Final Hybrid-G4 evaluation (validation role; test sealed). '
        'FDMA baseline and adaptive switch fixed on the selection set. FDMA has zero new updates; '
        'Hybrid received more cumulative training updates than FDMA.\n')
    files = {p.name: rt.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=cfg.NAMESPACE, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true', required=True)
    parser.parse_args(argv)
    evaluate()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
