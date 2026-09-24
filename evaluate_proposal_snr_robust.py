#!/usr/bin/env python3
"""Five-curve fixed-DL feedback-SNR sweep using baseline_evaluate unchanged.

Default: FB=[0,5,10,15,20,25], DL=25, seed=20260921, 8x200 shared samples.
No training. New output directories only; paper data/figures are never written.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from evaluate_specialist_envelope import (
    EVALUATION_SEED, check_paths, checkpoint_plan, project_path,
)
from train_proposal_specialist import K4_ALLOCATIONS
from train_proposal_snr_robust import RECIPE, ROBUST_ROOT, robust_config

ROOT = Path(__file__).resolve().parent
FB_SNRS = (0, 5, 10, 15, 20, 25)
DEFAULT_OUTPUT = 'runs_as_screen/evaluation/k4_pure_as_fb_robust'
HISTORICAL_LABELS = {
    'universal': 'Universal Proposed',
    'specialist_df0_da256': 'Allocation-specific specialist',
    'swin': 'Swin-CFNet', 'csinet': 'CsiNet+',
}


def verify_results(summary, samples, plan, count, seed):
    expected = {
        (entry['method_key'], *entry['allocations'][0], float(snr))
        for entry in plan for snr in FB_SNRS
    }
    actual = set()
    ids = {key: set() for key in expected}
    for row in summary:
        key = (row['method_key'], row['allocation_df'], row['allocation_da'], row['feedback_snr_db'])
        if key not in expected or key in actual:
            raise RuntimeError(f'Unexpected or duplicate result: {key}')
        actual.add(key)
        if (
            (row['k_users'], row['subcarriers'], row['feedback_budget']) != (4, 256, 256)
            or row['num_samples'] != count or row['seed'] != seed or row['downlink_snr_db'] != 25
            or not all(math.isfinite(float(row[name])) for name in ('sum_rate_mean', 'sum_rate_se', 'nmse_db_mean'))
        ):
            raise RuntimeError(f'Invalid protocol or metric: {key}')
    if actual != expected:
        raise RuntimeError(f'Missing results: {expected - actual}')
    for row in samples:
        key = (row['method_key'], row['allocation_df'], row['allocation_da'], row['feedback_snr_db'])
        if key not in ids or row['sample_index'] in ids[key]:
            raise RuntimeError(f'Unexpected or duplicate sample: {key}')
        ids[key].add(row['sample_index'])
    if any(value != set(range(count)) for value in ids.values()):
        raise RuntimeError('Incomplete shared sample-index ranges.')


def historical_comparison(summary, reference_path):
    """Read-only check: the paper CSV has means/SEs but no seed metadata."""
    if not reference_path.is_file():
        print('[Historical check] Reference CSV absent; no historical numbers merged.')
        return []
    with reference_path.open(newline='', encoding='utf-8-sig') as handle:
        reference = {
            (row['curve'], float(row['feedback_snr_db'])): float(row['sum_rate_mean'])
            for row in csv.DictReader(handle)
        }
    rows = []
    for row in summary:
        label = HISTORICAL_LABELS.get(row['method_key'])
        key = (label, row['feedback_snr_db'])
        if key in reference:
            rows.append({
                'curve': label, 'feedback_snr_db': row['feedback_snr_db'],
                'historical_sum_rate_mean': reference[key],
                'rerun_sum_rate_mean': row['sum_rate_mean'],
                'delta_bps_hz': row['sum_rate_mean'] - reference[key],
            })
    if rows:
        delta = max(abs(row['delta_bps_hz']) for row in rows)
        print(f'[Historical check] {len(rows)}/24 matched points; maximum absolute mean difference={delta:.6g}')
        if delta > 1e-3 or len(rows) != 24:
            print('[CHECK BEFORE PAPER USE] Check old sweep seed/checkpoint/protocol metadata; do not mix historical means.')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-root', default='runs_deployzf_v1')
    parser.add_argument('--specialist-root', default='runs_as_screen/proposal_specialists')
    parser.add_argument('--baseline-root', default='runs_as_screen/baselines')
    parser.add_argument('--robust-root', default=str(ROBUST_ROOT))
    parser.add_argument('--universal-checkpoint')
    parser.add_argument('--swin-checkpoint')
    parser.add_argument('--csinet-checkpoint')
    parser.add_argument('--original-checkpoint')
    parser.add_argument('--robust-checkpoint')
    parser.add_argument('--seed', type=int, default=EVALUATION_SEED, help='Match the existing sweep metadata.')
    parser.add_argument('--output-dir')
    parser.add_argument('--reference-csv', default='paper/data/k4_pure_as_feedback_snr.csv')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='8 samples at each SNR; separate output, not formal results.')
    args = parser.parse_args()

    plan = [entry for entry in checkpoint_plan(args) if entry['method_key'] in HISTORICAL_LABELS]
    for entry in plan:
        entry['curve'] = HISTORICAL_LABELS[entry['method_key']]
        if entry['model_key'] == 'proposed':
            entry['allocations'] = ((0, 256),)
        if entry['method_key'] == 'specialist_df0_da256':
            entry['label'] = 'Original pure-AS specialist'
            entry['method_type'] = 'original_specialist'
            if args.original_checkpoint:
                entry['path'] = project_path(args.original_checkpoint)
    robust = {
        'method_key': 'robust_specialist', 'model_key': 'proposed',
        'method_type': 'robust_specialist', 'label': 'FB-SNR-robust specialist',
        'curve': 'Feedback-SNR-robust specialist', 'allocations': ((0, 256),),
        'path': project_path(args.robust_checkpoint or robust_config(args.robust_root).BEST_PATH),
    }
    plan.insert(2, robust)
    check_paths(plan)
    saved_config = json.loads((robust['path'].parent / 'config.json').read_text(encoding='utf-8'))
    if (
        saved_config['training_recipe']['name'] != RECIPE
        or saved_config['training_recipe'].get('downlink_snr_db') != 25.0
        or saved_config['allocations'] != [[0, 256]]
        or robust['training_status'] != 'completed'
    ):
        raise RuntimeError('The robust curve requires the completed fixed-DL pure-AS fine-tuning run.')
    num_batches = 1 if args.preflight else 200
    count = 8 * num_batches
    print(f'[Protocol] seed={args.seed}; FB={FB_SNRS}; DL=25; batch_size=8; num_batches={num_batches}')
    print('[Protocol] Existing evaluator/sampling/noise/metrics unchanged; all five curves use the same batches.')
    print('[Seed provenance] 20260921 is the current screening/envelope default; the paper CSV has no seed metadata.')
    if args.dry_run:
        return 0

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = project_path(args.output_dir or (DEFAULT_OUTPUT + (f'_preflight_{stamp}' if args.preflight else '')))
    if output.exists():
        raise FileExistsError(f'Existing results will not be overwritten: {output}')
    import torch
    import baseline_evaluate as ev
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Run evaluation on the GPU server.')
    torch.set_float32_matmul_precision('high')
    device = ev.setup_gpu()
    scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
    scenario.validate()
    tag = 'k4_pure_as_fb_robust' + ('_preflight' if args.preflight else '')
    evaluation_args = ev.build_parser().parse_args([
        '--scenario', '4,256,256', '--models', 'proposed,swin,csinet',
        '--allocations', '0,256', '--checkpoint', 'best', '--no-include-proposed-best',
        '--feedback-snr', ','.join(map(str, FB_SNRS)), '--downlink-snr', '25',
        '--batch-size', '8', '--num-batches', str(num_batches), '--seed', str(args.seed),
        '--baseline-objective', 'task', '--extra-metrics', 'basic',
        '--proposal-root', str(project_path(args.proposal_root)),
        '--baseline-root', str(project_path(args.baseline_root)),
        '--output-dir', str(output), '--tag', tag,
    ])
    methods = []
    for entry in plan:
        if entry['model_key'] == 'proposed':
            method = ev.build_proposed_method(scenario, entry['path'], 'best', '0,256', device)
            expected_grid = K4_ALLOCATIONS if entry['method_key'] == 'universal' else ((0, 256),)
            if tuple(method.config.GAMMA_GRID) != expected_grid or (
                method.config.D_F_MAX_PER_USER, method.config.D_A_MAX_SHARED
            ) != (64, 256):
                raise RuntimeError(f"Checkpoint architecture/type mismatch: {entry['path']}")
        else:
            method = ev.build_baseline_method(
                entry['model_key'], scenario, entry['path'], 'best', device, evaluation_args.rzf_reg,
            )
        method.method_key = entry['method_key']
        method.display_name = entry['label']
        methods.append(method)

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'created_utc': stamp, 'preflight': args.preflight, 'seed': args.seed,
        'batch_size': 8, 'num_batches': num_batches, 'num_samples': count,
        'feedback_snr_db': FB_SNRS, 'downlink_snr_db': 25, 'strict_loading': True,
        'methods': [{**entry, 'path': str(entry['path']), 'checkpoint_step': method.checkpoint_step}
                    for entry, method in zip(plan, methods)],
        'sampling': 'Same H_dl/H_ul tensors for all five curves; original evaluator noise convention unchanged.',
        'reference_csv': str(project_path(args.reference_csv)),
    }
    with (output / 'sweep_manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    ev.set_seed(args.seed)
    snr_points = [(float(snr), 25.0) for snr in FB_SNRS]
    summary, samples = ev.evaluate_scenario(scenario, methods, snr_points, evaluation_args, device)
    verify_results(summary, samples, plan, count, args.seed)
    by_key = {entry['method_key']: entry for entry in plan}
    for row in summary + samples:
        row['method_type'] = by_key[row['method_key']]['method_type']
        row['curve'] = by_key[row['method_key']]['curve']
    ev.print_summary(summary)
    ev.save_outputs(summary, samples, [scenario], [method.method_key for method in methods], snr_points, evaluation_args)
    ev.write_csv(output / 'plot_k4_pure_as_fb_robust.csv', [
        {name: row[name] for name in ('curve', 'method_key', 'feedback_snr_db', 'downlink_snr_db',
                                     'allocation_df', 'allocation_da', 'num_samples', 'seed',
                                     'sum_rate_mean', 'sum_rate_se', 'nmse_db_mean')}
        for row in summary
    ])
    if not args.preflight:
        comparison = historical_comparison(summary, project_path(args.reference_csv))
        if comparison:
            ev.write_csv(output / 'historical_curve_check.csv', comparison)
    print(f'[Complete] {len(summary)} rows; {count} samples per point; {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
