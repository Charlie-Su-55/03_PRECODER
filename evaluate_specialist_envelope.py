#!/usr/bin/env python3
"""Evaluate the K4 specialist envelope on one shared set of channel samples.

Formal:    python evaluate_specialist_envelope.py
Preflight: python evaluate_specialist_envelope.py --preflight
Paths:     python evaluate_specialist_envelope.py --dry-run

No training is performed. Existing result directories are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from configs.baseline_config import make_config
from configs.train_config import get_suite
from train_proposal_specialist import K4_ALLOCATIONS, checkpoint_status, specialist_config

ROOT = Path(__file__).resolve().parent
CASE = 'k4_n256_d256'
EVALUATION_SEED = 20260921
DEFAULT_OUTPUT = 'runs_as_screen/evaluation/k4_n256_d256_specialist_envelope'


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def checkpoint_plan(args):
    proposal_root = project_path(args.proposal_root)
    baseline_root = project_path(args.baseline_root)
    specialist_root = project_path(args.specialist_root)
    universal_cfg = next(
        cfg for cfg in get_suite('proposal_adaptive', output_root=proposal_root)
        if (cfg.K_USERS, cfg.SUBCARRIERS, cfg.D_TOT) == (4, 256, 256)
    )
    plan = [{
        'method_key': 'universal', 'model_key': 'proposed',
        'method_type': 'universal', 'label': 'Proposed universal',
        'path': project_path(args.universal_checkpoint or universal_cfg.BEST_PATH),
        'allocations': K4_ALLOCATIONS,
    }]
    for allocation in K4_ALLOCATIONS:
        cfg = specialist_config(allocation, specialist_root)
        df, da = allocation
        plan.append({
            'method_key': f'specialist_df{df}_da{da}', 'model_key': 'proposed',
            'method_type': 'specialist', 'label': f'Specialist ({df},{da})',
            'path': Path(cfg.BEST_PATH), 'allocations': (allocation,),
        })
    for model, label in (('swin', 'Swin'), ('csinet', 'CsiNet+')):
        cfg = make_config(model, CASE, objective='task', output_root=str(baseline_root))
        plan.append({
            'method_key': model, 'model_key': model, 'method_type': label,
            'label': f'{label}-based FDMA',
            'path': project_path(getattr(args, f'{model}_checkpoint') or cfg.BEST_PATH),
            'allocations': ((64, 0),),
        })
    return plan


def check_paths(plan):
    errors = []
    for entry in plan:
        path = entry['path']
        status = checkpoint_status(path)
        entry['training_status'] = status.get('state', 'unknown')
        print(f"\n[{entry['method_key']}] {entry['allocations']}; status={entry['training_status']}")
        print(f'  {path}')
        if not path.is_file():
            errors.append(f'Missing checkpoint: {path}')
        elif entry['training_status'] not in {'completed', 'unknown'}:
            errors.append(f"Run is not completed ({entry['training_status']}): {path}")
        elif entry['training_status'] == 'unknown':
            print('  Completion metadata unavailable; provenance is recorded as unknown.')
    if len({entry['path'].resolve() for entry in plan}) != len(plan):
        errors.append('Different methods must not reuse the same checkpoint path.')
    if errors:
        raise RuntimeError('\n'.join(errors))


def check_results(summary, samples, plan, expected_samples):
    """Reject incomplete/mixed grids before publishing the formal CSV."""
    expected = {
        (entry['method_key'], df, da)
        for entry in plan for df, da in entry['allocations']
    }
    actual = set()
    sample_ids = {key: set() for key in expected}
    for row in summary:
        key = (row['method_key'], row['allocation_df'], row['allocation_da'])
        if key not in expected or key in actual:
            raise RuntimeError(f'Unexpected or duplicate summary row: {key}')
        actual.add(key)
        if (
            (row['k_users'], row['subcarriers'], row['feedback_budget']) != (4, 256, 256)
            or row['num_samples'] != expected_samples
            or row['feedback_snr_db'] != 25.0 or row['downlink_snr_db'] != 25.0
            or row['seed'] != EVALUATION_SEED
        ):
            raise RuntimeError(f'Unexpected evaluation setting: {key}')
        for field in ('sum_rate_mean', 'sum_rate_se', 'nmse_db_mean'):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f'Non-finite {field}: {key}')
    if actual != expected:
        raise RuntimeError(f'Missing summary rows: {expected - actual}')
    for row in samples:
        key = (row['method_key'], row['allocation_df'], row['allocation_da'])
        if key not in sample_ids or row['sample_index'] in sample_ids[key]:
            raise RuntimeError(f'Unexpected or duplicate sample row: {key}')
        sample_ids[key].add(row['sample_index'])
    target_ids = set(range(expected_samples))
    if any(ids != target_ids for ids in sample_ids.values()):
        raise RuntimeError('Methods do not have identical complete sample-index ranges.')


def plot_rows(summary):
    """One row per split; baseline columns are fixed FDMA references."""
    indexed = {
        (row['method_type'], row['allocation_df'], row['allocation_da']): row
        for row in summary
    }
    result = []
    for df, da in K4_ALLOCATIONS:
        row = {'allocation_df': df, 'allocation_da': da, 'as_fraction': da / 256.0}
        for prefix, source in (
            ('universal', indexed[('universal', df, da)]),
            ('specialist', indexed[('specialist', df, da)]),
            ('swin_reference', indexed[('Swin', 64, 0)]),
            ('csinet_reference', indexed[('CsiNet+', 64, 0)]),
        ):
            for metric in ('sum_rate_mean', 'sum_rate_se', 'nmse_db_mean'):
                row[f'{prefix}_{metric}'] = source[metric]
        row['specialist_minus_universal_bps_hz'] = (
            row['specialist_sum_rate_mean'] - row['universal_sum_rate_mean']
        )
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-root', default='runs_deployzf_v1')
    parser.add_argument('--baseline-root', default='runs_as_screen/baselines')
    parser.add_argument('--specialist-root', default='runs_as_screen/proposal_specialists')
    parser.add_argument('--universal-checkpoint', '--proposed-checkpoint', dest='universal_checkpoint')
    parser.add_argument('--swin-checkpoint')
    parser.add_argument('--csinet-checkpoint')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--preflight', action='store_true', help='One batch of 8 samples, separate output.')
    parser.add_argument('--dry-run', action='store_true', help='Inspect paths/status only; no torch import.')
    args = parser.parse_args()

    plan = checkpoint_plan(args)
    check_paths(plan)
    num_batches = 1 if args.preflight else 200
    sample_count = 8 * num_batches
    print(f'\n[Protocol] K=4, Nsc=256, Dtot=256; FB=DL=25 dB; seed={EVALUATION_SEED}')
    print(f'[Samples] batch_size=8, num_batches={num_batches}, total={sample_count}')
    print('[Methods] 1 universal x 5 splits + 5 specialists x 1 split + 2 FDMA baselines')
    print('[NMSE] Raw representation error; secondary diagnostic, not a ranking criterion.')
    if args.dry_run:
        return 0

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    default_output = DEFAULT_OUTPUT + (f'_preflight_{stamp}' if args.preflight else '')
    output = project_path(args.output_dir or default_output)
    if output.exists():
        raise FileExistsError(f'Results will not be overwritten: {output}. Use a new --output-dir.')

    import torch
    import baseline_evaluate as ev
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Run evaluation on the GPU server.')
    torch.set_float32_matmul_precision('high')
    device = ev.setup_gpu()
    scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
    scenario.validate()
    tag = 'specialist_envelope_preflight' if args.preflight else 'specialist_envelope'
    evaluation_args = ev.build_parser().parse_args([
        '--scenario', '4,256,256', '--models', 'proposed,swin,csinet',
        '--allocations', 'all', '--checkpoint', 'best',
        '--feedback-snr', '25', '--downlink-snr', '25',
        '--batch-size', '8', '--num-batches', str(num_batches),
        '--seed', str(EVALUATION_SEED), '--no-include-proposed-best',
        '--baseline-objective', 'task', '--extra-metrics', 'basic',
        '--proposal-root', str(project_path(args.proposal_root)),
        '--baseline-root', str(project_path(args.baseline_root)),
        '--output-dir', str(output), '--tag', tag,
    ])
    methods = []
    for entry in plan:
        if entry['model_key'] == 'proposed':
            method = ev.build_proposed_method(scenario, entry['path'], 'best', 'all', device)
            # Inspect the whole supported grid, not a requested subset of a generalist.
            if tuple(method.config.GAMMA_GRID) != entry['allocations']:
                raise RuntimeError(f"Checkpoint allocation/type mismatch: {entry['path']}")
            if (method.config.D_F_MAX_PER_USER, method.config.D_A_MAX_SHARED) != (64, 256):
                raise RuntimeError(f"Checkpoint architecture mismatch: {entry['path']}")
        else:
            method = ev.build_baseline_method(
                entry['model_key'], scenario, entry['path'], 'best', device, evaluation_args.rzf_reg,
            )
        method.method_key = entry['method_key']  # model_key remains the forward-dispatch key.
        method.display_name = entry['label']
        methods.append(method)

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'created_utc': stamp, 'preflight': args.preflight,
        'scenario': CASE, 'seed': EVALUATION_SEED,
        'batch_size': 8, 'num_batches': num_batches, 'num_samples': sample_count,
        'feedback_snr_db': 25, 'downlink_snr_db': 25, 'strict_loading': True,
        'methods': [
            {**entry, 'path': str(entry['path']), 'checkpoint_step': method.checkpoint_step}
            for entry, method in zip(plan, methods)
        ],
        'nmse_definition': 'Mean of per-sample dB errors of raw antenna-frequency representations against H_dl.',
        'nmse_role': 'Secondary reconstruction diagnostic; training objectives differ.',
        'sampling': 'One channel batch shared by all methods inside evaluate_scenario.',
        'noise': 'Existing relative-noise convention and per-condition seeding unchanged.',
        'envelope': 'One fixed specialist per split; no maximization over test outcomes.',
    }
    with (output / 'envelope_manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)

    # Generate each channel batch once and pass the very same tensors to all models.
    ev.set_seed(EVALUATION_SEED)
    snr_points = [(25.0, 25.0)]
    summary, samples = ev.evaluate_scenario(scenario, methods, snr_points, evaluation_args, device)
    check_results(summary, samples, plan, sample_count)
    by_key = {entry['method_key']: entry for entry in plan}
    for row in summary + samples:
        entry = by_key[row['method_key']]
        row['method_type'] = entry['method_type']
        row['as_fraction'] = row['allocation_da'] / 256.0
        row['training_status'] = entry['training_status']
    ev.print_summary(summary)
    ev.save_outputs(
        summary, samples, [scenario], [method.method_key for method in methods],
        snr_points, evaluation_args,
    )
    ev.write_csv(output / 'plot_specialist_envelope.csv', plot_rows(summary))
    print(f'\n[Complete] {len(summary)} rows, {sample_count} samples per row: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
