#!/usr/bin/env python3
"""Server entry point for low-load FDMA--AS screening.

check: inspect paths and print a plan without importing torch or loading weights.
run: require CUDA, reuse checkpoints, train missing models, then evaluate.
train-specialists-k4: train only missing K4 specialists, preserving existing runs.
preflight-specialists-k4: load all eight checkpoints and evaluate one shared batch.
eval-specialists-k4: evaluate the full specialist envelope on 1600 shared samples.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from configs.baseline_config import make_config
from configs.train_config import get_suite

ROOT = Path(__file__).resolve().parent
CASES = {
    'k4_n256_d256': (4, 256, 256, 'proposal_adaptive'),
    'k2_n128_d128': (2, 128, 128, 'proposal_k2_screen'),
}
MODELS = ('proposed', 'swin', 'csinet')


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def training_command(model, case, output_root):
    k, _, budget, suite = CASES[case]
    if model == 'proposed':
        return [
            sys.executable, '-u', 'main_train.py', '--suite', suite,
            '--only-k', str(k), '--only-budget', str(budget),
            '--output-root', str(output_root), '--fail-fast',
        ]
    return [
        sys.executable, '-u', 'baseline_train.py', '--models', model,
        '--scenarios', case, '--objective', 'task',
        '--output-root', str(output_root), '--fail-fast',
    ]


def checkpoint_state(path):
    if len(path.parents) < 3:
        return 'unknown'
    run_root = path.parents[2]
    possible = (
        path.parent / 'status.json',
        run_root / 'logs' / path.parent.name / 'status.json',
    )
    for status_path in possible:
        if status_path.is_file():
            payload = json.loads(status_path.read_text(encoding='utf-8'))
            return payload.get('state', 'unknown')
    return 'unknown'


def build_plan(args):
    k, nsc, budget, suite = CASES[args.case]
    new_root = project_path(args.work_root)
    configs = {}
    proposal_configs = get_suite(suite, output_root=new_root / 'proposal')
    configs['proposed'] = next(
        cfg for cfg in proposal_configs
        if (cfg.K_USERS, cfg.SUBCARRIERS, cfg.D_TOT) == (k, nsc, budget)
    )
    for model in ('swin', 'csinet'):
        configs[model] = make_config(
            model, args.case, objective='task',
            output_root=str(new_root / 'baselines'),
        )

    plan = []
    errors = []
    token = f'K{k}_Nsc{nsc}_Dfb{budget}_'
    for model in MODELS:
        cfg = configs[model]
        target = Path(cfg.BEST_PATH).resolve()
        train_root = new_root / ('proposal' if model == 'proposed' else 'baselines')
        explicit = getattr(args, f'{model}_checkpoint')
        candidates = set()
        if explicit:
            path = project_path(explicit)
            if not path.is_file():
                errors.append(f'{model}: explicit checkpoint does not exist: {path}')
            else:
                candidates.add(path)
        else:
            legacy_root = project_path(
                args.proposal_root if model == 'proposed' else args.baseline_root
            )
            prefixes = ('hybrid_gnn_adaptive_', 'adaptive_') if model == 'proposed' else (
                f'{model}_task_', f'{model}_plus_task_', f'{model}_cfnet_task_',
            )
            for prefix in prefixes:
                candidates.update(
                    path.resolve()
                    for path in (legacy_root / 'checkpoints').glob(
                        f'{prefix}*{token}*/best.pth'
                    )
                )
            # Only reuse this wrapper's new run after full training has completed.
            if target.is_file() and checkpoint_state(target) == 'completed':
                candidates.add(target)

        ordered = sorted(candidates)
        if len(ordered) > 1:
            paths = '\n  '.join(str(path) for path in ordered)
            errors.append(
                f'{model}: multiple checkpoints; specify --{model}-checkpoint PATH:\n  {paths}'
            )
        selected = ordered[0] if len(ordered) == 1 else None
        state = checkpoint_state(selected) if selected else 'not_selected'
        if selected and state not in {'completed', 'unknown'} and not explicit:
            errors.append(
                f'{model}: existing run is {state}: {selected}. Finish that run, or '
                f'explicitly select --{model}-checkpoint PATH for exploratory evaluation.'
            )
        plan.append({
            'model': model,
            'action': 'reuse' if selected else 'train_or_resume',
            'checkpoint': str(selected or target),
            'training_state': state,
            'total_steps': cfg.TOTAL_STEPS,
            'command': training_command(model, args.case, train_root),
        })
    return plan, errors


def print_plan(args, plan, errors):
    print(f'Case: {args.case}; adaptive proposal + task-trained Swin + CsiNet+')
    print('DL SNR: 25 dB; feedback SNR: 0,5,10,15,20,25 dB')
    print(f'{args.samples} held-out samples per condition; all five allocations')
    print('New training uses one seed (42); this is a screening experiment.')
    for job in plan:
        print(f"\n[{job['model']}] {job['action']} ({job['training_state']})")
        print(f"  {job['checkpoint']}")
        if job['action'] == 'train_or_resume':
            print(f"  Training budget: {job['total_steps']} steps")
            print('  ' + shlex.join(job['command']))
        elif job['training_state'] == 'unknown':
            print('  Completion metadata unavailable; verify training provenance.')
    for error in errors:
        print('\n[NEEDS ATTENTION] ' + error)


def run_command(command):
    print('\n[RUN] ' + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_comparison(summary_path, output_path, args):
    """Report every fixed allocation; never pick the test-set best allocation."""
    with summary_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    k, nsc, budget, _ = CASES[args.case]
    expected = {
        (model, df, da, float(snr), 25.0)
        for model in MODELS
        for df, da in (
            [(budget * (4 - q) // (4 * k), budget * q // 4) for q in range(5)]
            if model == 'proposed' else [(budget // k, 0)]
        )
        for snr in (0, 5, 10, 15, 20, 25)
    }
    actual = set()
    for row in rows:
        if (int(row['k_users']), int(row['subcarriers']), int(row['feedback_budget'])) != (k, nsc, budget):
            raise ValueError('Scenario mismatch in evaluation summary.')
        if int(row['num_samples']) != args.samples or not math.isfinite(float(row['sum_rate_mean'])):
            raise ValueError('Invalid sample count or non-finite rate in evaluation summary.')
        key = (
            row['model_key'], int(row['allocation_df']), int(row['allocation_da']),
            float(row['feedback_snr_db']), float(row['downlink_snr_db']),
        )
        if key in actual:
            raise ValueError(f'Duplicate evaluation row: {key}')
        actual.add(key)
    if actual != expected:
        raise ValueError(f'Evaluation grid mismatch: missing={expected - actual}, extra={actual - expected}')

    def condition(row):
        return (row['scenario_id'], row['feedback_snr_db'], row['downlink_snr_db'])

    baselines = {
        (condition(row), row['model_key']): row
        for row in rows if row['model_key'] in {'swin', 'csinet'}
    }
    comparisons = []
    for row in rows:
        if row['model_key'] != 'proposed':
            continue
        df, da = int(row['allocation_df']), int(row['allocation_da'])
        label = 'pure AS' if df == 0 else ('pure FDMA' if da == 0 else 'FDMA--AS')
        for baseline in ('swin', 'csinet'):
            ref = baselines[(condition(row), baseline)]
            if row['num_samples'] != ref['num_samples']:
                raise ValueError('Sample count mismatch in evaluation summary.')
            proposal_rate = float(row['sum_rate_mean'])
            baseline_rate = float(ref['sum_rate_mean'])
            comparisons.append({
                'scenario': row['scenario_id'],
                'allocation_df': df, 'allocation_da': da, 'configuration': label,
                'feedback_snr_db': row['feedback_snr_db'],
                'downlink_snr_db': row['downlink_snr_db'],
                'num_samples': row['num_samples'], 'baseline': baseline,
                'proposal_sum_rate': proposal_rate, 'baseline_sum_rate': baseline_rate,
                'delta_bps_hz': proposal_rate - baseline_rate,
            })
    if not comparisons:
        raise ValueError('No proposal rows found in evaluation summary.')
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print('\n25-dB feedback results (positive delta means higher sample mean):')
    for row in comparisons:
        if float(row['feedback_snr_db']) == 25 and row['baseline'] == 'swin':
            print(
                f"  {row['configuration']:10s} ({row['allocation_df']},{row['allocation_da']}): "
                f"{row['proposal_sum_rate']:.3f}; Swin={row['baseline_sum_rate']:.3f}; "
                f"delta={row['delta_bps_hz']:+.3f} bps/Hz"
            )


def main():
    specialist_commands = {
        'train-specialists-k4': ['train_proposal_specialist.py', '--remaining-k4'],
        'preflight-specialists-k4': ['evaluate_specialist_envelope.py', '--preflight'],
        'eval-specialists-k4': ['evaluate_specialist_envelope.py'],
    }
    if len(sys.argv) > 1 and sys.argv[1] in specialist_commands:
        run_command([
            sys.executable, '-u', *specialist_commands[sys.argv[1]], *sys.argv[2:],
        ])
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', *specialist_commands))
    parser.add_argument('--case', choices=tuple(CASES), default='k4_n256_d256')
    parser.add_argument('--proposal-root', default='runs_deployzf_v1')
    parser.add_argument('--baseline-root', default='runs_baselines')
    parser.add_argument('--work-root', default='runs_as_screen')
    for model in MODELS:
        parser.add_argument(f'--{model}-checkpoint')
    parser.add_argument('--samples', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260921)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0 or args.samples % args.batch_size:
        parser.error('--samples must be positive and divisible by --batch-size.')

    plan, errors = build_plan(args)
    print_plan(args, plan, errors)
    if errors:
        return 2
    if args.action == 'check':
        print('\nPath inspection only: no weights loaded, training or evaluation started.')
        return 0

    # All numerical work belongs on the GPU server.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Run this command on the GPU server.')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = project_path(args.work_root) / 'evaluation' / f'{args.case}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    k, nsc, budget, _ = CASES[args.case]
    evaluation = [
        sys.executable, '-u', 'baseline_evaluate.py',
        '--scenario', f'{k},{nsc},{budget}',
        '--models', 'proposed,swin,csinet', '--baseline-objective', 'task',
        '--checkpoint', 'best', '--allocations', 'all', '--no-include-proposed-best',
        '--feedback-snr', '0,5,10,15,20,25', '--downlink-snr', '25',
        '--batch-size', str(args.batch_size),
        '--num-batches', str(args.samples // args.batch_size),
        '--seed', str(args.seed), '--output-dir', str(output), '--tag', args.case,
    ]
    for job in plan:
        evaluation.extend([f"--{job['model']}-checkpoint", job['checkpoint']])
    manifest = {
        'created_utc': stamp, 'arguments': vars(args),
        'training_plan': plan, 'evaluation_command': evaluation,
        'purpose': 'Single-training-seed screening; all fixed allocations retained.',
    }
    (output / 'screen_plan.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    reused = [job['model'] for job in plan if job['action'] == 'reuse']
    if reused:
        # Catch incompatible reused weights before starting any long training.
        run_command(evaluation + [
            '--models', ','.join(reused), '--num-batches', '1',
            '--feedback-snr', '25', '--output-dir', str(output / 'preflight'),
            '--tag', 'preflight',
        ])
    for job in plan:
        if job['action'] == 'train_or_resume':
            run_command(job['command'])
        if not Path(job['checkpoint']).is_file():
            raise FileNotFoundError(job['checkpoint'])
    run_command(evaluation)
    write_comparison(output / f'summary_{args.case}.csv', output / 'comparison.csv', args)
    print(f'\nResults: {output}')
    print('Return comparison.csv, summary_*.csv and screen_plan.json for analysis.')
    print('Sample-level results are retained in samples_*.csv.gz.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
