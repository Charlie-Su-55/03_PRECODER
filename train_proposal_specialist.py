#!/usr/bin/env python3
"""Train fixed-allocation K4/Nsc256/Dtot256 specialists with the existing trainer.

Server:
    python train_proposal_specialist.py --remaining-k4 --dry-run
    python -u train_proposal_specialist.py --remaining-k4
    python -u train_proposal_specialist.py --allocation 16,192

Existing checkpoints are never resumed or overwritten by this entry point.
The batch command requires the existing pure-AS best.pth and never trains it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.train_config import ExperimentSpec, resolve_experiment

ROOT = Path(__file__).resolve().parent
K4_ALLOCATIONS = ((64, 0), (48, 64), (32, 128), (16, 192), (0, 256))


def specialist_config(allocation, output_root):
    """Retain the original architecture, recipe, validation rule and naming."""
    cfg = resolve_experiment(
        ExperimentSpec(
            model_name='proposal', train_mode='specialist',
            k_users=4, subcarriers=256, feedback_budget=256,
            recipe_name='k4_n256_standard', seed=42, allocation=allocation,
        ),
        output_root=Path(output_root),
    )
    if allocation not in K4_ALLOCATIONS or cfg.ALLOCATION_GRID != [allocation]:
        raise ValueError(f'Unsupported fixed allocation: {allocation}')
    if (cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED) != (64, 256):
        raise RuntimeError('Specialists must retain the complete max-width architecture.')
    if cfg.TOTAL_STEPS != 50_000 or cfg.BEST_METRIC != 'mean_all':
        raise RuntimeError('Expected the original 50000-step, mean_all recipe.')
    return cfg


def checkpoint_status(checkpoint_path):
    """Read completion status without deserializing checkpoint tensors."""
    path = Path(checkpoint_path)
    possible = [path.parent / 'status.json']
    if len(path.parents) >= 3:
        possible.append(path.parents[2] / 'logs' / path.parent.name / 'status.json')
    for status_path in possible:
        if status_path.is_file():
            return json.loads(status_path.read_text(encoding='utf-8'))
    return {'state': 'unknown'}


def existing_checkpoints(cfg):
    return [path for path in (Path(cfg.BEST_PATH), Path(cfg.SAVE_PATH)) if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root', default='runs_as_screen/proposal_specialists',
        help='Relative paths are resolved from this script directory.',
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        '--allocation', choices=[f'{df},{da}' for df, da in K4_ALLOCATIONS],
        help='One fixed allocation; defaults to 0,256 for the original single-run entry.',
    )
    selection.add_argument('--remaining-k4', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error('--stop-after must be positive.')
    if args.remaining_k4 and args.stop_after is not None:
        parser.error('--remaining-k4 always uses the full 50000-step budget.')

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root = output_root.resolve()
    allocations = (
        K4_ALLOCATIONS if args.remaining_k4
        else (tuple(map(int, (args.allocation or '0,256').split(','))),)
    )
    configs = [specialist_config(allocation, output_root) for allocation in allocations]

    pending = []
    conflicts = []
    for cfg in configs:
        allocation = cfg.ALLOCATION_GRID[0]
        present = existing_checkpoints(cfg)
        status = checkpoint_status(cfg.BEST_PATH)
        print(f'\n[Allocation] {allocation}; seed=42; steps=50000')
        print(f'[Best checkpoint] {cfg.BEST_PATH}')
        if present:
            print(f"[SKIP existing] state={status.get('state', 'unknown')}; preserved:", flush=True)
            for path in present:
                print(f'  {path}')
            if status.get('state') != 'completed':
                print('  Completion is not confirmed; formal evaluation checks this status.')
        elif any(Path(path).exists() for path in (cfg.SAVE_DIR, cfg.LOG_DIR)):
            conflicts.append(f'Existing artifacts without a checkpoint: {cfg.SAVE_DIR} / {cfg.LOG_DIR}')
        elif args.remaining_k4 and allocation == (0, 256):
            conflicts.append(f'Pure-AS checkpoint is required and will NOT be trained: {cfg.BEST_PATH}')
        else:
            print('[NEW] fixed allocation; full max widths=(64,256); mean_all validation')
            pending.append(cfg)

    if args.remaining_k4:
        pure_as = specialist_config((0, 256), output_root)
        if not Path(pure_as.BEST_PATH).is_file():
            conflicts.append(f'Missing protected pure-AS best.pth: {pure_as.BEST_PATH}')
    if conflicts:
        raise RuntimeError('\n'.join(conflicts))
    print(f'\n[Plan] {len(pending)} new run(s); all existing checkpoints skipped.', flush=True)
    if args.dry_run or not pending:
        return 0

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Run training on the GPU server.')
    from main_train import RunOptions, train_proposal

    for cfg in pending:
        # Recheck before each long job, and reserve a new directory atomically.
        if existing_checkpoints(cfg):
            print(f'[SKIP newly existing] {cfg.BEST_PATH}', flush=True)
            continue
        if Path(cfg.LOG_DIR).exists():
            raise FileExistsError(f'Existing logs will not be overwritten: {cfg.LOG_DIR}')
        Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=False)
        train_proposal(
            cfg,
            RunOptions(resume=False, skip_existing=True, stop_after=args.stop_after),
        )
        status = checkpoint_status(cfg.BEST_PATH)
        print(f"[Finished] state={status.get('state')}; {cfg.BEST_PATH}", flush=True)
        if args.stop_after is None and (
            status.get('state') != 'completed' or not Path(cfg.BEST_PATH).is_file()
        ):
            raise RuntimeError(f'Training did not produce a completed run: {cfg.SAVE_DIR}')

    print('\n[Checkpoint status]')
    for cfg in configs:
        state = checkpoint_status(cfg.BEST_PATH).get('state', 'unknown')
        print(f'{cfg.ALLOCATION_GRID[0]}: {state}; best_exists={Path(cfg.BEST_PATH).is_file()}')
        print(f'  {cfg.BEST_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
