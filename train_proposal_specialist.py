#!/usr/bin/env python3
"""Train the K=4, Nsc=256, Dtot=256 pure-AS specialist with the existing trainer.

Server:
    python -u train_proposal_specialist.py
    python train_proposal_specialist.py --dry-run

Fresh runs start from seed 42; reruns resume only this specialist's latest.pth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from configs.train_config import ExperimentSpec, resolve_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root', default='runs_as_screen/proposal_specialists',
        help='Relative paths are resolved from this script directory.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error('--stop-after must be positive.')

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root
    cfg = resolve_experiment(
        ExperimentSpec(
            model_name='proposal',
            train_mode='specialist',
            k_users=4,
            subcarriers=256,
            feedback_budget=256,
            recipe_name='k4_n256_standard',
            seed=42,
            allocation=(0, 256),
        ),
        output_root=output_root.resolve(),
    )
    # Keep the adaptive model's maximum-width layers, including unused FDMA layers.
    # The single allocation has index 0 and its own running-power statistic.
    if cfg.ALLOCATION_GRID != [(0, 256)] or (
        cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED
    ) != (64, 256):
        raise RuntimeError('Unexpected specialist allocation or model capacity.')
    if cfg.TOTAL_STEPS != 50_000 or cfg.BEST_METRIC != 'mean_all':
        raise RuntimeError('Expected the original 50000-step, mean_all recipe.')

    print(f'[Experiment] {cfg.EXP_NAME}', flush=True)
    print(f'[Allocation] {cfg.ALLOCATION_GRID}; seed={cfg.SEED}')
    print(f'[Recipe] {cfg.recipe.name}; steps={cfg.TOTAL_STEPS}; batch={cfg.BATCH_SIZE}')
    print(f'[Selection] mean validation sum rate over SNR={cfg.VAL_SNR_LIST}')
    print(f'[Best checkpoint] {cfg.BEST_PATH}')
    print(f'[Resume checkpoint] {cfg.SAVE_PATH}', flush=True)
    if args.dry_run:
        return 0

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Run training on the GPU server.')

    from main_train import RunOptions, train_proposal
    train_proposal(
        cfg,
        RunOptions(resume=True, skip_existing=True, stop_after=args.stop_after),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

