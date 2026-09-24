#!/usr/bin/env python3
"""Opt-in pure-AS K4 feedback-SNR-robust fine-tuning; no suite expansion.

Server: --dry-run, then --stop-after 500, then --resume after smoke checks.
All writes stay under runs_as_screen/proposal_snr_robust.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from configs.train_config import ExperimentSpec, resolve_experiment
from train_proposal_specialist import specialist_config

ROOT = Path(__file__).resolve().parent
ROBUST_ROOT = ROOT / 'runs_as_screen/proposal_snr_robust'
RECIPE = 'k4_n256_fb_robust_ft'


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def robust_config(output_root=ROBUST_ROOT):
    cfg = resolve_experiment(
        ExperimentSpec(
            model_name='proposal', train_mode='specialist',
            k_users=4, subcarriers=256, feedback_budget=256,
            recipe_name=RECIPE, seed=42, allocation=(0, 256),
        ),
        output_root=project_path(output_root),
    )
    if cfg.ALLOCATION_GRID != [(0, 256)] or (
        cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED, cfg.NUM_SUBBANDS
    ) != (64, 256, 4):
        raise RuntimeError('Expected the unchanged full-width pure-AS specialist architecture.')
    return cfg


def default_init_checkpoint():
    return Path(specialist_config(
        (0, 256), ROOT / 'runs_as_screen/proposal_specialists',
    ).BEST_PATH)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_run(cfg):
    """Inspect new-run logs/status only; do not add another checkpoint loader."""
    for path in (cfg.BEST_PATH, cfg.SAVE_PATH):
        if not Path(path).is_file():
            raise RuntimeError(f'Missing new checkpoint: {path}')
    status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
    if status.get('state') not in {'paused', 'completed'}:
        raise RuntimeError(f'Unexpected run status: {status}')
    records = [
        json.loads(line) for line in Path(cfg.METRICS_PATH).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    train_records = [row for row in records if row.get('event') == 'train_snr']
    val_records = [row for row in records if row.get('event') == 'validation']
    if not val_records:
        raise RuntimeError('Missing fixed-DL validation log.')
    for row in train_records:
        if not (0.0 <= row['feedback_snr_db'] <= 25.0) or row['downlink_snr_db'] != 25.0:
            raise RuntimeError(f'Training SNR protocol mismatch: {row}')
    for row in val_records:
        if row['feedback_snr_db'] != [0, 5, 10, 15, 20, 25] or row['downlink_snr_db'] != 25.0:
            raise RuntimeError(f'Validation SNR protocol mismatch: {row}')
        expected_keys = {f'alloc0_snr{snr}' for snr in cfg.VAL_SNR_LIST}
        if set(row['table']) != expected_keys or row['selection_metric_name'] != 'mean_all':
            raise RuntimeError('Validation must contain exactly six pure-AS feedback-SNR points.')
        if not math.isclose(
            row['selection_metric'], sum(row['table'].values()) / 6.0, rel_tol=1e-9, abs_tol=1e-9,
        ):
            raise RuntimeError('Checkpoint selection is not the six-point arithmetic mean.')
    if status['step'] >= 2 * cfg.LOG_INTERVAL and len({
        row['feedback_snr_db'] for row in train_records
    }) < 2:
        raise RuntimeError('Expected changing feedback SNR in training logs.')
    print(f"[Checks passed] state={status['state']}; step={status['step']}/{cfg.TOTAL_STEPS}")
    print('[Checks passed] FB range/log variation, fixed DL=25, six-point mean_all validation.')
    print(f'[New best]   {cfg.BEST_PATH}')
    print(f'[New latest] {cfg.SAVE_PATH}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--init-checkpoint', help='Weights-only source; defaults to the original pure-AS best.pth.')
    parser.add_argument('--resume', action='store_true', help='Resume ONLY this new run from its latest.pth.')
    parser.add_argument(
        '--output-root', default=str(ROBUST_ROOT),
        help='Must be the dedicated robust root or a new subdirectory within it.',
    )
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error('--stop-after must be positive.')
    if args.resume and args.init_checkpoint is not None:
        parser.error('--resume and --init-checkpoint are mutually exclusive.')

    output_root = project_path(args.output_root)
    if not output_root.is_relative_to(ROBUST_ROOT.resolve()):
        parser.error('Output must stay under runs_as_screen/proposal_snr_robust.')
    cfg = robust_config(output_root)
    protected_root = (ROOT / 'runs_as_screen/proposal_specialists').resolve()
    for path in (cfg.SAVE_DIR, cfg.LOG_DIR):
        resolved = Path(path).resolve()
        if resolved.is_relative_to(protected_root) or not resolved.is_relative_to(ROBUST_ROOT.resolve()):
            parser.error(f'Unsafe output path: {resolved}')
    snapshot = json.loads(json.dumps(cfg.to_dict()))
    manifest_path = Path(cfg.SAVE_DIR) / 'robust_run_manifest.json'
    manifest = None
    if args.resume:
        if not manifest_path.is_file() or not Path(cfg.SAVE_PATH).is_file():
            parser.error('--resume requires this robust run manifest and latest.pth.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['config'] != snapshot:
            parser.error('Resolved robust configuration differs from the paused run.')
        init_path = project_path(manifest['init_checkpoint'])
    else:
        init_path = project_path(args.init_checkpoint or default_init_checkpoint())

    print(f'[Experiment] {cfg.EXP_NAME}')
    print(f'[Mode] {"resume new robust latest.pth" if args.resume else "weights-only initialization, strict=True"}')
    print(f'[Protected initialization checkpoint] {init_path}')
    print(f'[Source exists] {init_path.is_file()}')
    print(f'[Best]   {cfg.BEST_PATH}')
    print(f'[Latest] {cfg.SAVE_PATH}')
    print('[Training] FB_SNR ~ Uniform(0,25); DL_SNR=25 throughout.')
    print('[Validation] FB SNRs=[0,5,10,15,20,25]; fixed DL SNR=25; mean_all.')
    print('[Resolved configuration]')
    print(json.dumps(snapshot, indent=2))
    if not init_path.is_file():
        raise FileNotFoundError(init_path)
    if not args.resume and any(Path(path).exists() for path in (cfg.SAVE_DIR, cfg.LOG_DIR)):
        raise FileExistsError('New run already has artifacts. Use --resume, or a NEW --output-root subdirectory.')
    if args.dry_run:
        print('[Dry run] No torch import, weight loading, experiment directories, or training.')
        return 0

    source_hash = sha256_file(init_path)
    if manifest is not None and source_hash != manifest['init_sha256']:
        raise RuntimeError('Original initialization checkpoint changed since smoke training.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Run this on the GPU server.')
    from main_train import RunOptions, train_proposal

    if not args.resume:
        Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=False)
        Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=False)
        manifest = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'init_checkpoint': str(init_path), 'init_sha256': source_hash,
            'initialization': 'weights-only, strict=True, fresh AdamW/OneCycleLR',
            'config': snapshot,
        }
        with manifest_path.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)

    try:
        train_proposal(cfg, RunOptions(
            resume=args.resume, skip_existing=True, stop_after=args.stop_after,
            init_checkpoint=None if args.resume else str(init_path),
        ))
    finally:
        if sha256_file(init_path) != source_hash:
            raise RuntimeError('Protected initialization checkpoint changed during this run.')
        print(f'[Protected source] SHA-256 unchanged: {source_hash}', flush=True)
    verify_run(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
