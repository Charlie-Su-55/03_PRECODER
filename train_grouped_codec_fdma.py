#!/usr/bin/env python3
"""Train only the independent F Legacy+FDMA arm using the existing task trainer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback
import uuid

from configs.grouped_codec_fdma_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, REPO_ROOT, fdma_config, validate_config,
    channel_data_api, make_channel_data, protocol,
)
from train_grouped_codec import (
    GroupedExecutionProtocol, apply_backend, fingerprint, _json, run_lock,
    _output_paths, _safe_path, _close, expected_cycle, sha256_file, _backend_info,
    _validate_identity, _validation_values, validate_model_identity,
    validate_model_state, make_model_factory, SOURCE_FILES as COMMON_SOURCE_FILES,
)

SOURCE_FILES = ('train_grouped_codec_fdma.py', 'configs/grouped_codec_fdma_train_config.py',
                'models/grouped_codec_fdma.py') + COMMON_SOURCE_FILES


class FDMAExecutionProtocol(GroupedExecutionProtocol):
    """Reuse the exact indexed gamma/noise and validation implementation, with F identity."""
    def __init__(self, cfg):
        validate_config(cfg)
        self.cfg = cfg
        self.config_snapshot = cfg.to_dict()


def validate_output_paths(cfg, *, allowed_root=OUTPUT_ROOT):
    return _output_paths(cfg, allowed_root=allowed_root)


def check_run_state(cfg, *, resume, allowed_root=OUTPUT_ROOT):
    validate_config(cfg)
    validate_output_paths(cfg, allowed_root=allowed_root)
    if not resume:
        for path in (cfg.SAVE_DIR, cfg.LOG_DIR):
            if Path(path).exists():
                raise FileExistsError(f'Refusing to overwrite/reinitialize {path}')
        return
    for key in ('SAVE_PATH', 'BEST_PATH', 'CONFIG_PATH', 'STATUS_PATH', 'METRICS_PATH'):
        if not Path(getattr(cfg, key)).is_file():
            raise FileNotFoundError(f'F resume requires {key}')
    snapshot = json.loads(Path(cfg.CONFIG_PATH).read_text(encoding='utf-8'))
    if _json(snapshot) != _json(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg))):
        raise ValueError('F resume full configuration/layout/protocol/fingerprint mismatch')
    status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
    if status.get('experiment') != cfg.EXP_NAME or status.get('state') not in ('paused', 'running'):
        raise ValueError('F resume requires this unfinished arm and seed')
    if status.get('fingerprint', fingerprint(cfg)) != fingerprint(cfg):
        raise ValueError('F resume status fingerprint mismatch')


def validate_resume_payload(payload, cfg, *, allow_completed=False):
    """F-specific identity plus the same single-group AdamW/20k scheduling contract."""
    validate_config(cfg)
    channel_data_api().validate_resume_config(payload, cfg)
    _validate_identity(payload, cfg)
    step = payload['step']
    upper = cfg.TOTAL_STEPS if allow_completed else cfg.TOTAL_STEPS - 1
    if type(step) is not int or not 0 < step <= upper or step % cfg.VAL_INTERVAL:
        raise ValueError('F latest must be a positive 500-update boundary in the fixed 20k plan')
    if _validation_values(payload, cfg) > payload['best_metric'] + 1e-10:
        raise ValueError('Latest validation exceeds retained best metric')
    phases = [dict(end_step=cfg.WARMUP_PCT * cfg.TOTAL_STEPS - 1,
                   start_lr='initial_lr', end_lr='max_lr',
                   start_momentum='max_momentum', end_momentum='base_momentum'),
              dict(end_step=cfg.TOTAL_STEPS - 1, start_lr='max_lr', end_lr='min_lr',
                   start_momentum='base_momentum', end_momentum='max_momentum')]
    expected = dict(total_steps=cfg.TOTAL_STEPS, last_epoch=step, _step_count=step + 1,
                    _schedule_phases=phases, _anneal_func_type='cos', cycle_momentum=True,
                    use_beta1=True, base_lrs=[cfg.LR / 25])
    scheduler = payload['scheduler']
    for key, value in expected.items():
        if scheduler.get(key) != value:
            raise ValueError(f'F scheduler {key} mismatch')
    lr, beta1 = expected_cycle(cfg, step)
    if not isinstance(scheduler.get('_last_lr'), list) or len(scheduler['_last_lr']) != 1:
        raise ValueError('One shared UE/BS LR required')
    _close(scheduler['_last_lr'][0], lr, 'scheduler LR')
    optimizer = payload['optimizer']
    if (not isinstance(optimizer.get('state'), dict) or not optimizer['state']
            or len(optimizer.get('param_groups', [])) != 1):
        raise ValueError('Complete AdamW history and one shared UE/BS group required')
    group = optimizer['param_groups'][0]
    for key, value in dict(lr=lr, initial_lr=cfg.LR / 25, max_lr=cfg.LR,
                          min_lr=cfg.LR / 25 / 10000, weight_decay=cfg.WEIGHT_DECAY,
                          eps=1e-8, base_momentum=.85, max_momentum=.95).items():
        _close(group.get(key), value, f'optimizer {key}')
    if len(group.get('betas', [])) != 2:
        raise ValueError('AdamW betas missing')
    _close(group['betas'][0], beta1, 'AdamW beta1')
    _close(group['betas'][1], .999, 'AdamW beta2')
    for key in ('amsgrad', 'maximize', 'capturable', 'differentiable'):
        if group.get(key) is not False:
            raise ValueError(f'Unexpected AdamW {key}')
    if group.get('foreach') is not None or group.get('fused') is not None:
        raise ValueError('No foreach/fused override')
    rng = payload['rng_state']
    if any(rng.get(key) is None for key in ('python', 'numpy', 'torch', 'cuda')):
        raise ValueError('Complete default RNG states required')
    if not isinstance(rng['cuda'], (tuple, list)) or not rng['cuda']:
        raise ValueError('CUDA RNG missing')
    return payload


def validate_best_metadata(best, latest, cfg):
    validate_config(cfg)
    _validate_identity(best, cfg)
    step = best.get('step')
    if type(step) is not int or not 0 < step <= latest['step'] or step % cfg.VAL_INTERVAL:
        raise ValueError('Random-init F best must be a positive validation boundary, never step0')
    _close(best.get('best_metric'), latest['best_metric'], 'best/latest selected metric')
    _close(_validation_values(best, cfg), best['best_metric'], 'best six-gamma mean_all')


def validate_runtime_resume(torch, model, payload, cfg, *, allow_completed=False):
    """Check actual state/moments/RNG without modifying global RNG or constructing an optimizer."""
    import numpy as np
    validate_resume_payload(payload, cfg, allow_completed=allow_completed)
    validate_model_identity(cfg, model)
    validate_model_state(torch, model, payload['state'])
    model.load_state_dict(payload['state'], strict=True)
    parameters = list(model.named_parameters())
    group = payload['optimizer']['param_groups'][0]
    ids, moments = group.get('params'), payload['optimizer']['state']
    if ids != list(range(len(parameters))) or not set(moments) <= set(ids):
        raise ValueError('AdamW parameter identity/order/count mismatch')
    inactive = set(model.inactive_parameter_names())
    if not inactive <= set(dict(parameters)):
        raise ValueError('Invalid inactive parameter declaration')
    for index, (name, parameter) in enumerate(parameters):
        if name in inactive:
            if index in moments:
                raise ValueError(f'Unexpected inactive original-FDMA optimizer state: {name}')
            continue
        saved = moments.get(index)
        if not isinstance(saved, dict) or set(saved) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError(f'Missing active AdamW history: {name}')
        counter = saved['step']
        if (not torch.is_tensor(counter) or counter.numel() != 1
                or not torch.isfinite(counter).all() or counter.item() != payload['step']):
            raise ValueError(f'AdamW update count mismatch: {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = saved[key]
            if (not torch.is_tensor(value) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                raise ValueError(f'AdamW {key} mismatch: {name}')
        if not (saved['exp_avg_sq'] >= 0).all():
            raise ValueError(f'Negative second moment: {name}')
    rng = payload['rng_state']
    if len(rng['cuda']) != torch.cuda.device_count():
        raise ValueError('Visible CUDA count differs from resume')
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    for device, value in [('cpu', rng['torch'])] + [(f'cuda:{i}', s) for i, s in enumerate(rng['cuda'])]:
        if (not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1
                or value.device.type != 'cpu'):
            raise ValueError(f'Malformed RNG tensor for {device}')
        torch.Generator(device=device).set_state(value)


def optimizer_scheduler_factory(cfg, model):
    import torch
    validate_config(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.LR,
        total_steps=cfg.TOTAL_STEPS, pct_start=cfg.WARMUP_PCT, anneal_strategy='cos')
    return optimizer, scheduler


def completed_resume_status(cfg, latest, best, previous_status):
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    if latest['step'] != cfg.TOTAL_STEPS:
        raise ValueError('Completion recovery requires all 20k planned updates')
    if (previous_status.get('experiment') != cfg.EXP_NAME
            or previous_status.get('state') not in ('running', 'paused')):
        raise ValueError('Completion recovery requires this unfinished F run')
    if previous_status.get('state') == 'paused' and previous_status.get('step') != latest['step']:
        raise ValueError('Paused/latest completion step mismatch')
    return dict(state='completed', experiment=cfg.EXP_NAME, step=cfg.TOTAL_STEPS,
        best_metric_name=cfg.BEST_METRIC, best_metric=best['best_metric'],
        best_path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
        completion_recovered=True, additional_optimizer_updates=0)


def _record(cfg, name, data):
    path = Path(cfg.LOG_DIR) / name
    _safe_path(path, OUTPUT_ROOT)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)


def run_training(cfg, args):
    """Independent run orchestration only; no copied loss/forward/update loop."""
    check_run_state(cfg, resume=args.resume)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('F training requires CUDA; no laptop CPU fallback')
    from main_train import RunOptions, set_seed, train_proposal
    from models.grouped_codec_fdma import GroupedCodecFDMAPrecoder
    data = make_channel_data(cfg)
    execution = FDMAExecutionProtocol(cfg)
    with run_lock(cfg, resume=args.resume, allowed_root=OUTPUT_ROOT):
        check_run_state(cfg, resume=args.resume)
        set_seed(cfg.SEED)
        model = GroupedCodecFDMAPrecoder(cfg)
        actual = validate_model_identity(cfg, model)
        record = dict(arm=cfg.arm, seed=cfg.SEED, config=cfg.to_dict(),
            config_fingerprint=fingerprint(cfg), mode='resume' if args.resume else 'random_initialization',
            historical_weights_loaded=False, actual_model_metadata=actual,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
            source_sha256={name: sha256_file(REPO_ROOT / name) for name in SOURCE_FILES},
            backend_before=_backend_info(torch), python=sys.version, platform=platform.platform(),
            scheduler_total_steps=cfg.TOTAL_STEPS, screening_only=True)
        if args.resume:
            payload = torch.load(cfg.SAVE_PATH, map_location='cpu', weights_only=False)
            finished = payload.get('step') == cfg.TOTAL_STEPS
            validate_runtime_resume(torch, model, payload, cfg, allow_completed=finished)
            best = torch.load(cfg.BEST_PATH, map_location='cpu', weights_only=False)
            validate_best_metadata(best, payload, cfg)
            validate_model_state(torch, model, best['state'])
            status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
            if status['state'] == 'paused' and status.get('step') != payload['step']:
                raise ValueError('Paused/latest update count mismatch')
            start = payload['step'] + 1
            record['resume_sha256'] = sha256_file(cfg.SAVE_PATH)
            if finished:
                if best['step'] == payload['step']:
                    for name, value in payload['state'].items():
                        if not torch.equal(value, best['state'][name]):
                            raise ValueError(f'Same-step best/latest state mismatch: {name}')
                completed = completed_resume_status(cfg, payload, best, status)
                record.update(mode='finalize_interrupted_completion', start_step=start,
                    planned_end_step=cfg.TOTAL_STEPS, additional_optimizer_updates=0)
                token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8]
                _record(cfg, f'invocation_{token}.json', record)
                result = dict(completed_updates=payload['step'], best_step=best['step'],
                    best_metric=best['best_metric'], latest_sha256=record['resume_sha256'],
                    best_sha256=sha256_file(cfg.BEST_PATH), completion_recovered=True,
                    additional_optimizer_updates=0, screening_only=True, backend_after=_backend_info(torch))
                _record(cfg, f'completion_{token}.json', result)
                from as_receiver_batch_support import save_json
                save_json(cfg.STATUS_PATH, completed, replace=True)
                print(json.dumps(result, indent=2))
                return
            factory = make_model_factory(cfg, model, resume_state=payload['state'])
            del best, payload
        else:
            start = 1
            factory = make_model_factory(cfg, model)
            Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=False)
        record.update(start_step=start,
            planned_end_step=min(cfg.TOTAL_STEPS, start + args.stop_after - 1)
                if args.stop_after is not None else cfg.TOTAL_STEPS,
            random_address_resume='train index equals next global_step; fixed validation index0',
            full_training_bitwise_resume_verified=False)
        token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8]
        _record(cfg, f'invocation_{token}.json', record)
        try:
            train_proposal(cfg, RunOptions(resume=args.resume, skip_existing=False,
                stop_after=args.stop_after, init_checkpoint=None), model_factory=factory,
                channel_data=data, optimizer_scheduler_factory=optimizer_scheduler_factory,
                execution_protocol=execution)
            latest = torch.load(cfg.SAVE_PATH, map_location='cpu', weights_only=False)
            validate_resume_payload(latest, cfg, allow_completed=True)
            if latest['step'] != record['planned_end_step']:
                raise RuntimeError('Unexpected F completed update count')
            best = torch.load(cfg.BEST_PATH, map_location='cpu', weights_only=False)
            validate_best_metadata(best, latest, cfg)
            result = dict(completed_updates=latest['step'], best_step=best['step'],
                best_metric=best['best_metric'], latest_sha256=sha256_file(cfg.SAVE_PATH),
                best_sha256=sha256_file(cfg.BEST_PATH), backend_after=_backend_info(torch), screening_only=True)
            _record(cfg, f'completion_{token}.json', result)
            print(json.dumps(result, indent=2))
        except BaseException as error:
            _record(cfg, f'failure_{token}.json', dict(error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(), automatic_fallback=False))
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    parser.add_argument('--seed', type=int, choices=SEEDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.dry_run or args.train or args.stop_after is not None):
        if args.resume:
            parser.error('Choose an explicit run mode')
        parser.print_help()
        return 0
    if args.seed is None:
        parser.error('Choose a seed')
    if args.resume and args.dry_run:
        parser.error('Dry-run does not resume checkpoints')
    if args.stop_after is not None and (not 0 < args.stop_after <= 20000 or args.stop_after % 500):
        parser.error('stop-after must be a positive multiple of 500 within the fixed 20k plan')
    cfg = fdma_config(args.seed)
    validate_config(cfg)
    validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps(dict(action='plan only; no weights/model imports/output writes',
            config=cfg.to_dict(), fingerprint=fingerprint(cfg), latest=cfg.SAVE_PATH,
            best=cfg.BEST_PATH, scheduler_total_steps=20000, historical_checkpoint=None), indent=2))
        return 0
    run_training(cfg, args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
