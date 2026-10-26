#!/usr/bin/env python3
"""Explicit AS-spatial 20k screening; shared trainer, opt-in LR groups only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import SimpleNamespace
import uuid

from configs.as_spatial_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, REPO_ROOT, as_spatial_config, validate_config,
    optimizer_group_spec, make_channel_data, channel_data_api,
)
from preflight_proposal_as_direct import (
    DEFAULT_MANIFEST, _backend_info, _sha256_stream, sha256_file,
    validate_checkpoint_metadata, verify_checkpoint_hash,
)
from train_proposal_as_direct import _safe_path, _close, expected_cycle, validate_model_state, optimizer_state_required
from train_proposal_as_receiver import (
    fingerprint, _json, validate_output_paths as _validate_output_paths,
    validate_best_metadata, run_lock,
)

UE_PREFIXES = ("encoder.", "cond_embed.")
SOURCE_FILES = (
    "train_proposal_as_spatial.py", "configs/as_spatial_train_config.py",
    "models/adaptive_hybrid_as_spatial.py", "models/adaptive_hybrid.py",
    "models/adaptive_hybrid_as_receiver.py", "models/swin_precoder.py",
    "models/baseline_common.py", "main_train.py", "configs/train_config.py",
    "utils/replayable_channel_data.py", "utils/sionna_rng.py", "utils/channel_generator.py",
    "train_proposal_as_receiver.py", "train_proposal_as_direct.py",
    "configs/as_receiver_train_config.py", "preflight_proposal_as_direct.py",
)


def validate_output_paths(cfg, *, allowed_root=OUTPUT_ROOT):
    _validate_output_paths(cfg, allowed_root=allowed_root)


def check_run_state(cfg, *, resume, allowed_root=OUTPUT_ROOT):
    validate_config(cfg)
    validate_output_paths(cfg, allowed_root=allowed_root)
    if not resume:
        for path in (cfg.SAVE_DIR, cfg.LOG_DIR):
            if Path(path).exists():
                raise FileExistsError(f"Refusing to overwrite/reinitialize {path}")
        return
    for field in ("SAVE_PATH", "BEST_PATH", "CONFIG_PATH", "STATUS_PATH", "METRICS_PATH"):
        if not Path(getattr(cfg, field)).is_file():
            raise FileNotFoundError(f"Resume requires {field}: {getattr(cfg, field)}")
    if _json(json.loads(Path(cfg.CONFIG_PATH).read_text(encoding="utf-8"))) != _json(
            dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg))):
        raise ValueError("Resume full config/source/data/group fingerprint mismatch")
    status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding="utf-8"))
    if status.get("experiment") != cfg.EXP_NAME or status.get("state") not in ("paused", "running"):
        raise ValueError("Resume status must identify this unfinished arm/seed")
    if status.get("fingerprint", fingerprint(cfg)) != fingerprint(cfg):
        raise ValueError("Resume status fingerprint mismatch")


def cycle_values(cfg, peak_lr, step):
    return expected_cycle(SimpleNamespace(LR=peak_lr, TOTAL_STEPS=cfg.TOTAL_STEPS,
                                         WARMUP_PCT=cfg.WARMUP_PCT), step)


def parameter_groups(cfg, model):
    """Exact named ownership, separate true LR groups; no parameter duplication."""
    named = dict(model.named_parameters())
    by_id = {id(parameter): name for name, parameter in named.items()}
    ue = list(model.ue_parameters())
    bs = list(model.bs_parameters())
    ue_ids, bs_ids = {id(p) for p in ue}, {id(p) for p in bs}
    if len(ue_ids) != len(ue) or len(bs_ids) != len(bs) or ue_ids & bs_ids or ue_ids | bs_ids != set(by_id):
        raise ValueError("UE/BS parameter groups must be complete, disjoint and unique")
    if any(not by_id[id(p)].startswith(UE_PREFIXES) for p in ue) or any(by_id[id(p)].startswith(UE_PREFIXES) for p in bs):
        raise ValueError("Parameter ownership differs from encoder+condition versus new BS")
    frozen = cfg.arm == "frozen_ue"
    if model.ue_frozen is not frozen or any(p.requires_grad is frozen for p in ue) or any(not p.requires_grad for p in bs):
        raise ValueError("UE/BS trainability differs from the declared arm")
    groups = []
    for spec in optimizer_group_spec(cfg):
        params = bs if spec["name"] == "bs" else ue
        groups.append(dict(params=params, group_name=spec["name"],
                           parameter_names=[by_id[id(p)] for p in params], lr=spec["peak_lr"]))
    return groups


def validate_model_identity(cfg, model):
    actual = model.architecture_metadata
    declared = cfg.architecture_metadata
    for key in ('architecture', 'model_class', 'arm', 'ue_ffn_dim', 'D_MODEL',
                'max_dimensions', 'allocation_grid', 'running_pwr_shape', 'cfg_idx',
                'ue_frozen', 'decoder', 'receiver', 'frozen_ue_power', 'joint_ue_power',
                'rzf_regularization', 'power_normalization'):
        if _json(actual.get(key)) != _json(declared[key]):
            raise ValueError(f'Actual AS-spatial model metadata differs for {key}')
    if model.pwr_momentum != declared['running_power_ema_momentum']:
        raise ValueError('Actual UE running-power momentum differs from fixed protocol')
    return actual


def optimizer_scheduler_factory(cfg, model):
    import torch
    groups = parameter_groups(cfg, model)
    optimizer = torch.optim.AdamW(groups, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[spec["peak_lr"] for spec in optimizer_group_spec(cfg)],
        total_steps=cfg.TOTAL_STEPS, pct_start=cfg.WARMUP_PCT, anneal_strategy="cos",
    )
    return optimizer, scheduler


def validate_resume_payload(payload, cfg, *, allow_completed=False):
    """Validate all groups and the full 20k schedule, not only one shared LR."""
    channel_data_api().validate_resume_config(payload, cfg)
    if _json(payload.get("allocation_grid")) != _json(cfg.ALLOCATION_GRID):
        raise ValueError("Resume allocation must remain singleton pure AS")
    step = payload["step"]
    upper = cfg.TOTAL_STEPS if allow_completed else cfg.TOTAL_STEPS - 1
    if type(step) is not int or not 0 < step <= upper or step % cfg.VAL_INTERVAL:
        raise ValueError("Resume step must be a 500-update boundary within 20k")
    if "running_pwr" not in payload["state"]:
        raise ValueError("Missing singleton UE running power")
    specs = optimizer_group_spec(cfg)
    scheduler = payload["scheduler"]
    phases = [dict(end_step=cfg.WARMUP_PCT * cfg.TOTAL_STEPS - 1,
                   start_lr="initial_lr", end_lr="max_lr", start_momentum="max_momentum", end_momentum="base_momentum"),
              dict(end_step=cfg.TOTAL_STEPS - 1, start_lr="max_lr", end_lr="min_lr",
                   start_momentum="base_momentum", end_momentum="max_momentum")]
    for key, expected in {
        "total_steps": cfg.TOTAL_STEPS, "last_epoch": step, "_step_count": step + 1,
        "_schedule_phases": phases, "_anneal_func_type": "cos", "cycle_momentum": True,
        "use_beta1": True, "base_lrs": [spec["peak_lr"] / 25 for spec in specs],
    }.items():
        if scheduler.get(key) != expected:
            raise ValueError(f"Resume scheduler {key} mismatch")
    last_lr = scheduler.get("_last_lr")
    optimizer = payload["optimizer"]
    groups = optimizer.get("param_groups", [])
    if not optimizer.get("state") or len(groups) != len(specs) or not isinstance(last_lr, list) or len(last_lr) != len(specs):
        raise ValueError("Missing optimizer history or incorrect LR group count")
    all_names, all_ids = [], []
    for index, (group, spec) in enumerate(zip(groups, specs)):
        peak = spec["peak_lr"]
        lr, beta1 = cycle_values(cfg, peak, step)
        if group.get("group_name") != spec["name"]:
            raise ValueError("Optimizer group ordering/name mismatch")
        names, ids = group.get("parameter_names"), group.get("params")
        if not isinstance(names, list) or not names or not isinstance(ids, list) or len(names) != len(ids):
            raise ValueError("Explicit optimizer parameter names/IDs required")
        if any(not isinstance(name, str) or name not in payload["state"] or
               name.startswith(UE_PREFIXES) != (spec["name"] == "ue") for name in names):
            raise ValueError("Optimizer group contains an invalid UE/BS parameter")
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Invalid optimizer parameter ID")
        all_names.extend(names)
        all_ids.extend(ids)
        for key, expected in {"lr": lr, "initial_lr": peak/25, "max_lr": peak,
                              "min_lr": peak/25/10000, "weight_decay": cfg.WEIGHT_DECAY,
                              "eps": 1e-8, "base_momentum": 0.85, "max_momentum": 0.95}.items():
            _close(group.get(key), expected, f"group {spec['name']} {key}")
        _close(last_lr[index], lr, "scheduler group LR")
        if len(group.get("betas", [])) != 2:
            raise ValueError("AdamW betas missing")
        _close(group["betas"][0], beta1, "AdamW beta1")
        _close(group["betas"][1], 0.999, "AdamW beta2")
        if any(group.get(key) is not False for key in ("amsgrad", "maximize", "capturable", "differentiable")):
            raise ValueError("Unexpected AdamW mode")
        if group.get("foreach") is not None or group.get("fused") is not None:
            raise ValueError("Unexpected foreach/fused override")
    if len(set(all_names)) != len(all_names) or all_ids != list(range(len(all_ids))):
        raise ValueError("Duplicated or reordered optimizer parameter identities")
    if not isinstance(optimizer["state"], dict) or not set(optimizer["state"]) <= set(all_ids):
        raise ValueError("Optimizer history refers to unknown parameter IDs")
    rng = payload["rng_state"]
    if any(rng.get(key) is None for key in ("python", "numpy", "torch", "cuda")) or not isinstance(rng["cuda"], (list, tuple)) or not rng["cuda"]:
        raise ValueError("Complete default Python/NumPy/CPU/all-CUDA RNG required")


def validate_runtime_resume(torch, model, payload, cfg):
    import numpy as np
    validate_resume_payload(payload, cfg)
    validate_model_state(torch, model, payload["state"])
    model.load_state_dict(payload["state"], strict=True)
    expected = parameter_groups(cfg, model)
    saved_groups = payload["optimizer"]["param_groups"]
    moments = payload["optimizer"]["state"]
    for current, saved in zip(expected, saved_groups):
        if current["parameter_names"] != saved["parameter_names"]:
            raise ValueError("Actual parameter grouping/order differs from resume state")
        for name, parameter, index in zip(current["parameter_names"], current["params"], saved["params"]):
            required = optimizer_state_required(name)
            if index not in moments:
                if required:
                    raise ValueError(f"Missing active optimizer moments: {name}")
                continue
            if not required:
                raise ValueError(f"Unexpected moments for bypassed UE branch: {name}")
            saved_state = moments[index]
            if set(saved_state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"Incomplete AdamW history: {name}")
            counter = saved_state["step"]
            if not torch.is_tensor(counter) or counter.numel() != 1 or not torch.isfinite(counter).all() or counter.item() != payload["step"]:
                raise ValueError(f"Optimizer counter mismatch: {name}")
            for key in ("exp_avg", "exp_avg_sq"):
                value = saved_state[key]
                if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                    raise ValueError(f"Optimizer moment shape/dtype/finiteness mismatch: {name}")
            if not (saved_state["exp_avg_sq"] >= 0).all():
                raise ValueError("Negative AdamW second moment")
    rng = payload["rng_state"]
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume requires saved visible CUDA RNG count")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    for device, value in [("cpu", rng["torch"])] + [(f"cuda:{i}", state) for i, state in enumerate(rng["cuda"])]:
        if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
            raise ValueError(f"Malformed RNG tensor for {device}")
        torch.Generator(device=device).set_state(value)


def ue_state_digest(torch, state):
    import hashlib
    digest = hashlib.sha256()
    keys = sorted(name for name in state if name.startswith(UE_PREFIXES) or name == "running_pwr")
    if not keys or "running_pwr" not in keys:
        raise ValueError("UE state identity requires encoder, condition and running power")
    for name in keys:
        value = state[name].detach().cpu().contiguous()
        digest.update(_json([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_model_factory(cfg, model, *, source_state=None, resume_state=None):
    if (source_state is None) == (resume_state is None):
        raise ValueError("Choose exactly one fresh UE source or strict resume state")
    used = False
    def factory(actual_cfg, device):
        nonlocal used, source_state, resume_state
        if used or fingerprint(cfg) != fingerprint(actual_cfg):
            raise RuntimeError("Factory reused or configuration changed")
        used = True
        fresh = source_state is not None
        if fresh:
            model.initialize_ue_from_full(source_state)
        else:
            model.load_state_dict(resume_state, strict=True)
        model.set_ue_frozen(cfg.arm == "frozen_ue")
        parameter_groups(cfg, model)
        model.to(device)
        source_state = resume_state = None
        if fresh:
            from main_train import set_seed
            set_seed(cfg.SEED)
        return model
    return factory


def write_record(cfg, filename, record):
    path = Path(cfg.LOG_DIR) / filename
    _safe_path(path, OUTPUT_ROOT)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_training(cfg, args):
    check_run_state(cfg, resume=args.resume)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("AS-spatial screening requires CUDA; no CPU fallback")
    from main_train import RunOptions, set_seed, train_proposal
    from models.adaptive_hybrid_as_spatial import AdaptiveHybridPrecoderASSpatial
    data = make_channel_data(cfg)
    channel_data_api().validate_channel_data(cfg, data)
    with run_lock(cfg, resume=args.resume, allowed_root=OUTPUT_ROOT):
        check_run_state(cfg, resume=args.resume)
        set_seed(cfg.SEED)  # Identical new-BS initialization between same-seed arms.
        model = AdaptiveHybridPrecoderASSpatial(cfg)
        validate_model_identity(cfg, model)
        parameter_groups(cfg, model)
        record = dict(utc=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            source_code_sha256={name: sha256_file(REPO_ROOT/name) for name in SOURCE_FILES},
            manifest_sha256=sha256_file(DEFAULT_MANIFEST),
            config=cfg.to_dict(), config_fingerprint=fingerprint(cfg),
            actual_model_metadata=model.architecture_metadata,
            mode="resume_latest" if args.resume else "fresh_source_UE_random_BS",
            optimizer_groups=optimizer_group_spec(cfg), scheduler_total_steps=cfg.TOTAL_STEPS,
            stop_after=args.stop_after, python=sys.version, platform=platform.platform(),
            backend_before_trainer=_backend_info(torch), screening_only=True,
            full_training_bitwise_resume_verified=False)
        if args.resume:
            with Path(cfg.SAVE_PATH).open("rb") as stream:
                record["resume_checkpoint_sha256"] = _sha256_stream(stream)
                stream.seek(0)
                payload = torch.load(stream, map_location="cpu", weights_only=False)
            validate_runtime_resume(torch, model, payload, cfg)
            best = torch.load(cfg.BEST_PATH, map_location="cpu", weights_only=False)
            validate_best_metadata(best, payload, cfg)
            validate_model_state(torch, model, best["state"])
            del best
            status = json.loads(Path(cfg.STATUS_PATH).read_text())
            if status["state"] == "paused" and status.get("step") != payload["step"]:
                raise ValueError("Paused status/latest step mismatch")
            start = payload["step"] + 1
            factory = make_model_factory(cfg, model, resume_state=payload["state"])
        else:
            path = cfg.source_identity["path"] if args.checkpoint is None else str((REPO_ROOT / args.checkpoint).resolve())
            with Path(path).open("rb") as stream:
                verify_checkpoint_hash(path, cfg.source_identity["checkpoint_sha256"], stream=stream)
                stream.seek(0)
                payload = torch.load(stream, map_location="cpu", weights_only=False)
            validate_checkpoint_metadata(payload, cfg.source_identity)
            record.update(source_path=path, source_sha256=cfg.source_identity["checkpoint_sha256"],
                          source_step=payload["step"], source_bs_weights_loaded=False)
            start = 1
            factory = make_model_factory(cfg, model, source_state=payload["state"])
        record["ue_state_sha256_before"] = ue_state_digest(torch, payload["state"])
        record["start_step"] = start
        record["planned_end_step"] = min(cfg.TOTAL_STEPS, start+args.stop_after-1) if args.stop_after is not None else cfg.TOTAL_STEPS
        del payload
        if not args.resume:
            Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=False)
        invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
        write_record(cfg, f"invocation_{invocation}.json", record)
        try:
            train_proposal(cfg, RunOptions(resume=args.resume, skip_existing=False,
                stop_after=args.stop_after, init_checkpoint=None), model_factory=factory,
                channel_data=data, optimizer_scheduler_factory=optimizer_scheduler_factory)
            latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
            validate_resume_payload(latest, cfg, allow_completed=True)
            if latest["step"] != record["planned_end_step"]:
                raise RuntimeError("Trainer did not stop at the requested fixed-budget update")
            ue_after = ue_state_digest(torch, latest["state"])
            if cfg.arm == "frozen_ue" and ue_after != record["ue_state_sha256_before"]:
                raise RuntimeError("Frozen UE parameters/condition/running power changed")
            result = dict(normal_return=True, arm=cfg.arm, seed=cfg.SEED,
                completed_updates=latest["step"], total_updates=cfg.TOTAL_STEPS,
                latest_sha256=sha256_file(cfg.SAVE_PATH), config_fingerprint=fingerprint(cfg),
                ue_state_sha256_before=record["ue_state_sha256_before"], ue_state_sha256_after=ue_after,
                frozen_ue_unchanged=ue_after==record["ue_state_sha256_before"] if cfg.arm=="frozen_ue" else None,
                backend_after_trainer=_backend_info(torch), invocation=invocation)
            if not args.resume:
                verify_checkpoint_hash(record["source_path"], cfg.source_identity["checkpoint_sha256"])
            write_record(cfg, f"completion_step{latest['step']}_{invocation}.json", result)
            print(json.dumps(result, indent=2, allow_nan=False))
        except Exception as error:
            write_record(cfg, f"failure_{invocation}.json", dict(normal_return=False,
                error_type=type(error).__name__, error=str(error), invocation=invocation,
                arm=cfg.arm, seed=cfg.SEED))
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--stop-after", type=int, metavar="N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Fresh only: same pinned source bytes at an explicit path")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.dry_run or args.train or args.stop_after is not None):
        if args.resume or args.checkpoint is not None:
            parser.error("Execution requires an explicit mode")
        parser.print_help()
        return 0
    if args.arm is None or args.seed is None:
        parser.error("Choose --arm and --seed explicitly")
    if args.resume and (args.dry_run or args.checkpoint is not None):
        parser.error("Resume requires train/stop-after and forbids fresh checkpoint override")
    if args.stop_after is not None and (args.stop_after<=0 or args.stop_after>20000 or args.stop_after%500):
        parser.error("stop-after must be a positive multiple of 500 within the fixed 20k budget")
    cfg = as_spatial_config(args.arm, args.seed)
    validate_config(cfg)
    validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps(dict(action="plan only; no weights or files written", config=cfg.to_dict(),
            fingerprint=fingerprint(cfg), latest=cfg.SAVE_PATH, best=cfg.BEST_PATH,
            source_path=str(args.checkpoint) if args.checkpoint else cfg.source_identity["path"],
            source_availability_not_checked=True, scheduler_total_steps=cfg.TOTAL_STEPS,
            optimizer_groups=optimizer_group_spec(cfg)), indent=2))
        return 0
    run_training(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
