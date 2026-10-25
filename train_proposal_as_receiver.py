#!/usr/bin/env python3
"""Isolated ASReceiver two-seed screening, using the existing task trainer.

Only --train or --stop-after explicitly runs CUDA training. --dry-run is
standard-library-only. The fixed 20k budget includes any 500-update smoke run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import uuid

from configs.as_receiver_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, REPO_ROOT, as_receiver_config,
    make_channel_data, validate_config, channel_data_api,
)
from preflight_proposal_as_direct import (
    DEFAULT_MANIFEST, _backend_info, _sha256_stream, sha256_file,
    validate_checkpoint_metadata, verify_checkpoint_hash,
)
# These helpers are configuration-agnostic. The old 5k/ASDirect validators,
# factory, training loop and CLI are deliberately NOT reused.
from train_proposal_as_direct import (
    _safe_path, _close, expected_cycle, validate_model_state, optimizer_state_required,
)

SOURCE_FILES = (
    "train_proposal_as_receiver.py", "configs/as_receiver_train_config.py",
    "models/adaptive_hybrid.py", "models/adaptive_hybrid_as_receiver.py",
    "main_train.py", "configs/train_config.py", "utils/replayable_channel_data.py",
    "utils/sionna_rng.py", "utils/channel_generator.py", "utils/layers.py",
    "preflight_proposal_as_direct.py", "train_proposal_as_direct.py",
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def fingerprint(cfg):
    return hashlib.sha256(_json(cfg.to_dict()).encode("utf-8")).hexdigest()[:16]


def validate_output_paths(cfg, *, allowed_root=OUTPUT_ROOT):
    root = Path(allowed_root).absolute()
    if Path(cfg.output_root).absolute() != root:
        raise ValueError("Wrong dedicated ASReceiver output root")
    if not cfg.EXP_NAME or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cfg.EXP_NAME):
        raise ValueError("Unsafe experiment name")
    paths = {"SAVE_DIR": root / "checkpoints" / cfg.EXP_NAME,
             "LOG_DIR": root / "logs" / cfg.EXP_NAME}
    for field, name in (("SAVE_PATH", "latest.pth"), ("BEST_PATH", "best.pth"),
                        ("CONFIG_PATH", "config.json"), ("STATUS_PATH", "status.json")):
        paths[field] = paths["SAVE_DIR"] / name
    paths["METRICS_PATH"] = paths["LOG_DIR"] / "metrics.jsonl"
    for field, expected in paths.items():
        actual = Path(getattr(cfg, field))
        _safe_path(actual, root)
        if actual.absolute() != expected:
            raise ValueError(f"Unexpected {field}: {actual}")
        _safe_path(actual.with_suffix(actual.suffix + ".tmp"), root)
    _safe_path(root / "locks" / (cfg.EXP_NAME + ".lock"), root)


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
    saved = json.loads(Path(cfg.CONFIG_PATH).read_text(encoding="utf-8"))
    if _json(saved) != _json(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg))):
        raise ValueError("Resume config/source/data protocol/fingerprint mismatch")
    status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding="utf-8"))
    if status.get("experiment") != cfg.EXP_NAME or status.get("state") not in ("paused", "running"):
        raise ValueError("Resume status must identify this unfinished arm and seed")
    if status.get("fingerprint", fingerprint(cfg)) != fingerprint(cfg):
        raise ValueError("Resume status fingerprint mismatch")


def validate_resume_payload(payload, cfg, *, allow_completed=False):
    """Complete JSON/schema checks; real tensor/RNG checks run only on CUDA host."""
    channel_data_api().validate_resume_config(payload, cfg)
    if "allocation_grid" not in payload or _json(payload["allocation_grid"]) != _json(cfg.ALLOCATION_GRID):
        raise ValueError("Latest allocation_grid must remain singleton [(0,256)]")
    step = payload["step"]
    upper = cfg.TOTAL_STEPS if allow_completed else cfg.TOTAL_STEPS - 1
    if type(step) is not int or not 0 < step <= upper or step % cfg.VAL_INTERVAL:
        raise ValueError("Latest step must be a 500-update boundary within the fixed 20k budget")
    state = payload["state"]
    if "running_pwr" not in state:
        raise ValueError("Latest running_pwr missing")
    extra = {name for name in state if name.startswith("as_receiver.")}
    if bool(extra) != (cfg.arm == "as_receiver") or any(name.endswith(".alpha") for name in state):
        raise ValueError("Latest receiver/ASDirect state keys do not match the selected arm")
    scheduler = payload["scheduler"]
    phases = [
        dict(end_step=cfg.WARMUP_PCT * cfg.TOTAL_STEPS - 1,
             start_lr="initial_lr", end_lr="max_lr", start_momentum="max_momentum", end_momentum="base_momentum"),
        dict(end_step=cfg.TOTAL_STEPS - 1, start_lr="max_lr", end_lr="min_lr",
             start_momentum="base_momentum", end_momentum="max_momentum"),
    ]
    for field, expected in {
        "total_steps": cfg.TOTAL_STEPS, "last_epoch": step, "_step_count": step + 1,
        "_schedule_phases": phases, "_anneal_func_type": "cos", "cycle_momentum": True,
        "use_beta1": True, "base_lrs": [cfg.LR / 25],
    }.items():
        if scheduler.get(field) != expected:
            raise ValueError(f"Latest scheduler {field} mismatch")
    lr, beta1 = expected_cycle(cfg, step)
    if not isinstance(scheduler.get("_last_lr"), list) or len(scheduler["_last_lr"]) != 1:
        raise ValueError("Scheduler must have one shared learning rate")
    _close(scheduler["_last_lr"][0], lr, "scheduler LR")
    optimizer = payload["optimizer"]
    if not optimizer.get("state") or len(optimizer.get("param_groups", [])) != 1:
        raise ValueError("Initialized AdamW state and a single shared group are required")
    group = optimizer["param_groups"][0]
    for field, expected in {
        "lr": lr, "initial_lr": cfg.LR / 25, "max_lr": cfg.LR,
        "min_lr": cfg.LR / 25 / 10000, "weight_decay": cfg.WEIGHT_DECAY,
        "eps": 1e-8, "base_momentum": 0.85, "max_momentum": 0.95,
    }.items():
        _close(group.get(field), expected, f"optimizer {field}")
    if len(group.get("betas", [])) != 2:
        raise ValueError("AdamW betas missing")
    _close(group["betas"][0], beta1, "AdamW beta1")
    _close(group["betas"][1], 0.999, "AdamW beta2")
    for field in ("amsgrad", "maximize", "capturable", "differentiable"):
        if group.get(field) is not False:
            raise ValueError(f"Unexpected AdamW {field}")
    if group.get("foreach") is not None or group.get("fused") is not None:
        raise ValueError("No foreach/fused override is permitted")
    rng = payload["rng_state"]
    if any(rng.get(k) is None for k in ("python", "numpy", "torch", "cuda")):
        raise ValueError("Complete default Python/NumPy/Torch/all-CUDA RNG states are required")
    if not isinstance(rng["cuda"], (tuple, list)) or not rng["cuda"]:
        raise ValueError("CUDA RNG state missing")


def validate_best_metadata(best, latest, cfg):
    required = {"state", "config", "config_fingerprint", "allocation_grid", "step", "best_metric"}
    if not isinstance(best, dict) or not required <= best.keys():
        raise ValueError("Retained best checkpoint is incomplete")
    if best["config_fingerprint"] != fingerprint(cfg) or _json(best["config"]) != _json(cfg.to_dict()):
        raise ValueError("Retained best arm/seed/source/data protocol mismatch")
    if _json(best["allocation_grid"]) != _json(cfg.ALLOCATION_GRID):
        raise ValueError("Retained best allocation mismatch")
    if type(best["step"]) is not int or not 0 < best["step"] <= latest["step"] or best["step"] % cfg.VAL_INTERVAL:
        raise ValueError("Retained best step is incompatible with latest")
    _close(best["best_metric"], latest["best_metric"], "best/latest selection metric")


def validate_runtime_resume(torch, model, payload, cfg):
    import numpy as np
    validate_resume_payload(payload, cfg)
    validate_model_state(torch, model, payload["state"])
    model.load_state_dict(payload["state"], strict=True)
    parameters = list(model.named_parameters())
    group = payload["optimizer"]["param_groups"][0]
    ids = group.get("params")
    if ids != list(range(len(parameters))):
        raise ValueError("Optimizer parameter IDs/order/count mismatch")
    moments = payload["optimizer"]["state"]
    if not isinstance(moments, dict) or not set(moments) <= set(ids):
        raise ValueError("Optimizer state refers to unknown parameters")
    for index, (name, parameter) in enumerate(parameters):
        required = optimizer_state_required(name)
        if index not in moments:
            if required:
                raise ValueError(f"Missing active optimizer state: {name}")
            continue
        if not required:
            raise ValueError(f"Unexpected optimizer history for bypassed task parameter: {name}")
        saved = moments[index]
        if not isinstance(saved, dict) or set(saved) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Incomplete AdamW state: {name}")
        counter = saved["step"]
        if not torch.is_tensor(counter) or counter.numel() != 1 or not torch.isfinite(counter).all() or counter.item() != payload["step"]:
            raise ValueError(f"Optimizer step mismatch: {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = saved[key]
            if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError(f"Optimizer {key} shape/dtype/finiteness mismatch: {name}")
        if not (saved["exp_avg_sq"] >= 0).all():
            raise ValueError(f"Negative AdamW second moment: {name}")
    rng = payload["rng_state"]
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume requires the saved visible CUDA device count")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    for device, value in [("cpu", rng["torch"])] + [(f"cuda:{i}", s) for i, s in enumerate(rng["cuda"])]:
        if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
            raise ValueError(f"Malformed RNG tensor for {device}")
        torch.Generator(device=device).set_state(value)


def initialize_model(cfg, model, *, source_state=None, resume_state=None):
    if (source_state is None) == (resume_state is None):
        raise ValueError("Choose exactly one initialization or resume state")
    if resume_state is not None:
        model.load_state_dict(resume_state, strict=True)
    elif cfg.arm == "as_receiver":
        from models.adaptive_hybrid_as_receiver import migrate_original_state_dict
        migrate_original_state_dict(model, source_state)
    else:
        model.load_state_dict(source_state, strict=True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("All model parameters must remain trainable")
    return model


def make_model_factory(cfg, model, *, source_state=None, resume_state=None):
    """Load/migrate before optimizer creation; avoid arm-dependent init RNG drift."""
    used = False
    def factory(actual_cfg, device):
        nonlocal used, source_state, resume_state
        if used or fingerprint(actual_cfg) != fingerprint(cfg):
            raise RuntimeError("Factory reused or called with a different config")
        used = True
        fresh = source_state is not None
        initialize_model(cfg, model, source_state=source_state, resume_state=resume_state)
        if fresh:
            import torch
            current = model.state_dict()
            for name, value in source_state.items():
                if not torch.equal(current[name].detach().cpu(), value.detach().cpu()):
                    raise ValueError(f"Migration changed original parameter/buffer: {name}")
        source_state = resume_state = None
        model.to(device)
        if fresh:
            from main_train import set_seed
            set_seed(cfg.SEED)
        # On resume, trainer restores checkpoint RNG AFTER factory/optimizer
        # construction. Do not reset that restored state after train_proposal.
        return model
    return factory


def _create_lock(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _recover_stale_lock(path, cfg, record):
    """Explicit resume only: same host + PID proven dead; never guess ownership."""
    recovery = path.with_suffix(path.suffix + ".recovery")
    recovery.mkdir()  # Atomic exclusion among cooperating recovery attempts.
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        expected = {"schema": "as_receiver_lock_v1", "hostname": platform.node(),
                    "experiment": cfg.EXP_NAME, "arm": cfg.arm, "seed": cfg.SEED}
        if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Unknown/foreign lock identity; refusing automatic recovery")
        pid = saved.get("pid")
        if type(pid) is not int or pid <= 0 or not isinstance(saved.get("token"), str):
            raise RuntimeError("Malformed lock PID/token; refusing automatic recovery")
        if os.name != "posix":
            raise RuntimeError("Automatic dead-PID lock recovery is supported only on POSIX")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError("Cannot prove lock owner exited") from exc
        else:
            raise RuntimeError("Lock owner is still alive; refusing concurrent training")
        archive = path.with_name(path.name + ".stale_" + uuid.uuid4().hex + ".json")
        path.rename(archive)
        print(f"[Resume lock] Preserved confirmed dead-owner lock at {archive}")
        _create_lock(path, record)  # Never overwrite a competing fresh owner.
    finally:
        recovery.rmdir()  # Exact empty guard created by this invocation.


@contextmanager
def run_lock(cfg, *, resume=False, allowed_root=OUTPUT_ROOT):
    path = Path(allowed_root) / "locks" / (cfg.EXP_NAME + ".lock")
    _safe_path(path, allowed_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "as_receiver_lock_v1", "pid": os.getpid(),
              "hostname": platform.node(), "experiment": cfg.EXP_NAME,
              "arm": cfg.arm, "seed": cfg.SEED, "token": uuid.uuid4().hex,
              "utc": datetime.now(timezone.utc).isoformat()}
    try:
        _create_lock(path, record)
    except FileExistsError:
        if not resume:
            raise
        _recover_stale_lock(path, cfg, record)
    try:
        yield
    finally:
        if json.loads(path.read_text(encoding="utf-8")).get("token") != record["token"]:
            raise RuntimeError("Lock ownership changed; refusing to remove another owner")
        path.unlink()


def write_record(cfg, filename, record):
    path = Path(cfg.LOG_DIR) / filename
    _safe_path(path, OUTPUT_ROOT)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _provenance(cfg, torch, args):
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "source_code_sha256": {p: sha256_file(REPO_ROOT / p) for p in SOURCE_FILES},
        "manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "python": sys.version, "platform": platform.platform(), "environment_before_trainer": _backend_info(torch),
        "config": cfg.to_dict(), "config_fingerprint": fingerprint(cfg),
        "mode": "resume_latest" if args.resume else "fresh_robust_weights_new_optimizer",
        "stop_after": args.stop_after, "scheduler_total_steps": cfg.TOTAL_STEPS,
        "screening_only": True, "full_training_bitwise_resume_verified": False,
        "channel_data_protocol": cfg.channel_data_protocol,
        "fresh_default_rng_seed_after_factory": cfg.SEED,
    }


def run_training(cfg, args):
    check_run_state(cfg, resume=args.resume)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("ASReceiver screening requires CUDA; no CPU fallback")
    from main_train import RunOptions, set_seed, train_proposal
    from models.adaptive_hybrid import AdaptiveHybridPrecoder
    from models.adaptive_hybrid_as_receiver import AdaptiveHybridPrecoderASReceiver
    data = make_channel_data(cfg)
    channel_data_api().validate_channel_data(cfg, data)
    with run_lock(cfg, resume=args.resume):
        check_run_state(cfg, resume=args.resume)
        cls = AdaptiveHybridPrecoderASReceiver if cfg.arm == "as_receiver" else AdaptiveHybridPrecoder
        # The new receiver's random projections must themselves be seeded;
        # a second reset after factory initialization aligns downstream draws.
        set_seed(cfg.SEED)
        model = cls(cfg)
        if type(model).__name__ != cfg.architecture_metadata["model_class"]:
            raise ValueError("Wrong model class")
        for branch in (model.encoder.fdma_backbone, model.encoder.aircomp_backbone):
            if len(branch.layers) != 4 or any(layer.linear1.out_features != 2048 or layer.linear2.in_features != 2048 for layer in branch.layers):
                raise ValueError("Both UE branches must remain Full FFN2048")
        if cfg.arm == "as_receiver":
            if model.receiver_enabled is not True:
                raise ValueError("ASReceiver training must use the enabled receiver path")
            actual_receiver = model.receiver_config_metadata
            for key, expected in {"width": 128, "heads": 4, "dropout": 0.0,
                                  "lambda_floor": cfg.AS_RECEIVER_LAMBDA_FLOOR,
                                  "solve_dtype": cfg.AS_RECEIVER_SOLVE_DTYPE,
                                  "scalar_gate": False}.items():
                if actual_receiver.get(key) != expected:
                    raise ValueError(f"Actual receiver {key} differs from fixed configuration")
        record = _provenance(cfg, torch, args)
        record["actual_model_class"] = type(model).__name__
        record["actual_receiver_metadata"] = model.receiver_config_metadata if cfg.arm == "as_receiver" else None
        if args.resume:
            with Path(cfg.SAVE_PATH).open("rb") as stream:
                record["resume_checkpoint_sha256"] = _sha256_stream(stream)
                stream.seek(0)
                payload = torch.load(stream, map_location="cpu", weights_only=False)
            validate_runtime_resume(torch, model, payload, cfg)
            with Path(cfg.BEST_PATH).open("rb") as stream:
                record["retained_best_sha256"] = _sha256_stream(stream)
                stream.seek(0)
                best = torch.load(stream, map_location="cpu", weights_only=False)
            validate_best_metadata(best, payload, cfg)
            validate_model_state(torch, model, best["state"])
            del best
            status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding="utf-8"))
            if status["state"] == "paused" and status.get("step") != payload["step"]:
                raise ValueError("Paused status/latest step mismatch")
            start = payload["step"] + 1
            factory = make_model_factory(cfg, model, resume_state=payload["state"])
        else:
            selected = cfg.source_identity["path"] if args.checkpoint is None else str((REPO_ROOT / args.checkpoint).resolve())
            with Path(selected).open("rb") as stream:
                verify_checkpoint_hash(selected, cfg.source_identity["checkpoint_sha256"], stream=stream)
                stream.seek(0)
                payload = torch.load(stream, map_location="cpu", weights_only=False)
            validate_checkpoint_metadata(payload, cfg.source_identity)
            extra = set(model.state_dict()) - set(payload["state"])
            if cfg.arm == "as_receiver":
                if not extra or any(not key.startswith("as_receiver.") for key in extra):
                    raise ValueError("Only explicit as_receiver.* additions may be absent from the original state")
            elif extra:
                raise ValueError("Full control must not add source-state keys")
            validate_model_state(torch, model, payload["state"], missing_gates=extra)
            record.update(loaded_source_path=selected,
                          loaded_source_sha256=cfg.source_identity["checkpoint_sha256"],
                          loaded_source_step=payload["step"])
            start = 1
            factory = make_model_factory(cfg, model, source_state=payload["state"])
        record["start_step"] = start
        record["planned_end_step"] = min(cfg.TOTAL_STEPS, start + args.stop_after - 1) if args.stop_after is not None else cfg.TOTAL_STEPS
        del payload
        if not args.resume:
            Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=False)
        invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
        write_record(cfg, f"invocation_{invocation}.json", record)
        try:
            train_proposal(cfg, RunOptions(resume=args.resume, skip_existing=False,
                                          stop_after=args.stop_after, init_checkpoint=None),
                           model_factory=factory, channel_data=data)
            latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
            validate_resume_payload(latest, cfg, allow_completed=True)
            if latest["step"] != record["planned_end_step"]:
                raise RuntimeError("Trainer ended at an unexpected update count")
            result = {"completed_updates": latest["step"], "total_updates": cfg.TOTAL_STEPS,
                      "arm": cfg.arm, "seed": cfg.SEED, "normal_return": True,
                      "environment_after_trainer": _backend_info(torch),
                      "latest_sha256": sha256_file(cfg.SAVE_PATH),
                      "config_fingerprint": fingerprint(cfg), "invocation": invocation,
                      "receiver_state_keys": sorted(k for k in latest["state"] if k.startswith("as_receiver."))}
            if not args.resume:
                verify_checkpoint_hash(record["loaded_source_path"], cfg.source_identity["checkpoint_sha256"])
            write_record(cfg, f"completion_step{latest['step']}_{invocation}.json", result)
            print(json.dumps(result, indent=2, allow_nan=False))
        except Exception as exc:
            write_record(cfg, f"failure_{invocation}.json", {
                "normal_return": False, "error_type": type(exc).__name__, "error": str(exc),
                "invocation": invocation, "arm": cfg.arm, "seed": cfg.SEED,
            })
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Stdlib plan only; no weights or writes")
    mode.add_argument("--train", action="store_true", help="Run remaining updates of fixed 20000 schedule")
    mode.add_argument("--stop-after", type=int, metavar="N", help="Pause after N additional updates, a positive multiple of 500")
    parser.add_argument("--resume", action="store_true", help="Strictly restore this arm/seed latest")
    parser.add_argument("--checkpoint", type=Path, help="Fresh only: relocated identical pinned robust bytes")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    active = args.dry_run or args.train or args.stop_after is not None
    if not active:
        if args.resume or args.checkpoint is not None:
            parser.error("--resume/--checkpoint requires explicit execution mode")
        parser.print_help()
        return 0
    if args.arm is None or args.seed is None:
        parser.error("Choose both --arm and --seed")
    if args.resume and (args.dry_run or args.checkpoint is not None):
        parser.error("--resume requires --train/--stop-after and forbids --checkpoint")
    if args.stop_after is not None and (args.stop_after <= 0 or args.stop_after > 20000 or args.stop_after % 500):
        parser.error("--stop-after must be a positive multiple of 500, at most 20000")
    cfg = as_receiver_config(args.arm, args.seed)
    validate_config(cfg)
    validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps({"action": "plan only; no weights read or files written",
                          "config": cfg.to_dict(), "fingerprint": fingerprint(cfg),
                          "checkpoint_path": str(args.checkpoint) if args.checkpoint else cfg.source_identity["path"],
                          "checkpoint_availability_not_checked": True,
                          "latest": cfg.SAVE_PATH, "best": cfg.BEST_PATH, "logs": cfg.LOG_DIR,
                          "scheduler_total_steps": cfg.TOTAL_STEPS,
                          "channel_data_protocol": cfg.channel_data_protocol}, indent=2))
        return 0
    run_training(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
