#!/usr/bin/env python3
"""Explicit two-arm, single-seed 5k screening; no action without a CLI mode.

The existing main_train loop owns data, loss, optimizer, scheduler and saves.
This adapter validates identity/resume before that loop can write any files.
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

from configs.as_direct_train_config import (
    ARMS, OUTPUT_ROOT, REPO_ROOT, as_direct_config, validate_config,
)
from preflight_proposal_as_direct import (
    DEFAULT_MANIFEST, _assert_source_state, _backend_info, _sha256_stream, sha256_file,
    validate_checkpoint_metadata, verify_checkpoint_hash,
)

GATES = tuple(f"layers.{index}.alpha" for index in range(5))
SOURCE_FILES = (
    "train_proposal_as_direct.py", "configs/as_direct_train_config.py",
    "preflight_proposal_as_direct.py", "main_train.py", "configs/train_config.py",
    "models/adaptive_hybrid.py", "models/adaptive_hybrid_as_direct.py",
    "utils/channel_generator.py", "utils/layers.py", "utils/gpu_setup.py",
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def fingerprint(cfg):
    return hashlib.sha256(_json(cfg.to_dict()).encode("utf-8")).hexdigest()[:16]


def _safe_path(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Output must be below {root}: {path}")
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Linked output path is not allowed: {part}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Resolved output escapes dedicated root: {path}")


def validate_output_paths(cfg, *, allowed_root=OUTPUT_ROOT):
    root = Path(allowed_root).absolute()
    if Path(cfg.output_root).absolute() != root:
        raise ValueError("Wrong dedicated ASDirect output root")
    if not cfg.EXP_NAME or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cfg.EXP_NAME):
        raise ValueError("Unsafe experiment name")
    paths = {"SAVE_DIR": root / "checkpoints" / cfg.EXP_NAME,
             "LOG_DIR": root / "logs" / cfg.EXP_NAME}
    for name, filename in (("SAVE_PATH", "latest.pth"), ("BEST_PATH", "best.pth"),
                           ("CONFIG_PATH", "config.json"), ("STATUS_PATH", "status.json")):
        paths[name] = paths["SAVE_DIR"] / filename
    paths["METRICS_PATH"] = paths["LOG_DIR"] / "metrics.jsonl"
    for name, expected in paths.items():
        actual = Path(getattr(cfg, name))
        _safe_path(actual, root)
        if actual.absolute() != expected:
            raise ValueError(f"Unexpected {name}: {actual}")
        _safe_path(actual.with_suffix(actual.suffix + ".tmp"), root)
    _safe_path(root / "locks" / (cfg.EXP_NAME + ".lock"), root)


def check_run_state(cfg, *, resume, allowed_root=OUTPUT_ROOT):
    validate_output_paths(cfg, allowed_root=allowed_root)
    if not resume:
        for path in (cfg.SAVE_DIR, cfg.LOG_DIR):
            if Path(path).exists():
                raise FileExistsError(f"Refusing to overwrite/reinitialize {path}")
        return
    for field in ("SAVE_PATH", "BEST_PATH", "CONFIG_PATH", "STATUS_PATH", "METRICS_PATH"):
        if not Path(getattr(cfg, field)).is_file():
            raise FileNotFoundError(f"Resume requires {field}: {getattr(cfg, field)}")
    snapshot = json.loads(Path(cfg.CONFIG_PATH).read_text(encoding="utf-8"))
    if _json(snapshot) != _json(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg))):
        raise ValueError("Resume config.json arm/source/config/fingerprint mismatch")
    status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding="utf-8"))
    if status.get("experiment") != cfg.EXP_NAME or status.get("state") not in ("paused", "running"):
        raise ValueError("Resume status must be this unfinished experiment")
    if status.get("fingerprint", fingerprint(cfg)) != fingerprint(cfg):
        raise ValueError("Resume status fingerprint mismatch")


def expected_cycle(cfg, step):
    """Original trainer's two-phase cosine OneCycle defaults (including beta1)."""
    warm_end = cfg.WARMUP_PCT * cfg.TOTAL_STEPS - 1
    if step <= warm_end:
        pct = step / warm_end
        start_lr, end_lr, start_m, end_m = cfg.LR / 25, cfg.LR, 0.95, 0.85
    else:
        pct = (step - warm_end) / (cfg.TOTAL_STEPS - 1 - warm_end)
        start_lr, end_lr, start_m, end_m = cfg.LR, cfg.LR / 25 / 10000, 0.85, 0.95
    def cosine(start, end):
        return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)
    return cosine(start_lr, end_lr), cosine(start_m, end_m)


def _close(actual, expected, label):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_resume_payload(payload, cfg, *, allow_completed=False):
    """Stdlib metadata/structure checks; tensor/RNG checks follow on CUDA host."""
    required = {"state", "config", "config_fingerprint", "allocation_grid", "step",
                "best_metric", "optimizer", "scheduler", "rng_state"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError("Latest checkpoint lacks complete resume fields")
    if payload["config_fingerprint"] != fingerprint(cfg) or _json(payload["config"]) != _json(cfg.to_dict()):
        raise ValueError("Latest arm/architecture/source/config/fingerprint mismatch")
    if _json(payload["allocation_grid"]) != _json(cfg.ALLOCATION_GRID):
        raise ValueError("Latest allocation_grid must remain singleton [(0,256)]")
    step = payload["step"]
    upper = cfg.TOTAL_STEPS if allow_completed else cfg.TOTAL_STEPS - 1
    if type(step) is not int or not 0 < step <= upper or step % cfg.VAL_INTERVAL:
        raise ValueError("Latest step must be an unfinished 500-step boundary (5000 only for completion audit)")
    if not math.isfinite(float(payload["best_metric"])):
        raise ValueError("Latest best_metric must be finite")
    state = payload["state"]
    if not isinstance(state, dict) or "running_pwr" not in state:
        raise ValueError("Latest state/running_pwr missing")
    actual_gates = {name for name in state if name.endswith(".alpha")}
    if actual_gates != (set(GATES) if cfg.arm == "as_direct" else set()):
        raise ValueError("Latest gate keys do not match arm")
    scheduler = payload["scheduler"]
    if not isinstance(scheduler, dict):
        raise ValueError("Missing scheduler state")
    phases = [
        dict(end_step=cfg.WARMUP_PCT * cfg.TOTAL_STEPS - 1,
             start_lr="initial_lr", end_lr="max_lr", start_momentum="max_momentum", end_momentum="base_momentum"),
        dict(end_step=cfg.TOTAL_STEPS - 1, start_lr="max_lr", end_lr="min_lr",
             start_momentum="base_momentum", end_momentum="max_momentum"),
    ]
    for name, expected in {
        "total_steps": 5000, "last_epoch": step, "_step_count": step + 1,
        "_schedule_phases": phases, "_anneal_func_type": "cos",
        "cycle_momentum": True, "use_beta1": True, "base_lrs": [cfg.LR / 25],
    }.items():
        if scheduler.get(name) != expected:
            raise ValueError(f"Latest scheduler {name} mismatch")
    lr, beta1 = expected_cycle(cfg, step)
    if not isinstance(scheduler.get("_last_lr"), list) or len(scheduler["_last_lr"]) != 1:
        raise ValueError("Latest scheduler requires one shared learning rate")
    _close(scheduler["_last_lr"][0], lr, "scheduler LR")
    optimizer = payload["optimizer"]
    if not isinstance(optimizer, dict) or not optimizer.get("state") or len(optimizer.get("param_groups", [])) != 1:
        raise ValueError("Latest optimizer must contain initialized state and one shared group")
    group = optimizer["param_groups"][0]
    for name, expected in {"lr": lr, "initial_lr": cfg.LR / 25, "max_lr": cfg.LR,
                           "min_lr": cfg.LR / 25 / 10000, "weight_decay": cfg.WEIGHT_DECAY,
                           "eps": 1e-8, "base_momentum": 0.85, "max_momentum": 0.95}.items():
        _close(group.get(name), expected, f"optimizer {name}")
    if len(group.get("betas", [])) != 2:
        raise ValueError("AdamW betas missing")
    _close(group["betas"][0], beta1, "AdamW beta1")
    _close(group["betas"][1], 0.999, "AdamW beta2")
    for name in ("amsgrad", "maximize", "capturable", "differentiable"):
        if group.get(name) is not False:
            raise ValueError(f"Unexpected AdamW {name}")
    if group.get("foreach") is not None or group.get("fused") is not None:
        raise ValueError("Explicit foreach/fused overrides do not match original trainer")
    rng = payload["rng_state"]
    if not isinstance(rng, dict) or any(rng.get(k) is None for k in ("python", "numpy", "torch", "cuda")):
        raise ValueError("Complete Python/NumPy/CPU/all-CUDA RNG required")
    if not isinstance(rng["cuda"], (list, tuple)) or not rng["cuda"]:
        raise ValueError("CUDA RNG state missing")


def validate_model_state(torch, model, state, *, missing_gates=()):
    expected = model.state_dict()
    if set(state) != set(expected) - set(missing_gates):
        raise ValueError("State key set mismatch; no filtering/partial loading allowed")
    for name, value in state.items():
        reference = expected[name]
        if not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"State shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite state: {name}")
    if tuple(state["running_pwr"].shape) != (1,) or not (state["running_pwr"] > 0).all():
        raise ValueError("running_pwr must be a positive singleton indexed at 0")


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
        # Inactive FDMA / unused legacy outputs legitimately have no moments.
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
        if not torch.is_tensor(counter) or counter.numel() != 1 or not torch.isfinite(counter).all():
            raise ValueError(f"Invalid optimizer step: {name}")
        count = counter.item()
        if count != int(count) or not 0 < count <= payload["step"] or (required and count != payload["step"]):
            raise ValueError(f"Optimizer step inconsistent with saved updates: {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = saved[key]
            if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError(f"Optimizer {key} shape/dtype/finiteness mismatch: {name}")
        if not (saved["exp_avg_sq"] >= 0).all():
            raise ValueError(f"Negative AdamW second moment: {name}")
    rng = payload["rng_state"]
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume requires the same number of visible CUDA RNG states")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    for device, value in [("cpu", rng["torch"])] + [(f"cuda:{i}", s) for i, s in enumerate(rng["cuda"])]:
        if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
            raise ValueError(f"Malformed RNG tensor for {device}")
        torch.Generator(device=device).set_state(value)  # Validate without changing global RNG.


def optimizer_state_required(name):
    """Only parameters bypassed by the fixed pure-AS task path may lack moments."""
    if name in {"encoder.fdma_sb_query", "pad_f", "pad_a", "residual_scale"}:
        return False
    inactive = ("encoder.fdma_input_proj.", "encoder.fdma_input_norm.",
                "encoder.fdma_backbone.", "encoder.fdma_attn_pool.",
                "encoder.fc_fdma.", "fdma_emb.", "residual_head.")
    return not name.startswith(inactive)


def validate_best_metadata(best, latest, cfg):
    """Retained selected weights must belong to this run, not just exist on disk."""
    required = {"state", "config", "config_fingerprint", "allocation_grid", "step", "best_metric"}
    if not isinstance(best, dict) or not required <= best.keys():
        raise ValueError("Retained best checkpoint is incomplete")
    if best["config_fingerprint"] != fingerprint(cfg) or _json(best["config"]) != _json(cfg.to_dict()):
        raise ValueError("Retained best checkpoint arm/source/config mismatch")
    if _json(best["allocation_grid"]) != _json(cfg.ALLOCATION_GRID):
        raise ValueError("Retained best allocation mismatch")
    if type(best["step"]) is not int or not 0 < best["step"] <= latest["step"] or best["step"] % cfg.VAL_INTERVAL:
        raise ValueError("Retained best step is incompatible with latest")
    _close(best["best_metric"], latest["best_metric"], "best/latest selection metric")


def initialize_model(cfg, model, *, source_state=None, resume_state=None):
    """Called only by factory before trainer creates its optimizer."""
    if (source_state is None) == (resume_state is None):
        raise ValueError("Choose exactly one initialization or resume state")
    if resume_state is not None:
        model.load_state_dict(resume_state, strict=True)
    elif cfg.arm == "as_direct":
        from models.adaptive_hybrid_as_direct import migrate_original_state_dict
        migrate_original_state_dict(model, source_state)
    else:
        model.load_state_dict(source_state, strict=True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("All model parameters must remain trainable")
    return model


def make_model_factory(cfg, model, *, source_state=None, resume_state=None):
    used = False
    def factory(actual_cfg, device):
        nonlocal used, source_state, resume_state
        if used or fingerprint(actual_cfg) != fingerprint(cfg):
            raise RuntimeError("Factory reused or called with a different config")
        used = True
        initialize_model(cfg, model, source_state=source_state, resume_state=resume_state)
        if source_state is not None:
            import torch
            _assert_source_state(torch, model, source_state, gates=GATES if cfg.arm == "as_direct" else ())
        source_state = resume_state = None
        return model.to(device)
    return factory


@contextmanager
def run_lock(cfg):
    path = OUTPUT_ROOT / "locks" / (cfg.EXP_NAME + ".lock")
    _safe_path(path, OUTPUT_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        yield
    finally:
        path.unlink()  # Exact lock created by this call; never remove a stale lock.


def write_record(cfg, filename, record):
    path = Path(cfg.LOG_DIR) / filename
    _safe_path(path, OUTPUT_ROOT)
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _provenance(cfg, torch, args):
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "source_code_sha256": {p: sha256_file(REPO_ROOT / p) for p in SOURCE_FILES},
        "manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "python": sys.version, "platform": platform.platform(), "environment_before_trainer": _backend_info(torch),
        "config": cfg.to_dict(), "config_fingerprint": fingerprint(cfg),
        "mode": "resume_latest" if args.resume else "fresh_robust_weights_new_optimizer",
        "stop_after": args.stop_after, "scheduler_total_steps": 5000,
        "screening_only": True, "uninterrupted_bitwise_equivalence_verified": False,
    }


def run_training(cfg, args):
    check_run_state(cfg, resume=args.resume)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("ASDirect screening requires CUDA; no CPU fallback")
    from main_train import RunOptions, train_proposal
    from models.adaptive_hybrid import AdaptiveHybridPrecoder
    from models.adaptive_hybrid_as_direct import AdaptiveHybridPrecoderASDirect
    with run_lock(cfg):
        check_run_state(cfg, resume=args.resume)
        cls = AdaptiveHybridPrecoderASDirect if cfg.arm == "as_direct" else AdaptiveHybridPrecoder
        model = cls(cfg)  # One CPU instance for validation, later moved by factory.
        if type(model).__name__ != cfg.architecture_metadata["model_class"]:
            raise ValueError("Wrong model class")
        for branch in (model.encoder.fdma_backbone, model.encoder.aircomp_backbone):
            if len(branch.layers) != 4 or any(layer.linear1.out_features != 2048 or layer.linear2.in_features != 2048 for layer in branch.layers):
                raise ValueError("Both UE branches must remain Full FFN2048")
        record = _provenance(cfg, torch, args)
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
            validate_model_state(torch, model, payload["state"], missing_gates=GATES if cfg.arm == "as_direct" else ())
            record["loaded_source_path"] = selected
            record["loaded_source_sha256"] = cfg.source_identity["checkpoint_sha256"]
            record["loaded_source_step"] = payload["step"]
            start = 1
            factory = make_model_factory(cfg, model, source_state=payload["state"])
        record["start_step"] = start
        record["planned_end_step"] = min(5000, start + args.stop_after - 1) if args.stop_after is not None else 5000
        # No source optimizer state is passed through. Trainer constructs its own.
        del payload
        if not args.resume:
            Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=False)
        invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
        write_record(cfg, f"invocation_{invocation}.json", record)
        try:
            train_proposal(cfg, RunOptions(resume=args.resume, skip_existing=False,
                                          stop_after=args.stop_after, init_checkpoint=None),
                           model_factory=factory)
            latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
            validate_resume_payload(latest, cfg, allow_completed=True)
            if latest["step"] != record["planned_end_step"]:
                raise RuntimeError("Trainer ended at an unexpected update count")
            gates = {key: {"alpha": float(latest["state"][key].item()),
                           "tanh_alpha": math.tanh(float(latest["state"][key].item()))}
                     for key in GATES} if cfg.arm == "as_direct" else {}
            result = {"completed_updates": latest["step"], "total_updates": 5000,
                      "arm": cfg.arm, "gates": gates, "normal_return": True,
                      "environment_after_trainer": _backend_info(torch),
                      "latest_sha256": sha256_file(cfg.SAVE_PATH),
                      "config_fingerprint": fingerprint(cfg), "invocation": invocation}
            if not args.resume:
                verify_checkpoint_hash(record["loaded_source_path"], cfg.source_identity["checkpoint_sha256"])
            write_record(cfg, f"completion_step{latest['step']}_{invocation}.json", result)
            print(json.dumps(result, indent=2, allow_nan=False))
        except Exception as exc:
            write_record(cfg, f"failure_{invocation}.json", {
                "normal_return": False, "error_type": type(exc).__name__, "error": str(exc),
                "invocation": invocation, "arm": cfg.arm,
            })
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Stdlib plan only: no weights, models or writes")
    mode.add_argument("--train", action="store_true", help="Run all remaining updates of the fixed 5000 schedule")
    mode.add_argument("--stop-after", type=int, metavar="N", help="Pause after N additional updates; positive multiple of 500")
    parser.add_argument("--resume", action="store_true", help="Strictly restore this arm's latest, never reinitialize")
    parser.add_argument("--checkpoint", type=Path, help="Fresh run only: relocated identical robust source bytes (SHA256 required)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    active = args.dry_run or args.train or args.stop_after is not None
    if not active:
        if args.resume or args.checkpoint is not None:
            parser.error("--resume/--checkpoint requires an explicit execution mode")
        parser.print_help()
        return 0
    if args.arm is None:
        parser.error("Choose --arm full_control or --arm as_direct")
    if args.resume and (args.dry_run or args.checkpoint is not None):
        parser.error("--resume requires --train/--stop-after and forbids --checkpoint")
    if args.stop_after is not None and (args.stop_after <= 0 or args.stop_after > 5000 or args.stop_after % 500):
        parser.error("--stop-after must be a positive multiple of 500, at most 5000; total schedule stays 5000")
    cfg = as_direct_config(args.arm)
    validate_config(cfg)
    validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps({"action": "plan only; no weights read or files written",
                          "config": cfg.to_dict(), "fingerprint": fingerprint(cfg),
                          "checkpoint_path": str(args.checkpoint) if args.checkpoint else cfg.source_identity["path"],
                          "checkpoint_availability_not_checked": True,
                          "latest": cfg.SAVE_PATH, "best": cfg.BEST_PATH, "logs": cfg.LOG_DIR,
                          "new_optimizer": "AdamW", "scheduler_total_steps": cfg.TOTAL_STEPS,
                          "existing_run": Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists()}, indent=2))
        return 0
    run_training(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
