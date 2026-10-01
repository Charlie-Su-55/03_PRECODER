#!/usr/bin/env python3
"""Opt-in Lite1024 pure-AS specialist; no action without an explicit CLI mode.

The training loop, data generation, losses, scheduler and checkpoint writer are
main_train's existing implementations. Import/dry-run needs only stdlib/configs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from configs.lite_train_config import (
    LITE_ROOT, REPO_ROOT, lite_config, validate_lite_recipe,
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def lite_fingerprint(cfg):
    # Same serialization as main_train.config_fingerprint, without importing it.
    return hashlib.sha256(_json(cfg.to_dict()).encode("utf-8")).hexdigest()[:16]


def _safe_descendant(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Output path must be below the Lite root: {path}")
    # Check lexical components BEFORE resolve; include the root and its parents
    # so a linked proposal_lite/runs_as_screen cannot redirect both comparisons.
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Symlink/junction not allowed in Lite output path: {part}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Resolved output escapes Lite root: {path}")


def validate_output_paths(cfg, *, allowed_root=LITE_ROOT):
    root = Path(allowed_root).absolute()
    if Path(cfg.output_root).resolve() != root.resolve():
        raise ValueError(f"Expected dedicated Lite output root: {root}")
    if not cfg.EXP_NAME or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cfg.EXP_NAME):
        raise ValueError("Unsafe experiment name")
    expected = {
        "SAVE_DIR": root / "checkpoints" / cfg.EXP_NAME,
        "LOG_DIR": root / "logs" / cfg.EXP_NAME,
    }
    expected.update({
        "SAVE_PATH": expected["SAVE_DIR"] / "latest.pth",
        "BEST_PATH": expected["SAVE_DIR"] / "best.pth",
        "CONFIG_PATH": expected["SAVE_DIR"] / "config.json",
        "STATUS_PATH": expected["SAVE_DIR"] / "status.json",
        "METRICS_PATH": expected["LOG_DIR"] / "metrics.jsonl",
    })
    for name, path in expected.items():
        _safe_descendant(path, root)
        actual = Path(getattr(cfg, name))
        _safe_descendant(actual, root)
        if actual.resolve() != path.resolve():
            raise ValueError(f"Unexpected {name}: {actual}")
        if name not in ("SAVE_DIR", "LOG_DIR"):
            _safe_descendant(path.with_suffix(path.suffix + ".tmp"), root)
    _safe_descendant(root / "locks" / (cfg.EXP_NAME + ".lock"), root)


def check_run_state(cfg, *, resume, allowed_root=LITE_ROOT):
    """Read-only checks; a missing latest can never become a fresh run."""
    validate_output_paths(cfg, allowed_root=allowed_root)
    if not resume:
        for path in (cfg.SAVE_DIR, cfg.LOG_DIR):
            if Path(path).exists():
                raise FileExistsError(f"Refusing to overwrite/reinitialize existing run: {path}")
        return
    for name in ("SAVE_PATH", "CONFIG_PATH", "STATUS_PATH", "BEST_PATH", "METRICS_PATH"):
        if not Path(getattr(cfg, name)).is_file():
            raise FileNotFoundError(f"Resume requires {name}: {getattr(cfg, name)}")
    snapshot = json.loads(Path(cfg.CONFIG_PATH).read_text(encoding="utf-8"))
    expected = dict(cfg.to_dict(), config_fingerprint=lite_fingerprint(cfg))
    if _json(snapshot) != _json(expected):
        raise ValueError("Resume config.json metadata/fingerprint/config mismatch")
    status = json.loads(Path(cfg.STATUS_PATH).read_text(encoding="utf-8"))
    if status.get("state") == "completed" or int(status.get("step", 0)) >= cfg.TOTAL_STEPS:
        raise RuntimeError("Completed Lite experiment must not be restarted")


def validate_resume_payload(payload, cfg):
    """Validate only this run's latest checkpoint before the trainer writes."""
    required = ("state", "config", "config_fingerprint", "allocation_grid", "step",
                "best_metric", "optimizer", "scheduler", "rng_state")
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError("Latest checkpoint lacks required resume fields")
    if payload["config_fingerprint"] != lite_fingerprint(cfg):
        raise ValueError("Latest checkpoint fingerprint mismatch")
    if _json(payload["config"]) != _json(cfg.to_dict()):
        raise ValueError("Latest checkpoint architecture metadata/config mismatch")
    if _json(payload["allocation_grid"]) != _json(cfg.ALLOCATION_GRID):
        raise ValueError("Latest checkpoint allocation grid mismatch")
    step = payload["step"]
    if type(step) is not int or not 0 < step < cfg.TOTAL_STEPS:
        raise ValueError("Latest step must be positive and unfinished; completed runs cannot resume")
    if not math.isfinite(float(payload["best_metric"])):
        raise ValueError("Latest best_metric is not finite")
    scheduler = payload["scheduler"]
    if not isinstance(scheduler, dict) or scheduler.get("total_steps") != cfg.TOTAL_STEPS or scheduler.get("last_epoch") != step:
        raise ValueError("Latest scheduler must retain the 50000-step schedule and saved step")
    optimizer = payload["optimizer"]
    if not isinstance(optimizer, dict) or not optimizer.get("param_groups") or not optimizer.get("state"):
        raise ValueError("Latest checkpoint lacks initialized optimizer state")
    rng = payload["rng_state"]
    if not isinstance(rng, dict) or any(rng.get(key) is None for key in ("python", "numpy", "torch")):
        raise ValueError("Latest checkpoint lacks complete Python/NumPy/Torch RNG state")
    state = payload["state"]
    if not isinstance(state, dict) or "running_pwr" not in state or tuple(state["running_pwr"].shape) != (1,):
        raise ValueError("Latest running_pwr must match singleton allocation index 0")


def lite_model_factory(cfg, device):
    """Construct before optimizer creation; also accepts the separate diagnostic grid."""
    from models.adaptive_hybrid_lite import AdaptiveHybridPrecoderLite

    model = AdaptiveHybridPrecoderLite(cfg, ue_ffn_dim=cfg.ue_ffn_dim).to(device)
    actual = dict(model.architecture_metadata, model_class=type(model).__name__)
    if actual != cfg.architecture_metadata:
        raise RuntimeError(f"Constructed architecture differs from config: {actual}")
    widths = {}
    for branch in ("fdma_backbone", "aircomp_backbone"):
        layers = getattr(model.encoder, branch).layers
        widths[branch] = [layer.linear1.out_features for layer in layers]
        if len(layers) != 4 or any(layer.linear1.out_features != 1024 or layer.linear2.in_features != 1024 for layer in layers):
            raise RuntimeError(f"Unexpected UE FFN widths: {widths}")
    bs_cross = [layer.linear1.out_features for layer in model.cross_subband_attn.layers]
    bs_refinement = [layer.ffn[0].out_features for layer in model.layers]
    if bs_cross != [2048] or any(layer.linear2.in_features != 2048 for layer in model.cross_subband_attn.layers) or bs_refinement != [512] * 5:
        raise RuntimeError("BS FFNs must remain 2048 / 512")
    if tuple(model.running_pwr.shape) != (len(cfg.GAMMA_GRID),):
        raise RuntimeError("running_pwr does not match the actual allocation grid")
    counts = {"encoder": sum(p.numel() for p in model.encoder.parameters()),
              "condition": sum(p.numel() for p in model.cond_embed.parameters()),
              "full": sum(p.numel() for p in model.parameters())}
    counts["ue"] = counts["encoder"] + counts["condition"]
    counts["bs_remainder"] = counts["full"] - counts["ue"]
    if counts != dict(encoder=6607778, condition=66816, full=16006619, ue=6674594, bs_remainder=9332025):
        raise RuntimeError(f"Unexpected real-size Lite parameter counts: {counts}")
    print(json.dumps({"constructed_model": actual, "ue_ffn_widths": widths,
                      "bs_cross_subband_ffn": bs_cross, "bs_refinement_ffn": bs_refinement,
                      "parameters": counts, "allocations": cfg.ALLOCATION_GRID,
                      "save_dir": cfg.SAVE_DIR}, indent=2))
    return model


def _backend_info(torch):
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def gpu_preflight(cfg):
    """Synthetic CUDA diagnostics only: no optimizer, channel generator or writes."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("--preflight requires CUDA; no CPU fallback")
    from main_train import SumRateLoss, extract_training_outputs, temporary_seed

    before = _json(cfg.to_dict())
    diagnostic_cfg = replace(cfg, allocations=cfg.full_allocation_grid,
                             spec=replace(cfg.spec, train_mode="adaptive", allocation=None))
    print("Preflight: independent full-grid model; formal singleton stays at index 0.")
    with temporary_seed(cfg.SEED):
        device = torch.device("cuda")
        model = lite_model_factory(diagnostic_cfg, device)
        # Report effective settings after the model's normal dependency imports.
        print(json.dumps(_backend_info(torch), indent=2))
        shape = (1, cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS)
        h_dl = torch.randn(shape, device=device, dtype=torch.complex64)
        h_ul = torch.randn(shape, device=device, dtype=torch.complex64)
        h_ul_est = h_ul + 0.01 * torch.randn_like(h_ul)
        criterion = SumRateLoss()

        def check_precoder(value, label):
            if tuple(value.shape) != (1, cfg.ANTENNAS, cfg.K_USERS, cfg.SUBCARRIERS) or not torch.isfinite(value).all().item():
                raise RuntimeError(f"{label}: invalid shape/non-finite output")
            power = value.abs().square().sum(dim=(1, 2))
            error = (power - cfg.TOTAL_POWER).abs().max().item()
            if not torch.allclose(power, torch.full_like(power, cfg.TOTAL_POWER), rtol=1e-6, atol=1e-6):
                raise RuntimeError(f"{label}: per-subcarrier power max absolute error={error:.9g}")
            return error

        for allocation in ((64, 0), (32, 128), (0, 256)):
            idx = diagnostic_cfg.GAMMA_GRID.index(allocation)
            model.train()
            model.zero_grad(set_to_none=True)
            legacy_w, h_hat = extract_training_outputs(model(h_dl, h_ul, 10.0, *allocation, idx, H_ul_est=h_ul_est))
            check_precoder(legacy_w, f"{allocation} legacy interface only")
            if tuple(h_hat.shape) != shape or not torch.isfinite(h_hat).all().item():
                raise RuntimeError(f"{allocation}: invalid H_hat")
            w = model.zf_solver(h_hat)
            w = w * torch.sqrt(cfg.TOTAL_POWER / (w.abs().square().sum(dim=(1, 2), keepdim=True) + 1e-9))
            error = check_precoder(w, f"{allocation} deployed RZF")
            loss, _ = criterion(w, h_dl, 10.0 ** (-10.0 / 10.0))
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"{allocation}: non-finite task loss")
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all().item():
                    raise RuntimeError(f"{allocation}: non-finite gradient in {name}")
            for branch, active in (("fdma_backbone", allocation[0] > 0), ("aircomp_backbone", allocation[1] > 0)):
                if active:
                    for layer in getattr(model.encoder, branch).layers:
                        for linear in (layer.linear1, layer.linear2):
                            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in linear.parameters()):
                                raise RuntimeError(f"{allocation}: missing/non-finite active {branch} FFN gradient")
            model.eval()
            with torch.no_grad():
                for estimate in (None, h_ul_est):
                    check_precoder(model(h_dl, h_ul, 10.0, *allocation, idx, H_ul_est=estimate), f"{allocation} eval")
            print(f"PASS allocation={allocation}, diagnostic index={idx}, task backward finite, power max_abs={error:.9g}")
            del legacy_w, h_hat, w, loss
    if _json(cfg.to_dict()) != before:
        raise RuntimeError("Preflight changed formal training config")
    print("GPU preflight passed. No optimizer step, checkpoint load/write or formal evaluation.")


@contextmanager
def _run_lock(cfg):
    # A persistent lock directory, but no stale lock silently removed on failure.
    lock = LITE_ROOT / "locks" / (cfg.EXP_NAME + ".lock")
    _safe_descendant(lock, LITE_ROOT)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        yield
    finally:
        lock.unlink()  # Only the exact lock created by this invocation.


def _write_provenance(cfg, *, resume, stop_after):
    paths = ("train_proposal_lite.py", "main_train.py", "configs/train_config.py",
             "configs/lite_train_config.py", "models/adaptive_hybrid.py",
             "models/adaptive_hybrid_lite.py", "utils/channel_generator.py",
             "utils/layers.py", "utils/__init__.py", "utils/gpu_setup.py")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "tracked_changes": git("diff", "HEAD", "--name-only"),
        "source_sha256": {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in paths},
        "resume": resume, "stop_after": stop_after, "config_fingerprint": lite_fingerprint(cfg),
        "checkpoint_source": str(Path(cfg.SAVE_PATH)) if resume else None,
        "fresh_initialization": not resume,
    }
    name = "provenance_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json"
    path = Path(cfg.LOG_DIR) / name
    _safe_descendant(path, LITE_ROOT)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print config/paths only; no heavy imports or writes")
    mode.add_argument("--preflight", action="store_true", help="CUDA synthetic forward/backward only; no training")
    mode.add_argument("--stop-after", type=int, metavar="N", help="Run N steps then pause within the unchanged 50k schedule")
    mode.add_argument("--train", action="store_true", help="Explicitly run the full remaining standard training schedule")
    parser.add_argument("--resume", action="store_true", help="With --train/--stop-after only; restore this Lite run's latest")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.resume and not (args.train or args.stop_after is not None):
        parser.error("--resume requires --train or --stop-after N")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if not (args.dry_run or args.preflight or args.train or args.stop_after is not None):
        parser.print_help()
        return 0
    cfg = lite_config()
    validate_lite_recipe(cfg)
    validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps({"action": "inspect only; fresh training requires --stop-after N or --train",
                          "config": cfg.to_dict(), "config_fingerprint": lite_fingerprint(cfg),
                          "ue_ffn_widths": [1024] * 4, "bs_ffn_widths": {"cross_subband": 2048, "refinement": 512},
                          "paths": {name: getattr(cfg, name) for name in ("SAVE_PATH", "BEST_PATH", "CONFIG_PATH", "STATUS_PATH", "METRICS_PATH")},
                          "existing_run": Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists(),
                          "fresh_optimizer": "AdamW", "scheduler_total_steps": cfg.TOTAL_STEPS,
                          "formal_running_power_index": 0}, indent=2))
        return 0
    if args.preflight:
        gpu_preflight(cfg)
        return 0
    check_run_state(cfg, resume=args.resume)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Lite training entry requires CUDA; no local CPU training fallback")
    from main_train import RunOptions, train_proposal
    with _run_lock(cfg):
        check_run_state(cfg, resume=args.resume)
        if args.resume:
            payload = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
            validate_resume_payload(payload, cfg)
            cuda_rng = payload["rng_state"].get("cuda")
            if not isinstance(cuda_rng, (list, tuple)) or len(cuda_rng) != torch.cuda.device_count():
                raise ValueError("Resume requires RNG states for the same number of visible CUDA devices")
            del payload
        else:
            Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=False)
        validate_output_paths(cfg)
        _write_provenance(cfg, resume=args.resume, stop_after=args.stop_after)
        train_proposal(cfg, RunOptions(resume=args.resume, skip_existing=False,
                                      stop_after=args.stop_after, init_checkpoint=None),
                       model_factory=lite_model_factory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
