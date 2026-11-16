#!/usr/bin/env python3
"""Train and independently calibrate the shared K8 pilot-only UL CE U-Net.

Training observations are synthesized only from pilot REs. H_UL is used as a
supervision target and for offline metrics; it is never passed to the model.
Validation and calibration use independent step-addressed data roles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from configs import twc_v1_k8_ce_train_config as cfg
from models.twc_v1_k8_neural import K8PilotResidualUNet

ROOT = Path(__file__).resolve().parent


def _load_local(relative, module_name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Avoid utils/__init__.py: it configures the shared GPU merely on package import.
_link = _load_local("utils/twc_v1_k8_link.py", "_twc_v1_k8_ce_train_link")
_protocol = _link.protocol
_data_module = _load_local("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_ce_train_data")
estimate_ul_ls_interpolated = _link.estimate_ul_ls_interpolated
transmit_pilots = _link.transmit_pilots
TWCv1K8ReplayableData = _data_module.TWCv1K8ReplayableData
EVALUATION_FEEDBACK_GAMMA_DB = _protocol.EVALUATION_FEEDBACK_GAMMA_DB
PILOT_POSITIONS_BY_USER = _protocol.PILOT_POSITIONS_BY_USER
feedback_noise_variance = _protocol.feedback_noise_variance


DEFAULT_RUN = ROOT / "runs" / "twc_v1_k8_ce" / f"seed_{cfg.CE_SEED}"
GAMMAS = tuple(float(v) for v in EVALUATION_FEEDBACK_GAMMA_DB)
EPS = tuple(float(v) for v in cfg.PILOT_ENERGIES)


def _json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_torch_save(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _standard_noise(shape, *, device, generator):
    pair = torch.randn((*shape, 2), dtype=torch.float32, device=device, generator=generator)
    return torch.view_as_complex(pair.contiguous())


def _pilots(batch, *, device):
    value = torch.zeros((batch, 8, 256), dtype=torch.complex64, device=device)
    for user, positions in enumerate(PILOT_POSITIONS_BY_USER):
        value[:, user, list(positions)] = 1.0 + 0.0j
    return value


def _nmse_per_link(estimate, truth):
    numerator = (estimate - truth).abs().square().sum(dim=(2, 3))
    denominator = truth.abs().square().sum(dim=(2, 3)).clamp_min(1e-20)
    return numerator / denominator


def _condition(batch_index):
    return GAMMAS[batch_index % len(GAMMAS)], EPS[(batch_index // len(GAMMAS)) % len(EPS)]


def _data(base_seed, device, namespace):
    sizes = {
        "ce_train": cfg.BATCH_SIZE,
        "codec_train": cfg.BATCH_SIZE,
        "ce_calibration": cfg.CALIBRATION_BATCH_SIZE,
        "validation": cfg.VALIDATION_BATCH_SIZE,
        "test": 1,
    }
    return TWCv1K8ReplayableData(base_seed=base_seed, batch_sizes=sizes,
        device=device, cdl_model=cfg.CDL_MODEL, speed=cfg.SPEED_MPS,
        seed_namespace=namespace)


def _noise_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def _make_observation(H_ul, gamma, ep, seed):
    noise = _standard_noise((H_ul.shape[0], 32, 256), device=H_ul.device,
                            generator=_noise_generator(H_ul.device, seed))
    Y = transmit_pilots(H_ul, ep, gamma, noise)
    H_ls = estimate_ul_ls_interpolated(Y, ep)
    return Y, H_ls


def _metric_row(estimate, H_ul, gamma, ep, role, batch_index, step):
    per_link = _nmse_per_link(estimate, H_ul)
    return {
        "role": role, "step": step, "batch_index": batch_index, "gamma_db": gamma,
        "pilot_energy_per_ue": ep, "N0": feedback_noise_variance(gamma),
        "sample_count": int(per_link.numel()),
        "mean_per_link_nmse": float(per_link.mean().cpu()),
        "mean_per_link_nmse_db": float((10 * torch.log10(per_link.mean().clamp_min(1e-20))).cpu()),
    }


def _append_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _evaluate(model, data, device, batches, role, run_seed, step):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch_index in range(batches):
            sample = data.sample(role, batch_index)
            H_ul = sample["H_ul"].to(device)
            # Every gamma/Ep point uses the same fixed validation channels.
            for gamma_index, gamma in enumerate(GAMMAS):
                for ep_index, ep in enumerate(EPS):
                    noise_seed = run_seed + 800_000 + batch_index * 100 + gamma_index * 10 + ep_index
                    Y, H_ls = _make_observation(H_ul, gamma, ep, noise_seed)
                    estimate = model(Y, _pilots(H_ul.shape[0], device=device), H_ls,
                        pilot_energy_per_ue=ep,
                        noise_variance=feedback_noise_variance(gamma))
                    row = _metric_row(estimate, H_ul, gamma, ep, role, batch_index, step)
                    row["sample_fingerprint"] = sample["sample_fingerprint"]
                    row["LS_mean_per_link_nmse"] = float(_nmse_per_link(H_ls, H_ul).mean().cpu())
                    rows.append(row)
    model.train()
    return rows


def _mean_metric(rows):
    if not rows:
        return math.inf
    return float(np.mean([row["mean_per_link_nmse"] for row in rows]))


def _rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint(path, model, optimizer, step, best_nmse, data, device):
    payload = {
        "format": "twc_v1_k8_ce_checkpoint_v1", "step": int(step),
        "best_validation_nmse": float(best_nmse),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": _rng_state(device), "config": _config(),
        "data_protocol": data.protocol,
    }
    _atomic_torch_save(path, payload)


def _config():
    return {key: getattr(cfg, key) for key in (
        "CE_SEED", "BASE_DATA_SEED", "SEED_NAMESPACE", "CDL_MODEL", "SPEED_MPS",
        "UPDATES", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP_NORM",
        "VALIDATION_EVERY", "VALIDATION_BATCH_SIZE", "VALIDATION_BATCHES",
        "CALIBRATION_BATCH_SIZE", "CALIBRATION_BATCHES", "SAVE_EVERY", "GAMMA_DB",
        "PILOT_ENERGIES")}


def _truncate_jsonl(path, max_step):
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if int(record["step"]) <= max_step:
                retained.append(json.dumps(record, sort_keys=True))
    path.write_text("".join(line + "\n" for line in retained), encoding="utf-8")


def _truncate_validation_csv(path, max_step):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["step"]) <= max_step]
    if not rows:
        path.unlink()
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    run_dir = args.run_dir.resolve()
    harmless_console_logs = {"training.console.log", "calibration.console.log"}
    existing_artifacts = ([path for path in run_dir.iterdir()
                           if path.name not in harmless_console_logs]
                          if run_dir.exists() else [])
    if existing_artifacts and not args.resume:
        raise FileExistsError(f"Run directory already contains files; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(cfg.CE_SEED)
    np.random.seed(cfg.CE_SEED)
    random.seed(cfg.CE_SEED)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.CE_SEED)
        torch.backends.cudnn.benchmark = False
    data = _data(cfg.BASE_DATA_SEED, str(device), cfg.SEED_NAMESPACE)
    model = K8PilotResidualUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)
    start_step, best_nmse = 0, math.inf
    latest_path = run_dir / "checkpoint_latest.pt"
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {latest_path}")
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        if payload.get("format") != "twc_v1_k8_ce_checkpoint_v1" or payload.get("config") != _config():
            raise ValueError("Checkpoint format/config differs from the frozen CE recipe")
        if payload.get("data_protocol") != data.protocol:
            raise ValueError("Checkpoint data protocol differs from this run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        best_nmse = float(payload["best_validation_nmse"])
        _restore_rng(payload["rng"], device)
        _truncate_jsonl(run_dir / "train.jsonl", start_step)
        _truncate_jsonl(run_dir / "validation_summary.jsonl", start_step)
        _truncate_validation_csv(run_dir / "validation.csv", start_step)

    validation_data = _data(cfg.BASE_DATA_SEED, str(device), cfg.SEED_NAMESPACE)
    validation_batches = args.validation_batches or cfg.VALIDATION_BATCHES
    manifest = {
        "run_id": run_dir.name, "status": "running", "stage": "A_UL_CE",
        "seed": cfg.CE_SEED, "device": str(device), "config": _config(),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        "starting_step": start_step, "target_updates": cfg.UPDATES,
        "optimizer_updates": start_step, "source_checkpoint": None,
        "data_protocol": data.protocol,
        "validation_protocol": validation_data.protocol,
        "validation_batches_per_epoch": validation_batches,
    }
    _json_write(run_dir / "manifest.json", manifest)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.monotonic()
    for step in range(start_step + 1, cfg.UPDATES + 1):
        sample = data.sample("ce_train", step)
        H_ul = sample["H_ul"].to(device)
        gamma, ep = _condition(step - 1)
        Y, H_ls = _make_observation(H_ul, gamma, ep,
            cfg.CE_SEED + 10_000 + step)
        prediction = model(Y, _pilots(H_ul.shape[0], device=device), H_ls,
            pilot_energy_per_ue=ep,
            noise_variance=feedback_noise_variance(gamma))
        loss = _nmse_per_link(prediction, H_ul).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite CE loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"Nonfinite CE gradient at update {step}")
        optimizer.step()
        train_record = {"step": step, "gamma_db": gamma, "pilot_energy_per_ue": ep,
            "N0": feedback_noise_variance(gamma), "loss_nmse": float(loss.detach().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sample_fingerprint": sample["sample_fingerprint"],
            "elapsed_seconds": time.monotonic() - start_time}
        with (run_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(train_record, sort_keys=True) + "\n")

        validate_now = step == 1 or step % cfg.VALIDATION_EVERY == 0 or step == cfg.UPDATES
        if validate_now:
            rows = _evaluate(model, validation_data, device, validation_batches,
                             "validation", cfg.CE_SEED, step)
            _append_csv(run_dir / "validation.csv", rows)
            val_mean = _mean_metric(rows)
            summary = {"step": step, "mean_per_link_nmse": val_mean,
                "mean_per_link_nmse_db": 10 * math.log10(max(val_mean, 1e-20)),
                "LS_mean_per_link_nmse": float(np.mean([r["LS_mean_per_link_nmse"] for r in rows])),
                "validation_samples": sum(r["sample_count"] for r in rows),
                "elapsed_seconds": time.monotonic() - start_time,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            with (run_dir / "validation_summary.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")
            if val_mean < best_nmse:
                best_nmse = val_mean
                _checkpoint(run_dir / "checkpoint_best.pt", model, optimizer,
                            step, best_nmse, data, device)
        if (step % cfg.SAVE_EVERY == 0 or step == cfg.UPDATES
                or args.max_steps and step >= args.max_steps):
            _checkpoint(latest_path, model, optimizer, step, best_nmse, data, device)
        if validate_now:
            manifest.update(status="running", optimizer_updates=step,
                elapsed_seconds=time.monotonic() - start_time,
                latest_checkpoint=str(latest_path.resolve()) if latest_path.exists() else None,
                last_validation_nmse=val_mean,
                peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
            _json_write(run_dir / "manifest.json", manifest)
            eta = ((time.monotonic() - start_time) / max(step - start_step, 1)
                   * max(cfg.UPDATES - step, 0))
            print(json.dumps({"event": "validation", "step": step,
                "target_updates": cfg.UPDATES, "validation_nmse": val_mean,
                "LS_nmse": summary["LS_mean_per_link_nmse"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "estimated_remaining_seconds": eta,
                "peak_memory_bytes": summary["peak_memory_bytes"]}, sort_keys=True), flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    final_step = min(cfg.UPDATES, args.max_steps) if args.max_steps else cfg.UPDATES
    manifest.update(status="completed" if final_step >= cfg.UPDATES else "paused",
        optimizer_updates=final_step, elapsed_seconds=time.monotonic() - start_time,
        peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        latest_checkpoint=str(latest_path.resolve()) if latest_path.exists() else None,
        best_checkpoint=str((run_dir / "checkpoint_best.pt").resolve())
            if (run_dir / "checkpoint_best.pt").exists() else None,
        last_validation_nmse=best_nmse)
    _json_write(run_dir / "manifest.json", manifest)
    return manifest


def calibrate(args):
    run_dir = args.run_dir.resolve()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "checkpoint_best.pt"
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format") != "twc_v1_k8_ce_checkpoint_v1":
        raise ValueError("Unexpected CE checkpoint format")
    model = K8PilotResidualUNet().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    calibration_data = _data(cfg.BASE_DATA_SEED, str(device), cfg.SEED_NAMESPACE)
    if calibration_data.protocol != payload["data_protocol"]:
        raise ValueError("Calibration data protocol differs from checkpoint training protocol")

    records, variance_rows = [], []
    with torch.no_grad():
        calibration_batches = args.calibration_batches or cfg.CALIBRATION_BATCHES
        for gamma in GAMMAS:
            for ep in EPS:
                errors, normalized = [], []
                for batch_index in range(calibration_batches):
                    address_index = (GAMMAS.index(gamma) * len(EPS) + EPS.index(ep)) * calibration_batches + batch_index
                    sample = calibration_data.sample("ce_calibration", address_index)
                    H_ul = sample["H_ul"].to(device)
                    Y, H_ls = _make_observation(H_ul, gamma, ep,
                        cfg.CE_SEED + 900_000 + address_index)
                    estimate = model(Y, _pilots(H_ul.shape[0], device=device), H_ls,
                        pilot_energy_per_ue=ep,
                        noise_variance=feedback_noise_variance(gamma))
                    err = estimate - H_ul
                    errors.append(err.detach().cpu())
                    normalized.append(_nmse_per_link(estimate, H_ul).detach().cpu())
                    records.append({"gamma_db": gamma, "pilot_energy_per_ue": ep,
                        "batch_index": batch_index, "sample_fingerprint": sample["sample_fingerprint"]})
                error = torch.cat(errors, dim=0)
                nmse = torch.cat(normalized, dim=0)
                variance_rows.append({
                    "gamma_db": gamma, "pilot_energy_per_ue": ep,
                    "N0": feedback_noise_variance(gamma),
                        "samples": int(error.shape[0]),
                    "mean_per_link_nmse": float(nmse.mean()),
                    "mean_per_link_nmse_db": float(10 * torch.log10(nmse.mean().clamp_min(1e-20))),
                    "error_variance_complex": float(error.abs().square().mean()),
                    "error_real_variance": float(error.real.var(unbiased=False)),
                    "error_imag_variance": float(error.imag.var(unbiased=False)),
                })

    csv_path = run_dir / "calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variance_rows[0]))
        writer.writeheader()
        writer.writerows(variance_rows)
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    report = {
        "status": "completed", "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]), "calibration_role": "ce_calibration",
        "calibration_protocol": calibration_data.protocol,
        "independent_from_training_and_validation": True,
        "condition_rows": len(variance_rows), "sample_batches": len(records),
        "batches_per_condition": calibration_batches,
        "sample_manifest_sha256": fingerprint,
        "metrics": str(csv_path.resolve()), "records": records,
        "interpretation": "Independent CE error statistics for soft-recovery design; test data untouched.",
    }
    _json_write(run_dir / "calibration.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=0,
        help="CPU/CUDA smoke limit; production uses the frozen 5000-update recipe")
    parser.add_argument("--validation-batches", type=int, default=0,
        help="smoke override; production uses all 18 gamma/Ep validation conditions")
    parser.add_argument("--calibration-batches", type=int, default=0,
        help="smoke override; production uses the frozen independent calibration sample count")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.max_steps < 0 or args.cpu_threads < 1 or args.validation_batches < 0 or args.calibration_batches < 0:
        parser.error("step/batch limits must be nonnegative and CPU threads positive")
    if args.calibrate:
        calibrate(args)
    else:
        result = train(args)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
