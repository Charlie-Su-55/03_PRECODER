#!/usr/bin/env python3
"""Stage-C matched terminal branch trainer (NMSE control or joint objective).

Training starts only from a completed 10k Stage-B warmup. Joint-loss scaling
must be selected using validation diagnostics and supplied explicitly; this
entry point never guesses lambda_target or rate_reference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from configs import twc_v1_k8_codec_stage_c_config as cfg

ROOT = Path(__file__).resolve().parent


def load_local(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_system = load_local("utils/twc_v1_k8_system.py", "_twc_v1_k8_stage_c_system")
_data = load_local("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_stage_c_data")
K8FeedbackChain, protocol = _system.K8FeedbackChain, _system.protocol
TWCv1K8ReplayableData = _data.TWCv1K8ReplayableData
RECIPE = {key: value for key, value in vars(cfg).items() if key.isupper()}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pt.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def standard_noise(batch, generator, device):
    value = torch.randn((batch, 32, 256, 2), generator=generator, device=device)
    return torch.view_as_complex(value.contiguous())


def make_data(device):
    sizes = {"ce_train": cfg.BATCH_SIZE, "codec_train": cfg.BATCH_SIZE,
        "ce_calibration": cfg.BATCH_SIZE, "validation": cfg.BATCH_SIZE, "test": 1}
    return TWCv1K8ReplayableData(base_seed=cfg.BASE_DATA_SEED, batch_sizes=sizes,
        device=str(device), cdl_model=cfg.CDL_MODEL, speed=cfg.SPEED_MPS,
        seed_namespace=cfg.SEED_NAMESPACE)


def validate(chain, data, arm, seed, batches):
    device = next(chain.codec_model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed + 6_000_000)
    groups_to_test = cfg.GROUP_COUNTS if arm == "hybrid" else (1,)
    rows = []
    chain.codec_model.eval()
    with torch.no_grad():
        for batch_index in range(batches):
            sample = data.sample("validation", batch_index)
            Hdl, Hul = sample["H_dl"].to(device), sample["H_ul"].to(device)
            for gamma in cfg.GAMMAS:
                ep = 16.0
                for groups in groups_to_test:
                    if arm == "hybrid":
                        layout = protocol.default_hybrid_layout(groups)
                        ratio = cfg.ENERGY_RATIOS[batch_index % len(cfg.ENERGY_RATIOS)]
                        available = 64.0 - ep
                        split = protocol.EnergySplit(ep,
                            available * ratio[0] / sum(ratio),
                            available * ratio[1] / sum(ratio))
                    else:
                        layout, split = protocol.FDMALayout(), None
                    output = chain(Hdl, Hul, feedback_gamma_db=gamma,
                        pilot_energy_per_ue=ep,
                        pilot_standard_noise=standard_noise(len(Hdl), generator, device),
                        data_standard_noise=standard_noise(len(Hdl), generator, device),
                        layout=layout, energy_split=split)
                    metrics = chain.metrics(output, Hdl)
                    rows.append({"validation_batch": batch_index, "gamma_db": gamma,
                        "pilot_energy_per_ue": ep, "group_count": groups, "layout": arm,
                        "dl_nmse": float(metrics["dl_nmse_per_user"].mean()),
                        "downlink_sum_rate_bps_hz": float(metrics["sum_rate_bps_hz"].mean()),
                        "loss_branch": None, "learning_rate": None,
                        "sample_fingerprint": sample["sample_fingerprint"]})
    chain.codec_model.train()
    return rows


def state_rng(device):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda": result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state, device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def main_run(args):
    branch = args.branch
    warmup_path = args.warmup_checkpoint.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if branch == "nmse":
        if args.lambda_target is not None or args.rate_reference is not None:
            raise ValueError("The NMSE control branch does not take rate-loss parameters")
        lambda_target, rate_reference = 0.0, None
    else:
        if (args.lambda_target is None or not math.isfinite(args.lambda_target)
                or args.lambda_target <= 0 or args.rate_reference is None
                or not math.isfinite(args.rate_reference) or args.rate_reference <= 0):
            raise ValueError("Joint branch requires positive finite --lambda-target and --rate-reference")
        lambda_target, rate_reference = args.lambda_target, args.rate_reference

    warmup = torch.load(warmup_path, map_location=device, weights_only=False)
    if (warmup.get("format") != "twc_v1_k8_codec_warmup_v1"
            or warmup.get("step") != 10_000
            or warmup.get("arm") != args.arm
            or warmup.get("seed") != args.seed):
        raise ValueError("Stage C requires a matching, completed Stage-B 10,000-update checkpoint")

    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.cpu_threads)
    data = make_data(device)
    if data.protocol != warmup.get("data_protocol"):
        raise ValueError("Warmup and Stage C data protocols differ")
    chain = K8FeedbackChain.from_artifacts(ROOT / "runs/twc_v1_k8_ce/seed_20261105/checkpoint_latest.pt",
        ROOT / "runs/twc_v1_k8_ce/seed_20261105/calibration.csv", device=device)
    model = chain.codec_model
    model.load_state_dict(warmup["codec_model"], strict=True)

    run_dir = args.run_dir.resolve()
    latest = run_dir / "checkpoint_latest.pt"
    safe_logs = {"training.console.log"}
    artifacts = [p for p in run_dir.iterdir() if p.name not in safe_logs] if run_dir.exists() else []
    if artifacts and not args.resume:
        raise FileExistsError(f"Run directory contains artifacts; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Both matched branches start with fresh, identical optimizer recipes.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    start = 0
    best_nmse = math.inf
    if args.resume:
        state = torch.load(latest, map_location=device, weights_only=False)
        expected = (state.get("format") == "twc_v1_k8_codec_stage_c_v1"
            and state.get("branch") == branch and state.get("arm") == args.arm
            and state.get("seed") == seed and state.get("warmup_checkpoint") == str(warmup_path))
        if not expected or state.get("recipe") != RECIPE:
            raise ValueError("Stage C resume checkpoint/recipe mismatch")
        if state.get("lambda_target") != lambda_target or state.get("rate_reference") != rate_reference:
            raise ValueError("Stage C resume objective scaling differs")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start = int(state["stage_updates"]); best_nmse = float(state["best_validation_nmse"])
        restore_rng(state["rng"], device)

    manifest = {"run_id": run_dir.name, "stage": "C_MATCHED_TERMINAL",
        "status": "running", "arm": args.arm, "branch": branch, "seed": seed,
        "device": str(device), "source_warmup_checkpoint": str(warmup_path),
        "source_warmup_updates": 10_000, "source_cumulative_updates": 10_000 + start,
        "stage_updates": start, "target_stage_updates": cfg.UPDATES,
        "lambda_target": lambda_target, "rate_reference_bps_hz": rate_reference,
        "lambda_ramp_updates": cfg.RATE_RAMP_UPDATES if branch == "joint" else 0,
        "scheduler": "constant_lambda_lr_over_20k (identical for matched branches)",
        "recipe": RECIPE, "data_protocol": data.protocol,
        "code_commit": os.popen("git rev-parse HEAD").read().strip(),
        "test_role_used": False}
    atomic_json(run_dir / "manifest.json", manifest)
    model.train()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    start_time = time.monotonic()
    for stage_step in range(start + 1, cfg.UPDATES + 1):
        sample = data.sample("codec_train", stage_step)
        Hdl, Hul = sample["H_dl"].to(device), sample["H_ul"].to(device)
        gamma = cfg.GAMMAS[(stage_step - 1) % len(cfg.GAMMAS)]
        ep = cfg.PILOT_ENERGIES[((stage_step - 1) // len(cfg.GAMMAS)) % len(cfg.PILOT_ENERGIES)]
        if args.arm == "hybrid":
            groups = cfg.GROUP_COUNTS[((stage_step - 1) // (len(cfg.GAMMAS) * len(cfg.PILOT_ENERGIES))) % len(cfg.GROUP_COUNTS)]
            layout = protocol.default_hybrid_layout(groups)
            ratio = cfg.ENERGY_RATIOS[((stage_step - 1) // 2) % len(cfg.ENERGY_RATIOS)]
            available = 64.0 - ep
            split = protocol.EnergySplit(ep, available * ratio[0] / sum(ratio),
                                          available * ratio[1] / sum(ratio))
        else:
            groups, layout, split = 1, protocol.FDMALayout(), None
        noise_gen = torch.Generator(device=device).manual_seed(seed * 10_000_000 + stage_step)
        output = chain(Hdl, Hul, feedback_gamma_db=gamma, pilot_energy_per_ue=ep,
            pilot_standard_noise=standard_noise(len(Hdl), noise_gen, device),
            data_standard_noise=standard_noise(len(Hdl), noise_gen, device),
            layout=layout, energy_split=split)
        metrics = chain.metrics(output, Hdl)
        nmse = metrics["dl_nmse_per_user"].mean()
        rate = metrics["sum_rate_bps_hz"].mean()
        if branch == "joint":
            ramp = min(max((stage_step - 1) / max(cfg.RATE_RAMP_UPDATES - 1, 1), 0.0), 1.0)
            lambda_now = lambda_target * ramp
            loss = nmse - lambda_now * rate / rate_reference
        else:
            lambda_now, loss = 0.0, nmse
        if not bool(torch.isfinite(loss) & torch.isfinite(nmse) & torch.isfinite(rate)):
            raise FloatingPointError(f"Nonfinite Stage C metric/loss at update {stage_step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        if not bool(torch.isfinite(grad)):
            raise FloatingPointError(f"Nonfinite Stage C gradient at update {stage_step}")
        optimizer.step(); scheduler.step()

        if stage_step == 1 or stage_step % cfg.VALIDATION_EVERY == 0 or stage_step == cfg.UPDATES:
            rows = validate(chain, data, args.arm, seed,
                args.validation_batches or cfg.VALIDATION_BATCHES)
            mean_val_nmse = float(np.mean([row["dl_nmse"] for row in rows]))
            for row in rows:
                row.update(branch=branch, loss_stage="NMSE_control" if branch == "nmse" else "NMSE_plus_rate",
                    loss_weight_nmse=1.0, loss_weight_rate=lambda_now,
                    rate_reference_bps_hz=rate_reference,
                    learning_rate=optimizer.param_groups[0]["lr"])
            validation_record = {"stage_update": stage_step,
                "source_cumulative_updates": 10_000 + stage_step,
                "elapsed_seconds": time.monotonic() - start_time,
                "train_loss": float(loss.detach()), "train_nmse": float(nmse.detach()),
                "train_rate_bps_hz": float(rate.detach()), "lambda_now": lambda_now,
                "lambda_target": lambda_target, "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": rows}
            with (run_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation_record, sort_keys=True) + "\n")
            if mean_val_nmse < best_nmse:
                best_nmse = mean_val_nmse
                atomic_torch_save(run_dir / "checkpoint_best_nmse.pt", {
                    "format": "twc_v1_k8_codec_stage_c_v1", "branch": branch,
                    "arm": args.arm, "seed": seed, "stage_updates": stage_step,
                    "model": model.state_dict(), "warmup_checkpoint": str(warmup_path),
                    "recipe": RECIPE, "lambda_target": lambda_target,
                    "rate_reference": rate_reference, "best_validation_nmse": best_nmse})
            state = {"format": "twc_v1_k8_codec_stage_c_v1", "branch": branch,
                "arm": args.arm, "seed": seed, "stage_updates": stage_step,
                "source_cumulative_updates": 10_000 + stage_step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": state_rng(device),
                "warmup_checkpoint": str(warmup_path), "data_protocol": data.protocol,
                "recipe": RECIPE, "lambda_target": lambda_target,
                "rate_reference": rate_reference, "best_validation_nmse": best_nmse}
            atomic_torch_save(latest, state)
            elapsed = time.monotonic() - start_time
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            manifest.update(stage_updates=stage_step,
                source_cumulative_updates=10_000 + stage_step, elapsed_seconds=elapsed,
                peak_cuda_allocated_bytes=peak, best_validation_nmse=best_nmse,
                latest_checkpoint=str(latest), status="running")
            atomic_json(run_dir / "manifest.json", manifest)
            print(json.dumps({"event": "validation", "stage_update": stage_step,
                "source_cumulative_updates": 10_000 + stage_step,
                "target_stage_updates": cfg.UPDATES, "lambda_now": lambda_now,
                "train_nmse": validation_record["train_nmse"],
                "train_rate_bps_hz": validation_record["train_rate_bps_hz"],
                "mean_validation_nmse": mean_val_nmse, "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": elapsed / stage_step * (cfg.UPDATES-stage_step),
                "peak_cuda_allocated_bytes": peak}, sort_keys=True), flush=True)

    manifest.update(status="completed", stage_updates=cfg.UPDATES,
        source_cumulative_updates=10_000 + cfg.UPDATES,
        elapsed_seconds=time.monotonic() - start_time,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    atomic_json(run_dir / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("fdma", "hybrid"), required=True)
    parser.add_argument("--branch", choices=("nmse", "joint"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-target", type=float)
    parser.add_argument("--rate-reference", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-batches", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = main_run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
