#!/usr/bin/env python3
"""Train one Stage-B NMSE warmup arm (FDMA or Hybrid), with fixed validation.

This is an entry point only. It is never invoked by imports or preflight tools.
Run separate arms/seeds and run directories; resume is explicit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from configs import twc_v1_k8_codec_warmup_config as cfg

ROOT = Path(__file__).resolve().parent

def load_local(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_system = load_local("utils/twc_v1_k8_system.py", "_twc_v1_k8_codec_train_system")
_data = load_local("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_codec_train_data")
K8FeedbackChain, protocol = _system.K8FeedbackChain, _system.protocol
TWCv1K8ReplayableData = _data.TWCv1K8ReplayableData

RECIPE = {key: value for key, value in vars(cfg).items() if key.isupper()}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def standard_noise(batch, generator, device):
    z = torch.randn((batch, 32, 256, 2), generator=generator, device=device)
    return torch.view_as_complex(z.contiguous())


def build_chain(device):
    return K8FeedbackChain.from_artifacts(ROOT / cfg.CE_CHECKPOINT,
        ROOT / cfg.CE_CALIBRATION, device=device)


def validation(chain, data, arm, seed, batches):
    device = next(chain.codec_model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed + 6_000_000)
    rows = []
    gammas = cfg.GAMMAS
    eps = (cfg.PILOT_ENERGIES[1],)
    groups_to_test = (1, 2, 4) if arm == "hybrid" else (1,)
    chain.codec_model.eval()
    with torch.no_grad():
        for bi in range(batches):
            sample = data.sample("validation", bi)
            Hdl, Hul = sample["H_dl"].to(device), sample["H_ul"].to(device)
            for gamma in gammas:
                for ep in eps:
                    for groups in groups_to_test:
                        if arm == "hybrid":
                            layout = protocol.default_hybrid_layout(groups)
                            ratio = cfg.ENERGY_RATIOS[(bi + int(gamma)) % len(cfg.ENERGY_RATIOS)]
                            data_e = 64.0 - ep
                            split = protocol.EnergySplit(ep, data_e * ratio[0] / sum(ratio),
                                data_e * ratio[1] / sum(ratio))
                        else:
                            layout, split = protocol.FDMALayout(), None
                        output = chain(Hdl, Hul, feedback_gamma_db=gamma,
                            pilot_energy_per_ue=ep, pilot_standard_noise=standard_noise(len(Hdl), gen, device),
                            data_standard_noise=standard_noise(len(Hdl), gen, device),
                            layout=layout, energy_split=split)
                        result = chain.metrics(output, Hdl)
                        rows.append({"validation_batch": bi, "gamma_db": gamma,
                            "pilot_energy_per_ue": ep, "group_count": groups,
                            "layout": arm, "dl_nmse": float(result["dl_nmse_per_user"].mean()),
                            "downlink_sum_rate_bps_hz": float(result["sum_rate_bps_hz"].mean()),
                            "loss_stage": "NMSE_warmup", "loss_weight_nmse": 1.0,
                            "loss_weight_rate": 0.0, "learning_rate": None,
                            "sample_fingerprint": sample["sample_fingerprint"]})
    chain.codec_model.train()
    return rows


def run(args):
    arm = args.arm
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_dir = args.run_dir.resolve()
    latest = run_dir / "checkpoint_latest.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Run directory is nonempty; pass --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.cpu_threads)
    batch_sizes = {"ce_train": cfg.BATCH_SIZE, "codec_train": cfg.BATCH_SIZE,
        "ce_calibration": cfg.BATCH_SIZE, "validation": cfg.BATCH_SIZE, "test": 1}
    data = TWCv1K8ReplayableData(base_seed=cfg.BASE_DATA_SEED, batch_sizes=batch_sizes,
        device=str(device), cdl_model=cfg.CDL_MODEL, speed=cfg.SPEED_MPS,
        seed_namespace=cfg.SEED_NAMESPACE)
    chain = build_chain(device)
    ce_step = 5000
    model = chain.codec_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)
    start = 0
    if args.resume:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        if ckpt.get("format") != "twc_v1_k8_codec_warmup_v1" or ckpt.get("arm") != arm or ckpt.get("seed") != seed:
            raise ValueError("Checkpoint arm/seed/format mismatch")
        if ckpt.get("data_protocol") != data.protocol or ckpt.get("config") != RECIPE:
            raise ValueError("Checkpoint recipe/data protocol mismatch")
        model.load_state_dict(ckpt["codec_model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start = int(ckpt["step"])
        torch.set_rng_state(ckpt["torch_cpu_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng"])
    manifest = {"run_id": run_dir.name, "stage": "B_CODEC_NMSE_WARMUP", "status": "running",
        "arm": arm, "seed": seed, "device": str(device), "source_ce_checkpoint_step": ce_step,
        "optimizer_updates": start, "target_updates": cfg.UPDATES, "data_protocol": data.protocol,
        "code_commit": os.popen("git rev-parse HEAD").read().strip(),
        "uncommitted_sources": "see git status; checkpoint records source commit only",
        "config": RECIPE}
    atomic_json(run_dir / "manifest.json", manifest)
    model.train()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for step in range(start + 1, cfg.UPDATES + 1):
        sample = data.sample("codec_train", step)
        Hdl, Hul = sample["H_dl"].to(device), sample["H_ul"].to(device)
        gamma = cfg.GAMMAS[(step - 1) % len(cfg.GAMMAS)]
        ep = cfg.PILOT_ENERGIES[((step - 1) // len(cfg.GAMMAS)) % len(cfg.PILOT_ENERGIES)]
        if arm == "hybrid":
            groups = (1, 2, 4)[((step - 1) // (len(cfg.GAMMAS) * len(cfg.PILOT_ENERGIES))) % 3]
            layout = protocol.default_hybrid_layout(groups)
            ratio = cfg.ENERGY_RATIOS[((step - 1) // 2) % 2]
            available = 64.0 - ep; total_ratio = sum(ratio)
            split = protocol.EnergySplit(ep, available * ratio[0] / total_ratio,
                                          available * ratio[1] / total_ratio)
        else:
            groups, layout, split = 1, protocol.FDMALayout(), None
        noise_gen = torch.Generator(device=device).manual_seed(seed * 10_000_000 + step)
        output = chain(Hdl, Hul, feedback_gamma_db=gamma, pilot_energy_per_ue=ep,
            pilot_standard_noise=standard_noise(len(Hdl), noise_gen, device),
            data_standard_noise=standard_noise(len(Hdl), noise_gen, device),
            layout=layout, energy_split=split)
        loss = chain.metrics(output, Hdl)["dl_nmse_per_user"].mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        if not bool(torch.isfinite(grad)): raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % cfg.VALIDATION_EVERY == 0 or step == cfg.UPDATES:
            rows = validation(chain, data, arm, seed, args.validation_batches or cfg.VALIDATION_BATCHES)
            with (run_dir / "validation.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "elapsed_seconds": time.monotonic()-started,
                    "learning_rate": optimizer.param_groups[0]["lr"], "loss_stage": "NMSE_warmup",
                    "validation": rows}, sort_keys=True) + "\n")
            payload = {"format": "twc_v1_k8_codec_warmup_v1", "arm": arm, "seed": seed,
                "step": step, "codec_model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "data_protocol": data.protocol, "config": RECIPE, "ce_checkpoint_step": ce_step}
            tmp = latest.with_suffix(".pt.tmp"); torch.save(payload, tmp); os.replace(tmp, latest)
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            elapsed = time.monotonic() - started
            manifest.update(optimizer_updates=step, elapsed_seconds=elapsed,
                peak_cuda_allocated_bytes=peak, latest_checkpoint=str(latest), status="running")
            atomic_json(run_dir / "manifest.json", manifest)
            print(json.dumps({"event": "validation", "step": step, "target": cfg.UPDATES,
                "elapsed_seconds": elapsed, "estimated_remaining_seconds": elapsed / step * (cfg.UPDATES-step),
                "peak_cuda_allocated_bytes": peak, "validation": rows}, sort_keys=True), flush=True)
    manifest.update(status="completed", optimizer_updates=cfg.UPDATES,
        elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    atomic_json(run_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("fdma", "hybrid"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-batches", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
