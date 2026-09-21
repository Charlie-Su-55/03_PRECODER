#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified trainer for CsiNet+-based and Swin-based FDMA baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from configs.baseline_config import BaselineConfig, list_scenarios, make_config
from models.csinet_plus_precoder import CsiNetPlus_E2E_Precoder
from models.swin_precoder import Swin_E2E_Precoder
from utils import CDLChannelGenerator_Feedback, setup_gpu


MODEL_REGISTRY = {
    "csinet": CsiNetPlus_E2E_Precoder,
    "swin": Swin_E2E_Precoder,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]
    return state


def _cpu_uint8_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_uint8_contiguous(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([
            _cpu_uint8_contiguous(x) for x in state["cuda"]
        ])


@contextmanager
def temporary_seed(seed: int):
    state = capture_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def config_fingerprint(cfg: BaselineConfig) -> str:
    payload = cfg.to_dict().copy()
    for key in (
        "OUTPUT_ROOT", "SAVE_DIR", "BEST_PATH", "SAVE_PATH",
        "LOG_DIR", "METRICS_PATH", "STATUS_PATH",
    ):
        payload.pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def optimizer_to(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class SumRateLoss(nn.Module):
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        W: torch.Tensor,
        H_dl_true: torch.Tensor,
        noise_power: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_eff = torch.einsum("bkns,bnjs->bkjs", H_dl_true.conj(), W)
        power = h_eff.abs().square()
        signal = torch.diagonal(power, dim1=1, dim2=2).permute(0, 2, 1)
        interference = power.sum(dim=2) - signal
        sinr = signal / (interference + noise_power + self.eps)
        rate = torch.log2(1.0 + sinr)
        avg_rate = rate.sum(dim=1).mean()
        return -avg_rate, avg_rate


def normalized_mse(H_hat: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
    error = (H_hat - H_true).abs().square().sum(dim=(1, 2, 3))
    reference = H_true.abs().square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return (error / reference).mean()


def nmse_db(H_hat: torch.Tensor, H_true: torch.Tensor) -> float:
    error = (H_hat - H_true).abs().square().sum(dim=(1, 2, 3))
    reference = H_true.abs().square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
    values = 10.0 * torch.log10((error / reference).clamp_min(1e-12))
    return float(values.mean().item())


def get_training_snr(step: int, cfg: BaselineConfig) -> float:
    rng = np.random.default_rng(cfg.SEED + 20_000_000 + step)
    if step <= cfg.SNR_STAGE1_END:
        low, high = cfg.STAGE1_SNR
    elif step <= cfg.SNR_STAGE2_END:
        low, high = cfg.STAGE2_SNR
    else:
        low, high = cfg.STAGE3_SNR
    return float(rng.uniform(low, high))


def build_channel_generator(cfg: BaselineConfig, device: torch.device):
    generator = CDLChannelGenerator_Feedback(
        Nt=cfg.ANTENNAS,
        Nsc=cfg.SUBCARRIERS,
        carrier_freq=cfg.CARRIER_FREQ,
        speed=cfg.SPEED,
        num_users=cfg.K_USERS,
    )
    if hasattr(generator, "to"):
        generator = generator.to(device)
    return generator


def generate_channel_batch(
    generator,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with temporary_seed(seed):
        with torch.no_grad():
            H_dl, H_ul = generator.generate_batch_data(batch_size)
    return H_dl.to(device), H_ul.to(device)


@torch.no_grad()
def validate(
    model: nn.Module,
    H_dl_val: torch.Tensor,
    H_ul_val: torch.Tensor,
    criterion: SumRateLoss,
    cfg: BaselineConfig,
) -> dict:
    model.eval()
    rates: Dict[str, float] = {}
    nmse_values: Dict[str, float] = {}

    for snr_db in cfg.VAL_SNR_LIST:
        feedback_seed = (
            cfg.SEED + 30_000_000 + cfg.K_USERS*100_000
            + cfg.D_TOT*100 + int(snr_db)*10
        )
        with temporary_seed(feedback_seed):
            output = model(
                H_dl_val,
                H_ul_val,
                torch.tensor(float(snr_db), dtype=torch.float32, device=H_dl_val.device),
            )
        W = output["precoder"]
        H_hat = output["channel_surrogate"]
        _, rate = criterion(W, H_dl_val, 10.0 ** (-float(snr_db) / 10.0))
        rates[str(snr_db)] = float(rate.item())
        nmse_values[str(snr_db)] = nmse_db(H_hat, H_dl_val)

    mean_rate = float(np.mean(list(rates.values())))
    mean_nmse_db = float(np.mean(list(nmse_values.values())))
    if cfg.OBJECTIVE == "task":
        selection_score = mean_rate
        selection_name = "mean_rate"
    else:
        selection_score = -mean_nmse_db
        selection_name = "negative_mean_nmse_db"

    return {
        "rates": rates,
        "nmse_db": nmse_values,
        "mean_rate": mean_rate,
        "mean_nmse_db": mean_nmse_db,
        "selection_score": selection_score,
        "selection_name": selection_name,
    }


def format_validation(step: int, result: dict, cfg: BaselineConfig) -> str:
    rate_line = " | ".join(
        f"{snr}dB:{result['rates'][str(snr)]:.2f}"
        for snr in cfg.VAL_SNR_LIST
    )
    return (
        f"Step {step} | Validation\n"
        f"    Sum rate: {rate_line} | mean:{result['mean_rate']:.3f}\n"
        f"    Mean NMSE: {result['mean_nmse_db']:.2f} dB\n"
        f"    Selection [{result['selection_name']}]: "
        f"{result['selection_score']:.3f}"
    )


def checkpoint_payload(
    cfg: BaselineConfig,
    step: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    best_score: float,
    validation: dict | None,
) -> dict:
    return {
        "step": step,
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_score": best_score,
        "validation": validation,
        "config": cfg.to_dict(),
        "config_fingerprint": config_fingerprint(cfg),
        "rng_state": capture_rng_state(),
    }


def load_resume(
    cfg: BaselineConfig,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
    expected = config_fingerprint(cfg)
    actual = checkpoint.get("config_fingerprint")
    if actual is not None and actual != expected:
        raise RuntimeError(
            f"Config fingerprint mismatch: checkpoint={actual}, current={expected}"
        )
    model.load_state_dict(checkpoint["state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint["step"]) + 1, float(checkpoint.get("best_score", -float("inf")))


def train_one(
    cfg: BaselineConfig,
    *,
    stop_after: int | None,
    no_resume: bool,
    skip_existing: bool,
) -> None:
    save_dir = Path(cfg.SAVE_DIR)
    log_dir = Path(cfg.LOG_DIR)
    best_path = Path(cfg.BEST_PATH)
    latest_path = Path(cfg.SAVE_PATH)
    status_path = Path(cfg.STATUS_PATH)

    if skip_existing and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            print(f"[Skip] {cfg.EXP_NAME}")
            return

    if no_resume and latest_path.exists():
        raise RuntimeError(
            f"{latest_path} already exists. Use a new --output-root or remove that experiment directory."
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "config.json", cfg.to_dict())
    write_json(status_path, {"state": "running", "experiment": cfg.EXP_NAME})

    set_seed(cfg.SEED)
    torch.set_float32_matmul_precision("high")
    device = setup_gpu()
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("CUDA is unavailable. Baseline training is aborted.")

    model = MODEL_REGISTRY[cfg.MODEL_NAME](cfg).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 100)
    print(f"[Experiment] {cfg.EXP_NAME}")
    print(f"[Model]      {cfg.MODEL_NAME}")
    print(f"[Objective]  {cfg.OBJECTIVE}")
    print(
        f"[Scenario]   K={cfg.K_USERS}, Nsc={cfg.SUBCARRIERS}, "
        f"Dtot={cfg.D_TOT}, Df={cfg.D_F}"
    )
    print(
        f"[Recipe]     steps={cfg.TOTAL_STEPS}, batch={cfg.BATCH_SIZE}, "
        f"lr={cfg.LR}, warmup={cfg.WARMUP_PCT:.2f}"
    )
    print(f"[Model]      trainable parameters={parameter_count/1e6:.3f} M")
    print(f"[Checkpoint] {cfg.SAVE_DIR}")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.LR,
        total_steps=cfg.TOTAL_STEPS,
        pct_start=cfg.WARMUP_PCT,
        anneal_strategy="cos",
    )

    start_step = 1
    best_score = -float("inf")
    if latest_path.exists() and not no_resume:
        start_step, best_score = load_resume(
            cfg, model, optimizer, scheduler, device
        )
        print(f"[Resume] step={start_step}, best_score={best_score:.4f}")

    generator = build_channel_generator(cfg, device)
    H_dl_val, H_ul_val = generate_channel_batch(
        generator, cfg.VAL_BATCH, device, cfg.SEED + 500_000
    )

    criterion = SumRateLoss()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    target_step = cfg.TOTAL_STEPS if stop_after is None else min(cfg.TOTAL_STEPS, int(stop_after))
    if start_step > target_step:
        print(f"[Nothing to do] start_step={start_step}, target_step={target_step}")
        return

    progress = tqdm(
        range(start_step, target_step + 1),
        desc=f"{cfg.MODEL_NAME}-{cfg.SCENARIO_NAME}",
        dynamic_ncols=True,
    )
    last_validation = None

    for step in progress:
        H_dl, H_ul = generate_channel_batch(
            generator, cfg.BATCH_SIZE, device, cfg.SEED + 10_000_000 + step
        )
        snr_db = get_training_snr(step, cfg)
        snr_tensor = torch.tensor(snr_db, dtype=torch.float32, device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with temporary_seed(cfg.SEED + 40_000_000 + step):
            output = model(H_dl, H_ul, snr_tensor)

        W = output["precoder"]
        H_hat = output["channel_surrogate"]
        rate_loss, train_rate = criterion(W, H_dl, 10.0 ** (-snr_db / 10.0))
        recon_loss = normalized_mse(H_hat, H_dl)
        loss = rate_loss if cfg.OBJECTIVE == "task" else recon_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % cfg.LOG_INTERVAL == 0:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                SR=f"{train_rate.item():.2f}",
                NMSE=f"{10*np.log10(max(recon_loss.item(), 1e-12)):.2f}",
                SNR=f"{snr_db:.1f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        should_validate = step % cfg.VAL_INTERVAL == 0 or step == target_step
        if not should_validate:
            continue

        validation = validate(model, H_dl_val, H_ul_val, criterion, cfg)
        last_validation = validation
        progress.write(format_validation(step, validation, cfg))

        append_jsonl(cfg.METRICS_PATH, {
            "step": step,
            "experiment": cfg.EXP_NAME,
            "objective": cfg.OBJECTIVE,
            **validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "grad_norm": float(
                grad_norm.detach().cpu().item()
                if torch.is_tensor(grad_norm) else grad_norm
            ),
        })

        payload = checkpoint_payload(
            cfg, step, model, optimizer, scheduler, best_score, validation
        )
        torch.save(payload, latest_path)

        if validation["selection_score"] > best_score:
            best_score = validation["selection_score"]
            payload["best_score"] = best_score
            torch.save(payload, best_path)
            progress.write(
                f"New best {validation['selection_name']}: "
                f"{best_score:.3f}; saving best.pth"
            )

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(
        f"[GPU Memory] peak allocated={peak_allocated:.2f} GB, "
        f"peak reserved={peak_reserved:.2f} GB"
    )

    state = "completed" if target_step >= cfg.TOTAL_STEPS else "paused"
    write_json(status_path, {
        "state": state,
        "experiment": cfg.EXP_NAME,
        "step": target_step,
        "total_steps": cfg.TOTAL_STEPS,
        "best_score": best_score,
        "last_validation": last_validation,
    })
    print(f"[{state.capitalize()}] {cfg.EXP_NAME}; best_score={best_score:.3f}")


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="csinet,swin")
    parser.add_argument("--scenarios", default="k8_n256_d256")
    parser.add_argument(
        "--objective", choices=["task", "reconstruction"], default="task"
    )
    parser.add_argument("--output-root", default="runs_baselines")
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()

    if args.list_scenarios:
        print("\n".join(list_scenarios()))
        return

    models = parse_csv(args.models)
    scenarios = parse_csv(args.scenarios)
    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {', '.join(MODEL_REGISTRY)}"
            )

    configs = [
        make_config(
            model_name,
            scenario_name,
            objective=args.objective,
            output_root=args.output_root,
        )
        for scenario_name in scenarios
        for model_name in models
    ]

    print(f"[Suite] {len(configs)} experiment(s)")
    for index, cfg in enumerate(configs, start=1):
        print(
            f"[{index:02d}] {cfg.EXP_NAME} | "
            f"steps={cfg.TOTAL_STEPS}, batch={cfg.BATCH_SIZE}"
        )
    if args.dry_run:
        return

    failures = []
    for index, cfg in enumerate(configs, start=1):
        print(f"\n[Suite] Starting {index}/{len(configs)}: {cfg.EXP_NAME}")
        try:
            train_one(
                cfg,
                stop_after=args.stop_after,
                no_resume=args.no_resume,
                skip_existing=args.skip_existing,
            )
        except Exception as error:
            failures.append((cfg.EXP_NAME, repr(error)))
            print(f"[FAILED] {cfg.EXP_NAME}: {error}")
            traceback.print_exc()
            if args.fail_fast:
                break

    if failures:
        print("\nSuite finished with failures:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        raise SystemExit(1)

    print("\nSuite completed successfully.")


if __name__ == "__main__":
    main()