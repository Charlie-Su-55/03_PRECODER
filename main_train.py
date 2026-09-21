#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic training entry point for the Hybrid FDMA-AirComp experiment suites.

Examples
--------
# List all declared suites and resolved checkpoint names
python main_train.py --list-suites
python main_train.py --suite proposal_all --dry-run

# Train every adaptive proposal checkpoint for K={4,8,16}, Dfb={128,256}
python main_train.py --suite proposal_adaptive --skip-existing

# Train all per-allocation specialists sequentially
python main_train.py --suite proposal_specialists --skip-existing

# Restrict a suite without editing train_config.py
python main_train.py --suite proposal_all --only-k 8 --only-budget 256

# Split a large suite across two GPUs/processes
CUDA_VISIBLE_DEVICES=0 python main_train.py --suite proposal_specialists --num-shards 2 --shard-id 0
CUDA_VISIBLE_DEVICES=1 python main_train.py --suite proposal_specialists --num-shards 2 --shard-id 1

Notes
-----
The proposal trainer is fully implemented against the current interface:
    AdaptiveHybridPrecoder(cfg)
    model(H_dl, H_ul, snr, d_f, d_a, cfg_idx)

CsiNet+ and Swin-CFNet parameters and experiment names are resolved by
train_config.py. Their trainer adapters require the exact model constructors and
forward signatures from the project and are therefore intentionally not guessed
here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from configs.train_config import TrainConfig, get_suite, list_suites


# =============================================================================
# 1. General utilities
# =============================================================================


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
        state["cuda"] = [
            cuda_state.cpu().clone()
            for cuda_state in torch.cuda.get_rng_state_all()
        ]

    return state


def _to_cpu_byte_tensor(value: Any) -> torch.Tensor:
    """Convert a serialized RNG state to a contiguous CPU ByteTensor."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.uint8)

    return (
        value.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore Python, NumPy, CPU Torch, and CUDA RNG states safely."""
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    cpu_rng_state = _to_cpu_byte_tensor(state["torch"])
    torch.set_rng_state(cpu_rng_state)

    if torch.cuda.is_available() and "cuda" in state:
        cuda_rng_states = [
            _to_cpu_byte_tensor(cuda_state)
            for cuda_state in state["cuda"]
        ]

        current_device_count = torch.cuda.device_count()

        if len(cuda_rng_states) == current_device_count:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        else:
            # More robust when a checkpoint was saved with a different
            # visible-GPU configuration.
            for device_idx, cuda_state in enumerate(
                cuda_rng_states[:current_device_count]
            ):
                torch.cuda.set_rng_state(cuda_state, device=device_idx)

@contextmanager
def temporary_seed(seed: int):
    """Use a deterministic local seed without changing the training RNG stream."""

    state = capture_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def config_fingerprint(cfg: TrainConfig) -> str:
    payload = json.dumps(
        cfg.to_dict(), sort_keys=True, ensure_ascii=False, default=json_default
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
    os.replace(tmp, path)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def allocation_index_for_step(step: int, n_allocations: int, seed: int) -> int:
    """Balanced deterministic allocation sampling.

    Every block of n_allocations steps visits each allocation exactly once in a
    seed-dependent shuffled order. No sampler state needs to be checkpointed.
    """

    if n_allocations <= 0:
        raise ValueError("n_allocations must be positive.")
    cycle, position = divmod(step - 1, n_allocations)
    order = list(range(n_allocations))
    random.Random(seed + cycle).shuffle(order)
    return order[position]


def get_snr(step: int, cfg: TrainConfig) -> float:
    if step <= cfg.STAGE1_STEPS:
        low, high = cfg.STAGE1_SNR
    elif step <= cfg.STAGE2_STEPS:
        low, high = cfg.STAGE2_SNR
    else:
        low, high = cfg.STAGE3_SNR
    return float(np.random.uniform(low, high))


# =============================================================================
# 2. Losses and validation
# =============================================================================


class SumRateLoss(nn.Module):
    def __init__(self, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        W: torch.Tensor,
        H_dl_true: torch.Tensor,
        noise_power: float | torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        h_eff = torch.einsum("bkns,bnjs->bkjs", H_dl_true.conj(), W)
        power = h_eff.abs().square()
        sig_pwr = torch.diagonal(power, dim1=1, dim2=2).permute(0, 2, 1)
        int_pwr = power.sum(dim=2) - sig_pwr
        sinr = sig_pwr / (int_pwr + noise_power + self.eps)
        rate = torch.log2(1.0 + sinr)
        avg_rate = rate.sum(dim=1).mean()
        return -avg_rate, float(avg_rate.detach().item())


def alignment_losses(
    H_hat: torch.Tensor,
    H_dl: torch.Tensor,
    dir_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-8
    H_hat_n = H_hat / (
        H_hat.abs().square().sum(dim=2, keepdim=True).sqrt() + eps
    )
    H_dl_n = H_dl / (
        H_dl.abs().square().sum(dim=2, keepdim=True).sqrt() + eps
    )
    inner = (H_hat_n * H_dl_n.conj()).sum(dim=2)
    if dir_mode == "abs":
        cos_sim = inner.abs().mean()
    elif dir_mode == "real":
        cos_sim = inner.real.mean()
    else:
        raise ValueError(f"Unsupported DIR_MODE: {dir_mode}")
    dir_loss = 1.0 - cos_sim
    mse_loss = torch.mean(torch.abs(H_hat - H_dl).square())
    return dir_loss, mse_loss, cos_sim


def extract_precoder(model_output: Any) -> torch.Tensor:
    """Support models returning W or tuples whose first element is W."""

    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise RuntimeError("Model returned an empty tuple/list.")
        return model_output[0]
    if not torch.is_tensor(model_output):
        raise TypeError(f"Unsupported model output type: {type(model_output)!r}")
    return model_output


def extract_training_outputs(model_output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(model_output, (tuple, list)) or len(model_output) < 2:
        raise RuntimeError(
            "Proposal training requires model(...) to return at least (W, H_hat)."
        )
    W, H_hat = model_output[0], model_output[1]
    if not torch.is_tensor(W) or not torch.is_tensor(H_hat):
        raise TypeError("W and H_hat must be torch tensors.")
    return W, H_hat


@torch.no_grad()
def validate_proposal(
    model: nn.Module,
    H_dl_val: torch.Tensor,
    H_ul_val: torch.Tensor,
    criterion: SumRateLoss,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[Tuple[int, int], float]:
    """Return {(allocation_index, snr_db): sum_rate}."""

    model.eval()
    table: Dict[Tuple[int, int], float] = {}
    for cfg_idx, (d_f, d_a) in enumerate(cfg.ALLOCATION_GRID):
        for snr_db in cfg.VAL_SNR_LIST:
            # Fix both channel set and feedback-noise realization across steps.
            val_seed = cfg.SEED + 1_000_000 + cfg_idx * 10_000 + int(snr_db) * 10
            with temporary_seed(val_seed):
                snr_t = torch.tensor(float(snr_db), device=device)
                output = model(H_dl_val, H_ul_val, snr_t, d_f, d_a, cfg_idx)
                W = extract_precoder(output)
            _, rate = criterion(W, H_dl_val, 10.0 ** (-float(snr_db) / 10.0))
            table[(cfg_idx, int(snr_db))] = rate
    return table


def validation_metric(
    table: Mapping[Tuple[int, int], float], cfg: TrainConfig
) -> float:
    if cfg.BEST_METRIC == "mean_all":
        return float(np.mean(list(table.values())))
    if cfg.BEST_METRIC == "snr_25":
        values = [value for (_, snr), value in table.items() if snr == 25]
        if not values:
            raise RuntimeError("BEST_METRIC='snr_25' but 25 dB is absent from validation.")
        return float(np.mean(values))
    raise ValueError(f"Unknown BEST_METRIC: {cfg.BEST_METRIC}")


def format_val_table(
    table: Mapping[Tuple[int, int], float], step: int, cfg: TrainConfig
) -> Tuple[str, float]:
    lines = [f"Step {step} | Validation matrix (bps/Hz)"]
    header = "    allocation     " + " | ".join(
        f"{snr:>4}dB" for snr in cfg.VAL_SNR_LIST
    ) + " | mean"
    lines.append(header)

    for cfg_idx, (d_f, d_a) in enumerate(cfg.ALLOCATION_GRID):
        values = [table[(cfg_idx, snr)] for snr in cfg.VAL_SNR_LIST]
        lines.append(
            f"    ({d_f:>3},{d_a:>3})       "
            + " | ".join(f"{value:6.2f}" for value in values)
            + f" | {np.mean(values):5.2f}"
        )

    metric = validation_metric(table, cfg)
    overall = float(np.mean(list(table.values())))
    lines.append(f"    overall mean: {overall:.3f}")
    lines.append(f"    selection metric [{cfg.BEST_METRIC}]: {metric:.3f}")
    return "\n".join(lines), metric


# =============================================================================
# 3. Proposal model and checkpoint handling
# =============================================================================


def build_proposal_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    # Lazy import keeps --dry-run usable even on machines without the full model tree.
    from models.adaptive_hybrid import AdaptiveHybridPrecoder

    return AdaptiveHybridPrecoder(cfg).to(device)


def build_channel_generator(cfg: TrainConfig, device: torch.device):
    from utils import CDLChannelGenerator_Feedback

    return CDLChannelGenerator_Feedback(
        Nt=cfg.ANTENNAS,
        Nsc=cfg.SUBCARRIERS,
        speed=cfg.SPEED,
        num_users=cfg.K_USERS,
    ).to(device)


def generate_validation_set(channel_gen, cfg: TrainConfig, device: torch.device):
    # Validation channels are independent of the training RNG and reproducible
    # across fresh runs and resumed runs.
    with temporary_seed(cfg.SEED + 500_000):
        with torch.no_grad():
            H_dl, H_ul = channel_gen.generate_batch_data(cfg.VAL_BATCH)
    return H_dl.to(device), H_ul.to(device)


def generate_training_batch(
    channel_gen,
    cfg: TrainConfig,
    device: torch.device,
    step: int,
):
    # The batch for a given global step is deterministic. This avoids changing
    # the data stream after resume even though the in-memory cache is not saved.
    with temporary_seed(cfg.SEED + 10_000_000 + step):
        with torch.no_grad():
            H_dl, H_ul = channel_gen.generate_batch_data(cfg.BATCH_SIZE)
    return H_dl.to(device), H_ul.to(device)


def checkpoint_payload(
    *,
    cfg: TrainConfig,
    step: int,
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    scheduler: Any | None,
    best_metric: float,
    val_table: Mapping[Tuple[int, int], float] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": step,
        "state": model.state_dict(),
        "best_metric": best_metric,
        "config_fingerprint": config_fingerprint(cfg),
        "config": cfg.to_dict(),
        "allocation_grid": cfg.ALLOCATION_GRID,
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if val_table is not None:
        payload["val_table"] = {
            f"alloc{idx}_snr{snr}": value
            for (idx, snr), value in val_table.items()
        }
    return payload


def load_latest_checkpoint(
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[int, float]:
    path = Path(cfg.SAVE_PATH)
    if not path.exists():
        return 1, float("-inf")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    expected = config_fingerprint(cfg)
    actual = checkpoint.get("config_fingerprint")
    if actual is not None and actual != expected:
        raise RuntimeError(
            "Checkpoint/config mismatch. Refusing unsafe resume.\n"
            f"checkpoint fingerprint: {actual}\n"
            f"current fingerprint:    {expected}\n"
            f"path: {path}"
        )

    model.load_state_dict(checkpoint["state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint.get("rng_state"))

    start_step = int(checkpoint["step"]) + 1
    best_metric = float(
        checkpoint.get("best_metric", checkpoint.get("best_rate", float("-inf")))
    )
    return start_step, best_metric


# =============================================================================
# 4. One resolved experiment
# =============================================================================

@dataclass
class RunOptions:
    resume: bool
    skip_existing: bool
    stop_after: int | None = None
    init_checkpoint: str | None = None

# @dataclass
# class RunOptions:
#     resume: bool
#     skip_existing: bool
#     stop_after: int | None = None


def train_proposal(cfg: TrainConfig, options: RunOptions) -> None:
    torch.set_float32_matmul_precision("high")
    set_seed(cfg.SEED)
    from utils import setup_gpu

    device = setup_gpu()

    save_dir = Path(cfg.SAVE_DIR)
    log_dir = Path(cfg.LOG_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = config_fingerprint(cfg)
    snapshot = cfg.to_dict()
    snapshot["config_fingerprint"] = fingerprint

    config_path = Path(cfg.CONFIG_PATH)
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing_fp = existing.get("config_fingerprint")
        if existing_fp is not None and existing_fp != fingerprint:
            raise RuntimeError(
                f"Existing config.json does not match the resolved run: {config_path}"
            )
    else:
        atomic_write_json(config_path, snapshot)

    status_path = Path(cfg.STATUS_PATH)
    if options.skip_existing and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            print(f"[SKIP] Completed experiment: {cfg.EXP_NAME}")
            return

    atomic_write_json(
        status_path,
        {"state": "running", "experiment": cfg.EXP_NAME, "fingerprint": fingerprint},
    )

    print("=" * 100)
    print(f"[Experiment] {cfg.EXP_NAME}")
    print(f"[Mode]       {cfg.TRAIN_MODE}")
    print(f"[Scenario]   K={cfg.K_USERS}, Nsc={cfg.SUBCARRIERS}, Dfb={cfg.D_TOT}")
    print(f"[Allocations] {cfg.ALLOCATION_GRID}")
    print(
        f"[Recipe]     {cfg.recipe.name}: steps={cfg.TOTAL_STEPS}, "
        f"batch={cfg.BATCH_SIZE}, lr={cfg.LR:g}, warmup={cfg.WARMUP_PCT:.2f}"
    )
    print(
        f"[Curriculum] phase1={cfg.PHASE1_STEPS}, "
        f"snr_stages=({cfg.STAGE1_STEPS}, {cfg.STAGE2_STEPS}, {cfg.TOTAL_STEPS})"
    )
    print(f"[Checkpoint] {cfg.SAVE_DIR}")

    model = build_proposal_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model]      trainable parameters={n_params / 1e6:.3f} M")

    criterion = SumRateLoss()
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

    # start_step = 1
    # best_metric = float("-inf")
    # if options.resume and Path(cfg.SAVE_PATH).exists():
    #     start_step, best_metric = load_latest_checkpoint(
    #         cfg, model, optimizer, scheduler, device
    #     )
    #     print(
    #         f"[Resume]     start_step={start_step}, "
    #         f"best_{cfg.BEST_METRIC}={best_metric:.3f}"
    #     )
    # elif not options.resume and Path(cfg.SAVE_PATH).exists():
    #     raise FileExistsError(
    #         f"Checkpoint exists but --no-resume was requested: {cfg.SAVE_PATH}"
    #     )
    
    start_step = 1
    best_metric = float("-inf")
    latest_path = Path(cfg.SAVE_PATH)

    if options.resume and latest_path.exists():
        # Resume an interrupted fine-tuning run, including optimizer/scheduler.
        start_step, best_metric = load_latest_checkpoint(
            cfg,
            model,
            optimizer,
            scheduler,
            device,
        )
        print(
            f"[Resume]     start_step={start_step}, "
            f"best_{cfg.BEST_METRIC}={best_metric:.3f}"
        )

    elif latest_path.exists():
        raise FileExistsError(
            f"Checkpoint already exists but resume is disabled: {latest_path}"
        )

    elif options.init_checkpoint is not None:
        # Start a new fine-tuning run from old model weights only.
        init_path = Path(options.init_checkpoint)

        if not init_path.is_file():
            raise FileNotFoundError(
                f"Initialization checkpoint does not exist: {init_path}"
            )

        checkpoint = torch.load(
            init_path,
            map_location="cpu",
            weights_only=False,
        )

        if "state" not in checkpoint:
            raise KeyError(
                f"Checkpoint does not contain a 'state' field: {init_path}"
            )

        model.load_state_dict(
            checkpoint["state"],
            strict=True,
        )

        source_config = checkpoint.get("config", {})
        print(f"[Initialize] model weights loaded from: {init_path}")
        print(
            "[Initialize] fresh AdamW optimizer and OneCycleLR schedule "
            "will be used."
        )

        if source_config:
            print(
                "[Initialize] source checkpoint step="
                f"{checkpoint.get('step', 'unknown')}"
            )

        start_step = 1
        best_metric = float("-inf")

    if start_step > cfg.TOTAL_STEPS:
        print(f"[DONE] Latest checkpoint already reached step {cfg.TOTAL_STEPS}.")
        atomic_write_json(
            status_path,
            {
                "state": "completed",
                "experiment": cfg.EXP_NAME,
                "step": cfg.TOTAL_STEPS,
                "best_metric": best_metric,
            },
        )
        return
    
    end_step = cfg.TOTAL_STEPS

    if options.stop_after is not None:
        if options.stop_after <= 0:
            raise ValueError("--stop-after must be a positive integer.")

        end_step = min(
            cfg.TOTAL_STEPS,
            start_step + options.stop_after - 1,
        )

        print(
            f"[Short Run]  Running steps {start_step}--{end_step} "
            f"of {cfg.TOTAL_STEPS}"
        )

    channel_gen = build_channel_generator(cfg, device)
    print("[Validation] Preparing fixed channel set...")
    H_dl_val, H_ul_val = generate_validation_set(channel_gen, cfg, device)
    # Reset CUDA peak-memory statistics before the training loop.
    # The fixed validation tensors remain allocated, so their memory is still
    # included in the current baseline.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Cache is keyed by global step. The deterministic per-step generator keeps
    # resumed runs aligned without storing large tensors inside checkpoints.
    batch_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if step not in batch_cache:
            last = min(cfg.TOTAL_STEPS, step + cfg.DATA_CACHE_SIZE - 1)
            for future_step in range(step, last + 1):
                if future_step not in batch_cache:
                    batch_cache[future_step] = generate_training_batch(
                        channel_gen, cfg, device, future_step
                    )
        batch = batch_cache.pop(step)
        return batch

    progress = tqdm(
        range(start_step, end_step + 1),
        desc=f"K{cfg.K_USERS}-D{cfg.D_TOT}-{cfg.TRAIN_MODE}",
        dynamic_ncols=True,
    )

    for step in progress:
        H_dl_batch, H_ul_batch = get_batch(step)

        cfg_idx = allocation_index_for_step(
            step, len(cfg.ALLOCATION_GRID), cfg.SEED
        )
        d_f, d_a = cfg.ALLOCATION_GRID[cfg_idx]

        model.train()
        optimizer.zero_grad(set_to_none=True)

        snr = get_snr(step, cfg)
        noise_power = 10.0 ** (-snr / 10.0)
        snr_tensor = torch.tensor(snr, device=device, dtype=torch.float32)

        output = model(
            H_dl_batch,
            H_ul_batch,
            snr_tensor,
            d_f,
            d_a,
            cfg_idx,
        )
        # W, H_hat = extract_training_outputs(output)

        # sum_rate_loss, train_rate = criterion(W, H_dl_batch, noise_power)
        # dir_loss, mse_loss, cos_sim = alignment_losses(
        #     H_hat, H_dl_batch, cfg.DIR_MODE
        # )

        # 模型原始训练输出仍保留，用于兼容当前 forward 接口。
        W_residual_train, H_hat = extract_training_outputs(output)

        # ============================================================
        # Deploy-consistent task path
        # 训练时直接优化验证/部署阶段真正使用的 RZF(H_hat)。
        # ============================================================
        proposal_model = (
            model.module
            if hasattr(model, "module")
            else model
        )

        W_deploy = proposal_model.zf_solver(H_hat)

        frob_deploy = (
            W_deploy.abs()
            .square()
            .sum(dim=(1, 2), keepdim=True)
        )

        W_deploy = W_deploy * torch.sqrt(
            cfg.TOTAL_POWER / (frob_deploy + 1e-9)
        )

        # 主 sum-rate loss 只作用于实际部署路径。
        sum_rate_loss, train_rate = criterion(
            W_deploy,
            H_dl_batch,
            noise_power,
        )

        dir_loss, mse_loss, cos_sim = alignment_losses(
            H_hat,
            H_dl_batch,
            cfg.DIR_MODE,
        )        

        if step <= cfg.PHASE1_STEPS:
            w_dir, w_mse = cfg.PH1_W_DIR, cfg.PH1_W_MSE
            phase = "phase1_anchor"
        else:
            w_dir, w_mse = cfg.PH2_W_DIR, cfg.PH2_W_MSE
            phase = "phase2_task"

        loss = sum_rate_loss + w_dir * dir_loss + w_mse * mse_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step={step}, allocation={(d_f, d_a)}, snr={snr:.2f}."
            )

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at step={step}.")
        optimizer.step()
        scheduler.step()

        if step % cfg.LOG_INTERVAL == 0:
            progress.set_postfix(
                {
                    "phase": phase,
                    "alloc": f"({d_f},{d_a})",
                    "SR": f"{train_rate:.2f}",
                    "cos": f"{float(cos_sim.detach()):.3f}",
                    "MSE": f"{float(mse_loss.detach()):.4f}",
                    "SNR": f"{snr:.1f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        if step % cfg.VAL_INTERVAL == 0 or step == end_step:
            val_table = validate_proposal(
                model, H_dl_val, H_ul_val, criterion, cfg, device
            )
            text, current_metric = format_val_table(val_table, step, cfg)
            progress.write(text)

            metrics_record = {
                "step": step,
                "selection_metric_name": cfg.BEST_METRIC,
                "selection_metric": current_metric,
                "best_metric_before_update": best_metric,
                "table": {
                    f"alloc{idx}_snr{snr}": value
                    for (idx, snr), value in val_table.items()
                },
            }
            append_jsonl(cfg.METRICS_PATH, metrics_record)

            if current_metric > best_metric:
                best_metric = current_metric
                progress.write(
                    f"New best {cfg.BEST_METRIC}: {best_metric:.3f}; saving best.pth"
                )
                atomic_torch_save(
                    checkpoint_payload(
                        cfg=cfg,
                        step=step,
                        model=model,
                        optimizer=None,
                        scheduler=None,
                        best_metric=best_metric,
                        val_table=val_table,
                    ),
                    cfg.BEST_PATH,
                )

            atomic_torch_save(
                checkpoint_payload(
                    cfg=cfg,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    val_table=val_table,
                ),
                cfg.SAVE_PATH,
            )

    # Report the true CUDA peak over training, validation, and checkpointing.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )
        peak_reserved_gb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        )

        print(
            f"[GPU Memory] peak allocated={peak_allocated_gb:.2f} GB, "
            f"peak reserved={peak_reserved_gb:.2f} GB"
        )
        
    if end_step < cfg.TOTAL_STEPS:
        atomic_write_json(
            status_path,
            {
                "state": "paused",
                "experiment": cfg.EXP_NAME,
                "step": end_step,
                "total_steps": cfg.TOTAL_STEPS,
                "best_metric_name": cfg.BEST_METRIC,
                "best_metric": best_metric,
                "latest_path": cfg.SAVE_PATH,
            },
        )

        print(
            f"[Paused] {cfg.EXP_NAME} at step {end_step}/"
            f"{cfg.TOTAL_STEPS}"
        )
        return
    
    atomic_write_json(
        status_path,
        {
            "state": "completed",
            "experiment": cfg.EXP_NAME,
            "step": cfg.TOTAL_STEPS,
            "best_metric_name": cfg.BEST_METRIC,
            "best_metric": best_metric,
            "best_path": cfg.BEST_PATH,
            "latest_path": cfg.SAVE_PATH,
        },
    )
    print(f"[Completed] {cfg.EXP_NAME}; best {cfg.BEST_METRIC}={best_metric:.3f}")


# =============================================================================
# 5. Suite filtering and CLI
# =============================================================================


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def filter_runs(
    runs: Sequence[TrainConfig],
    *,
    only_k: set[int] | None,
    only_budget: set[int] | None,
    only_mode: str | None,
    only_model: str | None,
    num_shards: int,
    shard_id: int,
) -> list[TrainConfig]:
    filtered = [
        cfg
        for cfg in runs
        if (only_k is None or cfg.K_USERS in only_k)
        and (only_budget is None or cfg.D_TOT in only_budget)
        and (only_mode is None or cfg.TRAIN_MODE == only_mode)
        and (only_model is None or cfg.MODEL_NAME == only_model)
    ]

    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= shard_id < num_shards:
        raise ValueError("Require 0 <= --shard-id < --num-shards.")

    return [cfg for index, cfg in enumerate(filtered) if index % num_shards == shard_id]


def print_run_plan(runs: Sequence[TrainConfig]) -> None:
    if not runs:
        print("No experiments matched the requested filters.")
        return
    print(f"Resolved {len(runs)} experiment(s):")
    for index, cfg in enumerate(runs):
        print(
            f"[{index:02d}] model={cfg.MODEL_NAME:<12} mode={cfg.TRAIN_MODE:<10} "
            f"K={cfg.K_USERS:<2} Nsc={cfg.SUBCARRIERS:<3} Dfb={cfg.D_TOT:<3} "
            f"alloc={cfg.ALLOCATION_GRID}"
        )
        print(f"     {cfg.EXP_NAME}")
        print(
            f"     recipe={cfg.recipe.name}, "
            f"steps={cfg.TOTAL_STEPS}, "
            f"batch={cfg.BATCH_SIZE}, "
            f"val_batch={cfg.VAL_BATCH}, "
            f"cache={cfg.DATA_CACHE_SIZE}, "
            f"lr={cfg.LR:g}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one declared Hybrid FDMA-AirComp experiment suite."
    )
    parser.add_argument("--suite", type=str, default="proposal_adaptive")
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-k", type=str, default=None, help="Example: 4,8")
    parser.add_argument("--only-budget", type=str, default=None, help="Example: 128,256")
    parser.add_argument(
        "--only-mode",
        choices=["adaptive", "specialist", "baseline"],
        default=None,
    )
    parser.add_argument(
        "--only-model",
        choices=["proposal", "csinet_plus", "swin_cfnet"],
        default=None,
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Skip declared baseline runs whose project-specific adapters are not implemented.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Initialize model weights from an existing checkpoint while "
            "starting a fresh optimizer and learning-rate schedule."
        ),
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Run only this many additional steps, save, and pause.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.list_suites:
        print("Available suites:")
        for name in list_suites():
            print(f"  - {name}")
        return 0

    runs = get_suite(args.suite, output_root=args.output_root)
    runs = filter_runs(
        runs,
        only_k=parse_int_set(args.only_k),
        only_budget=parse_int_set(args.only_budget),
        only_mode=args.only_mode,
        only_model=args.only_model,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    print_run_plan(runs)

    if args.dry_run:
        return 0
    if not runs:
        return 0

    # options = RunOptions(
    #     resume=not args.no_resume,
    #     skip_existing=args.skip_existing,
    #     stop_after=args.stop_after,
    # )
    options = RunOptions(
        resume=not args.no_resume,
        skip_existing=args.skip_existing,
        stop_after=args.stop_after,
        init_checkpoint=args.init_checkpoint,
    )
    failures = []

    for index, cfg in enumerate(runs, start=1):
        print(f"\n[Suite] Starting {index}/{len(runs)}: {cfg.EXP_NAME}")
        try:
            if cfg.MODEL_NAME == "proposal":
                train_proposal(cfg, options)
            else:
                message = (
                    f"Training adapter for '{cfg.MODEL_NAME}' is not implemented because "
                    "its constructor and forward/loss interface were not included. "
                    "The resolved baseline config is ready in train_config.py."
                )
                if args.skip_unsupported:
                    print(f"[SKIP] {message}")
                    continue
                raise NotImplementedError(message)
        except Exception as exc:  # Keep an overnight suite alive unless fail-fast is requested.
            failures.append((cfg.EXP_NAME, repr(exc)))
            try:
                atomic_write_json(
                    cfg.STATUS_PATH,
                    {
                        "state": "failed",
                        "experiment": cfg.EXP_NAME,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception:
                pass
            print(f"[FAILED] {cfg.EXP_NAME}: {exc}", file=sys.stderr)
            traceback.print_exc()
            if args.fail_fast:
                break
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        print("\nSuite finished with failures:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        return 1

    print("\nSuite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
