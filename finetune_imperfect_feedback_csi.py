#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pilot-CSI-aware fine-tuning for the proposed adaptive hybrid feedback model.

Physical feedback transmission always uses the true H_ul.
The BS receives only a pilot-aided LS estimate H_ul_est.

Training:
    - warm-start from an existing proposed checkpoint
    - full allocation grid
    - fixed feedback-data SNR
    - fixed downlink SNR
    - random pilot SNR
    - pure deploy-consistent sum-rate loss
    - no direction/MSE auxiliary losses

The resulting checkpoint is intended for the imperfect-feedback-link-CSI
robustness experiment and does not replace the original headline checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from configs.train_config import TrainConfig, get_suite
from main_train import (
    SumRateLoss,
    allocation_index_for_step,
    atomic_torch_save,
    atomic_write_json,
    build_channel_generator,
    build_proposal_model,
    extract_precoder,
    extract_training_outputs,
    generate_training_batch,
    generate_validation_set,
    set_seed,
)
from utils import setup_gpu
from utils.channel_estimation import estimate_feedback_channel_ls


def parse_snr_list(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"inf", "+inf", "perfect"}:
            values.append(float("inf"))
        else:
            values.append(float(item))
    if not values:
        raise ValueError("SNR list cannot be empty.")
    return values


def snr_label(value: float) -> str:
    if math.isinf(value):
        return "perfect"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def build_finetune_config(args: argparse.Namespace) -> TrainConfig:
    candidates = get_suite("proposal_adaptive", output_root=args.output_root)
    matches = [
        cfg for cfg in candidates
        if cfg.K_USERS == args.k
        and cfg.SUBCARRIERS == args.nsc
        and cfg.D_TOT == args.budget
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one proposal config for "
            f"K={args.k}, Nsc={args.nsc}, Dtot={args.budget}, found {len(matches)}."
        )

    base = matches[0]

    recipe = replace(
        base.recipe,
        name=f"k{args.k}_n{args.nsc}_pilotls_ft",
        total_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        val_interval=args.val_interval,
        val_batch=args.val_batch,
        warmup_pct=args.warmup_pct,
        grad_clip=args.grad_clip,
        phase1_pct=0.50,
        snr_stage1_pct=0.30,
        snr_stage2_pct=0.70,
        stage1_snr=(args.feedback_snr, args.feedback_snr),
        stage2_snr=(args.feedback_snr, args.feedback_snr),
        stage3_snr=(args.feedback_snr, args.feedback_snr),
        val_snr_list=(int(args.feedback_snr),),
        ph1_w_dir=0.0,
        ph1_w_mse=0.0,
        ph2_w_dir=0.0,
        ph2_w_mse=0.0,
        weight_decay=args.weight_decay,
        data_cache_size=1,
        log_interval=args.log_interval,
        best_metric="mean_all",
    )

    spec = replace(base.spec, recipe_name=recipe.name)

    return TrainConfig(
        spec=spec,
        allocations=base.allocations,
        full_allocation_grid=base.full_allocation_grid,
        recipe=recipe,
        physical=base.physical,
        proposal=base.proposal,
        baselines=base.baselines,
        output_root=Path(args.output_root),
    )


def sample_pilot_snr(step: int, seed: int, choices: List[float]) -> float:
    rng = random.Random(seed + 30_000_000 + step)
    return rng.choice(choices)


def deploy_precoder(model: nn.Module, H_hat: torch.Tensor, total_power: float) -> torch.Tensor:
    proposal_model = model.module if hasattr(model, "module") else model
    W = proposal_model.zf_solver(H_hat)
    frob = W.abs().square().sum(dim=(1, 2), keepdim=True)
    return W * torch.sqrt(total_power / (frob + 1e-9))


def make_checkpoint(
    cfg: TrainConfig,
    model: nn.Module,
    step: int,
    best_metric: float,
    source_checkpoint: str,
    source_step: int,
    args: argparse.Namespace,
    val_table: Mapping[Tuple[int, float], float] | None = None,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    payload = {
        "step": step,
        "state": model.state_dict(),
        "best_metric": best_metric,
        "config": cfg.to_dict(),
        "allocation_grid": cfg.ALLOCATION_GRID,
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_step": source_step,
        "finetune_protocol": {
            "experiment": "pilot_aided_imperfect_feedback_link_csi_finetune",
            "feedback_snr_db": args.feedback_snr,
            "downlink_snr_db": args.downlink_snr,
            "pilot_length": args.pilot_length or args.k,
            "train_pilot_snr_db": parse_snr_list(args.train_pilot_snr),
            "val_pilot_snr_db": parse_snr_list(args.val_pilot_snr),
            "loss": "deploy_consistent_sum_rate_only",
            "lr": args.lr,
            "steps": args.steps,
            "batch_size": args.batch_size,
        },
    }

    if val_table is not None:
        payload["val_table"] = {
            f"alloc{idx}_pilot{snr_label(pilot_snr)}": rate
            for (idx, pilot_snr), rate in val_table.items()
        }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    return payload


@torch.no_grad()
def build_validation_estimates(
    H_ul_val: torch.Tensor,
    pilot_snrs: List[float],
    pilot_length: int,
    seed: int,
) -> Dict[float, torch.Tensor]:
    estimates = {}

    for index, pilot_snr in enumerate(pilot_snrs):
        set_seed(seed + 70_000_000 + index)
        ce = estimate_feedback_channel_ls(
            H_ul_val,
            pilot_snr_db=pilot_snr,
            pilot_length=pilot_length,
        )
        estimates[pilot_snr] = ce.H_hat

    return estimates


@torch.no_grad()
def validate(
    model: nn.Module,
    H_dl_val: torch.Tensor,
    H_ul_val: torch.Tensor,
    H_ul_estimates: Dict[float, torch.Tensor],
    criterion: SumRateLoss,
    cfg: TrainConfig,
    pilot_snrs: List[float],
    feedback_snr: float,
    downlink_snr: float,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[Tuple[int, float], float], float]:
    model.eval()

    table: Dict[Tuple[int, float], float] = {}
    feedback_snr_tensor = torch.tensor(feedback_snr, dtype=torch.float32, device=device)
    downlink_noise_power = 10.0 ** (-downlink_snr / 10.0)

    for cfg_idx, (d_f, d_a) in enumerate(cfg.ALLOCATION_GRID):
        for pilot_snr in pilot_snrs:
            H_ul_est = H_ul_estimates[pilot_snr]

            # Keep feedback-data AWGN identical across pilot-SNR conditions
            # for a fixed allocation.
            set_seed(seed + 80_000_000 + cfg_idx)

            output = model(
                H_dl_val,
                H_ul_val,
                feedback_snr_tensor,
                d_f,
                d_a,
                cfg_idx,
                H_ul_est=H_ul_est,
            )

            W = extract_precoder(output)
            _, rate = criterion(W, H_dl_val, downlink_noise_power)
            table[(cfg_idx, pilot_snr)] = rate

    metric = float(np.mean(list(table.values())))
    return table, metric


def format_validation(
    table: Mapping[Tuple[int, float], float],
    cfg: TrainConfig,
    pilot_snrs: List[float],
    step: int,
) -> str:
    lines = [f"Step {step} | Pilot-CSI validation (bps/Hz)"]

    header = "    allocation  " + " | ".join(
        f"{snr_label(snr):>7}" for snr in pilot_snrs
    ) + " | mean"

    lines.append(header)

    for cfg_idx, (d_f, d_a) in enumerate(cfg.ALLOCATION_GRID):
        values = [table[(cfg_idx, snr)] for snr in pilot_snrs]

        lines.append(
            f"    ({d_f:>2},{d_a:>3})     "
            + " | ".join(f"{value:7.2f}" for value in values)
            + f" | {np.mean(values):5.2f}"
        )

    lines.append(f"    overall mean: {np.mean(list(table.values())):.3f}")
    return "\n".join(lines)


def load_source_checkpoint(model: nn.Module, path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if "state" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'state' field: {path}")

    model.load_state_dict(checkpoint["state"], strict=True)
    return int(checkpoint.get("step", -1))


def train(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    cfg = build_finetune_config(args)
    device = setup_gpu()

    train_pilot_snrs = parse_snr_list(args.train_pilot_snr)
    val_pilot_snrs = parse_snr_list(args.val_pilot_snr)
    pilot_length = args.pilot_length if args.pilot_length is not None else cfg.K_USERS

    if pilot_length < cfg.K_USERS:
        raise ValueError(f"pilot_length={pilot_length} must be >= K={cfg.K_USERS}.")

    save_dir = Path(cfg.SAVE_DIR)
    log_dir = Path(cfg.LOG_DIR)
    best_path = Path(cfg.BEST_PATH)
    latest_path = Path(cfg.SAVE_PATH)
    status_path = Path(cfg.STATUS_PATH)
    metrics_path = Path(cfg.METRICS_PATH)

    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = build_proposal_model(cfg, device)
    criterion = SumRateLoss()

    source_path = Path(args.init_checkpoint)
    source_step = -1

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.steps,
        pct_start=args.warmup_pct,
        anneal_strategy="cos",
    )

    start_step = 1
    best_metric = float("-inf")

    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but latest checkpoint does not exist: {latest_path}"
            )

        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_step = int(checkpoint["step"]) + 1
        best_metric = float(checkpoint.get("best_metric", float("-inf")))
        source_step = int(checkpoint.get("source_checkpoint_step", -1))
        source_path = Path(checkpoint.get("source_checkpoint", args.init_checkpoint))

        print(f"[Resume] start_step={start_step}, best_metric={best_metric:.3f}")

    else:
        if latest_path.exists() or best_path.exists():
            raise FileExistsError(
                f"Fine-tuning output already exists in {save_dir}. "
                "Use --resume or choose another --output-root."
            )

        source_step = load_source_checkpoint(model, source_path)

    print("=" * 110)
    print("[Pilot-CSI-aware fine-tuning]")
    print(f"[Scenario]       K={cfg.K_USERS}, Nsc={cfg.SUBCARRIERS}, Dtot={cfg.D_TOT}")
    print(f"[Allocations]    {cfg.ALLOCATION_GRID}")
    print(f"[Source]         {source_path}")
    print(f"[Source step]    {source_step}")
    print(f"[Train pilot]    {[snr_label(x) for x in train_pilot_snrs]}")
    print(f"[Val pilot]      {[snr_label(x) for x in val_pilot_snrs]}")
    print(f"[Pilot length]   {pilot_length}")
    print(f"[Feedback SNR]   {args.feedback_snr:.1f} dB")
    print(f"[Downlink SNR]   {args.downlink_snr:.1f} dB")
    print(f"[Steps]          {args.steps}")
    print(f"[Batch]          {args.batch_size}")
    print(f"[LR]             {args.lr:g}")
    print(f"[Loss]           deploy-consistent sum-rate only")
    print(f"[Output]         {save_dir}")
    print("=" * 110)

    atomic_write_json(
        cfg.CONFIG_PATH,
        {
            **cfg.to_dict(),
            "pilot_csi_finetune": {
                "source_checkpoint": str(source_path),
                "source_checkpoint_step": source_step,
                "pilot_length": pilot_length,
                "train_pilot_snr_db": train_pilot_snrs,
                "val_pilot_snr_db": val_pilot_snrs,
                "feedback_snr_db": args.feedback_snr,
                "downlink_snr_db": args.downlink_snr,
            },
        },
    )

    atomic_write_json(
        status_path,
        {
            "state": "running",
            "experiment": cfg.EXP_NAME,
            "source_checkpoint": str(source_path),
        },
    )

    channel_gen = build_channel_generator(cfg, device)

    print("[Validation] Preparing fixed validation channels...")
    H_dl_val, H_ul_val = generate_validation_set(channel_gen, cfg, device)

    H_ul_est_val = build_validation_estimates(
        H_ul_val,
        val_pilot_snrs,
        pilot_length,
        args.seed,
    )

    # Evaluate the source checkpoint before any update.
    if not args.resume:
        initial_table, initial_metric = validate(
            model,
            H_dl_val,
            H_ul_val,
            H_ul_est_val,
            criterion,
            cfg,
            val_pilot_snrs,
            args.feedback_snr,
            args.downlink_snr,
            args.seed,
            device,
        )

        print(format_validation(initial_table, cfg, val_pilot_snrs, step=0))
        print(f"[Initial metric] {initial_metric:.3f}")

        best_metric = initial_metric

        atomic_torch_save(
            make_checkpoint(
                cfg,
                model,
                step=0,
                best_metric=best_metric,
                source_checkpoint=str(source_path),
                source_step=source_step,
                args=args,
                val_table=initial_table,
            ),
            best_path,
        )

    progress = tqdm(
        range(start_step, args.steps + 1),
        desc=f"pilot-ft-K{cfg.K_USERS}-D{cfg.D_TOT}",
        dynamic_ncols=True,
    )

    feedback_snr_tensor = torch.tensor(
        args.feedback_snr,
        dtype=torch.float32,
        device=device,
    )

    downlink_noise_power = 10.0 ** (-args.downlink_snr / 10.0)

    for step in progress:
        H_dl_batch, H_ul_batch = generate_training_batch(
            channel_gen,
            cfg,
            device,
            step,
        )

        cfg_idx = allocation_index_for_step(
            step,
            len(cfg.ALLOCATION_GRID),
            cfg.SEED,
        )

        d_f, d_a = cfg.ALLOCATION_GRID[cfg_idx]
        pilot_snr = sample_pilot_snr(step, args.seed, train_pilot_snrs)

        # Pilot observation / LS estimation.
        set_seed(args.seed + 40_000_000 + step)

        with torch.no_grad():
            ce = estimate_feedback_channel_ls(
                H_ul_batch,
                pilot_snr_db=pilot_snr,
                pilot_length=pilot_length,
            )

        H_ul_est = ce.H_hat

        # Independent deterministic RNG stream for dropout and feedback-data AWGN.
        set_seed(args.seed + 50_000_000 + step)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        output = model(
            H_dl_batch,
            H_ul_batch,
            feedback_snr_tensor,
            d_f,
            d_a,
            cfg_idx,
            H_ul_est=H_ul_est,
        )

        _, H_hat = extract_training_outputs(output)

        W_deploy = deploy_precoder(
            model,
            H_hat,
            cfg.TOTAL_POWER,
        )

        loss, train_rate = criterion(
            W_deploy,
            H_dl_batch,
            downlink_noise_power,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step={step}, "
                f"allocation={(d_f, d_a)}, pilot_snr={pilot_snr}."
            )

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(
                f"Non-finite gradient norm at step={step}."
            )

        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0:
            progress.set_postfix(
                {
                    "alloc": f"({d_f},{d_a})",
                    "pilot": snr_label(pilot_snr),
                    "SR": f"{train_rate:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        if step % args.val_interval == 0 or step == args.steps:
            val_table, current_metric = validate(
                model,
                H_dl_val,
                H_ul_val,
                H_ul_est_val,
                criterion,
                cfg,
                val_pilot_snrs,
                args.feedback_snr,
                args.downlink_snr,
                args.seed,
                device,
            )

            progress.write(
                format_validation(
                    val_table,
                    cfg,
                    val_pilot_snrs,
                    step,
                )
            )

            record = {
                "step": step,
                "selection_metric": current_metric,
                "best_metric_before_update": best_metric,
                "table": {
                    f"alloc{idx}_pilot{snr_label(pilot_snr)}": rate
                    for (idx, pilot_snr), rate in val_table.items()
                },
            }

            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if current_metric > best_metric:
                best_metric = current_metric

                progress.write(
                    f"[Best] mean allocation×pilot rate={best_metric:.3f}; saving best.pth"
                )

                atomic_torch_save(
                    make_checkpoint(
                        cfg,
                        model,
                        step=step,
                        best_metric=best_metric,
                        source_checkpoint=str(source_path),
                        source_step=source_step,
                        args=args,
                        val_table=val_table,
                    ),
                    best_path,
                )

            atomic_torch_save(
                make_checkpoint(
                    cfg,
                    model,
                    step=step,
                    best_metric=best_metric,
                    source_checkpoint=str(source_path),
                    source_step=source_step,
                    args=args,
                    val_table=val_table,
                    optimizer=optimizer,
                    scheduler=scheduler,
                ),
                latest_path,
            )

    atomic_write_json(
        status_path,
        {
            "state": "completed",
            "experiment": cfg.EXP_NAME,
            "step": args.steps,
            "best_metric": best_metric,
            "best_path": str(best_path),
            "latest_path": str(latest_path),
            "source_checkpoint": str(source_path),
        },
    )

    print("=" * 110)
    print("[Completed]")
    print(f"Best metric : {best_metric:.3f}")
    print(f"Best path   : {best_path}")
    print(f"Latest path : {latest_path}")
    print("=" * 110)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pilot-CSI-aware fine-tuning for the proposed adaptive model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--nsc", type=int, default=128)
    parser.add_argument("--budget", type=int, default=128)

    parser.add_argument("--init-checkpoint", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="runs_imperfect_csi_ft")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-pct", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--val-batch", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument(
        "--train-pilot-snr",
        type=str,
        default="-5,0,5,10,15,20,inf",
    )

    parser.add_argument(
        "--val-pilot-snr",
        type=str,
        default="-5,0,5,10,15,20,inf",
    )

    parser.add_argument("--pilot-length", type=int, default=None)
    parser.add_argument("--feedback-snr", type=float, default=25.0)
    parser.add_argument("--downlink-snr", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.batch_size <= 0 or args.val_batch <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.val_interval <= 0 or args.log_interval <= 0:
        raise ValueError("validation/log intervals must be positive.")
    if not 0.0 < args.warmup_pct < 1.0:
        raise ValueError("--warmup-pct must lie in (0,1).")

    train(args)


if __name__ == "__main__":
    main()