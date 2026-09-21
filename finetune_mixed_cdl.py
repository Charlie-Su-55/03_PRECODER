#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mixed-CDL fine-tuning for the proposed adaptive Hybrid model.

Protocol
--------
- Warm-start from the frozen CDL-C checkpoint.
- Fine-tune on CDL-A/B/C/D/E.
- Every 25 training steps visits every (CDL profile, allocation) pair exactly once.
- Fixed feedback SNR and downlink SNR.
- Deploy-consistent sum-rate loss only.
- Validation uses fixed channels for every CDL profile.
- Best checkpoint is selected by the equal mean over all CDL profiles x allocations.

This is a robustness/adaptation experiment, not a new architecture.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
import main_train as train_base
from models.adaptive_hybrid import AdaptiveHybridPrecoder
from utils import CDLChannelGenerator_Feedback, setup_gpu


VALID_CDL = ("A", "B", "C", "D", "E")


def parse_profiles(text: str) -> Tuple[str, ...]:
    profiles = []
    for item in text.split(","):
        item = item.strip().upper()
        if not item:
            continue
        if item not in VALID_CDL:
            raise ValueError(f"Unsupported CDL profile '{item}'. Expected one of {VALID_CDL}.")
        if item not in profiles:
            profiles.append(item)
    if not profiles:
        raise ValueError("At least one CDL profile is required.")
    return tuple(profiles)


def condition_for_step(step: int, n_profiles: int, n_allocations: int, seed: int) -> Tuple[int, int]:
    """Balanced schedule: every block visits every profile x allocation pair once."""
    n_conditions = n_profiles * n_allocations
    cycle, position = divmod(step - 1, n_conditions)
    order = list(range(n_conditions))
    random.Random(seed + cycle).shuffle(order)
    condition = order[position]
    return condition // n_allocations, condition % n_allocations


def build_generators(scenario: base_eval.Scenario, profiles: Sequence[str], device: torch.device):
    generators = {}
    for profile in profiles:
        generators[profile] = CDLChannelGenerator_Feedback(
            Nt=scenario.antennas,
            Nsc=scenario.subcarriers,
            carrier_freq=scenario.carrier_freq,
            speed=scenario.speed,
            num_users=scenario.k_users,
            cdl_model=profile,
        ).to(device)
    return generators


def build_validation_sets(generators, profiles: Sequence[str], batch_size: int, seed: int, device: torch.device):
    validation_sets = {}
    for profile in profiles:
        val_seed = seed + 500_000 + base_eval.stable_int(f"CDL-{profile}")
        with train_base.temporary_seed(val_seed), torch.no_grad():
            H_dl, H_ul = generators[profile].generate_batch_data(batch_size)
        validation_sets[profile] = (H_dl.to(device), H_ul.to(device))
    return validation_sets


def deploy_precoder(model: nn.Module, H_hat: torch.Tensor, total_power: float) -> torch.Tensor:
    W = model.zf_solver(H_hat)
    power = W.abs().square().sum(dim=(1, 2), keepdim=True)
    return W * torch.sqrt(float(total_power) / (power + 1e-9))


@torch.no_grad()
def validate(model: nn.Module, validation_sets, profiles: Sequence[str], allocations, criterion, feedback_snr: float, downlink_snr: float, seed: int, device: torch.device):
    model.eval()
    table: Dict[Tuple[str, int], float] = {}
    snr_t = torch.tensor(float(feedback_snr), dtype=torch.float32, device=device)
    noise_power = 10.0 ** (-float(downlink_snr) / 10.0)

    for profile in profiles:
        H_dl, H_ul = validation_sets[profile]
        for cfg_idx, (d_f, d_a) in enumerate(allocations):
            val_seed = seed + 1_000_000 + base_eval.stable_int(f"CDL-{profile}") + cfg_idx * 10_000
            with train_base.temporary_seed(val_seed):
                output = model(H_dl, H_ul, snr_t, d_f, d_a, cfg_idx)
                W = train_base.extract_precoder(output)
            _, rate = criterion(W, H_dl, noise_power)
            table[(profile, cfg_idx)] = rate

    return table


def format_validation(table: Mapping[Tuple[str, int], float], profiles: Sequence[str], allocations, step: int) -> Tuple[str, float]:
    lines = [f"Step {step} | Mixed-CDL validation (bps/Hz)"]
    header = "    allocation   " + " | ".join(f"CDL-{p:>1}" for p in profiles) + " | mean"
    lines.append(header)

    for cfg_idx, (d_f, d_a) in enumerate(allocations):
        values = [table[(profile, cfg_idx)] for profile in profiles]
        lines.append(
            f"    ({d_f:>2},{d_a:>3})      "
            + " | ".join(f"{value:6.2f}" for value in values)
            + f" | {np.mean(values):5.2f}"
        )

    profile_means = []
    for profile in profiles:
        values = [table[(profile, cfg_idx)] for cfg_idx in range(len(allocations))]
        profile_means.append(float(np.mean(values)))

    lines.append("    profile mean " + " | ".join(f"{value:6.2f}" for value in profile_means))
    metric = float(np.mean(list(table.values())))
    lines.append(f"    overall mean: {metric:.3f}")
    return "\n".join(lines), metric


def checkpoint_payload(source_checkpoint, model, step: int, best_metric: float, profiles, allocations, args, val_table=None, optimizer=None, scheduler=None):
    payload = {
        "step": step,
        "state": model.state_dict(),
        "best_metric": best_metric,
        "config": source_checkpoint.get("config", {}),
        "allocation_grid": source_checkpoint.get("allocation_grid", list(allocations)),
        "source_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "finetune_type": "mixed_cdl",
        "mixed_cdl_profiles": list(profiles),
        "feedback_snr_db": args.feedback_snr,
        "downlink_snr_db": args.downlink_snr,
        "seed": args.seed,
    }

    if source_checkpoint.get("config_fingerprint") is not None:
        payload["config_fingerprint"] = source_checkpoint["config_fingerprint"]
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if val_table is not None:
        payload["val_table"] = {
            f"cdl{profile}_alloc{cfg_idx}": value
            for (profile, cfg_idx), value in val_table.items()
        }

    return payload


def save_checkpoint(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def append_metrics(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Mixed-CDL fine-tuning for the proposed CSI-feedback model.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--nsc", type=int, default=128)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--antennas", type=int, default=32)
    parser.add_argument("--carrier-freq", type=float, default=3.5e9)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--total-power", type=float, default=1.0)
    parser.add_argument("--num-subbands", type=int, default=4)

    parser.add_argument("--init-checkpoint", type=str, required=True)
    parser.add_argument("--profiles", type=str, default="A,B,C,D,E")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-batch", type=int, default=64)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-pct", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=0.5)

    parser.add_argument("--feedback-snr", type=float, default=25.0)
    parser.add_argument("--downlink-snr", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="runs_mixed_cdl_ft")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.steps <= 0 or args.batch_size <= 0 or args.val_batch <= 0 or args.val_interval <= 0:
        raise ValueError("steps, batch-size, val-batch, and val-interval must be positive.")

    profiles = parse_profiles(args.profiles)
    source_path = Path(args.init_checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    train_base.set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = setup_gpu()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    scenario = base_eval.Scenario(
        k_users=args.k,
        subcarriers=args.nsc,
        feedback_budget=args.budget,
        antennas=args.antennas,
        carrier_freq=args.carrier_freq,
        speed=args.speed,
        total_power=args.total_power,
        num_subbands=args.num_subbands,
    )
    scenario.validate()

    source_checkpoint = base_eval.load_checkpoint_cpu(source_path)
    base_eval.validate_checkpoint_scenario(source_checkpoint, source_path, scenario)
    cfg = base_eval.proposal_runtime_config(scenario, source_checkpoint)
    allocations = tuple(tuple(map(int, a)) for a in cfg.GAMMA_GRID)

    model = AdaptiveHybridPrecoder(cfg).to(device)
    model.load_state_dict(source_checkpoint["state"], strict=True)

    exp_name = (
        f"hybrid_gnn_adaptive_K{args.k}_Nsc{args.nsc}_Dfb{args.budget}"
        f"_mixedcdl_{''.join(profiles)}_ft_seed{args.seed}"
    )
    save_dir = Path(args.output_root) / "checkpoints" / exp_name
    log_dir = Path(args.output_root) / "logs" / exp_name
    best_path = save_dir / "best.pth"
    latest_path = save_dir / "latest.pth"
    metrics_path = log_dir / "metrics.jsonl"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps,
        pct_start=args.warmup_pct, anneal_strategy="cos"
    )
    criterion = train_base.SumRateLoss()

    start_step = 1
    best_metric = float("-inf")

    if args.resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"]) + 1
        best_metric = float(checkpoint["best_metric"])
        print(f"[Resume] step={start_step}, best={best_metric:.3f}")
    elif latest_path.exists():
        raise FileExistsError(f"{latest_path} already exists. Use --resume or choose another output root.")

    generators = build_generators(scenario, profiles, device)
    print("[Validation] Preparing fixed validation channels...")
    validation_sets = build_validation_sets(generators, profiles, args.val_batch, args.seed, device)

    print("=" * 108)
    print("[Mixed-CDL fine-tuning]")
    print(f"[Scenario]      K={args.k}, Nsc={args.nsc}, Dtot={args.budget}")
    print(f"[Source]        {source_path}")
    print(f"[Source step]   {source_checkpoint.get('step', 'unknown')}")
    print(f"[Profiles]      {profiles}")
    print(f"[Allocations]   {allocations}")
    print(f"[Feedback SNR]  {args.feedback_snr:.1f} dB")
    print(f"[Downlink SNR]  {args.downlink_snr:.1f} dB")
    print(f"[Steps]         {args.steps}")
    print(f"[Batch]         {args.batch_size}")
    print(f"[LR]            {args.lr:g}")
    print("[Loss]          deploy-consistent sum-rate only")
    print(f"[Output]        {save_dir}")
    print("=" * 108)

    if start_step == 1:
        table = validate(
            model, validation_sets, profiles, allocations, criterion,
            args.feedback_snr, args.downlink_snr, args.seed, device
        )
        text, best_metric = format_validation(table, profiles, allocations, 0)
        print(text)
        print(f"[Initial metric] {best_metric:.3f}")

        save_checkpoint(
            checkpoint_payload(
                source_checkpoint, model, 0, best_metric, profiles, allocations,
                args, val_table=table
            ),
            best_path,
        )

        append_metrics(metrics_path, {
            "step": 0,
            "selection_metric": best_metric,
            "table": {f"cdl{p}_alloc{i}": v for (p, i), v in table.items()},
        })

    feedback_snr_t = torch.tensor(args.feedback_snr, dtype=torch.float32, device=device)
    downlink_noise_power = 10.0 ** (-args.downlink_snr / 10.0)

    progress = tqdm(
        range(start_step, args.steps + 1),
        desc=f"mixed-cdl-K{args.k}-D{args.budget}",
        dynamic_ncols=True,
    )

    for step in progress:
        profile_idx, cfg_idx = condition_for_step(step, len(profiles), len(allocations), args.seed)
        profile = profiles[profile_idx]
        d_f, d_a = allocations[cfg_idx]

        data_seed = args.seed + 10_000_000 + step + base_eval.stable_int(f"CDL-{profile}")
        with train_base.temporary_seed(data_seed), torch.no_grad():
            H_dl, H_ul = generators[profile].generate_batch_data(args.batch_size)
        H_dl, H_ul = H_dl.to(device), H_ul.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        feedback_seed = args.seed + 40_000_000 + step
        with train_base.temporary_seed(feedback_seed):
            output = model(H_dl, H_ul, feedback_snr_t, d_f, d_a, cfg_idx)

        _, H_hat = train_base.extract_training_outputs(output)
        W_deploy = deploy_precoder(model, H_hat, args.total_power)
        loss, train_rate = criterion(W_deploy, H_dl, downlink_noise_power)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step={step}, CDL-{profile}, allocation={(d_f, d_a)}."
            )

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at step={step}.")

        optimizer.step()
        scheduler.step()

        progress.set_postfix(
            cdl=profile,
            alloc=f"({d_f},{d_a})",
            SR=f"{train_rate:.2f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if step % args.val_interval == 0 or step == args.steps:
            table = validate(
                model, validation_sets, profiles, allocations, criterion,
                args.feedback_snr, args.downlink_snr, args.seed, device
            )
            text, current_metric = format_validation(table, profiles, allocations, step)
            progress.write(text)

            append_metrics(metrics_path, {
                "step": step,
                "selection_metric": current_metric,
                "best_metric_before_update": best_metric,
                "table": {f"cdl{p}_alloc{i}": v for (p, i), v in table.items()},
            })

            if current_metric > best_metric:
                best_metric = current_metric
                progress.write(f"[Best] mean CDL×allocation rate={best_metric:.3f}; saving best.pth")
                save_checkpoint(
                    checkpoint_payload(
                        source_checkpoint, model, step, best_metric, profiles,
                        allocations, args, val_table=table
                    ),
                    best_path,
                )

            save_checkpoint(
                checkpoint_payload(
                    source_checkpoint, model, step, best_metric, profiles,
                    allocations, args, val_table=table,
                    optimizer=optimizer, scheduler=scheduler
                ),
                latest_path,
            )

    print("=" * 108)
    print("[Completed]")
    print(f"Best metric : {best_metric:.3f}")
    print(f"Best path   : {best_path}")
    print(f"Latest path : {latest_path}")
    print("=" * 108)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())