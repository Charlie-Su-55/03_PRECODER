#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_ablation.py
====================

Unified evaluation for the completed K=4, Nsc=128, Dtot=128 ablation runs.

It evaluates:
1. Full model from runs_deployzf_v1
2. no_condition
3. no_anchor_injection
4. no_cross_user_aggregation

All models use the same channel realizations and the same feedback-noise seeds.
The default protocol matches training: the same SNR value is used for feedback
noise and downlink receiver noise.

Outputs:
- ablation_summary.csv
- ablation_samples.csv.gz
- ablation_results.json

Example:
python evaluate_ablation.py \
  --snr-list 25 \
  --batch-size 48 \
  --num-batches 50 \
  --full-root runs_deployzf_v1 \
  --ablation-root runs_ablation_deployzf_v1 \
  --output-dir runs_evaluation/ablation_k4_d128
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from utils import CDLChannelGenerator_Feedback, setup_gpu

# The single-file ablation trainer contains the ablation model, configuration,
# and suite resolver. Importing it does not start training because its main
# function is protected by the usual __main__ guard.
import train_ablation as ablation_module

# The completed full-model checkpoint uses the normal project model.
from models.adaptive_hybrid import AdaptiveHybridPrecoder as FullAdaptiveModel


ALLOCATION_GRID: Tuple[Tuple[int, int], ...] = (
    (32, 0),
    (24, 32),
    (16, 64),
    (8, 96),
    (0, 128),
)

DISPLAY_NAMES = {
    "full": "Proposed Full Model",
    "no_condition": "w/o Condition Embedding",
    "no_anchor_injection": "w/o Iterative Anchor Injection",
    "no_cross_user_aggregation": "w/o Cross-User Aggregation",
}


def parse_float_csv(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one SNR value is required.")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_se_ci95(values: torch.Tensor) -> Tuple[float, float, float, float]:
    values = values.detach().float().cpu().flatten()
    mean = float(values.mean().item())
    if values.numel() <= 1:
        return mean, 0.0, mean, mean
    se = float(values.std(unbiased=True).item() / math.sqrt(values.numel()))
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def per_sample_sum_rate(
    W: torch.Tensor,
    H_dl_true: torch.Tensor,
    noise_power: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    W: [B, Nt, K, Nsc]
    H: [B, K, Nt, Nsc]
    Returns one mean sum-rate value per sample.
    """
    h_eff = torch.einsum(
        "bkns,bnjs->bkjs",
        H_dl_true.conj(),
        W,
    )
    power = h_eff.abs().square()
    signal = torch.diagonal(
        power,
        dim1=1,
        dim2=2,
    ).permute(0, 2, 1)
    interference = power.sum(dim=2) - signal
    sinr = signal / (interference + float(noise_power) + eps)
    rate = torch.log2(1.0 + sinr)
    return rate.sum(dim=1).mean(dim=-1)


def discover_full_checkpoint(
    root: str | Path,
    explicit_path: str | None,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    checkpoint_root = Path(root) / "checkpoints"
    patterns = (
        "hybrid_gnn_adaptive_K4_Nsc128_Dfb128*/best.pth",
        "adaptive_*K4*Nsc128*Dfb128*/best.pth",
    )

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(checkpoint_root.glob(pattern))

    candidates = sorted(set(path.resolve() for path in candidates))
    if not candidates:
        raise FileNotFoundError(
            f"No full-model K4/Nsc128/Dfb128 best.pth under "
            f"{checkpoint_root}"
        )
    if len(candidates) > 1:
        text = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            "Multiple full-model checkpoints matched. "
            "Use --full-checkpoint explicitly.\n"
            f"{text}"
        )
    return candidates[0]


def make_full_config():
    """
    Build the same K4/Nsc128/D128 runtime config used by the full model.
    The original model ignores the ablation switches.
    """
    spec = ablation_module.ExperimentSpec(
        model_name="proposal",
        train_mode="adaptive",
        k_users=4,
        subcarriers=128,
        feedback_budget=128,
        recipe_name="k4_n128_standard",
        seed=42,
        variant="",
    )
    return ablation_module.resolve_experiment(
        spec,
        output_root="runs_deployzf_v1",
    )


def load_checkpoint_state(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> Tuple[int, Mapping]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "state" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain a state dict: {checkpoint_path}"
        )
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()
    return int(checkpoint.get("step", -1)), checkpoint


def load_models(
    full_root: str,
    ablation_root: str,
    full_checkpoint: str | None,
    device: torch.device,
) -> Dict[str, dict]:
    loaded: Dict[str, dict] = {}

    full_cfg = make_full_config()
    full_path = discover_full_checkpoint(full_root, full_checkpoint)
    full_model = FullAdaptiveModel(full_cfg).to(device)
    full_step, _ = load_checkpoint_state(full_model, full_path)
    loaded["full"] = {
        "model": full_model,
        "cfg": full_cfg,
        "path": full_path,
        "step": full_step,
    }

    ablation_cfgs = ablation_module.get_suite(
        "proposal_ablation_k4_n128",
        output_root=ablation_root,
    )

    for cfg in ablation_cfgs:
        variant = cfg.ABLATION_NAME
        path = Path(cfg.BEST_PATH).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Missing best checkpoint for {variant}: {path}"
            )
        model = ablation_module.AdaptiveHybridPrecoder(cfg).to(device)
        step, _ = load_checkpoint_state(model, path)
        loaded[variant] = {
            "model": model,
            "cfg": cfg,
            "path": path,
            "step": step,
        }

    return loaded


def build_channel_generator(device: torch.device):
    generator = CDLChannelGenerator_Feedback(
        Nt=32,
        Nsc=128,
        carrier_freq=3.5e9,
        speed=1.0,
        num_users=4,
    )
    if hasattr(generator, "to"):
        generator = generator.to(device)
    return generator


@torch.no_grad()
def evaluate(
    loaded: Dict[str, dict],
    snr_list: Sequence[float],
    batch_size: int,
    num_batches: int,
    seed: int,
    device: torch.device,
) -> Tuple[List[dict], List[dict]]:
    generator = build_channel_generator(device)

    # rates[variant][snr][allocation] -> list of [B] tensors
    rates: Dict[str, Dict[float, Dict[Tuple[int, int], List[torch.Tensor]]]] = {
        variant: {
            snr: {allocation: [] for allocation in ALLOCATION_GRID}
            for snr in snr_list
        }
        for variant in loaded
    }

    progress = tqdm(
        range(num_batches),
        desc="evaluate-ablation-K4-Nsc128-D128",
        dynamic_ncols=True,
    )

    for batch_index in progress:
        channel_seed = seed + 4_128_128 + batch_index
        set_seed(channel_seed)
        H_dl, H_ul = generator.generate_batch_data(batch_size)
        H_dl = H_dl.to(device)
        H_ul = H_ul.to(device)

        for snr_db in snr_list:
            noise_power = 10.0 ** (-float(snr_db) / 10.0)
            snr_tensor = torch.tensor(
                float(snr_db),
                dtype=torch.float32,
                device=device,
            )

            for cfg_idx, (d_f, d_a) in enumerate(ALLOCATION_GRID):
                # Same seed across all variants at a given batch/SNR/allocation.
                feedback_seed = (
                    seed
                    + 100_000_000
                    + batch_index * 100_000
                    + int(round(snr_db * 100)) * 100
                    + cfg_idx
                )

                for variant, item in loaded.items():
                    set_seed(feedback_seed)
                    output = item["model"](
                        H_dl,
                        H_ul,
                        snr_tensor,
                        d_f,
                        d_a,
                        cfg_idx,
                    )
                    W = output[0] if isinstance(output, (tuple, list)) else output
                    if not torch.is_tensor(W):
                        raise RuntimeError(
                            f"{variant} returned unsupported output type "
                            f"{type(output)!r}"
                        )
                    rate_values = per_sample_sum_rate(
                        W,
                        H_dl,
                        noise_power,
                    )
                    rates[variant][snr_db][(d_f, d_a)].append(
                        rate_values.detach().cpu()
                    )

    summary_rows: List[dict] = []
    sample_rows: List[dict] = []

    for snr_db in snr_list:
        full_per_sample_avg = None

        for variant, item in loaded.items():
            allocation_tensors: Dict[Tuple[int, int], torch.Tensor] = {
                allocation: torch.cat(
                    rates[variant][snr_db][allocation]
                )
                for allocation in ALLOCATION_GRID
            }

            stacked = torch.stack(
                [allocation_tensors[a] for a in ALLOCATION_GRID],
                dim=1,
            )
            per_sample_grid_average = stacked.mean(dim=1)

            avg_mean, avg_se, avg_low, avg_high = mean_se_ci95(
                per_sample_grid_average
            )

            if variant == "full":
                full_per_sample_avg = per_sample_grid_average
                degradation_pct = 0.0
                paired_delta_mean = 0.0
                paired_delta_se = 0.0
            else:
                assert full_per_sample_avg is not None
                paired_delta = (
                    per_sample_grid_average - full_per_sample_avg
                )
                paired_delta_mean, paired_delta_se, _, _ = mean_se_ci95(
                    paired_delta
                )
                full_mean = float(full_per_sample_avg.mean().item())
                degradation_pct = (
                    100.0 * (avg_mean - full_mean) / full_mean
                )

            for allocation in ALLOCATION_GRID:
                values = allocation_tensors[allocation]
                mean, se, ci_low, ci_high = mean_se_ci95(values)
                summary_rows.append(
                    {
                        "variant": variant,
                        "variant_label": DISPLAY_NAMES[variant],
                        "checkpoint_step": item["step"],
                        "checkpoint_path": str(item["path"]),
                        "snr_db": float(snr_db),
                        "allocation_df": allocation[0],
                        "allocation_da": allocation[1],
                        "num_samples": int(values.numel()),
                        "sum_rate_mean": mean,
                        "sum_rate_se": se,
                        "sum_rate_ci95_low": ci_low,
                        "sum_rate_ci95_high": ci_high,
                        "five_config_avg_mean": avg_mean,
                        "five_config_avg_se": avg_se,
                        "five_config_avg_ci95_low": avg_low,
                        "five_config_avg_ci95_high": avg_high,
                        "degradation_vs_full_pct": degradation_pct,
                        "paired_delta_vs_full_mean": paired_delta_mean,
                        "paired_delta_vs_full_se": paired_delta_se,
                    }
                )

            for sample_index in range(stacked.shape[0]):
                sample_row = {
                    "variant": variant,
                    "variant_label": DISPLAY_NAMES[variant],
                    "snr_db": float(snr_db),
                    "sample_index": sample_index,
                    "five_config_avg": float(
                        per_sample_grid_average[sample_index].item()
                    ),
                }
                for allocation_index, allocation in enumerate(
                    ALLOCATION_GRID
                ):
                    sample_row[
                        f"rate_df{allocation[0]}_da{allocation[1]}"
                    ] = float(
                        stacked[sample_index, allocation_index].item()
                    )
                sample_rows.append(sample_row)

    return summary_rows, sample_rows


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary_rows: Sequence[Mapping], snr_db: float) -> None:
    selected = [
        row for row in summary_rows
        if float(row["snr_db"]) == float(snr_db)
    ]

    by_variant: Dict[str, Dict[Tuple[int, int], Mapping]] = {}
    for row in selected:
        by_variant.setdefault(row["variant"], {})[
            (row["allocation_df"], row["allocation_da"])
        ] = row

    print("\n" + "=" * 130)
    print(
        f"Ablation summary at operating SNR = {snr_db:g} dB "
        f"(same feedback and downlink SNR)"
    )
    print("=" * 130)
    print(
        f"{'Variant':<34} | "
        + " | ".join(f"{str(a):>11}" for a in ALLOCATION_GRID)
        + f" | {'Avg':>8} | {'Change':>9}"
    )
    print("-" * 130)

    for variant in (
        "full",
        "no_condition",
        "no_anchor_injection",
        "no_cross_user_aggregation",
    ):
        rows = by_variant[variant]
        first = rows[ALLOCATION_GRID[0]]
        values = [
            rows[allocation]["sum_rate_mean"]
            for allocation in ALLOCATION_GRID
        ]
        print(
            f"{DISPLAY_NAMES[variant]:<34} | "
            + " | ".join(f"{value:>11.3f}" for value in values)
            + f" | {first['five_config_avg_mean']:>8.3f}"
            + f" | {first['degradation_vs_full_pct']:>8.1f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate completed K4/Nsc128/D128 ablation checkpoints."
    )
    parser.add_argument("--snr-list", default="25")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--full-root", default="runs_deployzf_v1")
    parser.add_argument(
        "--ablation-root",
        default="runs_ablation_deployzf_v1",
    )
    parser.add_argument("--full-checkpoint", default=None)
    parser.add_argument(
        "--output-dir",
        default="runs_evaluation/ablation_k4_d128",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("Batch size and number of batches must be positive.")

    snr_list = parse_float_csv(args.snr_list)

    torch.set_float32_matmul_precision("high")
    device = setup_gpu()
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("CUDA is unavailable.")

    loaded = load_models(
        full_root=args.full_root,
        ablation_root=args.ablation_root,
        full_checkpoint=args.full_checkpoint,
        device=device,
    )

    print("=" * 100)
    print("[Ablation evaluator]")
    print(f"[SNRs] {snr_list}")
    print(
        f"[Samples] {args.num_batches} x {args.batch_size} "
        f"= {args.num_batches * args.batch_size}"
    )
    for variant, item in loaded.items():
        print(
            f"[Loaded] {DISPLAY_NAMES[variant]} | "
            f"step={item['step']} | {item['path']}"
        )

    summary_rows, sample_rows = evaluate(
        loaded=loaded,
        snr_list=snr_list,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "ablation_summary.csv"
    samples_path = output_dir / "ablation_samples.csv.gz"
    json_path = output_dir / "ablation_results.json"

    write_csv(summary_path, summary_rows)
    write_gzip_csv(samples_path, sample_rows)
    json_path.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for snr_db in snr_list:
        print_table(summary_rows, snr_db)

    print("\n[Saved]")
    print(f"  {summary_path}")
    print(f"  {samples_path}")
    print(f"  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())