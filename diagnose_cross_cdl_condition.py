#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose MU-MIMO channel conditioning across CDL-A/B/C/D/E.

For each test CDL profile, generate independent K-user channel realizations
using the current project generator and compute per-subcarrier singular-value
statistics of H in C^{K x Nt}.

The condition number is invariant to the scalar Frobenius normalization used
inside the Perfect-CSI RZF evaluator, so these statistics directly characterize
the spatial conditioning seen by that RZF reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
from utils import CDLChannelGenerator_Feedback, setup_gpu


VALID_CDL = ("A", "B", "C", "D", "E")


def parse_profiles(text: str):
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
    return profiles


def build_generator(args, profile: str, device: torch.device):
    return CDLChannelGenerator_Feedback(
        Nt=args.antennas,
        Nsc=args.nsc,
        carrier_freq=args.carrier_freq,
        speed=args.speed,
        num_users=args.k,
        cdl_model=profile,
    ).to(device)


@torch.no_grad()
def evaluate_profile(args, profile: str, device: torch.device):
    generator = build_generator(args, profile, device)

    log10_cond_all = []
    sigma_max_all = []
    sigma_min_all = []

    progress = tqdm(range(args.num_batches), desc=f"condition-CDL-{profile}", dynamic_ncols=True)

    for batch_idx in progress:
        seed = args.seed + base_eval.stable_int(f"CDL-{profile}") + batch_idx
        base_eval.set_seed(seed)

        H_dl, _ = generator.generate_batch_data(args.batch_size)
        H_dl = H_dl.to(device)

        # [B,K,Nt,Nsc] -> [B,Nsc,K,Nt]
        H_matrix = H_dl.permute(0, 3, 1, 2).contiguous()

        # Singular values: [B,Nsc,min(K,Nt)], descending order.
        singular_values = torch.linalg.svdvals(H_matrix)
        sigma_max = singular_values[..., 0]
        sigma_min = singular_values[..., -1]

        cond = sigma_max / sigma_min.clamp_min(1e-12)
        log10_cond = torch.log10(cond.clamp_min(1.0))

        log10_cond_all.append(log10_cond.flatten().cpu())
        sigma_max_all.append(sigma_max.flatten().cpu())
        sigma_min_all.append(sigma_min.flatten().cpu())

    log10_cond = torch.cat(log10_cond_all)
    sigma_max = torch.cat(sigma_max_all)
    sigma_min = torch.cat(sigma_min_all)

    result = {
        "cdl_model": profile,
        "num_channel_samples": args.batch_size * args.num_batches,
        "num_subcarrier_matrices": int(log10_cond.numel()),
        "mean_log10_condition": float(log10_cond.mean()),
        "median_log10_condition": float(log10_cond.median()),
        "p10_log10_condition": float(torch.quantile(log10_cond, 0.10)),
        "p90_log10_condition": float(torch.quantile(log10_cond, 0.90)),
        "mean_condition": float(torch.pow(10.0, log10_cond).mean()),
        "mean_sigma_max": float(sigma_max.mean()),
        "mean_sigma_min": float(sigma_min.mean()),
        "median_sigma_min": float(sigma_min.median()),
    }

    del generator
    torch.cuda.empty_cache()
    return result


def print_results(results):
    print("\n" + "=" * 122)
    print("Cross-CDL MU-MIMO conditioning")
    print("=" * 122)
    print(
        f"{'CDL':>5} | {'mean log10κ':>12} | {'median log10κ':>14} | "
        f"{'p10':>8} | {'p90':>8} | {'mean σmax':>10} | {'mean σmin':>10}"
    )
    print("-" * 122)

    for row in results:
        print(
            f"{row['cdl_model']:>5} | "
            f"{row['mean_log10_condition']:>12.4f} | "
            f"{row['median_log10_condition']:>14.4f} | "
            f"{row['p10_log10_condition']:>8.4f} | "
            f"{row['p90_log10_condition']:>8.4f} | "
            f"{row['mean_sigma_max']:>10.4f} | "
            f"{row['mean_sigma_min']:>10.4f}"
        )


def save_results(results, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "cross_cdl_condition_summary.csv"
    json_path = output_dir / "cross_cdl_condition_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    metadata = {
        "scenario": {
            "k": args.k,
            "antennas": args.antennas,
            "nsc": args.nsc,
            "carrier_freq": args.carrier_freq,
            "speed": args.speed,
        },
        "profiles": args.profiles,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_samples_per_profile": args.batch_size * args.num_batches,
        "seed": args.seed,
        "results": results,
    }

    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[Saved]")
    print(csv_path)
    print(json_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose MU channel conditioning across CDL profiles.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--antennas", type=int, default=32)
    parser.add_argument("--nsc", type=int, default=128)
    parser.add_argument("--carrier-freq", type=float, default=3.5e9)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--profiles", type=str, default="A,B,C,D,E")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output-dir", type=str, default="runs_evaluation/cross_cdl/condition_diagnostic")
    return parser


def main():
    args = build_parser().parse_args()
    profiles = parse_profiles(args.profiles)

    torch.set_float32_matmul_precision("high")
    device = setup_gpu()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    print("=" * 100)
    print("[Cross-CDL conditioning diagnostic]")
    print(f"[Scenario] K={args.k}, Nt={args.antennas}, Nsc={args.nsc}")
    print(f"[Profiles] {profiles}")
    print(f"[Samples/profile] {args.batch_size * args.num_batches}")
    print("=" * 100)

    results = [evaluate_profile(args, profile, device) for profile in profiles]
    print_results(results)
    save_results(results, args)


if __name__ == "__main__":
    main()