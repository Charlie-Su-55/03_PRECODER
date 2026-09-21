#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate a trained TUB-style VQ-SSCC checkpoint with a common RZF lambda sweep."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
from models.baseline_common import closed_form_rzf
from train_tub_vqsscc_baseline import TUBVQSSCC, channel_to_images, images_to_channel
from utils import setup_gpu


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def mean_se(values):
    x = torch.cat(values).float()
    mean = x.mean().item()
    se = (x.std(unbiased=True) / math.sqrt(x.numel())).item()
    return mean, se


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lambdas", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output-dir", default="runs_evaluation/tub_vqsscc_lambda_sweep")
    args = parser.parse_args()

    device = setup_gpu()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = checkpoint["config"]

    k = int(cfg["k"])
    nt = int(cfg["nt"])
    nsc = int(cfg["nsc"])
    nc = int(cfg["nc"])
    budget = int(cfg["budget"])
    base = int(cfg["base"])
    c_lat = int(cfg["c_lat"])
    num_embeddings = int(cfg["num_embeddings"])
    keep_k = int(cfg["keep_k"])
    total_power = float(cfg["total_power"])
    dl_snr = float(cfg["dl_snr"])

    model = TUBVQSSCC(
        base=base,
        c_lat=c_lat,
        num_embeddings=num_embeddings,
        keep_k=keep_k,
    ).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    model.vq.initialized = True
    model.eval()

    scenario = base_eval.Scenario(
        k_users=k,
        subcarriers=nsc,
        feedback_budget=budget,
        antennas=nt,
        carrier_freq=3.5e9,
        speed=1.0,
        total_power=total_power,
        num_subbands=4,
    )
    generator = base_eval.build_channel_generator(scenario, device)

    lambdas = parse_float_list(args.lambdas)
    dl_noise = 10.0 ** (-dl_snr / 10.0)

    sscc_rates = {lam: [] for lam in lambdas}
    ae_rates = {lam: [] for lam in lambdas}
    sscc_nmses = []

    for batch_idx in tqdm(range(args.num_batches), desc="TUB-SSCC lambda sweep", dynamic_ncols=True):
        base_eval.set_seed(args.seed + batch_idx)
        H_dl, _ = generator.generate_batch_data(args.batch_size)
        H_dl = H_dl.to(device)

        B, K = H_dl.shape[:2]
        x = channel_to_images(H_dl, nc)

        x_sscc, _ = model.sparse_forward(x, random_mask=False)
        H_sscc = images_to_channel(x_sscc, B, K, nsc)
        sscc_nmses.append(base_eval.per_sample_nmse_db(H_sscc, H_dl).cpu())

        x_ae = model.autoencoder(x)
        H_ae = images_to_channel(x_ae, B, K, nsc)

        for lam in lambdas:
            W_sscc = closed_form_rzf(
                H_sscc,
                regularization=lam,
                total_power=total_power,
            )
            W_ae = closed_form_rzf(
                H_ae,
                regularization=lam,
                total_power=total_power,
            )

            sscc_rates[lam].append(
                base_eval.per_sample_sum_rate(W_sscc, H_dl, dl_noise).cpu()
            )
            ae_rates[lam].append(
                base_eval.per_sample_sum_rate(W_ae, H_dl, dl_noise).cpu()
            )

    nmse_mean = torch.cat(sscc_nmses).mean().item()

    rows = []
    best_lambda = None
    best_rate = -float("inf")

    print("\n" + "=" * 88)
    print(
        f"TUB VQ-SSCC | checkpoint={checkpoint_path.name} | "
        f"N={args.batch_size * args.num_batches} | NMSE={nmse_mean:.3f} dB"
    )
    print("=" * 88)
    print(f"{'lambda':>12} {'SSCC rate':>14} {'SE':>10} {'Full-AE rate':>16} {'SE':>10}")
    print("-" * 88)

    for lam in lambdas:
        sscc_mean, sscc_se = mean_se(sscc_rates[lam])
        ae_mean, ae_se = mean_se(ae_rates[lam])

        rows.append({
            "lambda": lam,
            "sscc_rate_mean": sscc_mean,
            "sscc_rate_se": sscc_se,
            "full_ae_rate_mean": ae_mean,
            "full_ae_rate_se": ae_se,
            "sscc_nmse_db": nmse_mean,
            "num_samples": args.batch_size * args.num_batches,
            "checkpoint": str(checkpoint_path),
        })

        print(
            f"{lam:>12.4g} "
            f"{sscc_mean:>14.3f} "
            f"{sscc_se:>10.3f} "
            f"{ae_mean:>16.3f} "
            f"{ae_se:>10.3f}"
        )

        if sscc_mean > best_rate:
            best_rate = sscc_mean
            best_lambda = lam

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tub_vqsscc_lambda_sweep.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 88)
    print(f"Best SSCC lambda = {best_lambda:g}")
    print(f"Best SSCC rate   = {best_rate:.3f} bps/Hz")
    print(f"Saved: {output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()