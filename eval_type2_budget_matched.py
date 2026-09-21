#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
from models.type2_codebook import Type2CodebookBaseline
from utils import CDLChannelGenerator_Feedback, setup_gpu


def feedback_bits(Nt: int, L: int, n_subbands: int, amp_bits: int, phase_bits: int, oversample: int):
    n_beams = Nt * oversample
    beam_bits = math.ceil(math.log2(math.comb(n_beams, L)))
    amp_total = L * n_subbands * amp_bits - amp_bits
    phase_total = L * n_subbands * phase_bits - phase_bits
    return beam_bits + amp_total + phase_total


@torch.no_grad()
def reconstruct_channel(model: Type2CodebookBaseline, H_dl: torch.Tensor):
    H_eq = H_dl.conj()
    H_hat = []

    for k in range(model.K):
        top_idx, coeffs = model.encode_one_ue(H_eq[:, k])
        H_hat.append(model.decode_one_ue(top_idx, coeffs))

    H_hat = torch.stack(H_hat, dim=1).conj()

    # The benchmark channels are normalized to unit average power per UE.
    # Match that known normalization convention for the codebook reconstruction.
    power = H_hat.abs().square().mean(dim=(-2, -1), keepdim=True)
    H_hat = H_hat / torch.sqrt(power + 1e-12)

    return H_hat


def mean_se(x: torch.Tensor):
    x = x.float()
    mean = float(x.mean())
    se = float(x.std(unbiased=True) / math.sqrt(x.numel()))
    return mean, se


def main():
    parser = argparse.ArgumentParser(description="Budget-matched ideal Type-II-inspired digital baseline.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--nsc", type=int, default=128)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--dl-snr", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--num-subbands", type=int, default=4)
    parser.add_argument("--bits-per-symbol", type=int, default=4)
    parser.add_argument("--rzf-lambdas", type=str, default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output-dir", type=str, default="runs_evaluation/type2_budget_matched")
    args = parser.parse_args()

    device = setup_gpu()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    configs = [
        ("Type-I-inspired L=1", 1, 3, 3),
        ("Type-II-inspired L=2", 2, 3, 3),
    ]

    cfg = SimpleNamespace(
        K_USERS=args.k,
        ANTENNAS=args.nt,
        SUBCARRIERS=args.nsc,
        NUM_SUBBANDS=args.num_subbands,
        TOTAL_POWER=1.0,
    )

    generator = CDLChannelGenerator_Feedback(
        Nt=args.nt,
        Nsc=args.nsc,
        carrier_freq=3.5e9,
        speed=1.0,
        num_users=args.k,
        cdl_model="C",
    ).to(device)

    lambdas = [float(x) for x in args.rzf_lambdas.split(",")]
    noise_power = 10.0 ** (-args.dl_snr / 10.0)
    rows = []

    print("=" * 108)
    print("Budget-matched ideal digital Type-II evaluation")
    print(f"K={args.k}, Nt={args.nt}, Nsc={args.nsc}, Dtot={args.budget}, DL SNR={args.dl_snr:.1f} dB")
    print(f"Nominal digital mapping: {args.bits_per_symbol} bits/complex use")
    print("=" * 108)

    for label, L, amp_bits, phase_bits in configs:
        bits_ue = feedback_bits(args.nt, L, args.num_subbands, amp_bits, phase_bits, oversample=4)
        uses_ue = math.ceil(bits_ue / args.bits_per_symbol)
        total_uses = args.k * uses_ue

        if total_uses > args.budget:
            print(f"[SKIP] {label}: {total_uses}>{args.budget} channel uses")
            continue

        print(f"\n[{label}] {bits_ue} bits/UE -> {uses_ue} uses/UE -> {total_uses}/{args.budget} total uses")

        model = Type2CodebookBaseline(
            cfg,
            L=L,
            amp_bits=amp_bits,
            phase_bits=phase_bits,
            oversample=4,
            n_subbands=args.num_subbands,
            verbose=False,
        ).to(device)

        rate_store = {lam: [] for lam in lambdas}
        nmse_store = []

        for batch_idx in tqdm(range(args.num_batches), desc=label, dynamic_ncols=True):
            base_eval.set_seed(args.seed + batch_idx)
            H_dl, _ = generator.generate_batch_data(args.batch_size)
            H_dl = H_dl.to(device)

            H_hat = reconstruct_channel(model, H_dl)
            nmse_store.append(base_eval.per_sample_nmse_db(H_hat, H_dl).cpu())

            for lam in lambdas:
                W = base_eval.closed_form_rzf_reference(
                    H_hat,
                    regularization=lam,
                    total_power=1.0,
                )
                rate_store[lam].append(
                    base_eval.per_sample_sum_rate(W, H_dl, noise_power).cpu()
                )

        nmse = torch.cat(nmse_store)
        nmse_mean, nmse_se = mean_se(nmse)

        best_lambda = None
        best_rate = -float("inf")
        best_se = None

        for lam in lambdas:
            values = torch.cat(rate_store[lam])
            rate_mean, rate_se = mean_se(values)

            if rate_mean > best_rate:
                best_rate = rate_mean
                best_se = rate_se
                best_lambda = lam

        rows.append({
            "method": label,
            "L": L,
            "bits_per_ue": bits_ue,
            "nominal_bits_per_symbol": args.bits_per_symbol,
            "channel_uses_per_ue": uses_ue,
            "total_channel_uses": total_uses,
            "feedback_budget": args.budget,
            "delivery": "ideal_error_free",
            "best_rzf_lambda": best_lambda,
            "sum_rate_mean": best_rate,
            "sum_rate_se": best_se,
            "nmse_db_mean": nmse_mean,
            "nmse_db_se": nmse_se,
            "num_samples": args.batch_size * args.num_batches,
        })

        print(
            f"  best lambda={best_lambda:g}, rate={best_rate:.3f} ± {best_se:.3f}, "
            f"NMSE={nmse_mean:.3f} dB"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"type2_K{args.k}_Nsc{args.nsc}_D{args.budget}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 108)
    print(f"{'Method':<28} {'bits/UE':>8} {'uses':>8} {'lambda':>10} {'rate':>10} {'SE':>8} {'NMSE':>10}")
    print("-" * 108)

    for row in rows:
        print(
            f"{row['method']:<28} {row['bits_per_ue']:>8} {row['total_channel_uses']:>8} "
            f"{row['best_rzf_lambda']:>10g} {row['sum_rate_mean']:>10.3f} "
            f"{row['sum_rate_se']:>8.3f} {row['nmse_db_mean']:>10.3f}"
        )

    print("=" * 108)
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()