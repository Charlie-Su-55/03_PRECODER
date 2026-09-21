#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate robustness to imperfect DL CSI estimation at the UE.

Physical channel:
    H_dl_true

UE observation:
    H_dl_est = LS estimate from orthogonal DL pilots

Feedback:
    Neural methods encode H_dl_est.
    The physical feedback channel still uses H_ul and fixed feedback SNR.

Final KPI:
    Precoding sum rate is always evaluated on H_dl_true.

The DL-pilot overhead is reported separately and is not deducted from D_tot,
because D_tot denotes the UL feedback-payload budget.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
from utils import setup_gpu


def parse_snr_list(text: str):
    values = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        values.append(math.inf if item in {"inf", "infinity", "perfect"} else float(item))
    if not values:
        raise ValueError("Empty DL-pilot SNR list.")
    return values


def dft_pilots(num_ports: int, pilot_length: int, device: torch.device):
    if pilot_length < num_ports:
        raise ValueError(f"pilot_length={pilot_length} must be >= Nt={num_ports}.")
    n = torch.arange(num_ports, device=device, dtype=torch.float32).unsqueeze(1)
    t = torch.arange(pilot_length, device=device, dtype=torch.float32).unsqueeze(0)
    phase = -2.0 * math.pi * n * t / pilot_length
    return torch.exp(1j * phase).to(torch.complex64)


@torch.no_grad()
def estimate_dl_channel_ls(H_true: torch.Tensor, pilot_snr_db: float, pilot_length: int):
    if math.isinf(pilot_snr_db):
        return H_true.clone()

    B, K, Nt, Nsc = H_true.shape
    pilots = dft_pilots(Nt, pilot_length, H_true.device)

    # H_true [B,K,Nt,Nsc], pilots [Nt,Tp]
    # Y [B,K,Nsc,Tp]
    Y_clean = torch.einsum("bkns,nt->bkst", H_true, pilots)

    n0 = 10.0 ** (-float(pilot_snr_db) / 10.0)
    sigma = math.sqrt(n0 / 2.0)
    noise = sigma * (
        torch.randn(B, K, Nsc, pilot_length, device=H_true.device)
        + 1j * torch.randn(B, K, Nsc, pilot_length, device=H_true.device)
    )
    Y = Y_clean + noise.to(H_true.dtype)

    H_hat = torch.einsum("bkst,nt->bkns", Y, pilots.conj()) / pilot_length
    return H_hat


def mean_se(values: torch.Tensor):
    values = values.detach().float().cpu().flatten()
    mean = float(values.mean())
    se = 0.0 if values.numel() <= 1 else float(values.std(unbiased=True) / math.sqrt(values.numel()))
    return mean, se


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = base_eval.build_parser()
    group = parser.add_argument_group("Imperfect DL CSI")
    group.add_argument("--dl-pilot-snr-list", default="-20,-15,-10,-5,0,5,10,inf")
    group.add_argument("--dl-pilot-length", type=int, default=None)
    return parser


@torch.no_grad()
def main():
    args = build_parser().parse_args()

    scenarios = base_eval.parse_scenarios(args)
    if len(scenarios) != 1:
        raise ValueError("Use exactly one scenario.")
    scenario = scenarios[0]

    model_keys = [base_eval.normalize_model_name(x) for x in base_eval.parse_csv(args.models)]
    model_keys = list(dict.fromkeys(model_keys))
    pilot_snrs = parse_snr_list(args.dl_pilot_snr_list)
    pilot_length = args.dl_pilot_length or scenario.antennas

    if pilot_length < scenario.antennas:
        raise ValueError("For orthogonal LS pilots, pilot length must be >= number of BS antennas.")

    feedback_snr = float(base_eval.parse_float_csv(args.feedback_snr)[0])
    downlink_snr = float(args.downlink_snr)
    downlink_noise_power = 10.0 ** (-downlink_snr / 10.0)

    torch.set_float32_matmul_precision("high")
    device = setup_gpu()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    methods = base_eval.load_methods_for_scenario(
        scenario, model_keys, args, device, scenario_count=1
    )
    generator = base_eval.build_channel_generator(scenario, device)

    rate_store = {}
    ce_nmse_store = {snr: [] for snr in pilot_snrs}
    sample_rows = []

    def key(method, allocation, pilot_snr):
        return method.method_key, allocation[0], allocation[1], pilot_snr

    print("=" * 108)
    print("Imperfect DL-CSI evaluation")
    print(f"K={scenario.k_users}, Nt={scenario.antennas}, Nsc={scenario.subcarriers}, Dtot={scenario.feedback_budget}")
    print(f"DL pilot length={pilot_length}, feedback SNR={feedback_snr:g} dB, DL data SNR={downlink_snr:g} dB")
    print(f"DL pilot SNRs={pilot_snrs}")
    print("=" * 108)

    for batch_idx in tqdm(range(args.num_batches), desc="imperfect-dl-csi", dynamic_ncols=True):
        channel_seed = args.seed + base_eval.stable_int(scenario.scenario_id) + batch_idx
        base_eval.set_seed(channel_seed)

        H_dl_true, H_ul = generator.generate_batch_data(args.batch_size)
        H_dl_true = H_dl_true.to(device)
        H_ul = H_ul.to(device)

        perfect_cache = {}

        for pilot_idx, pilot_snr in enumerate(pilot_snrs):
            ce_seed = args.seed + 30_000_000 + batch_idx * 10_000 + pilot_idx
            base_eval.set_seed(ce_seed)
            H_dl_est = estimate_dl_channel_ls(H_dl_true, pilot_snr, pilot_length)

            ce_nmse = base_eval.per_sample_nmse_db(H_dl_est, H_dl_true)
            ce_nmse_store[pilot_snr].append(ce_nmse.cpu())

            snr_tensor = torch.tensor(feedback_snr, dtype=torch.float32, device=device)

            for method in methods:
                if method.model_key == "perfect_rzf":
                    lam = float(method.rzf_lambda)
                    if lam not in perfect_cache:
                        perfect_cache[lam] = base_eval.closed_form_rzf_reference(
                            H_dl_true, regularization=lam, total_power=scenario.total_power
                        )
                    W = perfect_cache[lam]
                    allocations = ((-1, -1),)
                else:
                    allocations = method.allocations

                for d_f, d_a in allocations:
                    if method.model_key == "proposed":
                        grid = [tuple(map(int, a)) for a in method.config.GAMMA_GRID]
                        cfg_idx = grid.index((d_f, d_a))

                        feedback_seed = args.seed + 100_000_000 + batch_idx * 100_000 + pilot_idx
                        base_eval.set_seed(feedback_seed)
                        W, _ = base_eval.forward_proposed_with_surrogate(
                            method.model, H_dl_est, H_ul, snr_tensor, d_f, d_a, cfg_idx
                        )

                    elif method.model_key in {"csinet", "swin"}:
                        feedback_seed = args.seed + 100_000_000 + batch_idx * 100_000 + pilot_idx
                        base_eval.set_seed(feedback_seed)
                        output = method.model(H_dl_est, H_ul, snr_tensor)
                        W, _ = base_eval.extract_output(method.model_key, output)

                    elif method.model_key == "perfect_rzf":
                        pass

                    else:
                        raise ValueError(method.model_key)

                    rates = base_eval.per_sample_sum_rate(W, H_dl_true, downlink_noise_power)
                    store_key = key(method, (d_f, d_a), pilot_snr)
                    rate_store.setdefault(store_key, []).append(rates.cpu())

                    for i in range(rates.numel()):
                        sample_rows.append({
                            "dl_pilot_snr_db": "inf" if math.isinf(pilot_snr) else pilot_snr,
                            "dl_ce_nmse_db": float(ce_nmse[i]),
                            "method": method.display_name,
                            "allocation_df": d_f,
                            "allocation_da": d_a,
                            "sum_rate": float(rates[i]),
                        })

    summary_rows = []

    for pilot_snr in pilot_snrs:
        ce_values = torch.cat(ce_nmse_store[pilot_snr])
        ce_mean, ce_se = mean_se(ce_values)

        for method in methods:
            allocations = ((-1, -1),) if method.model_key == "perfect_rzf" else method.allocations

            for d_f, d_a in allocations:
                values = torch.cat(rate_store[key(method, (d_f, d_a), pilot_snr)])
                rate_mean, rate_se = mean_se(values)

                summary_rows.append({
                    "dl_pilot_snr_db": "inf" if math.isinf(pilot_snr) else pilot_snr,
                    "dl_ce_nmse_db": ce_mean,
                    "dl_ce_nmse_se": ce_se,
                    "method": method.display_name,
                    "allocation_df": d_f,
                    "allocation_da": d_a,
                    "sum_rate_mean": rate_mean,
                    "sum_rate_se": rate_se,
                    "num_samples": values.numel(),
                })

    print("\n" + "=" * 118)
    print(f"{'Pilot SNR':>10} {'CE NMSE':>10} {'Method':<34} {'Alloc':>10} {'Rate':>10} {'SE':>8}")
    print("-" * 118)

    for row in summary_rows:
        alloc = "reference" if row["allocation_df"] < 0 else f"({row['allocation_df']},{row['allocation_da']})"
        print(
            f"{str(row['dl_pilot_snr_db']):>10} {row['dl_ce_nmse_db']:>10.3f} "
            f"{row['method']:<34} {alloc:>10} {row['sum_rate_mean']:>10.3f} {row['sum_rate_se']:>8.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"summary_imperfect_dl_csi_{scenario.scenario_id}.csv"
    samples_path = output_dir / f"samples_imperfect_dl_csi_{scenario.scenario_id}.csv.gz"

    write_csv(summary_path, summary_rows)
    if not args.no_save_samples:
        write_gzip_csv(samples_path, sample_rows)

    print("\n[Saved]")
    print(summary_path)
    if not args.no_save_samples:
        print(samples_path)


if __name__ == "__main__":
    main()