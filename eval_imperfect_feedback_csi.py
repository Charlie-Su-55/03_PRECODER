#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pilot-aided imperfect feedback-link CSI evaluation.

Physical feedback transmission always uses the true H_ul.
The BS only has access to an LS estimate H_ul_est obtained from orthogonal UL pilots.

Main protocol:
    fixed feedback-data SNR
    fixed downlink SNR
    sweep pilot SNR / feedback-link CSI quality

The script reuses checkpoint loading and metric utilities from baseline_evaluate.py
without modifying the frozen headline evaluator.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
from utils import setup_gpu
from utils.channel_estimation import estimate_feedback_channel_ls


SUPPORTED_MODELS = {"proposed", "csinet", "swin"}


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_pilot_snrs(text: str) -> List[float]:
    values = []
    for item in parse_csv(text):
        if item.lower() in {"inf", "+inf", "perfect"}:
            values.append(float("inf"))
        else:
            values.append(float(item))
    if not values:
        raise ValueError("At least one pilot SNR must be specified.")
    return values


def resolve_proposed_allocations(k: int, nsc: int, budget: int, selection: str) -> str:
    if selection != "recommended":
        return selection

    if (k, nsc, budget) == (8, 128, 128):
        return "16,0;4,96;0,128"

    if (k, nsc, budget) == (16, 256, 256):
        return "16,0;4,192;0,256"

    raise ValueError(
        f"No recommended allocation set is defined for K={k}, Nsc={nsc}, Dtot={budget}. "
        "Please pass --allocations explicitly."
    )


def mean_se_ci95(values: torch.Tensor) -> Tuple[float, float, float, float]:
    values = values.detach().float().cpu().flatten()

    if values.numel() == 0:
        return math.nan, math.nan, math.nan, math.nan

    if torch.isneginf(values).all():
        return float("-inf"), 0.0, float("-inf"), float("-inf")

    return base_eval.mean_se_ci95(values)


def forward_proposed_with_estimated_csi(
    model: torch.nn.Module,
    H_dl: torch.Tensor,
    H_ul: torch.Tensor,
    H_ul_est: torch.Tensor,
    feedback_snr: torch.Tensor,
    d_f: int,
    d_a: int,
    cfg_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run proposed deployment path while exposing H_hat for evaluation."""

    captured: Dict[str, torch.Tensor] = {}

    if not hasattr(model, "h_estimator"):
        raise RuntimeError("Proposed model has no h_estimator module.")

    def capture_h_raw(_module, _inputs, output):
        captured["h_raw"] = output

    handle = model.h_estimator.register_forward_hook(capture_h_raw)

    try:
        output = model(H_dl, H_ul, feedback_snr, d_f, d_a, cfg_idx, H_ul_est=H_ul_est)
    finally:
        handle.remove()

    W, H_hat = base_eval.extract_output("proposed", output)

    if H_hat is None:
        h_raw = captured.get("h_raw")
        if h_raw is None:
            raise RuntimeError("Could not capture proposed h_estimator output.")
        H_hat = base_eval.reconstruct_proposed_surrogate(model, h_raw, H_dl.shape[0])

    return W, H_hat


def add_store(
    stores: Dict[Tuple[str, int, int, float, str], List[torch.Tensor]],
    method_key: str,
    d_f: int,
    d_a: int,
    pilot_snr: float,
    metric: str,
    values: torch.Tensor,
) -> None:
    key = (method_key, d_f, d_a, pilot_snr, metric)
    stores.setdefault(key, []).append(values.detach().cpu())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Tuple[List[dict], List[dict], dict]:
    torch.set_float32_matmul_precision("high")

    device = setup_gpu()

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

    model_keys = parse_csv(args.models)

    for model_key in model_keys:
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model_key}'. Allowed: {sorted(SUPPORTED_MODELS)}")

    args.allocations = resolve_proposed_allocations(args.k, args.nsc, args.budget, args.allocations)

    methods = base_eval.load_methods_for_scenario(
        scenario, model_keys, args, device, scenario_count=1
    )

    channel_gen = base_eval.build_channel_generator(scenario, device)
    pilot_snrs = parse_pilot_snrs(args.pilot_snr)
    pilot_length = args.pilot_length if args.pilot_length is not None else scenario.k_users

    feedback_snr_tensor = torch.tensor(
        float(args.feedback_snr), dtype=torch.float32, device=device
    )
    downlink_noise_power = 10.0 ** (-float(args.downlink_snr) / 10.0)

    stores: Dict[Tuple[str, int, int, float, str], List[torch.Tensor]] = {}
    ce_stores: Dict[float, List[torch.Tensor]] = {snr: [] for snr in pilot_snrs}

    print("=" * 110)
    print("[Imperfect feedback-link CSI evaluation]")
    print(f"[Scenario]       {scenario.scenario_id}")
    print(f"[Models]         {model_keys}")
    print(f"[Allocations]    {args.allocations}")
    print(f"[Pilot SNR]      {pilot_snrs}")
    print(f"[Pilot length]   Tp={pilot_length}")
    print(f"[Feedback SNR]   {args.feedback_snr:.1f} dB")
    print(f"[Downlink SNR]   {args.downlink_snr:.1f} dB")
    print(f"[Samples]        {args.num_batches} x {args.batch_size} = {args.num_batches * args.batch_size}")
    print("=" * 110)

    for method in methods:
        if method.model is not None:
            method.model.eval()

        print(
            f"[Loaded] {method.display_name}, step={method.checkpoint_step}, "
            f"allocations={method.allocations}"
        )

    progress = tqdm(range(args.num_batches), desc=f"imperfect-csi-{scenario.scenario_id}", dynamic_ncols=True)

    for batch_index in progress:
        channel_seed = args.seed + base_eval.stable_int(scenario.scenario_id) + batch_index
        base_eval.set_seed(channel_seed)

        H_dl, H_ul = channel_gen.generate_batch_data(args.batch_size)
        H_dl = H_dl.to(device)
        H_ul = H_ul.to(device)

        for pilot_snr in pilot_snrs:
            # Use the same underlying pilot-noise realization at all pilot-SNR points.
            # Only the noise scaling changes.
            ce_seed = args.seed + 200_000_000 + batch_index
            base_eval.set_seed(ce_seed)

            ce = estimate_feedback_channel_ls(
                H_ul,
                pilot_snr_db=pilot_snr,
                pilot_length=pilot_length,
            )
            H_ul_est = ce.H_hat

            if math.isinf(pilot_snr):
                ce_nmse_values = torch.full(
                    (H_ul.shape[0],),
                    float("-inf"),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                ce_nmse_values = base_eval.per_sample_nmse_db(H_ul_est, H_ul)

            ce_stores[pilot_snr].append(ce_nmse_values.detach().cpu())

            for method in methods:
                for d_f, d_a in method.allocations:
                    # Keep feedback-data AWGN identical across pilot-SNR conditions.
                    feedback_seed = args.seed + 100_000_000 + batch_index * 100_000
                    base_eval.set_seed(feedback_seed)

                    if method.model_key == "proposed":
                        all_grid = [tuple(map(int, allocation)) for allocation in method.config.GAMMA_GRID]
                        cfg_idx = all_grid.index((d_f, d_a))

                        W, H_hat = forward_proposed_with_estimated_csi(
                            method.model,
                            H_dl,
                            H_ul,
                            H_ul_est,
                            feedback_snr_tensor,
                            d_f,
                            d_a,
                            cfg_idx,
                        )

                    elif method.model_key in {"csinet", "swin"}:
                        output = method.model(
                            H_dl,
                            H_ul,
                            feedback_snr_tensor,
                            H_ul_est=H_ul_est,
                        )
                        W, H_hat = base_eval.extract_output(method.model_key, output)

                    else:
                        raise ValueError(method.model_key)

                    rate_values = base_eval.per_sample_sum_rate(
                        W,
                        H_dl,
                        downlink_noise_power,
                    )

                    add_store(
                        stores,
                        method.method_key,
                        d_f,
                        d_a,
                        pilot_snr,
                        "sum_rate",
                        rate_values,
                    )

                    if H_hat is not None:
                        add_store(
                            stores,
                            method.method_key,
                            d_f,
                            d_a,
                            pilot_snr,
                            "representation_nmse_db",
                            base_eval.per_sample_nmse_db(H_hat, H_dl),
                        )

    method_by_key = {method.method_key: method for method in methods}
    summary_rows: List[dict] = []
    sample_rows: List[dict] = []

    conditions = sorted(
        {key[:4] for key in stores},
        key=lambda x: (x[3], x[0], x[1], x[2]),
    )

    ce_all = {
        pilot_snr: torch.cat(values)
        for pilot_snr, values in ce_stores.items()
    }

    for method_key, d_f, d_a, pilot_snr in conditions:
        method = method_by_key[method_key]

        metric_values: Dict[str, torch.Tensor] = {}
        for key, batch_values in stores.items():
            if key[:4] == (method_key, d_f, d_a, pilot_snr):
                metric_values[key[4]] = torch.cat(batch_values)

        rate_values = metric_values["sum_rate"]
        rate_mean, rate_se, rate_low, rate_high = mean_se_ci95(rate_values)

        ce_values = ce_all[pilot_snr]
        ce_mean, ce_se, ce_low, ce_high = mean_se_ci95(ce_values)

        row = {
            "scenario_id": scenario.scenario_id,
            "method_key": method.model_key,
            "method_label": method.display_name,
            "checkpoint_step": method.checkpoint_step,
            "checkpoint_path": method.checkpoint_path,
            "allocation_df": d_f,
            "allocation_da": d_a,
            "pilot_length": pilot_length,
            "pilot_snr_db": pilot_snr,
            "feedback_csi_nmse_db_mean": ce_mean,
            "feedback_csi_nmse_db_se": ce_se,
            "feedback_csi_nmse_db_ci95_low": ce_low,
            "feedback_csi_nmse_db_ci95_high": ce_high,
            "feedback_snr_db": args.feedback_snr,
            "downlink_snr_db": args.downlink_snr,
            "num_samples": int(rate_values.numel()),
            "sum_rate_mean": rate_mean,
            "sum_rate_se": rate_se,
            "sum_rate_ci95_low": rate_low,
            "sum_rate_ci95_high": rate_high,
            "seed": args.seed,
        }

        if "representation_nmse_db" in metric_values:
            mean, se, low, high = mean_se_ci95(metric_values["representation_nmse_db"])
            row["representation_nmse_db_mean"] = mean
            row["representation_nmse_db_se"] = se
            row["representation_nmse_db_ci95_low"] = low
            row["representation_nmse_db_ci95_high"] = high

        summary_rows.append(row)

        if not args.no_save_samples:
            for sample_index in range(rate_values.numel()):
                sample_row = {
                    "scenario_id": scenario.scenario_id,
                    "method_key": method.model_key,
                    "method_label": method.display_name,
                    "allocation_df": d_f,
                    "allocation_da": d_a,
                    "pilot_length": pilot_length,
                    "pilot_snr_db": pilot_snr,
                    "feedback_csi_nmse_db": float(ce_values[sample_index].item()),
                    "feedback_snr_db": args.feedback_snr,
                    "downlink_snr_db": args.downlink_snr,
                    "sample_index": sample_index,
                    "sum_rate": float(rate_values[sample_index].item()),
                }

                if "representation_nmse_db" in metric_values:
                    sample_row["representation_nmse_db"] = float(
                        metric_values["representation_nmse_db"][sample_index].item()
                    )

                sample_rows.append(sample_row)

    metadata = {
        "experiment": "pilot_aided_imperfect_feedback_link_csi",
        "scenario": {
            "k_users": scenario.k_users,
            "antennas": scenario.antennas,
            "subcarriers": scenario.subcarriers,
            "feedback_budget": scenario.feedback_budget,
            "carrier_freq": scenario.carrier_freq,
            "speed": scenario.speed,
        },
        "protocol": {
            "pilot_estimator": "orthogonal_DFT_pilots_LS",
            "pilot_length": pilot_length,
            "pilot_snr_db": pilot_snrs,
            "feedback_snr_db": args.feedback_snr,
            "downlink_snr_db": args.downlink_snr,
            "physical_feedback_channel": "true_H_ul",
            "receiver_channel_knowledge": "pilot_estimated_H_ul",
            "same_channel_realizations_across_methods": True,
            "same_pilot_noise_draw_across_pilot_snr": True,
            "same_feedback_noise_seed_across_pilot_snr": True,
        },
        "methods": [
            {
                "model_key": method.model_key,
                "method_label": method.display_name,
                "checkpoint_step": method.checkpoint_step,
                "checkpoint_path": method.checkpoint_path,
                "allocations": list(method.allocations),
            }
            for method in methods
        ],
        "seed": args.seed,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
    }

    return summary_rows, sample_rows, metadata


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print("Imperfect feedback-link CSI summary")
    print("=" * 132)
    print(
        f"{'method':<24} | {'allocation':>11} | {'pilot':>8} | "
        f"{'UL-CSI NMSE':>12} | {'rate':>9} | {'SE':>7}"
    )
    print("-" * 132)

    for row in rows:
        allocation = f"({row['allocation_df']},{row['allocation_da']})"

        if math.isinf(float(row["pilot_snr_db"])):
            pilot_text = "perfect"
            ce_text = "-inf"
        else:
            pilot_text = f"{row['pilot_snr_db']:.1f}"
            ce_text = f"{row['feedback_csi_nmse_db_mean']:.3f}"

        print(
            f"{row['method_label']:<24} | {allocation:>11} | {pilot_text:>8} | "
            f"{ce_text:>12} | {row['sum_rate_mean']:>9.3f} | {row['sum_rate_se']:>7.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pilot-aided imperfect feedback-link CSI evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--nsc", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--antennas", type=int, default=32)
    parser.add_argument("--carrier-freq", type=float, default=3.5e9)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--total-power", type=float, default=1.0)
    parser.add_argument("--num-subbands", type=int, default=4)

    parser.add_argument("--models", default="proposed,csinet,swin")
    parser.add_argument(
        "--allocations",
        default="recommended",
        help="Proposed allocation selection. 'recommended' uses pure FDMA, selected hybrid, and pure shared.",
    )

    parser.add_argument(
        "--pilot-snr",
        default="-10,-5,0,5,10,20,inf",
        help="Pilot SNR values in dB. Use 'inf' for perfect feedback-link CSI.",
    )
    parser.add_argument("--pilot-length", type=int, default=None)
    parser.add_argument("--feedback-snr", type=float, default=25.0)
    parser.add_argument("--downlink-snr", type=float, default=25.0)

    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--proposal-root", default="runs_deployzf_v1")
    parser.add_argument("--baseline-root", default="runs_baselines")
    parser.add_argument("--baseline-objective", choices=["task", "reconstruction"], default="task")
    parser.add_argument("--proposed-checkpoint", default=None)
    parser.add_argument("--csinet-checkpoint", default=None)
    parser.add_argument("--swin-checkpoint", default=None)
    parser.add_argument("--rzf-reg", type=float, default=1e-3)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--no-save-samples", action="store_true")
    parser.add_argument("--output-dir", default="runs_evaluation/imperfect_feedback_csi")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("batch-size and num-batches must be positive.")

    summary_rows, sample_rows, metadata = evaluate(args)
    print_summary(summary_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_id = f"K{args.k}_Nsc{args.nsc}_Dfb{args.budget}"

    summary_path = output_dir / f"summary_{scenario_id}.csv"
    samples_path = output_dir / f"samples_{scenario_id}.csv.gz"
    metadata_path = output_dir / f"metadata_{scenario_id}.json"

    write_csv(summary_path, summary_rows)

    if not args.no_save_samples:
        write_gzip_csv(samples_path, sample_rows)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n[Saved]")
    print(summary_path)

    if not args.no_save_samples:
        print(samples_path)

    print(metadata_path)


if __name__ == "__main__":
    main()