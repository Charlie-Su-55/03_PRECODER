#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot cross-CDL evaluation.

Existing checkpoints are trained on CDL-C.
This script evaluates them directly on CDL-A/B/C/D/E without retraining.

The existing baseline_evaluate.py is reused for:
- checkpoint loading
- model forward
- feedback transmission
- Perfect-CSI RZF
- sum-rate / NMSE / power metrics
- statistics

Only the test CDL profile is changed.
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
from utils import CDLChannelGenerator_Feedback, setup_gpu


VALID_CDL_MODELS = {"A", "B", "C", "D", "E"}
TRAINING_CDL_MODEL = "C"


def parse_cdl_models(text: str) -> List[str]:
    models = []

    for item in text.split(","):
        item = item.strip().upper()

        if not item:
            continue

        if item not in VALID_CDL_MODELS:
            raise ValueError(
                f"Unsupported CDL model '{item}'. "
                f"Expected one of {sorted(VALID_CDL_MODELS)}."
            )

        if item not in models:
            models.append(item)

    if not models:
        raise ValueError("At least one CDL model must be selected.")

    return models


def build_cdl_generator(
    scenario: base_eval.Scenario,
    device: torch.device,
    cdl_model: str,
):
    generator = CDLChannelGenerator_Feedback(
        Nt=scenario.antennas,
        Nsc=scenario.subcarriers,
        carrier_freq=scenario.carrier_freq,
        speed=scenario.speed,
        num_users=scenario.k_users,
        cdl_model=cdl_model,
    )

    if hasattr(generator, "to"):
        generator = generator.to(device)

    return generator


@torch.no_grad()
def evaluate_one_cdl(
    scenario: base_eval.Scenario,
    methods: Sequence[base_eval.LoadedMethod],
    cdl_model: str,
    feedback_snr: float,
    downlink_snr: float,
    batch_size: int,
    num_batches: int,
    seed: int,
    extra_metrics: str,
    device: torch.device,
) -> Tuple[List[dict], List[dict]]:
    generator = build_cdl_generator(
        scenario,
        device,
        cdl_model,
    )

    stores: Dict[
        Tuple[str, int, int, str],
        List[torch.Tensor],
    ] = {}

    def add_metric(
        method_key: str,
        d_f: int,
        d_a: int,
        metric: str,
        values: torch.Tensor,
    ) -> None:
        key = (
            method_key,
            d_f,
            d_a,
            metric,
        )

        stores.setdefault(key, []).append(
            values.detach().cpu()
        )

    progress = tqdm(
        range(num_batches),
        desc=f"cross-cdl-{cdl_model}-{scenario.scenario_id}",
        dynamic_ncols=True,
    )

    for batch_index in progress:
        channel_seed = (
            seed
            + base_eval.stable_int(
                f"{scenario.scenario_id}_CDL_{cdl_model}"
            )
            + batch_index
        )

        base_eval.set_seed(channel_seed)

        H_dl, H_ul = generator.generate_batch_data(
            batch_size
        )

        H_dl = H_dl.to(device)
        H_ul = H_ul.to(device)

        downlink_noise_power = 10.0 ** (
            -float(downlink_snr) / 10.0
        )

        perfect_cache: Dict[float, torch.Tensor] = {}

        for method in methods:
            if method.model_key == "perfect_rzf":
                assert method.rzf_lambda is not None

                if method.rzf_lambda not in perfect_cache:
                    perfect_cache[method.rzf_lambda] = (
                        base_eval.closed_form_rzf_reference(
                            H_dl,
                            regularization=method.rzf_lambda,
                            total_power=scenario.total_power,
                        )
                    )

                W = perfect_cache[method.rzf_lambda]
                H_hat = None
                allocations = ((-1, -1),)

            else:
                allocations = method.allocations

            for d_f, d_a in allocations:
                feedback_seed = (
                    seed
                    + 100_000_000
                    + base_eval.stable_int(
                        f"CDL_{cdl_model}"
                    )
                    + batch_index * 100_000
                    + base_eval.stable_int(
                        f"{feedback_snr:.6f}"
                    )
                )

                base_eval.set_seed(feedback_seed)

                if method.model_key == "proposed":
                    all_grid = [
                        tuple(map(int, allocation))
                        for allocation
                        in method.config.GAMMA_GRID
                    ]

                    cfg_idx = all_grid.index(
                        (d_f, d_a)
                    )

                    W, H_hat = (
                        base_eval.forward_proposed_with_surrogate(
                            method.model,
                            H_dl,
                            H_ul,
                            torch.tensor(
                                float(feedback_snr),
                                dtype=torch.float32,
                                device=device,
                            ),
                            d_f,
                            d_a,
                            cfg_idx,
                        )
                    )

                elif method.model_key in {"csinet", "swin"}:
                    output = method.model(
                        H_dl,
                        H_ul,
                        torch.tensor(
                            float(feedback_snr),
                            dtype=torch.float32,
                            device=device,
                        ),
                    )

                    W, H_hat = base_eval.extract_output(
                        method.model_key,
                        output,
                    )

                elif method.model_key == "perfect_rzf":
                    pass

                else:
                    raise ValueError(
                        method.model_key
                    )

                rate_values = base_eval.per_sample_sum_rate(
                    W,
                    H_dl,
                    downlink_noise_power,
                )

                add_metric(
                    method.method_key,
                    d_f,
                    d_a,
                    "sum_rate",
                    rate_values,
                )

                power_values = (
                    base_eval.per_sample_precoder_power(
                        W
                    )
                )

                add_metric(
                    method.method_key,
                    d_f,
                    d_a,
                    "precoder_power",
                    power_values,
                )

                if H_hat is not None:
                    nmse_values = (
                        base_eval.per_sample_nmse_db(
                            H_hat,
                            H_dl,
                        )
                    )

                    add_metric(
                        method.method_key,
                        d_f,
                        d_a,
                        "nmse_db",
                        nmse_values,
                    )

                    if extra_metrics == "full":
                        cos_real, cos_abs = (
                            base_eval.per_sample_cosine_metrics(
                                H_hat,
                                H_dl,
                            )
                        )

                        add_metric(
                            method.method_key,
                            d_f,
                            d_a,
                            "cos_real",
                            cos_real,
                        )

                        add_metric(
                            method.method_key,
                            d_f,
                            d_a,
                            "cos_abs",
                            cos_abs,
                        )

                        condition = (
                            base_eval.per_sample_mean_log10_condition(
                                H_hat
                            )
                        )

                        add_metric(
                            method.method_key,
                            d_f,
                            d_a,
                            "surrogate_log10_condition",
                            condition,
                        )

    method_by_key = {
        method.method_key: method
        for method in methods
    }

    conditions = sorted(
        {
            key[:3]
            for key in stores
        },
        key=lambda value: (
            value[0],
            value[1],
            value[2],
        ),
    )

    summary_rows = []
    sample_rows = []

    for method_key, d_f, d_a in conditions:
        method = method_by_key[method_key]

        metric_values: Dict[
            str,
            torch.Tensor,
        ] = {}

        for key, values in stores.items():
            if key[:3] != (
                method_key,
                d_f,
                d_a,
            ):
                continue

            metric_values[key[3]] = torch.cat(
                values
            )

        rate_values = metric_values["sum_rate"]

        (
            rate_mean,
            rate_se,
            rate_ci_low,
            rate_ci_high,
        ) = base_eval.mean_se_ci95(
            rate_values
        )

        row = {
            "training_cdl_model": TRAINING_CDL_MODEL,
            "test_cdl_model": cdl_model,
            "scenario_id": scenario.scenario_id,
            "k_users": scenario.k_users,
            "antennas": scenario.antennas,
            "subcarriers": scenario.subcarriers,
            "feedback_budget": scenario.feedback_budget,
            "model_key": method.model_key,
            "method_key": method.method_key,
            "method_label": method.display_name,
            "checkpoint_kind": method.checkpoint_kind,
            "checkpoint_step": method.checkpoint_step,
            "checkpoint_path": method.checkpoint_path,
            "allocation_df": d_f,
            "allocation_da": d_a,
            "feedback_snr_db": feedback_snr,
            "downlink_snr_db": downlink_snr,
            "num_samples": int(
                rate_values.numel()
            ),
            "sum_rate_mean": rate_mean,
            "sum_rate_se": rate_se,
            "sum_rate_ci95_low": rate_ci_low,
            "sum_rate_ci95_high": rate_ci_high,
            "rzf_lambda": (
                method.rzf_lambda
                if method.rzf_lambda is not None
                else getattr(
                    method.config,
                    "RZF_REG",
                    math.nan,
                )
            ),
            "seed": seed,
        }

        for metric_name, values in metric_values.items():
            if metric_name == "sum_rate":
                continue

            (
                mean,
                se,
                ci_low,
                ci_high,
            ) = base_eval.mean_se_ci95(
                values
            )

            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_se"] = se
            row[f"{metric_name}_ci95_low"] = ci_low
            row[f"{metric_name}_ci95_high"] = ci_high

        summary_rows.append(row)

        sample_count = int(
            rate_values.numel()
        )

        for sample_index in range(sample_count):
            sample_row = {
                "training_cdl_model": TRAINING_CDL_MODEL,
                "test_cdl_model": cdl_model,
                "scenario_id": scenario.scenario_id,
                "k_users": scenario.k_users,
                "subcarriers": scenario.subcarriers,
                "feedback_budget": scenario.feedback_budget,
                "model_key": method.model_key,
                "method_key": method.method_key,
                "method_label": method.display_name,
                "allocation_df": d_f,
                "allocation_da": d_a,
                "feedback_snr_db": feedback_snr,
                "downlink_snr_db": downlink_snr,
                "sample_index": sample_index,
                "sum_rate": float(
                    rate_values[
                        sample_index
                    ].item()
                ),
            }

            for metric_name, values in metric_values.items():
                if metric_name == "sum_rate":
                    continue

                sample_row[metric_name] = float(
                    values[
                        sample_index
                    ].item()
                )

            sample_rows.append(
                sample_row
            )

    del generator

    return summary_rows, sample_rows


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with gzip.open(
        path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    print()
    print("=" * 130)
    print(
        "Cross-CDL zero-shot evaluation summary"
    )
    print("=" * 130)

    print(
        f"{'CDL':>5} | "
        f"{'method':<28} | "
        f"{'allocation':>12} | "
        f"{'rate':>9} | "
        f"{'SE':>7} | "
        f"{'NMSE dB':>9}"
    )

    print("-" * 130)

    profile_order = {
        name: index
        for index, name
        in enumerate(
            ["A", "B", "C", "D", "E"]
        )
    }

    rows = sorted(
        summary_rows,
        key=lambda row: (
            profile_order[
                row["test_cdl_model"]
            ],
            row["method_label"],
            row["allocation_df"],
            row["allocation_da"],
        ),
    )

    for row in rows:
        allocation = (
            "reference"
            if row["allocation_df"] < 0
            else (
                f"({row['allocation_df']},"
                f"{row['allocation_da']})"
            )
        )

        nmse = row.get(
            "nmse_db_mean",
            math.nan,
        )

        nmse_text = (
            f"{float(nmse):.3f}"
            if math.isfinite(float(nmse))
            else ""
        )

        print(
            f"{row['test_cdl_model']:>5} | "
            f"{row['method_label']:<28} | "
            f"{allocation:>12} | "
            f"{row['sum_rate_mean']:>9.3f} | "
            f"{row['sum_rate_se']:>7.3f} | "
            f"{nmse_text:>9}"
        )


def save_outputs(
    summary_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    scenario: base_eval.Scenario,
    methods: Sequence[base_eval.LoadedMethod],
    cdl_models: Sequence[str],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        args.tag
        if args.tag
        else scenario.scenario_id
    )

    summary_path = (
        output_dir
        / f"summary_cross_cdl_{tag}.csv"
    )

    samples_path = (
        output_dir
        / f"samples_cross_cdl_{tag}.csv.gz"
    )

    metadata_path = (
        output_dir
        / f"metadata_cross_cdl_{tag}.json"
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    if not args.no_save_samples:
        write_gzip_csv(
            samples_path,
            sample_rows,
        )

    metadata = {
        "experiment": (
            "zero_shot_cross_cdl_generalization"
        ),
        "training_cdl_model": TRAINING_CDL_MODEL,
        "test_cdl_models": list(
            cdl_models
        ),
        "scenario": {
            "scenario_id": (
                scenario.scenario_id
            ),
            "k_users": (
                scenario.k_users
            ),
            "antennas": (
                scenario.antennas
            ),
            "subcarriers": (
                scenario.subcarriers
            ),
            "feedback_budget": (
                scenario.feedback_budget
            ),
            "carrier_freq": (
                scenario.carrier_freq
            ),
            "speed": (
                scenario.speed
            ),
            "total_power": (
                scenario.total_power
            ),
            "num_subbands": (
                scenario.num_subbands
            ),
        },
        "feedback_snr_db": (
            args.feedback_snr
        ),
        "downlink_snr_db": (
            args.downlink_snr
        ),
        "batch_size": (
            args.batch_size
        ),
        "num_batches": (
            args.num_batches
        ),
        "num_samples_per_profile": (
            args.batch_size
            * args.num_batches
        ),
        "allocations": (
            args.allocations
        ),
        "models": (
            args.models
        ),
        "rzf_lambdas": (
            args.rzf_lambdas
        ),
        "seed": (
            args.seed
        ),
        "methods": [
            {
                "model_key": (
                    method.model_key
                ),
                "method_label": (
                    method.display_name
                ),
                "checkpoint_path": (
                    method.checkpoint_path
                ),
                "checkpoint_step": (
                    method.checkpoint_step
                ),
                "checkpoint_kind": (
                    method.checkpoint_kind
                ),
                "allocations": [
                    list(allocation)
                    for allocation
                    in method.allocations
                ],
                "rzf_lambda": (
                    method.rzf_lambda
                ),
            }
            for method in methods
        ],
        "protocol": {
            "retraining": False,
            "feedback_link_csi": "perfect",
            "channel_normalization": (
                "legacy per-user "
                "average-power normalization"
            ),
            "H_dl_H_ul_relation": (
                "independent instantaneous "
                "draws from the same test "
                "CDL profile"
            ),
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            default=base_eval.json_default,
        ),
        encoding="utf-8",
    )

    print()
    print("[Saved]")
    print(summary_path)

    if not args.no_save_samples:
        print(samples_path)

    print(metadata_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot CDL-A/B/C/D/E "
            "generalization evaluation."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--nsc",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--budget",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--antennas",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--carrier-freq",
        type=float,
        default=3.5e9,
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--total-power",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num-subbands",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--cdl-models",
        type=str,
        default="A,B,C,D,E",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=(
            "proposed,csinet,"
            "swin,perfect_rzf"
        ),
    )

    parser.add_argument(
        "--allocations",
        type=str,
        default="all",
    )

    parser.add_argument(
        "--feedback-snr",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--downlink-snr",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest"],
        default="best",
    )

    parser.add_argument(
        "--proposal-root",
        type=str,
        default="runs_deployzf_v1",
    )

    parser.add_argument(
        "--baseline-root",
        type=str,
        default="runs_baselines",
    )

    parser.add_argument(
        "--baseline-objective",
        choices=[
            "task",
            "reconstruction",
        ],
        default="task",
    )

    parser.add_argument(
        "--proposed-checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--csinet-checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--swin-checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--rzf-lambdas",
        type=str,
        default="1e-4",
    )

    parser.add_argument(
        "--rzf-reg",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-batches",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--extra-metrics",
        choices=["basic", "full"],
        default="basic",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "runs_evaluation/cross_cdl"
        ),
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--no-save-samples",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.num_batches <= 0:
        raise ValueError(
            "--num-batches must be positive."
        )

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

    cdl_models = parse_cdl_models(
        args.cdl_models
    )

    model_keys = [
        base_eval.normalize_model_name(
            name
        )
        for name
        in base_eval.parse_csv(
            args.models
        )
    ]

    model_keys = list(
        dict.fromkeys(
            model_keys
        )
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    device = setup_gpu()

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA device is required."
        )

    print("=" * 110)
    print(
        "[Cross-CDL zero-shot evaluation]"
    )
    print(
        f"[Training CDL]  "
        f"CDL-{TRAINING_CDL_MODEL}"
    )
    print(
        f"[Test CDL]      "
        f"{['CDL-' + x for x in cdl_models]}"
    )
    print(
        f"[Scenario]      "
        f"{scenario.scenario_id}"
    )
    print(
        f"[Models]        "
        f"{model_keys}"
    )
    print(
        f"[Allocations]   "
        f"{args.allocations}"
    )
    print(
        f"[Feedback SNR]  "
        f"{args.feedback_snr:.1f} dB"
    )
    print(
        f"[Downlink SNR]  "
        f"{args.downlink_snr:.1f} dB"
    )
    print(
        f"[Samples/CDL]   "
        f"{args.batch_size} x "
        f"{args.num_batches} = "
        f"{args.batch_size * args.num_batches}"
    )
    print("=" * 110)

    methods = (
        base_eval.load_methods_for_scenario(
            scenario,
            model_keys,
            args,
            device,
            scenario_count=1,
        )
    )

    for method in methods:
        if method.model_key == "proposed":
            print(
                f"[Loaded] "
                f"{method.display_name}, "
                f"step={method.checkpoint_step}, "
                f"allocations="
                f"{method.allocations}"
            )

        elif method.model_key == "perfect_rzf":
            print(
                f"[Loaded] "
                f"{method.display_name}, "
                f"lambda="
                f"{method.rzf_lambda}"
            )

        else:
            print(
                f"[Loaded] "
                f"{method.display_name}, "
                f"step="
                f"{method.checkpoint_step}"
            )

    all_summary_rows = []
    all_sample_rows = []

    try:
        for index, cdl_model in enumerate(
            cdl_models,
            start=1,
        ):
            print()
            print("=" * 110)
            print(
                f"[{index}/{len(cdl_models)}] "
                f"CDL-C-trained checkpoints "
                f"-> test CDL-{cdl_model}"
            )
            print("=" * 110)

            summary_rows, sample_rows = (
                evaluate_one_cdl(
                    scenario=scenario,
                    methods=methods,
                    cdl_model=cdl_model,
                    feedback_snr=(
                        args.feedback_snr
                    ),
                    downlink_snr=(
                        args.downlink_snr
                    ),
                    batch_size=(
                        args.batch_size
                    ),
                    num_batches=(
                        args.num_batches
                    ),
                    seed=args.seed,
                    extra_metrics=(
                        args.extra_metrics
                    ),
                    device=device,
                )
            )

            all_summary_rows.extend(
                summary_rows
            )

            if not args.no_save_samples:
                all_sample_rows.extend(
                    sample_rows
                )

            torch.cuda.empty_cache()

    finally:
        for method in methods:
            if method.model is not None:
                del method.model

        torch.cuda.empty_cache()

    print_summary(
        all_summary_rows
    )

    save_outputs(
        summary_rows=all_summary_rows,
        sample_rows=all_sample_rows,
        scenario=scenario,
        methods=methods,
        cdl_models=cdl_models,
        args=args,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())