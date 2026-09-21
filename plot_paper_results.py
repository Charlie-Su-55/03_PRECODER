#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_paper_results.py

Create vector PDF figures from the CSV files produced by:
  1) baseline_evaluate.py
  2) evaluate_ablation.py

No GPU or checkpoint loading is required. Each figure is also saved as a PNG
preview next to the PDF.

Examples
--------
python plot_paper_results.py k16-snr \
  --input runs_evaluation/paper_k16/summary_k16_all.csv \
  --output-dir paper/figures

python plot_paper_results.py cross-k \
  --input runs_evaluation/paper_cross_k/summary_cross_k_25db.csv \
  --output-dir paper/figures

python plot_paper_results.py k16-baseline \
  --input runs_evaluation/paper_k16/summary_k16_all.csv \
  --output-dir paper/figures

python plot_paper_results.py ablation \
  --input runs_evaluation/paper_ablation_k4_d128/ablation_summary.csv \
  --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 10


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def i(row: dict[str, str], key: str, default: int = -1) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return int(float(value))


def ci_half_width(row: dict[str, str], prefix: str = "sum_rate") -> float:
    low = f(row, f"{prefix}_ci95_low")
    high = f(row, f"{prefix}_ci95_high")
    if math.isfinite(low) and math.isfinite(high):
        return max(0.0, 0.5 * (high - low))
    se = f(row, f"{prefix}_se")
    return 1.96 * se if math.isfinite(se) else 0.0


def matched_snr(row: dict[str, str]) -> bool:
    fb = f(row, "feedback_snr_db")
    dl = f(row, "downlink_snr_db")
    return math.isfinite(fb) and math.isfinite(dl) and abs(fb - dl) < 1e-9


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {png_path}")


def allocation_label(row: dict[str, str]) -> str:
    return f"({i(row, 'allocation_df')},{i(row, 'allocation_da')})"


def plot_k16_snr(rows: list[dict[str, str]], output_dir: Path, k_users: int) -> None:
    selected = [
        row for row in rows
        if i(row, "k_users") == k_users
        and row.get("model_key") == "proposed"
        and matched_snr(row)
    ]
    if not selected:
        raise RuntimeError(f"No matched-SNR proposed rows found for K={k_users}.")

    allocations = sorted(
        {
            (i(row, "allocation_df"), i(row, "allocation_da"))
            for row in selected
        },
        key=lambda pair: pair[1],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for index, allocation in enumerate(allocations):
        curve = [
            row for row in selected
            if (i(row, "allocation_df"), i(row, "allocation_da")) == allocation
        ]
        curve.sort(key=lambda row: f(row, "feedback_snr_db"))
        snrs = [f(row, "feedback_snr_db") for row in curve]
        means = [f(row, "sum_rate_mean") for row in curve]
        errors = [ci_half_width(row) for row in curve]
        ax.errorbar(
            snrs,
            means,
            yerr=errors,
            marker=markers[index % len(markers)],
            linewidth=1.6,
            markersize=5,
            capsize=2.5,
            label=f"$(D_f,D_a)={allocation}$",
        )

    ax.set_xlabel("Operating SNR (dB)")
    ax.set_ylabel("Average sum rate (bps/Hz)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_k{k_users}_allocation_vs_snr")


def plot_cross_k(
    rows: list[dict[str, str]],
    output_dir: Path,
    operating_snr: float,
) -> None:
    selected = [
        row for row in rows
        if row.get("model_key") == "proposed"
        and matched_snr(row)
        and abs(f(row, "feedback_snr_db") - operating_snr) < 1e-9
    ]
    if not selected:
        raise RuntimeError(
            f"No proposed matched-SNR rows found at {operating_snr:g} dB."
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {4: "o", 8: "s", 16: "^"}

    for k_users in sorted({i(row, "k_users") for row in selected}):
        curve = [row for row in selected if i(row, "k_users") == k_users]
        curve.sort(key=lambda row: f(row, "aircomp_fraction"))

        x = [100.0 * f(row, "aircomp_fraction") for row in curve]
        y = [f(row, "sum_rate_mean") for row in curve]
        err = [ci_half_width(row) for row in curve]

        ax.errorbar(
            x,
            y,
            yerr=err,
            marker=markers.get(k_users, "o"),
            linewidth=1.7,
            markersize=5,
            capsize=2.5,
            label=f"$K={k_users}$",
        )

        best_index = int(np.argmax(y))
        best_row = curve[best_index]

        if best_index == len(curve) - 1:
            offset = (-5, -16)
            horizontal_alignment = "right"
        else:
            offset = (0, 8)
            horizontal_alignment = "center"

        ax.annotate(
            f"best {allocation_label(best_row)}",
            (x[best_index], y[best_index]),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            fontsize=7,
        )

    ax.set_xlabel("AirComp resource fraction (%)")
    ax.set_ylabel("Average sum rate (bps/Hz)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(13, 28.4)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_allocation_cross_k_{operating_snr:g}db")


def first_row(rows: Iterable[dict[str, str]], message: str) -> dict[str, str]:
    rows = list(rows)
    if not rows:
        raise RuntimeError(message)
    return rows[0]


def plot_k16_baseline(
    rows: list[dict[str, str]],
    output_dir: Path,
    k_users: int,
    operating_snr: float,
) -> None:
    selected = [
        row for row in rows
        if i(row, "k_users") == k_users
        and matched_snr(row)
        and abs(f(row, "feedback_snr_db") - operating_snr) < 1e-9
    ]
    if not selected:
        raise RuntimeError(
            f"No matched-SNR rows found for K={k_users} at "
            f"{operating_snr:g} dB."
        )

    proposed = [row for row in selected if row.get("model_key") == "proposed"]
    pure_fdma = first_row(
        (row for row in proposed if i(row, "allocation_da") == 0),
        "Missing proposed pure-FDMA row.",
    )
    pure_aircomp = first_row(
        (row for row in proposed if i(row, "allocation_df") == 0),
        "Missing proposed pure-AirComp row.",
    )
    hybrid_rows = [
        row for row in proposed
        if i(row, "allocation_df") > 0 and i(row, "allocation_da") > 0
    ]
    if not hybrid_rows:
        raise RuntimeError("Missing proposed hybrid rows.")
    best_hybrid = max(hybrid_rows, key=lambda row: f(row, "sum_rate_mean"))

    csinet = first_row(
        (row for row in selected if row.get("model_key") == "csinet"),
        "Missing CsiNet+ row.",
    )
    swin = first_row(
        (row for row in selected if row.get("model_key") == "swin"),
        "Missing Swin row.",
    )

    chosen = [pure_fdma, csinet, pure_aircomp, best_hybrid, swin]
    labels = [
        f"Proposed\nFDMA {allocation_label(pure_fdma)}",
        "CsiNet+\nFDMA",
        f"Proposed\nAirComp {allocation_label(pure_aircomp)}",
        f"Proposed\nHybrid {allocation_label(best_hybrid)}",
        "Swin-CFNet\nFDMA",
    ]
    means = [f(row, "sum_rate_mean") for row in chosen]
    errors = [ci_half_width(row) for row in chosen]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    positions = np.arange(len(chosen))
    ax.bar(positions, means, yerr=errors, capsize=3)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Average sum rate (bps/Hz)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for position, value, error in zip(positions, means, errors):
        ax.text(
            position,
            value + error + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_k{k_users}_baseline_{operating_snr:g}db")




def plot_sota_snr(
    rows: list[dict[str, str]],
    output_dir: Path,
    fixed_dl_snr: float | None,
) -> None:
    """
    Compare fixed proposed hybrid allocations against pure-FDMA baselines.

    K=8  uses proposed hybrid (24,64).
    K=16 uses proposed hybrid (4,192).

    fixed_dl_snr=None:
        matched feedback/downlink operating-SNR sweep.

    fixed_dl_snr=<value>:
        feedback-SNR sweep at a fixed downlink SNR.
    """

    hybrid_allocations = {
        8: (24, 64),
        16: (4, 192),
    }

    for k_users, hybrid_allocation in hybrid_allocations.items():

        if fixed_dl_snr is None:
            selected = [
                row for row in rows
                if i(row, "k_users") == k_users
                and matched_snr(row)
            ]
            suffix = "matched"
            x_label = "Operating SNR (dB)"
        else:
            selected = [
                row for row in rows
                if i(row, "k_users") == k_users
                and abs(
                    f(row, "downlink_snr_db") - fixed_dl_snr
                ) < 1e-9
            ]
            suffix = f"dl{fixed_dl_snr:g}db"
            x_label = "Feedback SNR (dB)"

        if not selected:
            raise RuntimeError(
                f"No valid SOTA-SNR rows found for K={k_users}."
            )

        proposed_rows = [
            row for row in selected
            if row.get("model_key") == "proposed"
        ]

        proposed_hybrid = [
            row for row in proposed_rows
            if (
                i(row, "allocation_df"),
                i(row, "allocation_da"),
            ) == hybrid_allocation
        ]

        proposed_fdma = [
            row for row in proposed_rows
            if i(row, "allocation_da") == 0
        ]

        csinet_rows = [
            row for row in selected
            if row.get("model_key") == "csinet"
        ]

        swin_rows = [
            row for row in selected
            if row.get("model_key") == "swin"
        ]

        curves = [
            (
                proposed_hybrid,
                f"Proposed Hybrid {hybrid_allocation}",
                "o",
            ),
            (
                proposed_fdma,
                "Proposed Pure FDMA",
                "s",
            ),
            (
                csinet_rows,
                "CsiNet+ FDMA",
                "^",
            ),
            (
                swin_rows,
                "Swin-CFNet FDMA",
                "D",
            ),
        ]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        for curve_rows, label, marker in curves:
            if not curve_rows:
                raise RuntimeError(
                    f"Missing curve '{label}' for K={k_users}."
                )

            curve_rows = sorted(
                curve_rows,
                key=lambda row: f(row, "feedback_snr_db"),
            )

            x = [
                f(row, "feedback_snr_db")
                for row in curve_rows
            ]
            y = [
                f(row, "sum_rate_mean")
                for row in curve_rows
            ]
            errors = [
                ci_half_width(row)
                for row in curve_rows
            ]

            ax.errorbar(
                x,
                y,
                yerr=errors,
                marker=marker,
                linewidth=1.7,
                markersize=5,
                capsize=2.5,
                label=label,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Average sum rate (bps/Hz)")
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()

        save_figure(
            fig,
            output_dir,
            f"fig_sota_vs_snr_k{k_users}_{suffix}",
        )






def plot_ablation(
    rows: list[dict[str, str]],
    output_dir: Path,
    operating_snr: float,
) -> None:
    selected = [
        row for row in rows
        if abs(f(row, "snr_db") - operating_snr) < 1e-9
    ]
    if not selected:
        raise RuntimeError(f"No ablation rows found at {operating_snr:g} dB.")

    variant_order = [
        "full",
        "no_condition",
        "no_anchor_injection",
        "no_cross_user_aggregation",
    ]
    allocation_order = [
        (32, 0),
        (24, 32),
        (16, 64),
        (8, 96),
        (0, 128),
    ]

    lookup = {
        (
            row.get("variant"),
            i(row, "allocation_df"),
            i(row, "allocation_da"),
        ): row
        for row in selected
    }

    x = np.arange(len(allocation_order))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for variant_index, variant in enumerate(variant_order):
        variant_rows = [
            lookup[(variant, allocation[0], allocation[1])]
            for allocation in allocation_order
        ]
        means = [f(row, "sum_rate_mean") for row in variant_rows]
        errors = [ci_half_width(row) for row in variant_rows]
        average = f(variant_rows[0], "five_config_avg_mean")
        label = f"{variant_rows[0].get('variant_label', variant)} (Avg. {average:.2f})"
        offset = (variant_index - (len(variant_order) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=2,
            label=label,
        )

    ax.set_xticks(x, [str(allocation) for allocation in allocation_order])
    ax.set_xlabel("Allocation $(D_f,D_a)$")
    ax.set_ylabel("Average sum rate (bps/Hz)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_ablation_k4_d128")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-ready PDF figures from evaluation CSV files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--input", required=True)
        subparser.add_argument("--output-dir", default="paper/figures")

    p = subparsers.add_parser("k16-snr")
    add_common(p)
    p.add_argument("--k", type=int, default=16)

    p = subparsers.add_parser("cross-k")
    add_common(p)
    p.add_argument("--snr", type=float, default=25.0)

    p = subparsers.add_parser("k16-baseline")
    add_common(p)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--snr", type=float, default=25.0)



    p = subparsers.add_parser("sota-snr")
    add_common(p)
    p.add_argument(
        "--fixed-dl-snr",
        type=float,
        default=None,
        help=(
            "Leave unset for matched feedback/downlink SNR. "
            "Set, e.g., 25, for a feedback-SNR sweep at fixed DL SNR."
        ),
    )


    p = subparsers.add_parser("ablation")
    add_common(p)
    p.add_argument("--snr", type=float, default=25.0)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_csv(args.input)
    output_dir = ensure_dir(args.output_dir)

    if args.command == "k16-snr":
        plot_k16_snr(rows, output_dir, args.k)
    elif args.command == "cross-k":
        plot_cross_k(rows, output_dir, args.snr)
    elif args.command == "k16-baseline":
        plot_k16_baseline(rows, output_dir, args.k, args.snr)
    elif args.command == "sota-snr":
        plot_sota_snr(
            rows,
            output_dir,
            args.fixed_dl_snr,
        )
    elif args.command == "ablation":
        plot_ablation(rows, output_dir, args.snr)
    else:
        raise RuntimeError(args.command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
