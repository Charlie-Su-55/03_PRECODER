#!/usr/bin/env python3
"""Plot the verified K4/Nsc128/Dtot128 ablation summary; CSV-only, no models.

Usage: python paper/scripts/plot_ablation_verified.py --input SUMMARY.csv --output-dir OUT
All 20 bars use the stored per-allocation means and CI bounds. The repeated
paired_delta_vs_full_* columns are grid-level statistics and are never plotted.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ALLOCATIONS = ((32, 0), (24, 32), (16, 64), (8, 96), (0, 128))
VARIANTS = (
    "full", "no_condition", "no_anchor_injection", "no_cross_user_aggregation",
)
LABELS = {
    "full": "Full model",
    "no_condition": "w/o allocation/SNR conditioning",
    "no_anchor_injection": "w/o repeated direct FDMA input",
    "no_cross_user_aggregation": "w/o AS-context/cross-user path",
}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#9270B2")
STEM = "fig_ablation_k4_d128_verified"
FIELDS = (
    "sum_rate_mean", "sum_rate_ci95_low", "sum_rate_ci95_high",
    "five_config_avg_mean",
)


def load_summary(path: Path) -> dict:
    """Reject incomplete/mixed protocols; preserve unrounded CSV values."""
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(FIELDS) | {
            "variant", "allocation_df", "allocation_da", "snr_db", "num_samples",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            allocation = (int(row["allocation_df"]), int(row["allocation_da"]))
            key = (row["variant"], allocation)
            if key[0] not in VARIANTS or allocation not in ALLOCATIONS:
                raise ValueError(f"Line {line}: unexpected variant/allocation {key}")
            if key in records:
                raise ValueError(f"Line {line}: duplicate variant/allocation {key}")
            if float(row["snr_db"]) != 25.0 or int(row["num_samples"]) != 2400:
                raise ValueError(f"Line {line}: expected matched 25-dB, 2400-sample summary")
            values = {field: float(row[field]) for field in FIELDS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Line {line}: non-finite plotting value")
            if not 0 <= values["sum_rate_ci95_low"] <= values["sum_rate_mean"] <= values["sum_rate_ci95_high"]:
                raise ValueError(f"Line {line}: invalid sum-rate CI bounds")
            records[key] = values
    expected = {(variant, allocation) for variant in VARIANTS for allocation in ALLOCATIONS}
    if set(records) != expected:
        raise ValueError(f"Incomplete grid; missing {sorted(expected - set(records))}")
    for variant in VARIANTS:
        rows = [records[variant, allocation] for allocation in ALLOCATIONS]
        average = rows[0]["five_config_avg_mean"]
        if any(row["five_config_avg_mean"] != average for row in rows):
            raise ValueError(f"{variant}: inconsistent repeated grid average")
        # The original evaluator averages in float32; do not replace its value.
        if not math.isclose(sum(row["sum_rate_mean"] for row in rows) / 5, average,
                            rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{variant}: grid average disagrees with allocation means")
    return records


def draw_figure(records: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7,
    }):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        width = 0.19
        for index, variant in enumerate(VARIANTS):
            rows = [records[variant, allocation] for allocation in ALLOCATIONS]
            means = [row["sum_rate_mean"] for row in rows]
            errors = [
                [row["sum_rate_mean"] - row["sum_rate_ci95_low"] for row in rows],
                [row["sum_rate_ci95_high"] - row["sum_rate_mean"] for row in rows],
            ]
            x = [i + (index - 1.5) * width for i in range(5)]
            average = rows[0]["five_config_avg_mean"]
            ax.bar(x, means, width, yerr=errors, capsize=2.3,
                   error_kw={"elinewidth": 0.85, "capthick": 0.85, "ecolor": "#202020"},
                   color=COLORS[index], edgecolor="#404040", linewidth=0.4,
                   label=f"{LABELS[variant]}\n(grid avg. {average:.2f})", zorder=3)
        ax.set_xticks(range(5), [f"({df}, {da})" for df, da in ALLOCATIONS])
        ax.set_xlabel(r"Allocation $(D_f,D_a)$")
        ax.set_ylabel("Downlink sum rate (bps/Hz)")
        ax.set_ylim(0, 30)
        ax.set_xlim(-0.55, 4.55)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#bcbcbc", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2,
                  frameon=False, fontsize=9, columnspacing=1.8, handlelength=1.7,
                  labelspacing=0.7)
        fig.tight_layout(pad=0.6)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_summary(args.input.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    destinations = [output / f"{STEM}.{suffix}" for suffix in ("pdf", "png")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite an existing verified figure; use a new output directory")
    fig = draw_figure(records)
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(destinations[0], bbox_inches="tight",
                    metadata={"Title": "K4 allocation ablation with per-bar 95% confidence intervals"})
        fig.savefig(destinations[1], dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved all 4 variants x 5 allocations; error bars use stored per-bar CI bounds.")
    for path in destinations:
        print(path)


if __name__ == "__main__":
    main()
