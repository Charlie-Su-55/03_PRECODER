#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "k4_pure_as_feedback_snr.csv"
OUT = ROOT / "figures"

ORDER = [
    "Allocation-specific specialist",
    "Universal Proposed",
    "Swin-CFNet",
    "CsiNet+",
]

STYLE = {
    "Allocation-specific specialist": dict(marker="s", linestyle="-", linewidth=1.8, markersize=5.5),
    "Universal Proposed": dict(marker="o", linestyle="--", linewidth=1.8, markersize=5.5),
    "Swin-CFNet": dict(marker="D", linestyle="-.", linewidth=1.6, markersize=5.0),
    "CsiNet+": dict(marker="^", linestyle=":", linewidth=1.6, markersize=5.2),
}


def load_data():
    curves = defaultdict(list)
    with DATA.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            curves[row["curve"]].append({
                "snr": float(row["feedback_snr_db"]),
                "rate": float(row["sum_rate_mean"]),
                "se": float(row["sum_rate_se"]),
            })

    for key in curves:
        curves[key].sort(key=lambda x: x["snr"])

    expected = [0, 5, 10, 15, 20, 25]
    for key in ORDER:
        if key not in curves:
            raise RuntimeError(f"Missing curve: {key}")
        snrs = [int(x["snr"]) for x in curves[key]]
        if snrs != expected:
            raise RuntimeError(f"{key}: unexpected SNR points {snrs}")

    return curves


def main():
    curves = load_data()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.1))

    for key in ORDER:
        rows = curves[key]
        x = [r["snr"] for r in rows]
        y = [r["rate"] for r in rows]
        ci95 = [1.96 * r["se"] for r in rows]

        ax.errorbar(
            x, y, yerr=ci95,
            capsize=2.2,
            label=key,
            **STYLE[key],
        )

    ax.set_xlabel("Feedback SNR (dB)")
    ax.set_ylabel("Downlink sum rate (bps/Hz)")
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlim(-0.7, 25.7)
    ax.set_ylim(9.5, 30.5)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    ax.legend(
        fontsize=8,
        ncol=2,
        frameon=True,
        loc="lower right",
    )

    specialist_25 = curves["Allocation-specific specialist"][-1]["rate"]
    swin_25 = curves["Swin-CFNet"][-1]["rate"]
    gain = specialist_25 - swin_25

    ax.annotate(
        f"+{gain:.2f}",
        xy=(25, specialist_25),
        xytext=(-28, 12),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", linewidth=0.7),
    )

    fig.tight_layout()

    pdf = OUT / "fig_k4_pure_as_feedback_snr.pdf"
    png = OUT / "fig_k4_pure_as_feedback_snr.png"

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf}")
    print(f"Saved: {png}")
    print(f"25-dB specialist: {specialist_25:.3f}")
    print(f"25-dB Swin-CFNet: {swin_25:.3f}")
    print(f"Difference: {gain:+.3f} bps/Hz")


if __name__ == "__main__":
    main()
