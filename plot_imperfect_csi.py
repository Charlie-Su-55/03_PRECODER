#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-facing robustness figure for imperfect UL/DL CSI."""

import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt


UL_ZERO = Path("paper_results/data/imperfect_csi/ul_csi_zero_shot.csv")
UL_AWARE = Path("paper_results/data/imperfect_csi/ul_csi_aware.csv")
DL_CSI = Path("paper_results/data/imperfect_csi/dl_csi.csv")
OUT_DIR = Path("paper_results/figures")


def load_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    x = float(value)
    return x if math.isfinite(x) else None


def get_ul_curve(rows, method_key, df_dim=None, da_dim=None):
    curve = []
    for r in rows:
        if r["method_key"] != method_key:
            continue
        if df_dim is not None and int(r["allocation_df"]) != df_dim:
            continue
        if da_dim is not None and int(r["allocation_da"]) != da_dim:
            continue

        snr = finite_float(r["pilot_snr_db"])
        if snr is None:
            continue

        curve.append({
            "x": snr,
            "y": float(r["sum_rate_mean"]),
            "se": float(r["sum_rate_se"]),
            "n": int(r["num_samples"]),
        })

    return sorted(curve, key=lambda z: z["x"])


def get_dl_curve(rows, method):
    curve = []
    for r in rows:
        if r["method"] != method:
            continue

        snr = finite_float(r["dl_pilot_snr_db"])
        if snr is None:
            continue

        curve.append({
            "x": snr,
            "y": float(r["sum_rate_mean"]),
            "se": float(r["sum_rate_se"]),
            "nmse": float(r["dl_ce_nmse_db"]),
            "n": int(r["num_samples"]),
        })

    return sorted(curve, key=lambda z: z["x"])


def plot_curve(ax, curve, label, marker, linestyle="-"):
    x = [r["x"] for r in curve]
    y = [r["y"] for r in curve]
    err = [1.96 * r["se"] for r in curve]
    ax.errorbar(x, y, yerr=err, marker=marker, linestyle=linestyle,
                linewidth=1.6, markersize=5, capsize=2.2, label=label)


def validate(curves, expected_x):
    for name, curve in curves.items():
        actual = [int(r["x"]) for r in curve]
        if actual != expected_x:
            raise RuntimeError(f"{name}: unexpected x points {actual}")
        if any(r["n"] != 1600 for r in curve):
            raise RuntimeError(f"{name}: not all points use 1600 samples")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ul_zero = load_csv(UL_ZERO)
    ul_aware = load_csv(UL_AWARE)
    dl_rows = load_csv(DL_CSI)

    ul_x = [-10, -5, 0, 5, 10, 15, 20]
    dl_x = [-20, -15, -10, -5, 0, 5, 10]

    ul_curves = {
        "Proposed zero-shot": get_ul_curve(ul_zero, "proposed", 4, 96),
        "Proposed CSI-aware": get_ul_curve(ul_aware, "proposed", 4, 96),
        "Swin-CFNet": get_ul_curve(ul_zero, "swin", 16, 0),
        "CsiNet+": get_ul_curve(ul_zero, "csinet", 16, 0),
    }

    dl_curves = {
        "Proposed": get_dl_curve(dl_rows, "Proposed"),
        "Swin-CFNet": get_dl_curve(dl_rows, "Swin-based FDMA"),
        "CsiNet+": get_dl_curve(dl_rows, "CsiNet+-based FDMA"),
    }

    validate(ul_curves, ul_x)
    validate(dl_curves, dl_x)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), sharey=True)
    ax1, ax2 = axes

    plot_curve(ax1, ul_curves["Proposed zero-shot"], "Proposed, zero-shot", "o", "--")
    plot_curve(ax1, ul_curves["Proposed CSI-aware"], "Proposed, CSI-aware", "s")
    plot_curve(ax1, ul_curves["Swin-CFNet"], "Swin-CFNet", "D", "-.")
    plot_curve(ax1, ul_curves["CsiNet+"], "CsiNet+", "^", "-.")

    ax1.set_xlabel("UL pilot SNR (dB)")
    ax1.set_ylabel("Downlink sum rate (bps/Hz)")
    ax1.set_xticks(ul_x)
    ax1.set_ylim(0, 30)
    ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax1.set_title("(a) BS-side feedback-link CSI estimation", fontsize=9)
    ax1.legend(fontsize=6.8, ncol=2, frameon=True, loc="lower right")

    plot_curve(ax2, dl_curves["Proposed"], "Proposed", "o")
    plot_curve(ax2, dl_curves["Swin-CFNet"], "Swin-CFNet", "D", "-.")
    plot_curve(ax2, dl_curves["CsiNet+"], "CsiNet+", "^", "-.")

    ax2.set_xlabel("DL pilot SNR (dB)")
    ax2.set_xticks(dl_x)
    ax2.set_ylim(0, 30)
    ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax2.set_title("(b) UE-side DL CSI estimation", fontsize=9)
    ax2.legend(fontsize=7.0, frameon=True, loc="lower right")

    fig.tight_layout(w_pad=1.2)

    pdf_path = OUT_DIR / "fig_imperfect_csi_robustness.pdf"
    png_path = OUT_DIR / "fig_imperfect_csi_robustness.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    print("\nUL pilot SNR | Zero-shot | CSI-aware | Swin | CsiNet+")
    for i, snr in enumerate(ul_x):
        print(
            f"{snr:>4} dB | "
            f"{ul_curves['Proposed zero-shot'][i]['y']:>8.3f} | "
            f"{ul_curves['Proposed CSI-aware'][i]['y']:>9.3f} | "
            f"{ul_curves['Swin-CFNet'][i]['y']:>5.3f} | "
            f"{ul_curves['CsiNet+'][i]['y']:>7.3f}"
        )

    print("\nDL pilot SNR | CE NMSE | Proposed | Swin | CsiNet+")
    for i, snr in enumerate(dl_x):
        print(
            f"{snr:>4} dB | "
            f"{dl_curves['Proposed'][i]['nmse']:>7.2f} dB | "
            f"{dl_curves['Proposed'][i]['y']:>8.3f} | "
            f"{dl_curves['Swin-CFNet'][i]['y']:>5.3f} | "
            f"{dl_curves['CsiNet+'][i]['y']:>7.3f}"
        )


if __name__ == "__main__":
    main()