#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot paper-facing K=8 feedback-SNR robustness results."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


DATA_PATH = Path("paper_results/data/feedback_snr/k8_d128_dl25_feedback_snr.csv")
OUT_DIR = Path("paper_results/figures")
PAPER_CSV = Path("paper_results/data/feedback_snr/k8_d128_dl25_feedback_snr_paper_10to25.csv")


def load_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "model_key": row["model_key"],
                "allocation_df": int(float(row["allocation_df"])),
                "allocation_da": int(float(row["allocation_da"])),
                "feedback_snr_db": float(row["feedback_snr_db"]),
                "downlink_snr_db": float(row["downlink_snr_db"]),
                "num_samples": int(float(row["num_samples"])),
                "sum_rate_mean": float(row["sum_rate_mean"]),
                "sum_rate_se": float(row["sum_rate_se"]),
            })
    return rows


def get_curve(rows, model_key, allocation=None, snr_points=None):
    selected = []
    for row in rows:
        if row["model_key"] != model_key:
            continue
        if allocation is not None:
            df_dim, da_dim = allocation
            if row["allocation_df"] != df_dim or row["allocation_da"] != da_dim:
                continue
        if snr_points is not None and int(row["feedback_snr_db"]) not in snr_points:
            continue
        selected.append(row)
    return sorted(selected, key=lambda x: x["feedback_snr_db"])


def plot_curve(ax, curve, label, marker, linestyle="-"):
    x = [r["feedback_snr_db"] for r in curve]
    y = [r["sum_rate_mean"] for r in curve]
    yerr = [1.96 * r["sum_rate_se"] for r in curve]
    ax.errorbar(x, y, yerr=yerr, marker=marker, linestyle=linestyle,
                linewidth=1.7, markersize=5.5, capsize=2.5, label=label)


def curve_values(curve):
    return {int(r["feedback_snr_db"]): r["sum_rate_mean"] for r in curve}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(DATA_PATH)
    expected_snr = [10, 15, 20, 25]

    hybrid_4_96 = get_curve(rows, "proposed", (4, 96), expected_snr)
    pure_as = get_curve(rows, "proposed", (0, 128), expected_snr)
    pure_fdma = get_curve(rows, "proposed", (16, 0), expected_snr)
    swin = get_curve(rows, "swin", snr_points=expected_snr)
    csinet = get_curve(rows, "csinet", snr_points=expected_snr)

    curves = {
        "hybrid_4_96": hybrid_4_96,
        "pure_as": pure_as,
        "pure_fdma": pure_fdma,
        "swin": swin,
        "csinet": csinet,
    }

    for name, curve in curves.items():
        actual_snr = [int(r["feedback_snr_db"]) for r in curve]
        if actual_snr != expected_snr:
            raise RuntimeError(f"{name} has unexpected SNR points: {actual_snr}")
        for row in curve:
            if row["downlink_snr_db"] != 25:
                raise RuntimeError(f"{name}: DL SNR is not fixed at 25 dB.")
            if row["num_samples"] != 1600:
                raise RuntimeError(f"{name}: expected 1600 samples, got {row['num_samples']}.")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    plot_curve(ax, hybrid_4_96, r"FDMA--AS $(4,96)$", "o")
    plot_curve(ax, pure_as, "Pure AS", "^", "--")
    plot_curve(ax, pure_fdma, "Pure FDMA", "v", "--")
    plot_curve(ax, swin, "Swin-CFNet", "D", "-.")
    plot_curve(ax, csinet, "CsiNet+", "P", "-.")

    ax.set_xlabel("Feedback SNR (dB)")
    ax.set_ylabel("Downlink sum rate (bps/Hz)")
    ax.set_xticks(expected_snr)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8, ncol=2, frameon=True, loc="lower right")

    fig.tight_layout()

    pdf_path = OUT_DIR / "fig_feedback_snr_k8_dl25_10to25.pdf"
    png_path = OUT_DIR / "fig_feedback_snr_k8_dl25_10to25.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    values = {name: curve_values(curve) for name, curve in curves.items()}

    with open(PAPER_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["feedback_snr_db", "hybrid_4_96", "pure_as", "pure_fdma", "swin", "csinet"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for snr in expected_snr:
            writer.writerow({
                "feedback_snr_db": snr,
                "hybrid_4_96": values["hybrid_4_96"][snr],
                "pure_as": values["pure_as"][snr],
                "pure_fdma": values["pure_fdma"][snr],
                "swin": values["swin"][snr],
                "csinet": values["csinet"][snr],
            })

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {PAPER_CSV}")

    print("\nSNR | Hybrid(4,96) | Pure AS | Pure FDMA | Swin | CsiNet+")
    for snr in expected_snr:
        print(
            f"{snr:>3} | "
            f"{values['hybrid_4_96'][snr]:>12.3f} | "
            f"{values['pure_as'][snr]:>7.3f} | "
            f"{values['pure_fdma'][snr]:>9.3f} | "
            f"{values['swin'][snr]:>5.3f} | "
            f"{values['csinet'][snr]:>7.3f}"
        )


if __name__ == "__main__":
    main()