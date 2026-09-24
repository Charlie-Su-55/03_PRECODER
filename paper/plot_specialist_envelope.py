#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper figure: universal vs allocation-specific specialist envelope."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "k4_specialist_envelope.csv"
OUT = ROOT / "figures"


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        for key in ("df", "da"):
            r[key] = int(r[key])
        for key in ("as_fraction", "sum_rate_mean", "sum_rate_se",
                    "nmse_db_mean", "nmse_db_se"):
            r[key] = float(r[key])
    return rows


def select(rows, method_type):
    out = [r for r in rows if r["method_type"] == method_type]
    return sorted(out, key=lambda r: r["as_fraction"])


def xyerr(rows, ykey, sekey):
    x = [r["as_fraction"] for r in rows]
    y = [r[ykey] for r in rows]
    err = [1.96 * r[sekey] for r in rows]
    return x, y, err


def main():
    rows = load_rows(DATA)
    OUT.mkdir(parents=True, exist_ok=True)

    universal = select(rows, "universal")
    specialist = select(rows, "specialist")
    swin = select(rows, "swin")[0]
    csinet = select(rows, "csinet")[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    ax1, ax2 = axes

    # ------------------------------------------------------------
    # (a) Downlink sum rate
    # ------------------------------------------------------------
    x, y, e = xyerr(universal, "sum_rate_mean", "sum_rate_se")
    ax1.errorbar(x, y, yerr=e, marker="o", linewidth=1.7,
                 markersize=5.5, capsize=2.2, label="Universal Proposed")

    x, y, e = xyerr(specialist, "sum_rate_mean", "sum_rate_se")
    ax1.errorbar(x, y, yerr=e, marker="s", linewidth=1.7,
                 markersize=5.5, capsize=2.2,
                 label="Allocation-specific specialist")

    ax1.axhline(swin["sum_rate_mean"], linestyle="--", linewidth=1.4,
                label="Swin-CFNet")
    ax1.axhline(csinet["sum_rate_mean"], linestyle=":", linewidth=1.4,
                label="CsiNet+")

    ax1.set_xlabel("AS fraction")
    ax1.set_ylabel("Downlink sum rate (bps/Hz)")
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax1.set_title("(a) Task performance", fontsize=9)
    ax1.legend(fontsize=6.9, frameon=True, loc="lower right")

    # ------------------------------------------------------------
    # (b) Raw representation / reconstructed-channel NMSE
    # ------------------------------------------------------------
    x, y, e = xyerr(universal, "nmse_db_mean", "nmse_db_se")
    ax2.errorbar(x, y, yerr=e, marker="o", linewidth=1.7,
                 markersize=5.5, capsize=2.2, label="Universal Proposed")

    x, y, e = xyerr(specialist, "nmse_db_mean", "nmse_db_se")
    ax2.errorbar(x, y, yerr=e, marker="s", linewidth=1.7,
                 markersize=5.5, capsize=2.2,
                 label="Allocation-specific specialist")

    ax2.scatter([0], [swin["nmse_db_mean"]], marker="D", s=30,
                label="Swin-CFNet")
    ax2.scatter([0], [csinet["nmse_db_mean"]], marker="^", s=34,
                label="CsiNet+")

    ax2.set_xlabel("AS fraction")
    ax2.set_ylabel("Raw reconstruction NMSE (dB)")
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax2.set_title("(b) Reconstruction diagnostic", fontsize=9)
    ax2.legend(fontsize=6.9, frameon=True, loc="upper right")

    fig.tight_layout(w_pad=1.2)

    pdf = OUT / "fig_specialist_envelope.pdf"
    png = OUT / "fig_specialist_envelope.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf}")
    print(f"Saved: {png}")

    print("\nAS frac | Universal | Specialist | Spec. gain")
    for u, s in zip(universal, specialist):
        gain = s["sum_rate_mean"] - u["sum_rate_mean"]
        print(
            f"{u['as_fraction']:.2f} | "
            f"{u['sum_rate_mean']:.3f} | "
            f"{s['sum_rate_mean']:.3f} | "
            f"{gain:+.3f}"
        )

    pure_as = specialist[-1]
    delta_swin = pure_as["sum_rate_mean"] - swin["sum_rate_mean"]
    print(
        f"\nPure-AS specialist vs Swin-CFNet: "
        f"{pure_as['sum_rate_mean']:.3f} vs "
        f"{swin['sum_rate_mean']:.3f} "
        f"({delta_swin:+.3f} bps/Hz)"
    )


if __name__ == "__main__":
    main()