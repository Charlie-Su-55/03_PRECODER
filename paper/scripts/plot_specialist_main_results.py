#!/usr/bin/env python3
"""Render three formal K4 specialist figures from stored CSV summaries only.

Defaults are resolved relative to this script, independently of the current
working directory. No models are loaded and no experiments are run. The two
source summaries are kept separate: their 25-dB results are not interchangeable.
Existing figure files are protected unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENVELOPE = PAPER_DIR / "data/k4_specialist_envelope/summary_specialist_envelope.csv"
DEFAULT_SWEEP = PAPER_DIR / "data/k4_pure_as_fb_robust/summary_k4_pure_as_fb_robust.csv"
ALLOCATIONS = ((64, 0), (48, 64), (32, 128), (16, 192), (0, 256))
SNR_GRID = (0, 5, 10, 15, 20, 25)
BENCHMARK_METHODS = ("csinet", "specialist_df64_da0", "swin", "specialist_df0_da256")
SWEEP_METHODS = ("specialist_df0_da256", "robust_specialist", "swin", "csinet")
STEMS = (
    "fig_k4_specialist_benchmark_25db",
    "fig_k4_specialist_feedback_snr",
    "fig_k4_allocation_flexibility",
)
RATE_FIELDS = ("sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high")
PROTOCOL = {
    "k_users": 4, "antennas": 32, "subcarriers": 256,
    "feedback_budget": 256, "seed": 20260921, "num_samples": 1600,
}
CHECKPOINT_STEPS = {
    "csinet": 49500, "swin": 49000, "universal": 48000,
    "specialist_df64_da0": 49000, "specialist_df48_da64": 47500,
    "specialist_df32_da128": 46500, "specialist_df16_da192": 47000,
    "specialist_df0_da256": 50000, "robust_specialist": 20000,
}
LABELS = {
    "csinet": "CsiNet+ (FDMA)", "swin": "Swin-CFNet (FDMA)",
    "specialist_df64_da0": "Pure-FDMA specialist",
    "specialist_df0_da256": "Pure-AS specialist (50k)",
    "robust_specialist": "Pure-AS refined (50k + 20k)",
}
COLORS = {
    "csinet": "#0072B2", "swin": "#333333",
    "specialist_df64_da0": "#009E73",
    "specialist_df0_da256": "#D55E00", "robust_specialist": "#CC79A7",
    "universal": "#7570B3",
}
MARKERS = {
    "csinet": "v", "swin": "s", "specialist_df64_da0": "D",
    "specialist_df0_da256": "o", "robust_specialist": "^",
}
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7,
}


def specialist_key(allocation: tuple[int, int]) -> str:
    return f"specialist_df{allocation[0]}_da{allocation[1]}"


def _expected(kind: str) -> set:
    if kind == "envelope":
        keys = {(specialist_key(a), a, 25.0) for a in ALLOCATIONS}
        keys |= {("universal", a, 25.0) for a in ALLOCATIONS}
        return keys | {(m, (64, 0), 25.0) for m in ("csinet", "swin")}
    if kind == "sweep":
        return {
            (m, (64, 0) if m in ("csinet", "swin") else (0, 256), float(snr))
            for m in (*SWEEP_METHODS, "universal") for snr in SNR_GRID
        }
    raise ValueError(f"Unknown summary kind: {kind}")


def load_summary(path: Path, kind: str) -> dict:
    """Validate the complete formal protocol, preserving stored rate/CI values."""
    expected = _expected(kind)
    records = {}
    required = set(PROTOCOL) | set(RATE_FIELDS) | {
        "scenario_id", "method_key", "model_key", "allocation_df", "allocation_da",
        "feedback_snr_db", "downlink_snr_db", "fdma_fraction", "aircomp_fraction",
        "checkpoint_kind", "checkpoint_step",
    }
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            prefix = f"{Path(path).name}, line {line}"
            try:
                if any(int(row[field]) != value for field, value in PROTOCOL.items()):
                    raise ValueError("expected K4/Nt32/Nsc256/Dtot256, seed 20260921, 1600 samples")
                if row["scenario_id"] != "K4_Nsc256_Dfb256":
                    raise ValueError("unexpected scenario_id")
                if float(row["downlink_snr_db"]) != 25.0:
                    raise ValueError("expected downlink SNR 25 dB")
                allocation = (int(row["allocation_df"]), int(row["allocation_da"]))
                method = row["method_key"]
                feedback_snr = float(row["feedback_snr_db"])
                key = (method, allocation, feedback_snr)
                if key not in expected:
                    raise ValueError(f"unexpected method/allocation/feedback SNR: {key}")
                if key in records:
                    raise ValueError(f"duplicate method/allocation/feedback SNR: {key}")
                expected_model = method if method in ("csinet", "swin") else "proposed"
                if row["model_key"] != expected_model:
                    raise ValueError("method/model mismatch")
                if row["checkpoint_kind"] != "best" or int(row["checkpoint_step"]) != CHECKPOINT_STEPS[method]:
                    raise ValueError("unexpected checkpoint kind/step")
                for field, expected_fraction in (
                    ("fdma_fraction", allocation[0] * 4 / 256),
                    ("aircomp_fraction", allocation[1] / 256),
                ):
                    if float(row[field]) != expected_fraction:
                        raise ValueError(f"{field}/allocation mismatch")
                if "as_fraction" in row and float(row["as_fraction"]) != allocation[1] / 256:
                    raise ValueError("as_fraction/allocation mismatch")
                values = {field: float(row[field]) for field in RATE_FIELDS}
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("non-finite plotting value")
                mean, se, low, high = (values[field] for field in RATE_FIELDS)
                if not 0 <= low <= mean <= high or se < 0:
                    raise ValueError("invalid sum-rate CI bounds or SE")
                if not all(math.isclose(bound, target, rel_tol=1e-7, abs_tol=1e-7)
                           for bound, target in ((low, mean - 1.96 * se), (high, mean + 1.96 * se))):
                    raise ValueError("CI mismatch: expected stored mean +/- 1.96 SE")
                records[key] = values
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"{prefix}: {exc}") from exc
    if set(records) != expected:
        raise ValueError(f"Incomplete {kind} grid; missing {sorted(expected - set(records))}")
    return records


def benchmark_rows(records: dict) -> list[dict]:
    return [records[m, (0, 256) if m == "specialist_df0_da256" else (64, 0), 25.0]
            for m in BENCHMARK_METHODS]


def sweep_rows(records: dict, method: str) -> list[dict]:
    allocation = (64, 0) if method in ("csinet", "swin") else (0, 256)
    return [records[method, allocation, float(snr)] for snr in SNR_GRID]


def flexibility_rows(records: dict, universal: bool = False) -> list[dict]:
    return [records["universal" if universal else specialist_key(a), a, 25.0]
            for a in ALLOCATIONS]


def _values_and_errors(rows: list[dict]) -> tuple[list, list]:
    return ([r["sum_rate_mean"] for r in rows], [
        [r["sum_rate_mean"] - r["sum_rate_ci95_low"] for r in rows],
        [r["sum_rate_ci95_high"] - r["sum_rate_mean"] for r in rows],
    ])


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _axes_style(ax, axis="y") -> None:
    ax.grid(axis=axis, linestyle=":", linewidth=0.6, color="#bcbcbc", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def draw_benchmark(records: dict):
    plt = _pyplot()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        rows = benchmark_rows(records)
        for y, method, row in zip(range(4), BENCHMARK_METHODS, rows):
            means, errors = _values_and_errors([row])
            ax.errorbar(means, [y], xerr=errors, fmt=MARKERS[method], markersize=6,
                        color=COLORS[method], capsize=3.2, elinewidth=1.0, zorder=3)
            ax.annotate(f"{means[0]:.3f}", (means[0], y), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=10)
        ax.set_yticks(range(4), ["CsiNet+\n(FDMA)", "Pure-FDMA\nspecialist",
                               "Swin-CFNet\n(FDMA)", "Pure-AS\nspecialist"])
        ax.set_ylim(-0.45, 3.6)
        ax.set_xlim(20.8, 30.1)
        ax.set_xlabel("Downlink sum rate (bps/Hz)")
        ax.set_title("Matched training budget: 50k updates", fontsize=10, pad=12)
        _axes_style(ax, "x")
        fig.tight_layout(pad=0.8)
    return fig


def draw_feedback_snr(records: dict):
    plt = _pyplot()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        for method in SWEEP_METHODS:
            means, errors = _values_and_errors(sweep_rows(records, method))
            label = LABELS[method].replace(" (50k)", "\n(50k)").replace(
                " (50k + 20k)", "\n(50k + 20k)")
            ax.errorbar(SNR_GRID, means, yerr=errors, label=label,
                        color=COLORS[method], marker=MARKERS[method], markersize=4.5,
                        linestyle="--" if method == "robust_specialist" else "-",
                        linewidth=1.45, capsize=2.4, elinewidth=0.8, zorder=3)
        ax.set_xticks(SNR_GRID)
        ax.set_xlim(-0.7, 25.7)
        ax.set_ylim(10, 31)
        ax.set_xlabel("Feedback SNR (dB)")
        ax.set_ylabel("Downlink sum rate (bps/Hz)")
        _axes_style(ax)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False,
                  fontsize=9.5, columnspacing=0.8, handlelength=1.5, labelspacing=0.5)
        inset = ax.inset_axes([0.55, 0.08, 0.42, 0.33])
        for method in SWEEP_METHODS[:3]:
            means, errors = _values_and_errors(sweep_rows(records, method)[-2:])
            inset.errorbar(SNR_GRID[-2:], means, yerr=errors, color=COLORS[method],
                           marker=MARKERS[method], markersize=3,
                           linestyle="--" if method == "robust_specialist" else "-",
                           linewidth=1.0, capsize=2, elinewidth=0.7)
        inset.set_xlim(19.5, 25.5)
        inset.set_ylim(28.1, 29.8)
        inset.set_xticks((20, 25))
        inset.set_yticks((28.5, 29.0, 29.5))
        inset.tick_params(labelsize=9, pad=2)
        inset.set_title("20-25 dB detail", fontsize=9.5, pad=5)
        inset.grid(linestyle=":", linewidth=0.5, color="#cccccc")
        fig.tight_layout(pad=0.8)
    return fig


def draw_flexibility(records: dict):
    plt = _pyplot()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        shares = [a[1] / 256 * 100 for a in ALLOCATIONS]
        for universal, label, color, marker, linestyle in (
            (False, "Specialists (5 checkpoints)", COLORS["specialist_df0_da256"], "o", "-"),
            (True, "Universal model (1 checkpoint)", COLORS["universal"], "s", "--"),
        ):
            means, errors = _values_and_errors(flexibility_rows(records, universal))
            ax.errorbar(shares, means, yerr=errors, label=label, color=color,
                        marker=marker, markersize=4.5, linestyle=linestyle,
                        linewidth=1.5, capsize=2.7, elinewidth=0.9, zorder=3)
        ax.set_xticks(shares)
        ax.set_xlim(-3, 103)
        ax.set_ylim(19.5, 30.5)
        ax.set_xlabel(r"AS resource share, $D_a/(4D_f+D_a)$ (%)")
        ax.set_ylabel("Downlink sum rate (bps/Hz)")
        _axes_style(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=9.5)
        fig.tight_layout(pad=0.8)
    return fig


def render(envelope: dict, sweep: dict, output_dir: Path, overwrite: bool = False) -> list[Path]:
    """Write six standalone artifacts; validate all destinations before writing."""
    output_dir = Path(output_dir).expanduser().resolve()
    destinations = [output_dir / f"{stem}.{suffix}" for stem in STEMS for suffix in ("pdf", "png")]
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing figures; choose another --output-dir "
                              "or explicitly pass --overwrite: " + ", ".join(existing))
    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (draw, records) in enumerate(((draw_benchmark, envelope),
                                            (draw_feedback_snr, sweep),
                                            (draw_flexibility, envelope))):
        fig = draw(records)
        try:
            with plt.rc_context(STYLE):
                fig.savefig(destinations[index * 2], bbox_inches="tight", metadata={
                    "Title": STEMS[index],
                    "Subject": "Formal 1600-sample results; stored mean and 95% CI; fixed training seed",
                })
                fig.savefig(destinations[index * 2 + 1], dpi=220, bbox_inches="tight")
        finally:
            plt.close(fig)
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope-input", type=Path, default=DEFAULT_ENVELOPE)
    parser.add_argument("--sweep-input", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR / "figures")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing these six figure files")
    args = parser.parse_args()
    envelope = load_summary(args.envelope_input.expanduser().resolve(), "envelope")
    sweep = load_summary(args.sweep_input.expanduser().resolve(), "sweep")
    paths = render(envelope, sweep, args.output_dir, args.overwrite)
    print("CSV-only: 12 envelope and 30 SNR-sweep rows validated; stored 95% CI bounds preserved.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
