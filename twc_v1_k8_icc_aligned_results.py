"""Small, CSV-only paired summaries for the independent ICC-aligned screen.

The independent unit is one complete eight-user channel realization.  This
module imports neither Torch nor Sionna and never reads a checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

from physical_split_finish_io import write_union_csv

ARMS = ("fdma", "hybrid_g4")
GAMMAS = (0., 5., 10., 15., 20., 25.)
ID_FIELDS = frozenset(("arm", "gamma_db", "sample_index", "batch_index", "within_batch"))
REQUIRED_METRICS = ("dl_nmse_linear", "sum_rate_bps_hz", "desired_signal_power_mean",
                    "interference_power_mean", "rzf_power_max_error")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def moments(values):
    values = list(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, (float, int))
                         or not math.isfinite(v) for v in values):
        raise ValueError("Finite, nonempty numeric observations required")
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, se=se,
                ci95_low=None if se is None else mean - 1.96 * se,
                ci95_high=None if se is None else mean + 1.96 * se)


def validate_rows(rows, *, samples, batch_size=4, arms=ARMS, gammas=GAMMAS):
    """Refuse duplicates, omissions, mismatched IDs or nonfinite diagnostics."""
    if type(samples) is not int or samples < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive sample and batch counts required")
    if samples % batch_size:
        raise ValueError("Only complete physical microbatches may be summarized")
    rows = [dict(row) for row in rows]
    expected = {(arm, float(gamma), i) for arm in arms for gamma in gammas for i in range(samples)}
    index = {}
    for row in rows:
        if not ID_FIELDS.issubset(row) or not set(REQUIRED_METRICS).issubset(row):
            raise ValueError("Incomplete per-MU metric row")
        i = _integer(row["sample_index"], "sample_index")
        batch = _integer(row["batch_index"], "batch_index")
        within = _integer(row["within_batch"], "within_batch")
        gamma = row["gamma_db"]
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma):
            raise ValueError("Invalid gamma")
        key = (row["arm"], float(gamma), i)
        if key not in expected or batch != i // batch_size or within != i % batch_size:
            raise ValueError("Unexpected condition or sample/batch identity")
        if key in index:
            raise ValueError("Duplicate complete-MU sample key: " + repr(key))
        for field, value in row.items():
            if field in ID_FIELDS or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Nonfinite or nonnumeric metric: " + field)
            if (field.endswith("nmse_linear") or field.endswith("energy") or field.endswith("power_mean")) and value < 0:
                raise ValueError("Negative physical error/power diagnostic: " + field)
        index[key] = row
    if set(index) != expected:
        missing = sorted(expected - set(index))
        raise ValueError("Missing complete-MU observations: " + repr(missing[:3]))
    return rows, index


def summarize_rows(rows, *, samples, batch_size=4, arms=ARMS):
    rows, _ = validate_rows(rows, samples=samples, batch_size=batch_size, arms=arms)
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"], float(row["gamma_db"])].append(row)
    summary = []
    for (arm, gamma), group in sorted(groups.items()):
        result = dict(arm=arm, gamma_db=gamma, n_samples=len(group), batch_size=batch_size,
                      n_batches=len(group) // batch_size)
        metrics = sorted(set().union(*(set(row) for row in group)) - ID_FIELDS)
        for metric in metrics:
            present = [row.get(metric) for row in group]
            if all(value is None for value in present):
                continue
            if any(value is None for value in present):
                raise ValueError("Partially missing diagnostic within one condition: " + metric)
            values = moments(present)
            for suffix in ("mean", "se", "ci95_low", "ci95_high"):
                result[metric + "_" + suffix] = values[suffix]
            if metric.endswith("nmse_linear"):
                # This is log10(mean linear NMSE), never mean(dB NMSE).
                result[metric.replace("_linear", "_db") + "_from_linear_mean"] = (
                    -math.inf if values["mean"] == 0 else 10 * math.log10(values["mean"]))
        summary.append(result)
    return summary


def paired_rows(rows, *, samples, batch_size=4):
    """Hybrid minus FDMA; average gamma within MU before either interval."""
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    groups = [("point", (gamma,)) for gamma in GAMMAS]
    groups += [("low_0_5_10", GAMMAS[:3]), ("all_six", GAMMAS)]
    output = []
    for scope, gammas in groups:
        common = set.intersection(*(set(index[arm, gamma, 0]) - ID_FIELDS
                                    for arm in ARMS for gamma in gammas))
        for metric in sorted(common):
            if any(index[arm, gamma, i].get(metric) is None
                   for arm in ARMS for gamma in gammas for i in range(samples)):
                continue
            deltas = [statistics.mean(index["hybrid_g4", gamma, i][metric] -
                                     index["fdma", gamma, i][metric] for gamma in gammas)
                      for i in range(samples)]
            sample = moments(deltas)
            batches = [statistics.mean(deltas[start:start + batch_size])
                       for start in range(0, samples, batch_size)]
            batch = moments(batches)
            result = dict(comparison="hybrid_g4-fdma", metric=metric, scope=scope,
                          gamma_db=gammas[0] if scope == "point" else None,
                          n_samples=samples, n_batches=len(batches), batch_size=batch_size,
                          mean=sample["mean"], sample_se=sample["se"],
                          sample_ci95_low=sample["ci95_low"], sample_ci95_high=sample["ci95_high"],
                          batch_se=batch["se"], batch_ci95_low=batch["ci95_low"],
                          batch_ci95_high=batch["ci95_high"])
            output.append(result)
    return output


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            row = {name: (None if value == "" else value) for name, value in source.items()}
            for name, value in row.items():
                if name == "arm" or value is None:
                    continue
                row[name] = int(value) if name in ("sample_index", "batch_index", "within_batch") else float(value)
            rows.append(row)
    return rows


def write_result_tables(output_dir, rows, *, samples, batch_size=4, arms=ARMS):
    output = Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError("Owned result directory must already exist")
    summary = summarize_rows(rows, samples=samples, batch_size=batch_size, arms=arms)
    paired = paired_rows(rows, samples=samples, batch_size=batch_size) if tuple(arms) == ARMS else []
    targets = [output / name for name in ("samples.csv.gz", "summary.csv", "paired_deltas.csv")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite completed or partial metric evidence")
    write_union_csv(targets[0], rows)
    write_union_csv(targets[1], summary)
    write_union_csv(targets[2], paired)
    return dict(summary=summary, paired_deltas=paired,
                files={path.name: file_sha256(path) for path in targets})


def write_figures_and_drafts(output_dir, summary, paired):
    """Server-only static plotting from verified small tables; no models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    names = ("icc_aligned_six_points.pdf", "icc_aligned_six_points.png",
             "icc_aligned_paired_delta.pdf", "icc_aligned_paired_delta.png", "RESULTS_DRAFT.md")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Plots/drafts already exist; preserve partial stage evidence")
    colors = {"fdma": "#5685a6", "hybrid_g4": "#6ca88b"}
    labels = {"fdma": "Matched FDMA", "hybrid_g4": "Hybrid G4"}
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
        for arm in ARMS:
            points = sorted((row for row in summary if row["arm"] == arm), key=lambda row: row["gamma_db"])
            x = [row["gamma_db"] for row in points]
            axes[0].errorbar(x, [row["sum_rate_bps_hz_mean"] for row in points],
                            yerr=[1.96 * row["sum_rate_bps_hz_se"] for row in points],
                            label=labels[arm], color=colors[arm], marker="o", markersize=3, capsize=2)
            axes[1].plot(x, [row["dl_nmse_db_from_linear_mean"] for row in points],
                         label=labels[arm], color=colors[arm], marker="o", markersize=3)
        for ax in axes:
            ax.set_xlabel(r"Feedback $\gamma$ (dB)"); ax.set_xticks(GAMMAS)
            ax.grid(True, alpha=.2); ax.legend(frameon=False)
        axes[0].set_ylabel("Downlink sum rate (bps/Hz)")
        axes[1].set_ylabel("DL CSI NMSE (dB, linear mean)")
        fig.savefig(output / names[0]); fig.savefig(output / names[1], dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
        points = sorted((row for row in paired if row["metric"] == "sum_rate_bps_hz" and
                         row["scope"] == "point"), key=lambda row: row["gamma_db"])
        means = [row["mean"] for row in points]
        ax.errorbar([row["gamma_db"] for row in points], means,
                    yerr=[1.96 * row["batch_se"] for row in points], marker="o", markersize=3,
                    capsize=2, color=colors["hybrid_g4"])
        ax.axhline(0., color="0.5", linewidth=.7)
        ax.set(xlabel=r"Feedback $\gamma$ (dB)", ylabel="Hybrid G4 − FDMA (bps/Hz)", xticks=GAMMAS)
        ax.grid(True, alpha=.2)
        fig.savefig(output / names[2]); fig.savefig(output / names[3], dpi=180); plt.close(fig)
    low = next(row for row in paired if row["metric"] == "sum_rate_bps_hz" and row["scope"] == "low_0_5_10")
    all_six = next(row for row in paired if row["metric"] == "sum_rate_bps_hz" and row["scope"] == "all_six")
    text = ("# Internal result draft / 内部结果草稿\n\n"
            "This is a single-seed, independently initialized K8 adaptation, not an unchanged ICC reproduction. "
            "All six gamma points and signed differences are retained. RZF alpha is fixed at 0.03.\n\n"
            f"Hybrid−FDMA: low-three mean {low['mean']:.6g} bps/Hz, batch-paired 95% CI "
            f"[{low['batch_ci95_low']:.6g}, {low['batch_ci95_high']:.6g}]; all-six mean "
            f"{all_six['mean']:.6g}, batch-paired 95% CI [{all_six['batch_ci95_low']:.6g}, "
            f"{all_six['batch_ci95_high']:.6g}].\n\n"
            "本结果为单训练seed的K8独立机制适配，不代表ICC系统无改复现或跨训练seed稳定性。"
            "均保留六gamma点、负差值及逐样本/批配对区间。低三点和全六点先在同一个完整MU信道内平均，"
            "再跨MU或四MU批计算区间；用户和SNR点不增加独立样本量。当前帧真实功率仅为离线诊断。\n")
    (output / names[4]).write_text(text, encoding="utf-8")
    return {name: file_sha256(output / name) for name in names}


def make_small_package(output_dir, files, *, name="icc_aligned_small_results.zip"):
    """Explicit allowlist only: never include weights, raw samples or CSI."""
    output = Path(output_dir).resolve(); target = output / name
    if target.exists():
        raise FileExistsError("Small result package already exists")
    allowed = {".csv", ".json", ".md", ".pdf", ".png"}
    paths = []
    for relative in files:
        path = (output / relative).resolve()
        if path.parent != output or path.suffix not in allowed or not path.is_file() or path.is_symlink():
            raise ValueError("Only explicit small local tables/receipts/figures may be packaged")
        paths.append(path)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, arcname=path.name)
    return dict(path=str(target), sha256=file_sha256(target), files=[path.name for path in sorted(paths)])
