"""CSV-only ASReceiver batch tables, vector plots and bilingual internal drafts.

Never used to select checkpoints/seeds/SNRs. Negative results remain visible.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

from as_receiver_batch_support import save_json, sha256

SEEDS = (20261030, 20261031)
FB = (0, 5, 10, 15, 20, 25)
METHODS = ("full_control", "as_receiver", "robust_frozen", "swin", "as_receiver_off")
COMPARISONS = (("as_receiver", "full_control"), ("full_control", "robust_frozen"),
               ("as_receiver", "robust_frozen"), ("as_receiver", "swin"), ("as_receiver", "as_receiver_off"))
TAG = "k4_as_receiver_v1"


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}")
    return value


def interpret(summary, paired, seed):
    expected = {(method, snr) for method in METHODS for snr in FB}
    actual = [(row["method_key"], int(float(row["feedback_snr_db"]))) for row in summary]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Expected all five methods and six SNRs; no omissions/duplicates")
    for row in summary:
        if (int(row["num_samples"]) != 1600 or int(row["training_seed"]) != seed
                or int(row["seed"]) != 20261101 or str(row.get("preflight", "")).lower() not in ("false", "0")):
            raise ValueError("Not the prescribed formal screening test")
        for key in ("sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high"):
            finite(row, key)
    scopes = [("per_snr", str(snr)) for snr in FB] + [("low_snr_mean", "0,5,10"), ("all_snr_mean", "0,5,10,15,20,25")]
    expected_pairs = {(method, reference, scope, points) for method, reference in COMPARISONS for scope, points in scopes}
    actual_pairs = []
    for row in paired:
        key = (row["method_key"], row["reference_method_key"], row["estimand"], row["feedback_snr_points"])
        actual_pairs.append(key)
        if (int(row["num_samples"]) != 1600 or int(row["num_batches"]) != 200
                or int(row["training_seed"]) != seed or int(row["seed"]) != 20261101
                or str(row["training_seeds_pooled"]).lower() not in ("false", "0")):
            raise ValueError("Invalid paired sample/seed definition")
        for name in ("delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high",
                     "batch_delta_mean", "batch_delta_se", "batch_delta_ci95_low", "batch_delta_ci95_high"):
            finite(row, name)
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Missing/duplicate paired conditions; retain all comparisons and SNRs")
    def select(estimand, snr=None):
        selected = [row for row in paired if row["method_key"] == "as_receiver"
                    and row["reference_method_key"] == "full_control"
                    and row["estimand"] == estimand
                    and (snr is None or float(row["feedback_snr_db"]) == snr)]
        if len(selected) != 1:
            raise ValueError("Missing/duplicate predeclared paired estimand")
        row = selected[0]
        if int(row["num_samples"]) != 1600 or int(row["num_batches"]) != 200 or int(row["training_seed"]) != seed:
            raise ValueError("Paired units/training seed do not match")
        for key in ("delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high",
                    "batch_delta_mean", "batch_delta_se", "batch_delta_ci95_low", "batch_delta_ci95_high"):
            finite(row, key)
        return row
    low, high = select("low_snr_mean"), select("per_snr", 25)
    low_pass = finite(low, "delta_mean") >= 0.5 and finite(low, "batch_delta_ci95_low") > 0.0
    high_pass = finite(high, "batch_delta_ci95_low") >= -0.1
    indexed = {key: row for key, row in zip(actual, summary)}
    gap = [dict(feedback_snr_db=snr,
                as_receiver_minus_swin=finite(indexed[("as_receiver", snr)], "sum_rate_mean")
                    - finite(indexed[("swin", snr)], "sum_rate_mean")) for snr in FB]
    return dict(training_seed=seed, low_snr_as_receiver_minus_full=low, fb25_as_receiver_minus_full=high,
                low_snr_gain_criterion_met=low_pass, fb25_degradation_criterion_met=high_pass,
                engineering_continuation_criterion_met=low_pass and high_pass,
                six_point_gap_to_swin=gap, seeds_pooled=False,
                caveat="Fixed-checkpoint sampling intervals, not multi-training-seed inference; Swin training budget differs")


def organize_results(folder, evaluations):
    folder = Path(folder)
    output = folder / "export"
    if output.exists():
        # Resume after an interrupted export does not overwrite previous drafts.
        output = folder / ("export_retry_" + __import__("uuid").uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    interpretations, summaries, pairs = {}, {}, {}
    for seed in SEEDS:
        source = Path(evaluations[str(seed)])
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("freeze_checks", {}).get("passed") is not True:
            raise ValueError("Failed or unfrozen evaluation cannot enter results summary")
        summaries[seed] = rows(source / f"summary_{TAG}.csv")
        pairs[seed] = rows(source / f"paired_deltas_{TAG}.csv")
        interpretations[str(seed)] = interpret(summaries[seed], pairs[seed], seed)
        for basename in (f"summary_{TAG}.csv", f"paired_deltas_{TAG}.csv", f"parameters_{TAG}.csv",
                         f"receiver_parameters_{TAG}.csv"):
            original = source / basename
            if original.is_file():
                shutil.copyfile(original, output / f"seed{seed}_{basename}")
    save_json(output / "engineering_summary.json", interpretations)
    # Import only after reading/validating completed results; never import models.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#397EA8", "#BA715A")
    fig, ax = plt.subplots(figsize=(6.3, 3.4), layout="constrained")
    for seed, color in zip(SEEDS, colors):
        selected = sorted([row for row in pairs[seed] if row["method_key"] == "as_receiver"
                    and row["reference_method_key"] == "full_control" and row["estimand"] == "per_snr"],
                    key=lambda row: float(row["feedback_snr_db"]))
        if [int(float(row["feedback_snr_db"])) for row in selected] != list(FB):
            raise ValueError("Incomplete six-point paired plot")
        mean = [finite(row, "delta_mean") for row in selected]
        err = [[value - finite(row, "batch_delta_ci95_low") for value, row in zip(mean, selected)],
               [finite(row, "batch_delta_ci95_high") - value for value, row in zip(mean, selected)]]
        ax.errorbar(FB, mean, yerr=err, marker="o", capsize=3, linewidth=1.3, color=color,
                    label=f"Fine-tuning seed {seed}")
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Feedback SNR (dB)", ylabel="ASReceiver - Full sum rate (bps/Hz)", xticks=FB)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(output / "as_receiver_full_paired_batch_ci.pdf")
    fig.savefig(output / "as_receiver_full_paired_batch_ci.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        for method in METHODS:
            selected = sorted([row for row in summaries[seed] if row["method_key"] == method],
                              key=lambda row: float(row["feedback_snr_db"]))
            ax.plot(FB, [finite(row, "sum_rate_mean") for row in selected], marker="o", markersize=3,
                    linewidth=1.1, label=method.replace("_", " "))
        ax.set(xlabel="Feedback SNR (dB)", title=f"Fine-tuning seed {seed}", xticks=FB)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Sum rate (bps/Hz)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "as_receiver_all_methods.pdf")
    fig.savefig(output / "as_receiver_all_methods.png", dpi=160)
    plt.close(fig)
    english, chinese = [], []
    for seed in SEEDS:
        record = interpretations[str(seed)]
        low, high = record["low_snr_as_receiver_minus_full"], record["fb25_as_receiver_minus_full"]
        values = [finite(low, "delta_mean"), finite(low, "batch_delta_ci95_low"), finite(low, "batch_delta_ci95_high"),
                  finite(high, "delta_mean"), finite(high, "batch_delta_ci95_low"), finite(high, "batch_delta_ci95_high")]
        a, b, c, d, e, f = values
        outcome = "met" if record["engineering_continuation_criterion_met"] else "did not meet"
        english.append(f"For fine-tuning seed {seed}, the ASReceiver-minus-Full mean difference over 0/5/10 dB was {a:+.4f} bps/Hz (batch-paired 95% CI [{b:+.4f}, {c:+.4f}]). At 25 dB, the difference was {d:+.4f} bps/Hz [{e:+.4f}, {f:+.4f}]. This run {outcome} the predeclared engineering continuation criterion.")
        chinese.append(f"微调 seed {seed}：0/5/10 dB 样本内平均后，ASReceiver−Full 为 {a:+.4f} bps/Hz，batch 配对95%区间为 [{b:+.4f}, {c:+.4f}]；25 dB 差值为 {d:+.4f}，区间为 [{e:+.4f}, {f:+.4f}]。本次{'达到' if record['engineering_continuation_criterion_met'] else '未达到'}预先固定的工程扩展标准。")
    caveat_en = "Both fine-tuning seeds are reported separately on the same 1600 test channel realizations; they are not pooled into 3200 independent samples. Both sample- and batch-level paired intervals are retained; decisions use the predeclared batch intervals. These intervals concern evaluation sampling of fixed checkpoints, not across-training-seed stability. Receiver-off is an inference intervention on one trained checkpoint, not a separately retrained ablation. No SNR point or negative result is removed. This is not a publication acceptance criterion."
    caveat_zh = "两个微调seed分别报告、共用1600测试信道，不合并成3200独立样本。逐样本与batch配对区间均保留，工程判断固定使用batch区间。区间仅针对固定checkpoint的评估采样，不代表跨训练seed稳定性。关闭新增支路是固定权重推理干预，不是重训消融。保留全部SNR与负结果；工程标准不等于期刊录用标准。"
    for name, paragraphs in (("results_draft_en.md", english + [caveat_en]), ("results_draft_zh.md", chinese + [caveat_zh])):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write("\n\n".join(paragraphs) + "\n")
    files = sorted(path for path in output.iterdir() if path.is_file())
    return dict(export_directory=str(output), per_seed=interpretations,
                files={str(path): sha256(path) for path in files}, _evidence=files,
                not_a_publication_guarantee=True, training_seeds_pooled=False)
