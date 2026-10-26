"""CSV-only ASSpatial result organization; no model/checkpoint imports or selection."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from as_receiver_batch_support import save_json, sha256
from as_receiver_results import rows, finite

SEEDS = (20261030, 20261031)
FB = (0, 5, 10, 15, 20, 25)
ARMS = ("frozen_ue", "joint")
METHODS = ARMS + ("full_control", "as_receiver", "swin")
PAIRS = tuple((arm, ref) for arm in ARMS for ref in ("full_control", "as_receiver", "swin")) + (("joint", "frozen_ue"),)
TAG = "k4_as_spatial_v1"
TEST_SEED = 20261103


def interpret(summary, paired, seed):
    expected = {(method, snr) for method in METHODS for snr in FB}
    keys = [(r["method_key"], int(float(r["feedback_snr_db"]))) for r in summary]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("All five methods and six SNRs are required exactly once")
    for row in summary:
        if (int(row["training_seed"]) != seed or int(row["seed"]) != TEST_SEED or int(row["num_samples"]) != 1600
                or str(row.get("preflight", "")).lower() not in ("false", "0")):
            raise ValueError("Not this independent screening protocol")
        for metric in ("sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high"):
            finite(row, metric)
    scopes = [("per_snr", str(fb)) for fb in FB] + [("low_snr_mean", "0,5,10"), ("all_snr_mean", "0,5,10,15,20,25")]
    expected_pairs = {(m, r, scope, points) for m, r in PAIRS for scope, points in scopes}
    indexed = {}
    for row in paired:
        key = (row["method_key"], row["reference_method_key"], row["estimand"], row["feedback_snr_points"])
        if key not in expected_pairs or key in indexed:
            raise ValueError("Unexpected/duplicate paired condition")
        if (int(row["training_seed"]) != seed or int(row["seed"]) != TEST_SEED
                or int(row["num_samples"]) != 1600 or int(row["num_batches"]) != 200
                or str(row["training_seeds_pooled"]).lower() not in ("false", "0")):
            raise ValueError("Paired sample/seed definition differs")
        for field in ("delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high",
                      "batch_delta_mean", "batch_delta_se", "batch_delta_ci95_low", "batch_delta_ci95_high"):
            finite(row, field)
        indexed[key] = row
    if set(indexed) != expected_pairs:
        raise ValueError("Missing paired comparisons; never omit a negative result")
    outcome = {}
    for arm in ARMS:
        low = indexed[arm, "full_control", "low_snr_mean", "0,5,10"]
        high = indexed[arm, "full_control", "per_snr", "25"]
        outcome[arm] = dict(low_snr_minus_full=low, fb25_minus_full=high,
            engineering_continuation_criterion_met=finite(low, "delta_mean") >= 0.5
                and finite(low, "batch_delta_ci95_low") > 0 and finite(high, "batch_delta_ci95_low") >= -0.1,
            six_point_gap_to_swin=[indexed[arm, "swin", "per_snr", str(fb)] for fb in FB])
    return dict(training_seed=seed, arms=outcome, training_seeds_pooled=False,
                inference_scope="fixed-checkpoint evaluation sampling; not multi-training-seed uncertainty",
                failure_interpretation="new BS is randomly initialized; failure does not establish a UE performance ceiling")


def organize_results(folder, evaluations):
    folder = Path(folder)
    output = folder / "export"
    if output.exists():
        output = folder / ("export_retry_" + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    summaries, paired, interpretations = {}, {}, {}
    for seed in SEEDS:
        source = Path(evaluations[str(seed)])
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("freeze_checks", {}).get("passed") is not True:
            raise ValueError("Only completed frozen evaluations may be summarized")
        summaries[seed] = rows(source / f"summary_{TAG}.csv")
        paired[seed] = rows(source / f"paired_deltas_{TAG}.csv")
        interpretations[str(seed)] = interpret(summaries[seed], paired[seed], seed)
        for basename in (f"summary_{TAG}.csv", f"paired_deltas_{TAG}.csv", f"parameters_{TAG}.csv",
                         f"spatial_parameters_{TAG}.csv", f"soft_codeword_summary_{TAG}.csv"):
            original = source / basename
            if not original.is_file():
                raise FileNotFoundError(original)
            shutil.copyfile(original, output / f"seed{seed}_{basename}")
        shutil.copyfile(source / "manifest.json", output / f"seed{seed}_evaluation_manifest.json")
    save_json(output / "engineering_summary.json", interpretations)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#397EA8", "#BA715A")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        for arm, color in zip(ARMS, colors):
            selected = sorted([r for r in paired[seed] if r["method_key"] == arm and r["reference_method_key"] == "full_control"
                               and r["estimand"] == "per_snr"], key=lambda r: float(r["feedback_snr_db"]))
            means = [finite(r, "delta_mean") for r in selected]
            errors = [[v - finite(r, "batch_delta_ci95_low") for v, r in zip(means, selected)],
                      [finite(r, "batch_delta_ci95_high") - v for v, r in zip(means, selected)]]
            ax.errorbar(FB, means, yerr=errors, marker="o", capsize=3, linewidth=1.2, color=color, label=arm)
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Feedback SNR (dB)", title=f"Training seed {seed}", xticks=FB)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("ASSpatial - Full rate (bps/Hz)")
    axes[-1].legend(frameon=False)
    fig.savefig(output / "as_spatial_full_batch_paired_ci.pdf")
    fig.savefig(output / "as_spatial_full_batch_paired_ci.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        for method in METHODS:
            selected = sorted([r for r in summaries[seed] if r["method_key"] == method], key=lambda r: float(r["feedback_snr_db"]))
            ax.plot(FB, [finite(r, "sum_rate_mean") for r in selected], marker="o", markersize=3, linewidth=1.1, label=method)
        ax.set(xlabel="Feedback SNR (dB)", title=f"Training seed {seed}", xticks=FB)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Sum rate (bps/Hz)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "as_spatial_all_methods.pdf")
    fig.savefig(output / "as_spatial_all_methods.png", dpi=160)
    plt.close(fig)
    english, chinese = [], []
    for seed in SEEDS:
        for arm in ARMS:
            result = interpretations[str(seed)]["arms"][arm]
            low, high = result["low_snr_minus_full"], result["fb25_minus_full"]
            a, b, c = (finite(low, key) for key in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high"))
            d, e, f = (finite(high, key) for key in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high"))
            english.append(f"Seed {seed}, {arm}: the 0/5/10-dB within-sample average gain over Full was {a:+.4f} bps/Hz (batch-paired 95% CI [{b:+.4f}, {c:+.4f}]); at 25 dB the difference was {d:+.4f} [{e:+.4f}, {f:+.4f}]. The predeclared engineering criterion was {'met' if result['engineering_continuation_criterion_met'] else 'not met'}.")
            chinese.append(f"训练 seed {seed}、{arm}：相对 Full，0/5/10 dB 先在样本内平均后的差值为 {a:+.4f} bps/Hz，batch 配对95%区间 [{b:+.4f}, {c:+.4f}]；25 dB 差值为 {d:+.4f}，区间 [{e:+.4f}, {f:+.4f}]。{'达到' if result['engineering_continuation_criterion_met'] else '未达到'}预先固定的工程筛选标准。")
    english.append("Both seeds and all SNRs are reported, including negative gains. They share 1600 test realizations but are not pooled into 3200 independent samples. Sample- and batch-level intervals are both retained; the criterion uses batch intervals and is not a publication guarantee. The new BS starts randomly; unsuccessful screening does not prove a ceiling of the UE representation. This is not an equal-parameter or equal-history comparison. True codewords enter offline error diagnostics only, never the deployed decoder.")
    chinese.append("两个训练seed分别报告全部SNR与负增益；共用1600测试信道，不合并成3200独立样本。逐样本和batch区间均保留，工程判断使用batch区间，不代表期刊验收。新BS随机初始化，筛选失败不能直接证明UE表示达到性能上限。本实验不是等参数或等历史训练预算比较。真实码字仅用于离线误差统计，不进入可部署decoder。")
    for name, paragraphs in (("results_draft_en.md", english), ("results_draft_zh.md", chinese)):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write("\n\n".join(paragraphs) + "\n")
    files = sorted(p for p in output.iterdir() if p.is_file())
    return dict(export_directory=str(output), per_seed=interpretations,
                files={str(p): sha256(p) for p in files}, _evidence=files, training_seeds_pooled=False)
