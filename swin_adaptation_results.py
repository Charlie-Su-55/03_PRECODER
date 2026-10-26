"""CSV-only reporting for K4 pure-AS decoder extension / Swin adaptation.

Never used for checkpoint selection. Both training seeds and all six SNRs stay
visible, including negative deltas. No model imports or performance execution.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from as_receiver_batch_support import save_json, sha256
from as_receiver_results import rows, finite

SEEDS = (20261030, 20261031)
FB = (0, 5, 10, 15, 20, 25)
TEST_SEED = 20261110
TAG = "k4_swin_adaptation_v1"
METHODS = ("frozen_ue", "joint", "full_control", "as_receiver", "swin", "swin_adapted")
FOCUS = (("joint", "swin"), ("joint", "swin_adapted"), ("swin_adapted", "swin"))
PAIRS = FOCUS + (("joint", "full_control"), ("joint", "as_receiver"), ("joint", "frozen_ue"))
LABELS = {"frozen_ue": "ASSpatial frozen UE", "joint": "ASSpatial joint",
          "full_control": "Full +20k", "as_receiver": "ASReceiver +20k",
          "swin": "Original Swin", "swin_adapted": "Swin +20k (best includes step0)"}


def interpret(summary, paired, training_seed):
    expected = {(m, fb) for m in METHODS for fb in FB}
    keys = [(r["method_key"], int(float(r["feedback_snr_db"]))) for r in summary]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Require all six methods x six SNRs exactly once")
    for row in summary:
        if (int(row["training_seed"]) != training_seed or int(row["seed"]) != TEST_SEED
                or int(row["num_samples"]) != 1600
                or str(row.get("preflight", "")).lower() not in ("false", "0")):
            raise ValueError("Not the declared 1600-channel independent screening test")
        for metric in ("sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high"):
            finite(row, metric)
    scopes = [("per_snr", str(s)) for s in FB] + [("low_snr_mean", "0,5,10"),
                                                  ("all_snr_mean", "0,5,10,15,20,25")]
    expected_pairs = {(a, b, scope, points) for a, b in PAIRS for scope, points in scopes}
    indexed = {}
    for row in paired:
        key = (row["method_key"], row["reference_method_key"], row["estimand"], row["feedback_snr_points"])
        if key not in expected_pairs or key in indexed:
            raise ValueError("Unexpected or duplicate paired estimand")
        if (int(row["training_seed"]) != training_seed or int(row["seed"]) != TEST_SEED
                or int(row["num_samples"]) != 1600 or int(row["num_batches"]) != 200
                or str(row["training_seeds_pooled"]).lower() not in ("false", "0")):
            raise ValueError("Invalid paired units or pooled training seeds")
        for field in ("delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high",
                      "batch_delta_mean", "batch_delta_se", "batch_delta_ci95_low", "batch_delta_ci95_high"):
            finite(row, field)
        indexed[key] = row
    if set(indexed) != expected_pairs:
        raise ValueError("Missing comparisons; negative results may not be omitted")
    comparisons = {}
    for method, ref in PAIRS:
        comparisons[method + "_minus_" + ref] = dict(
            low_snr=indexed[method, ref, "low_snr_mean", "0,5,10"],
            all_snr=indexed[method, ref, "all_snr_mean", "0,5,10,15,20,25"],
            per_snr=[indexed[method, ref, "per_snr", str(fb)] for fb in FB])
    low = comparisons["joint_minus_full_control"]["low_snr"]
    high = comparisons["joint_minus_full_control"]["per_snr"][-1]
    return dict(training_seed=training_seed, comparisons=comparisons,
        engineering_criterion_vs_full=(finite(low, "delta_mean") >= .5
            and finite(low, "batch_delta_ci95_low") > 0
            and finite(high, "batch_delta_ci95_low") >= -.1),
        all_rates=summary, training_seeds_pooled=False,
        scope="K4 pure-AS decoder extension only; fixed-checkpoint sampling intervals",
        budget="Equal additional updates, not matched historical total training budgets")


def result_paragraphs(interpretations, training_records):
    en, zh = [], []
    for seed in SEEDS:
        item = interpretations[str(seed)]
        step = training_records[str(seed)]["best_step"]
        en.append(f"Training seed {seed}: Swin completed 20,000 additional updates; validation selected step {step} from candidates including the original source at step 0.")
        zh.append(f"训练seed {seed}：Swin完成20,000个新增更新；固定验证集从包含原始权重step0的候选中选择step {step}。")
        for method, ref in FOCUS:
            selected = item["comparisons"][method + "_minus_" + ref]
            low, high = selected["low_snr"], selected["per_snr"][-1]
            a, b, c = (finite(low, k) for k in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high"))
            d, e, f = (finite(high, k) for k in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high"))
            en.append(f"Seed {seed}, {LABELS[method]} minus {LABELS[ref]}: the within-channel 0/5/10-dB average difference is {a:+.4f} bps/Hz (batch-paired 95% CI [{b:+.4f}, {c:+.4f}]); at 25 dB it is {d:+.4f} [{e:+.4f}, {f:+.4f}].")
            zh.append(f"seed {seed}，{LABELS[method]}减{LABELS[ref]}：0/5/10 dB先在同一信道内平均的差值为{a:+.4f} bps/Hz（batch配对95%区间[{b:+.4f}, {c:+.4f}]）；25 dB差值为{d:+.4f}，区间[{e:+.4f}, {f:+.4f}]。")
    en.append("Internal draft for the K4 pure-AS decoder extension, not a replacement for the original hybrid, multi-user-load or imperfect-CSI experiments. This control matches only the additional 20k-update budget, not the historical total budget or parameter count. All six SNR points and both training seeds are reported separately, including negative results. Sample- and batch-paired intervals are both retained; neither describes uncertainty across training seeds. Common feedback RNG does not imply equal absolute noise or equal UE energy. No checkpoint or SNR was selected using test results. These outputs do not update the manuscript automatically.")
    zh.append("本内部草稿仅定位于K4 pure-AS decoder扩展，不能替换原Hybrid、多用户负载或误差实验。这里只匹配新增20k更新预算，不是历史总预算或等参数比较。六个SNR点和两个训练seed分别报告，负结果完整保留；逐样本与batch配对区间均保留，均不代表跨训练seed的不确定性。相同反馈RNG不等于相同绝对噪声或UE总能量；未按测试结果选择权重或SNR。本批不会自动更新论文主稿。")
    return en, zh


def verified_result_record(source, seed):
    from evaluate_swin_adaptation import protocol, artifact_hashes
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("freeze_checks", {}).get("passed") is not True
            or manifest.get("protocol") != protocol(seed)
            or manifest.get("output_sha256") != artifact_hashes(source)):
        raise ValueError("Require completed, frozen, hash-verified evaluations under the fixed protocol")
    return manifest


def organize_results(folder, evaluations, training_records):
    folder = Path(folder)
    output = folder / ("export_" + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    summaries, paired, interpretations = {}, {}, {}
    for seed in SEEDS:
        source = Path(evaluations[str(seed)])
        manifest = verified_result_record(source, seed)
        summaries[seed] = rows(source / f"summary_{TAG}.csv")
        paired[seed] = rows(source / f"paired_deltas_{TAG}.csv")
        interpretations[str(seed)] = interpret(summaries[seed], paired[seed], seed)
        for prefix in ("summary", "paired_deltas", "parameters"):
            shutil.copyfile(source / f"{prefix}_{TAG}.csv", output / f"seed{seed}_{prefix}_{TAG}.csv")
        shutil.copyfile(source / "manifest.json", output / f"seed{seed}_evaluation_manifest.json")
        if verified_result_record(source, seed) != manifest:
            raise ValueError("Evaluation evidence changed during result organization")
    save_json(output / "comparison_summary.json", interpretations)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        for method in METHODS:
            selected = sorted((r for r in summaries[seed] if r["method_key"] == method), key=lambda r: float(r["feedback_snr_db"]))
            ax.plot(FB, [finite(r, "sum_rate_mean") for r in selected], marker="o", markersize=3, linewidth=1.1, label=LABELS[method])
        ax.set(xlabel="Feedback SNR (dB)", title=f"Training seed {seed}", xticks=FB)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Sum rate (bps/Hz)")
    axes[-1].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("swin_adaptation_all_six_points." + suffix), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        for a, b in FOCUS:
            selected = interpretations[str(seed)]["comparisons"][a + "_minus_" + b]["per_snr"]
            means = [finite(r, "delta_mean") for r in selected]
            errors = [[v - finite(r, "batch_delta_ci95_low") for v, r in zip(means, selected)],
                      [finite(r, "batch_delta_ci95_high") - v for v, r in zip(means, selected)]]
            ax.errorbar(FB, means, yerr=errors, capsize=3, marker="o", markersize=3,
                        linewidth=1.1, label=LABELS[a] + " − " + LABELS[b])
        ax.axhline(0, color=".5", linestyle="--", linewidth=.8)
        ax.set(xlabel="Feedback SNR (dB)", title=f"Training seed {seed}", xticks=FB)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Paired rate difference (bps/Hz); batch 95% CI")
    axes[-1].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("swin_adaptation_batch_paired_ci." + suffix), dpi=160)
    plt.close(fig)
    english, chinese = result_paragraphs(interpretations, training_records)
    for name, paragraphs in (("results_draft_en.md", english), ("results_draft_zh.md", chinese)):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write("\n\n".join(paragraphs) + "\n")
    files = sorted(p for p in output.iterdir() if p.is_file())
    return dict(export_directory=str(output), per_seed=interpretations, training_seeds_pooled=False,
                files={str(p): sha256(p) for p in files}, _evidence=files)
