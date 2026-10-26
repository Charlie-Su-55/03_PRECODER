"""CSV-only internal reporting for Joint+Swin256, never checkpoint selection."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from as_receiver_batch_support import save_json, sha256
from as_receiver_results import rows, finite

SEEDS = (20261030, 20261031)
FB = (0, 5, 10, 15, 20, 25)
TEST_SEED = 20261120
TAG = "k4_joint_swin256_v1"
METHODS = ("plain_swin256", "joint_swin256", "spatial_joint", "full_control", "swin", "swin_adapted")
PAIRS = tuple(("joint_swin256", reference) for reference in
              ("plain_swin256", "spatial_joint", "swin_adapted", "swin", "full_control"))
FOCUS = PAIRS[:3]
LABELS = dict(plain_swin256="Plain Swin256", joint_swin256="Joint+Swin256",
              spatial_joint="Existing ASSpatial joint", full_control="Full +20k",
              swin="Original Swin", swin_adapted="Swin +20k (best includes step0)")


def interpret(summary, paired, training_seed):
    if type(training_seed) is not int or training_seed not in SEEDS:
        raise ValueError("Two fixed training seeds must be reported separately")
    expected = {(key, fb) for key in METHODS for fb in FB}
    keys = [(row["method_key"], float(row["feedback_snr_db"])) for row in summary]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("All six methods by six SNRs required exactly once")
    for row in summary:
        if (int(row["training_seed"]) != training_seed or int(row["seed"]) != TEST_SEED
                or int(row["num_samples"]) != 1600 or float(row["downlink_snr_db"]) != 25
                or str(row.get("preflight", "")).lower() not in ("false", "0")):
            raise ValueError("Expected the declared independent 1600-channel test, not preflight")
        mean, se, low, high = [finite(row, "sum_rate_" + name) for name in ("mean", "se", "ci95_low", "ci95_high")]
        if se < 0 or not low <= mean <= high:
            raise ValueError("Invalid sum-rate uncertainty")
    scopes = [("per_snr", str(fb)) for fb in FB] + [
        ("low_snr_mean", "0,5,10"), ("all_snr_mean", "0,5,10,15,20,25")]
    expected_pairs = {(method, reference, scope, points) for method, reference in PAIRS for scope, points in scopes}
    indexed = {}
    for row in paired:
        key = (row["method_key"], row["reference_method_key"], row["estimand"], row["feedback_snr_points"])
        if key not in expected_pairs or key in indexed:
            raise ValueError("Unexpected or duplicate paired estimate")
        if (int(row["training_seed"]) != training_seed or int(row["seed"]) != TEST_SEED
                or int(row["num_samples"]) != 1600 or int(row["num_batches"]) != 200
                or float(row["downlink_snr_db"]) != 25
                or str(row["training_seeds_pooled"]).lower() not in ("false", "0")):
            raise ValueError("Invalid paired sample/batch units or training-seed pooling")
        for prefix in ("delta_", "batch_delta_"):
            mean, se, low, high = [finite(row, prefix + name) for name in ("mean", "se", "ci95_low", "ci95_high")]
            if se < 0 or not low <= mean <= high:
                raise ValueError("Invalid paired uncertainty")
        indexed[key] = row
    if set(indexed) != expected_pairs:
        raise ValueError("Missing paired comparison; retain negative outcomes")
    comparisons = {}
    for method, reference in PAIRS:
        comparisons[method + "_minus_" + reference] = dict(
            low_snr=indexed[method, reference, "low_snr_mean", "0,5,10"],
            all_snr=indexed[method, reference, "all_snr_mean", "0,5,10,15,20,25"],
            per_snr=[indexed[method, reference, "per_snr", str(fb)] for fb in FB])
    primary = comparisons["joint_swin256_minus_plain_swin256"]
    low, high = primary["low_snr"], primary["per_snr"][-1]
    return dict(training_seed=training_seed, comparisons=comparisons, all_rates=summary,
                engineering_criterion_vs_plain=(finite(low, "delta_mean") >= .5
                    and finite(low, "batch_delta_ci95_low") > 0
                    and finite(high, "batch_delta_ci95_low") >= -.1),
                engineering_rule="low mean>=0.5; low batch CI lower>0; FB25 batch CI lower>=-0.1",
                training_seeds_pooled=False,
                scope="K4 pure-AS BS-decoder screening; fixed-checkpoint evaluation sampling only",
                budget="Same added20k for new arms; not equal historical budgets or parameters across all controls")


def result_paragraphs(interpretations, training_records):
    english, chinese = [], []
    for seed in SEEDS:
        item = interpretations[str(seed)]
        record = training_records[str(seed)]
        selection = ", ".join(f"{LABELS[arm]} best={record[arm]['best_step']}" for arm in METHODS[:2])
        english.append(f"Training seed {seed}: both new arms completed 20,000 additional updates; fixed validation selected {selection}.")
        chinese.append(f"训练seed {seed}：两个新臂均完成20,000个新增更新；固定验证集选择{selection}。")
        for method, reference in PAIRS:
            selected = item["comparisons"][method + "_minus_" + reference]
            low, high = selected["low_snr"], selected["per_snr"][-1]
            a, b, c = [finite(low, name) for name in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high")]
            d, e, f = [finite(high, name) for name in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high")]
            english.append(f"Seed {seed}, {LABELS[method]} minus {LABELS[reference]}: the within-channel 0/5/10-dB average difference is {a:+.4f} bps/Hz (batch-paired 95% CI [{b:+.4f}, {c:+.4f}]); the 25-dB difference is {d:+.4f} [{e:+.4f}, {f:+.4f}].")
            chinese.append(f"seed {seed}，{LABELS[method]}减{LABELS[reference]}：0/5/10 dB先在同一信道内平均的速率差为{a:+.4f} bps/Hz（batch配对95%区间[{b:+.4f}, {c:+.4f}]）；25 dB速率差为{d:+.4f}，区间[{e:+.4f}, {f:+.4f}]。")
        passed = item["engineering_criterion_vs_plain"]
        english.append(f"Seed {seed}: the predeclared engineering criterion against Plain Swin256 is {'met' if passed else 'not met'}; this is not a cross-training-seed significance claim.")
        chinese.append(f"seed {seed}：相对Plain Swin256的预先固定工程标准{'满足' if passed else '未满足'}；这不构成跨训练seed的显著性结论。")
    english.append("Internal K4 pure-AS decoder screening only, not a replacement for the original hybrid, user-load, or imperfect-CSI experiments. All six SNR points, both training seeds, and negative outcomes are retained. Sample- and batch-paired intervals are both saved; neither estimates training-seed uncertainty. Stored parameter counts are reported, not equal-parameter or latency comparisons. Recovered/corrected codeword errors use detached truth only after decoding and are secondary offline diagnostics, never decoder inputs or selection criteria. Common feedback RNG does not imply equal absolute noise or equal UE energy. No manuscript is updated automatically.")
    chinese.append("这只是K4 pure-AS decoder的内部筛选，不能替换原Hybrid、用户负载或不完美CSI实验。完整保留六个SNR点、两个训练seed及负结果；逐样本和batch配对区间均保存，均不代表训练seed不确定性。参数量按真实结构报告，不宣称等参数或时延优势。恢复/校正码字误差仅在解码完成后使用分离真值作次要离线诊断，真值不进入decoder，也不用于选择权重。相同反馈RNG不等于相同绝对噪声或UE总能量。本任务不会自动修改论文。")
    return english, chinese


def organize_results(folder, evaluations, training_records):
    from evaluate_proposal_joint_swin import protocol, artifact_hashes
    folder = Path(folder)
    if set(evaluations) != {str(seed) for seed in SEEDS} or set(training_records) != {str(seed) for seed in SEEDS}:
        raise ValueError("Both independently reported training seeds are required")
    summaries, paired, interpretations = {}, {}, {}
    for seed in SEEDS:
        source = Path(evaluations[str(seed)])
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status") != "completed" or manifest.get("results_verified") is not True
                or manifest.get("freeze_checks", {}).get("passed") is not True
                or manifest.get("protocol") != protocol(seed)
                or manifest.get("output_sha256") != artifact_hashes(source)):
            raise ValueError("Evaluation protocol/freeze/artifact evidence not verified")
        for arm in METHODS[:2]:
            trained = training_records[str(seed)][arm]
            loaded = manifest["methods"][arm]
            for name in ("best_step", "best_sha256", "latest_sha256"):
                saved_name = "selected_best_step" if name == "best_step" else name
                if trained[name] != loaded[saved_name]:
                    raise ValueError("Training/evaluation selected checkpoint identity differs")
        summaries[seed] = rows(source / f"summary_{TAG}.csv")
        paired[seed] = rows(source / f"paired_deltas_{TAG}.csv")
        interpretations[str(seed)] = interpret(summaries[seed], paired[seed], seed)
    output = folder / ("export_" + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    for seed in SEEDS:
        source = Path(evaluations[str(seed)])
        for prefix in ("summary", "paired_deltas", "parameters", "component_parameters", "soft_codeword_summary"):
            shutil.copyfile(source / f"{prefix}_{TAG}.csv", output / f"seed{seed}_{prefix}_{TAG}.csv")
        shutil.copyfile(source / "manifest.json", output / f"seed{seed}_evaluation_manifest.json")
    save_json(output / "comparison_summary.json", interpretations)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for axis, seed in zip(axes, SEEDS):
        for method in METHODS:
            selected = sorted((row for row in summaries[seed] if row["method_key"] == method),
                              key=lambda row: float(row["feedback_snr_db"]))
            axis.plot(FB, [finite(row, "sum_rate_mean") for row in selected], marker="o", markersize=3,
                      linewidth=1.1, label=LABELS[method])
        axis.set(xlabel="Feedback SNR (dB)", title=f"Training seed {seed}", xticks=FB)
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Sum rate (bps/Hz)")
    axes[-1].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("joint_swin256_all_six_points." + suffix), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for axis, seed in zip(axes, SEEDS):
        for method, reference in FOCUS:
            selected = interpretations[str(seed)]["comparisons"][method + "_minus_" + reference]["per_snr"]
            means = [finite(row, "delta_mean") for row in selected]
            errors = [[mean - finite(row, "batch_delta_ci95_low") for mean, row in zip(means, selected)],
                      [finite(row, "batch_delta_ci95_high") - mean for mean, row in zip(means, selected)]]
            axis.errorbar(FB, means, yerr=errors, capsize=3, marker="o", markersize=3,
                          linewidth=1.1, label=LABELS[method] + " minus " + LABELS[reference])
        axis.axhline(0, color=".5", linestyle="--", linewidth=.8)
        axis.set(xlabel="Feedback SNR (dB)", title=f"Training seed {seed}", xticks=FB)
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Paired rate difference (bps/Hz); batch 95% CI")
    axes[-1].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("joint_swin256_batch_paired_ci." + suffix), dpi=160)
    plt.close(fig)
    english, chinese = result_paragraphs(interpretations, training_records)
    for name, paragraphs in (("results_draft_en.md", english), ("results_draft_zh.md", chinese)):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write("\n\n".join(paragraphs) + "\n")
    files = sorted(path for path in output.iterdir() if path.is_file())
    return dict(export_directory=str(output), per_seed=interpretations, training_seeds_pooled=False,
                files={str(path): sha256(path) for path in files}, _evidence=files)
