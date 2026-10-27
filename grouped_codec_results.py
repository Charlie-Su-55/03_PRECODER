"""CSV-only grouped-codec screening report; no model, channel or checkpoint loading."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import uuid

from as_receiver_batch_support import save_json, sha256
from as_receiver_results import rows, finite

SEEDS = (20261030, 20261031)
GAMMAS, TEST_SEED, TAG = (0, 5, 10, 15, 20, 25), 20261210, "k4_grouped_codec_v1"
METHODS = ("legacy_all4", "ordered_all4", "legacy_group2", "ordered_group2", "ordered_fdma")
LABELS = dict(legacy_all4="A: Legacy UE, all-4 AS", ordered_all4="B: Ordered UE, all-4 AS",
              legacy_group2="C: Legacy UE, grouped AS", ordered_group2="D: Ordered UE, grouped AS",
              ordered_fdma="E: Ordered-UE Joint-FDMA")
CONTRASTS = {
    "B_minus_A": {"ordered_all4": 1, "legacy_all4": -1},
    "D_minus_C": {"ordered_group2": 1, "legacy_group2": -1},
    "C_minus_A": {"legacy_group2": 1, "legacy_all4": -1},
    "D_minus_B": {"ordered_group2": 1, "ordered_all4": -1},
    "D_minus_E": {"ordered_group2": 1, "ordered_fdma": -1},
    "B_minus_E": {"ordered_all4": 1, "ordered_fdma": -1},
    "interaction_D_minus_C_minus_B_plus_A": {
        "ordered_group2": 1, "legacy_group2": -1, "ordered_all4": -1, "legacy_all4": 1},
}
CONTRAST_LABELS = dict(B_minus_A="B - A: ordered UE, all-4", D_minus_C="D - C: ordered UE, grouped",
    C_minus_A="C - A: grouping, legacy", D_minus_B="D - B: grouping, ordered",
    D_minus_E="D - E: grouped AS vs Joint-FDMA", B_minus_E="B - E: all-4 AS vs Joint-FDMA",
    interaction_D_minus_C_minus_B_plus_A="D - C - B + A: interaction")


def _uncertainty(row, prefix):
    mean, se, low, high = [finite(row, prefix + name) for name in ("mean", "se", "ci95_low", "ci95_high")]
    if se < 0 or not math.isclose(low, mean - 1.96 * se, rel_tol=1e-6, abs_tol=1e-6) or not math.isclose(high, mean + 1.96 * se, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Invalid predeclared normal-approximation interval")


def interpret(summary, paired, training_seed):
    if type(training_seed) is not int or training_seed not in SEEDS:
        raise ValueError("Report the two training seeds separately")
    expected = {(key, gamma) for key in METHODS for gamma in GAMMAS}
    actual = [(r["method_key"], finite(r, "gamma_db")) for r in summary]
    if len(actual) != 30 or set(actual) != expected:
        raise ValueError("All five methods and six gamma points are required exactly once")
    for row in summary:
        if "feedback_snr_db" in row or "allocation_df" in row or "allocation_da" in row:
            raise ValueError("Old received-SNR/allocation protocol cannot be relabeled as gamma")
        if (int(row["training_seed"]) != training_seed or int(row["seed"]) != TEST_SEED
                or int(row["num_samples"]) != 1600 or int(row["num_batches"]) != 200
                or int(row["batch_size"]) != 8 or finite(row, "downlink_snr_db") != 25
                or str(row.get("preflight", "")).lower() not in ("false", "0")):
            raise ValueError("Expected the independent 1600-channel screening test, not preflight")
        gamma = finite(row, "gamma_db")
        if not math.isclose(finite(row, "noise_variance_N0"), .25 * 10 ** (-gamma / 10), rel_tol=1e-10):
            raise ValueError("Absolute noise protocol changed")
        _uncertainty(row, "sum_rate_")
    scopes = [("per_gamma", str(g)) for g in GAMMAS] + [
        ("low_gamma_mean", "0,5,10"), ("all_gamma_mean", "0,5,10,15,20,25")]
    expected_pairs = {(name, scope, points) for name in CONTRASTS for scope, points in scopes}
    indexed = {}
    for row in paired:
        key = (row["contrast_id"], row["estimand"], row["gamma_points"])
        if key not in expected_pairs or key in indexed:
            raise ValueError("Unexpected/duplicate linear contrast")
        coefficients = row["coefficients"]
        if isinstance(coefficients, str): coefficients = json.loads(coefficients)
        if coefficients != CONTRASTS[key[0]]:
            raise ValueError("Predeclared linear contrast coefficients changed")
        if (int(row["training_seed"]) != training_seed or int(row["seed"]) != TEST_SEED
                or int(row["num_samples"]) != 1600 or int(row["num_batches"]) != 200
                or int(row["batch_size"]) != 8 or finite(row, "downlink_snr_db") != 25
                or str(row["training_seeds_pooled"]).lower() not in ("false", "0")
                or str(row["batch_ci_estimable"]).lower() not in ("true", "1")):
            raise ValueError("Invalid sample/batch units or training-seed pooling")
        for prefix in ("delta_", "batch_delta_"): _uncertainty(row, prefix)
        if not math.isclose(finite(row, "delta_mean"), finite(row, "batch_delta_mean"), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Equal-size batch means must retain the same signed estimate")
        indexed[key] = row
    if set(indexed) != expected_pairs:
        raise ValueError("Missing contrast: negative and interaction estimates must be retained")
    comparisons = {name: dict(per_gamma=[indexed[name, "per_gamma", str(g)] for g in GAMMAS],
        low_gamma=indexed[name, "low_gamma_mean", "0,5,10"],
        all_gamma=indexed[name, "all_gamma_mean", "0,5,10,15,20,25"]) for name in CONTRASTS}
    primary = comparisons["D_minus_E"]
    low, high = primary["low_gamma"], primary["per_gamma"][-1]
    return dict(training_seed=training_seed, comparisons=comparisons, all_rates=summary,
        engineering_criterion_D_minus_E=(finite(low, "delta_mean") >= .5 and
            finite(low, "batch_delta_ci95_low") > 0 and finite(high, "batch_delta_ci95_low") >= -.1),
        engineering_rule="D-E: low gamma mean>=0.5; low batch CI lower>0; gamma25 batch CI lower>=-0.1",
        training_seeds_pooled=False,
        physical_scope="Equal per-frame UE energy64, occupied RE256, shared absolute N0; gamma is not historical received FB SNR",
        interaction_scope="D-C-B+A formed directly for each common complete multi-user channel",
        layout_scope="Grouping changes codeword length, coordinate power and separation; not an isolated demixer ablation",
        budget_scope="All five arms randomly initialized, 20k updates; convergence is not established")


def result_paragraphs(interpretations, training_records):
    english, chinese = [], []
    for seed in SEEDS:
        item, training = interpretations[str(seed)], training_records[str(seed)]
        selected = ", ".join(f"{arm}: best step {training[arm]['best_step']}" for arm in METHODS)
        english.append(f"Training seed {seed}: all five randomly initialized arms completed 20,000 updates; validation selections were {selected}.")
        chinese.append(f"训练seed {seed}：五个模型均从随机初始化完成20,000次更新；固定验证集选点为 {selected}。")
        for name in CONTRASTS:
            comparison = item["comparisons"][name]
            low, high = comparison["low_gamma"], comparison["per_gamma"][-1]
            a, b, c = [finite(low, k) for k in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high")]
            d, e, f = [finite(high, k) for k in ("delta_mean", "batch_delta_ci95_low", "batch_delta_ci95_high")]
            english.append(f"Seed {seed}, {CONTRAST_LABELS[name]}: the within-channel gamma=0/5/10-dB mean contrast is {a:+.4f} bps/Hz (batch-paired 95% CI [{b:+.4f}, {c:+.4f}]); gamma=25 dB gives {d:+.4f} [{e:+.4f}, {f:+.4f}].")
            chinese.append(f"seed {seed}，{CONTRAST_LABELS[name]}：同一信道内gamma=0/5/10 dB平均差为{a:+.4f} bps/Hz，batch配对95%区间[{b:+.4f}, {c:+.4f}]；gamma=25 dB差值为{d:+.4f}，区间[{e:+.4f}, {f:+.4f}]。")
        passed = item["engineering_criterion_D_minus_E"]
        english.append(f"Seed {seed}: the predeclared D-minus-E engineering rule is {'met' if passed else 'not met'}; it is not a multi-training-seed significance claim.")
        chinese.append(f"seed {seed}：D减E的预先固定工程标准{'满足' if passed else '未满足'}；这不是跨训练seed显著性结论。")
    english.append("This is a new physical protocol: energy64 per UE per frame, 256 occupied feedback REs, and identical absolute N0 across methods. Gamma=10log10(64/(256N0)) is a transmit-block reference, not the historical received feedback SNR; no old curves are mixed in. E is Ordered-UE Joint-FDMA, not historical Swin. Grouping changes active codeword length, coordinate energy and multi-user separation jointly. All six gamma points, both seeds, all seven contrasts and negative outcomes are retained. Sample- and batch-paired intervals describe evaluation sampling for fixed checkpoints, not training-seed uncertainty. Active latent errors are offline within-representation diagnostics, not a common CSI ranking. Parameter counts include stored inactive legacy branches and do not measure latency; evaluation peak memory is not training peak memory. Twenty thousand updates do not establish convergence. No manuscript changes are made.")
    chinese.append("本轮采用独立物理协议：每帧每UE能量64、总占用反馈RE为256，各模型绝对N0相同。gamma=10log10(64/(256N0))是发射块参考量，不是历史接收feedback SNR，不能拼接旧曲线。E为Ordered-UE Joint-FDMA，不是旧Swin。分组同时改变活动码字长度、每坐标能量和用户分离，不是纯解混消融。保留六个gamma点、两个训练seed、七项对比及负增益。逐样本和batch配对区间只描述固定checkpoint的测试采样，不代表训练seed不确定性。活动latent误差仅供各表示内部诊断，不能当作统一CSI排名；参数统计包含存储但未执行的旧FDMA分支，不代表时延；评估峰值显存不等于训练峰值显存。20k更新不证明已收敛，不自动修改论文。")
    return english, chinese


def verify_shared_batch_inputs(manifest, reference=None):
    records = manifest.get("shared_batch_inputs")
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError("Actual shared-channel/noise hashes for all200 batches required")
    expected = {"batch_index", "sample_index_start", "sample_index_stop", "H_dl_sha256", "H_ul_sha256", "standard_noise_sha256"}
    for index, row in enumerate(records):
        if (not isinstance(row, dict) or set(row) != expected or row["batch_index"] != index
                or row["sample_index_start"] != 8 * index or row["sample_index_stop"] != 8 * (index + 1)):
            raise ValueError("Incomplete/duplicate/reordered actual input batch identity")
        for name in ("H_dl_sha256", "H_ul_sha256", "standard_noise_sha256"):
            if not isinstance(row[name], str) or not re.fullmatch(r"[0-9a-f]{64}", row[name]):
                raise ValueError("Invalid actual channel/noise digest")
    if reference is not None and records != reference:
        raise ValueError("The two training seeds were not evaluated on identical actual channels and noise")
    return records


def organize_results(folder, evaluations, training_records):
    from evaluate_grouped_codec import protocol, artifact_hashes
    if set(evaluations) != {str(s) for s in SEEDS} or set(training_records) != {str(s) for s in SEEDS}:
        raise ValueError("Both independently reported training seeds are required")
    interpretations, summaries, input_reference = {}, {}, None
    for seed in SEEDS:
        source = Path(evaluations[str(seed)])
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status") != "completed" or manifest.get("results_verified") is not True
                or manifest.get("freeze_checks", {}).get("passed") is not True
                or manifest.get("shared_channel_noise_inputs_unchanged") is not True
                or manifest.get("protocol") != protocol(seed)
                or manifest.get("output_sha256") != artifact_hashes(source)):
            raise ValueError("Grouped evaluation protocol/freeze/input/artifact evidence not verified")
        input_reference = verify_shared_batch_inputs(manifest, input_reference)
        for arm in METHODS:
            trained, loaded = training_records[str(seed)][arm], manifest["methods"][arm]
            for name in ("best_step", "best_sha256", "latest_sha256"):
                saved = "selected_best_step" if name == "best_step" else name
                if trained[name] != loaded[saved]:
                    raise ValueError("Selected training/evaluation checkpoint identity differs")
        summaries[seed] = rows(source / f"summary_{TAG}.csv")
        interpretations[str(seed)] = interpret(summaries[seed], rows(source / f"paired_deltas_{TAG}.csv"), seed)
    output = Path(folder) / ("export_" + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    for seed in SEEDS:
        source = Path(evaluations[str(seed)])
        for prefix in ("summary", "paired_deltas", "parameters"):
            shutil.copyfile(source / f"{prefix}_{TAG}.csv", output / f"seed{seed}_{prefix}_{TAG}.csv")
        shutil.copyfile(source / "manifest.json", output / f"seed{seed}_evaluation_manifest.json")
    save_json(output / "comparison_summary.json", interpretations)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    xlabel = r"$\gamma=10\log_{10}(64/(256N_0))$ (dB)"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for axis, seed in zip(axes, SEEDS):
        for method in METHODS:
            selected = sorted((r for r in summaries[seed] if r["method_key"] == method), key=lambda r: finite(r, "gamma_db"))
            axis.errorbar(GAMMAS, [finite(r, "sum_rate_mean") for r in selected],
                          yerr=[1.96 * finite(r, "sum_rate_se") for r in selected],
                          capsize=2, marker="o", markersize=3, linewidth=1.1, label=LABELS[method])
        axis.set(xlabel=xlabel, title=f"Training seed {seed}", xticks=GAMMAS)
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Sum rate (bps/Hz); sample 95% CI")
    axes[-1].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"): fig.savefig(output / ("grouped_codec_all_points." + suffix), dpi=160)
    plt.close(fig)
    groups = (("B_minus_A", "D_minus_C"), ("C_minus_A", "D_minus_B", "interaction_D_minus_C_minus_B_plus_A"),
              ("D_minus_E", "B_minus_E"))
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    for col, seed in enumerate(SEEDS):
        for row, contrast_names in enumerate(groups):
            axis = axes[row, col]
            for name in contrast_names:
                selected = interpretations[str(seed)]["comparisons"][name]["per_gamma"]
                axis.errorbar(GAMMAS, [finite(r, "delta_mean") for r in selected],
                              yerr=[1.96 * finite(r, "batch_delta_se") for r in selected],
                              capsize=3, marker="o", markersize=3, linewidth=1.1, label=CONTRAST_LABELS[name])
            axis.axhline(0, color=".5", linestyle="--", linewidth=.8)
            axis.set(xlabel=xlabel, title=f"Training seed {seed}", xticks=GAMMAS,
                     ylabel="Rate contrast; batch 95% CI (bps/Hz)")
            axis.grid(alpha=.18); axis.legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"): fig.savefig(output / ("grouped_codec_all_contrasts." + suffix), dpi=160)
    plt.close(fig)
    english, chinese = result_paragraphs(interpretations, training_records)
    for name, paragraphs in (("results_draft_en.md", english), ("results_draft_zh.md", chinese)):
        with (output / name).open("x", encoding="utf-8") as stream: stream.write("\n\n".join(paragraphs) + "\n")
    files = sorted(path for path in output.iterdir() if path.is_file())
    return dict(export_directory=str(output), per_seed=interpretations, training_seeds_pooled=False,
                files={str(path): sha256(path) for path in files}, _evidence=files)
