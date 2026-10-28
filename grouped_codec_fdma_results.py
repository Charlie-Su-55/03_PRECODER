"""CSV/log-only six-arm Legacy+FDMA report; never loads weights or runs models.

Plotting is lazy. make_plots=False supports synthetic tests without matplotlib.
Original evaluation CSVs are copied byte-for-byte; signed estimates are retained.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import uuid
import zipfile

from as_receiver_batch_support import save_json, sha256
from as_receiver_results import rows, finite
from grouped_codec_results import (
    SEEDS, GAMMAS, METHODS as OLD_METHODS, LABELS as OLD_LABELS,
    CONTRASTS as OLD_CONTRASTS, CONTRAST_LABELS as OLD_CONTRAST_LABELS,
    _uncertainty, verify_shared_batch_inputs, interpret as interpret_old)

TAG = "k4_grouped_codec_fdma_v1"
DATASETS = {"old_completion": 20261210, "new_replication": 20261220}
METHODS = (*OLD_METHODS, "legacy_fdma")
LABELS = dict(OLD_LABELS, legacy_fdma="F: Legacy-UE Joint-FDMA")
CONTRASTS = dict(OLD_CONTRASTS, **{
    "C_minus_F": {"legacy_group2": 1, "legacy_fdma": -1},
    "A_minus_F": {"legacy_all4": 1, "legacy_fdma": -1},
    "F_minus_E": {"legacy_fdma": 1, "ordered_fdma": -1},
    "C_minus_E": {"legacy_group2": 1, "ordered_fdma": -1},
    "interaction_C_minus_F_minus_D_plus_E": {
        "legacy_group2": 1, "legacy_fdma": -1, "ordered_group2": -1, "ordered_fdma": 1},
})
CONTRAST_LABELS = dict(OLD_CONTRAST_LABELS, **{
    "C_minus_F": "C - F: grouping, legacy vs Joint-FDMA",
    "A_minus_F": "A - F: all-4, legacy vs Joint-FDMA",
    "F_minus_E": "F - E: legacy vs ordered Joint-FDMA",
    "C_minus_E": "C - E: legacy grouped vs ordered Joint-FDMA",
    "interaction_C_minus_F_minus_D_plus_E": "(C - F) - (D - E): interaction",
})
RULE = "low gamma mean >= 0.5; low batch CI lower > 0; gamma25 batch CI lower >= -0.1"
VALIDATION_STEPS = tuple(range(500, 20001, 500))
SMALL_LIMIT = 32 * 1024 * 1024


def _require(row, names):
    if not isinstance(row, dict) or any(name not in row for name in names):
        raise ValueError("Missing required report fields: " + ",".join(names))


def _integer(row, name):
    value = finite(row, name)
    if value != int(value):
        raise ValueError("Nonintegral " + name)
    return int(value)


def _units(row, training_seed, dataset, *, summary):
    _require(row, ("dataset", "training_seed", "seed", "num_samples", "num_batches",
                   "batch_size", "downlink_snr_db"))
    if (row["dataset"] != dataset or _integer(row, "training_seed") != training_seed
            or _integer(row, "seed") != DATASETS[dataset]
            or _integer(row, "num_samples") != 1600 or _integer(row, "num_batches") != 200
            or _integer(row, "batch_size") != 8 or finite(row, "downlink_snr_db") != 25):
        raise ValueError("Wrong dataset, seed, channel/batch units or downlink SNR")
    if summary and str(row.get("preflight", "")).lower() not in ("false", "0"):
        raise ValueError("Preflight cannot be reported as the completed test")
    if any(key in row for key in ("feedback_snr_db", "allocation_df", "allocation_da")):
        raise ValueError("Historical received-SNR/allocation fields cannot be relabeled as gamma")


def _criterion(comparison):
    low, high = comparison["low_gamma"], comparison["per_gamma"][-1]
    return (finite(low, "delta_mean") >= .5 and
            finite(low, "batch_delta_ci95_low") > 0 and
            finite(high, "batch_delta_ci95_low") >= -.1)


def interpret(summary, paired, training_seed, dataset):
    """Validate complete signed tables without modifying or re-estimating values."""
    if type(training_seed) is not int or training_seed not in SEEDS or dataset not in DATASETS:
        raise ValueError("Report each prescribed training seed and dataset separately")
    expected = {(key, gamma) for key in METHODS for gamma in GAMMAS}
    actual = []
    for row in summary:
        _require(row, ("method_key", "gamma_db", "noise_variance_N0",
                       "sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high"))
        _units(row, training_seed, dataset, summary=True)
        gamma = finite(row, "gamma_db")
        actual.append((row["method_key"], gamma))
        if not math.isclose(finite(row, "noise_variance_N0"), .25 * 10 ** (-gamma / 10), rel_tol=1e-10):
            raise ValueError("Absolute-noise protocol changed")
        _uncertainty(row, "sum_rate_")
    if len(actual) != 36 or set(actual) != expected:
        raise ValueError("All six methods and six gamma points are required exactly once")
    scopes = [("per_gamma", str(g)) for g in GAMMAS] + [
        ("low_gamma_mean", "0,5,10"), ("all_gamma_mean", "0,5,10,15,20,25")]
    expected_pairs = {(name, scope, points) for name in CONTRASTS for scope, points in scopes}
    indexed = {}
    for row in paired:
        _require(row, ("contrast_id", "estimand", "gamma_points", "coefficients",
                       "training_seeds_pooled", "datasets_pooled", "batch_ci_estimable",
                       "sample_ci_unit", "batch_ci_unit", "delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high",
                       "batch_delta_mean", "batch_delta_se", "batch_delta_ci95_low", "batch_delta_ci95_high"))
        _units(row, training_seed, dataset, summary=False)
        key = (row["contrast_id"], row["estimand"], row["gamma_points"])
        if key not in expected_pairs or key in indexed:
            raise ValueError("Unexpected or duplicate contrast/estimand")
        coefficients = row["coefficients"]
        if isinstance(coefficients, str):
            coefficients = json.loads(coefficients)
        if coefficients != CONTRASTS[key[0]]:
            raise ValueError("Signed, predeclared contrast coefficients changed")
        if (str(row["training_seeds_pooled"]).lower() not in ("false", "0") or
                str(row["datasets_pooled"]).lower() not in ("false", "0") or
                str(row["batch_ci_estimable"]).lower() not in ("true", "1") or
                row["sample_ci_unit"] != "complete_multi_user_channel" or
                row["batch_ci_unit"] != "sample_index//8"):
            raise ValueError("No seed/dataset pooling; complete-channel and eight-channel batch CI units required")
        for prefix in ("delta_", "batch_delta_"):
            _uncertainty(row, prefix)
        if not math.isclose(finite(row, "delta_mean"), finite(row, "batch_delta_mean"),
                            rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Equal-size batch and channel means must retain the same signed estimate")
        indexed[key] = row
    if set(indexed) != expected_pairs:
        raise ValueError("All twelve contrasts and eight estimands are required, including negative values")
    comparisons = {name: {
        "per_gamma": [indexed[name, "per_gamma", str(g)] for g in GAMMAS],
        "low_gamma": indexed[name, "low_gamma_mean", "0,5,10"],
        "all_gamma": indexed[name, "all_gamma_mean", "0,5,10,15,20,25"],
    } for name in CONTRASTS}
    return dict(dataset=dataset, test_seed=DATASETS[dataset], training_seed=training_seed,
        comparisons=comparisons, all_rates=summary,
        engineering_criterion_C_minus_F=_criterion(comparisons["C_minus_F"]),
        engineering_criterion_D_minus_E=_criterion(comparisons["D_minus_E"]),
        engineering_rule=RULE, primary_contrast="C_minus_F",
        primary_role="primary_confirmatory" if dataset == "new_replication" else "paired_completion",
        old_D_minus_E_role="retained historical engineering rule; never replaced by C-F",
        training_seeds_pooled=False, datasets_pooled=False,
        interval_scope="Fixed-checkpoint evaluation uncertainty; not uncertainty across training seeds",
        interaction_scope="Each interaction is a direct within-channel linear contrast before gamma averaging",
        physical_scope="Energy64 per UE, occupied RE256, common absolute N0; gamma is not received feedback SNR",
        layout_scope="Grouping changes codeword length, coordinate energy and separation jointly",
        budget_scope="Fixed 20k training budget; neither trends nor best-step selection establish convergence")


def _safe_path(path, *, file=False):
    path = Path(path).absolute()
    for current in (path, *path.parents):
        if (current.is_symlink() or
                (hasattr(current, "is_junction") and current.is_junction()) or
                (current.exists() and getattr(current.lstat(), "st_file_attributes", 0) & 0x400)):
            raise ValueError("Links/reparse points are not report inputs or destinations")
    if file and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_json(path, expected=None):
    path = _safe_path(path, file=True)
    before = sha256(path)
    if expected is not None and before != expected:
        raise ValueError("Input SHA256 mismatch: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha256(path) != before:
        raise ValueError("Input changed while reading: " + str(path))
    return value, before


def training_curves(training_records):
    """Read twelve exact metrics paths; report a trend, not a convergence claim."""
    if set(training_records) != {str(seed) for seed in SEEDS}:
        raise ValueError("Both training seeds are required")
    result, source_files = {}, {}
    for seed in SEEDS:
        records = training_records[str(seed)]
        if set(records) != set(METHODS):
            raise ValueError("All twelve completed training records are required")
        result[str(seed)] = {}
        for method in METHODS:
            record = records[method]
            _require(record, ("completed_updates", "best_step", "metrics_path", "metrics_sha256"))
            if _integer(record, "completed_updates") != 20000:
                raise ValueError("Only completed 20k budgets can be reported")
            path = _safe_path(record["metrics_path"], file=True)
            before = sha256(path)
            if before != record["metrics_sha256"]:
                raise ValueError("Training metrics hash mismatch")
            points = []
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("event") != "validation":
                        if item.get("event") != "train_gamma":
                            raise ValueError(f"Unknown training metrics event at {path}:{line_number}")
                        continue
                    _require(item, ("step", "selection_metric_name", "selection_metric",
                                    "table", "gamma_db", "downlink_snr_db"))
                    if (item["selection_metric_name"] != "mean_all" or item["gamma_db"] != list(GAMMAS)
                            or finite(item, "downlink_snr_db") != 25):
                        raise ValueError("Validation selection or fixed six-gamma protocol changed")
                    table = item["table"]
                    expected = {f"layout0_gamma{gamma}" for gamma in GAMMAS}
                    if not isinstance(table, dict) or set(table) != expected:
                        raise ValueError("Validation table must retain all six gamma rates")
                    rates = {str(gamma): finite(table, f"layout0_gamma{gamma}") for gamma in GAMMAS}
                    metric = finite(item, "selection_metric")
                    if not math.isclose(metric, sum(rates.values()) / 6, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError("Validation mean_all differs from its six recorded rates")
                    points.append(dict(step=_integer(item, "step"), mean_all=metric, rates=rates))
            if sha256(path) != before:
                raise ValueError("Training metrics changed during reading")
            if tuple(point["step"] for point in points) != VALIDATION_STEPS:
                raise ValueError("Require exactly forty ordered validation points, steps 500..20000")
            best = max(points, key=lambda point: point["mean_all"])
            if _integer(record, "best_step") != best["step"]:
                raise ValueError("Selected best step differs from first strict mean_all maximum")
            tail = [point for point in points if point["step"] >= 15000]
            xs, ys = [p["step"] / 1000 for p in tail], [p["mean_all"] for p in tail]
            xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((x-xmean)*(y-ymean) for x,y in zip(xs,ys)) / sum((x-xmean)**2 for x in xs)
            result[str(seed)][method] = dict(
                completed_updates=20000, best_step=best["step"], best_mean_all=best["mean_all"],
                metrics_path=str(path), metrics_sha256=before, points=points,
                last_five_validation_points=points[-5:],
                last_5000_updates=dict(start_step=15000, end_step=20000, num_points=11,
                    boundary_policy="inclusive 15000 and 20000; eleven 500-step validation points",
                    mean_all_start=ys[0], mean_all_end=ys[-1], end_minus_start=ys[-1]-ys[0],
                    ols_slope_per_1000_updates=slope, minimum=min(ys), maximum=max(ys)),
                interpretation="Descriptive fixed-validation trend only; convergence is not established")
            source_files[str(path)] = before
    if len(source_files) != 12:
        raise ValueError("Twelve distinct exact metrics paths are required")
    return dict(per_seed=result, source_files=source_files,
                training_seeds_pooled=False, convergence_established=False)


def _equal_numeric_rows(old, new, keys):
    for key in keys:
        if finite(old, key) != finite(new, key):
            raise ValueError("Frozen old result was numerically changed: " + key)


def verify_old_completion(old_evidence, interpretations):
    """Compare original A-E summary and seven signed contrasts, never rewrite them."""
    _require(old_evidence, ("summary_path", "summary_sha256", "evaluations", "training", "frozen_files"))
    _, summary_digest = _read_json(old_evidence["summary_path"], old_evidence["summary_sha256"])
    historical, source_files = {}, {str(_safe_path(old_evidence["summary_path"])): summary_digest}
    for seed in SEEDS:
        evidence = old_evidence["evaluations"][str(seed)]
        summary_path = _safe_path(evidence["summary_path"], file=True)
        paired_path = _safe_path(Path(evidence["directory"]) / "paired_deltas_k4_grouped_codec_v1.csv", file=True)
        for path in (summary_path, paired_path):
            actual = sha256(path)
            expected = old_evidence["frozen_files"].get(str(path))
            if expected is None or expected != actual:
                raise ValueError("Frozen old small-table hash is missing or changed")
            source_files[str(path)] = actual
        old_summary, old_paired = rows(summary_path), rows(paired_path)
        prior = interpret_old(old_summary, old_paired, seed)
        current = interpretations["old_completion"][str(seed)]
        new_summary = {(r["method_key"], finite(r, "gamma_db")): r for r in current["all_rates"]}
        for row in old_summary:
            _equal_numeric_rows(row, new_summary[row["method_key"], finite(row, "gamma_db")],
                ("sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high"))
        for name in OLD_CONTRASTS:
            before, after = prior["comparisons"][name], current["comparisons"][name]
            for first, second in zip(before["per_gamma"]+[before["low_gamma"],before["all_gamma"]],
                                     after["per_gamma"]+[after["low_gamma"],after["all_gamma"]]):
                _equal_numeric_rows(first, second,
                    tuple(prefix+part for prefix in ("delta_","batch_delta_")
                          for part in ("mean","se","ci95_low","ci95_high")))
        if prior["engineering_criterion_D_minus_E"] != current["engineering_criterion_D_minus_E"]:
            raise ValueError("Historical D-E criterion was changed")
        historical[str(seed)] = dict(
            engineering_criterion_D_minus_E=prior["engineering_criterion_D_minus_E"],
            old_seven_contrasts_unchanged=True, old_A_to_E_rates_unchanged=True,
            source_summary=str(summary_path), source_paired=str(paired_path))
    for path, digest in source_files.items():
        if sha256(path) != digest:
            raise ValueError("Frozen historical table changed during report verification")
    return dict(per_seed=historical, source_files=source_files, old_files_modified=False)


def _copy_small(source, target, expected=None):
    source = _safe_path(source, file=True)
    if source.suffix.lower() not in (".csv", ".json", ".md", ".pdf", ".png") or source.stat().st_size > SMALL_LIMIT:
        raise ValueError("Only complete small CSV/JSON/document/figure artifacts may enter the bundle")
    digest = sha256(source)
    if expected is not None and digest != expected:
        raise ValueError("Copy source no longer matches verified evidence")
    with target.open("xb") as output, source.open("rb") as input_stream:
        shutil.copyfileobj(input_stream, output)
    if sha256(target) != digest or sha256(source) != digest:
        raise ValueError("Input changed while copying")
    return digest


def _write_csv(path, records):
    if not records:
        raise ValueError("Empty derived table")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plots(output, interpretations, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip(METHODS, ("#2864a5","#dc7b25","#38945a","#ad4794","#7159a6","#665b4c")))
    def finish(fig, name):
        for suffix in ("pdf", "png"):
            fig.savefig(output / (name+"."+suffix), dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for axis, seed in zip(axes, SEEDS):
        for method in METHODS:
            points = curves["per_seed"][str(seed)][method]["points"]
            axis.plot([p["step"] for p in points], [p["mean_all"] for p in points],
                      color=colors[method], linewidth=1.1, label=LABELS[method])
        axis.set(xlabel="Training updates", title=f"Training seed {seed}")
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Shared fixed validation mean_all (bps/Hz)")
    axes[-1].legend(frameon=False, fontsize=7)
    finish(fig, "validation_mean_all_six_methods")
    groups = (("B_minus_A","D_minus_C","F_minus_E"),
              ("C_minus_A","D_minus_B","interaction_D_minus_C_minus_B_plus_A"),
              ("D_minus_E","B_minus_E","C_minus_E"),
              ("C_minus_F","A_minus_F","interaction_C_minus_F_minus_D_plus_E"))
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=True, layout="constrained")
        for axis, seed in zip(axes, SEEDS):
            for method in METHODS:
                selected = sorted((r for r in interpretations[dataset][str(seed)]["all_rates"]
                                   if r["method_key"] == method), key=lambda r: finite(r,"gamma_db"))
                axis.errorbar(GAMMAS, [finite(r,"sum_rate_mean") for r in selected],
                    yerr=[1.96*finite(r,"sum_rate_se") for r in selected],
                    color=colors[method], capsize=2, marker="o", markersize=3, linewidth=1.1,
                    label=LABELS[method])
            axis.set(xlabel="Transmit-reference gamma (dB)", xticks=GAMMAS,
                     title=f"{dataset}; training seed {seed}")
            axis.grid(alpha=.18)
        axes[0].set_ylabel("Sum rate (bps/Hz); sample 95% CI")
        axes[-1].legend(frameon=False, fontsize=7)
        finish(fig, dataset+"_six_method_all_gamma")
        fig, axes = plt.subplots(4,2,figsize=(11,12),layout="constrained")
        for column, seed in enumerate(SEEDS):
            for row_index, names in enumerate(groups):
                axis = axes[row_index,column]
                for name in names:
                    selected = interpretations[dataset][str(seed)]["comparisons"][name]["per_gamma"]
                    axis.errorbar(GAMMAS,[finite(r,"delta_mean") for r in selected],
                        yerr=[1.96*finite(r,"batch_delta_se") for r in selected],
                        capsize=2,marker="o",markersize=3,linewidth=1,label=CONTRAST_LABELS[name])
                axis.axhline(0,color=".5",linestyle="--",linewidth=.8)
                axis.set(xlabel="Transmit-reference gamma (dB)",xticks=GAMMAS,
                         title=f"Training seed {seed}",ylabel="Rate contrast; batch 95% CI")
                axis.grid(alpha=.18)
                axis.legend(frameon=False,fontsize=6)
        finish(fig,dataset+"_all_twelve_paired_contrasts")


def result_paragraphs(interpretations, curves, historical):
    english = ["Internal six-arm Legacy+FDMA report. The new independent test (seed 20261220) is the primary C-F comparison; the old-test F completion (seed 20261210) is secondary. No training seeds or test datasets are pooled. In old_completion only F is loaded and evaluated now; A-E records are reused after actual input-hash matching, not newly loaded."]
    chinese = ["六模型 Legacy+FDMA 内部结果草稿。新独立测试集（seed 20261220）以 C−F 为主要比较；旧测试集补全 F（seed 20261210）为次要报告。不合并训练 seed 或测试数据集。old_completion 本次仅载入并评估 F；A–E 是逐批输入哈希匹配后复用的旧记录，不声称本次重新载入。"]
    for seed in SEEDS:
        english.append(f"Training seed {seed}: A-E are the unchanged completed historical models; only F is newly trained. All six budgets are 20,000 updates.")
        chinese.append(f"训练 seed {seed}：A–E 为保持不变的已完成历史模型，仅 F 为新增训练；六模型预算均为 20,000 次更新。")
        for method in METHODS:
            item = curves["per_seed"][str(seed)][method]
            tail = item["last_5000_updates"]
            last = ", ".join(f"{p['step']}:{p['mean_all']:.6f}" for p in item["last_five_validation_points"])
            english.append(f"{LABELS[method]}, seed {seed}: best step {item['best_step']}; last five validation mean_all points {last}. Inclusive steps 15000-20000 (11 points): endpoint change {tail['end_minus_start']:+.6f} bps/Hz, OLS slope {tail['ols_slope_per_1000_updates']:+.6f} bps/Hz per 1000 updates. This describes a trend, not convergence.")
            chinese.append(f"{LABELS[method]}，seed {seed}：最佳 step {item['best_step']}；最后五个验证 mean_all 点为 {last}。15000–20000（含端点、11 点）：末减初 {tail['end_minus_start']:+.6f} bps/Hz，OLS 斜率 {tail['ols_slope_per_1000_updates']:+.6f} bps/Hz/1000 更新。仅描述趋势，不证明收敛。")
    for dataset in ("new_replication", "old_completion"):
        for seed in SEEDS:
            item = interpretations[dataset][str(seed)]
            for name in CONTRASTS:
                for scope in ("low_gamma", "all_gamma"):
                    value = item["comparisons"][name][scope]
                    mean = finite(value,"delta_mean")
                    sample_ci = (finite(value,"delta_ci95_low"),finite(value,"delta_ci95_high"))
                    batch_ci = (finite(value,"batch_delta_ci95_low"),finite(value,"batch_delta_ci95_high"))
                    english.append(f"{dataset}, training seed {seed}, {CONTRAST_LABELS[name]}, {scope}: {mean:+.6f} bps/Hz; sample-paired 95% CI [{sample_ci[0]:+.6f}, {sample_ci[1]:+.6f}], batch-paired 95% CI [{batch_ci[0]:+.6f}, {batch_ci[1]:+.6f}].")
                    chinese.append(f"{dataset}，训练 seed {seed}，{CONTRAST_LABELS[name]}，{scope}：{mean:+.6f} bps/Hz；逐样本配对 95% 区间 [{sample_ci[0]:+.6f}, {sample_ci[1]:+.6f}]，batch 配对 95% 区间 [{batch_ci[0]:+.6f}, {batch_ci[1]:+.6f}]。")
            cf, de = item["engineering_criterion_C_minus_F"], item["engineering_criterion_D_minus_E"]
            english.append(f"{dataset}, seed {seed}: C-F rule {'met' if cf else 'not met'} ({item['primary_role']}); retained D-E rule {'met' if de else 'not met'}. Both rules use {RULE}. These are per-seed engineering criteria, not significance across training seeds.")
            chinese.append(f"{dataset}，seed {seed}：C−F 判据{'满足' if cf else '未满足'}（{item['primary_role']}）；保留的 D−E 判据{'满足' if de else '未满足'}。两者均要求低三点均值≥0.5、低三点 batch 区间下界>0、gamma=25 的 batch 区间下界≥−0.1。这是逐训练 seed 工程判据，不是跨训练 seed 显著性。")
    old_outcomes = {seed: historical["per_seed"][seed]["engineering_criterion_D_minus_E"] for seed in historical["per_seed"]}
    english.append(f"The original A-E rates and all seven original signed contrasts were verified unchanged. Original D-E rule outcomes are {old_outcomes}; no prior negative outcome is replaced or suppressed. All twelve contrasts, six gamma points, both training seeds and both datasets remain in the unrounded CSV tables.")
    chinese.append(f"已核对原 A–E rate 和全部七项原有有符号对比保持不变。原 D−E 判据结果为 {old_outcomes}；不替换或隐去原负结果。未舍入 CSV 保留全部十二项对比、六个 gamma 点、两个训练 seed 和两个测试数据集。")
    english.append("All confidence intervals quantify evaluation sampling for fixed checkpoints, not training-seed uncertainty. Low/all-gamma contrasts average within each complete multi-user channel before estimating uncertainty; interactions are direct linear contrasts, not differences of independent intervals. The physical protocol fixes per-UE energy64, occupied RE256 and N0=0.25*10^(-gamma/10). Grouping simultaneously changes codeword length, coordinate energy and separation; C-F is not a pure demixer ablation. E and F are Joint-FDMA controls, not historical Swin. Latent errors are representation-specific offline diagnostics. Stored parameter counts include inactive legacy FDMA modules and do not measure latency. Fixed 20k budgets do not establish convergence. No extension, retraining of A-E or manuscript editing is performed. Weights and raw .csv.gz remain on the server; this bundle contains only small tables, technical JSON, figures and drafts.")
    chinese.append("所有置信区间仅描述固定 checkpoint 的测试采样不确定性，不代表训练 seed 不确定性。低/全部 gamma 对比先在同一个完整多用户信道内平均再估计区间；交互项直接对每个样本作线性对比，不是相减独立区间。物理协议固定每 UE 能量64、总占用 RE256、N0=0.25×10^(−gamma/10)。分组同时改变码字长度、坐标能量和分离条件，C−F 不是纯解混消融。E、F 是 Joint-FDMA 对照而非历史 Swin。latent 误差仅为表示内离线诊断；存储参数含未活动的旧 FDMA 模块，不代表时延。固定20k预算不证明收敛。不延训、不重训A–E、不修改论文。权重和 raw .csv.gz 留在服务器；小包仅含表格、技术JSON、图和草稿。")
    return english, chinese


def organize_results(folder, evaluations, training_records, old_evidence, *, make_plots=True):
    """Verify immutable inputs, then create one new small report directory."""
    from evaluate_grouped_codec_fdma import protocol, artifact_hashes
    if set(evaluations) != set(DATASETS):
        raise ValueError("Both old completion and new replication datasets are required")
    curves = training_curves(training_records)
    interpretations, sources, manifests, dataset_inputs, parameter_tables = {}, {}, {}, {}, {}
    for dataset in DATASETS:
        if set(evaluations[dataset]) != {str(seed) for seed in SEEDS}:
            raise ValueError("Both independently reported training seeds are required per dataset")
        interpretations[dataset], input_reference = {}, None
        for seed in SEEDS:
            source = _safe_path(evaluations[dataset][str(seed)])
            manifest, manifest_sha = _read_json(source/"manifest.json")
            if (manifest.get("status") != "completed" or manifest.get("results_verified") is not True
                    or manifest.get("freeze_checks",{}).get("passed") is not True
                    or manifest.get("shared_channel_noise_inputs_unchanged") is not True
                    or manifest.get("samples_hashed_pairing") is not True
                    or manifest.get("protocol") != protocol(seed, dataset)
                    or manifest.get("output_sha256") != artifact_hashes(source)):
                raise ValueError("Evaluation protocol/freeze/input/artifact evidence not verified")
            input_reference = verify_shared_batch_inputs(manifest, input_reference)
            expected_loaded = {"legacy_fdma"} if dataset == "old_completion" else set(METHODS)
            if set(manifest["freeze_checks"].get("loaded_methods", ())) != expected_loaded:
                raise ValueError("Freeze evidence must cover precisely the models loaded in this evaluation")
            if dataset == "old_completion":
                old_manifest = old_evidence["evaluations"][str(seed)]["manifest"]
                if (manifest.get("frozen_old_rows_preserved") is not True or
                        manifest["shared_batch_inputs"] != old_manifest.get("shared_batch_inputs")):
                    raise ValueError("Old completion did not preserve actual frozen batch input identities")
            for method in METHODS:
                trained, loaded = training_records[str(seed)][method], manifest["methods"][method]
                if loaded.get("loaded_in_this_run") is not (method in expected_loaded):
                    raise ValueError("Historical reuse must not be described as a new model load")
                if method in OLD_METHODS:
                    old_training = old_evidence["training"][str(seed)][method]
                    for field in ("completed_updates", "best_step", "best_sha256", "latest_sha256",
                                  "metrics_path", "metrics_sha256"):
                        if trained[field] != old_training[field]:
                            raise ValueError("Historical training record changed")
                for name in ("best_step","best_sha256","latest_sha256"):
                    saved = "selected_best_step" if name == "best_step" else name
                    if trained[name] != loaded[saved]:
                        raise ValueError("Training/evaluation checkpoint identity differs")
            summary = rows(_safe_path(source/f"summary_{TAG}.csv",file=True))
            paired = rows(_safe_path(source/f"paired_deltas_{TAG}.csv",file=True))
            interpretations[dataset][str(seed)] = interpret(summary,paired,seed,dataset)
            parameters = rows(_safe_path(source/f"parameters_{TAG}.csv",file=True))
            if (len(parameters) != 6 or {r["method_key"] for r in parameters} != set(METHODS)
                    or any(_integer(r,"training_seed") != seed for r in parameters)):
                raise ValueError("Complete per-seed actual parameter counts are required")
            for row in parameters:
                if (_integer(row,"full_parameters") <= 0 or _integer(row,"ue_parameters") <= 0
                        or _integer(row,"bs_parameters") <= 0 or _integer(row,"full_parameters") !=
                        _integer(row,"ue_parameters") + _integer(row,"bs_parameters")):
                    raise ValueError("Inconsistent actual model parameter counts")
            parameter_tables[dataset,seed] = {r["method_key"]: r for r in parameters}
            sources[dataset,seed] = source
            manifests[dataset,seed] = (manifest,manifest_sha)
        dataset_inputs[dataset] = input_reference
    if dataset_inputs["old_completion"] == dataset_inputs["new_replication"]:
        raise ValueError("Independent replication unexpectedly reused all old actual input batches")
    historical = verify_old_completion(old_evidence, interpretations)
    parameter_fields = ("full_parameters", "ue_parameters", "bs_parameters")
    for seed in SEEDS:
        old_parameter_path = _safe_path(old_evidence["evaluations"][str(seed)]["parameters_path"], file=True)
        original_parameters = {r["method_key"]: r for r in rows(old_parameter_path)}
        if set(original_parameters) != set(OLD_METHODS):
            raise ValueError("Original actual parameter table must contain the five frozen methods")
        digest = sha256(old_parameter_path)
        if old_evidence["frozen_files"].get(str(old_parameter_path)) != digest:
            raise ValueError("Original actual parameter CSV hash changed")
        historical["source_files"][str(old_parameter_path)] = digest
        for method in METHODS:
            first, second = (parameter_tables[dataset,seed][method] for dataset in DATASETS)
            _equal_numeric_rows(first, second, parameter_fields)
            if method in OLD_METHODS:
                _equal_numeric_rows(original_parameters[method], first, parameter_fields)
    parent = _safe_path(folder)
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    output = parent/("export_"+uuid.uuid4().hex)
    output.mkdir(exist_ok=False)
    source_files = dict(curves["source_files"], **historical["source_files"])
    for (dataset,seed), source in sources.items():
        manifest, manifest_sha = manifests[dataset,seed]
        for prefix in ("summary","paired_deltas","parameters"):
            filename = f"{prefix}_{TAG}.csv"
            expected = manifest["output_sha256"].get(filename)
            if expected is None:
                raise ValueError("Small output artifact absent from manifest")
            digest = _copy_small(source/filename,output/f"{dataset}_seed{seed}_{filename}",expected)
            source_files[str(source/filename)] = digest
        _copy_small(source/"manifest.json",output/f"{dataset}_seed{seed}_evaluation_manifest.json",manifest_sha)
        source_files[str(source/"manifest.json")] = manifest_sha
    save_json(output/"comparison_summary.json",dict(per_dataset=interpretations,historical=historical,
        primary_dataset="new_replication",primary_contrast="C_minus_F",
        training_seeds_pooled=False,datasets_pooled=False))
    save_json(output/"training_trends.json",curves)
    save_json(output/"source_evidence.json",dict(source_files=source_files,
        old_batch_summary_path=old_evidence["summary_path"],
        old_batch_summary_sha256=old_evidence["summary_sha256"],
        frozen_historical_files=old_evidence["frozen_files"],
        training_records=training_records,plots_generated=bool(make_plots),
        historical_source_hashes_are_provenance_not_current_code_requirements=True))
    learning, trends = [], []
    for seed in SEEDS:
        for method in METHODS:
            item = curves["per_seed"][str(seed)][method]
            for point in item["points"]:
                learning.append(dict(training_seed=seed,method_key=method,step=point["step"],
                    mean_all=point["mean_all"],**{f"gamma{g}_rate":point["rates"][str(g)] for g in GAMMAS}))
            trends.append(dict(training_seed=seed,method_key=method,best_step=item["best_step"],
                best_mean_all=item["best_mean_all"],**item["last_5000_updates"],
                convergence_established=False))
    _write_csv(output/"validation_mean_all_all_points.csv",learning)
    _write_csv(output/"training_last5000_trends.csv",trends)
    if make_plots:
        _plots(output,interpretations,curves)
    english,chinese = result_paragraphs(interpretations,curves,historical)
    for filename, paragraphs in (("results_draft_en.md",english),("results_draft_zh.md",chinese)):
        with (output/filename).open("x",encoding="utf-8") as stream:
            stream.write("\n\n".join(paragraphs)+"\n")
    for path,digest in source_files.items():
        if sha256(_safe_path(path,file=True)) != digest:
            raise ValueError("A verified input changed during report generation")
    files = sorted(p for p in output.iterdir() if p.is_file())
    save_json(output/"bundle_manifest.json",dict(
        policy="Only small complete CSV/JSON/Markdown/PDF/PNG; no weights or raw compressed samples",
        files={p.name:{"size":p.stat().st_size,"sha256":sha256(p)} for p in files}))
    files.append(output/"bundle_manifest.json")
    archive = output/"small_results_bundle.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            if path.suffix.lower() not in (".csv",".json",".md",".pdf",".png") or path.stat().st_size > SMALL_LIMIT:
                raise ValueError("Forbidden or unexpectedly large report bundle member")
            bundle.write(path,arcname=path.name)
    files.append(archive)
    return dict(export_directory=str(output),per_dataset=interpretations,
        training_trends=curves,historical=historical,
        parameters={dataset:{str(seed):list(parameter_tables[dataset,seed].values()) for seed in SEEDS}
                    for dataset in DATASETS},primary_dataset="new_replication",
        primary_contrast="C_minus_F",training_seeds_pooled=False,datasets_pooled=False,
        files={str(path):sha256(path) for path in files},_evidence=files)
