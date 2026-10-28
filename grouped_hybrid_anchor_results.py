"""CSV/log-only grouped Hybrid Anchor results; no weights or model execution."""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

from as_receiver_batch_support import save_json,sha256
from as_receiver_results import rows,finite
from grouped_codec_results import _uncertainty
from grouped_codec_fdma_results import _safe_path,_read_json,_copy_small,_write_csv
from evaluate_grouped_hybrid_anchor import (
    ARMS,SEEDS,METHODS,GAMMAS,TEST_SEED,TAG,CONTRASTS,LAYOUTS,OPTIONAL_METRICS,
    _available,protocol,artifact_hashes,verify_shared_inputs,SOURCE_BEST_SHA256)

LABELS={"fdma_control":"FDMA +20k control","hybrid_f16_a96":"Hybrid Df16 / Da96 per group",
        "hybrid_f32_a64":"Hybrid Df32 / Da64 per group","frozen_F":"Frozen source F"}
CONTRAST_LABELS={"H16_minus_Fnew":"Hybrid16 - new FDMA","H32_minus_Fnew":"Hybrid32 - new FDMA",
                "H16_minus_H32":"Hybrid16 - Hybrid32","Fnew_minus_Ffrozen":"New FDMA - frozen F"}
PRIMARY=("H16_minus_Fnew","H32_minus_Fnew")
RULE="low gamma mean>=0.5; low batch95CI lower>0; gamma25 batch95CI lower>=-0.1"
VALIDATION_STEPS=tuple(range(0,20001,500))


def _integer(row,key):
    value=finite(row,key)
    if value!=int(value):raise ValueError("Nonintegral "+key)
    return int(value)


def _units(row,seed):
    if (_integer(row,"training_seed")!=seed or _integer(row,"seed")!=TEST_SEED
            or _integer(row,"num_samples")!=1600 or _integer(row,"num_batches")!=200
            or _integer(row,"batch_size")!=8 or finite(row,"downlink_snr_db")!=25):
        raise ValueError("Expected1600 complete channels,200 batches,test seed20261228,DL25")
    if any(name in row for name in ("feedback_snr_db","allocation_df","allocation_da")):
        raise ValueError("Grouped Hybrid is not a legacy allocation or received-feedback-SNR protocol")


def interpret(summary,paired,training_seed):
    if type(training_seed)is not int or training_seed not in SEEDS:
        raise ValueError("Two training seeds must be reported separately")
    expected={(m,g) for m in METHODS for g in GAMMAS}
    actual=[]
    for row in summary:
        _units(row,training_seed)
        method,gamma=row["method_key"],finite(row,"gamma_db")
        actual.append((method,gamma))
        if method not in METHODS or gamma not in GAMMAS:raise ValueError("Unexpected method/gamma")
        df,da=LAYOUTS[method]
        if (_integer(row,"Df_per_user")!=df or _integer(row,"Da_per_group")!=da
                or _integer(row,"occupied_feedback_RE")!=256 or
                _integer(row,"active_symbols_per_ue")!=df+da or 4*df+2*da!=256):
            raise ValueError("Grouped exclusive/shared budget changed")
        if (str(row.get("preflight","")).lower()not in ("false","0")
                or str(row.get("prior_frozen","")).lower()not in ("true","1")):
            raise ValueError("Formal evaluation requires frozen prior, not preflight")
        if not math.isclose(finite(row,"noise_variance_N0"),.25*10**(-gamma/10),rel_tol=1e-10):
            raise ValueError("Absolute N0 changed")
        _uncertainty(row,"sum_rate_")
        for metric in OPTIONAL_METRICS:
            cells=[row.get(metric+"_"+part)for part in ("mean","se","ci95_low","ci95_high")]
            if _available(method,metric):
                if any(value in ("",None)for value in cells):raise ValueError("Missing available diagnostic")
                _uncertainty(row,metric+"_")
            elif any(value not in ("",None)for value in cells):
                raise ValueError("Unavailable frozen-F or empty-support diagnostic must remain NA")
    if len(actual)!=24 or set(actual)!=expected:raise ValueError("All4 methods x6 gamma conditions required")
    scopes=[("per_gamma",str(g))for g in GAMMAS]+[("low_gamma_mean","0,5,10"),("all_gamma_mean","0,5,10,15,20,25")]
    expected_pairs={(name,scope,points)for name in CONTRASTS for scope,points in scopes}
    indexed={}
    for row in paired:
        _units(row,training_seed)
        key=(row["contrast_id"],row["estimand"],row["gamma_points"])
        if key not in expected_pairs or key in indexed:raise ValueError("Unexpected/duplicate signed contrast")
        coefficients=row["coefficients"]
        if isinstance(coefficients,str):coefficients=json.loads(coefficients)
        if coefficients!=CONTRASTS[key[0]]:raise ValueError("Predeclared contrast direction changed")
        if (str(row.get("training_seeds_pooled","")).lower()not in ("false","0")
                or str(row.get("batch_ci_estimable","")).lower()not in ("true","1")
                or row.get("sample_ci_unit")!="complete_multi_user_channel"
                or row.get("batch_ci_unit")!="sample_index//8"):
            raise ValueError("Complete-channel and batch pairing required; no training-seed pooling")
        for prefix in ("delta_","batch_delta_"):_uncertainty(row,prefix)
        if not math.isclose(finite(row,"delta_mean"),finite(row,"batch_delta_mean"),rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError("Equal-size batch means must preserve the signed estimate")
        indexed[key]=row
    if set(indexed)!=expected_pairs:raise ValueError("All4 contrasts x8 estimands, including negatives, required")
    comparisons={name:dict(per_gamma=[indexed[name,"per_gamma",str(g)]for g in GAMMAS],
        low_gamma=indexed[name,"low_gamma_mean","0,5,10"],
        all_gamma=indexed[name,"all_gamma_mean","0,5,10,15,20,25"])for name in CONTRASTS}
    criteria={}
    for name in PRIMARY:
        low,high=comparisons[name]["low_gamma"],comparisons[name]["per_gamma"][-1]
        criteria[name]=(finite(low,"delta_mean")>=.5 and finite(low,"batch_delta_ci95_low")>0
                        and finite(high,"batch_delta_ci95_low")>=-.1)
    return dict(training_seed=training_seed,test_seed=TEST_SEED,all_rates=summary,comparisons=comparisons,
        engineering_criteria=criteria,engineering_rule=RULE,training_seeds_pooled=False,
        criterion_scope="Per-training-seed engineering screen, not journal acceptance or multi-seed significance",
        prior_scope="Committed historical diagonal power prior; linear error proxy, not posterior variance or calibrated uncertainty",
        reference_scope="Primary comparisons use synchronized +20k fdma_control, not frozen source F",
        model_scope="Joint+Swin already existed; this experiment adds exclusive-user anchors and historical-power-aware reception",
        budget_scope="Six fixed continuation runs; trends do not establish convergence")


def training_curves(training_records):
    if set(training_records)!={str(s)for s in SEEDS}:raise ValueError("Both training seeds required")
    per_seed,source_files={},{}
    for seed in SEEDS:
        records=training_records[str(seed)]
        if set(records)!=set(ARMS):raise ValueError("Exactly the six new continuation logs are required")
        per_seed[str(seed)]={}
        for arm in ARMS:
            record=records[arm]
            if _integer(record,"completed_updates")!=20000:raise ValueError("Each continuation must complete20k")
            path=_safe_path(record["metrics_path"],file=True)
            digest=sha256(path)
            if digest!=record["metrics_sha256"]:raise ValueError("Exact validation log hash changed")
            points=[]
            with path.open(encoding="utf-8")as stream:
                for line in stream:
                    if not line.strip():continue
                    item=json.loads(line)
                    if item.get("event")!="validation":
                        if item.get("event")!="train_gamma":raise ValueError("Unknown metrics event")
                        continue
                    if (item.get("selection_metric_name")!="mean_all" or item.get("gamma_db")!=list(GAMMAS)
                            or finite(item,"downlink_snr_db")!=25):raise ValueError("Validation protocol changed")
                    table=item["table"]
                    if set(table)!={f"layout0_gamma{g}"for g in GAMMAS}:raise ValueError("Six complete validation nodes required")
                    rates={str(g):finite(table,f"layout0_gamma{g}")for g in GAMMAS}
                    mean=finite(item,"selection_metric")
                    if not math.isclose(mean,sum(rates.values())/6,rel_tol=1e-10,abs_tol=1e-10):
                        raise ValueError("Validation selection must equal six-node mean_all")
                    points.append(dict(step=_integer(item,"step"),mean_all=mean,rates=rates))
            if sha256(path)!=digest:raise ValueError("Training log changed during reading")
            if tuple(p["step"]for p in points)!=VALIDATION_STEPS:
                raise ValueError("Exactly41 ordered validation points including step0 and every500 required")
            best=max(points,key=lambda p:p["mean_all"])
            if _integer(record,"best_step")!=best["step"]:
                raise ValueError("Selected step must be first strict validation maximum, including step0")
            tail=[p for p in points if p["step"]>=15000]
            xs,ys=[p["step"]/1000 for p in tail],[p["mean_all"]for p in tail]
            xm,ym=sum(xs)/len(xs),sum(ys)/len(ys)
            slope=sum((x-xm)*(y-ym)for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
            per_seed[str(seed)][arm]=dict(best_step=best["step"],best_mean_all=best["mean_all"],
                completed_updates=20000,points=points,last_five_validation_points=points[-5:],
                last_5000_updates=dict(start_step=15000,end_step=20000,num_points=11,
                    boundary_policy="inclusive endpoints;11 validation points",
                    mean_all_start=ys[0],mean_all_end=ys[-1],end_minus_start=ys[-1]-ys[0],
                    ols_slope_per_1000_updates=slope,minimum=min(ys),maximum=max(ys)),
                metrics_path=str(path),metrics_sha256=digest,convergence_established=False)
            source_files[str(path)]=digest
    if len(source_files)!=6:raise ValueError("Six distinct exact metrics paths required")
    return dict(per_seed=per_seed,source_files=source_files,training_seeds_pooled=False,
                validation_samples=256,validation_batch_size=8,convergence_established=False)


def _figures(output,interpretations,curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":9,
                         "axes.spines.top":False,"axes.spines.right":False})
    colors=dict(zip(METHODS,("#2864a5","#d47b27","#3a945c","#777777")))
    def finish(fig,name):
        for suffix in ("pdf","png"):fig.savefig(output/(name+"."+suffix),dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,layout="constrained")
    for axis,seed in zip(axes,SEEDS):
        for method in METHODS:
            selected=sorted((r for r in interpretations[str(seed)]["all_rates"]if r["method_key"]==method),
                            key=lambda r:finite(r,"gamma_db"))
            axis.errorbar(GAMMAS,[finite(r,"sum_rate_mean")for r in selected],
                yerr=[1.96*finite(r,"sum_rate_se")for r in selected],color=colors[method],
                marker="o",markersize=3,capsize=2,linewidth=1.2,label=LABELS[method])
        axis.set(xlabel="Transmit-reference gamma (dB)",xticks=GAMMAS,title=f"Training seed {seed}")
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Sum rate (bps/Hz); sample 95% CI")
    axes[-1].legend(frameon=False,fontsize=8)
    finish(fig,"hybrid_anchor_four_methods_all_gamma")
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for column,seed in enumerate(SEEDS):
        for row,names in enumerate((PRIMARY,("H16_minus_H32","Fnew_minus_Ffrozen"))):
            axis=axes[row,column]
            for name in names:
                selected=interpretations[str(seed)]["comparisons"][name]["per_gamma"]
                axis.errorbar(GAMMAS,[finite(r,"delta_mean")for r in selected],
                    yerr=[1.96*finite(r,"batch_delta_se")for r in selected],marker="o",markersize=3,
                    capsize=2,linewidth=1,label=CONTRAST_LABELS[name])
            axis.axhline(0,color=".5",linestyle="--",linewidth=.8)
            axis.set(xlabel="Transmit-reference gamma (dB)",xticks=GAMMAS,
                     title=f"Training seed {seed}",ylabel="Paired rate contrast; batch 95% CI")
            axis.grid(alpha=.18);axis.legend(frameon=False,fontsize=8)
    finish(fig,"hybrid_anchor_all_paired_contrasts")
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,layout="constrained")
    for axis,seed in zip(axes,SEEDS):
        for arm in ARMS:
            points=curves["per_seed"][str(seed)][arm]["points"]
            axis.plot([p["step"]for p in points],[p["mean_all"]for p in points],color=colors[arm],
                      linewidth=1.1,label=LABELS[arm])
        axis.set(xlabel="Additional optimizer updates",title=f"Training seed {seed}")
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Fixed256-channel validation mean_all (bps/Hz)")
    axes[-1].legend(frameon=False,fontsize=8)
    finish(fig,"hybrid_anchor_validation_mean_all")


def result_paragraphs(interpretations,curves,source_records):
    en=["Internal grouped Hybrid Anchor report. The primary controls are the synchronized additional20k fdma_control runs, not frozen source F. Joint+Swin already existed; the new intervention combines exclusive-user anchors, group-shared symbols and historical diagonal-power-aware reception. No claim of an independently decodable base layer is made."]
    zh=["Grouped Hybrid Anchor 内部结果草稿。主要对照是同步追加20k的新 fdma_control，而不是冻结源 F。Joint+Swin 此前已经存在；本轮加入独占用户 anchor、组内共享符号及历史对角功率感知接收，不宣称独占部分已经是可独立解码的基础层。"]
    for seed in SEEDS:
        source=source_records[str(seed)]
        en.append(f"Training seed {seed}: all three new arms migrated from source F best step {source['best_step']} (SHA256 {source['best_sha256']}) and completed20000 new optimizer updates with fresh optimizer/scheduler state.")
        zh.append(f"训练 seed {seed}：三个新配置均从源 F best step {source['best_step']}（SHA256 {source['best_sha256']}）迁移，以全新 optimizer/scheduler 完成20000次追加更新。")
        for arm in ARMS:
            item=curves["per_seed"][str(seed)][arm];tail=item["last_5000_updates"]
            last=", ".join(f"{p['step']}:{p['mean_all']:.6f}"for p in item["last_five_validation_points"])
            en.append(f"{arm}, seed {seed}: selected step {item['best_step']} from step0 and every500 updates. Last five validation mean_all points: {last}. Inclusive15000-20000 (11 points): end-minus-start {tail['end_minus_start']:+.6f} bps/Hz, OLS slope {tail['ols_slope_per_1000_updates']:+.6f} per1000 updates. This is a trend, not evidence of convergence.")
            zh.append(f"{arm}，seed {seed}：从step0及每500步选择best step {item['best_step']}。最后五个验证 mean_all 点：{last}。15000–20000含端点11点：末减初 {tail['end_minus_start']:+.6f} bps/Hz，OLS斜率 {tail['ols_slope_per_1000_updates']:+.6f}/1000更新。仅描述趋势，不证明收敛。")
        item=interpretations[str(seed)]
        for name in CONTRASTS:
            comparison=item["comparisons"][name]
            for scope,value in (("low gamma0/5/10",comparison["low_gamma"]),("all six gamma",comparison["all_gamma"]),
                                ("gamma25",comparison["per_gamma"][-1])):
                mean=finite(value,"delta_mean")
                a,b,c,d=[finite(value,k)for k in ("delta_ci95_low","delta_ci95_high","batch_delta_ci95_low","batch_delta_ci95_high")]
                en.append(f"Seed {seed}, {CONTRAST_LABELS[name]}, {scope}: {mean:+.6f} bps/Hz; sample-paired95% CI [{a:+.6f},{b:+.6f}], batch-paired95% CI [{c:+.6f},{d:+.6f}].")
                zh.append(f"seed {seed}，{CONTRAST_LABELS[name]}，{scope}：{mean:+.6f} bps/Hz；逐样本配对95%区间[{a:+.6f},{b:+.6f}]，batch配对95%区间[{c:+.6f},{d:+.6f}]。")
        for name,passed in item["engineering_criteria"].items():
            en.append(f"Seed {seed}, {name}: engineering rule {'met'if passed else'not met'} ({RULE}); this is not journal acceptance or multi-training-seed significance.")
            zh.append(f"seed {seed}，{name}：工程判据{'满足'if passed else'未满足'}（低三点均值≥0.5、低三点batch区间下界>0、25dB的batch区间下界≥−0.1）；这不是期刊验收或跨训练seed显著性。")
    en.append("All six gamma points, four methods, four signed comparisons and both independently reported training seeds are retained, including negative results. Low/all-gamma differences are formed within each complete multi-user channel before aggregation; users and subcarriers do not inflate sample size. Intervals describe evaluation sampling conditional on selected checkpoints, not training-seed uncertainty. Each layout jointly normalizes exclusive+shared transmitted energy to64 per UE with4Df+2Da=256; gamma is a transmit reference, not historical received feedback SNR. The receiver proxy is diagonal-prior linear error, not a true posterior variance or established empirical calibration. Frozen F has no corresponding anchor/prior-proxy diagnostics and therefore uses NA, not zero. Diagnostics are detached after decoding and never become decoder truth inputs. Stored parameter counts are not latency. No extra ablation training, test-point stitching, adaptive extension, historical Swin overlay or manuscript editing is performed. Raw samples and weights stay on the server.")
    zh.append("完整保留六个gamma点、四方法、四项有符号对比及两个分别报告的训练seed，包括负结果。低/全gamma差值先在同一完整多用户信道内计算，再聚合；不以用户或子载波扩大样本数。区间只描述所选checkpoint条件下的测试采样，不代表训练seed不确定性。每配置独占+共享发射码字联合归一化到每UE能量64，预算4Df+2Da=256；gamma是发射参考量，不是历史接收feedback SNR。接收proxy是对角先验下线性误差，不称真实后验方差或已完成经验校准。冻结F没有对应anchor/先验proxy，记为NA而非零。诊断在解码完成后detach，真实码字不进入decoder。存储参数量不代表时延。不新增消融训练、不逐测试点拼接最优曲线、不自适应延训、不搬入历史Swin曲线、不改论文；原始样本和权重留在服务器。")
    return en,zh


def organize_results(folder,evaluations,training_records,source_records,*,make_plots=True):
    if set(evaluations)!={str(s)for s in SEEDS}or set(source_records)!={str(s)for s in SEEDS}:
        raise ValueError("Two explicit evaluation/source records required")
    curves=training_curves(training_records)
    interpretations,manifests,sources,parameter_tables={}, {}, {}, {}
    reference=None
    source_files=dict(curves["source_files"])
    for seed in SEEDS:
        source=_safe_path(evaluations[str(seed)])
        manifest,digest=_read_json(source/"manifest.json")
        if (manifest.get("status")!="completed" or manifest.get("results_verified")is not True
                or manifest.get("freeze_checks",{}).get("passed")is not True
                or manifest.get("freeze_checks",{}).get("errors")!=[]
                or set(manifest.get("freeze_checks",{}).get("loaded_methods",()))!=set(METHODS)
                or manifest.get("shared_channel_noise_inputs_unchanged")is not True
                or manifest.get("samples_hashed_pairing")is not True
                or manifest.get("prior_frozen")is not True or manifest.get("protocol")!=protocol(seed)
                or manifest.get("output_sha256")!=artifact_hashes(source)):
            raise ValueError("Evaluation protocol/freeze/artifact verification failed")
        if (manifest.get("model_buffers_before")!=manifest.get("model_buffers_after")
                or not manifest.get("input_sha256_before")
                or manifest.get("input_sha256_before")!=manifest.get("input_sha256_after")):
            raise ValueError("Inputs, prior buffers or model mode changed")
        snapshots=manifest["model_buffers_before"]
        if set(snapshots)!=set(METHODS)or any(not s.get("training")or any(s["training"].values())for s in snapshots.values()):
            raise ValueError("All four methods require frozen eval-mode evidence")
        reference=verify_shared_inputs(manifest,reference)
        pinned=source_records[str(seed)]
        if pinned["best_sha256"]!=SOURCE_BEST_SHA256[seed]:
            raise ValueError("Wrong same-seed pinned source F")
        for name in ("best_path","latest_path","config_path","status_path","best_sha256","latest_sha256",
                     "config_sha256","status_sha256","best_step","latest_step"):
            if manifest["source_F"][name]!=pinned[name]:raise ValueError("Source F evidence changed")
        for arm in ARMS:
            trained,loaded=training_records[str(seed)][arm],manifest["methods"][arm]
            if (trained["best_step"]!=loaded["selected_best_step"] or
                    loaded["prior_update_count"]!=trained["best_step"] or
                    loaded["latest_step"]!=20000 or loaded["strict_loading"]is not True):
                raise ValueError("Selected weights/prior count/completion identity mismatch")
            for name in ("best_sha256","latest_sha256"):
                if trained[name]!=loaded[name]:raise ValueError("Training/evaluation checkpoint digest differs")
        frozen=manifest["methods"]["frozen_F"]
        if (frozen["best_sha256"]!=pinned["best_sha256"] or frozen["selected_best_step"]!=pinned["best_step"]
                or frozen.get("prior_update_count")is not None or frozen.get("strict_loading")is not True):
            raise ValueError("Frozen reference is not the exact pinned best")
        summary=rows(source/f"summary_{TAG}.csv")
        interpretations[str(seed)]=interpret(summary,rows(source/f"paired_deltas_{TAG}.csv"),seed)
        parameters=rows(source/f"parameters_{TAG}.csv")
        if len(parameters)!=4 or {r["method_key"]for r in parameters}!=set(METHODS):
            raise ValueError("Four actual parameter rows required")
        indexed={}
        for row in parameters:
            full,ue,bs=(_integer(row,name)for name in ("full_parameters","ue_parameters","bs_parameters"))
            if _integer(row,"training_seed")!=seed or min(full,ue,bs)<=0 or full!=ue+bs:
                raise ValueError("Actual parameter accounting mismatch")
            indexed[row["method_key"]]=(full,ue,bs)
        if len({indexed[arm]for arm in ARMS})!=1:raise ValueError("Three new layouts must have identical capacity")
        parameter_tables[str(seed)]=parameters
        sources[str(seed)]=source;manifests[str(seed)]=(manifest,digest)
    parent=_safe_path(folder)
    if not parent.is_dir():raise FileNotFoundError(parent)
    output=parent/("export_"+uuid.uuid4().hex);output.mkdir(exist_ok=False)
    for seed in SEEDS:
        source=sources[str(seed)];manifest,digest=manifests[str(seed)]
        for prefix in ("summary","paired_deltas","parameters"):
            name=f"{prefix}_{TAG}.csv"
            actual=_copy_small(source/name,output/f"seed{seed}_{name}",manifest["output_sha256"][name])
            source_files[str(source/name)]=actual
        _copy_small(source/"manifest.json",output/f"seed{seed}_evaluation_manifest.json",digest)
        source_files[str(source/"manifest.json")]=digest
    save_json(output/"comparison_summary.json",dict(per_seed=interpretations,training_seeds_pooled=False))
    save_json(output/"training_trends.json",curves)
    save_json(output/"source_evidence.json",dict(source_records=source_records,training_records=training_records,
        source_files=source_files,plots_generated=bool(make_plots),weights_loaded_by_report=False))
    learning,trends,diagnostics=[],[],[]
    for seed in SEEDS:
        for arm in ARMS:
            item=curves["per_seed"][str(seed)][arm]
            for point in item["points"]:
                learning.append(dict(training_seed=seed,method_key=arm,step=point["step"],mean_all=point["mean_all"],
                    **{f"gamma{g}_rate":point["rates"][str(g)]for g in GAMMAS}))
            trends.append(dict(training_seed=seed,method_key=arm,best_step=item["best_step"],
                best_mean_all=item["best_mean_all"],**item["last_5000_updates"],convergence_established=False))
        for row in interpretations[str(seed)]["all_rates"]:
            names=("exclusive_energy_mean_per_ue_mean","shared_energy_mean_per_ue_mean",
                   "recovered_mse_active_mean","proxy_mse_active_mean","recovered_mse_exclusive_mean",
                   "proxy_mse_exclusive_mean","recovered_mse_shared_mean","proxy_mse_shared_mean",
                   "anchor_rms_mean_mean","downlink_desired_mean_mean","downlink_interference_mean_mean")
            diagnostics.append(dict(training_seed=seed,method_key=row["method_key"],gamma_db=row["gamma_db"],
                diagnostic_schema=row["diagnostic_schema"],**{name:row[name]for name in names}))
    _write_csv(output/"validation_mean_all_all_points.csv",learning)
    _write_csv(output/"training_last5000_trends.csv",trends)
    _write_csv(output/"receiver_diagnostics_summary.csv",diagnostics)
    if make_plots:_figures(output,interpretations,curves)
    english,chinese=result_paragraphs(interpretations,curves,source_records)
    for name,paragraphs in (("results_draft_en.md",english),("results_draft_zh.md",chinese)):
        with (output/name).open("x",encoding="utf-8")as stream:stream.write("\n\n".join(paragraphs)+"\n")
    for path,digest in source_files.items():
        if sha256(_safe_path(path,file=True))!=digest:raise ValueError("Report input changed during export")
    files=sorted(path for path in output.iterdir()if path.is_file())
    return dict(export_directory=str(output),per_seed=interpretations,training_trends=curves,
        parameters=parameter_tables,training_seeds_pooled=False,files={str(p):sha256(p)for p in files},_evidence=files)
