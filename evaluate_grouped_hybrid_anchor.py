#!/usr/bin/env python3
"""Fixed grouped-Hybrid-Anchor evaluation; no training, source discovery or old-batch replay."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import traceback

from configs.grouped_hybrid_anchor_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT as BATCH_ROOT, SOURCE_BEST_SHA256,
    anchor_config, source_identity, channel_data_api, protocol as physical)
from configs.grouped_codec_fdma_train_config import fdma_config
from evaluate_proposal_as_direct import (
    sample_statistics, check_tensor_state, load_payload, read_json, write_manifest)
from evaluate_proposal_lite import _compare, model_buffer_snapshot, validate_frozen
from evaluate_proposal_as_receiver import create_output, artifact_hashes as _artifact_hashes, runtime_record as _runtime_record
from preflight_proposal_as_direct import sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = BATCH_ROOT / "evaluation"
TAG = "k4_grouped_hybrid_anchor_v1"
METHODS = (*ARMS, "frozen_F")
GAMMAS, TEST_SEED = (0,5,10,15,20,25), 20261228
LAYOUTS = {"fdma_control": (64,0), "hybrid_f16_a96": (16,96),
           "hybrid_f32_a64": (32,64), "frozen_F": (64,0)}
CONTRASTS = {
    "H16_minus_Fnew": {"hybrid_f16_a96": 1, "fdma_control": -1},
    "H32_minus_Fnew": {"hybrid_f32_a64": 1, "fdma_control": -1},
    "H16_minus_H32": {"hybrid_f16_a96": 1, "hybrid_f32_a64": -1},
    "Fnew_minus_Ffrozen": {"fdma_control": 1, "frozen_F": -1},
}
COMMON_METRICS = (
    "sum_rate", "nmse_db", "precoder_power", "precoder_power_max_error",
    "energy_ue0", "energy_ue1", "energy_ue2", "energy_ue3",
    "exclusive_energy_mean_per_ue", "shared_energy_mean_per_ue", "empirical_noise_power",
    "recovered_mse_active", "recovered_nmse_linear_active", "recovered_nmse_db_active",
    "corrected_mse_active", "corrected_nmse_linear_active", "corrected_nmse_db_active",
    "recovered_mse_exclusive", "downlink_desired_mean", "downlink_interference_mean")
OPTIONAL_METRICS = ("recovered_mse_shared", "proxy_mse_active", "proxy_mse_exclusive",
                    "proxy_mse_shared", "prior_power_mean_active", "anchor_rms_mean")
METRICS = COMMON_METRICS + OPTIONAL_METRICS
OUTPUT_ARTIFACTS = tuple(f"{name}_{TAG}.{suffix}" for name,suffix in (
    ("summary","csv"), ("samples","csv.gz"), ("paired_deltas","csv"),
    ("paired_samples","csv.gz"), ("parameters","csv"), ("metadata","json"), ("results","json")))
SOURCE_FILES = (
    "evaluate_grouped_hybrid_anchor.py", "configs/grouped_hybrid_anchor_train_config.py",
    "train_grouped_hybrid_anchor.py", "models/grouped_hybrid_anchor.py", "utils/grouped_hybrid_feedback.py",
    "main_train.py", "models/grouped_codec_precoder.py", "models/grouped_codec_fdma.py",
    "models/adaptive_hybrid_joint_swin.py", "models/adaptive_hybrid.py", "models/swin_precoder.py",
    "models/baseline_common.py", "configs/grouped_codec_train_config.py",
    "configs/grouped_codec_fdma_train_config.py", "train_grouped_codec.py", "train_grouped_codec_fdma.py",
    "evaluate_grouped_codec_fdma.py", "baseline_evaluate.py", "utils/replayable_channel_data.py",
    "utils/sionna_rng.py", "utils/channel_generator.py", "utils/layers.py",
    "evaluate_proposal_as_direct.py", "evaluate_proposal_as_receiver.py", "evaluate_proposal_lite.py")


def make_test_data():
    return channel_data_api().ReplayableChannelData(
        base_seed=physical.namespace_seed(TEST_SEED,"channel_test"), k_users=4,nt=32,nsc=256,
        batch_size=8,validation_batch_size=8,carrier_freq=3.5e9,speed=1.,cdl_model="C",device="cuda:0")


def verify_test_namespace():
    current = make_test_data()
    current_seeds = {current.seed_for("training",i) for i in range(1,201)}
    historical = (20261101,20261103,20261120,20261210,20261220)
    for seed in historical:
        old = channel_data_api().ReplayableChannelData(base_seed=seed,k_users=4,nt=32,nsc=256,
            batch_size=8,validation_batch_size=8,carrier_freq=3.5e9,speed=1.,cdl_model="C",device="cuda:0")
        if current.protocol["data_id"] == old.protocol["data_id"] or current_seeds & {
                old.seed_for("training",i) for i in range(1,201)}:
            raise ValueError("New test namespace overlaps a historical screening address")
    return dict(historical_test_seeds_checked=list(historical),actual_channel_seed_sets_disjoint=True,
                new_namespace="grouped_hybrid_anchor_v1",training_seed_omitted=True,gamma_omitted=True)


def protocol(training_seed):
    if type(training_seed) is not int or training_seed not in SEEDS:
        raise ValueError("One prescribed training seed required")
    return dict(version="grouped-hybrid-anchor-evaluation-v1",training_seed=training_seed,seed=TEST_SEED,
        methods=list(METHODS),gamma_db=list(GAMMAS),downlink_snr_db=25,preflight=False,
        batch_size=8,num_batches=200,num_samples=1600,scenario=dict(K=4,Nt=32,Nsc=256,Dtot=256),
        physical_protocol=physical.protocol_metadata(),channel_data_protocol=make_test_data().protocol,
        namespace_check=verify_test_namespace(),channel_address="training(batch_index+1)",
        noise_address=dict(namespace="grouped_hybrid_anchor_v1",seed=TEST_SEED,split="test",
                           index="zero-based batch_index",omits=["method","training_seed","gamma"]),
        gamma_definition="10log10(64/(256*N0)); transmit reference, not received feedback SNR",
        noise_variance="N0=0.25*10**(-gamma/10); one common eps and one physical noise realization",
        grouped_budget="4*Df_per_user+2*Da_per_group=256; never old 4*Df+Da",
        layout_by_method={m:dict(Df_per_user=LAYOUTS[m][0],Da_per_group=LAYOUTS[m][1]) for m in METHODS},
        numerical_policy=dict(real_dtype="float32",complex_dtype="complex64",matmul_precision="highest",
                              allow_tf32=False,amp=False,microbatch=False),
        prior_frozen=True,source_F_best_sha256=SOURCE_BEST_SHA256[training_seed],
        sample_ci_unit="complete_multi_user_channel",batch_ci_unit="sample_index//8",
        training_seeds_pooled=False,test_selects_checkpoint=False,
        diagnostic_scope="Truth is detached after decoder/RZF; proxy is diagonal-prior linear error, not posterior variance",
        absent_diagnostics="frozen_F proxy/prior/anchor and empty shared support use null/CSV blank, never fabricated zero",
        downlink_diagnostic_units="mean desired/interfering power over four users and 256 subcarriers")


def checkpoint_plan(training_seed):
    protocol(training_seed)
    entries=[]
    for arm in ARMS:
        cfg=anchor_config(arm,training_seed)
        entries.append(dict(method_key=arm,training_seed=training_seed,best_path=cfg.BEST_PATH,
            latest_path=cfg.SAVE_PATH,config_path=cfg.CONFIG_PATH,status_path=cfg.STATUS_PATH,
            expected_source_best_sha256=SOURCE_BEST_SHA256[training_seed],new_optimizer_updates=20000))
    source=source_identity(training_seed)
    entries.append(dict(method_key="frozen_F",training_seed=training_seed,
        **{name:source[name] for name in ("best_path","latest_path","config_path","status_path")},
        expected_best_sha256=SOURCE_BEST_SHA256[training_seed],new_optimizer_updates=0))
    return entries


def sample_identity(method,gamma,index,training_seed):
    if method not in METHODS or gamma not in GAMMAS or type(index) is not int or index<0:
        raise ValueError("Invalid explicit Hybrid Anchor result identity")
    if training_seed not in SEEDS: raise ValueError("Unexpected training seed")
    df,da=LAYOUTS[method]
    return dict(method_key=method,training_seed=training_seed,seed=TEST_SEED,gamma_db=gamma,
        downlink_snr_db=25,Df_per_user=df,Da_per_group=da,occupied_feedback_RE=256,
        active_symbols_per_ue=df+da,sample_index=index,batch_index=index//8,
        noise_variance_N0=.25*10**(-gamma/10),
        diagnostic_schema="frozen_F_no_anchor_or_power_prior_proxy" if method=="frozen_F" else "historical_power_prior_anchor_v1")


def _available(method,metric):
    if metric=="recovered_mse_shared": return LAYOUTS[method][1]>0
    if metric=="proxy_mse_shared": return method!="frozen_F" and LAYOUTS[method][1]>0
    return method!="frozen_F"


def summarize_samples(samples,count,training_seed):
    if type(count) is not int or count<8 or count%8:
        raise ValueError("Use complete batches of eight multi-user channels")
    indexed={(method,gamma):{} for method in METHODS for gamma in GAMMAS}
    for row in samples:
        key=(row.get("method_key"),row.get("gamma_db")); index=row.get("sample_index")
        if key not in indexed or type(index) is not int or not 0<=index<count or index in indexed[key]:
            raise ValueError("Missing/duplicate/unexpected full sample key")
        if any(name in row for name in ("feedback_snr_db","allocation_df","allocation_da")):
            raise ValueError("This grouped Hybrid protocol is not a legacy allocation")
        for name,value in sample_identity(key[0],key[1],index,training_seed).items():
            _compare(value,row.get(name),"Sample identity."+name)
        for name in COMMON_METRICS:
            if name not in row or row[name] is None or isinstance(row[name],bool) or not math.isfinite(float(row[name])):
                raise ValueError("Missing/nonfinite common diagnostic: "+name)
        for name in OPTIONAL_METRICS:
            if name not in row: raise ValueError("Missing explicit diagnostic availability: "+name)
            if _available(key[0],name):
                if row[name] is None or isinstance(row[name],bool) or not math.isfinite(float(row[name])):
                    raise ValueError("Available diagnostic must be finite: "+name)
            elif row[name] is not None:
                raise ValueError("Unavailable diagnostics must be null, not zero: "+name)
        if any(abs(float(row[f"energy_ue{k}"])-64)>1e-4 for k in range(4)):
            raise ValueError("Joint per-UE energy must remain64")
        if abs(row["exclusive_energy_mean_per_ue"]+row["shared_energy_mean_per_ue"]-64)>1e-4:
            raise ValueError("Exclusive and shared energy must sum to64, not64 each")
        if (min(row["exclusive_energy_mean_per_ue"],row["shared_energy_mean_per_ue"])<0 or
                LAYOUTS[key[0]][1]==0 and row["shared_energy_mean_per_ue"]!=0):
            raise ValueError("Invalid support energy")
        if not 0<=row["precoder_power_max_error"]<=1e-5 or abs(row["precoder_power"]-256)>256e-5:
            raise ValueError("Per-subcarrier precoder power violated")
        nonnegative=("empirical_noise_power","downlink_desired_mean","downlink_interference_mean",
            "recovered_mse_active","corrected_mse_active","recovered_mse_exclusive")+OPTIONAL_METRICS
        if any(row[name] is not None and row[name]<0 for name in nonnegative):
            raise ValueError("Negative squared-power/error diagnostic")
        indexed[key][index]=row
    summary=[]
    for (method,gamma),records in indexed.items():
        if set(records)!=set(range(count)): raise ValueError("Incomplete method/gamma sample indices")
        result=sample_identity(method,gamma,0,training_seed)
        result.pop("sample_index");result.pop("batch_index")
        result.update(num_samples=count,num_batches=count//8,batch_size=8,preflight=count!=1600,
                      prior_frozen=True,rzf_regularization=1e-3)
        for metric in METRICS:
            values=[records[i][metric] for i in range(count)]
            if values[0] is None:
                stats=dict(mean=None,se=None,ci95_low=None,ci95_high=None)
            else: stats=sample_statistics(values)
            result.update({metric+"_"+name:value for name,value in stats.items()})
        summary.append(result)
    for gamma in GAMMAS:
        for index in range(count):
            expected=indexed[METHODS[0],gamma][index]["empirical_noise_power"]
            if any(not math.isclose(indexed[m,gamma][index]["empirical_noise_power"],expected,rel_tol=1e-6,abs_tol=1e-10)
                   for m in METHODS[1:]):
                raise ValueError("Methods did not use the same realized absolute physical noise")
    return summary,indexed


def paired_rows(summary,samples,count,training_seed):
    actual,indexed=summarize_samples(samples,count,training_seed)
    supplied={(r.get("method_key"),r.get("gamma_db")):r for r in summary}
    if len(summary)!=24 or len(supplied)!=24: raise ValueError("Exactly24 unique summary conditions required")
    for row in actual:
        candidate=supplied.get((row["method_key"],row["gamma_db"]),{})
        for name,value in row.items():
            if isinstance(value,float):
                if candidate.get(name) is None or not math.isfinite(float(candidate[name])) or not math.isclose(
                        value,float(candidate[name]),rel_tol=1e-6,abs_tol=1e-6):
                    raise ValueError("Summary/raw mismatch: "+name)
            else: _compare(value,candidate.get(name),"Summary/raw."+name)
    output,raw=[],[]
    scopes=[("per_gamma",(g,)) for g in GAMMAS]+[("low_gamma_mean",(0,5,10)),("all_gamma_mean",GAMMAS)]
    for name,coefficients in CONTRASTS.items():
        for estimand,gammas in scopes:
            values=[statistics.fmean(sum(c*indexed[m,g][i]["sum_rate"] for m,c in coefficients.items())
                    for g in gammas) for i in range(count)]
            batches=[statistics.fmean(values[i:i+8]) for i in range(0,count,8)]
            record=dict(contrast_id=name,coefficients=json.dumps(coefficients,sort_keys=True),
                estimand=estimand,gamma_points=",".join(map(str,gammas)),
                gamma_db=gammas[0] if len(gammas)==1 else None,seed=TEST_SEED,training_seed=training_seed,
                downlink_snr_db=25,num_samples=count,num_batches=count//8,batch_size=8,
                sample_ci_unit="complete_multi_user_channel",batch_ci_unit="sample_index//8",
                batch_ci_estimable=count>8,training_seeds_pooled=False,
                role="primary" if name in ("H16_minus_Fnew","H32_minus_Fnew") and estimand=="low_gamma_mean"
                     else "secondary25" if name in ("H16_minus_Fnew","H32_minus_Fnew") and gammas==(25,) else "diagnostic")
            for prefix,sequence in (("delta_",values),("batch_delta_",batches)):
                record.update({prefix+k:v for k,v in sample_statistics(sequence).items()})
            output.append(record)
            raw.extend(dict(contrast_id=name,estimand=estimand,gamma_points=record["gamma_points"],
                sample_index=i,batch_index=i//8,delta=value,seed=TEST_SEED,training_seed=training_seed)
                for i,value in enumerate(values))
    return output,raw


def extract_diagnostics(torch,diagnostics,noise,gamma,method):
    """Offline only: no simulator truth/proxy calibration is fed back to the model."""
    batch=noise.shape[0]
    z,corrected,truth=[diagnostics[name] for name in ("z_hat","z_corrected","true_z")]
    for value in (z,corrected,truth,diagnostics["energy"],diagnostics["noise_power"]):
        if (not torch.is_tensor(value) or value.requires_grad or not torch.isfinite(value).all().item()):
            raise ValueError("Detached finite diagnostic tensors required")
    if any(tuple(v.shape)!=(batch,4,256) for v in (z,corrected,truth)):
        raise ValueError("Canonical diagnostic shape must be B,4,256")
    active=diagnostics["active_mask"]
    if active.dtype!=torch.bool or tuple(active.shape)!=(4,256):
        raise ValueError("Canonical boolean support mask required")
    if method=="frozen_F":
        exclusive,shared=active,torch.zeros_like(active)
    else:
        exclusive,shared=diagnostics["exclusive_mask"],diagnostics["shared_mask"]
    if (exclusive.dtype!=torch.bool or shared.dtype!=torch.bool or exclusive.shape!=active.shape or
            shared.shape!=active.shape or torch.any(exclusive&shared).item() or
            not torch.equal(exclusive|shared,active)):
        raise ValueError("Exclusive/shared masks must partition actual support")
    df,da=LAYOUTS[method]
    # The frozen reference uses the same genuine FDMA endpoint mapping, not a
    # disguised model/config identity.
    geometry=physical.tensor_geometry(physical.LayoutSpec("fdma_control" if method=="frozen_F" else method),active.device)
    if any(not torch.equal(value,geometry[name]) for name,value in
           (("active_mask",active),("exclusive_mask",exclusive),("shared_mask",shared))):
        raise ValueError("Canonical support differs from the declared physical layout")
    if (not torch.equal(exclusive.sum(-1),torch.full((4,),df,device=active.device)) or
            not torch.equal(shared.sum(-1),torch.full((4,),da,device=active.device))):
        raise ValueError("Declared per-user exclusive/shared codeword support differs")
    expanded=active[None].expand_as(truth)
    if any(torch.count_nonzero(value[~expanded]).item() for value in (z,corrected,truth)):
        raise ValueError("Inactive canonical coordinates must remain exactly zero")
    energy=diagnostics["energy"]
    if tuple(energy.shape)!=(batch,4) or not torch.allclose(energy,truth.abs().square().sum(-1),rtol=0,atol=1e-4):
        raise ValueError("Normalized truth and per-UE energy diagnostics disagree")
    per_coordinate_error=(z-truth).abs().square()
    def masked_mean(value,mask):
        count=int(mask.sum().item())
        return None if not count else value.masked_fill(~mask[None],0).sum((1,2))/count
    exclusive_energy=truth.abs().square().masked_fill(~exclusive[None],0).sum(-1)
    shared_energy=truth.abs().square().masked_fill(~shared[None],0).sum(-1)
    values={f"energy_ue{k}":energy[:,k] for k in range(4)}
    values.update(exclusive_energy_mean_per_ue=exclusive_energy.mean(-1),
                  shared_energy_mean_per_ue=shared_energy.mean(-1),
                  recovered_mse_exclusive=masked_mean(per_coordinate_error,exclusive),
                  recovered_mse_shared=masked_mean(per_coordinate_error,shared))
    denominator=truth.abs().square().sum((1,2)).clamp_min(1e-12)
    for prefix,value in (("recovered",z),("corrected",corrected)):
        squared=(value-truth).abs().square().masked_fill(~expanded,0).sum((1,2))
        ratio=squared/denominator
        values.update({prefix+"_mse_active":squared/int(active.sum().item()),
                       prefix+"_nmse_linear_active":ratio,prefix+"_nmse_db_active":10*torch.log10(ratio.clamp_min(1e-12))})
    N0=.25*10**(-gamma/10)
    if not math.isclose(float(diagnostics["N0"]),N0,rel_tol=1e-6,abs_tol=1e-12):
        raise ValueError("Absolute feedback noise variance changed")
    measured=diagnostics["noise_power"]
    if tuple(measured.shape)!=(batch,) or not torch.allclose(measured,
            (noise*math.sqrt(N0/2)).abs().square().mean((1,2)),rtol=1e-6,atol=1e-10):
        raise ValueError("Realized physical noise was resampled or rescaled")
    values["empirical_noise_power"]=measured
    if method=="frozen_F":
        values.update({name:None for name in OPTIONAL_METRICS if name!="recovered_mse_shared"})
    else:
        proxy,prior=diagnostics["error_proxy"],diagnostics["prior_power"]
        if tuple(prior.shape)==(4,256): prior=prior[None].expand(batch,-1,-1)
        for value in (proxy,prior,diagnostics["anchor_rms"],diagnostics["error_feature"]):
            if not torch.is_tensor(value) or value.requires_grad or not torch.isfinite(value).all().item():
                raise ValueError("Detached finite public receiver diagnostics required")
        if proxy.shape!=truth.shape or prior.shape!=truth.shape or torch.any(proxy<0).item() or torch.any(prior<0).item():
            raise ValueError("Invalid diagonal-prior linear error proxy")
        if any(torch.count_nonzero(value[~expanded]).item() for value in (proxy,prior)):
            raise ValueError("Inactive proxy/prior coordinates must be zero")
        feature=diagnostics["error_feature"]
        if feature.shape!=proxy.shape or not torch.allclose(
                feature,torch.log1p(proxy/prior.clamp_min(1e-8)),rtol=1e-6,atol=1e-7):
            raise ValueError("Public error-feature transform differs from the declared proxy transform")
        if not torch.allclose(prior.sum(-1),torch.full((batch,4),64.,device=prior.device),rtol=0,atol=1e-4):
            raise ValueError("Interpolated per-UE historical prior must sum to64")
        for name,expected in (("energy_exclusive",exclusive_energy),("energy_shared",shared_energy)):
            if not torch.allclose(diagnostics[name],expected,rtol=0,atol=1e-4):
                raise ValueError("Route energy diagnostic differs from jointly normalized transmitted codeword")
        anchor=diagnostics["anchor_rms"]
        if tuple(anchor.shape)!=(batch,4) or torch.any(anchor<0).item():
            raise ValueError("Per-user anchor residual RMS required")
        values.update(proxy_mse_active=masked_mean(proxy,active),proxy_mse_exclusive=masked_mean(proxy,exclusive),
            proxy_mse_shared=masked_mean(proxy,shared),prior_power_mean_active=masked_mean(prior,active),
            anchor_rms_mean=anchor.mean(-1))
    return values


def downlink_diagnostics(torch,W,H_dl):
    power=torch.einsum("bkns,bnjs->bkjs",H_dl.conj(),W).abs().square()
    signal=torch.diagonal(power,dim1=1,dim2=2).permute(0,2,1)
    interference=(power.sum(2)-signal).clamp_min(0)
    return dict(downlink_desired_mean=signal.mean((1,2)),
                downlink_interference_mean=interference.mean((1,2)))


def artifact_hashes(output):
    return _artifact_hashes(output,expected_artifacts=OUTPUT_ARTIFACTS)


def verify_shared_inputs(manifest,reference=None):
    entries=manifest.get("shared_batch_inputs")
    if not isinstance(entries,list) or len(entries)!=200: raise ValueError("All200 actual input hashes required")
    expected={"batch_index","sample_index_start","sample_index_stop","H_dl_sha256","H_ul_sha256","standard_noise_sha256"}
    for index,row in enumerate(entries):
        if (set(row)!=expected or row["batch_index"]!=index or row["sample_index_start"]!=8*index
                or row["sample_index_stop"]!=8*(index+1)):
            raise ValueError("Incomplete/reordered batch identity")
        for name in ("H_dl_sha256","H_ul_sha256","standard_noise_sha256"):
            value=row[name]
            if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid actual tensor SHA256")
    if reference is not None and entries!=reference:
        raise ValueError("Two training seeds must use identical actual test channels and standard noise")
    return entries


def run_evaluation(training_seed,*,output_dir=None):
    p=protocol(training_seed)
    output=create_output(training_seed,output_dir=output_dir,root=OUTPUT_ROOT)
    manifest=dict(status="pending",protocol=p,output_directory=str(output),methods={},shared_batch_inputs=[],
        model_execution={},prior_frozen=True,training_seeds_pooled=False)
    hashes,snapshots,models,samples={},{},{},[]
    failure=None
    try:
        manifest["runtime"]=_runtime_record(source_files=SOURCE_FILES)
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no laptop CPU fallback")
        from train_grouped_hybrid_anchor import (
            load_source,verify_training_metadata,build_model,validate_model_state,apply_backend)
        from evaluate_grouped_codec_fdma import build_fdma_model
        from preflight_joint_swin import tensor_digest
        import baseline_evaluate as ev
        manifest["runtime_before_backend"]=_runtime_record(torch,source_files=SOURCE_FILES)
        apply_backend(torch)
        manifest["runtime"]=_runtime_record(torch,source_files=SOURCE_FILES)
        device=torch.device("cuda:0");ev.set_seed(TEST_SEED)
        source_payload,source_record=load_source(torch,training_seed)
        manifest["source_F"]=source_record
        source=source_identity(training_seed)
        for name in ("best","latest","config","status"):
            path=source[name+"_path"]
            hashes[str(path)]=sha256_file(path)
            if hashes[str(path)]!=source_record[name+"_sha256"]:
                raise ValueError("Verified source identity changed before evaluation loading")
        if hashes[source["best_path"]]!=SOURCE_BEST_SHA256[training_seed]:
            raise ValueError("Pinned source F changed after verification")
        parameters=[]
        for entry in checkpoint_plan(training_seed):
            method=entry["method_key"]
            if method=="frozen_F":
                model,cfg=build_fdma_model(torch,training_seed,source_payload,device)
                record=dict(source_record,selected_best_step=source_payload["step"],prior_update_count=None,
                    historical_receiver=True,new_optimizer_updates=0)
            else:
                latest=load_payload(torch,entry["latest_path"],hashes)
                best=load_payload(torch,entry["best_path"],hashes)
                sidecars={name:read_json(entry[name+"_path"],hashes) for name in ("config","status")}
                record=verify_training_metadata(method,training_seed,best,latest,sidecars)
                model,cfg=build_model(torch,method,training_seed,best,device)
                validate_model_state(torch,model,latest["state"],step=latest["step"])
                check_tensor_state(torch,latest["state"],best["state"],equal=latest["step"]==best["step"])
                record.update(selected_best_step=best["step"],prior_update_count=best["step"],new_optimizer_updates=20000)
                del latest,best
            model.eval()
            models[method]=(model,cfg);snapshots[method]=model_buffer_snapshot(model)
            validate_frozen(snapshots[method],snapshots[method])
            record.update(best_path=entry["best_path"],latest_path=entry["latest_path"],
                best_sha256=hashes[entry["best_path"]],latest_sha256=hashes[entry["latest_path"]],
                strict_loading=True,model_class=type(model).__name__,architecture_metadata=model.architecture_metadata)
            manifest["methods"][method]=record
            parameters.append(dict(method_key=method,training_seed=training_seed,
                full_parameters=sum(v.numel() for v in model.parameters()),
                ue_parameters=sum(v.numel() for v in model.ue_parameters()),
                bs_parameters=sum(v.numel() for v in model.bs_parameters()),
                note="Actual stored numel; inactive legacy branch included; not MACs or latency"))
        del source_payload
        _compare(hashes,{path:sha256_file(path) for path in hashes},"Inputs changed during loading")
        data=make_test_data().prepare();torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for batch_index in range(200):
                H_dl,H_ul=data.training(batch_index+1)
                noise=physical.standard_noise(TEST_SEED,"test",batch_index,8,device=str(device))
                originals=tuple(t.clone() for t in (H_dl,H_ul,noise))
                manifest["shared_batch_inputs"].append(dict(batch_index=batch_index,sample_index_start=8*batch_index,
                    sample_index_stop=8*(batch_index+1),**{name+"_sha256":tensor_digest({name:value})
                    for name,value in zip(("H_dl","H_ul","standard_noise"),originals)}))
                for gamma in GAMMAS:
                    for method,(model,cfg) in models.items():
                        result=model.forward_with_diagnostics(H_dl,H_ul,float(gamma),standard_noise=noise)
                        W,Hhat,diagnostics=result["precoder"],result["channel_surrogate"],result["diagnostics"]
                        if not torch.isfinite(W).all().item() or not torch.isfinite(Hhat).all().item():
                            raise ValueError("Nonfinite deployment output")
                        if method!="frozen_F":
                            count=int(diagnostics["prior_update_count"])
                            if count!=manifest["methods"][method]["selected_best_step"]:
                                raise ValueError("Evaluation prior count changed or differs from selected weights")
                        values=extract_diagnostics(torch,diagnostics,noise,gamma,method)
                        values.update(sum_rate=ev.per_sample_sum_rate(W,H_dl,10**(-25/10)),
                            nmse_db=ev.per_sample_nmse_db(Hhat,H_dl),
                            precoder_power=ev.per_sample_precoder_power(W),
                            precoder_power_max_error=(W.abs().square().sum((1,2))-1).abs().amax(1),
                            **downlink_diagnostics(torch,W,H_dl))
                        arrays={name:(None if value is None else value.detach().cpu().tolist()) for name,value in values.items()}
                        samples.extend(dict(sample_identity(method,gamma,8*batch_index+i,training_seed),
                            **{name:(None if value is None else value[i]) for name,value in arrays.items()}) for i in range(8))
                        manifest["model_execution"].setdefault(method,diagnostics["execution"])
                if not all(torch.equal(a,b) for a,b in zip(originals,(H_dl,H_ul,noise))):
                    raise ValueError("Shared channel/noise tensors were modified")
        summary,_=summarize_samples(samples,1600,training_seed)
        paired,raw_pairs=paired_rows(summary,samples,1600,training_seed)
        ev.write_csv(output/f"summary_{TAG}.csv",summary);ev.write_gzip_csv(output/f"samples_{TAG}.csv.gz",samples)
        ev.write_csv(output/f"paired_deltas_{TAG}.csv",paired);ev.write_gzip_csv(output/f"paired_samples_{TAG}.csv.gz",raw_pairs)
        ev.write_csv(output/f"parameters_{TAG}.csv",parameters)
        for name,value in (("metadata",p),("results",dict(summary=summary,paired=paired))):
            with (output/f"{name}_{TAG}.json").open("x",encoding="utf-8") as stream:
                json.dump(value,stream,indent=2,allow_nan=False)
        manifest.update(status="completed",results_verified=True,summary_rows=24,paired_rows=32,
            shared_channel_noise_inputs_unchanged=True,samples_hashed_pairing=True,
            memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    except Exception as error:
        failure=error;manifest.update(status="failed",error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
    finally:
        errors,after_hashes,after_buffers=[],{},{}
        for path,digest in hashes.items():
            try:
                after_hashes[path]=sha256_file(path)
                if after_hashes[path]!=digest: errors.append("Changed frozen input: "+path)
            except Exception as error: errors.append(str(error))
        for method,(model,cfg) in models.items():
            try:
                after_buffers[method]=model_buffer_snapshot(model)
                validate_frozen(snapshots[method],after_buffers[method])
                if method!="frozen_F":
                    model.power_prior.validate_state(manifest["methods"][method]["selected_best_step"])
            except Exception as error: errors.append(method+":"+str(error))
        if manifest.get("runtime",{}).get("source_sha256") is not None:
            for name,digest in manifest["runtime"]["source_sha256"].items():
                try:
                    if sha256_file(ROOT/name)!=digest: errors.append("Source changed: "+name)
                except Exception as error: errors.append(str(error))
        manifest.update(input_sha256_before=hashes,input_sha256_after=after_hashes,
            model_buffers_before=snapshots,model_buffers_after=after_buffers,
            freeze_checks=dict(passed=not errors and len(models)==4,errors=errors,loaded_methods=list(models)))
        if errors or failure is None and len(models)!=4:
            failure=failure or RuntimeError("Frozen-state check failed")
            manifest.update(status="failed",error=str(failure))
        if failure is None:
            try: manifest["output_sha256"]=artifact_hashes(output)
            except Exception as error: failure=error;manifest.update(status="failed",error=str(error))
        write_manifest(output,manifest)
    if failure is not None:
        raise RuntimeError(f"Hybrid Anchor evaluation failed; evidence {output/'manifest.json'}") from failure
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run",action="store_true");modes.add_argument("--evaluate",action="store_true")
    parser.add_argument("--seed",type=int,choices=SEEDS);parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    if not (args.dry_run or args.evaluate): parser.print_help();return 0
    if args.seed is None: parser.error("Explicit --seed required")
    if args.dry_run:
        if args.output_dir is not None:
            from evaluate_proposal_lite import validate_output_path
            validate_output_path(args.output_dir,OUTPUT_ROOT)
        print(json.dumps(dict(protocol=protocol(args.seed),checkpoints=checkpoint_plan(args.seed),
            source_verified=False,checkpoint_bytes_not_read=True,models_not_imported=True,outputs_not_created=True),indent=2))
        return 0
    run_evaluation(args.seed,output_dir=args.output_dir)
    return 0


if __name__=="__main__": raise SystemExit(main())
