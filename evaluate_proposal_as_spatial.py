#!/usr/bin/env python3
"""Fixed AS-spatial screening evaluation with same-forward offline diagnostics.

No models, weights or channels are used by help/--dry-run. Actual execution
requires CUDA and an explicit --preflight or --evaluate. Existing controls are
strictly loaded, never retrained. True codewords never enter the decoder.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import traceback

from configs.as_spatial_train_config import ARMS, SEEDS, as_spatial_config, channel_data_api
import evaluate_proposal_as_receiver as shared
from evaluate_proposal_as_direct import (archived_swin, check_tensor_state, load_payload, read_json,
                                         sample_statistics, verify_archived_metadata, write_manifest)
from evaluate_proposal_lite import _compare, model_buffer_snapshot, validate_frozen, validate_output_path
from preflight_proposal_as_direct import DEFAULT_MANIFEST, _backend_info, sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_as_spatial_v1"
TAG = "k4_as_spatial_v1"
METHODS = ("frozen_ue", "joint", "full_control", "as_receiver", "swin")
COMPARISONS = tuple((arm, reference) for arm in ARMS for reference in ("full_control", "as_receiver", "swin")) + (("joint", "frozen_ue"),)
PRIMARY_PAIRS = tuple((arm, "full_control") for arm in ARMS)
FB_SNRS, TEST_SEED, PREFLIGHT_SEED = shared.FB_SNRS, 20261103, 20261104
SOFT_METRICS = ("soft_mse", "soft_nmse_linear", "soft_nmse_db", "reliability_mean")
OUTPUT_ARTIFACTS = tuple(f"{prefix}_{TAG}.{suffix}" for prefix,suffix in (
    ("summary","csv"),("samples","csv.gz"),("results","json"),("metadata","json"),
    ("paired_deltas","csv"),("paired_samples","csv.gz"),("parameters","csv"),
    ("spatial_parameters","csv"),("soft_codeword_summary","csv"),("soft_codeword_samples","csv.gz")))
SOURCE_FILES = tuple(dict.fromkeys(shared.SOURCE_FILES + (
    "evaluate_proposal_as_spatial.py", "models/adaptive_hybrid_as_spatial.py",
    "configs/as_spatial_train_config.py", "train_proposal_as_spatial.py", "utils/metrics.py")))


def make_test_data(preflight=False):
    return channel_data_api().ReplayableChannelData(
        base_seed=PREFLIGHT_SEED if preflight else TEST_SEED, k_users=4, nt=32, nsc=256,
        batch_size=8, validation_batch_size=8, carrier_freq=3.5e9, speed=1.0, cdl_model="C", device="cuda:0")


def protocol(training_seed, preflight=False):
    p = shared.protocol(training_seed, preflight)
    p.update(seed=PREFLIGHT_SEED if preflight else TEST_SEED,
             channel_data_protocol=make_test_data(preflight).protocol,
             diagnostic_scope="Same forward; detached true/recovered codewords are offline diagnostics only")
    return p


def checkpoint_plan(training_seed):
    shared.validate_seed(training_seed)
    entries = []
    for arm in ARMS:
        cfg = as_spatial_config(arm, training_seed)
        entries.append(dict(method_key=arm, model_key="proposed", arm=arm, training_seed=training_seed,
                            path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH, config_path=cfg.CONFIG_PATH,
                            status_path=cfg.STATUS_PATH, allocations=[[0,256]],
                            label="ASSpatial frozen UE" if arm=="frozen_ue" else "ASSpatial joint"))
    # Reuse the precise existing, same-seed control paths; exclude frozen robust/off.
    entries.extend(item for item in shared.checkpoint_plan(training_seed)
                   if item["method_key"] in ("full_control","as_receiver","swin"))
    if tuple(item["method_key"] for item in entries) != METHODS or len({item["path"] for item in entries}) != 5:
        raise ValueError("Expected five distinct fixed checkpoints; no alternate-weight search")
    return entries


def verify_arm_metadata(arm, seed, best, latest, sidecars):
    if arm not in ARMS:
        return shared.verify_arm_metadata(arm, seed, best, latest, sidecars)
    from train_proposal_as_spatial import validate_resume_payload, validate_best_metadata
    cfg = as_spatial_config(arm, seed)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    _compare(20000, latest["step"], "ASSpatial completed updates")
    _compare(dict(cfg.to_dict(),config_fingerprint=shared.fingerprint(cfg)), sidecars["config"], "ASSpatial sidecar config")
    for key,value in dict(state="completed",step=20000,experiment=cfg.EXP_NAME,
                          best_metric_name="mean_all",best_metric=best["best_metric"]).items():
        _compare(value, sidecars["status"].get(key), f"ASSpatial status.{key}")
    return dict(config=best["config"],config_fingerprint=best["config_fingerprint"],
                selected_best_step=best["step"],latest_step=latest["step"],completed_updates=20000,
                best_metric=best["best_metric"],training_status=sidecars["status"],training_seed=seed,
                architecture_metadata=cfg.architecture_metadata)


def validate_loaded_architecture(saved, actual):
    """Reject equal-shaped but differently described decoders/physical settings."""
    for name in ("architecture", "model_class", "arm", "ue_frozen", "ue_ffn_dim", "D_MODEL",
                 "max_dimensions", "allocation_grid", "running_pwr_shape", "cfg_idx",
                 "decoder", "receiver", "frozen_ue_power", "joint_ue_power",
                 "rzf_regularization", "power_normalization"):
        if name not in saved or name not in actual:
            raise ValueError(f"Missing ASSpatial architecture field: {name}")
        _compare(saved[name],actual[name],f"Constructed ASSpatial architecture.{name}")
    for name in ("lambda_floor", "solve_dtype"):
        _compare(saved[name],actual["receiver"][name],f"Constructed ASSpatial receiver.{name}")


def build_method(torch, ev, scenario, entry, payload, device):
    if entry["method_key"] not in ARMS:
        return shared.build_method(torch, ev, scenario, entry, payload, device)
    from models.adaptive_hybrid_as_spatial import AdaptiveHybridPrecoderASSpatial
    ev.validate_checkpoint_scenario(payload, Path(entry["path"]), scenario)
    cfg = ev.proposal_runtime_config(scenario, payload)
    metadata = payload["config"]["architecture_metadata"]
    for name,value in dict(architecture="as_spatial_v1",arm=entry["arm"],max_dimensions=[64,256],
                           allocation_grid=[[0,256]],ue_ffn_dim=2048).items():
        _compare(value,metadata.get(name),f"ASSpatial architecture.{name}")
    cfg.arm = entry["arm"]
    cfg.AS_SPATIAL_LAMBDA_FLOOR = metadata["lambda_floor"]
    cfg.AS_SPATIAL_SOLVE_DTYPE = metadata["solve_dtype"]
    model = AdaptiveHybridPrecoderASSpatial(cfg)
    validate_loaded_architecture(metadata, model.architecture_metadata)
    check_tensor_state(torch, model.state_dict(), payload["state"])
    model.load_state_dict(payload["state"], strict=True)  # Complete trained state; no UE initialization.
    check_tensor_state(torch, model.state_dict(), payload["state"], equal=True)
    _compare([(0,256)], cfg.GAMMA_GRID, "ASSpatial supported grid")
    _compare((64,256),(cfg.D_F_MAX_PER_USER,cfg.D_A_MAX_SHARED),"ASSpatial max architecture")
    _compare((1,),tuple(model.running_pwr.shape),"ASSpatial running power")
    if not (model.running_pwr>0).all().item():
        raise ValueError("Nonpositive running power")
    for branch in ("fdma_backbone","aircomp_backbone"):
        _compare([2048]*4,[layer.linear1.out_features for layer in getattr(model.encoder,branch).layers],branch)
    model.to(device).eval()
    return ev.LoadedMethod(model_key="proposed",method_key=entry["method_key"],display_name=entry["label"],
                           model=model,checkpoint_path=entry["path"],checkpoint_step=payload["step"],
                           checkpoint_kind="best",allocations=((0,256),),config=cfg,rzf_lambda=1e-3)


def paired_rows(summary, samples, plan, count, seed, training_seed):
    return shared.paired_rows(summary,samples,plan,count,seed,training_seed,
                              method_keys=METHODS,comparisons=COMPARISONS,primary_pairs=PRIMARY_PAIRS)


def summarize_soft_rows(rows, count, seed, training_seed):
    """One soft-codeword error per complete multi-user sample, not per symbol."""
    expected = {(arm,fb) for arm in ARMS for fb in FB_SNRS}
    indexed = {key:{} for key in expected}
    if type(count) is not int or count<8 or count%8:
        raise ValueError("Soft diagnostics require complete batches of eight")
    for row in rows:
        key = row.get("method_key"),row.get("feedback_snr_db")
        i = row.get("sample_index")
        if key not in expected or type(i) is not int or not 0<=i<count or i in indexed[key]:
            raise ValueError("Missing/duplicate/invalid soft-codeword sample key")
        for name,value in dict(seed=seed,training_seed=training_seed,downlink_snr_db=25,
                               batch_index=i//8,allocation_df=0,allocation_da=256).items():
            _compare(value,row.get(name),f"soft diagnostic.{name}")
        if any(not math.isfinite(float(row[name])) for name in SOFT_METRICS+("noise_est","regularization")):
            raise ValueError("Nonfinite soft-codeword diagnostic")
        if row["soft_mse"]<0 or row["soft_nmse_linear"]<0 or not 0<=row["reliability_mean"]<=1:
            raise ValueError("Invalid soft-codeword error/reliability range")
        indexed[key][i]=row
    output=[]
    for (arm,fb),samples in indexed.items():
        if set(samples)!=set(range(count)):
            raise ValueError("Incomplete soft-codeword diagnostic coverage")
        row=dict(method_key=arm,feedback_snr_db=fb,downlink_snr_db=25,training_seed=training_seed,seed=seed,
                 num_samples=count,num_batches=count//8,batch_size=8,batch_ci_estimable=count>8,
                 unit="one complete multi-user channel; all users/symbols reduced within sample")
        for metric in SOFT_METRICS:
            values=[samples[i][metric] for i in range(count)]
            batches=[statistics.fmean(values[i:i+8]) for i in range(0,count,8)]
            row.update({metric+"_"+key:value for key,value in sample_statistics(values).items()})
            row.update({"batch_"+metric+"_"+key:value for key,value in sample_statistics(batches).items()})
        for metric in ("noise_est","regularization"):
            for start in range(0,count,8):
                if any(samples[i][metric]!=samples[start][metric] for i in range(start,start+8)):
                    raise ValueError("Batch-shared diagnostic scalar changed within batch")
            values=[samples[i][metric] for i in range(0,count,8)]
            row.update({metric+"_batch_mean":statistics.fmean(values),metric+"_batch_min":min(values),metric+"_batch_max":max(values)})
        output.append(row)
    return output


class SoftCodewordCollector:
    """Read-only reductions from a completed forward; no model hooks or RNG draws."""
    def __init__(self, torch, seed, training_seed):
        self.torch,self.seed,self.training_seed=torch,seed,training_seed
        self.rows=[]
        self.rng_checks=dict(calls=0,passed=True)
        self.rng=channel_data_api()._load_source("utils/sionna_rng.py","_precoder_sionna_rng_isolation_v1")

    def __call__(self, context, payload):
        torch=self.torch
        before=self.rng.snapshot_rng_state().fingerprint()
        try:
            key=context["method_key"]
            if key not in ARMS:
                raise ValueError("Unexpected model supplied soft-codeword diagnostics")
            for name in ("feedback_snr_db","allocation_df","allocation_da"):
                _compare(context[name],payload.get(name),f"soft payload.{name}")
            if context["downlink_snr_db"]!=25 or (context["allocation_df"],context["allocation_da"])!=(0,256):
                raise ValueError("Unexpected soft diagnostic condition")
            for name in ("z_hat","true_z_a","reliability","noise_est","regularization"):
                value=payload[name]
                if not torch.is_tensor(value) or value.requires_grad or not torch.isfinite(value).all().item():
                    raise ValueError(f"Expected detached finite diagnostic tensor: {name}")
            z,true,rel=payload["z_hat"].permute(0,2,1),payload["true_z_a"],payload["reliability"].permute(0,2,1)
            if tuple(z.shape)!=(8,4,256) or true.shape!=z.shape or rel.shape!=z.shape:
                raise ValueError("Soft codeword axes must resolve to [8,4,256]")
            if not z.is_complex() or not true.is_complex() or rel.is_complex():
                raise ValueError("Unexpected soft-codeword/reliability dtype")
            mse=(z-true).abs().square().mean(dim=(1,2))
            nmse=mse/true.abs().square().mean(dim=(1,2)).clamp_min(1e-12)
            values=dict(soft_mse=mse,soft_nmse_linear=nmse,soft_nmse_db=10*torch.log10(nmse.clamp_min(1e-12)),
                        reliability_mean=rel.mean(dim=(1,2)))
            arrays={name:value.detach().cpu().tolist() for name,value in values.items()}
            noise=float(payload["noise_est"].item()); regularization=float(payload["regularization"].item())
            for i in range(8):
                self.rows.append(dict(context,sample_index=context["batch_index"]*8+i,seed=self.seed,
                                      training_seed=self.training_seed,noise_est=noise,regularization=regularization,
                                      **{name:value[i] for name,value in arrays.items()}))
        finally:
            after=self.rng.snapshot_rng_state().fingerprint()
            self.rng_checks["calls"]+=1
            if before!=after:
                self.rng_checks.update(passed=False,first_failure_context=dict(context),before=before,after=after)
                raise RuntimeError("Offline soft-codeword collector changed external RNG")


def artifact_hashes(output):
    return shared.artifact_hashes(output,expected_artifacts=OUTPUT_ARTIFACTS)


def runtime_record(torch=None):
    return shared.runtime_record(torch,source_files=SOURCE_FILES)


def run_evaluation(training_seed, *, preflight=False, output_dir=None):
    shared.validate_seed(training_seed)
    output=shared.create_output(training_seed,preflight=preflight,output_dir=output_dir,root=OUTPUT_ROOT)
    p=protocol(training_seed,preflight)
    manifest=dict(status="pending",protocol=p,output_directory=str(output),methods={},
                  primary="Each new arm-Full within-sample FB0/5/10 mean; FB25 degradation check",
                  statistics="Separate training seeds; ddof1 sample and batch-mean paired95%CI, no pooling/oracle",
                  diagnostic="True/recovered codewords detached after deployed outputs; no truth is passed to decoder; no extra forward",
                  soft_nmse="Per complete channel mean(|z_hat-Z_a|^2)/max(mean(|Z_a|^2),1e-12); dB floor1e-12",
                  power="precoder_power is full-band total, nominal256",nmse="Raw H representation NMSE secondary only",
                  comparison_limit="Neither equal parameters nor equal Swin training budget; shared RNG is not equal absolute noise")
    hashes,snapshots,methods={}, {}, []
    summary=samples=collector=None
    failure=None
    try:
        manifest["runtime"]=runtime_record()
        manifest["plan"]=plan=checkpoint_plan(training_seed)
        hashes[str(DEFAULT_MANIFEST)]=sha256_file(DEFAULT_MANIFEST)
        import torch
        manifest["runtime"]["backend"]=_backend_info(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback forbidden")
        import baseline_evaluate as ev
        manifest["runtime"]=runtime_record(torch)
        scenario=ev.Scenario(k_users=4,subcarriers=256,feedback_budget=256); scenario.validate()
        args=shared.evaluation_args(ev,output,p,tag=TAG)
        device=torch.device("cuda:0")
        ev.set_seed(p["seed"])
        parameters,spatial_parameters=[],[]
        for entry in plan:
            key=entry["method_key"]
            if key!="swin":
                latest=load_payload(torch,entry["latest_path"],hashes)
                best=load_payload(torch,entry["path"],hashes)
                sidecars={name:read_json(entry[name+"_path"],hashes) for name in ("config","status")}
                record=verify_arm_metadata(entry["arm"],training_seed,best,latest,sidecars)
                check_tensor_state(torch,latest["state"],best["state"])
                same_step=latest["step"]==best["step"]
                if same_step:
                    check_tensor_state(torch,latest["state"],best["state"],equal=True)
                record.update(latest_path=entry["latest_path"],latest_sha256=hashes[entry["latest_path"]],best_latest_same_step=same_step)
                del latest
            else:
                best=load_payload(torch,entry["path"],hashes,entry["expected_best_sha256"])
                record=verify_archived_metadata(entry,best)
            method=build_method(torch,ev,scenario,entry,best,device)
            methods.append(method)
            snapshots[key]=model_buffer_snapshot(method.model)
            validate_frozen(snapshots[key],snapshots[key])
            record.update(model_class=type(method.model).__name__,method_key=key,model_key=method.model_key,
                          best_path=entry["path"],best_sha256=hashes[entry["path"]],strict_loading=True,
                          allocation=entry["allocations"][0],rzf_regularization=1e-3)
            ue_count=None if key=="swin" else sum(x.numel() for x in method.model.encoder.parameters())+sum(x.numel() for x in method.model.cond_embed.parameters())
            parameters.append(dict(method_key=key,training_seed=training_seed,full_parameters=sum(x.numel() for x in method.model.parameters()),
                                   encoder_plus_condition_parameters=ue_count,note="Stored parameters, not equal-parameter or latency comparison"))
            if key in ARMS:
                record.update(loaded_architecture_metadata=method.model.architecture_metadata,ue_frozen=method.model.ue_frozen)
                _compare(key=="frozen_ue",method.model.ue_frozen,"loaded UE freeze mode")
                for name,value in method.model.named_parameters():
                    if name.startswith("decoder."):
                        spatial_parameters.append(dict(method_key=key,name=name,numel=value.numel(),l2_norm=float(value.detach().float().norm().item())))
            manifest["methods"][key]=record
            del best
        _compare(hashes,{path:sha256_file(path) for path in hashes},"Inputs changed during loading")
        ev.write_csv(output/f"parameters_{TAG}.csv",parameters)
        ev.write_csv(output/f"spatial_parameters_{TAG}.csv",spatial_parameters)
        collector=SoftCodewordCollector(torch,p["seed"],training_seed)
        points=[(float(fb),25.0) for fb in FB_SNRS]
        ev.set_seed(p["seed"])
        with torch.no_grad():
            summary,samples=ev.evaluate_scenario(scenario,methods,points,args,device,
                                                 channel_data=make_test_data(preflight),diagnostic_sink=collector)
        deltas,paired=paired_rows(summary,samples,plan,p["num_samples"],p["seed"],training_seed)
        soft_summary=summarize_soft_rows(collector.rows,p["num_samples"],p["seed"],training_seed)
        for row in summary+samples+deltas+paired+soft_summary+collector.rows:
            row.update(training_seed=training_seed,seed=p["seed"],preflight=preflight,dataset_role=p["dataset_role"])
        ev.save_outputs(summary,samples,[scenario],list(METHODS),points,args)
        ev.write_csv(output/f"paired_deltas_{TAG}.csv",deltas)
        ev.write_gzip_csv(output/f"paired_samples_{TAG}.csv.gz",paired)
        ev.write_csv(output/f"soft_codeword_summary_{TAG}.csv",soft_summary)
        ev.write_gzip_csv(output/f"soft_codeword_samples_{TAG}.csv.gz",collector.rows)
        manifest.update(status="completed",summary_rows=len(summary),sample_rows=len(samples),paired_rows=len(deltas),
                        soft_rows=len(collector.rows),soft_summary_rows=len(soft_summary),results_verified=True)
    except Exception as exc:
        failure=exc
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        if summary is not None:
            try:
                ev.save_outputs(summary,samples,[scenario],list(METHODS),points,args)
            except Exception as save_error:
                manifest["failure_data_save_error"]=str(save_error)
        if collector is not None and collector.rows:
            try:
                ev.write_gzip_csv(output/f"soft_codeword_partial_{TAG}.csv.gz",collector.rows)
            except Exception as save_error:
                manifest["failure_soft_save_error"]=str(save_error)
    finally:
        errors,after_hashes,after_buffers=[],{},{}
        for path,expected in hashes.items():
            try:
                after_hashes[path]=sha256_file(path)
                if after_hashes[path]!=expected:
                    errors.append(f"Input changed: {path}")
            except Exception as exc:
                errors.append(f"Cannot recheck {path}: {exc}")
        for method in methods:
            try:
                key=method.method_key
                after_buffers[key]=model_buffer_snapshot(method.model)
                validate_frozen(snapshots[key],after_buffers[key])
                if key in ARMS:
                    _compare(key=="frozen_ue",method.model.ue_frozen,"UE freeze mode changed")
                elif key=="as_receiver":
                    _compare(True,method.model.receiver_enabled,"ASReceiver unexpectedly disabled")
            except Exception as exc:
                errors.append(f"Frozen check {method.method_key}: {exc}")
        if collector is not None:
            manifest["diagnostic_rng_checks"]=collector.rng_checks
            if not collector.rng_checks["passed"]:
                errors.append("Offline diagnostic callback changed RNG")
        manifest.update(input_sha256_before=hashes,input_sha256_after=after_hashes,
                        model_buffers_before=snapshots,model_buffers_after=after_buffers,
                        freeze_checks=dict(passed=not errors and len(methods)==5,errors=errors))
        if errors or (failure is None and len(methods)!=5):
            failure=failure or RuntimeError("Frozen/input checks failed: "+"; ".join(errors))
            manifest.update(status="failed",error=str(failure))
        if failure is None and manifest["status"]=="completed":
            try:
                manifest["output_sha256"]=artifact_hashes(output)
            except Exception as exc:
                failure=exc
                manifest.update(status="failed",error=f"Artifact evidence failed: {exc}")
        write_manifest(output,manifest)
    if failure is not None:
        raise RuntimeError(f"ASSpatial evaluation failed; evidence: {output/'manifest.json'}") from failure
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run",action="store_true"); mode.add_argument("--preflight",action="store_true"); mode.add_argument("--evaluate",action="store_true")
    parser.add_argument("--seed",type=int,choices=SEEDS); parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    if not (args.dry_run or args.preflight or args.evaluate):
        parser.print_help(); return 0
    if args.seed is None:
        parser.error("Explicit training --seed required; test seed is fixed independently")
    if args.dry_run:
        import json
        if args.output_dir is not None:
            validate_output_path(args.output_dir.absolute(),OUTPUT_ROOT)
        print(json.dumps(dict(plan=checkpoint_plan(args.seed),evaluation=protocol(args.seed),preflight=protocol(args.seed,True),
                              output_root=str(OUTPUT_ROOT),models_not_imported=True,checkpoint_bytes_not_read=True,
                              training_seeds_pooled=False,offline_truth_not_decoder_input=True),indent=2))
        return 0
    run_evaluation(args.seed,preflight=args.preflight,output_dir=args.output_dir)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
