#!/usr/bin/env python3
"""Six fixed-checkpoint screening comparisons; no models in help/dry-run.

Actual execution requires CUDA. All six methods are freshly evaluated on the
same replayable channel batches; historical CSV means are never concatenated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from configs.swin_adapt_train_config import SEEDS, swin_adapt_config
from configs.as_spatial_train_config import channel_data_api
import evaluate_proposal_as_receiver as shared
import evaluate_proposal_as_spatial as spatial
from evaluate_proposal_as_direct import (
    check_tensor_state, load_payload, read_json, verify_archived_metadata, write_manifest,
)
from evaluate_proposal_lite import _compare, model_buffer_snapshot, validate_frozen, validate_output_path
from preflight_proposal_as_direct import DEFAULT_MANIFEST, sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_swin_adaptation_v1"
TAG = "k4_swin_adaptation_v1"
METHODS = spatial.METHODS + ("swin_adapted",)
COMPARISONS = (("joint", "swin"), ("joint", "swin_adapted"), ("swin_adapted", "swin"),
               ("joint", "full_control"), ("joint", "as_receiver"), ("joint", "frozen_ue"))
PRIMARY_PAIRS = (("joint", "swin_adapted"),)
FB_SNRS, TEST_SEED, PREFLIGHT_SEED = shared.FB_SNRS, 20261110, 20261111
OUTPUT_ARTIFACTS = tuple(f"{name}_{TAG}.{suffix}" for name, suffix in (
    ("summary", "csv"), ("samples", "csv.gz"), ("results", "json"), ("metadata", "json"),
    ("paired_deltas", "csv"), ("paired_samples", "csv.gz"), ("parameters", "csv")))
SOURCE_FILES = tuple(dict.fromkeys(spatial.SOURCE_FILES + (
    "evaluate_swin_adaptation.py", "configs/swin_adapt_train_config.py", "train_swin_adaptation.py", "baseline_train.py")))


def make_test_data(preflight=False):
    return channel_data_api().ReplayableChannelData(
        base_seed=PREFLIGHT_SEED if preflight else TEST_SEED, k_users=4, nt=32, nsc=256,
        batch_size=8, validation_batch_size=8, carrier_freq=3.5e9, speed=1.0,
        cdl_model="C", device="cuda:0")


def protocol(training_seed, preflight=False):
    result = shared.protocol(training_seed, preflight)
    result.update(seed=PREFLIGHT_SEED if preflight else TEST_SEED,
                  channel_data_protocol=make_test_data(preflight).protocol,
                  methods=list(METHODS), screening_not_formal_result=True)
    return result


def checkpoint_plan(training_seed):
    shared.validate_seed(training_seed)
    entries = spatial.checkpoint_plan(training_seed)
    cfg = swin_adapt_config(training_seed)
    entries.append(dict(method_key="swin_adapted", model_key="swin", training_seed=training_seed,
                        path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH, config_path=cfg.CONFIG_PATH,
                        status_path=cfg.STATUS_PATH, allocations=[[64, 0]],
                        label="Swin-based FDMA +20k adaptation"))
    if tuple(item["method_key"] for item in entries) != METHODS or len({item["path"] for item in entries}) != 6:
        raise ValueError("Six fixed, distinct checkpoint paths required")
    return entries


def verify_adapted_metadata(seed, best, latest, sidecars):
    from train_swin_adaptation import fingerprint, validate_resume_payload, validate_best_metadata
    cfg = swin_adapt_config(seed)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    _compare(20000, latest.get("step"), "Swin adapted completion")
    _compare(cfg.to_dict(), sidecars["config"], "Swin adapted sidecar config")
    if type(best.get("step")) is not int or not 0 <= best["step"] <= 20000 or best["step"] % 500:
        raise ValueError("Swin selected best must be the validated source (step0) or a 500-step validation")
    status = sidecars["status"]
    for field, expected in dict(state="completed", step=20000, experiment=cfg.EXP_NAME,
                                total_steps=20000, best_score=best["best_score"],
                                selected_best_step=best["step"], best_selection_name="mean_all").items():
        _compare(expected, status.get(field), f"Swin adapted status.{field}")
    return dict(config=best["config"], config_fingerprint=fingerprint(cfg),
                selected_best_step=best["step"], latest_step=latest["step"], completed_updates=20000,
                selected_source_step0=best["step"] == 0, best_metric=best["best_score"],
                selection_metric="mean_all", training_status=status, training_seed=seed,
                source_identity=cfg.source_identity, validation=best.get("validation"))


def build_method(torch, ev, scenario, entry, payload, device):
    if entry["method_key"] != "swin_adapted":
        return spatial.build_method(torch, ev, scenario, entry, payload, device)
    from models.swin_precoder import Swin_E2E_Precoder
    cfg = swin_adapt_config(entry["training_seed"])
    _compare(cfg.to_dict(), payload["config"], "Adapted Swin saved complete configuration")
    for name, expected in dict(K_USERS=4, ANTENNAS=32, SUBCARRIERS=256, D_TOT=256,
                                D_F=64, RZF_REG=1e-3, SHARE_USER_WEIGHTS=True).items():
        _compare(expected, getattr(cfg, name), f"Swin runtime.{name}")
    model = Swin_E2E_Precoder(cfg)
    check_tensor_state(torch, model.state_dict(), payload["state"])
    model.load_state_dict(payload["state"], strict=True)
    check_tensor_state(torch, model.state_dict(), payload["state"], equal=True)
    model.to(device).eval()
    return ev.LoadedMethod(model_key="swin", method_key="swin_adapted", display_name=entry["label"],
                           model=model, checkpoint_path=entry["path"], checkpoint_step=payload["step"],
                           checkpoint_kind="best", allocations=((64, 0),), config=cfg, rzf_lambda=1e-3)


def paired_rows(summary, samples, plan, count, seed, training_seed):
    return shared.paired_rows(summary, samples, plan, count, seed, training_seed,
                              method_keys=METHODS, comparisons=COMPARISONS, primary_pairs=PRIMARY_PAIRS)


def artifact_hashes(output):
    return shared.artifact_hashes(output, expected_artifacts=OUTPUT_ARTIFACTS)


def runtime_record(torch=None):
    return shared.runtime_record(torch, source_files=SOURCE_FILES)


def run_evaluation(training_seed, *, preflight=False, output_dir=None):
    shared.validate_seed(training_seed)
    output = shared.create_output(training_seed, preflight=preflight, output_dir=output_dir, root=OUTPUT_ROOT)
    p = protocol(training_seed, preflight)
    manifest = dict(status="pending", protocol=p, output_directory=str(output), methods={},
                    primary="joint-swin_adapted within-channel FB0/5/10 average; FB25 degradation check",
                    statistics="Signed ddof1 paired sample and sample_index//8 batch-mean 95%CI; no seed pooling",
                    ci_scope="Sampling uncertainty conditional on fixed checkpoints, not training-seed stability",
                    selection="Best from fixed validation protocol, including Swin source step0; test never selects checkpoints",
                    comparison_limit="Additional-20k matched adaptation stage, not equal total historical training or equal parameters",
                    physical_limit="Same feedback RNG is not equal absolute noise power or equal total UE energy",
                    power="precoder_power is full-band total; nominal256, not per-subcarrier1",
                    nmse="Raw representation diagnostic only, not a model ranking or selection criterion")
    hashes, snapshots, methods = {}, {}, []
    summary = samples = None
    failure = None
    try:
        manifest["runtime"] = runtime_record()
        manifest["plan"] = plan = checkpoint_plan(training_seed)
        hashes[str(DEFAULT_MANIFEST)] = sha256_file(DEFAULT_MANIFEST)
        import torch
        manifest["runtime"] = runtime_record(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback forbidden")
        import baseline_evaluate as ev
        scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
        scenario.validate()
        args = shared.evaluation_args(ev, output, p, tag=TAG)
        device = torch.device("cuda:0")
        ev.set_seed(p["seed"])
        parameters = []
        for entry in plan:
            key = entry["method_key"]
            if key == "swin":
                best = load_payload(torch, entry["path"], hashes, entry["expected_best_sha256"])
                record = verify_archived_metadata(entry, best)
            else:
                latest = load_payload(torch, entry["latest_path"], hashes)
                best = load_payload(torch, entry["path"], hashes)
                sidecars = {name: read_json(entry[name + "_path"], hashes) for name in ("config", "status")}
                record = (verify_adapted_metadata(training_seed, best, latest, sidecars)
                          if key == "swin_adapted" else spatial.verify_arm_metadata(entry["arm"], training_seed, best, latest, sidecars))
                check_tensor_state(torch, latest["state"], best["state"])
                if latest["step"] == best["step"]:
                    check_tensor_state(torch, latest["state"], best["state"], equal=True)
                record.update(latest_path=entry["latest_path"], latest_sha256=hashes[entry["latest_path"]],
                              best_latest_same_step=latest["step"] == best["step"])
                del latest
            method = build_method(torch, ev, scenario, entry, best, device)
            if key == "swin_adapted" and best["step"] == 0:
                source = next(item for item in methods if item.method_key == "swin")
                check_tensor_state(torch, source.model.state_dict(), method.model.state_dict(), equal=True)
                record["step0_model_equals_pinned_source"] = True
            methods.append(method)
            snapshots[key] = model_buffer_snapshot(method.model)
            validate_frozen(snapshots[key], snapshots[key])
            record.update(model_class=type(method.model).__name__, method_key=key, model_key=method.model_key,
                          best_path=entry["path"], best_sha256=hashes[entry["path"]], strict_loading=True,
                          allocation=entry["allocations"][0], rzf_regularization=1e-3)
            if key in spatial.ARMS:
                record.update(loaded_architecture_metadata=method.model.architecture_metadata,
                              ue_frozen=method.model.ue_frozen)
            ue_count = sum(x.numel() for x in method.model.encoder.parameters())
            if method.model_key == "proposed":
                ue_count += sum(x.numel() for x in method.model.cond_embed.parameters())
            parameters.append(dict(method_key=key, training_seed=training_seed,
                                   full_parameters=sum(x.numel() for x in method.model.parameters()),
                                   encoder_plus_condition_parameters=ue_count,
                                   counting_scope="Swin encoder; proposal encoder+condition; stored full model",
                                   note="Not equal parameters, MACs or measured latency"))
            manifest["methods"][key] = record
            del best
        _compare(hashes, {path: sha256_file(path) for path in hashes}, "Inputs changed during loading")
        ev.write_csv(output / f"parameters_{TAG}.csv", parameters)
        points = [(float(fb), 25.0) for fb in FB_SNRS]
        ev.set_seed(p["seed"])
        with torch.no_grad():
            summary, samples = ev.evaluate_scenario(scenario, methods, points, args, device,
                                                    channel_data=make_test_data(preflight))
        deltas, paired = paired_rows(summary, samples, plan, p["num_samples"], p["seed"], training_seed)
        for row in summary + samples + deltas + paired:
            row.update(training_seed=training_seed, seed=p["seed"], preflight=preflight, dataset_role=p["dataset_role"])
        ev.save_outputs(summary, samples, [scenario], list(METHODS), points, args)
        ev.write_csv(output / f"paired_deltas_{TAG}.csv", deltas)
        ev.write_gzip_csv(output / f"paired_samples_{TAG}.csv.gz", paired)
        manifest.update(status="completed", summary_rows=len(summary), sample_rows=len(samples),
                        paired_rows=len(deltas), results_verified=True)
    except Exception as exc:
        failure = exc
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if summary is not None:
            try:
                ev.save_outputs(summary, samples, [scenario], list(METHODS), points, args)
            except Exception as save_error:
                manifest["failure_data_save_error"] = str(save_error)
    finally:
        errors, after_hashes, after_buffers = [], {}, {}
        for path, expected in hashes.items():
            try:
                after_hashes[path] = sha256_file(path)
                if after_hashes[path] != expected:
                    errors.append(f"Input changed: {path}")
            except Exception as exc:
                errors.append(f"Cannot recheck {path}: {exc}")
        for method in methods:
            try:
                key = method.method_key
                after_buffers[key] = model_buffer_snapshot(method.model)
                validate_frozen(snapshots[key], after_buffers[key])
                if key in spatial.ARMS:
                    _compare(key == "frozen_ue", method.model.ue_frozen, "UE freeze mode changed")
                elif key == "as_receiver":
                    _compare(True, method.model.receiver_enabled, "ASReceiver disabled")
            except Exception as exc:
                errors.append(f"Frozen check {method.method_key}: {exc}")
        manifest.update(input_sha256_before=hashes, input_sha256_after=after_hashes,
                        model_buffers_before=snapshots, model_buffers_after=after_buffers,
                        freeze_checks=dict(passed=not errors and len(methods) == 6, errors=errors))
        if errors or (failure is None and len(methods) != 6):
            failure = failure or RuntimeError("Frozen/input checks failed: " + "; ".join(errors))
            manifest.update(status="failed", error=str(failure))
        if failure is None and manifest["status"] == "completed":
            try:
                manifest["output_sha256"] = artifact_hashes(output)
            except Exception as exc:
                failure = exc
                manifest.update(status="failed", error=f"Artifact evidence failed: {exc}")
        write_manifest(output, manifest)
    if failure is not None:
        raise RuntimeError(f"Swin adaptation evaluation failed; evidence: {output / 'manifest.json'}") from failure
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for option in ("dry-run", "preflight", "evaluate"):
        modes.add_argument("--" + option, action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not (args.dry_run or args.preflight or args.evaluate):
        parser.print_help()
        return 0
    if args.seed is None:
        parser.error("Explicit training --seed required; test seed is fixed independently")
    if args.dry_run:
        import json
        if args.output_dir is not None:
            validate_output_path(args.output_dir.absolute(), OUTPUT_ROOT)
        print(json.dumps(dict(plan=checkpoint_plan(args.seed), evaluation=protocol(args.seed),
                              preflight=protocol(args.seed, True), output_root=str(OUTPUT_ROOT),
                              models_not_imported=True, checkpoint_bytes_not_read=True,
                              training_seeds_pooled=False, best_step0_allowed=True), indent=2))
        return 0
    run_evaluation(args.seed, preflight=args.preflight, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
