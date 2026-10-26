#!/usr/bin/env python3
"""Independent two-seed ASReceiver screening: fixed models and replayable test data.

Default help / --dry-run are standard-library only. Actual model evaluation
requires explicit CUDA --preflight or --evaluate. Training seeds remain separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import traceback
import uuid

from configs.as_receiver_train_config import SEEDS, as_receiver_config, channel_data_api
from evaluate_proposal_as_direct import (
    archived_swin, check_tensor_state, load_payload, read_json, sample_statistics,
    verify_archived_metadata, write_manifest,
)
from evaluate_proposal_lite import (
    _compare, check_proposal_model, model_buffer_snapshot, validate_frozen,
    validate_output_path,
)
from preflight_proposal_as_direct import (
    DEFAULT_MANIFEST, _backend_info, load_source_record, sha256_file,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_as_receiver_v1"
TAG = "k4_as_receiver_v1"
METHODS = ("full_control", "as_receiver", "robust_frozen", "swin", "as_receiver_off")
COMPARISONS = (("as_receiver", "full_control"), ("full_control", "robust_frozen"),
               ("as_receiver", "robust_frozen"), ("as_receiver", "swin"),
               ("as_receiver", "as_receiver_off"))
FB_SNRS, DL_SNR, BATCH_SIZE = (0, 5, 10, 15, 20, 25), 25, 8
TEST_SEED, PREFLIGHT_SEED = 20261101, 20261102
SCENARIO_ID = "K4_Nsc256_Dfb256"
SOURCE_FILES = (
    "evaluate_proposal_as_receiver.py", "baseline_evaluate.py", "evaluate_proposal_as_direct.py",
    "evaluate_proposal_lite.py", "train_proposal_as_receiver.py", "preflight_proposal_as_direct.py",
    "configs/as_receiver_train_config.py", "configs/train_config.py", "configs/baseline_config.py",
    "models/adaptive_hybrid.py", "models/adaptive_hybrid_as_receiver.py",
    "models/swin_precoder.py", "models/baseline_common.py", "utils/layers.py",
    "utils/channel_generator.py", "utils/replayable_channel_data.py", "utils/sionna_rng.py",
)
OUTPUT_ARTIFACTS = (
    f"summary_{TAG}.csv", f"samples_{TAG}.csv.gz", f"results_{TAG}.json",
    f"metadata_{TAG}.json", f"paired_deltas_{TAG}.csv", f"paired_samples_{TAG}.csv.gz",
    f"parameters_{TAG}.csv", f"receiver_parameters_{TAG}.csv",
)


def validate_seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"Training seed must be one of {SEEDS}; test seed is fixed separately")
    return seed


def make_test_data(preflight=False):
    return channel_data_api().ReplayableChannelData(
        base_seed=PREFLIGHT_SEED if preflight else TEST_SEED, k_users=4, nt=32, nsc=256,
        batch_size=8, validation_batch_size=8, carrier_freq=3.5e9, speed=1.0,
        cdl_model="C", device="cuda:0",
    )


def protocol(training_seed, preflight=False):
    validate_seed(training_seed)
    return dict(training_seed=training_seed, seed=PREFLIGHT_SEED if preflight else TEST_SEED,
                preflight=preflight, batch_size=8, num_batches=1 if preflight else 200,
                num_samples=8 if preflight else 1600, feedback_snr_db=list(FB_SNRS),
                downlink_snr_db=25, channel_profile="CDL-C",
                scenario=dict(K=4, Nt=32, Nsc=256, Dtot=256, Nsb=4),
                dataset_role="preflight_not_screening_result" if preflight else "independent_screening_test",
                channel_data_protocol=make_test_data(preflight).protocol,
                channel_address="training(batch_index+1); identical data protocol across training seeds")


def checkpoint_plan(training_seed):
    validate_seed(training_seed)
    entries = []
    for arm in ("full_control", "as_receiver"):
        cfg = as_receiver_config(arm, training_seed)
        entries.append(dict(method_key=arm, model_key="proposed", arm=arm,
                            path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
                            config_path=cfg.CONFIG_PATH, status_path=cfg.STATUS_PATH,
                            allocations=[[0, 256]], training_seed=training_seed,
                            label="Full +20k control" if arm == "full_control" else "ASReceiver +20k"))
    for key, source in (("robust_frozen", load_source_record()), ("swin", archived_swin())):
        entries.append(dict(method_key=key, model_key=source["model_key"], path=source["path"],
                            expected_best_sha256=source["checkpoint_sha256"], archive=source,
                            allocations=source["allocations"],
                            label="Frozen robust pure-AS" if key == "robust_frozen" else "Swin-based FDMA"))
    entries.append(dict(entries[1], method_key="as_receiver_off", label="ASReceiver (receiver disabled)",
                        inference_intervention=True, trained_method="as_receiver"))
    if len({entry["path"] for entry in entries}) != 4:
        raise ValueError("Expected four distinct checkpoints; receiver-off must share ASReceiver best")
    return entries


def fingerprint(cfg):
    return channel_data_api().configuration_fingerprint(cfg.to_dict())


def verify_arm_metadata(arm, training_seed, best, latest, sidecars):
    """20k identity only; do not apply the historical 5k selected-step/hash rules."""
    from train_proposal_as_receiver import validate_resume_payload, validate_best_metadata
    cfg = as_receiver_config(arm, training_seed)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    _compare(20000, latest.get("step"), f"{arm}.latest.step")
    if type(best.get("step")) is not int or not 0 < best["step"] <= 20000:
        raise ValueError("Selected best step must lie in the completed 20k plan")
    expected = dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg))
    _compare(expected, sidecars["config"], f"{arm}.sidecar.config")
    status = sidecars["status"]
    for key, value in {"state": "completed", "step": 20000, "experiment": cfg.EXP_NAME,
                       "best_metric_name": "mean_all", "best_metric": best["best_metric"]}.items():
        _compare(value, status.get(key), f"{arm}.status.{key}")
    if "total_steps" in status:
        _compare(20000, status["total_steps"], "status.total_steps")
    return dict(config=best["config"], config_fingerprint=best["config_fingerprint"],
                selected_best_step=best["step"], latest_step=latest["step"], completed_updates=20000,
                best_metric=best["best_metric"], training_status=status, training_seed=training_seed,
                architecture_metadata=cfg.architecture_metadata)


def build_method(torch, ev, scenario, entry, payload, device):
    key, path = entry["method_key"], Path(entry["path"])
    if key == "swin":
        method = ev.build_baseline_method("swin", scenario, path, "best", device, 1e-3)
        check_tensor_state(torch, method.model.state_dict(), payload["state"], equal=True)
    else:
        from models.adaptive_hybrid import AdaptiveHybridPrecoder
        from models.adaptive_hybrid_as_receiver import AdaptiveHybridPrecoderASReceiver
        ev.validate_checkpoint_scenario(payload, path, scenario)
        cfg = ev.proposal_runtime_config(scenario, payload)
        receiver = key in ("as_receiver", "as_receiver_off")
        if receiver:
            metadata = payload["config"]["architecture_metadata"]
            _compare("as_receiver_v1", metadata.get("architecture"), "receiver.architecture")
            settings = metadata["receiver_settings"]
            cfg.AS_RECEIVER_LAMBDA_FLOOR = settings["lambda_floor"]
            cfg.AS_RECEIVER_SOLVE_DTYPE = settings["solve_dtype"]
        cls = AdaptiveHybridPrecoderASReceiver if receiver else AdaptiveHybridPrecoder
        model = cls(cfg)
        check_tensor_state(torch, model.state_dict(), payload["state"])
        model.load_state_dict(payload["state"], strict=True)  # Trained full state; never migrate/reset.
        check_tensor_state(torch, model.state_dict(), payload["state"], equal=True)
        check_proposal_model(model, cfg, 2048)
        if receiver:
            model.receiver_enabled = key != "as_receiver_off"
        model.to(device).eval()
        method = ev.LoadedMethod(model_key="proposed", method_key=key, display_name=entry["label"],
                                 model=model, checkpoint_path=str(path), checkpoint_step=payload["step"],
                                 checkpoint_kind="best", allocations=((0, 256),), config=cfg, rzf_lambda=1e-3)
    method.method_key, method.display_name = key, entry["label"]
    method.model.eval()
    _compare(entry["allocations"], method.allocations, f"{key}.allocation")
    return method


def verify_results(summary, samples, plan, count, seed, training_seed, *, method_keys=METHODS):
    validate_seed(training_seed)
    if type(count) is not int or count < 8 or count % 8:
        raise ValueError("Sample count must comprise complete eight-channel batches")
    by_key = {entry["method_key"]: entry for entry in plan}
    if not method_keys or len(set(method_keys)) != len(method_keys) or tuple(by_key) != tuple(method_keys) or len(plan) != len(method_keys):
        raise ValueError("Exactly the declared fixed methods, without duplicates, are required")
    expected = {(key, fb, 25) for key in method_keys for fb in FB_SNRS}
    indexed, summaries = {key: {} for key in expected}, {}

    def condition(row):
        key = row.get("method_key"), row.get("feedback_snr_db"), row.get("downlink_snr_db")
        if key not in expected:
            raise ValueError(f"Unexpected condition/oracle row: {key}")
        entry = by_key[key[0]]
        for name, value in dict(scenario_id=SCENARIO_ID, k_users=4, subcarriers=256, feedback_budget=256,
                                model_key=entry["model_key"], allocation_df=entry["allocations"][0][0],
                                allocation_da=entry["allocations"][0][1]).items():
            _compare(value, row.get(name), f"{key}.{name}")
        for name, value in (("seed", seed), ("training_seed", training_seed)):
            if name in row:
                _compare(value, row[name], f"{key}.{name}")
        return key

    for row in summary:
        key = condition(row)
        if key in summaries:
            raise ValueError(f"Duplicate summary condition: {key}")
        for name, value in dict(num_samples=count, antennas=32, seed=seed, rzf_lambda=0.001).items():
            _compare(value, row.get(name), f"{key}.{name}")
        summaries[key] = row
    if set(summaries) != expected:
        raise ValueError("Missing summary conditions")
    for row in samples:
        key, index = condition(row), row.get("sample_index")
        if type(index) is not int or not 0 <= index < count or index in indexed[key]:
            raise ValueError(f"Duplicate/invalid sample index: {key}, {index}")
        for metric in ("sum_rate", "nmse_db", "precoder_power"):
            if metric not in row or not math.isfinite(float(row[metric])):
                raise ValueError(f"Missing/nonfinite metric: {key}, {metric}")
        indexed[key][index] = row
    for key, rows in indexed.items():
        if set(rows) != set(range(count)):
            raise ValueError(f"Missing sample IDs: {key}")
        for metric in ("sum_rate", "nmse_db", "precoder_power"):
            statistics_ = sample_statistics([rows[i][metric] for i in range(count)])
            for suffix, value in statistics_.items():
                actual = summaries[key].get(metric + "_" + suffix)
                if actual is None or not math.isfinite(float(actual)) or not math.isclose(float(actual), value, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"Summary/raw mismatch: {key}.{metric}_{suffix}")
    return indexed


def paired_rows(summary, samples, plan, count, seed, training_seed, *, method_keys=METHODS,
                comparisons=COMPARISONS, primary_pairs=(("as_receiver", "full_control"),)):
    indexed = verify_results(summary, samples, plan, count, seed, training_seed, method_keys=method_keys)
    if (len(set(comparisons)) != len(comparisons) or any(a not in method_keys or b not in method_keys or a == b for a,b in comparisons)
            or any(pair not in comparisons for pair in primary_pairs)):
        raise ValueError("Invalid declared paired comparisons")
    rows, raw = [], []
    scopes = [("per_snr", (fb,)) for fb in FB_SNRS] + [("low_snr_mean", (0, 5, 10)), ("all_snr_mean", FB_SNRS)]
    for method, reference in comparisons:
        for estimand, snrs in scopes:
            differences = [statistics.fmean(float(indexed[method, fb, 25][i]["sum_rate"])
                                            - float(indexed[reference, fb, 25][i]["sum_rate"])
                                            for fb in snrs) for i in range(count)]
            batches = [statistics.fmean(differences[i:i+8]) for i in range(0, count, 8)]
            row = dict(comparison=f"{method}-{reference}", method_key=method, reference_method_key=reference,
                       training_seed=training_seed, seed=seed, scenario_id=SCENARIO_ID, estimand=estimand,
                       feedback_snr_points=",".join(map(str, snrs)), feedback_snr_db=snrs[0] if len(snrs) == 1 else None,
                       downlink_snr_db=25, num_samples=count, num_batches=count//8, batch_size=8,
                       batch_ci_estimable=len(batches)>1, sample_ci_unit="complete_multi_user_channel",
                       batch_ci_unit="sample_index//8", training_seeds_pooled=False,
                       role="primary" if (method, reference) in primary_pairs and estimand == "low_snr_mean"
                       else "secondary_25db_degradation" if (method, reference) in primary_pairs and snrs == (25,)
                       else "inference_intervention" if reference == "as_receiver_off" else "diagnostic")
            row.update({"delta_"+key: value for key, value in sample_statistics(differences).items()})
            row.update({"batch_delta_"+key: value for key, value in sample_statistics(batches).items()})
            rows.append(row)
            raw.extend(dict(comparison=row["comparison"], estimand=estimand, feedback_snr_points=row["feedback_snr_points"],
                            sample_index=i, batch_index=i//8, delta=value, training_seed=training_seed, seed=seed)
                       for i, value in enumerate(differences))
    return rows, raw


def create_output(training_seed, *, preflight=False, output_dir=None, root=OUTPUT_ROOT):
    validate_seed(training_seed)
    root = Path(root).absolute()
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    path = (root / f"{'preflight' if preflight else 'screening'}_seed{training_seed}_{token}"
            if output_dir is None else Path(output_dir).absolute())
    validate_output_path(path, root)
    path.mkdir(parents=True, exist_ok=False)
    return path


def runtime_record(torch=None, *, source_files=SOURCE_FILES):
    software = {}
    for package in ("torch", "numpy", "sionna"):
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = "unavailable"
    return dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_sha256={p: sha256_file(ROOT / p) for p in source_files}, python=sys.version,
                platform=platform.platform(), software=software, backend=None if torch is None else _backend_info(torch))


def artifact_hashes(output, *, expected_artifacts=OUTPUT_ARTIFACTS):
    """Bind all completed artifacts to safe basename keys; exclude self-manifest."""
    output = Path(output)
    actual = {path.name for path in output.iterdir()
              if path.name != "manifest.json" and path.name.endswith((".csv", ".csv.gz", ".json"))}
    if actual != set(expected_artifacts):
        raise ValueError(f"Incomplete/unexpected evaluation artifacts: missing={set(expected_artifacts)-actual}, extra={actual-set(expected_artifacts)}")
    hashes = {}
    for name in expected_artifacts:
        path = output / name
        if (not path.is_file() or path.is_symlink()
                or (hasattr(path, "is_junction") and path.is_junction())
                or getattr(path.lstat(), "st_file_attributes", 0) & 0x400
                or path.resolve().parent != output.resolve()):
            raise ValueError(f"Missing/linked evaluation artifact: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def evaluation_args(ev, output, p, *, tag=TAG):
    args = ev.build_parser().parse_args([
        "--scenario", "4,256,256", "--models", "proposed,swin", "--allocations", "0,256",
        "--checkpoint", "best", "--no-include-proposed-best", "--baseline-objective", "task",
        "--feedback-snr", "0,5,10,15,20,25", "--downlink-snr", "25", "--rzf-reg", "0.001",
        "--batch-size", "8", "--num-batches", str(p["num_batches"]), "--seed", str(p["seed"]),
        "--extra-metrics", "basic", "--output-dir", str(output), "--tag", tag,
    ])
    if args.include_proposed_best or args.no_save_samples:
        raise ValueError("Oracle output forbidden; individual samples required")
    return args


def run_evaluation(training_seed, *, preflight=False, output_dir=None):
    validate_seed(training_seed)
    output = create_output(training_seed, preflight=preflight, output_dir=output_dir)
    p = protocol(training_seed, preflight)
    manifest = dict(status="pending", protocol=p, output_directory=str(output), methods={},
                    primary="ASReceiver-Full within-sample FB0/5/10 average", secondary="FB25 signed degradation check",
                    statistics="ddof=1 SE, mean +/-1.96SE; sample-level and batch-mean paired CIs both retained",
                    seed_scope="No pooling of training seeds; same fixed test data protocol across seeds",
                    ci_scope="Evaluation sampling conditional on fixed checkpoints, not training-seed uncertainty",
                    intervention="receiver-off uses the identical trained ASReceiver best with only receiver_enabled=False",
                    power="precoder_power is total over 256 subcarriers, nominal256; not per-subcarrier1",
                    nmse="Raw task-surrogate representation diagnostic only; not used for selection",
                    comparison_limit="Swin is not equal-training-budget; common feedback RNG is not equal absolute noise or total UE energy")
    hashes, snapshots, methods = {}, {}, []
    summary = samples = None
    failure = None
    try:
        manifest["runtime"] = runtime_record()
        manifest["plan"] = plan = checkpoint_plan(training_seed)
        hashes[str(DEFAULT_MANIFEST)] = sha256_file(DEFAULT_MANIFEST)
        import torch
        manifest["runtime"]["backend"] = _backend_info(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback forbidden")
        import baseline_evaluate as ev
        manifest["runtime"] = runtime_record(torch)
        scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
        scenario.validate()
        args, device = evaluation_args(ev, output, p), torch.device("cuda:0")
        ev.set_seed(p["seed"])
        parameters, receiver_parameters, saved_receiver_state = [], [], None
        for entry in plan:
            key = entry["method_key"]
            if key in ("full_control", "as_receiver", "as_receiver_off"):
                arm = entry["arm"]
                latest = load_payload(torch, entry["latest_path"], hashes)
                best = load_payload(torch, entry["path"], hashes)
                sidecars = {name: read_json(entry[name+"_path"], hashes) for name in ("config", "status")}
                record = verify_arm_metadata(arm, training_seed, best, latest, sidecars)
                check_tensor_state(torch, latest["state"], best["state"])
                same_step = latest["step"] == best["step"]
                if same_step:
                    check_tensor_state(torch, latest["state"], best["state"], equal=True)
                record.update(latest_path=entry["latest_path"], latest_sha256=hashes[entry["latest_path"]],
                              best_latest_same_step=same_step, same_step_state_equality_required=same_step)
                del latest
            else:
                best = load_payload(torch, entry["path"], hashes, entry["expected_best_sha256"])
                record = verify_archived_metadata(entry, best)
            method = build_method(torch, ev, scenario, entry, best, device)
            methods.append(method)
            snapshots[key] = model_buffer_snapshot(method.model)
            validate_frozen(snapshots[key], snapshots[key])
            record.update(model_class=type(method.model).__name__, method_key=key, model_key=method.model_key,
                          best_path=entry["path"], best_sha256=hashes[entry["path"]], strict_loading=True,
                          allocation=entry["allocations"][0], rzf_regularization=1e-3,
                          inference_intervention=key=="as_receiver_off")
            parameter_row = dict(method_key=key, full_parameters=sum(x.numel() for x in method.model.parameters()),
                                 encoder_plus_condition_parameters=None, training_seed=training_seed,
                                 note="Parameter count, not MACs or latency")
            if key != "swin":
                record.update(cfg_idx=0, max_dimensions=[64,256], widths=check_proposal_model(method.model, method.config, 2048))
                parameter_row["encoder_plus_condition_parameters"] = sum(x.numel() for x in method.model.encoder.parameters()) + sum(x.numel() for x in method.model.cond_embed.parameters())
            if key.startswith("as_receiver"):
                record["receiver_enabled"] = method.model.receiver_enabled
                record["loaded_architecture_metadata"] = method.model.architecture_metadata
                actual_receiver = method.model.receiver_config_metadata
                for name, expected in {"width": 128, "heads": 4, "dropout": 0.0,
                                       "lambda_floor": 1e-6, "solve_dtype": "complex64"}.items():
                    _compare(expected, actual_receiver.get(name), f"receiver.runtime.{name}")
                if key == "as_receiver":
                    saved_receiver_state = {name: value.detach().cpu().clone() for name, value in best["state"].items()}
                else:
                    check_tensor_state(torch, saved_receiver_state, best["state"], equal=True)
                    record["identical_trained_state_to_as_receiver"] = True
                    saved_receiver_state = None
                for name, value in method.model.named_parameters():
                    if name.startswith("as_receiver."):
                        receiver_parameters.append(dict(method_key=key, name=name, numel=value.numel(),
                                                        l2_norm=float(value.detach().float().norm().item()),
                                                        max_abs=float(value.detach().abs().max().item()),
                                                        receiver_enabled=method.model.receiver_enabled))
            manifest["methods"][key] = record
            parameters.append(parameter_row)
            del best
        _compare(hashes, {path: sha256_file(path) for path in hashes}, "Inputs changed during loading")
        ev.write_csv(output/f"parameters_{TAG}.csv", parameters)
        ev.write_csv(output/f"receiver_parameters_{TAG}.csv", receiver_parameters)
        data = make_test_data(preflight)
        _compare(p["channel_data_protocol"], data.protocol, "evaluation data protocol")
        points = [(float(fb), 25.0) for fb in FB_SNRS]
        ev.set_seed(p["seed"])
        with torch.no_grad():
            summary, samples = ev.evaluate_scenario(scenario, methods, points, args, device, channel_data=data)
        deltas, paired = paired_rows(summary, samples, plan, p["num_samples"], p["seed"], training_seed)
        for row in summary+samples+deltas+paired:
            row.update(training_seed=training_seed, seed=p["seed"], preflight=preflight, dataset_role=p["dataset_role"])
        ev.save_outputs(summary, samples, [scenario], list(METHODS), points, args)
        ev.write_csv(output/f"paired_deltas_{TAG}.csv", deltas)
        ev.write_gzip_csv(output/f"paired_samples_{TAG}.csv.gz", paired)
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
                if key.startswith("as_receiver"):
                    _compare(key != "as_receiver_off", method.model.receiver_enabled, "Receiver switch changed")
            except Exception as exc:
                errors.append(f"Frozen check {method.method_key}: {exc}")
        manifest.update(input_sha256_before=hashes, input_sha256_after=after_hashes,
                        model_buffers_before=snapshots, model_buffers_after=after_buffers,
                        freeze_checks=dict(passed=not errors and len(methods)==len(METHODS), errors=errors))
        if errors or (failure is None and len(methods) != len(METHODS)):
            failure = failure or RuntimeError("Frozen/input checks failed: " + "; ".join(errors))
            manifest.update(status="failed", error=str(failure))
        if failure is None and manifest.get("status") == "completed":
            try:
                manifest["output_sha256"] = artifact_hashes(output)
            except Exception as exc:
                failure = exc
                manifest.update(status="failed", error=f"Artifact evidence failed: {exc}")
        write_manifest(output, manifest)
    if failure is not None:
        raise RuntimeError(f"ASReceiver evaluation failed; evidence: {output/'manifest.json'}") from failure
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Training seed; test seed fixed independently")
    parser.add_argument("--output-dir", type=Path, help="New directory strictly inside the evaluation root")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.dry_run or args.preflight or args.evaluate):
        parser.print_help()
        return 0
    if args.seed is None:
        parser.error("Explicit --seed is required for each separately reported training seed")
    if args.dry_run:
        if args.output_dir is not None:
            validate_output_path(args.output_dir.absolute(), OUTPUT_ROOT)
        print(json.dumps(dict(plan=checkpoint_plan(args.seed), evaluation=protocol(args.seed),
                              preflight=protocol(args.seed, True), output_root=str(OUTPUT_ROOT),
                              output_dir=None if args.output_dir is None else str(args.output_dir),
                              checkpoint_bytes_not_read=True, models_not_imported=True,
                              oracle_rows=False, training_seeds_pooled=False), indent=2))
        return 0
    run_evaluation(args.seed, preflight=args.preflight, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
