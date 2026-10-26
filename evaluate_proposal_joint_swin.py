#!/usr/bin/env python3
"""Six fixed Joint+Swin256 comparisons on new shared replayable channels.

Help/dry-run are standard-library only. Explicit preflight/evaluate require
CUDA; no training or checkpoint selection occurs in this evaluator.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import traceback

from configs.joint_swin_train_config import ARMS, SEEDS, joint_swin_config
from configs.as_spatial_train_config import channel_data_api
import evaluate_proposal_as_receiver as shared
import evaluate_proposal_as_spatial as spatial
import evaluate_swin_adaptation as adapted
from evaluate_proposal_as_direct import (
    check_tensor_state, load_payload, read_json, sample_statistics, verify_archived_metadata, write_manifest,
)
from evaluate_proposal_lite import _compare, model_buffer_snapshot, validate_frozen, validate_output_path
from preflight_proposal_as_direct import DEFAULT_MANIFEST, sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_joint_swin256_v1"
TAG = "k4_joint_swin256_v1"
METHODS = ("plain_swin256", "joint_swin256", "spatial_joint", "full_control", "swin", "swin_adapted")
COMPARISONS = tuple(("joint_swin256", reference) for reference in
                    ("plain_swin256", "spatial_joint", "swin_adapted", "swin", "full_control"))
PRIMARY_PAIRS = (("joint_swin256", "plain_swin256"),)
FB_SNRS, TEST_SEED, PREFLIGHT_SEED = shared.FB_SNRS, 20261120, 20261121
DIAGNOSTIC_METRICS = ("recovered_mse", "recovered_nmse_linear", "recovered_nmse_db",
                      "corrected_mse", "corrected_nmse_linear", "corrected_nmse_db",
                      "correction_mse", "reliability_mean")
OUTPUT_ARTIFACTS = tuple(f"{name}_{TAG}.{suffix}" for name, suffix in (
    ("summary", "csv"), ("samples", "csv.gz"), ("results", "json"), ("metadata", "json"),
    ("paired_deltas", "csv"), ("paired_samples", "csv.gz"), ("parameters", "csv"),
    ("component_parameters", "csv"), ("soft_codeword_summary", "csv"), ("soft_codeword_samples", "csv.gz")))
SOURCE_FILES = tuple(dict.fromkeys(adapted.SOURCE_FILES + (
    "evaluate_proposal_joint_swin.py", "models/adaptive_hybrid_joint_swin.py",
    "configs/joint_swin_train_config.py", "train_proposal_joint_swin.py")))


def make_test_data(preflight=False):
    return channel_data_api().ReplayableChannelData(
        base_seed=PREFLIGHT_SEED if preflight else TEST_SEED, k_users=4, nt=32, nsc=256,
        batch_size=8, validation_batch_size=8, carrier_freq=3.5e9, speed=1.0,
        cdl_model="C", device="cuda:0")


def protocol(training_seed, preflight=False):
    result = shared.protocol(training_seed, preflight)
    result.update(seed=PREFLIGHT_SEED if preflight else TEST_SEED,
                  channel_data_protocol=make_test_data(preflight).protocol,
                  methods=list(METHODS), screening_not_formal_result=True,
                  diagnostic_scope="Recovered/corrected errors from same forward, detached truth after decoder only")
    return result


def checkpoint_plan(training_seed):
    shared.validate_seed(training_seed)
    entries = []
    for arm in ARMS:
        cfg = joint_swin_config(arm, training_seed)
        entries.append(dict(method_key=arm, model_key="proposed", arm=arm, training_seed=training_seed,
                            path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
                            config_path=cfg.CONFIG_PATH, status_path=cfg.STATUS_PATH,
                            allocations=[[0, 256]], label="Plain Swin256" if arm == "plain_swin256" else "Joint+Swin256"))
    old = {row["method_key"]: row for row in adapted.checkpoint_plan(training_seed)}
    entries.append(dict(old["joint"], method_key="spatial_joint", label="ASSpatial joint (existing)"))
    entries.extend(old[key] for key in ("full_control", "swin", "swin_adapted"))
    if tuple(row["method_key"] for row in entries) != METHODS or len({row["path"] for row in entries}) != 6:
        raise ValueError("Six distinct, fixed checkpoint identities required")
    return entries


def verify_new_metadata(arm, seed, best, latest, sidecars):
    from train_proposal_joint_swin import fingerprint, validate_resume_payload, validate_best_metadata
    cfg = joint_swin_config(arm, seed)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    _compare(20000, latest.get("step"), "Joint+Swin256 completed updates")
    if type(best.get("step")) is not int or not 0 < best["step"] <= 20000 or best["step"] % 500:
        raise ValueError("New random BS must select a positive 500-step validation, not source step0")
    _compare(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg)), sidecars["config"], "Joint+Swin256 sidecar config")
    for field, expected in dict(state="completed", step=20000, experiment=cfg.EXP_NAME,
                                best_metric_name="mean_all", best_metric=best["best_metric"]).items():
        _compare(expected, sidecars["status"].get(field), f"Joint+Swin256 status.{field}")
    return dict(config=best["config"], config_fingerprint=best["config_fingerprint"],
                selected_best_step=best["step"], latest_step=latest["step"], completed_updates=20000,
                best_metric=best["best_metric"], selection_metric="mean_all", training_seed=seed,
                training_status=sidecars["status"], architecture_metadata=cfg.architecture_metadata)


def build_method(torch, ev, scenario, entry, payload, device):
    key = entry["method_key"]
    if key not in ARMS:
        original = dict(entry, method_key="joint") if key == "spatial_joint" else entry
        method = adapted.build_method(torch, ev, scenario, original, payload, device)
        method.method_key, method.display_name = key, entry["label"]
        return method
    from models.adaptive_hybrid_joint_swin import AdaptiveHybridPrecoderJointSwin
    from train_proposal_joint_swin import validate_model_identity
    cfg = joint_swin_config(entry["arm"], entry["training_seed"])
    _compare(cfg.to_dict(), payload["config"], "Saved new-arm config")
    model = AdaptiveHybridPrecoderJointSwin(cfg)
    validate_model_identity(cfg, model)
    check_tensor_state(torch, model.state_dict(), payload["state"])
    model.load_state_dict(payload["state"], strict=True)
    check_tensor_state(torch, model.state_dict(), payload["state"], equal=True)
    _compare([(0, 256)], cfg.GAMMA_GRID, "New supported grid")
    _compare((64, 256), (cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED), "New max architecture")
    _compare((1,), tuple(model.running_pwr.shape), "New singleton running power")
    if not (model.running_pwr > 0).all().item():
        raise ValueError("Nonpositive running power")
    model.to(device).eval()
    return ev.LoadedMethod(model_key="proposed", method_key=key, display_name=entry["label"],
                           model=model, checkpoint_path=entry["path"], checkpoint_step=payload["step"],
                           checkpoint_kind="best", allocations=((0, 256),), config=cfg, rzf_lambda=1e-3)


def paired_rows(summary, samples, plan, count, seed, training_seed):
    return shared.paired_rows(summary, samples, plan, count, seed, training_seed,
                              method_keys=METHODS, comparisons=COMPARISONS, primary_pairs=PRIMARY_PAIRS)


def summarize_diagnostics(rows, count, seed, training_seed):
    if type(count) is not int or count < 8 or count % 8:
        raise ValueError("Diagnostics require complete eight-channel batches")
    indexed = {(arm, fb): {} for arm in ARMS for fb in FB_SNRS}
    for row in rows:
        key, index = (row.get("method_key"), row.get("feedback_snr_db")), row.get("sample_index")
        if key not in indexed or type(index) is not int or not 0 <= index < count or index in indexed[key]:
            raise ValueError("Unexpected or duplicate diagnostic sample")
        for name, value in dict(seed=seed, training_seed=training_seed, batch_index=index // 8,
                                downlink_snr_db=25, allocation_df=0, allocation_da=256).items():
            _compare(value, row.get(name), f"Diagnostic {name}")
        if any(not math.isfinite(float(row[name])) for name in DIAGNOSTIC_METRICS + ("noise_est", "regularization")):
            raise ValueError("Nonfinite diagnostic")
        for name in ("recovered_mse", "recovered_nmse_linear", "corrected_mse", "corrected_nmse_linear", "correction_mse"):
            if row[name] < 0:
                raise ValueError("Negative squared error")
        if not 0 <= row["reliability_mean"] <= 1:
            raise ValueError("Reliability outside [0,1]")
        if key[0] == "plain_swin256" and (row["correction_mse"] != 0 or
                any(row["recovered_" + suffix] != row["corrected_" + suffix]
                    for suffix in ("mse", "nmse_linear", "nmse_db"))):
            raise ValueError("Plain arm must leave the recovered codeword unchanged")
        indexed[key][index] = row
    output = []
    for (arm, fb), records in indexed.items():
        if set(records) != set(range(count)):
            raise ValueError("Missing diagnostic sample IDs")
        row = dict(method_key=arm, feedback_snr_db=fb, downlink_snr_db=25, seed=seed,
                   training_seed=training_seed, num_samples=count, num_batches=count // 8, batch_size=8,
                   batch_ci_estimable=count > 8, unit="complete_multi_user_channel")
        for metric in DIAGNOSTIC_METRICS:
            values = [records[i][metric] for i in range(count)]
            batches = [statistics.fmean(values[i:i + 8]) for i in range(0, count, 8)]
            row.update({metric + "_" + suffix: value for suffix, value in sample_statistics(values).items()})
            row.update({"batch_" + metric + "_" + suffix: value for suffix, value in sample_statistics(batches).items()})
        for metric in ("noise_est", "regularization"):
            values = []
            for start in range(0, count, 8):
                if any(records[i][metric] != records[start][metric] for i in range(start, start + 8)):
                    raise ValueError("Batch-shared diagnostic scalar changed within a batch")
                values.append(records[start][metric])
            row.update({metric + "_batch_mean": statistics.fmean(values), metric + "_batch_min": min(values),
                        metric + "_batch_max": max(values)})
        output.append(row)
    return output


def validate_execution_record(arm, record):
    if arm not in ARMS:
        raise ValueError("Unknown execution-record arm")
    expected = dict(recovered_shape=[8, 256, 4],
                    joint_context_shape=None if arm == "plain_swin256" else [8, 1024, 128],
                    spatial_shape=[8, 4, 512, 128], angular_delay_shape=[8, 4, 2, 256, 32])
    _compare(expected, record, "Actual diagnostic runtime tensor shapes")
    return expected


class CodewordCollector:
    """Offline detached reductions after decoding; never an input to a model."""
    def __init__(self, torch, seed, training_seed):
        self.torch, self.seed, self.training_seed = torch, seed, training_seed
        self.rows = []
        self.rng_checks = dict(calls=0, passed=True)
        self.execution_records = {}
        self.rng = channel_data_api()._load_source("utils/sionna_rng.py", "_precoder_sionna_rng_isolation_v1")

    def __call__(self, context, payload):
        if context["method_key"] not in ARMS:
            return  # Existing ASSpatial also exposes diagnostics; not a new intervention.
        torch = self.torch
        before = self.rng.snapshot_rng_state().fingerprint()
        try:
            for name in ("feedback_snr_db", "allocation_df", "allocation_da"):
                _compare(context[name], payload.get(name), f"Diagnostic payload {name}")
            execution = validate_execution_record(context["method_key"], payload.get("execution"))
            identity = (context["method_key"], context["feedback_snr_db"])
            observed = self.execution_records.setdefault(identity, dict(method_key=identity[0],
                feedback_snr_db=identity[1], calls=0, runtime_shapes=execution,
                scope="Actual tensor shapes; module-call evidence belongs to separate CUDA preflight"))
            _compare(observed["runtime_shapes"], execution, "Runtime shape changed across batches")
            observed["calls"] += 1
            for name in ("z_hat", "z_corrected", "true_z_a", "reliability", "noise_est", "regularization"):
                value = payload[name]
                if not torch.is_tensor(value) or value.requires_grad or not torch.isfinite(value).all().item():
                    raise ValueError(f"Expected finite, detached diagnostic tensor: {name}")
            recovered, corrected = payload["z_hat"].permute(0, 2, 1), payload["z_corrected"].permute(0, 2, 1)
            true, reliability = payload["true_z_a"], payload["reliability"].permute(0, 2, 1)
            if any(tuple(value.shape) != (8, 4, 256) for value in (recovered, corrected, true, reliability)):
                raise ValueError("Expected codewords aligned as [8,4,256]")
            if not all(value.is_complex() for value in (recovered, corrected, true)) or reliability.is_complex():
                raise ValueError("Codeword/reliability dtype mismatch")
            denominator = true.abs().square().mean(dim=(1, 2)).clamp_min(1e-12)
            metrics = {}
            for label, value in (("recovered", recovered), ("corrected", corrected)):
                mse = (value - true).abs().square().mean(dim=(1, 2))
                nmse = mse / denominator
                metrics.update({label + "_mse": mse, label + "_nmse_linear": nmse,
                                label + "_nmse_db": 10 * torch.log10(nmse.clamp_min(1e-12))})
            metrics.update(correction_mse=(corrected - recovered).abs().square().mean(dim=(1, 2)),
                           reliability_mean=reliability.mean(dim=(1, 2)))
            values = {name: value.detach().cpu().tolist() for name, value in metrics.items()}
            for i in range(8):
                self.rows.append(dict(context, sample_index=context["batch_index"] * 8 + i,
                                      seed=self.seed, training_seed=self.training_seed,
                                      noise_est=float(payload["noise_est"].item()),
                                      regularization=float(payload["regularization"].item()),
                                      **{name: value[i] for name, value in values.items()}))
        finally:
            after = self.rng.snapshot_rng_state().fingerprint()
            self.rng_checks["calls"] += 1
            if before != after:
                self.rng_checks.update(passed=False, first_failure_context=dict(context), before=before, after=after)
                raise RuntimeError("Offline diagnostic collector changed external RNG")


def artifact_hashes(output):
    return shared.artifact_hashes(output, expected_artifacts=OUTPUT_ARTIFACTS)


def runtime_record(torch=None):
    return shared.runtime_record(torch, source_files=SOURCE_FILES)


def run_evaluation(training_seed, *, preflight=False, output_dir=None):
    shared.validate_seed(training_seed)
    output = shared.create_output(training_seed, preflight=preflight, output_dir=output_dir, root=OUTPUT_ROOT)
    p = protocol(training_seed, preflight)
    manifest = dict(status="pending", protocol=p, output_directory=str(output), methods={},
                    primary="joint_swin256-plain_swin256 within-channel FB0/5/10 mean; FB25 degradation check",
                    statistics="Signed paired ddof1 sample and sample_index//8 batch intervals; seeds not pooled",
                    ci_scope="Evaluation sampling conditional on fixed checkpoints, not multi-seed uncertainty",
                    diagnostic_scope="True codewords detached after reconstruction; offline errors only, no extra forward",
                    nmse="Raw representation diagnostic, not checkpoint selection", power="Full-band total, nominal256",
                    comparison_limit="Not equal parameters or historical training budgets; common RNG is not equal absolute noise")
    hashes, snapshots, methods = {}, {}, []
    summary = samples = collector = None
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
        parameters, components = [], []
        for entry in plan:
            key = entry["method_key"]
            if key == "swin":
                best = load_payload(torch, entry["path"], hashes, entry["expected_best_sha256"])
                record = verify_archived_metadata(entry, best)
            else:
                latest = load_payload(torch, entry["latest_path"], hashes)
                best = load_payload(torch, entry["path"], hashes)
                sidecars = {name: read_json(entry[name + "_path"], hashes) for name in ("config", "status")}
                if key in ARMS:
                    record = verify_new_metadata(entry["arm"], training_seed, best, latest, sidecars)
                elif key == "swin_adapted":
                    record = adapted.verify_adapted_metadata(training_seed, best, latest, sidecars)
                else:
                    record = spatial.verify_arm_metadata(entry["arm"], training_seed, best, latest, sidecars)
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
            if key in ARMS:
                record["loaded_architecture_metadata"] = method.model.architecture_metadata
            ue = sum(value.numel() for value in method.model.encoder.parameters())
            if method.model_key == "proposed":
                ue += sum(value.numel() for value in method.model.cond_embed.parameters())
            parameters.append(dict(method_key=key, training_seed=training_seed,
                                   full_parameters=sum(value.numel() for value in method.model.parameters()),
                                   encoder_plus_condition_parameters=ue,
                                   note="Stored parameters, not matched complexity or measured latency"))
            counts = {}
            for name, value in method.model.named_parameters():
                component = name.split(".", 1)[0]
                counts[component] = counts.get(component, 0) + value.numel()
            components.extend(dict(method_key=key, component=name, parameters=count) for name, count in sorted(counts.items()))
            manifest["methods"][key] = record
            del best
        _compare(hashes, {path: sha256_file(path) for path in hashes}, "Inputs changed during loading")
        ev.write_csv(output / f"parameters_{TAG}.csv", parameters)
        ev.write_csv(output / f"component_parameters_{TAG}.csv", components)
        collector = CodewordCollector(torch, p["seed"], training_seed)
        points = [(float(fb), 25.0) for fb in FB_SNRS]
        ev.set_seed(p["seed"])
        with torch.no_grad():
            summary, samples = ev.evaluate_scenario(scenario, methods, points, args, device,
                channel_data=make_test_data(preflight), diagnostic_sink=collector)
        deltas, paired = paired_rows(summary, samples, plan, p["num_samples"], p["seed"], training_seed)
        diagnostics = summarize_diagnostics(collector.rows, p["num_samples"], p["seed"], training_seed)
        for row in summary + samples + deltas + paired + diagnostics + collector.rows:
            row.update(training_seed=training_seed, seed=p["seed"], preflight=preflight, dataset_role=p["dataset_role"])
        ev.save_outputs(summary, samples, [scenario], list(METHODS), points, args)
        ev.write_csv(output / f"paired_deltas_{TAG}.csv", deltas)
        ev.write_gzip_csv(output / f"paired_samples_{TAG}.csv.gz", paired)
        ev.write_csv(output / f"soft_codeword_summary_{TAG}.csv", diagnostics)
        ev.write_gzip_csv(output / f"soft_codeword_samples_{TAG}.csv.gz", collector.rows)
        manifest.update(status="completed", results_verified=True, summary_rows=len(summary),
                        sample_rows=len(samples), paired_rows=len(deltas), soft_summary_rows=len(diagnostics))
    except Exception as exc:
        failure = exc
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if summary is not None:
            try:
                ev.save_outputs(summary, samples, [scenario], list(METHODS), points, args)
            except Exception as save_error:
                manifest["failure_data_save_error"] = str(save_error)
        if collector is not None and collector.rows:
            try:
                ev.write_gzip_csv(output / f"soft_codeword_partial_{TAG}.csv.gz", collector.rows)
            except Exception as save_error:
                manifest["failure_diagnostic_save_error"] = str(save_error)
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
                if key in ARMS:
                    from train_proposal_joint_swin import validate_model_identity
                    validate_model_identity(method.config, method.model)
                elif key == "spatial_joint":
                    _compare(False, method.model.ue_frozen, "ASSpatial joint changed arm")
            except Exception as exc:
                errors.append(f"Frozen check {method.method_key}: {exc}")
        if collector is not None:
            manifest["diagnostic_rng_checks"] = collector.rng_checks
            manifest["diagnostic_execution_records"] = list(collector.execution_records.values())
            if not collector.rng_checks["passed"]:
                errors.append("Offline diagnostic callback changed RNG")
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
        raise RuntimeError(f"Joint+Swin256 evaluation failed; evidence: {output / 'manifest.json'}") from failure
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
        parser.error("Explicit training seed required; evaluation seed is fixed independently")
    if args.dry_run:
        import json
        if args.output_dir is not None:
            validate_output_path(args.output_dir.absolute(), OUTPUT_ROOT)
        print(json.dumps(dict(plan=checkpoint_plan(args.seed), evaluation=protocol(args.seed),
                              preflight=protocol(args.seed, True), output_root=str(OUTPUT_ROOT),
                              models_not_imported=True, checkpoint_bytes_not_read=True,
                              training_seeds_pooled=False, diagnostic_truth_not_decoder_input=True), indent=2))
        return 0
    run_evaluation(args.seed, preflight=args.preflight, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
