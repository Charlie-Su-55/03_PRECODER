#!/usr/bin/env python3
"""Fixed four-method ASDirect screening test; stdlib dry-run, CUDA opt-in only."""
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
import uuid

from configs.as_direct_train_config import as_direct_config
from preflight_proposal_as_direct import (
    DEFAULT_MANIFEST, _backend_info, _sha256_stream, load_source_record,
    sha256_file, validate_checkpoint_metadata,
)
from train_proposal_as_direct import fingerprint, validate_best_metadata, validate_resume_payload
from evaluate_proposal_lite import (
    _compare, _saved_fingerprint, check_proposal_model, model_buffer_snapshot,
    validate_frozen, validate_output_path,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_as_direct_v1"
TAG = "k4_as_direct_v1"
METHODS = ("full_control", "as_direct", "robust_frozen", "swin")
FB_SNRS = (0, 5, 10, 15, 20, 25)
DL_SNR = 25
BATCH_SIZE = 8
SCENARIO_ID = "K4_Nsc256_Dfb256"
LATEST_SHA256 = {
    "full_control": "3362abf4812f1fd61c2dc66795766bbfcf26845f59161be739328e15258492cc",
    "as_direct": "211b55c28a1501b34e4197c69098d402a6d63828b3c2a79cf4154987b5f77981",
}
COMPARISONS = (("as_direct", "full_control"), ("full_control", "robust_frozen"),
               ("as_direct", "robust_frozen"), ("as_direct", "swin"))
GATES = tuple(f"layers.{i}.alpha" for i in range(5))
SOURCE_FILES = (
    "evaluate_proposal_as_direct.py", "baseline_evaluate.py", "evaluate_proposal_lite.py",
    "train_proposal_as_direct.py", "preflight_proposal_as_direct.py",
    "configs/as_direct_train_config.py", "configs/train_config.py", "configs/baseline_config.py",
    "models/adaptive_hybrid.py", "models/adaptive_hybrid_as_direct.py", "models/swin_precoder.py",
    "models/baseline_common.py", "utils/channel_generator.py", "utils/layers.py",
)


def protocol(preflight=False):
    return dict(preflight=preflight, seed=20261027 if preflight else 20261026,
                batch_size=8, num_batches=1 if preflight else 200,
                num_samples=8 if preflight else 1600, feedback_snr_db=list(FB_SNRS),
                downlink_snr_db=25, channel_profile="CDL-C",
                dataset_role="preflight_not_screening_result" if preflight else "independent_screening_test",
                scenario=dict(K=4, Nt=32, Nsc=256, Dtot=256, Nsb=4))


def archived_swin(manifest_path=DEFAULT_MANIFEST):
    path = Path(manifest_path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Archived manifest exceeds 2 MiB")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("methods", []) if item.get("method_key") == "swin"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one archived Swin")
    entry = matches[0]
    for field, expected in {
        "model_key": "swin", "method_type": "Swin", "training_status": "completed",
        "training_mode": "task_trained_analog_FDMA", "allocations": [[64, 0]],
        "checkpoint_step": 49000, "cfg_index": None,
        "checkpoint_sha256": "c913f3477f6af74665d7ba1b11a7a8e06bdea8c051822205ace1a8a071489276",
    }.items():
        _compare(expected, entry.get(field), f"archived_swin.{field}")
    if not isinstance(entry.get("path"), str) or not entry["path"]:
        raise ValueError("Archived Swin path missing")
    config = entry.get("loaded_checkpoint_config", {})
    for field, expected in {"MODEL_NAME": "swin", "OBJECTIVE": "task", "TOTAL_STEPS": 50000,
                            "K_USERS": 4, "ANTENNAS": 32, "SUBCARRIERS": 256, "D_TOT": 256,
                            "D_F": 64, "RZF_REG": 0.001}.items():
        _compare(expected, config.get(field), f"archived_swin.config.{field}")
    return entry


def checkpoint_plan():
    result = []
    for arm in METHODS[:2]:
        cfg = as_direct_config(arm)
        result.append(dict(method_key=arm, model_key="proposed", path=cfg.BEST_PATH,
                           latest_path=cfg.SAVE_PATH, expected_latest_sha256=LATEST_SHA256[arm],
                           config_path=cfg.CONFIG_PATH, status_path=cfg.STATUS_PATH,
                           allocations=[[0, 256]],
                           label="Full +5k control" if arm == "full_control" else "ASDirect +5k"))
    for key, source in (("robust_frozen", load_source_record()), ("swin", archived_swin())):
        result.append(dict(method_key=key, model_key=source["model_key"], path=source["path"],
                           expected_best_sha256=source["checkpoint_sha256"], archive=source,
                           allocations=source["allocations"],
                           label="Frozen robust pure-AS" if key == "robust_frozen" else "Swin-based FDMA"))
    if len({entry["path"] for entry in result}) != 4:
        raise ValueError("Four methods must have distinct checkpoint paths")
    return result


def verify_arm_metadata(arm, best, latest, sidecars, cfg=None):
    cfg = as_direct_config(arm) if cfg is None else cfg
    _compare(arm, cfg.arm, "arm config identity")
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    for kind, payload in (("best", best), ("latest", latest)):
        if type(payload.get("step")) is not int or payload["step"] != 5000:
            raise ValueError(f"{arm}.{kind}.step must be 5000")
    _compare(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg)), sidecars["config"], f"{arm}.sidecar.config")
    status = sidecars["status"]
    for name, expected in {"state": "completed", "step": 5000, "experiment": cfg.EXP_NAME,
                           "best_metric_name": "mean_all", "best_metric": best["best_metric"]}.items():
        _compare(expected, status.get(name), f"{arm}.status.{name}")
    if "total_steps" in status:
        _compare(5000, status["total_steps"], f"{arm}.status.total_steps")
    return dict(config=best["config"], config_fingerprint=best["config_fingerprint"],
                best_step=best["step"], latest_step=latest["step"], completed_updates=5000,
                best_metric=best["best_metric"], training_status=status)


def verify_archived_metadata(entry, payload):
    source = entry["archive"]
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Archived checkpoint lacks saved config")
    _compare(source["loaded_checkpoint_config"], payload["config"], f"{entry['method_key']}.config")
    if type(payload.get("step")) is not int:
        raise ValueError("Archived checkpoint step must be an integer")
    _compare(source["checkpoint_step"], payload["step"], "archived checkpoint step")
    _compare(_saved_fingerprint(payload["config"], entry["model_key"]),
             payload.get("config_fingerprint"), "archived checkpoint fingerprint")
    if entry["method_key"] == "robust_frozen":
        validate_checkpoint_metadata(payload, source)
    elif payload.get("allocation_grid") is not None:
        _compare([[64, 0]], payload["allocation_grid"], "Swin allocation grid")
    return dict(config=payload["config"], config_fingerprint=payload["config_fingerprint"],
                best_step=payload["step"], training_status=source["training_status"],
                training_recipe_updates=20000 if entry["method_key"] == "robust_frozen" else 50000,
                completion_evidence="Archived diagnostic manifest; no new completion experiment performed")


def check_tensor_state(torch, reference, actual, *, equal=False):
    """Reject implicit dtype conversion, missing gates, slicing and partial loads."""
    if not isinstance(actual, dict) or set(reference) != set(actual):
        raise ValueError("State key mismatch (including trained gate keys)")
    for name, expected in reference.items():
        value = actual[name]
        if not torch.is_tensor(expected) or not torch.is_tensor(value) or expected.shape != value.shape or expected.dtype != value.dtype:
            raise ValueError(f"State shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise ValueError(f"Nonfinite state: {name}")
        if equal and not torch.equal(expected.detach().cpu(), value.detach().cpu()):
            raise ValueError(f"State values differ: {name}")


def load_payload(torch, path, hashes, expected=None):
    with Path(path).open("rb") as stream:
        actual = _sha256_stream(stream)
        hashes[str(path)] = actual
        if expected is not None and actual != expected:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}; expected={expected}; actual={actual}")
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=False)


def read_json(path, hashes):
    raw = Path(path).read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8"))


def build_method(torch, ev, scenario, entry, payload, device):
    key, path = entry["method_key"], Path(entry["path"])
    if key == "swin":
        method = ev.build_baseline_method("swin", scenario, path, "best", device, 1e-3)
        # The existing strict loader is retained; also reject any dtype conversion.
        check_tensor_state(torch, method.model.state_dict(), payload["state"], equal=True)
    else:
        from models.adaptive_hybrid import AdaptiveHybridPrecoder
        from models.adaptive_hybrid_as_direct import AdaptiveHybridPrecoderASDirect
        ev.validate_checkpoint_scenario(payload, path, scenario)
        cfg = ev.proposal_runtime_config(scenario, payload)
        cls = AdaptiveHybridPrecoderASDirect if key == "as_direct" else AdaptiveHybridPrecoder
        model = cls(cfg)
        check_tensor_state(torch, model.state_dict(), payload["state"])
        model.load_state_dict(payload["state"], strict=True)  # Never migrate or zero trained gates.
        check_tensor_state(torch, model.state_dict(), payload["state"], equal=True)
        check_proposal_model(model, cfg, 2048)
        model.to(device).eval()
        method = ev.LoadedMethod(model_key="proposed", method_key=key, display_name=entry["label"],
                                 model=model, checkpoint_path=str(path), checkpoint_step=payload["step"],
                                 checkpoint_kind="best", allocations=((0, 256),), config=cfg,
                                 rzf_lambda=1e-3)
    method.method_key, method.display_name = key, entry["label"]
    method.model.eval()
    _compare(entry["allocations"], method.allocations, f"{key}.loaded_allocations")
    return method


def sample_statistics(values):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Statistics need finite, nonempty samples")
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    if not math.isfinite(mean) or (se is not None and not math.isfinite(se)):
        raise ValueError("Nonfinite derived statistics")
    return dict(mean=mean, se=se, ci95_low=None if se is None else mean - 1.96 * se,
                ci95_high=None if se is None else mean + 1.96 * se)


def verify_results(summary, samples, plan, count, seed):
    if type(count) is not int or count < 8 or count % BATCH_SIZE:
        raise ValueError("Sample count must contain complete batches of eight")
    by_key = {entry["method_key"]: entry for entry in plan}
    if tuple(by_key) != METHODS or len(plan) != 4:
        raise ValueError("Exactly the four fixed methods are required")
    expected = {(key, fb, DL_SNR) for key in METHODS for fb in FB_SNRS}
    indexed, summaries = {key: {} for key in expected}, {}

    def condition(row):
        key = (row.get("method_key"), row.get("feedback_snr_db"), row.get("downlink_snr_db"))
        if key not in expected:
            raise ValueError(f"Unexpected method/SNR condition: {key}")
        method = by_key[key[0]]
        for field, value in {"scenario_id": SCENARIO_ID, "k_users": 4, "subcarriers": 256,
                             "feedback_budget": 256, "model_key": method["model_key"],
                             "allocation_df": method["allocations"][0][0],
                             "allocation_da": method["allocations"][0][1]}.items():
            _compare(value, row.get(field), f"{key}.{field}")
        if "seed" in row:
            _compare(seed, row["seed"], f"{key}.seed")
        return key

    for row in summary:
        key = condition(row)
        if key in summaries:
            raise ValueError(f"Duplicate summary condition: {key}")
        for field, value in {"num_samples": count, "antennas": 32, "seed": seed, "rzf_lambda": 0.001}.items():
            _compare(value, row.get(field), f"{key}.{field}")
        summaries[key] = row
    if set(summaries) != expected:
        raise ValueError(f"Missing summary conditions: {expected - set(summaries)}")
    for row in samples:
        key, index = condition(row), row.get("sample_index")
        if type(index) is not int or not 0 <= index < count or index in indexed[key]:
            raise ValueError(f"Duplicate/invalid sample ID: {key}, {index}")
        for metric in ("sum_rate", "nmse_db", "precoder_power"):
            if metric not in row or not math.isfinite(float(row[metric])):
                raise ValueError(f"Missing/nonfinite {metric}: {key}, {index}")
        indexed[key][index] = row
    for key, rows in indexed.items():
        if set(rows) != set(range(count)):
            raise ValueError(f"Missing sample IDs: {key}")
        for metric in ("sum_rate", "nmse_db", "precoder_power"):
            values = sample_statistics([rows[i][metric] for i in range(count)])
            for suffix, expected_value in values.items():
                actual = summaries[key].get(metric + "_" + suffix)
                if actual is None or not math.isfinite(float(actual)) or not math.isclose(float(actual), expected_value, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"Summary/raw mismatch: {key}.{metric}_{suffix}")
    return indexed


def paired_rows(summary, samples, plan, count, seed):
    indexed = verify_results(summary, samples, plan, count, seed)
    rows, raw_deltas = [], []
    scopes = [("per_snr", (fb,)) for fb in FB_SNRS] + [
        ("low_snr_mean", (0, 5, 10)), ("all_snr_mean", FB_SNRS)]
    for method, reference in COMPARISONS:
        for estimand, snrs in scopes:
            # Align by complete condition + sample_index before averaging SNRs.
            differences = [statistics.fmean(
                float(indexed[method, fb, DL_SNR][i]["sum_rate"]) -
                float(indexed[reference, fb, DL_SNR][i]["sum_rate"]) for fb in snrs)
                for i in range(count)]
            batches = [statistics.fmean(differences[i:i + BATCH_SIZE]) for i in range(0, count, BATCH_SIZE)]
            per_sample, per_batch = sample_statistics(differences), sample_statistics(batches)
            row = dict(comparison=f"{method}-{reference}", method_key=method, reference_method_key=reference,
                       scenario_id=SCENARIO_ID, estimand=estimand, feedback_snr_points=",".join(map(str, snrs)),
                       feedback_snr_db=snrs[0] if len(snrs) == 1 else None, downlink_snr_db=25,
                       num_samples=count, num_batches=count // 8, batch_size=8, seed=seed,
                       sample_ci_unit="complete_multi_user_channel", batch_ci_unit="sample_index//8",
                       batch_ci_estimable=len(batches) > 1,
                       role="primary" if (method, reference, estimand) == ("as_direct", "full_control", "low_snr_mean") else
                       "secondary_25db_degradation_check" if (method, reference, snrs) == ("as_direct", "full_control", (25,)) else "diagnostic")
            row.update({"delta_" + key: value for key, value in per_sample.items()})
            row.update({"batch_delta_" + key: value for key, value in per_batch.items()})
            rows.append(row)
            raw_deltas.extend(dict(comparison=row["comparison"], estimand=estimand,
                                   feedback_snr_points=row["feedback_snr_points"], sample_index=i,
                                   batch_index=i // 8, delta=value, seed=seed)
                              for i, value in enumerate(differences))
    return rows, raw_deltas


def create_output(root=OUTPUT_ROOT, *, preflight=False, run_id=None):
    root = Path(root)
    token = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    if not isinstance(token, str) or not token or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in token):
        raise ValueError("Unsafe run_id")
    output = root / (("preflight_" if preflight else "screening_") + token)
    validate_output_path(output, root)
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_manifest(output, manifest):
    # A partially failed run must still leave standards-compliant evidence.
    bad = []
    def clean(value, key="manifest"):
        if isinstance(value, float) and not math.isfinite(value):
            bad.append(key)
            return None
        if isinstance(value, dict):
            return {name: clean(item, key + "." + name) for name, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item, f"{key}[{i}]") for i, item in enumerate(value)]
        return value
    value = clean(manifest)
    if bad:
        value.update(status="failed", nonfinite_report_fields=bad)
    with (Path(output) / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
    if bad:
        raise ValueError(f"Nonfinite manifest fields: {bad}")


def runtime_record(torch=None):
    software = {}
    for package in ("torch", "numpy", "sionna"):
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = "unavailable"
    return dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_sha256={p: sha256_file(ROOT / p) for p in SOURCE_FILES},
                python=sys.version, platform=platform.platform(), software=software,
                backend=None if torch is None else _backend_info(torch))


def evaluation_args(ev, output, p):
    args = ev.build_parser().parse_args([
        "--scenario", "4,256,256", "--models", "proposed,swin", "--allocations", "0,256",
        "--checkpoint", "best", "--no-include-proposed-best", "--baseline-objective", "task",
        "--feedback-snr", "0,5,10,15,20,25", "--downlink-snr", "25", "--rzf-reg", "0.001",
        "--batch-size", "8", "--num-batches", str(p["num_batches"]), "--seed", str(p["seed"]),
        "--extra-metrics", "basic", "--output-dir", str(output), "--tag", TAG,
        "--proposal-root", str(ROOT / "runs_as_screen/proposal_as_direct_v1"),
        "--baseline-root", str(ROOT / "runs_as_screen/baselines"),
    ])
    if args.include_proposed_best or args.no_save_samples:
        raise ValueError("Oracle rows forbidden; per-sample output required")
    return args


def run_evaluation(*, preflight=False):
    output = create_output(preflight=preflight)
    p = protocol(preflight)
    manifest = dict(protocol=p, status="pending", methods={}, output_directory=str(output),
                    primary="ASDirect-Full, within-sample average over FB=0/5/10 before inference",
                    secondary="ASDirect-Full at FB25 degradation check; no post-hoc decision threshold",
                    statistics="ddof=1 SE, normal mean +/-1.96 SE. Both sample-level paired and batch-mean paired intervals retained; batch index=sample_index//8. One batch => batch SE/CI unavailable.",
                    ci_scope="Fixed checkpoint evaluation sampling only, not multi-training-seed stability",
                    comparisons="Swin is not an equal-training-budget comparison; shared feedback RNG does not imply equal absolute noise or equal total UE energy.",
                    sampling="Each complete multi-user H_dl/H_ul sample shared by four methods and six FB points; no per-sample oracle or checkpoint reselection.",
                    channel_seed="seed + stable_int(scenario_id) + batch_index",
                    feedback_seed="seed + 100000000 + batch_index*100000 + stable_int(f'{feedback_snr:.6f}')",
                    power="precoder_power sums all frequencies; nominal 256, NOT per-subcarrier power 1",
                    nmse="Raw representation NMSE is a secondary task-surrogate diagnostic, never checkpoint selection")
    hashes, before, methods, summary, samples = {}, {}, [], None, None
    failure = None
    try:
        manifest["runtime"] = runtime_record()
        manifest["plan"] = plan = checkpoint_plan()
        hashes[str(DEFAULT_MANIFEST)] = sha256_file(DEFAULT_MANIFEST)
        import torch
        manifest["runtime"]["backend"] = _backend_info(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        import baseline_evaluate as ev
        manifest["runtime"] = runtime_record(torch)
        scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
        scenario.validate()
        args = evaluation_args(ev, output, p)
        device = torch.device("cuda")
        ev.set_seed(p["seed"])
        parameters, gates = [], []
        for entry in plan:
            key = entry["method_key"]
            if key in LATEST_SHA256:
                latest = load_payload(torch, entry["latest_path"], hashes, entry["expected_latest_sha256"])
                best = load_payload(torch, entry["path"], hashes)  # Best bytes have their own actual hash.
                sidecars = {name: read_json(entry[name + "_path"], hashes) for name in ("config", "status")}
                record = verify_arm_metadata(key, best, latest, sidecars)
                check_tensor_state(torch, latest["state"], best["state"], equal=True)
                record.update(best_latest_state_equal=True, selection_metric_equal=True,
                              latest_sha256=hashes[entry["latest_path"]], latest_path=entry["latest_path"])
                del latest
            else:
                best = load_payload(torch, entry["path"], hashes, entry["expected_best_sha256"])
                record = verify_archived_metadata(entry, best)
            method = build_method(torch, ev, scenario, entry, best, device)
            methods.append(method)
            before[key] = model_buffer_snapshot(method.model)
            validate_frozen(before[key], before[key])
            record.update(model_class=type(method.model).__name__, method_key=key, model_key=method.model_key,
                          best_path=entry["path"], best_sha256=hashes[entry["path"]], strict_loading=True,
                          allocation=entry["allocations"][0], rzf_regularization=1e-3)
            row = dict(method_key=key, full_parameters=sum(x.numel() for x in method.model.parameters()),
                       encoder_plus_condition_parameters=None, note="Parameter count, not latency or MACs")
            if key != "swin":
                record.update(cfg_idx=0, max_dimensions=[64, 256],
                              widths=check_proposal_model(method.model, method.config, 2048))
                row["encoder_plus_condition_parameters"] = sum(x.numel() for x in method.model.encoder.parameters()) + sum(x.numel() for x in method.model.cond_embed.parameters())
            if key == "as_direct":
                for name in GATES:
                    value = float(best["state"][name].item())
                    gates.append(dict(method_key=key, parameter=name, alpha=value, tanh_alpha=math.tanh(value)))
                record["trained_gates"] = list(gates)
            manifest["methods"][key] = record
            parameters.append(row)
            del best
        manifest["input_sha256_before"] = dict(hashes)
        manifest["model_buffers_before"] = before
        _compare(hashes, {path: sha256_file(path) for path in hashes}, "inputs changed during loading")
        ev.write_csv(output / f"parameters_{TAG}.csv", parameters)
        ev.write_csv(output / f"gates_{TAG}.csv", gates)
        ev.set_seed(p["seed"])
        points = [(float(fb), float(DL_SNR)) for fb in FB_SNRS]
        with torch.no_grad():
            summary, samples = ev.evaluate_scenario(scenario, methods, points, args, device)
        deltas, raw_deltas = paired_rows(summary, samples, plan, p["num_samples"], p["seed"])
        for row in summary + samples + deltas + raw_deltas:
            row.update(dataset_role=p["dataset_role"], preflight=preflight, seed=p["seed"])
        ev.save_outputs(summary, samples, [scenario], list(METHODS), points, args)
        ev.write_csv(output / f"paired_deltas_{TAG}.csv", deltas)
        ev.write_gzip_csv(output / f"paired_samples_{TAG}.csv.gz", raw_deltas)
        manifest.update(status="completed", summary_rows=len(summary), sample_rows=len(samples),
                        paired_rows=len(deltas), results_verified=True)
    except Exception as exc:
        failure = exc
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        # Retain returned raw data if postprocessing/verification fails.
        if summary is not None:
            try:
                ev.save_outputs(summary, samples, [scenario], list(METHODS), points, args)
            except Exception as save_error:
                manifest["failure_data_save_error"] = str(save_error)
    finally:
        errors = []
        after_hashes, after_buffers = {}, {}
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
                validate_frozen(before[key], after_buffers[key])
                if key == "as_direct":
                    actual = {name: float(value.detach().cpu().item()) for name, value in method.model.named_parameters() if name in GATES}
                    expected = {row["parameter"]: row["alpha"] for row in manifest["methods"][key]["trained_gates"]}
                    manifest["gate_values_after"] = actual
                    _compare(expected, actual, "trained gates changed during evaluation")
            except Exception as exc:
                errors.append(f"Frozen model check {method.method_key}: {exc}")
        manifest.update(input_sha256_before=dict(hashes), input_sha256_after=after_hashes,
                        model_buffers_before=before, model_buffers_after=after_buffers,
                        freeze_checks=dict(passed=not errors and len(methods) == 4,
                                           errors=errors, checked_methods=[m.method_key for m in methods]))
        if errors or (failure is None and len(methods) != 4):
            manifest["status"] = "failed"
            failure = failure or RuntimeError("Frozen/input checks failed: " + "; ".join(errors))
            manifest.setdefault("error", f"{type(failure).__name__}: {failure}")
        write_manifest(output, manifest)
    if failure is not None:
        raise RuntimeError(f"ASDirect evaluation failed; evidence: {output / 'manifest.json'}") from failure
    print(f"Completed {p['dataset_role']}: 24 conditions x {p['num_samples']} samples; {output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Stdlib plan only; no weights, models or writes")
    mode.add_argument("--preflight", action="store_true", help="CUDA; seed20261027, one batch of eight per condition")
    mode.add_argument("--evaluate", action="store_true", help="CUDA; seed20261026, 200 batches / 1600 samples per condition")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.dry_run or args.preflight or args.evaluate):
        parser.print_help()
        return 0
    if args.dry_run:
        print(json.dumps(dict(plan=checkpoint_plan(), preflight=protocol(True), evaluation=protocol(False),
                              output_root=str(OUTPUT_ROOT), checkpoint_bytes_not_read=True,
                              selection="fixed best checkpoints; no proposed_best/oracle rows",
                              primary="as_direct-full_control: per-sample FB0/5/10 mean difference",
                              uncertainty="both channel-paired and batch-mean-paired CI; preflight batch CI unavailable"), indent=2))
        return 0
    run_evaluation(preflight=args.preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
