#!/usr/bin/env python3
"""New equal-UE-energy / absolute-N0 grouped-codec evaluation, never old FB-SNR.

Help and dry-run are standard-library only. Actual evaluation requires CUDA.
Five layouts share each actual channel pair and full standard-noise tensor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import traceback

from configs.grouped_codec_train_config import ARMS, SEEDS, grouped_codec_config
from configs.as_spatial_train_config import channel_data_api
import evaluate_proposal_as_receiver as shared
from evaluate_proposal_as_direct import check_tensor_state, load_payload, read_json, sample_statistics, write_manifest
from evaluate_proposal_lite import _compare, model_buffer_snapshot, validate_frozen, validate_output_path
from preflight_proposal_as_direct import sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_grouped_codec_v1"
TAG = "k4_grouped_codec_v1"
METHODS = ("legacy_all4", "ordered_all4", "legacy_group2", "ordered_group2", "ordered_fdma")
GAMMAS, TEST_SEED, PREFLIGHT_SEED = (0, 5, 10, 15, 20, 25), 20261210, 20261211
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
METRICS = ("sum_rate", "nmse_db", "precoder_power", "precoder_power_max_error",
           "energy_ue0", "energy_ue1", "energy_ue2", "energy_ue3", "empirical_noise_power",
           "recovered_mse_active", "recovered_nmse_linear_active", "recovered_nmse_db_active",
           "corrected_mse_active", "corrected_nmse_linear_active", "corrected_nmse_db_active",
           "reliability_mean_active", "receiver_regularization")
OUTPUT_ARTIFACTS = tuple(f"{name}_{TAG}.{suffix}" for name, suffix in (
    ("summary", "csv"), ("samples", "csv.gz"), ("results", "json"), ("metadata", "json"),
    ("paired_deltas", "csv"), ("paired_samples", "csv.gz"), ("parameters", "csv")))
SOURCE_FILES = (
    "evaluate_grouped_codec.py", "configs/grouped_codec_train_config.py", "train_grouped_codec.py",
    "models/grouped_codec_precoder.py", "utils/grouped_feedback_protocol.py", "main_train.py",
    "models/adaptive_hybrid.py", "models/adaptive_hybrid_joint_swin.py", "models/swin_precoder.py",
    "models/baseline_common.py", "baseline_evaluate.py", "utils/replayable_channel_data.py",
    "utils/sionna_rng.py", "utils/channel_generator.py", "utils/layers.py",
    "evaluate_proposal_as_receiver.py", "evaluate_proposal_as_direct.py", "evaluate_proposal_lite.py",
    "preflight_joint_swin.py",
)


def physical_api():
    return channel_data_api()._load_source("utils/grouped_feedback_protocol.py", "_grouped_feedback_protocol_v1")


def make_test_data(preflight=False):
    return channel_data_api().ReplayableChannelData(base_seed=PREFLIGHT_SEED if preflight else TEST_SEED,
        k_users=4, nt=32, nsc=256, batch_size=8, validation_batch_size=8,
        carrier_freq=3.5e9, speed=1.0, cdl_model="C", device="cuda:0")


def layout_metadata(arm):
    spec = physical_api().layout_for_arm(arm)
    return spec.to_dict()


def layout_name(arm):
    return "group2" if arm.endswith("group2") else "fdma" if arm.endswith("fdma") else "all4"


def expected_n0(gamma_db):
    if not math.isfinite(float(gamma_db)):
        raise ValueError("Nonfinite gamma")
    return .25 * 10 ** (-float(gamma_db) / 10)


def protocol(training_seed, preflight=False):
    shared.validate_seed(training_seed)
    return dict(training_seed=training_seed, seed=PREFLIGHT_SEED if preflight else TEST_SEED,
        preflight=preflight, batch_size=8, num_batches=1 if preflight else 200,
        num_samples=8 if preflight else 1600, gamma_db=list(GAMMAS), downlink_snr_db=25,
        scenario=dict(K=4, Nt=32, Nsc=256, occupied_feedback_RE=256), channel_profile="CDL-C",
        uplink_csi="perfect", energy_per_frame_per_ue=64,
        gamma_definition="10*log10(64/(256*N0)); transmit block reference, NOT old received feedback SNR",
        N0_definition="0.25*10**(-gamma_db/10); identical absolute variance for every method",
        standard_noise="complex eps with independent unit-Normal real/imag; multiply sqrt(N0/2) exactly once",
        noise_address=dict(split="test", index="zero-based batch_index", seed=PREFLIGHT_SEED if preflight else TEST_SEED,
                           omits=["arm", "training_seed", "gamma"]),
        channel_data_protocol=make_test_data(preflight).protocol,
        channel_address="training(batch_index+1), independent fixed test base_seed; DL/UL continuous pair",
        active_layouts={arm: layout_metadata(arm) for arm in METHODS},
        physical_protocol=physical_api().protocol_metadata(),
        numerical_policy=dict(real_dtype="float32", complex_dtype="complex64", matmul_precision="highest",
                              allow_tf32=False, amp=False, microbatch=False),
        dataset_role="preflight_not_screening_result" if preflight else "independent_screening_test",
        energy_absolute_tolerance=1e-4, per_subcarrier_power_absolute_tolerance=1e-5,
        training_seeds_pooled=False, old_protocol_curves_allowed=False)


def checkpoint_plan(training_seed):
    shared.validate_seed(training_seed)
    entries = []
    for arm in METHODS:
        cfg = grouped_codec_config(arm, training_seed)
        entries.append(dict(method_key=arm, arm=arm, training_seed=training_seed, path=cfg.BEST_PATH,
            latest_path=cfg.SAVE_PATH, config_path=cfg.CONFIG_PATH, status_path=cfg.STATUS_PATH,
            active_layout=layout_name(arm), layout_spec=layout_metadata(arm)))
    if len({entry["path"] for entry in entries}) != 5:
        raise ValueError("Five independent checkpoints required")
    return entries


def verify_training_metadata(arm, seed, best, latest, sidecars):
    from train_grouped_codec import fingerprint, validate_resume_payload, validate_best_metadata
    cfg = grouped_codec_config(arm, seed)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    _compare(20000, latest.get("step"), "Completed random-initialization budget")
    if type(best.get("step")) is not int or not 0 < best["step"] <= 20000 or best["step"] % 500:
        raise ValueError("Best must be a positive 500-update validation boundary")
    _compare(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg)), sidecars["config"], "Full sidecar config")
    for name, value in dict(state="completed", step=20000, experiment=cfg.EXP_NAME,
                             best_metric_name="mean_all", best_metric=best["best_metric"]).items():
        _compare(value, sidecars["status"].get(name), "Training status." + name)
    return dict(config=best["config"], config_fingerprint=best["config_fingerprint"],
                selected_best_step=best["step"], latest_step=latest["step"], completed_updates=20000,
                best_metric=best["best_metric"], training_status=sidecars["status"],
                random_initialization=True, historical_weights_loaded=False)


def build_model(torch, arm, seed, payload, device):
    from models.grouped_codec_precoder import GroupedCodecPrecoder
    from train_grouped_codec import validate_model_identity, validate_model_state
    cfg = grouped_codec_config(arm, seed)
    _compare(cfg.to_dict(), payload["config"], "Loaded complete configuration")
    model = GroupedCodecPrecoder(cfg)
    validate_model_identity(cfg, model)
    validate_model_state(torch, model, payload["state"])
    check_tensor_state(torch, model.state_dict(), payload["state"])
    model.load_state_dict(payload["state"], strict=True)
    check_tensor_state(torch, model.state_dict(), payload["state"], equal=True)
    model.to(device).eval()
    return model, cfg


def sample_identity(method, gamma, index, seed, training_seed):
    active = 256 if method.endswith("all4") else 128 if method.endswith("group2") else 64
    return dict(method_key=method, gamma_db=gamma, downlink_snr_db=25, active_layout=layout_name(method),
                occupied_feedback_RE=256, active_symbols_per_ue=active, active_users_per_RE=active // 64,
                noise_variance_N0=expected_n0(gamma), sample_index=index, batch_index=index // 8,
                seed=seed, training_seed=training_seed)


def summarize_samples(samples, count, seed, training_seed):
    if type(count) is not int or count < 8 or count % 8:
        raise ValueError("Expected complete eight-channel batches")
    indexed = {(arm, gamma): {} for arm in METHODS for gamma in GAMMAS}
    for row in samples:
        key, index = (row.get("method_key"), row.get("gamma_db")), row.get("sample_index")
        if key not in indexed or type(index) is not int or not 0 <= index < count or index in indexed[key]:
            raise ValueError("Invalid/duplicate method-gamma-sample key")
        if "feedback_snr_db" in row or "allocation_df" in row or "allocation_da" in row:
            raise ValueError("Old feedback-SNR/allocation fields forbidden in this protocol")
        for name, value in sample_identity(key[0], key[1], index, seed, training_seed).items():
            _compare(value, row.get(name), "Sample identity." + name)
        if any(not math.isfinite(float(row[name])) for name in METRICS):
            raise ValueError("Missing/nonfinite metric")
        if any(abs(float(row[f"energy_ue{k}"]) - 64) > 1e-4 for k in range(4)):
            raise ValueError("Per-frame per-UE energy deviates from64")
        if not 0 <= row["precoder_power_max_error"] <= 1e-5:
            raise ValueError("Per-subcarrier power constraint violated")
        if abs(row["precoder_power"] - 256) > 256e-5:
            raise ValueError("Full-band precoder power should be256, not1")
        if not 0 <= row["reliability_mean_active"] <= 1:
            raise ValueError("Invalid active reliability proxy")
        if row["empirical_noise_power"] < 0:
            raise ValueError("Negative empirical noise power")
        ridge = row["noise_variance_N0"] / (64 / row["active_symbols_per_ue"]) + 1e-6
        if not math.isclose(row["receiver_regularization"], ridge, rel_tol=1e-6, abs_tol=1e-10):
            raise ValueError("Receiver ridge inconsistent with absoluteN0 and active coordinate energy")
        if any(row[name] < 0 for name in METRICS if "_mse_" in name or "_linear_" in name):
            raise ValueError("Negative squared/linear diagnostic error")
        indexed[key][index] = row
    summary = []
    for key, records in indexed.items():
        if set(records) != set(range(count)):
            raise ValueError("Missing test samples")
        row = sample_identity(*key, 0, seed, training_seed)
        row.pop("sample_index"); row.pop("batch_index")
        row.update(num_samples=count, num_batches=count // 8, batch_size=8, rzf_regularization=1e-3)
        for metric in METRICS:
            values = [records[i][metric] for i in range(count)]
            row.update({metric + "_" + name: value for name, value in sample_statistics(values).items()})
        summary.append(row)
    # Same standard noise and same absolute N0, not just the same seed label.
    for gamma in GAMMAS:
        for i in range(count):
            reference = indexed[METHODS[0], gamma][i]["empirical_noise_power"]
            if any(not math.isclose(indexed[arm, gamma][i]["empirical_noise_power"], reference,
                                    rel_tol=1e-6, abs_tol=1e-10) for arm in METHODS[1:]):
                raise ValueError("Physical noise differs across methods")
    return summary, indexed


def paired_rows(summary, samples, count, seed, training_seed):
    actual, indexed = summarize_samples(samples, count, seed, training_seed)
    provided = {(row["method_key"], row["gamma_db"]): row for row in summary}
    if len(provided) != len(summary) or len(summary) != 30:
        raise ValueError("Exactly30 unique summary conditions required")
    for row in actual:
        candidate = provided.get((row["method_key"], row["gamma_db"]), {})
        for name, value in row.items():
            if isinstance(value, float):
                if not math.isfinite(float(candidate.get(name, math.nan))) or not math.isclose(value, float(candidate[name]), rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("Summary/raw mismatch: " + name)
            else:
                _compare(value, candidate.get(name), "Summary/raw." + name)
    rows, raw = [], []
    scopes = [("per_gamma", (g,)) for g in GAMMAS] + [("low_gamma_mean", (0, 5, 10)), ("all_gamma_mean", GAMMAS)]
    for contrast_id, coefficients in CONTRASTS.items():
        for estimand, gammas in scopes:
            values = [statistics.fmean(sum(coefficient * indexed[arm, gamma][i]["sum_rate"]
                       for arm, coefficient in coefficients.items()) for gamma in gammas) for i in range(count)]
            batches = [statistics.fmean(values[i:i + 8]) for i in range(0, count, 8)]
            row = dict(contrast_id=contrast_id, coefficients=json.dumps(coefficients, sort_keys=True), estimand=estimand,
                gamma_points=",".join(map(str, gammas)), gamma_db=gammas[0] if len(gammas) == 1 else None,
                downlink_snr_db=25, seed=seed, training_seed=training_seed, num_samples=count,
                num_batches=count // 8, batch_size=8, batch_ci_estimable=count > 8, training_seeds_pooled=False,
                sample_ci_unit="complete_multi_user_channel", batch_ci_unit="sample_index//8",
                role="primary" if contrast_id == "D_minus_E" and estimand == "low_gamma_mean" else
                     "secondary25" if contrast_id == "D_minus_E" and gammas == (25,) else "diagnostic")
            row.update({"delta_" + name: value for name, value in sample_statistics(values).items()})
            row.update({"batch_delta_" + name: value for name, value in sample_statistics(batches).items()})
            rows.append(row)
            raw.extend(dict(contrast_id=contrast_id, estimand=estimand, gamma_points=row["gamma_points"],
                            sample_index=i, batch_index=i // 8, delta=value, seed=seed, training_seed=training_seed)
                       for i, value in enumerate(values))
    return rows, raw


def artifact_hashes(output):
    return shared.artifact_hashes(output, expected_artifacts=OUTPUT_ARTIFACTS)


def runtime_record(torch=None):
    return shared.runtime_record(torch, source_files=SOURCE_FILES)


def extract_diagnostics(torch, diagnostics, noise, gamma, arm):
    names = ("z_hat", "z_corrected", "true_z", "reliability", "energy", "noise_power")
    for name in names:
        value = diagnostics[name]
        if not torch.is_tensor(value) or value.requires_grad or not torch.isfinite(value).all().item():
            raise ValueError("Detached finite diagnostic required: " + name)
    z, corrected, truth, reliability = (diagnostics[name] for name in names[:4])
    if any(tuple(value.shape) != (8, 4, 256) for value in (z, corrected, truth, reliability)):
        raise ValueError("Canonical diagnostic shape must be[B,4,256]")
    mask = diagnostics["active_mask"]
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or tuple(mask.shape) != (4, 256):
        raise ValueError("Explicit canonical active mask required")
    expected_mask = torch.tensor(physical_api().layout_for_arm(arm).canonical_to_physical,
                                 device=mask.device) >= 0
    if not torch.equal(mask, expected_mask):
        raise ValueError("Canonical activity mask differs from the declared physical layout")
    inactive = ~mask.unsqueeze(0).expand_as(z)
    if any(torch.count_nonzero(value[inactive]).item() for value in (z, corrected, truth, reliability)):
        raise ValueError("Inactive canonical slots must remain strictly zero")
    active = mask.unsqueeze(0).expand_as(z)
    denominator = truth.abs().square().masked_fill(~active, 0).sum(dim=(1, 2)).clamp_min(1e-12)
    active_count = int(mask.sum().item())
    values = {}
    for label, value in (("recovered", z), ("corrected", corrected)):
        squared = (value - truth).abs().square().masked_fill(~active, 0).sum(dim=(1, 2))
        ratio = squared / denominator
        values.update({label + "_mse_active": squared / active_count,
                       label + "_nmse_linear_active": ratio,
                       label + "_nmse_db_active": 10 * torch.log10(ratio.clamp_min(1e-12))})
    energy = diagnostics["energy"]
    if tuple(energy.shape) != (8, 4) or tuple(diagnostics["noise_power"].shape) != (8,):
        raise ValueError("Energy/noise diagnostic dimensions differ")
    if not torch.allclose(energy, truth.abs().square().sum(-1), rtol=0, atol=1e-4):
        raise ValueError("Reported per-UE energy differs from transmitted canonical codewords")
    N0 = expected_n0(gamma)
    value = diagnostics["N0"]
    actual_n0 = float(value.item()) if torch.is_tensor(value) else float(value)
    if not math.isclose(N0, actual_n0, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("Model used an unexpected absolute N0")
    ridge = diagnostics["lambda_rx"]
    if not torch.is_tensor(ridge) or ridge.requires_grad or ridge.numel() != 1:
        raise ValueError("Detached public receiver regularization scalar required")
    expected_ridge = N0 / (64 / physical_api().layout_for_arm(arm).active_per_user) + 1e-6
    if not math.isclose(float(ridge.item()), expected_ridge, rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError("Recovery did not use declared N0/nominal-coordinate-power regularization")
    empirical = (noise * math.sqrt(N0 / 2)).abs().square().mean(dim=(1, 2))
    if not torch.allclose(empirical, diagnostics["noise_power"], rtol=1e-6, atol=1e-10):
        raise ValueError("Physical noise was rescaled or resampled")
    values.update({f"energy_ue{k}": energy[:, k] for k in range(4)})
    values.update(empirical_noise_power=diagnostics["noise_power"],
                  receiver_regularization=ridge.expand(8),
                  reliability_mean_active=reliability.masked_fill(~active, 0).sum(dim=(1, 2)) / active_count)
    return values


def run_evaluation(training_seed, *, preflight=False, output_dir=None):
    shared.validate_seed(training_seed)
    output = shared.create_output(training_seed, preflight=preflight, output_dir=output_dir, root=OUTPUT_ROOT)
    p = protocol(training_seed, preflight)
    manifest = dict(status="pending", protocol=p, plan=checkpoint_plan(training_seed), output_directory=str(output), methods={},
        shared_batch_inputs=[],
        ci_scope="Fixed-checkpoint sampling; separate training seeds, not3200 independent samples",
        latent_scope="Active-only latent errors diagnostic within each representation, not common CSI error",
        layout_scope="Layout also changes code length, coordinate energy and separation; not pure demixing gain",
        budget_scope="20k updates do not establish convergence; E is matched Ordered-UE Joint-FDMA, not historical Swin")
    hashes, snapshots, models, samples, execution = {}, {}, {}, [], {}
    failure = None
    try:
        manifest["runtime"] = runtime_record()
        import torch
        manifest["runtime_before_backend"] = runtime_record(torch)
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback forbidden")
        from train_grouped_codec import apply_backend
        apply_backend(torch)
        manifest["runtime"] = runtime_record(torch)
        from preflight_joint_swin import tensor_digest
        import baseline_evaluate as ev
        device = torch.device("cuda:0")
        ev.set_seed(p["seed"])
        parameters = []
        for item in checkpoint_plan(training_seed):
            arm = item["arm"]
            latest = load_payload(torch, item["latest_path"], hashes)
            best = load_payload(torch, item["path"], hashes)
            sidecars = {name: read_json(item[name + "_path"], hashes) for name in ("config", "status")}
            record = verify_training_metadata(arm, training_seed, best, latest, sidecars)
            check_tensor_state(torch, latest["state"], best["state"], equal=latest["step"] == best["step"])
            model, cfg = build_model(torch, arm, training_seed, best, device)
            from train_grouped_codec import validate_model_state
            validate_model_state(torch, model, latest["state"])
            models[arm] = model, cfg
            snapshots[arm] = model_buffer_snapshot(model)
            validate_frozen(snapshots[arm], snapshots[arm])
            record.update(model_class=type(model).__name__, best_path=item["path"], latest_path=item["latest_path"],
                best_sha256=hashes[item["path"]], latest_sha256=hashes[item["latest_path"]], strict_loading=True,
                active_layout=item["active_layout"], layout_spec=item["layout_spec"], architecture_metadata=model.architecture_metadata)
            manifest["methods"][arm] = record
            parameters.append(dict(method_key=arm, training_seed=training_seed,
                full_parameters=sum(value.numel() for value in model.parameters()),
                ue_parameters=sum(value.numel() for value in model.ue_parameters()),
                bs_parameters=sum(value.numel() for value in model.bs_parameters()),
                note="Stored parameters include any inactive legacy FDMA branch; not MACs or latency"))
            del latest, best
        _compare(hashes, {path: sha256_file(path) for path in hashes}, "Inputs changed during loading")
        ev.write_csv(output / f"parameters_{TAG}.csv", parameters)
        data = make_test_data(preflight).prepare()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for batch_index in range(p["num_batches"]):
                H_dl, H_ul = data.training(batch_index + 1)
                noise = physical_api().standard_noise(p["seed"], "test", batch_index, 8, device=str(device))
                original_inputs = tuple(value.clone() for value in (H_dl, H_ul, noise))
                manifest["shared_batch_inputs"].append(dict(batch_index=batch_index,
                    sample_index_start=batch_index * 8, sample_index_stop=(batch_index + 1) * 8,
                    **{name + "_sha256": tensor_digest({name: value}) for name, value in
                       zip(("H_dl", "H_ul", "standard_noise"), original_inputs)}))
                for gamma in GAMMAS:
                    for arm, (model, _) in models.items():
                        result = model.forward_with_diagnostics(H_dl, H_ul, float(gamma), standard_noise=noise)
                        W, H_hat, diagnostics = result["precoder"], result["channel_surrogate"], result["diagnostics"]
                        if not torch.isfinite(W).all().item() or not torch.isfinite(H_hat).all().item():
                            raise ValueError("Nonfinite deployed output")
                        values = extract_diagnostics(torch, diagnostics, noise, gamma, arm)
                        values.update(sum_rate=ev.per_sample_sum_rate(W, H_dl, 10 ** (-25 / 10)),
                            nmse_db=ev.per_sample_nmse_db(H_hat, H_dl), precoder_power=ev.per_sample_precoder_power(W),
                            precoder_power_max_error=(W.abs().square().sum(dim=(1, 2)) - 1).abs().amax(dim=1))
                        arrays = {name: value.detach().cpu().tolist() for name, value in values.items()}
                        for i in range(8):
                            samples.append(dict(sample_identity(arm, gamma, batch_index * 8 + i, p["seed"], training_seed),
                                                **{name: value[i] for name, value in arrays.items()}))
                        actual = diagnostics.get("execution")
                        if not isinstance(actual, dict) or not actual:
                            raise ValueError("Actual model execution shape record missing")
                        identity = arm + ":" + str(gamma)
                        saved = execution.setdefault(identity, dict(method_key=arm, gamma_db=gamma, calls=0, runtime=actual))
                        _compare(saved["runtime"], actual, "Runtime shape/layout changed across batches")
                        saved["calls"] += 1
                if not all(torch.equal(before, after) for before, after in zip(original_inputs, (H_dl, H_ul, noise))):
                    raise ValueError(f"A model modified shared channel/noise inputs in batch{batch_index}")
        summary, _ = summarize_samples(samples, p["num_samples"], p["seed"], training_seed)
        paired, raw_paired = paired_rows(summary, samples, p["num_samples"], p["seed"], training_seed)
        for row in summary + samples + paired + raw_paired:
            row.update(preflight=preflight, dataset_role=p["dataset_role"])
        ev.write_csv(output / f"summary_{TAG}.csv", summary)
        ev.write_gzip_csv(output / f"samples_{TAG}.csv.gz", samples)
        ev.write_csv(output / f"paired_deltas_{TAG}.csv", paired)
        ev.write_gzip_csv(output / f"paired_samples_{TAG}.csv.gz", raw_paired)
        for name, value in (("results", dict(summary=summary, paired=paired)), ("metadata", p)):
            with (output / f"{name}_{TAG}.json").open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
        manifest.update(status="completed", results_verified=True, summary_rows=len(summary), paired_rows=len(paired),
                        shared_channel_noise_inputs_unchanged=True,
                        diagnostic_execution_records=list(execution.values()),
                        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device), scope="all five evaluation models; not training peak"))
    except Exception as exc:
        failure = exc
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if samples:
            try: ev.write_gzip_csv(output / f"partial_samples_{TAG}.csv.gz", samples)
            except Exception as save_error: manifest["partial_save_error"] = str(save_error)
    finally:
        errors, after_hashes, after_buffers = [], {}, {}
        for path, expected in hashes.items():
            try:
                after_hashes[path] = sha256_file(path)
                if after_hashes[path] != expected: errors.append("Input changed: " + path)
            except Exception as exc: errors.append(str(exc))
        for arm, (model, cfg) in models.items():
            try:
                after_buffers[arm] = model_buffer_snapshot(model)
                validate_frozen(snapshots[arm], after_buffers[arm])
                from train_grouped_codec import validate_model_identity
                validate_model_identity(cfg, model)
            except Exception as exc: errors.append(arm + ":" + str(exc))
        manifest.update(input_sha256_before=hashes, input_sha256_after=after_hashes,
                        model_buffers_before=snapshots, model_buffers_after=after_buffers,
                        freeze_checks=dict(passed=not errors and len(models) == 5, errors=errors))
        if errors or (failure is None and len(models) != 5):
            failure = failure or RuntimeError("Frozen/input checks failed: " + ";".join(errors))
            manifest.update(status="failed", error=str(failure))
        if failure is None:
            try: manifest["output_sha256"] = artifact_hashes(output)
            except Exception as exc:
                failure = exc
                manifest.update(status="failed", error=str(exc))
        write_manifest(output, manifest)
    if failure is not None:
        raise RuntimeError(f"Grouped-codec evaluation failed; evidence: {output / 'manifest.json'}") from failure
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for option in ("dry-run", "preflight", "evaluate"): modes.add_argument("--" + option, action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS); parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not (args.dry_run or args.preflight or args.evaluate): parser.print_help(); return 0
    if args.seed is None: parser.error("Explicit training seed required")
    if args.dry_run:
        if args.output_dir is not None: validate_output_path(args.output_dir.absolute(), OUTPUT_ROOT)
        print(json.dumps(dict(plan=checkpoint_plan(args.seed), evaluation=protocol(args.seed), preflight=protocol(args.seed, True),
            models_not_imported=True, checkpoint_bytes_not_read=True, old_protocol_curves_allowed=False), indent=2))
        return 0
    run_evaluation(args.seed, preflight=args.preflight, output_dir=args.output_dir)
    return 0


if __name__ == "__main__": raise SystemExit(main())
