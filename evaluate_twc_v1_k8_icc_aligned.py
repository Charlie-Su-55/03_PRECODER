"""Frozen six-gamma monitor/final validation for the independent ICC adaptation.

Default/help/dry-run use the standard library.  Actual model construction and
CUDA execution are explicit and delegated to this batch's own strict loader.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

from twc_v1_k8_icc_aligned_results import (ARMS, GAMMAS, file_sha256,
    make_small_package, write_figures_and_drafts, write_result_tables)
from physical_split_finish_io import write_union_csv

ROOT = Path(__file__).resolve().parent


def _json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii")); digest.update(str(tuple(value.shape)).encode("ascii"))
    # uint8 view works for real/complex tensors without a NumPy dtype guess.
    digest.update(value.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_state(module):
    return {"parameters": {name: _tensor_hash(value) for name, value in module.named_parameters()},
            "buffers": {name: _tensor_hash(value) for name, value in module.named_buffers()}}


def checked_input_hashes(value, *, H_dl, H_ul):
    """Require actual noise/pilot bytes, not equal seeds as a pairing claim."""
    if not isinstance(value, dict):
        raise ValueError("Actual standard-noise and pilot-polarity hashes required")
    for key in ("standard_noise", "pilot_polarity"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Missing or invalid actual tensor hash: " + key)
    return dict(H_dl=H_dl, H_ul=H_ul, standard_noise=value["standard_noise"],
                pilot_polarity=value["pilot_polarity"])


@contextmanager
def frozen_evaluation(models, ce):
    """Freeze inference, then restore monitor callers' modes/default RNG even on failure."""
    import numpy as np
    import torch
    modules = {"ce": ce}
    for arm, (model, statistics) in models.items():
        modules[arm] = model
        modules[arm + ".statistics"] = statistics
    modes = {name: {child: child.training for child in module.modules()}
             for name, module in modules.items()}
    states = {name: _module_state(module) for name, module in modules.items()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
           [state.clone() for state in torch.cuda.get_rng_state_all()])
    try:
        for module in modules.values():
            module.eval()
        with torch.no_grad():
            yield
    finally:
        changed = [name for name, module in modules.items() if _module_state(module) != states[name]]
        bad_mode = [name for name, module in modules.items() if any(child.training for child in module.modules())]
        for local_modes in modes.values():
            for child, training in local_modes.items():
                child.training = training
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
        if changed or bad_mode:
            raise RuntimeError("Frozen model/CE/statistics mutated: " + repr(changed) + "; non-eval=" + repr(bad_mode))


def _vectors(out, H_dl):
    """Offline scoring only; simulator truth never re-enters the decoder."""
    import torch
    from train_twc_v1_k8_icc_aligned import load_local
    downlink = load_local("utils/twc_v1_k8_downlink.py", "_icc_aligned_downlink")
    design_rzf, downlink_nmse, evaluate_sum_rate = downlink.design_rzf, downlink.downlink_nmse, downlink.evaluate_sum_rate
    h = out["H_hat"]
    nmse = downlink_nmse(h, H_dl).mean(dim=1)
    W, power = design_rzf(h, alpha=.03)
    metrics = evaluate_sum_rate(W, H_dl, downlink_snr_db=25.)
    error = (power - 1.).abs().amax(dim=1)
    if not bool((error <= 1e-5).all()):
        raise FloatingPointError("Per-subcarrier total RZF power differs from one by >1e-5")
    result = dict(dl_nmse_linear=nmse, sum_rate_bps_hz=metrics["sum_rate_bps_hz"],
        desired_signal_power_mean=metrics["desired_signal_power_mean"],
        interference_power_mean=metrics["interference_power_mean"], rzf_power_max_error=error)
    for user in range(8):
        result[f"user_{user}_rate_bps_hz"] = metrics["user_rate_bps_hz"][:, user]
    waveform = out["waveform"]
    active, soft = waveform.active, out["soft"][..., :waveform.active.shape[-1]]
    per_user_power = active.abs().square().sum(dim=-1)
    if not bool(torch.isfinite(per_user_power).all() & torch.isclose(
            per_user_power, torch.full_like(per_user_power, 48.), rtol=1e-5, atol=1e-5).all()):
        raise FloatingPointError("Data energy is not48 per user and frame")
    frame_energy = waveform.x_grid.abs().square().sum(dim=-1)
    if not bool(torch.isclose(frame_energy, torch.full_like(frame_energy, 64.), rtol=1e-5, atol=1e-5).all()):
        raise FloatingPointError("Pilot+data frame energy is not64 per user")
    result["data_energy"] = per_user_power.mean(dim=1)
    result["frame_energy"] = frame_energy.mean(dim=1)
    def normalized_error(a, b):
        return (((a - b).abs().square().sum(dim=-1) /
                 b.abs().square().sum(dim=-1).clamp_min(1e-20)).mean(dim=1))
    result["soft_nmse_linear"] = normalized_error(soft, active)
    if active.shape[-1] == 44:
        result["soft_key_nmse_linear"] = normalized_error(soft[..., :4], active[..., :4])
        result["soft_shared_nmse_linear"] = normalized_error(soft[..., 4:], active[..., 4:])
        key_energy = active[..., :4].abs().square().sum(dim=-1)
        shared_energy = active[..., 4:].abs().square().sum(dim=-1)
        result["key_energy"] = key_energy.mean(dim=1)
        result["shared_energy"] = shared_energy.mean(dim=1)
        result["key_energy_user_min"] = key_energy.amin(dim=1)
        result["key_energy_user_max"] = key_energy.amax(dim=1)
        result["shared_energy_user_min"] = shared_energy.amin(dim=1)
        result["shared_energy_user_max"] = shared_energy.amax(dim=1)
        for user in range(8):
            result[f"user_{user}_key_energy"] = key_energy[:, user]
            result[f"user_{user}_shared_energy"] = shared_energy[:, user]
    if "H_ul_est" in out:
        result["estimated_ul_energy"] = out["H_ul_est"].abs().square().mean(dim=(1, 2, 3))
    if any(not bool(torch.isfinite(vector).all()) for vector in result.values()):
        raise FloatingPointError("Nonfinite frozen validation metric")
    return {name: vector.detach().cpu().tolist() for name, vector in result.items()}


def evaluate_models(models, ce, data, *, mode, output_dir, checkpoint_records=None,
                    code_identity=None, forward=None, make_figures=False):
    """Evaluate supplied models on the batch's deterministic monitor/final addresses.

    ``models`` maps arm to (codec, ReceiverStatistics).  Both final methods
    consume the same actual input batch and noise address, not merely an equal
    top-level seed.  An individual arm may be monitored during training.
    """
    import torch
    from configs import twc_v1_k8_icc_aligned_config as cfg
    import train_twc_v1_k8_icc_aligned as runtime
    if mode not in ("monitor", "final"):
        raise ValueError("Only monitor/final frozen scoring is supported")
    if not models or any(arm not in ARMS for arm in models):
        raise ValueError("Unknown/missing ICC-aligned method")
    if mode == "final" and set(models) != set(ARMS):
        raise ValueError("Final validation requires both predeclared methods")
    if forward is None:
        forward = runtime.forward_codec
    output = Path(output_dir).resolve()
    if not output.is_relative_to(cfg.OUTPUT_ROOT.resolve()) or output == cfg.OUTPUT_ROOT.resolve():
        raise ValueError("Evaluation must use a child of the independent ICC-aligned root")
    output.mkdir(parents=True, exist_ok=False)
    records = checkpoint_records or {}
    frozen_files = {}
    for identity in records.values():
        identities = list(identity.get("frozen_files", []))
        if identity.get("path") and identity.get("sha256"):
            identities.append(dict(path=identity["path"], sha256=identity["sha256"]))
        for entry in identities:
            path, digest = Path(entry["path"]), entry["sha256"]
            if file_sha256(path) != digest:
                raise ValueError("Checkpoint/CE input hash changed before validation: " + str(path))
            frozen_files[str(path)] = digest
    calibration_path = cfg.OUTPUT_ROOT / "ce_calibration" / "calibration.json"
    if calibration_path.is_file():
        frozen_files[str(calibration_path)] = file_sha256(calibration_path)
    core = runtime.core_identity()
    runtime.guard_core(core)
    started = time.perf_counter()
    manifest = dict(format="twc_v1_k8_icc_aligned_evaluation_v1", mode=mode, status="running",
        fixed_rzf_alpha=.03, DL_SNR=25., gamma_db=list(GAMMAS), arms=list(models),
        complete_MU_sample_definition="one independent eight-user DL/UL realization",
        sample_and_batch_intervals="mean +/-1.96*SE; ddof=1; complete MU frame; batch4",
        across_gamma="mean within same MU before statistics", CE_and_prior_frozen=True,
        training_seed_replications=1, test_used=False, checkpoints=records,
        code_identity=code_identity or runtime.source_identity(), core=core,
        environment=runtime.environment(), data_protocol=data.protocol,
        no_current_frame_power_side_information=True, alpha_calibration_performed=False)
    rows, fingerprints, prior_rows, parameter_rows = [], [], [], []
    total_batches = cfg.MONITOR_BATCHES if mode == "monitor" else cfg.FINAL_BATCHES
    first = cfg.ROLE_RANGES[mode][0]
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Actual ICC-aligned validation requires CUDA; no CPU fallback")
        torch.cuda.reset_peak_memory_stats()
        with frozen_evaluation(models, ce):
            for arm, (model, statistics) in models.items():
                prior = statistics.nominal_power().detach().cpu()
                prior_rows.extend(dict(arm=arm, user=user, coordinate=coord, prior_power=float(prior[user, coord]))
                                  for user in range(8) for coord in range(prior.shape[-1]))
                encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
                total_parameters = sum(parameter.numel() for parameter in model.parameters())
                parameter_rows.append(dict(arm=arm, total_parameters=total_parameters,
                    UE_encoder_parameters=encoder_parameters, BS_parameters=total_parameters - encoder_parameters,
                    CE_parameters=sum(parameter.numel() for parameter in ce.parameters()),
                    note="stored parameters; not MACs, latency or speedup"))
            for bi in range(total_batches):
                sample = data.sample("validation", first + bi)
                hdl_before, hul_before = _tensor_hash(sample["H_dl"]), _tensor_hash(sample["H_ul"])
                for gi, gamma in enumerate(GAMMAS):
                    for arm, (model, statistics) in models.items():
                        out = forward(model, statistics, ce, sample, gamma_db=gamma,
                                      noise_key=(mode, bi, gi))
                        vectors = _vectors(out, sample["H_dl"])
                        batch = len(vectors["dl_nmse_linear"])
                        if batch != cfg.MICRO_BATCH:
                            raise ValueError("Actual evaluation microbatch changed")
                        for within in range(batch):
                            rows.append(dict(arm=arm, gamma_db=gamma,
                                sample_index=bi * batch + within, batch_index=bi,
                                within_batch=within, **{name: value[within] for name, value in vectors.items()}))
                        hashes = checked_input_hashes(out.get("input_hashes"), H_dl=hdl_before, H_ul=hul_before)
                        fingerprints.append(dict(arm=arm, gamma_db=gamma, batch_index=bi, fingerprint=hashes))
                if (_tensor_hash(sample["H_dl"]) != hdl_before or _tensor_hash(sample["H_ul"]) != hul_before):
                    raise RuntimeError("Paired channel input mutated during frozen validation")
                fingerprints.append(dict(batch_index=bi, address=sample.get("address"),
                    H_dl=hdl_before, H_ul=hul_before, pair=sample.get("sample_fingerprint")))
        # A top-level seed does not prove pairing: all required actual input
        # hashes were collected above and must agree across methods.
        paired_fingerprints = {}
        for entry in fingerprints:
            if "arm" not in entry:
                continue
            key = (entry["batch_index"], entry["gamma_db"])
            if key in paired_fingerprints and paired_fingerprints[key] != entry["fingerprint"]:
                raise RuntimeError("Methods did not use identical channel/standard-noise inputs")
            paired_fingerprints[key] = entry["fingerprint"]
        tables = write_result_tables(output, rows, samples=total_batches * cfg.MICRO_BATCH,
                                     batch_size=cfg.MICRO_BATCH, arms=tuple(models))
        write_union_csv(output / "coordinate_prior.csv", prior_rows)
        write_union_csv(output / "parameters.csv", parameter_rows)
        _json(output / "input_fingerprints.json", fingerprints)
        files = dict(tables["files"])
        for name in ("coordinate_prior.csv", "parameters.csv", "input_fingerprints.json"):
            files[name] = file_sha256(output / name)
        if make_figures:
            if set(models) != set(ARMS):
                raise ValueError("Final paired figures require both arms")
            files.update(write_figures_and_drafts(output, tables["summary"], tables["paired_deltas"]))
        manifest.update(status="passed", samples_per_condition=total_batches * cfg.MICRO_BATCH,
            batch_size=cfg.MICRO_BATCH, n_batches=total_batches, n_conditions=len(models) * len(GAMMAS),
            elapsed_seconds=time.perf_counter() - started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), files=files,
            freezing_verified=True, checkpoint_hashes_unchanged=True,
            outputs={"summary": str(output / "summary.csv"), "paired": str(output / "paired_deltas.csv")})
        for path, digest in frozen_files.items():
            if file_sha256(path) != digest:
                raise RuntimeError("Input checkpoint/CE changed during validation: " + path)
        runtime.guard_core(core)
        _json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc(),
                        elapsed_seconds=time.perf_counter() - started)
        if not (output / "manifest.json").exists():
            _json(output / "manifest.json", manifest)
        raise


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--mode", choices=("monitor", "final"), default="final")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not (args.dry_run or args.run):
        parser.print_help(); return 0
    from configs import twc_v1_k8_icc_aligned_config as cfg
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE, mode=args.mode, arms=ARMS, gamma=GAMMAS,
            samples_per_condition=(cfg.MONITOR_BATCHES if args.mode == "monitor" else cfg.FINAL_BATCHES) * cfg.MICRO_BATCH,
            batch_size=cfg.MICRO_BATCH, fixed_alpha=.03, test_used=False,
            model_loading=False, writes=False), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--run requires an explicit new --output-dir")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    from train_twc_v1_k8_icc_aligned import make_data, load_ce, load_codec, configure_cuda
    device = configure_cuda()
    ce, ce_record = load_ce(device)
    methods, records = {}, {"ce": ce_record}
    for arm in ARMS:
        model, statistics, record = load_codec(arm, device)
        methods[arm], records[arm] = (model, statistics), record
    manifest = evaluate_models(methods, ce, make_data(device), mode=args.mode,
        output_dir=args.output_dir, checkpoint_records=records, make_figures=args.mode == "final")
    print(json.dumps(dict(status=manifest["status"], output=str(args.output_dir.resolve())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
