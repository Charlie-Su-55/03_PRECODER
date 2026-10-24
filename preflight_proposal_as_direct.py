#!/usr/bin/env python3
"""Explicit CUDA-only, real-weight ASDirect preflight. No optimizer or training.

Import, help and --dry-run use only the standard library. The source is the
single archived robust_specialist; --checkpoint relocates identical bytes,
never selects a replacement model. Only --run imports the numerical stack.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback
from types import SimpleNamespace
import uuid


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / (
    "paper/data/diagnostics/k4_low_feedback_snr/"
    "validation256_cudnn_deterministic_20260930T165938_038117Z/diagnostic_manifest.json"
)
OUTPUT_ROOT = ROOT / "runs_as_screen/diagnostics/as_direct_v1"
SEED = 20261024
RTOL = ATOL = 1e-6
EXPECTED_SOURCE_SHA256 = "e21ea1feae595e0898b6be724f7062b05c09b2d1ef03fe9ffec7ac91869064bf"
FULL_GRID = [[64, 0], [48, 64], [32, 128], [16, 192], [0, 256]]
SOURCE_FILES = (
    "preflight_proposal_as_direct.py", "models/adaptive_hybrid.py",
    "models/adaptive_hybrid_as_direct.py", "main_train.py",
    "configs/train_config.py", "utils/layers.py", "utils/gpu_setup.py",
)


def _json_value(value):
    # Saved dataclass snapshots can contain tuples; manifests contain lists.
    return json.loads(json.dumps(value, allow_nan=False))


def _expect(mapping, dotted_path, expected):
    value = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Missing configuration field: {dotted_path}")
        value = value[key]
    if value != expected or (isinstance(expected, (int, float)) and isinstance(value, bool)):
        raise ValueError(f"{dotted_path}: expected {expected!r}, got {value!r}")


def validate_source_config(config):
    """Validate the archived Full singleton, without importing model/config code."""
    if not isinstance(config, Mapping):
        raise ValueError("loaded_checkpoint_config must be an object")
    config = _json_value(config)
    required = {
        "experiment.model_name": "proposal", "experiment.train_mode": "specialist",
        "experiment.k_users": 4, "experiment.subcarriers": 256,
        "experiment.feedback_budget": 256, "experiment.allocation": [0, 256],
        "experiment.recipe_name": "k4_n256_fb_robust_ft", "experiment.seed": 42,
        "experiment.variant": "", "allocations": [[0, 256]],
        "full_allocation_grid": FULL_GRID, "physical.ANTENNAS": 32,
        "physical.NUM_SUBBANDS": 4, "physical.TOTAL_POWER": 1.0,
        "physical.CARRIER_FREQ": 3.5e9, "physical.SPEED": 1.0,
        "proposal_model.D_MODEL": 256, "proposal_model.NUM_HEADS": 8,
        "proposal_model.NUM_ENCODER_LAYERS": 4, "proposal_model.DROPOUT": 0.1,
        "proposal_model.NUM_UNFOLD_LAYERS": 5, "proposal_model.GNN_AGG_DIM": 48,
        "resolved.d_f_max_per_user": 64, "resolved.d_a_max_shared": 256,
        "training_recipe.name": "k4_n256_fb_robust_ft",
        "training_recipe.total_steps": 20000, "training_recipe.downlink_snr_db": 25.0,
        "training_recipe.best_metric": "mean_all",
    }
    for field, expected in required.items():
        _expect(config, field, expected)

    def check_architecture(value, prefix="config"):
        if isinstance(value, dict):
            for key, child in value.items():
                field = f"{prefix}.{key}"
                if key.lower() == "ue_ffn_dim" and (child != 2048 or isinstance(child, bool)):
                    raise ValueError(f"{field}: source must have Full UE FFN2048")
                if key == "architecture" and child not in ("full", "original", "AdaptiveHybridPrecoder"):
                    raise ValueError(f"{field}: not Original Full: {child!r}")
                if key == "model_class" and child != "AdaptiveHybridPrecoder":
                    raise ValueError(f"{field}: not Original Full: {child!r}")
                check_architecture(child, field)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_architecture(child, f"{prefix}[{index}]")

    check_architecture(config)
    return config


def runtime_config(config):
    """Map saved values, not current recipe defaults; never expand GAMMA_GRID."""
    config = validate_source_config(config)
    experiment, physical = config["experiment"], config["physical"]
    return SimpleNamespace(
        K_USERS=experiment["k_users"], ANTENNAS=physical["ANTENNAS"],
        SUBCARRIERS=experiment["subcarriers"], D_TOT=experiment["feedback_budget"],
        NUM_SUBBANDS=physical["NUM_SUBBANDS"], TOTAL_POWER=physical["TOTAL_POWER"],
        GAMMA_GRID=[tuple(point) for point in config["allocations"]],
        D_F_MAX_PER_USER=config["resolved"]["d_f_max_per_user"],
        D_A_MAX_SHARED=config["resolved"]["d_a_max_shared"],
        **{key: config["proposal_model"][key] for key in (
            "D_MODEL", "NUM_HEADS", "NUM_ENCODER_LAYERS", "DROPOUT",
            "NUM_UNFOLD_LAYERS", "GNN_AGG_DIM",
        )},
    )


def load_source_record(manifest_path=DEFAULT_MANIFEST):
    path = Path(manifest_path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Source manifest exceeds the 2 MiB small-JSON limit")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("methods", [])
               if isinstance(item, dict) and item.get("method_key") == "robust_specialist"]
    if len(matches) != 1:
        raise ValueError("Manifest must contain exactly one robust_specialist")
    source = matches[0]
    for field, value in {
        "model_key": "proposed", "method_type": "robust_specialist",
        "training_status": "completed", "training_mode": "fixed_allocation_robust_refinement",
        "allocations": [[0, 256]], "cfg_index": 0, "checkpoint_step": 20000,
    }.items():
        _expect(source, field, value)
    if not isinstance(source.get("path"), str) or not source["path"].strip():
        raise ValueError("robust_specialist.path must be a nonempty checkpoint path")
    if not isinstance(source.get("checkpoint_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", source["checkpoint_sha256"]
    ):
        raise ValueError("robust_specialist.checkpoint_sha256 must be 64 lowercase hex digits")
    if source["checkpoint_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("robust_specialist.checkpoint_sha256 differs from the audited source identity")
    source["loaded_checkpoint_config"] = validate_source_config(source.get("loaded_checkpoint_config"))
    return source


def _sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return _sha256_stream(stream)


def verify_checkpoint_hash(path, expected, *, stream=None):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid expected checkpoint SHA256")
    actual = sha256_file(path) if stream is None else _sha256_stream(stream)
    if actual != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch: expected={expected}, actual={actual}, path={path}")
    return actual


def build_plan(checkpoint=None, *, manifest_path=DEFAULT_MANIFEST):
    source = load_source_record(manifest_path)
    selected = source["path"]
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        selected = str((path if path.is_absolute() else ROOT / path).resolve())
    return {
        "source": source, "checkpoint_path": selected,
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "output_root": str(OUTPUT_ROOT), "mode": "synthetic_cuda_preflight_not_performance_evaluation",
        "weight_read_in_dry_run": False, "checkpoint_present_not_checked_in_dry_run": True,
        "protocol": {
            "seed": SEED, "num_samples": 2, "batch_size": 2,
            "K": 4, "Nt": 32, "Nsc": 256, "Dtot": 256, "Nsb": 4,
            "allocation": [0, 256], "cfg_idx": 0,
            "feedback_snr_db": [0, 10, 25], "downlink_snr_db": 25,
            "rtol": RTOL, "atol": ATOL,
            "rng_seeds": {"inputs": SEED, "train_and_backward": SEED + 1, "eval": SEED + 2},
            "inputs": "Two synthetic complex64 multi-user channels, reused for all conditions",
        },
    }


def _config_differences(actual, expected, prefix="config"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(actual) | set(expected)):
            field = f"{prefix}.{key}"
            if key not in actual or key not in expected:
                differences.append(f"{field}: missing or unexpected")
            else:
                differences.extend(_config_differences(actual[key], expected[key], field))
        return differences
    if actual != expected or (isinstance(actual, bool) != isinstance(expected, bool)):
        return [f"{prefix}: actual={actual!r}, archived={expected!r}"]
    return []


def validate_checkpoint_metadata(payload, source):
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    config = validate_source_config(payload.get("config"))
    differences = _config_differences(config, source["loaded_checkpoint_config"])
    if differences:
        raise ValueError("Checkpoint config differs from manifest: " + "; ".join(differences))
    _expect(payload, "step", source["checkpoint_step"])
    if _json_value(payload.get("allocation_grid")) != [[0, 256]]:
        raise ValueError("Checkpoint allocation_grid must be singleton [(0,256)], not the full grid")
    return config


def create_run_directory(root=OUTPUT_ROOT, *, run_id=None):
    root = Path(root)
    if run_id is None:
        run_id = "preflight_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("run_id must be a safe directory basename")
    root.mkdir(parents=True, exist_ok=True)
    result = root / run_id
    result.mkdir(exist_ok=False)
    return result


def save_report(output_dir, report):
    # Failure diagnostics must survive overflow/NaN without invalid JSON.
    nonfinite = []

    def sanitize(value, field="report"):
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append(field)
            return None
        if isinstance(value, dict):
            return {key: sanitize(child, f"{field}.{key}") for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(child, f"{field}[{index}]") for index, child in enumerate(value)]
        return value

    cleaned = sanitize(report)
    if nonfinite:
        report.update(passed=False, status="nonfinite_report_metrics", nonfinite_report_fields=nonfinite)
        cleaned.update(passed=False, status=report["status"], nonfinite_report_fields=nonfinite)
    encoded = json.dumps(cleaned, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (Path(output_dir) / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "source_hash_convention": "raw file bytes (line endings included)",
        "python": sys.version, "platform": platform.platform(),
    }


def _backend_info(torch):
    return {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "default_dtype": str(torch.get_default_dtype()), "cpu_threads": torch.get_num_threads(),
        "sdp_backends": {name: getattr(torch.backends.cuda, name)() for name in (
            "flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled",
        )},
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF",
            "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        )},
    }


def _comparison(torch, actual, expected):
    result = {"shape": list(actual.shape), "reference_shape": list(expected.shape), "passed": False,
              "max_abs": None, "max_rel": None, "relative_denominator_floor": 1e-12}
    if actual.shape != expected.shape:
        return result
    result["finite"] = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if result["finite"]:
        difference = (actual - expected).abs()
        result.update(max_abs=difference.max().item(),
                      max_rel=(difference / expected.abs().clamp_min(1e-12)).max().item(),
                      outside_tolerance=int((~torch.isclose(actual, expected, rtol=RTOL, atol=ATOL)).sum().item()),
                      passed=bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)))
    return result


def _power_check(torch, value, total_power):
    result = {"passed": False, "max_abs": None, "shape": list(value.shape)}
    if tuple(value.shape) != (2, 32, 4, 256) or not torch.isfinite(value).all():
        return result
    power = value.abs().square().sum(dim=(1, 2))
    result.update(max_abs=(power - total_power).abs().max().item(),
                  passed=bool(torch.allclose(power, torch.full_like(power, total_power), rtol=RTOL, atol=ATOL)))
    return result


def _normalized_rzf(torch, model, h_hat):
    precoder = model.zf_solver(h_hat)
    power = precoder.abs().square().sum(dim=(1, 2), keepdim=True)
    return precoder * torch.sqrt(model.cfg.TOTAL_POWER / (power + 1e-9))


def _assert_source_state(torch, model, state, *, gates=()):
    current = model.state_dict()
    if set(current) != set(state) | set(gates):
        raise ValueError("Model/source state key mismatch (only gate additions are permitted)")
    for name, tensor in state.items():
        saved = current[name]
        if saved.dtype != tensor.dtype or saved.shape != tensor.shape or not torch.equal(saved.detach().cpu(), tensor):
            raise ValueError(f"Source state not exactly preserved: {name}")
    for name in gates:
        tensor = current[name]
        if tensor.shape != () or not torch.isfinite(tensor) or tensor.item() != 0.0:
            raise ValueError(f"Expected zero scalar gate: {name}")


def _buffer_snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def _restore_buffers(torch, model, snapshot):
    current = dict(model.named_buffers())
    if set(current) != set(snapshot):
        raise ValueError("Registered buffer names changed")
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(snapshot[name])


def _rng_comparison(torch, first, second):
    return {
        "cpu": torch.equal(first["torch"], second["torch"]),
        "all_cuda": len(first["cuda"]) == len(second["cuda"]) and all(
            torch.equal(left, right) for left, right in zip(first["cuda"], second["cuda"])
        ),
        "python": first["python"] == second["python"],
        "numpy": (first["numpy"][0] == second["numpy"][0]
                  and first["numpy"][1].tolist() == second["numpy"][1].tolist()
                  and first["numpy"][2:] == second["numpy"][2:]),
    }


def _gradient_report(torch, model):
    gates, missing, nonfinite = {}, [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        if name in model.gate_state_keys:
            gates[name] = (parameter.grad.item() if parameter.grad is not None
                           and torch.isfinite(parameter.grad).all() else None)
    finite_gates = len(gates) == 5 and all(value is not None and math.isfinite(value) for value in gates.values())
    norm = math.hypot(*gates.values()) if finite_gates else None
    return {"gate_gradients": gates, "gate_gradient_norm": norm,
            "missing_gradient_names": missing, "nonfinite_gradient_names": nonfinite,
            "passed": finite_gates and math.isfinite(norm) and norm > 0.0 and not nonfinite}


def _cuda_preflight(plan, report):
    import torch

    report["backend_before_model_import"] = _backend_info(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("ASDirect --run requires CUDA; CPU fallback is prohibited")
    source = plan["source"]
    path = Path(plan["checkpoint_path"])
    # Hash and deserialize the same opened source file, not a re-resolved path.
    with path.open("rb") as stream:
        report["checkpoint_sha256_before"] = verify_checkpoint_hash(
            path, source["checkpoint_sha256"], stream=stream
        )
        stream.seek(0)
        # This is a user-owned, archived, hash-pinned checkpoint (not arbitrary pickle).
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    config = validate_checkpoint_metadata(payload, source)
    report["loaded_checkpoint_config"] = config
    report["checkpoint_step"] = payload["step"]
    state = payload.get("state")
    if not isinstance(state, Mapping) or not isinstance(state.get("running_pwr"), torch.Tensor):
        raise ValueError("Checkpoint state/running_pwr is missing or invalid")
    if tuple(state["running_pwr"].shape) != (1,):
        raise ValueError("Source running_pwr must have length 1 (cfg_idx=0)")
    del payload  # Optimizer/scheduler state, if saved, is never used.

    from models.adaptive_hybrid import AdaptiveHybridPrecoder
    from models.adaptive_hybrid_as_direct import AdaptiveHybridPrecoderASDirect, migrate_original_state_dict
    from main_train import SumRateLoss, capture_rng_state, restore_rng_state, set_seed

    report["backend_effective"] = _backend_info(torch)
    report["software"] = {}
    for package in ("torch", "numpy", "sionna"):
        try:
            report["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["software"][package] = "not recorded by package metadata"
    cfg = runtime_config(config)
    original = AdaptiveHybridPrecoder(cfg)
    direct = AdaptiveHybridPrecoderASDirect(cfg)
    # Validate Original's keys/shapes/dtypes explicitly (strict=True alone permits dtype conversion).
    target = original.state_dict()
    if set(state) != set(target):
        raise ValueError("Original checkpoint key set differs from the Full architecture")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f"Original source shape/dtype mismatch: {name}")
    original.load_state_dict(state, strict=True)
    migrate_original_state_dict(direct, state)
    gates = tuple(f"layers.{index}.alpha" for index in range(5))
    if direct.gate_state_keys != gates:
        raise ValueError("Unexpected ASDirect gate whitelist")
    _assert_source_state(torch, original, state)
    _assert_source_state(torch, direct, state, gates=gates)
    for model in (original, direct):
        for branch in (model.encoder.fdma_backbone, model.encoder.aircomp_backbone):
            if len(branch.layers) != 4 or any(layer.linear1.out_features != 2048 or layer.linear2.in_features != 2048
                                              for layer in branch.layers):
                raise ValueError("Source must use four Full FFN2048 layers per UE branch")
        model.to(torch.device("cuda:0"))
    original_buffers, direct_buffers = _buffer_snapshot(original), _buffer_snapshot(direct)
    if set(original_buffers) != set(direct_buffers) or any(
        not torch.equal(value, direct_buffers[name]) for name, value in original_buffers.items()
    ):
        raise ValueError("Original and ASDirect registered buffers differ after migration")
    added = sum(p.numel() for p in direct.parameters()) - sum(p.numel() for p in original.parameters())
    if added != 5:
        raise ValueError(f"Expected +5 parameters, got {added}")
    report["loading"] = {"original_class": type(original).__name__, "direct_class": type(direct).__name__,
                         "strict": True, "migration": "migrate_original_state_dict", "added_parameters": added,
                         "gate_keys": list(gates), "non_gate_state_exact": True,
                         "ue_ffn_dim": 2048, "running_pwr_shape": [1], "cfg_idx": 0,
                         "max_dimensions": [64, 256], "supported_allocations": [[0, 256]]}
    set_seed(SEED)
    h_dl = torch.randn((2, 4, 32, 256), device="cuda:0", dtype=torch.complex64)
    h_ul = torch.randn_like(h_dl)
    report["synthetic_input_sha256"] = {
        name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in (("H_dl", h_dl), ("H_ul", h_ul))
    }
    criterion = SumRateLoss()
    rows = report["conditions"] = []
    for feedback_snr in plan["protocol"]["feedback_snr_db"]:
        row = {"feedback_snr_db": feedback_snr, "downlink_snr_db": 25, "passed": False,
               "comparisons": {}, "power": {}, "rng": {}, "buffer_changes": {}}
        rows.append(row)
        try:
            for training in (True, False):
                mode = "train" if training else "eval"
                _restore_buffers(torch, original, original_buffers)
                _restore_buffers(torch, direct, direct_buffers)
                original.train(training)
                direct.train(training)
                set_seed(SEED + (1 if training else 2))
                start = capture_rng_state()
                with torch.no_grad():
                    expected = original(h_dl, h_ul, feedback_snr, 0, 256, 0)
                    end_original = capture_rng_state()
                    restore_rng_state(start)
                    actual = direct(h_dl, h_ul, feedback_snr, 0, 256, 0)
                    end_direct = capture_rng_state()
                    row["rng"][mode] = _rng_comparison(torch, end_original, end_direct)
                    if training:
                        row["comparisons"]["H_hat"] = _comparison(torch, actual[1], expected[1])
                        row["comparisons"]["W_train_legacy_interface_only"] = _comparison(torch, actual[0], expected[0])
                        actual_w = _normalized_rzf(torch, direct, actual[1])
                        expected_w = _normalized_rzf(torch, original, expected[1])
                        row["comparisons"]["deployed_RZF"] = _comparison(torch, actual_w, expected_w)
                        row["power"]["direct_deployed_RZF"] = _power_check(torch, actual_w, cfg.TOTAL_POWER)
                        row["power"]["original_deployed_RZF"] = _power_check(torch, expected_w, cfg.TOTAL_POWER)
                    else:
                        row["comparisons"]["W_eval"] = _comparison(torch, actual, expected)
                        row["power"]["direct_eval"] = _power_check(torch, actual, cfg.TOTAL_POWER)
                        row["power"]["original_eval"] = _power_check(torch, expected, cfg.TOTAL_POWER)
                for label, model, snapshot in (("original", original, original_buffers), ("direct", direct, direct_buffers)):
                    changed = [name for name, value in model.named_buffers() if not torch.equal(value, snapshot[name])]
                    row["buffer_changes"][f"{mode}_{label}"] = changed
                    if set(changed) - ({"running_pwr"} if training else set()):
                        raise RuntimeError(f"Unexpected buffer changes in {mode}/{label}: {changed}")
                if any(not torch.equal(value, dict(direct.named_buffers())[name]) for name, value in original.named_buffers()):
                    raise RuntimeError(f"Paired model buffers diverged in {mode}")

            _restore_buffers(torch, direct, direct_buffers)
            direct.train()
            direct.zero_grad(set_to_none=True)
            set_seed(SEED + 1)
            legacy_w, h_hat = direct(h_dl, h_ul, feedback_snr, 0, 256, 0)
            if tuple(h_hat.shape) != (2, 4, 32, 256) or not torch.isfinite(h_hat).all():
                raise FloatingPointError("Invalid H_hat on the task-backward path")
            w = _normalized_rzf(torch, direct, h_hat)
            row["power"]["backward_deployed_RZF"] = _power_check(torch, w, cfg.TOTAL_POWER)
            loss, rate = criterion(w, h_dl, 10.0 ** (-25.0 / 10.0))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite deploy-consistent task loss")
            row["task_loss"] = loss.item()
            row["synthetic_rate_not_performance_result"] = rate
            loss.backward()
            row["backward"] = _gradient_report(torch, direct)
            print(f"FB={feedback_snr} dB gate gradients: " + json.dumps(row["backward"]["gate_gradients"]))
            del legacy_w, h_hat, w, loss
            row["passed"] = (all(item["passed"] for item in row["comparisons"].values())
                             and all(item["passed"] for item in row["power"].values())
                             and all(all(item.values()) for item in row["rng"].values())
                             and row["backward"]["passed"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        finally:
            direct.zero_grad(set_to_none=True)
        print(f"FB={feedback_snr} dB passed={row['passed']}; errors=" + json.dumps(row["comparisons"]))

    # Training forward intentionally updates running_pwr; report then restore it.
    for label, model, snapshot in (("original", original, original_buffers), ("direct", direct, direct_buffers)):
        changed = [name for name, value in model.named_buffers() if not torch.equal(value, snapshot[name])]
        report[f"{label}_final_buffer_changes_before_restore"] = changed
        if set(changed) - {"running_pwr"}:
            raise RuntimeError(f"Unexpected final buffer changes: {label}/{changed}")
        _restore_buffers(torch, model, snapshot)
        _assert_source_state(torch, model, state, gates=gates if label == "direct" else ())
    report["parameters_unchanged_and_buffers_restored"] = True
    report["synthetic_inputs_unchanged"] = all(
        hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        == report["synthetic_input_sha256"][name]
        for name, value in (("H_dl", h_dl), ("H_ul", h_ul))
    )
    report["backend_final"] = _backend_info(torch)
    report["backend_unchanged_during_checks"] = report["backend_final"] == report["backend_effective"]
    torch.cuda.synchronize()
    return (all(row["passed"] for row in rows) and report["backend_unchanged_during_checks"]
            and report["synthetic_inputs_unchanged"])


def run_preflight(checkpoint=None):
    # Fixed output root, no caller-selected arbitrary filesystem destination.
    if not OUTPUT_ROOT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Preflight output root resolves outside 03_PRECODER")
    output_dir = create_run_directory()
    print(f"ASDirect preflight output: {output_dir}", flush=True)
    report = {"passed": False, "status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "formal_evaluation": False, "training": False, "optimizer_created": False,
              "weight_files_written": False, "tolerances": {"rtol": RTOL, "atol": ATOL}}
    plan = None
    try:
        report["provenance"] = _provenance()
        plan = build_plan(checkpoint)
        report["plan"] = plan
        report["passed"] = bool(_cuda_preflight(plan, report))
        report["status"] = "passed" if report["passed"] else "checks_failed"
    except Exception as exc:
        report.update(passed=False, status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        if plan is not None and "checkpoint_sha256_before" in report:
            try:
                report["checkpoint_sha256_after"] = verify_checkpoint_hash(
                    plan["checkpoint_path"], plan["source"]["checkpoint_sha256"]
                )
            except Exception as exc:
                report.update(passed=False, status="checkpoint_integrity_failed", integrity_error=str(exc))
        save_report(output_dir, report)
    print(f"ASDirect CUDA preflight: {report['status']}; report={output_dir / 'report.json'}")
    if not report["passed"]:
        print(report.get("error", "See per-condition comparison/backward evidence in report.json"), file=sys.stderr)
    return 0 if report["passed"] else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Standard-library plan only; no weight read or output")
    mode.add_argument("--run", action="store_true", help="Explicit CUDA real-checkpoint synthetic preflight; no training")
    parser.add_argument("--checkpoint", help="Exact relocated source file; must match archived SHA256; no search")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.dry_run or args.run):
        parser.print_help()
        return 0
    if args.dry_run:
        try:
            print(json.dumps(build_plan(args.checkpoint), indent=2, allow_nan=False))
            return 0
        except Exception as exc:
            print(f"Dry-run failed: {exc}", file=sys.stderr)
            return 1
    return run_preflight(checkpoint=args.checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
