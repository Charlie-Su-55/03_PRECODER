#!/usr/bin/env python3
"""Explicit CUDA, channel-only acceptance for isolated Sionna 2.0.1 RNG.

Help and --dry-run use only the standard library. No models, checkpoint I/O,
training, or performance evaluation are imported or run by this tool.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import stat
import subprocess
import sys
import traceback
import uuid


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/diagnostics/sionna_rng_v1"
SEED = 20261028
ATOL = RTOL = 1e-6
SHAPE = (2, 4, 32, 256)
SOURCE_FILES = (
    "diagnose_sionna_rng.py", "utils/sionna_rng.py", "utils/channel_generator.py",
)
WORKER_MARKER = "SIONNA_RNG_WORKER_JSON:"


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    return seed


def plan(seed=SEED):
    seed = validate_seed(seed)
    return {
        "mode": "channel-only CUDA acceptance; no model or checkpoint",
        "supported_sionna": "2.0.1", "seed": seed,
        "additional_seeds": [(seed + offset) % 2**32 for offset in (1, 2, 3)],
        "channel": {"K": 4, "Nt": 32, "Nsc": 256, "batch_size": 2,
                    "profile": "C", "carrier_frequency": 3.5e9, "speed": 1.0,
                    "generator": "CDLChannelGenerator_Feedback",
                    "sampling": "one generate_batch_data call; consecutive DL then UL"},
        "output_root": str(OUTPUT_ROOT), "rtol": RTOL, "atol": ATOL,
        "acceptance": ["same-seed reused and reconstructed generator", "different seeds",
                       "interleaved validation and generation order", "fresh Python process",
                       "normal, exceptional and nested state/stream restoration, including seed=None"],
        "scope_limit": "Channel reproducibility only, not full training checkpoint recovery.",
        "dry_run_reads_weights": False, "dry_run_creates_output": False,
    }


def create_output(root=OUTPUT_ROOT, *, run_id=None):
    root = Path(root).absolute()
    if run_id is None:
        run_id = "channels_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("run_id must be a single safe directory name")
    output = root / run_id
    for part in (output, *output.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Output symlink/junction refused: {part}")
        if part.exists():
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
            if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError(f"Output reparse-point ancestor refused: {part}")
    if output.resolve().parent != root.resolve():
        raise ValueError("Output must be an immediate child of the diagnostic root")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _load_file(relative, name):
    # Deliberately avoid utils/__init__.py, which changes GPU settings.
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # An actual source module, not a dependency substitute.
    spec.loader.exec_module(module)
    return module


def _source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _environment():
    versions = {}
    for name in ("sionna", "torch", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    return {"python": sys.version, "platform": platform.platform(), "versions": versions,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_sha256": _source_hashes()}


def _json_safe(value, nonfinite, path="report"):
    """Keep failed numerical evidence writable without treating it as a PASS."""
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append({"field": path, "value": str(value)})
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item, nonfinite, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, nonfinite, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def write_report(output, report):
    nonfinite = []
    safe = _json_safe(report, nonfinite)
    if nonfinite:
        report.update(status="failed", passed=False, nonfinite_fields=nonfinite)
        safe.update(status="failed", passed=False, nonfinite_fields=nonfinite)
    path = Path(output) / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(safe, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return safe


def _backend(torch):
    return {
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "cuda_devices": torch.cuda.device_count(),
        "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "environment": {name: os.environ.get(name) for name in (
            "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
            "NVIDIA_TF32_OVERRIDE", "PYTHONHASHSEED")},
    }


def _tensor_record(tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _pair_record(pair):
    return {name: _tensor_record(tensor) for name, tensor in zip(("H_dl", "H_ul"), pair)}


def _compare(torch, reference, actual):
    results = {}
    for name, left, right in zip(("H_dl", "H_ul"), reference, actual):
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            results[name] = {"passed": False, "reason": "shape or dtype mismatch",
                             "reference": _tensor_record(left), "actual": _tensor_record(right)}
            continue
        difference = (left - right).abs().to(torch.float64)
        denominator = left.abs().to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
        left_info, right_info = _tensor_record(left), _tensor_record(right)
        results[name] = {
            "passed": bool(torch.allclose(left, right, rtol=RTOL, atol=ATOL))
                      and left_info["sha256"] == right_info["sha256"],
            "exact_hash_match": left_info["sha256"] == right_info["sha256"],
            "within_tolerance": bool(torch.allclose(left, right, rtol=RTOL, atol=ATOL)),
            "max_abs": difference.max().item(),
            "max_rel": (difference / denominator).max().item(),
            "reference": left_info, "actual": right_info,
        }
    return {"passed": all(item["passed"] for item in results.values()), "channels": results,
            "rtol": RTOL, "atol": ATOL, "requires_exact_hash_match": True}


def _make_generator(channel_class):
    return channel_class(Nt=32, Nsc=256, carrier_freq=3.5e9, speed=1.0,
                         num_users=4, cdl_model="C").to("cuda")


def _sample(rng, torch, channel_class, seed, generator=None):
    with rng.isolated_sionna_seed(seed), torch.no_grad():
        if generator is None:
            generator = _make_generator(channel_class)
        # Do not reset between H_dl and H_ul: use the unchanged public generator.
        pair = generator.generate_batch_data(batch_size=2, device="cuda")
        for value in pair:
            if tuple(value.shape) != SHAPE or not torch.isfinite(value).all().item():
                raise RuntimeError("Channel output has incorrect shape or non-finite values")
        return tuple(value.detach().cpu().clone() for value in pair)


def _draw_streams(torch, np, config, *, private):
    values = {
        "python": [random.random() for _ in range(8)],
        "numpy": np.random.random(8).tolist(),
        "torch_cpu": _tensor_record(torch.rand(8, device="cpu")),
        "torch_cuda": [_tensor_record(torch.rand(8, device=f"cuda:{index}"))
                       for index in range(torch.cuda.device_count())],
    }
    if private:
        values["sionna_python"] = [config.py_rng.random() for _ in range(8)]
        values["sionna_numpy"] = config.np_rng.random(8).tolist()
        values["sionna_torch"] = {
            device: _tensor_record(torch.rand(8, device=device, generator=config.torch_rng(device)))
            for device in ["cpu"] + [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        }
    return values


def _warm_private(torch, config):
    # Public APIs materialize lazy generators before recording the external stream.
    config.py_rng
    config.np_rng
    for device in ["cpu"] + [f"cuda:{index}" for index in range(torch.cuda.device_count())]:
        config.torch_rng(device)


def _state_check(rng, torch, np, config, action, *, private=True):
    before = rng.snapshot_rng_state()
    result = {"before": before.fingerprint()}
    try:
        expected = _draw_streams(torch, np, config, private=private)
        before.restore()
        action_result = action()
        after = rng.snapshot_rng_state()
        actual = _draw_streams(torch, np, config, private=private)
        result.update({"after": after.fingerprint(),
                       "state_unchanged": before.fingerprint() == after.fingerprint(),
                       "next_stream_draws_equal": expected == actual,
                       "expected_draws": expected, "actual_draws": actual,
                       "action": action_result})
        result["passed"] = result["state_unchanged"] and result["next_stream_draws_equal"]
        if isinstance(action_result, dict) and "passed" in action_result:
            result["passed"] = result["passed"] and action_result["passed"]
    finally:
        before.restore()
    return result


def _encode_pair(pair):
    result = {}
    for name, tensor in zip(("H_dl", "H_ul"), pair):
        array = tensor.contiguous().numpy()
        result[name] = dict(_tensor_record(tensor),
                           gzip_base64=base64.b64encode(gzip.compress(array.tobytes())).decode("ascii"))
    return result


def _decode_pair(torch, np, data):
    result = []
    for name in ("H_dl", "H_ul"):
        item = data[name]
        if tuple(item["shape"]) != SHAPE or item["dtype"] not in ("complex64", "complex128"):
            raise RuntimeError("Fresh-process channel shape/dtype is invalid")
        raw = gzip.decompress(base64.b64decode(item["gzip_base64"], validate=True))
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise RuntimeError("Fresh-process channel transport hash mismatch")
        array = np.frombuffer(raw, dtype=item["dtype"]).reshape(SHAPE).copy()
        if not np.isfinite(array).all():
            raise RuntimeError("Fresh-process channel contains non-finite values")
        result.append(torch.from_numpy(array))
    return tuple(result)


def _worker(seed):
    # Internal subprocess path only; no output directory or saved channel arrays.
    with redirect_stdout(sys.stderr):
        rng = _load_file("utils/sionna_rng.py", "_sionna_rng_worker_utility")
        torch, _, config = rng.runtime()
        if not torch.cuda.is_available():
            raise RuntimeError("Sionna channel diagnostic requires CUDA; CPU fallback is forbidden")
        channel_class = _load_file("utils/channel_generator.py", "_sionna_rng_worker_channels").CDLChannelGenerator_Feedback
        pair = _sample(rng, torch, channel_class, seed)
        payload = {"backend": _backend(torch), "environment": _environment(),
                   "sionna_device": str(config.device), "sionna_precision": config.precision,
                   "channels": _encode_pair(pair)}
    print(WORKER_MARKER + json.dumps(payload, allow_nan=False))


def _fresh_process(torch, np, reference, seed, backend, config):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--run", "--seed", str(seed), "--_worker"]
    child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=180)
    lines = [line[len(WORKER_MARKER):] for line in child.stdout.splitlines() if line.startswith(WORKER_MARKER)]
    evidence = {"command": command, "returncode": child.returncode,
                "stderr_tail": child.stderr[-12000:], "stdout_tail_without_payload": "\n".join(
                    line for line in child.stdout.splitlines() if not line.startswith(WORKER_MARKER))[-4000:]}
    if child.returncode != 0 or len(lines) != 1:
        return dict(evidence, passed=False, reason="Fresh-process worker failed or returned malformed evidence")
    data = json.loads(lines[0])
    pair = _decode_pair(torch, np, data["channels"])
    comparisons = _compare(torch, reference, pair)
    same_backend = data["backend"] == backend
    same_config = data["sionna_device"] == str(config.device) and data["sionna_precision"] == config.precision
    same_source = data["environment"]["source_sha256"] == _source_hashes()
    return dict(evidence, passed=comparisons["passed"] and same_backend and same_config and same_source,
                comparison=comparisons, backend_equal=same_backend, config_equal=same_config,
                source_hashes_equal=same_source,
                worker_backend=data["backend"], worker_environment=data["environment"])


def run_diagnostic(seed=SEED):
    seed = validate_seed(seed)
    output = create_output()
    report = {"status": "running", "passed": False, "plan": plan(seed), "checks": {},
              "output_directory": str(output),
              "scope": "Channel replay only; no claim of full training resume reproducibility."}
    torch = rng = initial = None
    try:
        report["environment"] = _environment()
        rng = _load_file("utils/sionna_rng.py", "_sionna_rng_diagnostic_utility")
        torch, np, config = rng.runtime()
        report["backend"] = _backend(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("Sionna channel diagnostic requires CUDA; CPU fallback is forbidden")
        initial = rng.snapshot_rng_state()
        report["initial_rng"] = initial.fingerprint()
        report["sionna_config"] = {"device": str(config.device), "precision": config.precision,
                                    "seed": config.seed}
        config_source = Path(inspect.getsourcefile(type(config)))
        report["installed_sionna_config_source"] = {
            "path": str(config_source), "sha256": hashlib.sha256(config_source.read_bytes()).hexdigest()}
        channel_class = _load_file("utils/channel_generator.py", "_sionna_rng_diagnostic_channels").CDLChannelGenerator_Feedback
        other, validation, nested_seed = [(seed + offset) % 2**32 for offset in (1, 2, 3)]
        with rng.isolated_sionna_seed(seed):
            generator = _make_generator(channel_class)
        reference = _sample(rng, torch, channel_class, seed, generator)
        report["reference_channels"] = _pair_record(reference)
        checks = report["checks"]
        checks["same_seed_reused_generator"] = _compare(torch, reference, _sample(rng, torch, channel_class, seed, generator))
        checks["same_seed_reconstructed_generator"] = _compare(torch, reference, _sample(rng, torch, channel_class, seed))
        different = _sample(rng, torch, channel_class, other, generator)
        difference = _compare(torch, reference, different)
        checks["different_seed_changes_both_channels"] = {
            "passed": all(not value["exact_hash_match"] for value in difference["channels"].values()),
            "comparison": difference, "different_seed": other}
        checks["consecutive_dl_ul_not_identical"] = {
            "passed": _tensor_record(reference[0])["sha256"] != _tensor_record(reference[1])["sha256"],
            "sampling": "Unchanged generate_batch_data samples CDL twice without reseeding between calls"}
        _sample(rng, torch, channel_class, validation, generator)
        checks["interleaved_validation"] = _compare(torch, reference, _sample(rng, torch, channel_class, seed, generator))
        reordered = {point: _sample(rng, torch, channel_class, point, generator) for point in (validation, other, seed)}
        checks["generation_order_seed"] = _compare(torch, reference, reordered[seed])
        checks["generation_order_other_seed"] = _compare(torch, different, reordered[other])

        def normal():
            _sample(rng, torch, channel_class, seed, generator)

        def exceptional():
            try:
                with rng.isolated_sionna_seed(seed):
                    generator.generate_batch_data(batch_size=2, device="cuda")
                    raise RuntimeError("intentional RNG isolation acceptance exception")
            except RuntimeError as error:
                if str(error) != "intentional RNG isolation acceptance exception":
                    raise

        def nested():
            with rng.isolated_sionna_seed(seed):
                generator.generate_batch_data(batch_size=2, device="cuda")
                outer = rng.snapshot_rng_state()
                normal()
                ordinary_equal = outer.fingerprint() == rng.snapshot_rng_state().fingerprint()
                try:
                    with rng.isolated_sionna_seed(nested_seed):
                        generator.generate_batch_data(batch_size=2, device="cuda")
                        raise RuntimeError("intentional nested RNG exception")
                except RuntimeError as error:
                    if str(error) != "intentional nested RNG exception":
                        raise
                exceptional_equal = outer.fingerprint() == rng.snapshot_rng_state().fingerprint()
                return {"passed": ordinary_equal and exceptional_equal,
                        "inner_normal_restores_outer": ordinary_equal,
                        "inner_exception_restores_outer": exceptional_equal}

        # Original laziness is checked without forcing creation of private RNGs.
        for name, action in (("normal", normal), ("exception", exceptional), ("nested", nested)):
            checks[f"original_state_{name}_exit"] = _state_check(rng, torch, np, config, action, private=False)
        # Explicitly exercise the None seed and materialized private streams.
        config.seed = None
        _warm_private(torch, config)
        _draw_streams(torch, np, config, private=True)
        for name, action in (("normal", normal), ("exception", exceptional), ("nested", nested)):
            checks[f"external_none_{name}_exit"] = _state_check(rng, torch, np, config, action)
        with rng.isolated_sionna_seed(other):
            _warm_private(torch, config)
            _draw_streams(torch, np, config, private=True)
            for name, action in (("normal", normal), ("exception", exceptional), ("nested", nested)):
                checks[f"external_seeded_{name}_exit"] = _state_check(rng, torch, np, config, action)
        checks["fresh_process"] = _fresh_process(torch, np, reference, seed, report["backend"], config)
        report["passed"] = all(item["passed"] for item in checks.values())
        report["status"] = "passed" if report["passed"] else "failed"
    except Exception as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        if initial is not None:
            try:
                initial.restore()
                report["final_rng"] = rng.snapshot_rng_state().fingerprint()
                restored = report["final_rng"] == report["initial_rng"]
                report["initial_rng_restored"] = restored
                if not restored:
                    report.update(status="failed", passed=False)
            except Exception as error:
                report.update(status="failed", passed=False, restore_error=f"{type(error).__name__}: {error}")
        if torch is not None:
            try:
                report["backend_final"] = _backend(torch)
                report["backend_unchanged"] = report.get("backend") == report["backend_final"]
                if not report["backend_unchanged"]:
                    report.update(status="failed", passed=False)
            except Exception as error:
                report.update(status="failed", passed=False, backend_error=f"{type(error).__name__}: {error}")
        try:
            report["source_hashes_final"] = _source_hashes()
            report["source_unchanged"] = report["source_hashes_final"] == report.get("environment", {}).get("source_sha256")
            if not report["source_unchanged"]:
                report.update(status="failed", passed=False)
        except Exception as error:
            report.update(status="failed", passed=False, source_hash_error=f"{type(error).__name__}: {error}")
        write_report(output, report)
        for name, result in report["checks"].items():
            print(f"{'PASS' if result['passed'] else 'FAIL'} {name}")
        print(f"{report['status'].upper()}: {output / 'report.json'}")
    if not report["passed"]:
        raise RuntimeError(f"Channel reproducibility acceptance failed; see {output / 'report.json'}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the plan using only the standard library; no files written")
    mode.add_argument("--run", action="store_true", help="Explicit CUDA channel-only acceptance; no training or models")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Explicit replay seed (default {SEED})")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_seed(args.seed)
    except ValueError as error:
        parser.error(str(error))
    if args._worker and not args.run:
        parser.error("Internal worker requires --run")
    if args.dry_run:
        print(json.dumps(plan(args.seed), indent=2, allow_nan=False))
    elif args.run:
        if args._worker:
            _worker(args.seed)
        else:
            run_diagnostic(args.seed)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
