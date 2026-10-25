#!/usr/bin/env python3
"""Explicit CUDA acceptance of the optional training-channel data interface.

No trainer, model, optimizer or checkpoint is loaded. Help and --dry-run use
only the standard library. This checks the same data-interface operations that
train_proposal calls, not full model/optimizer/scheduler resume equivalence.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

import diagnose_sionna_rng as acceptance


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/diagnostics/training_channel_data_v1"
BASE_SEED = 20261029
STEPS = (1, 2, 5, 500, 501, 5000)
WORKER_STEPS = (501, 5000)
WORKER_MARKER = "TRAINING_CHANNEL_DATA_WORKER_JSON:"
SOURCE_FILES = (
    "diagnose_training_channel_data.py", "utils/replayable_channel_data.py",
    "utils/sionna_rng.py", "utils/channel_generator.py", "main_train.py",
    "diagnose_sionna_rng.py",
)


def _data_module():
    # Avoid utils/__init__.py and its unrelated GPU-backend configuration.
    name = "_training_channel_data_diagnostic_interface"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "utils/replayable_channel_data.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load replayable channel-data interface")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class DiagnosticConfig:
    """Local interface-contract configuration; never passed into a trainer."""

    K_USERS = 4
    ANTENNAS = 32
    SUBCARRIERS = 256
    CARRIER_FREQ = 3.5e9
    SPEED = 1.0
    BATCH_SIZE = 2
    VAL_BATCH = 2
    TOTAL_STEPS = 5000

    def __init__(self, protocol):
        self.channel_data_protocol = protocol

    def to_dict(self):
        return {
            "diagnostic_only": True,
            "K_USERS": self.K_USERS, "ANTENNAS": self.ANTENNAS,
            "SUBCARRIERS": self.SUBCARRIERS, "CARRIER_FREQ": self.CARRIER_FREQ,
            "SPEED": self.SPEED, "BATCH_SIZE": self.BATCH_SIZE,
            "VAL_BATCH": self.VAL_BATCH, "TOTAL_STEPS": self.TOTAL_STEPS,
            "channel_data_protocol": self.channel_data_protocol,
        }


def diagnostic_data(base_seed=BASE_SEED):
    acceptance.validate_seed(base_seed)
    module = _data_module()
    data = module.ReplayableChannelData(
        base_seed=base_seed, k_users=4, nt=32, nsc=256, batch_size=2,
        validation_batch_size=2, carrier_freq=3.5e9, speed=1.0,
        cdl_model="C", device="cuda:0",
    )
    cfg = DiagnosticConfig(data.protocol)
    module.validate_channel_data(cfg, data)
    return cfg, data


def plan(base_seed=BASE_SEED):
    cfg, data = diagnostic_data(base_seed)
    return {
        "mode": "channel-data-interface-only CUDA acceptance",
        "supported_sionna": "2.0.1", "base_seed": base_seed,
        "diagnostic_config": cfg.to_dict(), "protocol": data.protocol,
        "channel": {"shape": list(acceptance.SHAPE), "profile": "C",
                    "generator": "CDLChannelGenerator_Feedback",
                    "sampling": "one isolated seed per consecutive DL/UL pair"},
        "selected_global_steps": list(STEPS),
        "fresh_process_starts_at": WORKER_STEPS[0],
        "output_root": str(OUTPUT_ROOT),
        "rtol": acceptance.RTOL, "atol": acceptance.ATOL,
        "acceptance": ["fixed validation replay", "sequential selected steps",
                       "interleaved validation", "reordered step prefetch",
                       "fresh process starts at a requested step without prior samples",
                       "external default and Sionna RNG unchanged"],
        "scope_limit": "Data-interface channel replay only; not full training resume.",
        "dry_run_reads_weights": False, "dry_run_creates_output": False,
    }


def create_output(root=OUTPUT_ROOT, *, run_id=None):
    if run_id is None:
        run_id = ("data_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")
                  + uuid.uuid4().hex[:8])
    return acceptance.create_output(root, run_id=run_id)


def _source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def _environment():
    result = acceptance._environment()
    result["source_sha256"] = _source_hashes()
    return result


def _guarded(rng, checks, name, action):
    """Record a side effect before restoring it; never hide a failed check."""
    before = rng.snapshot_rng_state()
    try:
        return action()
    finally:
        after = rng.snapshot_rng_state()
        comparison = {
            "passed": before.fingerprint() == after.fingerprint(),
            "before": before.fingerprint(), "after": after.fingerprint(),
        }
        checks[name] = comparison
        if not comparison["passed"]:
            # Preserve the evidence and avoid contaminating later diagnostics.
            before.restore()


def _cpu_pair(torch, pair):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise RuntimeError("Data interface must return the H_dl/H_ul pair")
    result = []
    for tensor in pair:
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != acceptance.SHAPE
                or not torch.is_complex(tensor) or not torch.isfinite(tensor).all().item()):
            raise RuntimeError("Channel pair has an incorrect shape, dtype or non-finite values")
        result.append(tensor.detach().contiguous().cpu().clone())
    return tuple(result)


def _sample(rng, torch, checks, name, operation):
    # Only this small diagnostic copies selected batches to CPU for hashing.
    return _guarded(rng, checks, name, lambda: _cpu_pair(torch, operation()))


def _worker(base_seed):
    with redirect_stdout(sys.stderr):
        rng = acceptance._load_file("utils/sionna_rng.py", "_training_channel_data_worker_rng")
        torch, _, config = rng.runtime()
        if not torch.cuda.is_available():
            raise RuntimeError("Training-channel diagnostic requires CUDA; CPU fallback is forbidden")
        checks = {}
        cfg, data = _guarded(rng, checks, "construction", lambda: diagnostic_data(base_seed))
        _guarded(rng, checks, "prepare", data.prepare)
        # First sampling operation starts at 501: do not recreate earlier batches.
        pairs = {
            str(step): _sample(rng, torch, checks, f"training_{step}",
                               lambda point=step: data.training(point))
            for step in WORKER_STEPS
        }
        validation = _sample(rng, torch, checks, "validation", data.validation)
        payload = {
            "protocol": data.protocol, "config": cfg.to_dict(),
            "channels": {key: acceptance._encode_pair(value) for key, value in pairs.items()},
            "validation": acceptance._encode_pair(validation),
            "rng_checks": checks, "passed": all(item["passed"] for item in checks.values()),
            "backend": acceptance._backend(torch), "environment": _environment(),
            "sionna_device": str(config.device), "sionna_precision": config.precision,
        }
    print(WORKER_MARKER + json.dumps(payload, allow_nan=False))


def _fresh_process(torch, np, reference, validation, base_seed, backend, config, protocol):
    command = [sys.executable, "-B", str(Path(__file__).resolve()),
               "--run", "--seed", str(base_seed), "--_worker"]
    child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                           check=False, timeout=240)
    lines = [line[len(WORKER_MARKER):] for line in child.stdout.splitlines()
             if line.startswith(WORKER_MARKER)]
    evidence = {
        "command": command, "returncode": child.returncode,
        "starts_at_global_step": WORKER_STEPS[0],
        "stderr_tail": child.stderr[-12000:],
        "stdout_tail_without_payload": "\n".join(
            line for line in child.stdout.splitlines() if not line.startswith(WORKER_MARKER))[-4000:],
    }
    if child.returncode != 0 or len(lines) != 1:
        return dict(evidence, passed=False, reason="Fresh-process worker failed or malformed evidence")
    payload = json.loads(lines[0])
    comparisons = {
        f"training_{step}": acceptance._compare(
            torch, reference[step], acceptance._decode_pair(torch, np, payload["channels"][str(step)]))
        for step in WORKER_STEPS
    }
    comparisons["validation"] = acceptance._compare(
        torch, validation, acceptance._decode_pair(torch, np, payload["validation"]))
    same_backend = payload["backend"] == backend
    same_sionna = (payload["sionna_device"] == str(config.device)
                   and payload["sionna_precision"] == config.precision)
    same_source = payload["environment"]["source_sha256"] == _source_hashes()
    same_protocol = payload["protocol"] == protocol
    passed = (all(item["passed"] for item in comparisons.values()) and payload["passed"]
              and same_backend and same_sionna and same_source and same_protocol)
    return dict(evidence, passed=passed, comparisons=comparisons,
                protocol_equal=same_protocol, backend_equal=same_backend,
                sionna_config_equal=same_sionna, source_hashes_equal=same_source,
                worker_rng_checks=payload["rng_checks"], worker_backend=payload["backend"],
                worker_environment=payload["environment"])


def run_diagnostic(base_seed=BASE_SEED):
    acceptance.validate_seed(base_seed)
    output = create_output()
    report = {
        "status": "running", "passed": False, "output_directory": str(output),
        "checks": {}, "rng_checks": {},
        "scope": "Optional data-interface channel replay only; no full training resume claim.",
    }
    initial = rng = torch = None
    try:
        report["environment"] = _environment()
        rng = acceptance._load_file("utils/sionna_rng.py", "_training_channel_data_diagnostic_rng")
        torch, np, config = rng.runtime()
        report["backend"] = acceptance._backend(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("Training-channel diagnostic requires CUDA; CPU fallback is forbidden")
        initial = rng.snapshot_rng_state()
        report["initial_rng"] = initial.fingerprint()
        report["sionna_config"] = {"device": str(config.device), "precision": config.precision,
                                    "seed": config.seed}
        config_source = Path(inspect.getsourcefile(type(config)))
        report["installed_sionna_config_source"] = {
            "path": str(config_source), "sha256": hashlib.sha256(config_source.read_bytes()).hexdigest()}
        checks, rng_checks = report["checks"], report["rng_checks"]
        cfg, data = _guarded(rng, rng_checks, "construction_and_contract",
                             lambda: diagnostic_data(base_seed))
        report.update(plan=plan(base_seed), protocol=data.protocol, diagnostic_config=cfg.to_dict())
        _guarded(rng, rng_checks, "prepare", data.prepare)
        validation = _sample(rng, torch, rng_checks, "validation_initial", data.validation)
        reference = {
            step: _sample(rng, torch, rng_checks, f"sequential_training_{step}",
                          lambda point=step: data.training(point))
            for step in STEPS
        }
        report["reference_channels"] = {
            "validation": acceptance._pair_record(validation),
            "training": {str(step): acceptance._pair_record(pair) for step, pair in reference.items()},
        }
        checks["consecutive_dl_ul_distinct"] = {
            "passed": all(acceptance._tensor_record(pair[0])["sha256"]
                          != acceptance._tensor_record(pair[1])["sha256"]
                          for pair in (validation, *reference.values())),
            "sampling": "Existing generate_batch_data samples DL then UL without intermediate reseeding",
        }
        for step in STEPS:
            current_validation = _sample(rng, torch, rng_checks, f"validation_before_{step}", data.validation)
            checks[f"validation_fixed_before_{step}"] = acceptance._compare(torch, validation, current_validation)
            current = _sample(rng, torch, rng_checks, f"interleaved_training_{step}",
                              lambda point=step: data.training(point))
            checks[f"training_after_validation_{step}"] = acceptance._compare(torch, reference[step], current)
        for step in reversed(STEPS):
            current = _sample(rng, torch, rng_checks, f"reordered_training_{step}",
                              lambda point=step: data.training(point))
            checks[f"reordered_prefetch_{step}"] = acceptance._compare(torch, reference[step], current)
        checks["fresh_process"] = _fresh_process(
            torch, np, reference, validation, base_seed, report["backend"], config, data.protocol)
        report["passed"] = (all(item["passed"] for item in checks.values())
                            and all(item["passed"] for item in rng_checks.values()))
        report["status"] = "passed" if report["passed"] else "failed"
    except Exception as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        if initial is not None:
            try:
                report["final_rng_before_cleanup"] = rng.snapshot_rng_state().fingerprint()
                report["external_rng_unchanged"] = report["final_rng_before_cleanup"] == report["initial_rng"]
                initial.restore()
                report["initial_rng_restored"] = rng.snapshot_rng_state().fingerprint() == report["initial_rng"]
                if not report["external_rng_unchanged"] or not report["initial_rng_restored"]:
                    report.update(status="failed", passed=False)
            except Exception as error:
                report.update(status="failed", passed=False, restore_error=f"{type(error).__name__}: {error}")
        if torch is not None:
            try:
                report["backend_final"] = acceptance._backend(torch)
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
        acceptance.write_report(output, report)
        for name, result in report["checks"].items():
            print(f"{'PASS' if result['passed'] else 'FAIL'} {name}")
        rng_passes = sum(bool(item["passed"]) for item in report["rng_checks"].values())
        print(f"External RNG checks: {rng_passes}/{len(report['rng_checks'])}")
        print(f"{report['status'].upper()}: {output / 'report.json'}")
    if not report["passed"]:
        raise RuntimeError(f"Channel-data interface acceptance failed; see {output / 'report.json'}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print standard-library-only plan; no files written")
    mode.add_argument("--run", action="store_true", help="CUDA channel-data acceptance only; no models or optimizer")
    parser.add_argument("--seed", "--base-seed", dest="seed", type=int, default=BASE_SEED)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        acceptance.validate_seed(args.seed)
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
