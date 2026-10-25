#!/usr/bin/env python3
"""Fixed two-arm/two-seed AS receiver batch. No action without explicit --run.

Run this on the CUDA server, not on the laptop. Each child process releases GPU
memory on exit. This is an engineering screening batch, not an adaptive search.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import traceback
import uuid

from as_receiver_batch_support import CLEAN_TARGETS, archive_exact_target, safe_inside, save_json, sha256
from configs.as_receiver_train_config import ARMS, SEEDS, OUTPUT_ROOT, as_receiver_config

ROOT = Path(__file__).resolve().parent
BATCH_ROOT = OUTPUT_ROOT / "batches"
EVAL_ROOT = ROOT / "runs_as_screen/evaluation/k4_as_receiver_v1"
SOURCE_FILES = (
    "run_as_receiver_batch.py", "as_receiver_batch_support.py", "as_receiver_results.py",
    "models/adaptive_hybrid_as_receiver.py", "models/adaptive_hybrid.py",
    "configs/as_receiver_train_config.py", "configs/train_config.py",
    "train_proposal_as_receiver.py", "evaluate_proposal_as_receiver.py",
    "main_train.py", "baseline_evaluate.py", "utils/replayable_channel_data.py",
    "utils/sionna_rng.py", "utils/channel_generator.py", "utils/metrics.py",
    "utils/layers.py", "tests/test_adaptive_hybrid_as_receiver.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_record():
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                sha256={name: sha256(ROOT / name) for name in SOURCE_FILES})


def plan():
    configs = [as_receiver_config(arm, seed) for seed in SEEDS for arm in ARMS]
    return dict(batch="as_receiver_v1_two_seed20k", runs=[dict(arm=c.arm, seed=c.SEED,
                total_updates=c.TOTAL_STEPS, technical_smoke_included_updates=500,
                best=c.BEST_PATH, latest=c.SAVE_PATH) for c in configs],
                cleanup_exact_directories=list(CLEAN_TARGETS),
                evaluation=dict(seed=20261101, samples_per_condition=1600, batch_size=8,
                                feedback_snr=[0, 5, 10, 15, 20, 25], downlink_snr=25,
                                includes_receiver_off_intervention=True, training_seeds_pooled=False),
                criteria=dict(primary="ASReceiver-Full low(0/5/10) within-sample mean; batch CI",
                              low_mean_min=0.5, low_ci95_lower_strictly_greater=0.0,
                              fb25_ci95_lower_min=-0.1,
                              interpretation="engineering continuation criterion, not a publication guarantee"),
                no_posthoc_seed_checkpoint_or_snr_search=True,
                lightweight_plan_no_weights_or_writes=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def controller_lock(resume):
    path = safe_inside(OUTPUT_ROOT / "controller.lock", ROOT)
    if path.exists():
        old = read_json(path)
        if (os.name != "posix" or not resume or old.get("host") != socket.gethostname()
                or type(old.get("pid")) is not int or old["pid"] <= 0):
            raise RuntimeError("Existing/unknown controller lock; do not launch a competing batch")
        guard = path.with_name(path.name + ".recovery")
        guard.mkdir(exist_ok=False)
        try:
            if read_json(path) != old:
                raise RuntimeError("Controller lock changed during recovery")
            try:
                os.kill(old["pid"], 0)
            except ProcessLookupError:
                save_json(path.with_name(path.name + ".stale_" + uuid.uuid4().hex + ".json"), old)
                path.unlink()  # Exact confirmed-dead lock; never training artifacts.
            else:
                raise RuntimeError("Controller PID still alive or not verifiably dead")
        finally:
            guard.rmdir()
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = dict(pid=os.getpid(), host=socket.gethostname(), created=now(), token=uuid.uuid4().hex)
    save_json(path, owner)
    try:
        yield
    finally:
        if path.is_file() and read_json(path) == owner:
            path.unlink()


def _child(folder, label, command, *, environment=None):
    log = folder / (label + "_" + uuid.uuid4().hex[:8] + ".log")
    print(f"[{now()}] {label}: {' '.join(map(str, command))}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}); evidence {log}")
    return str(log)


def _run_stage(summary, folder, key, action):
    existing = summary["stages"].get(key, {})
    if existing.get("status") == "completed":
        for filename, digest in existing.get("evidence_sha256", {}).items():
            if not Path(filename).is_file() or sha256(filename) != digest:
                raise RuntimeError(f"Completed stage evidence changed: {key}: {filename}")
        print(f"[resume] verified completed stage {key}", flush=True)
        return existing["result"]
    entry = dict(status="running", started=now(), attempt=existing.get("attempt", 0) + 1)
    summary["stages"][key] = entry
    save_json(folder / "batch_summary.json", summary, replace=True)
    try:
        result = action()
        evidence = result.pop("_evidence", [])
        entry.update(status="completed", finished=now(), result=result,
                     evidence_sha256={str(path): sha256(path) for path in evidence})
    except Exception as error:
        entry.update(status="failed", finished=now(), error=f"{type(error).__name__}: {error}",
                     traceback=traceback.format_exc())
        raise
    finally:
        save_json(folder / "batch_summary.json", summary, replace=True)
    return entry["result"]


def _environment_stage():
    import torch
    from preflight_proposal_as_direct import load_source_record, verify_checkpoint_hash, _backend_info
    from evaluate_proposal_as_direct import archived_swin
    from diagnose_sionna_rng import _load_file, _backend
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; never fall back to laptop CPU")
    if importlib.metadata.version("sionna") != "2.0.1":
        raise RuntimeError("Accepted Sionna 2.0.1 runtime required")
    importlib.metadata.version("matplotlib")  # Fail before long work if final plots cannot be produced.
    rng = _load_file("utils/sionna_rng.py", "_as_receiver_environment_rng")
    rng.runtime()  # Verify accepted private-RNG implementation, without channel generation.
    robust, swin = load_source_record(), archived_swin()
    # Archived metadata uses checkpoint_sha256 for both records.
    hashes = {row["path"]: verify_checkpoint_hash(row["path"], row["checkpoint_sha256"])
              for row in (robust, swin)}
    return dict(cuda=_backend_info(torch), channel_backend=_backend(torch), checkpoint_sha256=hashes,
                torch=torch.__version__, sionna=importlib.metadata.version("sionna"),
                matplotlib=importlib.metadata.version("matplotlib"), source=source_record())


def _model_stage():
    import torch
    import baseline_evaluate as ev
    from models.adaptive_hybrid import AdaptiveHybridPrecoder
    from models.adaptive_hybrid_as_receiver import AdaptiveHybridPrecoderASReceiver, migrate_original_state_dict
    from preflight_proposal_as_direct import (load_source_record, validate_checkpoint_metadata,
        verify_checkpoint_hash, _comparison, _backend_info)
    from main_train import capture_rng_state, restore_rng_state, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for real source check")
    source = load_source_record()
    with Path(source["path"]).open("rb") as stream:
        verify_checkpoint_hash(source["path"], source["checkpoint_sha256"], stream=stream)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(payload, source)
    cfg = as_receiver_config("as_receiver", SEEDS[0])
    old = AdaptiveHybridPrecoder(cfg).cuda().eval()
    new = AdaptiveHybridPrecoderASReceiver(cfg).cuda().eval()
    old.load_state_dict(payload["state"], strict=True)
    migrate_original_state_dict(new, payload["state"])
    for key, tensor in payload["state"].items():
        if not torch.equal(new.state_dict()[key].cpu(), tensor):
            raise RuntimeError(f"Source tensor changed: {key}")
    set_seed(20261029)
    shape = (2, 4, 32, 256)
    h_dl = torch.complex(torch.randn(shape, device="cuda"), torch.randn(shape, device="cuda")) / 2**0.5
    h_ul = torch.complex(torch.randn(shape, device="cuda"), torch.randn(shape, device="cuda")) / 2**0.5
    buffers = {label: {key: value.clone() for key, value in model.named_buffers()}
               for label, model in (("full", old), ("receiver", new))}
    rows = []
    for allocation in ((64, 0), (32, 128), (0, 256)):
        for snr in (0, 10, 25):
            initial = capture_rng_state()
            with torch.no_grad():
                w_old, h_old = ev.forward_proposed_with_surrogate(old, h_dl, h_ul,
                    torch.tensor(float(snr), device="cuda"), *allocation, 0)
                restore_rng_state(initial)
                w_new, h_new = ev.forward_proposed_with_surrogate(new, h_dl, h_ul,
                    torch.tensor(float(snr), device="cuda"), *allocation, 0)
            row = dict(allocation=list(allocation), feedback_snr=snr,
                       H_hat=_comparison(torch, h_new, h_old),
                       deployed_RZF=_comparison(torch, w_new, w_old),
                       power_max_abs=float((w_new.abs().square().sum((1, 2)) - 1).abs().max()))
            rows.append(row)
            if not row["H_hat"]["passed"] or not row["deployed_RZF"]["passed"] or row["power_max_abs"] > 2e-6:
                raise RuntimeError("Zero projection changed source outputs: " + json.dumps(row))
    for label, model in (("full", old), ("receiver", new)):
        if any(not torch.equal(value, buffers[label][key]) for key, value in model.named_buffers()):
            raise RuntimeError("Source-check eval buffers changed")
    verify_checkpoint_hash(source["path"], source["checkpoint_sha256"])
    return dict(passed=True, zero_projection_comparisons=rows, checkpoint_sha256=source["checkpoint_sha256"],
                added_parameters=sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()),
                encoder_parameters_unchanged=sum(p.numel() for p in new.encoder.parameters()) == sum(p.numel() for p in old.encoder.parameters()),
                architecture=new.architecture_metadata, backend=_backend_info(torch), rtol=1e-6, atol=1e-6)


def inspect_training(arm, seed):
    import torch
    from train_proposal_as_receiver import validate_resume_payload, validate_best_metadata, fingerprint
    cfg = as_receiver_config(arm, seed)
    latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
    best = torch.load(cfg.BEST_PATH, map_location="cpu", weights_only=False)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    status = read_json(cfg.STATUS_PATH)
    if latest["step"] != cfg.TOTAL_STEPS or status.get("state") != "completed":
        raise RuntimeError("Training did not complete exactly 20000 updates")
    return dict(arm=arm, seed=seed, completed_updates=latest["step"], best_step=best["step"],
                best_metric=best["best_metric"], fingerprint=fingerprint(cfg),
                best_path=cfg.BEST_PATH, best_sha256=sha256(cfg.BEST_PATH),
                latest_path=cfg.SAVE_PATH, latest_sha256=sha256(cfg.SAVE_PATH), status=status)


def _hidden_stage(name, output, arm=None, seed=None):
    output = safe_inside(output, BATCH_ROOT)
    if output.exists():
        raise FileExistsError(output)
    try:
        if name == "environment":
            report = _environment_stage()
        elif name == "model":
            report = _model_stage()
        elif name == "training_record":
            report = inspect_training(arm, seed)
        elif name == "model_tests":
            import unittest
            os.environ["AS_RECEIVER_TEST_DEVICE"] = "cuda"
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_adaptive_hybrid_as_receiver.py")
            outcome = unittest.TextTestRunner(verbosity=2).run(suite)
            report = dict(passed=outcome.wasSuccessful() and not outcome.skipped,
                          checks=outcome.testsRun, failures=len(outcome.failures),
                          errors=len(outcome.errors), skipped=outcome.skipped)
            if not report["passed"]:
                save_json(output, report)
                raise RuntimeError("New model CUDA checks did not all pass")
        else:
            raise ValueError("Unknown internal stage")
        report.setdefault("passed", True)
        save_json(output, report)
    except Exception as error:
        if not output.exists():
            save_json(output, dict(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        raise


def _stage_child(folder, name, *, arm=None, seed=None):
    output = folder / (name + "_" + uuid.uuid4().hex[:8] + ".json")
    cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--_stage", name,
           "--_output", str(output)]
    if arm is not None:
        cmd += ["--_arm", arm, "--_seed", str(seed)]
    log = _child(folder, name, cmd)
    report = read_json(output)
    if report.get("passed") is not True:
        raise RuntimeError(f"Stage failed: {output}")
    return dict(report=report, report_path=str(output), log=log, _evidence=[output, log])


def _data_acceptance(folder, environment):
    # Reuse only a successful report for exactly this current implementation.
    diagnostic_root = ROOT / "runs_as_screen/diagnostics/training_channel_data_v1"
    expected = {name: sha256(ROOT / name) for name in (
        "diagnose_training_channel_data.py", "utils/replayable_channel_data.py", "utils/sionna_rng.py",
        "utils/channel_generator.py", "main_train.py", "diagnose_sionna_rng.py")}
    def accepted(path):
        record = read_json(path)
        versions = {name: importlib.metadata.version(name) for name in ("sionna", "torch", "numpy")}
        return (record.get("passed") is True and record.get("external_rng_unchanged") is True
                and record.get("environment", {}).get("source_sha256") == expected
                and record.get("environment", {}).get("versions") == versions
                and record.get("backend") == environment["channel_backend"])
    for path in sorted(diagnostic_root.glob("data_*/report.json")) if diagnostic_root.exists() else []:
        if accepted(path):
            return dict(report_path=str(path), reused=True, _evidence=[path])
    before = set(diagnostic_root.glob("data_*/report.json")) if diagnostic_root.exists() else set()
    log = _child(folder, "channel_data_acceptance", [sys.executable, "-B", "-u", "diagnose_training_channel_data.py", "--run"])
    reports = set(diagnostic_root.glob("data_*/report.json")) - before
    if len(reports) != 1 or not accepted(next(iter(reports))):
        raise RuntimeError("New channel-data report missing/not accepted")
    path = next(iter(reports))
    return dict(report_path=str(path), reused=False, log=log, _evidence=[path, log])


def _train_one(folder, arm, seed):
    cfg = as_receiver_config(arm, seed)
    base = [sys.executable, "-B", "-u", "train_proposal_as_receiver.py", "--arm", arm, "--seed", str(seed)]
    logs = []
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
            raise RuntimeError("Incomplete fresh run has no latest; refusing to overwrite/reinitialize")
        logs.append(_child(folder, f"{arm}_{seed}_first500", base + ["--stop-after", "500"]))
    status = read_json(cfg.STATUS_PATH)
    if status.get("state") != "completed":
        # Entrypoint strictly validates latest/config/optimizer/schedule/RNG.
        # The fixed scheduler has 20000 total updates: smoke is not extra.
        logs.append(_child(folder, f"{arm}_{seed}_continue", base + ["--resume", "--train"]))
    result = _stage_child(folder, "training_record", arm=arm, seed=seed)
    result["training_logs"] = logs
    result["_evidence"] += logs + [cfg.SAVE_PATH, cfg.BEST_PATH, cfg.STATUS_PATH, cfg.CONFIG_PATH]
    return result


def verify_evaluation_record(output, seed):
    from evaluate_proposal_as_receiver import protocol, artifact_hashes
    output = safe_inside(output, EVAL_ROOT)
    record = read_json(output / "manifest.json")
    if (record.get("status") != "completed" or record.get("freeze_checks", {}).get("passed") is not True
            or record.get("protocol") != protocol(seed)):
        raise ValueError("Evaluation protocol/status/freeze identity mismatch")
    for name, digest in record.get("runtime", {}).get("source_sha256", {}).items():
        if sha256(safe_inside(ROOT / name, ROOT)) != digest:
            raise ValueError(f"Evaluation source differs: {name}")
    if not record.get("runtime", {}).get("source_sha256"):
        raise ValueError("Evaluation source provenance missing")
    for name, digest in record.get("input_sha256_after", {}).items():
        if sha256(name) != digest:
            raise ValueError(f"Evaluation input changed: {name}")
    if not record.get("input_sha256_after"):
        raise ValueError("Evaluation input provenance missing")
    saved = record.get("output_sha256", {})
    if artifact_hashes(output) != saved:
        raise ValueError("Evaluation artifact set/hashes do not match saved evidence")
    required = {"summary_k4_as_receiver_v1.csv", "samples_k4_as_receiver_v1.csv.gz",
                "paired_deltas_k4_as_receiver_v1.csv", "paired_samples_k4_as_receiver_v1.csv.gz"}
    if not required <= saved.keys():
        raise ValueError("Evaluation output integrity manifest missing")
    for name, digest in saved.items():
        if Path(name).name != name or sha256(safe_inside(output / name, output)) != digest:
            raise ValueError(f"Evaluation output changed: {name}")
    return [output / "manifest.json"] + [output / name for name in saved]


def _evaluate_one(folder, seed):
    parent = safe_inside(EVAL_ROOT / folder.name / f"seed_{seed}", ROOT)
    # A controller crash after evaluator success must not repeat formal work.
    for path in sorted(parent.glob("attempt_*/manifest.json")) if parent.exists() else []:
        record = read_json(path)
        if record.get("status") == "completed" and record.get("freeze_checks", {}).get("passed") is True:
            evidence = verify_evaluation_record(path.parent, seed)
            return dict(output_directory=str(path.parent), reused=True, _evidence=evidence)
    output = parent / ("attempt_" + uuid.uuid4().hex[:8])
    log = _child(folder, f"evaluation_{seed}", [sys.executable, "-B", "-u", "evaluate_proposal_as_receiver.py",
                 "--seed", str(seed), "--evaluate", "--output-dir", str(output)])
    manifest = output / "manifest.json"
    record = read_json(manifest)
    if record.get("status") != "completed" or record.get("freeze_checks", {}).get("passed") is not True:
        raise RuntimeError(f"Evaluation failed/frozen check missing: {manifest}")
    evidence = verify_evaluation_record(output, seed)
    return dict(output_directory=str(output), log=log, reused=False, _evidence=evidence + [log])


def _reports(folder, evaluations):
    from as_receiver_results import organize_results
    return organize_results(folder, evaluations)


def run_batch(*, resume=False):
    if os.name != "posix":
        raise RuntimeError("Execute this batch on the POSIX CUDA server; local Windows is preparation-only")
    with controller_lock(resume):
        pointer = OUTPUT_ROOT / "last_batch.json"
        if resume:
            if not pointer.is_file():
                raise FileNotFoundError("No recorded batch to resume")
            folder = safe_inside(read_json(pointer)["directory"], BATCH_ROOT)
            summary = read_json(folder / "batch_summary.json")
            if summary["source"]["sha256"] != source_record()["sha256"]:
                raise RuntimeError("Relevant source changed; cannot silently resume this fixed batch")
        else:
            if pointer.exists():
                raise RuntimeError("Batch already recorded; use --resume, never start a competing replacement")
            for seed in SEEDS:
                for arm in ARMS:
                    cfg = as_receiver_config(arm, seed)
                    if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
                        raise RuntimeError("Existing unowned arm outputs; refusing to adopt/overwrite")
            name = "batch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
            folder = safe_inside(BATCH_ROOT / name, ROOT)
            folder.mkdir(parents=True, exist_ok=False)
            summary = dict(status="running", created=now(), source=source_record(), plan=plan(), stages={},
                           return_files=[str(folder / "batch_summary.json"), str(folder / "results_bundle.tar.gz")])
            save_json(folder / "batch_summary.json", summary)
            save_json(pointer, dict(directory=str(folder), created=now()))
        summary.update(status="running", resumed=resume)
        try:
            for index, target in enumerate(CLEAN_TARGETS):
                _run_stage(summary, folder, f"cleanup_{index}",
                           lambda target=target: archive_exact_target(target, folder / "archives"))
            environment = _run_stage(summary, folder, "environment", lambda: _stage_child(folder, "environment"))["report"]
            if resume:
                current = _stage_child(folder, "environment")
                summary.setdefault("resume_environment_checks", []).append(current)
                if (current["report"]["channel_backend"] != environment["channel_backend"]
                        or current["report"]["checkpoint_sha256"] != environment["checkpoint_sha256"]
                        or any(current["report"][name] != environment[name] for name in ("torch", "sionna", "matplotlib"))):
                    raise RuntimeError("Server environment/source differs from the recorded batch")
            _run_stage(summary, folder, "channel_data", lambda: _data_acceptance(folder, environment))
            _run_stage(summary, folder, "new_model_checks", lambda: _stage_child(folder, "model_tests"))
            _run_stage(summary, folder, "real_source_zero_projection", lambda: _stage_child(folder, "model"))
            training = {}
            for seed in SEEDS:
                for arm in ARMS:
                    key = f"train_{arm}_{seed}"
                    training[key] = _run_stage(summary, folder, key,
                        lambda arm=arm, seed=seed: _train_one(folder, arm, seed))["report"]
            summary["training"] = training
            evaluations = {str(seed): _run_stage(summary, folder, f"evaluate_{seed}",
                          lambda seed=seed: _evaluate_one(folder, seed))["output_directory"] for seed in SEEDS}
            summary["results"] = _run_stage(summary, folder, "organize_results", lambda: _reports(folder, evaluations))
            summary.update(status="completed", finished=now())
        except BaseException as error:
            summary.update(status="failed", failed=now(), error=f"{type(error).__name__}: {error}",
                           traceback=traceback.format_exc(),
                           failure_return_files=[str(folder / "batch_summary.json")],
                           resume_command="python -B -u run_as_receiver_batch.py --run --resume")
            raise
        finally:
            save_json(folder / "batch_summary.json", summary, replace=True)
            # Failed-attempt packages are never reused as the final result package.
            bundle = folder / ("results_bundle.tar.gz" if summary["status"] == "completed" else
                               "failure_bundle_" + uuid.uuid4().hex[:8] + ".tar.gz")
            summary["small_result_bundle"] = str(bundle)
            summary["return_files"] = [str(folder / "batch_summary.json"), str(bundle)]
            save_json(folder / "batch_summary.json", summary, replace=True)
            if not bundle.exists():
                with tarfile.open(bundle, "x:gz") as archive:
                    archive.add(folder / "batch_summary.json", arcname="batch_summary.json", recursive=False)
                    export_path = summary.get("results", {}).get("export_directory")
                    export = safe_inside(export_path, folder) if export_path else None
                    if export is not None and export.is_dir():
                        for path in sorted(export.iterdir()):
                            if path.is_file() and path.suffix in (".csv", ".json", ".md", ".pdf", ".png"):
                                archive.add(path, arcname="export/" + path.name, recursive=False)
            print(f"BATCH_SUMMARY={folder / 'batch_summary.json'}", flush=True)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_stage", choices=("environment", "model_tests", "model", "training_record"), help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--_seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._stage:
        if args._output is None or args.dry_run or args.run or args.summary or args.resume:
            parser.error("Internal stage requires only an explicit output path")
        _hidden_stage(args._stage, args._output, args._arm, args._seed)
    elif args.resume and not args.run:
        parser.error("--resume requires --run")
    elif args.dry_run:
        print(json.dumps(plan(), indent=2))
    elif args.summary:
        folder = safe_inside(read_json(OUTPUT_ROOT / "last_batch.json")["directory"], BATCH_ROOT)
        print((folder / "batch_summary.json").read_text(encoding="utf-8"))
    elif args.run:
        run_batch(resume=args.resume)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
