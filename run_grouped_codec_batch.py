#!/usr/bin/env python3
"""Five layouts x two seeds: explicit server batch, resumable hash-verified stages."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import uuid

from as_receiver_batch_support import safe_inside, save_json, sha256
from run_as_receiver_batch import _child, _run_stage, now, read_json
from run_as_spatial_batch import _package
from train_proposal_as_receiver import run_lock
from configs.grouped_codec_train_config import ARMS, SEEDS, OUTPUT_ROOT, grouped_codec_config

ROOT = Path(__file__).resolve().parent
BATCH_ROOT = OUTPUT_ROOT / "batches"
EVAL_ROOT = ROOT / "runs_as_screen/evaluation/k4_grouped_codec_v1"


def source_record():
    from train_grouped_codec import SOURCE_FILES as training
    from evaluate_grouped_codec import SOURCE_FILES as evaluation
    files = tuple(dict.fromkeys(training + evaluation + (
        "run_grouped_codec_batch.py", "preflight_grouped_codec.py", "grouped_codec_results.py",
        "tests/test_grouped_feedback_protocol.py", "tests/test_grouped_codec_precoder.py",
        "run_as_receiver_batch.py", "run_as_spatial_batch.py", "as_receiver_batch_support.py")))
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                sha256={name: sha256(ROOT / name) for name in files})


def plan():
    from evaluate_grouped_codec import protocol, CONTRASTS
    from preflight_grouped_codec import plan as preflight_plan
    return dict(batch="grouped_codec_v1", training=[dict(arm=arm, seed=seed,
        total_updates=20000, included_smoke_updates=500, best=grouped_codec_config(arm, seed).BEST_PATH)
        for seed in SEEDS for arm in ARMS], total_official_updates=200000,
        evaluations={str(seed): protocol(seed) for seed in SEEDS}, preflight=preflight_plan(),
        initialization="all UE and BS random; common BS identical within seed; no old weights",
        contrasts=CONTRASTS, criteria=dict(reference="D ordered_group2 minus E ordered_fdma",
            low_mean_min=.5, low_batch_ci95_lower_strictly_greater=0, gamma25_batch_ci95_lower_min=-.1),
        historical_runs_retrained=False, cleanup=False, visits_08=False,
        training_seeds_pooled=False, adaptive_search=False, manuscript_modified=False)


def _environment():
    import torch
    from diagnose_sionna_rng import _load_file, _backend
    from preflight_proposal_as_direct import _backend_info
    from train_grouped_codec import apply_backend
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no laptop CPU fallback")
    if importlib.metadata.version("sionna") != "2.0.1":
        raise RuntimeError("Accepted Sionna2.0.1 required")
    importlib.metadata.version("matplotlib")
    _load_file("utils/sionna_rng.py", "_grouped_environment_rng").runtime()
    apply_backend(torch)
    return dict(torch=torch.__version__, sionna=importlib.metadata.version("sionna"),
        matplotlib=importlib.metadata.version("matplotlib"), channel_backend=_backend(torch),
        cuda=_backend_info(torch), source=source_record(), historical_weights_loaded=False)


def _training_record(arm, seed):
    import torch
    from evaluate_grouped_codec import verify_training_metadata, build_model
    from train_grouped_codec import validate_model_state
    from evaluate_proposal_as_direct import check_tensor_state
    cfg = grouped_codec_config(arm, seed)
    latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
    best = torch.load(cfg.BEST_PATH, map_location="cpu", weights_only=False)
    record = verify_training_metadata(arm, seed, best, latest,
        {"config": read_json(cfg.CONFIG_PATH), "status": read_json(cfg.STATUS_PATH)})
    model, _ = build_model(torch, arm, seed, best, torch.device("cpu"))
    check_tensor_state(torch, model.state_dict(), latest["state"])
    validate_model_state(torch, model, latest["state"])
    if latest["step"] == best["step"]:
        check_tensor_state(torch, latest["state"], best["state"], equal=True)
    return dict(arm=arm, training_seed=seed, completed_updates=latest["step"], best_step=best["step"],
        best_metric=best["best_metric"], best_path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
        best_sha256=sha256(cfg.BEST_PATH), latest_sha256=sha256(cfg.SAVE_PATH), verified_selection=record)


def _stage_child(folder, name, arm=None, seed=None):
    output = folder / (name + "_" + uuid.uuid4().hex[:8] + ".json")
    command = [sys.executable, "-B", "-u", "run_grouped_codec_batch.py", "--_stage", name,
               "--_output", str(output)]
    if arm is not None:
        command += ["--_arm", arm, "--_seed", str(seed)]
    log = _child(folder, name, command)
    report = read_json(output)
    if report.get("passed") is not True:
        raise RuntimeError(f"Stage failed: {output}")
    return dict(report=report, report_path=str(output), log=log, _evidence=[output, log])


def _preflight(folder):
    environment = dict(os.environ, GROUPED_CODEC_TEST_DEVICE="cuda")
    logs = []
    for name in ("test_grouped_feedback_protocol.py", "test_grouped_codec_precoder.py"):
        logs.append(_child(folder, name[:-3], [sys.executable, "-B", "-m", "unittest", "discover",
            "-s", "tests", "-p", name, "-v"], environment=environment))
    output = folder / ("cuda_batch48_" + uuid.uuid4().hex[:8])
    retained = folder / (output.name + ".json")
    try:
        logs.append(_child(folder, "cuda_batch48", [sys.executable, "-B", "-u", "preflight_grouped_codec.py",
            "--run", "--output-dir", str(output)]))
    finally:
        if (output / "report.json").is_file():
            save_json(retained, read_json(output / "report.json"))
    report = read_json(output / "report.json")
    if report.get("passed") is not True:
        raise RuntimeError("New-physics/model/batch48 checks failed; training forbidden")
    return dict(report=report, _evidence=logs + [output / "report.json", retained])


def _train_one(folder, arm, seed):
    cfg = grouped_codec_config(arm, seed)
    command = [sys.executable, "-B", "-u", "train_grouped_codec.py", "--arm", arm, "--seed", str(seed)]
    logs = []
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
            raise RuntimeError("Existing arm directory without latest; preserve and investigate")
        logs.append(_child(folder, f"{arm}_{seed}_first500", command + ["--stop-after", "500"]))
    if read_json(cfg.STATUS_PATH).get("state") != "completed":
        logs.append(_child(folder, f"{arm}_{seed}_continue", command + ["--resume", "--train"]))
    result = _stage_child(folder, "training_record", arm, seed)
    result["_evidence"] += logs + [cfg.BEST_PATH, cfg.SAVE_PATH, cfg.STATUS_PATH, cfg.CONFIG_PATH]
    return result


def verify_evaluation(output, seed):
    from evaluate_grouped_codec import protocol, artifact_hashes
    output = safe_inside(output, EVAL_ROOT)
    manifest = read_json(output / "manifest.json")
    if (manifest.get("status") != "completed" or not manifest.get("results_verified")
            or not manifest.get("freeze_checks", {}).get("passed")
            or not manifest.get("shared_channel_noise_inputs_unchanged")
            or manifest.get("protocol") != protocol(seed)):
        raise ValueError("Evaluation protocol/status/freeze check failed")
    inputs = manifest.get("input_sha256_after", {})
    code = manifest.get("runtime", {}).get("source_sha256", {})
    if not inputs or not code or inputs != manifest.get("input_sha256_before"):
        raise ValueError("Missing or changed evaluation provenance")
    for filename, digest in inputs.items():
        if sha256(safe_inside(filename, ROOT)) != digest:
            raise ValueError(f"Changed evaluation input: {filename}")
    for filename, digest in code.items():
        if sha256(safe_inside(ROOT / filename, ROOT)) != digest:
            raise ValueError(f"Changed evaluation source: {filename}")
    if artifact_hashes(output) != manifest.get("output_sha256"):
        raise ValueError("Changed evaluation artifacts")
    return [output / "manifest.json"] + [output / name for name in manifest["output_sha256"]]


def _evaluate_one(folder, seed):
    parent = safe_inside(EVAL_ROOT / folder.name / f"seed_{seed}", EVAL_ROOT)
    pointer = folder / f"evaluation_seed{seed}.json"
    if pointer.exists():
        previous = safe_inside(read_json(pointer)["directory"], parent)
        if (previous / "manifest.json").exists() and read_json(previous / "manifest.json").get("status") == "completed":
            return dict(output_directory=str(previous), reused=True, _evidence=verify_evaluation(previous, seed))
    output = parent / ("attempt_" + uuid.uuid4().hex[:8])
    save_json(pointer, {"directory": str(output)}, replace=pointer.exists())
    log = _child(folder, f"evaluate_{seed}", [sys.executable, "-B", "-u", "evaluate_grouped_codec.py",
        "--seed", str(seed), "--evaluate", "--output-dir", str(output)])
    return dict(output_directory=str(output), log=log, _evidence=verify_evaluation(output, seed) + [log])


def run_batch(resume=False):
    if os.name != "posix":
        raise RuntimeError("Execute the batch on the CUDA server, not the laptop")
    owner = SimpleNamespace(EXP_NAME="grouped_codec_batch", arm="controller", SEED=0)
    with run_lock(owner, resume=resume, allowed_root=OUTPUT_ROOT):
        pointer = OUTPUT_ROOT / "last_batch.json"
        if resume:
            folder = safe_inside(read_json(pointer)["directory"], BATCH_ROOT)
            summary = read_json(folder / "batch_summary.json")
            if summary["source"]["sha256"] != source_record()["sha256"]:
                raise RuntimeError("Relevant source changed; refuse silent protocol changes")
        else:
            if pointer.exists():
                raise FileExistsError("Existing batch; explicitly use --resume")
            for seed in SEEDS:
                for arm in ARMS:
                    cfg = grouped_codec_config(arm, seed)
                    if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
                        raise FileExistsError("Unowned new-arm output; refuse overwrite")
            folder = safe_inside(BATCH_ROOT / ("batch_" + uuid.uuid4().hex), ROOT)
            folder.mkdir(parents=True, exist_ok=False)
            summary = dict(status="running", created=now(), source=source_record(), plan=plan(),
                stages={}, cleanup="none", training={}, historical_runs_retrained=False)
            save_json(folder / "batch_summary.json", summary)
            save_json(pointer, {"directory": str(folder)})
        summary["status"] = "running"
        try:
            environment = _run_stage(summary, folder, "environment", lambda: _stage_child(folder, "environment"))["report"]
            if resume:
                current = _stage_child(folder, "environment")["report"]
                for key in ("torch", "sionna", "matplotlib", "channel_backend", "cuda"):
                    if current[key] != environment[key]:
                        raise RuntimeError(f"Resume backend mismatch: {key}")
            summary["preflight"] = _run_stage(summary, folder, "new_cuda_preflight", lambda: _preflight(folder))["report"]
            for seed in SEEDS:
                summary["training"].setdefault(str(seed), {})
                for arm in ARMS:
                    summary["training"][str(seed)][arm] = _run_stage(summary, folder, f"train_{arm}_{seed}",
                        lambda arm=arm, seed=seed: _train_one(folder, arm, seed))["report"]
            evaluations = {str(seed): _run_stage(summary, folder, f"evaluate_{seed}",
                lambda seed=seed: _evaluate_one(folder, seed))["output_directory"] for seed in SEEDS}
            from grouped_codec_results import organize_results
            summary["results"] = _run_stage(summary, folder, "organize_results",
                lambda: organize_results(folder, evaluations, summary["training"]))
            summary.update(status="completed", finished=now())
        except BaseException as error:
            summary.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                resume_command="python -B -u run_grouped_codec_batch.py --run --resume")
            raise
        finally:
            try:
                _package(folder, summary)
            except BaseException as error:
                summary.update(status="failed", package_error=f"{type(error).__name__}: {error}",
                    return_files=[str(folder / "batch_summary.json")])
                save_json(folder / "batch_summary.json", summary, replace=True)
                raise
    return folder


def print_summary():
    pointer = OUTPUT_ROOT / "last_batch.json"
    if not pointer.exists():
        print("not_started: no experimental outcome inferred")
        return 0
    folder = safe_inside(read_json(pointer)["directory"], BATCH_ROOT)
    summary = read_json(folder / "batch_summary.json")
    print(json.dumps(dict(status=summary["status"], error=summary.get("error"),
        summary_path=str(folder / "batch_summary.json"),
        stages={k: v["status"] for k, v in summary["stages"].items()},
        return_files=summary.get("return_files", [])), indent=2))
    for seed, arms in summary.get("training", {}).items():
        for arm, record in arms.items():
            print(f"train {seed} {arm}: latest={record['completed_updates']} best={record['best_step']} sha256={record['best_sha256']}")
    for arm, record in summary.get("preflight", {}).get("arms", {}).items():
        print(f"preflight {arm}: params={record['parameters']} peak_MiB={record.get('peak_allocated_bytes',0)/2**20:.1f} passed={record['passed']}")
    from evaluate_grouped_codec import METHODS, GAMMAS
    for seed, result in summary.get("results", {}).get("per_seed", {}).items():
        print(f"seed {seed} | gamma={list(GAMMAS)} | D-E engineering rule={result['engineering_criterion_D_minus_E']}")
        for method in METHODS:
            values = sorted((r for r in result["all_rates"] if r["method_key"] == method), key=lambda r: float(r["gamma_db"]))
            print(method + ": " + " ".join(f"{float(r['sum_rate_mean']):.4f}" for r in values))
        for name, comparison in result["comparisons"].items():
            print(name + " delta/sample95CI/batch95CI:")
            for label, row in [(str(g), r) for g, r in zip(GAMMAS, comparison["per_gamma"])] + [
                    ("low0/5/10", comparison["low_gamma"]), ("all6", comparison["all_gamma"])]:
                print(f" {label}: {float(row['delta_mean']):+.4f} sample[{float(row['delta_ci95_low']):+.4f},{float(row['delta_ci95_high']):+.4f}] batch[{float(row['batch_delta_ci95_low']):+.4f},{float(row['batch_delta_ci95_high']):+.4f}]")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for name in ("dry-run", "run", "summary"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_stage", choices=("environment", "training_record"), help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--_seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._stage:
        if args._output is None or args.run or args.resume or args.dry_run or args.summary:
            parser.error("Internal stage requires only explicit output and identity")
        if args._stage == "training_record" and (args._arm is None or args._seed is None):
            parser.error("Training record requires arm and seed")
        output = safe_inside(args._output, BATCH_ROOT)
        if output.exists():
            raise FileExistsError(output)
        try:
            result = _environment() if args._stage == "environment" else _training_record(args._arm, args._seed)
            save_json(output, dict(result, passed=True))
        except BaseException as error:
            if not output.exists():
                save_json(output, dict(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
            raise
    elif args.resume and not args.run:
        parser.error("--resume requires --run")
    elif args.dry_run:
        print(json.dumps(plan(), indent=2))
    elif args.summary:
        return print_summary()
    elif args.run:
        run_batch(args.resume)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
