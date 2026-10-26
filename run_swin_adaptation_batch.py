#!/usr/bin/env python3
"""Archive verified ASSpatial, adapt only Swin, and evaluate six fixed methods.

Default/help and dry-run never load weights. Explicit --run is CUDA-server only.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/swin_adaptation_v1"
BATCH_ROOT = OUTPUT_ROOT / "batches"
EVAL_ROOT = ROOT / "runs_as_screen/evaluation/k4_swin_adaptation_v1"
SEEDS = (20261030, 20261031)
SOURCE_FILES = (
    "run_swin_adaptation_batch.py", "swin_adaptation_results.py", "as_spatial_archive.py",
    "configs/swin_adapt_train_config.py", "train_swin_adaptation.py", "evaluate_swin_adaptation.py",
    "baseline_train.py", "baseline_evaluate.py", "configs/baseline_config.py",
    "models/swin_precoder.py", "models/baseline_common.py", "models/adaptive_hybrid.py",
    "models/adaptive_hybrid_as_receiver.py", "models/adaptive_hybrid_as_spatial.py",
    "evaluate_proposal_as_spatial.py", "evaluate_proposal_as_receiver.py", "evaluate_proposal_as_direct.py",
    "evaluate_proposal_lite.py", "configs/as_spatial_train_config.py", "configs/as_receiver_train_config.py",
    "train_proposal_as_spatial.py", "train_proposal_as_receiver.py", "main_train.py",
    "preflight_proposal_as_direct.py", "utils/replayable_channel_data.py", "utils/sionna_rng.py",
    "utils/channel_generator.py", "utils/layers.py", "utils/metrics.py",
    "run_as_receiver_batch.py", "run_as_spatial_batch.py", "as_receiver_batch_support.py",
)


def source_record():
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                sha256={name: sha256(ROOT / name) for name in SOURCE_FILES})


def plan():
    from configs.swin_adapt_train_config import swin_adapt_config
    from evaluate_swin_adaptation import protocol, checkpoint_plan
    return dict(batch="swin_adaptation_v1", training=[dict(seed=seed,
        total_updates=20000, included_smoke_updates=500, source_step0_competes_for_best=True,
        best=swin_adapt_config(seed).BEST_PATH, latest=swin_adapt_config(seed).SAVE_PATH) for seed in SEEDS],
        evaluations={str(seed): protocol(seed) for seed in SEEDS},
        checkpoints={str(seed): checkpoint_plan(seed) for seed in SEEDS},
        frozen_proposals=["Full", "ASReceiver", "ASSpatial frozen", "ASSpatial joint"],
        old_experiments_retrained=False, cleanup=False,
        archive_existing="completed ASSpatial small CSV/JSON/manifest with provenance hashes",
        archive_new="paper/data/k4_swin_adaptation_verified/<batch>",
        inference_scope="K4 pure-AS decoder extension only; two training seeds reported separately",
        budget="Equal additional updates, NOT equal historical total training budget",
        no_test_selection=True, manuscript_modified=False)


def _environment():
    from run_as_receiver_batch import _environment_stage
    report = _environment_stage()  # Dependencies/CUDA/source hashes only; no old test suites.
    report["source"] = source_record()
    report["old_model_or_rng_tests_repeated"] = False
    return report


def _inspect_training(seed):
    import torch
    from configs.swin_adapt_train_config import swin_adapt_config
    from evaluate_swin_adaptation import verify_adapted_metadata
    cfg = swin_adapt_config(seed)
    latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
    best = torch.load(cfg.BEST_PATH, map_location="cpu", weights_only=False)
    status = read_json(cfg.STATUS_PATH)
    record = verify_adapted_metadata(seed, best, latest,
        {"config": read_json(cfg.CONFIG_PATH), "status": status})
    return dict(training_seed=seed, completed_updates=latest["step"], best_step=best["step"],
        best_is_source_step0=best["step"] == 0, best_score=best["best_score"],
        best_path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
        best_sha256=sha256(cfg.BEST_PATH), latest_sha256=sha256(cfg.SAVE_PATH), status=status,
        verified_selection=record)


def _stage_child(folder, name, seed=None):
    output = folder / (name + "_" + uuid.uuid4().hex[:8] + ".json")
    command = [sys.executable, "-B", "-u", "run_swin_adaptation_batch.py",
               "--_stage", name, "--_output", str(output)]
    if seed is not None:
        command += ["--_seed", str(seed)]
    log = _child(folder, name, command)
    report = read_json(output)
    if report.get("passed") is not True:
        raise RuntimeError(f"Stage did not pass: {output}")
    return dict(report=report, report_path=str(output), log=log, _evidence=[output, log])


def _train_one(folder, seed):
    from configs.swin_adapt_train_config import swin_adapt_config
    cfg = swin_adapt_config(seed)
    base = [sys.executable, "-B", "-u", "train_swin_adaptation.py", "--seed", str(seed)]
    logs = []
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
            raise RuntimeError("Unowned Swin outputs without latest; preserve and investigate")
        logs.append(_child(folder, f"swin_{seed}_first500", base + ["--stop-after", "500"]))
    if read_json(cfg.STATUS_PATH).get("state") != "completed":
        logs.append(_child(folder, f"swin_{seed}_continue", base + ["--resume", "--train"]))
    record = _stage_child(folder, "training_record", seed)
    record["_evidence"] += logs + [cfg.SAVE_PATH, cfg.BEST_PATH, cfg.CONFIG_PATH, cfg.STATUS_PATH]
    record["training_logs"] = logs
    return record


def verify_evaluation(output, seed):
    from evaluate_swin_adaptation import protocol, artifact_hashes
    output = safe_inside(output, EVAL_ROOT)
    record = read_json(output / "manifest.json")
    if (record.get("status") != "completed" or record.get("freeze_checks", {}).get("passed") is not True
            or record.get("protocol") != protocol(seed)):
        raise ValueError("New evaluation status/freeze/protocol mismatch")
    source = record.get("runtime", {}).get("source_sha256", {})
    inputs = record.get("input_sha256_after", {})
    if not source or not inputs or inputs != record.get("input_sha256_before"):
        raise ValueError("Missing/changed evaluation input or source provenance")
    for name, digest in source.items():
        if sha256(safe_inside(ROOT / name, ROOT)) != digest:
            raise ValueError(f"New evaluation source changed: {name}")
    for name, digest in inputs.items():
        if sha256(safe_inside(name, ROOT)) != digest:
            raise ValueError(f"Evaluation input changed: {name}")
    if artifact_hashes(output) != record.get("output_sha256"):
        raise ValueError("Evaluation artifacts changed")
    return [output / "manifest.json"] + [output / name for name in record["output_sha256"]]


def _evaluate_one(folder, seed):
    parent = safe_inside(EVAL_ROOT / folder.name / f"seed_{seed}", ROOT)
    for path in sorted(parent.glob("attempt_*/manifest.json")) if parent.exists() else []:
        if read_json(path).get("status") == "completed":
            return dict(output_directory=str(path.parent), reused=True, _evidence=verify_evaluation(path.parent, seed))
    output = parent / ("attempt_" + uuid.uuid4().hex[:8])
    log = _child(folder, f"evaluate_{seed}", [sys.executable, "-B", "-u", "evaluate_swin_adaptation.py",
        "--seed", str(seed), "--evaluate", "--output-dir", str(output)])
    return dict(output_directory=str(output), reused=False, log=log,
                _evidence=verify_evaluation(output, seed) + [log])


def _assert_inputs(expected):
    if not expected:
        raise ValueError("Missing frozen proposal input evidence")
    for path, digest in expected.items():
        if sha256(safe_inside(path, ROOT)) != digest:
            raise RuntimeError(f"Frozen historical input changed: {path}")


def _archive_new(folder, evaluations, summary):
    from as_spatial_archive import archive_evaluations
    from evaluate_swin_adaptation import protocol, OUTPUT_ARTIFACTS
    files = summary["results"]["files"]
    for filename, digest in files.items():
        if sha256(safe_inside(filename, folder)) != digest:
            raise ValueError(f"Result export changed before archival: {filename}")
    small_evidence = {"report/" + Path(name).name: name for name in files
                      if Path(name).suffix in (".csv", ".json")}
    return archive_evaluations(evaluations, root=ROOT,
        archive_root=ROOT / "paper/data/k4_swin_adaptation_verified", archive_id=folder.name,
        expected_protocols={str(seed): protocol(seed) for seed in SEEDS}, required_artifacts=OUTPUT_ARTIFACTS,
        provenance=dict(batch="swin_adaptation_v1", source=summary["source"], training=summary["training"],
                        historical_archive=summary["historical_archive"], scope=summary["plan"]["inference_scope"],
                        draft_and_figure_files=files),
        evidence_files=small_evidence)


def run_batch(resume=False):
    if os.name != "posix":
        raise RuntimeError("Run on the CUDA server only, not the preparation laptop")
    from configs.swin_adapt_train_config import swin_adapt_config
    from as_spatial_archive import archive_completed_as_spatial
    identity = SimpleNamespace(EXP_NAME="swin_adaptation_batch_v1", arm="controller", SEED=0)
    with run_lock(identity, resume=resume, allowed_root=OUTPUT_ROOT):
        pointer = OUTPUT_ROOT / "last_batch.json"
        if resume:
            folder = safe_inside(read_json(pointer)["directory"], BATCH_ROOT)
            summary = read_json(folder / "batch_summary.json")
            if summary["source"]["sha256"] != source_record()["sha256"]:
                raise RuntimeError("Relevant source changed; refuse silent protocol change on resume")
        else:
            if pointer.exists():
                raise FileExistsError("Existing batch; explicitly use --resume")
            for seed in SEEDS:
                cfg = swin_adapt_config(seed)
                if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
                    raise FileExistsError("Unowned Swin adaptation output; refuse overwrite")
            folder = safe_inside(BATCH_ROOT / ("batch_" + uuid.uuid4().hex), ROOT)
            folder.mkdir(parents=True, exist_ok=False)
            summary = dict(status="running", created=now(), source=source_record(), plan=plan(), stages={},
                           cleanup="none; preserve all historical experiments")
            save_json(folder / "batch_summary.json", summary)
            save_json(pointer, {"directory": str(folder)})
        summary["status"] = "running"
        try:
            archive = _run_stage(summary, folder, "archive_completed_as_spatial", archive_completed_as_spatial)
            summary["historical_archive"] = archive
            frozen_inputs = archive["frozen_input_sha256"]
            _assert_inputs(frozen_inputs)
            env = _run_stage(summary, folder, "environment", lambda: _stage_child(folder, "environment"))["report"]
            if resume:
                actual = _stage_child(folder, "environment")["report"]
                for key in ("torch", "sionna", "matplotlib", "channel_backend", "checkpoint_sha256"):
                    if actual[key] != env[key]:
                        raise RuntimeError(f"Resume environment/source mismatch: {key}")
            summary["training"] = {}
            for seed in SEEDS:
                summary["training"][str(seed)] = _run_stage(summary, folder, f"train_swin_{seed}",
                    lambda seed=seed: _train_one(folder, seed))["report"]
            _assert_inputs(frozen_inputs)
            evaluations = {str(seed): _run_stage(summary, folder, f"evaluate_{seed}",
                           lambda seed=seed: _evaluate_one(folder, seed))["output_directory"] for seed in SEEDS}
            from swin_adaptation_results import organize_results
            summary["results"] = _run_stage(summary, folder, "organize_results",
                lambda: organize_results(folder, evaluations, summary["training"]))
            summary["paper_archive"] = _run_stage(summary, folder, "archive_new_control",
                lambda: _archive_new(folder, evaluations, summary))
            _assert_inputs(frozen_inputs)
            summary.update(status="completed", finished=now())
        except BaseException as error:
            summary.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                           resume_command="python -B -u run_swin_adaptation_batch.py --run --resume")
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
        print(json.dumps(dict(status="not_started", message="No local batch results; no experiment outcome inferred.")))
        return 0
    folder = safe_inside(read_json(pointer)["directory"], BATCH_ROOT)
    summary = read_json(folder / "batch_summary.json")
    print(json.dumps(dict(status=summary["status"], error=summary.get("error"),
        stages={key: row["status"] for key, row in summary["stages"].items()},
        training=summary.get("training", {}),
        historical_archive=summary.get("historical_archive", {}).get("archive_directory"),
        paper_archive=summary.get("paper_archive", {}).get("archive_directory"),
        return_files=summary.get("return_files", [])), indent=2, ensure_ascii=False))
    results = summary.get("results", {})
    if results:
        print("\n=== All six SNRs and signed paired results, separate training seeds ===")
        print(json.dumps(results["per_seed"], indent=2, ensure_ascii=False))
        print("\n=== Actual stored parameters (not latency) ===")
        for seed in SEEDS:
            filename = Path(results["export_directory"]) / f"seed{seed}_parameters_k4_swin_adaptation_v1.csv"
            print(filename.read_text(encoding="utf-8"))
        print((Path(results["export_directory"]) / "results_draft_zh.md").read_text(encoding="utf-8"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for option in ("dry-run", "run", "summary"):
        mode.add_argument("--" + option, action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_stage", choices=("environment", "training_record"), help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._stage:
        if args._output is None or args.run or args.resume or args.dry_run or args.summary:
            parser.error("Internal stage requires only an explicit output")
        if args._stage == "training_record" and args._seed is None:
            parser.error("Training record requires a seed")
        output = safe_inside(args._output, BATCH_ROOT)
        if output.exists():
            raise FileExistsError(output)
        try:
            result = _environment() if args._stage == "environment" else _inspect_training(args._seed)
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
