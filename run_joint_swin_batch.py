#!/usr/bin/env python3
"""One explicit CUDA batch: new checks, four 20k runs, six-method tests and reports."""
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
from configs.joint_swin_train_config import ARMS, SEEDS, OUTPUT_ROOT, joint_swin_config

ROOT = Path(__file__).resolve().parent
BATCH_ROOT = OUTPUT_ROOT / "batches"
EVAL_ROOT = ROOT / "runs_as_screen/evaluation/k4_joint_swin256_v1"


def source_record():
    from train_proposal_joint_swin import SOURCE_FILES as training
    from evaluate_proposal_joint_swin import SOURCE_FILES as evaluation
    files = tuple(dict.fromkeys(training + evaluation + (
        "run_joint_swin_batch.py", "preflight_joint_swin.py", "joint_swin_results.py",
        "tests/test_adaptive_hybrid_joint_swin.py", "run_as_receiver_batch.py",
        "run_as_spatial_batch.py", "as_receiver_batch_support.py")))
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                sha256={name: sha256(ROOT / name) for name in files})


def plan():
    from evaluate_proposal_joint_swin import protocol, checkpoint_plan
    from preflight_joint_swin import plan as preflight_plan
    return dict(batch="joint_swin256_v1", training=[dict(arm=arm, seed=seed,
        total_updates=20000, included_smoke_updates=500, best=joint_swin_config(arm, seed).BEST_PATH)
        for seed in SEEDS for arm in ARMS],
        evaluations={str(seed): protocol(seed) for seed in SEEDS},
        checkpoints={str(seed): checkpoint_plan(seed) for seed in SEEDS}, preflight=preflight_plan(),
        old_controls_retrained=False, cleanup=False, visits_08=False,
        criteria=dict(reference="same-seed plain_swin256", low_mean_min=.5,
                      low_batch_ci95_lower_strictly_greater=0, fb25_batch_ci95_lower_min=-.1,
                      interpretation="engineering screen; not a Swin-superiority or journal acceptance claim"),
        training_seeds_pooled=False, test_seed_checkpoint_snr_search=False, manuscript_modified=False)


def _environment():
    import torch
    from run_as_receiver_batch import _environment_stage
    from evaluate_proposal_joint_swin import checkpoint_plan, build_method
    from evaluate_proposal_as_direct import load_payload, verify_archived_metadata, check_tensor_state
    import evaluate_proposal_as_spatial as spatial
    import evaluate_swin_adaptation as adapted
    import baseline_evaluate as ev
    report = _environment_stage()  # Dependency/source checks only; no old model/RNG test suites.
    hashes, records = {}, []
    scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
    scenario.validate()
    for seed in SEEDS:
        source_swin_state = None
        for entry in checkpoint_plan(seed):
            key = entry["method_key"]
            if key in ARMS:
                continue
            if key == "swin":
                best = load_payload(torch, entry["path"], hashes, entry["expected_best_sha256"])
                record = verify_archived_metadata(entry, best)
                source_swin_state = best["state"]
            else:
                latest = load_payload(torch, entry["latest_path"], hashes)
                best = load_payload(torch, entry["path"], hashes)
                sidecars = {name: read_json(entry[name + "_path"]) for name in ("config", "status")}
                hashes.update({entry[name + "_path"]: sha256(entry[name + "_path"]) for name in sidecars})
                record = (adapted.verify_adapted_metadata(seed, best, latest, sidecars) if key == "swin_adapted"
                          else spatial.verify_arm_metadata(entry["arm"], seed, best, latest, sidecars))
                check_tensor_state(torch, latest["state"], best["state"])
                if best["step"] == latest["step"]:
                    check_tensor_state(torch, latest["state"], best["state"], equal=True)
                del latest
            # Real strict loading, one historical control on CPU at a time; no forward/evaluation.
            method = build_method(torch, ev, scenario, entry, best, torch.device("cpu"))
            if key == "swin_adapted" and best["step"] == 0:
                check_tensor_state(torch, source_swin_state, best["state"], equal=True)
            records.append(dict(training_seed=seed, method=key, best_step=best["step"],
                                path=entry["path"], sha256=hashes[entry["path"]], strict_load=True,
                                metadata=record))
            del method, best
        del source_swin_state
    hashes.update(report["checkpoint_sha256"])
    report.update(frozen_inputs=hashes, old_controls=records, source=source_record(), old_tests_repeated=False)
    return report


def _training_record(arm, seed):
    import torch
    from evaluate_proposal_joint_swin import verify_new_metadata
    from train_proposal_joint_swin import validate_model_identity, validate_model_state
    from evaluate_proposal_as_direct import check_tensor_state
    from models.adaptive_hybrid_joint_swin import AdaptiveHybridPrecoderJointSwin
    cfg = joint_swin_config(arm, seed)
    latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
    best = torch.load(cfg.BEST_PATH, map_location="cpu", weights_only=False)
    record = verify_new_metadata(arm, seed, best, latest,
        {"config": read_json(cfg.CONFIG_PATH), "status": read_json(cfg.STATUS_PATH)})
    model = AdaptiveHybridPrecoderJointSwin(cfg)
    validate_model_identity(cfg, model)
    # Completion metadata already validated step20k; strict tensors independently checked.
    validate_model_state(torch, model, latest["state"])
    validate_model_state(torch, model, best["state"])
    if latest["step"] == best["step"]:
        check_tensor_state(torch, latest["state"], best["state"], equal=True)
    model.load_state_dict(best["state"], strict=True)
    return dict(arm=arm, training_seed=seed, completed_updates=latest["step"], best_step=best["step"],
        best_metric=best["best_metric"], best_path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
        best_sha256=sha256(cfg.BEST_PATH), latest_sha256=sha256(cfg.SAVE_PATH), verified_selection=record)


def _stage_child(folder, name, arm=None, seed=None):
    output = folder / (name + "_" + uuid.uuid4().hex[:8] + ".json")
    command = [sys.executable, "-B", "-u", "run_joint_swin_batch.py", "--_stage", name,
               "--_output", str(output)]
    if arm is not None:
        command += ["--_arm", arm, "--_seed", str(seed)]
    log = _child(folder, name, command)
    report = read_json(output)
    if report.get("passed") is not True:
        raise RuntimeError(f"Stage failed: {output}")
    return dict(report=report, report_path=str(output), log=log, _evidence=[output, log])


def _preflight(folder):
    environment = dict(os.environ, JOINT_SWIN_TEST_DEVICE="cuda")
    log = _child(folder, "new_structural_tests", [sys.executable, "-B", "-m", "unittest", "discover",
        "-s", "tests", "-p", "test_adaptive_hybrid_joint_swin.py", "-v"], environment=environment)
    output = folder / ("cuda_batch48_" + uuid.uuid4().hex[:8])
    retained_report = folder / (output.name + ".json")
    try:
        cuda_log = _child(folder, "cuda_batch48", [sys.executable, "-B", "-u", "preflight_joint_swin.py",
            "--run", "--output-dir", str(output)])
    finally:
        # The existing small-bundle helper includes top-level JSON, not nested
        # directories. Retain failed/OOM evidence too, before _child re-raises.
        if (output / "report.json").is_file():
            save_json(retained_report, read_json(output / "report.json"))
    report = read_json(output / "report.json")
    if report.get("passed") is not True:
        raise RuntimeError("Batch48 CUDA preflight failed; no training permitted")
    return dict(report=report, report_path=str(output / "report.json"),
                _evidence=[log, cuda_log, output / "report.json", retained_report])


def _train_one(folder, arm, seed):
    cfg = joint_swin_config(arm, seed)
    command = [sys.executable, "-B", "-u", "train_proposal_joint_swin.py", "--arm", arm, "--seed", str(seed)]
    logs = []
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
            raise RuntimeError("Existing new-arm output without latest: preserve and investigate")
        logs.append(_child(folder, f"{arm}_{seed}_first500", command + ["--stop-after", "500"]))
    if read_json(cfg.STATUS_PATH).get("state") != "completed":
        logs.append(_child(folder, f"{arm}_{seed}_continue", command + ["--resume", "--train"]))
    result = _stage_child(folder, "training_record", arm, seed)
    result["_evidence"] += logs + [cfg.BEST_PATH, cfg.SAVE_PATH, cfg.STATUS_PATH, cfg.CONFIG_PATH]
    return result


def verify_evaluation(output, seed):
    from evaluate_proposal_joint_swin import protocol, artifact_hashes
    output = safe_inside(output, EVAL_ROOT)
    manifest = read_json(output / "manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("freeze_checks", {}).get("passed") is not True
            or manifest.get("protocol") != protocol(seed)):
        raise ValueError("Evaluation protocol/status/freeze mismatch")
    inputs = manifest.get("input_sha256_after", {})
    code = manifest.get("runtime", {}).get("source_sha256", {})
    if not inputs or not code or inputs != manifest.get("input_sha256_before"):
        raise ValueError("Missing/changed evaluation provenance")
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
    log = _child(folder, f"evaluate_{seed}", [sys.executable, "-B", "-u", "evaluate_proposal_joint_swin.py",
        "--seed", str(seed), "--evaluate", "--output-dir", str(output)])
    return dict(output_directory=str(output), reused=False, log=log,
                _evidence=verify_evaluation(output, seed) + [log])


def _assert_frozen(hashes):
    if not hashes:
        raise ValueError("Missing frozen historical input identities")
    for filename, digest in hashes.items():
        if sha256(safe_inside(filename, ROOT)) != digest:
            raise RuntimeError(f"Historical input changed: {filename}")


def run_batch(resume=False):
    if os.name != "posix":
        raise RuntimeError("Run only on the CUDA server, not the preparation laptop")
    owner = SimpleNamespace(EXP_NAME="joint_swin256_batch", arm="controller", SEED=0)
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
                    cfg = joint_swin_config(arm, seed)
                    if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
                        raise FileExistsError("Unowned new-arm output; refuse overwrite")
            folder = safe_inside(BATCH_ROOT / ("batch_" + uuid.uuid4().hex), ROOT)
            folder.mkdir(parents=True, exist_ok=False)
            summary = dict(status="running", created=now(), source=source_record(), plan=plan(), stages={},
                           cleanup="none", training={}, old_controls_retrained=False)
            save_json(folder / "batch_summary.json", summary)
            save_json(pointer, {"directory": str(folder)})
        summary["status"] = "running"
        try:
            environment = _run_stage(summary, folder, "environment", lambda: _stage_child(folder, "environment"))["report"]
            _assert_frozen(environment["frozen_inputs"])
            if resume:
                current = _stage_child(folder, "environment")["report"]
                for key in ("torch", "sionna", "matplotlib", "channel_backend", "frozen_inputs"):
                    if current[key] != environment[key]:
                        raise RuntimeError(f"Resume environment/identity mismatch: {key}")
            summary["preflight"] = _run_stage(summary, folder, "new_cuda_preflight", lambda: _preflight(folder))["report"]
            for seed in SEEDS:
                summary["training"][str(seed)] = {}
                for arm in ARMS:
                    summary["training"][str(seed)][arm] = _run_stage(summary, folder, f"train_{arm}_{seed}",
                        lambda arm=arm, seed=seed: _train_one(folder, arm, seed))["report"]
            _assert_frozen(environment["frozen_inputs"])
            evaluations = {str(seed): _run_stage(summary, folder, f"evaluate_{seed}",
                lambda seed=seed: _evaluate_one(folder, seed))["output_directory"] for seed in SEEDS}
            from joint_swin_results import organize_results
            summary["results"] = _run_stage(summary, folder, "organize_results",
                lambda: organize_results(folder, evaluations, summary["training"]))
            _assert_frozen(environment["frozen_inputs"])
            summary.update(status="completed", finished=now())
        except BaseException as error:
            summary.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                           resume_command="python -B -u run_joint_swin_batch.py --run --resume")
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
        print("not_started: no experiment outcome inferred")
        return 0
    folder = safe_inside(read_json(pointer)["directory"], BATCH_ROOT)
    summary = read_json(folder / "batch_summary.json")
    print(json.dumps(dict(status=summary["status"], error=summary.get("error"),
        summary_path=str(folder / "batch_summary.json"),
        stages={k: v["status"] for k, v in summary["stages"].items()},
        return_files=summary.get("return_files", [])), indent=2))
    for seed, arms in summary.get("training", {}).items():
        for arm, record in arms.items():
            print(f"train {seed} {arm}: latest={record['completed_updates']} best={record['best_step']} best_sha256={record['best_sha256']}")
    for arm, record in summary.get("preflight", {}).get("arms", {}).items():
        print(f"preflight {arm}: params={record['parameters']} peak_MiB={record.get('peak_allocated_bytes',0)/2**20:.1f} passed={record['passed']}")
    from evaluate_proposal_joint_swin import METHODS, FB_SNRS
    for seed, result in summary.get("results", {}).get("per_seed", {}).items():
        print(f"seed {seed} | FB={list(FB_SNRS)} | structural criterion vs plain={result['engineering_criterion_vs_plain']}")
        for method in METHODS:
            values = sorted((r for r in result["all_rates"] if r["method_key"] == method), key=lambda r: float(r["feedback_snr_db"]))
            print(method + ": " + " ".join(f"{float(r['sum_rate_mean']):.4f}" for r in values))
        for pair, result_pair in result["comparisons"].items():
            points = result_pair["per_snr"]
            print(pair + " delta[batch95%CI]: " + "; ".join(
                f"{float(r['delta_mean']):+.4f}[{float(r['batch_delta_ci95_low']):+.4f},{float(r['batch_delta_ci95_high']):+.4f}]" for r in points))
            r = result_pair["low_snr"]
            print(f"  low0/5/10={float(r['delta_mean']):+.4f} [{float(r['batch_delta_ci95_low']):+.4f},{float(r['batch_delta_ci95_high']):+.4f}]")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for option in ("dry-run", "run", "summary"):
        mode.add_argument("--" + option, action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_stage", choices=("environment", "training_record"), help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--_seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._stage:
        if args._output is None or args.run or args.resume or args.dry_run or args.summary:
            parser.error("Internal stage requires only its explicit output and identity")
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
