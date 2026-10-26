#!/usr/bin/env python3
"""One isolated ASSpatial screening batch; only explicit --run executes CUDA work."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback
from types import SimpleNamespace
import uuid

from as_receiver_batch_support import safe_inside, save_json, sha256
from run_as_receiver_batch import _child, _run_stage, now, read_json
from train_proposal_as_receiver import run_lock
from configs.as_spatial_train_config import ARMS, SEEDS, OUTPUT_ROOT, as_spatial_config

ROOT = Path(__file__).resolve().parent
BATCH_ROOT = OUTPUT_ROOT / "batches"
EVAL_ROOT = ROOT / "runs_as_screen/evaluation/k4_as_spatial_v1"
SOURCE_FILES = (
    "run_as_spatial_batch.py", "as_spatial_results.py", "configs/as_spatial_train_config.py",
    "train_proposal_as_spatial.py", "evaluate_proposal_as_spatial.py",
    "models/adaptive_hybrid_as_spatial.py", "models/adaptive_hybrid.py",
    "models/adaptive_hybrid_as_receiver.py", "models/swin_precoder.py", "models/baseline_common.py",
    "main_train.py", "baseline_evaluate.py", "utils/replayable_channel_data.py", "utils/sionna_rng.py",
    "utils/channel_generator.py", "utils/layers.py", "utils/metrics.py",
    "run_as_receiver_batch.py", "as_receiver_batch_support.py", "train_proposal_as_receiver.py",
    "evaluate_proposal_as_receiver.py", "configs/as_receiver_train_config.py",
    "tests/test_adaptive_hybrid_as_spatial.py",
)


def source_record():
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                sha256={name: sha256(ROOT / name) for name in SOURCE_FILES})


def plan():
    from evaluate_proposal_as_spatial import protocol, checkpoint_plan
    return dict(batch="as_spatial_v1", new_runs=[dict(arm=arm, training_seed=seed,
                total_updates=20000, included_smoke_updates=500,
                best=as_spatial_config(arm, seed).BEST_PATH) for seed in SEEDS for arm in ARMS],
                tests={str(seed): protocol(seed) for seed in SEEDS},
                checkpoints={str(seed): checkpoint_plan(seed) for seed in SEEDS},
                old_controls_retrained=False, cleanup=False, architecture="new BS; original UE source",
                criteria=dict(reference="same-seed Full +20k", low_mean_min=0.5,
                              low_batch_ci_lower_strict_min=0.0, fb25_batch_ci_lower_min=-0.1),
                no_posthoc_seed_or_snr_search=True, training_seeds_pooled=False)


def _reference_evidence():
    """Read/validate completed controls, never invoke an old training/evaluation run."""
    import torch
    from configs.as_receiver_train_config import as_receiver_config
    from evaluate_proposal_as_receiver import verify_arm_metadata
    from evaluate_proposal_as_direct import archived_swin
    from preflight_proposal_as_direct import load_source_record, verify_checkpoint_hash, validate_checkpoint_metadata
    evidence, rows = {}, []
    source = load_source_record()
    verify_checkpoint_hash(source["path"], source["checkpoint_sha256"])
    payload = torch.load(source["path"], map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(payload, source)
    evidence[source["path"]] = sha256(source["path"])
    del payload
    swin = archived_swin()
    verify_checkpoint_hash(swin["path"], swin["checkpoint_sha256"])
    evidence[swin["path"]] = sha256(swin["path"])
    for seed in SEEDS:
        for arm in ("full_control", "as_receiver"):
            cfg = as_receiver_config(arm, seed)
            for filename in (cfg.BEST_PATH, cfg.SAVE_PATH, cfg.CONFIG_PATH, cfg.STATUS_PATH):
                if not Path(filename).is_file():
                    raise FileNotFoundError(f"Required completed old control missing; no auto-retraining: {filename}")
                evidence[filename] = sha256(filename)
            best = torch.load(cfg.BEST_PATH, map_location="cpu", weights_only=False)
            latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
            record = verify_arm_metadata(arm, seed, best, latest,
                        {"config": read_json(cfg.CONFIG_PATH), "status": read_json(cfg.STATUS_PATH)})
            rows.append(dict(arm=arm, training_seed=seed, selected_step=record["selected_best_step"],
                             best_path=cfg.BEST_PATH, best_sha256=evidence[cfg.BEST_PATH]))
            del best, latest
    return dict(checkpoint_and_sidecar_sha256=evidence, completed_old_controls=rows)


def _environment():
    from run_as_receiver_batch import _environment_stage
    report = _environment_stage()  # Existing CUDA/version/source identity check, not its model/data tests.
    report["references"] = _reference_evidence()
    report["source"] = source_record()
    report["old_experiments_rerun"] = False
    report["data_protocol"] = as_spatial_config(ARMS[0], SEEDS[0]).channel_data_protocol
    return report


def _model_checks():
    import torch
    import unittest
    from main_train import set_seed, SumRateLoss
    from train_proposal_as_spatial import ue_state_digest
    from models.adaptive_hybrid_as_spatial import AdaptiveHybridPrecoderASSpatial
    from preflight_proposal_as_direct import load_source_record, verify_checkpoint_hash, validate_checkpoint_metadata
    if not torch.cuda.is_available():
        raise RuntimeError("ASSpatial checks require real CUDA; no CPU fallback")
    os.environ["AS_SPATIAL_TEST_DEVICE"] = "cuda"
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_adaptive_hybrid_as_spatial.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"Focused ASSpatial CUDA checks failed/skipped: failures={len(result.failures)}, errors={len(result.errors)}, skips={result.skipped}")
    source = load_source_record()
    verify_checkpoint_hash(source["path"], source["checkpoint_sha256"])
    payload = torch.load(source["path"], map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(payload, source)
    models, counts = [], {}
    for arm in ARMS:
        cfg = as_spatial_config(arm, SEEDS[0])
        set_seed(cfg.SEED)
        model = AdaptiveHybridPrecoderASSpatial(cfg)
        model.initialize_ue_from_full(payload["state"])
        counts[arm] = dict(total=sum(p.numel() for p in model.parameters()),
                          ue_encoder_condition=sum(p.numel() for p in model.ue_parameters()),
                          new_bs=sum(p.numel() for p in model.bs_parameters()),
                          trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        models.append(model)
    first, second = models
    for key, tensor in first.state_dict().items():
        if not torch.equal(tensor, second.state_dict()[key]):
            raise RuntimeError(f"Two arms did not receive identical initial UE/BS state: {key}")
    for model in models:
        for key, value in model.state_dict().items():
            if key.startswith(("encoder.", "cond_embed.")) or key == "running_pwr":
                if not torch.equal(value, payload["state"][key]):
                    raise RuntimeError(f"Source UE migration altered {key}")
    initial_bs_hashes, real_source_checks = {}, []
    set_seed(20261105)
    shape = (2, 4, 32, 256)
    h_dl = torch.randn(shape, dtype=torch.complex64, device="cuda")
    h_ul = torch.randn(shape, dtype=torch.complex64, device="cuda")
    for arm, model in zip(ARMS, models):
        digest = hashlib.sha256()
        for name, value in model.decoder.state_dict().items():
            digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        initial_bs_hashes[arm] = digest.hexdigest()
        model.cuda().train()
        before_ue = ue_state_digest(torch, model.state_dict())
        set_seed(20261106)
        _, h_hat = model(h_dl, h_ul, 10.0, 0, 256, 0)
        w = model.zf_solver(h_hat)
        w = w * torch.sqrt(model.cfg.TOTAL_POWER / (w.abs().square().sum((1, 2), keepdim=True) + 1e-9))
        loss, _ = SumRateLoss()(w, h_dl, 10**(-25/10))
        if not torch.isfinite(h_hat).all() or not torch.isfinite(w).all() or not torch.isfinite(loss):
            raise RuntimeError("Nonfinite real-source synthetic task path")
        loss.backward()  # No optimizer, weight update, CDL or performance evaluation.
        active = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        if not active or any(not torch.isfinite(grad).all() for _, grad in active):
            raise RuntimeError("Nonfinite/missing real-source gradients")
        if not any(name.startswith("decoder.") and grad.abs().sum().item() > 0 for name, grad in active):
            raise RuntimeError("New BS has no nonzero task gradient")
        power_error = float((w.abs().square().sum((1, 2)) - model.cfg.TOTAL_POWER).abs().max())
        if power_error > 2e-6:
            raise RuntimeError("Per-subcarrier power constraint failed")
        after_ue = ue_state_digest(torch, model.state_dict())
        if arm == "frozen_ue" and (before_ue != after_ue or any(n.startswith(("encoder.", "cond_embed.")) for n, _ in active)):
            raise RuntimeError("Frozen real-source UE/state changed or received gradients")
        if arm == "joint" and not any(n.startswith("encoder.") and g.abs().sum().item() > 0 for n, g in active):
            raise RuntimeError("Joint UE task gradient disconnected")
        real_source_checks.append(dict(arm=arm, finite=True, power_max_abs=power_error,
                ue_state_unchanged=before_ue == after_ue, gradient_parameter_count=len(active)))
        model.cpu()
    if len(set(initial_bs_hashes.values())) != 1:
        raise RuntimeError("Initial BS hashes differ between arms")
    verify_checkpoint_hash(source["path"], source["checkpoint_sha256"])
    return dict(passed=True, source_sha256=source["checkpoint_sha256"],
                same_initial_state_between_arms=True, source_ue_exact=True, actual_parameters=counts,
                initial_bs_sha256=initial_bs_hashes, real_source_synthetic_checks=real_source_checks,
                focused_cuda_checks=result.testsRun, comparison_to_full_zero_output="not_applicable_new_random_decoder")


def _inspect_training(arm, seed):
    import torch
    from train_proposal_as_spatial import validate_resume_payload, validate_best_metadata
    cfg = as_spatial_config(arm, seed)
    latest = torch.load(cfg.SAVE_PATH, map_location="cpu", weights_only=False)
    best = torch.load(cfg.BEST_PATH, map_location="cpu", weights_only=False)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    status = read_json(cfg.STATUS_PATH)
    if latest["step"] != 20000 or status.get("state") != "completed" or status.get("step") != 20000:
        raise ValueError("Expected a completed 20k new run")
    return dict(arm=arm, training_seed=seed, completed_updates=latest["step"], best_step=best["step"],
                best_path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
                best_sha256=sha256(cfg.BEST_PATH), latest_sha256=sha256(cfg.SAVE_PATH),
                best_metric=best["best_metric"], status=status)


def _stage_child(folder, name, arm=None, seed=None):
    output = folder / (name + "_" + uuid.uuid4().hex[:8] + ".json")
    command = [sys.executable, "-B", "-u", "run_as_spatial_batch.py", "--_stage", name, "--_output", str(output)]
    if arm is not None:
        command += ["--_arm", arm, "--_seed", str(seed)]
    log = _child(folder, name, command)
    report = read_json(output)
    if report.get("passed") is not True:
        raise RuntimeError(f"Stage did not pass: {output}")
    return dict(report=report, report_path=str(output), log=log, _evidence=[output, log])


def _train_one(folder, arm, seed):
    cfg = as_spatial_config(arm, seed)
    base = [sys.executable, "-B", "-u", "train_proposal_as_spatial.py", "--arm", arm, "--seed", str(seed)]
    logs = []
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
            raise RuntimeError("Unfinished output without latest: preserve and investigate, do not overwrite")
        logs.append(_child(folder, f"{arm}_{seed}_first500", base + ["--stop-after", "500"]))
    if read_json(cfg.STATUS_PATH).get("state") != "completed":
        logs.append(_child(folder, f"{arm}_{seed}_continue", base + ["--resume", "--train"]))
    record = _stage_child(folder, "training_record", arm, seed)
    record["_evidence"] += logs + [cfg.SAVE_PATH, cfg.BEST_PATH, cfg.CONFIG_PATH, cfg.STATUS_PATH]
    record["training_logs"] = logs
    return record


def verify_evaluation(output, seed):
    from evaluate_proposal_as_spatial import protocol, artifact_hashes
    output = safe_inside(output, EVAL_ROOT)
    record = read_json(output / "manifest.json")
    if (record.get("status") != "completed" or record.get("freeze_checks", {}).get("passed") is not True
            or record.get("protocol") != protocol(seed)):
        raise ValueError("Evaluation status/freeze/protocol mismatch")
    code = record.get("runtime", {}).get("source_sha256", {})
    inputs = record.get("input_sha256_after", {})
    if not code or not inputs:
        raise ValueError("Evaluation source/input evidence missing")
    for name, digest in code.items():
        if sha256(safe_inside(ROOT / name, ROOT)) != digest:
            raise ValueError(f"Evaluation source changed: {name}")
    for name, digest in inputs.items():
        if sha256(name) != digest:
            raise ValueError(f"Evaluation input changed: {name}")
    if artifact_hashes(output) != record.get("output_sha256"):
        raise ValueError("Evaluation artifact hashes changed")
    return [output / "manifest.json"] + [output / name for name in record["output_sha256"]]


def _evaluate_one(folder, seed):
    parent = safe_inside(EVAL_ROOT / folder.name / f"seed_{seed}", ROOT)
    for path in sorted(parent.glob("attempt_*/manifest.json")) if parent.exists() else []:
        if read_json(path).get("status") == "completed":
            return dict(output_directory=str(path.parent), reused=True, _evidence=verify_evaluation(path.parent, seed))
    output = parent / ("attempt_" + uuid.uuid4().hex[:8])
    log = _child(folder, f"evaluate_{seed}", [sys.executable, "-B", "-u", "evaluate_proposal_as_spatial.py",
                 "--seed", str(seed), "--evaluate", "--output-dir", str(output)])
    return dict(output_directory=str(output), reused=False, log=log,
                _evidence=verify_evaluation(output, seed) + [log])


def _assert_references(record):
    for path, digest in record["references"]["checkpoint_and_sidecar_sha256"].items():
        if sha256(path) != digest:
            raise RuntimeError(f"Pinned old control/source changed during this batch: {path}")


def _package(folder, summary):
    # Never treat an interrupted or older package as the current summary.
    prefix = "results_bundle_" if summary["status"] == "completed" else "failure_bundle_"
    bundle = folder / (prefix + uuid.uuid4().hex + ".tar.gz")
    temporary = bundle.with_name(bundle.name + ".partial")
    summary["return_files"] = [str(folder / "batch_summary.json"), str(bundle)]
    save_json(folder / "batch_summary.json", summary, replace=True)
    export = summary.get("results", {}).get("export_directory")
    candidates = list(safe_inside(export, folder).iterdir()) if export else []
    candidates += [p for p in folder.iterdir() if p.suffix in (".json", ".log") and p.name != "batch_summary.json"]
    selected = [folder / "batch_summary.json"]
    for path in sorted(candidates):
        safe_inside(path, folder)
        if path.is_file() and path.suffix in (".csv", ".json", ".md", ".pdf", ".png", ".log") and path.stat().st_size < 8 * 1024**2:
            selected.append(path)
    expected = {path.relative_to(folder).as_posix(): sha256(path) for path in selected}
    with tarfile.open(temporary, "x:gz") as archive:
        for path in selected:
            archive.add(path, arcname=path.relative_to(folder).as_posix(), recursive=False)
    with tarfile.open(temporary, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise RuntimeError("Small result bundle member list mismatch")
        for member in members:
            if not member.isfile():
                raise RuntimeError("Small result bundle contains a non-file")
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected[member.name]:
                raise RuntimeError("Small result bundle content verification failed")
    if any(sha256(folder / name) != digest for name, digest in expected.items()):
        raise RuntimeError("Export evidence changed while packaging")
    os.link(temporary, bundle)  # Atomic, exclusive publication; never overwrite a previous package.
    temporary.unlink()  # Only this invocation's verified temporary archive, not experiment data.
    print(f"BATCH_SUMMARY={folder / 'batch_summary.json'}", flush=True)


def run_batch(resume=False):
    if os.name != "posix":
        raise RuntimeError("Run only on the CUDA server, not the preparation laptop")
    identity = SimpleNamespace(EXP_NAME="as_spatial_batch_v1", arm="controller", SEED=0)
    with run_lock(identity, resume=resume, allowed_root=OUTPUT_ROOT):
        pointer = OUTPUT_ROOT / "last_batch.json"
        if resume:
            folder = safe_inside(read_json(pointer)["directory"], BATCH_ROOT)
            summary = read_json(folder / "batch_summary.json")
            if summary["source"]["sha256"] != source_record()["sha256"]:
                raise RuntimeError("Relevant source changed; do not silently resume a different experiment")
        else:
            if pointer.exists():
                raise FileExistsError("Batch already exists: explicitly use --resume")
            for seed in SEEDS:
                for arm in ARMS:
                    cfg = as_spatial_config(arm, seed)
                    if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
                        raise FileExistsError("Unowned new-arm outputs exist; refuse overwrite/adoption")
            folder = safe_inside(BATCH_ROOT / ("batch_" + uuid.uuid4().hex), ROOT)
            folder.mkdir(parents=True, exist_ok=False)
            summary = dict(status="running", created=now(), source=source_record(), plan=plan(), stages={},
                           cleanup="none; all historical experiments preserved")
            save_json(folder / "batch_summary.json", summary)
            save_json(pointer, {"directory": str(folder)})
        summary["status"] = "running"
        try:
            env = _run_stage(summary, folder, "environment_and_old_controls", lambda: _stage_child(folder, "environment"))["report"]
            if resume:
                actual = _stage_child(folder, "environment")["report"]
                for name in ("channel_backend", "torch", "sionna", "matplotlib", "references"):
                    if actual[name] != env[name]:
                        raise RuntimeError(f"Resume environment/reference mismatch: {name}")
            _assert_references(env)
            _run_stage(summary, folder, "new_model_checks", lambda: _stage_child(folder, "model"))
            summary["training"] = {}
            for seed in SEEDS:
                for arm in ARMS:
                    key = f"train_{arm}_{seed}"
                    summary["training"][key] = _run_stage(summary, folder, key,
                        lambda arm=arm, seed=seed: _train_one(folder, arm, seed))["report"]
            _assert_references(env)
            evaluations = {str(seed): _run_stage(summary, folder, f"evaluate_{seed}",
                            lambda seed=seed: _evaluate_one(folder, seed))["output_directory"] for seed in SEEDS}
            from as_spatial_results import organize_results
            summary["results"] = _run_stage(summary, folder, "organize_results", lambda: organize_results(folder, evaluations))
            _assert_references(env)
            summary.update(status="completed", finished=now())
        except BaseException as error:
            summary.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                           resume_command="python -B -u run_as_spatial_batch.py --run --resume")
            raise
        finally:
            try:
                _package(folder, summary)
            except BaseException as packaging_error:
                summary.update(status="failed", package_error=f"{type(packaging_error).__name__}: {packaging_error}",
                               package_traceback=traceback.format_exc(),
                               return_files=[str(folder / "batch_summary.json")])
                save_json(folder / "batch_summary.json", summary, replace=True)
                raise
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_stage", choices=("environment", "model", "training_record"), help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--_seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._stage:
        if args._output is None or args.run or args.dry_run or args.summary or args.resume:
            parser.error("Internal stage requires only its explicit output")
        output = safe_inside(args._output, BATCH_ROOT)
        if output.exists():
            raise FileExistsError(output)
        try:
            report = (_environment() if args._stage == "environment" else _model_checks()
                      if args._stage == "model" else _inspect_training(args._arm, args._seed))
            save_json(output, dict(report, passed=True))
        except BaseException as error:
            if not output.exists():
                save_json(output, dict(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
            raise
    elif args.resume and not args.run:
        parser.error("--resume requires --run")
    elif args.dry_run:
        print(json.dumps(plan(), indent=2))
    elif args.summary:
        folder = safe_inside(read_json(OUTPUT_ROOT / "last_batch.json")["directory"], BATCH_ROOT)
        print((folder / "batch_summary.json").read_text(encoding="utf-8"))
    elif args.run:
        run_batch(args.resume)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
