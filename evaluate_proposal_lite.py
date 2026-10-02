#!/usr/bin/env python3
"""Independent four-model Lite1024 test; no action without an explicit mode.

Only the Lite loader is new. Sampling, physical links, eval-only H_hat capture,
RZF and metrics are delegated to baseline_evaluate without patches or probes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from configs.baseline_config import make_config
from configs.lite_train_config import lite_config
from train_proposal_specialist import checkpoint_status, specialist_config

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_pure_as_lite1024"
SEED = 20261002
FB_SNRS = (0, 5, 10, 15, 20, 25)
DL_SNR = 25
BATCH_SIZE = 8
SCENARIO_ID = "K4_Nsc256_Dfb256"
TAG = "k4_pure_as_lite1024"
BASELINE_PATH_FIELDS = {"OUTPUT_ROOT", "SAVE_DIR", "BEST_PATH", "SAVE_PATH",
                        "LOG_DIR", "METRICS_PATH", "STATUS_PATH"}


def checkpoint_plan(root=ROOT):
    root = Path(root)
    experiment = "hybrid_gnn_specialist_K4_Nsc256_Dfb256_f000_a100_df0_da256_sb4_k4_n256_standard"
    choices = (
        ("original", "proposed", "Original pure-AS specialist (UE FFN2048)", "proposal_specialists", experiment + "_seed42", (0, 256)),
        ("lite1024", "proposed", "Lite1024 pure-AS specialist", "proposal_lite", experiment + "_ueffn1024_seed42", (0, 256)),
        ("swin", "swin", "Swin-based FDMA", "baselines", "swin_task_K4_Nsc256_Dfb256_Df64_sharedUE_seed42", (64, 0)),
        ("csinet", "csinet", "CsiNet+-based FDMA", "baselines", "csinet_task_K4_Nsc256_Dfb256_Df64_sharedUE_seed42", (64, 0)),
    )
    return [dict(method_key=key, model_key=model, label=label,
                 path=root / "runs_as_screen" / directory / "checkpoints" / name / "best.pth",
                 allocations=(allocation,))
            for key, model, label, directory, name, allocation in choices]


def _plain(value):
    return json.loads(json.dumps(value, default=str))


def _compare(expected, actual, prefix):
    """Name the exact conflicting/missing field, not just the experiment folder."""
    expected, actual = _plain(expected), _plain(actual)
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                raise ValueError(f"{prefix}.{key}: expected {expected.get(key, '<absent>')!r}, found {actual.get(key, '<absent>')!r}")
            _compare(expected[key], actual[key], f"{prefix}.{key}")
    elif expected != actual:
        raise ValueError(f"{prefix}: expected {expected!r}, found {actual!r}")


def validate_training_config(entry, config):
    key = entry["method_key"]
    if entry["model_key"] == "proposed":
        expected = (lite_config() if key == "lite1024" else specialist_config((0, 256), ROOT / "runs_as_screen/proposal_specialists")).to_dict()
        for name in ("experiment", "training_recipe", "allocations", "full_allocation_grid", "physical", "proposal_model"):
            _compare(expected[name], config.get(name), f"{key}.config.{name}")
        for name in ("d_f_max_per_user", "d_a_max_shared", "phase1_steps", "stage1_steps", "stage2_steps"):
            _compare(expected["resolved"][name], config.get("resolved", {}).get(name), f"{key}.config.resolved.{name}")
        if key == "lite1024":
            _compare(expected["architecture_metadata"], config.get("architecture_metadata"), f"{key}.architecture_metadata")
        elif "architecture_metadata" in config:
            raise ValueError("original: unexpected architecture_metadata; inspect the supposed original checkpoint")
    else:
        expected = make_config(entry["model_key"], "k4_n256_d256", objective="task").to_dict()
        # Baseline validation batch is 64, not the proposals' 32. Paths are
        # historical provenance, not grounds to reject a relocated checkpoint.
        for name in set(expected) - BASELINE_PATH_FIELDS:
            _compare(expected[name], config.get(name), f"{key}.config.{name}")


def load_sidecars(entry):
    path = Path(entry["path"])
    folders = [path.parent, path.parents[2] / "logs" / path.parent.name]
    result = {"paths": {}, "all_paths": []}
    for name in ("config", "status"):
        found = [folder / (name + ".json") for folder in folders if (folder / (name + ".json")).is_file()]
        if not found:
            raise FileNotFoundError(f"{entry['method_key']}: missing {name} sidecar; checked: " + ", ".join(map(str, (folder / (name + '.json') for folder in folders))))
        values = [json.loads(item.read_text(encoding="utf-8")) for item in found]
        for value in values[1:]:
            _compare(values[0], value, f"{entry['method_key']}.ambiguous_{name}_sidecars")
        result[name], result["paths"][name] = values[0], str(found[0])
        result["all_paths"].extend(map(str, found))
    # Reuse the project's checkpoint/log completion resolver; reject unknown.
    _compare(result["status"], checkpoint_status(path), f"{entry['method_key']}.resolved_status")
    return result


def _saved_fingerprint(config, model_key):
    payload = dict(config)
    if model_key != "proposed":
        payload = {key: value for key, value in payload.items() if key not in BASELINE_PATH_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=model_key != "proposed", default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def verify_checkpoint(entry, best, latest, sidecars):
    key = entry["method_key"]
    for kind, payload in (("best", best), ("latest", latest)):
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError(f"{key}.{kind}: missing saved config")
        validate_training_config(entry, payload["config"])
        step = payload.get("step")
        if type(step) is not int or not 1 <= step <= 50000:
            raise ValueError(f"{key}.{kind}.step: invalid {step!r}")
        actual_fp = payload.get("config_fingerprint")
        expected_fp = _saved_fingerprint(payload["config"], entry["model_key"])
        if actual_fp != expected_fp:
            raise ValueError(f"{key}.{kind}.config_fingerprint: expected {expected_fp}, found {actual_fp!r}")
        grid = payload.get("allocation_grid")
        if entry["model_key"] == "proposed" or grid is not None:
            _compare(entry["allocations"], grid, f"{key}.{kind}.allocation_grid")
    _compare(best["config"], latest["config"], f"{key}.best_vs_latest.config")
    side_config = dict(sidecars["config"])
    side_fp = side_config.pop("config_fingerprint", None)
    if side_fp is not None:
        _compare(best["config_fingerprint"], side_fp, f"{key}.sidecar.config_fingerprint")
    _compare(best["config"], side_config, f"{key}.sidecar.config")
    status = sidecars["status"]
    _compare("completed", status.get("state"), f"{key}.status.state")
    _compare(50000, status.get("step"), f"{key}.status.step")
    _compare(50000, latest["step"], f"{key}.latest.step")
    if "total_steps" in status:
        _compare(50000, status["total_steps"], f"{key}.status.total_steps")
    if key == "lite1024":
        _compare(49500, best["step"], "lite1024.best.step vs user-reported log (stop on discrepancy)")
    return dict(checkpoint_step=best["step"], completion_step=latest["step"],
                training_status=status, config=best["config"],
                checkpoint_allocation_grid=best.get("allocation_grid"),
                config_fingerprint=best["config_fingerprint"], sidecars=sidecars,
                provenance_limit="Saved configuration/completion verified; not a reconstruction of historical execution source or all initialization events.")


def inspect_plan(plan):
    """No torch import/deserialization: paths + saved JSON only, all errors shown."""
    errors, records = [], {}
    for entry in plan:
        key, path = entry["method_key"], Path(entry["path"])
        print(f"[{key}] allocation={entry['allocations'][0]}\n  {path}")
        for item in (path, path.with_name("latest.pth")):
            if not item.is_file():
                errors.append(f"Missing checkpoint: {item}")
        try:
            sidecars = load_sidecars(entry)
            validate_training_config(entry, sidecars["config"])
            _compare("completed", sidecars["status"].get("state"), f"{key}.status.state")
            _compare(50000, sidecars["status"].get("step"), f"{key}.status.step")
            records[key] = sidecars
            print(f"  config={sidecars['paths']['config']}; status={sidecars['paths']['status']}")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if len({Path(entry['path']).resolve() for entry in plan}) != 4:
        errors.append("The four methods must have four distinct checkpoint paths")
    if errors:
        raise ValueError("\n".join(errors))
    return records


def validate_output_path(path, root=OUTPUT_ROOT):
    path, root = Path(path).absolute(), Path(root).absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Output must be a new subdirectory of {root}: {path}")
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Output symlink/junction refused: {part}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Resolved output escapes fixed root: {path}")
    if path.exists():
        raise FileExistsError(f"Existing output will not be overwritten: {path}")


def check_proposal_model(model, cfg, width):
    _compare([(0, 256)], cfg.GAMMA_GRID, "proposal.runtime.supported_grid")
    _compare((64, 256), (cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED), "proposal.runtime.max_dimensions")
    _compare((1,), tuple(model.running_pwr.shape), "proposal.running_pwr.shape")
    widths = {}
    for branch in ("fdma_backbone", "aircomp_backbone"):
        layers = getattr(model.encoder, branch).layers
        widths[branch] = [layer.linear1.out_features for layer in layers]
        _compare([width] * 4, widths[branch], f"proposal.{branch}.FFN")
        _compare([width] * 4, [layer.linear2.in_features for layer in layers], f"proposal.{branch}.linear2")
    widths["bs_cross_subband"] = [layer.linear1.out_features for layer in model.cross_subband_attn.layers]
    widths["bs_refinement"] = [layer.ffn[0].out_features for layer in model.layers]
    _compare([2048], widths["bs_cross_subband"], "proposal.BS.cross_subband")
    _compare([512] * 5, widths["bs_refinement"], "proposal.BS.refinement")
    return widths


def build_lite_method(ev, scenario, path, checkpoint, device):
    from models.adaptive_hybrid_lite import AdaptiveHybridPrecoderLite
    entry = next(item for item in checkpoint_plan() if item["method_key"] == "lite1024")
    validate_training_config(entry, checkpoint["config"])
    ev.validate_checkpoint_scenario(checkpoint, Path(path), scenario)
    cfg = ev.proposal_runtime_config(scenario, checkpoint)
    model = AdaptiveHybridPrecoderLite(cfg, ue_ffn_dim=1024).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()
    check_proposal_model(model, cfg, 1024)
    return ev.LoadedMethod(model_key="proposed", method_key="lite1024", display_name=entry["label"],
                           model=model, checkpoint_path=str(path), checkpoint_step=int(checkpoint["step"]),
                           checkpoint_kind="best", allocations=((0, 256),), config=cfg)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_buffer_snapshot(model):
    import torch
    buffers = {}
    for name, value in model.named_buffers(remove_duplicate=False):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        buffers[name] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())
    return dict(buffers=buffers, training={name: module.training for name, module in model.named_modules()})


def validate_frozen(before, after):
    if any(before["training"].values()) or any(after["training"].values()):
        raise ValueError("Evaluation requires every module to remain in eval mode")
    _compare(before, after, "evaluation.model_buffers_and_mode")


def _statistics(values):
    if len(values) < 2 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Statistics require at least two finite channel samples")
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    if not math.isfinite(mean) or not math.isfinite(se):
        raise ValueError("Non-finite derived statistics")
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def verify_results(summary, samples, plan, count):
    if count < 2:
        raise ValueError("Need at least two complete multi-user channel samples")
    by_key = {entry["method_key"]: entry for entry in plan}
    if set(by_key) != {"original", "lite1024", "swin", "csinet"} or len(plan) != 4:
        raise ValueError("Expected exactly the four fixed configurations")
    expected = {(key, float(fb), float(DL_SNR)) for key in by_key for fb in FB_SNRS}

    common = ("method_key", "model_key", "scenario_id", "k_users", "subcarriers", "feedback_budget",
              "allocation_df", "allocation_da", "feedback_snr_db", "downlink_snr_db")

    def require(row, fields, label):
        if not isinstance(row, dict):
            raise ValueError(f"{label}: expected a record")
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"{label}: missing fields {missing}")

    def condition(row):
        key = (row["method_key"], float(row["feedback_snr_db"]), float(row["downlink_snr_db"]))
        if key not in expected:
            raise ValueError(f"Unexpected condition: {key}")
        entry = by_key[key[0]]
        for field, value in (("scenario_id", SCENARIO_ID), ("k_users", 4), ("subcarriers", 256),
                             ("feedback_budget", 256), ("model_key", entry["model_key"])):
            _compare(value, row.get(field), f"{key}.{field}")
        _compare(entry["allocations"][0], (row["allocation_df"], row["allocation_da"]), f"{key}.allocation")
        if "seed" in row:
            _compare(SEED, row["seed"], f"{key}.seed")
        return key

    summaries, indexed = {}, {key: {} for key in expected}
    for row in summary:
        require(row, common + ("num_samples", "antennas", "seed") + tuple(
            metric + "_" + suffix for metric in ("sum_rate", "nmse_db", "precoder_power")
            for suffix in ("mean", "se", "ci95_low", "ci95_high")), "summary")
        key = condition(row)
        if key in summaries:
            raise ValueError(f"Duplicate summary condition: {key}")
        for field, value in (("num_samples", count), ("antennas", 32), ("seed", SEED)):
            _compare(value, row.get(field), f"{key}.{field}")
        summaries[key] = row
    if set(summaries) != expected:
        raise ValueError(f"Missing summary conditions: {expected - set(summaries)}")
    for row in samples:
        require(row, common + ("sample_index", "sum_rate", "nmse_db", "precoder_power"), "sample")
        key, index = condition(row), row["sample_index"]
        if type(index) is not int or not 0 <= index < count or index in indexed[key]:
            raise ValueError(f"Duplicate/invalid sample key: {key}, sample_index={index}")
        for metric in ("sum_rate", "nmse_db", "precoder_power"):
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"Non-finite {metric}: {key}, sample_index={index}")
        indexed[key][index] = row
    for key, rows in indexed.items():
        if set(rows) != set(range(count)):
            raise ValueError(f"Missing samples: {key}, missing={sorted(set(range(count)) - set(rows))[:10]}")
        for metric in ("sum_rate", "nmse_db", "precoder_power"):
            stats = _statistics([float(rows[i][metric]) for i in range(count)])
            for suffix, value in zip(("mean", "se", "ci95_low", "ci95_high"), stats):
                actual = float(summaries[key][metric + "_" + suffix])
                # Existing evaluator reduces in float32. Keep its values; this
                # check permits only the documented reduction-rounding scale.
                if not math.isfinite(actual) or not math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"Summary/raw mismatch {key}.{metric}_{suffix}: summary={actual}, raw={value}")
    return indexed


def paired_rows(summary, samples, plan, count):
    indexed = verify_results(summary, samples, plan, count)
    result = []
    for fb in FB_SNRS:
        for reference in ("original", "swin"):
            differences = [float(indexed["lite1024", fb, DL_SNR][i]["sum_rate"]) - float(indexed[reference, fb, DL_SNR][i]["sum_rate"]) for i in range(count)]
            mean, se, low, high = _statistics(differences)
            result.append(dict(comparison=f"lite1024-{reference}", method_key="lite1024", reference_method_key=reference,
                               scenario_id=SCENARIO_ID, feedback_snr_db=fb, downlink_snr_db=DL_SNR,
                               num_samples=count, seed=SEED, delta_mean=mean, delta_se=se,
                               delta_ci95_low=low, delta_ci95_high=high))
    return result


def _runtime_record(torch):
    versions = {}
    for name in ("torch", "numpy", "sionna"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    sources = ("evaluate_proposal_lite.py", "baseline_evaluate.py", "models/adaptive_hybrid.py",
               "models/adaptive_hybrid_lite.py", "models/swin_precoder.py", "models/csinet_plus_precoder.py",
               "models/baseline_common.py",
               "configs/train_config.py", "configs/lite_train_config.py", "configs/baseline_config.py",
               "utils/channel_generator.py", "utils/layers.py", "utils/gpu_setup.py")
    return dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                tracked_changes=subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines(),
                source_sha256={name: sha256(ROOT / name) for name in sources}, python=platform.python_version(), software=versions,
                backend=dict(cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(),
                             float32_matmul_precision=torch.get_float32_matmul_precision(),
                             matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                             cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                             cudnn_benchmark=torch.backends.cudnn.benchmark,
                             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                             flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                             mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                             math_sdp=torch.backends.cuda.math_sdp_enabled()))


def run_evaluation(plan, sidecars, *, preflight):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; real checkpoint loading/evaluation is server-only")
    import baseline_evaluate as ev
    device = torch.device("cuda")
    scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
    scenario.validate()
    role = "preflight_not_formal_test" if preflight else "independent_held_out_test"
    batches, count = (1, 8) if preflight else (200, 1600)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = OUTPUT_ROOT / (("preflight_" if preflight else "formal_") + stamp)
    validate_output_path(output)
    args = ev.build_parser().parse_args([
        "--scenario", "4,256,256", "--models", "proposed,swin,csinet", "--allocations", "0,256",
        "--checkpoint", "best", "--no-include-proposed-best", "--baseline-objective", "task",
        "--feedback-snr", ",".join(map(str, FB_SNRS)), "--downlink-snr", "25",
        "--batch-size", "8", "--num-batches", str(batches), "--seed", str(SEED),
        "--extra-metrics", "basic", "--output-dir", str(output), "--tag", TAG,
        "--proposal-root", str(ROOT / "runs_as_screen/proposal_specialists"),
        "--baseline-root", str(ROOT / "runs_as_screen/baselines"),
    ])
    paths = {Path(item) for value in sidecars.values() for item in value["all_paths"]}
    paths.update(Path(entry["path"]) for entry in plan)
    paths.update(Path(entry["path"]).with_name("latest.pth") for entry in plan)
    hashes_before = {str(path): sha256(path) for path in sorted(paths)}
    methods, records, parameters = [], {}, []
    ev.set_seed(SEED)
    for entry in plan:
        key, path = entry["method_key"], Path(entry["path"])
        best = ev.load_checkpoint_cpu(path)
        latest = ev.load_checkpoint_cpu(path.with_name("latest.pth"))
        record = verify_checkpoint(entry, best, latest, sidecars[key])
        del latest
        if key == "lite1024":
            method = build_lite_method(ev, scenario, path, best, device)
        elif key == "original":
            method = ev.build_proposed_method(scenario, path, "best", "0,256", device)
        else:
            method = ev.build_baseline_method(key, scenario, path, "best", device, args.rzf_reg)
        del best
        method.method_key, method.display_name = key, entry["label"]
        _compare(entry["allocations"], method.allocations, f"{key}.loaded_allocations")
        method.model.eval()
        record.update(model_class=type(method.model).__name__, checkpoint_path=str(path),
                      checkpoint_sha256=hashes_before[str(path)], latest_path=str(path.with_name("latest.pth")),
                      latest_sha256=hashes_before[str(path.with_name("latest.pth"))], strict_loading=True,
                      method_key=key, model_key=method.model_key, allocation=list(entry["allocations"][0]))
        full = sum(p.numel() for p in method.model.parameters())
        parameter_row = dict(method_key=key, full_parameters=full, encoder_plus_condition_parameters="",
                             counting_note="Parameter counts only; not MACs, latency or speedup")
        if entry["model_key"] == "proposed":
            # Reporting metadata only: inherited zf_solver already hard-codes
            # 1e-3. evaluate_scenario does not use this field for proposed
            # forward dispatch; do not change its model/config/solver.
            method.rzf_lambda = 1e-3
            record["rzf_regularization"] = method.rzf_lambda
            width = 1024 if key == "lite1024" else 2048
            record["structure_widths"] = check_proposal_model(method.model, method.config, width)
            record["cfg_idx"] = 0
            record["max_dimensions"] = [64, 256]
            parameter_row["encoder_plus_condition_parameters"] = sum(p.numel() for p in method.model.encoder.parameters()) + sum(p.numel() for p in method.model.cond_embed.parameters())
            parameter_row["counting_note"] += "; shared condition counted with UE, also used by BS"
            expected_counts = (16006619, 6674594) if key == "lite1024" else (20209115, 10877090)
            _compare(expected_counts, (full, parameter_row["encoder_plus_condition_parameters"]), f"{key}.parameter_counts")
        else:
            record["structure_widths"] = {name: getattr(method.config, name) for name in ("SWIN_EMBED_DIM", "SWIN_MLP_RATIO", "CSINET_ENC_CHANNELS", "CSINET_DEC_CHANNELS")}
        records[key] = record
        parameters.append(parameter_row)
        methods.append(method)
        print(f"[Loaded] {key}: {record['model_class']}, best step={method.checkpoint_step}, completion step={record['completion_step']}, strict=True")
    buffers_before = {method.method_key: model_buffer_snapshot(method.model) for method in methods}
    for value in buffers_before.values():
        validate_frozen(value, value)
    _compare(hashes_before, {str(path): sha256(path) for path in sorted(paths)}, "checkpoint/sidecar changed during loading")
    manifest = dict(preflight=preflight, dataset_role=role, created_utc=stamp, status="pending",
                    scenario=dict(k_users=4, antennas=32, subcarriers=256, feedback_budget=256, num_subbands=4),
                    feedback_snr_db=list(FB_SNRS), downlink_snr_db=DL_SNR, seed=SEED,
                    batch_size=BATCH_SIZE, num_batches=batches, num_samples_per_condition=count,
                    methods=records, runtime=_runtime_record(torch),
                    input_sha256_before=hashes_before, model_buffers_before=buffers_before,
                    sampling="One channel batch shared by every model and SNR; one sample is a complete multi-user realization.",
                    channel_seed="seed + stable_int(scenario_id) + batch_index",
                    feedback_seed="seed + 100000000 + batch_index*100000 + stable_int(f'{feedback_snr:.6f}')",
                    noise="Unchanged relative received-SNR convention. Common RNG seeds imply common base noise only with compatible random-call shape/order, NOT equal absolute AS/FDMA noise power.",
                    statistics="SE uses sample standard deviation (ddof=1); normal-approximation 95% CI = mean +/- 1.96 SE; paired differences are formed per sample_index before aggregation.",
                    nmse="Mean of per-sample dB raw antenna-frequency representation errors versus H_dl; secondary diagnostic, not task ranking.",
                    power="Per-sample total precoder power across all subcarriers; nominal 256 for per-subcarrier power 1.",
                    selection="Fixed best checkpoints; no test-based selection, oracle allocation, point deletion, seed search or historical CSV merging.")
    validate_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        ev.set_seed(SEED)
        points = [(float(fb), float(DL_SNR)) for fb in FB_SNRS]
        try:
            summary, samples = ev.evaluate_scenario(scenario, methods, points, args, device)
        finally:
            # Check even when evaluation fails; never hide a side effect behind
            # a later schema/statistics error or a failed forward pass.
            manifest["input_sha256_after"] = {str(path): sha256(path) for path in sorted(paths)}
            manifest["model_buffers_after"] = {method.method_key: model_buffer_snapshot(method.model) for method in methods}
            _compare(hashes_before, manifest["input_sha256_after"], "checkpoint/sidecar modified during evaluation")
            for key in buffers_before:
                validate_frozen(buffers_before[key], manifest["model_buffers_after"][key])
            manifest["checkpoint_and_buffers_unchanged"] = True
        for row in summary + samples:
            row.update(dataset_role=role, preflight=preflight, seed=SEED)
        deltas = paired_rows(summary, samples, plan, count)
        for row in deltas + parameters:
            row.update(dataset_role=role, preflight=preflight)
        ev.save_outputs(summary, samples, [scenario], [entry["method_key"] for entry in plan], points, args)
        plot_fields = ("method_key", "method_label", "allocation_df", "allocation_da", "feedback_snr_db", "downlink_snr_db", "sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high", "num_samples", "seed", "dataset_role", "preflight")
        ev.write_csv(output / f"plot_{TAG}.csv", [{name: row[name] for name in plot_fields} for row in summary])
        ev.write_csv(output / f"paired_deltas_{TAG}.csv", deltas)
        ev.write_csv(output / f"parameters_{TAG}.csv", parameters)
        manifest["status"] = "completed"
        manifest["summary_rows"] = len(summary)
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Failed runs are kept as evidence, never silently overwritten on retry.
        with (output / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str, allow_nan=False)
    print(f"[Completed] {role}: 24 conditions x {count} samples. Output: {output}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inspect fixed paths and sidecar JSON only; no weights or writes")
    mode.add_argument("--preflight", action="store_true", help="CUDA: four real checkpoints, six SNRs, 8 samples/condition")
    mode.add_argument("--evaluate", action="store_true", help="CUDA: independent test, 1600 samples/condition, fixed seed")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (args.dry_run or args.preflight or args.evaluate):
        parser.print_help()
        return 0
    plan = checkpoint_plan()
    print(f"Protocol: {SCENARIO_ID}, Nt32/Nsb4; FB={FB_SNRS}, DL=25, seed={SEED}, B=8")
    print(f"Output root: {OUTPUT_ROOT}; preflight=8 samples, formal=1600 samples per condition")
    try:
        sidecars = inspect_plan(plan)
        if args.dry_run:
            print("Dry-run passed: files/sidecars only. Checkpoint tensors, internal steps and real loading NOT verified.")
            return 0
        run_evaluation(plan, sidecars, preflight=args.preflight)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
