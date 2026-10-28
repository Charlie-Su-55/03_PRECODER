#!/usr/bin/env python3
"""Independent Legacy+FDMA completion and new-test replication; old files are read-only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import traceback

import evaluate_grouped_codec as old
from as_receiver_batch_support import safe_inside
from as_spatial_archive import _Verifier, _verify_evaluations
from configs.grouped_codec_fdma_train_config import SEEDS, fdma_config
from evaluate_proposal_as_direct import check_tensor_state, load_payload, read_json, sample_statistics, write_manifest
from evaluate_proposal_lite import _compare, model_buffer_snapshot, validate_frozen, validate_output_path
from preflight_proposal_as_direct import sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs_as_screen/evaluation/k4_grouped_codec_fdma_v1"
TAG = "k4_grouped_codec_fdma_v1"
OLD_BATCH_RELATIVE = Path("runs_as_screen/grouped_codec_v1/batches/batch_b314d7520c844f259ffa5db18ab0b149/batch_summary.json")
DATASETS = {"old_completion": 20261210, "new_replication": 20261220}
METHODS = old.METHODS + ("legacy_fdma",)
GAMMAS, METRICS = old.GAMMAS, old.METRICS
CONTRASTS = dict(deepcopy(old.CONTRASTS), **{
    "C_minus_F": {"legacy_group2": 1, "legacy_fdma": -1},
    "A_minus_F": {"legacy_all4": 1, "legacy_fdma": -1},
    "F_minus_E": {"legacy_fdma": 1, "ordered_fdma": -1},
    "C_minus_E": {"legacy_group2": 1, "ordered_fdma": -1},
    "interaction_C_minus_F_minus_D_plus_E": {
        "legacy_group2": 1, "legacy_fdma": -1, "ordered_group2": -1, "ordered_fdma": 1}})
OUTPUT_ARTIFACTS = tuple(f"{name}_{TAG}.{suffix}" for name, suffix in (
    ("summary", "csv"), ("samples", "csv.gz"), ("results", "json"), ("metadata", "json"),
    ("paired_deltas", "csv"), ("paired_samples", "csv.gz"), ("parameters", "csv")))
SOURCE_FILES = tuple(dict.fromkeys(old.SOURCE_FILES + (
    "evaluate_grouped_codec_fdma.py", "configs/grouped_codec_fdma_train_config.py",
    "models/grouped_codec_fdma.py", "train_grouped_codec_fdma.py", "as_spatial_archive.py")))
INTEGER_FIELDS = ("sample_index", "batch_index", "occupied_feedback_RE", "active_symbols_per_ue",
                  "active_users_per_RE", "seed", "training_seed", "num_samples", "num_batches", "batch_size")


def make_test_data(dataset):
    if dataset not in DATASETS: raise ValueError("Unknown independent dataset")
    return old.channel_data_api().ReplayableChannelData(base_seed=DATASETS[dataset],
        k_users=4, nt=32, nsc=256, batch_size=8, validation_batch_size=8,
        carrier_freq=3.5e9, speed=1.0, cdl_model="C", device="cuda:0")


def protocol(training_seed, dataset):
    old.shared.validate_seed(training_seed)
    if dataset not in DATASETS: raise ValueError("Explicit old_completion or new_replication required")
    value = deepcopy(old.protocol(training_seed))
    value.update(dataset=dataset, seed=DATASETS[dataset], channel_data_protocol=make_test_data(dataset).protocol,
        methods=list(METHODS), computed_methods=["legacy_fdma"] if dataset == "old_completion" else list(METHODS),
        dataset_role="paired_old_test_completion" if dataset == "old_completion" else "independent_replication",
        old_batch_summary=str(ROOT / OLD_BATCH_RELATIVE),
        checkpoint_selection="Validation only; the two test sets never select checkpoints",
        primary_comparison="C_minus_F on new_replication; old D_minus_E remains reported unchanged",
        training_seeds_pooled=False, datasets_pooled=False)
    value["noise_address"]["seed"] = DATASETS[dataset]
    value["active_layouts"]["legacy_fdma"] = old.physical_api().LayoutSpec("fdma").to_dict()
    return value


def checkpoint_plan(training_seed, dataset, old_evidence=None):
    p = protocol(training_seed, dataset)
    entries = deepcopy(old.checkpoint_plan(training_seed))
    for item in entries:
        item["forward_required"] = dataset == "new_replication"
        item["source_batch"] = str(ROOT / OLD_BATCH_RELATIVE)
        if old_evidence is not None:
            recorded = old_evidence["training"][str(training_seed)][item["arm"]]
            _compare(item["path"], recorded["best_path"], "Pinned old best path")
            _compare(item["latest_path"], recorded["latest_path"], "Pinned old latest path")
            item.update(expected_best_sha256=recorded["best_sha256"], expected_latest_sha256=recorded["latest_sha256"])
    cfg = fdma_config(training_seed)
    entries.append(dict(method_key="legacy_fdma", arm="legacy_fdma", training_seed=training_seed,
        path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH, config_path=cfg.CONFIG_PATH, status_path=cfg.STATUS_PATH,
        active_layout="fdma", layout_spec=p["active_layouts"]["legacy_fdma"], forward_required=True,
        source_batch=None))
    if len({item["path"] for item in entries}) != 6: raise ValueError("Six distinct selected checkpoints required")
    return entries


def csv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing/duplicate CSV header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row width")
    return rows


def typed_rows(rows, *, raw):
    output = []
    for original in rows:
        row = dict(original)
        for name in INTEGER_FIELDS:
            if name in row: row[name] = int(row[name])
        for name in ("gamma_db", "downlink_snr_db", "noise_variance_N0", "rzf_regularization"):
            if name in row: row[name] = float(row[name])
        for metric in METRICS:
            names = (metric,) if raw else tuple(metric + "_" + suffix for suffix in ("mean", "se", "ci95_low", "ci95_high"))
            for name in names:
                if name not in row: raise ValueError("Missing physical/result metric: " + name)
                row[name] = float(row[name])
        if "preflight" in row:
            if str(row["preflight"]).lower() not in ("false", "true", "0", "1"):
                raise ValueError("Invalid preflight marker")
            row["preflight"] = str(row["preflight"]).lower() in ("true", "1")
        output.append(row)
    return output


def verify_stage_evidence(stages, verifier):
    expected = {"environment", "new_cuda_preflight", "organize_results"}
    expected |= {f"train_{arm}_{seed}" for seed in SEEDS for arm in old.METHODS}
    expected |= {f"evaluate_{seed}" for seed in SEEDS}
    _compare(sorted(expected), sorted(stages), "Fixed old stage set")
    for name, stage in stages.items():
        _compare("completed", stage.get("status"), "Old stage " + name)
        verifier.hashes(stage.get("evidence_sha256"))
        result = stage.get("result", {})
        if "report_path" in result:
            report = verifier.json(result["report_path"])
            _compare(result.get("report"), report, "Saved old stage report")
            _compare(True, report.get("passed"), "Old stage passed")


def verify_old_batch(path=None, *, root=ROOT):
    """Hash and parse only the fixed old batch. Never discover another run or load weights."""
    root = Path(root).absolute()
    expected = root / OLD_BATCH_RELATIVE
    selected = expected if path is None else Path(path).absolute()
    if selected != expected: raise ValueError("Only the explicitly authorized fixed old batch is accepted")
    verifier = _Verifier(root)
    summary = verifier.json(selected)
    _compare("completed", summary.get("status"), "Fixed old batch completion")
    verify_stage_evidence(summary.get("stages", {}), verifier)
    source = summary.get("source", {}).get("sha256")
    if not isinstance(source, dict) or not source: raise ValueError("Old source provenance missing")
    for relative, digest in source.items(): verifier.file(safe_inside(root / relative, root), digest)
    preflight = summary.get("preflight", {})
    _compare(True, preflight.get("passed"), "Old preflight completion")
    _compare(preflight, summary["stages"]["new_cuda_preflight"]["result"]["report"], "Old preflight record")
    training, evaluations = {}, {}
    from train_grouped_codec import fingerprint
    for seed in SEEDS:
        training[str(seed)] = {}
        for arm in old.METHODS:
            record = summary.get("training", {}).get(str(seed), {}).get(arm)
            if not isinstance(record, dict): raise ValueError("Missing old training record")
            _compare(record, summary["stages"][f"train_{arm}_{seed}"]["result"]["report"], "Old training stage/summary")
            for key, value in dict(arm=arm, training_seed=seed, completed_updates=20000).items():
                _compare(value, record.get(key), "Old training " + key)
            step = record.get("best_step")
            if type(step) is not int or not 0 < step <= 20000 or step % 500: raise ValueError("Invalid old best step")
            cfg = old.grouped_codec_config(arm, seed, output_root=root / "runs_as_screen/grouped_codec_v1")
            for key, expected_path in (("best_path", cfg.BEST_PATH), ("latest_path", cfg.SAVE_PATH)):
                _compare(str(expected_path), record.get(key), "Fixed old checkpoint path")
                verifier.file(expected_path, record[key.replace("path", "sha256")])
            config, status = verifier.json(cfg.CONFIG_PATH), verifier.json(cfg.STATUS_PATH)
            _compare(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg)), config, "Full old saved configuration")
            for key, value in dict(state="completed", step=20000, experiment=cfg.EXP_NAME,
                                    best_metric_name="mean_all", best_metric=record["best_metric"]).items():
                _compare(value, status.get(key), "Old status " + key)
            selected_meta = record["verified_selection"]
            _compare(cfg.to_dict(), selected_meta["config"], "Old verified source config")
            _compare(status, selected_meta["training_status"], "Old verified status")
            metrics = verifier.file(cfg.METRICS_PATH)
            training[str(seed)][arm] = dict(record, metrics_path=str(metrics), metrics_sha256=verifier.checked[str(metrics)]["sha256"])
        directory = summary["stages"][f"evaluate_{seed}"]["result"]["output_directory"]
        directory = safe_inside(directory, root / "runs_as_screen/evaluation/k4_grouped_codec_v1" / expected.parent.name / f"seed_{seed}")
        pointer = verifier.json(expected.parent / f"evaluation_seed{seed}.json")
        _compare(str(directory), pointer.get("directory"), "Exact old evaluation pointer")
        evaluations[str(seed)] = str(directory)
    _, _, _, manifests = _verify_evaluations(evaluations, verifier,
        {str(seed): old.protocol(seed) for seed in SEEDS}, old.OUTPUT_ARTIFACTS)
    from grouped_codec_results import interpret, verify_shared_batch_inputs
    reference = None
    evidence = {}
    for seed in SEEDS:
        key = str(seed); directory = Path(evaluations[key]); manifest = manifests[key]
        reference = verify_shared_batch_inputs(manifest, reference)
        _compare(True, manifest.get("shared_channel_noise_inputs_unchanged"), "Old shared inputs frozen")
        _compare(sorted(old.METHODS), sorted(manifest["methods"]), "Old five method identities")
        for arm in old.METHODS:
            recorded, loaded = training[key][arm], manifest["methods"][arm]
            for field, expected_value in dict(best_path=recorded["best_path"], latest_path=recorded["latest_path"],
                    best_sha256=recorded["best_sha256"], latest_sha256=recorded["latest_sha256"],
                    selected_best_step=recorded["best_step"], latest_step=20000,
                    config=recorded["verified_selection"]["config"]).items():
                _compare(expected_value, loaded.get(field), "Old evaluated checkpoint identity " + field)
        raw_path = directory / f"samples_{old.TAG}.csv.gz"
        summary_path = directory / f"summary_{old.TAG}.csv"
        samples = typed_rows(csv_rows(raw_path), raw=True)
        summaries = typed_rows(csv_rows(summary_path), raw=False)
        old.paired_rows(summaries, samples, 1600, DATASETS["old_completion"], seed)
        conclusion = interpret(csv_rows(summary_path), csv_rows(directory / f"paired_deltas_{old.TAG}.csv"), seed)
        _compare(summary["results"]["per_seed"][key]["engineering_criterion_D_minus_E"],
                 conclusion["engineering_criterion_D_minus_E"], "Preserved old D-E conclusion")
        evidence[key] = dict(directory=str(directory), manifest=manifest, manifest_path=str(directory / "manifest.json"),
            raw_path=str(raw_path), raw_sha256=manifest["output_sha256"][raw_path.name],
            summary_path=str(summary_path), parameters_path=str(directory / f"parameters_{old.TAG}.csv"))
    verifier.unchanged()
    return dict(status="verified", summary_path=str(selected), summary_sha256=verifier.checked[str(selected)]["sha256"],
        summary=summary, evaluations=evidence, training=training,
        frozen_files={name: record["sha256"] for name, record in verifier.checked.items()},
        weights_deserialized=False, old_batch_id=expected.parent.name)


def verify_training_metadata(seed, best, latest, sidecars):
    from train_grouped_codec_fdma import fingerprint, validate_resume_payload, validate_best_metadata
    cfg = fdma_config(seed)
    validate_resume_payload(latest, cfg, allow_completed=True)
    validate_best_metadata(best, latest, cfg)
    _compare(20000, latest.get("step"), "F completed budget")
    _compare(dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg)), sidecars["config"], "Full F sidecar config")
    for name, value in dict(state="completed", step=20000, experiment=cfg.EXP_NAME,
                             best_metric_name="mean_all", best_metric=best["best_metric"]).items():
        _compare(value, sidecars["status"].get(name), "F status." + name)
    return dict(config=best["config"], config_fingerprint=best["config_fingerprint"],
        selected_best_step=best["step"], latest_step=latest["step"], completed_updates=20000,
        best_metric=best["best_metric"], training_status=sidecars["status"],
        random_initialization=True, historical_weights_loaded=False)


def build_fdma_model(torch, seed, payload, device):
    from models.grouped_codec_fdma import GroupedCodecFDMAPrecoder
    from train_grouped_codec_fdma import validate_model_identity, validate_model_state
    cfg = fdma_config(seed)
    _compare(cfg.to_dict(), payload["config"], "F full checkpoint configuration")
    model = GroupedCodecFDMAPrecoder(cfg)
    validate_model_identity(cfg, model); validate_model_state(torch, model, payload["state"])
    model.load_state_dict(payload["state"], strict=True)
    check_tensor_state(torch, model.state_dict(), payload["state"], equal=True)
    return model.to(device).eval(), cfg


def summarize_samples(samples, count, seed, training_seed, dataset):
    if dataset not in DATASETS or seed != DATASETS[dataset] or count < 8 or count % 8:
        raise ValueError("Explicit dataset and complete eight-channel batches required")
    indexed = {(arm, gamma): {} for arm in METHODS for gamma in GAMMAS}
    for row in samples:
        key, index = (row.get("method_key"), row.get("gamma_db")), row.get("sample_index")
        if key not in indexed or type(index) is not int or not 0 <= index < count or index in indexed[key]:
            raise ValueError("Invalid/duplicate complete dataset-method-gamma-sample key")
        if row.get("dataset") != dataset or any(name in row for name in ("feedback_snr_db", "allocation_df", "allocation_da")):
            raise ValueError("Dataset mixing or obsolete physical protocol")
        for name, value in old.sample_identity(key[0], key[1], index, seed, training_seed).items():
            _compare(value, row.get(name), "Sample identity." + name)
        if any(not math.isfinite(float(row[name])) for name in METRICS): raise ValueError("Nonfinite metric")
        if any(abs(row[f"energy_ue{k}"] - 64) > 1e-4 for k in range(4)): raise ValueError("UE energy changed")
        if not 0 <= row["precoder_power_max_error"] <= 1e-5 or abs(row["precoder_power"] - 256) > 256e-5:
            raise ValueError("Precoder power constraint changed")
        if not 0 <= row["reliability_mean_active"] <= 1 or row["empirical_noise_power"] < 0:
            raise ValueError("Invalid physical diagnostics")
        ridge = row["noise_variance_N0"] / (64 / row["active_symbols_per_ue"]) + 1e-6
        if not math.isclose(row["receiver_regularization"], ridge, rel_tol=1e-6, abs_tol=1e-10): raise ValueError("Wrong receiver ridge")
        if any(row[name] < 0 for name in METRICS if "_mse_" in name or "_linear_" in name): raise ValueError("Negative squared diagnostic")
        indexed[key][index] = row
    summaries = []
    for key, records in indexed.items():
        if set(records) != set(range(count)): raise ValueError("Incomplete paired sample IDs")
        row = old.sample_identity(*key, 0, seed, training_seed)
        row.pop("sample_index"); row.pop("batch_index")
        row.update(dataset=dataset, num_samples=count, num_batches=count // 8, batch_size=8, rzf_regularization=1e-3)
        for metric in METRICS:
            row.update({metric + "_" + name: value for name, value in sample_statistics([records[i][metric] for i in range(count)]).items()})
        summaries.append(row)
    for gamma in GAMMAS:
        for i in range(count):
            reference = indexed[METHODS[0], gamma][i]["empirical_noise_power"]
            if any(not math.isclose(indexed[arm, gamma][i]["empirical_noise_power"], reference, rel_tol=1e-6, abs_tol=1e-10) for arm in METHODS[1:]):
                raise ValueError("Different physical noise in supposedly paired observations")
    return summaries, indexed


def paired_rows(summary, samples, count, seed, training_seed, dataset):
    actual, indexed = summarize_samples(samples, count, seed, training_seed, dataset)
    supplied = {(row["method_key"], row["gamma_db"]): row for row in summary}
    if len(supplied) != 36 or len(summary) != 36: raise ValueError("Exactly36 summary conditions required")
    for row in actual:
        candidate = supplied.get((row["method_key"], row["gamma_db"]), {})
        for name, value in row.items():
            if isinstance(value, float):
                if not math.isfinite(float(candidate.get(name, math.nan))) or not math.isclose(value, float(candidate[name]), rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("Summary/raw mismatch " + name)
            else: _compare(value, candidate.get(name), "Summary/raw " + name)
    output, raw = [], []
    for name, coefficients in CONTRASTS.items():
        for estimand, gammas in [("per_gamma", (g,)) for g in GAMMAS] + [("low_gamma_mean", (0, 5, 10)), ("all_gamma_mean", GAMMAS)]:
            values = [statistics.fmean(sum(c * indexed[arm, gamma][i]["sum_rate"] for arm, c in coefficients.items()) for gamma in gammas) for i in range(count)]
            batches = [statistics.fmean(values[i:i+8]) for i in range(0, count, 8)]
            row = dict(contrast_id=name, coefficients=json.dumps(coefficients, sort_keys=True), estimand=estimand,
                gamma_points=",".join(map(str, gammas)), gamma_db=gammas[0] if len(gammas) == 1 else None,
                dataset=dataset, seed=seed, training_seed=training_seed, downlink_snr_db=25,
                num_samples=count, num_batches=count//8, batch_size=8, batch_ci_estimable=count>8,
                training_seeds_pooled=False, datasets_pooled=False,
                sample_ci_unit="complete_multi_user_channel", batch_ci_unit="sample_index//8",
                role=("primary_confirmatory" if dataset == "new_replication" else "paired_completion") if name == "C_minus_F" and estimand == "low_gamma_mean"
                     else "secondary25" if name == "C_minus_F" and gammas == (25,) else "historical_primary" if name == "D_minus_E" and estimand == "low_gamma_mean" else "diagnostic")
            row.update({"delta_"+k:v for k,v in sample_statistics(values).items()})
            row.update({"batch_delta_"+k:v for k,v in sample_statistics(batches).items()})
            output.append(row)
            raw.extend(dict(dataset=dataset, contrast_id=name, estimand=estimand, gamma_points=row["gamma_points"],
                sample_index=i, batch_index=i//8, delta=value, seed=seed, training_seed=training_seed) for i,value in enumerate(values))
    return output, raw


def artifact_hashes(output):
    return old.shared.artifact_hashes(output, expected_artifacts=OUTPUT_ARTIFACTS)


def runtime_record(torch=None):
    return old.shared.runtime_record(torch, source_files=SOURCE_FILES)


def match_old_batch_inputs(record, old_manifest):
    records = old_manifest.get("shared_batch_inputs", [])
    index = record.get("batch_index")
    if type(index) is not int or not 0 <= index < len(records) or record != records[index]:
        raise ValueError("Actual old-completion Hdl/Hul/standard-noise hashes do not match the fixed archived batch")


def extract_fdma_diagnostics(torch, diagnostics, noise, gamma, cfg):
    """Same active-only diagnostic math, driven by F's actual LayoutSpec, never a fake arm."""
    z, corrected, truth, reliability = [diagnostics[name] for name in ("z_hat", "z_corrected", "true_z", "reliability")]
    for value in (z, corrected, truth, reliability, diagnostics["energy"], diagnostics["noise_power"]):
        if not torch.is_tensor(value) or value.requires_grad or not torch.isfinite(value).all().item(): raise ValueError("Detached finite diagnostics required")
    if any(tuple(value.shape) != (8,4,256) for value in (z,corrected,truth,reliability)): raise ValueError("Canonical diagnostics shape mismatch")
    mask = diagnostics["active_mask"]
    expected = torch.tensor(cfg.layout_spec.canonical_to_physical, device=mask.device) >= 0
    if mask.dtype != torch.bool or not torch.equal(mask,expected): raise ValueError("F actual layout mask mismatch")
    active = mask[None].expand_as(z)
    if any(torch.count_nonzero(value[~active]).item() for value in (z,corrected,truth,reliability)): raise ValueError("Inactive F slots not zero")
    count = int(mask.sum().item()); denominator = truth.abs().square().sum((1,2)).clamp_min(1e-12)
    values = {}
    for prefix,value in (("recovered",z),("corrected",corrected)):
        squared=(value-truth).abs().square().masked_fill(~active,0).sum((1,2)); ratio=squared/denominator
        values.update({prefix+"_mse_active":squared/count,prefix+"_nmse_linear_active":ratio,
                       prefix+"_nmse_db_active":10*torch.log10(ratio.clamp_min(1e-12))})
    energy, measured = diagnostics["energy"], diagnostics["noise_power"]
    if tuple(energy.shape)!=(8,4) or tuple(measured.shape)!=(8,): raise ValueError("F physical diagnostic shape mismatch")
    if not torch.allclose(energy,truth.abs().square().sum(-1),rtol=0,atol=1e-4): raise ValueError("F energy accounting mismatch")
    N0=old.expected_n0(gamma); ridge=diagnostics["lambda_rx"]
    if not math.isclose(float(diagnostics["N0"]),N0,rel_tol=1e-6,abs_tol=1e-12): raise ValueError("F absoluteN0 changed")
    if not torch.is_tensor(ridge) or ridge.requires_grad or ridge.numel()!=1 or not math.isclose(float(ridge.item()),N0/(64/cfg.layout_spec.active_per_user)+1e-6,rel_tol=1e-6,abs_tol=1e-10): raise ValueError("F receiver ridge changed")
    if not torch.allclose(measured,(noise*math.sqrt(N0/2)).abs().square().mean((1,2)),rtol=1e-6,atol=1e-10): raise ValueError("F physical noise resampled/rescaled")
    values.update({f"energy_ue{k}":energy[:,k] for k in range(4)})
    values.update(empirical_noise_power=measured,receiver_regularization=ridge.expand(8),
                  reliability_mean_active=reliability.masked_fill(~active,0).sum((1,2))/count)
    return values


def run_evaluation(training_seed,dataset,*,output_dir=None):
    p=protocol(training_seed,dataset)
    output=old.shared.create_output(training_seed,output_dir=output_dir,root=OUTPUT_ROOT/dataset)
    manifest=dict(status="pending",protocol=p,output_directory=str(output),methods={},shared_batch_inputs=[],
        dataset=dataset, datasets_pooled=False,training_seeds_pooled=False,
        ci_scope="Fixed checkpoints; sample and batch evaluation uncertainty, not training-seed uncertainty")
    hashes,snapshots,models,samples={},{},{},[]; failure=None
    try:
        manifest["runtime"]=runtime_record()
        evidence=verify_old_batch()
        hashes.update(evidence["frozen_files"])
        manifest["old_batch"]=dict(summary_path=evidence["summary_path"],summary_sha256=evidence["summary_sha256"],
            evaluation_manifest=evidence["evaluations"][str(training_seed)]["manifest_path"],
            raw_path=evidence["evaluations"][str(training_seed)]["raw_path"],raw_sha256=evidence["evaluations"][str(training_seed)]["raw_sha256"])
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback forbidden")
        from train_grouped_codec_fdma import apply_backend, validate_model_identity, validate_model_state
        manifest["runtime_before_backend"]=runtime_record(torch); apply_backend(torch); manifest["runtime"]=runtime_record(torch)
        import baseline_evaluate as ev
        from preflight_joint_swin import tensor_digest
        device=torch.device("cuda:0"); ev.set_seed(p["seed"])
        parameters=[]
        previous=evidence["evaluations"][str(training_seed)]
        if dataset=="old_completion":
            samples=typed_rows(csv_rows(previous["raw_path"]),raw=True)
            for row in samples: row.update(dataset=dataset,source="verified_original_raw")
            parameters=csv_rows(previous["parameters_path"])
            manifest["reused_methods"]=deepcopy(previous["manifest"]["methods"])
            manifest["methods"]=deepcopy(previous["manifest"]["methods"])
            for record in manifest["methods"].values(): record["loaded_in_this_run"]=False
        for item in checkpoint_plan(training_seed,dataset,evidence):
            if not item["forward_required"]: continue
            arm=item["arm"]
            latest=load_payload(torch,item["latest_path"],hashes,expected=item.get("expected_latest_sha256"))
            best=load_payload(torch,item["path"],hashes,expected=item.get("expected_best_sha256"))
            sidecars={name:read_json(item[name+"_path"],hashes) for name in ("config","status")}
            if arm=="legacy_fdma":
                record=verify_training_metadata(training_seed,best,latest,sidecars)
                model,cfg=build_fdma_model(torch,training_seed,best,device)
                validate_model_state(torch,model,latest["state"])
            else:
                record=old.verify_training_metadata(arm,training_seed,best,latest,sidecars)
                model,cfg=old.build_model(torch,arm,training_seed,best,device)
                from train_grouped_codec import validate_model_state as old_state
                old_state(torch,model,latest["state"])
            check_tensor_state(torch,latest["state"],best["state"],equal=latest["step"]==best["step"])
            models[arm]=(model,cfg);snapshots[arm]=model_buffer_snapshot(model)
            validate_frozen(snapshots[arm],snapshots[arm])
            record.update(model_class=type(model).__name__,best_path=item["path"],latest_path=item["latest_path"],
                best_sha256=hashes[item["path"]],latest_sha256=hashes[item["latest_path"]],strict_loading=True,
                layout_spec=cfg.layout_spec.to_dict(),architecture_metadata=model.architecture_metadata,loaded_in_this_run=True)
            manifest["methods"][arm]=record
            parameters.append(dict(method_key=arm,training_seed=training_seed,
                full_parameters=sum(v.numel() for v in model.parameters()),ue_parameters=sum(v.numel() for v in model.ue_parameters()),
                bs_parameters=sum(v.numel() for v in model.bs_parameters()),note="Stored parameter counts, not MACs or latency"))
            del best,latest
        data=make_test_data(dataset).prepare();torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for batch_index in range(200):
                H_dl,H_ul=data.training(batch_index+1)
                noise=old.physical_api().standard_noise(p["seed"],"test",batch_index,8,device=str(device))
                originals=tuple(value.clone() for value in (H_dl,H_ul,noise))
                batch_record=dict(batch_index=batch_index,sample_index_start=8*batch_index,sample_index_stop=8*(batch_index+1),
                    **{name+"_sha256":tensor_digest({name:value}) for name,value in zip(("H_dl","H_ul","standard_noise"),originals)})
                if dataset=="old_completion": match_old_batch_inputs(batch_record,previous["manifest"])
                manifest["shared_batch_inputs"].append(batch_record)
                for gamma in GAMMAS:
                    for arm,(model,cfg) in models.items():
                        result=model.forward_with_diagnostics(H_dl,H_ul,float(gamma),standard_noise=noise)
                        W,Hhat,d=result["precoder"],result["channel_surrogate"],result["diagnostics"]
                        if not torch.isfinite(W).all().item() or not torch.isfinite(Hhat).all().item(): raise ValueError("Nonfinite model output")
                        values=(extract_fdma_diagnostics(torch,d,noise,gamma,cfg) if arm=="legacy_fdma" else old.extract_diagnostics(torch,d,noise,gamma,arm))
                        values.update(sum_rate=ev.per_sample_sum_rate(W,H_dl,10**(-25/10)),nmse_db=ev.per_sample_nmse_db(Hhat,H_dl),
                            precoder_power=ev.per_sample_precoder_power(W),precoder_power_max_error=(W.abs().square().sum((1,2))-1).abs().amax(1))
                        arrays={key:value.detach().cpu().tolist() for key,value in values.items()}
                        samples.extend(dict(old.sample_identity(arm,gamma,8*batch_index+i,p["seed"],training_seed),
                            dataset=dataset,source="new_forward",**{key:value[i] for key,value in arrays.items()}) for i in range(8))
                if not all(torch.equal(a,b) for a,b in zip(originals,(H_dl,H_ul,noise))): raise ValueError("Shared inputs modified")
        summary,_=summarize_samples(samples,1600,p["seed"],training_seed,dataset)
        paired,raw_pairs=paired_rows(summary,samples,1600,p["seed"],training_seed,dataset)
        for row in summary+samples+paired+raw_pairs: row.update(preflight=False,dataset_role=p["dataset_role"])
        for row in parameters: row["dataset"]=dataset
        ev.write_csv(output/f"summary_{TAG}.csv",summary);ev.write_gzip_csv(output/f"samples_{TAG}.csv.gz",samples)
        ev.write_csv(output/f"paired_deltas_{TAG}.csv",paired);ev.write_gzip_csv(output/f"paired_samples_{TAG}.csv.gz",raw_pairs)
        ev.write_csv(output/f"parameters_{TAG}.csv",parameters)
        for name,value in (("metadata",p),("results",dict(summary=summary,paired=paired))):
            with (output/f"{name}_{TAG}.json").open("x",encoding="utf-8") as stream: json.dump(value,stream,indent=2,allow_nan=False)
        manifest.update(status="completed",results_verified=True,summary_rows=36,paired_rows=96,
            shared_channel_noise_inputs_unchanged=True,samples_hashed_pairing=True,
            old_forward_calls=0 if dataset=="old_completion" else 5*200*6,
            frozen_old_rows_preserved=dataset=="old_completion",
            memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    except Exception as exc:
        failure=exc;manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        errors,after_hashes,after_buffers=[],{},{}
        for path,expected in hashes.items():
            try:
                after_hashes[path]=sha256_file(path)
                if after_hashes[path]!=expected: errors.append("Changed frozen input: "+path)
            except Exception as exc: errors.append(str(exc))
        for arm,(model,cfg) in models.items():
            try:
                after_buffers[arm]=model_buffer_snapshot(model);validate_frozen(snapshots[arm],after_buffers[arm])
                if arm=="legacy_fdma": validate_model_identity(cfg,model)
                else:
                    from train_grouped_codec import validate_model_identity as old_identity
                    old_identity(cfg,model)
            except Exception as exc: errors.append(arm+":"+str(exc))
        required=1 if dataset=="old_completion" else 6
        manifest.update(input_sha256_before=hashes,input_sha256_after=after_hashes,
            model_buffers_before=snapshots,model_buffers_after=after_buffers,
            freeze_checks=dict(passed=not errors and len(models)==required,errors=errors,loaded_methods=list(models)))
        if errors or (failure is None and len(models)!=required):
            failure=failure or RuntimeError("Frozen-input/model check failed");manifest.update(status="failed",error=str(failure))
        if failure is None:
            try: manifest["output_sha256"]=artifact_hashes(output)
            except Exception as exc: failure=exc;manifest.update(status="failed",error=str(exc))
        write_manifest(output,manifest)
    if failure is not None: raise RuntimeError(f"FDMA evaluation failed; preserved evidence {output/'manifest.json'}") from failure
    print(f"OUTPUT_DIRECTORY={output}");return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group();modes.add_argument("--dry-run",action="store_true");modes.add_argument("--evaluate",action="store_true")
    parser.add_argument("--seed",type=int,choices=SEEDS);parser.add_argument("--dataset",choices=tuple(DATASETS));parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    if not (args.dry_run or args.evaluate):parser.print_help();return 0
    if args.seed is None or args.dataset is None:parser.error("Explicit --seed and --dataset required")
    if args.dry_run:
        if args.output_dir is not None:validate_output_path(args.output_dir.absolute(),OUTPUT_ROOT/args.dataset)
        print(json.dumps(dict(protocol=protocol(args.seed,args.dataset),checkpoints=checkpoint_plan(args.seed,args.dataset),
            fixed_old_batch=str(ROOT/OLD_BATCH_RELATIVE),old_batch_verified=False,models_not_imported=True,
            checkpoint_bytes_not_read=True,outputs_not_created=True),indent=2));return 0
    run_evaluation(args.seed,args.dataset,output_dir=args.output_dir);return 0


if __name__=="__main__":raise SystemExit(main())
