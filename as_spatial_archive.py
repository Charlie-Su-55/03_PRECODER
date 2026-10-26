"""Immutable, byte-preserving small-result archives; standard library only.

Verification streams checkpoint/raw-sample hashes but NEVER deserializes weights
or executes experiments. Historical code hashes are provenance, not a demand
that today's code equal the code used by an already completed experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat

from as_receiver_batch_support import safe_inside, save_json, sha256

ROOT = Path(__file__).resolve().parent
SEEDS = (20261030, 20261031)
ARMS = ("frozen_ue", "joint")
METHODS = ARMS + ("full_control", "as_receiver", "swin")
TAG = "k4_as_spatial_v1"
ARTIFACTS = tuple(f"{name}_{TAG}.{extension}" for name, extension in (
    ("summary", "csv"), ("samples", "csv.gz"), ("results", "json"),
    ("metadata", "json"), ("paired_deltas", "csv"), ("paired_samples", "csv.gz"),
    ("parameters", "csv"), ("spatial_parameters", "csv"),
    ("soft_codeword_summary", "csv"), ("soft_codeword_samples", "csv.gz"),
))
MAX_SMALL_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def _same(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label} mismatch")


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Expected a complete SHA256 digest")
    return value.lower()


def _relative(name):
    if not isinstance(name, str):
        raise ValueError("Relative paths must be strings")
    path = Path(name)
    if not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive/source relative path: {name!r}")
    return path


def _historical_source(record):
    if not isinstance(record, dict) or not record:
        raise ValueError("Historical source hashes missing")
    for name, digest in record.items():
        _relative(name)
        _digest(digest)
    return record


class _Verifier:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.checked = {}

    def file(self, path, expected=None):
        path = Path(path)
        path = safe_inside(path if path.is_absolute() else self.root / path, self.root)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Not a regular evidence file: {path}")
        identity = (info.st_size, info.st_mtime_ns, info.st_ino)
        old = self.checked.get(str(path))
        if old is None or old["identity"] != identity:
            value = dict(identity=identity, sha256=sha256(path), size=info.st_size)
            after = path.stat()
            if identity != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f"Evidence changed while hashing: {path}")
            self.checked[str(path)] = value
        value = self.checked[str(path)]
        if expected is not None:
            _same(value["sha256"], _digest(expected), f"SHA256 {path}")
        return path

    def json(self, path):
        path = self.file(path)
        if path.stat().st_size > MAX_SMALL_BYTES:
            raise ValueError(f"Metadata unexpectedly large: {path}")
        def invalid(value):
            raise ValueError(f"Nonfinite JSON value: {value}")
        result = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
        if not isinstance(result, dict):
            raise ValueError(f"Expected object metadata: {path}")
        return result

    def hashes(self, mapping):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Evidence hash map missing")
        return {str(self.file(path, digest)): _digest(digest) for path, digest in mapping.items()}

    def unchanged(self):
        for name, item in self.checked.items():
            path = safe_inside(name, self.root)
            info = path.stat()
            if item["identity"] != (info.st_size, info.st_mtime_ns, info.st_ino):
                raise ValueError(f"Source evidence changed during archival: {name}")


def _verify_evaluations(evaluations, verifier, expected_protocols, required_artifacts):
    evaluations = {str(key): value for key, value in evaluations.items()}
    expected_protocols = {str(key): value for key, value in expected_protocols.items()}
    if not evaluations or set(map(str, evaluations)) != set(map(str, expected_protocols)):
        raise ValueError("Evaluation/protocol keys must match")
    required = set(required_artifacts)
    if not required or len(required) != len(required_artifacts):
        raise ValueError("Explicit unique artifact names required")
    for name in required:
        path = _relative(name)
        if path.name != name or not (name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".json")):
            raise ValueError("Only evaluation CSV/JSON/raw CSV.gz artifact names are allowed")
    files, inputs, histories, manifests = {}, {}, {}, {}
    for key, directory in evaluations.items():
        key = str(key)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Unsafe evaluation label")
        directory = safe_inside(directory, verifier.root)
        manifest_path = directory / "manifest.json"
        record = verifier.json(manifest_path)
        _same(record.get("status"), "completed", "Evaluation completion")
        _same(record.get("results_verified"), True, "Evaluation result checks")
        _same(record.get("protocol"), expected_protocols[key], "Evaluation protocol")
        _same(record.get("output_directory"), str(directory), "Evaluation output identity")
        _same(record.get("freeze_checks", {}).get("passed"), True, "Frozen evaluation")
        _same(record.get("freeze_checks", {}).get("errors"), [], "Frozen evaluation errors")
        before, after = record.get("model_buffers_before"), record.get("model_buffers_after")
        if not isinstance(before, dict) or not before:
            raise ValueError("Frozen buffer/mode evidence missing")
        _same(before, after, "Before/after buffer and mode evidence")
        for model in before.values():
            if not isinstance(model.get("training"), dict) or not model["training"] or any(model["training"].values()):
                raise ValueError("Evaluation contains a training-mode module")
        if not isinstance(record.get("methods"), dict) or set(record["methods"]) != set(before):
            raise ValueError("Method/frozen-evidence identity mismatch")
        _same(record.get("input_sha256_before"), record.get("input_sha256_after"), "Before/after input hashes")
        verified_inputs = verifier.hashes(record.get("input_sha256_after"))
        for name, digest in verified_inputs.items():
            if name in inputs and inputs[name] != digest:
                raise ValueError("Conflicting evaluation input identity")
            inputs[name] = digest
        for method in record["methods"].values():
            _same(method.get("strict_loading"), True, "Strict checkpoint loading")
            best = str(verifier.file(method["best_path"], method["best_sha256"]))
            _same(inputs.get(best), _digest(method["best_sha256"]), "Selected checkpoint input identity")
        histories[key] = dict(source_sha256=_historical_source(record.get("runtime", {}).get("source_sha256")),
                              runtime=record.get("runtime"), protocol=record["protocol"])
        saved = record.get("output_sha256")
        if not isinstance(saved, dict) or set(saved) != required:
            raise ValueError("Missing/extra output artifact hash entries")
        actual = set()
        for entry in directory.iterdir():
            safe_inside(entry, verifier.root)
            if not entry.is_file():
                raise ValueError("Unexpected non-file in completed evaluation")
            actual.add(entry.name)
        _same(actual, required | {"manifest.json"}, "Evaluation artifact file set")
        files[f"evaluations/{key}/manifest.json"] = manifest_path
        for name in sorted(required):
            path = verifier.file(directory / name, saved[name])
            if not name.endswith(".csv.gz"):
                files[f"evaluations/{key}/{name}"] = path
        manifests[key] = record
    return files, inputs, histories, manifests


def _inventory(files, verifier):
    inventory = {}
    total = 0
    for name, source in sorted(files.items()):
        relative = _relative(name)
        if relative.suffix not in (".csv", ".json") or name.endswith(".csv.gz"):
            raise ValueError("Archive excludes weights, compressed raw samples and arbitrary files")
        source = verifier.file(source)
        if source.suffix not in (".csv", ".json") or source.name.endswith(".csv.gz"):
            raise ValueError("Cannot disguise a forbidden source artifact as a small CSV/JSON")
        info = verifier.checked[str(source)]
        if info["size"] > MAX_SMALL_BYTES:
            raise ValueError(f"Small result file exceeds explicit limit; never truncate: {source}")
        total += info["size"]
        inventory[name] = dict(source_absolute=str(source), source_repository_relative=source.relative_to(verifier.root).as_posix(),
                               size=info["size"], sha256=info["sha256"])
    if total > MAX_ARCHIVE_BYTES:
        raise ValueError("Small archive exceeds explicit total size limit")
    return inventory


def _publish(files, verifier, archive_root, archive_id, provenance, evaluations, inputs, histories):
    if not isinstance(archive_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", archive_id) or archive_id in (".", ".."):
        raise ValueError("Unsafe immutable archive identifier")
    archive_root = safe_inside(archive_root, verifier.root)
    destination = safe_inside(archive_root / archive_id, archive_root)
    inventory = _inventory(files, verifier)
    manifest = dict(schema="verified-small-evaluation-archive-v1", status="verified",
                    archive_id=archive_id, files=inventory, provenance=provenance or {},
                    evaluations={str(key): str(value) for key, value in evaluations.items()},
                    frozen_input_sha256=inputs, historical_source_provenance=histories,
                    code_hash_policy="Recorded historical hashes retained; current code equality is NOT required",
                    weights_deserialized=False, experiments_executed=False,
                    numeric_policy="Original file bytes, all rows/columns, no rounding or rewriting",
                    excluded_from_copy="weights and raw/compressed samples; their hashes are verified, originals retained")
    reused = destination.exists()
    if reused:
        if not destination.is_dir():
            raise ValueError("Archive destination is not a directory")
        archived = verifier.json(destination / "archive_manifest.json")
        _same(archived, manifest, "Existing immutable archive identity")
        actual = set()
        for path in destination.rglob("*"):
            safe_inside(path, verifier.root)
            if path.is_file():
                actual.add(path.relative_to(destination).as_posix())
            elif not path.is_dir():
                raise ValueError("Unexpected archive entry")
        _same(actual, set(inventory) | {"archive_manifest.json"}, "Existing immutable archive members")
        for name, row in inventory.items():
            archived_file = verifier.file(destination / name, row["sha256"])
            _same(archived_file.stat().st_size, row["size"], "Archived file size")
    else:
        verifier.unchanged()
        destination.mkdir(parents=True, exist_ok=False)
        # Publish a completion manifest LAST. A partial destination is retained
        # and rejected on retry, never silently overwritten or declared valid.
        for name, row in inventory.items():
            target = safe_inside(destination / name, destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            with Path(row["source_absolute"]).open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            _same(sha256(target), row["sha256"], "Copied bytes")
            _same(sha256(row["source_absolute"]), row["sha256"], "Unchanged source bytes")
        verifier.unchanged()
        save_json(destination / "archive_manifest.json", manifest)
    verifier.unchanged()
    return dict(archive_directory=str(destination), reused=reused,
                evaluations=manifest["evaluations"], frozen_input_sha256=inputs,
                source_evidence=sorted(verifier.checked),
                historical_source_provenance=histories,
                _evidence=[str(destination / name) for name in inventory] + [str(destination / "archive_manifest.json")])


def archive_evaluations(evaluations, *, root=ROOT, archive_root, archive_id,
                        expected_protocols, required_artifacts, provenance=None, evidence_files=None):
    """Verify completed evaluations, then copy only CSV/JSON original bytes.

    required_artifacts includes raw .csv.gz for verification, never for copying.
    expected_protocols is an explicit {label: exact protocol} mapping supplied by
    the completed experiment's owner. No model/trainer/evaluator is invoked.
    """
    verifier = _Verifier(root)
    files, inputs, histories, _ = _verify_evaluations(
        evaluations, verifier, expected_protocols, required_artifacts
    )
    for name, path in (evidence_files or {}).items():
        if name in files or name == "archive_manifest.json":
            raise ValueError("Evidence destination collides with evaluation files")
        files[name] = path
    return _publish(files, verifier, archive_root, archive_id, provenance, evaluations, inputs, histories)


def _spatial_protocol(record, seed):
    expected = dict(training_seed=seed, seed=20261103, preflight=False, batch_size=8,
                    num_batches=200, num_samples=1600, feedback_snr_db=[0, 5, 10, 15, 20, 25],
                    downlink_snr_db=25, channel_profile="CDL-C",
                    scenario=dict(K=4, Nt=32, Nsc=256, Dtot=256, Nsb=4),
                    dataset_role="independent_screening_test")
    if not isinstance(record, dict):
        raise ValueError("ASSpatial evaluation protocol missing")
    for key, value in expected.items():
        _same(record.get(key), value, f"ASSpatial protocol {key}")
    data = record.get("channel_data_protocol", {})
    _same(data.get("protocol_version"), "training-channel-data-v1", "Channel protocol")
    _same(data.get("base_seed"), 20261103, "Independent channel seed")
    _same(data.get("sionna_version"), "2.0.1", "Historical Sionna version")
    _same(data.get("batch_sizes"), dict(training=8, validation=8), "Channel batches")
    for key, value in dict(k_users=4, nt=32, nsc=256, cdl_model="C").items():
        _same(data.get("physical", {}).get(key), value, "Channel physical setting")
    _digest(data.get("data_id"))
    return record


def _verify_spatial(root):
    verifier = _Verifier(root)
    pointer_path = verifier.root / "runs_as_screen/proposal_as_spatial_v1/last_batch.json"
    pointer = verifier.json(pointer_path)
    batch = safe_inside(pointer["directory"], verifier.root / "runs_as_screen/proposal_as_spatial_v1/batches")
    summary_path = batch / "batch_summary.json"
    summary = verifier.json(summary_path)
    _same(summary.get("status"), "completed", "ASSpatial batch status")
    historical = dict(batch_source=summary.get("source"))
    _historical_source(summary.get("source", {}).get("sha256"))
    stages = summary.get("stages", {})
    required_stages = {"environment_and_old_controls", "new_model_checks", "organize_results"}
    required_stages |= {f"train_{arm}_{seed}" for seed in SEEDS for arm in ARMS}
    required_stages |= {f"evaluate_{seed}" for seed in SEEDS}
    _same(set(stages), required_stages, "Completed batch stage set")
    evidence_files = {"evidence/batch_summary.json": summary_path,
                      "evidence/last_batch.json": pointer_path}
    for key, stage in stages.items():
        _same(stage.get("status"), "completed", f"Stage {key}")
        verifier.hashes(stage.get("evidence_sha256"))
        result = stage.get("result", {})
        if "report_path" in result:
            report_path = verifier.file(result["report_path"])
            report = verifier.json(report_path)
            _same(report, result.get("report"), f"Stage {key} report")
            _same(report.get("passed"), True, f"Stage {key} pass")
            evidence_files[f"evidence/{key}.json"] = report_path
    model_report = stages["new_model_checks"]["result"]["report"]
    _same(model_report.get("same_initial_state_between_arms"), True, "Initial arm identity")
    _same(model_report.get("source_ue_exact"), True, "Source UE migration")
    source_sha = _digest(model_report.get("source_sha256"))
    source_identity = None
    training = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"train_{arm}_{seed}"
            report = stages[key]["result"]["report"]
            _same(summary.get("training", {}).get(key), report, f"{key} summary/report")
            for field, value in dict(arm=arm, training_seed=seed, completed_updates=20000).items():
                _same(report.get(field), value, f"{key} {field}")
            status = report.get("status", {})
            _same(status.get("state"), "completed", f"{key} status")
            _same(status.get("step"), 20000, f"{key} step")
            best_step = report.get("best_step")
            if type(best_step) is not int or not 0 < best_step <= 20000 or best_step % 500:
                raise ValueError("Invalid selected best update")
            best = verifier.file(report["best_path"], report["best_sha256"])
            latest = verifier.file(report["latest_path"], report["latest_sha256"])
            safe_inside(best, verifier.root / "runs_as_screen/proposal_as_spatial_v1")
            _same(best.name, "best.pth", "Selected best filename")
            _same(latest.name, "latest.pth", "Latest filename")
            _same(best.parent, latest.parent, "Training checkpoint directory")
            config_path, status_path = best.parent / "config.json", best.parent / "status.json"
            config, actual_status = verifier.json(config_path), verifier.json(status_path)
            _same(actual_status, status, f"{key} status sidecar")
            fingerprint = config.get("config_fingerprint")
            snapshot = {name: value for name, value in config.items() if name != "config_fingerprint"}
            computed = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                                                allow_nan=False).encode()).hexdigest()[:16]
            _same(fingerprint, computed, f"{key} full config fingerprint")
            _same(snapshot.get("arm"), arm, f"{key} config arm")
            _same(snapshot.get("experiment", {}).get("seed"), seed, f"{key} config seed")
            _same(snapshot.get("allocations"), [[0, 256]], f"{key} pure AS")
            _same(snapshot.get("training_recipe", {}).get("total_steps"), 20000, f"{key} budget")
            _same(snapshot.get("architecture_metadata", {}).get("architecture"), "as_spatial_v1", "Architecture")
            identity = snapshot.get("source_identity", {})
            _same(_digest(identity.get("checkpoint_sha256")), source_sha, f"{key} source identity")
            verifier.file(identity["path"], source_sha)
            if source_identity is None:
                source_identity = identity
            _same(identity, source_identity, "Common Full UE source")
            evidence_files[f"training/{key}/config.json"] = config_path
            evidence_files[f"training/{key}/status.json"] = status_path
            training[key] = dict(report=report, config=snapshot, config_fingerprint=fingerprint)
    evaluations = {str(seed): stages[f"evaluate_{seed}"]["result"]["output_directory"] for seed in SEEDS}
    for seed, directory in evaluations.items():
        safe_inside(directory, verifier.root / "runs_as_screen/evaluation/k4_as_spatial_v1" / batch.name / f"seed_{seed}")
    expected_protocols = {str(seed): _spatial_protocol(verifier.json(Path(evaluations[str(seed)]) / "manifest.json").get("protocol"), seed)
                          for seed in SEEDS}
    files, inputs, histories, manifests = _verify_evaluations(evaluations, verifier, expected_protocols, ARTIFACTS)
    inputs[str(verifier.file(source_identity["path"], source_sha))] = source_sha
    from as_spatial_results import interpret
    for seed in SEEDS:
        directory, manifest = Path(evaluations[str(seed)]), manifests[str(seed)]
        _same(set(manifest["methods"]), set(METHODS), "ASSpatial five methods")
        _same(manifest.get("diagnostic_rng_checks", {}).get("passed"), True, "Diagnostic RNG preservation")
        for arm in ARMS:
            train = training[f"train_{arm}_{seed}"]
            method = manifest["methods"][arm]
            for field, expected in dict(config=train["config"], config_fingerprint=train["config_fingerprint"],
                                        completed_updates=20000, latest_step=20000,
                                        selected_best_step=train["report"]["best_step"],
                                        best_sha256=train["report"]["best_sha256"],
                                        latest_sha256=train["report"]["latest_sha256"]).items():
                _same(method.get(field), expected, f"Evaluation/training {arm}.{field}")
        def csv_rows(name):
            with (directory / name).open(encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        interpret(csv_rows(f"summary_{TAG}.csv"), csv_rows(f"paired_deltas_{TAG}.csv"), seed)
        parameters = csv_rows(f"parameters_{TAG}.csv")
        _same({row["method_key"] for row in parameters}, set(METHODS), "Actual parameter methods")
        if len(parameters) != 5 or any(int(row["full_parameters"]) <= 0 for row in parameters):
            raise ValueError("Actual parameter counts incomplete")
        spatial = csv_rows(f"spatial_parameters_{TAG}.csv")
        _same({row["method_key"] for row in spatial}, set(ARMS), "New-BS parameter statistics")
        if any(not row["name"].startswith("decoder.") or int(row["numel"]) <= 0 or
               not math.isfinite(float(row["l2_norm"])) for row in spatial):
            raise ValueError("Invalid new-BS parameter statistics")
        if len({(row["method_key"], row["name"]) for row in spatial}) != len(spatial):
            raise ValueError("Duplicate new-BS parameter statistics")
        for arm in ARMS:
            actual = model_report.get("actual_parameters", {}).get(arm, {})
            parameter_row = next(row for row in parameters if row["method_key"] == arm)
            _same(int(parameter_row["full_parameters"]), actual.get("total"), "Instantiated total parameter count")
            _same(int(parameter_row["encoder_plus_condition_parameters"]), actual.get("ue_encoder_condition"), "Instantiated UE parameter count")
            spatial_count = sum(int(row["numel"]) for row in spatial if row["method_key"] == arm)
            _same(spatial_count, actual.get("new_bs"), "Instantiated new-BS parameter count")
            _same(actual.get("total"), actual.get("ue_encoder_condition", -1) + spatial_count, "UE + BS parameter sum")
    historical.update(source_identity=source_identity, batch_directory=str(batch),
                      checkpoint_validation_scope="Recorded completion and current byte hashes; no weight deserialization")
    files.update(evidence_files)
    verifier.unchanged()
    return verifier, files, inputs, histories, evaluations, historical, batch


def verify_completed_as_spatial(*, root=ROOT):
    """Read-only validation of the existing completed pointer-selected batch."""
    verifier, files, inputs, histories, evaluations, historical, batch = _verify_spatial(root)
    return dict(batch_directory=str(batch), evaluations=evaluations,
                frozen_input_sha256=inputs, source_evidence=sorted(verifier.checked),
                historical_source_provenance=histories, provenance=historical,
                selected_files={name: str(path) for name, path in files.items()},
                _evidence=sorted(verifier.checked))


def archive_completed_as_spatial(*, root=ROOT, destination_root=None):
    verifier, files, inputs, histories, evaluations, historical, batch = _verify_spatial(root)
    destination_root = Path(root) / "paper/data/as_spatial_verified" if destination_root is None else Path(destination_root)
    return _publish(files, verifier, destination_root, batch.name, historical, evaluations, inputs, histories)
