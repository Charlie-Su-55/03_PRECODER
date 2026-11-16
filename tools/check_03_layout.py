#!/usr/bin/env python3
"""Read-only layout checks; no training imports, network access or file writes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "archive/early_k8/manifest.json"
SOURCE_COMMIT = "31c1fb30e6e2995d847a02226a6cbf19ff86b5b0"
ACTIVE = [f"{verb}_twc_v1_k8_low_snr.py" for verb in
          ("run", "train", "evaluate", "publish")]
ARCHIVED = {
    "diagnose_twc_v1_k8_paired_link.py",
    "diagnose_twc_v1_k8_stage_c_lambda.py",
    "evaluate_twc_v1_k8_paired_validation.py",
    "train_twc_v1_k8_ce.py", "train_twc_v1_k8_codec_stage_c.py",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(name, root_file=False):
    _require(isinstance(name, str) and bool(name), "invalid relative path")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and str(path) == name and
             not any(p in (".", "..") for p in path.parts) and
             re.fullmatch(r"[A-Za-z0-9_./-]+", name),
             f"unsafe relative path: {name!r}")
    if root_file:
        _require(len(path.parts) == 1 and path.suffix == ".py",
                 f"expected a root Python filename: {name}")
    return path


def _file(root, name):
    path = root
    for part in _relative(name).parts:
        path = path / part
        _require(not path.is_symlink(), f"symlink is not a checked file: {name}")
    _require(path.is_file(), f"missing regular file: {name}")
    return path


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root=ROOT):
    """Verify the checked-in layout and report local copies without modifying them."""
    root = Path(root).resolve(strict=True)
    data = _json(_file(root, MANIFEST))
    _require(data["format"] == "precoder_entrypoint_layout_v1" and
             data["source_commit"] == SOURCE_COMMIT and
             data["archive_purpose"] == "historical_reference_only",
             "unexpected layout format, baseline or archive purpose")
    active, records = data["active_entrypoints"], data["archived_files"]
    protected, hidden = data["protected_source_sha256"], data["workspace_hidden_paths"]
    _require(active == ACTIVE and len(records) == 5 and len(protected) == 86 and
             len(hidden) == 103 and len(set(hidden)) == 103,
             "unexpected active, archive, protected or exclusion inventory")
    _require({record["original"] for record in records} == ARCHIVED,
             "unexpected archived entrypoints")
    retained = []
    for record in records:
        original = record["original"]
        _relative(original, root_file=True)
        _require(set(record) == {"original", "archive", "sha256"} and
                 record["archive"] == "archive/early_k8/" + original and
                 original not in protected and original not in active,
                 f"invalid archive mapping: {original}")
        _require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) and
                 _sha(_file(root, record["archive"])) == record["sha256"],
                 f"archive SHA256 mismatch: {original}")
        local = root / original
        if local.is_symlink() or local.exists():
            kind = "symlink" if local.is_symlink() else (
                "file" if local.is_file() else "other")
            retained.append({"path": original, "kind": kind, "preserved": True})
    for name, digest in protected.items():
        _require(_relative(name).suffix == ".py" and
                 re.fullmatch(r"[0-9a-f]{64}", digest) and
                 _sha(_file(root, name)) == digest,
                 f"protected source SHA256 mismatch: {name}")
    for name in active:
        _relative(name, root_file=True)
        _require(name in protected and name not in hidden, f"hidden active entrypoint: {name}")
        _file(root, name)
    for name in hidden:
        _relative(name, root_file=True)
        _require(name not in active and name not in ARCHIVED,
                 f"invalid hidden entrypoint: {name}")
        _file(root, name)
    _require(data["workspace"] == "03_CURRENT.code-workspace", "unexpected workspace path")
    workspace = _json(_file(root, data["workspace"]))
    _require(set(workspace) == {"folders", "settings"} and
             len(workspace["folders"]) == 1 and
             set(workspace["folders"][0]) <= {"name", "path"} and
             workspace["folders"][0].get("path") == "." and
             set(workspace["settings"]) == {"files.exclude"}, "unexpected workspace scope")
    excluded = workspace["settings"]["files.exclude"]
    _require(set(excluded) == set(hidden) and all(v is True for v in excluded.values()),
             "workspace exclusions differ from the recorded root entrypoints")
    return {"status": "passed", "archived_files": len(records),
            "protected_sources": len(protected), "workspace_hidden_paths": len(hidden),
            "active_entrypoints": len(active), "retained_root_copies": retained,
            "warnings": ["Local copies were preserved; inspect them before any manual cleanup."]
                        if retained else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.root), indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Layout check failed: {exc}\n")


if __name__ == "__main__":
    main()
