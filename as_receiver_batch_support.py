"""Stdlib-only batch evidence and exact-target archival (not a general cleaner)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import uuid

ROOT = Path(__file__).resolve().parent
CLEAN_TARGETS = (
    "runs_as_screen/diagnostics/as_direct_v1/preflight_20261025T112943_010217Z_9ea0286d",
    "runs_as_screen/evaluation/k4_as_direct_v1/preflight_20261025T173028_269988Z_e8b82cda",
    "terminal_logs/as_direct_cpu_OZ1waP0M",
    "terminal_logs/as_direct_cuda_2yCQhEoi",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_inside(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if path == root or root not in path.parents:
        raise ValueError(f"Not a strict descendant of allowed root: {path}")
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Link/junction refused: {part}")
        if part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Reparse point refused: {part}")
    if root.resolve() not in path.resolve().parents:
        raise ValueError("Resolved path escaped root")
    return path


def save_json(path, data, *, replace=False):
    path = Path(path)
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        return
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)  # Only a controller-owned status file, never results/weights.


def inventory(target, root):
    target = safe_inside(target, root)
    if not target.is_dir():
        raise ValueError("Cleanup target must be a real directory")
    files, directories = [], []
    for current, names, filenames in os.walk(target, followlinks=False):
        current = safe_inside(current, root)
        for name in names:
            child = safe_inside(current / name, root)
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ValueError("Special directory entry refused")
            directories.append(str(child.relative_to(root)).replace("\\", "/"))
        for name in filenames:
            child = safe_inside(current / name, root)
            info = child.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("Special file/hard link refused")
            files.append(dict(path=child.relative_to(root).as_posix(), size=info.st_size,
                              sha256=sha256(child)))
    return sorted(files, key=lambda row: row["path"]), sorted(directories)


def _expected_content(relative, files, root):
    if not files:
        raise ValueError("Empty/unknown target preserved")
    if sum(row["size"] for row in files) > 128 * 1024 * 1024:
        raise ValueError("Unexpectedly large preflight/log directory preserved")
    names = [Path(row["path"]).name for row in files]
    if any(name.lower().endswith((".pth", ".pt", ".ckpt", ".npz", ".npy")) for name in names):
        raise ValueError("Weight/channel cache present; preserve entire target")
    if relative.startswith("terminal_logs/"):
        # Only text terminal transcripts; no scripts, arbitrary JSON or binary artifacts.
        if any(Path(name).suffix.lower() not in (".txt", ".log", ".out", ".err") for name in names):
            raise ValueError("Unexpected terminal-log file type preserved")
        for row in files:
            text = (root / row["path"]).read_text(encoding="utf-8")
            if "\x00" in text:
                raise ValueError("Binary terminal content preserved")
        return
    if any(not name.endswith((".json", ".csv", ".csv.gz", ".log", ".txt")) for name in names):
        raise ValueError("Unknown preflight artifact type preserved")
    folder = root / relative
    if "/diagnostics/" in relative:
        record = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if record.get("passed") is not True or record.get("status") not in ("passed", "completed"):
            raise ValueError("Unconfirmed/failed preflight preserved")
    else:
        record = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        protocol = record.get("protocol", {})
        if (record.get("status") != "completed" or protocol.get("num_samples") != 8
                or "preflight" not in str(protocol).lower()):
            raise ValueError("Not the completed eight-sample preflight; preserve")
        # A mislabeled directory must not hide a formal raw sample set.
        for row in files:
            path = root / row["path"]
            if path.name.endswith((".csv", ".csv.gz")):
                opener = gzip.open if path.name.endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8", newline="") as stream:
                    for item in csv.DictReader(stream):
                        for key in ("num_samples", "sample_count", "n_samples"):
                            if item.get(key) and float(item[key]) > 8:
                                raise ValueError("Formal sample count found; preserve")
                        if item.get("sample_index") and int(item["sample_index"]) >= 8:
                            raise ValueError("Formal sample index found; preserve")


def archive_exact_target(relative, archive_root, *, root=ROOT):
    """Inventory -> compressed copy -> full verification -> exact listed unlink.

    No glob deletion, extraction, directory moves or recursive rmtree. A valid
    archive survives interruptions. Changed/unknown files keep originals.
    """
    if relative not in CLEAN_TARGETS:
        raise ValueError("Cleanup target not explicitly authorized")
    root = Path(root).absolute()
    target = safe_inside(root / relative, root)
    result = dict(target=relative, status="pending", deleted_files=[])
    if not target.exists():
        return dict(result, status="skipped", reason="directory does not exist")
    try:
        files, directories = inventory(target, root)
        _expected_content(relative, files, root)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        return dict(result, status="preserved", reason=str(error))
    archive_root = safe_inside(archive_root, root)
    archive_root.mkdir(parents=True, exist_ok=True)
    token = Path(relative).name + "_" + uuid.uuid4().hex[:8]
    archive = archive_root / (token + ".tar.gz")
    listing = archive_root / (token + ".manifest.json")
    result.update(files=files, directories=directories, archive=str(archive), listing=str(listing))
    save_json(listing, dict(target=relative, files=files, directories=directories))
    with tarfile.open(archive, "x:gz") as output:
        for row in files:
            output.add(root / row["path"], arcname=row["path"], recursive=False)
    expected = {row["path"]: row for row in files}
    with tarfile.open(archive, "r:gz") as saved:
        members = saved.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise RuntimeError("Archive member list mismatch; originals retained")
        for member in members:
            if not member.isfile() or member.size != expected[member.name]["size"]:
                raise RuntimeError("Archive type/size mismatch; originals retained")
            digest = hashlib.sha256()
            with saved.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected[member.name]["sha256"]:
                raise RuntimeError("Archive hash mismatch; originals retained")
    result.update(archive_sha256=sha256(archive), archive_verified=True)
    save_json(archive_root / (token + ".verified.json"), result)
    if inventory(target, root) != (files, directories):
        return dict(result, status="preserved", reason="Contents changed during archive")
    # Recheck each exact file before unlink; persist partial-progress evidence.
    progress = archive_root / (token + ".removal_progress.json")
    result["removal_progress"] = str(progress)
    save_json(progress, result)
    try:
        for row in files:
            path = safe_inside(root / row["path"], root)
            if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
                raise RuntimeError("Original changed before removal; remaining files retained")
            path.unlink()
            result["deleted_files"].append(row["path"])
            save_json(progress, result, replace=True)
        for name in sorted(directories, key=lambda value: value.count("/"), reverse=True):
            safe_inside(root / name, root).rmdir()  # Empty only, never recursive.
        target.rmdir()
    except Exception as error:
        result.update(status="partial_removal_failed", error=str(error))
        save_json(progress, result, replace=True)
        raise RuntimeError(f"Cleanup stopped; recoverable archive and deletion evidence: {progress}") from error
    result["status"] = "archived_and_removed"
    save_json(archive_root / (token + ".completed.json"), result)
    result["_evidence"] = [str(archive), str(listing), str(progress),
        str(archive_root / (token + ".verified.json")), str(archive_root / (token + ".completed.json"))]
    return result
