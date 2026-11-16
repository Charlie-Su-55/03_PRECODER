"""Filesystem checks for the historical layout and unchanged scientific sources."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_03_layout", ROOT / "tools/check_03_layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class LayoutChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkout"
        self.root.mkdir()
        self.manifest = json.loads((ROOT / layout.MANIFEST).read_text())
        paths = {layout.MANIFEST, self.manifest["workspace"]}
        paths.update(self.manifest["protected_source_sha256"])
        paths.update(self.manifest["workspace_hidden_paths"])
        paths.update(record["archive"] for record in self.manifest["archived_files"])
        for name in paths:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def save_manifest(self, value):
        (self.root / layout.MANIFEST).write_text(json.dumps(value))

    def snapshot(self):
        return {str(p.relative_to(self.root)): (
            p.lstat().st_mode, p.lstat().st_mtime_ns,
            os.readlink(p) if p.is_symlink() else p.read_bytes())
            for p in self.root.rglob("*") if p.is_symlink() or p.is_file()}

    def test_real_checkout_preserves_baseline_git_blobs_and_current_sources(self):
        result = layout.check(ROOT)
        self.assertEqual((result["archived_files"], result["protected_sources"],
                          result["workspace_hidden_paths"], result["active_entrypoints"]),
                         (5, 86, 103, 4))
        for record in self.manifest["archived_files"]:
            original = subprocess.check_output(
                ["git", "show", f'{layout.SOURCE_COMMIT}:{record["original"]}'], cwd=ROOT)
            self.assertEqual((ROOT / record["archive"]).read_bytes(), original)
        for name, digest in self.manifest["protected_source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        excluded = json.loads((ROOT / self.manifest["workspace"]).read_text())["settings"]["files.exclude"]
        self.assertEqual(set(excluded), set(self.manifest["workspace_hidden_paths"]))
        self.assertTrue(set(layout.ACTIVE).isdisjoint(excluded))

    def test_cli_runs_without_site_packages_and_keeps_every_local_file(self):
        for name in (".git/config", ".gitignore", "notes/local-untracked.py"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("local content must remain unchanged\n")
        before = self.snapshot()
        result = subprocess.run(
            [sys.executable, "-B", "-S", str(ROOT / "tools/check_03_layout.py"),
             "--root", str(self.root)], cwd=self.temp.name,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "passed")
        self.assertEqual(self.snapshot(), before)

    def test_changed_or_symlinked_protected_and_archived_files_are_rejected(self):
        paths = ["configs/twc_v1_k8_low_snr_config.py", layout.ACTIVE[0],
                 self.manifest["archived_files"][0]["archive"]]
        for name in paths:
            path = self.root / name
            original = path.read_bytes()
            with self.subTest(name=name, change="bytes"):
                path.write_bytes(original + b"\n# modified\n")
                with self.assertRaises(ValueError):
                    layout.check(self.root)
                path.write_bytes(original)
            with self.subTest(name=name, change="symlink"):
                external = Path(self.temp.name) / "external.py"
                external.write_bytes(original)
                path.unlink()
                path.symlink_to(external)
                with self.assertRaises(ValueError):
                    layout.check(self.root)
                path.unlink()
                path.write_bytes(original)

    def test_workspace_cannot_hide_active_entries_directories_or_unrecorded_paths(self):
        workspace_path = self.root / self.manifest["workspace"]
        original = workspace_path.read_text()
        for change in ("active", "directory", "unrecorded", "false", "scope"):
            with self.subTest(change=change):
                workspace = json.loads(original)
                excluded = workspace["settings"]["files.exclude"]
                if change == "active":
                    excluded[layout.ACTIVE[0]] = True
                elif change == "directory":
                    excluded["models"] = True
                elif change == "unrecorded":
                    excluded["local_untracked.py"] = True
                elif change == "false":
                    excluded[next(iter(excluded))] = False
                else:
                    workspace["folders"][0]["path"] = ".."
                workspace_path.write_text(json.dumps(workspace))
                with self.assertRaises(ValueError):
                    layout.check(self.root)
        workspace_path.write_text(original)

    def test_manifest_cannot_redirect_paths_change_baseline_or_duplicate_keys(self):
        for change in ("parent", "absolute", "baseline", "sha", "active", "duplicate"):
            with self.subTest(change=change):
                manifest = copy.deepcopy(self.manifest)
                if change == "parent":
                    manifest["archived_files"][0]["archive"] = "../outside.py"
                elif change == "absolute":
                    digest = manifest["protected_source_sha256"].pop(next(iter(manifest["protected_source_sha256"])))
                    manifest["protected_source_sha256"]["/tmp/outside.py"] = digest
                elif change == "baseline":
                    manifest["source_commit"] = "0" * 40
                elif change == "sha":
                    manifest["archived_files"][0]["sha256"] = "0" * 64
                elif change == "active":
                    manifest["active_entrypoints"][0] = "train_obsolete.py"
                self.save_manifest(manifest)
                if change == "duplicate":
                    path = self.root / layout.MANIFEST
                    path.write_text('{"format":"ignored",' + path.read_text()[1:])
                with self.assertRaises(ValueError):
                    layout.check(self.root)
        self.save_manifest(self.manifest)

    def test_retained_edited_and_outside_symlink_copies_are_preserved_and_reported(self):
        original_names = [record["original"] for record in self.manifest["archived_files"]]
        (self.root / original_names[0]).write_text("locally edited historical source\n")
        external = Path(self.temp.name) / "private-outside.py"
        external.write_text("outside sentinel; do not follow the retained link\n")
        (self.root / original_names[1]).symlink_to(external)
        (self.root / original_names[2]).symlink_to("missing-untracked-source.py")
        before = self.snapshot()
        external_before = external.stat().st_mtime_ns, external.read_bytes()
        result = layout.check(self.root)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["warnings"])
        self.assertEqual([entry["kind"] for entry in result["retained_root_copies"]],
                         ["file", "symlink", "symlink"])
        self.assertTrue(all(entry["preserved"] for entry in result["retained_root_copies"]))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual((external.stat().st_mtime_ns, external.read_bytes()), external_before)


if __name__ == "__main__":
    unittest.main()
