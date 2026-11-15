#!/usr/bin/env python3
"""Plan or archive five retired 03 entrypoints without touching live research.

Git removes these exact obsolete sources when the cleanup commit is merged.
This tool handles only unchanged *untracked* copies left on an SSH checkout.
It never removes tracked, staged, edited, linked or unknown files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'https://github.com/Charlie-Su-55/03_PRECODER.git'
SOURCE_COMMIT = '895fc71b40baab5cba1744ff457d45f277c46b0f'
RETIRED = {
    'diagnose_cross_cdl_condition.py': '30b1580d003262c706fc555581e0f00d26b06c7260369b938516d966d02d54d5',
    'old/diagnostic_eval.py': '6392dcc72bcf824faafa10e99f6d9263a266761d41084af4296e96239e2642de',
    'old/train_adaptive_8ue.py': '6f66737d67f365a2e49bb393da94cc3445997f0917ed83dc1630b62ddb07bfa1',
    'old/train_csinet_baseline_8ue.py': '7e6c701be15b043a9c03ac6a1f985bf747517607cd9d97c5a7d956d21ce0f169',
    'old/train_swin_baseline_8ue.py': 'd83c7e49409e13ac5a3c8bd7cadb57314c46c786a5edfb51867ae349a3b6126b',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    return subprocess.run(['git', *args], cwd=root, capture_output=True,
                          check=True, env=env).stdout


def safe_child(root, relative):
    name = PurePosixPath(relative)
    if (name.is_absolute() or str(name) != relative or '..' in name.parts
            or '\\' in relative or not name.parts):
        raise ValueError('Canonical relative path required')
    cursor = root
    for part in name.parts:
        cursor = cursor/part
        if cursor.is_symlink():
            raise ValueError('Linked paths are preserved')
    if not cursor.resolve().is_relative_to(root):
        raise ValueError('Path escaped the authorized repository')
    return cursor


def checked_root(value, *, apply):
    root = Path(value).absolute()
    if root.is_symlink() or root.resolve() != root:
        raise ValueError('Use the physical repository path')
    if Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()) != root:
        raise ValueError('Exact repository root required')
    if git(root, 'remote', 'get-url', 'origin').decode().strip() != ORIGIN:
        raise ValueError('Official 03 origin required')
    if apply and git(root, 'branch', '--show-current').decode().strip() != 'main':
        raise ValueError('SSH apply requires the main checkout')
    return root


def active_processes(root):
    """Read Linux same-user process metadata; never kill or print command lines."""
    if not Path('/proc').is_dir():
        return [dict(reason='Reliable Linux process inventory unavailable')]
    blockers = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            name = (proc/'comm').read_text().strip()
            try:
                executable = (proc/'exe').resolve().name
            except OSError:
                executable = name
            if not any(re.fullmatch(r'python[0-9.]*', x) for x in (name, executable)):
                continue
            cwd = (proc/'cwd').resolve(strict=True)
            command = (proc/'cmdline').read_bytes()
            if cwd == root or cwd.is_relative_to(root) or os.fsencode(root) in command:
                blockers.append(dict(pid=int(proc.name), reason='Active 03 Python process'))
        except FileNotFoundError:
            continue
        except (PermissionError, OSError):
            blockers.append(dict(pid=int(proc.name), reason='Process inspection uncertain'))
    return blockers


def inventory(root):
    tracked = set(filter(None, git(root, 'ls-files', '-z').decode().split('\0')))
    staged = set(filter(None, git(root, 'diff', '--cached', '--name-only', '-z').decode().split('\0')))
    records = []
    for name, expected in RETIRED.items():
        row = dict(path=name, expected_sha256=expected)
        try:
            path = safe_child(root, name)
            if not path.exists():
                row['status'] = 'absent'
            elif name in staged:
                row['status'] = 'staged_preserved'
            elif name in tracked:
                row['status'] = 'tracked_preserved'
            else:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    row['status'] = 'nonregular_or_hardlinked_preserved'
                elif info.st_size > 1024*1024:
                    row['status'] = 'unexpected_size_preserved'
                else:
                    row['actual_sha256'] = sha256(path)
                    row['status'] = ('archive_candidate' if row['actual_sha256'] == expected
                                     else 'modified_or_unknown_preserved')
                    row['file_identity'] = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
        except (OSError, ValueError):
            row['status'] = 'unsafe_or_linked_preserved'
        records.append(row)
    return records


def git_identity(root):
    index = Path(git(root, 'rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute():
        index = root/index
    if index.is_symlink():
        raise ValueError('Regular local index required')
    return dict(head=git(root, 'rev-parse', 'HEAD').decode().strip(),
                index_sha256=sha256(index) if index.exists() else None)


def private_directory(root, relative):
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = safe_child(root, cursor.relative_to(root).joinpath(part).as_posix())
        if cursor.exists():
            if not cursor.is_dir():
                raise ValueError('Archive parent must be a directory')
        else:
            cursor.mkdir(mode=0o700)
            cursor.chmod(0o700)
    return cursor


def archive_one(root, row, target):
    """Copy and verify first; a failed copy never removes the original."""
    path = safe_child(root, row['path'])
    info = path.lstat()
    identity = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
    if identity != row['file_identity'] or sha256(path) != row['expected_sha256']:
        raise ValueError('Candidate changed after inventory')
    destination = private_directory(root, (target/'files').relative_to(root).as_posix())/row['path']
    private_directory(root, destination.parent.relative_to(root).as_posix())
    with path.open('rb') as source, destination.open('xb') as saved:
        os.chmod(destination, 0o600)
        saved.write(source.read())
        saved.flush()
        os.fsync(saved.fileno())
    if sha256(destination) != row['expected_sha256']:
        raise ValueError('Archive verification failed; original preserved')
    current = path.lstat()
    if ([current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns] != identity
            or sha256(path) != row['expected_sha256']):
        raise ValueError('Original changed during archive; preserved')
    path.unlink()
    return destination.relative_to(root).as_posix()


def cleanup(root=ROOT, *, apply=False):
    root = checked_root(root, apply=apply)
    before = git_identity(root)
    records = inventory(root)
    blockers = active_processes(root)
    result = dict(format='precoder_obsolete_cleanup_v1', apply=bool(apply),
                  source_commit=SOURCE_COMMIT, records=records, blockers=blockers,
                  archive=None, tracked_files_removed=0, source_history_preserved_in_git=True)
    if apply and blockers:
        raise ValueError('Active or uncertain 03 Python processes; nothing archived')
    candidates = [r for r in records if r['status'] == 'archive_candidate']
    if apply and candidates:
        # Deliberately avoid external worktree gitdirs and all user output trees.
        local_git = safe_child(root, '.git')
        if not local_git.is_dir():
            raise ValueError('Apply requires a regular in-repository .git directory')
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]
        target = private_directory(root, '.git/precoder_cleanup/'+name)
        result['archive'] = target.relative_to(root).as_posix()
        for row in candidates:
            # Recheck Git ownership immediately before each source mutation.
            current = {r['path']:r for r in inventory(root)}[row['path']]
            if current != row:
                row['status'] = 'changed_during_apply_preserved'
                continue
            try:
                row['archive_path'] = archive_one(root, row, target)
                row['status'] = 'archived'
            except (OSError, ValueError) as exc:
                row['status'] = 'archive_failed_original_preserved'
                row['error_type'] = type(exc).__name__
        result['git_before'] = before
        result['git_after'] = git_identity(root)
        if result['git_after'] != before:
            raise RuntimeError('Git HEAD/index changed during cleanup; inspect the preserved archive')
        receipt = target/'receipt.json'
        with receipt.open('x') as stream:
            os.chmod(receipt, 0o600)
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
    elif git_identity(root) != before:
        raise RuntimeError('Git HEAD/index changed during read-only inventory')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--apply', action='store_true', help='Archive exact unchanged untracked retired copies')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(cleanup(args.root, apply=args.apply), indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, 'Cleanup stopped: '+str(exc)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
