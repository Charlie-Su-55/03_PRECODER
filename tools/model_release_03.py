#!/usr/bin/env python3
"""Archive scientific run bytes for Release assets; never load model pickles.

This is an integrity-preserving backup, not a scientific checkpoint loader.
Existing strict scientific loaders remain responsible for model validation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

FORMAT = 'precoder_model_archive_v1'
PART_BYTES = 1024**3
MAX_MANIFEST = 64*1024**2
OLD_RUNS = (
    'twc_v1_k8_icc_aligned_savefix_v1', 'twc_v1_k8_icc_rate_v1',
    'twc_v1_k8_icc_alpha_v1', 'twc_v1_k8_w_residual_v1',
    'twc_v1_k8_ul_diagnostics_v1', 'twc_v1_k8_ul_propagation_v1',
    'twc_v1_k8_ul_calibration_v1', 'twc_v1_k8_ul_adapt_v1',
    'twc_v1_k8_ue_bs_rate_v1', 'twc_v1_k8_rate_headroom_v1',
    'twc_v1_k8_hybrid_direction_v1', 'twc_v1_k8_low_snr_interference_v1')
NEW_RUN = 'twc_v1_k8_ce_refine_v1'
SOURCE_CE = 'runs/twc_v1_k8_icc_aligned_savefix_v1/training/ce'
DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def _safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == '.'):
        raise ValueError('Unsafe archive path')
    return name


def _unlinked(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Symlink paths are not permitted')
    return path


def _regular(path):
    path = _unlinked(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('A regular file is required')
    return path


def _sha(path):
    with _regular(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path):
    path = _regular(path)
    if path.stat().st_size > MAX_MANIFEST:
        raise ValueError('Metadata exceeds the bounded JSON size')
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError('Duplicate JSON key')
            output[key] = value
        return output
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _under(root, name):
    name = _safe_name(name)
    target = _unlinked(root/name)
    if not target.is_relative_to(root):
        raise ValueError('Path escapes the repository')
    return target


def _source_refs(value):
    if isinstance(value, dict):
        if 'frozen_files' in value:
            entries = value['frozen_files']
            if not isinstance(entries, list):
                raise ValueError('Frozen source inventory must be a list')
            for entry in entries:
                if (not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}
                        or not isinstance(entry['path'], str)
                        or not DIGEST.fullmatch(str(entry['sha256']))):
                    raise ValueError('Frozen source entry is malformed')
                yield entry
        for key, item in value.items():
            if key != 'frozen_files':
                yield from _source_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _source_refs(item)


def _terminal(directory, step=None, production=False):
    record = _json(directory/'manifest.json')
    if (record != _json(directory/'terminal_receipt.json')
            or record.get('status') != 'completed'
            or (step is not None and record.get('step') != step)
            or (production and record.get('production_evidence') is not True)
            or record.get('checkpoint_sha256') != _sha(directory/'checkpoint_latest.pt')):
        raise ValueError('Completed terminal checkpoint/receipt bytes are required')
    _regular(directory/'initialization.json')
    return record


def _ce_history(directory):
    for step in range(0, 60001, 5000):
        name = f'checkpoint_step_{step:05d}'
        snapshot = _json(directory/(name+'.json'))
        if (snapshot.get('step') != step
                or snapshot.get('checkpoint_sha256') != _sha(directory/(name+'.pt'))):
            raise ValueError('An immutable CE curve checkpoint is missing or changed')
        monitor = directory/'monitor'/f'step_{step:05d}'
        record = _json(monitor/'manifest.json')
        files = record.get('files', {})
        if (record.get('status') != 'completed' or record.get('step') != step
                or set(files) != {'summary.csv', 'input_fingerprints.json'}
                or any(_sha(monitor/name) != digest for name, digest in files.items())):
            raise ValueError('Immutable CE curve monitor evidence is missing or changed')


def _select(root, profile):
    mandatory = OLD_RUNS[-1] if profile == 'old' else NEW_RUN
    summary = _json(_under(root, 'runs/'+mandatory)/'summary.json')
    if summary.get('status') != 'completed' or summary.get('namespace') != mandatory:
        raise ValueError('The requested experiment must be completed before backup')
    trajectories, step = (('low_snr_control', 'low_snr_interference'), 10000) if profile == 'old' else (('nmse', 'geometry'), 60000)
    records = [_terminal(root/'runs'/mandatory/'training'/name, step, True)
        for name in trajectories]
    for name, record in zip(trajectories, records):
        directory = root/'runs'/mandatory/'training'/name
        if profile == 'old':
            if record.get('progress_sha256') != _sha(directory/'progress.jsonl'):
                raise ValueError('Complete Hybrid progress evidence is missing or changed')
        else:
            _ce_history(directory)
    roots = []
    for name in OLD_RUNS if profile == 'old' else (NEW_RUN,):
        relative = 'runs/'+name
        path = _under(root, relative)
        if path.exists():
            if not path.is_dir():
                raise ValueError('Run path must be a directory')
            # Never silently back up a run still being written.
            if (path/'summary.json').exists() and _json(path/'summary.json').get('status') != 'completed':
                raise ValueError('Selected historical run is not completed: '+name)
            roots.append(relative)
            print('Preparing run: '+relative, file=sys.stderr, flush=True)
    if profile == 'new':
        _terminal(_under(root, SOURCE_CE), 5000)
        roots.append(SOURCE_CE)
    return roots, records


class _SplitWriter:
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.stream, self.count, self.digest = None, 0, None
        self.parts = []

    def _close(self):
        if self.stream is not None:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
            self.parts[-1].update(size=self.count, sha256=self.digest.hexdigest())
            if self.limit >= 1024**2:
                print(f'Archive part complete: {self.parts[-1]["name"]} ({self.count} bytes)',
                    file=sys.stderr, flush=True)
            self.stream = None

    def write(self, data):
        view = memoryview(data)
        for_start = len(view)
        while view:
            if self.stream is None or self.count == self.limit:
                self._close()
                name = f'models.tar.part{len(self.parts):04d}'
                self.stream = (self.directory/name).open('xb')
                self.parts.append(dict(name=name)); self.count = 0
                self.digest = hashlib.sha256()
            chunk = view[:self.limit-self.count]
            self.stream.write(chunk); self.digest.update(chunk)
            self.count += len(chunk); view = view[len(chunk):]
        return for_start

    def flush(self):
        if self.stream is not None: self.stream.flush()


class _HashedReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()
    def read(self, size=-1):
        value = self.stream.read(size); self.digest.update(value); return value


def prepare(root, profile, output_dir, part_bytes=PART_BYTES):
    """Create immutable split-tar assets from completed run bytes, without torch."""
    root, output = _unlinked(root), _unlinked(output_dir)
    if profile not in ('old', 'new') or type(part_bytes) is not int or not 0 < part_bytes <= PART_BYTES:
        raise ValueError('Declared profile and a part limit up to 1GiB required')
    if output.exists() or output.is_relative_to(root/'runs'):
        raise ValueError('A new output directory outside runs is required')
    roots, records = _select(root, profile)
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
        capture_output=True, text=True, check=False)
    commit = result.stdout.strip()
    if result.returncode or re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('A Git repository with a committed source identity is required')
    paths, stamps, required_hashes = {}, {}, {}
    def stamp(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    for relative in roots:
        for path in _under(root, relative).rglob('*'):
            _unlinked(path)
            if path.is_dir(): continue
            _regular(path)
            name = path.relative_to(root).as_posix()
            paths[name] = path
            stamps[name] = stamp(path)
    for record in records:
        for entry in _source_refs(record):
            path = _regular(entry['path'])
            if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in paths:
                raise ValueError('Frozen source dependency is outside the selected archive')
            relative = path.relative_to(root).as_posix()
            if relative in required_hashes:
                if required_hashes[relative] != entry['sha256']:
                    raise ValueError('Conflicting frozen source inventories')
                continue
            if _sha(path) != entry['sha256']:
                raise ValueError('Frozen source dependency bytes changed')
            required_hashes[relative] = entry['sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.model-release-', dir=output.parent))
    writer = _SplitWriter(stage, part_bytes)
    try:
        members = []
        with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for name, path in sorted(paths.items()):
                if stamp(path) != stamps[name]:
                    raise ValueError('Source changed after the archive inventory was captured')
                before = path.stat()
                info = tarfile.TarInfo(name)
                info.size, info.mode = before.st_size, 0o600
                with path.open('rb') as stream:
                    checked = _HashedReader(stream)
                    archive.addfile(info, checked)
                if stamp(path) != stamps[name]:
                    raise ValueError('Source changed during archive creation')
                if name in required_hashes and checked.digest.hexdigest() != required_hashes[name]:
                    raise ValueError('Archived frozen source bytes differ from the receipt')
                members.append(dict(path=name, size=info.size, sha256=checked.digest.hexdigest()))
        writer._close()
        final_paths = [_unlinked(p) for relative in roots
            for p in _under(root, relative).rglob('*')]
        final_names = {p.relative_to(root).as_posix() for p in final_paths if not p.is_dir()}
        if final_names != set(paths) or any(stamp(path) != stamps[name] for name, path in paths.items()):
            raise ValueError('Source inventory changed during archive creation')
        manifest = dict(format=FORMAT, profile=profile, source_commit=commit,
            roots=roots, members=members, parts=writer.parts,
            total_bytes=sum(item['size'] for item in members))
        (stage/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
        verify(stage)
        if output.exists(): raise FileExistsError('Archive output appeared during preparation')
        os.rename(stage, output)
        return manifest
    finally:
        writer._close()
        if stage.exists(): shutil.rmtree(stage)


class _PartsReader(io.RawIOBase):
    def __init__(self, directory, parts):
        self.paths = iter(directory/item['name'] for item in parts)
        self.stream = None
    def readable(self): return True
    def readinto(self, buffer):
        while True:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None: return 0
                self.stream = path.open('rb')
            count = self.stream.readinto(buffer)
            if count: return count
            self.stream.close(); self.stream = None
    def close(self):
        if self.stream is not None: self.stream.close()
        super().close()


def _manifest(directory):
    directory = _unlinked(directory)
    manifest = _json(directory/'manifest.json')
    if (not isinstance(manifest, dict)
            or set(manifest) != {'format', 'profile', 'source_commit', 'roots', 'members', 'parts', 'total_bytes'}
            or manifest['format'] != FORMAT or manifest['profile'] not in ('old', 'new')
            or not isinstance(manifest['source_commit'], str)
            or re.fullmatch(r'[0-9a-f]{40}', manifest['source_commit']) is None
            or not isinstance(manifest['members'], list) or not manifest['members']
            or not isinstance(manifest['parts'], list) or not manifest['parts']):
        raise ValueError('Declared model archive manifest required')
    roots = manifest['roots']
    allowed = {'runs/'+name for name in OLD_RUNS} if manifest['profile'] == 'old' else {'runs/'+NEW_RUN, SOURCE_CE}
    mandatory = 'runs/'+OLD_RUNS[-1] if manifest['profile'] == 'old' else 'runs/'+NEW_RUN
    if (not isinstance(roots, list) or any(not isinstance(name, str) for name in roots)
            or len(set(roots)) != len(roots) or not set(roots) <= allowed
            or mandatory not in roots or manifest['profile'] == 'new' and set(roots) != allowed):
        raise ValueError('Archive roots differ from the declared profile')
    names = set()
    for entry in manifest['members']:
        if (not isinstance(entry, dict) or set(entry) != {'path', 'size', 'sha256'}
                or not any(_safe_name(entry['path']).startswith(root+'/') for root in roots)
                or entry['path'] in names or type(entry['size']) is not int or entry['size'] < 0
                or not DIGEST.fullmatch(str(entry['sha256']))):
            raise ValueError('Invalid archive member inventory')
        names.add(entry['path'])
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise ValueError('Archive file paths have an ancestor conflict')
    for index, part in enumerate(manifest['parts']):
        if (not isinstance(part, dict) or set(part) != {'name', 'size', 'sha256'}
                or part['name'] != f'models.tar.part{index:04d}'
                or type(part['size']) is not int or not 0 < part['size'] <= PART_BYTES
                or not DIGEST.fullmatch(str(part['sha256']))):
            raise ValueError('Invalid split archive inventory')
        path = _regular(directory/part['name'])
        if path.stat().st_size != part['size'] or _sha(path) != part['sha256']:
            raise ValueError('Archive part bytes changed')
        if part['size'] >= 1024**2:
            print('Verified archive part: '+part['name'], file=sys.stderr, flush=True)
    if (set(p.name for p in directory.iterdir()) != {'manifest.json', *(p['name'] for p in manifest['parts'])}
            or type(manifest['total_bytes']) is not int
            or manifest['total_bytes'] != sum(item['size'] for item in manifest['members'])):
        raise ValueError('Archive assets or byte total changed')
    return manifest


def _walk(directory, manifest, consume=None):
    expected = {item['path']:item for item in manifest['members']}
    seen = set()
    with io.BufferedReader(_PartsReader(directory, manifest['parts'])) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for member in archive:
                name = _safe_name(member.name)
                item = expected.get(name)
                if (item is None or name in seen or not member.isreg() or member.sparse
                        or member.size != item['size']):
                    raise ValueError('Tar member differs from the declared regular-file inventory')
                seen.add(name)
                source = archive.extractfile(member)
                with source:
                    checked = _HashedReader(source)
                    if consume is None:
                        while checked.read(1024**2): pass
                    else: consume(item, checked)
                if checked.digest.hexdigest() != item['sha256']:
                    raise ValueError('Archive member bytes changed')
    if seen != set(expected): raise ValueError('Archive members are missing')


def verify(bundle_dir):
    directory = _unlinked(bundle_dir)
    manifest = _manifest(directory)
    _walk(directory, manifest)
    return manifest


verify_bundle = verify


def restore(bundle_dir, destination_repo):
    """Preflight every target; preserve identical files and refuse all conflicts."""
    directory, root = _unlinked(bundle_dir), _unlinked(destination_repo)
    manifest = verify(directory)
    for item in manifest['members']:
        path = _under(root, item['path'])
        if path.exists() and (not path.is_file() or _sha(path) != item['sha256']):
            raise FileExistsError('Conflicting destination is preserved: '+item['path'])
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise FileExistsError('A destination parent is not a directory')
            if parent == root: break
    def write(item, source):
        target = _under(root, item['path'])
        if target.exists():
            if _sha(target) != item['sha256']: raise FileExistsError('Destination changed after preflight')
            while source.read(1024**2): pass
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.restore-', suffix='.tmp', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as output:
                while block := source.read(1024**2): output.write(block)
                output.flush(); os.fsync(output.fileno())
            if _sha(temporary) != item['sha256']: raise ValueError('Restore bytes changed')
            # Atomic no-clobber publication: a concurrently created file is preserved.
            os.link(temporary, target)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    _walk(directory, manifest, write)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--verify', action='store_true')
    action.add_argument('--restore', action='store_true')
    parser.add_argument('--profile', choices=('old', 'new'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bundle-dir', type=Path)
    parser.add_argument('--destination-repo', type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        if not args.profile or args.output_dir is None: parser.error('--prepare requires --profile and --output-dir')
        result = prepare(Path(__file__).resolve().parents[1], args.profile, args.output_dir)
    elif args.restore:
        if args.bundle_dir is None or args.destination_repo is None: parser.error('--restore requires --bundle-dir and --destination-repo')
        result = restore(args.bundle_dir, args.destination_repo)
    else:
        if args.bundle_dir is None: parser.error('--verify requires --bundle-dir')
        result = verify(args.bundle_dir)
    print(json.dumps(dict(verified=True, profile=result['profile'], files=len(result['members']),
        assets=len(result['parts'])+1, bytes=result['total_bytes'], model_loaded=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
