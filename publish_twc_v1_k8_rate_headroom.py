#!/usr/bin/env python3
"""Explicit scalar-directory publication without changing HEAD or the index."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import publish_twc_v1_k8_ul_propagation as git_io

NAMESPACE = 'twc_v1_k8_rate_headroom_v1'
ROOT = Path(__file__).resolve().parent
REPOSITORY = git_io.REPOSITORY
PUBLIC_FILES = ('README.md', 'samples.csv', 'summary.csv', 'comparisons.csv',
                'input_fingerprints.json', 'provenance.json')
MAX_FILE_BYTES, MAX_TOTAL_BYTES = 32*1024**2, 48*1024**2
IDENTIFIERS = {'arm', 'method', 'scope', 'metric', 'contrast', 'reference',
               'comparison', 'component', 'condition', 'section'}


def _json(raw):
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite JSON constant: '+value)))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _owned(path):
    path = Path(path).absolute()
    allowed = ROOT/'runs'/NAMESPACE
    if path.resolve() == allowed.resolve() or not path.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('Owned headroom child path required')
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError('Linked result paths are forbidden')
        if ancestor == ROOT:
            break
    return path


def _report(path):
    path = _owned(path)
    parent = ROOT/'runs'/NAMESPACE/'reports'
    if (path.parent.resolve() != parent.resolve()
            or not re.fullmatch(r'attempt_[0-9a-f]{12}', path.name)):
        raise ValueError('Owned headroom report attempt required')
    return path


def _validate_payload(name, raw):
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError('Bounded, nonempty scalar artifact required')
    text = raw.decode('utf-8')
    if name.endswith('.json'):
        git_io._assert_public_json(_json(text))
    elif name.endswith('.csv'):
        reader = csv.DictReader(io.StringIO(text, newline=''))
        if (not reader.fieldnames or len(reader.fieldnames) > 256
                or len(set(reader.fieldnames)) != len(reader.fieldnames)):
            raise ValueError('Bounded unique scalar CSV columns required')
        for index, row in enumerate(reader):
            if index >= 100000 or None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed or oversized scalar CSV')
            git_io._assert_public_json(row)
            for field, value in row.items():
                if field in IDENTIFIERS:
                    if not re.fullmatch(r'[A-Za-z0-9_.:+ -]{1,256}', value):
                        raise ValueError('Controlled scalar identifier required')
                elif value:
                    try:
                        finite = math.isfinite(float(value))
                    except ValueError:
                        finite = False
                    if not finite:
                        raise ValueError('Only finite numeric CSV values may be published')
    elif name == 'README.md':
        git_io._assert_public_json(text)
    else:
        raise ValueError('Unexpected public artifact')


def _provenance(raw):
    record = _json(raw)
    if (not isinstance(record, dict) or record.get('namespace') != NAMESPACE
            or record.get('production_evidence') is not True
            or any(type(record.get(name)) is not int or record[name] != 0 for name in
                   ('optimizer_updates', 'UE_updates', 'BS_updates', 'CE_updates', 'EMA_updates'))
            or any(record.get(name) is not False for name in
                   ('test_used', 'contains_model_weights', 'contains_channel_tensors'))
            or not re.fullmatch(r'[0-9a-f]{64}', record.get('source_core_sha256', ''))
            or not re.fullmatch(r'[0-9a-f]{64}', record.get('preregistration_sha256', ''))
            or set(record.get('source_records', {})) != {'fdma', 'hybrid_g4'}
            or set(record.get('frozen_signatures', {})) != {'fdma', 'hybrid_g4'}):
        raise ValueError('Frozen zero-update source provenance required')
    for value in record['frozen_signatures'].values():
        if (not isinstance(value, dict) or set(value) != {'before', 'after', 'states'}
                or not value['before'] or value['before'] != value['after']):
            raise ValueError('Frozen model signatures changed')
        states = value['states']
        if (not isinstance(states, dict) or set(states) !=
                {'codec', 'encoder', 'decoder', 'statistics', 'CE'}
                or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
                       for v in states.values())):
            raise ValueError('Exact frozen state hashes required')
        expected = hashlib.sha256(json.dumps(states, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if value['before'] != expected:
            raise ValueError('Frozen signature does not bind its state hashes')
    for source in record['source_records'].values():
        if (not isinstance(source, dict) or source.get('production_evidence') is not True
                or source.get('step') != 10000):
            raise ValueError('Completed production joint10000 source required')
    return record


def create_export(directory, payloads):
    """Create an exclusive report; its verified manifest is written last.

    A failed creation leaves an incomplete directory which cannot be published
    or silently replaced. No source files, prior results or network are touched.
    """
    path = _report(directory)
    if set(payloads) != set(PUBLIC_FILES):
        raise ValueError('Exact public scalar inventory required')
    if sum(map(len, payloads.values())) > MAX_TOTAL_BYTES:
        raise ValueError('Scalar report exceeds size budget')
    for name, raw in payloads.items():
        _validate_payload(name, raw)
    provenance = _provenance(payloads['provenance.json'])
    manifest = dict(format=NAMESPACE+'_export_v1', namespace=NAMESPACE,
        export_id=path.name[8:], files={name: hashlib.sha256(raw).hexdigest()
            for name, raw in payloads.items()},
        source_core_sha256=provenance['source_core_sha256'],
        preregistration_sha256=provenance['preregistration_sha256'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    for name, raw in payloads.items():
        with (path/name).open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    with (path/'export_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return verify_export(path)


def verify_export(directory):
    path = _report(directory)
    if {item.name for item in path.iterdir()} != {*PUBLIC_FILES, 'export_manifest.json'}:
        raise ValueError('Unexpected or incomplete scalar report inventory')
    for name in (*PUBLIC_FILES, 'export_manifest.json'):
        target = path/name
        if target.is_symlink() or not target.is_file() or target.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Bounded regular public files required')
    manifest = _json((path/'export_manifest.json').read_bytes())
    if (set(manifest) != {'format', 'namespace', 'export_id', 'files',
                         'source_core_sha256', 'preregistration_sha256'}
            or manifest.get('format') != NAMESPACE+'_export_v1'
            or manifest.get('namespace') != NAMESPACE
            or manifest.get('export_id') != path.name[8:]
            or set(manifest.get('files', {})) != set(PUBLIC_FILES)):
        raise ValueError('Exact headroom export identity required')
    if sum((path/name).stat().st_size for name in PUBLIC_FILES) > MAX_TOTAL_BYTES:
        raise ValueError('Scalar report exceeds size budget')
    for name in PUBLIC_FILES:
        raw = (path/name).read_bytes()
        _validate_payload(name, raw)
        if hashlib.sha256(raw).hexdigest() != manifest['files'][name]:
            raise ValueError('Public report bytes changed: '+name)
    provenance = _provenance((path/'provenance.json').read_bytes())
    if any(manifest[key] != provenance[key] for key in
           ('source_core_sha256', 'preregistration_sha256')):
        raise ValueError('Export/source identity mismatch')
    return manifest


def _atomic_record(path, value):
    path = _owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='headroom_publication_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _completed(directory):
    import diagnose_twc_v1_k8_rate_headroom as diagnostic
    diagnostic.verify_completed_export(directory)


def _origin():
    if git_io._git(ROOT, ['remote', 'get-url', 'origin']) != REPOSITORY:
        raise ValueError('Official 03 origin required')


def publish(directory):
    import fcntl
    path = _report(directory)
    manifest = verify_export(path)
    _completed(path)
    _origin()
    branch = 'results/'+NAMESPACE+'/'+manifest['export_id']
    uploads = _owned(ROOT/'runs'/NAMESPACE/'uploads')
    uploads.mkdir(parents=True, exist_ok=True)
    with _owned(uploads/'publication.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = git_io._git_state(ROOT)
        receipt_path = _owned(uploads/(manifest['export_id']+'.json'))
        old = _json(receipt_path.read_bytes()) if receipt_path.exists() else None
        digest = _sha(path/'export_manifest.json')
        if old and (old['branch'] != branch or old['export_manifest_sha256'] != digest):
            raise ValueError('Saved publication identity changed')
        files = {name: path/name for name in (*PUBLIC_FILES, 'export_manifest.json')}
        commit, tree = git_io._orphan_commit(ROOT, files, commit=old['commit'] if old else None)
        receipt = dict(status='prepared', branch=branch, commit=commit, tree=tree,
            export_manifest_sha256=digest,
            url='https://github.com/Charlie-Su-55/03_PRECODER/tree/'+branch)
        _atomic_record(receipt_path, receipt)
        ref = 'refs/heads/'+branch
        remote = git_io._git(ROOT, ['ls-remote', '--heads', 'origin', ref])
        if remote and remote.split() != [commit, ref]:
            raise ValueError('Existing result ref differs; no overwrite permitted')
        code = 0
        if not remote:
            result = subprocess.run(['git', 'push', '--force-with-lease='+ref+':',
                'origin', commit+':'+ref], cwd=ROOT, env=git_io._git_environment(), check=False)
            code = result.returncode
        if git_io._git_state(ROOT) != state:
            raise RuntimeError('Source HEAD/index changed during publication')
        receipt.update(status='uploaded' if code == 0 else 'failed', git_exit_code=code)
        _atomic_record(receipt_path, receipt)
        if code:
            raise RuntimeError('Result upload failed; local scalar report and prepared commit are preserved')
    print('Uploaded branch: '+branch+'\nResult commit: '+commit+'\nRead URL: '+receipt['url'])
    print('Deleting the ref later does not guarantee Git-history erasure.')
    return receipt


def delete_remote(branch, expected_commit):
    if (not re.fullmatch('results/'+NAMESPACE+r'/[0-9a-f]{12}', branch)
            or not re.fullmatch(r'[0-9a-f]{40}', expected_commit)):
        raise ValueError('Exact owned result ref and expected commit required')
    _origin()
    before = git_io._git_state(ROOT)
    ref = 'refs/heads/'+branch
    remote = git_io._git(ROOT, ['ls-remote', '--heads', 'origin', ref])
    if remote.split() != [expected_commit, ref]:
        raise ValueError('Result ref changed or absent; nothing deleted')
    result = subprocess.run(['git', 'push', '--force-with-lease='+ref+':'+expected_commit,
        'origin', ':'+ref], cwd=ROOT, env=git_io._git_environment(), check=False)
    if git_io._git_state(ROOT) != before:
        raise RuntimeError('Source HEAD/index changed during deletion')
    if result.returncode:
        raise RuntimeError('Exact result-ref deletion failed; local evidence preserved')
    print('Deleted exact result ref; local evidence remains. History erasure is not guaranteed.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-only', action='store_true')
    action.add_argument('--publish', action='store_true')
    action.add_argument('--delete-remote', action='store_true')
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--branch')
    parser.add_argument('--expected-commit')
    args = parser.parse_args(argv)
    if args.delete_remote:
        if args.export_dir or not args.branch or not args.expected_commit:
            parser.error('--delete-remote requires --branch and --expected-commit only')
        delete_remote(args.branch, args.expected_commit)
    elif not args.export_dir or args.branch or args.expected_commit:
        parser.error('--verify-only/--publish requires --export-dir only')
    elif args.publish:
        publish(args.export_dir)
    else:
        manifest = verify_export(args.export_dir)
        _completed(args.export_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
