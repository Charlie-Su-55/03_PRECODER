#!/usr/bin/env python3
"""Export scalar diagnostics and explicitly publish an isolated result branch.

The calculation never pushes. Publication uses Git plumbing without checking
out a branch, touching the user's index, or adding the working directory.
The repository is public; deleting a result ref is not a promise of erasure.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

REPOSITORY = 'https://github.com/Charlie-Su-55/03_PRECODER.git'
WEB_REPOSITORY = 'https://github.com/Charlie-Su-55/03_PRECODER'
SCALAR_FILES = ('samples.csv.gz', 'summary.csv', 'comparisons.csv',
                'calibration_bins.csv.gz', 'input_fingerprints.json')
PUBLIC_FILES = (*SCALAR_FILES, 'provenance.json', 'README.md')
MAX_FILE_BYTES, MAX_TOTAL_BYTES = 16*1024**2, 32*1024**2


def _config():
    from configs import twc_v1_k8_ul_propagation_config as cfg
    return cfg


def _runtime():
    import ul_propagation_runtime as runtime
    return runtime


def _queue():
    import run_twc_v1_k8_ul_propagation as queue
    return queue


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _report(path):
    cfg, runtime = _config(), _runtime()
    path = runtime.inside(path)
    subtree = (cfg.OUTPUT_ROOT/'reports').resolve()
    if path == subtree or not path.is_relative_to(subtree):
        raise ValueError('Own propagation report child required')
    if not re.fullmatch(r'attempt_[0-9a-f]{12,32}', path.name):
        raise ValueError('Explicit propagation report attempt identifier required')
    return path


def _assert_public_json(value):
    """Reject accidental machine metadata or credentials in selected public JSON."""
    forbidden = {'password', 'token', 'secret', 'credentials', 'hostname', 'username',
                 'environment', 'frozen_files', 'source_root', 'output_root', 'log_path'}
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in forbidden:
                raise ValueError('Nonpublic metadata field in result export')
            _assert_public_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_public_json(item)
    elif isinstance(value, str):
        if (value.startswith(('/', '\\\\')) or re.match(r'^[A-Za-z]:[\\/]', value)
                or 'PRIVATE KEY' in value or any(mark in value for mark in
                    ('/home/', '/workspace/', '/Users/', '://127.0.0.1'))):
            raise ValueError('Machine path, credential or private key in result export')


def _public_provenance(summary_path, manifest_path, manifest):
    cfg, runtime = _config(), _runtime()
    summary = runtime.read_json(summary_path)
    preregistration = Path(summary_path).parent/'preregistration.json'
    if (summary.get('namespace') != cfg.NAMESPACE
            or summary.get('core') != manifest.get('core')
            or summary.get('recipe_fingerprint') != manifest.get('recipe_fingerprint')
            or runtime.sha256(preregistration) != summary.get('preregistration_sha256')):
        raise ValueError('Diagnostic preregistration/source identity changed')
    allowed_source = ('kind', 'arm', 'model_commit', 'checkpoint_sha256', 'core', 'step',
                      'state_sha256', 'CE_state_sha256', 'statistics_state_sha256')
    records = manifest.get('source_records', {})
    if set(records) != set(cfg.ARMS):
        raise ValueError('Both frozen source records required')
    signatures = manifest.get('frozen_signatures', {})
    if set(signatures) != set(cfg.ARMS) or any(
            row.get('before') != row.get('after') for row in signatures.values()):
        raise ValueError('Frozen source signatures do not match')
    public = dict(format=cfg.NAMESPACE+'_public_provenance_v1', namespace=cfg.NAMESPACE,
        diagnostic_code_commit=summary.get('code', {}).get('commit'), core=manifest['core'],
        recipe_fingerprint=manifest['recipe_fingerprint'],
        source_manifest_sha256=runtime.sha256(manifest_path),
        preregistration_sha256=runtime.sha256(preregistration),
        data_protocol=manifest['data_protocol'], arms=list(cfg.ARMS), gamma_db=list(cfg.GAMMAS),
        n_batches=cfg.BATCHES, batch_size=cfg.MICRO_BATCH, samples_per_condition=cfg.SAMPLES,
        optimizer_updates=0, UE_updates=0, CE_updates=0, EMA_updates=0,
        no_downlink_forward=True, original_filter_unchanged=True,
        true_UL_used_for_scoring_only=True, test_used=False,
        source_records={arm: {key: row[key] for key in allowed_source if key in row}
                        for arm, row in records.items()}, frozen_signatures=signatures,
        physical=cfg.recipe()['physical'], contains_model_weights=False,
        contains_channel_tensors=False, public_repository=True,
        deletion_is_not_guaranteed_history_erasure=True)
    _assert_public_json(public)
    return public


def _check_scalar_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Small regular whitelisted scalar artifact required: '+path.name)
    if path.suffix == '.json':
        _assert_public_json(json.loads(path.read_text()))
    elif path.name != 'README.md':
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(reader.fieldnames) > 512:
                raise ValueError('Bounded scalar CSV schema required')
            count = 0
            for row in reader:
                if None in row:
                    raise ValueError('Malformed scalar CSV')
                _assert_public_json(row)
                text_columns = {'arm', 'scope', 'comparison', 'metric', 'component', 'contrast'}
                for field, value in row.items():
                    if field in text_columns:
                        if value and (len(value) > 256 or not re.fullmatch(r'[A-Za-z0-9_.:+ -]+', value)):
                            raise ValueError('Controlled scalar CSV identifier required')
                    elif value:
                        try:
                            finite = math.isfinite(float(value))
                        except (ValueError, TypeError):
                            finite = False
                        if not finite:
                            raise ValueError('Only finite scalar numeric CSV values may be published')
                count += 1
                if count > 100000:
                    raise ValueError('Scalar CSV row limit exceeded')
        if path.name == 'samples.csv.gz':
            import twc_v1_k8_ul_propagation_results as results
            cfg = _config()
            results.validate_rows(results.read_raw_csv(path), samples=cfg.SAMPLES,
                batch_size=cfg.MICRO_BATCH)


def export_directory(summary_path, evaluation_dir, output_dir):
    """Create an immutable public scalar directory, with no archive or network."""
    cfg, runtime = _config(), _runtime()
    evaluation = runtime.inside(evaluation_dir)
    manifest = _queue().verify_manifest(evaluation)
    public = _public_provenance(summary_path, evaluation/'manifest.json', manifest)
    for name in SCALAR_FILES:
        _check_scalar_file(evaluation/name)
    output = _report(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SCALAR_FILES:
        with (output/name).open('xb') as stream:
            stream.write((evaluation/name).read_bytes())
        hashes[name] = runtime.sha256(output/name)
    runtime.atomic_json(output/'provenance.json', public, exclusive=True)
    hashes['provenance.json'] = runtime.sha256(output/'provenance.json')
    text = ('# Frozen UL error-propagation diagnostic\n\n'
        'Public scalar observations and redacted provenance only. No model weights, '
        'channel tensors, credentials or machine-specific paths are included.\n\n'
        'All training/CE/UE/EMA updates are zero. True UL is offline scoring input; '
        'the normal receiver filter is unchanged. This is not a deployable oracle '
        'or a downlink-rate result. Intervals condition on one fixed-model seed.\n\n'
        'Deleting this branch does not guarantee removal from Git history or caches. '
        'Original local evidence remains on the experiment server.\n')
    with (output/'README.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    hashes['README.md'] = runtime.sha256(output/'README.md')
    export = dict(format=cfg.NAMESPACE+'_export_v1', export_id=output.name[8:],
        namespace=cfg.NAMESPACE, files=hashes, contains_model_weights=False,
        contains_channel_tensors=False, public_repository=True,
        source_manifest_sha256=runtime.sha256(evaluation/'manifest.json'))
    runtime.atomic_json(output/'export_manifest.json', export, exclusive=True)
    verify_export_directory(output)
    return dict(export, path=str(output), manifest=str(output/'export_manifest.json'),
        manifest_sha256=runtime.sha256(output/'export_manifest.json'))


def verify_export_directory(path):
    path = _report(path)
    manifest_path = path/'export_manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Regular public export manifest required')
    manifest = _runtime().read_json(manifest_path)
    cfg = _config()
    expected_fields = {'format', 'export_id', 'namespace', 'files', 'contains_model_weights',
        'contains_channel_tensors', 'public_repository', 'source_manifest_sha256'}
    if (set(manifest) != expected_fields or manifest.get('format') != cfg.NAMESPACE+'_export_v1'
            or manifest.get('namespace') != cfg.NAMESPACE
            or manifest.get('export_id') != path.name[8:]
            or manifest.get('contains_model_weights') is not False
            or manifest.get('contains_channel_tensors') is not False
            or manifest.get('public_repository') is not True
            or set(manifest.get('files', {})) != set(PUBLIC_FILES)):
        raise ValueError('Explicit scalar export identity/inventory required')
    if {item.name for item in path.iterdir()} != set(PUBLIC_FILES) | {'export_manifest.json'}:
        raise ValueError('Unexpected file or directory in public result export')
    total = 0
    for name, digest in manifest['files'].items():
        file = path/name
        _check_scalar_file(file)
        total += file.stat().st_size
        if _sha(file) != digest:
            raise ValueError('Public result export bytes changed: '+name)
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Public scalar result export exceeds its size budget')
    return manifest


def _git_environment():
    # Preserve configured CA/proxy/credential helpers, while suppressing traces
    # that could reveal credentials entered at the actual Git terminal prompt.
    return {key: value for key, value in os.environ.items()
            if not key.startswith('GIT_TRACE') and key != 'GIT_CURL_VERBOSE'}


def _git(root, args, *, data=None):
    result = subprocess.run(['git', *args], cwd=root, input=data, capture_output=True,
        check=False, env=_git_environment())
    if result.returncode:
        raise RuntimeError('Local Git plumbing failed: '+args[0])
    return result.stdout.decode().strip()


def _git_state(root):
    head = _git(root, ['rev-parse', 'HEAD'])
    path = Path(_git(root, ['rev-parse', '--git-path', 'index']))
    if not path.is_absolute():
        path = Path(root)/path
    return head, path.read_bytes() if path.exists() else None


def _branch(export_id):
    if not re.fullmatch(r'[0-9a-f]{12,32}', export_id):
        raise ValueError('Exact propagation export identifier required')
    return 'results/'+_config().NAMESPACE+'/'+export_id


def _orphan_commit(root, files, *, commit=None):
    """Real plumbing only; deliberately no index, checkout, add or parent."""
    records = []
    for name, path in sorted(files.items()):
        if Path(name).name != name or any(char in name for char in '\n\t'):
            raise ValueError('Flat controlled public filenames required')
        blob = _git(root, ['hash-object', '-w', '--stdin'], data=Path(path).read_bytes())
        records.append(f'100644 blob {blob}\t{name}\n')
    tree = _git(root, ['mktree'], data=''.join(records).encode())
    if commit is not None:
        if (not re.fullmatch(r'[0-9a-f]{40}', commit)
                or _git(root, ['show', '-s', '--format=%T', commit]) != tree
                or _git(root, ['show', '-s', '--format=%P', commit])):
            raise ValueError('Prepared publication commit changed')
        return commit, tree
    commit = _git(root, ['-c', 'user.name=03 diagnostics exporter',
        '-c', 'user.email=diagnostics@localhost', '-c', 'commit.gpgsign=false',
        'commit-tree', tree], data=b'Public zero-update UL propagation scalar results\n')
    return commit, tree


def _push_commit(root, commit, branch):
    state = _git_state(root)
    result = subprocess.run(['git', 'push', 'origin', commit+':refs/heads/'+branch],
        cwd=root, check=False, env=_git_environment())
    if _git_state(root) != state:
        raise RuntimeError('Source HEAD/index changed during publication; inspect local state')
    return result.returncode


def _completed_export(path=None):
    queue, runtime, cfg = _queue(), _runtime(), _config()
    summary, summary_path = queue.load_summary()
    runtime.guard_core(summary['core'])
    if summary.get('status') != 'completed':
        raise ValueError('A completed propagation batch is required before publication')
    if runtime.sha256(summary_path.parent/'preregistration.json') != summary.get('preregistration_sha256'):
        raise ValueError('Original diagnostic preregistration changed')
    for record in summary['stages'].values():
        if record.get('status') != 'completed':
            raise ValueError('Incomplete local diagnostic stage')
        queue.verify_evidence(record)
    record = summary['stages'].get('directory_export', {}).get('details', {})
    selected = _report(record.get('path', ''))
    if path is not None and _report(path) != selected:
        raise ValueError('Only the completed batch scalar export may be published')
    verify_export_directory(selected)
    provenance = runtime.read_json(selected/'provenance.json')
    if (provenance.get('preregistration_sha256') != summary['preregistration_sha256']
            or provenance.get('core') != summary['core']):
        raise ValueError('Public export is not the completed local diagnostic')
    return selected


def publish(path=None):
    cfg, runtime = _config(), _runtime()
    output = _completed_export(path)
    manifest = verify_export_directory(output)
    if _git(cfg.ROOT, ['remote', 'get-url', 'origin']) != REPOSITORY:
        raise ValueError('Official 03 GitHub origin required for publication')
    branch = _branch(manifest['export_id'])
    receipt_path = runtime.inside(cfg.OUTPUT_ROOT/'uploads'/(manifest['export_id']+'.json'))
    with runtime.process_lock(cfg.OUTPUT_ROOT/'publication.lock'):
        before = _git_state(cfg.ROOT)
        expected_hash = runtime.sha256(output/'export_manifest.json')
        old = runtime.read_json(receipt_path) if receipt_path.exists() else None
        if old and (old.get('branch') != branch or old.get('export_manifest_sha256') != expected_hash):
            raise ValueError('Saved publication identity changed')
        files = {name: output/name for name in (*PUBLIC_FILES, 'export_manifest.json')}
        commit, tree = _orphan_commit(cfg.ROOT, files, commit=old.get('commit') if old else None)
        if _git_state(cfg.ROOT) != before:
            raise RuntimeError('Source HEAD/index changed before publication')
        receipt = dict(format=cfg.NAMESPACE+'_publication_v1', status='prepared',
            branch=branch, commit=commit, tree=tree, export_manifest_sha256=expected_hash,
            export_id=manifest['export_id'], public_repository=True,
            url=WEB_REPOSITORY+'/tree/'+branch)
        runtime.atomic_json(receipt_path, receipt)
        code = _push_commit(cfg.ROOT, commit, branch)
        receipt.update(status='uploaded' if code == 0 else 'failed', git_exit_code=code)
        runtime.atomic_json(receipt_path, receipt)
        if code:
            raise RuntimeError('Git upload failed; the local result export is preserved. '
                'Check the Git error above, proxy/network/CA trust and repository write '
                'authentication, then repeat --publish. '
                'Do not send credentials in chat.')
    print('Uploaded branch: '+branch)
    print('Result commit: '+commit)
    print('Read URL: '+receipt['url'])
    print('Deleting the branch later does not guarantee Git-history erasure.')
    return receipt


def delete_remote(branch, expected_commit):
    cfg = _config()
    prefix = 'results/'+cfg.NAMESPACE+'/'
    if (not branch.startswith(prefix) or _branch(branch[len(prefix):]) != branch
            or not re.fullmatch(r'[0-9a-f]{40}', expected_commit)):
        raise ValueError('Exact diagnostic result branch and expected commit required')
    if _git(cfg.ROOT, ['remote', 'get-url', 'origin']) != REPOSITORY:
        raise ValueError('Official 03 GitHub origin required')
    remote = _git(cfg.ROOT, ['ls-remote', '--heads', 'origin', 'refs/heads/'+branch])
    if remote.split() != [expected_commit, 'refs/heads/'+branch]:
        raise ValueError('Remote result ref is missing or changed; nothing deleted')
    # The expected-value lease closes the race between ls-remote and deletion.
    # This permits only deletion of this exact result ref at this exact commit;
    # it never grants an overwrite of main or another result branch.
    reference = 'refs/heads/'+branch
    result = subprocess.run(['git', 'push', '--force-with-lease='+reference+':'+expected_commit,
        'origin', ':'+reference], cwd=cfg.ROOT, check=False, env=_git_environment())
    if result.returncode:
        raise RuntimeError('Remote result-ref deletion failed; local evidence preserved')
    print('Deleted result ref only. Local evidence remains; Git-history erasure is not guaranteed.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--publish', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    action.add_argument('--delete-remote', action='store_true')
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--branch')
    parser.add_argument('--expected-commit')
    args = parser.parse_args(argv)
    if args.delete_remote:
        if not args.branch or not args.expected_commit or args.export_dir:
            parser.error('--delete-remote requires --branch and --expected-commit only')
        delete_remote(args.branch, args.expected_commit)
    elif args.branch or args.expected_commit:
        parser.error('Branch deletion arguments require --delete-remote')
    elif args.verify_only:
        path = _completed_export(args.export_dir)
        print(json.dumps(verify_export_directory(path), indent=2, sort_keys=True))
    else:
        publish(args.export_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
