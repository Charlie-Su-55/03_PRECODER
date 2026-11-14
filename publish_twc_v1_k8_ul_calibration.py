#!/usr/bin/env python3
"""Export checked scalar calibration evidence; explicitly publish a result ref.

Calculation never uploads. Git plumbing leaves the current branch, index and
dirty worktree in place. A deleted public ref is not a promise of erasure.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
import re
import subprocess

import publish_twc_v1_k8_ul_propagation as transport

REPOSITORY = transport.REPOSITORY
WEB_REPOSITORY = transport.WEB_REPOSITORY
SCALAR_FILES = ('model.json', 'cal_samples.csv.gz', 'cal_input_fingerprints.json',
    'samples.csv.gz', 'summary.csv', 'comparisons.csv', 'input_fingerprints.json')
PUBLIC_FILES = (*SCALAR_FILES, 'provenance.json', 'README.md')
MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_EXPANDED_BYTES = 16*1024**2, 48*1024**2, 64*1024**2
_sha, _git, _git_state = transport._sha, transport._git, transport._git_state
_push_commit, _git_environment = transport._push_commit, transport._git_environment


def _config():
    from configs import twc_v1_k8_ul_calibration_config as cfg
    return cfg


def _runtime():
    import ul_calibration_runtime as runtime
    return runtime


def _queue():
    import run_twc_v1_k8_ul_calibration as queue
    return queue


def _report(path):
    cfg, runtime = _config(), _runtime()
    path = runtime.inside(path)
    subtree = (cfg.OUTPUT_ROOT/'reports').resolve()
    if (path == subtree or not path.is_relative_to(subtree)
            or not re.fullmatch(r'attempt_[0-9a-f]{12,32}', path.name)):
        raise ValueError('Owned calibration report attempt required')
    return path


def _assert_public_json(value):
    transport._assert_public_json(value)
    if isinstance(value, dict):
        for item in value.values():
            _assert_public_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Finite public JSON numbers required')


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate public JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Finite public JSON constants required')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _check_scalar_file(path):
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > MAX_FILE_BYTES):
        raise ValueError('Bounded regular public scalar artifact required')
    if path.suffix == '.json':
        value = _json(path.read_text())
        _assert_public_json(value)
        if path.name == 'model.json':
            from twc_v1_k8_ul_calibration import validate_model
            validate_model(value)
        return
    if path.name == 'README.md':
        return
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        raw = stream.read(MAX_EXPANDED_BYTES+1)
    if len(raw) > MAX_EXPANDED_BYTES:
        raise ValueError('Expanded scalar artifact exceeds size budget')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
    if (not reader.fieldnames or len(reader.fieldnames) > 2048
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or any(not field for field in reader.fieldnames)):
        raise ValueError('Bounded scalar CSV schema required')
    text_columns = {'arm', 'variant', 'role', 'scope', 'comparison', 'metric',
                    'component', 'contrast', 'method', 'positive', 'negative'}
    count = 0
    for row in reader:
        if None in row:
            raise ValueError('Malformed scalar CSV')
        _assert_public_json(row)
        for field, value in row.items():
            if field in text_columns:
                if value and not re.fullmatch(r'[A-Za-z0-9_.:+ -]{1,256}', value):
                    raise ValueError('Controlled scalar identifiers required')
            elif value:
                try:
                    valid = math.isfinite(float(value))
                except (ValueError, TypeError):
                    valid = False
                if not valid:
                    raise ValueError('Finite scalar CSV numbers required')
        count += 1
        if count > 100000:
            raise ValueError('Scalar CSV row limit exceeded')
    if path.name == 'samples.csv.gz':
        import twc_v1_k8_ul_calibration_results as results
        cfg = _config()
        results.validate_rows(results.read_raw_csv(path), samples=cfg.VALIDATION_SAMPLES,
            batch_size=cfg.MICRO_BATCH)


def _public_provenance(summary_path, calibration, evaluation, cal, final):
    cfg, runtime, queue = _config(), _runtime(), _queue()
    summary = runtime.read_json(summary_path)
    queue.verify_preregistration(summary, summary_path)
    if (summary.get('namespace') != cfg.NAMESPACE
            or summary.get('core') != cal.get('core') or cal.get('core') != final.get('core')
            or summary.get('recipe_fingerprint') != final.get('recipe_fingerprint')
            or cal.get('recipe_fingerprint') != final.get('recipe_fingerprint')):
        raise ValueError('Calibration/validation preregistered identity changed')
    source_keys = ('kind', 'arm', 'model_commit', 'checkpoint_sha256', 'core', 'step',
                   'state_sha256', 'CE_state_sha256', 'statistics_state_sha256')
    records, signatures = final.get('source_records', {}), final.get('frozen_signatures', {})
    if set(records) != set(cfg.ARMS) or set(signatures) != set(cfg.ARMS):
        raise ValueError('Both frozen validation source records required')
    common = cal.get('common_CE_signature')
    if not isinstance(common, str) or not re.fullmatch(r'[0-9a-f]{64}', common):
        raise ValueError('Common CE calibration identity required')
    for arm, state in signatures.items():
        if (state.get('before') != state.get('after')
                or state.get('before', {}).get('CE') != common):
            raise ValueError('Frozen source/CE identity changed')
        binding = {'codec': 'state_sha256', 'CE': 'CE_state_sha256',
                   'statistics': 'statistics_state_sha256'}
        if any(state['before'].get(k) != records[arm].get(v) for k, v in binding.items()):
            raise ValueError('Frozen signatures no longer bind their source records')
    calibration_states = cal.get('frozen_signatures', {})
    if (set(calibration_states) != {'fdma'}
            or calibration_states['fdma'].get('before') != calibration_states['fdma'].get('after')
            or calibration_states['fdma'].get('before', {}).get('CE') != common):
        raise ValueError('Common calibration CE was not frozen')
    if (final.get('model_sha256') != runtime.sha256(calibration/'model.json')
            or final.get('calibration_manifest_sha256') != runtime.sha256(calibration/'manifest.json')):
        raise ValueError('Validation no longer binds the locked calibration')
    public = dict(format=cfg.NAMESPACE+'_public_provenance_v1', namespace=cfg.NAMESPACE,
        diagnostic_code_commit=summary.get('code', {}).get('commit'), core=final['core'],
        recipe_fingerprint=final['recipe_fingerprint'],
        calibration_manifest_sha256=runtime.sha256(calibration/'manifest.json'),
        source_manifest_sha256=runtime.sha256(evaluation/'manifest.json'),
        preregistration_sha256=summary['preregistration_sha256'],
        data_protocol=final['data_protocol'], arms=list(cfg.ARMS), variants=list(cfg.VARIANTS),
        gamma_db=list(cfg.GAMMAS), batch_size=cfg.MICRO_BATCH,
        calibration_batches=cfg.CALIBRATION_BATCHES, validation_batches=cfg.VALIDATION_BATCHES,
        calibration_samples=cfg.CALIBRATION_SAMPLES, validation_samples=cfg.VALIDATION_SAMPLES,
        calibration_role=cfg.CHANNEL_ROLES['calibration'], validation_role=cfg.CHANNEL_ROLES['validation'],
        model_sha256=final['model_sha256'], common_CE_signature=common,
        optimizer_updates=0, UE_updates=0, CE_updates=0, BS_updates=0, EMA_updates=0,
        calibration_is_statistical_fitting=True, true_UL_validation_scoring_only=True,
        calibration_locked_before_validation=True, test_used=False,
        source_records={arm: {k: row[k] for k in source_keys if k in row} for arm, row in records.items()},
        calibration_frozen_signatures=cal['frozen_signatures'], frozen_signatures=signatures,
        physical=cfg.recipe()['physical'], public_repository=True,
        contains_model_weights=False, contains_neural_weights=False,
        contains_calibration_coefficients=True, contains_channel_tensors=False,
        deletion_is_not_guaranteed_history_erasure=True)
    _assert_public_json(public)
    return public


def export_directory(summary_path, calibration_dir, evaluation_dir, output_dir):
    """Immutable checked CSV/JSON handoff; no archive, checkout or upload."""
    runtime, cfg, queue = _runtime(), _config(), _queue()
    calibration, evaluation = runtime.inside(calibration_dir), runtime.inside(evaluation_dir)
    cal = queue.verify_calibration(calibration)
    final = queue.verify_phase_binding(calibration, evaluation)
    public = _public_provenance(summary_path, calibration, evaluation, cal, final)
    sources = {'model.json': calibration/'model.json',
        'cal_samples.csv.gz': calibration/'cal_samples.csv.gz',
        'cal_input_fingerprints.json': calibration/'input_fingerprints.json',
        **{name: evaluation/name for name in ('samples.csv.gz', 'summary.csv',
            'comparisons.csv', 'input_fingerprints.json')}}
    for source in sources.values():
        _check_scalar_file(source)
    output = _report(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, source in sources.items():
        with (output/name).open('xb') as stream:
            stream.write(source.read_bytes())
        hashes[name] = runtime.sha256(output/name)
    runtime.atomic_json(output/'provenance.json', public, exclusive=True)
    hashes['provenance.json'] = runtime.sha256(output/'provenance.json')
    text = ('# Frozen UL independent calibration and validation\n\n'
        'Public scalar observations, a small statistical calibration model and '
        'redacted provenance. No neural weights, channel tensors, credentials or machine paths.\n\n'
        'Calibration and validation roles are independent. Calibration uses truth labels; '
        'deployable receiver inputs use estimated channels and known pilot observations only. '
        'All neural/optimizer/EMA updates are zero; statistical moment fitting is learning. '
        'Validation retains all physical arms, receiver controls and gamma points. '
        'Intervals condition on one frozen training seed and a fixed fitted calibrator.\n\n'
        'Deleting this ref does not guarantee removal from history or caches. '
        'Original local experiment evidence remains.\n')
    (output/'README.md').write_text(text, encoding='utf-8')
    hashes['README.md'] = runtime.sha256(output/'README.md')
    export = dict(format=cfg.NAMESPACE+'_export_v1', export_id=output.name[8:],
        namespace=cfg.NAMESPACE, files=hashes, contains_model_weights=False,
        contains_neural_weights=False, contains_calibration_coefficients=True,
        contains_channel_tensors=False, public_repository=True,
        source_manifest_sha256=runtime.sha256(evaluation/'manifest.json'),
        calibration_manifest_sha256=runtime.sha256(calibration/'manifest.json'))
    runtime.atomic_json(output/'export_manifest.json', export, exclusive=True)
    verify_export_binding(summary_path, calibration, evaluation, output)
    return dict(export, path=str(output), manifest=str(output/'export_manifest.json'),
        manifest_sha256=runtime.sha256(output/'export_manifest.json'))


def verify_export_directory(path):
    path, cfg = _report(path), _config()
    manifest_path = path/'export_manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Regular public export manifest required')
    m = _json(manifest_path.read_text())
    expected = {'format', 'export_id', 'namespace', 'files', 'contains_model_weights',
        'contains_neural_weights', 'contains_calibration_coefficients',
        'contains_channel_tensors', 'public_repository', 'source_manifest_sha256',
        'calibration_manifest_sha256'}
    if (set(m) != expected or m.get('format') != cfg.NAMESPACE+'_export_v1'
            or m.get('namespace') != cfg.NAMESPACE or m.get('export_id') != path.name[8:]
            or m.get('contains_model_weights') is not False or m.get('contains_channel_tensors') is not False
            or m.get('contains_neural_weights') is not False or m.get('contains_calibration_coefficients') is not True
            or m.get('public_repository') is not True or set(m.get('files', {})) != set(PUBLIC_FILES)
            or any(not re.fullmatch(r'[0-9a-f]{64}', m[k]) for k in
                ('source_manifest_sha256', 'calibration_manifest_sha256'))):
        raise ValueError('Exact public export identity and inventory required')
    if {p.name for p in path.iterdir()} != set(PUBLIC_FILES)|{'export_manifest.json'}:
        raise ValueError('Unexpected public result artifact')
    total = 0
    for name, digest in m['files'].items():
        file = path/name
        _check_scalar_file(file)
        total += file.stat().st_size
        if _sha(file) != digest:
            raise ValueError('Public result bytes changed: '+name)
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Public scalar result size budget exceeded')
    return m


def verify_export_binding(summary_path, calibration_dir, evaluation_dir, export_dir):
    """Bind a self-consistent export to this exact preregistered local batch."""
    runtime, queue = _runtime(), _queue()
    calibration, evaluation = runtime.inside(calibration_dir), runtime.inside(evaluation_dir)
    output = _report(export_dir)
    cal = queue.verify_calibration(calibration)
    final = queue.verify_phase_binding(calibration, evaluation)
    manifest = verify_export_directory(output)
    expected = _public_provenance(summary_path, calibration, evaluation, cal, final)
    if (runtime.read_json(output/'provenance.json') != expected
            or manifest['calibration_manifest_sha256'] != runtime.sha256(calibration/'manifest.json')
            or manifest['source_manifest_sha256'] != runtime.sha256(evaluation/'manifest.json')):
        raise ValueError('Public export provenance no longer binds this batch')
    sources = {'model.json': calibration/'model.json',
        'cal_samples.csv.gz': calibration/'cal_samples.csv.gz',
        'cal_input_fingerprints.json': calibration/'input_fingerprints.json',
        **{name: evaluation/name for name in ('samples.csv.gz', 'summary.csv',
            'comparisons.csv', 'input_fingerprints.json')}}
    if any(manifest['files'][name] != runtime.sha256(source) for name, source in sources.items()):
        raise ValueError('Public export payload no longer binds this batch')
    return manifest


def _branch(export_id):
    if not re.fullmatch(r'[0-9a-f]{12,32}', export_id):
        raise ValueError('Exact calibration export ID required')
    return 'results/'+_config().NAMESPACE+'/'+export_id


def _orphan_commit(root, files, *, commit=None):
    if commit is not None:
        return transport._orphan_commit(root, files, commit=commit)
    records = []
    for name, path in sorted(files.items()):
        if Path(name).name != name or any(c in name for c in '\n\t'):
            raise ValueError('Flat controlled public filenames required')
        blob = _git(root, ['hash-object', '-w', '--stdin'], data=Path(path).read_bytes())
        records.append(f'100644 blob {blob}\t{name}\n')
    tree = _git(root, ['mktree'], data=''.join(records).encode())
    commit = _git(root, ['-c', 'user.name=03 calibration exporter',
        '-c', 'user.email=diagnostics@localhost', '-c', 'commit.gpgsign=false',
        'commit-tree', tree], data=b'Public independent UL statistical calibration and validation\n')
    return commit, tree


def _completed_export(path=None):
    queue, runtime = _queue(), _runtime()
    summary, summary_path = queue.load_summary()
    runtime.guard_core(summary['core'])
    queue.verify_preregistration(summary, summary_path)
    required = set(queue.stage_names())
    if (summary.get('status') != 'completed' or not required <= set(summary['stages'])
            or set(summary['stages'])-required-{'startup_policy'}):
        raise ValueError('A completed calibration/validation batch is required')
    for record in summary['stages'].values():
        if record.get('status') != 'completed':
            raise ValueError('Incomplete calibration/validation stage')
        queue.verify_evidence(record)
    cal = summary['stages']['calibration']['details']['output_dir']
    final = summary['stages']['locked_validation']['details']['output_dir']
    queue.verify_phase_binding(cal, final)
    selected = _report(summary['stages']['directory_export']['details']['path'])
    if path is not None and _report(path) != selected:
        raise ValueError('Only the completed batch export may be published')
    verify_export_binding(summary_path, cal, final, selected)
    return selected


def publish(path=None):
    cfg, runtime = _config(), _runtime()
    output = _completed_export(path)
    manifest = verify_export_directory(output)
    if _git(cfg.ROOT, ['remote', 'get-url', 'origin']) != REPOSITORY:
        raise ValueError('Official 03 origin required for publication')
    branch = _branch(manifest['export_id'])
    receipt_path = runtime.inside(cfg.OUTPUT_ROOT/'uploads'/(manifest['export_id']+'.json'))
    with runtime.process_lock(cfg.OUTPUT_ROOT/'publication.lock'):
        before = _git_state(cfg.ROOT)
        digest = runtime.sha256(output/'export_manifest.json')
        old = runtime.read_json(receipt_path) if receipt_path.exists() else None
        if old and (old.get('branch') != branch or old.get('export_manifest_sha256') != digest):
            raise ValueError('Saved publication identity changed')
        files = {name: output/name for name in (*PUBLIC_FILES, 'export_manifest.json')}
        commit, tree = _orphan_commit(cfg.ROOT, files, commit=old.get('commit') if old else None)
        if _git_state(cfg.ROOT) != before:
            raise RuntimeError('Source HEAD/index changed before publication')
        receipt = dict(format=cfg.NAMESPACE+'_publication_v1', status='prepared',
            branch=branch, commit=commit, tree=tree, export_manifest_sha256=digest,
            export_id=manifest['export_id'], public_repository=True, url=WEB_REPOSITORY+'/tree/'+branch)
        runtime.atomic_json(receipt_path, receipt)
        code = _push_commit(cfg.ROOT, commit, branch)
        receipt.update(status='uploaded' if code == 0 else 'failed', git_exit_code=code)
        runtime.atomic_json(receipt_path, receipt)
        if code:
            raise RuntimeError('Git upload failed; local result export and prepared commit are preserved. '
                'Check the Git error, network and write authentication; then repeat --publish.')
    print('Uploaded branch: '+branch)
    print('Result commit: '+commit)
    print('Read URL: '+receipt['url'])
    return receipt


def delete_remote(branch, expected_commit):
    cfg = _config()
    prefix = 'results/'+cfg.NAMESPACE+'/'
    if (not branch.startswith(prefix) or _branch(branch[len(prefix):]) != branch
            or not re.fullmatch(r'[0-9a-f]{40}', expected_commit)):
        raise ValueError('Exact calibration result ref and expected commit required')
    if _git(cfg.ROOT, ['remote', 'get-url', 'origin']) != REPOSITORY:
        raise ValueError('Official 03 origin required')
    reference = 'refs/heads/'+branch
    if _git(cfg.ROOT, ['ls-remote', '--heads', 'origin', reference]).split() != [expected_commit, reference]:
        raise ValueError('Result ref is missing or changed; nothing deleted')
    result = subprocess.run(['git', 'push', '--force-with-lease='+reference+':'+expected_commit,
        'origin', ':'+reference], cwd=cfg.ROOT, check=False, env=_git_environment())
    if result.returncode:
        raise RuntimeError('Exact result ref deletion failed; local evidence remains')
    print('Deleted result ref only; local evidence remains. History erasure is not guaranteed.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    for name in ('publish', 'verify-only', 'delete-remote'):
        actions.add_argument('--'+name, action='store_true')
    p.add_argument('--export-dir', type=Path)
    p.add_argument('--branch')
    p.add_argument('--expected-commit')
    args = p.parse_args(argv)
    if args.delete_remote:
        if not args.branch or not args.expected_commit or args.export_dir:
            p.error('--delete-remote requires --branch and --expected-commit only')
        delete_remote(args.branch, args.expected_commit)
    elif args.branch or args.expected_commit:
        p.error('Ref arguments require --delete-remote')
    elif args.verify_only:
        print(json.dumps(verify_export_directory(_completed_export(args.export_dir)), indent=2))
    else:
        publish(args.export_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
