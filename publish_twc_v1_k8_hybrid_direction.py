#!/usr/bin/env python3
"""Explicit scalar-directory publication without changing HEAD or the index."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

import publish_twc_v1_k8_ul_propagation as git_io

NAMESPACE = 'twc_v1_k8_hybrid_direction_v1'
ROOT = Path(__file__).resolve().parent
REPOSITORY = git_io.REPOSITORY
PUBLIC_FILES = ('README.md', 'samples.csv.gz', 'summary.csv', 'comparisons.csv',
                'input_fingerprints.json', 'provenance.json')
MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_EXPANDED_BYTES = 32*1024**2, 48*1024**2, 64*1024**2
IDENTIFIERS = {'arm', 'method', 'scope', 'metric', 'contrast', 'reference',
               'comparison', 'component', 'condition', 'section', 'variant', 'role',
               'positive', 'negative'}
README_TEXT = ('# Hybrid-only direction-assisted rate extension\n\n'
    'Public paired complete-MU scalar observations, tensor fingerprints and source provenance. '
    'No neural weights, channel tensors, credentials or machine paths.\n\n'
    'Only Hybrid receives 10000 new UE+BS updates. The old joint10000 FDMA and '
    'Hybrid source are frozen references. Cumulative training budgets are unequal. '
    'Extra updates and a fixed direction auxiliary loss both change; this study '
    'does not isolate the auxiliary loss effect. CE and the CE-error table remain '
    'frozen; Hybrid retains a causal historical-power EMA. All six gamma nodes '
    'and negative results are preserved; test remains sealed. Intervals describe '
    'paired validation sampling conditional on fixed models and one source seed.\n\n'
    'Removing a public result ref does not guarantee Git-history or cache erasure. '
    'Original local evidence stays on the server.\n')


def _config():
    from configs import twc_v1_k8_hybrid_direction_config as cfg
    return cfg


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate public JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite JSON constant: '+value)))


def _public_json(value):
    git_io._assert_public_json(value)
    if isinstance(value, dict):
        for item in value.values(): _public_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _public_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Finite public JSON values required')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _owned(path):
    path = Path(path).absolute()
    allowed = ROOT/'runs'/NAMESPACE
    if path.resolve() == allowed.resolve() or not path.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('Owned hybrid-direction child path required')
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
        raise ValueError('Owned hybrid-direction report attempt required')
    return path


def _validate_payload(name, raw):
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError('Bounded, nonempty scalar artifact required')
    if name.endswith('.gz'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            expanded = stream.read(MAX_EXPANDED_BYTES+1)
        if len(expanded) > MAX_EXPANDED_BYTES:
            raise ValueError('Expanded scalar artifact exceeds size budget')
        text = expanded.decode('utf-8')
    else:
        text = raw.decode('utf-8')
    if name.endswith('.json'):
        _public_json(_json(text))
    elif name.endswith(('.csv', '.csv.gz')):
        reader = csv.DictReader(io.StringIO(text, newline=''))
        if (not reader.fieldnames or len(reader.fieldnames) > 2048
                or len(set(reader.fieldnames)) != len(reader.fieldnames)):
            raise ValueError('Bounded unique scalar CSV columns required')
        for index, row in enumerate(reader):
            if index >= 100000 or None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed or oversized scalar CSV')
            _public_json(row)
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
        if text != README_TEXT: raise ValueError('Fixed public readme required')
    else:
        raise ValueError('Unexpected public artifact')


def _provenance(raw):
    record = _json(raw)
    variants = {'fdma_frozen', 'hybrid_source', 'hybrid_direction'}
    if (not isinstance(record, dict) or record.get('namespace') != NAMESPACE
            or record.get('production_evidence') is not True
            or set(record.get('evaluation_updates', {})) != {'optimizer', 'UE', 'BS', 'CE', 'EMA'}
            or any(type(value) is not int or value != 0
                   for value in record['evaluation_updates'].values())
            or type(record.get('FDMA_new_training_updates')) is not int
            or record['FDMA_new_training_updates'] != 0
            or type(record.get('Hybrid_new_training_updates')) is not int
            or record['Hybrid_new_training_updates'] != 10000
            or any(record.get(name) is not False for name in
                   ('test_used', 'contains_model_weights', 'contains_channel_tensors',
                    'budgets_equal', 'single_factor_loss_attribution'))
            or any(not re.fullmatch(r'[0-9a-f]{64}', record.get(name, '')) for name in
                   ('source_core_sha256', 'preregistration_sha256',
                    'evaluation_manifest_sha256', 'recipe_fingerprint'))
            or not re.fullmatch(r'[0-9a-f]{40}', record.get('experiment_code_commit', ''))
            or set(record.get('source_records', {})) != variants
            or set(record.get('frozen_signatures', {})) != variants):
        raise ValueError('Hybrid-only training and frozen three-view evaluation provenance required')
    for value in record['frozen_signatures'].values():
        if (not isinstance(value, dict) or set(value) != {'before', 'after'}
                or not isinstance(value['before'], dict)
                or value['before'] != value['after']
                or set(value['before']) != {'codec', 'encoder', 'decoder', 'statistics', 'CE', 'CE_table'}
                or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
                       for key, v in value['before'].items() if key != 'CE_table')):
            raise ValueError('Frozen evaluation state signatures changed')
        table = value['before']['CE_table']
        if (not isinstance(table, dict) or set(table) != {'ce_errors', 'ce_calibrated'}
                or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
                       for v in table.values())):
            raise ValueError('Exact frozen CE-table hashes required')
    return record


def _public_record(record, depth=0):
    """Select audit scalars explicitly; never copy source file paths."""
    allowed = {'arm', 'method', 'kind', 'step', 'status', 'source_namespace',
        'source_kind', 'production_evidence',
        'evidence_kind', 'checkpoint_sha256', 'sha256', 'manifest_sha256',
        'terminal_receipt_sha256', 'initialization_sha256', 'progress_sha256',
        'encoder_state_sha256', 'decoder_state_sha256', 'state_sha256',
        'model_state_sha256', 'statistics_state_sha256', 'CE_state_sha256',
        'CE_table_state_sha256', 'base_ema_updates', 'total_ema_updates',
        'additional_EMA_commits', 'base_encoder_state_sha256',
        'base_decoder_state_sha256', 'base_statistics_state_sha256',
        'core', 'counters', 'training_trace', 'training_code', 'fingerprint', 'recipe_fingerprint'}
    if not isinstance(record, dict): raise ValueError('Explicit source record required')
    selected = {key: value for key, value in record.items() if key in allowed}
    if depth < 2:
        for name in ('parent_record', 'source_identity'):
            if isinstance(record.get(name), dict):
                selected[name] = _public_record(record[name], depth+1)
    _public_json(selected)
    return selected


def _completed_report():
    path = ROOT/'runs'/NAMESPACE/'summary.json'
    if path.is_symlink() or not path.is_file():
        raise ValueError('Completed local Hybrid extension summary required')
    summary = _json(path.read_bytes())
    if summary.get('namespace') != NAMESPACE or summary.get('status') != 'completed':
        raise ValueError('Completed local Hybrid extension required before publication')
    return _report(summary['report_directory'])


def export_directory(evaluation_dir, summary_path):
    """Build a checked scalar report without any network request."""
    import evaluate_twc_v1_k8_hybrid_direction as evaluator
    cfg = _config()
    evaluation = _owned(evaluation_dir)
    summary_path = _owned(summary_path)
    summary = _json(summary_path.read_bytes())
    prereg = summary_path.parent/'preregistration.json'
    if (prereg.is_symlink() or not prereg.is_file()
            or summary.get('namespace') != NAMESPACE
            or summary.get('preregistration_sha256') != _sha(prereg)):
        raise ValueError('Local preregistration binding changed')
    manifest = evaluator.verify_manifest(evaluation)
    if (manifest.get('status') != 'passed' or manifest.get('production_evidence') is not True
            or manifest.get('core') != summary.get('core')
            or manifest.get('recipe_fingerprint') != cfg.fingerprint()
            or summary.get('recipe') != cfg.recipe()
            or summary.get('recipe_fingerprint') != cfg.fingerprint()):
        raise ValueError('Completed production evaluation/recipe required')
    from diagnose_twc_v1_k8_rate_headroom import SOURCE_CORE_SHA256, SOURCE_PINS
    provenance = dict(format=NAMESPACE+'_provenance_v1', namespace=NAMESPACE,
        production_evidence=True, core=manifest['core'], recipe=cfg.recipe(),
        recipe_fingerprint=cfg.fingerprint(),
        experiment_code_commit=summary['code']['commit'],
        source_core_sha256=SOURCE_CORE_SHA256, parent_source_pins=SOURCE_PINS,
        source_records={key: _public_record(value) for key, value in manifest['source_records'].items()},
        frozen_signatures=manifest['frozen_signatures'],
        data_protocol=manifest['data_protocol'],
        evaluation_manifest_sha256=_sha(evaluation/'manifest.json'),
        evaluation_files=manifest['files'],
        preregistration_sha256=_sha(prereg),
        evaluation_updates=dict(optimizer=0, UE=0, BS=0, CE=0, EMA=0),
        FDMA_new_training_updates=0, Hybrid_new_training_updates=cfg.UPDATES,
        budgets_equal=False, single_factor_loss_attribution=False,
        fixed_terminal_selection=True, test_used=False,
        contains_model_weights=False, contains_channel_tensors=False,
        all_models_verified_before_data=manifest['all_models_verified_before_data'],
        paired_inputs_verified=manifest['paired_inputs_verified'],
        samples_per_condition=manifest['samples_per_condition'],
        batch_size=manifest['batch_size'], n_batches=manifest['n_batches'],
        gamma_db=manifest['gamma_db'], variants=manifest['variants'])
    _public_json(provenance)
    payloads = {name: (evaluation/name).read_bytes() for name in
        ('samples.csv.gz', 'summary.csv', 'comparisons.csv', 'input_fingerprints.json')}
    payloads.update({'README.md': README_TEXT.encode(), 'provenance.json': json.dumps(
        provenance, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()})
    report = ROOT/'runs'/NAMESPACE/'reports'/('attempt_'+uuid.uuid4().hex[:12])
    create_export(report, payloads)
    return report


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
        preregistration_sha256=provenance['preregistration_sha256'],
        recipe_fingerprint=provenance['recipe_fingerprint'],
        evaluation_manifest_sha256=provenance['evaluation_manifest_sha256'])
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
    return verify_export_directory(path)


def verify_export_directory(directory):
    path = _report(directory)
    if {item.name for item in path.iterdir()} != {*PUBLIC_FILES, 'export_manifest.json'}:
        raise ValueError('Unexpected or incomplete scalar report inventory')
    for name in (*PUBLIC_FILES, 'export_manifest.json'):
        target = path/name
        if target.is_symlink() or not target.is_file() or target.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Bounded regular public files required')
    manifest = _json((path/'export_manifest.json').read_bytes())
    if (set(manifest) != {'format', 'namespace', 'export_id', 'files',
                         'source_core_sha256', 'preregistration_sha256',
                         'recipe_fingerprint', 'evaluation_manifest_sha256'}
            or manifest.get('format') != NAMESPACE+'_export_v1'
            or manifest.get('namespace') != NAMESPACE
            or manifest.get('export_id') != path.name[8:]
            or set(manifest.get('files', {})) != set(PUBLIC_FILES)):
        raise ValueError('Exact hybrid-direction export identity required')
    if sum((path/name).stat().st_size for name in PUBLIC_FILES) > MAX_TOTAL_BYTES:
        raise ValueError('Scalar report exceeds size budget')
    for name in PUBLIC_FILES:
        raw = (path/name).read_bytes()
        _validate_payload(name, raw)
        if hashlib.sha256(raw).hexdigest() != manifest['files'][name]:
            raise ValueError('Public report bytes changed: '+name)
    provenance = _provenance((path/'provenance.json').read_bytes())
    if any(manifest[key] != provenance[key] for key in
           ('source_core_sha256', 'preregistration_sha256',
            'recipe_fingerprint', 'evaluation_manifest_sha256')):
        raise ValueError('Export/source identity mismatch')
    return manifest


def _atomic_record(path, value):
    path = _owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='hybrid-direction_publication_', suffix='.json', dir=path.parent)
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
    import run_twc_v1_k8_hybrid_direction as runner
    runner.verify_completed_export(directory)


def _origin():
    if git_io._git(ROOT, ['remote', 'get-url', 'origin']) != REPOSITORY:
        raise ValueError('Official 03 origin required')


def publish(directory=None):
    import fcntl
    path = _completed_report() if directory is None else _report(directory)
    manifest = verify_export_directory(path)
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
    elif args.branch or args.expected_commit:
        parser.error('--branch/--expected-commit are only valid with --delete-remote')
    elif args.publish:
        publish(args.export_dir)
    else:
        path = args.export_dir if args.export_dir is not None else _completed_report()
        manifest = verify_export_directory(path)
        _completed(path)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
