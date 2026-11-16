#!/usr/bin/env python3
"""Checked CE scalar-directory export and explicit isolated Git publication."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess

import publish_twc_v1_k8_ul_propagation as git_io

ROOT = Path(__file__).resolve().parent
NAMESPACE = 'twc_v1_k8_ce_refine_v1'
REPOSITORY = 'https://github.com/Charlie-Su-55/03_PRECODER.git'
WEB_REPOSITORY = 'https://github.com/Charlie-Su-55/03_PRECODER'
PUBLIC_FILES = ('README.md', 'samples.csv.gz', 'summary.csv', 'comparisons.csv',
    'input_fingerprints.json', 'provenance.json', 'calibration.json', 'curve_summary.csv')
MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_EXPANDED_BYTES = 32*1024**2, 48*1024**2, 128*1024**2
IDENTIFIERS = {'variant', 'role', 'scope', 'metric', 'comparison', 'positive', 'negative',
    'trajectory', 'condition', 'partition', 'status', 'model', 'section'}
README_TEXT = ('# Independent multi-SNR CE continuation\n\n'
    'Two CE trajectories each receive60000 NEW optimizer updates from the same pinned '
    '5000-update pilot-only U-Net. The NMSE control and geometry-assisted candidate use '
    'matched channels, noise, pilots, SNR schedule and optimizer recipe. UE, BS decoder, '
    'FDMA and historical-power EMA receive zero updates. The original CE and historical '
    'experiments stay unchanged.\n\n'
    'The geometry objective adds spatial projective direction and normalized inter-user '
    'Gram error, with both coefficients fixed at1. The CE already accepted SNR before '
    'this study; continuation is not the first introduction of SNR conditioning.\n\n'
    'Independent calibration precedes frozen final validation at six trained SNR nodes '
    'and five off-node midpoints. Fixed terminal checkpoints are used; learning curves do '
    'not select the best checkpoint. Intervals condition on fixed models and one source '
    'training seed. Analytical shared-recovery scores use a declared uniform prior; '
    'they are not codec sum rates or a demonstrated downlink improvement. Test remains sealed.\n\n'
    'Only scalar observations, CE error calibration statistics, fingerprints and redacted '
    'provenance are included. No model weights, channel tensors, credentials or private '
    'machine paths. Deleting a result branch does not guarantee history or cache erasure.\n')


def _config():
    from configs import twc_v1_k8_ce_refine_config as cfg
    return cfg


def _runtime():
    import ce_refine_runtime as rt
    return rt


def _json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out: raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _public(value):
    git_io._assert_public_json(value)
    if isinstance(value, dict):
        for item in value.values(): _public(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _public(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Finite public values required')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _owned(path):
    path = Path(path).absolute()
    allowed = ROOT/'runs'/NAMESPACE
    if path.resolve() == allowed.resolve() or not path.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('Owned CE-result child path required')
    for item in (path, *path.parents):
        if item.is_symlink(): raise ValueError('Linked result paths are forbidden')
        if item == ROOT: break
    return path


def _report(path):
    path = _owned(path)
    if path.parent != ROOT/'runs'/NAMESPACE/'reports' or not re.fullmatch(r'attempt_[0-9a-f]{12}', path.name):
        raise ValueError('Own CE scalar report attempt required')
    return path


def _validate_payload(name, raw):
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError('Nonempty bounded scalar payload required')
    if name.endswith('.gz'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            expanded = stream.read(MAX_EXPANDED_BYTES+1)
        if len(expanded) > MAX_EXPANDED_BYTES: raise ValueError('Expanded CSV exceeds size budget')
        text = expanded.decode('utf-8')
    else:
        text = raw.decode('utf-8')
    if name == 'README.md':
        if text != README_TEXT: raise ValueError('Controlled CE report README required')
    elif name.endswith('.json'):
        _public(_json(text))
    elif name.endswith(('.csv', '.csv.gz')):
        reader = csv.DictReader(io.StringIO(text, newline=''))
        if (not reader.fieldnames or len(reader.fieldnames) > 512
                or len(set(reader.fieldnames)) != len(reader.fieldnames)):
            raise ValueError('Bounded unique scalar CSV schema required')
        for index, row in enumerate(reader):
            if index >= 100000 or None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed or oversized scalar CSV')
            _public(row)
            for field, value in row.items():
                if field in IDENTIFIERS:
                    if not re.fullmatch(r'[A-Za-z0-9_.:+ -]{1,256}', value):
                        raise ValueError('Controlled CSV identifier required')
                elif value:
                    try: valid = math.isfinite(float(value))
                    except (ValueError, TypeError): valid = False
                    if not valid: raise ValueError('Finite numeric scalar CSV field required')
    else:
        raise ValueError('Only whitelisted scalar formats permitted')


def _csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _compare_table(actual, expected):
    def key(row):
        return tuple(str(row.get(k, '')) for k in ('variant', 'comparison', 'role', 'scope', 'gamma_db', 'metric'))
    def canonical(row):
        return tuple(str(row.get(k, '') if row.get(k) is not None else '')
            for k in ('variant', 'comparison', 'role', 'scope', 'gamma_db', 'metric'))
    observed = {}
    for row in actual:
        k = key(row)
        # CSV encodes e.g. 0.0; canonical numeric gamma avoids text-only differences.
        if row.get('gamma_db'):
            k = (*k[:4], str(float(row['gamma_db'])), k[5])
        if k in observed: raise ValueError('Duplicate aggregated CE statistic')
        observed[k] = row
    if len(observed) != len(expected): raise ValueError('Aggregated CE statistic count changed')
    for row in expected:
        k = canonical(row)
        if row.get('gamma_db') is not None:
            k = (*k[:4], str(float(row['gamma_db'])), k[5])
        supplied = observed.get(k)
        if supplied is None or set(supplied) != set(row):
            raise ValueError('Aggregated CE statistic schema changed')
        for field, target in row.items():
            value = supplied[field]
            if target is None:
                if value != '': raise ValueError('Null CE statistic changed')
            elif isinstance(target, str):
                if value != target: raise ValueError('CE statistic identifier changed')
            elif not math.isclose(float(value), float(target), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('CE statistic does not match paired samples')


def _verify_statistics(path):
    import twc_v1_k8_ce_refine_results as results
    cfg = _config()
    rows = results.read_raw_csv(path/'samples.csv.gz')
    summaries, comparisons = [], []
    for role in ('final', 'midpoint'):
        part = [row for row in rows if row['role'] == role]
        summaries += results.summarize_rows(part, samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
            batch_size=cfg.MICRO_BATCH, role=role)
        comparisons += results.paired_comparisons(part, samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
            batch_size=cfg.MICRO_BATCH, role=role)
    if len(rows) != len(results.DEFAULT_VARIANTS)*len(cfg.EVAL_GAMMAS)*cfg.FINAL_BATCHES*cfg.MICRO_BATCH:
        raise ValueError('Complete final and off-node midpoint CE samples required')
    _compare_table(_csv(path/'summary.csv'), summaries)
    _compare_table(_csv(path/'comparisons.csv'), comparisons)


def _verify_provenance(record):
    cfg = _config()
    required = dict(namespace=NAMESPACE, production_evidence=True, UE_updates=0, BS_updates=0,
        FDMA_updates=0, EMA_updates=0, CE_training_updates=cfg.UPDATES*len(cfg.TRAJECTORIES),
        all_models_verified_before_data=True, paired_inputs_verified=True, test_used=False, fixture=False,
        fixed_terminal_new_updates=cfg.UPDATES, CE_total_updates_per_new_model=cfg.SOURCE_UPDATES+cfg.UPDATES,
        calibration_locked_before_final=True, fingerprint=cfg.fingerprint())
    if any(record.get(k) != v for k, v in required.items()):
        raise ValueError('Completed production CE study provenance required; fixtures cannot publish')
    before, after = record.get('freeze_before'), record.get('freeze_after')
    if not isinstance(before, dict) or set(before) != {'source_ce5000', 'nmse65000', 'geometry65000'} or before != after:
        raise ValueError('All frozen CE signatures must match before and after final evaluation')
    if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) for value in before.values()):
        raise ValueError('Exact frozen CE state hashes required')
    sources = record.get('source_models', {})
    if set(sources) != set(before): raise ValueError('All strict source/terminal CE identities required')
    source = sources['source_ce5000']
    if (source.get('checkpoint_sha256') != cfg.SOURCE_CHECKPOINT_SHA256
            or source.get('source_state_sha256') != cfg.SOURCE_STATE_SHA256
            or source.get('source_commit') != cfg.SOURCE_COMMIT
            or source.get('source_core') != cfg.SOURCE_CORE
            or before['source_ce5000'] != cfg.SOURCE_STATE_SHA256):
        raise ValueError('Original CE5000 provenance changed')
    for trajectory in cfg.TRAJECTORIES:
        variant = trajectory+'65000'; item = sources[variant]
        if (item.get('step') != cfg.UPDATES or item.get('counters') != cfg.counters(cfg.UPDATES)
                or item.get('fingerprint') != cfg.fingerprint(trajectory)
                or item.get('core') != record.get('core') or item.get('production_evidence') is not True
                or item.get('CE_total_updates') != cfg.SOURCE_UPDATES+cfg.UPDATES
                or item.get('source_checkpoint_sha256') != cfg.SOURCE_CHECKPOINT_SHA256
                or item.get('source_state_sha256') != cfg.SOURCE_STATE_SHA256
                or item.get('source_commit') != cfg.SOURCE_COMMIT
                or item.get('model_state_sha256') != before[variant]
                or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('checkpoint_sha256', '')))):
            raise ValueError('CE terminal counters, identity or warm start changed')


def verify_export_directory(path):
    path = _report(path)
    manifest_path = path/'export_manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file(): raise ValueError('Regular scalar export manifest required')
    manifest = _json(manifest_path.read_bytes())
    expected = {'format', 'namespace', 'export_id', 'files', 'source_manifest_sha256',
        'preregistration_sha256', 'contains_model_weights', 'contains_channel_tensors'}
    if (set(manifest) != expected or manifest.get('format') != NAMESPACE+'_export_v1'
            or manifest.get('namespace') != NAMESPACE or manifest.get('export_id') != path.name[8:]
            or manifest.get('contains_model_weights') is not False
            or manifest.get('contains_channel_tensors') is not False
            or set(manifest.get('files', {})) != set(PUBLIC_FILES)
            or {p.name for p in path.iterdir()} != set(PUBLIC_FILES)|{'export_manifest.json'}):
        raise ValueError('Controlled CE export identity and inventory required')
    total = 0
    for name, expected_hash in manifest['files'].items():
        file = path/name
        if file.is_symlink() or not file.is_file(): raise ValueError('Regular scalar files required')
        if file.stat().st_size > MAX_FILE_BYTES: raise ValueError('Scalar file exceeds size budget')
        raw = file.read_bytes(); _validate_payload(name, raw); total += len(raw)
        if _sha(file) != expected_hash: raise ValueError('Scalar report bytes changed: '+name)
    if total > MAX_TOTAL_BYTES: raise ValueError('Scalar report exceeds total size budget')
    provenance = _json((path/'provenance.json').read_bytes())
    _verify_provenance(provenance)
    if provenance.get('preregistration_sha256') != manifest['preregistration_sha256']:
        raise ValueError('Scalar report preregistration binding changed')
    calibration = _json((path/'calibration.json').read_bytes())
    if (calibration.get('namespace') != NAMESPACE or calibration.get('production_evidence') is not True
            or calibration.get('record_id') != _config().digest({k:v for k,v in calibration.items() if k!='record_id'})):
        raise ValueError('Locked production CE calibration required')
    cfg = _config()
    if (calibration.get('nodes') != list(cfg.GAMMAS)
            or calibration.get('samples_per_node') != cfg.CALIBRATION_BATCHES*cfg.MICRO_BATCH
            or calibration.get('model_states') != provenance['freeze_before']
            or calibration.get('locked_before_final') is not True
            or provenance.get('calibration_sha256') != _sha(path/'calibration.json')
            or provenance.get('calibration_record_id') != calibration['record_id']):
        raise ValueError('CE calibration/model binding changed')
    errors = calibration.get('per_RE_error', {})
    if set(errors) != set(provenance['freeze_before']): raise ValueError('All three CE calibration tables required')
    for table in errors.values():
        if not isinstance(table, list) or len(table) != 6: raise ValueError('Six CE calibration nodes required')
        for node in table:
            if not isinstance(node, list) or len(node) != 8: raise ValueError('Eight UE calibration rows required')
            for row in node:
                if (not isinstance(row, list) or len(row) != 256
                        or any(isinstance(v, bool) or not isinstance(v, (int,float))
                               or not math.isfinite(v) or v < 0 for v in row)):
                    raise ValueError('Finite nonnegative 256-RE calibration statistics required')
    _verify_statistics(path)
    return manifest


def export_directory(summary_path, evaluation_dir, output_dir):
    cfg, rt = _config(), _runtime()
    evaluation, output = _owned(evaluation_dir), _report(output_dir)
    summary_path = _owned(summary_path)
    if summary_path.parent != ROOT/'runs'/NAMESPACE: raise ValueError('Own root CE summary required')
    summary = rt.read_json(summary_path)
    prereg = summary_path.parent/'preregistration.json'
    if (summary.get('namespace') != NAMESPACE or summary.get('recipe_fingerprint') != cfg.fingerprint()
            or _sha(prereg) != summary.get('preregistration_sha256')):
        raise ValueError('CE study registration changed')
    manifest_path = evaluation/'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Bounded regular evaluation manifest required')
    manifest = _json(manifest_path.read_bytes())
    if (manifest.get('namespace') != NAMESPACE or manifest.get('core') != summary.get('core')
            or manifest.get('production_evidence') is not True
            or manifest.get('recipe_fingerprint') != cfg.fingerprint()):
        raise ValueError('Completed production CE evaluation required')
    payloads = {}
    for name in PUBLIC_FILES:

        source = evaluation/name
        if name != 'README.md' and (source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_FILE_BYTES):
            raise ValueError('Bounded regular CE evaluation payload required')
        raw = README_TEXT.encode() if name == 'README.md' else source.read_bytes()
        if name != 'README.md' and manifest.get('files', {}).get(name) != hashlib.sha256(raw).hexdigest():
            raise ValueError('CE evaluation payload binding changed')
        if name == 'provenance.json':
            record = _json(raw); _verify_provenance(record)
            record.update(preregistration_sha256=_sha(prereg),
                evaluation_manifest_sha256=_sha(evaluation/'manifest.json'))
            raw = json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'
        _validate_payload(name, raw); payloads[name] = raw
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, raw in payloads.items():
        (output/name).write_bytes(raw); hashes[name] = _sha(output/name)
    export = dict(format=NAMESPACE+'_export_v1', namespace=NAMESPACE, export_id=output.name[8:],
        files=hashes, source_manifest_sha256=_sha(evaluation/'manifest.json'),
        preregistration_sha256=_sha(prereg), contains_model_weights=False, contains_channel_tensors=False)
    rt.atomic_json(output/'export_manifest.json', export, exclusive=True)
    verify_export_directory(output)
    return dict(export, report_directory=str(output), path=str(output),
        manifest_sha256=_sha(output/'export_manifest.json'))


def _completed(directory=None):
    import run_twc_v1_k8_ce_refine as queue
    queue.verify_completed_export(directory)
    return _report(queue.load_summary()['report_directory'])


def _origin():
    if git_io._git(ROOT, ['remote', 'get-url', 'origin']) != REPOSITORY:
        raise ValueError('Official 03 repository origin required')


def _orphan(files, commit=None):
    records = []
    for name, path in sorted(files.items()):
        blob = git_io._git(ROOT, ['hash-object', '-w', '--stdin'], data=Path(path).read_bytes())
        records.append(f'100644 blob {blob}\t{name}\n')
    tree = git_io._git(ROOT, ['mktree'], data=''.join(records).encode())
    if commit:
        if (not re.fullmatch(r'[0-9a-f]{40}', commit)
                or git_io._git(ROOT, ['show', '-s', '--format=%T', commit]) != tree
                or git_io._git(ROOT, ['show', '-s', '--format=%P', commit])):
            raise ValueError('Prepared CE publication changed')
        return commit, tree
    commit = git_io._git(ROOT, ['-c', 'user.name=03 CE results exporter',
        '-c', 'user.email=ce-results@localhost', '-c', 'commit.gpgsign=false', 'commit-tree', tree],
        data=b'Public independent CE continuation scalar results\n')
    return commit, tree


def publish(directory=None):
    rt = _runtime(); path = _completed(directory); manifest = verify_export_directory(path); _origin()
    branch = 'results/'+NAMESPACE+'/'+manifest['export_id']; ref = 'refs/heads/'+branch
    uploads = _owned(ROOT/'runs'/NAMESPACE/'uploads'); uploads.mkdir(parents=True, exist_ok=True)
    receipt_path = uploads/(manifest['export_id']+'.json')
    with rt.process_lock(uploads/'publication.lock'):
        before = git_io._git_state(ROOT)
        old = rt.read_json(receipt_path) if receipt_path.exists() else None
        export_hash = _sha(path/'export_manifest.json')
        if old and (old.get('branch') != branch or old.get('export_manifest_sha256') != export_hash):
            raise ValueError('Saved CE publication identity changed')
        commit, tree = _orphan({name:path/name for name in (*PUBLIC_FILES, 'export_manifest.json')},
            commit=old.get('commit') if old else None)
        receipt = dict(status='prepared', branch=branch, commit=commit, tree=tree,
            export_manifest_sha256=export_hash, export_id=manifest['export_id'],
            url=WEB_REPOSITORY+'/tree/'+branch)
        rt.atomic_json(receipt_path, receipt)
        remote = git_io._git(ROOT, ['ls-remote', '--heads', 'origin', ref])
        if remote and remote.split() != [commit, ref]: raise ValueError('Existing result ref differs; no overwrite allowed')
        code = 0
        if not remote:
            code = subprocess.run(['git', 'push', '--force-with-lease='+ref+':',
                'origin', commit+':'+ref], cwd=ROOT, env=git_io._git_environment(), check=False).returncode
        if git_io._git_state(ROOT) != before: raise RuntimeError('Source HEAD or index changed during CE publication')
        receipt.update(status='uploaded' if code==0 else 'failed', git_exit_code=code)
        rt.atomic_json(receipt_path, receipt)
        if code: raise RuntimeError('Git upload failed; local report and prepared commit are preserved')
    print('Uploaded branch: '+branch+'\nResult commit: '+commit+'\nRead URL: '+receipt['url'])
    return receipt


def delete_remote(branch, expected_commit):
    if (not re.fullmatch('results/'+NAMESPACE+r'/[0-9a-f]{12}', branch)
            or not re.fullmatch(r'[0-9a-f]{40}', expected_commit)):
        raise ValueError('Exact CE result ref and expected commit required')
    _origin(); ref = 'refs/heads/'+branch
    if git_io._git(ROOT, ['ls-remote', '--heads', 'origin', ref]).split() != [expected_commit, ref]:
        raise ValueError('Result ref changed or absent; nothing deleted')
    before = git_io._git_state(ROOT)
    result = subprocess.run(['git', 'push', '--force-with-lease='+ref+':'+expected_commit,
        'origin', ':'+ref], cwd=ROOT, env=git_io._git_environment(), check=False)
    if git_io._git_state(ROOT) != before: raise RuntimeError('Source state changed during exact-ref deletion')
    if result.returncode: raise RuntimeError('Exact result-ref deletion failed; local evidence preserved')
    print('Deleted exact result ref; local results preserved. Git-history erasure is not guaranteed.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-only', action='store_true')
    action.add_argument('--publish', action='store_true')
    action.add_argument('--delete-remote', action='store_true')
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--branch'); parser.add_argument('--expected-commit')
    args = parser.parse_args(argv)
    if args.delete_remote:
        if args.export_dir or not args.branch or not args.expected_commit:
            parser.error('--delete-remote requires --branch and --expected-commit only')
        delete_remote(args.branch, args.expected_commit)
    elif args.branch or args.expected_commit:
        parser.error('--branch/--expected-commit are only used for deletion')
    elif args.publish: publish(args.export_dir)
    else:
        selected = _completed(args.export_dir); print(json.dumps(verify_export_directory(selected), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
