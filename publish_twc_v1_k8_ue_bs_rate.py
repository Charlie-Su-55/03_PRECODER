#!/usr/bin/env python3
"""Export UE+BS joint-rate evidence and archived controls as a separate result ref.

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
SCALAR_FILES = ('samples.csv.gz', 'summary.csv', 'comparisons.csv',
    'input_fingerprints.json', 'training_pairing.json')
PUBLIC_FILES = (*SCALAR_FILES, 'provenance.json', 'README.md')
# 49,152 scalar observations can exceed 16 MiB when compressed. Keep a
# bounded report while allowing the declared full final sample budget.
MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_EXPANDED_BYTES = 32*1024**2, 48*1024**2, 64*1024**2
_sha, _git, _git_state = transport._sha, transport._git, transport._git_state
_push_commit, _git_environment = transport._push_commit, transport._git_environment
README_TEXT = ('# UE+BS joint-rate comparison\n\n'
    'Public complete-MU scalar observations and redacted provenance. '
    'No neural weights, channel tensors, credentials or machine paths.\n\n'
    'FDMA and Hybrid each train UE and BS for the same number of new updates. '
    'The same-source archived BS-only controls have matching update counts and '
    'training inputs. Joint training also updates the causal historical power '
    'prior, while the archived controls keep it frozen: this is a pipeline '
    'comparison, not a single-factor attribution to UE gradients. '
    'CE and its error table are frozen. Source, archived controls and new fixed '
    'terminals replay one fresh final validation set; all six SNR nodes and '
    'negative results remain. Test stays sealed. '
    'Intervals condition on one source training seed.\n\n'
    'Removing this public result ref does not guarantee erasure from history '
    'or caches. Original local evidence remains.\n')


def _config():
    from configs import twc_v1_k8_ue_bs_rate_config as cfg
    return cfg


def _runtime():
    import ue_bs_rate_runtime as runtime
    return runtime


def _queue():
    import run_twc_v1_k8_ue_bs_rate as queue
    return queue


def _report(path):
    cfg, runtime = _config(), _runtime()
    path = runtime.inside(path)
    subtree = (cfg.OUTPUT_ROOT/'reports').resolve()
    if (path == subtree or not path.is_relative_to(subtree)
            or not re.fullmatch(r'attempt_[0-9a-f]{12,32}', path.name)):
        raise ValueError('Owned joint-rate report attempt required')
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
        return
    if path.name == 'README.md':
        if path.read_text(encoding='utf-8') != README_TEXT:
            raise ValueError('Fixed public result readme required')
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
                    'component', 'contrast', 'method', 'phase', 'positive', 'negative'}
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
        import twc_v1_k8_ue_bs_rate_results as results
        cfg = _config()
        results.validate_rows(results.read_raw_csv(path), samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
            batch_size=cfg.MICRO_BATCH)


def _sources(evaluation):
    return {name:evaluation/name for name in SCALAR_FILES}


def _record_checkpoint(record):
    values = [record[key] for key in ('checkpoint_sha256', 'sha256') if key in record]
    if (not values or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
            for v in values) or len(set(values)) != 1):
        raise ValueError('An unambiguous checkpoint content identity is required')
    return values[0]


def _public_record(record):
    allowed = ('kind', 'arm', 'method', 'phase', 'source_commit', 'model_commit',
        'checkpoint_sha256', 'sha256', 'core', 'step', 'state_sha256',
        'CE_state_sha256', 'statistics_state_sha256', 'base_decoder_state_sha256',
        'source_state_sha256', 'source_decoder_sha256', 'initial_decoder_sha256',
        'model_schema', 'parameter_schema', 'initialization_sha256',
        'calibration_model_sha256', 'calibration_manifest_sha256',
        'fingerprint', 'recipe_fingerprint', 'counters', 'training_code', 'frozen_identity',
        'model_state_sha256', 'decoder_state_sha256', 'manifest_sha256',
        'terminal_receipt_sha256', 'training_trace', 'progress_sha256',
        'production_evidence', 'evidence_kind', 'encoder_state_sha256',
        'base_ema_updates', 'ema_updates', 'total_ema_updates', 'additional_EMA_commits',
        'source_kind', 'base_encoder_state_sha256', 'base_statistics_state_sha256',
        'statistics_schema', 'CE_table_state_sha256')
    if not isinstance(record, dict):
        raise ValueError('Source record dictionary required')
    result = {key: record[key] for key in allowed if key in record}
    _assert_public_json(result)
    return result


def _public_provenance(summary_path, evaluation, final):
    cfg,runtime,queue=_config(),_runtime(),_queue()
    summary=runtime.read_json(summary_path)
    queue.verify_preregistration(summary,summary_path)
    if (summary.get('namespace')!=cfg.NAMESPACE or summary.get('core')!=final.get('core')
            or summary.get('recipe')!=cfg.study_recipe()
            or summary.get('recipe_fingerprint')!=cfg.study_fingerprint()):
        raise ValueError('Joint-rate preregistration/core/recipe changed')
    for flag in ('all_terminals_verified_before_data','paired_inputs_verified',
            'frozen_models_verified','source_initialization_matched','common_CE_verified'):
        if final.get(flag) is not True:raise ValueError('Final joint-rate acceptance missing: '+flag)
    if final.get('production_evidence') is not True:
        raise ValueError('CPU evaluation fixture cannot be published as scientific evidence')
    pairing=_json((evaluation/'training_pairing.json').read_text())
    if (final.get('training_pairing')!=pairing or pairing.get('status')!='passed'
            or pairing.get('production_evidence') is not True
            or pairing.get('new_snapshot_pairing_verified') is not True
            or pairing.get('common_trace_matches_archived_controls') is not True
            or pairing.get('archived_raw_progress_replayed') is not False
            or pairing.get('common_training_trace')!=cfg.TRAINING_TRACE_PIN
            or pairing.get('hashed_microbatches')!=cfg.UPDATES*cfg.ACCUMULATION
            or pairing.get('hashed_MU_frames')!=cfg.UPDATES*cfg.ACCUMULATION*cfg.MICRO_BATCH
            or pairing.get('gamma_counts')!=cfg.counters(cfg.UPDATES)['gamma_counts']
            or set(pairing.get('trajectories',{}))!=set(cfg.ARMS)
            or pairing.get('archived_controls')!={a:cfg.control_identity(a) for a in cfg.ARMS}):
        raise ValueError('Production matched training-input evidence required')
    records=final.get('source_records',{})
    signatures=final.get('frozen_signatures',{})
    terminals=final.get('terminal_records',{})
    controls=final.get('control_records',{})
    if (set(records)!=set(cfg.METHODS) or set(signatures)!=set(cfg.METHODS)
            or set(terminals)!=set(cfg.ARMS) or set(controls)!=set(cfg.ARMS)):
        raise ValueError('All six source/control/joint views are required')
    common_CE,common_table=None,None
    for arm in cfg.ARMS:
        runtime.verify_control_files(arm)
        for method in cfg.METHODS:
            if set(records[method])!=set(cfg.ARMS) or set(signatures[method])!=set(cfg.ARMS):
                raise ValueError('Complete physical arm/method coverage required')
            record,state=records[method][arm],signatures[method][arm]
            before=state.get('before',{})
            if (set(before)!={'codec','encoder','decoder','statistics','CE','CE_table'}
                    or before!=state.get('after')
                    or any(not re.fullmatch(r'[0-9a-f]{64}',v) for v in before.values())
                    or before['codec']!=record.get('state_sha256')):
                raise ValueError('Evaluation model/CE/prior state changed')
            if common_CE is None:common_CE,common_table=before['CE'],before['CE_table']
            if common_CE!=before['CE'] or common_table!=before['CE_table']:
                raise ValueError('CE or CE-error table differs across paired methods')
            if method=='joint':
                terminal=terminals[arm]
                if (record.get('step')!=cfg.UPDATES or terminal.get('step')!=cfg.UPDATES
                        or _record_checkpoint(record)!=_record_checkpoint(terminal)):
                    raise ValueError('Fixed joint terminal identity changed')
            elif method=='bs_only':
                if (_record_checkpoint(record)!=_record_checkpoint(controls[arm])
                        or record.get('step')!=cfg.UPDATES
                        or _record_checkpoint(record)!=cfg.SOURCE_CONTROL_PINS[arm]['checkpoint_sha256']):
                    raise ValueError('Archived BS-only control identity changed')
        source=signatures['source'][arm]['before'];control=signatures['bs_only'][arm]['before']
        if any(source[name]!=control[name] for name in ('encoder','statistics','CE','CE_table')):
            raise ValueError('Archived BS-only control did not preserve its source UE/prior')
    public=dict(format=cfg.NAMESPACE+'_public_provenance_v1',namespace=cfg.NAMESPACE,
        experiment_code_commit=summary.get('code',{}).get('commit'),core=final['core'],
        recipe_fingerprint=cfg.study_fingerprint(),
        recipe_fingerprints={arm:cfg.fingerprint(arm) for arm in cfg.ARMS},
        source_manifest_sha256=runtime.sha256(evaluation/'manifest.json'),
        preregistration_sha256=summary['preregistration_sha256'],
        data_protocol=final['data_protocol'],arms=list(cfg.ARMS),methods=list(cfg.METHODS),
        gamma_db=list(cfg.GAMMAS),batch_size=cfg.MICRO_BATCH,accumulation=cfg.ACCUMULATION,
        new_UE_updates_per_trajectory=cfg.UPDATES,new_BS_updates_per_trajectory=cfg.UPDATES,
        total_new_UE_updates=len(cfg.ARMS)*cfg.UPDATES,total_new_BS_updates=len(cfg.ARMS)*cfg.UPDATES,
        CE_updates=0,EMA_new_commits_per_joint_trajectory=cfg.UPDATES,
        control_retrained=False,control_prior_frozen=True,joint_prior_causally_updated=True,
        single_factor_UE_gradient_attribution=False,
        final_batches=cfg.FINAL_BATCHES,final_samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
        balanced_gamma_counts=cfg.counters(cfg.UPDATES)['gamma_counts'],
        all_training_terminals_precede_final_data=True,fixed_terminal_selection=True,
        validation_used_for_fitting=False,test_used=False,training_pairing=pairing,
        source_records={method:{arm:_public_record(record) for arm,record in values.items()}
            for method,values in records.items()},
        terminal_records={arm:_public_record(value) for arm,value in terminals.items()},
        control_records={arm:_public_record(value) for arm,value in controls.items()},
        frozen_signatures=signatures,physical=cfg.recipe('fdma')['physical'],
        public_repository=True,contains_model_weights=False,contains_neural_weights=False,
        contains_channel_tensors=False,deletion_is_not_guaranteed_history_erasure=True)
    _assert_public_json(public)
    return public


def export_directory(summary_path, evaluation_dir, output_dir):
    """Create an immutable, verified scalar report; no archive or network IO."""
    runtime, cfg, queue = _runtime(), _config(), _queue()
    evaluation = runtime.inside(evaluation_dir)
    final = queue.verify_manifest(evaluation)
    public = _public_provenance(summary_path, evaluation, final)
    sources = _sources(evaluation)
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
    (output/'README.md').write_text(README_TEXT, encoding='utf-8')
    hashes['README.md'] = runtime.sha256(output/'README.md')
    manifest = dict(format=cfg.NAMESPACE+'_export_v1', export_id=output.name[8:],
        namespace=cfg.NAMESPACE, files=hashes, contains_model_weights=False,
        contains_neural_weights=False, contains_channel_tensors=False,public_repository=True,
        source_manifest_sha256=runtime.sha256(evaluation/'manifest.json'))
    runtime.atomic_json(output/'export_manifest.json', manifest, exclusive=True)
    verify_export_binding(summary_path, evaluation, output)
    return dict(manifest, path=str(output), manifest=str(output/'export_manifest.json'),
        manifest_sha256=runtime.sha256(output/'export_manifest.json'))


def verify_export_directory(path):
    path, cfg = _report(path), _config()
    manifest_path = path/'export_manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Regular public export manifest required')
    manifest = _json(manifest_path.read_text())
    expected = {'format', 'export_id', 'namespace', 'files', 'contains_model_weights',
        'contains_neural_weights','contains_channel_tensors','public_repository',
        'source_manifest_sha256'}
    if (set(manifest) != expected or manifest.get('format') != cfg.NAMESPACE+'_export_v1'
            or manifest.get('namespace') != cfg.NAMESPACE or manifest.get('export_id') != path.name[8:]
            or manifest.get('contains_model_weights') is not False or manifest.get('contains_channel_tensors') is not False
            or manifest.get('contains_neural_weights') is not False
            or manifest.get('public_repository') is not True or set(manifest.get('files', {})) != set(PUBLIC_FILES)
            or any(not re.fullmatch(r'[0-9a-f]{64}', manifest[key]) for key in
                ('source_manifest_sha256',))):
        raise ValueError('Exact public export identity and inventory required')
    if {p.name for p in path.iterdir()} != set(PUBLIC_FILES)|{'export_manifest.json'}:
        raise ValueError('Unexpected public result artifact')
    total = 0
    for name, digest in manifest['files'].items():
        file = path/name
        _check_scalar_file(file)
        total += file.stat().st_size
        if _sha(file) != digest:
            raise ValueError('Public result bytes changed: '+name)
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Public scalar result size budget exceeded')
    return manifest


def verify_export_binding(summary_path, evaluation_dir, export_dir):
    """Reject a self-consistent report from a different preregistered batch."""
    runtime, queue = _runtime(), _queue()
    evaluation = runtime.inside(evaluation_dir)
    output = _report(export_dir)
    final = queue.verify_manifest(evaluation)
    manifest = verify_export_directory(output)
    expected = _public_provenance(summary_path, evaluation, final)
    if (runtime.read_json(output/'provenance.json') != expected
            or manifest['source_manifest_sha256'] != runtime.sha256(evaluation/'manifest.json')):
        raise ValueError('Public export provenance no longer binds this batch')
    if any(manifest['files'][name] != runtime.sha256(source)
            for name,source in _sources(evaluation).items()):
        raise ValueError('Public export payload no longer binds this batch')
    return manifest


def _branch(export_id):
    if not re.fullmatch(r'[0-9a-f]{12,32}', export_id):
        raise ValueError('Exact joint-rate export ID required')
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
    commit = _git(root, ['-c', 'user.name=03 joint-rate exporter',
        '-c', 'user.email=diagnostics@localhost', '-c', 'commit.gpgsign=false',
        'commit-tree', tree], data=b'Public UE+BS joint-rate comparison\n')
    return commit, tree


def _completed_export(path=None):
    queue, runtime = _queue(), _runtime()
    summary, summary_path = queue.load_summary()
    runtime.guard_core(summary['core'])
    queue.verify_preregistration(summary, summary_path)
    required = set(queue.stage_names())
    if (summary.get('status') != 'completed' or not required <= set(summary['stages'])
            or set(summary['stages'])-required-{'startup_policy'}):
        raise ValueError('A completed matched joint-rate batch is required')
    for record in summary['stages'].values():
        if record.get('status') != 'completed':
            raise ValueError('Incomplete matched joint-rate stage')
        queue.verify_evidence(record)
    final = summary['stages']['final_validation']['details']['output_dir']
    selected = _report(summary['stages']['directory_export']['details']['path'])
    if path is not None and _report(path) != selected:
        raise ValueError('Only the completed batch export may be published')
    verify_export_binding(summary_path, final, selected)
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
        raise ValueError('Exact joint-rate result ref and expected commit required')
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
