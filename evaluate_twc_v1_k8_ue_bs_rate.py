#!/usr/bin/env python3
"""Fresh paired validation of source, archived BS control and joint terminals.

All six source/control/terminal views are loaded and accepted before a final
channel is requested.  UE-dependent waveform, prior and receiver outputs may
change after joint training; physical channels and random streams may not.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
from physical_split_finish_io import write_union_csv

EVALUATION_FILES = ('samples.csv.gz', 'summary.csv', 'comparisons.csv',
    'input_fingerprints.json', 'source_records.json', 'training_pairing.json')
COMMON_FIELDS = frozenset(('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity',
    'gamma', 'estimated_UL', 'CE_variance'))
METHOD_FIELDS = frozenset(('wave_active', 'Y_rx', 'prior'))
INPUT_FIELDS = COMMON_FIELDS | METHOD_FIELDS
RECOVERY_FIELDS = frozenset(('soft', 'reliability', 'estimated_UL', 'variance'))


def _config():
    from configs import twc_v1_k8_ue_bs_rate_config as cfg
    return cfg


def _runtime():
    import ue_bs_rate_runtime as runtime
    return runtime


def _trainer():
    import train_twc_v1_k8_ue_bs_rate as trainer
    return trainer


def _results():
    import twc_v1_k8_ue_bs_rate_results as results
    return results


def _trace():
    # The legacy format deliberately identifies the replayed common inputs,
    # not the new experiment or its trainable parameters.
    import train_twc_v1_k8_ul_adapt as trace
    return trace


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def checked_output(path):
    cfg, runtime = _config(), _runtime()
    path = runtime.inside(path)
    subtree = (cfg.OUTPUT_ROOT/'evaluation').resolve()
    if path == subtree or not path.is_relative_to(subtree):
        raise ValueError('Proper owned UE-BS final evaluation attempt required')
    return path


def _ce_table_digest(stats):
    return hashlib.sha256(_canonical({name: _tensor_hash(getattr(stats, name))
        for name in ('ce_errors', 'ce_calibrated')})).hexdigest()


def _records_signature(codec, stats, ce):
    modules = dict(codec=codec, encoder=codec.encoder, decoder=codec.decoder,
        statistics=stats, CE=ce)
    for module in modules.values():
        if any(child.training for child in module.modules()) or any(p.requires_grad for p in module.parameters()):
            raise ValueError('Final source/control/joint models must be frozen in evaluation mode')
    return dict({name: _runtime().state_digest(module) for name, module in modules.items()},
        CE_table=_ce_table_digest(stats))


def _check_files(record, *, fixture):
    runtime = _runtime()
    if not fixture:
        runtime.verify_files(record)
        return
    if not isinstance(record, dict) or not record.get('frozen_files'):
        raise ValueError('Explicit read-only CPU fixture evidence required')
    for item in record['frozen_files']:
        path = Path(item['path'])
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != item['sha256']:
            raise ValueError('Read-only source/control/terminal bytes changed')


def _check_record(record, arm, method, *, fixture):
    cfg = _config()
    expected_step = 5000 if method == 'source' else cfg.UPDATES
    if (not isinstance(record, dict) or record.get('arm') != arm
            or record.get('method') != method or record.get('step') != expected_step):
        raise ValueError('Exact source/control/fixed joint terminal identity and budget required')
    if not fixture:
        if record.get('production_evidence') is not True:
            raise ValueError('CPU fixture cannot pass production evaluation')
        if method == 'joint' and record.get('core') != _runtime().core_identity():
            raise ValueError('Joint terminal scientific core changed')
        if method == 'bs_only':
            pins = cfg.SOURCE_CONTROL_PINS[arm]
            for name in ('checkpoint_sha256', 'manifest_sha256', 'terminal_receipt_sha256',
                    'initialization_sha256', 'progress_sha256'):
                if record.get(name) != pins[name]:
                    raise ValueError('Archived matched BS-only control changed: '+name)
    _check_files(record, fixture=fixture)


def _check_state_record(record, signatures, *, fixture):
    if fixture:
        return
    fields = dict(codec='state_sha256', encoder='encoder_state_sha256',
        decoder='decoder_state_sha256', statistics='statistics_state_sha256', CE='CE_state_sha256')
    for name, field in fields.items():
        if record.get(field) != signatures[name]:
            raise ValueError('Loaded model signatures do not bind accepted record: '+field)


def _base_manifest(fixture, core):
    cfg = _config()
    return dict(format=cfg.NAMESPACE+'_evaluation_v1', status='running', role='final',
        core=core, recipe_fingerprints={arm: cfg.fingerprint(arm) for arm in cfg.ARMS},
        arms=list(cfg.ARMS), methods=list(cfg.METHODS), gamma_db=list(cfg.GAMMAS),
        batch_size=cfg.MICRO_BATCH, n_batches=cfg.FINAL_BATCHES,
        samples_per_condition=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
        fixed_rzf_alpha=cfg.ALPHA, DL_SNR=cfg.DL_SNR, terminal_updates=cfg.UPDATES,
        test_used=False, checkpoint_selection=False, neural_updates_during_evaluation=0,
        UE_updates_during_evaluation=0, CE_updates=0, EMA_updates_during_evaluation=0,
        production_evidence=not fixture, source_checkpoints_verified=not fixture,
        all_terminals_verified_before_data=True,
        attribution='joint UE+BS with causal prior versus archived BS-only with frozen prior')


def score_out(out, H_dl):
    from evaluate_twc_v1_k8_ul_adapt import score_out as unchanged_score
    return unchanged_score(out, H_dl)


def _validate_vectors(vectors, batch):
    import torch
    if not set(_results().REQUIRED_METRICS) <= set(vectors):
        raise ValueError('All final rate, CSI and soft-error vectors required')
    if any(not torch.is_tensor(value) or tuple(value.shape) != (batch,) or value.requires_grad
            or not bool(torch.isfinite(value).all()) for value in vectors.values()):
        raise ValueError('Finite detached complete-MU metric vectors required')


def _recovery_hashes(out):
    variance = (out['recovered']['variance'] if 'recovered' in out else
        out['reliability'][..., :out['waveform'].active.shape[-1]].reciprocal())
    return dict(soft=_tensor_hash(out['soft']), reliability=_tensor_hash(out['reliability']),
        estimated_UL=_tensor_hash(out['H_ul_est']), variance=_tensor_hash(variance))


def _add_fingerprint(index, arm, method, bi, sample, out, device):
    hashes = dict(out['input_hashes'], H_dl=_tensor_hash(sample['H_dl']), H_ul=_tensor_hash(sample['H_ul']))
    if set(hashes) != INPUT_FIELDS:
        raise ValueError('Runtime omitted actual shared or UE-dependent input hashes')
    gamma = float(out['gamma_db'][0])
    key = arm, bi, gamma
    common = {name: hashes[name] for name in COMMON_FIELDS}
    if key not in index:
        index[key] = dict(role='final', arm=arm, batch_index=bi, gamma_db=gamma,
            address=sample['address'], device=str(device), input_hashes=common,
            conditions=[], method_inputs={}, recovery_hashes={}, decoder_output_hashes={})
    row = index[key]
    if row['input_hashes'] != common or row['address'] != sample['address'] or method in row['conditions']:
        raise ValueError('Methods did not replay identical newly addressed common inputs')
    row['conditions'].append(method)
    row['method_inputs'][method] = {name: hashes[name] for name in METHOD_FIELDS}
    row['recovery_hashes'][method] = _recovery_hashes(out)
    row['decoder_output_hashes'][method] = _tensor_hash(out['H_hat'])


def fingerprint_map(entries, *, batches=None, device=None):
    import torch
    cfg, runtime = _config(), _runtime()
    batches = cfg.FINAL_BATCHES if batches is None else batches
    expected = {(arm, bi, float(g)) for arm in cfg.ARMS for bi in range(batches) for g in cfg.GAMMAS}
    methods = set(cfg.METHODS)
    index = {}
    metadata = None
    for row in entries:
        key = row.get('arm'), row.get('batch_index'), row.get('gamma_db')
        if key not in expected or key in index or row.get('role') != 'final':
            raise ValueError('Duplicate/unexpected final physical condition')
        if device is None:
            device = row.get('device')
        if metadata is None:
            metadata = runtime.make_data(device, role='final')
        if (row.get('device') != str(device)
                or row.get('address') != metadata.address('validation', cfg.ROLE_RANGES['final'][0]+key[1])
                or set(row.get('input_hashes', {})) != COMMON_FIELDS
                or any(not _valid_hash(v) for v in row['input_hashes'].values())
                or set(row.get('conditions', [])) != methods or len(row.get('conditions', [])) != len(methods)):
            raise ValueError('Incomplete final input identity/address/method coverage')
        expected_gamma = _tensor_hash(torch.full((cfg.MICRO_BATCH,), key[2], dtype=torch.float32))
        if row['input_hashes']['gamma'] != expected_gamma:
            raise ValueError('Actual gamma tensor differs from its declared final point')
        for name, fields in (('method_inputs', METHOD_FIELDS), ('recovery_hashes', RECOVERY_FIELDS)):
            views = row.get(name, {})
            if set(views) != methods or any(set(v) != fields or any(not _valid_hash(h) for h in v.values()) for v in views.values()):
                raise ValueError('Complete per-method input/recovery hashes required')
        if (set(row.get('decoder_output_hashes', {})) != methods
                or any(not _valid_hash(v) for v in row['decoder_output_hashes'].values())):
            raise ValueError('Complete per-method decoder output hashes required')
        if (row['method_inputs']['source'] != row['method_inputs']['bs_only']
                or row['recovery_hashes']['source'] != row['recovery_hashes']['bs_only']):
            raise ValueError('Archived BS-only control changed its frozen source waveform or receiver')
        index[key] = row
    if set(index) != expected:
        raise ValueError('Missing final physical condition')
    for bi in range(batches):
        for gamma in cfg.GAMMAS:
            pair = [index[arm, bi, float(gamma)]['input_hashes'] for arm in cfg.ARMS]
            if any(v != pair[0] for v in pair[1:]):
                raise ValueError('Physical arms did not use paired common channel/CE/noise/pilot inputs')
    for arm in cfg.ARMS:
        for name in ('H_dl', 'H_ul'):
            values = []
            for bi in range(batches):
                shared = {index[arm, bi, float(g)]['input_hashes'][name] for g in cfg.GAMMAS}
                if len(shared) != 1:
                    raise ValueError('Gamma views changed the complete-MU channel sample')
                values.append(next(iter(shared)))
            if len(set(values)) != batches:
                raise ValueError('Distinct final batch addresses repeated actual channel payloads')
    return index


def _check_prior_source_relationship(signatures):
    cfg = _config()
    for arm in cfg.ARMS:
        source = signatures['source'][arm]['before']
        control = signatures['bs_only'][arm]['before']
        if any(source[name] != control[name] for name in ('encoder', 'statistics')):
            raise ValueError('Archived BS-only control does not retain the source UE/prior')
    ce = {signatures[method][arm]['before']['CE'] for method in cfg.METHODS for arm in cfg.ARMS}
    ce_tables = {signatures[method][arm]['before']['CE_table'] for method in cfg.METHODS for arm in cfg.ARMS}
    if len(ce) != 1 or len(ce_tables) != 1:
        raise ValueError('CE network or paid-pilot CE-error table differs across methods/arms')


def _check_source_bindings(records, *, fixture):
    if fixture:
        return
    for arm in _config().ARMS:
        initial = dict(records['source'][arm], method='joint')
        if records['joint'][arm].get('source_identity') != initial:
            raise ValueError('Joint terminal no longer binds its exact pinned source')


def _check_past_disjoint(entries, *, fixture):
    cfg, runtime = _config(), _runtime()
    # The original calibration samples are available locally on the server.
    path = Path(cfg.CALIBRATION_DIR)/'input_fingerprints.json'
    if fixture and not path.is_file():
        return
    previous = runtime.read_json(path)
    for field in ('H_dl', 'H_ul'):
        values = {r['fingerprint'][field] for r in previous}
        if values.intersection(r['input_hashes'][field] for r in entries):
            raise ValueError('Fresh final channels overlap completed calibration labels')


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate training-journal JSON field')
        result[key] = value
    return result


def verify_training_pairing(terminal_records, control_records, *, fixture=False):
    """Audit new snapshots and compare common traces with pinned controls.

    Archived control traces are accepted by their exact recorded hashes; this
    does not independently replay historical tensors or its raw progress.
    """
    import torch
    cfg, runtime, trace_api = _config(), _runtime(), _trace()
    if fixture:
        return dict(format=cfg.NAMESPACE+'_training_pairing_v1', status='test_fixture_skipped',
            production_evidence=False, new_snapshot_pairing_verified=False,
            common_trace_matches_archived_controls=False, archived_raw_progress_replayed=False,
            trajectories={})
    if set(terminal_records) != set(cfg.ARMS) or set(control_records) != set(cfg.ARMS):
        raise ValueError('Both new joint terminals and archived controls required')
    expected_steps = set(range(cfg.SAVE_EVERY, cfg.UPDATES+1, cfg.SAVE_EVERY))
    trajectories, snapshots, common_protocol = {}, {}, None
    for arm in cfg.ARMS:
        terminal, control = terminal_records[arm], control_records[arm]
        protocol = terminal.get('data_protocol')
        metadata = runtime.make_training_data('cuda:0')
        if (not isinstance(protocol, dict) or protocol != metadata.protocol
                or protocol.get('execution', {}).get('device') != 'cuda:0'
                or common_protocol is not None and protocol != common_protocol):
            raise ValueError('Joint training changed the archived common replay protocol')
        common_protocol = protocol
        if control.get('training_trace') != cfg.TRAINING_TRACE_PIN:
            raise ValueError('Archived control common-input trace pin changed')
        directory = runtime.inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm, 'joint'))
        path = runtime.inside(directory/'progress.jsonl')
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 64*1024**2
                or runtime.sha256(path) != terminal.get('progress_sha256')):
            raise ValueError('Complete immutable new joint training journal required')
        selected, rows, last_step = {}, 0, 0
        base = terminal.get('base_ema_updates')
        if type(base) is not int or base < 0:
            raise ValueError('Inherited EMA counter required for joint terminal')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if len(line) > 1024**2:
                    raise ValueError('Training journal row exceeds bounded schema')
                row = json.loads(line, object_pairs_hook=_unique_pairs)
                step = row.get('step')
                if (type(step) is not int or step not in expected_steps or step < last_step
                        or row.get('arm') != arm or row.get('method') != 'joint'
                        or row.get('counters') != cfg.counters(step, base)):
                    raise ValueError('Joint saved-boundary identity/counter changed')
                last_step = step
                items = row.get('input_fingerprints')
                if not isinstance(items, list) or len(items) != cfg.ACCUMULATION:
                    raise ValueError('Every saved joint update requires all actual microbatches')
                trace = trace_api.validate_training_trace(row.get('training_trace_before_update'),
                    protocol, microbatches=(step-1)*cfg.ACCUMULATION)
                for micro, item in enumerate(items):
                    index = cfg.micro_index(step, micro)
                    if (not isinstance(item, dict) or item.get('micro_index') != index
                            or item.get('address') != metadata.address('codec_train', index)
                            or not set(trace_api.TRACE_FIELDS) <= set(item)):
                        raise ValueError('Actual common training address/microbatch changed')
                    for field in ('channel', 'standard_noise', 'pilot_polarity', 'gamma'):
                        if not _valid_hash(item[field]):
                            raise ValueError('Actual common training tensor SHA256 required')
                    expected_gamma = _tensor_hash(torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32))
                    if item['gamma'] != expected_gamma:
                        raise ValueError('Actual joint gamma differs from its balanced schedule')
                    trace = trace_api.advance_training_trace(trace, item, protocol)
                if trace != row.get('training_trace'):
                    raise ValueError('Actual joint snapshots do not close their rolling trace')
                selected[step] = row
                rows += 1
        if set(selected) != expected_steps:
            raise ValueError('Incomplete joint saved-boundary snapshots')
        final_trace = trace_api.validate_training_trace(terminal.get('training_trace'), protocol,
            microbatches=cfg.UPDATES*cfg.ACCUMULATION)
        if final_trace != selected[cfg.UPDATES]['training_trace'] or final_trace != cfg.TRAINING_TRACE_PIN:
            raise ValueError('Joint common training stream differs from archived matched controls')
        snapshots[arm] = selected
        trajectories[arm] = dict(progress_sha256=terminal['progress_sha256'],
            logged_boundary_updates=len(selected), logged_microbatches=len(selected)*cfg.ACCUMULATION,
            retained_journal_rows=rows, superseded_journal_rows=rows-len(selected),
            base_ema_updates=base, training_trace=final_trace,
            snapshot_sha256=hashlib.sha256(_canonical([dict(step=step,
                input_fingerprints=[{k: item[k] for k in trace_api.TRACE_FIELDS}
                    for item in selected[step]['input_fingerprints']],
                training_trace_before_update=selected[step]['training_trace_before_update'],
                training_trace=selected[step]['training_trace']) for step in sorted(selected)])).hexdigest())
    for step in sorted(expected_steps):
        for micro in range(cfg.ACCUMULATION):
            common = [{k: snapshots[arm][step]['input_fingerprints'][micro][k]
                for k in trace_api.TRACE_FIELDS} for arm in cfg.ARMS]
            if any(value != common[0] for value in common[1:]):
                raise ValueError('Joint physical arms did not train on paired common inputs')
        if len({_canonical(snapshots[arm][step]['training_trace']) for arm in cfg.ARMS}) != 1:
            raise ValueError('Joint saved-boundary trace endpoints differ across physical arms')
    return dict(format=cfg.NAMESPACE+'_training_pairing_v1', status='passed', production_evidence=True,
        new_snapshot_pairing_verified=True, common_trace_matches_archived_controls=True,
        archived_raw_progress_replayed=False, common_training_trace=cfg.plain(cfg.TRAINING_TRACE_PIN),
        hashed_microbatches=cfg.UPDATES*cfg.ACCUMULATION,
        hashed_MU_frames=cfg.UPDATES*cfg.ACCUMULATION*cfg.MICRO_BATCH,
        logged_boundary_updates=len(expected_steps), logged_microbatches=len(expected_steps)*cfg.ACCUMULATION,
        gamma_counts=cfg.gamma_counts(cfg.UPDATES), trajectories=trajectories,
        archived_controls={arm: cfg.control_identity(arm) for arm in cfg.ARMS},
        policy='new actual snapshots plus pinned historical trace endpoints; no independent tensor replay')


def _write_gzip(path, rows):
    fields = sorted(set().union(*(set(row) for row in rows)))
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output_dir, *, require_cuda=True, source_loader=None, control_loader=None,
        terminal_loader=None, data_factory=None, forward=None):
    import torch
    cfg, runtime, results = _config(), _runtime(), _results()
    output = checked_output(output_dir)
    core = runtime.core_identity()
    runtime.guard_core(core)
    fixture = not require_cuda or any(v is not None for v in
        (source_loader, control_loader, terminal_loader, data_factory, forward))
    device = runtime.configure_cuda() if require_cuda else torch.device('cpu')
    source_loader = source_loader or (lambda arm, dev: runtime.load_variant(arm, 'source', dev))
    control_loader = control_loader or runtime.load_control
    terminal_loader = terminal_loader or _trainer().load_finetuned
    data_factory = data_factory or (lambda dev: runtime.make_data(dev, role='final'))
    forward = forward or runtime.forward_codec
    loaders = dict(source=source_loader, bs_only=control_loader, joint=terminal_loader)
    admitted = {method: {} for method in cfg.METHODS}
    admitted_signatures = {method: {} for method in cfg.METHODS}
    # Every real source/control/terminal acceptance precedes the final factory.
    for method in cfg.METHODS:
        for arm in cfg.ARMS:
            codec, stats, ce, record = loaders[method](arm, device)
            _check_record(record, arm, method, fixture=fixture)
            state = _records_signature(codec, stats, ce)
            _check_state_record(record, state, fixture=fixture)
            admitted[method][arm] = cfg.plain(record)
            admitted_signatures[method][arm] = dict(before=state, after=state)
            del codec, stats, ce
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    _check_prior_source_relationship(admitted_signatures)
    _check_source_bindings(admitted, fixture=fixture)
    pairing = verify_training_pairing(admitted['joint'], admitted['bs_only'], fixture=fixture)
    output.mkdir(parents=True, exist_ok=False)
    manifest = _base_manifest(fixture, core)
    manifest.update(terminal_records=admitted['joint'], control_records=admitted['bs_only'],
        training_pairing=pairing)
    started = time.perf_counter()
    rows, fingerprints, records, signatures, protocol = [], {}, {}, {}, None
    try:
        with torch.inference_mode():
            for method in cfg.METHODS:
                records[method], signatures[method] = {}, {}
                for arm in cfg.ARMS:
                    codec, stats, ce, record = loaders[method](arm, device)
                    _check_record(record, arm, method, fixture=fixture)
                    if cfg.plain(record) != admitted[method][arm]:
                        raise ValueError('Preflight model identity changed before final evaluation')
                    before = _records_signature(codec, stats, ce)
                    _check_state_record(record, before, fixture=fixture)
                    data = data_factory(device)
                    metadata = runtime.make_data(device, role='final')
                    if data.protocol != metadata.protocol or protocol is not None and data.protocol != protocol:
                        raise ValueError('Final replay protocol differs from the declared fresh namespace')
                    protocol = data.protocol
                    records[method][arm] = cfg.plain(record)
                    for bi in range(cfg.FINAL_BATCHES):
                        absolute = cfg.ROLE_RANGES['final'][0]+bi
                        sample = data.sample('validation', absolute)
                        if sample.get('address') != metadata.address('validation', absolute):
                            raise ValueError('Final data address changed to monitor/training/test')
                        truth_before = (_tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul']))
                        for gi, gamma in enumerate(cfg.GAMMAS):
                            out = forward(codec, stats, ce, sample, gamma_db=gamma,
                                noise_key=('final', bi, gi), method=method)
                            out.setdefault('layout', stats.layout)
                            _add_fingerprint(fingerprints, arm, method, bi, sample, out, device)
                            output_hash, recovery_hash = _tensor_hash(out['H_hat']), _recovery_hashes(out)
                            vectors = score_out(out, sample['H_dl'])
                            _validate_vectors(vectors, cfg.MICRO_BATCH)
                            if output_hash != _tensor_hash(out['H_hat']) or recovery_hash != _recovery_hashes(out):
                                raise ValueError('Offline scoring mutated decoder or receiver outputs')
                            values = {name: value.detach().cpu().tolist() for name, value in vectors.items()}
                            for within in range(cfg.MICRO_BATCH):
                                rows.append(dict(role='final', arm=arm, method=method,
                                    gamma_db=float(gamma), batch_index=bi, within_batch=within,
                                    sample_index=bi*cfg.MICRO_BATCH+within,
                                    **{name: value[within] for name, value in values.items()}))
                        if truth_before != (_tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])):
                            raise ValueError('Offline final scoring mutated simulator truth')
                        runtime.atomic_json(output/'progress.json', dict(method=method, arm=arm,
                            completed_batches=bi+1, total_batches=cfg.FINAL_BATCHES,
                            updated=runtime.now(), neural_updates=0))
                        print(f'{method}/{arm}: final batch {bi+1}/{cfg.FINAL_BATCHES}', flush=True)
                    after = _records_signature(codec, stats, ce)
                    if before != after:
                        raise ValueError('Final inference changed frozen codec/CE/prior state')
                    _check_files(record, fixture=fixture)
                    signatures[method][arm] = dict(before=before, after=after)
                    del codec, stats, ce, data, sample, out
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
        _check_prior_source_relationship(signatures)
        entries = list(fingerprints.values())
        fingerprint_map(entries, device=device)
        _check_past_disjoint(entries, fixture=fixture)
        summary = results.summarize_rows(rows, samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH, batch_size=cfg.MICRO_BATCH)
        comparisons = results.paired_comparisons(rows, samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH, batch_size=cfg.MICRO_BATCH)
        _write_gzip(output/'samples.csv.gz', rows)
        write_union_csv(output/'summary.csv', summary)
        write_union_csv(output/'comparisons.csv', comparisons)
        runtime.atomic_json(output/'input_fingerprints.json', entries, exclusive=True)
        runtime.atomic_json(output/'source_records.json', records, exclusive=True)
        runtime.atomic_json(output/'training_pairing.json', pairing, exclusive=True)
        runtime.guard_core(core)
        manifest.update(status='test_fixture' if fixture else 'passed', data_protocol=protocol,
            source_records=records, frozen_signatures=signatures,
            paired_inputs_verified=True, frozen_models_verified=True,
            source_initialization_matched=True, common_CE_verified=True,
            elapsed_seconds=time.perf_counter()-started,
            files={name: runtime.sha256(output/name) for name in EVALUATION_FILES})
        runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-started)
        runtime.atomic_json(output/'manifest.json', manifest)
        raise


def verify_manifest(directory, *, allow_fixture=False):
    cfg, runtime = _config(), _runtime()
    directory = checked_output(directory)
    manifest = runtime.read_json(directory/'manifest.json')
    fixture = manifest.get('status') == 'test_fixture'
    if fixture and not allow_fixture:
        raise ValueError('CPU fixture cannot serve as production final evidence')
    expected = _base_manifest(fixture, runtime.core_identity())
    expected['status'] = 'test_fixture' if fixture else 'passed'
    for name, value in expected.items():
        actual = manifest.get(name)
        if actual != value or isinstance(value, (bool, int)) and type(actual) is not type(value):
            raise ValueError('Final evaluation identity/budget changed: '+name)
    runtime.guard_core(manifest['core'])
    if set(manifest.get('files', {})) != set(EVALUATION_FILES):
        raise ValueError('Exact final artifact inventory required')
    for name in EVALUATION_FILES:
        path = runtime.inside(directory/name)
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != manifest['files'][name]:
            raise ValueError('Final scalar artifact changed: '+name)
    records = runtime.read_json(directory/'source_records.json')
    if records != manifest.get('source_records') or set(records) != set(cfg.METHODS):
        raise ValueError('Complete source/control/joint model records required')
    if manifest.get('terminal_records') != records.get('joint') or manifest.get('control_records') != records.get('bs_only'):
        raise ValueError('Fixed final terminal/control records changed')
    signatures = manifest.get('frozen_signatures', {})
    if set(signatures) != set(cfg.METHODS):
        raise ValueError('Complete per-model frozen signatures required')
    for method in cfg.METHODS:
        if set(records[method]) != set(cfg.ARMS) or set(signatures[method]) != set(cfg.ARMS):
            raise ValueError('Complete method/arm final evidence required')
        for arm in cfg.ARMS:
            record, state = records[method][arm], signatures[method][arm]
            _check_record(record, arm, method, fixture=fixture)
            if (set(state) != {'before', 'after'} or state['before'] != state['after']
                    or set(state['before']) != {'codec', 'encoder', 'decoder', 'statistics', 'CE', 'CE_table'}
                    or any(not _valid_hash(v) for v in state['before'].values())):
                raise ValueError('Frozen per-model codec/CE/prior signature changed')
            _check_state_record(record, state['before'], fixture=fixture)
            if not fixture and method == 'joint':
                terminal = _trainer().verify_terminal_record(arm)
                if terminal.get('checkpoint_sha256') != record.get('checkpoint_sha256'):
                    raise ValueError('Accepted joint checkpoint identity changed')
                initial = dict(records['source'][arm], method='joint')
                if record.get('source_identity') != initial:
                    raise ValueError('Joint terminal no longer binds its exact pinned source')
    _check_prior_source_relationship(signatures)
    _check_source_bindings(records, fixture=fixture)
    pairing = runtime.read_json(directory/'training_pairing.json')
    if manifest.get('training_pairing') != pairing or pairing != verify_training_pairing(
            records['joint'], records['bs_only'], fixture=fixture):
        raise ValueError('Final matched training-input evidence changed')
    protocol = manifest.get('data_protocol', {})
    device = protocol.get('execution', {}).get('device')
    if protocol != runtime.make_data(device, role='final').protocol:
        raise ValueError('Fresh final data protocol changed')
    entries = runtime.read_json(directory/'input_fingerprints.json')
    fingerprint_map(entries, device=device)
    _check_past_disjoint(entries, fixture=fixture)
    for name in ('paired_inputs_verified', 'frozen_models_verified', 'source_initialization_matched', 'common_CE_verified'):
        if manifest.get(name) is not True:
            raise ValueError('Required final pairing/freezing evidence missing')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        parser.print_help()
        return 0
    if args.output_dir is None:
        parser.error('--run requires --output-dir')
    evaluate(args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
