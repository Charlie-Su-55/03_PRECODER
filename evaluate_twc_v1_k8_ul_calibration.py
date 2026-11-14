#!/usr/bin/env python3
"""Fit common pilot-only CE moments, then score locked frozen receivers.

Actual UL enters only calibration labels and offline diagnostics. Validation
never changes thresholds, moments, neural weights or the historical prior.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
from pathlib import Path
import time

from physical_split_finish_io import write_union_csv
from evaluate_twc_v1_k8_icc_aligned import _tensor_hash

CALIBRATION_FILES = ('model.json', 'cal_samples.csv.gz', 'input_fingerprints.json', 'source_records.json')
VALIDATION_FILES = ('samples.csv.gz', 'summary.csv', 'comparisons.csv',
                    'input_fingerprints.json', 'source_records.json')
FINGERPRINT_FIELDS = frozenset(('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity',
    'gamma', 'wave_active', 'Y_rx', 'nominal_power', 'estimated_UL', 'CE_variance'))


def _config():
    from configs import twc_v1_k8_ul_calibration_config as cfg
    return cfg


def _runtime():
    import ul_calibration_runtime as runtime
    return runtime


def _math():
    import twc_v1_k8_ul_calibration as math
    return math


def _metrics():
    import twc_v1_k8_ul_calibration_metrics as metrics
    return metrics


def _results():
    import twc_v1_k8_ul_calibration_results as results
    return results


def checked_output(path, role):
    cfg, runtime = _config(), _runtime()
    if role not in ('calibration', 'validation'):
        raise ValueError('Declared calibration or validation phase required')
    path = runtime.inside(path)
    subtree = (cfg.OUTPUT_ROOT/role).resolve()
    if path == subtree or not path.is_relative_to(subtree):
        raise ValueError('Proper owned '+role+' attempt required')
    return path


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _manifest(role, fixture, core):
    cfg = _config()
    calibration = role == 'calibration'
    batches = cfg.CALIBRATION_BATCHES if calibration else cfg.VALIDATION_BATCHES
    return dict(format=cfg.NAMESPACE+'_manifest_v1', status='running', phase=role,
        role=role, channel_role=cfg.CHANNEL_ROLES[role], core=core,
        recipe_fingerprint=cfg.fingerprint(), arms=['fdma'] if calibration else list(cfg.ARMS),
        variants=[] if calibration else list(cfg.VARIANTS), gamma_db=list(cfg.GAMMAS),
        batch_size=cfg.MICRO_BATCH, n_batches=batches,
        samples_per_condition=batches*cfg.MICRO_BATCH, optimizer_updates=0,
        UE_updates=0, CE_updates=0, BS_updates=0, EMA_updates=0, test_used=False,
        common_CE_calibration=True, source_checkpoints_verified=not fixture,
        production_evidence=not fixture, no_downlink_forward=calibration,
        calibration_labels_true_UL_only=calibration,
        true_UL_used_for_scoring_only=not calibration)


def _write_gzip(path, rows):
    fields = sorted(set().union(*(set(row) for row in rows)))
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _calibration_rows(path):
    cfg = _config()
    rows, seen = [], set()
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        for original in csv.DictReader(stream):
            row = {name:(value if name == 'role' else int(value) if name in
                    ('sample_index', 'batch_index', 'within_batch', 'n_coordinates') else float(value))
                for name, value in original.items()}
            required = {'role', 'gamma_db', 'sample_index', 'batch_index', 'within_batch', 'n_coordinates'}
            if not required <= set(row):
                raise ValueError('Complete calibration sufficient-statistic identity required')
            key = row['gamma_db'], row['sample_index']
            if (row['role'] != 'calibration' or row['gamma_db'] not in cfg.GAMMAS
                    or type(row['sample_index']) is not int or not 0 <= row['sample_index'] < cfg.CALIBRATION_SAMPLES
                    or key in seen or row['batch_index'] != row['sample_index']//cfg.MICRO_BATCH
                    or row['within_batch'] != row['sample_index'] % cfg.MICRO_BATCH
                    or row['n_coordinates'] != 8*256):
                raise ValueError('Independent calibration MU coverage or geometry changed')
            seen.add(key)
            rows.append(row)
    expected = {(float(g), i) for g in cfg.GAMMAS for i in range(cfg.CALIBRATION_SAMPLES)}
    if seen != expected:
        raise ValueError('Calibration sufficient-statistic coverage incomplete')
    return rows


def validate_vectors(vectors, batch):
    import torch
    if not vectors:
        raise ValueError('Per-MU diagnostic vectors required')
    for name, value in vectors.items():
        if value is None:
            continue
        if (not isinstance(name, str) or not torch.is_tensor(value)
                or tuple(value.shape) != (batch,) or value.requires_grad
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Finite detached complete-MU vector required: '+str(name))


def _fingerprint(sample, observation, gamma, noise, polarity, role, bi, arm, device):
    return dict(role=role, arm=arm, batch_index=bi, gamma_db=float(gamma[0]),
        address=sample['address'], device=str(device),
        true_UL_receiver_input=False,
        fingerprint=dict(H_dl=_tensor_hash(sample['H_dl']), H_ul=_tensor_hash(sample['H_ul']),
            standard_noise=_tensor_hash(noise), pilot_polarity=_tensor_hash(polarity),
            gamma=_tensor_hash(gamma), wave_active=_tensor_hash(observation['wave'].active),
            Y_rx=_tensor_hash(observation['Y']), nominal_power=_tensor_hash(observation['prior']),
            estimated_UL=_tensor_hash(observation['estimated']), CE_variance=_tensor_hash(observation['ce_variance'])))


def _public_hashes(observation):
    return {name: _tensor_hash(observation[field]) for name, field in
        (('Y_rx', 'Y'), ('nominal_power', 'prior'), ('estimated_UL', 'estimated'), ('CE_variance', 'ce_variance'))}


def _recovery_hashes(recovered):
    values = {name: _tensor_hash(recovered[field]) for name, field in
        (('soft', 'soft'), ('variance', 'variance'), ('reliability', 'reliability'),
         ('estimated_UL', 'H_ul_est'), ('Y_rx', 'Y_res'), ('nominal_power', 'prior'))}
    if set(recovered.get('filters', {})) != {'exclusive', 'shared'}:
        raise ValueError('Explicit receiver filter tensors required')
    values.update({'filter_'+name:_tensor_hash(value) for name,value in recovered['filters'].items()})
    return values


def fingerprint_map(entries, *, role, batches=None):
    """Require complete paired inputs and unchanged validation forward inputs."""
    cfg, runtime = _config(), _runtime()
    if role not in cfg.CHANNEL_ROLES:
        raise ValueError('Declared role required')
    if batches is None:
        batches = cfg.CALIBRATION_BATCHES if role == 'calibration' else cfg.VALIDATION_BATCHES
    arms = ('fdma',) if role == 'calibration' else cfg.ARMS
    expected = {(arm, bi, float(gamma)) for arm in arms for bi in range(batches) for gamma in cfg.GAMMAS}
    index, metadata, device = {}, None, None
    for row in entries:
        key = row.get('arm'), row.get('batch_index'), row.get('gamma_db')
        values = row.get('fingerprint', {})
        if (key not in expected or key in index or row.get('role') != role
                or row.get('true_UL_receiver_input') is not False
                or set(values) != FINGERPRINT_FIELDS or any(not _valid_hash(v) for v in values.values())):
            raise ValueError('Incomplete/duplicate calibration input fingerprint')
        if device is None:
            device = row['device']
            metadata = runtime.make_data(device)
        if row.get('device') != device or row.get('address') != metadata.address(cfg.CHANNEL_ROLES[role], key[1]):
            raise ValueError('Independent data role or sample address changed')
        if role == 'validation':
            before, after = row.get('recovery_hashes', {}), row.get('recovery_hashes_after', {})
            if set(before) != set(cfg.VARIANTS) or before != after:
                raise ValueError('Recovered inputs changed during frozen decoder/scoring')
            required = {'soft', 'variance', 'reliability', 'estimated_UL', 'Y_rx', 'nominal_power',
                        'filter_exclusive', 'filter_shared'}
            for variant in cfg.VARIANTS:
                if set(before[variant]) != required or any(not _valid_hash(v) for v in before[variant].values()):
                    raise ValueError('Complete receiver output hashes required')
                if (before[variant]['Y_rx'] != values['Y_rx']
                        or before[variant]['nominal_power'] != values['nominal_power']):
                    raise ValueError('Variant changed the actual observation or prior')
            for name in ('soft', 'estimated_UL', 'Y_rx', 'nominal_power', 'filter_exclusive', 'filter_shared'):
                if before['original'][name] != before['reliability_only'][name]:
                    raise ValueError('Reliability-only control changed receiver inputs')
            if before['original']['estimated_UL'] != values['estimated_UL']:
                raise ValueError('Original receiver did not use original CE')
        index[key] = row
    if set(index) != expected:
        raise ValueError('Incomplete calibration/validation fingerprint coverage')
    if role == 'validation':
        common = ('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma', 'estimated_UL', 'CE_variance')
        for bi in range(batches):
            for gamma in cfg.GAMMAS:
                values = [index[arm, bi, float(gamma)]['fingerprint'] for arm in arms]
                if any(any(v[name] != values[0][name] for name in common) for v in values[1:]):
                    raise ValueError('Physical arms did not share common paid CE/channel/noise inputs')
    return index


def _verify(directory, role, *, allow_fixture=False):
    cfg, runtime = _config(), _runtime()
    directory = checked_output(directory, role)
    manifest = runtime.read_json(directory/'manifest.json')
    fixture = manifest.get('status') == 'test_fixture'
    expected = _manifest(role, fixture, runtime.core_identity())
    expected.update(status='test_fixture' if fixture else 'passed')
    if fixture and not allow_fixture:
        raise ValueError('CPU fixture is not production calibration evidence')
    for name, value in expected.items():
        actual = manifest.get(name)
        if actual != value or isinstance(value, (bool, int)) and type(actual) is not type(value):
            raise ValueError('Calibration phase identity changed: '+name)
    runtime.guard_core(manifest['core'])
    files = CALIBRATION_FILES if role == 'calibration' else VALIDATION_FILES
    if set(manifest.get('files', {})) != set(files):
        raise ValueError('Exact calibration artifact inventory required')
    for name in files:
        path = runtime.inside(directory/name)
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != manifest['files'][name]:
            raise ValueError('Locked calibration artifact bytes changed: '+name)
    records = runtime.read_json(directory/'source_records.json')
    if records != manifest.get('source_records') or set(records) != set(expected['arms']):
        raise ValueError('Calibration source record inventory changed')
    for arm, record in records.items():
        if not fixture:
            runtime.verify_source_record(arm, record)
        signature = manifest.get('frozen_signatures', {}).get(arm, {})
        if (signature.get('before') != signature.get('after')
                or set(signature.get('before', {})) != {'codec', 'statistics', 'CE'}
                or any(not _valid_hash(v) for v in signature['before'].values())
                or signature['before'].get('CE') != manifest.get('common_CE_signature')):
            raise ValueError('Frozen common CE or source states changed')
        if not fixture and any(signature['before'][name] != record.get(field) for name, field in
                (('codec', 'state_sha256'), ('statistics', 'statistics_state_sha256'), ('CE', 'CE_state_sha256'))):
            raise ValueError('Frozen signatures do not bind to source checkpoint states')
    if manifest.get('data_protocol') != runtime.make_data('cuda:0' if not fixture else 'cpu').protocol:
        raise ValueError('Calibration protocol identity changed')
    for name in ('frozen_models_verified', 'paired_inputs_verified'):
        if manifest.get(name) is not True:
            raise ValueError('Required calibration evidence flag missing: '+name)
    entries = runtime.read_json(directory/'input_fingerprints.json')
    fingerprint_map(entries, role=role)
    if role == 'calibration':
        model = _math().validate_model(runtime.read_json(directory/'model.json'))
        if runtime.sha256(directory/'model.json') != manifest.get('model_sha256'):
            raise ValueError('Locked fitted model SHA changed')
        if manifest.get('calibration_only_fit') is not True:
            raise ValueError('Calibration-only fitting required')
        recomputed = _math().fit_model(_calibration_rows(directory/'cal_samples.csv.gz'))
        if recomputed != model:
            raise ValueError('Fitted model does not derive from locked calibration-only statistics')
    else:
        caldir = checked_output(manifest.get('calibration_dir', ''), 'calibration')
        calibration = verify_calibration(caldir, allow_fixture=allow_fixture)
        if (runtime.sha256(caldir/'manifest.json') != manifest.get('calibration_manifest_sha256')
                or manifest.get('model_sha256') != calibration['model_sha256']
                or manifest.get('model_sha256_before') != calibration['model_sha256']
                or manifest.get('model_sha256_after') != calibration['model_sha256']
                or manifest.get('model_frozen') is not True
                or manifest.get('common_CE_signature') != calibration['common_CE_signature']):
            raise ValueError('Validation calibration/model lock changed')
        _disjoint(runtime.read_json(caldir/'input_fingerprints.json'), entries)
    return manifest


def verify_calibration(directory, *, allow_fixture=False):
    return _verify(directory, 'calibration', allow_fixture=allow_fixture)


def verify_validation(directory, *, allow_fixture=False):
    return _verify(directory, 'validation', allow_fixture=allow_fixture)


def _disjoint(calibration, validation):
    for name in ('H_dl', 'H_ul'):
        before = {r['fingerprint'][name] for r in calibration}
        after = {r['fingerprint'][name] for r in validation}
        if before.intersection(after):
            raise ValueError('Calibration and validation channel payloads overlap')
    addresses = {r['address']['sionna_seed'] for r in calibration}
    if addresses.intersection(r['address']['sionna_seed'] for r in validation):
        raise ValueError('Calibration and validation data addresses overlap')


def _context(require_cuda, loader, data_factory):
    import torch
    runtime = _runtime()
    fixture = not require_cuda or loader is not None or data_factory is not None
    device = runtime.configure_cuda() if require_cuda else torch.device('cpu')
    return fixture, device, loader or runtime.load_source, data_factory or runtime.make_data


def _data(data_factory, device, protocol=None):
    runtime = _runtime()
    data, metadata = data_factory(device), runtime.make_data(device)
    if data.protocol != metadata.protocol or protocol is not None and data.protocol != protocol:
        raise ValueError('Calibration data factory protocol changed')
    return data, metadata


def _sample(data, metadata, role, bi):
    cfg = _config()
    sample = data.sample(cfg.CHANNEL_ROLES[role], bi)
    if sample.get('address') != metadata.address(cfg.CHANNEL_ROLES[role], bi):
        raise ValueError('Calibration/validation channel address changed')
    return sample


def calibrate(output_dir, *, require_cuda=True, loader=None, data_factory=None):
    import torch
    cfg, runtime, math = _config(), _runtime(), _math()
    output = checked_output(output_dir, 'calibration')
    core = runtime.core_identity()
    runtime.guard_core(core)
    fixture, device, loader, data_factory = _context(require_cuda, loader, data_factory)
    output.mkdir(parents=True, exist_ok=False)
    started, rows, fingerprints = time.perf_counter(), [], []
    manifest = _manifest('calibration', fixture, core)
    try:
        with torch.inference_mode():
            codec, stats, ce, record = loader('fdma', device)
            if not fixture:
                runtime.verify_source_record('fdma', record)
            before = runtime.frozen_signature(codec, stats, ce)
            data, metadata = _data(data_factory, device)
            for bi in range(cfg.CALIBRATION_BATCHES):
                sample = _sample(data, metadata, 'calibration', bi)
                sample_hashes = _tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])
                raw = codec.encoder(sample['H_dl'])
                for gi, gamma in enumerate(cfg.GAMMAS):
                    gv = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
                    noise, polarity = runtime.batch_random('calibration', bi, gi, device)
                    observation = runtime.observe(raw, stats, ce, sample, gv, noise, polarity)
                    feature = math.pilot_feature(observation['Y'], observation['wave'].pilot, observation['estimated'], gv)
                    records = math.calibration_statistics(observation['estimated'], sample['H_ul'],
                        observation['ce_variance'], feature, gv)
                    if len(records) != cfg.MICRO_BATCH:
                        raise ValueError('One sufficient-statistic record per complete MU required')
                    for within, row in enumerate(records):
                        rows.append(dict(row, batch_index=bi, within_batch=within,
                            sample_index=bi*cfg.MICRO_BATCH+within))
                    fingerprints.append(_fingerprint(sample, observation, gv, noise, polarity,
                        'calibration', bi, 'fdma', device))
                if sample_hashes != (_tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])):
                    raise ValueError('Calibration label construction mutated channel inputs')
                runtime.atomic_json(output/'progress.json', dict(role='calibration', batch=bi+1,
                    total_batches=cfg.CALIBRATION_BATCHES, updated=runtime.now(), optimizer_updates=0))
                print(f'calibration: batch {bi+1}/{cfg.CALIBRATION_BATCHES}', flush=True)
            model = math.fit_model(rows)
            math.validate_model(model)
            after = runtime.frozen_signature(codec, stats, ce)
            if before != after:
                raise ValueError('Frozen calibration model/prior changed')
            if not fixture:
                runtime.verify_source_record('fdma', record)
        fingerprint_map(fingerprints, role='calibration')
        runtime.atomic_json(output/'model.json', model, exclusive=True)
        _write_gzip(output/'cal_samples.csv.gz', rows)
        runtime.atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
        runtime.atomic_json(output/'source_records.json', {'fdma': record}, exclusive=True)
        runtime.guard_core(core)
        manifest.update(status='test_fixture' if fixture else 'passed', calibration_only_fit=True,
            frozen_models_verified=True, paired_inputs_verified=True,
            common_CE_signature=before['CE'], model_sha256=runtime.sha256(output/'model.json'),
            frozen_signatures={'fdma':dict(before=before, after=after)}, source_records={'fdma':record},
            data_protocol=data.protocol, elapsed_seconds=time.perf_counter()-started,
            files={name: runtime.sha256(output/name) for name in CALIBRATION_FILES})
        runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}',
            elapsed_seconds=time.perf_counter()-started)
        runtime.atomic_json(output/'manifest.json', manifest)
        raise


def _decode(codec, recovered, observation, layout, gamma, truth):
    import torch
    from evaluate_twc_v1_k8_icc_rate import metric_vectors
    out = codec.decoder(recovered['Y_res'], recovered['H_ul_est'], recovered['soft'],
        recovered['reliability'], observation['wave'].pilot, gamma_db=gamma,
        layout=layout, prior=recovered['prior'], ce_error=recovered['ce_variance'])
    score = dict(out, soft=recovered['soft'], H_ul_est=recovered['H_ul_est'], waveform=observation['wave'])
    vectors = metric_vectors(score, truth)
    return {name: torch.as_tensor(value, dtype=torch.float32, device=gamma.device).detach()
            for name, value in vectors.items()}


def evaluate(output_dir, calibration_dir, *, require_cuda=True, loader=None, data_factory=None):
    import torch
    cfg, runtime, math, metrics, results = _config(), _runtime(), _math(), _metrics(), _results()
    output = checked_output(output_dir, 'validation')
    # Verify all calibration bytes before creating any validation data object.
    fixture = not require_cuda or loader is not None or data_factory is not None
    calibration_dir = checked_output(calibration_dir, 'calibration')
    calibration = verify_calibration(calibration_dir, allow_fixture=fixture)
    calibration_manifest_sha = runtime.sha256(calibration_dir/'manifest.json')
    locked_sha = calibration['model_sha256']
    model = math.validate_model(runtime.read_json(calibration_dir/'model.json'))
    model_before = cfg.plain(model)
    calibration_inputs = runtime.read_json(calibration_dir/'input_fingerprints.json')
    calibration_channels = {name:{row['fingerprint'][name] for row in calibration_inputs}
        for name in ('H_dl', 'H_ul')}
    core = runtime.core_identity()
    runtime.guard_core(core)
    fixture, device, loader, data_factory = _context(require_cuda, loader, data_factory)
    output.mkdir(parents=True, exist_ok=False)
    started, rows, fingerprints, records, signatures = time.perf_counter(), [], [], {}, {}
    manifest = _manifest('validation', fixture, core)
    manifest.update(calibration_dir=str(calibration_dir), calibration_manifest_sha256=calibration_manifest_sha,
        model_sha256=locked_sha, model_sha256_before=locked_sha)
    protocol = None
    try:
        with torch.inference_mode():
            for arm in cfg.ARMS:
                codec, stats, ce, record = loader(arm, device)
                if not fixture:
                    runtime.verify_source_record(arm, record)
                before = runtime.frozen_signature(codec, stats, ce)
                if before['CE'] != calibration['common_CE_signature']:
                    raise ValueError('Common CE differs from fitted calibration source')
                records[arm] = record
                data, metadata = _data(data_factory, device, protocol)
                protocol = data.protocol
                for bi in range(cfg.VALIDATION_BATCHES):
                    sample = _sample(data, metadata, 'validation', bi)
                    sample_hashes = _tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])
                    if any(value in calibration_channels[name]
                            for name, value in zip(('H_dl', 'H_ul'), sample_hashes)):
                        raise ValueError('Calibration and validation channel payloads overlap')
                    raw = codec.encoder(sample['H_dl'])
                    for gi, gamma in enumerate(cfg.GAMMAS):
                        if runtime.sha256(calibration_dir/'model.json') != locked_sha:
                            raise ValueError('Frozen calibration model changed before validation')
                        if model != model_before:
                            raise ValueError('Frozen in-memory calibration model changed before validation')
                        gv = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
                        noise, polarity = runtime.batch_random('validation', bi, gi, device)
                        observation = runtime.observe(raw, stats, ce, sample, gv, noise, polarity)
                        entry = _fingerprint(sample, observation, gv, noise, polarity, 'validation', bi, arm, device)
                        public_before = _public_hashes(observation)
                        public_observation = {name: observation[name]
                            for name in ('Y', 'estimated', 'ce_variance', 'prior')}
                        public_observation['pilots'] = observation['wave'].pilot
                        recovered = math.recover_variants(public_observation, stats.layout, gv, model)
                        if model != model_before:
                            raise ValueError('Public recovery changed the frozen in-memory calibration model')
                        if set(recovered) != set(cfg.VARIANTS):
                            raise ValueError('All three prespecified receiver variants required')
                        entry['recovery_hashes'] = {v:_recovery_hashes(recovered[v]) for v in cfg.VARIANTS}
                        diagnostics = metrics.diagnostic_vectors(observation['estimated'], sample['H_ul'],
                            observation['ce_variance'], observation['Y'], observation['wave'].pilot, gv, stats.layout)
                        for variant in cfg.VARIANTS:
                            # These characterize the unchanged source CE and
                            # geometry, not the corrected mean of this variant.
                            vectors = {'source_'+name:value for name,value in diagnostics.items()}
                            effective_error = (recovered[variant]['H_ul_est']-sample['H_ul']).abs().square()
                            truth_power = sample['H_ul'].abs().square().reshape(cfg.MICRO_BATCH, -1).sum(-1)
                            vectors['effective_ce_nmse_linear'] = (
                                effective_error.reshape(cfg.MICRO_BATCH, -1).sum(-1)/truth_power.clamp_min(1e-20)).detach()
                            vectors.update(metrics.recovery_vectors(recovered[variant], observation['wave'].active, stats.layout))
                            downlink = _decode(codec, recovered[variant], observation, stats.layout, gv, sample['H_dl'])
                            for name, value in downlink.items():
                                if name in vectors and vectors[name] is not None and not torch.allclose(vectors[name], value, rtol=2e-5, atol=1e-7):
                                    raise ValueError('Inconsistent duplicate receiver/downlink metric: '+name)
                                vectors[name] = value
                            validate_vectors(vectors, cfg.MICRO_BATCH)
                            values = {name:(value.detach().cpu().tolist() if value is not None else [None]*cfg.MICRO_BATCH)
                                      for name, value in vectors.items()}
                            for within in range(cfg.MICRO_BATCH):
                                rows.append(dict(role='validation', arm=arm, variant=variant,
                                    gamma_db=float(gamma), batch_index=bi, within_batch=within,
                                    sample_index=bi*cfg.MICRO_BATCH+within,
                                    **{name:value[within] for name,value in values.items()}))
                        entry['recovery_hashes_after'] = {v:_recovery_hashes(recovered[v]) for v in cfg.VARIANTS}
                        if entry['recovery_hashes'] != entry['recovery_hashes_after'] or public_before != _public_hashes(observation):
                            raise ValueError('Offline diagnostics/decoder changed receiver observations')
                        fingerprints.append(entry)
                    if sample_hashes != (_tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])):
                        raise ValueError('Validation scoring mutated simulator truth')
                    runtime.atomic_json(output/'progress.json', dict(role='validation', arm=arm,
                        completed_conditions=len(fingerprints), total_conditions=len(cfg.ARMS)*cfg.VALIDATION_BATCHES*len(cfg.GAMMAS),
                        updated=runtime.now(), optimizer_updates=0))
                    print(f'{arm}: validation batch {bi+1}/{cfg.VALIDATION_BATCHES}', flush=True)
                after = runtime.frozen_signature(codec, stats, ce)
                if before != after:
                    raise ValueError('Frozen validation model/prior changed')
                signatures[arm] = dict(before=before, after=after)
                if not fixture:
                    runtime.verify_source_record(arm, record)
                del codec, stats, ce, data, sample, raw, observation, recovered
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        fingerprint_map(fingerprints, role='validation')
        _disjoint(calibration_inputs, fingerprints)
        # Revalidate the complete calibration stage, not only model.json.
        verify_calibration(calibration_dir, allow_fixture=fixture)
        if runtime.sha256(calibration_dir/'manifest.json') != calibration_manifest_sha:
            raise ValueError('Calibration evidence changed during validation')
        if runtime.sha256(calibration_dir/'model.json') != locked_sha:
            raise ValueError('Validation modified the fitted calibration model')
        summary = results.summarize_rows(rows, samples=cfg.VALIDATION_SAMPLES, batch_size=cfg.MICRO_BATCH)
        comparisons = results.paired_comparisons(rows, samples=cfg.VALIDATION_SAMPLES, batch_size=cfg.MICRO_BATCH)
        _write_gzip(output/'samples.csv.gz', rows)
        write_union_csv(output/'summary.csv', summary)
        write_union_csv(output/'comparisons.csv', comparisons)
        runtime.atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
        runtime.atomic_json(output/'source_records.json', records, exclusive=True)
        runtime.guard_core(core)
        if not fixture:
            runtime.verify_source_records(records)
        manifest.update(status='test_fixture' if fixture else 'passed',
            frozen_models_verified=True, paired_inputs_verified=True, independent_roles_verified=True,
            common_CE_signature=calibration['common_CE_signature'], model_frozen=True,
            model_sha256_after=runtime.sha256(calibration_dir/'model.json'),
            frozen_signatures=signatures, source_records=records, data_protocol=protocol,
            elapsed_seconds=time.perf_counter()-started,
            files={name:runtime.sha256(output/name) for name in VALIDATION_FILES})
        runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}',
            elapsed_seconds=time.perf_counter()-started)
        runtime.atomic_json(output/'manifest.json', manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--calibrate', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path)
    args = parser.parse_args(argv)
    if args.calibrate:
        if args.calibration_dir is not None:
            parser.error('--calibration-dir is only valid with --evaluate')
        calibrate(args.output_dir)
    else:
        if args.calibration_dir is None:
            parser.error('--evaluate requires --calibration-dir')
        evaluate(args.output_dir, args.calibration_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
