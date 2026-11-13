#!/usr/bin/env python3
"""Score normal-filter conditional error propagation without changing recovery."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path
import time

from physical_split_finish_io import write_union_csv
from evaluate_twc_v1_k8_icc_aligned import _tensor_hash


def _config():
    from configs import twc_v1_k8_ul_propagation_config as cfg
    return cfg


def _runtime():
    import ul_propagation_runtime as runtime
    return runtime


def _metrics():
    import twc_v1_k8_ul_propagation_metrics as metrics
    return metrics


def _results():
    import twc_v1_k8_ul_propagation_results as results
    return results


def checked_output(path):
    runtime, cfg = _runtime(), _config()
    path = runtime.inside(path)
    subtree = (cfg.OUTPUT_ROOT/'evaluation').resolve()
    if path == subtree or not path.is_relative_to(subtree):
        raise ValueError('Proper own propagation evaluation child required')
    return path


def validate_vectors(vectors, batch):
    import torch
    required = _results().REQUIRED_METRICS
    if not vectors or not set(required) <= set(vectors):
        raise ValueError('Actual, reported and propagated per-MU errors required')
    for name, value in vectors.items():
        if (not torch.is_tensor(value) or value.shape != (batch,)
                or value.requires_grad or not bool(torch.isfinite(value).all())):
            raise ValueError('Finite detached per-MU propagation scalar required: '+name)


def _valid_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def fingerprint_map(entries, *, batches):
    """Validate actual paired inputs and unchanged normal recovery artifacts."""
    cfg, runtime = _config(), _runtime()
    expected = {(arm, bi, float(gamma)) for arm in cfg.ARMS
                for bi in range(batches) for gamma in cfg.GAMMAS}
    fields = {'H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma',
              'wave_active', 'Y_rx', 'nominal_power', 'estimated_UL', 'CE_variance'}
    index, metadata, device = {}, None, None
    for row in entries:
        key = (row['arm'], row['batch_index'], float(row['gamma_db']))
        hashes = row['fingerprint']
        if key not in expected or key in index or set(hashes) != fields:
            raise ValueError('Duplicate/incomplete propagation input fingerprint')
        if any(not _valid_hash(value) for value in hashes.values()):
            raise ValueError('Actual propagation-input tensor SHA256 required')
        if device is None:
            device = row['device']
            metadata = runtime.make_data(device)
        if (row['device'] != device
                or row['address'] != metadata.address(cfg.CHANNEL_ROLE, row['batch_index'])):
            raise ValueError('Propagation input address/device changed')
        if (row.get('true_UL_used_to_design_filter') is not False
                or row.get('same_filter_for_all_scoring') is not True):
            raise ValueError('True channel must not redesign the normal receiver')
        for before, after in (('filter_hashes', 'filter_hashes_after'),
                              ('recovery_hashes', 'recovery_hashes_after')):
            required = {'exclusive', 'shared'} if before == 'filter_hashes' else {'soft', 'variance', 'Y_rx', 'prior'}
            values = row.get(before, {})
            if (set(values) != required or any(not _valid_hash(value) for value in values.values())
                    or values != row.get(after)):
                raise ValueError('Normal filter/soft/variance changed during offline scoring')
        if (row['recovery_hashes']['Y_rx'] != hashes['Y_rx']
                or row['recovery_hashes']['prior'] != hashes['nominal_power']):
            raise ValueError('Normal recovery did not retain the actual observation/prior')
        index[key] = row
    if set(index) != expected:
        raise ValueError('Incomplete propagation fingerprint coverage')
    common = ('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma', 'estimated_UL', 'CE_variance')
    for bi in range(batches):
        for gamma in cfg.GAMMAS:
            paired = [index[arm, bi, float(gamma)]['fingerprint'] for arm in cfg.ARMS]
            if any(any(value[name] != paired[0][name] for name in common) for value in paired[1:]):
                raise ValueError('Physical arms did not use paired channel/noise/pilot/CE inputs')
    return index


def evaluate(output_dir, *, require_cuda=True, loader=None, data_factory=None):
    """Production is CUDA-only; injected CPU checks are explicitly nonproduction."""
    import torch
    cfg, runtime, metrics, results = _config(), _runtime(), _metrics(), _results()
    output = checked_output(output_dir)
    core = runtime.core_identity()
    runtime.guard_core(core)
    fixture = not require_cuda or loader is not None or data_factory is not None
    device = runtime.configure_cuda() if require_cuda else torch.device('cpu')
    loader = runtime.load_source if loader is None else loader
    data_factory = runtime.make_data if data_factory is None else data_factory
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = dict(format=cfg.NAMESPACE+'_manifest_v1', phase='propagation', status='running',
        core=core, recipe_fingerprint=cfg.fingerprint(), arms=list(cfg.ARMS),
        gamma_db=list(cfg.GAMMAS), batch_size=cfg.MICRO_BATCH, n_batches=cfg.BATCHES,
        samples_per_condition=cfg.SAMPLES, optimizer_updates=0, UE_updates=0,
        CE_updates=0, EMA_updates=0, test_used=False, no_downlink_forward=True,
        source_checkpoints_verified=not fixture, production_evidence=not fixture,
        true_UL_used_for_scoring_only=True)
    rows, fingerprints, bins, records, signatures = [], [], [], {}, {}
    protocol, completed = None, 0
    total = len(cfg.ARMS)*cfg.BATCHES*len(cfg.GAMMAS)
    maximum_checks = {}
    try:
        with torch.inference_mode():
            for arm in cfg.ARMS:
                codec, stats, ce, record = loader(arm, device)
                if not fixture:
                    runtime.verify_source_record(arm, record)
                before = runtime.frozen_signature(codec, stats, ce)
                records[arm] = record
                data, metadata = data_factory(device), runtime.make_data(device)
                if data.protocol != metadata.protocol or protocol is not None and data.protocol != protocol:
                    raise ValueError('Propagation data factory/protocol changed')
                protocol = data.protocol
                for bi in range(cfg.BATCHES):
                    sample = data.sample(cfg.CHANNEL_ROLE, bi)
                    if sample.get('address') != metadata.address(cfg.CHANNEL_ROLE, bi):
                        raise ValueError('Propagation channel address changed')
                    hdl, hul = _tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])
                    raw = codec.encoder(sample['H_dl'])
                    for gi, gamma in enumerate(cfg.GAMMAS):
                        gamma_vector = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
                        noise, polarity = runtime.batch_random(bi, gi, device)
                        observation = runtime.observe(raw, stats, ce, sample, gamma_vector, noise, polarity)
                        # Only this normal receiver is executed. Truth never enters its inputs.
                        recovered = runtime.physics().recover(observation['Y'], observation['estimated'],
                            stats.layout, observation['prior'], observation['ce_variance'], gamma_vector)
                        recovery_hashes = {name: _tensor_hash(recovered[field]) for name, field in
                            (('soft', 'soft'), ('variance', 'variance'), ('Y_rx', 'Y_res'), ('prior', 'prior'))}
                        # Rebuild/hash the same public normal filter before any truth scoring.
                        filters = metrics.reconstruct_normal_filters(observation, recovered, stats.layout)
                        filter_hashes = {name: _tensor_hash(value) for name, value in filters.items()}
                        analysis = metrics.analyze(observation, recovered, sample['H_ul'], noise,
                            stats.layout, gamma_vector)
                        after_filters = {name: _tensor_hash(value) for name, value in analysis['filters'].items()}
                        after_recovery = {name: _tensor_hash(recovered[field]) for name, field in
                            (('soft', 'soft'), ('variance', 'variance'), ('Y_rx', 'Y_res'), ('prior', 'prior'))}
                        if filter_hashes != after_filters or recovery_hashes != after_recovery:
                            raise ValueError('Offline scoring changed the original filter or recovery')
                        vectors = analysis['vectors']
                        validate_vectors(vectors, cfg.MICRO_BATCH)
                        values = {name: value.detach().cpu().tolist() for name, value in vectors.items()}
                        for within in range(cfg.MICRO_BATCH):
                            rows.append(dict(arm=arm, gamma_db=float(gamma), batch_index=bi,
                                within_batch=within, sample_index=bi*cfg.MICRO_BATCH+within,
                                **{name: value[within] for name, value in values.items()}))
                        for entry in metrics.calibration_bins(analysis, observation['prior'],
                                stats.layout, edges=cfg.CALIBRATION_EDGES):
                            bins.append(dict(entry, arm=arm, gamma_db=float(gamma), batch_index=bi))
                        for name, value in analysis['checks'].items():
                            maximum_checks[name] = max(maximum_checks.get(name, 0.), float(value.max()))
                        fingerprints.append(dict(arm=arm, batch_index=bi, gamma_db=float(gamma),
                            address=sample['address'], device=str(device),
                            true_UL_used_to_design_filter=False, same_filter_for_all_scoring=True,
                            fingerprint=dict(H_dl=hdl, H_ul=hul, standard_noise=_tensor_hash(noise),
                                pilot_polarity=_tensor_hash(polarity), gamma=_tensor_hash(gamma_vector),
                                wave_active=_tensor_hash(observation['wave'].active),
                                Y_rx=_tensor_hash(observation['Y']), nominal_power=_tensor_hash(observation['prior']),
                                estimated_UL=_tensor_hash(observation['estimated']),
                                CE_variance=_tensor_hash(observation['ce_variance'])),
                            filter_hashes=filter_hashes, filter_hashes_after=after_filters,
                            recovery_hashes=recovery_hashes, recovery_hashes_after=after_recovery))
                        completed += 1
                    if _tensor_hash(sample['H_dl']) != hdl or _tensor_hash(sample['H_ul']) != hul:
                        raise ValueError('Offline scoring mutated true input channels')
                    runtime.atomic_json(output/'progress.json', dict(updated=runtime.now(), arm=arm,
                        completed_conditions=completed, total_conditions=total, optimizer_updates=0))
                    print(f'{arm}: batch {bi+1}/{cfg.BATCHES}; conditions {completed}/{total}', flush=True)
                after = runtime.frozen_signature(codec, stats, ce)
                if before != after:
                    raise ValueError('Frozen source model or prior changed')
                signatures[arm] = dict(before=before, after=after)
                if not fixture:
                    runtime.verify_source_record(arm, record)
                del codec, stats, ce, data, sample, raw, observation, recovered, analysis, filters
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        fingerprint_map(fingerprints, batches=cfg.BATCHES)
        summary = results.summarize_rows(rows, samples=cfg.SAMPLES, batch_size=cfg.MICRO_BATCH)
        comparisons = results.paired_comparisons(rows, samples=cfg.SAMPLES, batch_size=cfg.MICRO_BATCH)
        calibration = results.aggregate_calibration_bins(bins)
        write_union_csv(output/'summary.csv', summary)
        write_union_csv(output/'comparisons.csv', comparisons)
        import csv
        for name, values in (('samples.csv.gz', rows), ('calibration_bins.csv.gz', calibration)):
            with gzip.open(output/name, 'wt', encoding='utf-8', newline='') as stream:
                fields = sorted(set().union(*(set(row) for row in values)))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)
        runtime.atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
        runtime.atomic_json(output/'source_records.json', records, exclusive=True)
        files = ('samples.csv.gz', 'summary.csv', 'comparisons.csv', 'calibration_bins.csv.gz',
                 'input_fingerprints.json', 'source_records.json')
        runtime.guard_core(core)
        if not fixture:
            runtime.verify_source_records(records)
        manifest.update(status='test_fixture' if fixture else 'passed', paired_inputs_verified=True,
            frozen_models_verified=True, original_filter_unchanged=True,
            filter_reconstruction_verified=True, error_decomposition_verified=True,
            reconstruction_max_errors=maximum_checks, frozen_signatures=signatures,
            source_records=records, data_protocol=protocol,
            elapsed_seconds=time.perf_counter()-started, files={name: runtime.sha256(output/name) for name in files})
        runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}',
            elapsed_seconds=time.perf_counter()-started)
        runtime.atomic_json(output/'manifest.json', manifest)
        raise


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
