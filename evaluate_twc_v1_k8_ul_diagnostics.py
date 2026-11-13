#!/usr/bin/env python3
"""Frozen-codeword receiver diagnostics, without a BS forward or updates.

The true-UL condition is privileged offline sensitivity analysis. Both
conditions receive exactly one paid waveform and one noisy observation.
"""
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
    from configs import twc_v1_k8_ul_diagnostics_config as cfg
    return cfg


def _runtime():
    import ul_diagnostics_runtime as runtime
    return runtime


def _metrics():
    import twc_v1_k8_ul_diagnostics_metrics as metrics
    return metrics


def _results():
    import twc_v1_k8_ul_diagnostics_results as results
    return results


def checked_output(path):
    runtime, cfg = _runtime(), _config()
    path = runtime.inside(path)
    subtree = (cfg.OUTPUT_ROOT/'evaluation').resolve()
    if path == subtree or not path.is_relative_to(subtree):
        raise ValueError('Proper own diagnostic evaluation child required')
    return path


def validate_vectors(vectors, batch):
    import torch
    if not vectors or 'soft_nmse_linear' not in vectors:
        raise ValueError('Complete active-codeword soft NMSE required')
    for name, value in vectors.items():
        if (not torch.is_tensor(value) or value.shape != (batch,)
                or value.requires_grad or not bool(torch.isfinite(value).all())):
            raise ValueError('Finite detached per-MU diagnostic required: '+name)


def fingerprint_map(entries, *, batches):
    cfg = _config()
    expected = {(arm, variant, bi, float(gamma)) for arm in cfg.ARMS
        for variant in cfg.VARIANTS for bi in range(batches) for gamma in cfg.GAMMAS}
    keys = {'H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma',
            'wave_active', 'Y_rx', 'nominal_power', 'estimated_UL', 'calibrated_CE_variance'}
    index = {}
    metadata = _runtime().make_data('cuda:0')
    protocol_device = None
    for row in entries:
        key = (row['arm'], row['variant'], row['batch_index'], float(row['gamma_db']))
        hashes = row['fingerprint']
        if key not in expected or key in index or set(hashes) != keys:
            raise ValueError('Duplicate/incomplete diagnostic input fingerprint')
        if protocol_device is None:
            protocol_device = row['device']
            metadata = _runtime().make_data(protocol_device)
        if (row['device'] != protocol_device
                or row['address'] != metadata.address(cfg.CHANNEL_ROLE, row['batch_index'])):
            raise ValueError('Diagnostic input address/device changed')
        if any(not isinstance(value, str) or len(value) != 64
               or any(char not in '0123456789abcdef' for char in value) for value in hashes.values()):
            raise ValueError('Actual tensor SHA256 required')
        recovery = row['recovery_inputs']
        if set(recovery) != {'channel', 'ce_variance', 'Y_rx', 'nominal_power'}:
            raise ValueError('Explicit recovery-input hashes required')
        if any(not isinstance(value, str) or len(value) != 64
               or any(char not in '0123456789abcdef' for char in value) for value in recovery.values()):
            raise ValueError('Actual recovery-input SHA256 required')
        channel_key = 'estimated_UL' if row['variant'] == 'estimated_ul' else 'H_ul'
        if (recovery['channel'] != hashes[channel_key] or recovery['Y_rx'] != hashes['Y_rx']
                or recovery['nominal_power'] != hashes['nominal_power']):
            raise ValueError('Diagnostic recovery input boundary changed')
        if row['variant'] == 'estimated_ul':
            if recovery['ce_variance'] != hashes['calibrated_CE_variance']:
                raise ValueError('Normal CE variance changed')
        elif row.get('receiver_ce_variance_max') != 0.:
            raise ValueError('Privileged true-UL condition requires zero CE variance')
        index[key] = row
    if set(index) != expected:
        raise ValueError('Incomplete diagnostic input coverage')
    for bi in range(batches):
        for gamma in cfg.GAMMAS:
            references = []
            for arm in cfg.ARMS:
                rows = [index[arm, variant, bi, float(gamma)] for variant in cfg.VARIANTS]
                if rows[0]['fingerprint'] != rows[1]['fingerprint']:
                    raise ValueError('Diagnostic variants did not reuse the same observation/codeword')
                references.append(rows[0]['fingerprint'])
            shared = ('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma')
            if any(any(row[name] != references[0][name] for name in shared) for row in references[1:]):
                raise ValueError('Physical arms used different channel/noise/pilot inputs')
    return index


def evaluate(output_dir, *, require_cuda=True, loader=None, data_factory=None):
    """Production requires real CUDA/pinned sources; CPU injection is labelled.

    CPU fixtures exercise the real waveform, Fourier LS, CE module, recovery
    and statistics. Their manifest status is never accepted by the runner.
    """
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
    manifest = dict(format=cfg.NAMESPACE+'_manifest_v1', phase='diagnostic', status='running',
        core=core, recipe_fingerprint=cfg.fingerprint(), arms=list(cfg.ARMS),
        variants=list(cfg.VARIANTS), gamma_db=list(cfg.GAMMAS), batch_size=cfg.MICRO_BATCH,
        n_batches=cfg.BATCHES, samples_per_condition=cfg.SAMPLES,
        optimizer_updates=0, UE_updates=0, CE_updates=0, EMA_updates=0,
        test_used=False, privileged_truth_for_diagnostics_only=True,
        no_downlink_forward=True, source_checkpoints_verified=not fixture,
        production_evidence=not fixture)
    rows, fingerprints, bins, source_records, signatures = [], [], [], {}, {}
    protocol = None
    completed, total = 0, len(cfg.ARMS)*cfg.BATCHES*len(cfg.GAMMAS)
    try:
        with torch.inference_mode():
            for arm in cfg.ARMS:
                codec, stats, ce, record = loader(arm, device)
                if not fixture:
                    runtime.verify_source_record(arm, record)
                before = runtime.frozen_signature(codec, stats, ce)
                source_records[arm] = record
                data, metadata = data_factory(device), runtime.make_data(device)
                if data.protocol != metadata.protocol:
                    raise ValueError('Diagnostic data factory changed the declared protocol')
                if protocol is not None and data.protocol != protocol:
                    raise ValueError('Diagnostic arms changed the data protocol')
                protocol = data.protocol
                for bi in range(cfg.BATCHES):
                    sample = data.sample(cfg.CHANNEL_ROLE, bi)
                    if sample.get('address') != metadata.address(cfg.CHANNEL_ROLE, bi):
                        raise ValueError('Diagnostic data address changed')
                    hdl, hul = _tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])
                    raw = codec.encoder(sample['H_dl'])
                    for gi, gamma in enumerate(cfg.GAMMAS):
                        gamma_vector = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
                        noise, polarity = runtime.batch_random(bi, gi, device)
                        observation = runtime.observe(raw, stats, ce, sample, gamma_vector, noise, polarity)
                        variants, variant_inputs = runtime.recover_variants(
                            observation, sample['H_ul'], stats.layout, gamma_vector)
                        common = dict(H_dl=hdl, H_ul=hul, standard_noise=_tensor_hash(noise),
                            pilot_polarity=_tensor_hash(polarity), gamma=_tensor_hash(gamma_vector),
                            wave_active=_tensor_hash(observation['wave'].active),
                            Y_rx=_tensor_hash(observation['Y']), nominal_power=_tensor_hash(observation['prior']),
                            estimated_UL=_tensor_hash(observation['estimated']),
                            calibrated_CE_variance=_tensor_hash(observation['ce_variance']))
                        for variant in cfg.VARIANTS:
                            recovered = variants[variant]
                            channel, variance = variant_inputs[variant]
                            vectors = metrics.recovery_metrics(observation['wave'], recovered,
                                observation['prior'], stats.layout)
                            vectors.update(metrics.channel_metrics(channel, sample['H_ul'], variance,
                                stats.layout, observation['prior']))
                            validate_vectors(vectors, cfg.MICRO_BATCH)
                            values = {name: value.detach().cpu().tolist() for name, value in vectors.items()}
                            for within in range(cfg.MICRO_BATCH):
                                rows.append(dict(arm=arm, variant=variant, gamma_db=float(gamma),
                                    batch_index=bi, within_batch=within,
                                    sample_index=bi*cfg.MICRO_BATCH+within,
                                    **{name: value[within] for name, value in values.items()}))
                            for kind, entries in (
                                ('recovery', metrics.recovery_calibration_bins(observation['wave'], recovered,
                                    observation['prior'], stats.layout, edges=cfg.CALIBRATION_EDGES)),
                                ('channel', metrics.channel_calibration_bins(channel, sample['H_ul'], variance,
                                    stats.layout, edges=cfg.CALIBRATION_EDGES))):
                                for entry in entries:
                                    bins.append(dict(entry, arm=arm, variant=variant, gamma_db=float(gamma),
                                        batch_index=bi, component=kind+'_'+entry['component']))
                            fingerprints.append(dict(arm=arm, variant=variant, batch_index=bi,
                                gamma_db=float(gamma), address=sample['address'], device=str(device),
                                fingerprint=dict(common), receiver_ce_variance_max=float(variance.max()),
                                recovery_inputs=dict(channel=_tensor_hash(channel),
                                    ce_variance=_tensor_hash(variance), Y_rx=_tensor_hash(recovered['Y_res']),
                                    nominal_power=_tensor_hash(recovered['prior']))))
                        completed += 1
                    if _tensor_hash(sample['H_dl']) != hdl or _tensor_hash(sample['H_ul']) != hul:
                        raise ValueError('Diagnostic processing mutated true input channels')
                    runtime.atomic_json(output/'progress.json', dict(updated=runtime.now(), arm=arm,
                        completed_conditions=completed, total_conditions=total, optimizer_updates=0))
                    print(f'{arm}: batch {bi+1}/{cfg.BATCHES}; conditions {completed}/{total}', flush=True)
                after = runtime.frozen_signature(codec, stats, ce)
                if before != after:
                    raise ValueError('A frozen source model/prior changed during diagnostics')
                signatures[arm] = dict(before=before, after=after)
                if not fixture:
                    runtime.verify_source_record(arm, record)
                del codec, stats, ce, data, sample, raw, observation, variants
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        fingerprint_map(fingerprints, batches=cfg.BATCHES)
        summary = results.summarize_rows(rows, samples=cfg.SAMPLES, batch_size=cfg.MICRO_BATCH)
        comparisons = results.paired_comparisons(rows, samples=cfg.SAMPLES, batch_size=cfg.MICRO_BATCH)
        calibration = results.aggregate_calibration_bins(bins)
        write_union_csv(output/'summary.csv', summary)
        write_union_csv(output/'comparisons.csv', comparisons)
        for name, values in (('samples.csv.gz', rows), ('calibration_bins.csv.gz', calibration)):
            with gzip.open(output/name, 'wt', encoding='utf-8', newline='') as stream:
                import csv
                fields = sorted(set().union(*(set(row) for row in values)))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)
        runtime.atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
        runtime.atomic_json(output/'source_records.json', source_records, exclusive=True)
        files = ('samples.csv.gz', 'summary.csv', 'comparisons.csv', 'calibration_bins.csv.gz',
                 'input_fingerprints.json', 'source_records.json')
        runtime.guard_core(core)
        if not fixture:
            runtime.verify_source_records(source_records)
        manifest.update(status='test_fixture' if fixture else 'passed',
            paired_inputs_verified=True, frozen_models_verified=True,
            frozen_signatures=signatures, source_records=source_records, data_protocol=protocol,
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
