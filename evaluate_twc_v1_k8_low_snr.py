#!/usr/bin/env python3
"""Fresh four-view validation and offline UL scoring after matched low-SNR training."""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

from evaluate_twc_v1_k8_icc_aligned import _tensor_hash

EVALUATION_FILES = ('samples.csv.gz', 'summary.csv', 'comparisons.csv',
    'input_fingerprints.json', 'source_records.json')
COMMON_FIELDS = ('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma',
    'estimated_UL', 'CE_variance')
INPUT_FIELDS = frozenset((*COMMON_FIELDS, 'wave_active', 'Y_rx', 'prior'))


def _config():
    from configs import twc_v1_k8_low_snr_config as cfg
    return cfg


def _runtime():
    import low_snr_runtime as runtime
    return runtime


def _results():
    import twc_v1_k8_low_snr_results as results
    return results


def checked_output(path):
    path = _runtime().inside(path)
    allowed = (_config().OUTPUT_ROOT/'evaluation').resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError('Owned final evaluation attempt required')
    return path


def _load(variant, device):
    if variant in _config().TRAJECTORIES:
        import train_twc_v1_k8_low_snr as trainer
        return trainer.load_finetuned(variant, device)
    return _runtime().load_source(_results().physical_arm(variant), device)


def _signature(codec, stats, ce):
    runtime = _runtime()
    modules = dict(codec=codec, encoder=codec.encoder, decoder=codec.decoder,
        statistics=stats, CE=ce)
    for module in modules.values():
        if any(m.training for m in module.modules()) or any(
                p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError('All final networks and statistics must be frozen')
    return dict({name: runtime.state_digest(module) for name, module in modules.items()},
        CE_table={name: _tensor_hash(getattr(stats, name)) for name in ('ce_errors', 'ce_calibrated')})


def _verify_record(variant, record, *, fixture):
    cfg, runtime = _config(), _runtime()
    if not isinstance(record, dict) or not record.get('frozen_files'):
        raise ValueError('Explicit read-only source/terminal files required')
    runtime.verify_files(record)
    if fixture:
        if record.get('production_evidence') is not False:
            raise ValueError('Explicit CPU fixture evidence required')
        return
    if record.get('production_evidence') is not True:
        raise ValueError('CPU fixture cannot pass production final validation')
    if variant in _config().TRAJECTORIES:
        import train_twc_v1_k8_low_snr as trainer
        accepted = trainer.verify_terminal_record(variant)
        for name in ('checkpoint_sha256', 'core', 'step', 'statistics_state_sha256'):
            if record.get(name) != accepted.get(name):
                raise ValueError('New fixed Hybrid terminal identity changed: '+name)
        if record.get('step') != cfg.UPDATES:
            raise ValueError('Fixed new terminal update budget required')
    else:
        runtime.verify_source_record(_results().physical_arm(variant), record)


def _bind_signature(variant, record, signature, *, fixture):
    if fixture: return
    for state, field in (('codec', 'state_sha256'), ('encoder', 'encoder_state_sha256'),
            ('decoder', 'decoder_state_sha256'), ('statistics', 'statistics_state_sha256'),
            ('CE', 'CE_state_sha256')):
        if signature.get(state) != record.get(field):
            raise ValueError('Frozen loaded state differs from accepted checkpoint: '+state)
    frozen = (record if variant in _config().TRAJECTORIES else record['parent_record'])['frozen_identity']
    if any(signature.get('CE_table', {}).get(name) != frozen.get(name)
            for name in ('ce_errors', 'ce_calibrated')):
        raise ValueError('Loaded calibrated CE table differs from fixed parent')


def check_training_pairing(records, *, fixture=False):
    """Require both complete matched trajectories before any final channel.

    Initialization JSON hashes legitimately differ because they name different
    trajectories. Compare their immutable source states and actual physical
    input traces instead of requiring those JSON files to have equal bytes.
    """
    cfg = _config()
    if not set(cfg.TRAJECTORIES).issubset(records):
        raise ValueError('Both matched training terminals are required')
    pair = [records[name] for name in cfg.TRAJECTORIES]
    required_source = ('state_sha256', 'base_ema_updates', 'base_encoder_state_sha256',
        'base_decoder_state_sha256', 'base_statistics_state_sha256')
    source = pair[0].get('source_identity')
    if (not isinstance(source, dict) or any(key not in source for key in required_source)
            or source.get('base_ema_updates') != 40000 or pair[1].get('source_identity') != source):
        raise ValueError('Matched trajectories must start from identical source and prior')
    trace = pair[0].get('training_trace')
    if (not isinstance(trace, dict) or pair[1].get('training_trace') != trace
            or trace.get('microbatches') != cfg.UPDATES*cfg.ACCUMULATION):
        raise ValueError('Matched trajectories require the same complete actual input trace')
    for name, record in zip(cfg.TRAJECTORIES, pair):
        if (record.get('step') != cfg.UPDATES or record.get('method') != name
                or record.get('production_evidence') is not (not fixture)
                or record.get('base_ema_updates') != 40000
                or record.get('total_ema_updates') != 40000+cfg.UPDATES
                or record.get('counters') != cfg.counters(cfg.UPDATES, base_ema_updates=40000)):
            raise ValueError('Complete matched trajectory counters and source required')
        if not fixture:
            if (record.get('recipe') != cfg.recipe(name) or record.get('fingerprint') != cfg.fingerprint(name)
                    or record.get('core') != _runtime().core_identity()):
                raise ValueError('Matched trajectory recipe/core changed')
    return dict(status='passed', production_evidence=not fixture,
        source_identity_sha256=cfg.digest(source), shared_training_trace=trace,
        trajectories={name: dict(step=record['step'], base_ema_updates=40000,
            source_model_state_sha256=source.get('state_sha256'),
            source_encoder_state_sha256=source['base_encoder_state_sha256'],
            source_decoder_state_sha256=source['base_decoder_state_sha256'],
            source_statistics_state_sha256=source['base_statistics_state_sha256'],
            gamma_counts=cfg.gamma_counts(cfg.UPDATES), training_trace=record['training_trace'])
            for name, record in zip(cfg.TRAJECTORIES, pair)})


def _leakage(H_hat, H_true):
    from twc_v1_k8_low_snr import leakage_penalty
    cfg = _config()
    _, metrics = _runtime().helpers().rate_objective(H_hat, H_true,
        alpha=cfg.ALPHA, downlink_snr_db=cfg.DL_SNR)
    return leakage_penalty(metrics['desired_signal_power'], metrics['interference_power'], metrics['noise_power'])


def score_ul(out, true_UL):
    """Score the unchanged estimated-channel filter; never redesign it with truth."""
    import torch
    import twc_v1_k8_rate_headroom as decomposition
    import twc_v1_k8_ul_propagation_metrics as propagation
    observation, recovered = out['observation'], out['recovered']
    fields = ('soft', 'variance', 'H_ul_est', 'Y_res', 'prior')
    before = {name: _tensor_hash(recovered[name]) for name in fields}
    filters_before = {name: _tensor_hash(value) for name, value in
        propagation.reconstruct_normal_filters(observation, recovered, out['layout']).items()}
    separation = decomposition.receiver_error_decomposition(observation, recovered, true_UL,
        out['standard_noise'], out['layout'], out['gamma_db'])
    ce = propagation.analyze(observation, recovered, true_UL, out['standard_noise'],
        out['layout'], out['gamma_db'])
    if before != {name: _tensor_hash(recovered[name]) for name in fields}:
        raise ValueError('Offline UL scoring changed the actual receiver outputs')
    if filters_before != {name: _tensor_hash(value) for name, value in separation['filters'].items()}:
        raise ValueError('Offline true UL changed the normal estimated-channel filter')
    nominal = observation['prior'].to(torch.float64)[None]
    vectors = {}
    components = ('actual_error_power', 'realized_self_power', 'realized_cross_user_power',
        'realized_noise_power', 'realized_self_cross_user_cross', 'realized_self_noise_cross',
        'realized_cross_user_noise_cross')
    ce_components = ('realized_bias_power', 'realized_ce_power', 'realized_noise_power',
        'realized_bias_ce_cross', 'realized_bias_noise_cross', 'realized_ce_noise_cross')
    for section, selected in separation['sections'].items():
        for name in components:
            vectors['ul_'+section+'_nominal_scaled_'+name] = (
                separation['coordinate'][name][..., selected]/nominal[..., selected]).mean((1, 2))
        for name in ce_components:
            vectors['ul_'+section+'_ce_partition_'+name] = (
                ce['coordinate'][name][..., selected]/nominal[..., selected]).mean((1, 2))
    checks = {name: float(value.max()) for name, value in
        dict(separation['checks'], **ce['checks']).items()}
    return vectors, filters_before, checks


def fingerprint_map(entries, *, batches, metadata):
    cfg = _config()
    expected = {(v, b, float(g)) for v in cfg.EVAL_VARIANTS
        for b in range(batches) for g in cfg.GAMMAS}
    index = {}
    for row in entries:
        key = row['variant'], row['batch_index'], float(row['gamma_db'])
        if key not in expected or key in index or set(row['fingerprint']) != INPUT_FIELDS:
            raise ValueError('Incomplete or duplicate actual final input hashes')
        if row['address'] != metadata.address('validation', cfg.ROLE_RANGES['final'][0]+key[1]):
            raise ValueError('Fresh final channel address changed')
        for value in row['fingerprint'].values():
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Actual tensor SHA256 required')
        if (row.get('true_UL_used_for_scoring_only') is not True
                or set(row.get('filter_hashes', {})) != {'exclusive', 'shared'}
                or any(not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)
                    for value in row['filter_hashes'].values())):
            raise ValueError('Fixed estimated-channel filters and scoring-only truth required')
        index[key] = row
    if set(index) != expected: raise ValueError('Incomplete four-view final replay')
    for b in range(batches):
        for g in cfg.GAMMAS:
            rows = [index[v, b, float(g)] for v in cfg.EVAL_VARIANTS]
            if any(row['address'] != rows[0]['address'] or any(
                    row['fingerprint'][k] != rows[0]['fingerprint'][k] for k in COMMON_FIELDS) for row in rows[1:]):
                raise ValueError('Final variants did not use identical physical input bytes')
        for variant in cfg.EVAL_VARIANTS:
            first = index[variant, b, float(cfg.GAMMAS[0])]['fingerprint']
            if any(index[variant, b, float(g)]['fingerprint'][k] != first[k]
                    for g in cfg.GAMMAS for k in ('H_dl', 'H_ul', 'prior')):
                raise ValueError('Repeated gamma views changed channels or frozen prior')
    return index


def _csv(path, rows):
    fields = sorted(set().union(*(set(row) for row in rows)))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate(output_dir, *, require_cuda=True, loader=None, data_factory=None):
    import torch
    from evaluate_twc_v1_k8_ul_adapt import score_out
    from twc_v1_k8_low_snr import direction_error
    cfg, runtime, results = _config(), _runtime(), _results()
    fixture = not require_cuda or loader is not None or data_factory is not None
    core = runtime.core_identity(); runtime.guard_core(core)
    device = runtime.configure_cuda() if require_cuda else torch.device('cpu')
    loader = _load if loader is None else loader
    data_factory = runtime.make_data if data_factory is None else data_factory
    output = checked_output(output_dir); output.mkdir(parents=True, exist_ok=False)
    manifest = dict(format=cfg.NAMESPACE+'_evaluation_v1', output_dir=str(output), status='running',
        core=core, recipe_fingerprint=cfg.study_fingerprint(), study_recipe_fingerprint=cfg.study_fingerprint(), variants=list(cfg.EVAL_VARIANTS),
        gamma_db=list(cfg.GAMMAS), batch_size=cfg.MICRO_BATCH, n_batches=cfg.FINAL_BATCHES,
        samples_per_condition=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
        production_evidence=not fixture, source_checkpoints_verified=not fixture,
        test_used=False, checkpoint_selection=False, optimizer_updates=0, UE_updates=0,
        BS_updates=0, CE_updates=0, EMA_updates=0, FDMA_new_training_updates=0,
        Hybrid_new_training_updates=cfg.UPDATES*len(cfg.TRAJECTORIES),
        updates_per_trajectory=cfg.UPDATES, paired_trajectory_budgets_equal=True, budgets_equal=False,
        single_factor_loss_attribution=False, all_models_verified_before_data=True)
    runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
    loaded, records, before, signatures = {}, {}, {}, {}
    rows, fingerprints, checks = [], [], {}
    try:
        # Every loader/strict acceptance precedes even construction of final data.
        for variant in cfg.EVAL_VARIANTS:
            codec, stats, ce, record = loader(variant, device)
            _verify_record(variant, record, fixture=fixture)
            loaded[variant] = codec, stats, ce
            records[variant] = record
            before[variant] = _signature(codec, stats, ce)
            _bind_signature(variant, record, before[variant], fixture=fixture)
        ce_identities = {(s['CE'], tuple(sorted(s['CE_table'].items()))) for s in before.values()}
        if len(ce_identities) != 1: raise ValueError('CE weights and calibrated table must remain common')
        pairing = check_training_pairing(records, fixture=fixture)
        data, metadata = data_factory(device), runtime.make_data(device)
        if data.protocol != metadata.protocol:
            raise ValueError('Declared fresh final data protocol required')
        first, last = cfg.ROLE_RANGES['final']
        if last-first+1 != cfg.FINAL_BATCHES: raise ValueError('Exact final address budget required')
        manifest['data_protocol'] = data.protocol
        with torch.inference_mode():
            for bi in range(cfg.FINAL_BATCHES):
                sample = data.sample('validation', first+bi)
                if sample.get('address') != metadata.address('validation', first+bi):
                    raise ValueError('Final data address changed')
                channels = {k: _tensor_hash(sample[k]) for k in ('H_dl', 'H_ul')}
                for gi, gamma in enumerate(cfg.GAMMAS):
                    for variant in cfg.EVAL_VARIANTS:
                        codec, stats, ce = loaded[variant]
                        out = runtime.forward_codec(codec, stats, ce, sample, gamma,
                            ('final', bi, gi))
                        vectors = score_out(out, sample['H_dl'])
                        vectors['direction_error'] = direction_error(out['H_hat'], sample['H_dl']).detach()
                        vectors['dl_leakage_penalty'] = _leakage(out['H_hat'], sample['H_dl']).detach()
                        ul_vectors, filter_hashes, ul_checks = score_ul(out, sample['H_ul'])
                        vectors.update(ul_vectors)
                        for name, value in ul_checks.items():
                            checks[name] = max(checks.get(name, 0.), value)
                        if any(not torch.is_tensor(v) or v.shape != (cfg.MICRO_BATCH,) or
                                v.requires_grad or not bool(torch.isfinite(v).all()) for v in vectors.values()):
                            raise ValueError('Finite detached complete-MU evaluation vectors required')
                        values = {name: value.cpu().tolist() for name, value in vectors.items()}
                        for within in range(cfg.MICRO_BATCH):
                            rows.append(dict(variant=variant, arm=results.physical_arm(variant), role='final',
                                gamma_db=float(gamma), sample_index=bi*cfg.MICRO_BATCH+within,
                                batch_index=bi, within_batch=within,
                                **{name: value[within] for name, value in values.items()}))
                        fingerprints.append(dict(variant=variant, batch_index=bi, gamma_db=float(gamma),
                            address=sample['address'], fingerprint=dict(channels, **out['input_hashes']),
                            filter_hashes=filter_hashes, true_UL_used_for_scoring_only=True))
                if any(_tensor_hash(sample[k]) != value for k, value in channels.items()):
                    raise ValueError('Scoring changed simulator channel inputs')
                runtime.atomic_json(output/'progress.json', dict(completed_batches=bi+1,
                    total_batches=cfg.FINAL_BATCHES, updated=runtime.now(), neural_updates=0))
                print(f'final validation {bi+1}/{cfg.FINAL_BATCHES}', flush=True)
        for variant, modules in loaded.items():
            after = _signature(*modules)
            if after != before[variant]: raise ValueError('Final evaluation changed frozen model/prior/CE')
            _verify_record(variant, records[variant], fixture=fixture)
            signatures[variant] = dict(before=before[variant], after=after)
        fingerprint_map(fingerprints, batches=cfg.FINAL_BATCHES, metadata=metadata)
        samples = cfg.FINAL_BATCHES*cfg.MICRO_BATCH
        _csv(output/'samples.csv.gz', rows)
        _csv(output/'summary.csv', results.summarize_rows(rows, samples=samples, batch_size=cfg.MICRO_BATCH))
        _csv(output/'comparisons.csv', results.paired_comparisons(rows, samples=samples, batch_size=cfg.MICRO_BATCH))
        runtime.atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
        runtime.atomic_json(output/'source_records.json', records, exclusive=True)
        runtime.guard_core(core)
        manifest.update(status='test_fixture' if fixture else 'passed', paired_inputs_verified=True,
            frozen_signatures=signatures, source_records=records, training_pairing=pairing, closure_checks=checks,
            files={name: runtime.sha256(output/name) for name in EVALUATION_FILES})
        runtime.atomic_json(output/'manifest.json', manifest)
        return verify_manifest(output)
    except BaseException as exc:
        manifest.update(status='failed', error=type(exc).__name__+': '+str(exc))
        runtime.atomic_json(output/'manifest.json', manifest)
        raise


def verify_manifest(directory):
    cfg, runtime = _config(), _runtime()
    output = checked_output(directory)
    manifest = runtime.read_json(output/'manifest.json')
    if (manifest.get('format') != cfg.NAMESPACE+'_evaluation_v1' or
            manifest.get('status') not in ('passed', 'test_fixture') or
            manifest.get('output_dir') != str(output) or manifest.get('core') != runtime.core_identity() or
            manifest.get('recipe_fingerprint') != cfg.study_fingerprint() or
            manifest.get('study_recipe_fingerprint') != cfg.study_fingerprint() or
            manifest.get('variants') != list(cfg.EVAL_VARIANTS) or
            manifest.get('gamma_db') != list(cfg.GAMMAS) or
            manifest.get('n_batches') != cfg.FINAL_BATCHES or
            manifest.get('samples_per_condition') != cfg.FINAL_BATCHES*cfg.MICRO_BATCH or
            manifest.get('batch_size') != cfg.MICRO_BATCH or
            manifest.get('paired_inputs_verified') is not True or
            manifest.get('all_models_verified_before_data') is not True or
            any(manifest.get(name) is not False for name in
                ('test_used', 'checkpoint_selection', 'budgets_equal', 'single_factor_loss_attribution')) or
            any(type(manifest.get(name)) is not int or manifest[name] != 0 for name in
                ('optimizer_updates', 'UE_updates', 'BS_updates', 'CE_updates', 'EMA_updates', 'FDMA_new_training_updates')) or
            manifest.get('Hybrid_new_training_updates') != cfg.UPDATES*len(cfg.TRAJECTORIES) or
            manifest.get('updates_per_trajectory') != cfg.UPDATES or
            manifest.get('paired_trajectory_budgets_equal') is not True or
            set(manifest.get('files', {})) != set(EVALUATION_FILES)):
        raise ValueError('Complete frozen Hybrid-only final manifest required')
    fixture = manifest['status'] == 'test_fixture'
    if manifest.get('production_evidence') is not (not fixture) or manifest.get('source_checkpoints_verified') is not (not fixture):
        raise ValueError('Production versus fixture evaluation identity changed')
    runtime.guard_core(manifest['core'])
    for name, expected in manifest['files'].items():
        path = output/name
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != expected:
            raise ValueError('Final scalar artifact changed: '+name)
    records = runtime.read_json(output/'source_records.json')
    if records != manifest.get('source_records') or set(records) != set(cfg.EVAL_VARIANTS):
        raise ValueError('Final source-record artifact changed')
    if manifest.get('data_protocol') != runtime.make_data('cpu' if fixture else 'cuda:0').protocol:
        raise ValueError('Fresh final data protocol changed')
    for variant in cfg.EVAL_VARIANTS:
        value = manifest.get('frozen_signatures', {}).get(variant, {})
        if not value.get('before') or value.get('before') != value.get('after'):
            raise ValueError('Frozen final model signatures changed')
        _verify_record(variant, records[variant], fixture=fixture)
        _bind_signature(variant, records[variant], value['before'], fixture=fixture)
    if manifest.get('training_pairing') != check_training_pairing(records, fixture=fixture):
        raise ValueError('Matched training identity changed')
    closure = manifest.get('closure_checks', {})
    if not closure or any(not isinstance(v, (int, float)) or not __import__('math').isfinite(v) or v < 0 for v in closure.values()):
        raise ValueError('Finite UL reconstruction checks required')
    fingerprint_map(runtime.read_json(output/'input_fingerprints.json'), batches=cfg.FINAL_BATCHES,
        metadata=runtime.make_data('cpu' if fixture else 'cuda:0'))
    results = _results()
    results.validate_rows(results.read_raw_csv(output/'samples.csv.gz'),
        samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH, batch_size=cfg.MICRO_BATCH)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.evaluate or args.output_dir is None:
        parser.error('Explicit --evaluate --output-dir required')
    evaluate(args.output_dir)
    return 0


if __name__ == '__main__': raise SystemExit(main())
