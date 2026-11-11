"""Frozen paired evaluation for BS rate-objective adaptation; no model fitting."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

from physical_split_finish_io import write_union_csv
import twc_v1_k8_icc_aligned_results as tables
from evaluate_twc_v1_k8_icc_aligned import (
    _module_state, _tensor_hash, _vectors as original_vectors,
    checked_input_hashes, frozen_evaluation,
)


def _runtime():
    import train_twc_v1_k8_icc_rate as runtime
    return runtime


def _config():
    from configs import twc_v1_k8_icc_rate_config as cfg
    return cfg


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def metric_vectors(out, truth):
    """Score true DL offline; the BS forward never receives truth as an input."""
    vectors = original_vectors(out, truth)
    users = list(zip(*(vectors[f'user_{u}_rate_bps_hz'] for u in range(8))))
    vectors['min_user_rate_bps_hz'] = [min(values) for values in users]
    # Ratio of the mean interference and desired powers within each complete MU.
    vectors['interference_to_signal_ratio'] = [i / max(s, 1e-20) for i, s in zip(
        vectors['interference_power_mean'], vectors['desired_signal_power_mean'])]
    predicted = out['H_hat'].abs().square().mean(dim=(1,2,3))
    true_energy = truth.abs().square().mean(dim=(1,2,3)).clamp_min(1e-20)
    vectors['predicted_dl_energy'] = predicted.detach().cpu().tolist()
    vectors['predicted_to_true_dl_energy_ratio'] = (predicted/true_energy).detach().cpu().tolist()
    return vectors


def fingerprint_map(entries, *, batches, arms=None):
    """Validate actual input bytes, completeness and cross-arm pairing."""
    cfg = _config()
    arms = tuple(cfg.ARMS if arms is None else arms)
    result = {}
    expected = {(arm, bi, float(gamma)) for arm in arms for bi in range(batches)
                for gamma in cfg.GAMMAS}
    for entry in entries:
        if 'arm' not in entry:
            continue
        key = (entry['arm'], entry['batch_index'], float(entry['gamma_db']))
        if key not in expected or key in result:
            raise ValueError('Unexpected/duplicate actual input fingerprint')
        value = entry['fingerprint']
        if set(value) != {'H_dl', 'H_ul', 'standard_noise', 'pilot_polarity'}:
            raise ValueError('Complete channel/noise/pilot hashes required')
        for digest in value.values():
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Invalid actual input hash')
        result[key] = value
    if set(result) != expected:
        raise ValueError('Missing actual input fingerprint')
    for bi in range(batches):
        for gamma in cfg.GAMMAS:
            hashes = [result[arm, bi, float(gamma)] for arm in arms]
            if any(value != hashes[0] for value in hashes[1:]):
                raise ValueError('Methods did not use identical actual inputs')
    return result


def _check_frozen_files(records):
    files = {}
    for record in records.values():
        entries = list(record.get('frozen_files', []))
        if record.get('path') and record.get('sha256'):
            entries.append(dict(path=record['path'], sha256=record['sha256']))
        for entry in entries:
            path, digest = Path(entry['path']), entry['sha256']
            if not path.is_file() or tables.file_sha256(path) != digest:
                raise ValueError('Input checkpoint/receipt changed: ' + str(path))
            files[str(path)] = digest
    return files


def evaluate_models(models, ce, data, *, mode, output_dir, checkpoint_records=None,
                    code_identity=None, forward=None, scorer=None, require_cuda=True):
    """One or both supplied arms; final CLI loads only one codec at a time.

    CPU injection exists for actual tiny tensor tests, not as a server fallback.
    """
    import torch
    runtime, cfg = _runtime(), _config()
    if mode not in ('monitor', 'final'):
        raise ValueError('Only monitor/final evaluation is supported')
    if not models or any(arm not in cfg.ARMS for arm in models):
        raise ValueError('Unknown/missing rate experiment arm')
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('Server evaluation requires actual CUDA')
    output = runtime.inside(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    core = runtime.core_identity()
    runtime.guard_core(core)
    records = checkpoint_records or {}
    frozen_files = _check_frozen_files(records)
    forward = runtime.forward_codec if forward is None else forward
    scorer = metric_vectors if scorer is None else scorer
    total_batches = cfg.MONITOR_BATCHES if mode == 'monitor' else cfg.FINAL_BATCHES
    first = cfg.ROLE_RANGES[mode][0]
    started = time.perf_counter()
    manifest = dict(format=cfg.NAMESPACE + '_evaluation_v1', status='running', mode=mode,
        arms=list(models), core=core, code_identity=code_identity or runtime.source_identity(),
        data_protocol=data.protocol, gamma_db=list(cfg.GAMMAS), batch_size=cfg.MICRO_BATCH,
        n_batches=total_batches, samples_per_condition=total_batches * cfg.MICRO_BATCH,
        fixed_rzf_alpha=cfg.ALPHA, DL_SNR=cfg.DL_SNR, checkpoints=records,
        training_seed_replications=1, test_used=False, alpha_calibration_performed=False,
        statistics='complete MU; mean gamma within MU; sample and batch4 mean +/-1.96SE',
        frozen_UE_CE_prior=True, current_frame_power_receiver_input=False)
    try:
        rows, fingerprints, prior_rows, parameter_rows = [], [], [], []
        with frozen_evaluation(models, ce):
            for arm, (model, stats) in models.items():
                prior = stats.nominal_power().detach().cpu()
                prior_rows.extend(dict(arm=arm, user=u, coordinate=j, prior_power=float(prior[u, j]))
                                  for u in range(prior.shape[0]) for j in range(prior.shape[1]))
                encoder_n = sum(p.numel() for p in model.encoder.parameters())
                decoder_n = sum(p.numel() for p in model.decoder.parameters())
                parameter_rows.append(dict(arm=arm, UE_encoder_parameters=encoder_n,
                    BS_parameters=decoder_n, total_parameters=encoder_n + decoder_n,
                    CE_parameters=sum(p.numel() for p in ce.parameters())))
            for bi in range(total_batches):
                sample = data.sample('validation', first + bi)
                hdl, hul = _tensor_hash(sample['H_dl']), _tensor_hash(sample['H_ul'])
                for gi, gamma in enumerate(cfg.GAMMAS):
                    for arm, (model, stats) in models.items():
                        out = forward(model, stats, ce, sample, gamma_db=gamma,
                                      noise_key=(mode, bi, gi))
                        vectors = scorer(out, sample['H_dl'])
                        if any(len(value) != cfg.MICRO_BATCH for value in vectors.values()):
                            raise ValueError('Actual evaluation microbatch changed')
                        for within in range(cfg.MICRO_BATCH):
                            rows.append(dict(arm=arm, gamma_db=float(gamma),
                                sample_index=bi * cfg.MICRO_BATCH + within,
                                batch_index=bi, within_batch=within,
                                **{name: value[within] for name, value in vectors.items()}))
                        fingerprints.append(dict(arm=arm, gamma_db=float(gamma), batch_index=bi,
                            fingerprint=checked_input_hashes(out.get('input_hashes'), H_dl=hdl, H_ul=hul)))
                if _tensor_hash(sample['H_dl']) != hdl or _tensor_hash(sample['H_ul']) != hul:
                    raise RuntimeError('Evaluation input channels mutated')
        fingerprint_map(fingerprints, batches=total_batches, arms=tuple(models))
        result = tables.write_result_tables(output, rows,
            samples=total_batches * cfg.MICRO_BATCH, batch_size=cfg.MICRO_BATCH, arms=tuple(models))
        write_union_csv(output/'coordinate_prior.csv', prior_rows)
        write_union_csv(output/'parameters.csv', parameter_rows)
        runtime.atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
        files = dict(result['files'])
        for name in ('coordinate_prior.csv', 'parameters.csv', 'input_fingerprints.json'):
            files[name] = tables.file_sha256(output/name)
        _check_frozen_files(records)
        runtime.guard_core(core)
        manifest.update(status='passed', elapsed_seconds=time.perf_counter()-started,
            n_conditions=len(models)*len(cfg.GAMMAS), files=files,
            freezing_verified=True, checkpoint_hashes_unchanged=True,
            outputs=dict(summary=str(output/'summary.csv'), paired=str(output/'paired_deltas.csv')))
        runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                        traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started)
        if not (output/'manifest.json').exists():
            runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
        raise


def verify_evaluation(directory, *, arms=None):
    """Verify a phase before consuming its raw metrics or claiming completion."""
    cfg, runtime = _config(), _runtime()
    directory = runtime.inside(directory)
    manifest = read_json(directory/'manifest.json')
    if manifest.get('status') != 'passed' or manifest.get('core') != runtime.core_identity():
        raise ValueError('Evaluation status/core changed')
    expected_arms = tuple(cfg.ARMS if arms is None else arms)
    if set(manifest.get('arms', [])) != set(expected_arms):
        raise ValueError('Evaluation arm identity changed')
    if (manifest.get('mode') != 'final' or manifest.get('batch_size') != cfg.MICRO_BATCH or
            manifest.get('n_batches') != cfg.FINAL_BATCHES):
        raise ValueError('Final evaluation sample budget changed')
    for name, digest in manifest.get('files', {}).items():
        if Path(name).name != name or tables.file_sha256(directory/name) != digest:
            raise ValueError('Evaluation artifact changed: ' + name)
    if not {'samples.csv.gz', 'summary.csv', 'paired_deltas.csv', 'input_fingerprints.json'} <= set(manifest.get('files', {})):
        raise ValueError('Evaluation evidence incomplete')
    fingerprints = read_json(directory/'input_fingerprints.json')
    fingerprint_map(fingerprints, batches=cfg.FINAL_BATCHES, arms=expected_arms)
    rows = tables.read_raw_csv(directory/'samples.csv.gz')
    tables.validate_rows(rows, samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
                         batch_size=cfg.MICRO_BATCH, arms=expected_arms)
    return manifest, rows, fingerprints


def merge_serial_arms(output, arm_directories, *, phase):
    """Merge two independently replayed arms after checking actual paired bytes."""
    cfg, runtime = _config(), _runtime()
    if phase not in ('baseline', 'final') or set(arm_directories) != set(cfg.ARMS):
        raise ValueError('Exactly baseline/final and both arms are required')
    output = runtime.inside(output)
    rows, fingerprints, manifests = [], [], {}
    protocol = None
    for arm in cfg.ARMS:
        manifest, arm_rows, arm_fingerprints = verify_evaluation(arm_directories[arm], arms=(arm,))
        if protocol is not None and manifest['data_protocol'] != protocol:
            raise ValueError('Serial arms changed data protocol')
        protocol = manifest['data_protocol']
        rows.extend(arm_rows); fingerprints.extend(arm_fingerprints); manifests[arm] = manifest
    fingerprint_map(fingerprints, batches=cfg.FINAL_BATCHES)
    result = tables.write_result_tables(output, rows, samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
                                        batch_size=cfg.MICRO_BATCH)
    runtime.atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
    files = dict(result['files']); files['input_fingerprints.json'] = tables.file_sha256(output/'input_fingerprints.json')
    manifest = dict(format=cfg.NAMESPACE+'_serial_evaluation_v1', status='passed', mode='final', phase=phase,
        arms=list(cfg.ARMS), core=runtime.core_identity(), data_protocol=protocol,
        batch_size=cfg.MICRO_BATCH, n_batches=cfg.FINAL_BATCHES,
        samples_per_condition=cfg.FINAL_BATCHES*cfg.MICRO_BATCH, files=files,
        serial_arm_manifests=manifests, test_used=False, fixed_rzf_alpha=cfg.ALPHA,
        DL_SNR=cfg.DL_SNR, training_seed_replications=1)
    runtime.atomic_json(output/'manifest.json', manifest, exclusive=True)
    return manifest


def run_phase(phase, output_dir):
    """Explicit CUDA path: only one codec/CE is resident at any time."""
    import torch
    cfg, runtime = _config(), _runtime()
    if phase not in ('baseline', 'final'):
        raise ValueError('Unknown evaluation phase')
    device = runtime.configure_cuda()
    output = runtime.inside(output_dir); output.mkdir(parents=True, exist_ok=False)
    arm_dirs = {}
    for arm in cfg.ARMS:
        codec, stats, ce, record = (runtime.load_source(arm, device) if phase == 'baseline'
                                  else runtime.load_finetuned(arm, device))
        arm_dirs[arm] = output/arm
        evaluate_models({arm:(codec, stats)}, ce, runtime.make_data(device), mode='final',
            output_dir=arm_dirs[arm], checkpoint_records={arm:record})
        del codec, stats, ce
        gc.collect(); torch.cuda.empty_cache()
    return merge_serial_arms(output, arm_dirs, phase=phase)


def comparison_rows(baseline_rows, final_rows, *, samples, batch_size=4):
    """Predeclared paired main effects and interaction, never independent gammas."""
    _, base = tables.validate_rows(baseline_rows, samples=samples, batch_size=batch_size)
    _, fine = tables.validate_rows(final_rows, samples=samples, batch_size=batch_size)
    gammas = tables.GAMMAS
    scopes = [('point', (g,)) for g in gammas] + [('low_0_5_10', gammas[:3]), ('all_six', gammas)]
    comparisons = {
        'source_hybrid_minus_fdma': ((-1.,base,'fdma'),(1.,base,'hybrid_g4')),
        'rate_hybrid_minus_fdma': ((-1.,fine,'fdma'),(1.,fine,'hybrid_g4')),
        'fdma_rate_minus_source': ((-1.,base,'fdma'),(1.,fine,'fdma')),
        'hybrid_rate_minus_source': ((-1.,base,'hybrid_g4'),(1.,fine,'hybrid_g4')),
        'interaction_hybrid_gain_minus_fdma_gain': (
            (-1.,fine,'fdma'),(1.,fine,'hybrid_g4'),(1.,base,'fdma'),(-1.,base,'hybrid_g4')),
    }
    output=[]
    for scope, selected in scopes:
        for label, terms in comparisons.items():
            metrics = set.intersection(*(set(index[arm, g, 0])-tables.ID_FIELDS
                                        for _,index,arm in terms for g in selected))
            for metric in sorted(metrics):
                if any(index[arm,g,i].get(metric) is None for _,index,arm in terms
                       for g in selected for i in range(samples)):
                    continue
                delta=[statistics.mean(sum(coef*index[arm,g,i][metric] for coef,index,arm in terms)
                                       for g in selected) for i in range(samples)]
                individual=tables.moments(delta)
                grouped=tables.moments([statistics.mean(delta[i:i+batch_size]) for i in range(0,samples,batch_size)])
                output.append(dict(comparison=label, metric=metric, scope=scope,
                    gamma_db=selected[0] if scope=='point' else None, n_samples=samples,
                    n_batches=samples//batch_size, batch_size=batch_size, mean=individual['mean'],
                    sample_se=individual['se'], sample_ci95_low=individual['ci95_low'],
                    sample_ci95_high=individual['ci95_high'], batch_se=grouped['se'],
                    batch_ci95_low=grouped['ci95_low'], batch_ci95_high=grouped['ci95_high']))
    return output


def compare_evaluations(baseline_directory, final_directory, output_dir):
    cfg, runtime = _config(), _runtime()
    baseline, b_rows, b_fingerprints=verify_evaluation(baseline_directory)
    final, f_rows, f_fingerprints=verify_evaluation(final_directory)
    if baseline.get('phase')!='baseline' or final.get('phase')!='final':
        raise ValueError('Predeclared source and rate phases required')
    if baseline['data_protocol']!=final['data_protocol']:
        raise ValueError('Source and rate data protocol differ')
    if fingerprint_map(b_fingerprints,batches=cfg.FINAL_BATCHES)!=fingerprint_map(f_fingerprints,batches=cfg.FINAL_BATCHES):
        raise ValueError('Source and rate used different actual channel/noise inputs')
    output=runtime.inside(output_dir);output.mkdir(parents=True,exist_ok=False)
    rows=comparison_rows(b_rows,f_rows,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,batch_size=cfg.MICRO_BATCH)
    write_union_csv(output/'comparisons.csv',rows)
    runtime.atomic_json(output/'preregistered_comparison.json',dict(
        primary='interaction_hybrid_gain_minus_fdma_gain',
        primary_metric='sum_rate_bps_hz',primary_scope='all_six',
        interpretation='positive interaction is a relative gain, not proof Hybrid exceeds FDMA',
        source_baseline=str(baseline_directory),rate_final=str(final_directory),
        metric='sum_rate_bps_hz',scopes=['point','low_0_5_10','all_six'],
        all_user_metrics_retained=True,selection='fixed terminal, never best monitor',test_used=False),exclusive=True)
    files={p.name:tables.file_sha256(p) for p in output.iterdir() if p.is_file()}
    manifest=dict(status='passed',format=cfg.NAMESPACE+'_comparison_v1',core=runtime.core_identity(),
        files=files,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,training_seed_replications=1)
    runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
    return manifest


def cli(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group();actions.add_argument('--run',action='store_true');actions.add_argument('--dry-run',action='store_true')
    parser.add_argument('--phase',choices=('baseline','final'),default='final')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    cfg=_config()
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,phase=args.phase,arms=cfg.ARMS,
            gamma=cfg.GAMMAS,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
            same_fresh_baseline_final_data=True,serial_models=True,weights_read=False,writes=False),indent=2));return 0
    if not args.run:parser.print_help();return 0
    if args.output_dir is None:parser.error('--run requires --output-dir')
    result=run_phase(args.phase,args.output_dir)
    print(json.dumps(dict(status=result['status'],output=str(args.output_dir)),indent=2));return 0


if __name__=='__main__':
    sys.exit(cli())
