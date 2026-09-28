#!/usr/bin/env python3
"""K4 low-feedback-SNR validation diagnostic, never a formal test or training job.

Eight fixed configurations; FB=0/5/10/15/20/25, DL=25. A separate cached
256-channel validation set is reserved for configuration selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
from unittest.mock import patch

from evaluate_specialist_envelope import checkpoint_plan, check_paths, project_path
from evaluate_proposal_snr_robust import FB_SNRS, verify_results
from train_proposal_snr_robust import robust_config, RECIPE

ROOT = Path(__file__).resolve().parent
SEED = 20360928  # Not formal test seed 20260921 or training validation seed 500042.
BATCH_SIZE, SAMPLE_COUNT = 8, 256
OUTPUT_ROOT = ROOT / 'runs_as_screen/diagnostics/k4_low_feedback_snr'
ROLE = 'configuration_selection_validation_not_formal_test'


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-root', default='runs_deployzf_v1')
    parser.add_argument('--specialist-root', default='runs_as_screen/proposal_specialists')
    parser.add_argument('--baseline-root', default='runs_as_screen/baselines')
    parser.add_argument('--robust-root', default='runs_as_screen/proposal_snr_robust')
    parser.add_argument('--swin-checkpoint')
    parser.add_argument('--csinet-checkpoint')
    parser.add_argument('--robust-checkpoint')
    parser.add_argument('--seed', type=int, choices=[SEED], default=SEED,
                        help='Reserved validation seed; fixed to prevent test-set reuse.')
    parser.add_argument('--output-dir', help='New directory below runs_as_screen/diagnostics/k4_low_feedback_snr.')
    parser.add_argument('--print-plan', action='store_true', help='No file checks, torch import or writes.')
    parser.add_argument('--dry-run', action='store_true', help='Check checkpoint paths/status only; no weights loaded.')
    parser.add_argument('--preflight', action='store_true',
                        help='First 8 cached validation samples; also compare probed vs original evaluation.')
    parser.set_defaults(universal_checkpoint=None)
    return parser


def diagnostic_plan(args):
    plan = [e for e in checkpoint_plan(args) if e['method_key'] != 'universal']
    plan.insert(5, dict(
        method_key='robust_specialist', model_key='proposed',
        method_type='robust_specialist', label='Existing FB-SNR-robust pure-AS',
        path=project_path(args.robust_checkpoint or robust_config(args.robust_root).BEST_PATH),
        allocations=((0, 256),),
    ))
    return plan


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, default=str, allow_nan=False)


def software_versions():
    versions = {}
    for name in ('torch', 'numpy', 'sionna'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'distribution metadata unavailable'
    return versions


def rate_statistics(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Missing/non-finite diagnostic rates.')
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def low_snr_rows(samples, plan, count):
    """SE over per-channel averages across the three SNRs, not 3*n iid draws."""
    indexed = {}
    for row in samples:
        if row['feedback_snr_db'] not in (0, 5, 10):
            continue
        key = row['method_key'], row['sample_index'], row['feedback_snr_db']
        if key in indexed:
            raise ValueError('Duplicate low-SNR sample.')
        indexed[key] = float(row['sum_rate'])
    vectors = {}
    for entry in plan:
        key = entry['method_key']
        vectors[key] = [
            statistics.mean(indexed[key, i, float(snr)] for snr in (0, 5, 10))
            for i in range(count)
        ]
    rows = []
    for entry in plan:
        key = entry['method_key']
        mean, se = rate_statistics(vectors[key])
        row = dict(method_key=key, method_type=entry['method_type'],
                   allocation_df=entry['allocations'][0][0], allocation_da=entry['allocations'][0][1],
                   dataset_role=ROLE, num_samples=count, seed=SEED,
                   feedback_snr_db='0,5,10', downlink_snr_db=25,
                   low_snr_rate_mean=mean, low_snr_rate_se=se)
        for reference in ('swin', 'csinet', 'specialist_df0_da256'):
            delta = [a-b for a, b in zip(vectors[key], vectors[reference])]
            row['delta_vs_'+reference+'_mean'], row['delta_vs_'+reference+'_se'] = rate_statistics(delta)
        rows.append(row)
    return rows


def validate_training_metadata(entry, config):
    """Record actual loaded config, not a recipe inferred solely from filenames."""
    if entry['model_key'] == 'proposed':
        experiment, recipe = config['experiment'], config['training_recipe']
        expected_recipe = RECIPE if entry['method_key'] == 'robust_specialist' else 'k4_n256_standard'
        if (experiment['train_mode'] != 'specialist' or experiment['seed'] != 42
                or config['allocations'] != [list(entry['allocations'][0])]
                or recipe['name'] != expected_recipe
                or recipe['total_steps'] != (20000 if expected_recipe == RECIPE else 50000)):
            raise RuntimeError('Unexpected specialist training provenance: '+entry['method_key'])
        if expected_recipe == RECIPE and (
            recipe.get('downlink_snr_db') != 25
            or any(recipe[x] != 0 for x in ('ph1_w_dir', 'ph1_w_mse', 'ph2_w_dir', 'ph2_w_mse'))
        ):
            raise RuntimeError('Robust checkpoint is not the completed fixed-DL, rate-only recipe.')
        return 'fixed_allocation_robust_refinement' if expected_recipe == RECIPE else 'fixed_allocation_base_specialist'
    if config['OBJECTIVE'] != 'task' or config['SEED'] != 42:
        raise RuntimeError('Expected existing seed42 task-trained FDMA baseline.')
    return 'task_trained_analog_FDMA'


def cached_channels(ev, scenario, device):
    """Persist the actual channel tensors once; future diagnostics reuse them.

    NPZ is dataset storage, not a checkpoint. Never read/write a formal test set.
    An incomplete cache is an error, never silently regenerated or overwritten.
    """
    import numpy as np
    import torch
    folder = OUTPUT_ROOT / f'validation_channels_seed{SEED}_b8_n256'
    manifest_path, data_path = folder / 'channels.json', folder / 'channels.npz'
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=False)
        ev.set_seed(SEED)
        generator = ev.build_channel_generator(scenario, device)
        downlink, uplink = [], []
        for batch in range(SAMPLE_COUNT // BATCH_SIZE):
            ev.set_seed(SEED + ev.stable_int(scenario.scenario_id) + batch)
            with torch.no_grad():
                dl, ul = generator.generate_batch_data(BATCH_SIZE)
            downlink.append(dl.detach().cpu().numpy())
            uplink.append(ul.detach().cpu().numpy())
        with data_path.open('xb') as handle:
            np.savez(handle, H_dl=np.concatenate(downlink), H_ul=np.concatenate(uplink))
        write_json_new(manifest_path, dict(
            seed=SEED, num_samples=SAMPLE_COUNT, batch_size=BATCH_SIZE,
            scenario_id=scenario.scenario_id, dataset_role=ROLE,
            sha256=sha256(data_path),
            generation='Unmodified evaluator channel generator; batch seed = seed + stable_int(scenario_id) + batch_index.',
            software=software_versions(),
        ))
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    required = dict(seed=SEED, num_samples=SAMPLE_COUNT, batch_size=BATCH_SIZE,
                    scenario_id=scenario.scenario_id, dataset_role=ROLE)
    if any(metadata.get(k) != v for k, v in required.items()) or sha256(data_path) != metadata['sha256']:
        raise RuntimeError('Cached validation dataset metadata/hash mismatch; preserve and inspect it.')
    with np.load(data_path, allow_pickle=False) as data:
        dl, ul = data['H_dl'], data['H_ul']
    for array in (dl, ul):
        if array.shape != (256, 4, 32, 256) or array.dtype != np.complex64 or not np.isfinite(array).all():
            raise RuntimeError('Unexpected cached channel shape/dtype/values.')
    return (torch.from_numpy(dl), torch.from_numpy(ul)), dict(
        **metadata, path=str(data_path), manifest_path=str(manifest_path),
    )


class CachedBatchSource:
    def __init__(self, channels, device, state):
        self.channels, self.device, self.state = channels, device, state
        self.batch = 0

    def generate_batch_data(self, size):
        if size != BATCH_SIZE or (self.batch+1)*size > SAMPLE_COUNT:
            raise RuntimeError('Unexpected channel batch request.')
        begin = self.batch * size
        dl, ul = (x[begin:begin+size].to(self.device) for x in self.channels)
        self.state.update(batch_index=self.batch, H_dl=dl, H_ul=ul)
        self.batch += 1
        return dl, ul


def compare_probe_results(probed, plain):
    if len(probed) != len(plain):
        raise RuntimeError('Probe changed the sample count.')
    for a, b in zip(probed, plain):
        for field in ('method_key', 'allocation_df', 'allocation_da', 'feedback_snr_db', 'sample_index'):
            if a[field] != b[field]:
                raise RuntimeError('Probe changed evaluation ordering.')
        for field in ('sum_rate', 'nmse_db', 'precoder_power'):
            if not math.isclose(a[field], b[field], rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError('Probe changed '+field)


def main():
    args = build_parser().parse_args()
    plan = diagnostic_plan(args)
    if args.print_plan:
        print(json.dumps(plan, indent=2, default=str))
        return 0
    check_paths(plan)
    if any(e['training_status'] != 'completed' for e in plan):
        raise RuntimeError('All eight existing runs must have completed status metadata; no training is launched.')
    if args.dry_run:
        print('[Dry run] Eight fixed configurations; validation only; no torch import/weight loading/writes.')
        return 0

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    tag = 'low_feedback_validation_preflight' if args.preflight else 'low_feedback_validation'
    output = project_path(args.output_dir or OUTPUT_ROOT / f'{tag}_{stamp}')
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve() or output.exists():
        raise FileExistsError('Choose a NEW output directory beneath '+str(OUTPUT_ROOT))
    import torch
    import baseline_evaluate as ev
    from feedback_link_diagnostics import FeedbackLinkProbe
    if not torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic on the GPU server, not the local laptop.')
    torch.set_float32_matmul_precision('high')
    device = ev.setup_gpu()
    scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
    scenario.validate()
    num_batches = 1 if args.preflight else SAMPLE_COUNT // BATCH_SIZE
    count = BATCH_SIZE * num_batches
    evaluation_args = ev.build_parser().parse_args([
        '--scenario', '4,256,256', '--models', 'proposed,swin,csinet',
        '--allocations', 'all', '--checkpoint', 'best', '--no-include-proposed-best',
        '--feedback-snr', ','.join(map(str, FB_SNRS)), '--downlink-snr', '25',
        '--batch-size', str(BATCH_SIZE), '--num-batches', str(num_batches),
        '--seed', str(SEED), '--baseline-objective', 'task', '--extra-metrics', 'basic',
        '--output-dir', str(output), '--tag', tag,
        '--proposal-root', str(project_path(args.proposal_root)),
        '--baseline-root', str(project_path(args.baseline_root)),
    ])
    channels, channel_manifest = cached_channels(ev, scenario, device)
    methods, captured_configs = [], {}
    original_loader = ev.load_checkpoint_cpu

    def capture_config(path):
        checkpoint = original_loader(path)
        captured_configs[str(path)] = checkpoint['config']
        return checkpoint

    for entry in plan:
        entry['checkpoint_sha256'] = sha256(entry['path'])
        with patch.object(ev, 'load_checkpoint_cpu', capture_config):
            if entry['model_key'] == 'proposed':
                method = ev.build_proposed_method(scenario, entry['path'], 'best', 'all', device)
                if tuple(method.config.GAMMA_GRID) != entry['allocations'] or (
                    method.config.D_F_MAX_PER_USER, method.config.D_A_MAX_SHARED, method.config.NUM_SUBBANDS
                ) != (64, 256, 4):
                    raise RuntimeError('Specialist type/max-width architecture mismatch.')
            else:
                method = ev.build_baseline_method(
                    entry['model_key'], scenario, entry['path'], 'best', device, evaluation_args.rzf_reg,
                )
        saved_config = captured_configs[str(entry['path'])]
        entry['training_mode'] = validate_training_metadata(entry, saved_config)
        entry['loaded_checkpoint_config'] = saved_config
        entry['checkpoint_step'] = method.checkpoint_step
        entry['cfg_index'] = 0 if entry['model_key'] == 'proposed' else None
        method.method_key, method.display_name = entry['method_key'], entry['label']
        if method.model.training or tuple(method.allocations) != entry['allocations']:
            raise RuntimeError('Unexpected model mode or allocation.')
        methods.append(method)

    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        created_utc=stamp, dataset_role=ROLE, formal_test=False, preflight=args.preflight,
        seed=SEED, num_samples=count, batch_size=BATCH_SIZE, num_batches=num_batches,
        feedback_snr_db=FB_SNRS, downlink_snr_db=25, strict_loading=True,
        channels=channel_manifest, methods=[{**e, 'path': str(e['path'])} for e in plan],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sampling='Identical cached H_dl/H_ul tensors for every method and SNR; original evaluator feedback seeds unchanged.',
        inactive_branches='No row is recorded for a physically inactive branch; missing is not zero noise.',
        snr_location='Active-stream batch-mean received power / expected complex noise variance at BS antennas, before MRC/DFT/neural decoder.',
        as_power_reference='Power of the coherent sum over all four UE contributions, including cross terms.',
        mrc_metrics='Actual equalized symbols vs transmitted normalized symbols; noise projected through original MRC. No neural post-detection SNR.',
        low_snr_metric='Per-channel mean rate at 0/5/10 dB, then mean and SE over channel indices. Signed paired gains retained.',
        allocation_selection='Eight fixed configurations only; no sample-wise max or oracle deployment curve.',
        source_sha256={name: sha256(ROOT / name) for name in (
            'diagnose_low_feedback_snr.py', 'feedback_link_diagnostics.py', 'baseline_evaluate.py',
            'models/adaptive_hybrid.py', 'models/baseline_common.py',
            'models/swin_precoder.py', 'models/csinet_plus_precoder.py', 'utils/channel_generator.py',
        )},
    )
    write_json_new(output / 'diagnostic_manifest.json', manifest)
    ev.write_csv(output / 'channel_sample_index.csv', [
        dict(sample_index=i, dataset_role=ROLE, validation_seed=SEED,
             batch_index=i//BATCH_SIZE, within_batch_index=i%BATCH_SIZE,
             channel_generation_seed=SEED+ev.stable_int(scenario.scenario_id)+i//BATCH_SIZE,
             channel_cache_sha256=channel_manifest['sha256'])
        for i in range(count)
    ])
    snr_points = [(float(snr), 25.0) for snr in FB_SNRS]
    batch_state = {}

    def source(*_args):
        return CachedBatchSource(channels, device, batch_state)

    ev.set_seed(SEED)
    with torch.no_grad(), patch.object(ev, 'build_channel_generator', source):
        with FeedbackLinkProbe(methods, batch_state) as probe:
            summary, samples = ev.evaluate_scenario(scenario, methods, snr_points, evaluation_args, device)
        probe.verify_coverage(num_batches, FB_SNRS)
        if args.preflight:
            cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
            ev.set_seed(SEED)
            _, plain_samples = ev.evaluate_scenario(scenario, methods, snr_points, evaluation_args, device)
            compare_probe_results(samples, plain_samples)
            if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
                not torch.equal(a, b) for a, b in zip(gpu_rng, torch.cuda.get_rng_state_all())
            ):
                raise RuntimeError('Probes changed RNG consumption.')
    verify_results(summary, samples, plan, count, SEED)
    low_rows = low_snr_rows(samples, plan, count)
    by_key = {e['method_key']: e for e in plan}
    for row in summary + samples + probe.rows:
        entry = by_key[row['method_key']]
        row.update(dataset_role=ROLE, seed=SEED, method_type=entry['method_type'],
                   training_mode=entry['training_mode'], preflight=args.preflight)
    for row in low_rows:
        row['preflight'] = args.preflight
    for entry in plan:
        if sha256(entry['path']) != entry['checkpoint_sha256']:
            raise RuntimeError('A protected checkpoint changed during evaluation: '+str(entry['path']))
    ev.print_summary(summary)
    ev.save_outputs(summary, samples, [scenario], [m.method_key for m in methods], snr_points, evaluation_args)
    ev.write_csv(output / 'physical_link_batches.csv', probe.rows)
    ev.write_csv(output / 'low_snr_validation.csv', low_rows)
    write_json_new(output / 'diagnostic_complete.json', dict(
        status='complete', dataset_role=ROLE, preflight=args.preflight,
        summary_rows=len(summary), num_samples_per_point=count,
        physical_rows=len(probe.rows), checkpoint_hashes_unchanged=True,
        probe_transparency='passed (metrics and RNG)' if args.preflight else 'run --preflight separately',
    ))
    print('\n[Validation only] 0/5/10 dB mean +/- SE, signed delta vs Swin:')
    for row in low_rows:
        print(f"{row['method_key']}: {row['low_snr_rate_mean']:.6f} +/- {row['low_snr_rate_se']:.6f}; "
              f"delta={row['delta_vs_swin_mean']:+.6f}")
    print('[Complete] No training or formal paper outputs. Results: '+str(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
