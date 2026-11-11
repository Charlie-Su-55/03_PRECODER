#!/usr/bin/env python3
"""Frozen-weight alpha calibration, locked selection and held-out validation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
import zipfile

from run_twc_v1_k8_icc_aligned import gpu_admission


def _runtime():
    import icc_alpha_runtime as runtime
    return runtime


def _config():
    from configs import twc_v1_k8_icc_alpha_config as cfg
    return cfg


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    return ('calibration', 'selection_lock', 'final_validation', 'result_package')


def files_identity(paths):
    runtime = _runtime()
    result = {}
    for path in paths:
        path = runtime.inside(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Owned alpha evidence missing: ' + str(path))
        result[str(path)] = runtime.sha256(path)
    if not result:
        raise ValueError('Nonempty completed alpha evidence required')
    return result


def verify_evidence(record):
    runtime = _runtime()
    if not record.get('files'):
        raise ValueError('Completed alpha stage has no evidence')
    for name, digest in record['files'].items():
        if runtime.sha256(runtime.inside(name)) != digest:
            raise ValueError('Completed alpha evidence changed: ' + name)


def stage(summary, path, name, action):
    runtime = _runtime()
    previous = summary['stages'].get(name)
    if previous and previous['status'] == 'completed':
        verify_evidence(previous)
        return previous
    attempts = [] if previous is None else previous.get('attempts', []) + [previous]
    record = dict(status='running', started=now(), attempt_id=uuid.uuid4().hex[:12])
    summary['stages'][name] = record
    summary.update(status='running', current_stage=name, updated=now())
    runtime.atomic_json(path, summary)
    try:
        details, files = action(record)
        record.update(status='completed', finished=now(), details=details,
                      files=files_identity(files), attempts=attempts)
        runtime.atomic_json(path, summary)
        return record
    except BaseException as exc:
        record.update(status='failed', finished=now(), error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc(), attempts=attempts)
        summary.update(status='failed', error=record['error'], updated=now())
        runtime.atomic_json(path, summary)
        raise


def child(summary, path, record, script, args):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(summary['core'])
    log = path.parent / (summary['current_stage'] + '_' + record['attempt_id'] + '.log')
    record['log_path'] = str(log)
    runtime.atomic_json(path, summary)
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.run([sys.executable, '-B', '-u', script, *args], cwd=cfg.ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT, check=False)
    runtime.guard_core(summary['core'])
    if process.returncode:
        tail = '\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-16:])[-4000:]
        record['child_error'] = tail
        raise RuntimeError(f'{script} failed exit={process.returncode}; exact log={log}\n{tail}')
    return log


def load_summary():
    runtime, cfg = _runtime(), _config()
    pointer = runtime.read_json(cfg.OUTPUT_ROOT / 'last_batch.json')
    if set(pointer) != {'namespace', 'summary'} or pointer['namespace'] != cfg.NAMESPACE:
        raise ValueError('Own alpha-experiment pointer required')
    path = runtime.inside(pointer['summary'])
    summary = runtime.read_json(path)
    if summary.get('namespace') != cfg.NAMESPACE:
        raise ValueError('Alpha batch namespace changed')
    return summary, path


def final_attempts_exist():
    root = _config().OUTPUT_ROOT / 'evaluation' / 'final'
    return root.exists() and any(root.iterdir())


def require_locked_selection():
    """Validate existing selection; never create a lock on the final path."""
    runtime, cfg = _runtime(), _config()
    path = cfg.OUTPUT_ROOT / 'selection.json'
    if not path.is_file():
        raise ValueError('Locked alpha selection required before final evaluation; reselection forbidden')
    return runtime.load_selection(path)


def verify_manifest(directory, phase):
    runtime, cfg = _runtime(), _config()
    directory = runtime.inside(directory)
    manifest = runtime.read_json(directory / 'manifest.json')
    if (manifest.get('status') != 'passed' or manifest.get('phase') != phase
            or manifest.get('core') != runtime.core_identity()):
        raise ValueError('Alpha evaluation status/phase/core changed')
    batches = cfg.CALIBRATION_BATCHES if phase == 'calibration' else cfg.FINAL_BATCHES
    expected_identity = dict(recipe_fingerprint=cfg.fingerprint(),
        gamma_db=list(cfg.GAMMAS), batch_size=cfg.MICRO_BATCH, n_batches=batches,
        samples_per_condition=batches*cfg.MICRO_BATCH, no_optimizer=True, test_used=False)
    for name, value in expected_identity.items():
        if manifest.get(name) != value:
            raise ValueError('Alpha evaluation recipe/budget identity changed: ' + name)
    evidence = manifest.get('files', {})
    if not evidence:
        raise ValueError('Alpha evaluation evidence incomplete')
    expected = ({'samples.csv.gz', 'scores.csv', 'selection_candidate.json', 'input_fingerprints.json'}
                if phase == 'calibration' else
                {'samples.csv.gz', 'summary.csv', 'comparisons.csv', 'input_fingerprints.json'})
    if not expected <= set(evidence):
        raise ValueError('Alpha evaluation artifacts incomplete')
    for name, digest in evidence.items():
        if Path(name).name != name or runtime.sha256(runtime.inside(directory / name)) != digest:
            raise ValueError('Alpha evaluation artifact changed: ' + name)
    if phase == 'final':
        locked = require_locked_selection()
        if manifest.get('selection_record_id') != locked['record_id']:
            raise ValueError('Final evaluation used a different locked selection')
    return manifest


def evaluation_action(summary, path, phase, record):
    runtime, cfg = _runtime(), _config()
    if phase == 'calibration':
        output = cfg.OUTPUT_ROOT / 'calibration' / ('attempt_' + record['attempt_id'])
        args = ['--run', '--phase', 'calibration', '--output-dir', str(output)]
    elif phase == 'final':
        require_locked_selection()
        output = cfg.OUTPUT_ROOT / 'evaluation' / 'final' / ('attempt_' + record['attempt_id'])
        args = ['--run', '--phase', 'final', '--output-dir', str(output),
                '--selection', str(cfg.OUTPUT_ROOT / 'selection.json')]
    else:
        raise ValueError('Exactly calibration or final phase is allowed')
    log = child(summary, path, record, 'evaluate_twc_v1_k8_icc_alpha.py', args)
    manifest = verify_manifest(output, phase)
    files = [p for p in output.rglob('*') if p.is_file()] + [log]
    return dict(output_dir=str(output), manifest=manifest), files


def selection_action(calibration_directory):
    runtime, cfg = _runtime(), _config()
    lock = cfg.OUTPUT_ROOT / 'selection.json'
    if lock.exists():
        selected = require_locked_selection()
        if runtime.inside(selected['calibration_dir']) != runtime.inside(calibration_directory):
            raise ValueError('Existing selection belongs to different calibration evidence')
    else:
        if final_attempts_exist():
            raise ValueError('A final attempt already exists; missing selection cannot be recreated')
        selected = runtime.lock_selection(calibration_directory)
    return selected, [lock]


def package_action(summary_path, calibration_directory, final_directory, record):
    """Explicit scalar-only evidence archive, including reproducible raw metrics."""
    runtime, cfg = _runtime(), _config()
    selected = require_locked_selection()
    if runtime.inside(selected['calibration_dir']) != runtime.inside(calibration_directory):
        raise ValueError('Package calibration differs from locked evidence')
    verify_manifest(calibration_directory, 'calibration')
    verify_manifest(final_directory, 'final')
    output = runtime.inside(cfg.OUTPUT_ROOT / 'packages' / ('attempt_' + record['attempt_id']))
    output.mkdir(parents=True, exist_ok=False)
    calibration_directory, final_directory = Path(calibration_directory), Path(final_directory)
    # Only these basenames enter the archive. CSV.gz contains scalar metrics,
    # not channel tensors, model weights, optimizer state or arbitrary files.
    snapshots = {
        'selection.json': cfg.OUTPUT_ROOT / 'selection.json',
        'preregistration.json': summary_path.parent / 'preregistration.json',
        'calibration_manifest.json': calibration_directory / 'manifest.json',
        'calibration_scores.csv': calibration_directory / 'scores.csv',
        'calibration_candidate.json': calibration_directory / 'selection_candidate.json',
        'calibration_input_fingerprints.json': calibration_directory / 'input_fingerprints.json',
        'calib_samples.csv.gz': calibration_directory / 'samples.csv.gz',
        'final_manifest.json': final_directory / 'manifest.json',
        'summary.csv': final_directory / 'summary.csv',
        'comparisons.csv': final_directory / 'comparisons.csv',
        'final_input_fingerprints.json': final_directory / 'input_fingerprints.json',
        'final_samples.csv.gz': final_directory / 'samples.csv.gz',
    }
    hashes = {}
    for name, source in snapshots.items():
        source = runtime.inside(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing explicit package evidence: ' + str(source))
        target = output / name
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        hashes[name] = runtime.sha256(target)
    runtime.atomic_json(output / 'experiment_recipe.json', cfg.recipe(), exclusive=True)
    hashes['experiment_recipe.json'] = runtime.sha256(output / 'experiment_recipe.json')
    whitelist = dict(format=cfg.NAMESPACE + '_package_v1', files=hashes,
        digest='SHA256 of exact uncompressed file bytes; archive hash is in the outer receipt',
        contents='Scalar MU metrics, aggregate tables, locked selection, fingerprints and provenance only',
        channel_tensors_included=False, weights_included=False, optimizer_updates=0)
    runtime.atomic_json(output / 'package_whitelist.json', whitelist, exclusive=True)
    names = [*hashes, 'package_whitelist.json']
    archive = output / ('icc_alpha_small_results_' + record['attempt_id'] + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(names):
            stream.write(output / name, arcname=name)
    result = dict(path=str(archive), sha256=runtime.sha256(archive), files=sorted(names))
    receipt = summary_path.parent / ('package_' + record['attempt_id'] + '.json')
    runtime.atomic_json(receipt, result, exclusive=True)
    return result, [archive, receipt]


def run(args):
    runtime, cfg = _runtime(), _config()
    if sys.platform == 'win32':
        raise RuntimeError('Server alpha inference queue requires Linux CUDA')
    runtime.guard_core(runtime.core_identity())
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with runtime.process_lock(cfg.OUTPUT_ROOT / 'controller.lock'):
        if args.resume:
            summary, path = load_summary()
            runtime.guard_core(summary['core'])
            # Even a failed/partial final attempt makes reselection forbidden.
            if final_attempts_exist():
                require_locked_selection()
            for record in summary['stages'].values():
                if record['status'] == 'completed':
                    verify_evidence(record)
        else:
            if any((cfg.OUTPUT_ROOT / name).exists() for name in
                   ('last_batch.json', 'selection.json', 'calibration', 'evaluation', 'batches')):
                raise FileExistsError('Own alpha experiment exists; explicit --resume required')
            folder = cfg.OUTPUT_ROOT / 'batches' / ('batch_' + uuid.uuid4().hex)
            folder.mkdir(parents=True, exist_ok=False)
            path = folder / 'batch_summary.json'
            summary = dict(namespace=cfg.NAMESPACE, status='starting', created=now(), updated=now(),
                core=runtime.core_identity(), code=runtime.source_identity(), recipe=cfg.recipe(),
                stages={}, test_used=False, optimizer_updates=0, original_outputs_preserved=True)
            runtime.atomic_json(folder / 'preregistration.json', summary, exclusive=True)
            runtime.atomic_json(path, summary, exclusive=True)
            runtime.atomic_json(cfg.OUTPUT_ROOT / 'last_batch.json',
                dict(namespace=cfg.NAMESPACE, summary=str(path)), exclusive=True)
        if summary.get('status') == 'completed':
            print(compact_summary(summary))
            return
        try:
            summary['current_stage'] = 'startup_policy'
            admission = gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                min_free_mib=args.min_free_mib, max_gpu_util=args.max_gpu_util)
            policy = path.parent / ('startup_policy_' + uuid.uuid4().hex[:12] + '.json')
            runtime.atomic_json(policy, admission, exclusive=True)
            summary.setdefault('startup_history', []).append(admission)
            summary['stages']['startup_policy'] = dict(status='completed', details=admission,
                files=files_identity([policy]), finished=now())
            runtime.atomic_json(path, summary)
            calibration = stage(summary, path, 'calibration',
                lambda record: evaluation_action(summary, path, 'calibration', record))
            stage(summary, path, 'selection_lock',
                lambda record: selection_action(calibration['details']['output_dir']))
            final = stage(summary, path, 'final_validation',
                lambda record: evaluation_action(summary, path, 'final', record))
            stage(summary, path, 'result_package', lambda record: package_action(path,
                calibration['details']['output_dir'], final['details']['output_dir'], record))
            runtime.guard_core(summary['core'])
            summary.update(status='completed', current_stage='result_package',
                           finished=now(), updated=now())
            runtime.atomic_json(path, summary)
            print(compact_summary(summary))
        except BaseException as exc:
            summary.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                           traceback=traceback.format_exc(), updated=now())
            if summary.get('current_stage') == 'startup_policy':
                summary['stages']['startup_policy'] = dict(status='failed', error=summary['error'],
                    traceback=summary['traceback'], finished=now())
            runtime.atomic_json(path, summary)
            raise


def compact_summary(summary):
    runtime, cfg = _runtime(), _config()
    lines = [f'ICC alpha control | {summary.get("status", "unknown")} | stage={summary.get("current_stage", "unknown")}',
        'Namespace: ' + cfg.NAMESPACE, 'Updated: ' + summary.get('updated', 'unknown'),
        'Core commit: ' + summary.get('code', {}).get('commit', 'unknown'),
        'Frozen inference only: optimizer/UE/CE/EMA updates=0; original weights preserved',
        'Calibration: one global alpha per arm, equal six-gamma mean; final samples never select alpha',
        'Gamma: ' + ','.join(str(g) for g in cfg.GAMMAS)]
    lock = cfg.OUTPUT_ROOT / 'selection.json'
    if lock.exists():
        selected = runtime.load_selection(lock)
        for arm in cfg.ARMS:
            alpha = selected['selected'][arm]
            score = next(row for row in selected['scores'] if row['arm'] == arm and row['alpha'] == alpha)
            lines.append(f'Locked alpha {arm}: {alpha:.6g}; calibration six-point rate={score["mean"]:.5f}')
        lines.append('Selection record: ' + selected['record_id'])
    else:
        lines.append('Selection: not locked')
    if summary.get('status') == 'failed':
        record = summary.get('stages', {}).get(summary.get('current_stage'), {})
        lines.append('ERROR: ' + str(record.get('error', summary.get('error', 'unknown')))[:1200])
        if record.get('log_path'):
            lines.append('Exact log: ' + record['log_path'])
        if record.get('child_error'):
            lines.append(record['child_error'][-1500:])
    final = summary.get('stages', {}).get('final_validation', {})
    if final.get('status') == 'completed':
        output = Path(final['details']['output_dir'])
        with (output / 'summary.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        for policy in cfg.POLICIES:
            for arm in cfg.ARMS:
                method = cfg.method_name(policy, arm)
                points = sorted((row for row in rows if row['arm'] == method),
                                key=lambda row: float(row['gamma_db']))
                lines.append(method + ' rate: ' + ','.join(
                    f'{float(row["sum_rate_bps_hz_mean"]):.5f}' for row in points))
        with (output / 'comparisons.csv').open(newline='', encoding='utf-8') as stream:
            comparisons = list(csv.DictReader(stream))
        labels = list(dict.fromkeys(row['comparison'] for row in comparisons))
        for label in labels:
            for scope in ('all_six', 'low_0_5_10'):
                row = next(row for row in comparisons if row['comparison'] == label
                    and row['metric'] == 'sum_rate_bps_hz' and row['scope'] == scope)
                lines.append(f'{label} {scope}: {float(row["mean"]):+.5f} '
                    f'batchCI=[{float(row["batch_ci95_low"]):+.5f},{float(row["batch_ci95_high"]):+.5f}]')
        lines.append('Full results: ' + str(output))
    package = summary.get('stages', {}).get('result_package', {})
    lines.append('Small package: ' + str(package.get('details', {}).get('path', 'not completed')))
    lines.append('Intervals: paired validation sampling conditional on fixed models/selected alpha; '
                 'one training seed; calibration uncertainty excluded.')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--dry-run', action='store_true')
    actions.add_argument('--summary', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--min-free-mib', type=int, default=12000)
    parser.add_argument('--max-gpu-util', type=int, default=20)
    args = parser.parse_args(argv)
    cfg = _config()
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.details and not args.summary:
        parser.error('--details requires --summary')
    if args.summary:
        if not (cfg.OUTPUT_ROOT / 'last_batch.json').exists():
            print('ICC alpha control: not_started; no output created')
            return 0
        summary, _ = load_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False) if args.details else compact_summary(summary))
        return 0
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE, output=str(cfg.OUTPUT_ROOT),
            stages=stage_names(), recipe=cfg.recipe(), optimizer_updates=0,
            selection_precedes_final=True, weights_read=False, writes=False), indent=2))
        return 0
    if not args.run:
        parser.print_help()
        return 0
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
