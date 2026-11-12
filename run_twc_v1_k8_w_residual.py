#!/usr/bin/env python3
"""Explicit paired queue for matched RZF continuation and residual-W training."""
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
    import w_residual_runtime as runtime
    return runtime


def _config():
    from configs import twc_v1_k8_w_residual_config as cfg
    return cfg


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    cfg = _config()
    return ('source_preflight', *(f'{cfg.key(arm, method)}_{target}'
        for arm in cfg.ARMS for method in cfg.TRAIN_METHODS for target in (100, cfg.UPDATES)),
        'final_validation', 'result_package')


def files_identity(paths):
    runtime = _runtime()
    result = {}
    for path in paths:
        path = runtime.inside(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Owned residual-W evidence missing: ' + str(path))
        result[str(path)] = runtime.sha256(path)
    if not result:
        raise ValueError('Nonempty completed residual-W evidence required')
    return result


def verify_evidence(record):
    runtime = _runtime()
    if not record.get('files'):
        raise ValueError('Completed residual-W stage has no evidence')
    for name, digest in record['files'].items():
        if runtime.sha256(runtime.inside(name)) != digest:
            raise ValueError('Completed residual-W evidence changed: ' + name)


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
        raise ValueError('Own residual-W pointer required')
    path = runtime.inside(pointer['summary'])
    summary = runtime.read_json(path)
    if summary.get('namespace') != cfg.NAMESPACE:
        raise ValueError('Residual-W batch namespace changed')
    return summary, path


def verify_manifest(directory, phase):
    runtime, cfg = _runtime(), _config()
    directory = runtime.inside(directory)
    manifest = runtime.read_json(directory / 'manifest.json')
    if (manifest.get('status') != 'passed' or manifest.get('phase') != phase
            or manifest.get('core') != runtime.core_identity()
            or manifest.get('recipe_fingerprint') != cfg.study_fingerprint()):
        raise ValueError('Residual-W preflight/evaluation identity changed')
    evidence = manifest.get('files', {})
    if not evidence:
        raise ValueError('Residual-W preflight/evaluation evidence incomplete')
    if phase == 'preflight':
        if ('source_checks.json' not in evidence or manifest.get('optimizer_updates') != 0
                or manifest.get('test_used') is not False):
            raise ValueError('Residual-W source preflight evidence incomplete')
    elif phase == 'final':
        required = {'samples.csv.gz', 'summary.csv', 'comparisons.csv',
                    'input_fingerprints.json', 'training_pairing.json'}
        if not required <= set(evidence):
            raise ValueError('Residual-W final artifacts incomplete')
        expected = dict(batch_size=cfg.MICRO_BATCH, n_batches=cfg.FINAL_BATCHES,
            samples_per_condition=cfg.MICRO_BATCH*cfg.FINAL_BATCHES,
            gamma_db=list(cfg.GAMMAS), methods=list(cfg.METHODS), test_used=False,
            no_optimizer=True, frozen_models_verified=True, transmitted_precoder_scored=True,
            training_pairing_verified=True)
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError('Residual-W final budget/policy changed: ' + name)
    else:
        raise ValueError('Declared residual-W phase required')
    for name, digest in evidence.items():
        if Path(name).name != name or runtime.sha256(runtime.inside(directory/name)) != digest:
            raise ValueError('Residual-W preflight/evaluation artifact changed: ' + name)
    if phase == 'final' and runtime.read_json(directory/'training_pairing.json').get('status') != 'passed':
        raise ValueError('Production residual-W training pairing evidence required')
    return manifest


def evaluation_action(summary, path, phase, record):
    cfg = _config()
    if phase == 'preflight':
        output = cfg.OUTPUT_ROOT/'preflight'/('attempt_'+record['attempt_id'])
    elif phase == 'final':
        output = cfg.OUTPUT_ROOT/'evaluation'/'final'/('attempt_'+record['attempt_id'])
    else:
        raise ValueError('Preflight/final phase required')
    log = child(summary, path, record, 'evaluate_twc_v1_k8_w_residual.py',
        ['--run', '--phase', phase, '--output-dir', str(output)])
    manifest = verify_manifest(output, phase)
    return dict(output_dir=str(output), manifest=manifest), [
        *[p for p in output.rglob('*') if p.is_file()], log]


def training_action(summary, path, arm, method, target, record):
    runtime, cfg = _runtime(), _config()
    directory = runtime.inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm, method))
    args = ['--run', '--arm', arm, '--method', method, '--stop-after', str(target)]
    if (directory/'checkpoint_latest.pt').exists():
        manifest = runtime.read_json(directory/'manifest.json')
        if manifest['step'] > target:
            raise ValueError('Cannot rewind an unrecorded residual-W training stage')
        args.append('--resume')
    log = child(summary, path, record, 'train_twc_v1_k8_w_residual.py', args)
    manifest = runtime.read_json(directory/'manifest.json')
    expected = dict(arm=arm, method=method, step=target, counters=cfg.counters(target),
                    core=summary['core'], recipe_fingerprint=cfg.fingerprint(arm, method))
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError('Residual-W training boundary changed: '+name)
    if manifest.get('checkpoint_sha256') != runtime.sha256(directory/'checkpoint_latest.pt'):
        raise ValueError('Residual-W checkpoint bytes/receipt changed')
    files = [log, directory/'initialization.json']
    if target == 100:
        receipt = directory/'boundary_00100.json'
    else:
        if manifest.get('status') != 'completed':
            raise ValueError('Residual-W fixed terminal is incomplete')
        receipt = directory/'terminal_receipt.json'
        files.append(directory/'checkpoint_latest.pt')
    if runtime.read_json(receipt) != manifest:
        raise ValueError('Residual-W immutable boundary receipt changed')
    files.append(receipt)
    for step in cfg.MONITOR_STEPS:
        if step <= target:
            receipt = directory/'monitor'/f'step_{step:05d}_receipt.json'
            monitor = runtime.read_json(receipt)
            for name, digest in monitor['files'].items():
                if runtime.sha256(runtime.inside(name)) != digest:
                    raise ValueError('Residual-W monitor evidence changed')
            files.extend([receipt, *map(Path, monitor['files'])])
    return manifest, files


def package_action(summary_path, preflight_directory, final_directory, record):
    """Archive explicit scalar diagnostics and receipts; never model weights."""
    runtime, cfg = _runtime(), _config()
    verify_manifest(preflight_directory, 'preflight')
    verify_manifest(final_directory, 'final')
    output = runtime.inside(cfg.OUTPUT_ROOT/'packages'/('attempt_'+record['attempt_id']))
    output.mkdir(parents=True, exist_ok=False)
    final_directory, preflight_directory = Path(final_directory), Path(preflight_directory)
    snapshots = {
        'preregistration.json': summary_path.parent/'preregistration.json',
        'preflight_manifest.json': preflight_directory/'manifest.json',
        'source_checks.json': preflight_directory/'source_checks.json',
        'final_manifest.json': final_directory/'manifest.json',
        'summary.csv': final_directory/'summary.csv',
        'comparisons.csv': final_directory/'comparisons.csv',
        'input_fingerprints.json': final_directory/'input_fingerprints.json',
        'training_pairing.json': final_directory/'training_pairing.json',
        'samples.csv.gz': final_directory/'samples.csv.gz',
    }
    # Initialization includes the immutable source/warm-start identity. Terminal
    # receipts include additional counters and checkpoint digests, not tensors.
    for arm in cfg.ARMS:
        for method in cfg.TRAIN_METHODS:
            key = cfg.key(arm, method)
            directory = cfg.OUTPUT_ROOT/'training'/key
            snapshots[key+'_initialization.json'] = directory/'initialization.json'
            snapshots[key+'_terminal_receipt.json'] = directory/'terminal_receipt.json'
            snapshots[key+'_progress.jsonl'] = directory/'progress.jsonl'
    hashes = {}
    for name, source in snapshots.items():
        source = runtime.inside(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError('Missing explicit residual-W package evidence: '+str(source))
        with (output/name).open('xb') as stream:
            stream.write(source.read_bytes())
        hashes[name] = runtime.sha256(output/name)
    runtime.atomic_json(output/'experiment_recipe.json', cfg.study_recipe(), exclusive=True)
    hashes['experiment_recipe.json'] = runtime.sha256(output/'experiment_recipe.json')
    runtime.atomic_json(output/'package_whitelist.json', dict(
        format=cfg.NAMESPACE+'_package_v1', files=hashes,
        digest='SHA256 of exact uncompressed file bytes; outer receipt hashes archive bytes',
        contains_model_weights=False, contains_channel_tensors=False), exclusive=True)
    names = [*hashes, 'package_whitelist.json']
    archive = output/('w_residual_small_results_'+record['attempt_id']+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(names):
            stream.write(output/name, arcname=name)
    result = dict(path=str(archive), sha256=runtime.sha256(archive), files=sorted(names))
    receipt = summary_path.parent/('package_'+record['attempt_id']+'.json')
    runtime.atomic_json(receipt, result, exclusive=True)
    return result, [archive, receipt]


def run(args):
    runtime, cfg = _runtime(), _config()
    if sys.platform == 'win32':
        raise RuntimeError('Residual-W server queue requires Linux CUDA')
    runtime.guard_core(runtime.core_identity())
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with runtime.process_lock(cfg.OUTPUT_ROOT/'controller.lock'):
        if args.resume:
            summary, path = load_summary()
            runtime.guard_core(summary['core'])
            for record in summary['stages'].values():
                if record['status'] == 'completed':
                    verify_evidence(record)
        else:
            if any((cfg.OUTPUT_ROOT/name).exists() for name in
                   ('last_batch.json', 'training', 'preflight', 'evaluation', 'batches')):
                raise FileExistsError('Own residual-W experiment exists; explicit --resume required')
            directory = cfg.OUTPUT_ROOT/'batches'/('batch_'+uuid.uuid4().hex)
            directory.mkdir(parents=True, exist_ok=False)
            path = directory/'batch_summary.json'
            summary = dict(namespace=cfg.NAMESPACE, status='starting', created=now(), updated=now(),
                core=runtime.core_identity(), code=runtime.source_identity(), recipe=cfg.study_recipe(),
                stages={}, test_used=False, original_outputs_preserved=True)
            runtime.atomic_json(directory/'preregistration.json', summary, exclusive=True)
            runtime.atomic_json(path, summary, exclusive=True)
            runtime.atomic_json(cfg.OUTPUT_ROOT/'last_batch.json',
                dict(namespace=cfg.NAMESPACE, summary=str(path)), exclusive=True)
        if summary.get('status') == 'completed':
            print(compact_summary(summary))
            return
        try:
            summary['current_stage'] = 'startup_policy'
            admission = gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                min_free_mib=args.min_free_mib, max_gpu_util=args.max_gpu_util)
            policy = path.parent/('startup_policy_'+uuid.uuid4().hex[:12]+'.json')
            runtime.atomic_json(policy, admission, exclusive=True)
            summary.setdefault('startup_history', []).append(admission)
            summary['stages']['startup_policy'] = dict(status='completed', details=admission,
                files=files_identity([policy]), finished=now())
            runtime.atomic_json(path, summary)
            preflight = stage(summary, path, 'source_preflight',
                lambda record: evaluation_action(summary, path, 'preflight', record))
            for arm in cfg.ARMS:
                for method in cfg.TRAIN_METHODS:
                    for target in (100, cfg.UPDATES):
                        stage(summary, path, f'{cfg.key(arm,method)}_{target}',
                            lambda record,a=arm,m=method,t=target: training_action(summary,path,a,m,t,record))
            final = stage(summary, path, 'final_validation',
                lambda record: evaluation_action(summary, path, 'final', record))
            stage(summary, path, 'result_package', lambda record: package_action(path,
                preflight['details']['output_dir'], final['details']['output_dir'], record))
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
    lines = [f'W residual control | {summary.get("status","unknown")} | stage={summary.get("current_stage","unknown")}',
        'Namespace: '+cfg.NAMESPACE, 'Updated: '+summary.get('updated','unknown'),
        'Core commit: '+summary.get('code',{}).get('commit','unknown'),
        f'Budget: each arm/method {cfg.UPDATES} NEW updates; rate5000 source replayed unchanged',
        'Protocol: K8/256RE/Ep16+Ed48; common inputs; UE/CE/prior frozen; test sealed']
    for arm in cfg.ARMS:
        for method in cfg.TRAIN_METHODS:
            key = cfg.key(arm,method)
            manifest = cfg.OUTPUT_ROOT/'training'/key/'manifest.json'
            if not manifest.is_file():
                lines.append(key+': not started')
                continue
            row = runtime.read_json(manifest)
            lines.append(f'{key}: saved_boundary={row.get("step","unknown")}/{cfg.UPDATES} '
                f'ETA={row.get("ETA_seconds","unknown")}s recent={row.get("updated","unknown")}')
    if summary.get('status') == 'failed':
        record = summary.get('stages',{}).get(summary.get('current_stage'),{})
        lines.append('ERROR: '+str(record.get('error',summary.get('error','unknown')))[:1200])
        if record.get('log_path'):
            lines.append('Exact log: '+record['log_path'])
        if record.get('child_error'):
            lines.append(record['child_error'][-1500:])
    final = summary.get('stages',{}).get('final_validation',{})
    if final.get('status') == 'completed':
        output = Path(final['details']['output_dir'])
        with (output/'summary.csv').open(newline='',encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        for policy in cfg.POLICIES:
            for arm in cfg.ARMS:
                name = cfg.method_name(policy,arm)
                points = sorted((row for row in rows if row['arm']==name),
                                key=lambda row:float(row['gamma_db']))
                lines.append(name+' rate: '+','.join(f'{float(row["sum_rate_bps_hz_mean"]):.5f}' for row in points))
        with (output/'comparisons.csv').open(newline='',encoding='utf-8') as stream:
            comparisons = list(csv.DictReader(stream))
        primary = [f'{arm}_w_residual_minus_rzf_continue' for arm in cfg.ARMS]
        primary += ['interaction_w_residual_minus_rzf_continue']
        secondary = [f'{policy}_hybrid_minus_fdma' for policy in cfg.POLICIES]
        for label in primary+secondary:
            for scope in ('all_six','low_0_5_10'):
                row = next(row for row in comparisons if row['comparison']==label
                    and row['metric']=='sum_rate_bps_hz' and row['scope']==scope)
                lines.append(f'{label} {scope}: {float(row["mean"]):+.5f} '
                    f'batchCI=[{float(row["batch_ci95_low"]):+.5f},{float(row["batch_ci95_high"]):+.5f}]')
        lines.append('Full results: '+str(output))
    package = summary.get('stages',{}).get('result_package',{})
    lines.append('Small package: '+str(package.get('details',{}).get('path','not completed')))
    lines.append('Intervals: fixed-model validation sampling; one training seed; relative gains do not imply Hybrid wins.')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--run',action='store_true')
    actions.add_argument('--dry-run',action='store_true')
    actions.add_argument('--summary',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--details',action='store_true')
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--min-free-mib',type=int,default=12000)
    parser.add_argument('--max-gpu-util',type=int,default=20)
    args = parser.parse_args(argv)
    cfg = _config()
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.details and not args.summary:
        parser.error('--details requires --summary')
    if args.summary:
        if not (cfg.OUTPUT_ROOT/'last_batch.json').exists():
            print('W residual control: not_started; no output created')
            return 0
        summary,_ = load_summary()
        print(json.dumps(summary,indent=2,ensure_ascii=False) if args.details else compact_summary(summary))
        return 0
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,output=str(cfg.OUTPUT_ROOT),
            stages=stage_names(),recipe=cfg.study_recipe(),weights_read=False,writes=False),indent=2))
        return 0
    if not args.run:
        parser.print_help()
        return 0
    run(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
