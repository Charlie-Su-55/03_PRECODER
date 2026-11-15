#!/usr/bin/env python3
"""Hybrid-only joint continuation with a frozen historical FDMA reference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import uuid


def _config():
    from configs import twc_v1_k8_hybrid_direction_config as cfg
    return cfg


def _runtime():
    import hybrid_direction_runtime as runtime
    return runtime


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    return ('startup_policy', 'source_preflight', 'hybrid_100', 'hybrid_10000',
        'final_validation', 'directory_export')


def evidence(paths):
    runtime = _runtime()
    files = {}
    for path in paths:
        path = runtime.inside(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Owned regular stage evidence required')
        files[str(path)] = runtime.sha256(path)
    if not files: raise ValueError('Nonempty completed stage evidence required')
    return files


def verify_evidence(record):
    if not record.get('files') or evidence(record['files']) != record['files']:
        raise ValueError('Completed stage evidence changed')


def summary_path():
    return _config().OUTPUT_ROOT/'summary.json'


def load_summary():
    runtime, cfg = _runtime(), _config()
    summary = runtime.read_json(runtime.inside(summary_path()))
    if summary.get('namespace') != cfg.NAMESPACE:
        raise ValueError('Own Hybrid continuation summary required')
    return summary


def verify_registration(summary):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(summary['core'])
    path = runtime.inside(cfg.OUTPUT_ROOT/'preregistration.json')
    if path.is_symlink() or not path.is_file() or runtime.sha256(path) != summary.get('preregistration_sha256'):
        raise ValueError('Immutable Hybrid preregistration changed')
    registered = runtime.read_json(path)
    for key in ('namespace', 'core', 'code', 'recipe', 'recipe_fingerprint', 'source_pins'):
        if registered.get(key) != summary.get(key):
            raise ValueError('Registered experiment identity changed: '+key)
    if summary['recipe'] != cfg.recipe() or summary['recipe_fingerprint'] != cfg.fingerprint():
        raise ValueError('Hybrid continuation recipe changed')
    if summary['source_pins'] != cfg.SOURCE_PINS:
        raise ValueError('Pinned historical reference/source changed')


def save_summary(summary, **updates):
    summary.update(updates, updated=now())
    _runtime().atomic_json(summary_path(), summary)


def stage(summary, name, action, *, reuse=True):
    old = summary['stages'].get(name)
    if old and old.get('status') == 'completed' and reuse:
        verify_evidence(old)
        return old
    attempts = [] if old is None else old.get('attempts', [])+[
        {k: v for k, v in old.items() if k != 'attempts'}]
    record = dict(status='running', started=now(), attempt_id=uuid.uuid4().hex[:12], attempts=attempts)
    summary['stages'][name] = record
    save_summary(summary, status='running', stage=name)
    try:
        details, paths = action(record)
        record.update(status='completed', finished=now(), details=details, files=evidence(paths))
        save_summary(summary)
        return record
    except BaseException as exc:
        record.update(status='failed', finished=now(), error=type(exc).__name__+': '+str(exc))
        save_summary(summary, status='failed', error=record['error'])
        raise


def child(summary, record, script, args):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(summary['core'])
    log = runtime.inside(cfg.OUTPUT_ROOT/'logs'/f"{summary['stage']}_{record['attempt_id']}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    record['log'] = str(log)
    save_summary(summary)
    with log.open('x') as stream:
        result = subprocess.run([sys.executable, '-B', '-u', script, *args], cwd=cfg.ROOT,
            stdout=stream, stderr=subprocess.STDOUT, check=False)
    runtime.guard_core(summary['core'])
    if result.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-15:])[-3500:]
        raise RuntimeError(f'{script} exit={result.returncode}; log={log}\n{tail}')
    return log


def source_preflight(record):
    runtime, cfg = _runtime(), _config()
    device = runtime.configure_cuda()
    accepted = {}
    # Strictly accept BOTH original terminals before any new training.
    for arm in ('fdma', 'hybrid_g4'):
        codec, stats, ce, source = runtime.load_source(arm, device)
        runtime.verify_source_record(arm, source)
        runtime.verify_files(source)
        accepted[arm] = dict(checkpoint_sha256=source['parent_record']['checkpoint_sha256'],
            encoder=runtime.state_digest(codec.encoder), decoder=runtime.state_digest(codec.decoder),
            statistics=runtime.state_digest(stats), CE=runtime.state_digest(ce),
            EMA_updates=int(stats.ema_updates))
        del codec, stats, ce
        gc.collect()
    import torch
    torch.cuda.empty_cache()
    path = cfg.OUTPUT_ROOT/('source_preflight_'+record['attempt_id']+'.json')
    runtime.atomic_json(path, dict(status='passed', production_evidence=True,
        source_pins=cfg.SOURCE_PINS, accepted=accepted, FDMA_new_updates=0,
        Hybrid_new_updates=cfg.UPDATES), exclusive=True)
    return dict(sources=accepted, FDMA_new_updates=0), [path]


def training_action(summary, target, record):
    from train_twc_v1_k8_hybrid_direction import verify_boundary, verify_terminal_record
    cfg = _config()
    directory = cfg.OUTPUT_ROOT/'training'/'hybrid_g4'
    args = ['--train', '--stop-after', str(target)]
    if (directory/'checkpoint_latest.pt').exists(): args.append('--resume')
    log = child(summary, record, 'train_twc_v1_k8_hybrid_direction.py', args)
    if target == 100:
        manifest = verify_boundary()
        files = [directory/'boundary_00100.json', directory/'initialization.json', log]
    else:
        manifest = verify_terminal_record()
        files = [directory/name for name in ('checkpoint_latest.pt', 'manifest.json',
            'terminal_receipt.json', 'initialization.json', 'progress.jsonl')]+[log]
    return dict(step=target, manifest=manifest), files


def evaluation_action(summary, record):
    from evaluate_twc_v1_k8_hybrid_direction import verify_manifest
    cfg = _config()
    output = cfg.OUTPUT_ROOT/'evaluation'/('attempt_'+record['attempt_id'])
    log = child(summary, record, 'evaluate_twc_v1_k8_hybrid_direction.py',
        ['--evaluate', '--output-dir', str(output)])
    manifest = verify_manifest(output)
    return dict(output_dir=str(output)), [output/'manifest.json', log,
        *[output/name for name in manifest['files']]]


def export_action(summary, record):
    import publish_twc_v1_k8_hybrid_direction as publisher
    evaluation = summary['stages']['final_validation']['details']['output_dir']
    report = Path(publisher.export_directory(evaluation, summary_path()))
    publisher.verify_export_directory(report)
    return dict(report_directory=str(report)), list(report.iterdir())


def verify_completed_export(path):
    import publish_twc_v1_k8_hybrid_direction as publisher
    from evaluate_twc_v1_k8_hybrid_direction import verify_manifest
    runtime, cfg = _runtime(), _config()
    summary = load_summary()
    verify_registration(summary)
    path = runtime.inside(path)
    if summary.get('status') != 'completed' or path != runtime.inside(summary['report_directory']):
        raise ValueError('Completed Hybrid result/export binding required')
    for name in stage_names():
        record = summary['stages'].get(name, {})
        if record.get('status') != 'completed': raise ValueError('Incomplete Hybrid experiment stage')
        verify_evidence(record)
    if (summary['stages']['directory_export']['details']['report_directory'] != str(path)
            or summary.get('export_manifest_sha256') != runtime.sha256(path/'export_manifest.json')):
        raise ValueError('Completed report identity changed')
    evaluation = Path(summary['stages']['final_validation']['details']['output_dir'])
    manifest = verify_manifest(evaluation)
    exported = publisher.verify_export_directory(path)
    public = runtime.read_json(path/'provenance.json')
    expected = dict(core=summary['core'], recipe=cfg.recipe(), recipe_fingerprint=cfg.fingerprint(),
        source_core_sha256=cfg.SOURCE_CORE_SHA256, parent_source_pins=cfg.SOURCE_PINS,
        source_records={name: publisher._public_record(value)
            for name, value in manifest['source_records'].items()},
        frozen_signatures=manifest['frozen_signatures'], data_protocol=manifest['data_protocol'],
        experiment_code_commit=summary['code']['commit'],
        evaluation_manifest_sha256=runtime.sha256(evaluation/'manifest.json'),
        evaluation_files=manifest['files'], preregistration_sha256=summary['preregistration_sha256'],
        FDMA_new_training_updates=0, Hybrid_new_training_updates=cfg.UPDATES,
        budgets_equal=False, single_factor_loss_attribution=False,
        all_models_verified_before_data=True, paired_inputs_verified=True)
    if any(public.get(key) != value for key, value in expected.items()):
        raise ValueError('Completed public report provenance changed')
    return exported


def run(args):
    runtime, cfg = _runtime(), _config()
    with runtime.process_lock(cfg.OUTPUT_ROOT/'controller.lock'):
        if args.resume:
            summary = load_summary()
            verify_registration(summary)
            if summary.get('status') == 'completed':
                verify_completed_export(summary['report_directory'])
                print(compact_summary(summary)); return
        else:
            if any(p.name != 'controller.lock' for p in cfg.OUTPUT_ROOT.iterdir()):
                raise FileExistsError('Hybrid output exists; inspect and use explicit --resume')
            summary = dict(namespace=cfg.NAMESPACE, status='starting', stage='startup_policy', created=now(),
                core=runtime.core_identity(), code=runtime.source_identity(), recipe=cfg.recipe(),
                recipe_fingerprint=cfg.fingerprint(), source_pins=cfg.SOURCE_PINS, stages={},
                FDMA_new_updates=0, Hybrid_new_updates=cfg.UPDATES, test_used=False)
            runtime.guard_core(summary['core'])
            runtime.atomic_json(cfg.OUTPUT_ROOT/'preregistration.json', summary, exclusive=True)
            summary['preregistration_sha256'] = runtime.sha256(cfg.OUTPUT_ROOT/'preregistration.json')
            save_summary(summary)
        def admission(record):
            from run_twc_v1_k8_icc_aligned import gpu_admission
            result = gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                min_free_mib=args.min_free_mib, max_gpu_util=args.max_gpu_util)
            path = cfg.OUTPUT_ROOT/('startup_policy_'+record['attempt_id']+'.json')
            runtime.atomic_json(path, result, exclusive=True)
            return result, [path]
        stage(summary, 'startup_policy', admission, reuse=False)
        stage(summary, 'source_preflight', source_preflight, reuse=False)
        for target in (100, cfg.UPDATES):
            stage(summary, 'hybrid_'+str(target), lambda record, t=target: training_action(summary, t, record))
        stage(summary, 'final_validation', lambda record: evaluation_action(summary, record))
        exported = stage(summary, 'directory_export', lambda record: export_action(summary, record))
        report = exported['details']['report_directory']
        save_summary(summary, status='completed', stage='directory_export', report_directory=report,
            export_manifest_sha256=runtime.sha256(Path(report)/'export_manifest.json'), finished=now())
        summary.pop('error', None)
        save_summary(summary)
        verify_completed_export(report)
        print(compact_summary(summary))


def compact_summary(summary):
    cfg, runtime = _config(), _runtime()
    lines = [f"Hybrid direction | {summary['status']} | stage={summary['stage']}",
        'Namespace: '+cfg.NAMESPACE, 'Updated: '+summary.get('updated', 'unknown'),
        'Core commit: '+summary['code']['commit'],
        'Hybrid only: 10000 NEW UE+BS updates; FDMA frozen at old joint10000',
        'Loss: -complete-MU rate + 100 * phase/norm invariant direction error',
        'Original K8/256RE/Ep16+Ed48; six gamma nodes; CE/table frozen; test sealed']
    directory = cfg.OUTPUT_ROOT/'training'/'hybrid_g4'
    if (directory/'manifest.json').exists():
        manifest = runtime.read_json(directory/'manifest.json')
        lines.append(f"Hybrid saved_boundary={manifest['step']}/{cfg.UPDATES} "
            f"ETA={manifest.get('ETA_seconds', 'unknown')}s; EMA={manifest.get('total_ema_updates', 'unknown')}")
    else: lines.append('Hybrid: not started')
    if 'error' in summary: lines.append('ERROR: '+summary['error'])
    if summary.get('status') == 'completed':
        report = Path(summary['report_directory'])
        with (report/'summary.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['scope'] == 'point' and row['metric'] == 'sum_rate_bps_hz':
                    lines.append(f"{row['variant']} gamma={row['gamma_db']} rate={float(row['mean']):.6f}")
        with (report/'comparisons.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['scope'] in ('all_six', 'low_0_5_10') and row['metric'] == 'sum_rate_bps_hz':
                    lines.append(f"{row['comparison']} {row['scope']}: {float(row['mean']):+.6f} "
                        f"batchCI=[{float(row['batch_ci95_low']):+.6f},{float(row['batch_ci95_high']):+.6f}]")
        lines.extend(['Report directory: '+str(report),
            'Upload: '+sys.executable+' -B publish_twc_v1_k8_hybrid_direction.py --publish'])
    lines.append('Exploratory unequal-budget comparison; extra training and new loss are not isolated.')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true'); action.add_argument('--summary', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--min-free-mib', type=int, default=12000)
    parser.add_argument('--max-gpu-util', type=int, default=20)
    args = parser.parse_args(argv)
    if args.resume and not args.run: parser.error('--resume requires --run')
    cfg = _config()
    if args.run: run(args)
    elif args.summary:
        if not summary_path().exists(): print('Hybrid direction | not started | FDMA remains frozen')
        else:
            summary = load_summary(); verify_registration(summary)
            if summary['status'] == 'completed': verify_completed_export(summary['report_directory'])
            print(compact_summary(summary))
    else:
        print(json.dumps(dict(namespace=cfg.NAMESPACE, output=str(cfg.OUTPUT_ROOT), stages=stage_names(),
            recipe=cfg.recipe(), recipe_fingerprint=cfg.fingerprint(), FDMA_new_updates=0,
            Hybrid_new_updates=cfg.UPDATES, weights_read=False, writes=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
