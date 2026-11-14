#!/usr/bin/env python3
"""Independent calibration, locked validation and explicit directory export."""
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

from run_twc_v1_k8_icc_aligned import gpu_admission

def _runtime():
    import ul_calibration_runtime as runtime
    return runtime


def _config():
    from configs import twc_v1_k8_ul_calibration_config as cfg
    return cfg


def _publisher():
    import publish_twc_v1_k8_ul_calibration as publisher
    return publisher


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    return ('calibration', 'locked_validation', 'directory_export')


def files_identity(paths):
    runtime = _runtime()
    result = {}
    for value in paths:
        path = runtime.inside(value)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Owned UL calibration evidence missing: '+str(path))
        result[str(path)] = runtime.sha256(path)
    if not result:
        raise ValueError('Nonempty UL calibration evidence required')
    return result


def verify_evidence(record):
    runtime = _runtime()
    if not record.get('files'):
        raise ValueError('Completed calibration stage stage has no evidence')
    for name, digest in record['files'].items():
        path = runtime.inside(name)
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != digest:
            raise ValueError('Completed calibration stage evidence changed: '+name)


def stage(summary, path, name, action, *, reuse_completed=True):
    runtime = _runtime()
    previous = summary['stages'].get(name)
    if previous and previous['status'] == 'completed' and reuse_completed:
        verify_evidence(previous)
        return previous
    attempts = [] if previous is None else previous.get('attempts', []) + [
        {key: value for key, value in previous.items() if key != 'attempts'}]
    record = dict(status='running', started=now(), attempt_id=uuid.uuid4().hex[:12],
                  attempts=attempts)
    summary['stages'][name] = record
    summary.update(status='running', current_stage=name, updated=now())
    runtime.atomic_json(path, summary)
    try:
        details, files = action(record)
        record.update(status='completed', finished=now(), details=details,
                      files=files_identity(files))
        runtime.atomic_json(path, summary)
        return record
    except BaseException as exc:
        record.update(status='failed', finished=now(), error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc())
        summary.update(status='failed', error=record['error'], updated=now())
        runtime.atomic_json(path, summary)
        raise


def child(summary, path, record, script, args):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(summary['core'])
    log = path.parent/(summary['current_stage']+'_'+record['attempt_id']+'.log')
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
    pointer = runtime.read_json(cfg.OUTPUT_ROOT/'last_batch.json')
    if set(pointer) != {'namespace', 'summary'} or pointer['namespace'] != cfg.NAMESPACE:
        raise ValueError('Own UL calibration pointer required')
    path = runtime.inside(pointer['summary'])
    batches = (cfg.OUTPUT_ROOT/'batches').resolve()
    if path.name != 'batch_summary.json' or not path.parent.is_relative_to(batches):
        raise ValueError('Own calibration batch summary path required')
    summary = runtime.read_json(path)
    if summary.get('namespace') != cfg.NAMESPACE:
        raise ValueError('UL calibration batch namespace changed')
    return summary, path


def verify_preregistration(summary, summary_path):
    """Bind the initial identity bytes before any calibration attempt is admitted."""
    runtime = _runtime()
    path = runtime.inside(summary_path.parent/'preregistration.json')
    if (path.is_symlink() or not path.is_file()
            or runtime.sha256(path) != summary.get('preregistration_sha256')):
        raise ValueError('UL calibration preregistration bytes changed')
    registered = runtime.read_json(path)
    keys = ('namespace', 'core', 'code', 'recipe', 'recipe_fingerprint',
            'optimizer_updates', 'UE_updates', 'CE_updates', 'BS_updates', 'EMA_updates',
            'validation_used_for_fitting', 'test_used', 'original_outputs_preserved')
    for key in keys:
        if key not in registered or key not in summary or registered[key] != summary[key]:
            raise ValueError('UL calibration preregistered identity changed: '+key)


def _evaluator():
    import evaluate_twc_v1_k8_ul_calibration as evaluator
    return evaluator


def verify_calibration(directory):
    return _evaluator().verify_calibration(directory)


def verify_validation(directory):
    return _evaluator().verify_validation(directory)


def verify_phase_binding(calibration_directory, validation_directory):
    runtime = _runtime()
    calibration = runtime.inside(calibration_directory)
    verify_calibration(calibration)
    manifest = verify_validation(validation_directory)
    if (runtime.inside(manifest.get('calibration_dir', '')) != calibration
            or manifest.get('model_sha256') != runtime.sha256(calibration/'model.json')
            or manifest.get('calibration_manifest_sha256') != runtime.sha256(calibration/'manifest.json')):
        raise ValueError('Calibration/validation artifact binding changed')
    return manifest


def calibration_action(summary, path, record):
    runtime, cfg, evaluator = _runtime(), _config(), _evaluator()
    output = runtime.inside(cfg.OUTPUT_ROOT/'calibration'/('attempt_'+record['attempt_id']))
    record['output_dir'] = str(output)
    runtime.atomic_json(path, summary)
    log = child(summary, path, record, 'evaluate_twc_v1_k8_ul_calibration.py',
        ['--calibrate', '--output-dir', str(output)])
    manifest = verify_calibration(output)
    return dict(output_dir=str(output), manifest=manifest), [
        output/'manifest.json', *[output/name for name in evaluator.CALIBRATION_FILES], log]


def validation_action(summary, path, record, calibration_directory):
    runtime, cfg, evaluator = _runtime(), _config(), _evaluator()
    calibration = runtime.inside(calibration_directory)
    verify_calibration(calibration)
    locked = dict(model_sha256=runtime.sha256(calibration/'model.json'),
        calibration_manifest_sha256=runtime.sha256(calibration/'manifest.json'))
    output = runtime.inside(cfg.OUTPUT_ROOT/'validation'/('attempt_'+record['attempt_id']))
    record.update(output_dir=str(output), **locked)
    runtime.atomic_json(path, summary)
    log = child(summary, path, record, 'evaluate_twc_v1_k8_ul_calibration.py',
        ['--evaluate', '--calibration-dir', str(calibration), '--output-dir', str(output)])
    manifest = verify_phase_binding(calibration, output)
    if any(manifest.get(key) != value for key, value in locked.items()):
        raise ValueError('Validation did not retain its pre-locked calibration identity')
    return dict(output_dir=str(output), manifest=manifest, **locked), [
        output/'manifest.json', *[output/name for name in evaluator.VALIDATION_FILES],
        calibration/'manifest.json', calibration/'model.json', log]


def export_action(summary_path, calibration_directory, validation_directory, record):
    """Export a checked public scalar directory, without archives or network IO."""
    runtime, cfg = _runtime(), _config()
    calibration = runtime.inside(calibration_directory)
    directory = runtime.inside(validation_directory)
    verify_preregistration(runtime.read_json(summary_path), summary_path)
    verify_phase_binding(calibration, directory)
    output = runtime.inside(cfg.OUTPUT_ROOT/'reports'/('attempt_'+record['attempt_id']))
    result = _publisher().export_directory(summary_path, calibration, directory, output)
    exported = runtime.inside(result['path'])
    manifest = runtime.inside(result['manifest'])
    if exported != output or manifest != output/'export_manifest.json':
        raise ValueError('Own checked calibration report directory required')
    evidence = result.get('files')
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError('Explicit public scalar report inventory required')
    files = [manifest]
    for name, digest in evidence.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Public report leaf filenames required')
        path = runtime.inside(exported/name)
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != digest:
            raise ValueError('Public report artifact changed: '+name)
        files.append(path)
    _publisher().verify_export_binding(summary_path, calibration, directory, exported)
    return result, files


def run(args):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(runtime.core_identity())
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with runtime.process_lock(cfg.OUTPUT_ROOT/'controller.lock'):
        if args.resume:
            summary, path = load_summary()
            runtime.guard_core(summary['core'])
            if (summary.get('recipe_fingerprint') != cfg.fingerprint()
                    or summary.get('recipe') != cfg.recipe()):
                raise ValueError('Saved Calibration/validation recipe changed')
            verify_preregistration(summary, path)
            for record in summary['stages'].values():
                if record.get('status') == 'completed':
                    verify_evidence(record)
            if set(summary['stages'])-set((*stage_names(), 'startup_policy')):
                raise ValueError('Unknown calibration batch stage')
            for name, verifier in (('calibration', verify_calibration),
                                   ('locked_validation', verify_validation)):
                previous = summary['stages'].get(name, {})
                if previous.get('status') == 'completed':
                    verifier(previous['details']['output_dir'])
            calibration = summary['stages'].get('calibration', {})
            validation = summary['stages'].get('locked_validation', {})
            if validation.get('status') == 'completed':
                if calibration.get('status') != 'completed':
                    raise ValueError('Completed validation requires its completed calibration')
                verify_phase_binding(calibration['details']['output_dir'], validation['details']['output_dir'])
            if summary.get('status') == 'completed' and any(
                    summary['stages'].get(name, {}).get('status') != 'completed' for name in stage_names()):
                raise ValueError('Completed calibration batch has incomplete stages')
            exported = summary['stages'].get('directory_export', {})
            if exported.get('status') == 'completed':
                if validation.get('status') != 'completed':
                    raise ValueError('Completed export requires completed validation')
                _publisher().verify_export_binding(path, calibration['details']['output_dir'],
                    validation['details']['output_dir'], exported['details']['path'])
        else:
            if any(path.name != 'controller.lock' for path in cfg.OUTPUT_ROOT.iterdir()):
                raise FileExistsError('UL calibration output exists; inspect it and use explicit --resume')
            directory = cfg.OUTPUT_ROOT/'batches'/('batch_'+uuid.uuid4().hex)
            directory.mkdir(parents=True, exist_ok=False)
            path = directory/'batch_summary.json'
            summary = dict(namespace=cfg.NAMESPACE, status='starting', created=now(), updated=now(),
                core=runtime.core_identity(), code=runtime.source_identity(), recipe=cfg.recipe(),
                recipe_fingerprint=cfg.fingerprint(), stages={}, test_used=False,
                optimizer_updates=0, UE_updates=0, CE_updates=0, BS_updates=0, EMA_updates=0,
                validation_used_for_fitting=False,
                original_outputs_preserved=True)
            preregistration = directory/'preregistration.json'
            runtime.atomic_json(preregistration, summary, exclusive=True)
            summary['preregistration_sha256'] = runtime.sha256(preregistration)
            runtime.atomic_json(path, summary, exclusive=True)
            runtime.atomic_json(cfg.OUTPUT_ROOT/'last_batch.json',
                dict(namespace=cfg.NAMESPACE, summary=str(path)), exclusive=True)
        if summary.get('status') == 'completed':
            print(compact_summary(summary))
            return
        try:
            def admission_action(record):
                admission = gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                    min_free_mib=args.min_free_mib, max_gpu_util=args.max_gpu_util)
                policy = path.parent/('startup_policy_'+record['attempt_id']+'.json')
                runtime.atomic_json(policy, admission, exclusive=True)
                return admission, [policy]
            stage(summary, path, 'startup_policy', admission_action, reuse_completed=False)
            calibration = stage(summary, path, 'calibration',
                lambda record: calibration_action(summary, path, record))
            validation = stage(summary, path, 'locked_validation',
                lambda record: validation_action(summary, path, record, calibration['details']['output_dir']))
            stage(summary, path, 'directory_export', lambda record: export_action(
                path, calibration['details']['output_dir'], validation['details']['output_dir'], record))
            runtime.guard_core(summary['core'])
            summary.update(status='completed', current_stage='directory_export',
                           finished=now(), updated=now())
            summary.pop('error', None)
            summary.pop('traceback', None)
            runtime.atomic_json(path, summary)
            print(compact_summary(summary))
        except BaseException as exc:
            summary.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                           traceback=traceback.format_exc(), updated=now())
            runtime.atomic_json(path, summary)
            raise


def compact_summary(summary):
    runtime, cfg = _runtime(), _config()
    lines = [f'UL independent calibration | {summary.get("status", "unknown")} | stage={summary.get("current_stage", "unknown")}',
        'Namespace: '+cfg.NAMESPACE, 'Updated: '+summary.get('updated', 'unknown'),
        'Core commit: '+summary.get('code', {}).get('commit', 'unknown'),
        'Frozen UE/CE/BS/prior: neural/optimizer/EMA updates=0; statistical calibration is fitted separately',
        f'Calibration: {cfg.CALIBRATION_SAMPLES} MU/gamma; independent validation: {cfg.VALIDATION_SAMPLES} MU/arm/gamma',
        'Variants: '+','.join(cfg.VARIANTS),
        'Gamma: '+','.join(map(str, cfg.GAMMAS)),
        'True UL labels are calibration/scoring only; validation never fits or selects parameters; test sealed']
    for name in ('calibration', 'locked_validation'):
        record = summary.get('stages', {}).get(name, {})
        output = record.get('output_dir') or record.get('details', {}).get('output_dir')
        if not output:
            continue
        directory = runtime.inside(output)
        if record.get('status') == 'running' and (directory/'progress.json').is_file():
            lines.append(name+' progress: '+json.dumps(runtime.read_json(directory/'progress.json'), sort_keys=True))
        if record.get('status') == 'completed':
            lines.append(name+' results: '+str(directory))
            if name == 'calibration':
                lines.append('Locked model SHA256: '+runtime.sha256(directory/'model.json'))
            else:
                with (directory/'summary.csv').open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                for arm in cfg.ARMS:
                    for variant in cfg.VARIANTS:
                        points = sorted((r for r in rows if r.get('arm') == arm
                            and r.get('variant') == variant and r.get('scope') == 'point'),
                            key=lambda r: float(r['gamma_db']))
                        for metric in ('sum_rate_bps_hz', 'dl_nmse_linear',
                                       'soft_actual_to_reported_mse_ratio'):
                            key = metric+'_mean'
                            if points and all(r.get(key) not in (None, '') for r in points):
                                lines.append(f'{arm}/{variant} {metric}: '+','.join(f'{float(r[key]):.6g}' for r in points))
    if summary.get('status') == 'failed':
        record = summary.get('stages', {}).get(summary.get('current_stage'), {})
        lines.append('ERROR: '+str(record.get('error', summary.get('error', 'unknown')))[:1200])
        if record.get('log_path'):
            lines.append('Exact log: '+record['log_path'])
        if record.get('child_error'):
            lines.append(record['child_error'][-1500:])
    report = summary.get('stages', {}).get('directory_export', {}).get('details', {}).get('path')
    lines.append('Report directory: '+str(report or 'not completed'))
    if report:
        report = runtime.inside(report)
        lines.append('Upload command: '+sys.executable+' -B publish_twc_v1_k8_ul_calibration.py --publish --export-dir '+str(report))
    lines.append('Intervals condition on frozen models and fitted calibration; not multiple training seeds or guaranteed Hybrid gain.')
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
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.details and not args.summary:
        parser.error('--details requires --summary')
    cfg = _config()
    if args.summary:
        if not (cfg.OUTPUT_ROOT/'last_batch.json').exists():
            print('UL independent calibration: not_started; no output created')
            return 0
        summary, _ = load_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False) if args.details else compact_summary(summary))
        return 0
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE, output=str(cfg.OUTPUT_ROOT),
            stages=stage_names(), recipe=cfg.recipe(), weights_read=False, writes=False), indent=2))
        return 0
    if not args.run:
        parser.print_help()
        return 0
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
