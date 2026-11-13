#!/usr/bin/env python3
"""Offline, zero-update queue for frozen-codeword UL recovery diagnostics."""
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

EVALUATION_FILES = (
    'samples.csv.gz', 'summary.csv', 'comparisons.csv',
    'calibration_bins.csv.gz', 'input_fingerprints.json', 'source_records.json',
)


def _runtime():
    import ul_diagnostics_runtime as runtime
    return runtime


def _config():
    from configs import twc_v1_k8_ul_diagnostics_config as cfg
    return cfg


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    return ('mechanism_diagnostics', 'result_package')


def files_identity(paths):
    runtime = _runtime()
    result = {}
    for value in paths:
        path = runtime.inside(value)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Owned UL diagnostic evidence missing: '+str(path))
        result[str(path)] = runtime.sha256(path)
    if not result:
        raise ValueError('Nonempty UL diagnostic evidence required')
    return result


def verify_evidence(record):
    runtime = _runtime()
    if not record.get('files'):
        raise ValueError('Completed diagnostic stage has no evidence')
    for name, digest in record['files'].items():
        path = runtime.inside(name)
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != digest:
            raise ValueError('Completed diagnostic evidence changed: '+name)


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
        raise ValueError('Own UL diagnostic pointer required')
    path = runtime.inside(pointer['summary'])
    batches = (cfg.OUTPUT_ROOT/'batches').resolve()
    if path.name != 'batch_summary.json' or not path.parent.is_relative_to(batches):
        raise ValueError('Own diagnostic batch summary path required')
    summary = runtime.read_json(path)
    if summary.get('namespace') != cfg.NAMESPACE:
        raise ValueError('UL diagnostic batch namespace changed')
    return summary, path


def verify_preregistration(summary, summary_path):
    """Bind the initial identity bytes before any diagnostic attempt is admitted."""
    runtime = _runtime()
    path = runtime.inside(summary_path.parent/'preregistration.json')
    if (path.is_symlink() or not path.is_file()
            or runtime.sha256(path) != summary.get('preregistration_sha256')):
        raise ValueError('UL diagnostic preregistration bytes changed')
    registered = runtime.read_json(path)
    keys = ('namespace', 'core', 'code', 'recipe', 'recipe_fingerprint',
            'optimizer_updates', 'UE_updates', 'CE_updates', 'EMA_updates',
            'test_used', 'original_outputs_preserved')
    for key in keys:
        if key not in registered or key not in summary or registered[key] != summary[key]:
            raise ValueError('UL diagnostic preregistered identity changed: '+key)


def verify_manifest(directory):
    runtime, cfg = _runtime(), _config()
    directory = runtime.inside(directory)
    subtree = (cfg.OUTPUT_ROOT/'evaluation').resolve()
    if directory == subtree or not directory.is_relative_to(subtree):
        raise ValueError('Own diagnostic evaluation attempt required')
    manifest = runtime.read_json(directory/'manifest.json')
    expected = dict(status='passed', phase='diagnostic', format=cfg.NAMESPACE+'_manifest_v1',
        core=runtime.core_identity(), recipe_fingerprint=cfg.fingerprint(),
        arms=list(cfg.ARMS), variants=list(cfg.VARIANTS), gamma_db=list(cfg.GAMMAS),
        batch_size=cfg.MICRO_BATCH, n_batches=cfg.BATCHES, samples_per_condition=cfg.SAMPLES,
        optimizer_updates=0, UE_updates=0, CE_updates=0, EMA_updates=0, test_used=False,
        paired_inputs_verified=True, frozen_models_verified=True,
        privileged_truth_for_diagnostics_only=True, production_evidence=True,
        source_checkpoints_verified=True, no_downlink_forward=True)
    for name, value in expected.items():
        actual = manifest.get(name)
        if actual != value or isinstance(value, (bool, int)) and type(actual) is not type(value):
            raise ValueError('UL diagnostic identity/budget changed: '+name)
    if manifest.get('data_protocol') != runtime.make_data('cuda:0').protocol:
        raise ValueError('UL diagnostic data protocol changed')
    evidence = manifest.get('files', {})
    if set(evidence) != set(EVALUATION_FILES):
        raise ValueError('Explicit UL diagnostic artifact inventory required')
    for name, digest in evidence.items():
        path = runtime.inside(directory/name)
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or runtime.sha256(path) != digest):
            raise ValueError('UL diagnostic artifact changed: '+name)
    records = runtime.read_json(directory/'source_records.json')
    if records != manifest.get('source_records') or set(records) != set(cfg.ARMS):
        raise ValueError('Both pinned diagnostic source records required')
    runtime.verify_source_records(records)
    return manifest


def evaluation_action(summary, path, record):
    runtime, cfg = _runtime(), _config()
    output = runtime.inside(cfg.OUTPUT_ROOT/'evaluation'/('attempt_'+record['attempt_id']))
    record['output_dir'] = str(output)
    runtime.atomic_json(path, summary)
    log = child(summary, path, record, 'evaluate_twc_v1_k8_ul_diagnostics.py',
        ['--run', '--output-dir', str(output)])
    manifest = verify_manifest(output)
    return dict(output_dir=str(output), manifest=manifest), [
        output/'manifest.json', *[output/name for name in EVALUATION_FILES], log]


def package_action(summary_path, evaluation_directory, record):
    """Package explicit scalar evidence; source weights and channel tensors stay local."""
    runtime, cfg = _runtime(), _config()
    directory = runtime.inside(evaluation_directory)
    verify_preregistration(runtime.read_json(summary_path), summary_path)
    verify_manifest(directory)
    output = runtime.inside(cfg.OUTPUT_ROOT/'packages'/('attempt_'+record['attempt_id']))
    output.mkdir(parents=True, exist_ok=False)
    snapshots = {'preregistration.json': summary_path.parent/'preregistration.json',
                 'diagnostic_manifest.json': directory/'manifest.json'}
    snapshots.update({name: directory/name for name in EVALUATION_FILES})
    hashes = {}
    for name, source in snapshots.items():
        source = runtime.inside(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing explicit diagnostic package artifact: '+str(source))
        with (output/name).open('xb') as stream:
            stream.write(source.read_bytes())
        hashes[name] = runtime.sha256(output/name)
    runtime.atomic_json(output/'experiment_recipe.json', cfg.recipe(), exclusive=True)
    hashes['experiment_recipe.json'] = runtime.sha256(output/'experiment_recipe.json')
    runtime.atomic_json(output/'package_whitelist.json', dict(
        format=cfg.NAMESPACE+'_package_v1', files=hashes,
        digest='SHA256 of exact uncompressed bytes; outer receipt hashes archive bytes',
        contains_model_weights=False, contains_channel_tensors=False,
        privileged_diagnostic_not_deployable=True), exclusive=True)
    names = [*hashes, 'package_whitelist.json']
    archive = output/('ul_diagnostics_small_results_'+record['attempt_id']+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(names):
            stream.write(output/name, arcname=name)
    result = dict(path=str(archive), sha256=runtime.sha256(archive), files=sorted(names))
    receipt = summary_path.parent/('package_'+record['attempt_id']+'.json')
    runtime.atomic_json(receipt, result, exclusive=True)
    return result, [archive, receipt]


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
                raise ValueError('Saved UL diagnostic recipe changed')
            verify_preregistration(summary, path)
            for record in summary['stages'].values():
                if record.get('status') == 'completed':
                    verify_evidence(record)
            previous = summary['stages'].get('mechanism_diagnostics', {})
            if previous.get('status') == 'completed':
                verify_manifest(previous['details']['output_dir'])
        else:
            if any(path.name != 'controller.lock' for path in cfg.OUTPUT_ROOT.iterdir()):
                raise FileExistsError('UL diagnostic output exists; inspect it and use explicit --resume')
            directory = cfg.OUTPUT_ROOT/'batches'/('batch_'+uuid.uuid4().hex)
            directory.mkdir(parents=True, exist_ok=False)
            path = directory/'batch_summary.json'
            summary = dict(namespace=cfg.NAMESPACE, status='starting', created=now(), updated=now(),
                core=runtime.core_identity(), code=runtime.source_identity(), recipe=cfg.recipe(),
                recipe_fingerprint=cfg.fingerprint(), stages={}, test_used=False,
                optimizer_updates=0, UE_updates=0, CE_updates=0, EMA_updates=0,
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
            diagnostic = stage(summary, path, 'mechanism_diagnostics',
                lambda record: evaluation_action(summary, path, record))
            stage(summary, path, 'result_package', lambda record: package_action(
                path, diagnostic['details']['output_dir'], record))
            runtime.guard_core(summary['core'])
            summary.update(status='completed', current_stage='result_package',
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
    lines = [f'UL mechanism diagnostics | {summary.get("status","unknown")} | stage={summary.get("current_stage","unknown")}',
        'Namespace: '+cfg.NAMESPACE, 'Updated: '+summary.get('updated', 'unknown'),
        'Core commit: '+summary.get('code', {}).get('commit', 'unknown'),
        'Frozen inference only: optimizer/UE/CE/EMA updates=0; original weights preserved',
        f'Paired data: {cfg.SAMPLES} MU per arm/gamma; gamma='+','.join(map(str, cfg.GAMMAS)),
        'Comparison: estimated UL versus privileged true-UL/zero-CE-error recovery; same waveform/Y/noise/prior']
    diagnostic = summary.get('stages', {}).get('mechanism_diagnostics', {})
    output = diagnostic.get('output_dir') or diagnostic.get('details', {}).get('output_dir')
    if output:
        directory = runtime.inside(output)
        progress = directory/'progress.json'
        if diagnostic.get('status') == 'running' and progress.is_file():
            lines.append('Progress: '+json.dumps(runtime.read_json(progress), sort_keys=True))
        if diagnostic.get('status') == 'completed':
            with (directory/'summary.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            for arm in cfg.ARMS:
                for variant in cfg.VARIANTS:
                    points = [row for row in rows if row.get('arm') == arm and row.get('variant') == variant]
                    if not points:
                        points = [row for row in rows if row.get('arm') == variant+'_'+arm]
                    points.sort(key=lambda row: float(row['gamma_db']))
                    for metric in ('soft_nmse_linear_mean', 'soft_key_nmse_linear_mean', 'soft_shared_nmse_linear_mean'):
                        if points and all(row.get(metric) not in (None, '') for row in points):
                            lines.append(f'{arm}/{variant} {metric}: '+','.join(f'{float(row[metric]):.6g}' for row in points))
            with (directory/'comparisons.csv').open(newline='', encoding='utf-8') as stream:
                comparisons = list(csv.DictReader(stream))
            for arm in cfg.ARMS:
                for scope in ('all_six', 'low_0_5_10'):
                    for metric in ('soft_nmse_linear', 'soft_shared_nmse_linear'):
                        row = next((row for row in comparisons
                            if row.get('comparison') == arm+'_true_ul_diagnostic_minus_estimated_ul'
                            and row.get('scope') == scope and row.get('metric') == metric), None)
                        if row is not None:
                            lines.append(f'{arm} trueUL-estimated {metric} {scope}: '
                                f'{float(row["mean"]):+.6g} batchCI=['
                                f'{float(row["batch_ci95_low"]):+.6g},'
                                f'{float(row["batch_ci95_high"]):+.6g}]')
            lines.append('For soft NMSE differences, negative means lower recovery error.')
            lines.append('Full diagnostics: '+str(directory))
    if summary.get('status') == 'failed':
        record = summary.get('stages', {}).get(summary.get('current_stage'), {})
        lines.append('ERROR: '+str(record.get('error', summary.get('error', 'unknown')))[:1200])
        if record.get('log_path'):
            lines.append('Exact log: '+record['log_path'])
        if record.get('child_error'):
            lines.append(record['child_error'][-1500:])
    package = summary.get('stages', {}).get('result_package', {})
    lines.append('Small package: '+str(package.get('details', {}).get('path', 'not completed')))
    lines.append('Readout: fixed-model validation sensitivity; privileged truth is not a deployable arm or performance upper bound; test sealed.')
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
            print('UL mechanism diagnostics: not_started; no output created')
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
