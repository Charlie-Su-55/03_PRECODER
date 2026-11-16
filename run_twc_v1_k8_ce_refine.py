#!/usr/bin/env python3
"""Independent paired CE continuation; no UE, BS-codec or FDMA updates."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid


def _config():
    from configs import twc_v1_k8_ce_refine_config as cfg
    return cfg


def _runtime():
    import ce_refine_runtime as runtime
    return runtime


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    cfg = _config()
    return ('startup_policy', 'source_preflight',
        *(f'{name}_{step}' for name in cfg.TRAJECTORIES for step in (100, cfg.UPDATES)),
        'training_pairing', 'calibration_and_validation', 'directory_export')


def summary_path():
    return _config().OUTPUT_ROOT/'summary.json'


def source_pins():
    cfg = _config()
    return dict(checkpoint_sha256=cfg.SOURCE_CHECKPOINT_SHA256,
        terminal_sha256=cfg.SOURCE_TERMINAL_SHA256, state_sha256=cfg.SOURCE_STATE_SHA256,
        updates=cfg.SOURCE_UPDATES, commit=cfg.SOURCE_COMMIT)


def evidence(paths):
    runtime = _runtime()
    records = {}
    for path in paths:
        path = runtime.inside(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Owned regular CE stage evidence required')
        records[str(path)] = runtime.sha256(path)
    if not records:
        raise ValueError('Nonempty completed-stage evidence required')
    return records


def verify_evidence(record):
    if not record.get('files') or evidence(record['files']) != record['files']:
        raise ValueError('Completed CE stage evidence changed')


def load_summary():
    value = _runtime().read_json(_runtime().inside(summary_path()))
    if value.get('namespace') != _config().NAMESPACE:
        raise ValueError('Own CE-continuation summary required')
    return value


def verify_registration(summary):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(summary['core'])
    path = runtime.inside(cfg.OUTPUT_ROOT/'preregistration.json')
    if not path.is_file() or runtime.sha256(path) != summary.get('preregistration_sha256'):
        raise ValueError('Immutable CE registration changed')
    registered = runtime.read_json(path)
    for key in ('namespace', 'core', 'code', 'recipe', 'recipe_fingerprint', 'source_pins'):
        if registered.get(key) != summary.get(key):
            raise ValueError('CE registered identity changed: '+key)
    if (summary['recipe'] != cfg.recipe() or summary['recipe_fingerprint'] != cfg.fingerprint()
            or summary['source_pins'] != source_pins()):
        raise ValueError('CE source, recipe or declared budget changed')


def save_summary(summary, **updates):
    summary.update(updates, updated=now())
    _runtime().atomic_json(summary_path(), summary)


def stage(summary, name, action, *, reuse=True):
    old = summary['stages'].get(name)
    if old and old.get('status') == 'completed' and reuse:
        verify_evidence(old)
        return old
    attempts = [] if old is None else old.get('attempts', [])+[
        {key: value for key, value in old.items() if key != 'attempts'}]
    record = dict(status='running', attempt_id=uuid.uuid4().hex[:12], started=now(), attempts=attempts)
    summary['stages'][name] = record
    save_summary(summary, status='running', stage=name)
    try:
        details, paths = action(record)
        record.update(status='completed', finished=now(), details=details, files=evidence(paths))
        save_summary(summary)
        return record
    except BaseException as exc:
        record.update(status='failed', finished=now(), error=f'{type(exc).__name__}: {exc}')
        save_summary(summary, status='failed', error=record['error'])
        raise


def child(summary, record, script, args):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(summary['core'])
    log = runtime.inside(cfg.OUTPUT_ROOT/'logs'/f"{summary['stage']}_{record['attempt_id']}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    record['log'] = str(log)
    save_summary(summary)
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', '-u', script, *args], cwd=cfg.ROOT,
            stdout=stream, stderr=subprocess.STDOUT, check=False)
    runtime.guard_core(summary['core'])
    if result.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-15:])[-3500:]
        raise RuntimeError(f'{script} exit={result.returncode}; log={log}\n{tail}')
    return log


def source_preflight(record):
    """Actual CUDA pilot-only CE forward/backward; zero optimizer updates."""
    import torch
    runtime, cfg = _runtime(), _config()
    from twc_v1_k8_ce_refine import objective
    device = runtime.configure_cuda()
    model, source = runtime.load_source(device)
    runtime.verify_source(source)
    if source.get('fixture'):
        raise ValueError('Synthetic training fixtures cannot serve as production sources')
    before = runtime.state_digest(model)
    gen = torch.Generator(device=device).manual_seed(cfg.SEED)
    channels = torch.view_as_complex(torch.randn(4, 8, 32, 256, 2,
        generator=gen, device=device).contiguous())/2**.5
    gamma, noise, polarity = runtime.batch_random('ce_train', 1, device)
    model.train().requires_grad_(True)
    prediction, _, _, _ = runtime.pilot_forward(model, {'H_ul': channels}, gamma, noise, polarity)
    loss, _ = objective(prediction, channels, 'geometry')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Finite actual CUDA CE gradients required')
    norm = sum(float(g.detach().abs().sum()) for g in gradients)
    if norm <= 0 or runtime.state_digest(model) != before:
        raise ValueError('Nonzero CE gradients with unchanged source weights required')
    runtime.verify_source(source)
    model.zero_grad(set_to_none=True)
    output = cfg.OUTPUT_ROOT/('source_preflight_'+record['attempt_id']+'.json')
    details = dict(status='passed', production_evidence=True, synthetic_CUDA_preflight=True,
        source_pins=source_pins(), source_state_sha256=before, CE_optimizer_updates=0,
        UE_updates=0, BS_updates=0, FDMA_updates=0, EMA_commits=0,
        absolute_gradient_sum=norm, parameters=sum(p.numel() for p in model.parameters()),
        model_performance_verified=False)
    runtime.atomic_json(output, details, exclusive=True)
    return details, [output]


def training_action(summary, trajectory, target, record):
    cfg, runtime = _config(), _runtime()
    from train_twc_v1_k8_ce_refine import verify_terminal
    cfg.key(trajectory)
    directory = cfg.OUTPUT_ROOT/'training'/trajectory
    args = ['--train', '--trajectory', trajectory, '--stop-after', str(target)]
    if (directory/'checkpoint_latest.pt').is_file():
        args.append('--resume')
    log = child(summary, record, 'train_twc_v1_k8_ce_refine.py', args)
    if target == 100:
        path = directory/'boundary_00100.json'
        accepted = runtime.read_json(path)
        if accepted.get('step') != 100 or accepted.get('production_evidence') is not True:
            raise ValueError('Actual first100 CE-update receipt required')
        paths = [path, directory/'initialization.json', log]
    else:
        accepted = verify_terminal(trajectory)
        paths = [directory/name for name in ('checkpoint_latest.pt', 'manifest.json',
            'terminal_receipt.json', 'initialization.json', 'progress.jsonl')]+[log]
    return dict(trajectory=trajectory, step=target, manifest=accepted), paths


def check_training_pairing(records):
    """Match every input through the accumulated physical-input trace."""
    cfg = _config()
    if set(records) != set(cfg.TRAJECTORIES):
        raise ValueError('Both completed CE trajectories required')
    first, second = (records[key] for key in cfg.TRAJECTORIES)
    for record in (first, second):
        if (record.get('production_evidence') is not True or record.get('step') != cfg.UPDATES
                or record.get('counters') != cfg.counters(cfg.UPDATES)):
            raise ValueError('Actual complete CE-only trajectory required')
    for field in ('source_identity', 'data_protocol', 'training_trace'):
        if not first.get(field) or first.get(field) != second.get(field):
            raise ValueError('Paired CE input/source identity changed: '+field)
    return dict(status='passed', actual_input_trace_matched=True,
        source_pins=source_pins(), trajectories=list(cfg.TRAJECTORIES),
        updates_each=cfg.UPDATES, counters=cfg.counters(cfg.UPDATES),
        training_trace=first['training_trace'], test_used=False)


def pairing_action(record):
    cfg, runtime = _config(), _runtime()
    from train_twc_v1_k8_ce_refine import verify_terminal
    paired = check_training_pairing({name: verify_terminal(name) for name in cfg.TRAJECTORIES})
    path = cfg.OUTPUT_ROOT/('training_pairing_'+record['attempt_id']+'.json')
    runtime.atomic_json(path, paired, exclusive=True)
    return paired, [path]


def evaluation_action(summary, record):
    cfg, runtime = _config(), _runtime()
    output = cfg.OUTPUT_ROOT/'evaluation'/('attempt_'+record['attempt_id'])
    log = child(summary, record, 'evaluate_twc_v1_k8_ce_refine.py',
        ['--run', '--output-dir', str(output)])
    manifest = runtime.read_json(output/'manifest.json')
    if (manifest.get('namespace') != cfg.NAMESPACE or manifest.get('status') != 'completed'
            or manifest.get('production_evidence') is not True):
        raise ValueError('Completed real CE calibration/validation required')
    files = [output/name for name in manifest['files']]+[output/'manifest.json', log]
    summary['evaluation_directory'] = str(output)
    save_summary(summary)
    return dict(evaluation_directory=str(output), manifest=manifest), files


def export_action(summary, record):
    cfg = _config()
    from publish_twc_v1_k8_ce_refine import export_directory
    output = cfg.OUTPUT_ROOT/'reports'/('attempt_'+record['attempt_id'])
    exported = export_directory(summary_path(), summary['evaluation_directory'], output)
    return dict(exported, report_directory=str(output)), list(output.iterdir())


def verify_completed_export(path=None):
    summary = load_summary()
    verify_registration(summary)
    if summary.get('status') != 'completed':
        raise ValueError('CE run not completed')
    for name in stage_names():
        record = summary['stages'].get(name, {})
        if record.get('status') != 'completed':
            raise ValueError('Incomplete CE stage: '+name)
        verify_evidence(record)
    directory = Path(summary['report_directory'])
    if path is not None and Path(path).resolve() != directory.resolve():
        raise ValueError('Registered completed CE report required')
    from publish_twc_v1_k8_ce_refine import verify_export_directory
    manifest = verify_export_directory(directory)
    if _runtime().sha256(directory/'export_manifest.json') != summary['export_manifest_sha256']:
        raise ValueError('Completed CE export changed')
    return manifest


def run(args):
    runtime, cfg = _runtime(), _config()
    with runtime.process_lock(cfg.OUTPUT_ROOT/'controller.lock'):
        if args.resume:
            summary = load_summary()
            verify_registration(summary)
            if summary.get('status') == 'completed':
                verify_completed_export()
                print(compact_summary(summary))
                return
        else:
            if any(p.name != 'controller.lock' for p in cfg.OUTPUT_ROOT.iterdir()):
                raise FileExistsError('CE output exists; preserve it and inspect before explicit --resume')
            summary = dict(namespace=cfg.NAMESPACE, status='starting', stage='startup_policy',
                created=now(), core=runtime.core_identity(), code=runtime.source_identity(),
                recipe=cfg.recipe(), recipe_fingerprint=cfg.fingerprint(), source_pins=source_pins(),
                stages={}, total_new_CE_updates=2*cfg.UPDATES, UE_updates=0, BS_updates=0,
                FDMA_updates=0, EMA_commits=0, test_used=False)
            runtime.guard_core(summary['core'])
            runtime.atomic_json(cfg.OUTPUT_ROOT/'preregistration.json', summary, exclusive=True)
            summary['preregistration_sha256'] = runtime.sha256(cfg.OUTPUT_ROOT/'preregistration.json')
            save_summary(summary)
        def admission(record):
            from run_twc_v1_k8_icc_aligned import gpu_admission
            accepted = gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                min_free_mib=args.min_free_mib, max_gpu_util=args.max_gpu_util)
            path = cfg.OUTPUT_ROOT/('startup_policy_'+record['attempt_id']+'.json')
            runtime.atomic_json(path, accepted, exclusive=True)
            return accepted, [path]
        stage(summary, 'startup_policy', admission, reuse=False)
        stage(summary, 'source_preflight', source_preflight, reuse=False)
        for name in cfg.TRAJECTORIES:
            for target in (100, cfg.UPDATES):
                stage(summary, f'{name}_{target}',
                    lambda record, t=name, n=target: training_action(summary, t, n, record))
        stage(summary, 'training_pairing', pairing_action)
        stage(summary, 'calibration_and_validation', lambda record: evaluation_action(summary, record))
        export = stage(summary, 'directory_export', lambda record: export_action(summary, record))
        directory = export['details']['report_directory']
        summary.pop('error', None)
        save_summary(summary, status='completed', stage='directory_export', report_directory=directory,
            export_manifest_sha256=runtime.sha256(Path(directory)/'export_manifest.json'), finished=now())
        verify_completed_export()
        print(compact_summary(summary))


def compact_summary(summary):
    cfg, runtime = _config(), _runtime()
    lines = [f"CE multi-SNR refinement | {summary['status']} | stage={summary['stage']}",
        'Namespace: '+cfg.NAMESPACE, 'Updated: '+summary.get('updated', 'unknown'),
        'Core commit: '+summary['code']['commit'],
        'CE only: two independent trajectories x60000 NEW updates; each source5000 -> terminal65000',
        'UE/BS-codec/FDMA/EMA updates=0; old CE and old experiments preserved; test sealed',
        'NMSE control versus NMSE+direction+inter-user Gram; balanced six SNR nodes',
        'Original pilot64/Ep16/rank8 LS/pilot-only U-Net; five off-node midpoints also evaluated',
        'Source CE already trained across0..25dB; midpoints test continuation interpolation, not unseen-range generalization']
    manifests = {}
    for name in cfg.TRAJECTORIES:
        path = cfg.OUTPUT_ROOT/'training'/name/'manifest.json'
        if not path.exists():
            lines.append(name+': not started')
            continue
        value = runtime.read_json(path)
        manifests[name] = value
        eta = str(value.get('ETA_seconds', 'unknown'))+'s' if value['step'] else 'pending measured save'
        lines.append(f"{name}: saved_boundary={value['step']}/{cfg.UPDATES} "
            f"total_CE={cfg.SOURCE_UPDATES+value['step']} ETA={eta}; "
            f"checkpoint_updated={value.get('updated', 'unknown')}")
    speeds = [value.get('seconds_per_update', 0) for value in manifests.values()
        if value.get('seconds_per_update', 0) > 0]
    if speeds:
        reference = speeds[-1]
        remaining = sum((cfg.UPDATES-manifests.get(name, {}).get('step', 0))*
            (manifests.get(name, {}).get('seconds_per_update', 0) or reference)
            for name in cfg.TRAJECTORIES)
        lines.append(f'Estimated remaining training (both trajectories): {remaining:.0f}s; evaluation excluded')
    if 'error' in summary:
        lines.append('ERROR: '+summary['error'])
    if summary.get('status') == 'completed':
        report = Path(summary['report_directory'])
        with (report/'summary.csv').open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['scope'] == 'all_six' and row['metric'] in (
                        'ul_nmse_linear', 'direction_error', 'gram_error', 'shared_error_linear'):
                    lines.append(f"{row['variant']} {row['metric']}: {float(row['mean']):.7g}")
        lines.extend(['Report directory: '+str(report),
            'Upload: '+sys.executable+' -B publish_twc_v1_k8_ce_refine.py --publish'])
    lines.append('CE/recovery diagnostics only; downlink sum-rate gains require later codec validation.')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true')
    action.add_argument('--summary', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--min-free-mib', type=int, default=12000)
    parser.add_argument('--max-gpu-util', type=int, default=20)
    args = parser.parse_args(argv)
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.min_free_mib < 0 or not 0 <= args.max_gpu_util <= 100:
        parser.error('GPU thresholds outside physical range')
    cfg = _config()
    if args.run:
        run(args)
    elif args.summary:
        if not summary_path().exists():
            print('CE multi-SNR refinement | not started | UE/BS-codec/FDMA remain untouched')
        else:
            summary = load_summary()
            verify_registration(summary)
            if summary.get('status') == 'completed':
                verify_completed_export()
            print(compact_summary(summary))
    else:
        print(json.dumps(dict(namespace=cfg.NAMESPACE, output=str(cfg.OUTPUT_ROOT),
            stages=stage_names(), recipe=cfg.recipe(), recipe_fingerprint=cfg.fingerprint(),
            total_new_CE_updates=2*cfg.UPDATES, UE_updates=0, BS_updates=0, FDMA_updates=0,
            weights_read=False, writes=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
