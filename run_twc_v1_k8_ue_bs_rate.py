#!/usr/bin/env python3
"""Explicit UE+BS joint-rate queue with archived matched-budget controls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from run_twc_v1_k8_icc_aligned import gpu_admission


def _config():
    from configs import twc_v1_k8_ue_bs_rate_config as cfg
    return cfg


def _runtime():
    import ue_bs_rate_runtime as runtime
    return runtime


def _trainer():
    import train_twc_v1_k8_ue_bs_rate as trainer
    return trainer


def _evaluator():
    import evaluate_twc_v1_k8_ue_bs_rate as evaluator
    return evaluator


def _publisher():
    import publish_twc_v1_k8_ue_bs_rate as publisher
    return publisher


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    cfg = _config()
    return ('source_control_preflight', *(f'{cfg.key(a,m)}_{t}' for a in cfg.ARMS
        for m in cfg.TRAIN_METHODS for t in (100, cfg.UPDATES)),
        'final_validation', 'directory_export')


def files_identity(paths):
    runtime = _runtime()
    result = {}
    for value in paths:
        path = runtime.inside(value)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Owned joint-rate evidence missing: '+str(path))
        result[str(path)] = runtime.sha256(path)
    if not result:
        raise ValueError('Nonempty joint-rate evidence required')
    return result


def verify_evidence(record):
    runtime = _runtime()
    if not record.get('files'):
        raise ValueError('Completed joint-rate stage has no evidence')
    for name, digest in record['files'].items():
        path = runtime.inside(name)
        if path.is_symlink() or not path.is_file() or runtime.sha256(path) != digest:
            raise ValueError('Completed joint-rate evidence changed: '+name)


def stage(summary, path, name, action, *, reuse_completed=True):
    runtime = _runtime()
    old = summary['stages'].get(name)
    if old and old.get('status') == 'completed' and reuse_completed:
        verify_evidence(old)
        return old
    attempts = [] if old is None else old.get('attempts', []) + [
        {k:v for k,v in old.items() if k != 'attempts'}]
    record = dict(status='running', started=now(), attempt_id=uuid.uuid4().hex[:12], attempts=attempts)
    summary['stages'][name] = record
    summary.update(status='running', current_stage=name, updated=now())
    runtime.atomic_json(path, summary)
    try:
        details, files = action(record)
        record.update(status='completed', finished=now(), details=details, files=files_identity(files))
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
    if set(pointer) != {'namespace','summary'} or pointer['namespace'] != cfg.NAMESPACE:
        raise ValueError('Own joint-rate pointer required')
    path = runtime.inside(pointer['summary'])
    if path.name != 'batch_summary.json' or not path.parent.is_relative_to((cfg.OUTPUT_ROOT/'batches').resolve()):
        raise ValueError('Own joint-rate batch summary path required')
    summary = runtime.read_json(path)
    if summary.get('namespace') != cfg.NAMESPACE:
        raise ValueError('Joint-rate namespace changed')
    return summary, path


def verify_preregistration(summary, summary_path):
    runtime = _runtime()
    path = runtime.inside(Path(summary_path).parent/'preregistration.json')
    if path.is_symlink() or not path.is_file() or runtime.sha256(path) != summary.get('preregistration_sha256'):
        raise ValueError('Joint-rate preregistration bytes changed')
    record = runtime.read_json(path)
    for key in ('namespace','core','code','recipe','recipe_fingerprint','optimizer_updates_each',
            'total_optimizer_updates','source_controls','training_trace_pin',
            'test_used','fixed_terminal_selection','original_outputs_preserved'):
        if key not in summary or key not in record or summary[key] != record[key]:
            raise ValueError('Joint-rate preregistered identity changed: '+key)


def _owned_directory(value, section):
    runtime, cfg = _runtime(), _config()
    directory = runtime.inside(value)
    parent = (cfg.OUTPUT_ROOT/section).resolve()
    if directory == parent or not directory.is_relative_to(parent):
        raise ValueError('Owned joint-rate '+section+' attempt required')
    return directory


def verify_preflight(directory):
    runtime, cfg = _runtime(), _config()
    directory = _owned_directory(directory, 'preflight')
    manifest = runtime.read_json(directory/'manifest.json')
    expected = dict(format=cfg.NAMESPACE+'_preflight_v1', status='passed',
        core=runtime.core_identity(), recipe_fingerprint=cfg.study_fingerprint(),
        optimizer_updates=0, test_used=False, matched_sources_verified=True,
        controls_verified=True)
    for key, value in expected.items():
        if manifest.get(key) != value or isinstance(value,(bool,int)) and type(manifest.get(key)) is not type(value):
            raise ValueError('Joint-rate preflight identity changed: '+key)
    if set(manifest.get('files', {})) != {'source_checks.json'}:
        raise ValueError('Exact source/control preflight artifact required')
    if runtime.sha256(directory/'source_checks.json') != manifest['files']['source_checks.json']:
        raise ValueError('Source/control preflight bytes changed')
    record = runtime.read_json(directory/'source_checks.json')
    if set(record.get('sources', {})) != set(cfg.ARMS) or set(record.get('controls', {})) != set(cfg.ARMS):
        raise ValueError('Both source arms and archived controls required')
    for arm in cfg.ARMS:
        runtime.verify_files(record['sources'][arm])
        runtime.verify_files(record['controls'][arm])
        runtime.verify_control_files(arm)
        checks = record.get('state_checks', {}).get(arm, {})
        if not checks or checks.get('source_encoder') != checks.get('control_encoder'):
            raise ValueError('Archived BS-only control changed the source UE encoder')
        if checks.get('source_statistics') != checks.get('control_statistics'):
            raise ValueError('Archived BS-only control changed the source historical prior')
        if checks.get('source_CE') != checks.get('control_CE'):
            raise ValueError('Source and archived control CE differ')
        base=record['sources'][arm].get('base_ema_updates')
        if (type(base) is not int or base<0
                or record['controls'][arm].get('base_ema_updates')!=base):
            raise ValueError('Pinned source/control EMA starting counter changed')
    if record.get('training_data_protocol') != runtime.make_data('cuda:0', role='codec_train').protocol:
        raise ValueError('Exact archived-control training replay protocol changed')
    if record.get('validation_data_protocol') != runtime.make_data('cuda:0', role='final').protocol:
        raise ValueError('Independent validation protocol changed')
    runtime.guard_core(manifest['core'])
    return manifest


def preflight_action(summary, path, record):
    import torch
    runtime, cfg = _runtime(), _config()
    device = runtime.configure_cuda()
    sources, controls, checks = {}, {}, {}
    with torch.no_grad():
        for arm in cfg.ARMS:
            runtime.verify_control_files(arm)
            codec, stats, ce, source = runtime.load_initial(arm, 'joint', device)
            runtime.verify_files(source)
            checks[arm] = dict(source_encoder=runtime.state_digest(codec.encoder),
                source_decoder=runtime.state_digest(codec.decoder),
                source_statistics=runtime.state_digest(stats),source_CE=runtime.state_digest(ce))
            sources[arm] = source
            del codec, stats, ce
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
            codec, stats, ce, control = runtime.load_control(arm, device)
            runtime.verify_files(control)
            checks[arm].update(control_encoder=runtime.state_digest(codec.encoder),
                control_decoder=runtime.state_digest(codec.decoder),
                control_statistics=runtime.state_digest(stats),control_CE=runtime.state_digest(ce))
            controls[arm] = control
            del codec, stats, ce
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
    for arm in cfg.ARMS:
        if any(checks[arm]['source_'+name]!=checks[arm]['control_'+name]
                for name in ('encoder','statistics','CE')):
            raise ValueError('Archived control must preserve its pinned source UE/CE/prior')
        base=sources[arm].get('base_ema_updates')
        if type(base) is not int or base<0 or controls[arm].get('base_ema_updates')!=base:
            raise ValueError('Source/control inherited EMA counter differs')
    if len({checks[arm]['source_CE'] for arm in cfg.ARMS})!=1:
        raise ValueError('All physical arms must use the pinned common CE')
    output = runtime.inside(cfg.OUTPUT_ROOT/'preflight'/('attempt_'+record['attempt_id']))
    output.mkdir(parents=True, exist_ok=False)
    runtime.atomic_json(output/'source_checks.json', dict(sources=sources,controls=controls,
        state_checks=checks,training_data_protocol=runtime.make_data('cuda:0',role='codec_train').protocol,
        validation_data_protocol=runtime.make_data('cuda:0',role='final').protocol),exclusive=True)
    manifest = dict(format=cfg.NAMESPACE+'_preflight_v1',status='passed',core=summary['core'],
        recipe_fingerprint=cfg.study_fingerprint(),optimizer_updates=0,test_used=False,
        matched_sources_verified=True,controls_verified=True,
        files={'source_checks.json':runtime.sha256(output/'source_checks.json')})
    runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
    verify_preflight(output)
    return dict(output_dir=str(output),manifest=manifest),[output/'manifest.json',output/'source_checks.json']


def verify_boundary(directory, arm, method='joint', *, core=None):
    if method != 'joint': raise ValueError('Only joint is a newly trained trajectory')
    return _trainer().verify_boundary(directory,arm,core=core)


def verify_terminal(arm, method):
    # Resume inspection runs before GPU admission. It verifies the exact
    # terminal files/identities without allocating models; the evaluator
    # strictly loads every terminal on actual CUDA before final data access.
    if method != 'joint': raise ValueError('Only joint terminal belongs to this batch')
    return _trainer().verify_terminal_record(arm)


def _source_ema_base(summary, arm):
    runtime=_runtime()
    preflight=summary.get('stages',{}).get('source_control_preflight',{})
    if preflight.get('status')!='completed':
        raise ValueError('Accepted source/control preflight required before training')
    verify_evidence(preflight)
    directory=_owned_directory(preflight['details']['output_dir'],'preflight')
    sources=runtime.read_json(directory/'source_checks.json')
    source=sources['sources'][arm]
    runtime.verify_files(source)
    base=source.get('base_ema_updates')
    if type(base) is not int or base<0:
        raise ValueError('Verified source EMA starting counter required')
    return base


def training_action(summary, path, arm, method, target, record):
    runtime, cfg = _runtime(), _config()
    directory = runtime.inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm,method))
    args = ['--run','--arm',arm,'--stop-after',str(target)]
    if (directory/'checkpoint_latest.pt').exists():
        manifest = runtime.read_json(directory/'manifest.json')
        if type(manifest.get('step')) is not int or manifest['step'] > target:
            raise ValueError('Cannot rewind an unrecorded joint-rate stage')
        args.append('--resume')
    log = child(summary, path, record, 'train_twc_v1_k8_ue_bs_rate.py', args)
    manifest = runtime.read_json(directory/'manifest.json')
    base=_source_ema_base(summary,arm)
    for name,value in dict(arm=arm,method=method,step=target,counters=cfg.counters(target,base),
            core=summary['core'],recipe=cfg.recipe(arm,method),fingerprint=cfg.fingerprint(arm,method)).items():
        if manifest.get(name) != value:
            raise ValueError('Joint-rate training boundary changed: '+name)
    if manifest.get('base_ema_updates')!=base:
        raise ValueError('Training EMA base no longer binds the accepted source')
    if manifest.get('production_evidence') is not True:
        raise ValueError('CPU fixture is not an joint-rate terminal')
    if manifest.get('checkpoint_sha256') != runtime.sha256(directory/'checkpoint_latest.pt'):
        raise ValueError('Joint-rate checkpoint bytes changed')
    files = [log,directory/'initialization.json']
    if target == 100:
        if verify_boundary(directory,arm,method,core=summary['core']) != manifest:
            raise ValueError('Immutable100 receipt differs from checkpoint')
        files.append(directory/'boundary_00100.json')
    else:
        verify_terminal(arm,method)
        if manifest.get('status') != 'completed' or runtime.read_json(directory/'terminal_receipt.json') != manifest:
            raise ValueError('Fixed10000 terminal is incomplete')
        files.extend([directory/'checkpoint_latest.pt',directory/'manifest.json',directory/'terminal_receipt.json'])
    for step in cfg.MONITOR_STEPS:
        if step <= target:
            receipt = directory/'monitor'/f'step_{step:05d}_receipt.json'
            monitor = runtime.read_json(receipt)
            verify_evidence(monitor)
            files.extend([receipt,*map(Path,monitor['files'])])
    return manifest, files


def verify_manifest(directory):
    return _evaluator().verify_manifest(directory)


def evaluation_action(summary, path, record):
    runtime, cfg = _runtime(), _config()
    # This check precedes the child and therefore every final data read.
    for arm in cfg.ARMS:
        runtime.verify_control_files(arm)
        for method in cfg.TRAIN_METHODS:
            verify_terminal(arm,method)
    output = runtime.inside(cfg.OUTPUT_ROOT/'evaluation'/('attempt_'+record['attempt_id']))
    record['output_dir'] = str(output)
    log = child(summary,path,record,'evaluate_twc_v1_k8_ue_bs_rate.py',
        ['--run','--output-dir',str(output)])
    manifest = verify_manifest(output)
    return dict(output_dir=str(output),manifest=manifest), [output/'manifest.json',
        *[output/name for name in _evaluator().EVALUATION_FILES],log]


def export_action(summary_path, evaluation_directory, record):
    runtime, cfg = _runtime(), _config()
    verify_preregistration(runtime.read_json(summary_path),summary_path)
    verify_manifest(evaluation_directory)
    output = runtime.inside(cfg.OUTPUT_ROOT/'reports'/('attempt_'+record['attempt_id']))
    result = _publisher().export_directory(summary_path,evaluation_directory,output)
    if runtime.inside(result['path']) != output or runtime.inside(result['manifest']) != output/'export_manifest.json':
        raise ValueError('Own joint-rate report attempt required')
    _publisher().verify_export_binding(summary_path,evaluation_directory,output)
    return result, [output/'export_manifest.json',*[output/n for n in result['files']]]


def _verify_completed(summary, path):
    runtime, cfg = _runtime(), _config()
    expected = stage_names()
    if set(summary['stages'])-set((*expected,'startup_policy')):
        raise ValueError('Unknown joint-rate batch stage')
    earlier_complete = True
    for name in expected:
        item = summary['stages'].get(name,{})
        if item.get('status') != 'completed':
            earlier_complete = False
            continue
        if not earlier_complete:
            raise ValueError('Completed joint-rate stages violate dependency order')
        verify_evidence(item)
        if name == 'source_control_preflight':
            verify_preflight(item['details']['output_dir'])
        elif name == 'final_validation':
            verify_manifest(item['details']['output_dir'])
        elif name == 'directory_export':
            _publisher().verify_export_binding(path,summary['stages']['final_validation']['details']['output_dir'],
                item['details']['path'])
        else:
            for arm in cfg.ARMS:
                for method in cfg.TRAIN_METHODS:
                    if name == cfg.key(arm,method)+'_100':
                        receipt = verify_boundary(cfg.OUTPUT_ROOT/'training'/cfg.key(arm,method),arm,method,core=summary['core'])
                        if item['details'] != receipt:
                            raise ValueError('Completed100 evidence changed')
                    elif name == cfg.key(arm,method)+'_'+str(cfg.UPDATES):
                        verify_terminal(arm,method)
                        terminal = runtime.read_json(cfg.OUTPUT_ROOT/'training'/cfg.key(arm,method)/'terminal_receipt.json')
                        if item['details'] != terminal:
                            raise ValueError('Completed terminal evidence changed')
    if summary.get('status') == 'completed' and not earlier_complete:
        raise ValueError('Completed joint-rate batch has incomplete stages')


def run(args):
    runtime, cfg = _runtime(), _config()
    runtime.guard_core(runtime.core_identity())
    cfg.OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    with runtime.process_lock(cfg.OUTPUT_ROOT/'controller.lock'):
        if args.resume:
            summary,path = load_summary()
            runtime.guard_core(summary['core'])
            if summary.get('recipe') != cfg.study_recipe() or summary.get('recipe_fingerprint') != cfg.study_fingerprint():
                raise ValueError('Saved joint-rate recipe changed')
            verify_preregistration(summary,path)
            _verify_completed(summary,path)
        else:
            if any(p.name != 'controller.lock' for p in cfg.OUTPUT_ROOT.iterdir()):
                raise FileExistsError('Joint-rate output exists; inspect and use explicit --resume')
            directory = cfg.OUTPUT_ROOT/'batches'/('batch_'+uuid.uuid4().hex)
            directory.mkdir(parents=True,exist_ok=False)
            path = directory/'batch_summary.json'
            summary = dict(namespace=cfg.NAMESPACE,status='starting',created=now(),updated=now(),
                core=runtime.core_identity(),code=runtime.source_identity(),recipe=cfg.study_recipe(),
                recipe_fingerprint=cfg.study_fingerprint(),optimizer_updates_each=cfg.UPDATES,
                total_optimizer_updates=cfg.UPDATES*len(cfg.ARMS)*len(cfg.TRAIN_METHODS),
                source_controls=cfg.SOURCE_CONTROL_PINS,training_trace_pin=cfg.TRAINING_TRACE_PIN,
                test_used=False,fixed_terminal_selection=True,original_outputs_preserved=True,stages={})
            prereg = directory/'preregistration.json'
            runtime.atomic_json(prereg,summary,exclusive=True)
            summary['preregistration_sha256'] = runtime.sha256(prereg)
            runtime.atomic_json(path,summary,exclusive=True)
            runtime.atomic_json(cfg.OUTPUT_ROOT/'last_batch.json',dict(namespace=cfg.NAMESPACE,summary=str(path)),exclusive=True)
        if summary.get('status') == 'completed':
            print(compact_summary(summary))
            return
        try:
            def admission(record):
                result = gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                    min_free_mib=args.min_free_mib,max_gpu_util=args.max_gpu_util)
                file = path.parent/('startup_policy_'+record['attempt_id']+'.json')
                runtime.atomic_json(file,result,exclusive=True)
                return result,[file]
            stage(summary,path,'startup_policy',admission,reuse_completed=False)
            stage(summary,path,'source_control_preflight',lambda r:preflight_action(summary,path,r))
            for arm in cfg.ARMS:
                for method in cfg.TRAIN_METHODS:
                    for target in (100,cfg.UPDATES):
                        stage(summary,path,f'{cfg.key(arm,method)}_{target}',
                            lambda r,a=arm,m=method,t=target:training_action(summary,path,a,m,t,r))
            final = stage(summary,path,'final_validation',lambda r:evaluation_action(summary,path,r))
            stage(summary,path,'directory_export',lambda r:export_action(path,final['details']['output_dir'],r))
            runtime.guard_core(summary['core'])
            summary.update(status='completed',current_stage='directory_export',finished=now(),updated=now())
            summary.pop('error',None)
            summary.pop('traceback',None)
            runtime.atomic_json(path,summary)
            print(compact_summary(summary))
        except BaseException as exc:
            summary.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),updated=now())
            runtime.atomic_json(path,summary)
            raise


def compact_summary(summary):
    runtime,cfg = _runtime(),_config()
    lines = [f'UE+BS joint rate | {summary.get("status","unknown")} | stage={summary.get("current_stage","unknown")}',
        'Namespace: '+cfg.NAMESPACE,'Updated: '+summary.get('updated','unknown'),
        'Core commit: '+summary.get('code',{}).get('commit','unknown'),
        f'Budget: two new joint trajectories; each {cfg.UPDATES} UE+BS updates; archived BS-only controls reused',
        'Protocol: fixed K8/256RE/Ep16+Ed48; RZF alpha=.03; DL25; micro4 x accumulation8; test sealed',
        'UE input unchanged; balanced six-node feedback SNR; CE/CE table frozen; causal prior updates',
        'Comparison includes the prior-update policy; gains cannot be attributed to UE gradients alone']
    for arm in cfg.ARMS:
        file=cfg.OUTPUT_ROOT/'training'/cfg.key(arm,'joint')/'manifest.json'
        if not file.is_file():
            lines.append('joint_'+arm+': not started');continue
        record=runtime.read_json(file)
        counters=record.get('counters',{})
        lines.append(f'joint_{arm}: saved_boundary={record.get("step","unknown")}/{cfg.UPDATES} '
            f'ETA={record.get("ETA_seconds","unknown")}s recent={record.get("updated","unknown")}')
        if counters.get('gamma_counts'):
            lines.append(' gamma_counts: '+json.dumps(counters['gamma_counts'],sort_keys=True))
        lines.append(' EMA new commits='+str(counters.get('EMA_commits','unknown'))+
            ' source base='+str(record.get('base_ema_updates','unknown')))
    if summary.get('status')=='failed':
        record=summary.get('stages',{}).get(summary.get('current_stage'),{})
        lines.append('ERROR: '+str(record.get('error',summary.get('error','unknown')))[:1200])
        if record.get('log_path'):lines.append('Exact log: '+record['log_path'])
        if record.get('child_error'):lines.append(record['child_error'][-1500:])
    final=summary.get('stages',{}).get('final_validation',{})
    if final.get('status')=='completed':
        output=runtime.inside(final['details']['output_dir'])
        with (output/'summary.csv').open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        for method in cfg.METHODS:
            for arm in cfg.ARMS:
                points=sorted((r for r in rows if r.get('method')==method and r.get('arm')==arm
                    and r.get('scope')=='point'),key=lambda r:float(r['gamma_db']))
                if points:lines.append(f'{method}/{arm} rate: '+','.join(
                    f'{float(r["sum_rate_bps_hz_mean"]):.6g}' for r in points))
        with (output/'comparisons.csv').open(newline='',encoding='utf-8') as stream:
            comparisons=list(csv.DictReader(stream))
        for row in comparisons:
            if row.get('metric')=='sum_rate_bps_hz' and row.get('scope') in ('all_six','low_0_5_10'):
                lines.append(f'{row["comparison"]} {row["scope"]}: {float(row["mean"]):+.6g} '
                    f'batchCI=[{float(row["batch_ci95_low"]):+.6g},{float(row["batch_ci95_high"]):+.6g}]')
    report=summary.get('stages',{}).get('directory_export',{}).get('details',{}).get('path')
    lines.append('Report directory: '+str(report or 'not completed'))
    if report:lines.append('Upload command: '+sys.executable+
        ' -B publish_twc_v1_k8_ue_bs_rate.py --publish --export-dir '+str(runtime.inside(report)))
    lines.append('Intervals: fixed models and one source training seed; no unseen-SNR generalization claim.')
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
    if args.resume and not args.run: parser.error('--resume requires --run')
    if args.details and not args.summary: parser.error('--details requires --summary')
    cfg = _config()
    if args.summary:
        if not (cfg.OUTPUT_ROOT/'last_batch.json').exists():
            print('UE+BS joint rate: not_started; no output created')
            return 0
        summary,_ = load_summary()
        print(json.dumps(summary,indent=2,ensure_ascii=False) if args.details else compact_summary(summary))
        return 0
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,output=str(cfg.OUTPUT_ROOT),stages=stage_names(),
            recipe=cfg.study_recipe(),weights_read=False,writes=False),indent=2))
        return 0
    if not args.run:
        parser.print_help()
        return 0
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
