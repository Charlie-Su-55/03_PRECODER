"""Explicit queue for frozen-front-end, paired BS rate-objective adaptation."""
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
    import train_twc_v1_k8_icc_rate as runtime
    return runtime


def _config():
    from configs import twc_v1_k8_icc_rate_config as cfg
    return cfg


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_names():
    cfg=_config()
    return ('source_baseline',*(f'{arm}_{target}' for arm in cfg.ARMS for target in (100,cfg.UPDATES)),
            'rate_final','paired_comparison','result_package')


def files_identity(paths):
    runtime=_runtime();result={}
    for path in paths:
        path=runtime.inside(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError('Owned stage evidence missing: '+str(path))
        result[str(path)]=runtime.sha256(path)
    if not result:raise ValueError('Nonempty completed evidence required')
    return result


def verify_evidence(record):
    runtime=_runtime()
    if not record.get('files'):raise ValueError('Completed stage has no evidence')
    for name,digest in record['files'].items():
        if runtime.sha256(runtime.inside(name))!=digest:
            raise ValueError('Completed stage evidence changed: '+name)


def stage(summary,path,name,action):
    runtime=_runtime()
    previous=summary['stages'].get(name)
    if previous and previous['status']=='completed':
        verify_evidence(previous);return previous
    attempts=[] if previous is None else previous.get('attempts',[])+[previous]
    record=dict(status='running',started=now(),attempt_id=uuid.uuid4().hex[:12])
    summary['stages'][name]=record
    summary.update(status='running',current_stage=name,updated=now())
    runtime.atomic_json(path,summary)
    try:
        details,files=action(record)
        record.update(status='completed',finished=now(),details=details,
                      files=files_identity(files),attempts=attempts)
        runtime.atomic_json(path,summary);return record
    except BaseException as exc:
        record.update(status='failed',finished=now(),error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc(),attempts=attempts)
        summary.update(status='failed',error=record['error'],updated=now())
        runtime.atomic_json(path,summary);raise


def child(summary,path,record,script,args):
    runtime,cfg=_runtime(),_config()
    runtime.guard_core(summary['core'])
    log=path.parent/(summary['current_stage']+'_'+record['attempt_id']+'.log')
    record['log_path']=str(log);runtime.atomic_json(path,summary)
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.run([sys.executable,'-B','-u',script,*args],cwd=cfg.ROOT,
                               stdout=stream,stderr=subprocess.STDOUT,check=False)
    runtime.guard_core(summary['core'])
    if process.returncode:
        tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-16:])[-4000:]
        record['child_error']=tail
        raise RuntimeError(f'{script} failed exit={process.returncode}; exact log={log}\n{tail}')
    return log


def load_summary():
    runtime,cfg=_runtime(),_config()
    pointer=runtime.read_json(cfg.OUTPUT_ROOT/'last_batch.json')
    if set(pointer)!={'namespace','summary'} or pointer['namespace']!=cfg.NAMESPACE:
        raise ValueError('Own rate-experiment pointer required')
    path=runtime.inside(pointer['summary']);summary=runtime.read_json(path)
    if summary.get('namespace')!=cfg.NAMESPACE:
        raise ValueError('Rate batch namespace changed')
    return summary,path


def training_action(summary,path,arm,target,record):
    runtime,cfg=_runtime(),_config()
    directory=cfg.OUTPUT_ROOT/'training'/arm
    args=['--run','--arm',arm,'--stop-after',str(target)]
    if (directory/'checkpoint_latest.pt').exists():
        manifest=runtime.read_json(directory/'manifest.json')
        if manifest['step']>target:raise ValueError('Cannot rewind an unrecorded training stage')
        args.append('--resume')
    log=child(summary,path,record,'train_twc_v1_k8_icc_rate.py',args)
    manifest=runtime.read_json(directory/'manifest.json')
    if manifest['step']!=target or manifest['counters']!=cfg.counters(target):
        raise ValueError('Rate training did not reach the prescribed complete boundary')
    files=[log,directory/'initialization.json']
    if target==100:
        files.append(directory/'boundary_00100.json')
    else:
        if manifest.get('status')!='completed':raise ValueError('Fixed terminal state incomplete')
        files.extend([directory/'terminal_receipt.json',directory/'checkpoint_latest.pt'])
    for step in cfg.MONITOR_STEPS:
        if step<=target:
            receipt=directory/'monitor'/f'step_{step:05d}_receipt.json'
            monitor=runtime.read_json(receipt)
            for name,digest in monitor['files'].items():
                if runtime.sha256(runtime.inside(name))!=digest:
                    raise ValueError('Rate monitor evidence changed')
            files.extend([receipt,*map(Path,monitor['files'])])
    return manifest,files


def _evaluation_action(summary,path,phase,record):
    runtime,cfg=_runtime(),_config()
    output=cfg.OUTPUT_ROOT/'evaluation'/phase/('attempt_'+record['attempt_id'])
    log=child(summary,path,record,'evaluate_twc_v1_k8_icc_rate.py',
              ['--run','--phase',phase,'--output-dir',str(output)])
    from evaluate_twc_v1_k8_icc_rate import verify_evaluation
    manifest,_,_=verify_evaluation(output)
    if manifest.get('phase')!=phase:raise ValueError('Evaluation phase changed')
    files=[p for p in output.rglob('*') if p.is_file()]+[log]
    return dict(output_dir=str(output),manifest=manifest),files


def run(args):
    runtime,cfg=_runtime(),_config()
    if sys.platform=='win32':raise RuntimeError('Server queue requires Linux CUDA')
    runtime.guard_core(runtime.core_identity())
    cfg.OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    with runtime.process_lock(cfg.OUTPUT_ROOT/'controller.lock'):
        if args.resume:
            summary,path=load_summary();runtime.guard_core(summary['core'])
            for record in summary['stages'].values():
                if record['status']=='completed':verify_evidence(record)
        else:
            if (cfg.OUTPUT_ROOT/'last_batch.json').exists() or (cfg.OUTPUT_ROOT/'training').exists():
                raise FileExistsError('Own rate experiment exists; explicit --resume required')
            folder=cfg.OUTPUT_ROOT/'batches'/('batch_'+uuid.uuid4().hex)
            folder.mkdir(parents=True,exist_ok=False);path=folder/'batch_summary.json'
            summary=dict(namespace=cfg.NAMESPACE,status='starting',created=now(),updated=now(),
                core=runtime.core_identity(),training_code=runtime.source_identity(),stages={},
                recipes={arm:cfg.recipe(arm) for arm in cfg.ARMS},test_used=False,
                original_result_preserved=True,source_root=str(cfg.SOURCE_ROOT),
                primary='interaction_hybrid_gain_minus_fdma_gain',primary_scope='all_six',
                performance_claims_verified=False)
            runtime.atomic_json(folder/'preregistration.json',summary,exclusive=True)
            runtime.atomic_json(path,summary,exclusive=True)
            runtime.atomic_json(cfg.OUTPUT_ROOT/'last_batch.json',
                                dict(namespace=cfg.NAMESPACE,summary=str(path)),exclusive=True)
        if summary.get('status')=='completed':print(compact_summary(summary));return
        try:
            summary['current_stage']='startup_policy'
            admission=gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                min_free_mib=args.min_free_mib,max_gpu_util=args.max_gpu_util)
            policy_path=path.parent/('startup_policy_'+uuid.uuid4().hex[:12]+'.json')
            runtime.atomic_json(policy_path,admission,exclusive=True)
            summary.setdefault('startup_history',[]).append(admission)
            summary['stages']['startup_policy']=dict(status='completed',details=admission,
                files=files_identity([policy_path]),finished=now())
            runtime.atomic_json(path,summary)
            baseline=stage(summary,path,'source_baseline',
                           lambda record:_evaluation_action(summary,path,'baseline',record))
            for arm in cfg.ARMS:
                for target in (100,cfg.UPDATES):
                    stage(summary,path,f'{arm}_{target}',
                          lambda record,a=arm,t=target:training_action(summary,path,a,t,record))
            final=stage(summary,path,'rate_final',
                        lambda record:_evaluation_action(summary,path,'final',record))
            def compare(record):
                from evaluate_twc_v1_k8_icc_rate import compare_evaluations
                output=cfg.OUTPUT_ROOT/'comparison'/('attempt_'+record['attempt_id'])
                manifest=compare_evaluations(baseline['details']['output_dir'],
                                            final['details']['output_dir'],output)
                return dict(output_dir=str(output),manifest=manifest),list(output.iterdir())
            comparison=stage(summary,path,'paired_comparison',compare)
            def package(record):
                from twc_v1_k8_icc_aligned_results import make_small_package
                output=Path(comparison['details']['output_dir'])
                names=['comparisons.csv','preregistered_comparison.json','manifest.json']
                for phase,item in [('baseline',baseline),('rate_final',final)]:
                    folder=Path(item['details']['output_dir'])
                    for filename in ('manifest.json','summary.csv','paired_deltas.csv','input_fingerprints.json'):
                        target=output/(phase+'_'+filename)
                        # Explicit byte snapshots; never raw CSI or weight files.
                        data=(folder/filename).read_bytes()
                        if target.exists():
                            if target.read_bytes()!=data:raise ValueError('Existing package snapshot changed: '+str(target))
                        else:
                            with target.open('xb') as stream:stream.write(data)
                        names.append(target.name)
                for arm in cfg.ARMS:
                    original=cfg.OUTPUT_ROOT/'training'/arm/'terminal_receipt.json'
                    target=output/('rate_training_receipt_'+arm+'.json')
                    exported=dict(original_path=str(original),original_sha256=runtime.sha256(original),content=runtime.read_json(original))
                    if target.exists():
                        if runtime.read_json(target)!=exported:raise ValueError('Existing training export changed')
                    else:runtime.atomic_json(target,exported,exclusive=True)
                    names.append(target.name)
                package=make_small_package(output,names,name='icc_rate_small_results_'+record['attempt_id']+'.zip')
                receipt=path.parent/('package_'+record['attempt_id']+'.json')
                runtime.atomic_json(receipt,package,exclusive=True)
                return package,[Path(package['path']),receipt]
            stage(summary,path,'result_package',package)
            runtime.guard_core(summary['core'])
            summary.update(status='completed',current_stage='result_package',finished=now(),updated=now())
            runtime.atomic_json(path,summary);print(compact_summary(summary))
        except BaseException as exc:
            summary.update(status='failed',error=f'{type(exc).__name__}: {exc}',
                           traceback=traceback.format_exc(),updated=now())
            if summary.get('current_stage')=='startup_policy':
                summary['stages']['startup_policy']=dict(status='failed',error=summary['error'],
                    traceback=summary['traceback'],finished=now())
            runtime.atomic_json(path,summary);raise


def compact_summary(summary):
    runtime,cfg=_runtime(),_config()
    lines=[f'ICC rate objective | {summary.get("status","unknown")} | stage={summary.get("current_stage","unknown")}',
           'Namespace: '+cfg.NAMESPACE,'Updated: '+summary.get('updated','unknown'),
           'Core commit: '+summary.get('training_code',{}).get('commit','unknown'),
           'Protocol: original K8/256RE/Ep16+Ed48; RZF alpha=.03; DL25; test sealed',
           f'Budget: each arm {cfg.UPDATES} NEW BS updates; UE/CE/prior frozen; no CE/UE updates',
           'Objective: negative complete-MU sum rate; source20k baseline replayed on same fresh final inputs']
    for arm in cfg.ARMS:
        path=cfg.OUTPUT_ROOT/'training'/arm/'manifest.json'
        if not path.is_file():lines.append(arm+': not started');continue
        row=runtime.read_json(path)
        lines.append(f'{arm}: saved_boundary={row.get("step","unknown")}/{cfg.UPDATES} '
                     f'ETA={row.get("ETA_seconds","unknown")}s recent={row.get("updated","unknown")}')
        if row.get('error'):lines.append('  error: '+str(row['error'])[:500])
    if summary.get('status')=='failed':
        stage_record=summary.get('stages',{}).get(summary.get('current_stage'),{})
        lines.append('ERROR: '+str(stage_record.get('error',summary.get('error','unknown')))[:1200])
        if stage_record.get('log_path'):lines.append('Exact log: '+stage_record['log_path'])
        if stage_record.get('child_error'):lines.append(stage_record['child_error'][-1500:])
    for label in ('source_baseline','rate_final'):
        item=summary.get('stages',{}).get(label,{})
        if item.get('status')!='completed':continue
        folder=Path(item['details']['output_dir'])
        with (folder/'summary.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        for arm in cfg.ARMS:
            points=sorted((r for r in rows if r['arm']==arm),key=lambda r:float(r['gamma_db']))
            lines.append(label+' '+arm+' rate: '+','.join(f'{float(r["sum_rate_bps_hz_mean"]):.5f}' for r in points))
    compare=summary.get('stages',{}).get('paired_comparison',{})
    if compare.get('status')=='completed':
        output=Path(compare['details']['output_dir'])
        with (output/'comparisons.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        for label in ('source_hybrid_minus_fdma','rate_hybrid_minus_fdma','fdma_rate_minus_source',
                      'hybrid_rate_minus_source','interaction_hybrid_gain_minus_fdma_gain'):
            for scope in ('low_0_5_10','all_six'):
                row=next(r for r in rows if r['comparison']==label and r['metric']=='sum_rate_bps_hz' and r['scope']==scope)
                lines.append(f'{label} {scope}: {float(row["mean"]):+.5f} '
                             f'batchCI=[{float(row["batch_ci95_low"]):+.5f},{float(row["batch_ci95_high"]):+.5f}]')
        lines.append('Full comparisons: '+str(output))
    package=summary.get('stages',{}).get('result_package',{})
    lines.append('Small package: '+str(package.get('details',{}).get('path','not completed')))
    lines.append('Intervals: fixed-model validation sampling; one training seed; relative gain does not imply Hybrid wins.')
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group();actions.add_argument('--run',action='store_true');actions.add_argument('--dry-run',action='store_true');actions.add_argument('--summary',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--details',action='store_true')
    parser.add_argument('--allow-shared-gpu',action='store_true');parser.add_argument('--min-free-mib',type=int,default=12000);parser.add_argument('--max-gpu-util',type=int,default=20)
    args=parser.parse_args(argv);cfg=_config()
    if args.resume and not args.run:parser.error('--resume requires --run')
    if args.details and not args.summary:parser.error('--details requires --summary')
    if args.summary:
        if not (cfg.OUTPUT_ROOT/'last_batch.json').exists():
            print('ICC rate objective: not_started; no output created');return 0
        summary,_=load_summary()
        print(json.dumps(summary,indent=2,ensure_ascii=False) if args.details else compact_summary(summary));return 0
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,output=str(cfg.OUTPUT_ROOT),source=str(cfg.SOURCE_ROOT),
            stages=stage_names(),recipes={arm:cfg.recipe(arm) for arm in cfg.ARMS},
            fresh_baseline_final_pairing=True,optimizer_updates_each=cfg.UPDATES,
            current_step100_included=True,test_used=False,writes=False,weights_read=False),indent=2));return 0
    if not args.run:parser.print_help();return 0
    run(args);return 0


if __name__=='__main__':
    raise SystemExit(main())
