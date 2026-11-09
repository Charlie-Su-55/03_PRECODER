#!/usr/bin/env python3
"""One explicit server queue: new CE, calibration, paired codecs and validation."""
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

from configs import twc_v1_k8_icc_aligned_config as cfg
import train_twc_v1_k8_icc_aligned as runtime

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
POINTER = OUTPUT_ROOT/'last_batch.json'
STAGES = ('cuda_preflight','ce_100','ce_5000','ce_calibration','fdma_100',
    'fdma_20000','hybrid_g4_100','hybrid_g4_20000','final_validation','result_package')


def now():
    return datetime.now(timezone.utc).isoformat()


def launcher_identity():
    return dict(commit=runtime.source_identity()['commit'],
        sha256={name:runtime.sha256(ROOT/name) for name in cfg.LAUNCH_FILES})


def files_identity(files):
    result={}
    for path in files:
        path=runtime.inside(path)
        if not path.is_file() or path.is_symlink():raise ValueError('Owned evidence file missing: '+str(path))
        result[str(path)]=runtime.sha256(path)
    if not result:raise ValueError('Nonempty completed-stage evidence required')
    return result


def verify_evidence(record):
    evidence=record.get('files')
    if not evidence:raise ValueError('Completed stage has no evidence')
    for name,digest in evidence.items():
        if runtime.sha256(runtime.inside(name))!=digest:
            raise ValueError('Completed stage evidence changed: '+name)


def gpu_admission(*,allow_shared_gpu,min_free_mib,max_gpu_util):
    if type(min_free_mib)is not int or min_free_mib<1 or not 0<=max_gpu_util<=100:
        raise ValueError('Valid startup free-memory/utilization policy required')
    command=['nvidia-smi','--id=0','--query-gpu=uuid,name,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits']
    line=subprocess.check_output(command,text=True).strip()
    fields=[v.strip() for v in next(csv.reader([line]))]
    if len(fields)!=4:raise ValueError('Unexpected nvidia-smi GPU0 response')
    gpu_id,name,free,util=fields;free=int(free);util=int(util)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    pids=[row[1].strip() for row in csv.reader(processes.splitlines()) if len(row)==2 and row[0].strip()==gpu_id]
    if free<min_free_mib or util>max_gpu_util or (pids and not allow_shared_gpu):
        raise RuntimeError(f'GPU startup policy rejected:free={free}MiB util={util}% compute_pids={pids}; no processes terminated')
    return dict(gpu=name,uuid=gpu_id,free_mib=free,utilization=util,compute_pids=pids,
        allow_shared_gpu=allow_shared_gpu,min_free_mib=min_free_mib,max_gpu_util=max_gpu_util,
        checked=now(),policy_only=True,continuous_monitor=False,hard_resource_limit=False)


def stage(summary,path,name,action):
    previous=summary['stages'].get(name)
    if previous and previous['status']=='completed':
        verify_evidence(previous);return previous
    attempts=[] if previous is None else previous.get('attempts',[])+[previous]
    record=dict(status='running',started=now(),attempt_id=uuid.uuid4().hex[:12])
    summary['stages'][name]=record;summary.update(status='running',current_stage=name,updated=now())
    runtime.atomic_json(path,summary)
    try:
        details,files=action(record)
        record.update(status='completed',finished=now(),details=details,files=files_identity(files))
        record['attempts']=attempts
        runtime.atomic_json(path,summary);return record
    except BaseException as exc:
        record.update(status='failed',finished=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),attempts=attempts)
        summary.update(status='failed',error=record['error'],updated=now())
        runtime.atomic_json(path,summary);raise


def child(summary,path,record,script,args):
    runtime.guard_core(summary['core'])
    log=path.parent/(summary['current_stage']+'_'+record['attempt_id']+'.log')
    record['log_path']=str(log);runtime.atomic_json(path,summary)
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.run([sys.executable,'-B','-u',script,*args],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,check=False)
    runtime.guard_core(summary['core'])
    if process.returncode:
        # Read only this exact failing child's log, never guess the newest log.
        tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-14:])[-4000:]
        record['child_error']=tail
        raise RuntimeError(f'{script} failed exit={process.returncode}; exact log={log}\n{tail}')
    return log


def training_action(summary,path,component,target,record):
    directory=OUTPUT_ROOT/'training'/component
    latest=directory/'checkpoint_latest.pt'
    args=['--run','--component',component,'--stop-after',str(target)]
    if latest.exists():
        manifest=runtime.read_json(directory/'manifest.json')
        if manifest['step']>target:raise ValueError('Cannot rewind an unrecorded training stage')
        args.append('--resume')
    log=child(summary,path,record,'train_twc_v1_k8_icc_aligned.py',args)
    manifest=runtime.read_json(directory/'manifest.json')
    if manifest['step']!=target or manifest['counters']!=cfg.counters(component,target):
        raise ValueError('Training did not finish the prescribed complete boundary')
    files=[directory/'initialization.json',log]
    if target==100:
        files.append(directory/'boundary_00100.json')
    else:
        files += [latest,directory/'terminal_receipt.json']
    if component!='ce':
        steps=[s for s in cfg.MONITOR_STEPS if s<=target]
        for step in steps:
            receipt=directory/'monitor'/f'step_{step:05d}_receipt.json'
            r=runtime.read_json(receipt)
            for name,digest in r['files'].items():
                if runtime.sha256(name)!=digest:raise ValueError('Monitor artifact hash changed')
            files += [receipt,*map(Path,r['files'])]
    return manifest,files


def load_summary():
    pointer=runtime.read_json(POINTER)
    if set(pointer)!={'summary','namespace'} or pointer['namespace']!=cfg.NAMESPACE:
        raise ValueError('Own ICC-aligned pointer required')
    path=runtime.inside(pointer['summary'])
    summary=runtime.read_json(path)
    if summary.get('namespace')!=cfg.NAMESPACE:raise ValueError('Unknown batch identity')
    return summary,path


def run(args):
    if sys.platform=='win32':raise RuntimeError('Actual batch runs only in the server CUDA environment')
    # Lock file is persistent and independent of every historical run.
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    with runtime.process_lock(OUTPUT_ROOT/'controller.lock'):
        if args.resume:
            summary,path=load_summary();runtime.guard_core(summary['core'])
            for record in summary['stages'].values():
                if record['status']=='completed':verify_evidence(record)
            summary['launcher_history'].append(launcher_identity())
        else:
            if POINTER.exists():raise FileExistsError('Existing ICC batch; use explicit --resume')
            if (OUTPUT_ROOT/'training').exists():raise FileExistsError('Unknown training tree; refuse a new batch')
            directory=OUTPUT_ROOT/'batches'/('batch_'+uuid.uuid4().hex)
            directory.mkdir(parents=True,exist_ok=False);path=directory/'batch_summary.json'
            summary=dict(namespace=cfg.NAMESPACE,status='starting',created=now(),updated=now(),
                core=runtime.core_identity(),training_code=runtime.source_identity(),
                launcher_history=[launcher_identity()],stages={},seed=cfg.SEED,
                recipes={c:cfg.recipe(c) for c in ('ce',)+cfg.ARMS},
                performance_claims_verified=False,test_used=False)
            runtime.atomic_json(directory/'preregistration.json',summary,exclusive=True)
            runtime.atomic_json(path,summary,exclusive=True)
            runtime.atomic_json(POINTER,dict(summary=str(path),namespace=cfg.NAMESPACE),exclusive=True)
        if summary.get('status')=='completed':
            print(compact_summary(summary));return
        try:
            summary['current_stage']='startup_policy'
            summary['startup_policy']=gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
                min_free_mib=args.min_free_mib,max_gpu_util=args.max_gpu_util)
            summary.setdefault('startup_history',[]).append(summary['startup_policy'])
            policy_receipt=path.parent/('startup_policy_'+uuid.uuid4().hex[:12]+'.json')
            runtime.atomic_json(policy_receipt,summary['startup_policy'],exclusive=True)
            prior_policy=summary['stages'].get('startup_policy')
            summary['stages']['startup_policy']=dict(status='completed',details=summary['startup_policy'],
                files=files_identity([policy_receipt]),attempts=[] if prior_policy is None else [prior_policy])
            runtime.atomic_json(path,summary)
            def preflight(record):
                output=OUTPUT_ROOT/'preflight'/('attempt_'+record['attempt_id'])
                log=child(summary,path,record,'preflight_twc_v1_k8_icc_aligned.py',
                    ['--run','--output',str(output)])
                report=runtime.read_json(output/'report.json')
                if report.get('status')!='completed':raise ValueError('New-chain CUDA admission did not pass')
                return report,[output/'report.json',log]
            stage(summary,path,'cuda_preflight',preflight)
            for target in (100,cfg.CE_UPDATES):
                stage(summary,path,f'ce_{target}',lambda record,t=target:training_action(summary,path,'ce',t,record))
            def calibration(record):
                locked=OUTPUT_ROOT/'ce_calibration'/'calibration.json'
                if locked.exists():
                    result=runtime.calibration_record();return result,[locked]
                output=OUTPUT_ROOT/'ce_calibration'/('attempt_'+record['attempt_id'])
                log=child(summary,path,record,'train_twc_v1_k8_icc_aligned.py',
                    ['--run','--calibrate-ce','--calibration-output',str(output)])
                result=runtime.read_json(output/'calibration.json')
                if result.get('record_id')!=cfg.digest({k:v for k,v in result.items() if k!='record_id'}):
                    raise ValueError('CE calibration receipt failed')
                runtime.atomic_json(locked,result,exclusive=True)
                runtime.calibration_record()
                return dict(record_id=result['record_id'],path=str(locked)),[locked,output/'calibration.json',log]
            stage(summary,path,'ce_calibration',calibration)
            for arm in cfg.ARMS:
                for target in (100,cfg.CODEC_UPDATES):
                    stage(summary,path,f'{arm}_{target}',lambda record,a=arm,t=target:training_action(summary,path,a,t,record))
            def final(record):
                output=OUTPUT_ROOT/'evaluation'/('attempt_'+record['attempt_id'])
                log=child(summary,path,record,'evaluate_twc_v1_k8_icc_aligned.py',
                    ['--run','--mode','final','--output-dir',str(output)])
                manifest=runtime.read_json(output/'manifest.json')
                if manifest['status']!='passed':raise ValueError('Final frozen validation incomplete')
                files=[p for p in output.iterdir() if p.is_file()]
                return dict(output_dir=str(output),manifest=manifest),files+[log]
            result=stage(summary,path,'final_validation',final)
            def package(record):
                from twc_v1_k8_icc_aligned_results import make_small_package
                output=Path(result['details']['output_dir'])
                # Small receipts/calibration accompany the curves; no weights,
                # large raw samples, CSI cache or historical outputs are copied.
                snapshots={'CE_calibration_export.json':OUTPUT_ROOT/'ce_calibration/calibration.json'}
                for component in ('ce',)+cfg.ARMS:
                    snapshots['training_receipt_'+component+'.json']=OUTPUT_ROOT/'training'/component/'terminal_receipt.json'
                for name,original in snapshots.items():
                    target=output/name
                    exported=dict(original_path=str(original),original_sha256=runtime.sha256(original),
                        content=runtime.read_json(original))
                    if target.exists():
                        if runtime.read_json(target)!=exported:raise ValueError('Existing package snapshot changed')
                    else:runtime.atomic_json(target,exported,exclusive=True)
                names=[p.name for p in output.iterdir() if p.suffix in ('.csv','.json','.pdf','.png','.md') and p.is_file()]
                package=make_small_package(output,names,name='icc_aligned_small_results_'+record['attempt_id']+'.zip')
                receipt=path.parent/('package_'+record['attempt_id']+'.json')
                runtime.atomic_json(receipt,package,exclusive=True)
                return package,[Path(package['path']),receipt]
            stage(summary,path,'result_package',package)
            runtime.guard_core(summary['core'])
            summary.update(status='completed',finished=now(),updated=now())
            runtime.atomic_json(path,summary);print(compact_summary(summary))
        except BaseException as exc:
            summary.update(status='failed',error=f'{type(exc).__name__}: {exc}',updated=now(),traceback=traceback.format_exc())
            if summary.get('current_stage')=='startup_policy':
                summary['stages']['startup_policy']=dict(status='failed',error=summary['error'],
                    traceback=summary['traceback'],finished=now())
            runtime.atomic_json(path,summary);raise


def read_table(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def compact_summary(summary):
    lines=[f'ICC-aligned K8 | {summary.get("status","unknown")} | stage={summary.get("current_stage","unknown")}',
        'Namespace: '+cfg.NAMESPACE,'Updated: '+summary.get('updated','unknown'),
        'Core training commit: '+summary.get('training_code',{}).get('commit','unknown'),
        'Protocol: K8/Nt32/Nsc256;Ep16+Ed48;256RE;N0=.25*10^(-gamma/10)',
        'Batch: micro4 x accumulation8=32MU; single seed; fixed terminal weights; test sealed']
    for component in ('ce',)+cfg.ARMS:
        manifest_path=OUTPUT_ROOT/'training'/component/'manifest.json'
        if not manifest_path.exists():lines.append(component+': unknown/not started');continue
        try:
            m=runtime.read_json(manifest_path);count=m.get('counters',{})
            lines.append(f'{component}: {m.get("status","unknown")} updates={count.get("optimizer_updates","?")} '
                f'micro={count.get("microbatches","?")} MU={count.get("MU_frames","?")} UE={count.get("UE_inputs","?")} '
                f'EMA={count.get("EMA_commits","?")} ETA={m.get("ETA_seconds","?")}s')
            lines.append(f'  step_seconds={m.get("seconds_per_update","?")} peak_bytes={m.get("memory",{}).get("peak_allocated_bytes","?")} '
                f'recent={m.get("updated","unknown")}')
        except Exception as exc:lines.append(component+': unreadable manifest: '+str(exc))
    for name,record in summary.get('stages',{}).items():
        if record['status']=='failed':
            lines += [f'FAILED STAGE: {name}',str(record.get('error','unknown'))[:900],
                'Exact stage log: '+record.get('log_path','none')]
            trace=record.get('child_error') or record.get('traceback','')
            lines += trace.splitlines()[-4:]
            break
    else:
        if summary.get('status')=='failed':lines.append('ERROR: '+summary.get('error','unknown'))
    final=summary.get('stages',{}).get('final_validation',{})
    if final.get('status')=='completed':
        output=Path(final['details']['output_dir'])
        try:
            table=read_table(output/'summary.csv')
            for arm in cfg.ARMS:
                points=sorted((r for r in table if r['arm']==arm),key=lambda r:float(r['gamma_db']))
                lines.append(arm+' gamma: '+','.join(r['gamma_db'] for r in points))
                lines.append('  rate: '+','.join(f'{float(r["sum_rate_bps_hz_mean"]):.5f}' for r in points))
                lines.append('  DLNMSEdB(meanlinear): '+','.join(f'{float(r["dl_nmse_db_from_linear_mean"]):.4f}' for r in points))
                if arm=='hybrid_g4':
                    lines.append('  soft key/shared: '+','.join(f'{float(r["soft_key_nmse_linear_mean"]):.4g}/'
                        f'{float(r["soft_shared_nmse_linear_mean"]):.4g}' for r in points))
            pairs=read_table(output/'paired_deltas.csv')
            for scope in ('low_0_5_10','all_six','point'):
                selected=[r for r in pairs if r['metric']=='sum_rate_bps_hz' and r['scope']==scope
                    and (scope!='point' or float(r['gamma_db'])==25)]
                for r in selected:lines.append(f'Hybrid-FDMA {scope}: {float(r["mean"]):+.5f} '
                    f'batchCI=[{float(r["batch_ci95_low"]):+.5f},{float(r["batch_ci95_high"]):+.5f}]')
            lines.append('Full results: '+str(output))
        except Exception as exc:lines.append('Result table status: '+str(exc))
    package=summary.get('stages',{}).get('result_package',{})
    lines.append('Small package: '+package.get('details',{}).get('path','unknown/not completed'))
    lines.append('Readout: fixed-checkpoint validation sampling CI; not multiple training seeds or an ICC replication claim.')
    result='\n'.join(lines[:40])
    return result.encode('utf-8')[:8000].decode('utf-8',errors='ignore')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--run',action='store_true');action.add_argument('--dry-run',action='store_true')
    action.add_argument('--summary',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--details',action='store_true')
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--min-free-mib',type=int,default=12000)
    parser.add_argument('--max-gpu-util',type=int,default=20)
    args=parser.parse_args(argv)
    if args.summary:
        if not POINTER.exists():print('ICC-aligned K8: not_started; no output created');return 0
        summary,path=load_summary()
        print(json.dumps(summary,indent=2,ensure_ascii=False) if args.details else compact_summary(summary));return 0
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,root=str(OUTPUT_ROOT),stages=STAGES,
            core_files=cfg.CORE_FILES,arms=cfg.ARMS,CE_updates=5000,codec_updates_each=20000,
            microbatch=4,accumulation=8,monitor_MU=128,final_MU=1024,
            startup_policy=dict(allow_shared_gpu=args.allow_shared_gpu,min_free_mib=args.min_free_mib,
                max_gpu_util=args.max_gpu_util),weights_read=False,writes=False),indent=2));return 0
    if not args.run:parser.print_help();return 0
    run(args);return 0


if __name__=='__main__':raise SystemExit(main())
