#!/usr/bin/env python3
"""Read-only training acceptance and isolated physical-split postprocessing."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import uuid
import zipfile

import physical_split_finish_validation as validation
from physical_split_finish_io import write_union_csv
from configs import twc_v1_k8_physical_split_config as cfg

ROOT=cfg.REPO_ROOT
OUTPUT_ROOT=validation.FINISH_ROOT
POINTER=OUTPUT_ROOT/'last_finish.json'
EVALUATOR='evaluate_twc_v1_k8_physical_split_finish.py'


def now():return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def inside(path):
    value=Path(path).resolve()
    if not OUTPUT_ROOT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Finish output root escaped03')
    if value==OUTPUT_ROOT.resolve() or not value.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Only a child of the independent finish root may be written')
    return value


def save_json(path,value,*,exclusive=False):
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    else:
        tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('x',encoding='utf-8') as stream:
                json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,path)
        finally:
            if tmp.exists():tmp.unlink()


def verify_files(evidence):
    if not isinstance(evidence,dict) or not evidence:raise ValueError('Nonempty completed-stage evidence required')
    for path,digest in evidence.items():
        p=Path(path).resolve()
        if not p.is_relative_to(ROOT.resolve()):raise ValueError('Evidence outside03 refused')
        if not p.is_file() or sha256(p)!=digest:raise ValueError('Read-only/completed artifact changed: '+str(p))


def process_start(pid):
    """Linux process start ticks disambiguate PID reuse; no process is stopped."""
    try:
        raw=Path('/proc')/str(pid)/'stat'
        return raw.read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None
    except (OSError,IndexError):raise RuntimeError('Cannot safely inspect finish lease owner')


@contextmanager
def _finish_lease_record(resume):
    """Persistent finish-only lease: never delete an old or a new lock."""
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT_ROOT/'finish_lease.json';host=platform.node();prior=None
    if path.exists():
        if not resume:raise FileExistsError('Existing finish lease requires explicit --resume')
        prior=read_json(path)
        if prior.get('format')!='physical_split_finish_lease_v1' or prior.get('host')!=host:
            raise ValueError('Unrecognized/cross-host finish lease; retained')
        if prior.get('status')=='active' and process_start(prior['pid'])==prior.get('process_start'):
            raise RuntimeError('Prior finish process is still active; no unlock or termination')
        save_json(OUTPUT_ROOT/'lease_history'/('lease_'+uuid.uuid4().hex+'.json'),prior,exclusive=True)
    owner=dict(format='physical_split_finish_lease_v1',host=host,pid=os.getpid(),
        process_start=process_start(os.getpid()),status='active',started=now(),
        previous_lease_sha256=sha256(path) if prior is not None else None)
    save_json(path,owner,exclusive=prior is None)
    try:yield
    finally:
        owner.update(status='inactive',finished=now())
        save_json(path,owner)


@contextmanager
def finish_lease(resume):
    """An OS mutex plus persistent receipt; no lock file is ever unlinked."""
    import fcntl
    if not OUTPUT_ROOT.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Finish root escaped03')
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    mutex_path=OUTPUT_ROOT/'finish_mutex.lock'
    if mutex_path.is_symlink():raise ValueError('Finish mutex symlink refused')
    with mutex_path.open('a+b') as mutex:
        try:fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another finish process holds the finish-only mutex')
        try:
            with _finish_lease_record(resume):yield
        finally:fcntl.flock(mutex,fcntl.LOCK_UN)


def plan():
    return dict(task='physical_split_finish_v1',source_batch=str(validation.BATCH_PATH),
        training_commit=validation.TRAINING_COMMIT,output_root=str(OUTPUT_ROOT),methods=list(cfg.METHODS),
        flow=['read-only training acceptance','CPU union-column aggregation','global alpha lock','frozen GPU validation','small result package'],
        training_updates_added=0,cleanup=False,training_launches=False,weights_loaded=False,models_built=False,
        alpha_addresses=[15360,15375],validation_addresses=[16384,16447],gamma=list(cfg.GAMMAS),
        fixed_alpha=.03,also_tuned=True,test_used=False)


def assert_identities(summary):
    if summary['training_code']!=validation.archived_training_code():raise ValueError('Archived training identity changed')
    if summary['postprocessing_code']!=validation.postprocessing_code_record():raise ValueError('Postprocessing code changed')
    validation.verify_training_source_files(summary['training_code'])


def execute_stage(summary,folder,name,action):
    existing=summary['stages'].get(name,{})
    if existing.get('status')=='completed':
        verify_files(existing.get('evidence_sha256'));return existing['result']
    history=existing.get('attempt_history',[])
    if existing:history=history+[{k:v for k,v in existing.items() if k!='attempt_history'}]
    entry=dict(status='running',started=now(),attempt_history=history,attempt=len(history)+1)
    summary['stages'][name]=entry;summary.update(active_stage=name,updated=now())
    save_json(folder/'finish_summary.json',summary)
    try:
        result=action();paths=result.pop('_evidence',[])
        if not paths:raise ValueError('Stage did not return evidence')
        entry.update(status='completed',finished=now(),result=result,
            evidence_sha256={str(Path(p)):sha256(p) for p in paths})
        verify_files(entry['evidence_sha256'])
        return result
    except BaseException as error:
        entry.update(status='failed',finished=now(),error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        for attr in ('log_path','child_error','child_traceback'):
            if hasattr(error,attr):entry[attr]=getattr(error,attr)
        raise
    finally:
        summary['updated']=now();save_json(folder/'finish_summary.json',summary)


def acceptance_stage(folder):
    report=folder/('training_acceptance_'+uuid.uuid4().hex[:8]+'.json')
    accepted=validation.accept_training(report)
    if accepted.get('additional_training_updates')!=0:raise ValueError('Acceptance cannot add training updates')
    return dict(record_id=accepted['record_id'],training=accepted['training'],
        additional_training_updates=0,_evidence=[report,OUTPUT_ROOT/'training_acceptance.json',
            *map(Path,accepted['frozen_files'])])


def aggregate_training(folder):
    accepted=validation.load_acceptance();output=inside(folder/('aggregation_'+uuid.uuid4().hex[:8]))
    output.mkdir(parents=True,exist_ok=False)
    return write_training_aggregation(output,accepted)


def write_training_aggregation(output,accepted,*,progress_root=None):
    """The CPU-only aggregation path, separately testable with synthetic receipts."""
    output=inside(output)
    if progress_root is None:progress_root=cfg.OUTPUT_ROOT/'training'
    monitors=[];means=[];progress=[];histories={};layout_histories={};shared=None
    for arm in cfg.ARMS:
        record=accepted['training'][arm];histories[arm]=record['training_input_history']
        layout_histories[arm]=record['layout_transmit_history']
        for step,entry in sorted(record['monitor_records'].items(),key=lambda p:int(p[0])):
            if sha256(entry['path'])!=entry['sha256']:raise ValueError('Original monitor changed')
            report=read_json(entry['path'])
            if shared is not None and shared!=report['shared_inputs']:raise ValueError('Cross-arm fixed monitor inputs differ')
            shared=report['shared_inputs']
            monitors.extend(dict(arm=arm,step=int(step),**row) for row in report['summary'])
            means.extend(dict(arm=arm,step=int(step),**row) for row in report['all_six'])
        path=Path(progress_root)/arm/'progress.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            progress.append(dict(arm=arm,step=row['step'],head_updates=row['update_counts']['head_updates'],
                bs_updates=row['update_counts']['bs_updates'],CSI_NMSE_linear=row['csi_nmse_linear'],
                codeword_NMSE_linear=row['head_nmse_linear'],rate=row['rate'],
                condition=json.dumps(row['condition'],sort_keys=True)))
    if len(set(histories.values()))!=1:raise ValueError('Training channel/noise histories differ')
    for family in ('fdma','g4'):
        if layout_histories[family+'_control']!=layout_histories[family+'_split']:
            raise ValueError('Same-layout X/Y/Hest histories differ')
    columns={}
    for name,rows in (('monitor_full_grid.csv',monitors),('monitor_all_six.csv',means),('training_progress.csv',progress)):
        columns[name]=write_union_csv(output/name,rows)
    receipt=dict(passed=True,training_code=accepted['training_code'],postprocessing_code=accepted['postprocessing_code'],
        additional_training_updates=0,source_batch=str(validation.BATCH_PATH),
        all_training_H_noise_histories_match=True,same_layout_X_Y_Hest_histories_match=True,
        monitor_pairing=True,row_counts=dict(monitor_full_grid=len(monitors),monitor_all_six=len(means),training_progress=len(progress)),
        columns=columns,missing_fields='empty; not zero',NMSE='linear',source_files_unchanged=True)
    save_json(output/'aggregation_receipt.json',receipt,exclusive=True)
    verify_files(accepted['frozen_files'])
    return dict(directory=str(output),row_counts=receipt['row_counts'],
        _evidence=[output/name for name in ('monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv','aggregation_receipt.json')])


class FrozenEvaluationError(RuntimeError):
    pass


def frozen_command(stage,output):
    if stage not in ('calibration','evaluation'):raise ValueError('Only frozen calibration/evaluation subprocesses are allowed')
    return [sys.executable,'-B','-u',str(ROOT/EVALUATOR),'--calibrate' if stage=='calibration' else '--evaluate','--output-dir',str(inside(output))]


def gpu_sharing_snapshot():
    raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True)
    utilization,free=[int(v.strip()) for v in raw.strip().split(',')]
    snapshot=dict(time=now(),utilization_percent=utilization,free_MiB=free,continuous_monitoring=False,processes_terminated=False)
    if utilization>20 or free<4096:raise RuntimeError('GPU startup sharing check failed: '+json.dumps(snapshot))
    return snapshot


def evaluate_stage(folder,stage):
    import evaluate_twc_v1_k8_physical_split_finish as ev
    pointer=folder/(stage+'_pointer.json');previous=read_json(pointer) if pointer.exists() else None
    candidate=inside(previous['directory']) if previous else None
    logs=[];snapshot=None
    if stage=='calibration' and ev.SELECTION_PATH.exists():
        selected=ev.load_selection();output=inside(selected['calibration_directory'])
    elif candidate is not None and (candidate/'manifest.json').is_file() and read_json(candidate/'manifest.json').get('status')=='completed':
        output=candidate
        if stage=='calibration':ev.seal_selection(output)
    else:
        if stage=='calibration' and ev.ATTEMPT_PATH.exists():
            raise RuntimeError('Partial alpha-selection attempt retained; no overwrite or automatic reselection')
        if previous:save_json(folder/('prior_'+stage+'_pointer_'+uuid.uuid4().hex[:8]+'.json'),previous,exclusive=True)
        output=inside(folder/(stage+'_'+uuid.uuid4().hex[:8]));save_json(pointer,dict(directory=str(output)))
        snapshot=gpu_sharing_snapshot();command=frozen_command(stage,output)
        log=folder/(stage+'_'+uuid.uuid4().hex[:8]+'.log');logs=[log]
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
        if child.returncode:
            manifest=read_json(output/'manifest.json') if (output/'manifest.json').is_file() else {}
            error=FrozenEvaluationError(stage+' failed: '+str(manifest.get('error','exit '+str(child.returncode))))
            error.log_path=str(log);error.child_error=manifest.get('error');error.child_traceback=manifest.get('traceback')
            raise error
    ev.verify_stage_files(output,stage)
    result=dict(directory=str(output),startup_snapshot=snapshot,
        _evidence=[pointer,output/'manifest.json',*logs,*[output/name for name in ev.ARTIFACTS[stage]]])
    if stage=='calibration':
        selected=ev.load_selection()
        result.update(record_id=selected['record_id'],selected_alphas=selected['selected_alphas'])
        result['_evidence'] += [ev.SELECTION_PATH,ev.ATTEMPT_PATH]
    return result


def export_package(folder,summary):
    import evaluate_twc_v1_k8_physical_split_finish as ev
    validation.load_acceptance()
    audit=Path(summary['aggregation']['directory']);calibration=Path(summary['alpha_selection']['directory'])
    evaluated=Path(summary['validation']['directory']);token=uuid.uuid4().hex[:8]
    package=folder/('physical_split_finish_'+token+'.zip');snapshot=folder/('package_summary_'+token+'.json')
    save_json(snapshot,dict(status='results_completed',training_code=summary['training_code'],
        postprocessing_code=summary['postprocessing_code'],source_batch=str(validation.BATCH_PATH),
        training_updates_added=0,aggregation=summary['aggregation'],alpha_selection=summary['alpha_selection'],
        validation=summary['validation'],package_path=str(package)),exclusive=True)
    files=[snapshot,OUTPUT_ROOT/'training_acceptance.json',ev.SELECTION_PATH,
        audit/'aggregation_receipt.json',calibration/'manifest.json',evaluated/'manifest.json']
    files += [audit/name for name in ('monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv')]
    files += [calibration/name for name in ev.ARTIFACTS['calibration']]
    files += [evaluated/name for name in ev.ARTIFACTS['evaluation'] if not name.endswith('.gz')]
    summary.update(result_package=str(package),return_files=[str(folder/'finish_summary.json'),
        str(OUTPUT_ROOT/'training_acceptance.json'),str(evaluated/'results.json')])
    save_json(folder/'finish_summary.json',summary)
    indexed={str(path.relative_to(OUTPUT_ROOT)):path for path in files};expected={name:sha256(path) for name,path in indexed.items()}
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in indexed.items():archive.write(path,name)
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist())!=set(expected) or any(hashlib.sha256(archive.read(name)).hexdigest()!=digest for name,digest in expected.items()):
            raise ValueError('Finish package verification failed')
    summary['result_package_sha256']=sha256(package)
    return dict(path=str(package),sha256=summary['result_package_sha256'],package_summary=str(snapshot),
        _evidence=[package,snapshot])


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual finish runs on the server; no local model/checkpoint execution')
    if not resume and POINTER.exists():raise FileExistsError('Existing finish requires explicit --resume')
    if resume and not POINTER.is_file():raise FileNotFoundError('No prior finish pointer')
    with finish_lease(resume):
        if resume:
            folder=inside(read_json(POINTER)['directory']);summary=read_json(folder/'finish_summary.json');assert_identities(summary)
            for entry in summary.get('stages',{}).values():
                if entry.get('status')=='completed':verify_files(entry.get('evidence_sha256'))
        else:
            folder=OUTPUT_ROOT/'finishes'/('finish_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(format='physical_split_finish_v1',status='running',created=now(),updated=now(),
                training_code=validation.archived_training_code(),postprocessing_code=validation.postprocessing_code_record(),
                source_batch=str(validation.BATCH_PATH),training_updates_added=0,stages={})
            save_json(POINTER,dict(directory=str(folder)),exclusive=True);save_json(folder/'finish_summary.json',summary)
        if summary.get('status')=='completed':
            validation.load_acceptance()
            import evaluate_twc_v1_k8_physical_split_finish as ev
            ev.load_selection();ev.verify_stage_files(summary['validation']['directory'],'evaluation')
            if sha256(summary['result_package'])!=summary['result_package_sha256']:raise ValueError('Completed result package changed')
            print('Finish evidence verified; no stage repeated and no training updates.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary.update(status='running',updated=now())
        try:
            summary['acceptance']=execute_stage(summary,folder,'training_acceptance',lambda:acceptance_stage(folder))
            summary['aggregation']=execute_stage(summary,folder,'CPU_aggregation',lambda:aggregate_training(folder))
            assert_identities(summary)
            summary['alpha_selection']=execute_stage(summary,folder,'alpha_calibration',lambda:evaluate_stage(folder,'calibration'))
            summary['validation']=execute_stage(summary,folder,'frozen_validation',lambda:evaluate_stage(folder,'evaluation'))
            assert_identities(summary)
            summary['packaging']=execute_stage(summary,folder,'result_packaging',lambda:export_package(folder,summary))
            summary.update(status='completed',finished=now(),updated=now())
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'finish_summary.json',summary);raise
        save_json(folder/'finish_summary.json',summary);return folder


def relevant_failure(summary):
    """Use the failing stage's exception, never a guessed newest training log."""
    stages=summary.get('stages',{});active=summary.get('active_stage')
    if active in stages and stages[active].get('status')=='failed':name=active
    else:
        failed=[(name,entry) for name,entry in stages.items() if entry.get('status')=='failed']
        name=max(failed,key=lambda p:p[1].get('finished',''))[0] if failed else None
    if name is None:return dict(stage=None,error=summary.get('error'),traceback=None)
    entry=stages[name]
    return dict(stage=name,error=entry.get('error'),traceback=entry.get('child_traceback') or entry.get('traceback'),
        log_path=entry.get('log_path'))


def summary_lines(summary,accepted=None,result=None,selection=None,stage_log_tail=None):
    failure=relevant_failure(summary)
    lines=[f"original_training={'verified completed' if accepted else 'pending read-only verification'}; finish={summary.get('status','unknown')}",
        'training_updates_added=0; original data/head/BS=6048/6048/5940 per arm; UE/CE=0',
        'failure_stage='+str(failure['stage'])+' error='+str(failure['error'] or 'none'),
        'training_commit='+str(summary.get('training_code',{}).get('commit',validation.TRAINING_COMMIT)),
        'postprocessing_commit='+str(summary.get('postprocessing_code',{}).get('commit','unknown')),
        'gamma=0,5,10,15,20,25; fixed alpha=.03 + global tuned; validation only; test sealed']
    if accepted:
        for arm in cfg.ARMS:
            r=accepted['training'][arm];c=r['update_counts']
            lines.append(f"{arm} completed data={c['data_iterations']} head={c['head_updates']} BS={c['bs_updates']} (no new updates)")
    if result:
        groups={}
        for row in result.get('summary',[]):groups.setdefault((row['arm'],row['energy_policy']),{}).setdefault(row['rzf_mode'],[]).append(row)
        for (arm,policy),modes in sorted(groups.items()):
            lines.append(arm+'/'+policy+' '+' '.join(mode+'='+','.join(f"{row['sum_rate_bps_hz_mean']:.3f}"
                for row in sorted(rows,key=lambda r:r['gamma_db'])) for mode,rows in sorted(modes.items())))
        for name in ('fdma_split_minus_control_fdma_fixed','g4_split_minus_control_hybrid_1_3_fixed',
            'g4_split_minus_control_hybrid_1_1_fixed','split_g4_minus_fdma_hybrid_1_3_fixed','split_g4_minus_fdma_hybrid_1_1_fixed'):
            rows=[r for r in result.get('paired',[]) if r['contrast']==name and r['metric']=='sum_rate_bps_hz' and r['estimand'] in ('low_0_5_10','all_six')]
            if rows:lines.append(name+' '+';'.join(r['estimand']+f":{r['sample_mean']:+.3f}[{r['batch_ci95_low']:+.3f},{r['batch_ci95_high']:+.3f}]" for r in rows))
        for arm in cfg.ARMS:
            rows=[r for r in result.get('receiver_diagnostics',[]) if r.get('arm')==arm and r.get('estimand')=='all_six'
                and r.get('metric') in ('physical_soft_codeword_nmse_linear_all','physical_head_codeword_nmse_linear_all')]
            if rows:lines.append(arm+' physical_linear_all='+ ';'.join(str(r['energy_policy'])+'/'+r['metric'].replace('_codeword_nmse_linear_all','')+'='+f"{r['sample_mean']:.5g}" for r in rows))
    else:lines.append('six-point curves / paired intervals / physical errors=not yet available')
    if selection:lines.append('tuned_alpha='+','.join(m+':'+str(selection['selected_alphas'][m]) for m in cfg.METHODS))
    suffix=[]
    if failure.get('traceback'):suffix=['failing_stage_traceback:']+str(failure['traceback']).splitlines()[-4:]
    if stage_log_tail:suffix+=['failing_stage_exact_log_tail:']+list(stage_log_tail)[-4:]
    suffix+=['package='+str(summary.get('result_package','unknown')),
        'summary='+str(summary.get('summary_path','unknown'))+'; --summary --details for full records']
    lines=lines[:max(0,40-len(suffix))];footer=[line[:650] for line in suffix];reserved=len(('\n'.join(footer)+'\n').encode())
    body=[];used=0
    for line in lines:
        line=line[:600];size=len((line+'\n').encode())
        if used+size+reserved>8000:break
        body.append(line);used+=size
    return body+footer


def exact_failure_log_tail(summary):
    """A fallback for child import/startup failure before its manifest exists."""
    failure=relevant_failure(summary);name=failure.get('stage')
    if name is None or summary['stages'][name].get('child_traceback'):return None
    filename=failure.get('log_path')
    if not filename:return None
    try:
        path=inside(filename)
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-4096))
            return stream.read().decode('utf-8',errors='replace').splitlines()[-4:]
    except (OSError,ValueError):return ['Recorded failing-stage log is unavailable: '+str(filename)]


def print_summary(details=False):
    if not POINTER.exists():print('finish not_started; training_updates_added=0; server read-only verification pending');return 0
    try:
        folder=inside(read_json(POINTER)['directory']);summary=read_json(folder/'finish_summary.json')
    except (OSError,ValueError,KeyError) as error:
        print('finish status unknown; cannot read finish pointer/summary: '+str(error));return 0
    summary['summary_path']=str(folder/'finish_summary.json')
    def small_read(path):
        try:return read_json(path) if Path(path).is_file() else None
        except (OSError,ValueError):return None
    accepted=small_read(OUTPUT_ROOT/'training_acceptance.json')
    directory=summary.get('validation',{}).get('directory')
    result=small_read(inside(Path(directory)/'results.json')) if directory else None
    selection=small_read(OUTPUT_ROOT/'alpha_selection.json')
    log_tail=exact_failure_log_tail(summary)
    if details:print(json.dumps(dict(summary=summary,acceptance=accepted,results=result,selection=selection,
        failing_stage_log_tail=log_tail),indent=2))
    else:print('\n'.join(summary_lines(summary,accepted,result,selection,log_tail)))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','run','summary'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--details',action='store_true');args=parser.parse_args(argv)
    if args.resume and not args.run:parser.error('--resume requires --run')
    if args.details and not args.summary:parser.error('--details requires --summary')
    if args.dry_run:print(json.dumps(plan(),indent=2))
    elif args.run:run(args.resume)
    elif args.summary:return print_summary(args.details)
    else:parser.print_help()
    return 0


if __name__=='__main__':raise SystemExit(main())
