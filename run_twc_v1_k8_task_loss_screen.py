#!/usr/bin/env python3
"""One explicit CUDA batch: source acceptance, shared beta, four arms, validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import zipfile

from configs import twc_v1_k8_task_loss_screen_config as cfg
from train_twc_v1_k8_task_loss_screen import (
    source_code_record,sha256,save_json,canonical,now,load_calibration,validate_calibration_record)
from train_twc_v1_k8_ordered_encoder_screen import gpu_snapshot
from run_as_receiver_batch import _child,_run_stage

ROOT=cfg.REPO_ROOT
BATCH_ROOT=cfg.OUTPUT_ROOT/'batches'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def plan():
    return dict(experiment=cfg.NAMESPACE,source_implementation_commit=cfg.SOURCE_COMMIT,
        internal_source_commit=cfg.INTERNAL_SOURCE_COMMIT,seed=cfg.SEED,
        arms=[dict(arm=arm,source=cfg.source_identity(arm),recipe=cfg.recipe(arm),
                   new_updates=6048,included_smoke=108) for arm in cfg.ARMS],
        total_new_updates=24192,training_addresses=[16049,22096],RZF_alpha=.01,CE_frozen=True,
        calibration=dict(status='pending; dry-run does not read/generate it',sources=2,batches_per_source=12,
            batch_size=4,role='codec_train',addresses=[500000,500011],gamma=list(cfg.GAMMAS),Ep=16,
            hybrid_ratio=[1,3],rule='0.25*pooled_median(NMSE_grad_norm)/pooled_median(rate_grad_norm)',
            shared_beta=True,warmup_updates=756,optimizer_updates=0,immutable_record=str(cfg.CALIBRATION_PATH)),
        validation=dict(methods=list(cfg.METHODS),samples=256,batch_size=4,addresses=[3072,3135],
            gamma=list(cfg.GAMMAS),Ep=16,DL=25,conditions=54),
        sharing_policy=dict(max_utilization_percent=20,min_free_MiB=4096,other_processes_allowed=True,
            continuous_monitoring=False),source_checkpoints_read=False,models_built=False,test_used=False)


def assert_code(summary):
    if summary['code']!=source_code_record():raise ValueError('Batch source code changed during execution')


def calibration_stage(folder):
    report=folder/('calibration_report_'+uuid.uuid4().hex[:8]+'.json')
    save_json(folder/'calibration_report_pointer.json',dict(report=str(report)))
    log=_child(folder,'source_and_calibration',[sys.executable,'-B','-u','train_twc_v1_k8_task_loss_screen.py',
        '--calibrate','--report',str(report)])
    calibration=load_calibration();validate_calibration_record(calibration)
    receipt=read_json(report)
    if receipt.get('status')!='passed' or receipt.get('record_id')!=calibration['record_id']:
        raise ValueError('Complete immutable calibration acceptance required')
    files=[report,log,cfg.CALIBRATION_PATH,cfg.OUTPUT_ROOT/'calibration_attempt.json']
    files+=list(map(Path,calibration['frozen_files']))
    return dict(record_id=calibration['record_id'],beta_target=calibration['beta_target'],
        median_nmse_grad_norm=calibration['median_nmse_grad_norm'],median_rate_grad_norm=calibration['median_rate_grad_norm'],
        _evidence=files)


def inspect_training(arm,calibration):
    folder=cfg.OUTPUT_ROOT/'training'/arm;record=read_json(folder/'manifest.json')
    for key,value in dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,source=cfg.source_identity(arm),
        code=source_code_record(),calibration=cfg.calibration_identity(calibration),
        recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration)).items():
        if canonical(record.get(key))!=canonical(value):raise ValueError('Training receipt '+key+' changed')
    step=record.get('step')
    if type(step) is not int or not 108<=step<=6048 or step%108 or record.get('cumulative_updates')!=16048+step:
        raise ValueError('Saved training update/address mismatch')
    path=folder/'checkpoint_latest.pt'
    if sha256(path)!=record.get('checkpoint_sha256'):raise ValueError('Saved checkpoint hash differs from receipt')
    return record


def verify_training(arm,calibration):
    record=inspect_training(arm,calibration);folder=cfg.OUTPUT_ROOT/'training'/arm
    if record.get('status')!='completed' or record['step']!=6048:raise ValueError('Terminal6048 completion required')
    initial=read_json(folder/'initialization.json');method='source_fdma' if cfg.is_fdma(arm) else 'source_g4'
    expected=calibration['state_freeze'][method]['codec_model_sha256']
    if initial.get('initialized_state_sha256')!=expected or initial.get('migration') is not False:
        raise ValueError('All arms must copy their complete same frozen source, including residuals')
    events=[json.loads(line) for line in (folder/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if ([row.get('step') for row in events]!=list(range(108,6049,108))
            or any(row.get('cumulative_updates')!=16048+row['step'] for row in events)):
        raise ValueError('Training log coverage differs from fixed6048 budget')
    record['checkpoint_path']=str(folder/'checkpoint_latest.pt')
    return dict(report=record,_evidence=[folder/name for name in
        ('checkpoint_latest.pt','manifest.json','initialization.json','progress.jsonl')])


def run_arm(folder,arm,calibration):
    destination=cfg.OUTPUT_ROOT/'training'/arm;logs=[]
    command=[sys.executable,'-B','-u','train_twc_v1_k8_task_loss_screen.py','--arm',arm]
    if not (destination/'checkpoint_latest.pt').exists():
        if destination.exists():raise FileExistsError('Unowned partial directory retained; no new initialization')
        logs.append(_child(folder,arm+'_first108',command+['--stop-after','108']))
    record=inspect_training(arm,calibration)
    if record.get('status')!='completed':
        logs.append(_child(folder,arm+'_continue',command+['--resume','--train']))
    result=verify_training(arm,calibration);result['_evidence']+=logs;return result


def verify_evaluation(directory,calibration):
    from evaluate_twc_v1_k8_task_loss_screen import artifact_hashes
    output=cfg.inside_root(directory);manifest=read_json(output/'manifest.json')
    if (manifest.get('status')!='completed' or manifest.get('screen_version')!=cfg.NAMESPACE
            or manifest.get('results_verified') is not True or manifest.get('freeze_checks',{}).get('passed') is not True
            or manifest.get('summary_rows')!=54 or manifest.get('sample_rows')!=54*256
            or manifest.get('samples')!=256 or manifest.get('batch_size')!=4
            or manifest.get('role')!='validation' or manifest.get('test_used') is not False
            or set(manifest.get('methods',{}))!=set(cfg.METHODS)
            or manifest.get('validation_batch_addresses')!=list(range(3072,3136))
            or len(manifest.get('shared_inputs',[]))!=64
            or manifest.get('calibration')!=cfg.calibration_identity(calibration)):
        raise ValueError('Incomplete six-method terminal validation')
    if artifact_hashes(output)!=manifest.get('output_sha256'):raise ValueError('Validation output changed')
    if manifest.get('source_sha256')!=source_code_record()['sha256'] or not manifest.get('input_sha256'):
        raise ValueError('Missing code/input identity evidence')
    for path,digest in manifest['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Validation input changed: '+path)
    return manifest


def evaluate(folder,calibration):
    pointer=folder/'evaluation_pointer.json'
    if pointer.exists():
        previous=cfg.inside_root(read_json(pointer)['directory'])
        if (previous/'manifest.json').exists() and read_json(previous/'manifest.json').get('status')=='completed':
            value=verify_evaluation(previous,calibration)
            return dict(directory=str(previous),_evidence=[pointer,previous/'manifest.json',*[previous/n for n in value['output_sha256']]])
    output=folder/('evaluation_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
    command=[sys.executable,'-B','-u','evaluate_twc_v1_k8_task_loss_screen.py','--evaluate','--output-dir',str(output)]
    for arm in cfg.ARMS:command+=['--checkpoint',arm+'='+str(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')]
    log=_child(folder,'paired_validation',command);value=verify_evaluation(output,calibration)
    return dict(directory=str(output),_evidence=[pointer,log,output/'manifest.json',*[output/n for n in value['output_sha256']]])


def export_small(folder,summary):
    output=Path(summary['evaluation']['directory'])
    files=[output/name for name in ('summary.csv','paired_deltas.csv','parameters.csv','results.json','manifest.json')]
    files+=[cfg.CALIBRATION_PATH,cfg.OUTPUT_ROOT/'calibration_attempt.json',folder/'batch_summary.json']
    receipt=folder/'calibration_report_pointer.json'
    if receipt.exists():files+=[receipt,Path(read_json(receipt)['report'])]
    old_package=summary.get('result_package')
    if old_package and Path(old_package).is_file() and summary.get('result_package_sha256')==sha256(old_package):return
    if old_package:summary.setdefault('package_history',[]).append(old_package)
    summary['result_package']=str(folder/('task_loss_results_'+uuid.uuid4().hex[:8]+'.zip'))
    summary['return_files']=[str(folder/'batch_summary.json'),str(cfg.CALIBRATION_PATH),str(output/'results.json')]
    save_json(folder/'batch_summary.json',summary)
    names={str(p.relative_to(cfg.OUTPUT_ROOT)):p for p in files};expected={n:sha256(p) for n,p in names.items()}
    package=Path(summary['result_package'])
    if package.exists():raise FileExistsError('Completed result package cannot be overwritten')
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in names.items():archive.write(path,name)
    import hashlib
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist())!=set(expected) or any(hashlib.sha256(archive.read(n)).hexdigest()!=h for n,h in expected.items()):
            raise ValueError('Small result package content verification failed')
    summary['result_package_sha256']=sha256(package);save_json(folder/'batch_summary.json',summary)


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual batch runs only on the CUDA server')
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='k8_task_loss_controller',arm='controller',SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json');assert_code(summary)
        else:
            if pointer.exists() or cfg.CALIBRATION_PATH.exists():raise FileExistsError('Existing batch requires explicit --resume')
            if any((cfg.OUTPUT_ROOT/'training'/arm).exists() for arm in cfg.ARMS):raise FileExistsError('Existing training output retained')
            folder=BATCH_ROOT/('batch_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),code=source_code_record(),plan=plan(),stages={},training={},startup_snapshots=[])
            save_json(pointer,dict(directory=str(folder)));save_json(folder/'batch_summary.json',summary)
        if summary.get('status')=='completed':
            calibration=load_calibration()
            for arm in cfg.ARMS:verify_training(arm,calibration)
            verify_evaluation(summary['evaluation']['directory'],calibration)
            if not summary.get('result_package_sha256') or sha256(summary['result_package'])!=summary['result_package_sha256']:
                raise ValueError('Completed result package missing or changed')
            print('Already complete; verified with no repeated updates.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary['status']='running'
        try:
            summary['startup_snapshots'].append(gpu_snapshot());save_json(folder/'batch_summary.json',summary)
            summary['calibration']=_run_stage(summary,folder,'source_and_calibration',lambda:calibration_stage(folder))
            calibration=load_calibration()
            for arm in cfg.ARMS:
                assert_code(summary);summary['training'][arm]=_run_stage(summary,folder,'train_'+arm,lambda a=arm:run_arm(folder,a,calibration))['report']
            assert_code(summary);summary['evaluation']=_run_stage(summary,folder,'evaluation',lambda:evaluate(folder,calibration))
            summary.update(status='completed',finished=now());export_small(folder,summary)
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'batch_summary.json',summary);raise
        save_json(folder/'batch_summary.json',summary);return folder


def print_summary():
    pointer=cfg.OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; no calibration, checkpoint or result inferred');return 0
    folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
    print(json.dumps({k:summary.get(k) for k in ('status','error','created','updated','finished','result_package','result_package_sha256','return_files')},indent=2))
    print('summary_path='+str(folder/'batch_summary.json'));print('new_experiment_commit='+summary['code']['commit'])
    print('source_implementation_commit='+cfg.SOURCE_COMMIT+'; internal_source_commit='+cfg.INTERNAL_SOURCE_COMMIT)
    print('stages='+json.dumps({k:v['status'] for k,v in summary['stages'].items()}))
    calibration_valid=False
    if cfg.CALIBRATION_PATH.is_file():
        try:
            calibration=read_json(cfg.CALIBRATION_PATH);validate_calibration_record(calibration);calibration_valid=True
            print('calibration='+json.dumps({k:calibration[k] for k in ('record_id','beta_target','median_nmse_grad_norm','median_rate_grad_norm','rule')}))
            for method in ('source_fdma','source_g4'):
                rows=[r for r in calibration['records'] if r['source_key']==method]
                print(method+' calibration='+json.dumps(rows));print(method+' freeze='+json.dumps(calibration['state_freeze'][method]))
        except Exception as error:print('calibration_invalid='+f'{type(error).__name__}: {error}')
    if not calibration_valid:
        print('calibration: incomplete / unknown')
        report_pointer=folder/'calibration_report_pointer.json'
        if report_pointer.exists():
            path=cfg.inside_root(read_json(report_pointer)['report'])
            if path.exists():
                try:print('calibration_failure='+json.dumps(read_json(path)))
                except Exception as error:print('calibration_report_invalid='+str(error))
    for arm in cfg.ARMS:
        path=cfg.OUTPUT_ROOT/'training'/arm/'manifest.json'
        if not path.exists():print(arm+': not_started');continue
        record=read_json(path)
        print(json.dumps(dict(arm=arm,**{k:record.get(k) for k in ('status','step','cumulative_updates','updated_utc',
            'last_loss','last_nmse','last_rate','beta','parameters','checkpoint_sha256','error')})))
    directory=summary.get('evaluation',{}).get('directory')
    if not directory or not (Path(directory)/'results.json').exists():print('paired validation: unknown / incomplete');return 0
    results=read_json(Path(directory)/'results.json');curves={}
    print('actual_parameters='+json.dumps(results['parameters']))
    for row in results['summary']:curves.setdefault((row['arm'],row['energy_policy']),[]).append(row)
    for key,rows in curves.items():
        print(str(key));rows=sorted(rows,key=lambda r:r['gamma_db'])
        for metric in ('dl_nmse','sum_rate_bps_hz','desired_signal_power','interference_power'):
            print(metric+'='+','.join(f'{r[metric+"_mean"]:.6f}' for r in rows))
        for user in range(8):
            metric=f'user_rate_ue{user}_bps_hz_mean'
            print(metric+'='+','.join(f'{r[metric]:.6f}' for r in rows))
        print('energy_max_abs_error='+str(max(r['energy_max_abs_error'] for r in rows))+
              '; rzf_power_max_abs_error='+str(max(r['rzf_power_max_abs_error'] for r in rows)))
    for row in results['paired']:
        if row['metric'] not in ('dl_nmse','sum_rate_bps_hz'):continue
        print(f"{row['contrast']} {row['metric']} {row['estimand']}: {row['sample_mean']:+.6f} "
              f"sample95[{row['sample_ci95_low']:+.6f},{row['sample_ci95_high']:+.6f}] "
              f"batch95[{row['batch_ci95_low']:+.6f},{row['batch_ci95_high']:+.6f}]")
    print('54 conditions;256 complete MU validation samples;sample/batch CIs separated;test sealed.')
    print('All signed results retained;single training seed;no final-journal superiority claim.')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    for name in ('dry-run','run','summary'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args(argv)
    if args.resume and not args.run:parser.error('--resume requires --run')
    if args.dry_run:print(json.dumps(plan(),indent=2))
    elif args.run:run(args.resume)
    elif args.summary:return print_summary()
    else:parser.print_help()
    return 0


if __name__=='__main__':raise SystemExit(main())
