#!/usr/bin/env python3
"""One matched-RZF batch: new beta, four continuations, alpha lock, validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import zipfile

from configs import twc_v1_k8_rzf_matched_training_config as cfg
import train_twc_v1_k8_rzf_matched_training as train
from train_twc_v1_k8_ordered_encoder_screen import gpu_snapshot
from run_as_receiver_batch import _child,_run_stage

ROOT=cfg.REPO_ROOT
BATCH_ROOT=cfg.OUTPUT_ROOT/'batches'
read_json,save_json,sha256=train.read_json,train.save_json,train.sha256
source_code_record,now=train.source_code_record,train.now


def plan():
    return dict(experiment=cfg.NAMESPACE,source_implementation_commit=cfg.SOURCE_COMMIT,
        diagnostic_commit=cfg.DIAGNOSTIC_COMMIT,training_alphas=cfg.TRAINING_ALPHAS,
        arms=[dict(arm=arm,source=cfg.source_identity(arm),recipe=cfg.recipe(arm),new_updates=12096,included_smoke=108) for arm in cfg.ARMS],
        total_new_updates=48384,training_addresses=[22097,34192],new_beta=dict(role='codec_train',addresses=[500100,500111],
            source_models=2,batches_per_source=12,samples_per_batch=4,warmup_in_budget=756,old_beta_inherited=False),
        monitor=dict(addresses=[8960,8975],samples=64,steps=list(cfg.PROGRESS_VALIDATION_STEPS),
            full_gamma=list(cfg.GAMMAS),G4_both_ratios=True,alpha='fixed training layout alpha',checkpoint_selection=False),
        final_alpha=dict(addresses=[6144,6159],samples=64,models=6,grid=list(cfg.ALPHAS),one_per_checkpoint=True),
        final_validation=dict(addresses=[7168,7231],samples=256,conditions=108,both_fixed_and_tuned=True),
        modes_and_RNG_restored_after_monitor=True,source_weights_loaded=False,models_built=False,test_used=False,
        shared_GPU_policy=dict(max_utilization=20,min_free_MiB=4096,continuous_monitoring=False))


def assert_code(summary):
    if summary['code']!=source_code_record():raise ValueError('Matched batch source code changed during execution')


def beta_stage(folder):
    report=folder/('beta_report_'+uuid.uuid4().hex[:8]+'.json')
    save_json(folder/'beta_report_pointer.json',dict(report=str(report)))
    log=_child(folder,'sources_and_new_beta',[sys.executable,'-B','-u','train_twc_v1_k8_rzf_matched_training.py',
        '--calibrate','--report',str(report)])
    calibration=train.load_calibration()
    if read_json(report).get('status')!='passed' or read_json(report).get('record_id')!=calibration['record_id']:
        raise ValueError('Complete new beta acceptance report required')
    return dict(record_id=calibration['record_id'],beta_target=calibration['beta_target'],
        original_training_alpha_identity=calibration['original_training_alpha_identity'],
        _evidence=[report,log,cfg.CALIBRATION_PATH,cfg.OUTPUT_ROOT/'beta_attempt.json',
                   *map(Path,calibration['frozen_files'])])


def inspect_training(arm,calibration):
    folder=cfg.OUTPUT_ROOT/'training'/arm;record=read_json(folder/'manifest.json')
    for key,value in dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,source=cfg.source_identity(arm),
        code=source_code_record(),calibration=cfg.calibration_identity(calibration),training_alpha=cfg.training_alpha(arm),
        recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration)).items():
        if train.canonical(record.get(key))!=train.canonical(value):raise ValueError('Matched receipt '+key+' changed')
    step=record.get('step')
    if type(step) is not int or not 108<=step<=12096 or step%108 or record.get('cumulative_updates')!=22096+step:
        raise ValueError('Matched saved update/address mismatch')
    if record.get('actual_condition_counts')!=train.expected_counts(arm,step):raise ValueError('Actual condition counts changed')
    if sha256(folder/'checkpoint_latest.pt')!=record.get('checkpoint_sha256'):raise ValueError('Latest checkpoint receipt changed')
    return record


def verify_training(arm,calibration):
    record=inspect_training(arm,calibration);folder=cfg.OUTPUT_ROOT/'training'/arm
    if record.get('status')!='completed' or record['step']!=12096:raise ValueError('Fixed terminal12096 completion required')
    initial=read_json(folder/'initialization.json');family='source_fdma' if cfg.is_fdma(arm) else 'source_g4'
    if (initial.get('initialized_state_sha256')!=calibration['state_freeze'][family]['codec_model_sha256']
            or initial.get('migration') is not False or initial.get('new_optimizer') is not True):
        raise ValueError('Both objectives must strictly copy the same complete NMSE source')
    events=[json.loads(line) for line in (folder/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row.get('step') for row in events]!=list(range(108,12097,108)):raise ValueError('Training log budget coverage differs')
    for row in events:
        if (row.get('cumulative_updates')!=22096+row['step'] or train.canonical(row.get('condition'))!=train.canonical(cfg.schedule(arm,row['step']))
                or row.get('standard_noise_seed')!=cfg.training_noise_seed(row['step'])):raise ValueError('Training address/condition/noise log changed')
    monitors=record.get('monitor_records',{})
    if set(monitors)!={str(s) for s in cfg.PROGRESS_VALIDATION_STEPS}:raise ValueError('Four full-grid monitor records required')
    reference=None;files=[]
    for step,entry in sorted(monitors.items(),key=lambda item:int(item[0])):
        path=cfg.inside_root(entry['path'])
        if sha256(path)!=entry['sha256']:raise ValueError('Monitor bytes changed')
        report=read_json(path)
        if (report.get('step')!=int(step) or report.get('cumulative_updates')!=22096+int(step)
                or report.get('arm')!=arm or report.get('alpha')!=cfg.training_alpha(arm)
                or len(report.get('summary',[]))!=(6 if cfg.is_fdma(arm) else 12)
                or len(report.get('all_six',[]))!=(1 if cfg.is_fdma(arm) else 2)
                or len(report.get('shared_inputs',[]))!=16
                or [r.get('address') for r in report['shared_inputs']]!=list(range(8960,8976))):
            raise ValueError('Monitor full-grid identity incomplete')
        if reference is not None and report['shared_inputs']!=reference:raise ValueError('Monitor changed its fixed input set/noise')
        reference=report['shared_inputs'];files.append(path)
    record['checkpoint_path']=str(folder/'checkpoint_latest.pt')
    return dict(report=record,_evidence=[folder/name for name in
        ('checkpoint_latest.pt','manifest.json','initialization.json','progress.jsonl')]+files)


def run_arm(folder,arm,calibration):
    destination=cfg.OUTPUT_ROOT/'training'/arm;logs=[]
    command=[sys.executable,'-B','-u','train_twc_v1_k8_rzf_matched_training.py','--arm',arm]
    if not (destination/'checkpoint_latest.pt').exists():
        if destination.exists():raise FileExistsError('Partial unowned training directory retained')
        logs.append(_child(folder,arm+'_first108',command+['--stop-after','108']))
    record=inspect_training(arm,calibration)
    if record.get('status')!='completed':logs.append(_child(folder,arm+'_continue',command+['--resume','--train']))
    result=verify_training(arm,calibration);result['_evidence']+=logs;return result


def training_audit(folder,calibration):
    histories={};monitor_inputs={};progress={};all_monitor=[];all_six=[];all_progress=[]
    for arm in cfg.ARMS:
        record=verify_training(arm,calibration)['report'];histories[arm]=record.get('training_input_history')
        progress[arm]=[json.loads(line) for line in (cfg.OUTPUT_ROOT/'training'/arm/'progress.jsonl').read_text().splitlines() if line.strip()]
        all_progress.extend(dict(arm=arm,**{k:row.get(k) for k in ('step','cumulative_updates','nmse','rate','beta','grad_norm','standard_noise_seed')},
            condition=json.dumps(row['condition'],sort_keys=True)) for row in progress[arm])
        for step,entry in record['monitor_records'].items():
            report=read_json(entry['path']);monitor_inputs.setdefault(arm,report['shared_inputs'])
            all_monitor.extend(dict(arm=arm,step=int(step),**row) for row in report['summary'])
            all_six.extend(dict(arm=arm,step=int(step),**row) for row in report['all_six'])
    if None in histories.values() or len(set(histories.values()))!=1:raise ValueError('Training actual-channel/noise-address histories differ')
    reference=monitor_inputs[cfg.ARMS[0]]
    if any(value!=reference for value in monitor_inputs.values()):raise ValueError('Monitor inputs differ across four models')
    for index in range(112):
        values={(rows[index]['sample_fingerprint'],rows[index]['standard_noise_seed']) for rows in progress.values()}
        if len(values)!=1:raise ValueError('Saved-step actual channel/noise pairing mismatch')
    receipt=dict(passed=True,training_input_histories=histories,all_training_channel_fingerprints_and_noise_addresses_match=True,
        saved_step_pairing_points=112,all_monitor_inputs_match=True,complete_noise_tensor_hashes_at_all_updates=False,
        note='Every-step rolling digest binds actual channel fingerprint and standard-noise address, not an approximate post-detection SNR.')
    save_json(folder/'training_pairing_audit.json',receipt)
    from evaluate_twc_v1_k8_rzf_probe import _write_csv
    _write_csv(folder/'monitor_full_grid.csv',all_monitor);_write_csv(folder/'monitor_all_six.csv',all_six)
    _write_csv(folder/'training_progress.csv',all_progress)
    return dict(report=receipt,_evidence=[folder/name for name in ('training_pairing_audit.json','monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv')])


def alpha_stage(folder):
    import evaluate_twc_v1_k8_rzf_matched_training as ev
    pointer=folder/'alpha_calibration_pointer.json';logs=[]
    if cfg.SELECTION_PATH.exists():
        selected=ev.load_selection();output=cfg.inside_root(selected['calibration_directory'])
    elif pointer.exists() and (Path(read_json(pointer)['directory'])/'manifest.json').exists() and read_json(Path(read_json(pointer)['directory'])/'manifest.json').get('status')=='completed':
        output=cfg.inside_root(read_json(pointer)['directory']);selected=ev.seal_selection(output)
    else:
        if (cfg.OUTPUT_ROOT/'selection_attempt.json').exists():raise RuntimeError('Partial final-alpha calibration retained; no automatic reselection')
        if pointer.exists():save_json(folder/('prior_alpha_pointer_'+uuid.uuid4().hex[:8]+'.json'),read_json(pointer))
        output=folder/('alpha_calibration_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
        logs=[_child(folder,'terminal_alpha_calibration',[sys.executable,'-B','-u','evaluate_twc_v1_k8_rzf_matched_training.py',
            '--calibrate','--output-dir',str(output)])];selected=ev.load_selection()
    ev.verify_stage_files(output,'calibration')
    return dict(directory=str(output),record_id=selected['record_id'],selected_alphas=selected['selected_alphas'],
        boundary_optima=selected['boundary_optima'],_evidence=[pointer,cfg.SELECTION_PATH,output/'manifest.json',
            cfg.OUTPUT_ROOT/'selection_attempt.json',*logs,*[output/name for name in ev.ARTIFACTS['calibration']]])


def validation_stage(folder):
    import evaluate_twc_v1_k8_rzf_matched_training as ev
    pointer=folder/'validation_pointer.json';logs=[]
    if pointer.exists() and (Path(read_json(pointer)['directory'])/'manifest.json').exists() and read_json(Path(read_json(pointer)['directory'])/'manifest.json').get('status')=='completed':
        output=cfg.inside_root(read_json(pointer)['directory'])
    else:
        if pointer.exists():save_json(folder/('prior_validation_pointer_'+uuid.uuid4().hex[:8]+'.json'),read_json(pointer))
        output=folder/('validation_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
        logs=[_child(folder,'terminal_fixed_and_tuned_validation',[sys.executable,'-B','-u','evaluate_twc_v1_k8_rzf_matched_training.py',
            '--evaluate','--output-dir',str(output)])]
    ev.verify_stage_files(output,'evaluation')
    return dict(directory=str(output),_evidence=[pointer,output/'manifest.json',*logs,*[output/n for n in ev.ARTIFACTS['evaluation']]])


def export_small(folder,summary):
    import evaluate_twc_v1_k8_rzf_matched_training as ev
    old=summary.get('result_package')
    if old and Path(old).is_file() and sha256(old)==summary.get('result_package_sha256'):return
    output=Path(summary['validation']['directory']);alpha=Path(summary['alpha_selection']['directory'])
    files=[cfg.CALIBRATION_PATH,cfg.SELECTION_PATH,folder/'batch_summary.json',output/'manifest.json',alpha/'manifest.json']
    files+=[alpha/n for n in ev.ARTIFACTS['calibration']]
    files+=[output/n for n in ev.ARTIFACTS['evaluation'] if n!='samples.csv.gz']
    files+=[folder/n for n in ('training_pairing_audit.json','monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv')]
    for arm in cfg.ARMS:
        record=read_json(cfg.OUTPUT_ROOT/'training'/arm/'manifest.json')
        files.extend(Path(entry['path']) for entry in record['monitor_records'].values())
    package=folder/('matched_rzf_results_'+uuid.uuid4().hex[:8]+'.zip')
    if old:summary.setdefault('package_history',[]).append(old)
    summary.update(result_package=str(package),return_files=[str(folder/'batch_summary.json'),str(cfg.CALIBRATION_PATH),
        str(cfg.SELECTION_PATH),str(output/'results.json')])
    save_json(folder/'batch_summary.json',summary)
    names={str(path.relative_to(cfg.OUTPUT_ROOT)):path for path in files};expected={n:sha256(p) for n,p in names.items()}
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in names.items():archive.write(path,name)
    import hashlib
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist())!=set(expected) or any(hashlib.sha256(archive.read(n)).hexdigest()!=d for n,d in expected.items()):
            raise ValueError('Matched small-package content changed')
    summary['result_package_sha256']=sha256(package);save_json(folder/'batch_summary.json',summary)


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual batch runs only on the CUDA server')
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='k8_matched_rzf_controller',arm='controller',SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json');assert_code(summary)
        else:
            if pointer.exists() or cfg.CALIBRATION_PATH.exists():raise FileExistsError('Existing matched batch requires explicit --resume')
            if any((cfg.OUTPUT_ROOT/'training'/a).exists() for a in cfg.ARMS):raise FileExistsError('Unowned training output retained')
            folder=BATCH_ROOT/('batch_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),code=source_code_record(),plan=plan(),stages={},training={},startup_snapshots=[])
            save_json(pointer,dict(directory=str(folder)));save_json(folder/'batch_summary.json',summary)
        if summary.get('status')=='completed':
            calibration=train.load_calibration()
            for arm in cfg.ARMS:verify_training(arm,calibration)
            import evaluate_twc_v1_k8_rzf_matched_training as ev
            ev.load_selection();ev.verify_stage_files(summary['validation']['directory'],'evaluation')
            if sha256(summary['result_package'])!=summary['result_package_sha256']:raise ValueError('Completed package changed')
            print('Completed stages verified; no training or calibration repeated.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary['status']='running'
        try:
            summary['startup_snapshots'].append(gpu_snapshot());save_json(folder/'batch_summary.json',summary)
            summary['beta_calibration']=_run_stage(summary,folder,'sources_and_new_beta',lambda:beta_stage(folder))
            calibration=train.load_calibration()
            for arm in cfg.ARMS:
                assert_code(summary);summary['training'][arm]=_run_stage(summary,folder,'train_'+arm,lambda a=arm:run_arm(folder,a,calibration))['report']
            summary['training_audit']=_run_stage(summary,folder,'training_pairing_audit',lambda:training_audit(folder,calibration))['report']
            assert_code(summary);summary['alpha_selection']=_run_stage(summary,folder,'terminal_alpha_selection',lambda:alpha_stage(folder))
            assert_code(summary);summary['validation']=_run_stage(summary,folder,'terminal_validation',lambda:validation_stage(folder))
            assert_code(summary);summary.update(status='completed',finished=now());export_small(folder,summary)
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'batch_summary.json',summary);raise
        save_json(folder/'batch_summary.json',summary);return folder


def print_summary():
    pointer=cfg.OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; no weight/CUDA/gradient acceptance inferred');return 0
    folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
    print(json.dumps({k:summary.get(k) for k in ('status','error','created','updated','finished','result_package','result_package_sha256','return_files')},indent=2))
    print('summary_path='+str(folder/'batch_summary.json'));print('new_experiment_commit='+summary['code']['commit'])
    print('source_training_commit='+cfg.SOURCE_COMMIT+'; original_RZF_diagnostic_commit='+cfg.DIAGNOSTIC_COMMIT)
    print('training_alpha='+json.dumps(cfg.TRAINING_ALPHAS));print('stages='+json.dumps({k:v['status'] for k,v in summary['stages'].items()}))
    if cfg.CALIBRATION_PATH.exists():
        try:
            beta=read_json(cfg.CALIBRATION_PATH);train.validate_calibration_record(beta)
            print('new_beta='+json.dumps({k:beta[k] for k in ('record_id','beta_target','median_nmse_grad_norm','median_rate_grad_norm')}))
        except Exception as error:print('new_beta_invalid='+str(error))
    else:print('new_beta=unknown')
    for arm in cfg.ARMS:
        path=cfg.OUTPUT_ROOT/'training'/arm/'manifest.json'
        if not path.exists():print(arm+': not_started');continue
        record=read_json(path)
        print(json.dumps(dict(arm=arm,**{k:record.get(k) for k in ('status','step','cumulative_updates','updated_utc',
            'training_alpha','beta','last_nmse','last_rate','last_grad_norm','checkpoint_sha256','error')})))
        for step,entry in sorted(record.get('monitor_records',{}).items(),key=lambda p:int(p[0])):
            if not Path(entry['path']).is_file():print('monitor_missing='+entry['path']);continue
            report=read_json(entry['path'])
            print(arm+' monitor '+step+' training='+json.dumps({k:report.get(k) for k in ('training_nmse','training_rate','training_grad_norm','beta')}))
            print(arm+' monitor '+step+' all_six='+json.dumps(report['all_six']))
            print(arm+' monitor '+step+' six_point_curves='+json.dumps(report['summary']))
    if cfg.SELECTION_PATH.exists():
        import evaluate_twc_v1_k8_rzf_matched_training as ev
        try:
            selected=read_json(cfg.SELECTION_PATH);ev.validate_selection_record(selected)
            print('final_evaluation_alpha='+json.dumps(selected['selected_alphas']))
            print('predeclared_grid_boundary_optima='+json.dumps(selected['boundary_optima']))
        except Exception as error:print('final_alpha_invalid='+str(error))
    else:print('final_evaluation_alpha=unknown')
    directory=summary.get('validation',{}).get('directory')
    if not directory or not (Path(directory)/'results.json').exists():
        for name in ('beta_report_pointer.json','alpha_calibration_pointer.json','validation_pointer.json'):
            if not (folder/name).exists():continue
            item=read_json(folder/name);path=Path(item['report']) if 'report' in item else Path(item['directory'])/'manifest.json'
            if path.is_file():
                value=read_json(path)
                if value.get('status') not in ('passed','completed'):print(name+' failure='+json.dumps({k:value.get(k) for k in ('status','error','traceback','freeze_checks')}))
        print('final validation unknown/incomplete');return 0
    result=read_json(Path(directory)/'results.json');curves={}
    for row in result['summary']:curves.setdefault((row['arm'],row['rzf_mode'],row['energy_policy']),[]).append(row)
    for key,rows in curves.items():
        rows=sorted(rows,key=lambda r:r['gamma_db']);print(str(key)+' alpha='+str(rows[0]['alpha']))
        for metric in ('dl_nmse','sum_rate_bps_hz','desired_signal_power','interference_power'):
            print(metric+'='+','.join(f'{r[metric+"_mean"]:.7f}' for r in rows))
    for row in result['paired']:
        if row['metric'] not in ('dl_nmse','sum_rate_bps_hz'):continue
        print(f"{row['contrast']} {row['metric']} gamma={row['estimand']}: {row['sample_mean']:+.7f} "
              f"sample95[{row['sample_ci95_low']:+.7f},{row['sample_ci95_high']:+.7f}] "
              f"batch95[{row['batch_ci95_low']:+.7f},{row['batch_ci95_high']:+.7f}]")
    print('Single-seed validation screen;fixed/tuned both retained;no best selection;no assumed Hybrid crossover;test sealed.')
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
