#!/usr/bin/env python3
"""One explicit CUDA batch for four controlled ordered-window continuations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import zipfile

from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
from train_twc_v1_k8_ordered_encoder_screen import canonical,sha256,save_json,source_code_record,gpu_snapshot,now
from run_as_receiver_batch import _child,_run_stage

ROOT=cfg.REPO_ROOT
BATCH_ROOT=cfg.OUTPUT_ROOT/'batches'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def plan():
    from train_twc_v1_k8_ordered_encoder_screen import EQUIVALENCE_TOLERANCES
    return dict(experiment=cfg.NAMESPACE,source_commit=cfg.SOURCE_COMMIT,seed=cfg.SEED,
        arms=[dict(arm=arm,source=cfg.source_identity(arm),architecture=cfg.architecture(arm),
            new_updates=3024,included_smoke=108,terminal_cumulative_updates=16048) for arm in cfg.ARMS],
        total_new_updates=12096,training_addresses=[13025,16048],loss='NMSE only',CE_frozen=True,
        validation=dict(samples=256,batch_size=4,addresses=[2048,2111],conditions=36,
                        gamma=list(cfg.GAMMAS),Ep=16,DL=25,ratios=list(cfg.ENERGY_RATIOS)),
        migration_tolerances=EQUIVALENCE_TOLERANCES,
        sharing_policy=dict(max_utilization_percent=20,min_free_MiB=4096,
                            other_processes_allowed=True,continuous_monitoring=False),
        source_checkpoints_read=False,models_built=False,test_used=False,Stage_C_started=False)


def verify_training(arm):
    folder=cfg.OUTPUT_ROOT/'training'/arm;record=read_json(folder/'manifest.json')
    expected=dict(format=cfg.CHECKPOINT_FORMAT,status='completed',arm=arm,seed=cfg.SEED,
        step=3024,cumulative_updates=16048,config_fingerprint=cfg.fingerprint(arm),source_commit=cfg.SOURCE_COMMIT)
    for name,value in expected.items():
        if record.get(name)!=value:raise ValueError('Training record mismatch: '+arm+' '+name)
    if canonical(record.get('recipe'))!=canonical(cfg.recipe(arm)) or record.get('code')!=source_code_record():
        raise ValueError('New training recipe/code identity changed')
    checkpoint=folder/'checkpoint_latest.pt'
    if sha256(checkpoint)!=record.get('checkpoint_sha256'):raise ValueError('Terminal checkpoint changed')
    record['checkpoint_path']=str(checkpoint)
    return dict(report=record,_evidence=[checkpoint,folder/'manifest.json',folder/'initialization.json',folder/'progress.jsonl'])


def run_arm(folder,arm):
    destination=cfg.OUTPUT_ROOT/'training'/arm;logs=[]
    command=[sys.executable,'-B','-u','train_twc_v1_k8_ordered_encoder_screen.py','--arm',arm]
    if not (destination/'checkpoint_latest.pt').exists():
        if destination.exists():raise FileExistsError('Unowned/partial output without checkpoint; preserve and stop')
        logs.append(_child(folder,arm+'_first108',command+['--stop-after','108']))
    if read_json(destination/'manifest.json').get('status')!='completed':
        logs.append(_child(folder,arm+'_continue',command+['--resume','--train']))
    result=verify_training(arm);result['_evidence']+=logs;return result


def migration_preflight(folder):
    report=folder/('migration_preflight_'+uuid.uuid4().hex[:8]+'.json')
    pointer=folder/'migration_preflight_pointer.json';save_json(pointer,dict(report=str(report)))
    log=_child(folder,'migration_preflight',[sys.executable,'-B','-u','train_twc_v1_k8_ordered_encoder_screen.py',
        '--preflight','--report',str(report)])
    value=read_json(report)
    if value.get('status')!='passed' or value.get('code')!=source_code_record():raise ValueError('Migration/source preflight failed')
    files=[report,log,pointer]
    for path,digest in value.get('frozen_files',{}).items():
        if sha256(path)!=digest:raise ValueError('Source evidence changed')
        files.append(path)
    return dict(report=value,_evidence=files)


def verify_evaluation(directory):
    from evaluate_twc_v1_k8_ordered_encoder_screen import artifact_hashes
    output=cfg.inside_root(directory);manifest=read_json(output/'manifest.json')
    if (manifest.get('status')!='completed' or manifest.get('results_verified') is not True
            or manifest.get('freeze_checks',{}).get('passed') is not True
            or manifest.get('summary_rows')!=36 or manifest.get('sample_rows')!=36*256
            or manifest.get('samples')!=256 or manifest.get('batch_size')!=4
            or manifest.get('role')!='validation' or manifest.get('test_used') is not False
            or manifest.get('source_commit')!=cfg.SOURCE_COMMIT
            or set(manifest.get('methods',{}))!=set(cfg.ARMS)
            or len(manifest.get('shared_inputs',[]))!=64):raise ValueError('Incomplete ordered paired validation')
    if manifest.get('validation_batch_addresses')!=list(range(2048,2112)):raise ValueError('Validation addresses changed')
    if artifact_hashes(output)!=manifest.get('output_sha256'):raise ValueError('Validation output changed')
    if manifest.get('source_sha256')!=source_code_record()['sha256'] or not manifest.get('input_sha256'):
        raise ValueError('Missing code/input identity evidence')
    for name,digest in manifest['source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Evaluation source changed: '+name)
    for path,digest in manifest['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Evaluation input changed')
    return manifest


def evaluate(folder):
    pointer=folder/'evaluation_pointer.json'
    if pointer.exists():
        previous=cfg.inside_root(read_json(pointer)['directory'])
        if (previous/'manifest.json').exists() and read_json(previous/'manifest.json').get('status')=='completed':
            value=verify_evaluation(previous)
            return dict(directory=str(previous),_evidence=[pointer,previous/'manifest.json',*[previous/name for name in value['output_sha256']]])
    output=folder/('evaluation_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
    command=[sys.executable,'-B','-u','evaluate_twc_v1_k8_ordered_encoder_screen.py','--evaluate','--output-dir',str(output)]
    for arm in cfg.ARMS:command+=['--checkpoint',arm+'='+str(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')]
    log=_child(folder,'paired_validation',command);value=verify_evaluation(output)
    return dict(directory=str(output),_evidence=[pointer,log,output/'manifest.json',*[output/name for name in value['output_sha256']]])


def export_small(folder,summary):
    output=Path(summary['evaluation']['directory']);manifest=verify_evaluation(output)
    bundle=folder/('small_results_'+uuid.uuid4().hex[:8]+'.zip')
    summary.update(result_package=str(bundle),return_files=[str(folder/'batch_summary.json'),str(bundle)])
    save_json(folder/'batch_summary.json',summary)
    files=[folder/'batch_summary.json',output/'manifest.json']+[output/name for name in manifest['output_sha256'] if not name.endswith('.gz')]
    for stage in summary['stages'].values():
        for path in stage.get('evidence_sha256',{}):
            path=Path(path)
            if path.parent==folder and path.suffix=='.json':files.append(path)
    files=sorted(set(files));expected={p.relative_to(folder).as_posix():sha256(p) for p in files}
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(folder).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        import hashlib
        if set(archive.namelist())!=set(expected):raise ValueError('Package names changed')
        for name,digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Package bytes changed')
    summary.update(result_package_sha256=sha256(bundle),package_note='small tables and evidence only; weights/raw samples stay on server')
    save_json(folder/'batch_summary.json',summary)


def assert_code(summary):
    if summary['code']!=source_code_record():raise ValueError('New batch code changed during execution')


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual batch runs only on the CUDA server')
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='k8_ordered_controller',arm='controller',SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json');assert_code(summary)
        else:
            if pointer.exists():raise FileExistsError('Existing batch requires explicit --resume')
            for arm in cfg.ARMS:
                if (cfg.OUTPUT_ROOT/'training'/arm).exists():raise FileExistsError('Unowned new training output exists')
            folder=BATCH_ROOT/('batch_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),source_commit=cfg.SOURCE_COMMIT,code=source_code_record(),
                         plan=plan(),stages={},training={},startup_snapshots=[])
            save_json(pointer,dict(directory=str(folder)));save_json(folder/'batch_summary.json',summary)
        if summary.get('status')=='completed':
            for arm in cfg.ARMS:verify_training(arm)
            verify_evaluation(summary['evaluation']['directory']);print('Already complete; no updates repeated.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary['status']='running'
        try:
            snapshot=gpu_snapshot();summary.setdefault('startup_snapshots',[]).append(snapshot)
            save_json(folder/('gpu_startup_'+uuid.uuid4().hex[:8]+'.json'),snapshot)
            save_json(folder/'batch_summary.json',summary)
            preflight=_run_stage(summary,folder,'migration_preflight',lambda:migration_preflight(folder))['report']
            summary['preflight']=dict(status=preflight['status'],parameters=preflight['parameters'],tolerances=preflight['tolerances'])
            for arm in cfg.ARMS:
                assert_code(summary)
                summary['training'][arm]=_run_stage(summary,folder,'train_'+arm,lambda a=arm:run_arm(folder,a))['report']
            assert_code(summary);summary['evaluation']=_run_stage(summary,folder,'evaluation',lambda:evaluate(folder))
            assert_code(summary);summary.update(status='completed',finished=now());export_small(folder,summary)
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'batch_summary.json',summary);raise
        save_json(folder/'batch_summary.json',summary);return folder


def print_summary():
    pointer=cfg.OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; no checkpoint/result inferred');return 0
    folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
    print(json.dumps({k:summary.get(k) for k in ('status','error','source_commit','created','finished','updated','result_package','result_package_sha256','return_files')},indent=2))
    print('new_experiment_commit='+summary['code']['commit']);print('summary_path='+str(folder/'batch_summary.json'))
    print('stages='+json.dumps({k:v['status'] for k,v in summary['stages'].items()}))
    print('migration_preflight='+json.dumps(summary.get('preflight','unknown')))
    migration=summary.get('stages',{}).get('migration_preflight',{}).get('result',{}).get('report',{})
    pointer=folder/'migration_preflight_pointer.json'
    if not migration and pointer.exists():
        report=cfg.inside_root(read_json(pointer)['report'])
        if report.is_file():migration=read_json(report)
    if migration:
        print('migration_status='+json.dumps({k:migration.get(k) for k in ('status','error','source_commit','backend','tolerances')}))
    def maxima(value):
        if isinstance(value,dict):
            if 'max_abs' in value:return [value]
            return [entry for item in value.values() for entry in maxima(item)]
        return []
    for arm,comparisons in migration.get('comparisons',{}).items():
        values=maxima(comparisons)
        print('migration_errors '+json.dumps(dict(arm=arm,max_abs=max((r['max_abs'] for r in values if r['max_abs'] is not None),default=None),
            max_rel=max((r['max_rel'] for r in values if r['max_rel'] is not None),default=None),all_passed=all(r['passed'] for r in values))))
    for arm in cfg.ARMS:
        path=cfg.OUTPUT_ROOT/'training'/arm/'manifest.json'
        if not path.exists():print(arm+': not_started');continue
        record=read_json(path)
        print(json.dumps(dict(arm=arm,**{k:record.get(k) for k in ('status','step','cumulative_updates','updated_utc','last_loss',
              'peak_cuda_allocated_bytes','parameters','checkpoint_sha256','error')})))
    directory=summary.get('evaluation',{}).get('directory')
    if not directory or not (Path(directory)/'results.json').exists():print('paired validation: unknown / not completed');return 0
    results=read_json(Path(directory)/'results.json');print('256 validation samples; final3024/cumulative16048; gamma='+str(list(cfg.GAMMAS)))
    print('actual_parameters='+json.dumps(results.get('parameters',[])))
    curves={}
    for row in results['summary']:curves.setdefault((row['arm'],row['energy_policy']),[]).append(row)
    for key,rows in curves.items():
        rows=sorted(rows,key=lambda r:r['gamma_db']);print(str(key))
        for metric in ('dl_nmse','sum_rate_bps_hz','desired_signal_power','interference_power'):
            print(metric+'='+','.join(f'{r[metric+"_mean"]:.6f}' for r in rows))
    for row in results['paired']:
        if row['estimand'] not in ('low_0_5_10','all_six','25') or row['metric'] not in ('dl_nmse','sum_rate_bps_hz'):continue
        print(f"{row['contrast']} {row['metric']} {row['estimand']}: {row['sample_mean']:+.6f} "
              f"sample95[{row['sample_ci95_low']:+.6f},{row['sample_ci95_high']:+.6f}] "
              f"batch95[{row['batch_ci95_low']:+.6f},{row['batch_ci95_high']:+.6f}]")
    print('One-seed validation screen; unequal parameters; gap_change computed from raw paired samples; negative results retained.')
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
