#!/usr/bin/env python3
"""Run the bounded four-arm K8 continuation and paired validation on CUDA."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid
import zipfile

from configs import twc_v1_k8_schedule_screen_config as cfg
from train_twc_v1_k8_schedule_screen import canonical, sha256, save_json, source_code_record
from run_as_receiver_batch import _run_stage, _child

ROOT=cfg.REPO_ROOT
BATCH_ROOT=cfg.OUTPUT_ROOT/'batches'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()


def plan():
    return dict(experiment=cfg.NAMESPACE,seed=cfg.SEED,
        arms=[dict(arm=a,new_updates=cfg.UPDATES,included_smoke=108,source=cfg.source_identity(a),
                   checkpoint=str(cfg.OUTPUT_ROOT/'training'/a/'checkpoint_latest.pt')) for a in cfg.ARMS],
        total_new_updates=4*cfg.UPDATES,batch_size=4,loss='NMSE only',model_changed=False,
        selection='final3024 only',role='validation',test_used=False,
        validation=dict(samples=256,batches=64,gamma=list(cfg.GAMMAS),Ep=16,ratios=list(cfg.ENERGY_RATIOS),
                        conditions=90,first_batch_index=1024),
        primary='interleaved minus blocked; G4 and otherG retention',
        specialist_interpretation='same additional budget but3xG4 exposure; not isolated proof of negative transfer')


def verify_training(arm):
    folder=cfg.OUTPUT_ROOT/'training'/arm;record=read_json(folder/'manifest.json')
    for key,value in dict(format=cfg.CHECKPOINT_FORMAT,status='completed',arm=arm,seed=cfg.SEED,
                          step=3024,cumulative_updates=13024,config_fingerprint=cfg.fingerprint(arm)).items():
        if record.get(key)!=value:raise ValueError('Training record mismatch: '+arm+' '+key)
    if canonical(record.get('recipe'))!=canonical(cfg.recipe(arm)) or record.get('code')!=source_code_record():
        raise ValueError('Training recipe/code identity changed')
    checkpoint=folder/'checkpoint_latest.pt'
    if sha256(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Training checkpoint changed')
    record['checkpoint_path']=str(checkpoint)
    return dict(report=record,_evidence=[folder/'manifest.json',checkpoint,folder/'progress.jsonl',folder/'initialization.json'])


def run_arm(folder,arm):
    destination=cfg.OUTPUT_ROOT/'training'/arm;logs=[]
    command=[sys.executable,'-B','-u','train_twc_v1_k8_schedule_screen.py','--arm',arm]
    if not (destination/'checkpoint_latest.pt').exists():
        if destination.exists():raise FileExistsError('Partial output without checkpoint; preserve and stop: '+str(destination))
        logs.append(_child(folder,arm+'_first108',command+['--stop-after','108']))
    if read_json(destination/'manifest.json').get('status')!='completed':
        logs.append(_child(folder,arm+'_continue',command+['--resume','--train']))
    result=verify_training(arm);result['_evidence']+=logs;return result


def verify_evaluation(directory):
    from evaluate_twc_v1_k8_schedule_screen import artifact_hashes
    directory=cfg.inside_root(directory);manifest=read_json(directory/'manifest.json')
    if (manifest.get('status')!='completed' or manifest.get('results_verified') is not True
            or manifest.get('freeze_checks',{}).get('passed') is not True
            or manifest.get('summary_rows')!=90 or manifest.get('sample_rows')!=90*256):
        raise ValueError('Incomplete paired validation acceptance')
    if artifact_hashes(directory)!=manifest.get('output_sha256'):raise ValueError('Validation artifact changed')
    if (not manifest.get('input_sha256') or set(manifest.get('methods',{}))!=set(cfg.ARMS)
            or manifest.get('source_sha256')!=source_code_record()['sha256']
            or manifest.get('samples')!=256 or manifest.get('batch_size')!=4
            or manifest.get('role')!='validation' or manifest.get('test_used') is not False
            or len(manifest.get('shared_inputs',[]))!=64):raise ValueError('Missing protocol/input/source evidence')
    for name,digest in manifest['source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Validation code changed')
    for path,digest in manifest.get('input_sha256',{}).items():
        if sha256(path)!=digest:raise ValueError('Validation input changed')
    return manifest


def evaluate(folder):
    pointer=folder/'evaluation_pointer.json'
    if pointer.exists():
        output=cfg.inside_root(read_json(pointer)['directory'])
        if (output/'manifest.json').is_file() and read_json(output/'manifest.json').get('status')=='completed':
            manifest=verify_evaluation(output)
            return dict(directory=str(output),_evidence=[pointer,output/'manifest.json',*[output/n for n in manifest['output_sha256']]])
    output=folder/('evaluation_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
    command=[sys.executable,'-B','-u','evaluate_twc_v1_k8_schedule_screen.py','--evaluate','--output-dir',str(output)]
    for arm in cfg.ARMS:command+=['--checkpoint',arm+'='+str(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')]
    log=_child(folder,'paired_validation',command);manifest=verify_evaluation(output)
    return dict(directory=str(output),_evidence=[pointer,log,output/'manifest.json',*[output/n for n in manifest['output_sha256']]])


def source_preflight(folder):
    report=folder/('source_check_'+uuid.uuid4().hex[:8]+'.json')
    log=_child(folder,'source_checks',[sys.executable,'-B','-u','train_twc_v1_k8_schedule_screen.py',
        '--check-sources','--report',str(report)])
    value=read_json(report)
    if value.get('status')!='passed':raise ValueError('Source/CE/CUDA checks failed')
    frozen=[Path(cfg.source_identity(a)['path']) for a in ('blocked_universal','fdma_control')]
    frozen += [Path(cfg.source_identity(a)['manifest_path']) for a in ('blocked_universal','fdma_control')]
    frozen += [Path(item['path']) for item in value['ce_identity'].values()]
    return dict(report=value,_evidence=[report,log,*frozen])


def export_small(folder,summary):
    evaluation=Path(summary['evaluation']['directory']);manifest=verify_evaluation(evaluation)
    bundle=folder/('small_results_'+uuid.uuid4().hex[:8]+'.zip')
    summary.update(result_package=str(bundle),return_files=[str(folder/'batch_summary.json'),str(bundle)])
    save_json(folder/'batch_summary.json',summary)
    files=[folder/'batch_summary.json',evaluation/'manifest.json']+[evaluation/n for n in manifest['output_sha256'] if not n.endswith('.gz')]
    expected={p.relative_to(folder).as_posix():sha256(p) for p in files}
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(folder).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        import hashlib
        if set(archive.namelist())!=set(expected):raise ValueError('Result package names changed')
        for name,digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Result package bytes changed')
    summary.update(result_package_sha256=sha256(bundle),package_contents='small tables/manifests only; raw samples and weights stay on server')
    save_json(folder/'batch_summary.json',summary)


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual batch is intended for the CUDA server')
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='k8_schedule_controller',arm='controller',SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
            if summary['code']!=source_code_record():raise ValueError('Batch code identity changed')
        else:
            if pointer.exists():raise FileExistsError('Existing batch; explicit --resume required')
            for arm in cfg.ARMS:
                if (cfg.OUTPUT_ROOT/'training'/arm).exists():raise FileExistsError('Existing unowned training output')
            folder=BATCH_ROOT/('batch_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),code=source_code_record(),plan=plan(),stages={},training={})
            save_json(pointer,dict(directory=str(folder)));save_json(folder/'batch_summary.json',summary)
        if summary.get('status')=='completed':
            for arm in cfg.ARMS:verify_training(arm)
            verify_evaluation(summary['evaluation']['directory'])
            print('Already completed; no updates repeated.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary['status']='running'
        try:
            summary['sources']=_run_stage(summary,folder,'sources',lambda:source_preflight(folder))['report']
            for arm in cfg.ARMS:
                summary['training'][arm]=_run_stage(summary,folder,'train_'+arm,lambda a=arm:run_arm(folder,a))['report']
            summary['evaluation']=_run_stage(summary,folder,'evaluation',lambda:evaluate(folder))
            summary.update(status='completed',finished=now());export_small(folder,summary)
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'batch_summary.json',summary);raise
        save_json(folder/'batch_summary.json',summary)
        return folder


def print_summary():
    pointer=cfg.OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; no run/checkpoint inferred');return 0
    folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
    print(json.dumps({k:summary.get(k) for k in ('status','error','created','finished','updated','result_package','result_package_sha256','return_files')},indent=2))
    print('summary_path='+str(folder/'batch_summary.json'))
    print('stages='+json.dumps({k:v['status'] for k,v in summary['stages'].items()}))
    for arm in cfg.ARMS:
        path=cfg.OUTPUT_ROOT/'training'/arm/'manifest.json'
        if path.exists():
            record=read_json(path)
            print(json.dumps(dict(arm=arm,**{k:record.get(k) for k in ('status','step','cumulative_updates','updated_utc',
                'last_loss','peak_cuda_allocated_bytes','parameters','checkpoint_sha256','error')})))
        else:print(arm+': not_started')
    result=summary.get('evaluation',{}).get('directory')
    if not result or not (Path(result)/'results.json').exists():print('paired validation: unknown / not completed');return 0
    results=read_json(Path(result)/'results.json');print('validation256; no test; final local3024/cumulative13024; gamma='+str(list(cfg.GAMMAS)))
    curves={}
    for row in results['summary']:
        curves.setdefault((row['arm'],row['group_count'],row['energy_policy']),[]).append(row)
    for key,rows in curves.items():
        ordered=sorted(rows,key=lambda r:r['gamma_db'])
        print(str(key)+' rate='+','.join(f"{r['sum_rate_bps_hz_mean']:.5f}" for r in ordered)
              +' NMSE='+','.join(f"{r['dl_nmse_mean']:.6f}" for r in ordered))
    for row in results['paired']:
        if row['estimand'] not in ('low_0_5_10','all_six','25'):continue
        print(f"{row['contrast']} {row['metric']} {row['estimand']}: {row['sample_mean']:+.6f} "
              f"sample95[{row['sample_ci95_low']:+.6f},{row['sample_ci95_high']:+.6f}] "
              f"batch95[{row['batch_ci95_low']:+.6f},{row['batch_ci95_high']:+.6f}]")
    print('Single-seed condition-order/concentration screen; fixed-G has3xG4 updates; negative results retained.')
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
