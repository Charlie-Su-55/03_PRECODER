#!/usr/bin/env python3
"""One explicit frozen RZF diagnostic batch; never trains or updates weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import zipfile

from configs import twc_v1_k8_rzf_probe_config as cfg
import evaluate_twc_v1_k8_rzf_probe as ev
from train_twc_v1_k8_ordered_encoder_screen import gpu_snapshot
from run_as_receiver_batch import _child,_run_stage

ROOT=cfg.REPO_ROOT
BATCH_ROOT=cfg.OUTPUT_ROOT/'batches'
read_json,save_json,sha256=ev.read_json,ev.save_json,ev.sha256
source_code_record,now=ev.source_code_record,ev.now


def plan():
    return dict(config=cfg.recipe(),source_implementation_commit=cfg.SOURCE_COMMIT,
        selection=dict(samples=64,addresses=[4096,4111],global_alpha_per_checkpoint=True,
            G4_both_ratios_pooled=True,exact_tie='smaller alpha',grid=list(cfg.ALPHAS)),
        validation=dict(samples=256,addresses=[5120,5183],conditions=72,
            reuse_Hhat_for_fixed_and_tuned=True,paired_rows=640),
        network_forwards=dict(calibration=576,evaluation=2304,not_multiplied_by_alphas=True),
        training=False,optimizer_created=False,beta_recalibrated=False,weights_loaded=False,
        Hhat_persisted=False,test_used=False,
        GPU_policy=dict(utilization_at_most=20,free_MiB_at_least=4096,other_processes_allowed=True,
                        continuous_monitoring=False))


def assert_code(summary):
    if summary['code']!=source_code_record():raise ValueError('Frozen diagnostic code changed during run')


def selection_stage(folder):
    pointer=folder/'calibration_pointer.json'
    if cfg.SELECTION_PATH.exists():
        selected=ev.load_selection()
        output=cfg.inside_root(selected['calibration_directory'])
        manifest=ev.verify_stage_files(output,'calibration')
    elif pointer.exists() and (Path(read_json(pointer)['directory'])/'manifest.json').exists() and read_json(Path(read_json(pointer)['directory'])/'manifest.json').get('status')=='completed':
        output=cfg.inside_root(read_json(pointer)['directory'])
        manifest=ev.verify_stage_files(output,'calibration')
        selected=ev.seal_selection(output)  # No new GPU calls; seals existing completed evidence.
    else:
        if (cfg.OUTPUT_ROOT/'selection_attempt.json').exists():
            raise RuntimeError('Partial numerical alpha calibration retained; no automatic reselection')
        if pointer.exists():save_json(folder/('prior_calibration_pointer_'+uuid.uuid4().hex[:8]+'.json'),read_json(pointer))
        output=folder/('calibration_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
        log=_child(folder,'sources_and_alpha_selection',[sys.executable,'-B','-u','evaluate_twc_v1_k8_rzf_probe.py',
            '--calibrate','--output-dir',str(output)])
        selected=ev.load_selection();manifest=ev.verify_stage_files(output,'calibration')
    if sha256(output/'manifest.json')!=selected['calibration_manifest_sha256']:raise ValueError('Locked calibration receipt changed')
    files=[pointer,cfg.SELECTION_PATH,cfg.OUTPUT_ROOT/'selection_attempt.json',output/'manifest.json']
    files+=[output/name for name in ev.ARTIFACTS['calibration']]+list(map(Path,selected['frozen_files']))
    return dict(directory=str(output),record_id=selected['record_id'],selected_alphas=selected['selected_alphas'],
        boundary_optima=selected['boundary_optima'],_evidence=files)


def validation_stage(folder):
    pointer=folder/'validation_pointer.json';logs=[]
    if pointer.exists() and (Path(read_json(pointer)['directory'])/'manifest.json').exists() and read_json(Path(read_json(pointer)['directory'])/'manifest.json').get('status')=='completed':
        output=cfg.inside_root(read_json(pointer)['directory'])
        manifest=ev.verify_stage_files(output,'evaluation')
    else:
        if pointer.exists():save_json(folder/('prior_validation_pointer_'+uuid.uuid4().hex[:8]+'.json'),read_json(pointer))
        output=folder/('validation_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
        logs=[_child(folder,'fixed_and_tuned_validation',[sys.executable,'-B','-u','evaluate_twc_v1_k8_rzf_probe.py',
            '--evaluate','--output-dir',str(output)])]
        manifest=ev.verify_stage_files(output,'evaluation')
    selected=ev.load_selection()
    if manifest.get('selection')!=cfg.selection_identity(selected):raise ValueError('Validation used a different alpha lock')
    return dict(directory=str(output),_evidence=[pointer,output/'manifest.json',*logs,
        *[output/name for name in ev.ARTIFACTS['evaluation']]])


def export_small(folder,summary):
    previous=summary.get('result_package')
    if previous and Path(previous).is_file() and sha256(previous)==summary.get('result_package_sha256'):return
    if previous:summary.setdefault('package_history',[]).append(previous)
    output=Path(summary['validation']['directory']);cal=Path(summary['selection']['directory'])
    files=[cfg.SELECTION_PATH,folder/'batch_summary.json',cal/'manifest.json',output/'manifest.json']
    files+=[cal/name for name in ev.ARTIFACTS['calibration']]
    files+=[output/name for name in ev.ARTIFACTS['evaluation'] if name!='samples.csv.gz']
    package=folder/('rzf_probe_results_'+uuid.uuid4().hex[:8]+'.zip')
    summary.update(result_package=str(package),return_files=[str(folder/'batch_summary.json'),str(cfg.SELECTION_PATH),
        str(output/'results.json'),str(output/'trends.json')])
    save_json(folder/'batch_summary.json',summary)
    names={str(path.relative_to(cfg.OUTPUT_ROOT)):path for path in files}
    expected={name:sha256(path) for name,path in names.items()}
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in names.items():archive.write(path,name)
    import hashlib
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist())!=set(expected) or any(hashlib.sha256(archive.read(n)).hexdigest()!=d for n,d in expected.items()):
            raise ValueError('Diagnostic small-package bytes changed')
    summary['result_package_sha256']=sha256(package);save_json(folder/'batch_summary.json',summary)


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual frozen diagnostic runs only on the CUDA server')
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='k8_rzf_probe_controller',arm='probe',SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json');assert_code(summary)
        else:
            if pointer.exists() or cfg.SELECTION_PATH.exists():raise FileExistsError('Existing diagnostic requires explicit --resume')
            folder=BATCH_ROOT/('batch_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),code=source_code_record(),plan=plan(),stages={},startup_snapshots=[])
            save_json(pointer,dict(directory=str(folder)));save_json(folder/'batch_summary.json',summary)
        if summary.get('status')=='completed':
            ev.load_selection();ev.verify_stage_files(summary['selection']['directory'],'calibration')
            ev.verify_stage_files(summary['validation']['directory'],'evaluation')
            if sha256(summary['result_package'])!=summary['result_package_sha256']:raise ValueError('Completed result package changed')
            print('Completed stages verified; no inference or alpha selection repeated.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary['status']='running'
        try:
            summary['startup_snapshots'].append(gpu_snapshot());save_json(folder/'batch_summary.json',summary)
            summary['selection']=_run_stage(summary,folder,'source_and_selection',lambda:selection_stage(folder))
            assert_code(summary)
            summary['validation']=_run_stage(summary,folder,'validation_and_trends',lambda:validation_stage(folder))
            assert_code(summary);summary.update(status='completed',finished=now());export_small(folder,summary)
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'batch_summary.json',summary);raise
        save_json(folder/'batch_summary.json',summary);return folder


def print_summary():
    pointer=cfg.OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; no source/CUDA/alpha acceptance inferred');return 0
    folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
    print(json.dumps({k:summary.get(k) for k in ('status','error','created','updated','finished','result_package','result_package_sha256','return_files')},indent=2))
    print('summary_path='+str(folder/'batch_summary.json'));print('diagnostic_commit='+summary['code']['commit'])
    print('source_implementation_commit='+cfg.SOURCE_COMMIT+'; internal='+cfg.INTERNAL_SOURCE_COMMIT+'; ancestor='+cfg.ANCESTOR_SOURCE_COMMIT)
    print('stages='+json.dumps({k:v['status'] for k,v in summary['stages'].items()}))
    if cfg.SELECTION_PATH.exists():
        try:
            selected=read_json(cfg.SELECTION_PATH);ev.validate_selection_record(selected)
            print('selected_alpha='+json.dumps(selected['selected_alphas']))
            print('predeclared_grid_boundary_optima='+json.dumps(selected['boundary_optima']))
            print('original_beta_unchanged='+json.dumps(selected.get('original_beta_identity')))
            print('alpha_calibration_scores='+json.dumps(selected['scores']))
        except Exception as error:print('selection_invalid='+str(error))
    else:print('alpha selection incomplete / unknown')
    directory=summary.get('validation',{}).get('directory')
    if not directory:
        for name in ('calibration_pointer.json','validation_pointer.json'):
            if (folder/name).exists():
                path=cfg.inside_root(read_json(folder/name)['directory'])
                if (path/'manifest.json').exists():
                    value=read_json(path/'manifest.json')
                    if value.get('status')!='completed':print(name+' failure='+json.dumps({k:value.get(k) for k in ('status','error','traceback','network_forward_calls','freeze_checks')}))
        print('final validation unknown / incomplete')
        try:
            # Read small existing logs only. No weight read or source acceptance
            # is inferred if the GPU/source stage has not completed.
            logs={arm:[json.loads(line) for line in Path(cfg.source_identity(arm)['progress_path']).read_text().splitlines() if line.strip()] for arm in cfg.ARMS}
            trends=ev.analyze_training_trends(logs,read_json(ev.source_cfg.CALIBRATION_PATH))
            print('monitor_log_only; complete source acceptance may still be pending')
            for arm,trend in trends['arms'].items():print(arm+' fixed_monitor='+json.dumps(trend['monitor'])+' last_four='+json.dumps(trend['last_four_monitor']))
        except Exception as error:print('monitor evidence unavailable; 无法判断: '+str(error))
        return 0
    output=Path(directory)
    if not (output/'results.json').exists():print('final validation incomplete');return 0
    results=read_json(output/'results.json');print('source_parameters='+json.dumps({k:v['parameters'] for k,v in results['methods'].items()}))
    curves={}
    for row in results['summary']:curves.setdefault((row['arm'],row['rzf_mode'],row['energy_policy']),[]).append(row)
    for key,rows in curves.items():
        rows=sorted(rows,key=lambda r:r['gamma_db']);print(str(key)+' alpha='+str(rows[0]['alpha']))
        for metric in ev.METRICS+ev.USER_RATES+('hhat_energy_total','hhat_fro_norm','hhat_to_true_norm_ratio','hhat_tone_user_energy_mean','relative_alpha_mean','relative_alpha_min','relative_alpha_max'):
            print(metric+'='+','.join(f'{r[metric+"_mean"]:.7f}' for r in rows))
        print('energy_max_abs_error='+str(max(r['energy_max_abs_error'] for r in rows))+
              '; rzf_power_max_abs_error='+str(max(r['rzf_power_max_abs_error'] for r in rows)))
    for row in results['paired']:
        if row['metric'] not in ('dl_nmse','sum_rate_bps_hz'):continue
        print(f"{row['contrast']} {row['metric']} gamma={row['estimand']}: {row['sample_mean']:+.7f} "
              f"sample95[{row['sample_ci95_low']:+.7f},{row['sample_ci95_high']:+.7f}] "
              f"batch95[{row['batch_ci95_low']:+.7f},{row['batch_ci95_high']:+.7f}]")
    for arm,trend in results['trends']['arms'].items():
        print(arm+' all_fixed_monitor='+json.dumps(trend['monitor'],ensure_ascii=False))
        print(arm+' last_four_fixed_monitor='+json.dumps(trend['last_four_monitor'],ensure_ascii=False))
        print(arm+' trend_assessment='+trend['assessment'])
        groups={key:dict(records=len(points),first=points[0],last=points[-1],
            beta_min=min(p['beta'] for p in points),beta_max=max(p['beta'] for p in points)) for key,points in trend['training_by_condition'].items()}
        print(arm+' training_conditions_first_last_only='+json.dumps(groups))
    print('Raw H_hat NMSE is alpha-invariant; no fitted sample scaling; no test used; no training or beta update.')
    print('Tuning may explain part of a gain; it does not prove a purely amplitude cause or full-SNR convergence.')
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
