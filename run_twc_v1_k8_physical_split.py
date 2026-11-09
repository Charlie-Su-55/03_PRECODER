#!/usr/bin/env python3
"""Cleanup, locked sources, CUDA checks and one four-arm physical-split batch."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import zipfile
from configs import twc_v1_k8_physical_split_config as cfg
import train_twc_v1_k8_physical_split as train
from train_twc_v1_k8_ordered_encoder_screen import gpu_snapshot
from run_as_receiver_batch import _child,_run_stage
from run_twc_v1_k8_decoder_screen import verify_completed_stages

read_json,save_json,sha256,now=train.read_json,train.save_json,train.sha256,train.now
BATCH_ROOT=cfg.OUTPUT_ROOT/'batches'


def plan():
    return dict(experiment=cfg.NAMESPACE,source_commit=cfg.SOURCE_COMMIT,arms=list(cfg.ARMS),
        updates_per_arm=cfg.update_counts(6048),included_head_only_iterations=108,
        training_addresses=[40241,46288],monitor_addresses=[14336,14351],monitor_steps=list(cfg.MONITOR_STEPS),
        alpha_calibration=[15360,15375],validation=[16384,16447],validation_MU_samples=256,
        methods=list(cfg.METHODS),fixed_alpha=.03,also_global_tuned=True,
        gradient_partition=dict(codeword='physical_head only',CSI='original BS + split bridge; physical inputs detach'),
        cleanup='only verified untracked generated caches; all experimental evidence/locks kept',
        models_built=False,weights_loaded=False,test_used=False)


def assert_code(summary):
    if summary['code']!=train.source_code_record():raise ValueError('Physical-split source code changed')


def cleanup_stage():
    from cleanup_twc_v1_k8 import run_cleanup
    report=run_cleanup(apply=True)
    if report.get('protected_and_tracked_state_unchanged') is not True:raise ValueError('Cleanup protection check failed')
    return dict(report_path=report['report_path'],deleted_files=len(report['deleted']),
        logical_deleted_bytes=report['logical_deleted_bytes'],observed_free_delta_bytes=report['observed_free_delta_bytes'],
        allocation_estimate_bytes=report['reclaimable_allocation_estimate_bytes'],
        checkpoints_outputs_archives_locks_untouched=True,_evidence=[Path(report['report_path'])])


def check_stage(folder,mode):
    report=folder/(mode+'_'+uuid.uuid4().hex[:8]+'.json');save_json(folder/(mode+'_pointer.json'),dict(report=str(report)))
    log=_child(folder,mode,[sys.executable,'-B','-u','train_twc_v1_k8_physical_split.py',
        '--accept-sources' if mode=='sources' else '--preflight','--report',str(report)])
    value=read_json(report)
    if value.get('status')!='passed':raise ValueError('Physical-split '+mode+' failed')
    files=[report,log]
    result=dict(report=str(report))
    if mode=='sources':
        locked=train.load_acceptance();result['record_id']=locked['record_id']
        files += [cfg.SOURCE_ACCEPTANCE_PATH,*map(Path,locked['frozen_files'])]
    else:result.update(parameters=value['parameters'],memory=value['memory'])
    return dict(**result,_evidence=files)


def inspect_training(arm,acceptance):
    folder=cfg.OUTPUT_ROOT/'training'/arm;r=read_json(folder/'manifest.json')
    for name,value in dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,code=train.source_code_record(),
        source=cfg.source_identity(arm,acceptance),recipe=cfg.recipe(arm,acceptance),
        source_acceptance=cfg.acceptance_identity(acceptance),config_fingerprint=cfg.fingerprint(arm,acceptance),
        ue_frozen=True,source_codec_updates=34192,source_bs_updates=6048,source_data_address=40240).items():
        if train.canonical(r.get(name))!=train.canonical(value):raise ValueError('Physical-split receipt '+name+' changed')
    step=r.get('step')
    if (type(step) is not int or not 108<=step<=6048 or step%108 or r.get('phase')!=cfg.phase(step)
            or r.get('update_counts')!=cfg.update_counts(step) or r.get('data_global_step')!=40240+step
            or r.get('actual_condition_counts')!=train.expected_counts(arm,step)):
        raise ValueError('Head/BS iteration budget or global address mismatch')
    if sha256(folder/'checkpoint_latest.pt')!=r.get('checkpoint_sha256'):raise ValueError('Latest checkpoint receipt SHA changed')
    return r


def verify_training(arm,acceptance):
    r=inspect_training(arm,acceptance);folder=cfg.OUTPUT_ROOT/'training'/arm
    if r['step']!=6048 or r.get('status')!='completed':raise ValueError('Fixed terminal6048 completion required')
    initial=read_json(folder/'initialization.json')
    if initial.get('new_two_optimizers') is not True or initial.get('optimizer_parameter_names')!=r['optimizer_parameter_names']:
        raise ValueError('Separate fresh optimizer initialization missing')
    if any(n.startswith('encoder.') for values in r['optimizer_parameter_names'].values() for n in values):
        raise ValueError('UE entered an optimizer')
    if set(r.get('monitor_records',{}))!={str(s) for s in cfg.MONITOR_STEPS}:raise ValueError('Full monitor trajectory missing')
    rows=[json.loads(line) for line in (folder/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [v.get('step') for v in rows]!=list(range(108,6049,108)):raise ValueError('Training progress coverage differs')
    for v in rows:
        if (v.get('update_counts')!=cfg.update_counts(v['step']) or v.get('phase')!=cfg.phase(v['step'])
                or train.canonical(v.get('condition'))!=train.canonical(cfg.schedule(arm,v['step']))):
            raise ValueError('Saved update/phase/condition trajectory differs')
    resumes=r.get('resume_records',{})
    if '108' not in resumes:raise ValueError('Automatic included108 pause/resume receipt missing')
    files=[]
    for step,entry in resumes.items():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Resume evidence changed')
        proof=read_json(entry['path'])
        if (proof.get('step')!=int(step) or proof.get('update_counts')!=cfg.update_counts(int(step))
                or proof.get('next_iteration')!=int(step)+1 or proof.get('RNG_restored') is not True
                or proof.get('restored_optimizer_states') is not True):raise ValueError('Dual-optimizer restoration not verified')
        files.append(Path(entry['path']))
    fixed_inputs=None
    for step,entry in r['monitor_records'].items():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Monitor evidence SHA changed')
        report=read_json(entry['path'])
        if (report.get('step')!=int(step) or report.get('update_counts')!=cfg.update_counts(int(step))
                or report.get('alpha')!=.03 or len(report.get('summary',[]))!=(6 if cfg.is_fdma(arm) else 12)
                or [v.get('address') for v in report.get('shared_inputs',[])]!=list(range(14336,14352))):
            raise ValueError('Monitor grid/address/phase changed')
        if fixed_inputs is not None and report['shared_inputs']!=fixed_inputs:raise ValueError('Monitor input set changed')
        fixed_inputs=report['shared_inputs'];files.append(Path(entry['path']))
    terminal=read_json(folder/'terminal_checkpoint.json')
    if terminal!=dict(path=str(folder/'checkpoint_latest.pt'),sha256=r['checkpoint_sha256'],
                     update_counts=cfg.update_counts(6048),data_global_step=46288):
        raise ValueError('Terminal head/BS receipt differs')
    r['checkpoint_path']=str(folder/'checkpoint_latest.pt')
    return dict(report=r,_evidence=[folder/name for name in ('checkpoint_latest.pt','manifest.json','initialization.json','progress.jsonl','terminal_checkpoint.json')]+files)


def run_arm(folder,arm,acceptance):
    target=cfg.OUTPUT_ROOT/'training'/arm;cmd=[sys.executable,'-B','-u','train_twc_v1_k8_physical_split.py','--arm',arm];logs=[]
    if not (target/'checkpoint_latest.pt').exists():
        if target.exists():raise FileExistsError('Unowned partial training output retained')
        logs.append(_child(folder,arm+'_head108',cmd+['--stop-after','108']))
    r=inspect_training(arm,acceptance)
    if r['step']==108 and r['update_counts']['bs_updates']!=0:raise ValueError('Warm-up changed BS update count')
    if r.get('status')!='completed' or r['step']!=6048:
        logs.append(_child(folder,arm+'_continue',cmd+['--resume','--train']))
    value=verify_training(arm,acceptance);value['_evidence']+=logs;return value


def training_audit(folder,acceptance):
    from evaluate_twc_v1_k8_rzf_probe import _write_csv
    records={a:verify_training(a,acceptance)['report'] for a in cfg.ARMS}
    if len({r['training_input_history'] for r in records.values()})!=1:raise ValueError('Actual common H/standard-noise histories differ')
    for family in ('fdma','g4'):
        if records[family+'_control']['layout_transmit_history']!=records[family+'_split']['layout_transmit_history']:
            raise ValueError('Same-layout X/Y/Hest histories differ')
    monitors=[];means=[];progress=[];shared=None
    for arm,r in records.items():
        for step,entry in r['monitor_records'].items():
            report=read_json(entry['path'])
            if shared is not None and shared!=report['shared_inputs']:raise ValueError('Cross-arm monitor pairing differs')
            shared=report['shared_inputs'];monitors.extend(dict(arm=arm,step=int(step),**v) for v in report['summary'])
            means.extend(dict(arm=arm,step=int(step),**v) for v in report['all_six'])
        for line in (cfg.OUTPUT_ROOT/'training'/arm/'progress.jsonl').read_text(encoding='utf-8').splitlines():
            v=json.loads(line);progress.append(dict(arm=arm,step=v['step'],head_updates=v['update_counts']['head_updates'],
                bs_updates=v['update_counts']['bs_updates'],CSI_NMSE_linear=v['csi_nmse_linear'],codeword_NMSE_linear=v['head_nmse_linear'],
                rate=v['rate'],condition=json.dumps(v['condition'],sort_keys=True)))
    receipt=dict(passed=True,actual_H_and_noise_all_iterations_match=True,same_layout_actual_X_Y_Hest_match=True,
        cross_layout_X_Y_not_assumed_equal=True,updates=cfg.update_counts(6048),UE_CE_frozen=True,test_used=False)
    save_json(folder/'training_audit.json',receipt)
    for name,rows in (('monitor_full_grid.csv',monitors),('monitor_all_six.csv',means),('training_progress.csv',progress)):_write_csv(folder/name,rows)
    return dict(report=receipt,_evidence=[folder/n for n in ('training_audit.json','monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv')])


def evaluation_stage(folder,stage):
    import evaluate_twc_v1_k8_physical_split as ev
    pointer=folder/(stage+'_pointer.json');logs=[];candidate=cfg.inside_root(read_json(pointer)['directory']) if pointer.exists() else None
    if stage=='calibration' and cfg.SELECTION_PATH.exists():
        output=cfg.inside_root(ev.load_selection()['calibration_directory'])
    elif candidate is not None and (candidate/'manifest.json').is_file() and read_json(candidate/'manifest.json').get('status')=='completed':
        output=candidate
        if stage=='calibration':ev.seal_selection(output)
    else:
        if stage=='calibration' and (cfg.OUTPUT_ROOT/'selection_attempt.json').exists():raise RuntimeError('Partial alpha lock retained; no silent reselection')
        if pointer.exists():save_json(folder/('prior_'+stage+'_pointer_'+uuid.uuid4().hex[:8]+'.json'),read_json(pointer))
        output=folder/(stage+'_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
        logs=[_child(folder,stage,[sys.executable,'-B','-u','evaluate_twc_v1_k8_physical_split.py',
            '--calibrate' if stage=='calibration' else '--evaluate','--output-dir',str(output)])]
    ev.verify_stage_files(output,stage)
    result=dict(directory=str(output),_evidence=[pointer,output/'manifest.json',*logs,*[output/n for n in ev.ARTIFACTS[stage]]])
    if stage=='calibration':
        selection=ev.load_selection();result.update(selected_alphas=selection['selected_alphas'],record_id=selection['record_id'])
        result['_evidence'] += [cfg.SELECTION_PATH,cfg.OUTPUT_ROOT/'selection_attempt.json']
    return result


def export_small(folder,summary):
    import evaluate_twc_v1_k8_physical_split as ev
    old=summary.get('result_package')
    if old and Path(old).is_file() and sha256(old)==summary.get('result_package_sha256'):return
    evaluation=Path(summary['validation']['directory']);calibration=Path(summary['alpha_selection']['directory'])
    files=[folder/'batch_summary.json',cfg.SOURCE_ACCEPTANCE_PATH,cfg.SELECTION_PATH,evaluation/'manifest.json',calibration/'manifest.json',
        Path(summary['cleanup']['report_path']),Path(summary['preflight']['report'])]
    files += [calibration/name for name in ev.ARTIFACTS['calibration']]
    files += [evaluation/name for name in ev.ARTIFACTS['evaluation'] if not name.endswith('.gz')]
    files += [folder/name for name in ('training_audit.json','monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv')]
    for arm in cfg.ARMS:
        r=read_json(cfg.OUTPUT_ROOT/'training'/arm/'manifest.json')
        files += [Path(e['path']) for e in r['monitor_records'].values()]
    package=folder/('physical_split_results_'+uuid.uuid4().hex[:8]+'.zip')
    summary.update(result_package=str(package),return_files=[str(folder/'batch_summary.json'),str(evaluation/'results.json'),
        str(cfg.SOURCE_ACCEPTANCE_PATH),str(cfg.SELECTION_PATH)])
    save_json(folder/'batch_summary.json',summary)
    names={str(p.relative_to(cfg.REPO_ROOT)):p for p in files};expected={n:sha256(p) for n,p in names.items()}
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in names.items():archive.write(path,name)
    import hashlib
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist())!=set(expected) or any(hashlib.sha256(archive.read(n)).hexdigest()!=v for n,v in expected.items()):
            raise ValueError('Small result package verification failed')
    summary['result_package_sha256']=sha256(package);save_json(folder/'batch_summary.json',summary)


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual batch requires the CUDA server')
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='k8_physical_split_controller',arm='controller',SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
            assert_code(summary);verify_completed_stages(summary)
        else:
            if pointer.exists() or cfg.SOURCE_ACCEPTANCE_PATH.exists():raise FileExistsError('Existing batch needs explicit --resume')
            if any((cfg.OUTPUT_ROOT/'training'/a).exists() for a in cfg.ARMS):raise FileExistsError('Unowned arm output retained')
            folder=BATCH_ROOT/('batch_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),updated=now(),code=train.source_code_record(),plan=plan(),stages={},training={})
            save_json(pointer,dict(directory=str(folder)));save_json(folder/'batch_summary.json',summary)
        if summary.get('status')=='completed':
            acceptance=train.load_acceptance()
            for arm in cfg.ARMS:verify_training(arm,acceptance)
            import evaluate_twc_v1_k8_physical_split as ev
            ev.load_selection();ev.verify_stage_files(summary['validation']['directory'],'evaluation')
            if sha256(summary['result_package'])!=summary['result_package_sha256']:raise ValueError('Completed package changed')
            print('Completed stages verified; no training repeated.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary.update(status='running',updated=now())
        try:
            summary['cleanup']=_run_stage(summary,folder,'cleanup',cleanup_stage)
            summary.setdefault('startup_snapshots',[]).append(gpu_snapshot());save_json(folder/'batch_summary.json',summary)
            summary['sources']=_run_stage(summary,folder,'sources',lambda:check_stage(folder,'sources'))
            acceptance=train.load_acceptance();summary['preflight']=_run_stage(summary,folder,'cuda_preflight',lambda:check_stage(folder,'preflight'))
            for arm in cfg.ARMS:
                assert_code(summary);summary['training'][arm]=_run_stage(summary,folder,'train_'+arm,lambda a=arm:run_arm(folder,a,acceptance))['report']
            summary['training_audit']=_run_stage(summary,folder,'training_audit',lambda:training_audit(folder,acceptance))['report']
            summary['alpha_selection']=_run_stage(summary,folder,'terminal_alpha_selection',lambda:evaluation_stage(folder,'calibration'))
            summary['validation']=_run_stage(summary,folder,'terminal_validation',lambda:evaluation_stage(folder,'evaluation'))
            assert_code(summary);summary.update(status='completed',finished=now(),updated=now());export_small(folder,summary)
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'batch_summary.json',summary);raise
        save_json(folder/'batch_summary.json',summary);return folder


def safe_read(path):
    try:
        path=cfg.inside_root(path)
        return read_json(path) if path.is_file() else None
    except (OSError,ValueError,KeyError,TypeError):return None


def summary_lines(summary,records,result=None,selection=None,tail=None):
    lines=[f"status={summary.get('status','unknown')} error={summary.get('error','none')}",
        'updated='+max([str(summary.get('updated','unknown')),*[str(r.get('updated_utc','')) for r in records.values()]]),
        'gamma=0,5,10,15,20,25; alpha fixed=.03 plus locked tuned; single-seed validation; test sealed',
        'stages='+','.join(k+':'+v.get('status','unknown') for k,v in summary.get('stages',{}).items())]
    for arm in cfg.ARMS:
        r=records.get(arm,{});counts=r.get('update_counts',{})
        lines.append(f"{arm} {r.get('status','not_started')} data={r.get('step','?')}/6048 head={counts.get('head_updates','?')}"
            f" BS={counts.get('bs_updates','?')}/5940 UE=0 CE=0 phase={r.get('phase','?')}")
    for arm in cfg.ARMS:
        entries=records.get(arm,{}).get('monitor_records',{});value='unknown'
        if entries:
            step=max(entries,key=int);r=safe_read(entries[step]['path'])
            if r:value=step+' '+ ';'.join(v['energy_policy']+':'+format(v['sum_rate_bps_hz_mean'],'.4f') for v in r.get('all_six',[]))
        lines.append(arm+' latest_monitor='+value)
    if result:
        grouped={}
        for r in result.get('summary',[]):grouped.setdefault((r['arm'],r['energy_policy']),{}).setdefault(r['rzf_mode'],[]).append(r)
        for (arm,policy),modes in sorted(grouped.items()):
            lines.append(arm+'/'+policy+' '+' '.join(mode+'='+','.join(format(r['sum_rate_bps_hz_mean'],'.3f') for r in sorted(rows,key=lambda x:x['gamma_db']))
                for mode,rows in sorted(modes.items())))
        priorities=['fdma_split_minus_control_fdma_fixed','g4_split_minus_control_hybrid_1_3_fixed',
            'g4_split_minus_control_hybrid_1_1_fixed','split_g4_minus_fdma_hybrid_1_3_fixed',
            'split_g4_minus_fdma_hybrid_1_1_fixed']
        for contrast in priorities:
            values=[r for r in result.get('paired',[]) if r['contrast']==contrast and r['metric']=='sum_rate_bps_hz'
                and r['estimand'] in ('low_0_5_10','all_six')]
            if values:lines.append(contrast+' '+';'.join(r['estimand']+f":{r['sample_mean']:+.3f}[{r['batch_ci95_low']:+.3f},{r['batch_ci95_high']:+.3f}]" for r in values))
        diagnostics=result.get('receiver_diagnostics',[])
        for arm in cfg.ARMS:
            items=[r for r in diagnostics if r.get('arm')==arm and r.get('estimand')=='all_six'
                and r.get('metric') in ('physical_soft_codeword_nmse_linear_all','physical_head_codeword_nmse_linear_all')]
            if items:lines.append(arm+' physical_linear_all='+ ';'.join(str(r.get('energy_policy'))+'/'+r['metric'].replace('_codeword_nmse_linear_all','')+'='+format(r['sample_mean'],'.5g') for r in items))
    else:lines.append('final six-point results, paired CI and physical errors=unknown')
    if selection:lines.append('alpha='+','.join(m+':'+str(selection['selected_alphas'][m]) for m in cfg.METHODS))
    else:lines.append('alpha=unknown')
    suffix=(['failure_log_tail:']+tail) if tail else []
    suffix+=['package='+str(summary.get('result_package','unknown')),
        'summary='+str(summary.get('summary_path','unknown'))+'; --summary --details for full data']
    lines=lines[:max(0,40-len(suffix))]
    # Reserve the final exact paths even if failure output contains giant lines.
    body=[];used=0
    footer=[line[:600] for line in suffix];reserved=len(('\n'.join(footer)+'\n').encode())
    for line in lines:
        line=line[:600];size=len((line+'\n').encode())
        if used+size+reserved>8000:break
        body.append(line);used+=size
    return body+footer


def print_summary(details=False):
    pointer=cfg.OUTPUT_ROOT/'last_batch.json'
    if not pointer.is_file():print('not_started; no server cleanup/source/CUDA/training acceptance inferred');return 0
    p=read_json(pointer);folder=cfg.inside_root(p['directory']);summary=read_json(folder/'batch_summary.json')
    summary['summary_path']=str(folder/'batch_summary.json')
    records={a:(safe_read(cfg.OUTPUT_ROOT/'training'/a/'manifest.json') or dict(status='unknown')) for a in cfg.ARMS}
    directory=summary.get('validation',{}).get('directory')
    results=safe_read(Path(directory)/'results.json') if directory else None;selection=safe_read(cfg.SELECTION_PATH)
    failure_tail=[]
    if summary.get('status')=='failed':
        # Exact files in this batch only, not a repository-wide newest-log search.
        logs=[p for p in folder.iterdir() if p.is_file() and p.suffix=='.log' and not p.is_symlink()]
        if logs:
            path=max(logs,key=lambda p:p.stat().st_mtime_ns)
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-4096));failure_tail=stream.read().decode('utf-8',errors='replace').splitlines()[-5:]
    if details:
        monitors={a:{s:safe_read(e['path']) for s,e in r.get('monitor_records',{}).items()} for a,r in records.items()}
        checks={}
        for mode in ('sources','preflight'):
            item=safe_read(folder/(mode+'_pointer.json'))
            if item:checks[mode]=safe_read(item['report'])
        print(json.dumps(dict(summary=summary,training=records,monitors=monitors,checks=checks,
            selection=selection,results=results,failure_log_tail=failure_tail),indent=2))
    else:print('\n'.join(summary_lines(summary,records,results,selection,failure_tail)))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    for name in ('dry-run','run','summary'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--details',action='store_true');args=parser.parse_args(argv)
    if args.resume and not args.run:parser.error('--resume requires --run')
    if args.details and not args.summary:parser.error('--details requires --summary')
    if args.dry_run:print(json.dumps(plan(),indent=2))
    elif args.run:run(args.resume)
    elif args.summary:return print_summary(args.details)
    else:parser.print_help()
    return 0


if __name__=='__main__':raise SystemExit(main())
