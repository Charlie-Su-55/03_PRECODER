#!/usr/bin/env python3
"""Single independent decoder screen; no server action without explicit --run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import zipfile

from configs import twc_v1_k8_decoder_screen_config as cfg
import train_twc_v1_k8_decoder_screen as train
from train_twc_v1_k8_ordered_encoder_screen import gpu_snapshot
from run_as_receiver_batch import _child,_run_stage

read_json,save_json,sha256,now=train.read_json,train.save_json,train.sha256,train.now
BATCH_ROOT=cfg.OUTPUT_ROOT/'batches'


def plan():
    return dict(experiment=cfg.NAMESPACE,source_implementation_commit=cfg.SOURCE_COMMIT,
        arms=[dict(arm=a,recipe=cfg.recipe(a),new_bs_updates=6048,included_smoke=108) for a in cfg.ARMS],
        total_BS_updates=36288,UE_updates=0,CE_updates=0,training_addresses=[34193,40240],
        source_SHA='server cross-check manifest/terminal/batch/state/log; no guessed pin',
        full_chain_migration='B4 CUDA acceptance pending; unchanged original tolerances',
        source_acceptance=str(cfg.SOURCE_ACCEPTANCE_PATH),test_used=False,
        monitor=dict(addresses=[10240,10255],samples=64,steps=list(cfg.MONITOR_STEPS),alpha=.03),
        final_alpha=dict(addresses=[11264,11279],samples=64,grid=list(cfg.ALPHAS),models=8),
        final_validation=dict(addresses=[12288,12351],samples=256,fixed_alpha=.03,also_tuned=True),
        weights_read=False,models_built=False,shared_GPU_policy=dict(max_utilization=20,min_free_MiB=4096,
        continuous_monitoring=False,other_processes_terminated=False))


def assert_code(summary):
    if summary['code']!=train.source_code_record():raise ValueError('Decoder batch code changed')


def verify_completed_stages(summary):
    for name,stage in summary.get('stages',{}).items():
        if stage.get('status')!='completed':continue
        evidence=stage.get('evidence_sha256')
        if not isinstance(evidence,dict) or not evidence:raise ValueError('Missing completed-stage evidence: '+name)
        train.verify_frozen(evidence)


def acceptance_stage(folder):
    report=folder/('source_acceptance_'+uuid.uuid4().hex[:8]+'.json')
    save_json(folder/'source_pointer.json',dict(report=str(report)))
    log=_child(folder,'source_acceptance',[sys.executable,'-B','-u','train_twc_v1_k8_decoder_screen.py',
        '--accept-sources','--report',str(report)])
    record=train.load_acceptance()
    return dict(record_id=record['record_id'],sources={m:v['identity'] for m,v in record['sources'].items()},
        _evidence=[report,log,cfg.SOURCE_ACCEPTANCE_PATH,*map(Path,record['frozen_files'])])


def preflight_stage(folder):
    report=folder/('cuda_preflight_'+uuid.uuid4().hex[:8]+'.json')
    save_json(folder/'preflight_pointer.json',dict(report=str(report)))
    log=_child(folder,'cuda_preflight',[sys.executable,'-B','-u','train_twc_v1_k8_decoder_screen.py',
        '--preflight','--report',str(report)])
    value=read_json(report)
    if value.get('status')!='passed':raise ValueError('Decoder CUDA acceptance failed')
    return dict(report=str(report),parameters=value['parameters'],memory=value['memory'],_evidence=[report,log])


def inspect_training(arm,acceptance):
    folder=cfg.OUTPUT_ROOT/'training'/arm;record=read_json(folder/'manifest.json')
    for name,value in dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,code=train.source_code_record(),
        recipe=cfg.recipe(arm,acceptance),config_fingerprint=cfg.fingerprint(arm,acceptance),
        source=cfg.source_identity(arm,acceptance),source_acceptance=cfg.acceptance_identity(acceptance),
        ue_frozen=True,source_codec_updates=34192).items():
        if train.canonical(record.get(name))!=train.canonical(value):raise ValueError('Decoder receipt '+name+' differs')
    step=record.get('step')
    if (type(step) is not int or not 108<=step<=6048 or step%108 or record.get('new_bs_updates')!=step
            or record.get('data_global_step')!=34192+step
            or record.get('actual_condition_counts')!=train.expected_counts(arm,step)):
        raise ValueError('BS-only step/address/counts mismatch')
    if sha256(folder/'checkpoint_latest.pt')!=record.get('checkpoint_sha256'):raise ValueError('Latest SHA receipt mismatch')
    return record


def verify_training(arm,acceptance):
    record=inspect_training(arm,acceptance);folder=cfg.OUTPUT_ROOT/'training'/arm
    if record.get('status')!='completed' or record['step']!=6048:raise ValueError('Fixed terminal6048 required')
    initial=read_json(folder/'initialization.json')
    if (initial.get('new_bs_optimizer') is not True or initial.get('optimizer_parameter_names')!=record['optimizer_parameter_names']
            or any(n.startswith('encoder.') for n in record['optimizer_parameter_names'])):
        raise ValueError('Fresh BS-only optimizer evidence differs')
    if not initial.get('frozen_state',{}).get('ue') or not initial.get('frozen_state',{}).get('ce'):
        raise ValueError('Frozen UE/CE initialization evidence missing')
    rows=[json.loads(s) for s in (folder/'progress.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if [r.get('step') for r in rows]!=list(range(108,6049,108)):raise ValueError('Incomplete decoder training log')
    for row in rows:
        if (row.get('data_global_step')!=34192+row['step']
                or train.canonical(row.get('condition'))!=train.canonical(cfg.schedule(arm,row['step']))):
            raise ValueError('Decoder data/condition trajectory differs')
    if cfg.variant(arm)!='base2':
        if not any(v>0 for r in rows[:1] for v in r.get('new_branch_gradients',{}).values()):
            raise ValueError('No finite new output-path learning evidence by first108 updates')
    files=[]
    if set(record.get('monitor_records',{}))!={str(s) for s in cfg.MONITOR_STEPS}:raise ValueError('Missing monitor checkpoint')
    reference=None
    for step,entry in record['monitor_records'].items():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Monitor hash differs')
        monitor=read_json(entry['path'])
        if (monitor.get('step')!=int(step) or monitor.get('alpha')!=.03
                or len(monitor.get('summary',[]))!=(6 if cfg.is_fdma(arm) else 12)
                or len(monitor.get('all_six',[]))!=(1 if cfg.is_fdma(arm) else 2)
                or [r.get('address') for r in monitor.get('shared_inputs',[])]!=list(range(10240,10256))):
            raise ValueError('Monitor full-grid coverage/alpha mismatch')
        if reference is not None and monitor['shared_inputs']!=reference:raise ValueError('Monitor input set changed')
        reference=monitor['shared_inputs'];files.append(Path(entry['path']))
    terminal=read_json(folder/'terminal_checkpoint.json')
    if (terminal.get('sha256')!=record['checkpoint_sha256'] or terminal.get('new_bs_updates')!=6048
            or terminal.get('source_codec_updates')!=34192 or terminal.get('data_global_step')!=40240):
        raise ValueError('Decoder terminal receipt differs')
    record['checkpoint_path']=str(folder/'checkpoint_latest.pt')
    return dict(report=record,_evidence=[folder/n for n in ('checkpoint_latest.pt','manifest.json','initialization.json',
        'progress.jsonl','terminal_checkpoint.json')]+files)


def run_arm(folder,arm,acceptance):
    destination=cfg.OUTPUT_ROOT/'training'/arm;logs=[]
    cmd=[sys.executable,'-B','-u','train_twc_v1_k8_decoder_screen.py','--arm',arm]
    if not (destination/'checkpoint_latest.pt').exists():
        if destination.exists():raise FileExistsError('Unowned partial arm retained: '+arm)
        logs.append(_child(folder,arm+'_first108',cmd+['--stop-after','108']))
    record=inspect_training(arm,acceptance)
    if record['step']==108 and cfg.variant(arm)!='base2':
        if not any(v>0 for v in record.get('new_branch_gradients',{}).values()):
            raise ValueError('New branch did not receive effective gradient by108: '+arm)
        original=read_json(destination/'initialization.json').get('new_state_sha256',{})
        if not original or not any(record.get('new_branch_state',{}).get(n,d)!=d for n,d in original.items()):
            raise ValueError('No new parameter changed by first108 updates: '+arm)
    if record.get('status')!='completed' or record['step']!=6048:
        logs.append(_child(folder,arm+'_continue',cmd+['--resume','--train']))
    result=verify_training(arm,acceptance);result['_evidence']+=logs;return result


def audit_training(folder,acceptance):
    from evaluate_twc_v1_k8_rzf_probe import _write_csv
    records={a:verify_training(a,acceptance)['report'] for a in cfg.ARMS}
    if len({r['training_input_history'] for r in records.values()})!=1:
        raise ValueError('Six-arm actual H/standard-noise histories not shared')
    for family in ('fdma','g4'):
        if len({records[a]['layout_transmit_history'] for a in cfg.ARMS if a.startswith(family+'_')})!=1:
            raise ValueError('Same-layout frozen-UE X/Y/Hest histories differ: '+family)
    monitors=[];means=[];progress=[];fixed_inputs=None
    for arm,r in records.items():
        for step,entry in r['monitor_records'].items():
            report=read_json(entry['path'])
            if fixed_inputs is not None and fixed_inputs!=report['shared_inputs']:raise ValueError('Monitor pairing mismatch')
            fixed_inputs=report['shared_inputs']
            monitors.extend(dict(arm=arm,step=int(step),**row) for row in report['summary'])
            means.extend(dict(arm=arm,step=int(step),**row) for row in report['all_six'])
        for line in (cfg.OUTPUT_ROOT/'training'/arm/'progress.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);progress.append(dict(arm=arm,step=row['step'],nmse=row['nmse'],rate=row['rate'],
                grad_norm=row['grad_norm'],condition=json.dumps(row['condition'],sort_keys=True)))
    proof=dict(passed=True,all_six_channel_standard_noise_hashes_match=True,
        per_layout_X_Y_Hest_hashes_match=True,complete_tensor_hashes_every_update=True,UE_CE_frozen=True,
        cross_layout_X_Y_not_assumed_equal=True,test_used=False)
    save_json(folder/'training_audit.json',proof)
    for name,rows in (('monitor_full_grid.csv',monitors),('monitor_all_six.csv',means),('training_progress.csv',progress)):
        _write_csv(folder/name,rows)
    return dict(report=proof,_evidence=[folder/n for n in ('training_audit.json','monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv')])


def evaluation_stage(folder,stage):
    import evaluate_twc_v1_k8_decoder_screen as ev
    pointer=folder/(stage+'_pointer.json');logs=[]
    candidate=cfg.inside_root(read_json(pointer)['directory']) if pointer.exists() else None
    if stage=='calibration' and cfg.SELECTION_PATH.exists():
        selected=ev.load_selection();output=cfg.inside_root(selected['calibration_directory'])
    elif candidate is not None and (candidate/'manifest.json').is_file() and read_json(candidate/'manifest.json').get('status')=='completed':
        output=candidate
        if stage=='calibration':ev.seal_selection(output)
    else:
        if stage=='calibration' and (cfg.OUTPUT_ROOT/'selection_attempt.json').exists():
            raise RuntimeError('Incomplete alpha attempt retained; no silent reselection')
        if pointer.exists():save_json(folder/('previous_'+stage+'_pointer_'+uuid.uuid4().hex[:8]+'.json'),read_json(pointer))
        output=folder/(stage+'_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)))
        logs=[_child(folder,stage,[sys.executable,'-B','-u','evaluate_twc_v1_k8_decoder_screen.py',
            '--calibrate' if stage=='calibration' else '--evaluate','--output-dir',str(output)])]
    ev.verify_stage_files(output,stage)
    result=dict(directory=str(output),_evidence=[pointer,output/'manifest.json',*logs,*[output/n for n in ev.ARTIFACTS[stage]]])
    if stage=='calibration':
        record=ev.load_selection();result.update(record_id=record['record_id'],selected_alphas=record['selected_alphas'],
            boundary_optima=record['boundary_optima']);result['_evidence']+=[cfg.SELECTION_PATH,cfg.OUTPUT_ROOT/'selection_attempt.json']
    return result


def export_small(folder,summary):
    import evaluate_twc_v1_k8_decoder_screen as ev
    old=summary.get('result_package')
    if old and Path(old).is_file() and sha256(old)==summary.get('result_package_sha256'):return
    output=Path(summary['validation']['directory']);alpha=Path(summary['alpha_selection']['directory'])
    paths=[cfg.SOURCE_ACCEPTANCE_PATH,cfg.SELECTION_PATH,folder/'batch_summary.json',output/'manifest.json',alpha/'manifest.json']
    paths += [alpha/n for n in ev.ARTIFACTS['calibration']]
    paths += [output/n for n in ev.ARTIFACTS['evaluation'] if not n.endswith('.gz')]
    paths += [folder/n for n in ('training_audit.json','monitor_full_grid.csv','monitor_all_six.csv','training_progress.csv')]
    paths += [Path(e['path']) for a in cfg.ARMS for e in read_json(cfg.OUTPUT_ROOT/'training'/a/'manifest.json')['monitor_records'].values()]
    paths += [Path(summary['preflight']['report'])]
    package=folder/('decoder_screen_results_'+uuid.uuid4().hex[:8]+'.zip')
    summary.update(result_package=str(package),return_files=[str(folder/'batch_summary.json'),str(cfg.SOURCE_ACCEPTANCE_PATH),
        str(cfg.SELECTION_PATH),str(output/'results.json')])
    save_json(folder/'batch_summary.json',summary)
    files={str(p.relative_to(cfg.OUTPUT_ROOT)):p for p in paths};expected={n:sha256(p) for n,p in files.items()}
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items():archive.write(path,name)
    import hashlib
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist())!=set(expected) or any(hashlib.sha256(archive.read(n)).hexdigest()!=h for n,h in expected.items()):
            raise ValueError('Small package content verification failed')
    summary['result_package_sha256']=sha256(package);save_json(folder/'batch_summary.json',summary)


def run(resume=False):
    if os.name!='posix':raise RuntimeError('Actual batch requires the CUDA server')
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='k8_decoder_screen_controller',arm='controller',SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json');assert_code(summary)
            verify_completed_stages(summary)
        else:
            if pointer.exists() or cfg.SOURCE_ACCEPTANCE_PATH.exists():raise FileExistsError('Existing decoder batch requires explicit --resume')
            if any((cfg.OUTPUT_ROOT/'training'/a).exists() for a in cfg.ARMS):raise FileExistsError('Unowned arm output retained')
            folder=BATCH_ROOT/('batch_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),updated=now(),code=train.source_code_record(),
                plan=plan(),stages={},training={},startup_snapshots=[])
            save_json(pointer,dict(directory=str(folder)));save_json(folder/'batch_summary.json',summary)
        if summary.get('status')=='completed':
            acceptance=train.load_acceptance()
            for arm in cfg.ARMS:verify_training(arm,acceptance)
            import evaluate_twc_v1_k8_decoder_screen as ev
            ev.load_selection();ev.verify_stage_files(summary['validation']['directory'],'evaluation')
            if sha256(summary['result_package'])!=summary['result_package_sha256']:raise ValueError('Result package changed')
            print('Completed evidence verified; no work repeated.');return folder
        if summary.get('error'):summary.setdefault('failure_history',[]).append(dict(time=now(),error=summary.pop('error')))
        summary.update(status='running',updated=now())
        try:
            summary['startup_snapshots'].append(gpu_snapshot());save_json(folder/'batch_summary.json',summary)
            summary['sources']=_run_stage(summary,folder,'sources',lambda:acceptance_stage(folder))
            acceptance=train.load_acceptance()
            summary['preflight']=_run_stage(summary,folder,'cuda_preflight',lambda:preflight_stage(folder))
            for arm in cfg.ARMS:
                assert_code(summary)
                summary['training'][arm]=_run_stage(summary,folder,'train_'+arm,lambda a=arm:run_arm(folder,a,acceptance))['report']
            summary['training_audit']=_run_stage(summary,folder,'training_audit',lambda:audit_training(folder,acceptance))['report']
            assert_code(summary);summary['alpha_selection']=_run_stage(summary,folder,'terminal_alpha_selection',lambda:evaluation_stage(folder,'calibration'))
            assert_code(summary);summary['validation']=_run_stage(summary,folder,'terminal_validation',lambda:evaluation_stage(folder,'evaluation'))
            assert_code(summary);summary.update(status='completed',updated=now(),finished=now());export_small(folder,summary)
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',updated=now())
            save_json(folder/'batch_summary.json',summary);raise
        save_json(folder/'batch_summary.json',summary);return folder


def summary_lines(summary,training,results=None,selection=None):
    """Bounded human summary; detailed machine records stay in their exact files."""
    updated=max([str(summary.get('updated',summary.get('created','unknown'))),
        *[str(r.get('updated_utc','')) for r in training.values()]])
    lines=[f"status={summary.get('status','unknown')} error={summary.get('error','none')}",
        f"updated={updated} commit={summary.get('code',{}).get('commit','unknown')}",
        'gamma=0,5,10,15,20,25; fixed_alpha=.03; single-seed BS-only validation; test sealed',
        'stages='+','.join(k+':'+v.get('status','unknown') for k,v in summary.get('stages',{}).items())]
    for arm in cfg.ARMS:
        r=training.get(arm,{})
        lines.append(f"{arm}: {r.get('status','not_started')} BSstep={r.get('step','?')}/6048 data={r.get('data_global_step','?')} UE=frozen"
                     +(' error='+str(r['error']) if r.get('error') else ''))
    for arm in cfg.ARMS:
        r=training.get(arm,{});entries=r.get('monitor_records',{})
        if entries:
            step=max(entries,key=int);p=Path(entries[step]['path'])
            try:
                p=cfg.inside_root(p)
                if p.is_file():
                    report=read_json(p);text=';'.join(str(x['energy_policy'])+':'+format(x.get('sum_rate_bps_hz_mean',0),'.4f') for x in report.get('all_six',[]))
                    lines.append(arm+' latest_monitor='+step+' sixmean='+text)
                    continue
            except (OSError,ValueError,KeyError,TypeError):
                pass
        lines.append(arm+' latest_monitor=unknown')
    if selection:lines.append('alpha='+','.join(m+':'+str(selection['selected_alphas'][m]) for m in cfg.METHODS))
    else:lines.append('alpha=unknown')
    if results:
        groups={}
        for r in results.get('summary',[]):groups.setdefault((r['arm'],r['energy_policy']),{}).setdefault(r['rzf_mode'],[]).append(r)
        for (arm,policy),modes in sorted(groups.items()):
            parts=[]
            for mode,rows in sorted(modes.items()):
                rows.sort(key=lambda x:x['gamma_db'])
                parts.append(mode+'='+','.join(format(r['sum_rate_bps_hz_mean'],'.3f') for r in rows))
            lines.append(arm+'/'+policy+' '+ ' '.join(parts))
        keys={}
        for r in results.get('paired',[]):
            if (r.get('metric')=='sum_rate_bps_hz' and r.get('estimand') in ('low_0_5_10','low_three','low3','all_six')
                    and ('dc2' in r.get('contrast','') or 'deep4' in r.get('contrast',''))):
                keys.setdefault(r['contrast'],[]).append(r)
        priorities=[layout+'_'+variant+'_minus_base2_'+policy+'_'+mode
            for layout,policy in (('fdma','fdma'),('g4','hybrid_1_3'),('g4','hybrid_1_1'))
            for variant in ('dc2','deep4') for mode in ('fixed','tuned')]
        chosen=[name for name in priorities if name in keys]
        # Two layouts, both variants and both hybrid ratios; other modes in details.
        chosen=chosen[:4]+[n for n in chosen[4:] if n.endswith('_fixed')][:4]
        for contrast in chosen:
            rows=keys[contrast]
            lines.append(contrast+' '+ ';'.join(str(r.get('rzf_mode',''))+'/'+str(r['estimand'])+':'+
                f"{r['sample_mean']:+.3f}[{r['batch_ci95_low']:+.3f},{r['batch_ci95_high']:+.3f}]" for r in rows[:4]))
    else:lines.append('final six-point results / paired batch CI=unknown')
    lines+=['package='+str(summary.get('result_package','unknown')),
        'summary_path='+str(summary.get('summary_path','unknown'))+'; --details for full monitors/per-user/paired tables']
    # Failures can contain arbitrarily long external messages; never flood terminal.
    if len(lines)>40:lines=lines[:38]+lines[-2:]
    budget=8000
    rendered=[]
    for line in lines:
        line=line[:650];size=len((line+'\n').encode('utf-8'))
        if sum(len((v+'\n').encode('utf-8')) for v in rendered)+size>budget:
            line='[remaining details in batch_summary.json; use --summary --details]'
            rendered.append(line);break
        rendered.append(line)
    return rendered


def print_summary(details=False):
    pointer=cfg.OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; real-source/CUDA acceptance pending');return 0
    try:folder=cfg.inside_root(read_json(pointer)['directory']);summary=read_json(folder/'batch_summary.json')
    except (OSError,ValueError,KeyError,TypeError) as error:
        print('status=unknown; unreadable batch pointer/summary: '+str(error));return 0
    summary['summary_path']=str(folder/'batch_summary.json')
    training={}
    for arm in cfg.ARMS:
        path=cfg.OUTPUT_ROOT/'training'/arm/'manifest.json'
        if path.is_file():
            try:training[arm]=read_json(path)
            except (OSError,ValueError) as error:training[arm]=dict(status='unknown',error='manifest unreadable: '+str(error))
    output=summary.get('validation',{}).get('directory');results=None
    if output:
        try:
            output=cfg.inside_root(output)
            if (output/'results.json').is_file():results=read_json(output/'results.json')
        except (OSError,ValueError) as error:summary['error']=str(summary.get('error',''))+'; results unreadable: '+str(error)
    selection=None
    if cfg.SELECTION_PATH.is_file():
        try:selection=read_json(cfg.SELECTION_PATH)
        except (OSError,ValueError) as error:
            summary['error']=(str(summary.get('error',''))+'; alpha lock unreadable: '+str(error))
    if details:
        monitors={a:{s:read_json(e['path']) if Path(e['path']).is_file() else dict(error='missing')
            for s,e in r.get('monitor_records',{}).items()} for a,r in training.items()}
        print(json.dumps(dict(summary=summary,training=training,monitors=monitors,selection=selection,results=results),indent=2))
    else:print('\n'.join(summary_lines(summary,training,results,selection)))
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
