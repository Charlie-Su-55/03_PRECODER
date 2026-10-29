#!/usr/bin/env python3
"""One explicit server batch: three hybrid arms x two seeds, without historical retraining."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import uuid
import zipfile

from as_receiver_batch_support import safe_inside,save_json,sha256
from run_as_receiver_batch import _child,_run_stage,now,read_json
from train_proposal_as_receiver import run_lock
from configs.grouped_hybrid_anchor_train_config import ARMS,SEEDS,OUTPUT_ROOT,anchor_config,source_identity

ROOT=Path(__file__).resolve().parent
BATCH_ROOT=OUTPUT_ROOT/'batches'
EVAL_ROOT=OUTPUT_ROOT/'evaluation'


def source_record():
    from train_grouped_hybrid_anchor import SOURCE_FILES as training
    from evaluate_grouped_hybrid_anchor import SOURCE_FILES as evaluation
    files=tuple(dict.fromkeys(training+evaluation+('run_grouped_hybrid_anchor_batch.py',
        'preflight_grouped_hybrid_anchor.py','grouped_hybrid_anchor_results.py',
        'run_as_receiver_batch.py','as_receiver_batch_support.py',
        'grouped_codec_fdma_results.py','grouped_codec_results.py','as_receiver_results.py')))
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={name:sha256(ROOT/name) for name in files})


def plan():
    from evaluate_grouped_hybrid_anchor import protocol,CONTRASTS
    from preflight_grouped_hybrid_anchor import plan as preflight_plan
    return dict(batch='grouped_hybrid_anchor_v1',training=[dict(arm=arm,seed=seed,
        total_new_updates=20000,included_smoke=500,best=anchor_config(arm,seed).BEST_PATH)
        for seed in SEEDS for arm in ARMS],total_new_official_updates=120000,
        sources={str(seed):source_identity(seed) for seed in SEEDS},preflight=preflight_plan(),
        evaluation={str(seed):protocol(seed) for seed in SEEDS},contrasts=CONTRASTS,
        main_reference='new fdma_control, not frozen source F',
        engineering_rule=dict(low3_mean_min=.5,low3_batch_ci95_lower_gt=0,gamma25_batch_ci95_lower_min=-.1),
        old_runs_retrained=False,cleanup=False,visits_08=False,manuscript_modified=False,
        prior_update='only after optimizer succeeds; step0/validation/evaluation read only',
        adaptive_search=False,training_seeds_pooled=False)


def environment():
    from run_grouped_codec_batch import _environment
    result=_environment();result['source']=source_record();return result


def source_evidence():
    import torch
    from train_grouped_hybrid_anchor import load_source
    records={};frozen={}
    for seed in SEEDS:
        payload,record=load_source(torch,seed);del payload
        records[str(seed)]=record
        for name in ('best','latest','config','status'):
            path=record[name+'_path'];digest=record[name+'_sha256']
            if sha256(path)!=digest:raise ValueError('Source evidence changed '+path)
            frozen[path]=digest
    return dict(records=records,frozen_files=frozen,read_only=True)


def training_record(arm,seed):
    import torch
    from train_grouped_hybrid_anchor import verify_training_metadata,build_model,validate_runtime_resume
    from evaluate_proposal_as_direct import check_tensor_state
    cfg=anchor_config(arm,seed)
    initialization=Path(cfg.LOG_DIR)/'initialization.json'
    frozen={path:sha256(path) for path in (cfg.BEST_PATH,cfg.SAVE_PATH,cfg.CONFIG_PATH,cfg.STATUS_PATH,cfg.METRICS_PATH,str(initialization))}
    best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
    latest=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
    selection=verify_training_metadata(arm,seed,best,latest,
        dict(config=read_json(cfg.CONFIG_PATH),status=read_json(cfg.STATUS_PATH)))
    model,actual_cfg=build_model(torch,arm,seed,best,torch.device('cpu'))
    check_tensor_state(torch,latest['state'],best['state'],equal=best['step']==latest['step'])
    validate_runtime_resume(torch,model,latest,actual_cfg,allow_completed=True)
    if any(sha256(path)!=digest for path,digest in frozen.items()):raise ValueError('Training evidence changed during inspection')
    return dict(arm=arm,training_seed=seed,completed_updates=latest['step'],best_step=best['step'],
        best_metric=best['best_metric'],best_path=cfg.BEST_PATH,latest_path=cfg.SAVE_PATH,
        best_sha256=frozen[cfg.BEST_PATH],latest_sha256=frozen[cfg.SAVE_PATH],
        metrics_path=cfg.METRICS_PATH,metrics_sha256=frozen[cfg.METRICS_PATH],verified_selection=selection,
        initialization_path=str(initialization),initialization_sha256=frozen[str(initialization)],
        best_prior_update_count=int(best['state']['power_prior.update_count'].item()),
        latest_prior_update_count=int(latest['state']['power_prior.update_count'].item()))


def stage_child(folder,name,arm=None,seed=None):
    output=folder/(name+'_'+uuid.uuid4().hex[:8]+'.json')
    command=[sys.executable,'-B','-u','run_grouped_hybrid_anchor_batch.py','--_stage',name,'--_output',str(output)]
    if arm is not None:command+=['--_arm',arm,'--_seed',str(seed)]
    log=_child(folder,name,command);report=read_json(output)
    if report.get('passed') is not True:raise ValueError('Stage did not pass '+str(output))
    evidence=[output,log]
    for path,digest in report.get('frozen_files',{}).items():
        path=safe_inside(path,ROOT)
        if sha256(path)!=digest:raise ValueError('Frozen input changed')
        evidence.append(path)
    return dict(report=report,report_path=str(output),_evidence=evidence)


def preflight(folder):
    output=folder/('cuda_batch48_'+uuid.uuid4().hex[:8])
    retained=folder/(output.name+'.json')
    try:log=_child(folder,'cuda_hybrid_batch48',[sys.executable,'-B','-u','preflight_grouped_hybrid_anchor.py','--run','--output-dir',str(output)])
    finally:
        if (output/'report.json').is_file():save_json(retained,read_json(output/'report.json'))
    report=read_json(output/'report.json')
    if report.get('passed') is not True:raise ValueError('New hybrid preflight failed; no training authorized by controller')
    return dict(report=report,_evidence=[log,retained,output/'report.json'])


def train_one(folder,arm,seed,compat_path=None):
    cfg=anchor_config(arm,seed);logs=[]
    command=[sys.executable,'-B','-u','train_grouped_hybrid_anchor.py','--arm',arm,'--seed',str(seed)]
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():raise ValueError('Output exists without latest; preserve and investigate')
        logs.append(_child(folder,f'{arm}_{seed}_first500',command+['--stop-after','500']))
    if read_json(cfg.STATUS_PATH).get('state')!='completed':
        continuation=command+['--resume','--train']
        if compat_path is not None:continuation+=['--resume-compat',str(compat_path)]
        logs.append(_child(folder,f'{arm}_{seed}_continue',continuation))
    result=stage_child(folder,'training_record',arm,seed)
    result['_evidence']+=logs+[cfg.BEST_PATH,cfg.SAVE_PATH,cfg.STATUS_PATH,cfg.CONFIG_PATH,cfg.METRICS_PATH,Path(cfg.LOG_DIR)/'initialization.json']
    return result


def verify_evaluation(output,seed):
    from evaluate_grouped_hybrid_anchor import protocol,artifact_hashes
    output=safe_inside(output,EVAL_ROOT);manifest=read_json(output/'manifest.json')
    if manifest.get('status')!='completed' or manifest.get('protocol')!=protocol(seed):
        raise ValueError('Evaluation status/protocol mismatch')
    if manifest.get('freeze_checks',{}).get('passed') is not True or manifest.get('results_verified') is not True:
        raise ValueError('Evaluation numerical/freeze checks incomplete')
    if manifest.get('shared_channel_noise_inputs_unchanged') is not True:
        raise ValueError('Shared input identity not verified')
    inputs=manifest.get('input_sha256_before',{})
    code=manifest.get('runtime',{}).get('source_sha256',{})
    if not inputs or inputs!=manifest.get('input_sha256_after') or not code:raise ValueError('Missing immutable evaluation provenance')
    for path,digest in inputs.items():
        if sha256(safe_inside(path,ROOT))!=digest:raise ValueError('Evaluation input changed '+path)
    for name,digest in code.items():
        if sha256(safe_inside(ROOT/name,ROOT))!=digest:raise ValueError('Evaluation source changed '+name)
    if artifact_hashes(output)!=manifest.get('output_sha256'):raise ValueError('Evaluation artifact changed')
    return [output/'manifest.json']+[output/name for name in manifest['output_sha256']]+list(inputs)+[ROOT/name for name in code]


def evaluate_one(folder,seed):
    parent=EVAL_ROOT/folder.name/f'seed_{seed}';pointer=folder/f'evaluation_seed{seed}.json'
    if pointer.exists():
        previous=safe_inside(read_json(pointer)['directory'],parent)
        if (previous/'manifest.json').is_file() and read_json(previous/'manifest.json').get('status')=='completed':
            return dict(output_directory=str(previous),reused=True,_evidence=verify_evaluation(previous,seed)+[pointer])
    output=parent/('attempt_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)),replace=pointer.exists())
    retained=folder/f'evaluation_{seed}_{output.name}.json'
    try:log=_child(folder,f'evaluate_{seed}',[sys.executable,'-B','-u','evaluate_grouped_hybrid_anchor.py',
        '--seed',str(seed),'--evaluate','--output-dir',str(output)])
    finally:
        if (output/'manifest.json').is_file():save_json(retained,read_json(output/'manifest.json'))
    return dict(output_directory=str(output),_evidence=verify_evaluation(output,seed)+[pointer,retained,log])


def package_small(folder,summary):
    """No weights, raw samples, or terminal logs; explicit small report allowlist."""
    folder=safe_inside(folder,BATCH_ROOT);bundle=folder/('results_'+uuid.uuid4().hex+'.zip')
    summary['return_files']=[str(folder/'batch_summary.json'),str(bundle)]
    save_json(folder/'batch_summary.json',summary,replace=True)
    suffixes={'.csv','.json','.md','.pdf','.png'}
    selected={safe_inside(p,folder) for p in folder.iterdir() if p.is_file() and p.suffix in suffixes}
    for name,digest in summary.get('results',{}).get('files',{}).items():
        path=safe_inside(name,folder)
        if path.suffix in suffixes:
            if sha256(path)!=digest:raise ValueError('Small export changed')
            selected.add(path)
    if any(p.stat().st_size>32*1024**2 for p in selected) or sum(p.stat().st_size for p in selected)>128*1024**2:
        raise ValueError('Small result package size limit exceeded')
    expected={p.relative_to(folder).as_posix():sha256(p) for p in selected}
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):archive.write(path,path.relative_to(folder).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        if len(archive.namelist())!=len(expected) or set(archive.namelist())!=set(expected):raise ValueError('Package member mismatch')
        for name,digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Package content mismatch')
    if any(sha256(folder/name)!=digest for name,digest in expected.items()):raise ValueError('Package source changed')
    summary.update(result_package=str(bundle),result_package_sha256=sha256(bundle),
        package_note='Archive contains the pre-package-hash summary; live summary records archive SHA256')
    save_json(folder/'batch_summary.json',summary,replace=True)


def validate_batch_resume(folder,summary,current,compat_path=None):
    from grouped_hybrid_resume_compat import differences, validate_controller_transition
    if compat_path is not None:
        return validate_controller_transition(folder,summary,current,compat_path)
    if summary.get('resume_compatibility'):
        raise ValueError('This repaired batch requires its registered --resume-compat record on every resume')
    if summary['source']['sha256']!=current['sha256']:
        raise ValueError('Resume controller source-code SHA256 changed: '+
            json.dumps(differences(summary['source']['sha256'],current['sha256']),sort_keys=True))
    return None


def run_batch(resume=False,compat_path=None):
    if os.name!='posix':raise RuntimeError('Explicit server CUDA batch only')
    owner=SimpleNamespace(EXP_NAME='grouped_hybrid_anchor_batch_v1',arm='controller',SEED=0)
    with run_lock(owner,resume=resume,allowed_root=OUTPUT_ROOT):
        pointer=OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=safe_inside(read_json(pointer)['directory'],BATCH_ROOT);summary=read_json(folder/'batch_summary.json')
            receipt=validate_batch_resume(folder,summary,source_record(),compat_path)
            if receipt is not None:summary['resume_compatibility']=receipt
        else:
            if pointer.exists():raise FileExistsError('Existing batch; use explicit --resume')
            for seed in SEEDS:
                for arm in ARMS:
                    cfg=anchor_config(arm,seed)
                    if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():raise FileExistsError('Unowned training output exists')
            folder=safe_inside(BATCH_ROOT/('batch_'+uuid.uuid4().hex),BATCH_ROOT);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),source=source_record(),plan=plan(),stages={},training={},cleanup='none')
            save_json(folder/'batch_summary.json',summary);save_json(pointer,dict(directory=str(folder)))
        prior={key:summary.pop(key) for key in ('error','traceback','package_error') if key in summary}
        if prior:summary.setdefault('failure_history',[]).append(dict(time=now(),**prior))
        summary['status']='running'
        try:
            env=_run_stage(summary,folder,'environment',lambda:stage_child(folder,'environment'))['report']
            if resume and compat_path is None:
                current=stage_child(folder,'environment')['report']
                for name in ('torch','sionna','matplotlib','channel_backend','cuda'):
                    if current[name]!=env[name]:raise ValueError('Resume environment changed '+name)
            source=_run_stage(summary,folder,'sources',lambda:stage_child(folder,'sources'))['report']
            summary['sources']=source['records']
            summary['preflight']=_run_stage(summary,folder,'cuda_preflight',lambda:preflight(folder))['report']
            for seed in SEEDS:
                for arm in ARMS:
                    record=_run_stage(summary,folder,f'train_{arm}_{seed}',lambda arm=arm,seed=seed:train_one(folder,arm,seed,compat_path))['report']
                    summary['training'].setdefault(str(seed),{})[arm]=record
            evaluations={str(seed):_run_stage(summary,folder,f'evaluate_{seed}',lambda seed=seed:evaluate_one(folder,seed))['output_directory'] for seed in SEEDS}
            summary['evaluations']=evaluations
            from grouped_hybrid_anchor_results import organize_results
            summary['results']=_run_stage(summary,folder,'results',lambda:organize_results(folder,evaluations,summary['training'],summary['sources']))
            _run_stage(summary,folder,'sources',lambda:stage_child(folder,'sources'))
            summary.update(status='completed',finished=now())
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),
                resume_command='python -B -u run_grouped_hybrid_anchor_batch.py --run --resume'+
                    ('' if compat_path is None else ' --resume-compat '+str(compat_path)))
            raise
        finally:
            try:package_small(folder,summary)
            except BaseException as error:
                summary.update(status='failed',package_error=str(error),return_files=[str(folder/'batch_summary.json')])
                save_json(folder/'batch_summary.json',summary,replace=True);raise
    return folder


def training_progress(arm,seed,*,cfg=None):
    """Observe small sidecars only; checkpoint-internal facts require completed-stage verification."""
    cfg=anchor_config(arm,seed) if cfg is None else cfg
    result=dict(arm=arm,seed=seed,state='unknown',latest_step=None,best_step=None,mean_all=None,
        weights_read=False,last_record_utc=None,warnings=[])
    status=Path(cfg.STATUS_PATH);metrics=Path(cfg.METRICS_PATH)
    if status.is_file():
        value=read_json(status)
        if value.get('experiment')!=cfg.EXP_NAME:raise ValueError('Live status identity mismatch')
        result.update(state=value.get('state','unknown'),status_step=value.get('step'))
    if metrics.is_file():
        result['last_record_utc']=datetime.fromtimestamp(metrics.stat().st_mtime,timezone.utc).isoformat()
        with metrics.open(encoding='utf-8') as stream:
            for line in stream:
                try:row=json.loads(line)
                except ValueError:result['warnings'].append('Partial live metrics line');continue
                step=row.get('step')
                if type(step) is not int or not 0<=step<=20000:continue
                result['latest_step']=max(result['latest_step'] or 0,step)
                value=row.get('selection_metric')
                if (row.get('selection_metric_name')=='mean_all' and isinstance(value,(int,float))
                    and not isinstance(value,bool) and math.isfinite(value)
                    and (result['mean_all'] is None or value>result['mean_all'])):
                    result.update(best_step=step,mean_all=value)
    result['scope']='metrics/status observation, not a weight or historical-prior verification'
    return result


def print_summary():
    pointer=OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; no results inferred');return 0
    folder=safe_inside(read_json(pointer)['directory'],BATCH_ROOT);summary=read_json(folder/'batch_summary.json')
    stages={key:value['status'] for key,value in summary.get('stages',{}).items()}
    print(json.dumps(dict(status=summary['status'],error=summary.get('error'),package_error=summary.get('package_error'),
        stages=stages,current_stage=[key for key,value in stages.items() if value in ('running','failed')],
        resume_compatibility=summary.get('resume_compatibility'),
        summary_path=str(folder/'batch_summary.json'),
        source_hash_acceptance=stages.get('sources','unknown'),cuda_acceptance=stages.get('cuda_preflight','unknown'),
        result_package=summary.get('result_package'),result_package_sha256=summary.get('result_package_sha256'),
        return_files=summary.get('return_files',[]),weights_read=False),indent=2))
    for seed in SEEDS:
        for arm in ARMS:
            print('live '+json.dumps(training_progress(arm,seed),sort_keys=True))
            record=summary.get('training',{}).get(str(seed),{}).get(arm)
            if record:print('verified_training '+json.dumps({k:record[k] for k in ('arm','training_seed','completed_updates',
                'best_step','best_metric','best_sha256','latest_sha256','best_prior_update_count','latest_prior_update_count')}))
    results=summary.get('results',{}).get('per_seed',{})
    if not results:print('rates / paired sample+batch CI / engineering decisions: unknown (not yet audited)')
    from evaluate_grouped_hybrid_anchor import METHODS,GAMMAS
    for seed,record in results.items():
        print('RESULT seed='+seed+' engineering='+json.dumps(record['engineering_criteria'],sort_keys=True))
        print('gamma='+str(list(GAMMAS)))
        for method in METHODS:
            selected=sorted((r for r in record['all_rates'] if r['method_key']==method),key=lambda r:float(r['gamma_db']))
            print(method+': '+' '.join(f"{float(r['sum_rate_mean']):.6f}" for r in selected))
        for name,comparison in record['comparisons'].items():
            print(name+' delta / sample95CI / batch95CI')
            values=list(zip((str(g) for g in GAMMAS),comparison['per_gamma']))+[
                ('low0/5/10',comparison['low_gamma']),('all6',comparison['all_gamma'])]
            for label,row in values:
                print(f" {label}: {float(row['delta_mean']):+.6f} "
                    f"sample[{float(row['delta_ci95_low']):+.6f},{float(row['delta_ci95_high']):+.6f}] "
                    f"batch[{float(row['batch_delta_ci95_low']):+.6f},{float(row['batch_delta_ci95_high']):+.6f}]")
    print('gamma is absolute-energy/N0 protocol, not historical received feedback SNR; primary reference=new fdma_control. Two training seeds are separate.')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','run','summary'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-compat',type=Path,help='Exact registered 266d9e5 repair record')
    parser.add_argument('--_stage',choices=('environment','sources','training_record'),help=argparse.SUPPRESS)
    parser.add_argument('--_output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--_arm',choices=ARMS,help=argparse.SUPPRESS);parser.add_argument('--_seed',type=int,choices=SEEDS,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.resume_compat is not None and (not args.resume or not args.run):parser.error('--resume-compat requires --run --resume')
    if args._stage:
        if args._output is None or args.run or args.resume or args.dry_run or args.summary:parser.error('Internal stage requires only explicit output/identity')
        if args._stage=='training_record' and(args._arm is None or args._seed is None):parser.error('Training identity required')
        output=safe_inside(args._output,BATCH_ROOT)
        if output.exists():raise FileExistsError(output)
        try:
            result=environment() if args._stage=='environment' else source_evidence() if args._stage=='sources' else training_record(args._arm,args._seed)
            save_json(output,dict(result,passed=True))
        except BaseException as error:
            if not output.exists():save_json(output,dict(passed=False,error=str(error),traceback=traceback.format_exc()))
            raise
    elif args.resume and not args.run:parser.error('--resume requires --run')
    elif args.dry_run:print(json.dumps(plan(),indent=2))
    elif args.summary:return print_summary()
    elif args.run:run_batch(args.resume,args.resume_compat)
    else:parser.print_help()
    return 0


if __name__=='__main__':raise SystemExit(main())
