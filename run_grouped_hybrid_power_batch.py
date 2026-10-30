#!/usr/bin/env python3
"""Explicit eight-run power screening batch; historical runs are read-only inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import uuid
import zipfile

from as_receiver_batch_support import safe_inside, save_json, sha256
from run_as_receiver_batch import _child, _run_stage, now, read_json
from train_proposal_as_receiver import run_lock
from train_grouped_codec import _json
from configs.grouped_hybrid_power_train_config import ARMS, SEEDS, OUTPUT_ROOT, power_config, source_identity

ROOT = Path(__file__).resolve().parent
BATCH_ROOT = OUTPUT_ROOT / 'batches'
EVAL_ROOT = OUTPUT_ROOT / 'evaluation'
OLD_BATCH = ROOT / 'runs_as_screen/grouped_hybrid_anchor_v1/batches/batch_cf66957ea2914720bbdd9ed15dbf3884'
OLD_BASE = '266d9e52428d136f9de5d44a95da4afeea812fb6'
OLD_REPAIR = 'ad4e801612c12fbe70a72eb6fde30938161ce47f'


def source_record():
    from train_grouped_hybrid_power import SOURCE_FILES as training
    from evaluate_grouped_hybrid_power import SOURCE_FILES as evaluation
    names = set(training) | set(evaluation) | {
        'run_grouped_hybrid_power_batch.py','preflight_grouped_hybrid_power.py',
        'grouped_hybrid_power_results.py','run_grouped_hybrid_anchor_batch.py',
        'grouped_hybrid_resume_compat.py','run_as_receiver_batch.py','as_receiver_batch_support.py'}
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sha256={name:sha256(ROOT/name) for name in sorted(names)})


def plan():
    from evaluate_grouped_hybrid_power import protocol, CONTRASTS
    return dict(batch='grouped_hybrid_power_v1', old_batch=str(OLD_BATCH), old_batch_read_only=True,
        training=[dict(arm=arm,seed=seed,total_new_updates=20000,included_smoke=500,
            source_prior_offset=20000,latest_prior_count=40000,best=power_config(arm,seed).BEST_PATH)
            for seed in SEEDS for arm in ARMS], total_new_official_updates=160000,
        sources={str(s):{a:{k:v for k,v in source_identity(a,s).items()
            if k!='complete_source_configuration'} for a in ('fdma_control','h32_control')} for s in SEEDS},
        evaluation={str(s):protocol(s) for s in SEEDS}, contrasts=CONTRASTS,
        primary_reference='continued fdma_control, not frozen source',
        engineering_rule=dict(low3_mean_min=.5,low3_batch_ci95_lower_gt=0,gamma25_batch_ci95_lower_min=-.1),
        adaptive_claim_requires='gamma gate versus independently trained static gate',
        old_runs_retrained=False,cleanup=False,visits_08=False,manuscript_modified=False,
        prior_update='source20000+local step, only after successful optimizer; validation/test frozen',
        automatic_budget_extension=False,training_seeds_pooled=False)


def environment():
    from run_grouped_codec_batch import _environment
    result=_environment();result['source']=source_record();return result


def _same(a,b,label):
    if _json(a)!=_json(b):raise ValueError(label+' changed')


def _verify_git_map(commit,digests):
    if commit not in (OLD_BASE,OLD_REPAIR) or not isinstance(digests,dict) or not digests:
        raise ValueError('Unknown historical source identity')
    for name,digest in digests.items():
        safe_inside(ROOT/name,ROOT)
        content=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Historical Git blob differs: '+name)


def inspect_old_batch(summary,folder=OLD_BATCH):
    """Read-only small-evidence boundary; no discovery or re-running old stages."""
    from configs.grouped_hybrid_anchor_train_config import ARMS as old_arms
    folder=safe_inside(folder,ROOT)
    if folder!=OLD_BATCH.absolute():raise ValueError('Only the fixed completed anchor batch is a source')
    if summary.get('status')!='completed' or summary.get('error') or summary.get('package_error'):
        raise ValueError('Old anchor batch must be completed without a current error')
    expected={'environment','sources','cuda_preflight','results'} | {
        f'train_{arm}_{seed}' for seed in SEEDS for arm in old_arms} | {f'evaluate_{s}' for s in SEEDS}
    if set(summary.get('stages',{}))!=expected:raise ValueError('Old batch stages incomplete/unexpected')
    if set(summary.get('training',{}))!={str(s) for s in SEEDS}:raise ValueError('Two old seeds required')
    frozen={str(folder/'batch_summary.json'):sha256(folder/'batch_summary.json')}
    for name,stage in summary['stages'].items():
        if stage.get('status')!='completed' or not stage.get('evidence_sha256'):
            raise ValueError('Old completed-stage evidence missing: '+name)
        for filename,digest in stage['evidence_sha256'].items():
            path=safe_inside(filename,ROOT)
            if str(path) in frozen and frozen[str(path)]!=digest:raise ValueError('Conflicting old evidence digest')
            frozen[str(path)]=digest
    for seed in SEEDS:
        records=summary['training'][str(seed)]
        if set(records)!=set(old_arms):raise ValueError('Old six-run training records incomplete')
        for arm,record in records.items():
            for key in ('completed_updates','best_step','best_prior_update_count','latest_prior_update_count'):
                if type(record.get(key)) is not int or record[key]!=20000:
                    raise ValueError(f'Old {arm}/{seed} {key} is not20000')
            if record.get('arm')!=arm or record.get('training_seed')!=seed:raise ValueError('Old run identity mismatch')
            stage=summary['stages'][f'train_{arm}_{seed}']['result']['report']
            _same(record,stage,'Old summary/stage training record')
    package=safe_inside(summary['result_package'],folder)
    frozen[str(package)]=summary['result_package_sha256']
    for filename,digest in frozen.items():
        if sha256(filename)!=digest:raise ValueError('Old source evidence changed: '+filename)
    return frozen


def verify_old_version_chain(summary):
    """Validate archived 266->ad4 provenance without invoking its old HEAD-only resumer."""
    source=summary['source'];_verify_git_map(source['commit'],source['sha256'])
    if source['commit']!=OLD_BASE:raise ValueError('Wrong original anchor batch commit')
    receipt=summary.get('resume_compatibility',{})
    if receipt.get('version_chain')!=[OLD_BASE,OLD_REPAIR]:raise ValueError('Missing exact completed-batch repair chain')
    directory=OLD_BATCH/'resume_compat_266d9e5'
    path=safe_inside(receipt['record_path'],OLD_BATCH)
    if path!=directory/'record.json':raise ValueError('Wrong historical compatibility record path')
    seal=safe_inside(directory/'record.sha256',OLD_BATCH)
    snapshot=safe_inside(directory/'batch_summary.before.json',OLD_BATCH)
    if sha256(path)!=receipt.get('record_sha256') or seal.read_text(encoding='ascii').strip()!=sha256(path):
        raise ValueError('Archived compatibility record changed')
    record=read_json(path);release=record['release']
    if release.get('old_commit')!=OLD_BASE or release.get('repair_commit')!=OLD_REPAIR:
        raise ValueError('Unexpected historical repair release')
    for key,commit in (('old_controller_sha256',OLD_BASE),('new_controller_sha256',OLD_REPAIR),
                       ('old_trainer_sha256',OLD_BASE),('new_trainer_sha256',OLD_REPAIR),
                       ('release_files_sha256',OLD_REPAIR)):
        _verify_git_map(commit,release[key])
    if sha256(snapshot)!=record.get('snapshot_sha256') or sha256(snapshot)!=receipt.get('snapshot_sha256'):
        raise ValueError('Historical pre-repair snapshot changed')
    _same(source,read_json(snapshot)['source'],'Historical snapshot source')
    frozen={str(path):sha256(path),str(seal):sha256(seal),str(snapshot):sha256(snapshot)}
    for filename,digest in record['immutable_files'].items():
        path=safe_inside(filename,ROOT)
        if sha256(path)!=digest:raise ValueError('Historical immutable evidence changed: '+filename)
        frozen[str(path)]=digest
    # The registered500-step active digests are intentionally NOT treated as
    # current20k hashes. Completed checkpoints are verified below, independently.
    return frozen


def source_evidence():
    import torch
    from configs.grouped_hybrid_anchor_train_config import ARMS as old_arms
    from run_grouped_hybrid_anchor_batch import training_record as old_training_record, verify_evaluation as old_eval
    from train_grouped_hybrid_power import load_source
    summary=read_json(OLD_BATCH/'batch_summary.json')
    frozen=inspect_old_batch(summary);frozen.update(verify_old_version_chain(summary))
    for seed in SEEDS:
        for arm in old_arms:
            actual=old_training_record(arm,seed)
            saved=summary['training'][str(seed)][arm]
            _same(actual,{k:saved[k] for k in actual},'Actual old completed checkpoint record')
        directory=summary['evaluations'][str(seed)]
        _same(directory,summary['stages'][f'evaluate_{seed}']['result']['output_directory'],'Old evaluation pointer')
        for path in old_eval(directory,seed):frozen[str(path)]=sha256(path)
    records={}
    for seed in SEEDS:
        records[str(seed)]={}
        for arm in ('fdma_control','h32_control'):
            payload,record=load_source(torch,arm,seed);del payload
            records[str(seed)][arm]=record
            old_arm='fdma_control' if arm=='fdma_control' else 'hybrid_f32_a64'
            saved=summary['training'][str(seed)][old_arm]
            for name in ('best_path','latest_path','best_sha256','latest_sha256','best_step'):
                _same(record[name],saved[name],'Pinned source/old completed record '+name)
            for name in ('best','latest','config','status'):
                frozen[record[name+'_path']]=record[name+'_sha256']
    for path,digest in frozen.items():
        if sha256(safe_inside(path,ROOT))!=digest:raise ValueError('Old evidence changed while inspected')
    return dict(records=records,old_batch=str(OLD_BATCH),old_training_runs_verified=6,
        old_evaluations_verified=2,old_weights_retrained=False,frozen_files=frozen,read_only=True)


def training_record(arm,seed):
    import torch
    from train_grouped_hybrid_power import verify_training_metadata,build_model,validate_runtime_resume
    from evaluate_proposal_as_direct import check_tensor_state
    cfg=power_config(arm,seed);initial=Path(cfg.LOG_DIR)/'initialization.json'
    frozen={str(p):sha256(p) for p in (cfg.BEST_PATH,cfg.SAVE_PATH,cfg.CONFIG_PATH,cfg.STATUS_PATH,cfg.METRICS_PATH,initial)}
    best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
    latest=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
    selection=verify_training_metadata(arm,seed,best,latest,dict(config=read_json(cfg.CONFIG_PATH),status=read_json(cfg.STATUS_PATH)))
    model,actual_cfg=build_model(torch,arm,seed,best,torch.device('cpu'))
    parameters=dict(full=sum(p.numel() for p in model.parameters()),ue=sum(p.numel() for p in model.ue_parameters()),
        bs=sum(p.numel() for p in model.bs_parameters()),gate=sum(p.numel() for p in model.power_gate.parameters()))
    gate={str(g):float(model.gate_log_ratio(g).detach().item()) for g in (0,5,10,15,20,25)}
    check_tensor_state(torch,latest['state'],best['state'],equal=best['step']==latest['step'])
    validate_runtime_resume(torch,model,latest,actual_cfg,allow_completed=True)
    if any(sha256(p)!=d for p,d in frozen.items()):raise ValueError('New training evidence changed during inspection')
    return dict(arm=arm,training_seed=seed,completed_updates=20000,best_step=best['step'],best_metric=best['best_metric'],
        best_path=cfg.BEST_PATH,latest_path=cfg.SAVE_PATH,best_sha256=frozen[cfg.BEST_PATH],latest_sha256=frozen[cfg.SAVE_PATH],
        metrics_path=cfg.METRICS_PATH,metrics_sha256=frozen[cfg.METRICS_PATH],verified_selection=selection,
        initialization_path=str(initial),initialization_sha256=frozen[str(initial)],
        best_prior_update_count=int(best['state']['power_prior.update_count'].item()),
        latest_prior_update_count=int(latest['state']['power_prior.update_count'].item()),
        source_prior_offset=20000,parameters=parameters,selected_gate_log_energy_ratio=gate)


def stage_child(folder,name,arm=None,seed=None):
    output=folder/(name+'_'+uuid.uuid4().hex[:8]+'.json')
    command=[sys.executable,'-B','-u','run_grouped_hybrid_power_batch.py','--_stage',name,'--_output',str(output)]
    if arm is not None:command+=['--_arm',arm,'--_seed',str(seed)]
    log=_child(folder,name,command);report=read_json(output)
    if report.get('passed') is not True:raise ValueError('Stage failed: '+str(output))
    evidence=[output,log]
    for path,digest in report.get('frozen_files',{}).items():
        path=safe_inside(path,ROOT)
        if sha256(path)!=digest:raise ValueError('Frozen source changed')
        evidence.append(path)
    return dict(report=report,_evidence=evidence)


def preflight(folder):
    output=folder/('cuda_batch48_'+uuid.uuid4().hex[:8]);retained=folder/(output.name+'.json')
    try:log=_child(folder,'cuda_power_batch48',[sys.executable,'-B','-u','preflight_grouped_hybrid_power.py','--run','--output-dir',str(output)])
    finally:
        if (output/'report.json').is_file():save_json(retained,read_json(output/'report.json'))
    report=read_json(output/'report.json')
    if report.get('passed') is not True:raise ValueError('Power CUDA preflight failed; no training starts')
    return dict(report=report,_evidence=[log,retained,output/'report.json'])


def train_one(folder,arm,seed):
    cfg=power_config(arm,seed);logs=[]
    command=[sys.executable,'-B','-u','train_grouped_hybrid_power.py','--arm',arm,'--seed',str(seed)]
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():raise ValueError('New output exists without latest; preserve and stop')
        logs.append(_child(folder,f'{arm}_{seed}_first500',command+['--stop-after','500']))
    if read_json(cfg.STATUS_PATH).get('state')!='completed':
        logs.append(_child(folder,f'{arm}_{seed}_continue',command+['--resume','--train']))
    result=stage_child(folder,'training_record',arm,seed)
    result['_evidence']+=logs+[cfg.BEST_PATH,cfg.SAVE_PATH,cfg.CONFIG_PATH,cfg.STATUS_PATH,cfg.METRICS_PATH,Path(cfg.LOG_DIR)/'initialization.json']
    return result


def verify_evaluation(output,seed):
    from evaluate_grouped_hybrid_power import protocol,artifact_hashes
    output=safe_inside(output,EVAL_ROOT);manifest=read_json(output/'manifest.json')
    _same(manifest.get('protocol'),protocol(seed),'New evaluation protocol')
    if (manifest.get('status')!='completed' or manifest.get('results_verified') is not True
            or manifest.get('freeze_checks',{}).get('passed') is not True
            or manifest.get('shared_channel_noise_inputs_unchanged') is not True):raise ValueError('Incomplete evaluation acceptance')
    inputs=manifest.get('input_sha256_before',{});code=manifest.get('runtime',{}).get('source_sha256',{})
    if not inputs or inputs!=manifest.get('input_sha256_after') or not code:raise ValueError('Missing input/source freeze evidence')
    for path,digest in inputs.items():
        if sha256(safe_inside(path,ROOT))!=digest:raise ValueError('Evaluation input changed')
    for name,digest in code.items():
        if sha256(safe_inside(ROOT/name,ROOT))!=digest:raise ValueError('Evaluation code changed')
    if artifact_hashes(output)!=manifest.get('output_sha256'):raise ValueError('Evaluation artifact changed')
    return [output/'manifest.json']+[output/n for n in manifest['output_sha256']]+list(inputs)+[ROOT/n for n in code]


def evaluate_one(folder,seed):
    parent=EVAL_ROOT/'formal'/folder.name/f'seed_{seed}';pointer=folder/f'evaluation_seed{seed}.json'
    if pointer.exists():
        previous=safe_inside(read_json(pointer)['directory'],parent)
        if (previous/'manifest.json').is_file() and read_json(previous/'manifest.json').get('status')=='completed':
            return dict(output_directory=str(previous),reused=True,_evidence=verify_evaluation(previous,seed)+[pointer])
    output=parent/('attempt_'+uuid.uuid4().hex[:8]);save_json(pointer,dict(directory=str(output)),replace=pointer.exists())
    retained=folder/f'evaluation_{seed}_{output.name}.json'
    try:log=_child(folder,f'evaluate_{seed}',[sys.executable,'-B','-u','evaluate_grouped_hybrid_power.py','--seed',str(seed),'--evaluate','--output-dir',str(output)])
    finally:
        if (output/'manifest.json').is_file():save_json(retained,read_json(output/'manifest.json'))
    return dict(output_directory=str(output),_evidence=verify_evaluation(output,seed)+[pointer,retained,log])


def package_small(folder,summary):
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
        raise ValueError('Small package size limit exceeded')
    expected={p.relative_to(folder).as_posix():sha256(p) for p in selected}
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):archive.write(path,path.relative_to(folder).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        if set(archive.namelist())!=set(expected) or len(archive.namelist())!=len(expected):raise ValueError('Package member mismatch')
        for name,digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Package bytes mismatch')
    if any(sha256(folder/name)!=digest for name,digest in expected.items()):raise ValueError('Package input changed')
    summary.update(result_package=str(bundle),result_package_sha256=sha256(bundle),
                   package_note='Archive contains pre-package-hash summary; live summary stores archive digest')
    save_json(folder/'batch_summary.json',summary,replace=True)


def run_batch(resume=False):
    if os.name!='posix':raise RuntimeError('Explicit CUDA server batch only')
    owner=SimpleNamespace(EXP_NAME='grouped_hybrid_power_batch_v1',arm='controller',SEED=0)
    with run_lock(owner,resume=resume,allowed_root=OUTPUT_ROOT):
        pointer=OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=safe_inside(read_json(pointer)['directory'],BATCH_ROOT);summary=read_json(folder/'batch_summary.json')
            _same(summary['source'],source_record(),'Batch code identity')
        else:
            if pointer.exists():raise FileExistsError('Existing batch: explicit --resume required')
            for seed in SEEDS:
                for arm in ARMS:
                    cfg=power_config(arm,seed)
                    if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():raise FileExistsError('Unowned new training output exists')
            folder=safe_inside(BATCH_ROOT/('batch_'+uuid.uuid4().hex),BATCH_ROOT);folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),source=source_record(),plan=plan(),stages={},training={},cleanup='none')
            save_json(folder/'batch_summary.json',summary);save_json(pointer,dict(directory=str(folder)))
        previous={key:summary.pop(key) for key in ('error','traceback','package_error') if key in summary}
        if previous:summary.setdefault('failure_history',[]).append(dict(time=now(),**previous))
        summary['status']='running'
        try:
            env=_run_stage(summary,folder,'environment',lambda:stage_child(folder,'environment'))['report']
            if resume:
                current=environment()
                for key in ('torch','sionna','matplotlib','channel_backend','cuda'):_same(env[key],current[key],'Resume environment '+key)
            evidence=_run_stage(summary,folder,'sources',lambda:stage_child(folder,'sources'))['report']
            summary['sources']=evidence['records']
            summary['preflight']=_run_stage(summary,folder,'cuda_preflight',lambda:preflight(folder))['report']
            for seed in SEEDS:
                for arm in ARMS:
                    item=_run_stage(summary,folder,f'train_{arm}_{seed}',lambda arm=arm,seed=seed:train_one(folder,arm,seed))['report']
                    summary['training'].setdefault(str(seed),{})[arm]=item
            evaluations={str(s):_run_stage(summary,folder,f'evaluate_{s}',lambda s=s:evaluate_one(folder,s))['output_directory'] for s in SEEDS}
            summary['evaluations']=evaluations
            from grouped_hybrid_power_results import organize_results
            summary['results']=_run_stage(summary,folder,'results',lambda:organize_results(folder,evaluations,summary['training'],summary['sources']))
            _run_stage(summary,folder,'sources',lambda:stage_child(folder,'sources'))
            summary.update(status='completed',finished=now())
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),
                           resume_command='python -B -u run_grouped_hybrid_power_batch.py --run --resume')
            raise
        finally:
            try:package_small(folder,summary)
            except BaseException as error:
                summary.update(status='failed',package_error=str(error),return_files=[str(folder/'batch_summary.json')])
                save_json(folder/'batch_summary.json',summary,replace=True);raise
    return folder


def print_summary():
    from run_grouped_hybrid_anchor_batch import training_progress
    pointer=OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():print('not_started; no results inferred');return 0
    folder=safe_inside(read_json(pointer)['directory'],BATCH_ROOT);summary=read_json(folder/'batch_summary.json')
    stages={k:v['status'] for k,v in summary.get('stages',{}).items()}
    print(json.dumps(dict(status=summary['status'],error=summary.get('error'),package_error=summary.get('package_error'),
        stages=stages,current_stage=[k for k,v in stages.items() if v in ('running','failed')],
        summary_path=str(folder/'batch_summary.json'),result_package=summary.get('result_package'),
        result_package_sha256=summary.get('result_package_sha256'),return_files=summary.get('return_files',[]),weights_read=False),indent=2))
    source_acceptance=summary.get('stages',{}).get('sources',{}).get('result',{}).get('report',{})
    print('source_acceptance '+json.dumps({k:source_acceptance.get(k,'unknown') for k in (
        'passed','old_batch','old_training_runs_verified','old_evaluations_verified','old_weights_retrained','read_only')},sort_keys=True))
    for seed in SEEDS:
        for arm in ARMS:
            progress=training_progress(arm,seed,cfg=power_config(arm,seed))
            progress['expected_cumulative_prior_if_committed']=None if progress['latest_step'] is None else 20000+progress['latest_step']
            print('live '+json.dumps(progress,sort_keys=True))
            record=summary.get('training',{}).get(str(seed),{}).get(arm)
            if record:print('verified_training '+json.dumps({k:record[k] for k in ('arm','training_seed','completed_updates','best_step','best_metric',
                'best_sha256','latest_sha256','best_prior_update_count','latest_prior_update_count','parameters','selected_gate_log_energy_ratio')}))
    results=summary.get('results',{});per_seed=results.get('per_seed',{})
    if not per_seed:print('rates / paired CI / gates / energies / engineering decisions: unknown (not yet evaluated)')
    from evaluate_grouped_hybrid_power import METHODS,GAMMAS
    for seed,record in per_seed.items():
        print('RESULT seed='+seed+' engineering='+json.dumps(record['engineering_criteria'],sort_keys=True))
        print('gamma='+str(list(GAMMAS)))
        for method in METHODS:
            rows=sorted((r for r in record['all_rates'] if r['method_key']==method),key=lambda r:float(r['gamma_db']))
            print(method+' rates: '+' '.join(f"{float(r['sum_rate_mean']):.6f}" for r in rows))
            for row in rows:
                keys=['gamma_db']+[k for k in row if ('energy' in k or 'gate' in k or 'g_log' in k) and k.endswith('_mean')]
                print(' energy/gate '+method+' '+json.dumps({k:row[k] for k in keys},sort_keys=True))
        for name,comparison in record['comparisons'].items():
            print(name+' delta / sample95CI / batch95CI')
            pairs=list(zip(map(str,GAMMAS),comparison['per_gamma']))+[('low0/5/10',comparison['low_gamma']),('all6',comparison['all_gamma'])]
            for label,row in pairs:
                print(f" {label}: {float(row['delta_mean']):+.6f} sample[{float(row['delta_ci95_low']):+.6f},{float(row['delta_ci95_high']):+.6f}] "
                      f"batch[{float(row['batch_delta_ci95_low']):+.6f},{float(row['batch_delta_ci95_high']):+.6f}]")
    for seed,records in results.get('training_trends',{}).get('per_seed',{}).items():
        for arm,trend in records.items():
            print('training_trend '+json.dumps(dict(seed=seed,arm=arm,**{k:trend[k] for k in (
                'best_step','best_mean_all','last_five_validation_points','last_5000_updates','convergence_established')}),sort_keys=True))
    print('gamma is transmit-energy/N0, not historical received FB SNR. Primary reference=continued FDMA. Seeds separate; negative results retained.')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','run','summary'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--_stage',choices=('environment','sources','training_record'),help=argparse.SUPPRESS)
    parser.add_argument('--_output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--_arm',choices=ARMS,help=argparse.SUPPRESS);parser.add_argument('--_seed',type=int,choices=SEEDS,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._stage:
        if args._output is None or args.run or args.resume or args.dry_run or args.summary:parser.error('Internal stage requires only output/identity')
        if args._stage=='training_record' and (args._arm is None or args._seed is None):parser.error('Training identity required')
        output=safe_inside(args._output,BATCH_ROOT)
        if output.exists():raise FileExistsError(output)
        try:save_json(output,dict(environment() if args._stage=='environment' else source_evidence() if args._stage=='sources' else training_record(args._arm,args._seed),passed=True))
        except BaseException as error:
            if not output.exists():save_json(output,dict(passed=False,error=str(error),traceback=traceback.format_exc()))
            raise
    elif args.resume and not args.run:parser.error('--resume requires --run')
    elif args.dry_run:print(json.dumps(plan(),indent=2))
    elif args.summary:return print_summary()
    elif args.run:run_batch(args.resume)
    else:parser.print_help()
    return 0


if __name__=='__main__':raise SystemExit(main())
