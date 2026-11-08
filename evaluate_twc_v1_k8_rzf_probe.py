#!/usr/bin/env python3
"""Frozen K8 RZF grid selection, independent validation and read-only trends."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import traceback

from configs import twc_v1_k8_rzf_probe_config as cfg
from configs import twc_v1_k8_task_loss_screen_config as source_cfg
import train_twc_v1_k8_task_loss_screen as source
from evaluate_twc_v1_k8_task_loss_screen import statistics_for,_tensor_sha,_state_sha,_snapshot,_write_csv

ROOT=cfg.REPO_ROOT
SOURCE_FILES=tuple(dict.fromkeys(source.SOURCE_FILES+(
    'configs/twc_v1_k8_rzf_probe_config.py','evaluate_twc_v1_k8_rzf_probe.py','run_twc_v1_k8_rzf_probe.py')))
POLICIES={'hybrid_1_3':(1,3),'hybrid_1_1':(1,1)}
METRICS=('dl_nmse','sum_rate_bps_hz','desired_signal_power','interference_power')
USER_RATES=tuple(f'user_rate_ue{k}_bps_hz' for k in range(8))
DIAGNOSTICS=('hhat_energy_total','hhat_fro_norm','hhat_to_true_norm_ratio','hhat_tone_user_energy_mean',
             'relative_alpha_mean','relative_alpha_min','relative_alpha_max',
             'data_energy_per_ue','frame_energy_per_ue','rzf_power_mean')
INVARIANT=('dl_nmse','hhat_energy_total','hhat_fro_norm','hhat_to_true_norm_ratio','hhat_tone_user_energy_mean')
ARTIFACTS={
    'calibration':('alpha_calibration.csv','alpha_global_scores.csv'),
    'evaluation':('summary.csv','samples.csv.gz','paired_deltas.csv','results.json','trends.json',
                  'training_by_condition.csv','fixed_monitor.csv')}


def now():return datetime.now(timezone.utc).isoformat()
def canonical(value):return cfg.normalize_json(value)
def sha256(path):return source.sha256(path)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_json(path,value):return source.save_json(path,value)


def source_code_record():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})


@lru_cache(maxsize=1)
def archived_source_hashes():
    return {name:hashlib.sha256(subprocess.check_output(
        ['git','show',cfg.SOURCE_COMMIT+':'+name],cwd=ROOT)).hexdigest() for name in source.SOURCE_FILES}


def base_conditions():
    return tuple((arm,1 if cfg.is_fdma(arm) else 4,policy,gamma) for arm in cfg.ARMS
        for policy in (('fdma',) if cfg.is_fdma(arm) else tuple(POLICIES)) for gamma in cfg.GAMMAS)


def conditions():
    return tuple((arm,mode,g,policy,gamma) for arm,g,policy,gamma in base_conditions() for mode in ('fixed','tuned'))


def select_global_alphas(rows):
    expected={(arm,g,p,gamma,alpha) for arm,g,p,gamma in base_conditions() for alpha in cfg.ALPHAS}
    indexed={}
    for row in rows:
        key=tuple(row.get(k) for k in ('arm','group_count','energy_policy','gamma_db','alpha'))
        value=row.get('sum_rate_bps_hz_mean')
        if (key not in expected or key in indexed or row.get('num_samples')!=64
                or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0):
            raise ValueError('Incomplete/duplicate/nonfinite alpha calibration condition')
        indexed[key]=value
    if set(indexed)!=expected:raise ValueError('Every method/ratio/gamma/alpha condition is required')
    selected={};scores=[];boundary={}
    for arm in cfg.ARMS:
        own=[key for key in base_conditions() if key[0]==arm]
        candidates=[]
        for alpha in cfg.ALPHAS:
            score=statistics.fmean(indexed[(*key,alpha)] for key in own)
            scores.append(dict(arm=arm,alpha=alpha,mean_sum_rate=score,equal_weight_conditions=len(own)))
            candidates.append((score,alpha))
        # Exact equality only: no isclose tie and no result-dependent grid extension.
        value=min(candidates,key=lambda item:(-item[0],item[1]))[1]
        selected[arm]=value;boundary[arm]=value in (cfg.ALPHAS[0],cfg.ALPHAS[-1])
    return dict(selected_alphas=selected,scores=scores,boundary_optima=boundary)


def validate_selection_record(record):
    identity=cfg.selection_identity(record)
    for key,value in select_global_alphas(record.get('calibration_summary',[])).items():
        if canonical(record.get(key))!=canonical(value):raise ValueError('Locked selection '+key+' changed')
    if (record.get('source_implementation_commit')!=cfg.SOURCE_COMMIT
            or record.get('sources')!={arm:cfg.source_identity(arm) for arm in cfg.ARMS}
            or record.get('validation_addresses')!=list(range(4096,4112))
            or len(record.get('shared_inputs',[]))!=16 or not record.get('code',{}).get('sha256')
            or not record.get('frozen_files') or record.get('freeze_checks',{}).get('passed') is not True):
        raise ValueError('Locked alpha source/data/freeze evidence incomplete')
    return identity


def write_once_json(path,value):
    path=cfg.inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def load_selection():
    record=read_json(cfg.SELECTION_PATH);validate_selection_record(record)
    if record['code']!=source_code_record():raise ValueError('Diagnostic code identity changed after alpha lock')
    directory=cfg.inside_root(record['calibration_directory'])
    if sha256(directory/'manifest.json')!=record['calibration_manifest_sha256']:
        raise ValueError('Locked calibration manifest changed')
    verify_stage_files(directory,'calibration')
    return record


def validate_source_metadata(payload,manifest,initial,arm,original_calibration,*,
                             archived_hashes,calibration_sha256):
    """Accept596 implementation while retaining the b6/8096 internal lineage."""
    identity=cfg.source_identity(arm)
    source_cfg.validate_resume_metadata(payload,arm,original_calibration,code_sha256=archived_hashes)
    if (payload['step']!=6048 or payload['cumulative_updates']!=22096
            or canonical(payload.get('calibration_record'))!=canonical(original_calibration)
            or payload.get('calibration_sha256')!=calibration_sha256):
        raise ValueError('Source terminal step or original beta calibration changed')
    old_code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived_hashes)
    expected=dict(format=source_cfg.CHECKPOINT_FORMAT,status='completed',arm=arm,seed=cfg.SEED,
        step=6048,cumulative_updates=22096,checkpoint_sha256=identity['sha256'],code=old_code,
        source=source_cfg.source_identity(arm),calibration=source_cfg.calibration_identity(original_calibration),
        recipe=source_cfg.recipe(arm,original_calibration),config_fingerprint=source_cfg.fingerprint(arm,original_calibration),
        ce_identity=payload['ce_identity'])
    for key,value in expected.items():
        if canonical(manifest.get(key))!=canonical(value):raise ValueError('Source manifest '+key+' mismatch')
    for key,value in dict(code=old_code,source=source_cfg.source_identity(arm),
        calibration=source_cfg.calibration_identity(original_calibration),recipe=source_cfg.recipe(arm,original_calibration),
        ce_identity=payload['ce_identity'],new_optimizer=True,strict_loading=True,migration=False).items():
        if canonical(initial.get(key))!=canonical(value):raise ValueError('Source initialization '+key+' mismatch')
    family='source_fdma' if cfg.is_fdma(arm) else 'source_g4'
    if initial.get('initialized_state_sha256')!=original_calibration['state_freeze'][family]['codec_model_sha256']:
        raise ValueError('Source initialization did not retain complete frozen ordered state')
    return dict(implementation_commit=cfg.SOURCE_COMMIT,internal_source_commit=cfg.INTERNAL_SOURCE_COMMIT,
                ancestor_source_commit=cfg.ANCESTOR_SOURCE_COMMIT,step=6048,cumulative_updates=22096,
                new_diagnostic_HEAD_may_differ=True)


def load_frozen_sources(torch,warmup,data_protocol,ce):
    archived=archived_source_hashes()
    for name,digest in archived.items():
        if sha256(ROOT/name)!=digest:raise ValueError('Accepted source runtime file changed: '+name)
    beta_path=source_cfg.CALIBRATION_PATH;beta=read_json(beta_path)
    source.validate_calibration_record(beta)  # Pure record verification; no calibration execution.
    if (beta['code']!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)
            or canonical(beta['data_protocol'])!=canonical(data_protocol) or beta['ce_identity']!=ce):
        raise ValueError('Original beta code/data/CE identity mismatch')
    frozen={str(beta_path):sha256(beta_path),**beta['frozen_files']}
    for item in ce.values():frozen[item['path']]=item['sha256']
    pointer=source_cfg.OUTPUT_ROOT/'last_batch.json';folder=source_cfg.inside_root(read_json(pointer)['directory'])
    summary_path=folder/'batch_summary.json';summary=read_json(summary_path)
    if (not folder.is_relative_to((source_cfg.OUTPUT_ROOT/'batches').resolve()) or summary.get('status')!='completed'
            or summary.get('error') or summary.get('code')!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)
            or set(summary.get('training',{}))!=set(cfg.ARMS)):
        raise ValueError('Completed task-loss source batch receipt required')
    beta_stage=summary.get('stages',{}).get('source_and_calibration',{})
    if (beta_stage.get('status')!='completed' or beta_stage.get('evidence_sha256',{}).get(str(beta_path))!=frozen[str(beta_path)]
            or summary.get('calibration',{}).get('record_id')!=beta['record_id']
            or summary.get('calibration',{}).get('beta_target')!=beta['beta_target']):
        raise ValueError('Completed original beta calibration receipt changed')
    frozen.update({str(pointer):sha256(pointer),str(summary_path):sha256(summary_path)})
    states={};methods={};progress={};chain=None
    for arm in cfg.ARMS:
        identity=cfg.source_identity(arm);path=Path(identity['path'])
        if sha256(path)!=identity['sha256']:raise ValueError('Pinned checkpoint SHA256 mismatch: '+arm)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        manifest=read_json(identity['manifest_path']);initial=read_json(identity['initialization_path'])
        verified=validate_source_metadata(payload,manifest,initial,arm,beta,
            archived_hashes=archived,calibration_sha256=frozen[str(beta_path)])
        if canonical(payload['data_protocol'])!=canonical(data_protocol) or payload['ce_identity']!=ce:
            raise ValueError('Source data/CE protocol mismatch')
        stage=summary.get('stages',{}).get('train_'+arm,{})
        if stage.get('status')!='completed' or not stage.get('evidence_sha256'):raise ValueError('Source training stage incomplete')
        batch_record=summary['training'][arm]
        if (batch_record.get('checkpoint_path')!=identity['path'] or batch_record.get('checkpoint_sha256')!=identity['sha256']
                or batch_record.get('step')!=6048 or batch_record.get('cumulative_updates')!=22096):
            raise ValueError('Source batch checkpoint pointer changed')
        for key in ('path','manifest_path','initialization_path','progress_path'):
            filename=identity[key];digest=sha256(filename)
            if stage['evidence_sha256'].get(filename)!=digest:raise ValueError('Source stage evidence changed: '+filename)
            frozen[filename]=digest
        # Numeric state checking is pure and does not bind the source to HEAD.
        source.previous.old._check_numerical_payload(torch,payload)
        if chain is None:chain=source.build_chain(torch,torch.device('cuda:0'),arm,payload).eval()
        model=chain.codec_model;source.previous.old.check_tensor_state(torch,model,payload['codec_model'])
        params=list(model.parameters());group=payload['optimizer']['param_groups'][0]
        if group['params']!=list(range(len(params))) or not set(payload['optimizer']['state'])<=set(group['params']):
            raise ValueError('Source optimizer parameter identity changed')
        for index,state in payload['optimizer']['state'].items():
            if any(state[k].shape!=params[index].shape or state[k].dtype!=params[index].dtype for k in ('exp_avg','exp_avg_sq')):
                raise ValueError('Source optimizer moment shape/dtype mismatch')
        model.load_state_dict(payload['codec_model'],strict=True);chain.eval()
        if any(module.training for module in chain.modules()):raise ValueError('All frozen modules must stay eval')
        residuals={name:_tensor_sha(model.state_dict()[name]) for name in (
            'encoder.key_window_residual.weight','encoder.shared_window_residual.weight')}
        methods[arm]=dict(identity=identity,verification=verified,strict_loading=True,migration=False,
            trained_window_residual_sha256=residuals,parameters=source.previous.parameters(model),
            state_sha256=_state_sha(model.state_dict()))
        states[arm]=payload['codec_model']
        progress[arm]=[json.loads(line) for line in Path(identity['progress_path']).read_text(encoding='utf-8').splitlines() if line.strip()]
        del payload
    for path,digest in frozen.items():
        if sha256(path)!=digest:raise ValueError('Source/beta evidence changed: '+path)
    trends=analyze_training_trends(progress,beta)
    return dict(chain=chain,states=states,methods=methods,progress=progress,trends=trends,beta=beta,frozen_files=frozen)


def analyze_training_trends(progress_by_arm,original_beta_calibration):
    if set(progress_by_arm)!=set(cfg.ARMS):raise ValueError('All four source progress logs required')
    result=dict(arms={},training_rows=[],monitor_rows=[],assessment='无法判断全SNR收敛；保留匹配续训选项。')
    for arm,events in progress_by_arm.items():
        if [row.get('step') for row in events]!=list(range(108,6049,108)):raise ValueError('Source progress coverage changed')
        groups={};monitors=[]
        for row in events:
            step=row['step'];condition=source_cfg.schedule(arm,step)
            if row.get('cumulative_updates')!=16048+step or canonical(row.get('condition'))!=canonical(condition):
                raise ValueError('Source progress local/global/condition mismatch')
            for metric in ('nmse','rate','beta'):
                value=row.get(metric)
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                    raise ValueError('Invalid training metric: '+metric)
            if row['beta']!=source_cfg.loss_weight(arm,step,original_beta_calibration):raise ValueError('Logged beta schedule changed')
            key=json.dumps({k:condition[k] for k in ('gamma_db','pilot_energy_per_ue','group_count','energy_ratio')},sort_keys=True)
            point=dict(arm=arm,step=step,cumulative_updates=row['cumulative_updates'],condition=key,
                       nmse=row['nmse'],rate=row['rate'],beta=row['beta'])
            groups.setdefault(key,[]).append(point);result['training_rows'].append(point)
            monitor=row.get('monitor')
            if monitor is not None:
                if (len(monitor)!=4 or {m.get('batch') for m in monitor}!=set(range(4))
                        or any(m.get('group_count')!=(1 if cfg.is_fdma(arm) else 4) for m in monitor)
                        or any(not math.isfinite(m.get(k,float('nan'))) or m[k]<0 for m in monitor for k in ('nmse','rate'))):
                    raise ValueError('Fixed monitor layout/coverage/metric mismatch')
                point=dict(arm=arm,step=step,nmse=statistics.fmean(m['nmse'] for m in monitor),
                    rate=statistics.fmean(m['rate'] for m in monitor),gamma_db=10.,pilot_energy_per_ue=16.,
                    energy_policy='fdma' if cfg.is_fdma(arm) else 'hybrid_1_3',rzf_alpha=.01,samples=16,batches=4)
                monitors.append(point);result['monitor_rows'].append(point)
        last=monitors[-4:]
        if len(last)<4:assessment='无法判断：固定monitor不足四个记录。'
        elif last[-1]['rate']>last[0]['rate'] or last[-1]['nmse']<last[0]['nmse']:
            assessment='小固定monitor仍有改善迹象；无法据此判断全SNR收敛，保留匹配续训选项。'
        else:assessment='末四个小固定monitor未见净改善；无法判断是否收敛，保留匹配续训选项。'
        result['arms'][arm]=dict(training_by_condition=groups,monitor=monitors,last_four_monitor=last,
            last_four_rate_change=None if len(last)<4 else last[-1]['rate']-last[0]['rate'],
            last_four_nmse_change=None if len(last)<4 else last[-1]['nmse']-last[0]['nmse'],assessment=assessment,
            limitation='gamma10/Ep16/16 fixed MU samples;G4 only1:3;not a full-SNR convergence test')
    return result


def summarize_samples(rows,count=256):
    indexed={key:{} for key in conditions()};fingerprints={};invariant={};selected={}
    for row in rows:
        key=tuple(row.get(k) for k in ('arm','rzf_mode','group_count','energy_policy','gamma_db'));i=row.get('sample_index')
        if key not in indexed or type(i) is not int or not 0<=i<count or i in indexed[key]:raise ValueError('Missing/duplicate diagnostic sample key')
        if (row.get('batch_index')!=i//4 or row.get('seed')!=cfg.SEED or row.get('pilot_energy_per_ue')!=16.
                or row.get('downlink_snr_db')!=25.):raise ValueError('Diagnostic protocol mismatch')
        fp=row.get('channel_fingerprint')
        if not isinstance(fp,str) or len(fp)!=64 or fingerprints.setdefault(i//4,fp)!=fp:raise ValueError('Unpaired channel fingerprint')
        alpha=row.get('alpha')
        if alpha not in cfg.ALPHAS or isinstance(alpha,bool):raise ValueError('Unknown diagnostic alpha')
        if key[1]=='fixed' and alpha!=.01:raise ValueError('Fixed RZF must retain alpha0.01')
        if key[1]=='tuned' and selected.setdefault(key[0],alpha)!=alpha:raise ValueError('Per-condition alpha selection forbidden')
        for metric in METRICS+USER_RATES+DIAGNOSTICS:
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Invalid diagnostic metric: '+metric)
        if not math.isclose(sum(row[k] for k in USER_RATES),row['sum_rate_bps_hz'],rel_tol=1e-5,abs_tol=1e-5):raise ValueError('User rates do not sum to MU rate')
        invkey=(key[0],key[2],key[3],key[4],i);values=tuple(row[k] for k in INVARIANT)
        if invariant.setdefault(invkey,values)!=values:raise ValueError('Raw H_hat NMSE/norms changed with alpha')
        if (abs(row['data_energy_per_ue']-48)>1e-4 or abs(row['frame_energy_per_ue']-64)>1e-4
                or abs(row['rzf_power_mean']-1)>1e-5 or not 0<=row.get('energy_max_abs_error',float('inf'))<=1e-4
                or not 0<=row.get('rzf_power_max_abs_error',float('inf'))<=1e-5):raise ValueError('Energy/RZF power constraint failed')
        indexed[key][i]=row
    summary=[]
    for key,records in indexed.items():
        if set(records)!=set(range(count)) or count%4:raise ValueError('Incomplete complete-MU sample coverage')
        item=dict(zip(('arm','rzf_mode','group_count','energy_policy','gamma_db'),key))
        item.update(alpha=records[0]['alpha'],num_samples=count,num_batches=count//4)
        for metric in METRICS+USER_RATES+DIAGNOSTICS:
            item.update({metric+'_'+k:v for k,v in statistics_for([records[i][metric] for i in range(count)]).items()})
        item.update(energy_max_abs_error=max(r['energy_max_abs_error'] for r in records.values()),
                    rzf_power_max_abs_error=max(r['rzf_power_max_abs_error'] for r in records.values()))
        summary.append(item)
    return summary,indexed


def contrasts():
    result=[]
    for arm in cfg.ARMS:
        for policy in (('fdma',) if cfg.is_fdma(arm) else tuple(POLICIES)):
            g=1 if cfg.is_fdma(arm) else 4
            result.append(dict(name=arm+'_tuned_minus_fixed_'+policy,coefficients={(arm,'tuned',g,policy):1,(arm,'fixed',g,policy):-1}))
    for mode in ('fixed','tuned'):
        fj=('fdma_joint',mode,1,'fdma');fn=('fdma_nmse',mode,1,'fdma')
        result.append(dict(name='fdma_joint_minus_fdma_nmse_'+mode,coefficients={fj:1,fn:-1}))
        for policy in POLICIES:
            gj=('g4_joint',mode,4,policy);gn=('g4_nmse',mode,4,policy)
            for name,weights in (('g4_joint_minus_g4_nmse',{gj:1,gn:-1}),
                ('g4_joint_minus_fdma_joint',{gj:1,fj:-1}),('gap_interaction',{gj:1,fj:-1,gn:-1,fn:1})):
                result.append(dict(name=name+'_'+mode+'_'+policy,coefficients=weights))
    return result


def paired_statistics(rows,count=256):
    _,indexed=summarize_samples(rows,count);paired=[]
    for contrast in contrasts():
        for scope,gammas in [(str(int(g)),(g,)) for g in cfg.GAMMAS]+[('low_0_5_10',cfg.GAMMAS[:3]),('all_six',cfg.GAMMAS)]:
            for metric in METRICS:
                values=[statistics.fmean(sum(c*indexed[(*key,gamma)][i][metric] for key,c in contrast['coefficients'].items())
                        for gamma in gammas) for i in range(count)]
                batches=[statistics.fmean(values[i:i+4]) for i in range(0,count,4)]
                row=dict(contrast=contrast['name'],estimand=scope,metric=metric,num_samples=count,num_batches=count//4,
                    coefficients=json.dumps([dict(method=list(k),coefficient=v) for k,v in contrast['coefficients'].items()]))
                for prefix,v in (('sample',values),('batch',batches)):
                    row.update({prefix+'_'+k:x for k,x in statistics_for(v).items()})
                paired.append(row)
    return paired


def pilot_only_observation(torch,output,indices):
    if bool((output['tx']['Y'][...,list(indices)]!=0).any()):raise ValueError('Data reception overlaps CE pilot inputs')
    pilot=torch.zeros_like(output['Y_rx']);pilot[...,list(indices)]=output['Y_rx'][...,list(indices)]
    return pilot


def score_cached_hhat(torch,downlink,Hhat,Htrue,alphas):
    """No network calls, no fitted scaling, no truth-dependent alpha selection."""
    before=Hhat.detach().clone();tau=Hhat.abs().square().sum((1,2))/8
    energy=Hhat.abs().square().sum((1,2,3));true_energy=Htrue.abs().square().sum((1,2,3))
    if not bool(torch.isfinite(tau).all() & (tau>0).all() & (true_energy>0).all()):
        raise ValueError('Finite nonzero raw channel norm required')
    base=dict(dl_nmse=downlink.downlink_nmse(Hhat,Htrue).mean(-1),hhat_energy_total=energy,hhat_fro_norm=energy.sqrt(),
        hhat_to_true_norm_ratio=energy.sqrt()/true_energy.sqrt(),hhat_tone_user_energy_mean=tau.mean(-1))
    result={}
    for alpha in dict.fromkeys(alphas):
        precoder,power=downlink.design_rzf(Hhat,float(alpha))
        rate=downlink.evaluate_sum_rate(precoder,Htrue,25.);relative=float(alpha)/tau
        result[alpha]=dict(base,sum_rate_bps_hz=rate['sum_rate_bps_hz'],desired_signal_power=rate['desired_signal_power_mean'],
            interference_power=rate['interference_power_mean'],relative_alpha_mean=relative.mean(-1),
            relative_alpha_min=relative.amin(-1),relative_alpha_max=relative.amax(-1),rzf_power_mean=power.mean(-1),
            rzf_power_max_abs_error=(power-1).abs().amax(-1),
            **{name:rate['user_rate_bps_hz'][:,k] for k,name in enumerate(USER_RATES)})
        if any(not bool(torch.isfinite(v).all()) for v in result[alpha].values()):raise FloatingPointError('Nonfinite grid score')
        if result[alpha]['rzf_power_max_abs_error'].max().item()>1e-5:raise ValueError('Per-tone RZF power mismatch')
    if not torch.equal(before,Hhat):raise ValueError('RZF scoring modified raw H_hat')
    return result


def run_stage(stage,output_dir,selection=None):
    if stage not in ('calibration','evaluation'):raise ValueError('Unknown diagnostic split')
    output=cfg.inside_root(output_dir)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',stage=stage,code=source_code_record(),test_used=False,optimizer_updates=0,
        source_implementation_commit=cfg.SOURCE_COMMIT,shared_inputs=[],network_forward_calls=0,rzf_solves=0)
    inputs={};failure=None;chain=None;ce_before=None
    try:
        torch,np,warmup=source.previous.old.runtime();device=torch.device('cuda:0')
        if stage=='evaluation':
            validate_selection_record(selection)
            if selection['code']!=source_code_record():raise ValueError('Locked diagnostic code changed')
            inputs[str(cfg.SELECTION_PATH)]=sha256(cfg.SELECTION_PATH)
        data=source.make_data(warmup,device);data.prepare();ce=source.ce_identity()
        loaded=load_frozen_sources(torch,warmup,data.protocol,ce);chain=loaded['chain'];inputs.update(loaded['frozen_files'])
        if stage=='calibration':
            write_once_json(cfg.OUTPUT_ROOT/'selection_attempt.json',dict(started=now(),code=source_code_record(),grid=list(cfg.ALPHAS)))
        states=loaded['states'];ce_before=_state_sha(chain.ce_model.state_dict())
        manifest.update(data_protocol=data.protocol,ce_identity=ce,methods=loaded['methods'],backend=source.backend(torch),
            original_beta_identity=source_cfg.calibration_identity(loaded['beta']))
        if stage=='evaluation':
            for path,digest in selection['frozen_files'].items():
                if inputs.get(path)!=digest:raise ValueError('Selection frozen sources changed before validation')
            if canonical(selection['data_protocol'])!=canonical(data.protocol):raise ValueError('Locked data protocol changed')
        batches=16 if stage=='calibration' else 64;start=4096 if stage=='calibration' else 5120
        manifest.update(validation_addresses=list(range(start,start+batches)))
        raw=[];totals={};downlink=warmup._system.downlink
        with torch.no_grad():
            for bi in range(batches):
                sample=data.sample('validation',start+bi);hdl,hul=sample['H_dl'],sample['H_ul']
                originals=(_tensor_sha(hdl),_tensor_sha(hul));noises={};refs={}
                shared=dict(batch_index=bi,address=start+bi,H_dl_sha256=originals[0],H_ul_sha256=originals[1],
                    sample_fingerprint=sample['sample_fingerprint'],noise={},ce_inputs={})
                for gi,gamma in enumerate(cfg.GAMMAS):
                    pair=[]
                    for kind in ('pilot','data'):
                        generator=torch.Generator(device='cpu').manual_seed(cfg.noise_seed(stage,bi,gi,kind))
                        noise=torch.view_as_complex(torch.randn(4,32,256,2,generator=generator)).to(device);pair.append(noise)
                        shared['noise'][f'{gamma}_{kind}']=_tensor_sha(noise)
                    noises[gamma]=tuple(pair)
                pilot_indices=sorted({q for row in warmup.protocol.PILOT_POSITIONS_BY_USER for q in row})
                known=chain._known_pilots(4,device);known_sha=_tensor_sha(known)
                for arm in cfg.ARMS:
                    chain.codec_model.load_state_dict(states[arm],strict=True);chain.eval();snapshot=_snapshot(chain)
                    for _,g,policy,gamma in (c for c in base_conditions() if c[0]==arm):
                        condition=dict(gamma_db=gamma,pilot_energy_per_ue=16.,group_count=g,energy_ratio=None if policy=='fdma' else POLICIES[policy])
                        pilot,noise=noises[gamma]
                        out=source.previous.old._run(chain,warmup,hdl,hul,condition,pilot,noise);manifest['network_forward_calls']+=1
                        ce_inputs=(pilot_only_observation(torch,out,pilot_indices),out['H_ul_ls'],out['H_ul_est'])
                        if gamma not in refs:
                            refs[gamma]=tuple(v.detach().clone() for v in ce_inputs)
                            shared['ce_inputs'][str(gamma)]=dict(pilot_y_sha256=_tensor_sha(ce_inputs[0]),known_pilots_sha256=known_sha,
                                H_ls_sha256=_tensor_sha(ce_inputs[1]),H_ul_est_sha256=_tensor_sha(ce_inputs[2]),Ep=16.,N0=out['n0'])
                        elif any(not torch.equal(a,b) for a,b in zip(refs[gamma],ce_inputs)):
                            raise ValueError('Actual CE inputs/estimate differ across frozen methods or ratios')
                        alphas=cfg.ALPHAS if stage=='calibration' else (.01,selection['selected_alphas'][arm])
                        scores=score_cached_hhat(torch,downlink,out['H_dl_hat'],hdl,alphas);manifest['rzf_solves']+=len(scores)
                        energy=out['energy']['actual_data_per_ue'];energy_error=(energy-48).abs().amax(-1)
                        if not bool(torch.isfinite(energy).all()) or energy_error.max().item()>1e-4:raise ValueError('UE energy mismatch')
                        if stage=='calibration':
                            for alpha,score in scores.items():
                                key=(arm,g,policy,gamma,alpha);sums=totals.setdefault(key,dict(count=0,rate=0.))
                                sums['count']+=4;sums['rate']+=math.fsum(score['sum_rate_bps_hz'].cpu().tolist())
                        else:
                            for mode,alpha in (('fixed',.01),('tuned',selection['selected_alphas'][arm])):
                                values=dict(scores[alpha],data_energy_per_ue=energy.mean(-1),frame_energy_per_ue=energy.mean(-1)+16.,
                                    energy_max_abs_error=energy_error)
                                arrays={k:v.cpu().tolist() for k,v in values.items()}
                                raw.extend(dict(arm=arm,rzf_mode=mode,group_count=g,energy_policy=policy,gamma_db=gamma,alpha=alpha,
                                    seed=cfg.SEED,pilot_energy_per_ue=16.,downlink_snr_db=25.,sample_index=bi*4+i,batch_index=bi,
                                    channel_fingerprint=sample['sample_fingerprint'],**{k:v[i] for k,v in arrays.items()}) for i in range(4))
                        del out,scores,ce_inputs
                    if snapshot!=_snapshot(chain) or _state_sha(chain.codec_model.state_dict())!=loaded['methods'][arm]['state_sha256']:
                        raise ValueError('Frozen codec state/buffers/modes changed')
                if originals!=(_tensor_sha(hdl),_tensor_sha(hul)):raise ValueError('Shared channel inputs changed')
                for gamma,pair in noises.items():
                    for kind,value in zip(('pilot','data'),pair):
                        if _tensor_sha(value)!=shared['noise'][f'{gamma}_{kind}']:raise ValueError('Standard noise changed')
                manifest['shared_inputs'].append(shared)
                del refs,noises
                print(f'{stage} batch {bi+1}/{batches}',flush=True)
        trends=loaded['trends']
        if stage=='calibration':
            summary=[dict(zip(('arm','group_count','energy_policy','gamma_db','alpha'),key),num_samples=value['count'],
                sum_rate_bps_hz_mean=value['rate']/value['count']) for key,value in totals.items()]
            chosen=select_global_alphas(summary);_write_csv(output/'alpha_calibration.csv',summary);_write_csv(output/'alpha_global_scores.csv',chosen['scores'])
            manifest.update(calibration_summary=summary,**chosen)
        else:
            save_json(output/'trends.json',trends);_write_csv(output/'training_by_condition.csv',trends['training_rows'])
            _write_csv(output/'fixed_monitor.csv',trends['monitor_rows'])
            summary,_=summarize_samples(raw);paired=paired_statistics(raw)
            _write_csv(output/'summary.csv',summary);_write_csv(output/'samples.csv.gz',raw,True);_write_csv(output/'paired_deltas.csv',paired)
            save_json(output/'results.json',dict(summary=summary,paired=paired,methods=loaded['methods'],
                selection=cfg.selection_identity(selection),trends=trends))
            manifest.update(summary_rows=len(summary),sample_rows=len(raw),paired_rows=len(paired),selection=cfg.selection_identity(selection))
        manifest.update(status='completed',finished=now())
    except BaseException as error:
        failure=error;manifest.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        errors=[]
        for path,digest in inputs.items():
            try:
                if not Path(path).is_file() or sha256(path)!=digest:errors.append('Source input changed: '+path)
            except OSError as error:errors.append('Source input unreadable: '+path+': '+str(error))
        try:
            if manifest['code']!=source_code_record():errors.append('Diagnostic source code changed')
        except Exception as error:errors.append('Diagnostic code unreadable: '+str(error))
        if chain is not None:
            if ce_before is None or _state_sha(chain.ce_model.state_dict())!=ce_before:errors.append('Frozen CE state changed or not initialized')
            if any(module.training for module in chain.modules()):errors.append('Eval modes changed')
        manifest.update(frozen_files=inputs,freeze_checks=dict(passed=not errors and failure is None,errors=errors))
        if errors:failure=failure or RuntimeError('; '.join(errors));manifest.update(status='failed',error=str(failure))
        if failure is None:manifest['output_sha256']={name:sha256(output/name) for name in ARTIFACTS[stage]}
        save_json(output/'manifest.json',manifest)
    if failure is not None:raise RuntimeError('Frozen RZF diagnostic failed; evidence '+str(output)) from failure
    return manifest


def verify_stage_files(output_dir,stage):
    output=cfg.inside_root(output_dir);manifest=read_json(output/'manifest.json')
    batches=16 if stage=='calibration' else 64;start=4096 if stage=='calibration' else 5120
    if (manifest.get('status')!='completed' or manifest.get('stage')!=stage or manifest.get('code')!=source_code_record()
            or manifest.get('freeze_checks',{}).get('passed') is not True or manifest.get('test_used') is not False
            or manifest.get('optimizer_updates')!=0 or manifest.get('source_implementation_commit')!=cfg.SOURCE_COMMIT
            or manifest.get('validation_addresses')!=list(range(start,start+batches))
            or len(manifest.get('shared_inputs',[]))!=batches or manifest.get('network_forward_calls')!=batches*36
            or set(manifest.get('methods',{}))!=set(cfg.ARMS)):
        raise ValueError('Incomplete frozen RZF stage receipt')
    if {name:sha256(output/name) for name in ARTIFACTS[stage]}!=manifest.get('output_sha256'):
        raise ValueError('Frozen RZF artifacts changed')
    for path,digest in manifest.get('frozen_files',{}).items():
        if sha256(path)!=digest:raise ValueError('Frozen source evidence changed: '+path)
    if stage=='calibration':
        if manifest.get('rzf_solves')!=batches*36*9:raise ValueError('Grid scoring did not reuse each H_hat')
    elif (manifest.get('summary_rows')!=72 or manifest.get('sample_rows')!=72*256 or manifest.get('paired_rows')!=640):
        raise ValueError('Final fixed/tuned condition/sample coverage changed')
    return manifest


def seal_selection(output_dir):
    """Seal completed calibration evidence after interruption, without new inference."""
    if cfg.SELECTION_PATH.exists():return load_selection()
    manifest=verify_stage_files(output_dir,'calibration')
    record=dict(format=cfg.SELECTION_FORMAT,status='passed',created=now(),grid=list(cfg.ALPHAS),
        code=source_code_record(),source_implementation_commit=cfg.SOURCE_COMMIT,
        sources={arm:cfg.source_identity(arm) for arm in cfg.ARMS},data_protocol=manifest['data_protocol'],
        original_beta_identity=manifest['original_beta_identity'],calibration_directory=str(cfg.inside_root(output_dir)),
        calibration_manifest_sha256=sha256(cfg.inside_root(output_dir)/'manifest.json'),
        validation_addresses=manifest['validation_addresses'],shared_inputs=manifest['shared_inputs'],
        frozen_files=manifest['frozen_files'],freeze_checks=manifest['freeze_checks'],
        calibration_summary=manifest['calibration_summary'],selected_alphas=manifest['selected_alphas'],
        scores=manifest['scores'],boundary_optima=manifest['boundary_optima'],test_used=False)
    record['record_id']=cfg.json_sha256(record);validate_selection_record(record);write_once_json(cfg.SELECTION_PATH,record)
    return record


def calibrate(output_dir):
    if cfg.SELECTION_PATH.exists():return load_selection()
    attempt=cfg.OUTPUT_ROOT/'selection_attempt.json'
    if attempt.exists():raise RuntimeError('Incomplete alpha-selection attempt retained; no silent reselection')
    run_stage('calibration',output_dir)
    return seal_selection(output_dir)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    for name in ('dry-run','calibrate','evaluate'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(config=cfg.recipe(),models_loaded=False,CSI_cache_written=False,training=False),indent=2));return 0
    if not(args.calibrate or args.evaluate):parser.print_help();return 0
    if args.output_dir is None:parser.error('Explicit unused --output-dir required')
    if args.calibrate:calibrate(args.output_dir)
    else:run_stage('evaluation',args.output_dir,load_selection())
    return 0


if __name__=='__main__':raise SystemExit(main())
