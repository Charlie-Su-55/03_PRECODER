#!/usr/bin/env python3
"""Matched K8 continuation from NMSE sources with fixed layout-specific RZF."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
from types import SimpleNamespace
import uuid

from configs import twc_v1_k8_rzf_matched_training_config as cfg
from configs import twc_v1_k8_task_loss_screen_config as task_cfg
import train_twc_v1_k8_task_loss_screen as task
import evaluate_twc_v1_k8_rzf_probe as probe

ROOT=cfg.REPO_ROOT
canonical,sha256,save_json=task.canonical,task.sha256,task.save_json
capture_rng,restore_rng=task.capture_rng,task.restore_rng
make_data,ce_identity,tensor_digest=task.make_data,task.ce_identity,task.tensor_digest
SOURCE_FILES=tuple(dict.fromkeys(probe.SOURCE_FILES+(
    'configs/twc_v1_k8_rzf_matched_training_config.py','train_twc_v1_k8_rzf_matched_training.py',
    'evaluate_twc_v1_k8_rzf_matched_training.py','run_twc_v1_k8_rzf_matched_training.py')))
SOURCE_METHODS=('source_fdma','source_g4')
RULE='0.25 * median(all NMSE gradient norms) / median(all sum-rate gradient norms)'


def now():return datetime.now(timezone.utc).isoformat()


def source_code_record():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})


@lru_cache(maxsize=2)
def archived_hashes(commit,files):
    return {name:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)).hexdigest() for name in files}


def validate_training_alpha_record(record,manifest,*,archived_hashes):
    """Old probe acceptance is bound to0e, never to this experiment's HEAD."""
    identity=probe.validate_selection_record(record)
    expected={arm:(.03 if arm.startswith('fdma') else .1) for arm in probe.cfg.ARMS}
    code=dict(commit=cfg.DIAGNOSTIC_COMMIT,sha256=archived_hashes)
    if record.get('code')!=code or record['selected_alphas']!=expected:
        raise ValueError('Read-only original RZF choice/code differs from the fixed decision')
    if (manifest.get('status')!='completed' or manifest.get('stage')!='calibration'
            or manifest.get('code')!=code or manifest.get('source_implementation_commit')!=cfg.SOURCE_COMMIT
            or manifest.get('freeze_checks',{}).get('passed') is not True
            or manifest.get('validation_addresses')!=list(range(4096,4112))
            or manifest.get('network_forward_calls')!=576 or manifest.get('rzf_solves')!=5184):
        raise ValueError('Original RZF calibration receipt incomplete')
    if any(canonical(record.get(k))!=canonical(manifest.get(k)) for k in
           ('selected_alphas','calibration_summary','scores','boundary_optima','shared_inputs','frozen_files')):
        raise ValueError('Original RZF lock differs from its calibration receipt')
    return identity


def load_training_alpha_record():
    old=read_json(probe.cfg.SELECTION_PATH)
    directory=probe.cfg.inside_root(old['calibration_directory'])
    manifest=read_json(directory/'manifest.json')
    hashes=archived_hashes(cfg.DIAGNOSTIC_COMMIT,tuple(probe.SOURCE_FILES))
    identity=validate_training_alpha_record(old,manifest,archived_hashes=hashes)
    for name,digest in hashes.items():
        if sha256(ROOT/name)!=digest:raise ValueError('Accepted pre-training source/diagnostic code changed: '+name)
    if sha256(directory/'manifest.json')!=old['calibration_manifest_sha256']:raise ValueError('Original alpha manifest SHA changed')
    for name,digest in manifest['output_sha256'].items():
        if sha256(directory/name)!=digest:raise ValueError('Original alpha artifact changed: '+name)
    frozen=dict(old['frozen_files'])
    frozen.update({str(probe.cfg.SELECTION_PATH):sha256(probe.cfg.SELECTION_PATH),
        str(directory/'manifest.json'):sha256(directory/'manifest.json')})
    frozen.update({str(directory/name):digest for name,digest in manifest['output_sha256'].items()})
    for path,digest in frozen.items():
        if sha256(path)!=digest:raise ValueError('Original alpha frozen evidence changed: '+path)
    return old,dict(record_identity=identity,record_sha256=sha256(probe.cfg.SELECTION_PATH),frozen_files=frozen)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_sources(torch,warmup,data_protocol,ce):
    alpha,evidence=load_training_alpha_record()
    archived=archived_hashes(cfg.SOURCE_COMMIT,tuple(task.SOURCE_FILES))
    beta=read_json(task_cfg.CALIBRATION_PATH);task.validate_calibration_record(beta)
    if (beta['code']!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)
            or canonical(beta['data_protocol'])!=canonical(data_protocol) or beta['ce_identity']!=ce):
        raise ValueError('Historical beta/source protocol mismatch')
    oldpointer=task_cfg.OUTPUT_ROOT/'last_batch.json';folder=task_cfg.inside_root(read_json(oldpointer)['directory'])
    summary_path=folder/'batch_summary.json';summary=read_json(summary_path)
    if (summary.get('status')!='completed' or summary.get('error')
            or summary.get('code')!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)):
        raise ValueError('Completed historical task-loss batch required')
    result={}
    for method in SOURCE_METHODS:
        identity=cfg.source_identity(method);arm=identity['arm'];path=Path(identity['path'])
        if arm not in ('fdma_nmse','g4_nmse'):raise ValueError('Joint source initialization forbidden')
        if sha256(path)!=identity['sha256']:raise ValueError('Pinned NMSE source weight SHA256 mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        manifest=read_json(identity['manifest_path']);initial=read_json(identity['initialization_path'])
        verified=probe.validate_source_metadata(payload,manifest,initial,arm,beta,
            archived_hashes=archived,calibration_sha256=sha256(task_cfg.CALIBRATION_PATH))
        task.previous.old._check_numerical_payload(torch,payload)
        if payload['ce_identity']!=ce or canonical(payload['data_protocol'])!=canonical(data_protocol):
            raise ValueError('NMSE source CE/data protocol mismatch')
        stage=summary.get('stages',{}).get('train_'+arm,{})
        record=summary.get('training',{}).get(arm,{})
        if (stage.get('status')!='completed' or record.get('checkpoint_path')!=identity['path']
                or record.get('checkpoint_sha256')!=identity['sha256'] or record.get('step')!=6048
                or record.get('cumulative_updates')!=22096):raise ValueError('NMSE source completion pointer changed')
        events=[json.loads(line) for line in Path(identity['progress_path']).read_text(encoding='utf-8').splitlines() if line.strip()]
        if ([r.get('step') for r in events]!=list(range(108,6049,108))
                or any(r.get('cumulative_updates')!=16048+r['step'] for r in events)):raise ValueError('NMSE source log coverage changed')
        frozen=dict(evidence['frozen_files'])
        for key in ('path','manifest_path','initialization_path','progress_path'):
            filename=identity[key];digest=sha256(filename)
            if stage.get('evidence_sha256',{}).get(filename)!=digest:raise ValueError('NMSE source sidecar changed')
            frozen[filename]=digest
        frozen.update({str(oldpointer):sha256(oldpointer),str(summary_path):sha256(summary_path),
                       str(task_cfg.CALIBRATION_PATH):sha256(task_cfg.CALIBRATION_PATH)})
        result[method]=(payload,dict(identity=identity,verification=verified,frozen_files=frozen,
            historical_beta_identity=task_cfg.calibration_identity(beta),training_alpha_identity=evidence['record_identity']))
    return result


def build_chain(torch,device,method,payload):
    source_arm='fdma_nmse' if method in ('fdma_nmse','fdma_joint','source_fdma') else 'g4_nmse' if method in ('g4_nmse','g4_joint','source_g4') else None
    if source_arm is None:raise ValueError('Unknown fixed matched-training method')
    return task.build_chain(torch,device,source_arm,payload)


gradient_l2_norm=task.gradient_l2_norm


def calibration_from_records(rows):
    if not isinstance(rows,list) or len(rows)!=24:raise ValueError('Exactly24 new gradient records required')
    expected={(method,500100+i) for method in SOURCE_METHODS for i in range(12)};seen=set();paired={}
    for row in rows:
        key=(row.get('source_key'),row.get('batch_index'))
        if key not in expected or key in seen:raise ValueError('Unexpected/duplicate new beta address')
        seen.add(key);offset=key[1]-500100
        if row.get('gamma_db')!=cfg.GAMMAS[offset%6] or row.get('rzf_alpha')!=cfg.training_alpha(key[0]):
            raise ValueError('New beta gamma/fixed RZF path changed')
        for name in ('nmse_grad_norm','rate_grad_norm'):
            value=row.get(name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('New beta requires finite nonzero effective gradients')
        hashes=tuple(row.get(k) for k in ('sample_fingerprint','pilot_sha256','data_sha256'))
        if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in hashes):
            raise ValueError('Actual shared calibration input hashes required')
        if key[1] in paired and paired[key[1]]!=hashes:raise ValueError('New beta layouts did not share actual inputs')
        paired[key[1]]=hashes
    ng=statistics.median(r['nmse_grad_norm'] for r in rows);rg=statistics.median(r['rate_grad_norm'] for r in rows)
    beta=.25*ng/rg
    if not math.isfinite(beta) or beta<=0:raise ValueError('Invalid new beta')
    return dict(beta_target=beta,median_nmse_grad_norm=ng,median_rate_grad_norm=rg,rule=RULE)


def validate_calibration_record(record):
    identity=cfg.calibration_identity(record)
    for key,value in calibration_from_records(record.get('records')).items():
        if record.get(key)!=value:raise ValueError('New beta fixed rule differs: '+key)
    if (record.get('sources')!={m:cfg.source_identity(m) for m in SOURCE_METHODS}
            or not record.get('original_training_alpha_identity')
            or record.get('training_alphas')!={m:cfg.training_alpha(m) for m in SOURCE_METHODS}
            or set(record.get('state_freeze',{}))!=set(SOURCE_METHODS)
            or any(v.get('passed') is not True for v in record['state_freeze'].values())
            or not record.get('code',{}).get('sha256') or not record.get('frozen_files') or not record.get('data_protocol')):
        raise ValueError('New beta source/state/alpha evidence incomplete')
    return identity


def load_calibration():
    record=read_json(cfg.CALIBRATION_PATH);validate_calibration_record(record)
    if record['code']!=source_code_record():raise ValueError('Matched experiment code changed after new beta calibration')
    for path,digest in record['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('New beta frozen evidence changed: '+path)
    return record


def write_once_json(path,value):
    path=cfg.inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def check_power(torch,out,metrics,data_energy):
    energy=out['energy']['actual_data_per_ue'];power=metrics['rzf_per_subcarrier_power']
    if not bool(torch.isfinite(energy).all() & torch.isfinite(power).all()):raise FloatingPointError('Nonfinite UE/RZF power')
    e=float((energy-data_energy).abs().max().item());p=float((power-1).abs().max().item())
    if e>1e-4 or p>1e-5:raise ValueError('UE energy/RZF power constraint changed')
    return dict(energy_max_abs_error=e,rzf_power_max_abs_error=p)


def calibrate(report_path):
    report_path=cfg.inside_root(report_path)
    if report_path.exists():raise FileExistsError(report_path)
    if cfg.CALIBRATION_PATH.exists():
        record=load_calibration();save_json(report_path,dict(status='passed',reused=True,record_id=record['record_id']));return record
    attempt=cfg.OUTPUT_ROOT/'beta_attempt.json'
    if attempt.exists():raise RuntimeError('Incomplete new beta attempt retained; no automatic recalibration')
    report=dict(status='running',records=[],sources={},state_freeze={},frozen_files={});failure=None
    try:
        torch,np,warmup=task.previous.old.runtime();device=torch.device('cuda:0')
        data=make_data(warmup,device);data.prepare();ce=ce_identity();code=source_code_record()
        sources=load_sources(torch,warmup,data.protocol,ce);oldalpha,ae=load_training_alpha_record()
        write_once_json(attempt,dict(started=now(),code=code,training_alpha_identity=ae['record_identity']))
        report.update(code=code,data_protocol=data.protocol,ce_identity=ce,backend=task.backend(torch))
        for item in ce.values():report['frozen_files'][item['path']]=item['sha256']
        for method,(payload,evidence) in sources.items():
            chain=build_chain(torch,device,method,payload).eval();model=chain.codec_model
            if any(not p.requires_grad for p in model.parameters()):raise ValueError('All codec parameters must remain trainable')
            expected=tensor_digest(torch,payload['codec_model'])
            if tensor_digest(torch,model.state_dict())!=expected:raise ValueError('Full NMSE source was not strictly copied')
            before=tensor_digest(torch,chain.state_dict());buffers=tensor_digest(torch,dict(chain.named_buffers()))
            modes={name:m.training for name,m in chain.named_modules()};params=tuple(model.parameters())
            report['sources'][method]=evidence;report['frozen_files'].update(evidence['frozen_files'])
            for offset in range(12):
                sample=data.sample('codec_train',500100+offset);hdl,hul=sample['H_dl'],sample['H_ul']
                pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(
                    cfg.noise_seed('beta_calibration',offset,offset%6,kind)),device) for kind in ('pilot','data')]
                condition=dict(gamma_db=cfg.GAMMAS[offset%6],pilot_energy_per_ue=16.,group_count=1 if method=='source_fdma' else 4,
                    energy_ratio=None if method=='source_fdma' else (1,3))
                out=task.previous.old._run(chain,warmup,hdl,hul,condition,*pair)
                metrics=chain.metrics(out,hdl,rzf_alpha=cfg.training_alpha(method),downlink_snr_db=25.)
                nmse=metrics['dl_nmse_per_user'].mean();rate=metrics['sum_rate_bps_hz'].mean()
                ng=torch.autograd.grad(nmse,params,retain_graph=True,allow_unused=True);nnorm=gradient_l2_norm(torch,ng);del ng
                rg=torch.autograd.grad(rate,params,allow_unused=True);rnorm=gradient_l2_norm(torch,rg);del rg
                if not bool(torch.isfinite(nmse)&torch.isfinite(rate)):raise ValueError('New beta objective nonfinite')
                check_power(torch,out,metrics,48.)
                row=dict(source_key=method,batch_index=500100+offset,gamma_db=condition['gamma_db'],rzf_alpha=cfg.training_alpha(method),
                    nmse_grad_norm=nnorm,rate_grad_norm=rnorm,sample_fingerprint=sample['sample_fingerprint'],
                    pilot_sha256=tensor_digest(torch,{'noise':pair[0]}),data_sha256=tensor_digest(torch,{'noise':pair[1]}))
                report['records'].append(row);print(json.dumps(row),flush=True);del out,metrics,nmse,rate
            passed=(before==tensor_digest(torch,chain.state_dict()) and buffers==tensor_digest(torch,dict(chain.named_buffers()))
                and modes=={name:m.training for name,m in chain.named_modules()} and not chain.ce_model.training
                and all(p.grad is None for p in chain.parameters()))
            report['state_freeze'][method]=dict(passed=passed,codec_model_sha256=expected,
                state_before=before,state_after=tensor_digest(torch,chain.state_dict()),parameters=task.previous.parameters(model))
            if not passed:raise ValueError('New beta calibration changed source/CE state')
            del chain,model,params;torch.cuda.empty_cache()
        stats=calibration_from_records(report['records'])
        for path,digest in report['frozen_files'].items():
            if sha256(path)!=digest:raise ValueError('New beta source changed')
        if code!=source_code_record():raise ValueError('Matched code changed during calibration')
        record=dict(format=cfg.CALIBRATION_FORMAT,status='passed',created=now(),**stats,records=report['records'],
            sources={m:cfg.source_identity(m) for m in SOURCE_METHODS},code=code,data_protocol=data.protocol,ce_identity=ce,
            frozen_files=report['frozen_files'],state_freeze=report['state_freeze'],backend=report['backend'],
            original_training_alpha_identity=ae['record_identity'],original_training_alpha_sha256=ae['record_sha256'],
            training_alphas={m:cfg.training_alpha(m) for m in SOURCE_METHODS},optimizer_updates=0,test_used=False)
        record['record_id']=cfg.json_sha256(record);validate_calibration_record(record);write_once_json(cfg.CALIBRATION_PATH,record)
        report.update(status='passed',record_id=record['record_id'],**stats,finished=now())
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=__import__('traceback').format_exc())
    finally:save_json(report_path,report)
    if failure is not None:raise RuntimeError('Matched beta/source acceptance failed; '+str(report_path)) from failure
    return record


def task_loss(nmse,rate,arm,step,calibration):
    weight=cfg.loss_weight(arm,step,calibration)
    return nmse if not cfg.is_joint(arm) else nmse-weight*rate


def condition_key(condition):
    return json.dumps({k:condition[k] for k in ('gamma_db','pilot_energy_per_ue','group_count','energy_ratio')},sort_keys=True)


def expected_counts(arm,step):
    return dict(Counter(condition_key(cfg.schedule(arm,i)) for i in range(1,step+1)))


def monitor(torch,np,warmup,chain,data,arm,step):
    """64 fixed MU channels, full grid, finally restores exact modes and RNG."""
    if step not in cfg.PROGRESS_VALIDATION_STEPS:raise ValueError('Only predeclared monitor steps allowed')
    if chain.ce_model.training or any(p.requires_grad for p in chain.ce_model.parameters()):raise ValueError('CE must stay frozen/eval')
    rng=capture_rng(torch,np);modes=[(m,m.training) for m in chain.modules()]
    ce_before=tensor_digest(torch,chain.ce_model.state_dict());codec_before=tensor_digest(torch,chain.codec_model.state_dict())
    buffers=tensor_digest(torch,dict(chain.named_buffers()))
    points={};inputs=[];device=next(chain.codec_model.parameters()).device
    try:
        chain.eval()
        with torch.no_grad():
            for bi in range(16):
                sample=data.sample('validation',8960+bi);hdl,hul=sample['H_dl'],sample['H_ul']
                item=dict(address=8960+bi,channel_fingerprint=sample['sample_fingerprint'],noise={})
                for gi,gamma in enumerate(cfg.GAMMAS):
                    pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(cfg.noise_seed('monitor',bi,gi,kind)),device)
                          for kind in ('pilot','data')]
                    item['noise'][str(gamma)]=[tensor_digest(torch,{'noise':v}) for v in pair]
                    for ratio in ((None,) if cfg.is_fdma(arm) else ((1,3),(1,1))):
                        policy='fdma' if ratio is None else 'hybrid_'+str(ratio[0])+'_'+str(ratio[1])
                        cond=dict(gamma_db=gamma,pilot_energy_per_ue=16.,group_count=1 if ratio is None else 4,energy_ratio=ratio)
                        out=task.previous.old._run(chain,warmup,hdl,hul,cond,*pair)
                        metric=chain.metrics(out,hdl,rzf_alpha=cfg.training_alpha(arm),downlink_snr_db=25.)
                        check_power(torch,out,metric,48.)
                        values=dict(dl_nmse=metric['dl_nmse_per_user'].mean(-1),sum_rate_bps_hz=metric['sum_rate_bps_hz'],
                            desired_signal_power=metric['desired_signal_power_mean'],interference_power=metric['interference_power_mean'])
                        target=points.setdefault((policy,gamma),{k:[] for k in values})
                        for key,value in values.items():
                            if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite fixed monitor metric')
                            target[key].extend(value.cpu().tolist())
                inputs.append(item)
        summary=[]
        for (policy,gamma),metrics in points.items():
            if any(len(values)!=64 for values in metrics.values()):raise ValueError('Incomplete monitor MU sample grid')
            row=dict(energy_policy=policy,gamma_db=gamma,num_samples=64,alpha=cfg.training_alpha(arm))
            for key,values in metrics.items():row.update({key+'_'+k:v for k,v in probe.statistics_for(values).items()})
            summary.append(row)
        means=[]
        for policy in sorted({key[0] for key in points}):
            row=dict(energy_policy=policy,num_samples=64,alpha=cfg.training_alpha(arm))
            for metric in probe.METRICS:
                values=[statistics.fmean(points[(policy,g)][metric][i] for g in cfg.GAMMAS) for i in range(64)]
                row.update({metric+'_'+k:v for k,v in probe.statistics_for(values).items()})
            means.append(row)
        return dict(step=step,cumulative_updates=22096+step,arm=arm,summary=summary,all_six=means,
            shared_inputs=inputs,alpha=cfg.training_alpha(arm),selection=False,test_used=False)
    finally:
        for module,mode in modes:module.training=mode
        restore_rng(torch,np,rng)
        if (ce_before!=tensor_digest(torch,chain.ce_model.state_dict()) or codec_before!=tensor_digest(torch,chain.codec_model.state_dict())
                or buffers!=tensor_digest(torch,dict(chain.named_buffers()))
                or chain.ce_model.training):raise ValueError('Monitor changed frozen CE/buffer state')


def validate_runtime_resume(torch,model,payload,arm,calibration,data_protocol,ce):
    cfg.validate_resume_metadata(payload,arm,calibration,code_sha256=source_code_record()['sha256'])
    task.previous.old._check_numerical_payload(torch,payload)
    if (payload.get('calibration_record')!=calibration or payload.get('calibration_sha256')!=sha256(cfg.CALIBRATION_PATH)
            or payload['ce_identity']!=ce or canonical(payload['data_protocol'])!=canonical(data_protocol)
            or payload.get('actual_condition_counts')!=expected_counts(arm,payload['step'])):
        raise ValueError('Matched resume calibration/data/condition counts changed')
    history=payload.get('training_input_history')
    if (not isinstance(history,str) or len(history)!=64 or any(c not in '0123456789abcdef' for c in history)
            or payload.get('training_input_history_steps')!=payload['step']):raise ValueError('Training input history missing')
    task.previous.old.check_tensor_state(torch,model,payload['codec_model'])
    params=list(model.parameters());group=payload['optimizer']['param_groups'][0]
    if group['params']!=list(range(len(params))) or not set(payload['optimizer']['state'])<=set(group['params']):raise ValueError('Resume optimizer parameter order changed')
    for index,state in payload['optimizer']['state'].items():
        if any(state[k].shape!=params[index].shape or state[k].dtype!=params[index].dtype for k in ('exp_avg','exp_avg_sq')):raise ValueError('Resume optimizer moment mismatch')
    model.load_state_dict(payload['codec_model'],strict=True)
    for entry in payload.get('monitor_records',{}).values():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Saved monitor evidence changed')
    if set(payload.get('monitor_records',{}))!={str(s) for s in cfg.PROGRESS_VALIDATION_STEPS if s<=payload['step']}:
        raise ValueError('Saved monitor trajectory incomplete')


def load_final_payload(torch,arm,path,calibration,data_protocol,ce):
    path=Path(path).resolve();expected=(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt').resolve()
    if path!=expected:raise ValueError('Explicit matched terminal checkpoint required')
    manifest=read_json(path.parent/'manifest.json')
    if manifest.get('status')!='completed' or sha256(path)!=manifest.get('checkpoint_sha256'):raise ValueError('Terminal completion receipt changed')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg.validate_resume_metadata(payload,arm,calibration,code_sha256=source_code_record()['sha256'])
    task.previous.old._check_numerical_payload(torch,payload)
    if (payload['step']!=12096 or payload['cumulative_updates']!=34192 or payload.get('calibration_record')!=calibration
            or payload.get('calibration_sha256')!=sha256(cfg.CALIBRATION_PATH) or payload['ce_identity']!=ce
            or canonical(payload['data_protocol'])!=canonical(data_protocol)
            or payload.get('actual_condition_counts')!=expected_counts(arm,12096)):
        raise ValueError('Matched final step/beta/data/condition counts differ')
    for key,value in dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=12096,cumulative_updates=34192,
        training_alpha=cfg.training_alpha(arm),actual_condition_counts=expected_counts(arm,12096),
        code=source_code_record(),calibration=cfg.calibration_identity(calibration),source=cfg.source_identity(arm),
        recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration)).items():
        if canonical(manifest.get(key))!=canonical(value):raise ValueError('Terminal manifest '+key+' changed')
    monitors=payload.get('monitor_records',{})
    if set(monitors)!={str(s) for s in cfg.PROGRESS_VALIDATION_STEPS}:raise ValueError('Four complete monitor records required')
    for entry in monitors.values():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Terminal monitor file changed')
    return payload,dict(identity=cfg.source_identity(arm),frozen_files={str(p):sha256(p) for p in (
        path,path.parent/'manifest.json',path.parent/'initialization.json',path.parent/'progress.jsonl',
        *[Path(entry['path']) for entry in monitors.values()])})


def train(arm,*,resume=False,stop_after=None):
    from train_proposal_as_receiver import run_lock
    torch,np,warmup=task.previous.old.runtime();device=torch.device('cuda:0');calibration=load_calibration()
    cal_hash=sha256(cfg.CALIBRATION_PATH);folder=cfg.OUTPUT_ROOT/'training'/arm
    owner=SimpleNamespace(EXP_NAME='k8_rzf_matched_'+arm,arm=arm,SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        if not resume and folder.exists():raise FileExistsError('Matched training output exists')
        if resume and not (folder/'checkpoint_latest.pt').exists():raise FileNotFoundError('Explicit resume needs latest')
        random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED);torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark=False
        data=make_data(warmup,device);ce=ce_identity();code=source_code_record();start=0
        sources=load_sources(torch,warmup,data.protocol,ce);method='source_fdma' if cfg.is_fdma(arm) else 'source_g4'
        source_payload,evidence=sources[method];monitors={};counts=Counter();history='0'*64
        if resume:
            manifest=read_json(folder/'manifest.json')
            if sha256(folder/'checkpoint_latest.pt')!=manifest.get('checkpoint_sha256'):raise ValueError('Latest receipt SHA changed')
            initial=read_json(folder/'initialization.json')
            for key,value in dict(source=cfg.source_identity(arm),code=code,calibration=cfg.calibration_identity(calibration),
                                  recipe=cfg.recipe(arm,calibration),ce_identity=ce).items():
                if canonical(initial.get(key))!=canonical(value):raise ValueError('Initialization '+key+' changed')
            payload=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            chain=build_chain(torch,device,arm,payload);validate_runtime_resume(torch,chain.codec_model,payload,arm,calibration,data.protocol,ce)
            start=payload['step'];monitors=payload['monitor_records'];counts=Counter(payload['actual_condition_counts'])
            history=payload['training_input_history']
            if history!=manifest.get('training_input_history'):raise ValueError('Resume input-history receipt changed')
        else:
            chain=build_chain(torch,device,arm,source_payload)
            task.previous.check_source_optimizer_parameters(torch,chain.codec_model,source_payload,ordered=False)
            if tensor_digest(torch,chain.codec_model.state_dict())!=tensor_digest(torch,source_payload['codec_model']):raise ValueError('Matched initialization did not exactly copy NMSE source')
        del sources,source_payload
        if start==12096:
            if manifest.get('error'):manifest.setdefault('failure_history',[]).append(dict(error=manifest.pop('error'),time=now()))
            manifest.update(status='completed',updated_utc=now(),completion_recovered=True,additional_updates=0)
            save_json(folder/('terminal_recovery_'+uuid.uuid4().hex[:8]+'.json'),manifest);save_json(folder/'manifest.json',manifest);return manifest
        model=chain.codec_model;optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        if resume:
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);restore_rng(torch,np,payload['rng']);del payload
        folder.mkdir(parents=True,exist_ok=True)
        if not resume:save_json(folder/'initialization.json',dict(source=cfg.source_identity(arm),source_evidence=evidence,
            code=code,calibration=cfg.calibration_identity(calibration),recipe=cfg.recipe(arm,calibration),ce_identity=ce,
            initialized_state_sha256=tensor_digest(torch,model.state_dict()),migration=False,new_optimizer=True,strict_loading=True))
        end=min(12096,start+stop_after) if stop_after else 12096
        manifest=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,status='running',step=start,cumulative_updates=22096+start,
            target=12096,source=cfg.source_identity(arm),code=code,calibration=cfg.calibration_identity(calibration),
            recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration),ce_identity=ce,
            parameters=task.previous.parameters(model),training_alpha=cfg.training_alpha(arm),monitor_records=monitors,
            actual_condition_counts=dict(counts),updated_utc=now(),test_used=False,selection='terminal12096 only')
        manifest['training_input_history']=history
        if resume:manifest['checkpoint_sha256']=sha256(folder/'checkpoint_latest.pt')
        save_json(folder/'manifest.json',manifest);model.train();ce_before=tensor_digest(torch,chain.ce_model.state_dict())
        try:
            for step in range(start+1,end+1):
                condition=cfg.schedule(arm,step);sample=data.sample('codec_train',condition['global_step']);hdl,hul=sample['H_dl'],sample['H_ul']
                gen=torch.Generator(device=device).manual_seed(cfg.training_noise_seed(step))
                pair=[warmup.standard_noise(4,gen,device) for _ in range(2)]
                out=task.previous.old._run(chain,warmup,hdl,hul,condition,*pair)
                metrics=chain.metrics(out,hdl,rzf_alpha=cfg.training_alpha(arm),downlink_snr_db=25.)
                nmse=metrics['dl_nmse_per_user'].mean();rate=metrics['sum_rate_bps_hz'].mean();loss=task_loss(nmse,rate,arm,step,calibration)
                if not bool(torch.isfinite(loss)&torch.isfinite(nmse)&torch.isfinite(rate)):raise FloatingPointError('Nonfinite matched objective')
                check_power(torch,out,metrics,64.-condition['pilot_energy_per_ue'])
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if chain.ce_model.training or any(p.grad is not None for p in chain.ce_model.parameters()):raise ValueError('CE freeze violated')
                optimizer.step();scheduler.step();counts[condition_key(condition)]+=1
                history=hashlib.sha256((history+str(condition['global_step'])+sample['sample_fingerprint']+
                    str(cfg.training_noise_seed(step))).encode()).hexdigest()
                if step in cfg.PROGRESS_VALIDATION_STEPS:
                    report=monitor(torch,np,warmup,chain,data,arm,step)
                    report.update(training_nmse=float(nmse.detach()),training_rate=float(rate.detach()),
                        beta=cfg.loss_weight(arm,step,calibration),training_grad_norm=float(grad))
                    target=folder/(f'monitor_step_{step:06d}_'+uuid.uuid4().hex[:8]+'.json');save_json(target,report)
                    monitors[str(step)]=dict(path=str(target),sha256=sha256(target))
                if step%108==0:
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite model parameter')
                    if ce_before!=tensor_digest(torch,chain.ce_model.state_dict()):raise ValueError('CE state changed')
                    if code!=source_code_record() or ce_identity()!=ce or sha256(cfg.CALIBRATION_PATH)!=cal_hash:raise ValueError('Frozen code/CE/new beta changed')
                    if dict(counts)!=expected_counts(arm,step):raise ValueError('Actual condition budget differs')
                    if step==end:load_calibration()
                    payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=step,cumulative_updates=22096+step,
                        codec_model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(torch,np),
                        source=cfg.source_identity(arm),recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration),
                        calibration=cfg.calibration_identity(calibration),calibration_record=calibration,calibration_sha256=cal_hash,
                        code_sha256=code['sha256'],data_protocol=data.protocol,ce_identity=ce,monitor_records=monitors,actual_condition_counts=dict(counts),
                        training_input_history=history,training_input_history_steps=step)
                    path=folder/'checkpoint_latest.pt';tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(path)
                    manifest.update(step=step,cumulative_updates=22096+step,status='completed' if step==12096 else 'paused' if step==end else 'running',
                        checkpoint_sha256=sha256(path),monitor_records=monitors,actual_condition_counts=dict(counts),
                        updated_utc=now(),last_loss=float(loss.detach()),last_nmse=float(nmse.detach()),last_rate=float(rate.detach()),
                        beta=cfg.loss_weight(arm,step,calibration),last_grad_norm=float(grad))
                    manifest['training_input_history']=history
                    save_json(folder/'manifest.json',manifest)
                    row=dict(step=step,cumulative_updates=22096+step,condition=condition,loss=float(loss.detach()),nmse=float(nmse.detach()),
                        rate=float(rate.detach()),beta=cfg.loss_weight(arm,step,calibration),grad_norm=float(grad),utc=now(),
                        monitor=monitors.get(str(step)),sample_fingerprint=sample['sample_fingerprint'],standard_noise_seed=cfg.training_noise_seed(step))
                    with (folder/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    print(json.dumps(row),flush=True)
            if end==12096:save_json(folder/'terminal_checkpoint.json',dict(path=str(folder/'checkpoint_latest.pt'),sha256=manifest['checkpoint_sha256'],
                local_step=12096,cumulative_updates=34192,note='latest is the fixed terminal state; no extra large checkpoint copy'))
        except BaseException as error:
            manifest.update(status='failed',error=f'{type(error).__name__}: {error}',updated_utc=now())
            save_json(folder/('failure_'+uuid.uuid4().hex[:8]+'.json'),manifest);save_json(folder/'manifest.json',manifest);raise
        return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=cfg.ARMS)
    mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','train','calibrate'):mode.add_argument('--'+name,action='store_true')
    mode.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(recipe=cfg.recipe(args.arm or cfg.ARMS[0]),models_loaded=False,weights_read=False),indent=2));return 0
    if args.calibrate:
        if args.report is None or args.resume or args.arm:parser.error('--calibrate requires an unused --report only')
        calibrate(args.report);return 0
    if args.report is not None:parser.error('--report requires --calibrate')
    if not(args.train or args.stop_after is not None):parser.print_help();return 0
    if args.arm is None:parser.error('Explicit --arm required')
    if args.stop_after is not None and (not 0<args.stop_after<=12096 or args.stop_after%108):parser.error('Pause must be108-step multiples')
    train(args.arm,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':raise SystemExit(main())
