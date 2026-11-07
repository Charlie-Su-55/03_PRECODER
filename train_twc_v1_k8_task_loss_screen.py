#!/usr/bin/env python3
"""Isolated K8 NMSE/task-loss continuation and one immutable gradient calibration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from types import SimpleNamespace
import subprocess

from configs import twc_v1_k8_task_loss_screen_config as cfg
from configs import twc_v1_k8_ordered_encoder_screen_config as source_cfg
import train_twc_v1_k8_ordered_encoder_screen as previous

ROOT=cfg.REPO_ROOT
canonical,sha256,save_json=previous.canonical,previous.sha256,previous.save_json
capture_rng,restore_rng=previous.capture_rng,previous.restore_rng
make_data,ce_identity,tensor_digest=previous.make_data,previous.ce_identity,previous.tensor_digest
SOURCE_FILES=tuple(dict.fromkeys(previous.SOURCE_FILES+(
    'configs/twc_v1_k8_task_loss_screen_config.py','train_twc_v1_k8_task_loss_screen.py',
    'evaluate_twc_v1_k8_task_loss_screen.py','run_twc_v1_k8_task_loss_screen.py')))
SOURCE_METHODS={'source_fdma':'ordered_fdma','source_g4':'ordered_g4'}
RULE='0.25 * median(all NMSE gradient norms) / median(all sum-rate gradient norms)'


def now():return datetime.now(timezone.utc).isoformat()


def source_code_record():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})


@lru_cache(maxsize=1)
def archived_source_hashes():
    return {name:hashlib.sha256(subprocess.check_output(
        ['git','show',cfg.SOURCE_COMMIT+':'+name],cwd=ROOT)).hexdigest() for name in previous.SOURCE_FILES}


def source_identity(method):
    arm={'source_fdma':'fdma_nmse','source_g4':'g4_nmse'}.get(method,method)
    return cfg.source_identity(arm)


def validate_source_metadata(payload,manifest,initial,method,*,archived_hashes):
    """The source implementation is b6dc7ed; its internal source is8096, not HEAD."""
    identity=source_identity(method);arm=identity['arm']
    source_cfg.validate_resume_metadata(payload,arm,code_sha256=archived_hashes)
    if payload['step']!=3024 or payload['cumulative_updates']!=16048:
        raise ValueError('Source local3024/cumulative16048 required')
    expected_code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived_hashes)
    expected=dict(format=source_cfg.CHECKPOINT_FORMAT,status='completed',arm=arm,seed=cfg.SEED,
        step=3024,cumulative_updates=16048,config_fingerprint=source_cfg.fingerprint(arm),
        checkpoint_sha256=identity['sha256'],source_commit=cfg.INTERNAL_SOURCE_COMMIT,
        new_experiment_commit=cfg.SOURCE_COMMIT,code=expected_code)
    for key,value in expected.items():
        if canonical(manifest.get(key))!=canonical(value):raise ValueError('Source manifest '+key+' changed')
    for key,value in dict(code=expected_code,source_commit=cfg.INTERNAL_SOURCE_COMMIT,
                         new_experiment_commit=cfg.SOURCE_COMMIT,new_optimizer=True).items():
        if canonical(initial.get(key))!=canonical(value):raise ValueError('Source initialization '+key+' changed')
    for key,value in (('recipe',source_cfg.recipe(arm)),('source',source_cfg.source_identity(arm))):
        if any(canonical(record.get(key))!=canonical(value) for record in (payload,manifest,initial)):
            raise ValueError('Source '+key+' metadata changed')
    if any(canonical(row.get('ce_identity'))!=canonical(payload['ce_identity']) for row in (manifest,initial)):
        raise ValueError('Source CE identity differs across evidence')
    return dict(implementation_commit=cfg.SOURCE_COMMIT,internal_source_commit=cfg.INTERNAL_SOURCE_COMMIT,
        source_arm=arm,step=3024,cumulative_updates=16048,new_HEAD_may_differ=True)


def source_batch_evidence(method):
    identity=source_identity(method);pointer=source_cfg.OUTPUT_ROOT/'last_batch.json'
    folder=source_cfg.inside_root(json.loads(pointer.read_text(encoding='utf-8'))['directory'])
    if not folder.is_relative_to((source_cfg.OUTPUT_ROOT/'batches').resolve()):raise ValueError('Source batch pointer escaped')
    summary_path=folder/'batch_summary.json';summary=json.loads(summary_path.read_text(encoding='utf-8'))
    if (summary.get('status')!='completed' or summary.get('error') or
            summary.get('code')!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived_source_hashes())
            or set(summary.get('training',{}))!=set(source_cfg.ARMS)):
        raise ValueError('Completed ordered-v2 source batch evidence required')
    stage=summary.get('stages',{}).get('train_'+identity['arm'],{})
    evidence=stage.get('evidence_sha256',{})
    record=summary['training'][identity['arm']]
    if (stage.get('status')!='completed' or not evidence or
            record.get('checkpoint_path')!=identity['path'] or record.get('checkpoint_sha256')!=identity['sha256']
            or record.get('step')!=3024 or record.get('cumulative_updates')!=16048):
        raise ValueError('Source batch checkpoint pointer or completion changed')
    required=[identity[key] for key in ('path','manifest_path','initialization_path','progress_path')]
    for path in required:
        if evidence.get(path)!=sha256(path):raise ValueError('Source batch evidence changed: '+path)
    # Terminal evaluation was part of the accepted source batch, not a new best selector.
    evaluation=summary.get('stages',{}).get('evaluation',{})
    if evaluation.get('status')!='completed' or not evaluation.get('evidence_sha256'):
        raise ValueError('Source batch terminal-validation receipt missing')
    for path,digest in evaluation['evidence_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Source validation evidence changed: '+path)
    return [pointer,summary_path,*map(Path,required),*map(Path,evaluation['evidence_sha256'])]


def load_sources(torch,warmup,data_protocol):
    expected=archived_source_hashes()
    for name,digest in expected.items():
        if sha256(ROOT/name)!=digest:raise ValueError('Accepted ordered-v2 source code changed: '+name)
    result={}
    for method in SOURCE_METHODS:
        identity=source_identity(method);path=Path(identity['path'])
        if sha256(path)!=identity['sha256']:raise ValueError('Pinned ordered-v2 weight SHA256 mismatch: '+str(path))
        payload=torch.load(path,map_location='cpu',weights_only=False)
        manifest=json.loads(Path(identity['manifest_path']).read_text(encoding='utf-8'))
        initial=json.loads(Path(identity['initialization_path']).read_text(encoding='utf-8'))
        verified=validate_source_metadata(payload,manifest,initial,method,archived_hashes=expected)
        previous.verify_completed_payload(torch,payload,identity['arm'])
        if canonical(payload['data_protocol'])!=canonical(data_protocol) or payload['ce_identity']!=ce_identity():
            raise ValueError('Source data/CE protocol changed')
        events=[json.loads(line) for line in Path(identity['progress_path']).read_text(encoding='utf-8').splitlines() if line.strip()]
        if ([row.get('step') for row in events]!=list(range(108,3025,108)) or
                any(row.get('cumulative_updates')!=13024+row['step'] for row in events)):
            raise ValueError('Source training log does not demonstrate completed3024 continuation')
        paths=source_batch_evidence(method)
        frozen={str(p):sha256(p) for p in paths}
        if frozen[str(path)]!=identity['sha256']:raise ValueError('Source changed while loading')
        result[method]=(payload,dict(identity=identity,verification=verified,frozen_files=frozen))
    return result


def build_chain(torch,device,arm,payload):
    """Reuse the accepted ordered codec's strict restore, never its migration."""
    source_arm=SOURCE_METHODS.get(arm,cfg.SOURCE_ARMS.get(arm))
    if source_arm is None:raise ValueError('Unknown fixed task-loss method')
    return previous.build_chain(torch,device,source_arm,payload,initializing_source=False)


def gradient_l2_norm(torch,gradients):
    """Norm over the full codec vector; None contributes exactly zero.

    FP64 accumulation is diagnostic only; model/gradient computation dtype is
    unchanged. No optimizer or gradient-scaling transform is applied.
    """
    gradients=list(gradients);present=[g for g in gradients if g is not None]
    if not present:raise ValueError('No effective gradient components')
    if any(not torch.is_tensor(g) or not bool(torch.isfinite(g).all()) for g in present):
        raise ValueError('Nonfinite gradient component')
    if len({g.device for g in present})!=1:raise ValueError('Gradient devices differ')
    total=sum((g.detach().abs().to(torch.float64).square().sum() for g in present),
              torch.zeros((),dtype=torch.float64,device=present[0].device))
    value=math.sqrt(float(total.item()))
    if not math.isfinite(value) or value<=0:raise ValueError('Finite nonzero effective gradient norm required')
    return value


def calibration_from_records(records):
    expected={(method,cfg.CALIBRATION_START_INDEX+i) for method in SOURCE_METHODS for i in range(12)}
    seen=set();paired={}
    if not isinstance(records,list) or len(records)!=24:raise ValueError('Exactly24 calibration rows required')
    for row in records:
        key=(row.get('source_key'),row.get('batch_index'))
        if key not in expected or key in seen:raise ValueError('Missing, duplicate or unexpected calibration address')
        seen.add(key);offset=key[1]-cfg.CALIBRATION_START_INDEX
        if row.get('gamma_db')!=cfg.GAMMAS[offset%6]:raise ValueError('Calibration gamma/address mismatch')
        for name in ('nmse_grad_norm','rate_grad_norm'):
            value=row.get(name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('Calibration norm is not finite and nonzero: '+name)
        fingerprint=tuple(row.get(name) for name in ('sample_fingerprint','pilot_sha256','data_sha256'))
        if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in fingerprint):
            raise ValueError('Calibration SHA256 input hashes missing or malformed')
        if key[1] in paired and paired[key[1]]!=fingerprint:raise ValueError('Calibration layouts did not share actual inputs')
        paired[key[1]]=fingerprint
    if seen!=expected:raise ValueError('Calibration condition coverage incomplete')
    nmse=statistics.median(row['nmse_grad_norm'] for row in records)
    rate=statistics.median(row['rate_grad_norm'] for row in records)
    beta=.25*nmse/rate
    if not math.isfinite(beta) or beta<=0:raise ValueError('Invalid calibrated beta')
    return dict(beta_target=beta,median_nmse_grad_norm=nmse,median_rate_grad_norm=rate,rule=RULE)


def validate_calibration_record(record):
    identity=cfg.calibration_identity(record)
    if (record.get('rule')!=RULE or record.get('implementation_commit')!=cfg.SOURCE_COMMIT
            or record.get('internal_source_commit')!=cfg.INTERNAL_SOURCE_COMMIT
            or not record.get('code',{}).get('sha256') or not record.get('data_protocol')
            or not record.get('frozen_files') or set(record.get('state_freeze',{}))!=set(SOURCE_METHODS)
            or any(row.get('passed') is not True for row in record['state_freeze'].values())):
        raise ValueError('Incomplete calibration source/state evidence')
    calculated=calibration_from_records(record.get('records'))
    for name,value in calculated.items():
        if record.get(name)!=value:raise ValueError('Calibration '+name+' differs from fixed rule')
    expected={method:source_identity(method) for method in SOURCE_METHODS}
    if canonical(record.get('sources'))!=canonical(expected):raise ValueError('Calibration source identities changed')
    return identity


def load_calibration():
    if not cfg.CALIBRATION_PATH.is_file():raise FileNotFoundError('Completed immutable calibration required')
    record=json.loads(cfg.CALIBRATION_PATH.read_text(encoding='utf-8'));validate_calibration_record(record)
    if record['code']!=source_code_record():raise ValueError('Calibration experiment code identity changed')
    for path,digest in record['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('Calibration frozen source evidence changed: '+path)
    return record


def write_once_json(path,value):
    path=cfg.inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    # An incomplete interrupted write remains evidence and cannot be overwritten.
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def backend(torch):
    return dict(torch=str(torch.__version__),cuda=torch.version.cuda,
        matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def calibration_noise_seed(offset,kind):
    if type(offset) is not int or not 0<=offset<12 or kind not in ('pilot','data'):raise ValueError('Calibration noise address invalid')
    return cfg.SEED*10_000_000+39_000_000+offset*10+int(kind=='data')


def calibrate(report_path):
    report_path=cfg.inside_root(report_path)
    if report_path.exists():raise FileExistsError(report_path)
    if cfg.CALIBRATION_PATH.exists():
        record=load_calibration();save_json(report_path,dict(status='passed',reused=True,record_id=record['record_id']))
        return record
    attempt=cfg.OUTPUT_ROOT/'calibration_attempt.json'
    if attempt.exists():raise RuntimeError('Incomplete prior calibration attempt retained; automatic recalibration refused')
    report=dict(status='running',records=[],sources={},state_freeze={},frozen_files={},rule=RULE)
    failure=None
    try:
        torch,np,warmup=previous.old.runtime();device=torch.device('cuda:0')
        data=make_data(warmup,device);data.prepare();ce=ce_identity();code=source_code_record()
        sources=load_sources(torch,warmup,data.protocol)
        # Source/environment rejection is not a calibration attempt. Once the
        # numerical pass starts, its immutable marker prevents silent reruns.
        write_once_json(attempt,dict(status='started',time=now(),code=code,rule=RULE))
        report.update(code=code,ce_identity=ce,data_protocol=data.protocol,backend=backend(torch),
            implementation_commit=cfg.SOURCE_COMMIT,internal_source_commit=cfg.INTERNAL_SOURCE_COMMIT)
        for entry in ce.values():report['frozen_files'][entry['path']]=entry['sha256']
        for method,(payload,evidence) in sources.items():
            report['sources'][method]=evidence;report['frozen_files'].update(evidence['frozen_files'])
            chain=build_chain(torch,device,method,payload).eval()
            model=chain.codec_model
            if any(not p.requires_grad for p in model.parameters()):raise ValueError('All codec parameters must be trainable')
            state_before=tensor_digest(torch,chain.state_dict())
            buffers_before=tensor_digest(torch,dict(chain.named_buffers()))
            expected=tensor_digest(torch,payload['codec_model'])
            if tensor_digest(torch,model.state_dict())!=expected:raise ValueError('Strict source copy differs, including window residuals')
            modes={name:module.training for name,module in chain.named_modules()}
            params=tuple(model.parameters())
            for offset in range(12):
                address=cfg.CALIBRATION_START_INDEX+offset;sample=data.sample('codec_train',address)
                Hdl,Hul=sample['H_dl'],sample['H_ul']
                pilot=warmup.standard_noise(4,torch.Generator(device=device).manual_seed(calibration_noise_seed(offset,'pilot')),device)
                noise=warmup.standard_noise(4,torch.Generator(device=device).manual_seed(calibration_noise_seed(offset,'data')),device)
                condition=dict(gamma_db=cfg.GAMMAS[offset%6],pilot_energy_per_ue=16.,
                    group_count=1 if method=='source_fdma' else 4,energy_ratio=None if method=='source_fdma' else (1,3))
                output=previous.old._run(chain,warmup,Hdl,Hul,condition,pilot,noise)
                metrics=chain.metrics(output,Hdl,rzf_alpha=.01,downlink_snr_db=25.)
                nmse=metrics['dl_nmse_per_user'].mean();rate=metrics['sum_rate_bps_hz'].mean()
                if not bool(torch.isfinite(nmse)&torch.isfinite(rate)):raise FloatingPointError('Nonfinite calibration objective')
                ng=torch.autograd.grad(nmse,params,retain_graph=True,allow_unused=True)
                nnorm=gradient_l2_norm(torch,ng);missing_nmse=sum(g is None for g in ng);del ng
                rg=torch.autograd.grad(rate,params,allow_unused=True)
                rnorm=gradient_l2_norm(torch,rg);missing_rate=sum(g is None for g in rg);del rg
                energy_error=float((output['energy']['actual_data_per_ue']-48).abs().max().item())
                power_error=float((metrics['rzf_per_subcarrier_power']-1).abs().max().item())
                if not math.isfinite(energy_error+power_error) or energy_error>1e-4 or power_error>1e-5:
                    raise ValueError('Calibration energy/RZF constraint failed')
                row=dict(source_key=method,batch_index=address,gamma_db=condition['gamma_db'],condition=condition,
                    nmse_grad_norm=nnorm,rate_grad_norm=rnorm,nmse_missing_gradients=missing_nmse,rate_missing_gradients=missing_rate,
                    trainable_parameter_tensors=len(params),sample_fingerprint=sample['sample_fingerprint'],
                    pilot_sha256=tensor_digest(torch,{'noise':pilot}),data_sha256=tensor_digest(torch,{'noise':noise}),
                    nmse=float(nmse.detach()),sum_rate=float(rate.detach()),energy_max_abs=energy_error,rzf_power_max_abs=power_error)
                report['records'].append(row);print(json.dumps(row),flush=True)
                del output,metrics,nmse,rate
            unchanged=state_before==tensor_digest(torch,chain.state_dict())
            buffers_same=buffers_before==tensor_digest(torch,dict(chain.named_buffers()))
            mode_same=modes=={name:module.training for name,module in chain.named_modules()}
            clean_grad=all(p.grad is None for p in chain.parameters())
            report['state_freeze'][method]=dict(passed=unchanged and buffers_same and mode_same and clean_grad and not chain.ce_model.training,
                state_before=state_before,state_after=tensor_digest(torch,chain.state_dict()),modes_unchanged=mode_same,
                buffer_sha256_before=buffers_before,buffer_sha256_after=tensor_digest(torch,dict(chain.named_buffers())),
                codec_model_sha256=expected,parameter_grad_buffers_empty=clean_grad,parameters=previous.parameters(model))
            if not report['state_freeze'][method]['passed']:raise ValueError('Calibration changed source/CE state or modes')
            del chain,model,params;torch.cuda.empty_cache()
        stats=calibration_from_records(report['records'])
        for path,digest in report['frozen_files'].items():
            if sha256(path)!=digest:raise ValueError('Calibration source evidence changed')
        if code!=source_code_record():raise ValueError('Experiment code changed during calibration')
        record=dict(format=cfg.CALIBRATION_FORMAT,status='passed',created=now(),**stats,records=report['records'],
            sources={method:source_identity(method) for method in SOURCE_METHODS},code=code,
            implementation_commit=cfg.SOURCE_COMMIT,internal_source_commit=cfg.INTERNAL_SOURCE_COMMIT,
            data_protocol=data.protocol,ce_identity=ce,state_freeze=report['state_freeze'],frozen_files=report['frozen_files'],
            backend=report['backend'],gradient_norm_accumulation='FP64 diagnostic accumulation; saved model dtype unchanged',
            optimizer_updates=0,test_used=False,recipe_candidates=1)
        record['record_id']=cfg.json_sha256(record);validate_calibration_record(record)
        write_once_json(cfg.CALIBRATION_PATH,record)
        report.update(status='passed',record_id=record['record_id'],**stats,finished=now())
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',finished=now())
    finally:save_json(report_path,report)
    if failure is not None:raise RuntimeError('Calibration failed; retained '+str(report_path)) from failure
    return record


def task_loss(nmse,rate,arm,step,calibration):
    weight=cfg.loss_weight(arm,step,calibration)
    # No rate term at all for controls; the NMSE scale remains exactly unchanged.
    return nmse if not cfg.is_joint(arm) else nmse-weight*rate


def validate_runtime_resume(torch,model,payload,arm,calibration,data_protocol,ce):
    cfg.validate_resume_metadata(payload,arm,calibration,code_sha256=source_code_record()['sha256'])
    previous.old._check_numerical_payload(torch,payload)
    if (canonical(payload.get('calibration_record'))!=canonical(calibration)
            or payload.get('calibration_sha256')!=sha256(cfg.CALIBRATION_PATH)
            or canonical(payload['data_protocol'])!=canonical(data_protocol) or payload['ce_identity']!=ce):
        raise ValueError('Resume calibration/data/CE identity changed')
    previous.old.check_tensor_state(torch,model,payload['codec_model'])
    parameters=list(model.parameters());group=payload['optimizer']['param_groups'][0]
    if group['params']!=list(range(len(parameters))) or not set(payload['optimizer']['state'])<=set(group['params']):
        raise ValueError('Resume optimizer parameter order changed')
    for index,state in payload['optimizer']['state'].items():
        for name in ('exp_avg','exp_avg_sq'):
            if state[name].shape!=parameters[index].shape or state[name].dtype!=parameters[index].dtype:
                raise ValueError('Resume optimizer moment shape/dtype mismatch')
    model.load_state_dict(payload['codec_model'],strict=True)


def load_final_payload(torch,arm,path,calibration,data_protocol,ce):
    expected=(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt').resolve();path=Path(path).resolve()
    if path!=expected:raise ValueError('Only explicit task-loss terminal checkpoints allowed')
    manifest=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='completed' or sha256(path)!=manifest.get('checkpoint_sha256'):
        raise ValueError('Completed terminal checkpoint receipt required')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg.validate_resume_metadata(payload,arm,calibration,code_sha256=source_code_record()['sha256'])
    previous.old._check_numerical_payload(torch,payload)
    if (payload['step']!=6048 or payload['cumulative_updates']!=22096 or
            payload.get('calibration_record')!=calibration or payload.get('calibration_sha256')!=sha256(cfg.CALIBRATION_PATH)
            or payload['ce_identity']!=ce or canonical(payload['data_protocol'])!=canonical(data_protocol)):
        raise ValueError('Terminal step/calibration/data/CE mismatch')
    for key,value in dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=6048,cumulative_updates=22096,
        code=source_code_record(),source=cfg.source_identity(arm),calibration=cfg.calibration_identity(calibration),
        recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration)).items():
        if canonical(manifest.get(key))!=canonical(value):raise ValueError('Terminal manifest '+key+' changed')
    return payload,dict(identity=cfg.source_identity(arm),checkpoint_sha256=sha256(path),
        frozen_files={str(p):sha256(p) for p in (path,path.parent/'manifest.json',path.parent/'initialization.json',path.parent/'progress.jsonl')})


def train(arm,*,resume=False,stop_after=None):
    from train_proposal_as_receiver import run_lock
    torch,np,warmup=previous.old.runtime();device=torch.device('cuda:0');calibration=load_calibration()
    calibration_hash=sha256(cfg.CALIBRATION_PATH)
    folder=cfg.OUTPUT_ROOT/'training'/arm;owner=SimpleNamespace(EXP_NAME='k8_task_loss_'+arm,arm=arm,SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        if not resume and folder.exists():raise FileExistsError('Training output exists')
        if resume and not (folder/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Resume requires latest')
        random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED);torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark=False
        data=make_data(warmup,device);ce=ce_identity();code=source_code_record();start=0
        source_method='source_fdma' if cfg.is_fdma(arm) else 'source_g4'
        sources=load_sources(torch,warmup,data.protocol);source,evidence=sources[source_method]
        if resume:
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if sha256(folder/'checkpoint_latest.pt')!=manifest.get('checkpoint_sha256'):raise ValueError('Latest checkpoint receipt changed')
            initial=json.loads((folder/'initialization.json').read_text(encoding='utf-8'))
            for key,value in dict(code=code,source=cfg.source_identity(arm),calibration=cfg.calibration_identity(calibration),
                                  recipe=cfg.recipe(arm,calibration),ce_identity=ce).items():
                if canonical(initial.get(key))!=canonical(value):raise ValueError('Initialization '+key+' changed')
            payload=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            chain=build_chain(torch,device,arm,payload)
            validate_runtime_resume(torch,chain.codec_model,payload,arm,calibration,data.protocol,ce);start=payload['step']
        else:
            chain=build_chain(torch,device,arm,source)
            if tensor_digest(torch,chain.codec_model.state_dict())!=tensor_digest(torch,source['codec_model']):
                raise ValueError('Initial model differs from source, including learned window residuals')
        del source,sources
        if start==6048:
            if manifest.get('error'):manifest.setdefault('failure_history',[]).append(dict(error=manifest.pop('error'),time=now()))
            manifest.update(status='completed',updated_utc=now(),completion_recovered=True,additional_updates=0)
            save_json(folder/('completion_recovery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'),manifest)
            save_json(folder/'manifest.json',manifest);return manifest
        model=chain.codec_model;optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,weight_decay=cfg.WEIGHT_DECAY)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        if resume:
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);restore_rng(torch,np,payload['rng']);del payload
        folder.mkdir(parents=True,exist_ok=True)
        if not resume:save_json(folder/'initialization.json',dict(source=cfg.source_identity(arm),source_evidence=evidence,
            code=code,calibration=cfg.calibration_identity(calibration),ce_identity=ce,recipe=cfg.recipe(arm,calibration),
            initialized_state_sha256=tensor_digest(torch,model.state_dict()),new_optimizer=True,strict_loading=True,migration=False))
        end=min(6048,start+stop_after) if stop_after else 6048
        manifest=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,status='running',step=start,cumulative_updates=16048+start,
            target=6048,source=cfg.source_identity(arm),code=code,calibration=cfg.calibration_identity(calibration),
            recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration),ce_identity=ce,
            parameters=previous.parameters(model),updated_utc=now(),checkpoint_selection='terminal6048 only',test_used=False)
        if resume:manifest['checkpoint_sha256']=sha256(folder/'checkpoint_latest.pt')
        save_json(folder/'manifest.json',manifest);model.train();ce_before=tensor_digest(torch,chain.ce_model.state_dict())
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for step in range(start+1,end+1):
                condition=cfg.schedule(arm,step);sample=data.sample('codec_train',condition['global_step']);Hdl,Hul=sample['H_dl'],sample['H_ul']
                generator=torch.Generator(device=device).manual_seed(cfg.training_noise_seed(step))
                output=previous.old._run(chain,warmup,Hdl,Hul,condition,warmup.standard_noise(4,generator,device),warmup.standard_noise(4,generator,device))
                metrics=chain.metrics(output,Hdl,rzf_alpha=.01,downlink_snr_db=25.)
                nmse=metrics['dl_nmse_per_user'].mean();rate=metrics['sum_rate_bps_hz'].mean()
                loss=task_loss(nmse,rate,arm,step,calibration)
                if not bool(torch.isfinite(loss)&torch.isfinite(nmse)&torch.isfinite(rate)):raise FloatingPointError('Nonfinite objective')
                if ((output['energy']['actual_data_per_ue']+condition['pilot_energy_per_ue']-64).abs().max().item()>1e-4
                        or (metrics['rzf_per_subcarrier_power']-1).abs().max().item()>1e-5):raise ValueError('Energy/RZF constraint failed')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if chain.ce_model.training or any(p.grad is not None for p in chain.ce_model.parameters()):raise ValueError('Frozen CE changed')
                optimizer.step();scheduler.step()
                if step%108==0:
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite model parameter')
                    if tensor_digest(torch,chain.ce_model.state_dict())!=ce_before:raise ValueError('CE state changed')
                    rng=capture_rng(torch,np)
                    monitor=previous.old.progress_validation(torch,warmup,chain,data,'fdma_control' if cfg.is_fdma(arm) else 'specialist_g4') if step%756==0 or step==end else None
                    restore_rng(torch,np,rng)
                    if code!=source_code_record() or ce_identity()!=ce or sha256(cfg.CALIBRATION_PATH)!=calibration_hash:
                        raise ValueError('Frozen code/CE/calibration identity changed')
                    if step==end:load_calibration()
                    payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=step,cumulative_updates=16048+step,
                        codec_model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(torch,np),
                        recipe=cfg.recipe(arm,calibration),config_fingerprint=cfg.fingerprint(arm,calibration),source=cfg.source_identity(arm),
                        code_sha256=code['sha256'],calibration=cfg.calibration_identity(calibration),calibration_record=calibration,
                        calibration_sha256=calibration_hash,ce_identity=ce,data_protocol=data.protocol)
                    path=folder/'checkpoint_latest.pt';temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(path)
                    manifest.update(step=step,cumulative_updates=16048+step,status='completed' if step==6048 else 'paused' if step==end else 'running',
                        checkpoint_sha256=sha256(path),updated_utc=now(),last_loss=float(loss.detach()),last_nmse=float(nmse.detach()),
                        last_rate=float(rate.detach()),beta=cfg.loss_weight(arm,step,calibration),last_grad_norm=float(norm),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
                    save_json(folder/'manifest.json',manifest)
                    row=dict(step=step,cumulative_updates=16048+step,utc=now(),condition=condition,loss=float(loss.detach()),
                        nmse=float(nmse.detach()),rate=float(rate.detach()),beta=cfg.loss_weight(arm,step,calibration),monitor=monitor,
                        sample_fingerprint=sample['sample_fingerprint'],pilot_data_noise_seed=cfg.training_noise_seed(step))
                    with (folder/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    print(json.dumps(row),flush=True)
        except BaseException as error:
            manifest.update(status='failed',error=f'{type(error).__name__}: {error}',updated_utc=now())
            save_json(folder/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'),manifest)
            save_json(folder/'manifest.json',manifest);raise
        return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=cfg.ARMS)
    modes=parser.add_mutually_exclusive_group()
    for name in ('dry-run','train','calibrate'):modes.add_argument('--'+name,action='store_true')
    modes.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:
        if args.resume or args.report:parser.error('Dry-run neither resumes nor writes reports')
        print(json.dumps(dict(recipe=cfg.recipe(args.arm or cfg.ARMS[0]),weights_read=False,models_built=False,calibration_generated=False),indent=2));return 0
    if args.calibrate:
        if args.report is None or args.arm or args.resume:parser.error('--calibrate requires only an unused --report path')
        calibrate(args.report);return 0
    if args.report is not None:parser.error('--report requires --calibrate')
    if not(args.train or args.stop_after is not None):parser.print_help();return 0
    if args.arm is None:parser.error('Explicit --arm required')
    if args.stop_after is not None and (not 0<args.stop_after<=6048 or args.stop_after%108):parser.error('Pause must be108-step multiples')
    train(args.arm,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':raise SystemExit(main())
