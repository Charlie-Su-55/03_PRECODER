#!/usr/bin/env python3
"""Independent BS-only decoder screen; source evidence is locked before training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
from types import SimpleNamespace
import uuid

from configs import twc_v1_k8_decoder_screen_config as cfg
import train_twc_v1_k8_rzf_matched_training as previous

ROOT=cfg.REPO_ROOT
canonical,sha256,save_json=previous.canonical,previous.sha256,previous.save_json
read_json,now=previous.read_json,previous.now
capture_rng,restore_rng=previous.capture_rng,previous.restore_rng
make_data,ce_identity,tensor_digest=previous.make_data,previous.ce_identity,previous.tensor_digest
SOURCE_METHODS=('source_fdma','source_g4')
NEW_FILES=('models/twc_v1_k8_decoder_refinement.py','utils/twc_v1_k8_decoder_residual.py',
    'configs/twc_v1_k8_decoder_screen_config.py','train_twc_v1_k8_decoder_screen.py',
    'evaluate_twc_v1_k8_decoder_screen.py','run_twc_v1_k8_decoder_screen.py',
    'tests/test_twc_v1_k8_decoder_screen.py','docs/TWC_V1_K8_DECODER_SCREEN.md','docs/TWC_V1_STATUS.md')
SOURCE_FILES=tuple(dict.fromkeys(previous.SOURCE_FILES+NEW_FILES))


def source_code_record():
    import subprocess
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})


def write_once_json(path,value):
    path=cfg.inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def verify_frozen(files):
    for path,digest in files.items():
        resolved=Path(path).resolve()
        if not resolved.is_relative_to(ROOT.resolve()):raise ValueError('Evidence path outside03 refused: '+path)
        if sha256(resolved)!=digest:raise ValueError('Frozen evidence changed: '+path)


def validate_source_records(payload,manifest,initial,terminal,events,batch,stage,arm,beta,archived,actual_sha):
    """Actual archived ad39017 restore schema, independent of the new HEAD."""
    old=previous.cfg
    old.validate_resume_metadata(payload,arm,beta,code_sha256=archived)
    source=old.source_identity(arm);recipe=old.recipe(arm,beta)
    code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)
    expected=dict(format=old.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=12096,
        cumulative_updates=34192,status='completed',source=source,recipe=recipe,
        config_fingerprint=old.fingerprint(arm,beta),code=code,calibration=old.calibration_identity(beta),
        checkpoint_sha256=actual_sha,training_alpha=old.training_alpha(arm),
        actual_condition_counts=previous.expected_counts(arm,12096),ce_identity=payload['ce_identity'])
    for name,value in expected.items():
        if canonical(manifest.get(name))!=canonical(value):raise ValueError('Source manifest '+name+' mismatch')
    for name,value in dict(source=source,recipe=recipe,code=code,calibration=old.calibration_identity(beta),
                           ce_identity=payload['ce_identity'],migration=False,new_optimizer=True,strict_loading=True).items():
        if canonical(initial.get(name))!=canonical(value):raise ValueError('Source initialization '+name+' mismatch')
    if (payload['step']!=12096 or payload['cumulative_updates']!=34192
            or canonical(payload.get('calibration_record'))!=canonical(beta)
            or payload.get('actual_condition_counts')!=previous.expected_counts(arm,12096)):
        raise ValueError('Source terminal step/calibration/conditions mismatch')
    if (terminal.get('local_step')!=12096 or terminal.get('cumulative_updates')!=34192
            or terminal.get('sha256')!=actual_sha or terminal.get('path')!=str(previous.cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')):
        raise ValueError('Source terminal receipt mismatch')
    if ([r.get('step') for r in events]!=list(range(108,12097,108))
            or any(r.get('cumulative_updates')!=22096+r['step']
                or canonical(r.get('condition'))!=canonical(old.schedule(arm,r['step']))
                or r.get('standard_noise_seed')!=old.training_noise_seed(r['step']) for r in events)):
        raise ValueError('Source complete training log/addresses mismatch')
    record=batch.get('training',{}).get(arm,{})
    if (batch.get('status')!='completed' or batch.get('error') or batch.get('code')!=code
            or set(batch.get('training',{}))!=set(old.ARMS) or stage.get('status')!='completed'
            or record.get('checkpoint_sha256')!=actual_sha or record.get('step')!=12096
            or record.get('cumulative_updates')!=34192 or record.get('checkpoint_path')!=terminal['path']):
        raise ValueError('Completed source batch/explicit checkpoint pointer mismatch')
    if (initial.get('initialized_state_sha256')!=beta['state_freeze']['source_fdma' if arm.startswith('fdma') else 'source_g4']['codec_model_sha256']
            or manifest.get('training_input_history')!=payload.get('training_input_history')
            or payload.get('training_input_history_steps')!=12096):
        raise ValueError('Source initialization/full input history mismatch')
    return dict(passed=True,implementation_commit=cfg.SOURCE_COMMIT,local_step=12096,
        source_codec_updates=34192,old_fingerprint=old.fingerprint(arm,beta),historical_beta_read_only=True)


def accept_sources(report_path):
    """Server-only cross-acceptance: no guessed SHA and no historical HEAD rewriting."""
    report_path=cfg.inside_root(report_path)
    if report_path.exists():raise FileExistsError(report_path)
    if cfg.SOURCE_ACCEPTANCE_PATH.exists():
        record=load_acceptance();save_json(report_path,dict(status='passed',reused=True,record_id=record['record_id']));return record
    report=dict(status='running',sources={},frozen_files={});failure=None
    try:
        torch,np,warmup=previous.task.previous.old.runtime();device=torch.device('cuda:0')
        data=make_data(warmup,device);ce=ce_identity()
        archived=previous.archived_hashes(cfg.SOURCE_COMMIT,tuple(previous.SOURCE_FILES))
        for name,digest in archived.items():
            if sha256(ROOT/name)!=digest:raise ValueError('Archived source implementation modified: '+name)
        old=previous.cfg;pointer=old.OUTPUT_ROOT/'last_batch.json'
        folder=old.inside_root(read_json(pointer)['directory'])
        batch=read_json(folder/'batch_summary.json')
        beta=read_json(old.CALIBRATION_PATH);previous.validate_calibration_record(beta)
        if (beta['code']!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)
                or canonical(beta['data_protocol'])!=canonical(data.protocol) or beta['ce_identity']!=ce):
            raise ValueError('Source beta/data/CE identity mismatch')
        # This reads the existing 596d2a4 and 0e0b6e8 chain without binding it to HEAD.
        ancestors=previous.load_sources(torch,warmup,data.protocol,ce)
        for _,evidence in ancestors.values():report['frozen_files'].update(evidence['frozen_files'])
        del ancestors
        verify_frozen(beta['frozen_files']);report['frozen_files'].update(beta['frozen_files'])
        for stage_name,stage in batch.get('stages',{}).items():
            if stage.get('status')!='completed' or not stage.get('evidence_sha256'):
                raise ValueError('Source batch unfinished stage: '+stage_name)
            verify_frozen(stage['evidence_sha256']);report['frozen_files'].update(stage['evidence_sha256'])
        for required in ('sources_and_new_beta','training_pairing_audit','terminal_alpha_selection','terminal_validation',
                         *('train_'+a for a in old.ARMS)):
            if required not in batch.get('stages',{}):raise ValueError('Missing source stage: '+required)
        for method in SOURCE_METHODS:
            arm='fdma_nmse' if method=='source_fdma' else 'g4_nmse';base=old.OUTPUT_ROOT/'training'/arm
            path=base/'checkpoint_latest.pt';actual=sha256(path)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            manifest=read_json(base/'manifest.json');initial=read_json(base/'initialization.json')
            terminal=read_json(base/'terminal_checkpoint.json')
            events=[json.loads(s) for s in (base/'progress.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
            stage=batch['stages']['train_'+arm]
            proof=validate_source_records(payload,manifest,initial,terminal,events,batch,stage,arm,beta,archived,actual)
            previous.task.previous.old._check_numerical_payload(torch,payload)
            if (canonical(payload['data_protocol'])!=canonical(data.protocol) or payload['ce_identity']!=ce
                    or payload.get('calibration_sha256')!=sha256(old.CALIBRATION_PATH)):
                raise ValueError('Source complete protocol/CE/beta SHA differs')
            chain=previous.build_chain(torch,torch.device('cpu'),arm,payload)
            previous.task.previous.check_source_optimizer_parameters(torch,chain.codec_model,payload,ordered=False)
            state_sha=tensor_digest(torch,chain.codec_model.state_dict());del chain
            frozen={}
            for name in ('checkpoint_latest.pt','manifest.json','initialization.json','progress.jsonl'):
                p=str(base/name)
                if stage['evidence_sha256'].get(p)!=sha256(p):raise ValueError('Source batch sidecar hash mismatch: '+p)
                frozen[p]=sha256(p)
            for entry in payload.get('monitor_records',{}).values():
                if sha256(entry['path'])!=entry['sha256']:raise ValueError('Source monitor changed')
                frozen[entry['path']]=entry['sha256']
            if set(payload.get('monitor_records',{}))!={str(s) for s in old.PROGRESS_VALIDATION_STEPS}:
                raise ValueError('Incomplete source monitor history')
            frozen[str(base/'terminal_checkpoint.json')]=sha256(base/'terminal_checkpoint.json')
            identity=dict(arm=arm,path=str(path),sha256=actual,manifest_path=str(base/'manifest.json'),
                initialization_path=str(base/'initialization.json'),progress_path=str(base/'progress.jsonl'),
                terminal_path=str(base/'terminal_checkpoint.json'),seed=cfg.SEED,local_step=12096,
                cumulative_updates=34192,source_implementation_commit=cfg.SOURCE_COMMIT)
            report['sources'][method]=dict(identity=identity,verification=dict(**proof,state_sha256=state_sha),frozen_files=frozen)
            report['frozen_files'].update(frozen);del payload
        for p in (pointer,folder/'batch_summary.json',old.CALIBRATION_PATH):
            report['frozen_files'][str(p)]=sha256(p)
        for item in ce.values():report['frozen_files'][item['path']]=item['sha256']
        report.update(format=cfg.SOURCE_ACCEPTANCE_FORMAT,status='passed',created=now(),code=source_code_record(),
            source_code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived),data_protocol=data.protocol,ce_identity=ce,
            backend=previous.task.backend(torch),no_checkpoint_written=True)
        record=dict(report);record['record_id']=cfg.json_sha256(record)
        cfg.acceptance_identity(record);verify_frozen(record['frozen_files'])
        write_once_json(cfg.SOURCE_ACCEPTANCE_PATH,record)
        report['record_id']=record['record_id']
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=__import__('traceback').format_exc())
    finally:save_json(report_path,report)
    if failure is not None:raise RuntimeError('Source acceptance failed: '+str(report_path)) from failure
    return record


def load_acceptance():
    record=read_json(cfg.SOURCE_ACCEPTANCE_PATH);cfg.acceptance_identity(record)
    if record.get('code')!=source_code_record():raise ValueError('Decoder-screen code changed after source lock')
    verify_frozen(record['frozen_files'])
    return record


def load_sources(torch,warmup,data_protocol,ce):
    record=load_acceptance()
    if canonical(record['data_protocol'])!=canonical(data_protocol) or record['ce_identity']!=ce:
        raise ValueError('Locked source data/CE protocol mismatch')
    result={}
    for method,item in record['sources'].items():
        path=item['identity']['path'];payload=torch.load(path,map_location='cpu',weights_only=False)
        if sha256(path)!=item['identity']['sha256']:raise ValueError('Source changed while loading')
        result[method]=(payload,item)
    return result


def freeze_training_parts(chain):
    chain.ce_model.requires_grad_(False).eval()
    chain.codec_model.encoder.requires_grad_(False).eval()
    for name,param in chain.codec_model.named_parameters():
        if not name.startswith('encoder.'):param.requires_grad_(True)
    return [(name,p) for name,p in chain.codec_model.named_parameters() if p.requires_grad]


def build_chain(torch,device,method,payload,*,include_stages=False,initialize=False):
    """Full old state is migrated only once; normal restores are strict."""
    from models.twc_v1_k8_decoder_refinement import build_codec,migrate_source_state
    import train_twc_v1_k8_codec_warmup as warmup
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg.SEED)
        chain=warmup.build_chain(torch.device('cpu'))
        codec=build_codec(variant='base2' if method in SOURCE_METHODS else cfg.variant(method),
            new_module_seed=cfg.SEED,include_stages=include_stages)
    chain.codec_model=codec
    if initialize:
        evidence=migrate_source_state(codec,payload['codec_model'])
        chain.migration_evidence=evidence
    else:
        previous.task.previous.old.check_tensor_state(torch,codec,payload['codec_model'])
        codec.load_state_dict(payload['codec_model'],strict=True)
        if tensor_digest(torch,codec.state_dict())!=tensor_digest(torch,payload['codec_model']):
            raise ValueError('Strict restore did not preserve every source value/shared decoder alias')
    if codec.decoder is not codec.joint_decoder.base_decoder:raise ValueError('Duplicated spatial decoder alias')
    freeze_training_parts(chain)
    return chain.to(device)


def frozen_part_digest(torch,chain):
    return dict(ue=tensor_digest(torch,chain.codec_model.encoder.state_dict()),
                ce=tensor_digest(torch,chain.ce_model.state_dict()))


def assert_frozen_parts(torch,chain,before,*,check_state=True):
    if ((check_state and before!=frozen_part_digest(torch,chain)) or chain.codec_model.encoder.training or chain.ce_model.training
            or any(p.requires_grad or p.grad is not None for module in (chain.codec_model.encoder,chain.ce_model)
                   for p in module.parameters())):raise ValueError('Frozen UE/condition/CE state, mode or gradient changed')


def condition_key(condition):return previous.condition_key(condition)


def expected_counts(arm,step):
    return dict(Counter(condition_key(cfg.schedule(arm,i)) for i in range(1,step+1)))


def monitor(torch,np,warmup,chain,data,arm,step):
    if step not in cfg.MONITOR_STEPS:raise ValueError('Undeclared monitor step')
    rng=capture_rng(torch,np);modes=[(m,m.training) for m in chain.modules()]
    state=tensor_digest(torch,chain.state_dict());buffers=tensor_digest(torch,dict(chain.named_buffers()))
    points={};inputs=[];device=next(chain.parameters()).device
    try:
        chain.eval()
        with torch.no_grad():
            for bi in range(16):
                sample=data.sample('validation',cfg.MONITOR_START_INDEX+bi);hdl,hul=sample['H_dl'],sample['H_ul']
                item=dict(address=cfg.MONITOR_START_INDEX+bi,channel_fingerprint=sample['sample_fingerprint'],noise={})
                for gi,gamma in enumerate(cfg.GAMMAS):
                    pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(cfg.noise_seed('monitor',bi,gi,k)),device) for k in ('pilot','data')]
                    item['noise'][str(gamma)]=[tensor_digest(torch,{'noise':x}) for x in pair]
                    for ratio in ((None,) if cfg.is_fdma(arm) else ((1,3),(1,1))):
                        policy='fdma' if ratio is None else 'hybrid_'+str(ratio[0])+'_'+str(ratio[1])
                        condition=dict(gamma_db=gamma,pilot_energy_per_ue=16.,group_count=1 if ratio is None else 4,energy_ratio=ratio)
                        out=previous.task.previous.old._run(chain,warmup,hdl,hul,condition,*pair)
                        metric=chain.metrics(out,hdl,rzf_alpha=.03,downlink_snr_db=25.)
                        previous.check_power(torch,out,metric,48.)
                        values=dict(dl_nmse=metric['dl_nmse_per_user'].mean(-1),sum_rate_bps_hz=metric['sum_rate_bps_hz'],
                            desired_signal_power=metric['desired_signal_power_mean'],interference_power=metric['interference_power_mean'])
                        target=points.setdefault((policy,gamma),{k:[] for k in values})
                        for key,value in values.items():
                            if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite monitor')
                            target[key].extend(value.cpu().tolist())
                inputs.append(item)
        rows=[];means=[]
        for (policy,gamma),values in points.items():
            if any(len(v)!=64 for v in values.values()):raise ValueError('Monitor sample count')
            row=dict(energy_policy=policy,gamma_db=gamma,num_samples=64,alpha=.03)
            for name,v in values.items():row.update({name+'_'+k:x for k,x in previous.probe.statistics_for(v).items()})
            rows.append(row)
        for policy in sorted({p for p,g in points}):
            row=dict(energy_policy=policy,num_samples=64,alpha=.03)
            for name in next(iter(points.values())):
                values=[statistics.fmean(points[(policy,g)][name][i] for g in cfg.GAMMAS) for i in range(64)]
                row.update({name+'_'+k:x for k,x in previous.probe.statistics_for(values).items()})
            means.append(row)
        return dict(arm=arm,step=step,new_bs_updates=step,source_codec_updates=34192,
            data_global_step=34192+step,alpha=.03,summary=rows,all_six=means,shared_inputs=inputs,selection=False)
    finally:
        for module,mode in modes:module.training=mode
        restore_rng(torch,np,rng)
        if state!=tensor_digest(torch,chain.state_dict()) or buffers!=tensor_digest(torch,dict(chain.named_buffers())):
            raise ValueError('Monitor changed model/buffers')


def validate_runtime_resume(torch,chain,payload,arm,acceptance,data_protocol,ce):
    cfg.validate_resume_metadata(payload,arm,acceptance,code_sha256=source_code_record()['sha256'])
    previous.task.previous.old._check_numerical_payload(torch,payload)
    if (payload.get('source_acceptance_sha256')!=sha256(cfg.SOURCE_ACCEPTANCE_PATH)
            or canonical(payload['data_protocol'])!=canonical(data_protocol) or payload['ce_identity']!=ce
            or payload.get('actual_condition_counts')!=expected_counts(arm,payload['step'])):
        raise ValueError('Decoder resume source lock/data/CE/counts differ')
    previous.task.previous.old.check_tensor_state(torch,chain.codec_model,payload['codec_model'])
    params=freeze_training_parts(chain);names=[name for name,_ in params]
    if names!=payload.get('optimizer_parameter_names'):raise ValueError('BS-only optimizer parameter identity changed')
    group=payload['optimizer']['param_groups'][0]
    if group['params']!=list(range(len(params))):raise ValueError('BS optimizer order changed')
    for i,state in payload['optimizer']['state'].items():
        if i not in group['params']:raise ValueError('Unknown optimizer state')
        for name in ('exp_avg','exp_avg_sq'):
            if state[name].shape!=params[i][1].shape or state[name].dtype!=params[i][1].dtype:
                raise ValueError('BS optimizer shape/dtype changed')
    if set(payload.get('monitor_records',{}))!={str(s) for s in cfg.MONITOR_STEPS if s<=payload['step']}:
        raise ValueError('Incomplete decoder monitor trajectory')
    for entry in payload.get('monitor_records',{}).values():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Monitor evidence changed')
    for name in ('training_input_history','layout_transmit_history'):
        value=payload.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing actual input/transmit rolling digest')
    chain.codec_model.load_state_dict(payload['codec_model'],strict=True)


def load_final_payload(torch,arm,*_unused):
    acceptance=load_acceptance();folder=cfg.OUTPUT_ROOT/'training'/arm
    manifest=read_json(folder/'manifest.json');path=folder/'checkpoint_latest.pt'
    if manifest.get('status')!='completed' or manifest.get('step')!=6048 or sha256(path)!=manifest.get('checkpoint_sha256'):
        raise ValueError('Complete fixed decoder terminal6048 required')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg.validate_resume_metadata(payload,arm,acceptance,code_sha256=source_code_record()['sha256'])
    previous.task.previous.old._check_numerical_payload(torch,payload)
    if payload['step']!=6048 or canonical(manifest.get('recipe'))!=canonical(cfg.recipe(arm,acceptance)):
        raise ValueError('Terminal decoder state/config mismatch')
    evidence=dict(identity=dict(arm=arm,path=str(path),sha256=sha256(path),step=6048,
        new_bs_updates=6048,source_codec_updates=34192),frozen_files={str(p):sha256(p) for p in
        (path,folder/'manifest.json',folder/'initialization.json',folder/'progress.jsonl',folder/'terminal_checkpoint.json')})
    return payload,evidence


def input_fingerprints(torch,out,sample,pair):
    return dict(channel=sample['sample_fingerprint'],
        pilot=tensor_digest(torch,{'noise':pair[0]}),data=tensor_digest(torch,{'noise':pair[1]}),
        X=tensor_digest(torch,{'X':out['tx']['X']}),Y=tensor_digest(torch,{'Y':out['Y_rx']}),
        Hest=tensor_digest(torch,{'Hest':out['H_ul_est']}))


def train(arm,*,resume=False,stop_after=None):
    from train_proposal_as_receiver import run_lock
    torch,np,warmup=previous.task.previous.old.runtime();device=torch.device('cuda:0')
    acceptance=load_acceptance();folder=cfg.OUTPUT_ROOT/'training'/arm
    owner=SimpleNamespace(EXP_NAME='k8_decoder_'+arm,arm=arm,SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        if not resume and folder.exists():raise FileExistsError(folder)
        if resume and not (folder/'checkpoint_latest.pt').exists():raise FileNotFoundError('Resume needs this arm latest')
        random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED);torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark=False
        data=make_data(warmup,device);ce=ce_identity();code=source_code_record()
        sources=load_sources(torch,warmup,data.protocol,ce);method='source_fdma' if cfg.is_fdma(arm) else 'source_g4'
        source,evidence=sources[method];start=0;monitors={};counts=Counter();history='0'*64;layout_history='0'*64
        if resume:
            initial=read_json(folder/'initialization.json');manifest=read_json(folder/'manifest.json')
            if sha256(folder/'checkpoint_latest.pt')!=manifest.get('checkpoint_sha256'):raise ValueError('Resume latest hash mismatch')
            for key,value in dict(code=code,source=cfg.source_identity(arm,acceptance),recipe=cfg.recipe(arm,acceptance),
                                  source_acceptance=cfg.acceptance_identity(acceptance),ce_identity=ce).items():
                if canonical(initial.get(key))!=canonical(value):raise ValueError('Resume initialization '+key+' changed')
            payload=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            chain=build_chain(torch,device,arm,payload)
            validate_runtime_resume(torch,chain,payload,arm,acceptance,data.protocol,ce)
            start=payload['step'];monitors=payload['monitor_records'];counts=Counter(payload['actual_condition_counts'])
            history=payload['training_input_history'];layout_history=payload['layout_transmit_history']
        else:
            chain=build_chain(torch,device,arm,source,initialize=True)
        frozen=frozen_part_digest(torch,chain)
        if frozen['ue']!=tensor_digest(torch,{k.removeprefix('encoder.'):v for k,v in source['codec_model'].items() if k.startswith('encoder.')}):
            raise ValueError('Source UE/condition embedding not fully preserved')
        del sources,source
        params=freeze_training_parts(chain);names=[n for n,p in params]
        optimizer=torch.optim.AdamW([p for n,p in params],lr=2e-4,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        if resume:
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);restore_rng(torch,np,payload['rng']);del payload
        if start==6048:
            # Recover only the small terminal receipt if interruption happened after latest.
            terminal=folder/'terminal_checkpoint.json'
            if not terminal.exists():write_once_json(terminal,dict(path=str(folder/'checkpoint_latest.pt'),
                sha256=manifest['checkpoint_sha256'],new_bs_updates=6048,source_codec_updates=34192,data_global_step=40240))
            if manifest.get('error'):
                manifest.setdefault('failure_history',[]).append(dict(error=manifest.pop('error'),time=now()))
            manifest.update(status='completed',updated_utc=now(),terminal_receipt_recovered=True,additional_updates=0)
            write_once_json(folder/('terminal_recovery_'+uuid.uuid4().hex[:8]+'.json'),manifest)
            save_json(folder/'manifest.json',manifest)
            return manifest
        folder.mkdir(parents=True,exist_ok=True)
        if not resume:
            extra=set(chain.migration_evidence['added_keys'])
            write_once_json(folder/'initialization.json',dict(code=code,source=cfg.source_identity(arm,acceptance),
                source_evidence=evidence,source_acceptance=cfg.acceptance_identity(acceptance),recipe=cfg.recipe(arm,acceptance),
                ce_identity=ce,migration=chain.migration_evidence,new_bs_optimizer=True,optimizer_parameter_names=names,
                new_state_sha256={n:tensor_digest(torch,{n:p}) for n,p in params if n in extra},
                frozen_state=frozen,initialized_state_sha256=tensor_digest(torch,chain.codec_model.state_dict())))
        end=min(6048,start+stop_after) if stop_after else 6048
        manifest=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,status='running',step=start,new_bs_updates=start,
            source_codec_updates=34192,data_global_step=34192+start,ue_frozen=True,target=6048,source=cfg.source_identity(arm,acceptance),
            code=code,source_acceptance=cfg.acceptance_identity(acceptance),recipe=cfg.recipe(arm,acceptance),
            config_fingerprint=cfg.fingerprint(arm,acceptance),ce_identity=ce,parameters=previous.task.previous.parameters(chain.codec_model),
            monitor_alpha=.03,monitor_records=monitors,actual_condition_counts=dict(counts),updated_utc=now(),
            optimizer_parameter_names=names,selection='fixed terminal6048; no best',test_used=False)
        if resume:manifest['checkpoint_sha256']=sha256(folder/'checkpoint_latest.pt')
        save_json(folder/'manifest.json',manifest)
        chain.train();freeze_training_parts(chain);assert_frozen_parts(torch,chain,frozen)
        try:
            for step in range(start+1,end+1):
                cond=cfg.schedule(arm,step);sample=data.sample('codec_train',cond['global_step'])
                gen=torch.Generator(device=device).manual_seed(cfg.training_noise_seed(step))
                pair=[warmup.standard_noise(4,gen,device) for _ in range(2)]
                out=previous.task.previous.old._run(chain,warmup,sample['H_dl'],sample['H_ul'],cond,*pair)
                metrics=chain.metrics(out,sample['H_dl'],rzf_alpha=.03,downlink_snr_db=25.)
                loss=metrics['dl_nmse_per_user'].mean();rate=metrics['sum_rate_bps_hz'].mean()
                if not bool(torch.isfinite(loss)&torch.isfinite(rate)):raise FloatingPointError('Nonfinite NMSE/rate')
                previous.check_power(torch,out,metrics,64.-cond['pilot_energy_per_ue'])
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for n,p in params],1.,error_if_nonfinite=True)
                assert_frozen_parts(torch,chain,frozen,check_state=False)
                grads={n:float(p.grad.detach().norm()) for n,p in params if p.grad is not None and
                    ('.joint_layers.2.' in n or '.joint_layers.3.' in n or '.second_residual_head.' in n or '.residual_projection.' in n)}
                optimizer.step();scheduler.step();counts[condition_key(cond)]+=1
                stamps=input_fingerprints(torch,out,sample,pair)
                history=hashlib.sha256((history+json.dumps({k:stamps[k] for k in ('channel','pilot','data')},sort_keys=True)).encode()).hexdigest()
                layout_history=hashlib.sha256((layout_history+json.dumps(stamps,sort_keys=True)).encode()).hexdigest()
                if step in cfg.MONITOR_STEPS:
                    report=monitor(torch,np,warmup,chain,data,arm,step)
                    report.update(training_nmse=float(loss.detach()),training_rate=float(rate.detach()),
                        beta=0.,training_grad_norm=float(norm))
                    target=folder/(f'monitor_step_{step:06d}_'+uuid.uuid4().hex[:8]+'.json');write_once_json(target,report)
                    monitors[str(step)]=dict(path=str(target),sha256=sha256(target))
                if step%108==0:
                    assert_frozen_parts(torch,chain,frozen)
                    if code!=source_code_record() or ce_identity()!=ce:raise ValueError('Code/CE changed during screen')
                    if step==end:verify_frozen(acceptance['frozen_files'])
                    if dict(counts)!=expected_counts(arm,step):raise ValueError('Condition update counts differ')
                    if any(not bool(torch.isfinite(p).all()) for n,p in params):raise FloatingPointError('Nonfinite BS parameter')
                    payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=step,
                        new_bs_updates=step,source_codec_updates=34192,data_global_step=34192+step,ue_frozen=True,
                        codec_model=chain.codec_model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        rng=capture_rng(torch,np),source=cfg.source_identity(arm,acceptance),recipe=cfg.recipe(arm,acceptance),
                        config_fingerprint=cfg.fingerprint(arm,acceptance),source_acceptance=cfg.acceptance_identity(acceptance),
                        source_acceptance_sha256=sha256(cfg.SOURCE_ACCEPTANCE_PATH),code_sha256=code['sha256'],
                        data_protocol=data.protocol,ce_identity=ce,monitor_records=monitors,actual_condition_counts=dict(counts),
                        optimizer_parameter_names=names,frozen_state=frozen,training_input_history=history,
                        layout_transmit_history=layout_history)
                    path=folder/'checkpoint_latest.pt';tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(path)
                    manifest.update(step=step,new_bs_updates=step,data_global_step=34192+step,
                        status='completed' if step==6048 else 'paused' if step==end else 'running',
                        checkpoint_sha256=sha256(path),monitor_records=monitors,actual_condition_counts=dict(counts),
                        training_input_history=history,layout_transmit_history=layout_history,updated_utc=now(),
                        last_nmse=float(loss.detach()),last_rate=float(rate.detach()),last_grad_norm=float(norm),new_branch_gradients=grads,
                        new_branch_state={n:tensor_digest(torch,{n:p}) for n,p in params if n in grads})
                    save_json(folder/'manifest.json',manifest)
                    row=dict(step=step,new_bs_updates=step,source_codec_updates=34192,data_global_step=34192+step,
                        condition=cond,nmse=float(loss.detach()),rate=float(rate.detach()),beta=0.,grad_norm=float(norm),
                        new_branch_gradients=grads,input_fingerprints=stamps,utc=now(),monitor=monitors.get(str(step)))
                    with (folder/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    print(json.dumps(row),flush=True)
            if end==6048:write_once_json(folder/'terminal_checkpoint.json',dict(path=str(folder/'checkpoint_latest.pt'),
                sha256=manifest['checkpoint_sha256'],new_bs_updates=6048,source_codec_updates=34192,data_global_step=40240))
        except BaseException as error:
            manifest.update(status='failed',error=f'{type(error).__name__}: {error}',updated_utc=now())
            save_json(folder/('failure_'+uuid.uuid4().hex[:8]+'.json'),manifest);save_json(folder/'manifest.json',manifest);raise
        return manifest


def preflight(report_path):
    """Real B4 migration, unchanged full-chain tolerances, frozen-UE BS gradients."""
    report_path=cfg.inside_root(report_path)
    if report_path.exists():raise FileExistsError(report_path)
    report=dict(status='running',checks={},gradient_checks={},memory={},parameters={},errors=[]);failure=None
    try:
        torch,np,warmup=previous.task.previous.old.runtime();device=torch.device('cuda:0')
        from models.twc_v1_k8_ordered_encoder import comparison_error
        from models.twc_v1_k8_decoder_refinement import residual_ops
        from train_twc_v1_k8_ordered_encoder_screen import EQUIVALENCE_TOLERANCES as tol
        acceptance=load_acceptance();data=make_data(warmup,device);data.prepare()
        report.update(code=source_code_record(),backend=previous.task.backend(torch),tolerances=tol,
            source_acceptance=cfg.acceptance_identity(acceptance),batch_size=4,optimizer_updates=0)
        small_g=torch.diag(torch.tensor([1.+2.j,2.-1.j],device=device,dtype=torch.complex64))
        small_r=torch.tensor([[.2+.7j],[-.3+.1j]],device=device,dtype=torch.complex64)
        expected=(small_g.diagonal().conj()*small_r[:,0])/(small_g.diagonal().abs().square()+.03)
        actual=residual_ops.solve_matched_residual(small_g,small_r,.03)
        report['complex_linear_check']=comparison_error(actual,expected,rtol=1e-6,atol=1e-6)
        if not report['complex_linear_check']['passed']:report['errors'].append('known complex conjugate/axes solve')
        sources=load_sources(torch,warmup,data.protocol,ce_identity())
        for method,(payload,evidence) in sources.items():
            source=previous.build_chain(torch,device,method,payload).eval()
            before=tensor_digest(torch,source.state_dict())
            sample=data.sample('codec_train',500200);hdl,hul=sample['H_dl'],sample['H_ul']
            for arm in (a for a in cfg.ARMS if cfg.is_fdma(a)==cfg.is_fdma(method)):
                torch.cuda.reset_peak_memory_stats(device)
                chain=build_chain(torch,device,arm,payload,initialize=True,include_stages=True).eval()
                frozen=frozen_part_digest(torch,chain)
                report['parameters'][arm]=previous.task.previous.parameters(chain.codec_model)
                report['checks'][arm]={}
                for gi,gamma in enumerate((0.,10.,25.)):
                    for ratio in ((None,) if cfg.is_fdma(arm) else ((1,3),(1,1))):
                        key=str(gamma)+'/'+str(ratio);cond=dict(gamma_db=gamma,pilot_energy_per_ue=16.,
                            group_count=1 if ratio is None else 4,energy_ratio=ratio)
                        pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(
                            cfg.noise_seed('preflight',0,gi,k)),device) for k in ('pilot','data')]
                        rng=capture_rng(torch,np)
                        with torch.no_grad():
                            oldout=previous.task.previous.old._run(source,warmup,hdl,hul,cond,*pair)
                            oldm=source.metrics(oldout,hdl,rzf_alpha=.03,downlink_snr_db=25.)
                            restore_rng(torch,np,rng)
                            out=previous.task.previous.old._run(chain,warmup,hdl,hul,cond,*pair)
                            met=chain.metrics(out,hdl,rzf_alpha=.03,downlink_snr_db=25.)
                            repeat=chain.metrics(out,hdl,rzf_alpha=.03,downlink_snr_db=25.)
                        checks={}
                        for name,actual,expected,t in (
                            ('transmit',out['tx']['X'],oldout['tx']['X'],'transmit'),
                            ('H_hat',out['H_dl_hat'],oldout['H_dl_hat'],'H_hat'),
                            ('RZF',met['precoder'],oldm['precoder'],'RZF'),
                            ('NMSE',met['dl_nmse_per_user'],oldm['dl_nmse_per_user'],'dl_nmse'),
                            ('rate',met['sum_rate_bps_hz'],oldm['sum_rate_bps_hz'],'rate')):
                            checks[name]=comparison_error(actual,expected,**tol[t])
                        checks['RZF_repeat']=dict(passed=bool(torch.equal(met['precoder'],repeat['precoder'])))
                        report['checks'][arm][key]=checks
                        try:checks['power']=dict(passed=True,**previous.check_power(torch,out,met,48.))
                        except (ValueError,FloatingPointError) as error:checks['power']=dict(passed=False,error=str(error))
                        checks['execution']={k:v for k,v in out['receiver_execution'].items() if k!='stages'}
                        report['checks'][arm][key]=checks
                        if any(v.get('passed') is False for v in checks.values()):
                            report['errors'].append(arm+'/'+key+' numerical identity')
                chain.train();freeze_training_parts(chain);chain.zero_grad(set_to_none=True)
                cond=dict(gamma_db=10.,pilot_energy_per_ue=16.,group_count=1 if cfg.is_fdma(arm) else 4,
                    energy_ratio=None if cfg.is_fdma(arm) else (1,3))
                pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(
                    cfg.noise_seed('preflight',0,1,k)),device) for k in ('pilot','data')]
                out=previous.task.previous.old._run(chain,warmup,hdl,hul,cond,*pair)
                loss=chain.metrics(out,hdl,rzf_alpha=.03,downlink_snr_db=25.)['dl_nmse_per_user'].mean();loss.backward()
                params=freeze_training_parts(chain)
                active={n:float(p.grad.norm()) for n,p in params if p.grad is not None}
                if any(not bool(torch.isfinite(p.grad).all()) for n,p in params if p.grad is not None):
                    raise FloatingPointError('Nonfinite BS preflight gradient')
                target=[v for n,v in active.items() if ('.second_residual_head.' in n if cfg.variant(arm)=='dc2'
                    else ('.joint_layers.2.' in n or '.joint_layers.3.' in n) if cfg.variant(arm)=='deep4'
                    else '.residual_head.' in n)]
                passed=bool(target) and any(v>0 for v in target)
                assert_frozen_parts(torch,chain,frozen)
                report['gradient_checks'][arm]=dict(passed=passed,loss=float(loss.detach()),
                    new_output_gradients={n:v for n,v in active.items() if '.second_residual_head.' in n or
                        '.joint_layers.2.' in n or '.joint_layers.3.' in n},frozen_UE_CE=True,
                    zero_head_upstream_zero_allowed=True)
                if not passed:report['errors'].append(arm+' output-head gradient')
                report['memory'][arm]=dict(batch=4,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),includes_reference_model=True)
                del chain,out,loss,met,oldout,oldm,repeat;torch.cuda.empty_cache()
            if before!=tensor_digest(torch,source.state_dict()):raise ValueError('Source mutated during preflight')
            del source;torch.cuda.empty_cache()
        for family in ('fdma','g4'):
            base=report['parameters'][family+'_base2']['full'];dc=report['parameters'][family+'_dc2']['full']
            if dc-base!=514:report['errors'].append(family+' DC2 actual parameter delta !=514')
        verify_frozen(acceptance['frozen_files'])
        if report['errors']:raise ValueError('Preflight failed checks: '+str(report['errors']))
        report.update(status='passed',finished=now())
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=__import__('traceback').format_exc())
    finally:save_json(report_path,report)
    if failure is not None:raise RuntimeError('Decoder CUDA preflight failed: '+str(report_path)) from failure
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=cfg.ARMS)
    mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','train','accept-sources','preflight'):mode.add_argument('--'+name,action='store_true')
    mode.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(recipe=cfg.recipe(args.arm or cfg.ARMS[0]),models_built=False,weights_read=False),indent=2));return 0
    if args.accept_sources or args.preflight:
        if args.report is None or args.resume or args.arm:parser.error('Check requires an unused --report only')
        (accept_sources if args.accept_sources else preflight)(args.report);return 0
    if args.report is not None:parser.error('--report requires a check')
    if not(args.train or args.stop_after is not None):parser.print_help();return 0
    if args.arm is None:parser.error('--arm required')
    if args.stop_after is not None and (not 0<args.stop_after<=6048 or args.stop_after%108):parser.error('stop-after must be a108 boundary')
    train(args.arm,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':raise SystemExit(main())
