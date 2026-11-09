#!/usr/bin/env python3
"""Independent physical-head/semantic-CSI training with two disjoint optimizers."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

from configs import twc_v1_k8_physical_split_config as cfg
import train_twc_v1_k8_decoder_screen as old

ROOT=cfg.REPO_ROOT
canonical,sha256,save_json,read_json,now=old.canonical,old.sha256,old.save_json,old.read_json,old.now
capture_rng,restore_rng=old.capture_rng,old.restore_rng
make_data,ce_identity,tensor_digest=old.make_data,old.ce_identity,old.tensor_digest
runtime=old.previous.task.previous.old.runtime
run_forward=old.previous.task.previous.old._run
NEW_FILES=('models/twc_v1_k8_physical_split.py','configs/twc_v1_k8_physical_split_config.py',
    'train_twc_v1_k8_physical_split.py','evaluate_twc_v1_k8_physical_split.py','run_twc_v1_k8_physical_split.py',
    'tests/test_twc_v1_k8_physical_split.py','tests/test_twc_v1_k8_physical_split_batch.py',
    'cleanup_twc_v1_k8.py','tests/test_cleanup_twc_v1_k8.py','docs/TWC_V1_K8_PHYSICAL_SPLIT.md','docs/TWC_V1_STATUS.md')
SOURCE_FILES=tuple(dict.fromkeys(old.SOURCE_FILES+NEW_FILES))
HEAD_PREFIX='joint_decoder.physical_head.'
BRIDGE_PREFIX='joint_decoder.physical_bridge.'


def source_code_record():
    import subprocess
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})


def write_once_json(path,value):
    path=cfg.inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


verify_frozen=old.verify_frozen


def validate_archived_source(payload,manifest,initial,terminal,events,batch,stage,arm,acceptance,archived,actual_sha):
    """The b6336de schema is checked against its own complete archived identity."""
    old.cfg.validate_resume_metadata(payload,arm,acceptance,code_sha256=archived)
    code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)
    for name,value in dict(format=old.cfg.CHECKPOINT_FORMAT,status='completed',arm=arm,seed=cfg.SEED,
        step=6048,new_bs_updates=6048,source_codec_updates=34192,data_global_step=40240,ue_frozen=True,
        code=code,recipe=old.cfg.recipe(arm,acceptance),config_fingerprint=old.cfg.fingerprint(arm,acceptance),
        source=old.cfg.source_identity(arm,acceptance),source_acceptance=old.cfg.acceptance_identity(acceptance),
        checkpoint_sha256=actual_sha,ce_identity=payload['ce_identity'],actual_condition_counts=old.expected_counts(arm,6048)).items():
        if canonical(manifest.get(name))!=canonical(value):raise ValueError('Historical decoder manifest '+name+' changed')
    for name,value in dict(code=code,source=old.cfg.source_identity(arm,acceptance),
        recipe=old.cfg.recipe(arm,acceptance),source_acceptance=old.cfg.acceptance_identity(acceptance),
        ce_identity=payload['ce_identity'],new_bs_optimizer=True).items():
        if canonical(initial.get(name))!=canonical(value):raise ValueError('Historical initialization '+name+' changed')
    if (initial.get('migration',{}).get('strict') is not True or initial['migration'].get('source_values_preserved') is not True
            or payload['step']!=6048 or payload.get('actual_condition_counts')!=old.expected_counts(arm,6048)
            or payload.get('training_input_history')!=manifest.get('training_input_history')
            or payload.get('layout_transmit_history')!=manifest.get('layout_transmit_history')):
        raise ValueError('Historical migration/step/input-history evidence changed')
    path=str(old.cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')
    if terminal!=dict(path=path,sha256=actual_sha,new_bs_updates=6048,source_codec_updates=34192,data_global_step=40240):
        raise ValueError('Historical decoder terminal receipt differs')
    if ([row.get('step') for row in events]!=list(range(108,6049,108))
            or any(row.get('data_global_step')!=34192+row['step'] or
                   canonical(row.get('condition'))!=canonical(old.cfg.schedule(arm,row['step'])) for row in events)):
        raise ValueError('Historical decoder complete progress coverage differs')
    item=batch.get('training',{}).get(arm,{})
    if (batch.get('status')!='completed' or batch.get('error') or batch.get('code')!=code
            or set(batch.get('training',{}))!=set(old.cfg.ARMS) or stage.get('status')!='completed'
            or item.get('checkpoint_path')!=path or item.get('checkpoint_sha256')!=actual_sha or item.get('step')!=6048):
        raise ValueError('Completed historical batch/precise source pointer required')
    if set(payload.get('monitor_records',{}))!={str(s) for s in old.cfg.MONITOR_STEPS}:
        raise ValueError('Historical decoder monitor trajectory incomplete')
    return dict(passed=True,source_implementation_commit=cfg.SOURCE_COMMIT,local_step=6048,
        source_codec_updates=34192,source_bs_updates=6048,source_data_address=40240)


def check_optimizer_tensors(torch,payload,key,parameters,updates):
    """Reuse the unchanged AdamW/constant-scheduler/RNG numerical validator."""
    optimizer=payload[key+'_optimizer'];scheduler=payload[key+'_scheduler']
    shadow=dict(step=updates,optimizer=optimizer,scheduler=scheduler,rng=payload['rng'])
    old.previous.task.previous.old._check_numerical_payload(torch,shadow)
    names=[n for n,p in parameters];params=[p for n,p in parameters]
    if payload['optimizer_parameter_names'].get(key)!=names:raise ValueError('Optimizer '+key+' names/order changed')
    indices=list(range(len(params)))
    if optimizer['param_groups'][0]['params']!=indices:raise ValueError('Optimizer '+key+' parameter index mismatch')
    for i,moments in optimizer['state'].items():
        if i not in indices:raise ValueError('Unknown '+key+' optimizer parameter')
        for name in ('exp_avg','exp_avg_sq'):
            if moments[name].shape!=params[i].shape or moments[name].dtype!=params[i].dtype:
                raise ValueError('Optimizer '+key+' moment shape/dtype mismatch')


def accept_sources(report_path):
    report_path=cfg.inside_root(report_path)
    if report_path.exists():raise FileExistsError(report_path)
    if cfg.SOURCE_ACCEPTANCE_PATH.exists():
        record=load_acceptance();save_json(report_path,dict(status='passed',reused=True,record_id=record['record_id']));return record
    report=dict(status='running',sources={},frozen_files={});failure=None
    try:
        torch,np,warmup=runtime();device=torch.device('cuda:0');data=make_data(warmup,device);ce=ce_identity()
        archived=old.previous.archived_hashes(cfg.SOURCE_COMMIT,tuple(old.SOURCE_FILES))
        for name,digest in archived.items():
            # STATUS is explicitly updated for the new task; its old hash remains
            # in historical records and is never rewritten to the current value.
            if name!='docs/TWC_V1_STATUS.md' and sha256(ROOT/name)!=digest:
                raise ValueError('Accepted decoder source code modified: '+name)
        historical=read_json(old.cfg.SOURCE_ACCEPTANCE_PATH);old.cfg.acceptance_identity(historical)
        if historical['code']!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived):
            raise ValueError('Historical source lock belongs to a different implementation')
        if canonical(historical['data_protocol'])!=canonical(data.protocol) or historical['ce_identity']!=ce:
            raise ValueError('Historical data/CE identity differs')
        verify_frozen(historical['frozen_files']);report['frozen_files'].update(historical['frozen_files'])
        pointer=old.cfg.OUTPUT_ROOT/'last_batch.json';folder=old.cfg.inside_root(read_json(pointer)['directory'])
        batch=read_json(folder/'batch_summary.json')
        for required in ('sources','cuda_preflight','training_audit','terminal_alpha_selection','terminal_validation',
                         *('train_'+arm for arm in old.cfg.ARMS)):
            stage=batch.get('stages',{}).get(required,{})
            if stage.get('status')!='completed' or not stage.get('evidence_sha256'):
                raise ValueError('Original decoder batch incomplete: '+required)
            verify_frozen(stage['evidence_sha256']);report['frozen_files'].update(stage['evidence_sha256'])
        # Independently read the old alpha lock; do not call its HEAD-bound loader.
        import evaluate_twc_v1_k8_decoder_screen as old_eval
        selection=read_json(old.cfg.SELECTION_PATH);old_eval.validate_selection_record(selection)
        if selection['code']!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived):
            raise ValueError('Historical final alpha identity differs')
        verify_frozen(selection['frozen_files'])
        ues={}
        for method in cfg.SOURCE_METHODS:
            identity=cfg.planned_source_identity(method);arm=identity['arm'];path=Path(identity['path']);base=path.parent
            actual=sha256(path);payload=torch.load(path,map_location='cpu',weights_only=False)
            manifest=read_json(base/'manifest.json');initial=read_json(base/'initialization.json');terminal=read_json(base/'terminal_checkpoint.json')
            events=[json.loads(line) for line in (base/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            stage=batch['stages']['train_'+arm]
            proof=validate_archived_source(payload,manifest,initial,terminal,events,batch,stage,arm,historical,archived,actual)
            if (payload.get('source_acceptance_sha256')!=sha256(old.cfg.SOURCE_ACCEPTANCE_PATH)
                    or canonical(payload['data_protocol'])!=canonical(data.protocol) or payload['ce_identity']!=ce):
                raise ValueError('Historical checkpoint complete source/data/CE binding differs')
            old.previous.task.previous.old._check_numerical_payload(torch,payload)
            chain=old.build_chain(torch,torch.device('cpu'),arm,payload,include_stages=False)
            params=old.freeze_training_parts(chain);names=[n for n,p in params]
            if names!=payload.get('optimizer_parameter_names'):raise ValueError('Original BS optimizer identity differs')
            group=payload['optimizer']['param_groups'][0]
            if group['params']!=list(range(len(params))):raise ValueError('Original optimizer index order differs')
            for i,moments in payload['optimizer']['state'].items():
                if i not in group['params']:raise ValueError('Unknown original optimizer index')
                if any(moments[k].shape!=params[i][1].shape or moments[k].dtype!=params[i][1].dtype for k in ('exp_avg','exp_avg_sq')):
                    raise ValueError('Original optimizer shape/dtype differs')
            parts=frozen_part_digest(torch,chain);ue=parts['ue']
            if parts!=payload.get('frozen_state'):raise ValueError('Original frozen UE/CE state differs')
            family='fdma' if cfg.is_fdma(method) else 'g4'
            if family in ues and ues[family]!=ue:raise ValueError('Old controls of one layout have different UE states')
            ues[family]=ue
            state_sha=tensor_digest(torch,chain.codec_model.state_dict());del chain
            frozen={}
            for name in ('checkpoint_latest.pt','manifest.json','initialization.json','progress.jsonl','terminal_checkpoint.json'):
                p=str(base/name)
                if stage['evidence_sha256'].get(p)!=sha256(p):raise ValueError('Original batch artifact SHA mismatch: '+p)
                frozen[p]=sha256(p)
            for entry in payload['monitor_records'].values():
                if sha256(entry['path'])!=entry['sha256']:raise ValueError('Original monitor changed')
                frozen[entry['path']]=entry['sha256']
            identity['sha256']=actual
            report['sources'][method]=dict(identity=identity,verification=dict(**proof,state_sha256=state_sha,ue_sha256=ue),frozen_files=frozen)
            report['frozen_files'].update(frozen);del payload
        for p in (pointer,folder/'batch_summary.json',old.cfg.SOURCE_ACCEPTANCE_PATH,old.cfg.SELECTION_PATH):
            report['frozen_files'][str(p)]=sha256(p)
        for item in ce.values():report['frozen_files'][item['path']]=item['sha256']
        report.update(format=cfg.SOURCE_ACCEPTANCE_FORMAT,status='passed',created=now(),code=source_code_record(),
            source_code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived),current_STATUS_is_new_task=True,
            data_protocol=data.protocol,ce_identity=ce,backend=old.previous.task.backend(torch),no_checkpoint_written=True)
        record=dict(report);record['record_id']=cfg.json_sha256(record);cfg.acceptance_identity(record)
        verify_frozen(record['frozen_files']);write_once_json(cfg.SOURCE_ACCEPTANCE_PATH,record);report['record_id']=record['record_id']
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=__import__('traceback').format_exc())
    finally:save_json(report_path,report)
    if failure is not None:raise RuntimeError('Physical-split source acceptance failed: '+str(report_path)) from failure
    return record


def load_acceptance():
    record=read_json(cfg.SOURCE_ACCEPTANCE_PATH);cfg.acceptance_identity(record)
    if record['code']!=source_code_record():raise ValueError('Physical-split source code changed after lock')
    verify_frozen(record['frozen_files']);return record


def load_sources(torch,warmup,data_protocol,ce):
    record=load_acceptance()
    if canonical(record['data_protocol'])!=canonical(data_protocol) or record['ce_identity']!=ce:
        raise ValueError('Physical-split data/CE identity differs')
    result={}
    for method,item in record['sources'].items():
        path=item['identity']['path'];payload=torch.load(path,map_location='cpu',weights_only=False)
        if sha256(path)!=item['identity']['sha256']:raise ValueError('Source changed while loading')
        result[method]=(payload,item)
    return result


def parameter_groups(chain):
    head=[];bs=[];model=chain.codec_model;variant=model.joint_decoder.variant
    for name,p in model.named_parameters():
        if name.startswith('encoder.'):continue
        if name.startswith(HEAD_PREFIX):head.append((name,p))
        elif variant=='control' and name.startswith(BRIDGE_PREFIX):continue
        else:bs.append((name,p))
    if not head or not bs or {id(p) for n,p in head}&{id(p) for n,p in bs}:raise ValueError('Disjoint nonempty head/BS parameter groups required')
    return dict(head=head,bs=bs)


def configure_phase(chain,iteration):
    """Reassert all modes, including after codec.train() and after monitor."""
    model=chain.codec_model;model.train();groups=parameter_groups(chain)
    for p in model.parameters():p.requires_grad_(False)
    model.encoder.eval();chain.ce_model.requires_grad_(False).eval()
    active=iteration>108
    if not active:model.joint_decoder.eval();model.decoder.eval()
    model.joint_decoder.physical_head.train()
    for n,p in groups['head']:p.requires_grad_(True)
    for n,p in groups['bs']:p.requires_grad_(active)
    if model.joint_decoder.variant=='control':model.joint_decoder.physical_bridge.eval()
    return groups


def build_chain(torch,device,method,payload,*,include_stages=True,initialize=False):
    if method in cfg.SOURCE_METHODS:
        return old.build_chain(torch,device,cfg.planned_source_identity(method)['arm'],payload,include_stages=include_stages)
    from models.twc_v1_k8_physical_split import build_codec,migrate_source_state
    import train_twc_v1_k8_codec_warmup as warmup
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg.SEED);chain=warmup.build_chain(torch.device('cpu'))
        codec=build_codec(variant=cfg.variant(method),new_module_seed=cfg.SEED,include_stages=include_stages)
    chain.codec_model=codec
    if initialize:chain.migration_evidence=migrate_source_state(codec,payload['codec_model'])
    else:
        old.previous.task.previous.old.check_tensor_state(torch,codec,payload['codec_model'])
        codec.load_state_dict(payload['codec_model'],strict=True)
        if tensor_digest(torch,codec.state_dict())!=tensor_digest(torch,payload['codec_model']):
            raise ValueError('Strict restore changed values/shared decoder aliases')
    if codec.decoder is not codec.joint_decoder.base_decoder:raise ValueError('Spatial decoder alias lost')
    chain.ce_model.eval().requires_grad_(False);codec.encoder.eval().requires_grad_(False)
    return chain.to(device)


frozen_part_digest=old.frozen_part_digest


def assert_frozen(torch,chain,before,*,check_state=True):
    old.assert_frozen_parts(torch,chain,before,check_state=check_state)


def condition_key(condition):return old.condition_key(condition)
def expected_counts(arm,step):return dict(Counter(condition_key(cfg.schedule(arm,i)) for i in range(1,step+1)))


def validate_runtime_resume(torch,chain,payload,arm,acceptance,data_protocol,ce):
    cfg.validate_resume_metadata(payload,arm,acceptance,code_sha256=source_code_record()['sha256'])
    if (payload.get('source_acceptance_sha256')!=sha256(cfg.SOURCE_ACCEPTANCE_PATH)
            or canonical(payload['data_protocol'])!=canonical(data_protocol) or payload['ce_identity']!=ce
            or payload.get('actual_condition_counts')!=expected_counts(arm,payload['step'])):
        raise ValueError('Dual-optimizer resume source/data/CE/count mismatch')
    groups=parameter_groups(chain);counts=cfg.update_counts(payload['step'])
    for key in ('head','bs'):check_optimizer_tensors(torch,payload,key,groups[key],counts[key+'_updates'])
    old.previous.task.previous.old.check_tensor_state(torch,chain.codec_model,payload['codec_model'])
    if payload.get('frozen_state')!=frozen_part_digest(torch,chain):
        raise ValueError('Resume changed recorded frozen UE/CE parameters or buffers')
    if set(payload.get('monitor_records',{}))!={str(s) for s in cfg.MONITOR_STEPS if s<=payload['step']}:
        raise ValueError('Incomplete physical-split monitor trajectory')
    for entry in payload.get('monitor_records',{}).values():
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Saved monitor bytes changed')
    for name in ('training_input_history','layout_transmit_history'):
        digest=payload.get(name)
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing actual per-update input history')
    if payload.get('phase')!=cfg.phase(payload['step']):raise ValueError('Resume phase differs')
    chain.codec_model.load_state_dict(payload['codec_model'],strict=True)


def load_final_payload(torch,arm):
    acceptance=load_acceptance();folder=cfg.OUTPUT_ROOT/'training'/arm;path=folder/'checkpoint_latest.pt'
    manifest=read_json(folder/'manifest.json')
    if manifest.get('status')!='completed' or manifest.get('step')!=6048 or sha256(path)!=manifest.get('checkpoint_sha256'):
        raise ValueError('Complete fixed physical-split terminal required')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg.validate_resume_metadata(payload,arm,acceptance,code_sha256=source_code_record()['sha256'])
    if canonical(manifest.get('recipe'))!=canonical(cfg.recipe(arm,acceptance)):raise ValueError('Terminal recipe differs')
    return payload,dict(identity=dict(arm=arm,path=str(path),sha256=sha256(path),update_counts=cfg.update_counts(6048)),
        frozen_files={str(folder/name):sha256(folder/name) for name in
            ('checkpoint_latest.pt','manifest.json','initialization.json','progress.jsonl','terminal_checkpoint.json')})


def monitor(torch,np,warmup,chain,data,arm,step):
    import statistics
    from models.twc_v1_k8_physical_split import physical_diagnostics
    if step not in cfg.MONITOR_STEPS:raise ValueError('Unregistered monitor iteration')
    if (chain.ce_model.training or chain.codec_model.encoder.training or
            any(p.requires_grad for module in (chain.ce_model,chain.codec_model.encoder) for p in module.parameters())):
        raise ValueError('UE/condition embedding and CE must remain frozen/eval during monitor')
    rng=capture_rng(torch,np);modes=[(m,m.training) for m in chain.modules()]
    state=tensor_digest(torch,chain.state_dict());buffers=tensor_digest(torch,dict(chain.named_buffers()))
    device=next(chain.parameters()).device;points={};inputs=[]
    try:
        chain.eval()
        with torch.no_grad():
            for bi in range(16):
                sample=data.sample('validation',cfg.MONITOR_START_INDEX+bi);item=dict(address=cfg.MONITOR_START_INDEX+bi,
                    channel_fingerprint=sample['sample_fingerprint'],noise={})
                for gi,gamma in enumerate(cfg.GAMMAS):
                    pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(cfg.noise_seed('monitor',bi,gi,k)),device) for k in ('pilot','data')]
                    item['noise'][str(gamma)]=[tensor_digest(torch,{'noise':x}) for x in pair]
                    for ratio in ((None,) if cfg.is_fdma(arm) else ((1,3),(1,1))):
                        policy='fdma' if ratio is None else 'hybrid_'+str(ratio[0])+'_'+str(ratio[1])
                        cond=dict(gamma_db=gamma,pilot_energy_per_ue=16.,group_count=1 if ratio is None else 4,energy_ratio=ratio)
                        out=run_forward(chain,warmup,sample['H_dl'],sample['H_ul'],cond,*pair)
                        metrics=chain.metrics(out,sample['H_dl'],rzf_alpha=.03,downlink_snr_db=25.)
                        old.previous.check_power(torch,out,metrics,48.)
                        values=dict(dl_nmse_linear=metrics['dl_nmse_per_user'].mean(-1),sum_rate_bps_hz=metrics['sum_rate_bps_hz'],
                            desired_signal_power=metrics['desired_signal_power_mean'],interference_power=metrics['interference_power_mean'])
                        values.update(physical_diagnostics(out));target=points.setdefault((policy,gamma),{k:[] for k in values})
                        for key,value in values.items():
                            if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite monitor')
                            target[key].extend(value.cpu().tolist())
                inputs.append(item)
        rows=[];means=[]
        for (policy,gamma),values in points.items():
            if any(len(v)!=64 for v in values.values()):raise ValueError('Monitor MU count')
            row=dict(energy_policy=policy,gamma_db=gamma,num_samples=64,alpha=.03)
            for name,v in values.items():row.update({name+'_'+k:x for k,x in old.previous.probe.statistics_for(v).items()})
            rows.append(row)
        for policy in sorted({p for p,g in points}):
            row=dict(energy_policy=policy,num_samples=64,alpha=.03)
            for name in next(iter(points.values())):
                values=[statistics.fmean(points[(policy,g)][name][i] for g in cfg.GAMMAS) for i in range(64)]
                row.update({name+'_'+k:x for k,x in old.previous.probe.statistics_for(values).items()})
            means.append(row)
        return dict(arm=arm,step=step,update_counts=cfg.update_counts(step),data_global_step=40240+step,
            alpha=.03,summary=rows,all_six=means,shared_inputs=inputs,selection=False,NMSE_units='linear',
            received_residual_units='unnormalized mean power over data REs and antennas')
    finally:
        for module,mode in modes:module.training=mode
        restore_rng(torch,np,rng)
        if state!=tensor_digest(torch,chain.state_dict()) or buffers!=tensor_digest(torch,dict(chain.named_buffers())):
            raise ValueError('Monitor changed state/buffers')


def train(arm,*,resume=False,stop_after=None):
    from train_proposal_as_receiver import run_lock
    from models.twc_v1_k8_physical_split import physical_codeword_loss
    torch,np,warmup=runtime();device=torch.device('cuda:0');acceptance=load_acceptance()
    folder=cfg.OUTPUT_ROOT/'training'/arm;owner=SimpleNamespace(EXP_NAME='k8_physical_split_'+arm,arm=arm,SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        if not resume and folder.exists():raise FileExistsError(folder)
        if resume and not (folder/'checkpoint_latest.pt').exists():raise FileNotFoundError('Explicit resume needs own latest')
        random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED);torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark=False
        data=make_data(warmup,device);ce=ce_identity();code=source_code_record();sources=load_sources(torch,warmup,data.protocol,ce)
        source,evidence=sources[cfg.source_method(arm)];start=0;monitors={};counts=Counter();history='0'*64;layout_history='0'*64;resumes={}
        if resume:
            initial=read_json(folder/'initialization.json');manifest=read_json(folder/'manifest.json')
            if sha256(folder/'checkpoint_latest.pt')!=manifest.get('checkpoint_sha256'):raise ValueError('Latest receipt SHA mismatch')
            for k,v in dict(code=code,source=cfg.source_identity(arm,acceptance),source_acceptance=cfg.acceptance_identity(acceptance),
                recipe=cfg.recipe(arm,acceptance),ce_identity=ce).items():
                if canonical(initial.get(k))!=canonical(v):raise ValueError('Initialization '+k+' changed')
            payload=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            chain=build_chain(torch,device,arm,payload);validate_runtime_resume(torch,chain,payload,arm,acceptance,data.protocol,ce)
            start=payload['step'];monitors=payload['monitor_records'];counts=Counter(payload['actual_condition_counts'])
            history=payload['training_input_history'];layout_history=payload['layout_transmit_history'];resumes=payload.get('resume_records',{})
        else:chain=build_chain(torch,device,arm,source,initialize=True)
        frozen=frozen_part_digest(torch,chain)
        if frozen['ue']!=evidence['verification']['ue_sha256']:raise ValueError('Full frozen source UE not preserved')
        del sources,source
        groups=parameter_groups(chain);names={k:[n for n,p in values] for k,values in groups.items()}
        optimizers={k:torch.optim.AdamW([p for n,p in values],lr=2e-4,weight_decay=1e-5) for k,values in groups.items()}
        schedulers={k:torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda _:1.) for k,opt in optimizers.items()}
        if resume:
            for key in groups:
                optimizers[key].load_state_dict(payload[key+'_optimizer']);schedulers[key].load_state_dict(payload[key+'_scheduler'])
                check_optimizer_tensors(torch,dict(payload,**{key+'_optimizer':optimizers[key].state_dict(),
                    key+'_scheduler':schedulers[key].state_dict()}),key,groups[key],cfg.update_counts(start)[key+'_updates'])
            restore_rng(torch,np,payload['rng'])
            target=folder/('resume_'+uuid.uuid4().hex[:8]+'.json')
            write_once_json(target,dict(step=start,next_iteration=start+1,next_data_address=40241+start,
                update_counts=cfg.update_counts(start),phase=cfg.phase(start),strict=True,restored_optimizer_states=True,
                scheduler_epochs={k:v.last_epoch for k,v in schedulers.items()},RNG_restored=True,
                uninterrupted_bitwise_equivalence_claim=False))
            resumes[str(start)]=dict(path=str(target),sha256=sha256(target));del payload
        if start==6048:
            terminal=folder/'terminal_checkpoint.json'
            if not terminal.exists():write_once_json(terminal,dict(path=str(folder/'checkpoint_latest.pt'),
                sha256=manifest['checkpoint_sha256'],update_counts=cfg.update_counts(6048),data_global_step=46288))
            if manifest.get('error'):manifest.setdefault('failure_history',[]).append(dict(time=now(),error=manifest.pop('error')))
            manifest.update(status='completed',updated_utc=now(),additional_updates=0,resume_records=resumes)
            write_once_json(folder/('terminal_recovery_'+uuid.uuid4().hex[:8]+'.json'),manifest);save_json(folder/'manifest.json',manifest);return manifest
        folder.mkdir(parents=True,exist_ok=True)
        if not resume:
            write_once_json(folder/'initialization.json',dict(code=code,source=cfg.source_identity(arm,acceptance),source_evidence=evidence,
                source_acceptance=cfg.acceptance_identity(acceptance),recipe=cfg.recipe(arm,acceptance),ce_identity=ce,
                migration=chain.migration_evidence,new_two_optimizers=True,optimizer_parameter_names=names,frozen_state=frozen,
                initialized_state_sha256=tensor_digest(torch,chain.codec_model.state_dict())))
        end=min(6048,start+stop_after) if stop_after else 6048
        manifest=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,status='running',step=start,update_counts=cfg.update_counts(start),
            phase=cfg.phase(start),source_codec_updates=34192,source_bs_updates=6048,source_data_address=40240,data_global_step=40240+start,
            ue_frozen=True,code=code,source=cfg.source_identity(arm,acceptance),source_acceptance=cfg.acceptance_identity(acceptance),
            recipe=cfg.recipe(arm,acceptance),config_fingerprint=cfg.fingerprint(arm,acceptance),ce_identity=ce,
            monitor_records=monitors,actual_condition_counts=dict(counts),resume_records=resumes,optimizer_parameter_names=names,
            parameters=old.previous.task.previous.parameters(chain.codec_model),selection='terminal6048 only; no best',
            monitor_alpha=.03,test_used=False,updated_utc=now())
        if resume:manifest['checkpoint_sha256']=sha256(folder/'checkpoint_latest.pt')
        save_json(folder/'manifest.json',manifest)
        try:
            for step in range(start+1,end+1):
                configure_phase(chain,step);cond=cfg.schedule(arm,step);sample=data.sample('codec_train',cond['global_step'])
                gen=torch.Generator(device=device).manual_seed(cfg.training_noise_seed(step));pair=[warmup.standard_noise(4,gen,device) for _ in range(2)]
                out=run_forward(chain,warmup,sample['H_dl'],sample['H_ul'],cond,*pair)
                metrics=chain.metrics(out,sample['H_dl'],rzf_alpha=.03,downlink_snr_db=25.)
                head_loss=physical_codeword_loss(out);csi_loss=metrics['dl_nmse_per_user'].mean();rate=metrics['sum_rate_bps_hz'].mean()
                if not bool(torch.isfinite(head_loss)&torch.isfinite(csi_loss)&torch.isfinite(rate)):raise FloatingPointError('Nonfinite separated losses')
                old.previous.check_power(torch,out,metrics,64.-cond['pilot_energy_per_ue'])
                for opt in optimizers.values():opt.zero_grad(set_to_none=True)
                head_loss.backward()
                if any(p.grad is not None for n,p in groups['bs']):raise ValueError('Codeword loss leaked to original BS/bridge')
                head_norm=torch.nn.utils.clip_grad_norm_([p for n,p in groups['head']],1.,error_if_nonfinite=True)
                if step>108:
                    before_head={n:p.grad.detach().clone() if p.grad is not None else None for n,p in groups['head']}
                    csi_loss.backward()
                    if any((before_head[n] is None)!=(p.grad is None) or (p.grad is not None and not torch.equal(before_head[n],p.grad))
                           for n,p in groups['head']):raise ValueError('CSI loss leaked to physical head')
                    bs_norm=torch.nn.utils.clip_grad_norm_([p for n,p in groups['bs']],1.,error_if_nonfinite=True)
                else:bs_norm=torch.tensor(0.,device=device)
                assert_frozen(torch,chain,frozen,check_state=False)
                optimizers['head'].step();schedulers['head'].step()
                if step>108:optimizers['bs'].step();schedulers['bs'].step()
                counts[condition_key(cond)]+=1;stamps=old.input_fingerprints(torch,out,sample,pair)
                history=hashlib.sha256((history+json.dumps({k:stamps[k] for k in ('channel','pilot','data')},sort_keys=True)).encode()).hexdigest()
                layout_history=hashlib.sha256((layout_history+json.dumps(stamps,sort_keys=True)).encode()).hexdigest()
                if step in cfg.MONITOR_STEPS:
                    report=monitor(torch,np,warmup,chain,data,arm,step)
                    report.update(training_csi_nmse_linear=float(csi_loss.detach()),training_codeword_nmse_linear=float(head_loss.detach()),
                        training_rate=float(rate.detach()),head_grad_norm=float(head_norm),bs_grad_norm=float(bs_norm))
                    p=folder/(f'monitor_step_{step:06d}_'+uuid.uuid4().hex[:8]+'.json');write_once_json(p,report)
                    monitors[str(step)]=dict(path=str(p),sha256=sha256(p))
                if step%108==0:
                    assert_frozen(torch,chain,frozen)
                    if source_code_record()!=code or ce_identity()!=ce:raise ValueError('Code/frozen CE changed')
                    if step==end:verify_frozen(acceptance['frozen_files'])
                    if dict(counts)!=expected_counts(arm,step):raise ValueError('Condition budget mismatch')
                    if any(not bool(torch.isfinite(p).all()) for p in chain.codec_model.parameters()):raise FloatingPointError('Nonfinite model state')
                    payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=step,phase=cfg.phase(step),update_counts=cfg.update_counts(step),
                        source_codec_updates=34192,source_bs_updates=6048,source_data_address=40240,data_global_step=40240+step,ue_frozen=True,
                        codec_model=chain.codec_model.state_dict(),rng=capture_rng(torch,np),optimizer_parameter_names=names,
                        source=cfg.source_identity(arm,acceptance),recipe=cfg.recipe(arm,acceptance),config_fingerprint=cfg.fingerprint(arm,acceptance),
                        source_acceptance=cfg.acceptance_identity(acceptance),source_acceptance_sha256=sha256(cfg.SOURCE_ACCEPTANCE_PATH),
                        code_sha256=code['sha256'],ce_identity=ce,data_protocol=data.protocol,frozen_state=frozen,
                        monitor_records=monitors,actual_condition_counts=dict(counts),resume_records=resumes,
                        training_input_history=history,layout_transmit_history=layout_history)
                    for key in optimizers:payload[key+'_optimizer']=optimizers[key].state_dict();payload[key+'_scheduler']=schedulers[key].state_dict()
                    for key in groups:check_optimizer_tensors(torch,payload,key,groups[key],cfg.update_counts(step)[key+'_updates'])
                    path=folder/'checkpoint_latest.pt';temp=path.with_suffix('.pt.tmp');torch.save(payload,temp);temp.replace(path)
                    manifest.update(step=step,update_counts=cfg.update_counts(step),phase=cfg.phase(step),data_global_step=40240+step,
                        checkpoint_sha256=sha256(path),status='completed' if step==6048 else 'paused' if step==end else 'running',
                        updated_utc=now(),monitor_records=monitors,resume_records=resumes,actual_condition_counts=dict(counts),
                        training_input_history=history,layout_transmit_history=layout_history,
                        last_csi_nmse_linear=float(csi_loss.detach()),last_codeword_nmse_linear=float(head_loss.detach()),last_rate=float(rate.detach()),
                        head_grad_norm=float(head_norm),bs_grad_norm=float(bs_norm),frozen_state=frozen)
                    save_json(folder/'manifest.json',manifest)
                    row=dict(step=step,update_counts=cfg.update_counts(step),phase=cfg.phase(step),data_global_step=40240+step,
                        condition=cond,head_nmse_linear=float(head_loss.detach()),csi_nmse_linear=float(csi_loss.detach()),rate=float(rate.detach()),
                        head_grad_norm=float(head_norm),bs_grad_norm=float(bs_norm),input_fingerprints=stamps,utc=now(),monitor=monitors.get(str(step)))
                    with (folder/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    print(json.dumps(row),flush=True)
            if end==6048:write_once_json(folder/'terminal_checkpoint.json',dict(path=str(folder/'checkpoint_latest.pt'),
                sha256=manifest['checkpoint_sha256'],update_counts=cfg.update_counts(6048),data_global_step=46288))
        except BaseException as error:
            manifest.update(status='failed',error=f'{type(error).__name__}: {error}',updated_utc=now())
            save_json(folder/('failure_'+uuid.uuid4().hex[:8]+'.json'),manifest);save_json(folder/'manifest.json',manifest);raise
        return manifest


def _gradient_summary(torch,values,parameters):
    gradients=torch.autograd.grad(values,[p for n,p in parameters],retain_graph=True,allow_unused=True)
    present=[g for g in gradients if g is not None]
    if any(not bool(torch.isfinite(g).all()) for g in present):raise FloatingPointError('Nonfinite gradient boundary check')
    return dict(connected_parameters=len(present),total_parameters=len(parameters),
        l2_norm=float(sum(g.detach().abs().double().square().sum() for g in present).sqrt()) if present else 0.,
        by_parameter={n:None if g is None else float(g.norm()) for (n,p),g in zip(parameters,gradients)})


def preflight(report_path):
    """Real B4 source replay and separated gradients; zero optimizer updates."""
    report_path=cfg.inside_root(report_path)
    if report_path.exists():raise FileExistsError(report_path)
    report=dict(status='running',checks={},gradients={},interventions={},memory={},parameters={},failures=[]);failure=None
    try:
        torch,np,warmup=runtime();device=torch.device('cuda:0')
        from models.twc_v1_k8_ordered_encoder import comparison_error
        from train_twc_v1_k8_ordered_encoder_screen import EQUIVALENCE_TOLERANCES as tolerances
        from models.twc_v1_k8_physical_split import physical_codeword_loss,physical_diagnostics
        acceptance=load_acceptance();data=make_data(warmup,device);data.prepare()
        report.update(code=source_code_record(),source_acceptance=cfg.acceptance_identity(acceptance),batch_size=4,
            optimizer_updates=0,backend=old.previous.task.backend(torch),tolerances=tolerances)
        sources=load_sources(torch,warmup,data.protocol,ce_identity())
        for method in ('old_fdma_base2','old_g4_base2'):
            payload,evidence=sources[method];source=build_chain(torch,device,method,payload,include_stages=False).eval()
            source_before=tensor_digest(torch,source.state_dict())
            sample=data.sample('codec_train',500300);hdl,hul=sample['H_dl'],sample['H_ul']
            for arm in (a for a in cfg.ARMS if cfg.is_fdma(a)==cfg.is_fdma(method)):
                torch.cuda.reset_peak_memory_stats(device)
                chain=build_chain(torch,device,arm,payload,initialize=True).eval();model=chain.codec_model;receiver=model.joint_decoder
                before=tensor_digest(torch,chain.state_dict());frozen=frozen_part_digest(torch,chain)
                report['checks'][arm]={};report['parameters'][arm]=old.previous.task.previous.parameters(model)
                report['parameters'][arm].update(physical_head=sum(p.numel() for p in receiver.physical_head.parameters()),
                    physical_bridge=sum(p.numel() for p in receiver.physical_bridge.parameters()))
                for gi,gamma in enumerate((0.,10.,25.)):
                    for ratio in ((None,) if cfg.is_fdma(arm) else ((1,3),(1,1))):
                        label=str(gamma)+'/'+str(ratio);report['current_condition']=dict(arm=arm,gamma_db=gamma,ratio=ratio)
                        condition=dict(gamma_db=gamma,pilot_energy_per_ue=16.,group_count=1 if ratio is None else 4,energy_ratio=ratio)
                        pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(
                            cfg.noise_seed('preflight',0,gi,k)),device) for k in ('pilot','data')]
                        rng=capture_rng(torch,np)
                        with torch.no_grad():
                            source_out=run_forward(source,warmup,hdl,hul,condition,*pair)
                            source_metrics=source.metrics(source_out,hdl,rzf_alpha=.03,downlink_snr_db=25.)
                            restore_rng(torch,np,rng);out=run_forward(chain,warmup,hdl,hul,condition,*pair)
                            metrics=chain.metrics(out,hdl,rzf_alpha=.03,downlink_snr_db=25.)
                            repeat=chain.metrics(out,hdl,rzf_alpha=.03,downlink_snr_db=25.)
                            diagnostics=physical_diagnostics(out)
                        checks={}
                        for name,left,right,tolerance in (
                            ('X',out['tx']['X'],source_out['tx']['X'],'transmit'),
                            ('H_hat',out['H_dl_hat'],source_out['H_dl_hat'],'H_hat'),
                            ('RZF',metrics['precoder'],source_metrics['precoder'],'RZF'),
                            ('NMSE',metrics['dl_nmse_per_user'],source_metrics['dl_nmse_per_user'],'dl_nmse'),
                            ('rate',metrics['sum_rate_bps_hz'],source_metrics['sum_rate_bps_hz'],'rate')):
                            checks[name]=comparison_error(left,right,**tolerances[tolerance])
                        stages=out['receiver_execution']['stages']
                        checks['initial_physical_equals_soft']=dict(passed=bool(torch.equal(stages['x_phy'],stages['soft'])))
                        checks['RZF_repeat']=dict(passed=bool(torch.equal(metrics['precoder'],repeat['precoder'])))
                        try:checks['power']=dict(passed=True,**old.previous.check_power(torch,out,metrics,48.))
                        except (ValueError,FloatingPointError) as error:checks['power']=dict(passed=False,error=str(error))
                        checks['diagnostics_linear_or_mean_power']={k:float(v.mean()) for k,v in diagnostics.items()}
                        report['checks'][arm][label]=checks
                        if any(value.get('passed') is False for value in checks.values() if isinstance(value,dict)):
                            report['failures'].append(arm+'/'+label+' source replay')
                configure_phase(chain,109);chain.zero_grad(set_to_none=True)
                condition=dict(gamma_db=10.,pilot_energy_per_ue=16.,group_count=1 if cfg.is_fdma(arm) else 4,
                    energy_ratio=None if cfg.is_fdma(arm) else (1,3))
                pair=[warmup.standard_noise(4,torch.Generator(device=device).manual_seed(
                    cfg.noise_seed('preflight',0,1,k)),device) for k in ('pilot','data')]
                out=run_forward(chain,warmup,hdl,hul,condition,*pair);groups=parameter_groups(chain)
                physical_loss=physical_codeword_loss(out);csi_loss=chain.metrics(out,hdl,rzf_alpha=.03,downlink_snr_db=25.)['dl_nmse_per_user'].mean()
                checks=dict(physical_to_head=_gradient_summary(torch,physical_loss,groups['head']),
                    physical_to_BS=_gradient_summary(torch,physical_loss,groups['bs']),
                    CSI_to_head=_gradient_summary(torch,csi_loss,groups['head']),
                    CSI_to_BS=_gradient_summary(torch,csi_loss,groups['bs']))
                if (checks['physical_to_head']['l2_norm']<=0 or checks['CSI_to_BS']['l2_norm']<=0
                        or checks['physical_to_BS']['connected_parameters'] or checks['CSI_to_head']['connected_parameters']):
                    report['failures'].append(arm+' incorrect gradient partition')
                if cfg.bridge_enabled(arm):
                    output_grad=[v for n,v in checks['CSI_to_BS']['by_parameter'].items() if n.startswith(BRIDGE_PREFIX+'2.') and v is not None]
                    if not output_grad or not any(v>0 for v in output_grad):report['failures'].append(arm+' zero bridge output gradient')
                report['gradients'][arm]=checks
                assert_frozen(torch,chain,frozen)
                del out,physical_loss,csi_loss
                if cfg.bridge_enabled(arm):
                    # Small, explicit receiver-only diagnostic perturbations; every
                    # changed tensor and RNG state is restored before acceptance.
                    chain.eval();rng=capture_rng(torch,np)
                    saved={n:p.detach().clone() for n,p in receiver.named_parameters() if n.startswith(('physical_head.','physical_bridge.'))}
                    try:
                        with torch.no_grad():
                            receiver.physical_head[-1].bias.copy_(torch.tensor([.07,-.04],device=device))
                            receiver.physical_bridge[-1].weight.fill_(.03)
                            outputs={}
                            for bridge_on in (True,False):
                                for use_soft in (False,True):
                                    restore_rng(torch,np,rng)
                                    with receiver.diagnostic_options(bridge_enabled=bridge_on,physical_from_soft=use_soft):
                                        value=run_forward(chain,warmup,hdl,hul,condition,*pair)
                                        outputs[(bridge_on,use_soft)]=value['H_dl_hat'].detach().clone()
                            enabled=float((outputs[(True,False)]-outputs[(True,True)]).abs().max())
                            disabled=bool(torch.equal(outputs[(False,False)],outputs[(False,True)]))
                        report['interventions'][arm]=dict(passed=enabled>0 and disabled,bridge_on_H_hat_max_abs_change=enabled,
                            bridge_off_exact_replay=disabled,temporary_parameters_restored=True,
                            context1_inputs_unchanged=True,not_a_retrained_ablation=True)
                        if enabled<=0 or not disabled:report['failures'].append(arm+' physical interface ineffective')
                    finally:
                        with torch.no_grad():
                            for n,p in receiver.named_parameters():
                                if n in saved:p.copy_(saved[n])
                        restore_rng(torch,np,rng)
                if before!=tensor_digest(torch,chain.state_dict()):raise ValueError('Preflight changed model/buffers')
                report['memory'][arm]=dict(batch_size=4,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),includes_reference_model=True)
                del chain,model,receiver,source_out,source_metrics,metrics,repeat;torch.cuda.empty_cache()
            if source_before!=tensor_digest(torch,source.state_dict()):raise ValueError('Read-only source changed')
            del source;torch.cuda.empty_cache()
        verify_frozen(acceptance['frozen_files'])
        if report['failures']:raise ValueError('Physical-split preflight failed: '+str(report['failures']))
        report.update(status='passed',finished=now(),pause_resume_validation='The included108 checkpoint is validated by the continuing trainer; no extra updates.')
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=__import__('traceback').format_exc())
    finally:save_json(report_path,report)
    if failure is not None:raise RuntimeError('Physical-split preflight failed: '+str(report_path)) from failure
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=cfg.ARMS)
    mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','train','accept-sources','preflight'):mode.add_argument('--'+name,action='store_true')
    mode.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:print(json.dumps(dict(recipe=cfg.recipe(args.arm or cfg.ARMS[0]),weights_read=False,models_built=False),indent=2));return 0
    if args.accept_sources or args.preflight:
        if args.report is None or args.arm or args.resume:parser.error('Check requires unused --report only')
        (accept_sources if args.accept_sources else preflight)(args.report);return 0
    if args.report is not None:parser.error('--report requires a check')
    if not(args.train or args.stop_after is not None):parser.print_help();return 0
    if args.arm is None:parser.error('--arm required')
    if args.stop_after is not None and (not 0<args.stop_after<=6048 or args.stop_after%108):parser.error('stop-after is an included108 boundary')
    train(args.arm,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':raise SystemExit(main())
