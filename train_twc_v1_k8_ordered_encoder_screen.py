#!/usr/bin/env python3
"""Isolated ordered-window continuation, with archived-source validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

from configs import twc_v1_k8_ordered_encoder_screen_config as cfg
from configs import twc_v1_k8_schedule_screen_config as source_cfg
import train_twc_v1_k8_schedule_screen as old

ROOT=cfg.REPO_ROOT
canonical,sha256,save_json=old.canonical,old.sha256,old.save_json
ce_identity,capture_rng,restore_rng=old.ce_identity,old.capture_rng,old.restore_rng
make_data,tensor_digest=old.make_data,old.tensor_digest
SOURCE_FILES=tuple(dict.fromkeys(old.SOURCE_FILES+(
    'models/twc_v1_k8_ordered_encoder.py','configs/twc_v1_k8_ordered_encoder_screen_config.py',
    'train_twc_v1_k8_ordered_encoder_screen.py','evaluate_twc_v1_k8_ordered_encoder_screen.py',
    'run_twc_v1_k8_ordered_encoder_screen.py')))
# Preset before any full-model CUDA checks; never enlarged automatically.
EQUIVALENCE_TOLERANCES={
    'raw':dict(rtol=1e-5,atol=1e-6), 'transmit':dict(rtol=1e-5,atol=1e-6),
    'H_hat':dict(rtol=1e-4,atol=1e-5), 'RZF':dict(rtol=1e-4,atol=1e-5),
    'dl_nmse':dict(rtol=1e-4,atol=1e-6), 'rate':dict(rtol=1e-4,atol=1e-4)}


def now():return datetime.now(timezone.utc).isoformat()


def source_code_record():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})


@lru_cache(maxsize=1)
def archived_source_hashes():
    """The historical source identity is8096, irrespective of the new HEAD."""
    subprocess.run(['git','cat-file','-e',cfg.SOURCE_COMMIT+'^{commit}'],cwd=ROOT,check=True,capture_output=True)
    result={}
    for name in old.SOURCE_FILES:
        blob=subprocess.check_output(['git','show',cfg.SOURCE_COMMIT+':'+name],cwd=ROOT)
        result[name]=hashlib.sha256(blob).hexdigest()
    return result


def validate_source_metadata(payload,manifest,initial,arm,*,archived_hashes):
    identity=cfg.source_identity(arm);source_arm=identity['arm']
    # This existing metadata guard checks its old recipe/fingerprint, not HEAD.
    source_cfg.validate_resume_metadata(payload,source_arm,code_sha256=archived_hashes)
    if payload['step']!=3024 or payload['cumulative_updates']!=13024:
        raise ValueError('Source must be local3024/cumulative13024')
    expected_code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived_hashes)
    for key,value in dict(status='completed',arm=source_arm,seed=cfg.SEED,step=3024,
                          cumulative_updates=13024,config_fingerprint=source_cfg.fingerprint(source_arm),
                          checkpoint_sha256=identity['sha256']).items():
        if manifest.get(key)!=value:raise ValueError('Source manifest '+key+' mismatch')
    if manifest.get('code')!=expected_code or initial.get('code')!=expected_code:
        raise ValueError('Source commit/code differs from archived8096 source')
    for name,value in (('recipe',source_cfg.recipe(source_arm)),('source',source_cfg.source_identity(source_arm))):
        if any(canonical(record.get(name))!=canonical(value) for record in (payload,manifest,initial)):
            raise ValueError('Source '+name+' changed')
    if any(canonical(record.get('ce_identity'))!=canonical(payload['ce_identity']) for record in (manifest,initial)):
        raise ValueError('Source CE/calibration identity differs across evidence')
    return dict(source_commit=cfg.SOURCE_COMMIT,new_HEAD_may_differ=True,source_arm=source_arm,
                local_step=3024,cumulative_updates=13024)


def check_source_batch(identity):
    """Read the exact previous pointer; never discover a different source run."""
    pointer=source_cfg.OUTPUT_ROOT/'last_batch.json'
    folder=Path(json.loads(pointer.read_text(encoding='utf-8'))['directory']).resolve()
    if not folder.is_relative_to((source_cfg.OUTPUT_ROOT/'batches').resolve()):raise ValueError('Source batch pointer escaped its root')
    path=folder/'batch_summary.json';summary=json.loads(path.read_text(encoding='utf-8'))
    if summary.get('status')!='completed' or summary.get('error'):raise ValueError('The previous screening batch is not complete')
    if summary.get('code')!=dict(commit=cfg.SOURCE_COMMIT,sha256=archived_source_hashes()):
        raise ValueError('Source batch archived code identity mismatch')
    if set(summary.get('training',{}))!=set(source_cfg.ARMS):raise ValueError('Previous four training records required')
    record=summary['training'][identity['arm']]
    if (record.get('checkpoint_sha256')!=identity['sha256'] or record.get('step')!=3024
            or record.get('cumulative_updates')!=13024 or record.get('checkpoint_path')!=identity['path']):
        raise ValueError('Source batch points to different checkpoint evidence')
    stage=summary.get('stages',{}).get('train_'+identity['arm'],{})
    evidence=stage.get('evidence_sha256',{})
    if stage.get('status')!='completed' or not evidence:raise ValueError('Completed source training-stage evidence required')
    required=[identity['path'],identity['manifest_path'],identity['initialization_path'],identity['progress_path']]
    for filename in required:
        if evidence.get(filename)!=sha256(filename):raise ValueError('Source stage evidence changed: '+filename)
    return path


def load_source(torch,warmup,arm,data_protocol):
    identity=cfg.source_identity(arm);path=Path(identity['path']);expected=archived_source_hashes()
    if sha256(path)!=identity['sha256']:raise ValueError('Pinned3024 source SHA256 mismatch: '+str(path))
    for name,digest in expected.items():
        if sha256(ROOT/name)!=digest:raise ValueError('Historical runtime source changed: '+name)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    manifest=json.loads(Path(identity['manifest_path']).read_text(encoding='utf-8'))
    initial=json.loads(Path(identity['initialization_path']).read_text(encoding='utf-8'))
    verified=validate_source_metadata(payload,manifest,initial,arm,archived_hashes=expected)
    if canonical(payload['data_protocol'])!=canonical(data_protocol):raise ValueError('Source data protocol mismatch')
    if payload['ce_identity']!=ce_identity():raise ValueError('Source CE/calibration files differ')
    # Fresh initialization does not restore source RNG; validate all saved states
    # without assuming its visible CUDA device count equals this server's count.
    rng=payload['rng'];random.Random().setstate(rng['python'])
    import numpy as np
    np.random.RandomState().set_state(rng['numpy'])
    for value in [rng['torch_cpu'],*rng['torch_cuda']]:
        if not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1 or value.device.type!='cpu':
            raise ValueError('Invalid source RNG byte state')
    scheduler=payload['scheduler']
    if scheduler.get('_step_count')!=3025 or scheduler.get('_last_lr')!=[cfg.LEARNING_RATE]:raise ValueError('Source scheduler identity mismatch')
    groups=payload['optimizer'].get('param_groups',[])
    if len(groups)!=1 or groups[0].get('lr')!=cfg.LEARNING_RATE or groups[0].get('weight_decay')!=cfg.WEIGHT_DECAY:
        raise ValueError('Source optimizer recipe mismatch')
    for state in payload['optimizer'].get('state',{}).values():
        if (set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].item()!=3024
                or not bool(torch.isfinite(state['exp_avg']).all() & torch.isfinite(state['exp_avg_sq']).all())
                or not bool((state['exp_avg_sq']>=0).all())):raise ValueError('Invalid source optimizer moments')
    events=[json.loads(line) for line in Path(identity['progress_path']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if ([row.get('step') for row in events]!=list(range(108,3025,108))
            or events[-1].get('cumulative_updates')!=13024):raise ValueError('Source update/log evidence mismatch')
    batch=check_source_batch(identity)
    frozen={str(p):sha256(p) for p in [path,Path(identity['manifest_path']),Path(identity['initialization_path']),Path(identity['progress_path']),batch]}
    if frozen[str(path)]!=identity['sha256']:raise ValueError('Source changed during loading')
    return payload,dict(identity=identity,verification=verified,frozen_files=frozen)


def build_chain(torch,device,arm,payload,*,initializing_source=False):
    import train_twc_v1_k8_codec_warmup as warmup
    from models.twc_v1_k8_ordered_encoder import build_codec,migrate_mean_state
    # Load even the CE artifact on CPU first; saved CPU RNG tensors are never
    # remapped to CUDA merely because the eventual model runs there.
    chain=warmup.build_chain(torch.device('cpu'))
    if cfg.is_ordered(arm):
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            codec=build_codec(ordered=True)
        chain.codec_model=codec
    if initializing_source and cfg.is_ordered(arm):
        chain.migration_evidence=migrate_mean_state(chain.codec_model,payload['codec_model'])
    else:
        old.check_tensor_state(torch,chain.codec_model,payload['codec_model'])
        chain.codec_model.load_state_dict(payload['codec_model'],strict=True)
    if initializing_source:
        names=[name for name,_ in chain.codec_model.named_parameters()]
        group=payload['optimizer']['param_groups'][0];moments=payload['optimizer']['state']
        if group.get('params')!=list(range(len(names))) or not moments or not set(moments)<=set(group['params']):
            raise ValueError('Source optimizer parameter identity mismatch')
        for index,state in moments.items():
            reference=payload['codec_model'][names[index]]
            for key in ('exp_avg','exp_avg_sq'):
                if state[key].shape!=reference.shape or state[key].dtype!=reference.dtype:
                    raise ValueError('Source optimizer tensor shape/dtype mismatch')
    chain.ce_model.eval().requires_grad_(False)
    return chain.to(device)


def parameters(model):
    full=sum(p.numel() for p in model.parameters());ue=sum(p.numel() for p in model.encoder.parameters())
    return dict(full=full,ue=ue,bs=full-ue,key_head=sum(p.numel() for p in model.encoder.key_head.parameters()),
                shared_head=sum(p.numel() for p in model.encoder.shared_head.parameters()))


def verify_completed_payload(torch,payload,arm):
    cfg.validate_resume_metadata(payload,arm,code_sha256=source_code_record()['sha256'])
    if payload['step']!=3024:raise ValueError('Only the predeclared terminal3024 weights may be evaluated')
    old._check_numerical_payload(torch,payload)


def validate_runtime_resume(torch,model,payload,arm):
    cfg.validate_resume_metadata(payload,arm,code_sha256=source_code_record()['sha256'])
    old._check_numerical_payload(torch,payload);old.check_tensor_state(torch,model,payload['codec_model'])
    params=list(model.parameters());group=payload['optimizer']['param_groups'][0]
    if group['params']!=list(range(len(params))):raise ValueError('Optimizer parameter order changed')
    if not set(payload['optimizer']['state'])<=set(group['params']):raise ValueError('Unknown optimizer parameter')
    for index,state in payload['optimizer']['state'].items():
        for name in ('exp_avg','exp_avg_sq'):
            if state[name].shape!=params[index].shape or state[name].dtype!=params[index].dtype:
                raise ValueError('Optimizer tensor shape/dtype mismatch')
    model.load_state_dict(payload['codec_model'],strict=True)


def gpu_snapshot():
    command=['nvidia-smi','-i','0','--query-gpu=uuid,utilization.gpu,memory.total,memory.used,memory.free','--format=csv,noheader,nounits']
    raw=subprocess.check_output(command,text=True).strip();parts=[p.strip() for p in raw.split(',')]
    if len(parts)!=5:raise RuntimeError('Cannot read GPU0 sharing snapshot')
    utilization,total,used,free=map(int,parts[1:])
    processes=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
    result=dict(time=now(),uuid=parts[0],utilization_percent=utilization,total_MiB=total,used_MiB=used,free_MiB=free,
        other_processes=processes,policy=dict(max_utilization_percent=20,min_free_MiB=4096),
        continuous_monitoring=False,colleague_processes_terminated=False)
    if utilization>20 or free<4096:raise RuntimeError('GPU startup sharing check failed: '+json.dumps(result))
    return result


def preflight(report_path):
    output=cfg.inside_root(report_path)
    if output.exists():raise FileExistsError(output)
    report=dict(status='running',source_commit=cfg.SOURCE_COMMIT,tolerances=EQUIVALENCE_TOLERANCES)
    frozen={};failure=None
    try:
        report['startup_gpu']=gpu_snapshot()
        torch,np,warmup=old.runtime();device=torch.device('cuda:0');data=make_data(warmup,device)
        data.prepare();ce=ce_identity();code=source_code_record()
        report.update(code=code,ce_identity=ce,data_protocol=data.protocol,sources={},comparisons={},parameters={},
            backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        for item in ce.values():frozen[item['path']]=item['sha256']
        sample=data.sample('validation',1536);Hdl,Hul=sample['H_dl'][:2],sample['H_ul'][:2]
        report['migration_sample']=dict(role='validation',batch_index=1536,used_samples=2,
            generated_batch_size=4,sample_fingerprint=sample['sample_fingerprint'],test_used=False)
        from models.twc_v1_k8_ordered_encoder import comparison_error
        for mean_arm,ordered_arm in (('mean_fdma_control','ordered_fdma'),('mean_g4_control','ordered_g4')):
            payload,evidence=load_source(torch,warmup,mean_arm,data.protocol)
            report['sources'][mean_arm]=evidence;frozen.update(evidence['frozen_files'])
            mean=build_chain(torch,device,mean_arm,payload,initializing_source=True).eval()
            ordered=build_chain(torch,device,ordered_arm,payload,initializing_source=True).eval();del payload
            left,right=parameters(mean.codec_model),parameters(ordered.codec_model)
            report['parameters'][mean_arm]=left;report['parameters'][ordered_arm]=right
            if (right['full']-left['full']!=cfg.EXPECTED_PARAMETER_DELTA
                    or right['ue']-left['ue']!=cfg.EXPECTED_PARAMETER_DELTA or right['bs']!=left['bs']):
                raise ValueError('Actual parameter changes exceed the two UE head weights')
            comparisons={};report['comparisons'][ordered_arm]=comparisons
            buffers_before=[tensor_digest(torch,dict(c.named_buffers())) for c in (mean,ordered)]
            with torch.no_grad():
                for group in (1,2,4):
                    for branch,a,b in zip(('key','shared'),mean.codec_model.encode(Hdl,group_count=group),ordered.codec_model.encode(Hdl,group_count=group)):
                        result=comparison_error(b,a,**EQUIVALENCE_TOLERANCES['raw']);comparisons[f'raw_G{group}_{branch}']=result
                        if not result['passed']:raise ValueError('Raw-code migration mismatch: '+ordered_arm)
                for gi,gamma in enumerate(cfg.GAMMAS):
                    gen=torch.Generator(device=device).manual_seed(cfg.SEED*10000000+1900000+gi)
                    pilot=warmup.standard_noise(2,gen,device);noise=warmup.standard_noise(2,gen,device)
                    for ratio in ((None,) if cfg.is_fdma(mean_arm) else cfg.ENERGY_RATIOS):
                        condition=dict(gamma_db=gamma,pilot_energy_per_ue=16.,group_count=1 if ratio is None else 4,energy_ratio=ratio)
                        rng=capture_rng(torch,np);a=old._run(mean,warmup,Hdl,Hul,condition,pilot,noise)
                        am=mean.metrics(a,Hdl);restore_rng(torch,np,rng)
                        b=old._run(ordered,warmup,Hdl,Hul,condition,pilot,noise);bm=ordered.metrics(b,Hdl)
                        fields=dict(transmit=(b['tx']['X'],a['tx']['X']),H_hat=(b['H_dl_hat'],a['H_dl_hat']),
                            RZF=(bm['precoder'],am['precoder']),dl_nmse=(bm['dl_nmse_per_user'],am['dl_nmse_per_user']),
                            rate=(bm['sum_rate_bps_hz'],am['sum_rate_bps_hz']))
                        key=f'gamma{gamma}_ratio{ratio}';comparisons[key]={}
                        for name,(actual,expected) in fields.items():
                            result=comparison_error(actual,expected,**EQUIVALENCE_TOLERANCES[name]);comparisons[key][name]=result
                            if not result['passed']:raise ValueError(f'Full-chain migration mismatch: {ordered_arm} {key} {name}')
                        for chain_output,metric in ((a,am),(b,bm)):
                            if ((chain_output['energy']['actual_data_per_ue']-48).abs().max().item()>1e-4
                                    or (metric['rzf_per_subcarrier_power']-1).abs().max().item()>1e-5):
                                raise ValueError('Migration check energy/RZF constraint failed')
            if buffers_before!=[tensor_digest(torch,dict(c.named_buffers())) for c in (mean,ordered)]:
                raise ValueError('Migration preflight modified eval buffers')
            report['migration_keys_'+ordered_arm]=ordered.migration_evidence
            del mean,ordered,a,b,am,bm;torch.cuda.empty_cache()
        report.update(status='passed',optimizer_updates=0,weights_saved=False,finished=now())
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',finished=now())
    finally:
        report['frozen_files']=frozen
        changed=[p for p,h in frozen.items() if not Path(p).is_file() or sha256(p)!=h]
        if changed:failure=failure or RuntimeError('Source files changed');report.update(status='failed',changed_inputs=changed)
        save_json(output,report)
    if failure is not None:raise RuntimeError('Preflight failed; retained '+str(output)) from failure
    return report


def training_manifest(arm,start,*,data_protocol,ce,code,resume_hash=None):
    if type(start) is not int or not 0<=start<3024 or start%108:raise ValueError('Saved incomplete108-step boundary required')
    if start and (not isinstance(resume_hash,str) or len(resume_hash)!=64):raise ValueError('Verified latest hash required on resume')
    result=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,status='running',step=start,target=3024,
        cumulative_updates=cfg.SOURCE_UPDATES+start,recipe=cfg.recipe(arm),config_fingerprint=cfg.fingerprint(arm),
        source=cfg.source_identity(arm),source_commit=cfg.SOURCE_COMMIT,code=code,new_experiment_commit=code['commit'],
        ce_identity=ce,data_protocol=data_protocol,updated_utc=now(),test_used=False,checkpoint_selection='terminal3024 only')
    if resume_hash is not None:result['checkpoint_sha256']=resume_hash
    return result


def completed_receipt(arm,payload,previous,*,checkpoint_sha,ce,code):
    """Finalize an already validated terminal checkpoint without another update."""
    cfg.validate_resume_metadata(payload,arm,code_sha256=code['sha256'])
    if (payload['step']!=3024 or previous.get('checkpoint_sha256')!=checkpoint_sha
            or canonical(payload['ce_identity'])!=canonical(ce)):
        raise ValueError('Completion recovery requires the verified terminal checkpoint receipt')
    result=canonical(previous)
    if 'error' in result:result.setdefault('failure_history',[]).append(dict(error=result.pop('error'),time=now()))
    result.update(status='completed',step=3024,cumulative_updates=16048,arm=arm,seed=cfg.SEED,
        code=code,new_experiment_commit=code['commit'],source_commit=cfg.SOURCE_COMMIT,
        updated_utc=now(),completion_recovered=True,additional_optimizer_updates=0)
    return result


def train(arm,*,resume=False,stop_after=None):
    from train_proposal_as_receiver import run_lock
    torch,np,warmup=old.runtime();device=torch.device('cuda:0');folder=cfg.OUTPUT_ROOT/'training'/arm
    owner=SimpleNamespace(EXP_NAME='k8_ordered_'+arm,arm=arm,SEED=cfg.SEED)
    with run_lock(owner,resume=resume,allowed_root=cfg.OUTPUT_ROOT):
        if not resume and folder.exists():raise FileExistsError('Training output exists')
        if resume and not (folder/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Resume requires latest')
        random.seed(cfg.SEED);np.random.seed(cfg.SEED);torch.manual_seed(cfg.SEED);torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark=False
        data=make_data(warmup,device);ce=ce_identity();code=source_code_record();start=0;resume_hash=None
        source,evidence=load_source(torch,warmup,arm,data.protocol)
        if resume:
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            resume_hash=sha256(folder/'checkpoint_latest.pt')
            if resume_hash!=manifest.get('checkpoint_sha256'):raise ValueError('Latest differs from saved receipt')
            initial=json.loads((folder/'initialization.json').read_text(encoding='utf-8'))
            for key,value in dict(source=cfg.source_identity(arm),ce_identity=ce,code=code,recipe=cfg.recipe(arm)).items():
                if canonical(initial.get(key))!=canonical(value):raise ValueError('Initialization '+key+' changed')
            payload=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            chain=build_chain(torch,device,arm,payload,initializing_source=False)
            validate_runtime_resume(torch,chain.codec_model,payload,arm)
            if payload['ce_identity']!=ce or canonical(payload['data_protocol'])!=canonical(data.protocol):raise ValueError('Resume CE/data identity changed')
            start=payload['step']
            if start==3024:
                receipt=completed_receipt(arm,payload,manifest,checkpoint_sha=resume_hash,ce=ce,code=code)
                receipt['parameters']=parameters(chain.codec_model)
                save_json(folder/('completion_recovery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'),receipt)
                save_json(folder/'manifest.json',receipt)
                return receipt
        else:
            chain=build_chain(torch,device,arm,source,initializing_source=True)
        del source
        model=chain.codec_model;optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,weight_decay=cfg.WEIGHT_DECAY)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        if resume:
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);restore_rng(torch,np,payload['rng']);del payload
        folder.mkdir(parents=True,exist_ok=True)
        if not resume:save_json(folder/'initialization.json',dict(source=cfg.source_identity(arm),source_evidence=evidence,
            source_commit=cfg.SOURCE_COMMIT,new_experiment_commit=code['commit'],code=code,ce_identity=ce,recipe=cfg.recipe(arm),
            migration=getattr(chain,'migration_evidence',dict(strict_mean_load=True)),parameters=parameters(model),
            initialized_state_sha256=tensor_digest(torch,model.state_dict()),new_optimizer=True))
        end=min(3024,start+stop_after) if stop_after else 3024
        if not start<end:raise ValueError('Completed arm is not retrained')
        manifest=training_manifest(arm,start,data_protocol=data.protocol,ce=ce,code=code,resume_hash=resume_hash)
        manifest['parameters']=parameters(model);save_json(folder/'manifest.json',manifest)
        ce_before=tensor_digest(torch,chain.ce_model.state_dict());model.train();torch.cuda.reset_peak_memory_stats(device)
        try:
            for step in range(start+1,end+1):
                condition=cfg.schedule(arm,step);sample=data.sample('codec_train',condition['global_step']);Hdl,Hul=sample['H_dl'],sample['H_ul']
                generator=torch.Generator(device=device).manual_seed(cfg.SEED*10000000+condition['global_step'])
                output=old._run(chain,warmup,Hdl,Hul,condition,warmup.standard_noise(4,generator,device),warmup.standard_noise(4,generator,device))
                metrics=chain.metrics(output,Hdl);loss=metrics['dl_nmse_per_user'].mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite NMSE')
                if ((output['energy']['actual_data_per_ue']+condition['pilot_energy_per_ue']-64).abs().max().item()>1e-4
                        or (metrics['rzf_per_subcarrier_power']-1).abs().max().item()>1e-5):raise ValueError('Energy/RZF constraint changed')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.GRAD_CLIP_NORM,error_if_nonfinite=True)
                if chain.ce_model.training or any(p.grad is not None for p in chain.ce_model.parameters()):raise ValueError('Frozen CE changed')
                optimizer.step();scheduler.step()
                if step%108==0 or step==end:
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite parameter')
                    if tensor_digest(torch,chain.ce_model.state_dict())!=ce_before:raise ValueError('CE buffer changed')
                    rng=capture_rng(torch,np)
                    monitor=old.progress_validation(torch,warmup,chain,data,'fdma_control' if cfg.is_fdma(arm) else 'specialist_g4') if step%756==0 or step==end else None
                    restore_rng(torch,np,rng)
                    payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=step,cumulative_updates=cfg.SOURCE_UPDATES+step,
                        codec_model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(torch,np),
                        recipe=cfg.recipe(arm),config_fingerprint=cfg.fingerprint(arm),source=cfg.source_identity(arm),
                        code_sha256=code['sha256'],ce_identity=ce,data_protocol=data.protocol)
                    target=folder/'checkpoint_latest.pt';temporary=target.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(target)
                    if ce_identity()!=ce or (step==end and sha256(cfg.source_identity(arm)['path'])!=cfg.source_identity(arm)['sha256']):raise ValueError('Frozen input changed')
                    manifest.update(step=step,cumulative_updates=cfg.SOURCE_UPDATES+step,status='completed' if step==3024 else 'paused' if step==end else 'running',
                        checkpoint_sha256=sha256(target),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        updated_utc=now(),last_loss=float(loss.detach()),last_grad_norm=float(gradient),last_condition=condition)
                    save_json(folder/'manifest.json',manifest)
                    record=dict(step=step,cumulative_updates=cfg.SOURCE_UPDATES+step,loss=float(loss.detach()),condition=condition,monitor=monitor,utc=now())
                    with (folder/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                    print(json.dumps(record),flush=True)
        except BaseException as error:
            manifest.update(status='failed',error=f'{type(error).__name__}: {error}',updated_utc=now())
            save_json(folder/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'),manifest)
            save_json(folder/'manifest.json',manifest);raise
        return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=cfg.ARMS)
    modes=parser.add_mutually_exclusive_group()
    for name in ('dry-run','train','preflight'):modes.add_argument('--'+name,action='store_true')
    modes.add_argument('--stop-after',type=int);parser.add_argument('--resume',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.preflight:
        if args.report is None:parser.error('--preflight requires an unused --report path')
        preflight(args.report);return 0
    if args.report is not None:parser.error('--report requires --preflight')
    if not(args.dry_run or args.train or args.stop_after is not None):parser.print_help();return 0
    if args.arm is None:parser.error('Explicit --arm required')
    if args.dry_run:
        if args.resume:parser.error('Dry-run does not resume')
        print(json.dumps(dict(recipe=cfg.recipe(args.arm),tolerances=EQUIVALENCE_TOLERANCES,weights_read=False),indent=2));return 0
    if args.stop_after is not None and (not 0<args.stop_after<=3024 or args.stop_after%108):parser.error('Pause must be108-step multiples')
    train(args.arm,resume=args.resume,stop_after=args.stop_after);return 0


if __name__=='__main__':raise SystemExit(main())
