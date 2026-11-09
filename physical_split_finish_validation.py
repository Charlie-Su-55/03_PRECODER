"""Read-only acceptance of one finished physical-split training batch.

Training identity remains the complete archived914719b source set.  This
independent postprocessing identity never substitutes the current HEAD into an
old receipt and never calls a training, optimizer-step or old HEAD-bound loader.
"""
from __future__ import annotations

from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import subprocess
import traceback

from configs import twc_v1_k8_physical_split_config as cfg
import train_twc_v1_k8_physical_split as training
import evaluate_twc_v1_k8_physical_split as old_evaluation

ROOT=Path(__file__).resolve().parent
TRAINING_COMMIT='914719b5b6e0323b6f5a07c1d875608ff6307cf1'
TRAINING_SOURCE_FILES=tuple(training.SOURCE_FILES)
FINISH_ROOT=ROOT/'runs/twc_v1_k8_physical_split_finish_v1'
OUTPUT_ROOT=FINISH_ROOT
BATCH_PATH=ROOT/'runs/twc_v1_k8_physical_split_v1/batches/batch_0626e5795f5941af8a5a67d2e465564a'
ACCEPTANCE_PATH=FINISH_ROOT/'training_acceptance.json'
ACCEPTANCE_FORMAT='twc_v1_k8_physical_split_finish_training_acceptance_v1'
POSTPROCESSING_FILES=('physical_split_finish_io.py','physical_split_finish_validation.py',
    'evaluate_twc_v1_k8_physical_split_finish.py','finish_twc_v1_k8_physical_split.py',
    'tests/test_physical_split_finish_io.py','tests/test_physical_split_finish_validation.py',
    'tests/test_finish_twc_v1_k8_physical_split.py','docs/TWC_V1_K8_PHYSICAL_SPLIT_FINISH.md')
make_data,ce_identity,tensor_digest=training.make_data,training.ce_identity,training.tensor_digest
canonical=cfg.normalize_json


def now():return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def json_sha256(value):return cfg.json_sha256(value)
def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _inside(path,root=ROOT):
    path=Path(path)
    if path.is_symlink():raise ValueError('Symlink is not an owned immutable evidence file: '+str(path))
    resolved=path.resolve();allowed=Path(root).resolve()
    if not resolved.is_relative_to(allowed):raise ValueError('Evidence escaped the03 repository: '+str(path))
    return resolved


def write_once_json(path,value):
    path=_inside(path,FINISH_ROOT)
    if path==FINISH_ROOT.resolve():raise ValueError('A new finish output file is required')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


@lru_cache(maxsize=1)
def training_code():
    return dict(commit=TRAINING_COMMIT,sha256={name:hashlib.sha256(subprocess.check_output(
        ['git','show',TRAINING_COMMIT+':'+name],cwd=ROOT)).hexdigest() for name in TRAINING_SOURCE_FILES})


archived_training_code=training_code


def postprocessing_code_record():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={name:sha256(ROOT/name) for name in POSTPROCESSING_FILES})


def verify_training_source_files(code=None):
    expected=training_code()
    if code is not None and canonical(code)!=canonical(expected):raise ValueError('Archived training identity differs')
    if len(expected['sha256'])!=len(TRAINING_SOURCE_FILES):raise ValueError('Incomplete archived training source list')
    # Raw bytes on the executing server, including STATUS. No CRLF normalization.
    for name,digest in expected['sha256'].items():
        if sha256(_inside(ROOT/name))!=digest:raise ValueError('Original training source bytes changed: '+name)
    return expected


verify_source_files=verify_training_source_files


def verify_files(files):
    if not isinstance(files,dict) or not files:raise ValueError('Nonempty immutable evidence hash map required')
    for filename,digest in files.items():
        if not _digest(digest) or sha256(_inside(filename))!=digest:
            raise ValueError('Immutable training/source evidence changed: '+str(filename))


def validate_source_batch(batch,code):
    if (code.get('commit')!=TRAINING_COMMIT or set(code.get('sha256',{}))!=set(TRAINING_SOURCE_FILES)
            or not all(_digest(v) for v in code['sha256'].values())):
        raise ValueError('The complete archived914719b training identity is required')
    if (batch.get('status')!='failed' or canonical(batch.get('code'))!=canonical(code)
            or not batch.get('error') or set(batch.get('training',{}))!=set(cfg.ARMS)):
        raise ValueError('The exact failed post-training physical-split batch is required')
    stages=batch.get('stages',{})
    required=('sources','cuda_preflight',*('train_'+arm for arm in cfg.ARMS))
    for name in required:
        stage=stages.get(name,{})
        if stage.get('status')!='completed' or not stage.get('evidence_sha256'):
            raise ValueError('Original completed stage evidence missing: '+name)
    audit=stages.get('training_audit',{})
    if audit.get('status')!='failed' or not audit.get('error'):
        raise ValueError('Only the recorded failed training_audit may be finished')
    if 'dict contains fields not in fieldnames' not in str(audit['error']):
        raise ValueError('The recorded audit failure is not the reviewed CSV union field mismatch')
    if any(stage.get('status') in ('failed','running') for name,stage in stages.items() if name!='training_audit'):
        raise ValueError('An earlier/non-audit stage is incomplete')
    for name in ('terminal_alpha_selection','terminal_validation'):
        stage=stages.get(name,{})
        if stage and (stage.get('status') not in ('not_started','pending') or stage.get('evidence_sha256') or stage.get('result')):
            raise ValueError('Original alpha selection/evaluation must not have started')
    if batch.get('alpha_selection') or batch.get('validation'):
        raise ValueError('Original alpha/evaluation results already exist')
    return dict(status=batch['status'],error=batch['error'],training_audit=canonical(audit))


def verify_training_metadata(payload,manifest,initial,terminal,events,arm,source_acceptance,code,actual_sha,
                             *,source_acceptance_sha256=None):
    """Pure JSON-safe terminal schema gate; tensor validation is separate."""
    if arm not in cfg.ARMS or not _digest(actual_sha):raise ValueError('Prescribed arm and actual checkpoint SHA required')
    if (code.get('commit')!=TRAINING_COMMIT or set(code.get('sha256',{}))!=set(TRAINING_SOURCE_FILES)
            or not all(_digest(v) for v in code['sha256'].values())):
        raise ValueError('The complete archived914719b training identity is required')
    cfg.acceptance_identity(source_acceptance)
    if canonical(source_acceptance.get('code'))!=canonical(code):raise ValueError('Original source lock belongs to another training implementation')
    cfg.validate_resume_metadata(payload,arm,source_acceptance,code_sha256=code['sha256'])
    counts=cfg.update_counts(6048)
    if payload['step']!=6048 or payload.get('update_counts')!=counts:
        raise ValueError('Complete head6048/BS5940 terminal state required')
    expected=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,status='completed',step=6048,
        phase=cfg.phase(6048),update_counts=counts,source_codec_updates=34192,source_bs_updates=6048,
        source_data_address=40240,data_global_step=46288,ue_frozen=True,code=code,
        source=cfg.source_identity(arm,source_acceptance),source_acceptance=cfg.acceptance_identity(source_acceptance),
        recipe=cfg.recipe(arm,source_acceptance),config_fingerprint=cfg.fingerprint(arm,source_acceptance),
        ce_identity=payload['ce_identity'],checkpoint_sha256=actual_sha,
        actual_condition_counts=training.expected_counts(arm,6048))
    for name,value in expected.items():
        if canonical(manifest.get(name))!=canonical(value):raise ValueError('Original terminal manifest '+name+' changed')
    for name,value in dict(code=code,source=expected['source'],source_acceptance=expected['source_acceptance'],
        recipe=expected['recipe'],ce_identity=payload['ce_identity'],new_two_optimizers=True).items():
        if canonical(initial.get(name))!=canonical(value):raise ValueError('Original initialization '+name+' changed')
    migration=initial.get('migration',{})
    if migration.get('strict') is not True or migration.get('source_values_preserved') is not True:
        raise ValueError('Strict source migration evidence missing')
    names=payload.get('optimizer_parameter_names')
    if (not isinstance(names,dict) or set(names)!= {'head','bs'} or any(not isinstance(v,list) or not v for v in names.values())
            or names!=manifest.get('optimizer_parameter_names') or names!=initial.get('optimizer_parameter_names')):
        raise ValueError('Complete stable dual-optimizer parameter identities required')
    flat=[name for group in names.values() for name in group]
    if len(set(flat))!=len(flat) or any(not isinstance(name,str) or name.startswith('encoder.') for name in flat):
        raise ValueError('Optimizer identities overlap or include frozen UE')
    if any(not name.startswith(training.HEAD_PREFIX) for name in names['head']) or any(name.startswith(training.HEAD_PREFIX) for name in names['bs']):
        raise ValueError('Head and BS optimizers are not separated')
    if arm.endswith('control') and any(name.startswith(training.BRIDGE_PREFIX) for name in names['bs']):
        raise ValueError('Control bridge entered the BS optimizer')
    if (payload.get('actual_condition_counts')!=expected['actual_condition_counts']
            or not _digest(payload.get('source_acceptance_sha256'))
            or source_acceptance_sha256 is not None and payload['source_acceptance_sha256']!=source_acceptance_sha256):
        raise ValueError('Checkpoint source lock or condition budget changed')
    for key in ('frozen_state','training_input_history','layout_transmit_history','monitor_records','resume_records'):
        if canonical(payload.get(key))!=canonical(manifest.get(key)):raise ValueError('Checkpoint/manifest '+key+' differs')
    if payload.get('frozen_state')!=initial.get('frozen_state'):
        raise ValueError('Frozen UE/CE differs from initialization')
    if not all(_digest(payload.get(name)) for name in ('training_input_history','layout_transmit_history')):
        raise ValueError('Actual input/transmit histories missing')
    path=str(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')
    if canonical(terminal)!=canonical(dict(path=path,sha256=actual_sha,update_counts=counts,data_global_step=46288)):
        raise ValueError('Original terminal receipt changed')
    if not isinstance(events,list) or [row.get('step') for row in events]!=list(range(108,6049,108)):
        raise ValueError('Complete56 saved progress records required')
    for row in events:
        step=row['step']
        if (row.get('update_counts')!=cfg.update_counts(step) or row.get('phase')!=cfg.phase(step)
                or row.get('data_global_step')!=40240+step
                or canonical(row.get('condition'))!=canonical(cfg.schedule(arm,step))):
            raise ValueError('Saved data/optimizer/phase schedule differs')
        fp=row.get('input_fingerprints',{})
        if set(fp)!={'channel','pilot','data','X','Y','Hest'} or not all(_digest(v) for v in fp.values()):
            raise ValueError('Complete actual channel/noise/transmit fingerprints required')
    if set(payload.get('monitor_records',{}))!={str(step) for step in cfg.MONITOR_STEPS} or '108' not in payload.get('resume_records',{}):
        raise ValueError('Original monitors or included108 resume receipt missing')
    verified=dict(manifest);verified['checkpoint_path']=path
    return verified


def _validate_rng(torch,record):
    import numpy as np
    if not isinstance(record,dict) or not {'python','numpy','torch_cpu','torch_cuda'}<=set(record):raise ValueError('Complete RNG state required')
    if not isinstance(record['torch_cuda'],list):raise ValueError('Saved CUDA RNG states must be a complete list')
    values=[record['torch_cpu'],*record['torch_cuda']]
    if any(not torch.is_tensor(v) or v.dtype!=torch.uint8 or v.ndim!=1 or v.device.type!='cpu' or v.numel()==0 for v in values):
        raise ValueError('Saved CPU/CUDA RNG states must be CPU byte vectors')
    random.Random().setstate(record['python']);np.random.RandomState().set_state(record['numpy'])
    # Validate the CPU generator state without altering any default RNG. CUDA
    # byte-state evidence is retained and compared across checkpoints separately;
    # frozen evaluation never restores it into this machine's visible devices.
    torch.Generator(device='cpu').set_state(record['torch_cpu'])
    return len(record['torch_cuda'])


def _validate_optimizer(torch,payload,key,parameters,updates):
    optimizer=payload[key+'_optimizer'];scheduler=payload[key+'_scheduler'];groups=optimizer.get('param_groups',[])
    if len(groups)!=1:raise ValueError('Exactly one '+key+' AdamW parameter group required')
    for field,value in dict(lr=2e-4,initial_lr=2e-4,weight_decay=1e-5,eps=1e-8,betas=(.9,.999)).items():
        if canonical(groups[0].get(field))!=canonical(value):raise ValueError(key+' optimizer '+field+' differs')
    if (scheduler.get('last_epoch')!=updates or scheduler.get('_step_count')!=updates+1
            or scheduler.get('base_lrs')!=[2e-4] or scheduler.get('_last_lr')!=[2e-4]):
        raise ValueError(key+' scheduler update count differs')
    names=[name for name,_ in parameters];params=[parameter for _,parameter in parameters]
    if payload['optimizer_parameter_names'].get(key)!=names or groups[0].get('params')!=list(range(len(params))):
        raise ValueError(key+' optimizer parameter names/order differ')
    moments=optimizer.get('state',{})
    if not isinstance(moments,dict) or bool(moments)!=(updates>0):raise ValueError(key+' optimizer moments/update count differ')
    for index,values in moments.items():
        if type(index) is not int or not 0<=index<len(params) or set(values)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Invalid '+key+' optimizer moment identity')
        if not torch.is_tensor(values['step']) or values['step'].numel()!=1 or values['step'].item()!=updates:
            raise ValueError(key+' AdamW step differs')
        for field in ('exp_avg','exp_avg_sq'):
            value=values[field]
            if not torch.is_tensor(value) or value.shape!=params[index].shape or value.dtype!=params[index].dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(key+' moment shape/dtype/finite mismatch')
        if not bool((values['exp_avg_sq']>=0).all()):raise ValueError('Negative '+key+' second moment')


def build_chain(torch,device,method,payload,*,include_stages=True):
    # This existing factory performs only construction/strict loading.  No old
    # loader, current-HEAD validator, training entry or migration is invoked.
    chain=training.build_chain(torch,device,method,payload,include_stages=include_stages,initialize=False)
    chain.eval().requires_grad_(False)
    if chain.codec_model.decoder is not chain.codec_model.joint_decoder.base_decoder:raise ValueError('Spatial decoder alias differs')
    if old_evaluation._state_sha(chain.codec_model.state_dict())!=old_evaluation._state_sha(payload['codec_model']):
        raise ValueError('Strict state loading changed trained values or decoder aliases')
    return chain


def _batch_evidence(batch):
    files={}
    for name,stage in batch['stages'].items():
        if stage.get('status')=='completed':
            verify_files(stage['evidence_sha256']);files.update(stage['evidence_sha256'])
    return files


def _verify_auxiliary(payload,arm,files):
    fixed=None
    for step,entry in payload['monitor_records'].items():
        path=_inside(entry['path']);verify_files({str(path):entry['sha256']});files[str(path)]=entry['sha256']
        monitor=read_json(path)
        if (monitor.get('step')!=int(step) or monitor.get('update_counts')!=cfg.update_counts(int(step))
                or monitor.get('alpha')!=.03 or len(monitor.get('summary',[]))!=(6 if cfg.is_fdma(arm) else 12)
                or len(monitor.get('all_six',[]))!=(1 if cfg.is_fdma(arm) else 2)
                or [v.get('address') for v in monitor.get('shared_inputs',[])]!=list(range(14336,14352))):
            raise ValueError('Original full monitor protocol differs: '+arm)
        if fixed is not None and fixed!=monitor['shared_inputs']:raise ValueError('Monitor inputs changed across steps')
        fixed=monitor['shared_inputs']
    for step,entry in payload['resume_records'].items():
        path=_inside(entry['path']);verify_files({str(path):entry['sha256']});files[str(path)]=entry['sha256'];proof=read_json(path)
        if (proof.get('step')!=int(step) or proof.get('next_iteration')!=int(step)+1
                or proof.get('update_counts')!=cfg.update_counts(int(step)) or proof.get('RNG_restored') is not True
                or proof.get('restored_optimizer_states') is not True
                or proof.get('scheduler_epochs')!={'head':int(step),'bs':max(int(step)-108,0)}):
            raise ValueError('Original dual-optimizer resume proof differs')
    return fixed


def _validate_loaded(torch,payload,method,chain):
    if method in cfg.ARMS:
        counts=cfg.update_counts(6048);groups=training.parameter_groups(chain)
        for key in ('head','bs'):_validate_optimizer(torch,payload,key,groups[key],counts[key+'_updates'])
    cuda_count=_validate_rng(torch,payload['rng'])
    frozen=training.frozen_part_digest(torch,chain)
    if frozen!=payload.get('frozen_state'):raise ValueError('Actual frozen UE/CE differs from checkpoint receipt')
    if any(module.training for module in chain.modules()) or any(p.requires_grad for p in chain.parameters()):
        raise ValueError('Frozen evaluation model modes/gradients differ')
    return dict(strict_loading=True,migration=False,state_sha256=old_evaluation._state_sha(payload['codec_model']),
        frozen_state=frozen,saved_cuda_rng_count=cuda_count,model_tensor_device='cpu' if next(chain.parameters()).device.type=='cpu' else 'cuda')


def accept_training(report_path):
    """Server CPU read/strict-load acceptance; writes only the new finish tree."""
    report_path=_inside(report_path,FINISH_ROOT)
    if report_path.exists():raise FileExistsError(report_path)
    if ACCEPTANCE_PATH.exists():
        accepted=load_acceptance();write_once_json(report_path,dict(status='passed',reused=True,record_id=accepted['record_id']))
        return accepted
    report=dict(status='running',created=now(),checks={},additional_training_updates=0);failure=None
    try:
        import torch
        code=verify_training_source_files();post=postprocessing_code_record()
        pointer=cfg.OUTPUT_ROOT/'last_batch.json'
        if _inside(read_json(pointer)['directory'])!=BATCH_PATH.resolve():raise ValueError('Original batch pointer is not the prescribed batch')
        batch_path=BATCH_PATH/'batch_summary.json';batch=read_json(batch_path);failure_evidence=validate_source_batch(batch,code)
        if cfg.SELECTION_PATH.exists() or (cfg.OUTPUT_ROOT/'selection_attempt.json').exists():
            raise ValueError('Original alpha calibration has already started')
        source_acceptance=read_json(cfg.SOURCE_ACCEPTANCE_PATH);cfg.acceptance_identity(source_acceptance)
        if canonical(source_acceptance.get('code'))!=canonical(code):raise ValueError('Original source lock code differs')
        files=_batch_evidence(batch);verify_files(source_acceptance['frozen_files']);files.update(source_acceptance['frozen_files'])
        for name,digest in code['sha256'].items():files[str(ROOT/name)]=digest
        for path in (pointer,batch_path,cfg.SOURCE_ACCEPTANCE_PATH):files[str(path)]=sha256(path)
        for path in BATCH_PATH.iterdir():
            if path.is_file():files[str(_inside(path))]=sha256(path)
        lock_root=cfg.OUTPUT_ROOT/'locks'
        if lock_root.is_dir():
            for path in lock_root.iterdir():
                if path.is_file():files[str(_inside(path))]=sha256(path)
        accepted_training={};progress={};monitor_inputs=None;ce=source_acceptance['ce_identity'];data_protocol=source_acceptance['data_protocol']
        for arm in cfg.ARMS:
            report['current_arm']=arm;base=cfg.OUTPUT_ROOT/'training'/arm;checkpoint=base/'checkpoint_latest.pt';actual=sha256(checkpoint)
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            manifest=read_json(base/'manifest.json');initial=read_json(base/'initialization.json');terminal=read_json(base/'terminal_checkpoint.json')
            events=[json.loads(line) for line in (base/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            verified=verify_training_metadata(payload,manifest,initial,terminal,events,arm,source_acceptance,code,actual,
                source_acceptance_sha256=files[str(cfg.SOURCE_ACCEPTANCE_PATH)])
            item=batch['training'][arm]
            if item.get('checkpoint_path')!=str(checkpoint) or item.get('checkpoint_sha256')!=actual or item.get('step')!=6048:
                raise ValueError('Original batch checkpoint pointer differs: '+arm)
            stage=batch['stages']['train_'+arm]
            for name in ('checkpoint_latest.pt','manifest.json','initialization.json','progress.jsonl','terminal_checkpoint.json'):
                path=base/name;digest=sha256(path)
                if stage['evidence_sha256'].get(str(path))!=digest:raise ValueError('Completed training stage bytes differ: '+str(path))
                files[str(path)]=digest
            actual_monitor=_verify_auxiliary(payload,arm,files)
            if monitor_inputs is not None and actual_monitor!=monitor_inputs:raise ValueError('Full monitor inputs differ across four arms')
            monitor_inputs=actual_monitor
            if canonical(payload['data_protocol'])!=canonical(data_protocol) or payload['ce_identity']!=ce:raise ValueError('Original data/CE protocol differs')
            chain=build_chain(torch,torch.device('cpu'),arm,payload,include_stages=False)
            numeric=_validate_loaded(torch,payload,arm,chain)
            expected_ue=source_acceptance['sources'][cfg.source_method(arm)]['verification']['ue_sha256']
            if numeric['frozen_state']['ue']!=expected_ue:raise ValueError('Trained UE no longer equals frozen Base2 source')
            verified['verification']=numeric;accepted_training[arm]=verified;report['checks'][arm]=numeric
            progress[arm]=events
            del chain,payload
        if len({value['training_input_history'] for value in accepted_training.values()})!=1:
            raise ValueError('Rolling channel/standard-noise histories differ across four arms')
        for family in ('fdma','g4'):
            if accepted_training[family+'_control']['layout_transmit_history']!=accepted_training[family+'_split']['layout_transmit_history']:
                raise ValueError('Same-layout transmit/receive histories differ')
        for index in range(56):
            if len({tuple(progress[arm][index]['input_fingerprints'][name] for name in ('channel','pilot','data')) for arm in cfg.ARMS})!=1:
                raise ValueError('Saved channel/standard-noise samples differ across four arms')
            for family in ('fdma','g4'):
                if len({tuple(progress[family+'_'+variant][index]['input_fingerprints'][name] for name in ('X','Y','Hest')) for variant in ('control','split')})!=1:
                    raise ValueError('Saved same-layout X/Y/Hest samples differ')
        saved_counts={value['verification']['saved_cuda_rng_count'] for value in accepted_training.values()}
        if len(saved_counts)!=1:raise ValueError('Four training checkpoints disagree on saved CUDA RNG count')
        for name in ('visible_cuda_device_count','cuda_device_count'):
            historical=source_acceptance.get('backend',{}).get(name)
            if historical is not None and (type(historical) is not int or historical!=next(iter(saved_counts))):
                raise ValueError('Saved CUDA RNG count differs from the historical backend count')
        verify_files(files)
        record=dict(format=ACCEPTANCE_FORMAT,status='passed',created=now(),training=accepted_training,
            source_batch=str(BATCH_PATH),training_code=code,postprocessing_code=post,
            source_acceptance=cfg.acceptance_identity(source_acceptance),old_source_acceptance=source_acceptance,
            data_protocol=data_protocol,ce_identity=ce,frozen_files=files,source_failure=failure_evidence,
            additional_training_updates=0,CPU_strict_loading_only=True,CUDA_model_evaluation_executed=False)
        record.update(saved_cuda_rng_count=next(iter(saved_counts)),
            training_RNG_restored_for_frozen_evaluation=False,
            training_data_pairing=dict(passed=True,all_four_channel_noise_histories_match=True,
                same_layout_X_Y_Hest_histories_match=True,saved_pairing_points=56,full_monitors_match=True))
        record['record_id']=json_sha256(record);write_once_json(ACCEPTANCE_PATH,record)
        report.update(status='passed',record_id=record['record_id'],training_code=code,postprocessing_code=post,finished=now())
    except BaseException as error:
        failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:write_once_json(report_path,report)
    if failure is not None:raise RuntimeError('Original training acceptance failed: '+str(report_path)) from failure
    return record


def load_acceptance():
    record=read_json(ACCEPTANCE_PATH)
    if (record.get('format')!=ACCEPTANCE_FORMAT or record.get('status')!='passed'
            or record.get('record_id')!=json_sha256({k:v for k,v in record.items() if k!='record_id'})
            or canonical(record.get('training_code'))!=canonical(training_code())
            or record.get('postprocessing_code')!=postprocessing_code_record()
            or record.get('source_batch')!=str(BATCH_PATH) or set(record.get('training',{}))!=set(cfg.ARMS)
            or record.get('additional_training_updates')!=0):raise ValueError('Immutable finish acceptance identity differs')
    verify_training_source_files(record['training_code']);verify_files(record['frozen_files'])
    cfg.acceptance_identity(record['old_source_acceptance'])
    return record


def load_methods(torch,warmup,data_protocol,ce,acceptance):
    """Pool strict frozen variants for the independent GPU evaluator only."""
    if canonical(data_protocol)!=canonical(acceptance['data_protocol']) or ce!=acceptance['ce_identity']:
        raise ValueError('Finish evaluation data/CE protocol differs')
    verify_files(acceptance['frozen_files']);old_lock=acceptance['old_source_acceptance']
    chains={};states={};methods={};parameters=[]
    for method in cfg.METHODS:
        if method in cfg.ARMS:path=Path(acceptance['training'][method]['checkpoint_path']);digest=acceptance['training'][method]['checkpoint_sha256']
        else:path=Path(old_lock['sources'][method]['identity']['path']);digest=old_lock['sources'][method]['identity']['sha256']
        if sha256(path)!=digest:raise ValueError('Locked evaluation checkpoint changed: '+method)
        payload=torch.load(path,map_location='cpu',weights_only=False);kind=cfg.variant(method)
        if kind not in chains:chains[kind]=build_chain(torch,torch.device('cuda:0'),method,payload,include_stages=True)
        chain=chains[kind];expected=chain.codec_model.state_dict();state=payload['codec_model']
        if set(expected)!=set(state):raise ValueError('Complete frozen variant state required')
        for name,value in state.items():
            if not torch.is_tensor(value) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Frozen variant tensor differs: '+name)
        chain.codec_model.load_state_dict(state,strict=True);chain.eval().requires_grad_(False)
        numeric=_validate_loaded(torch,payload,method,chain)
        if method in cfg.ARMS and numeric['saved_cuda_rng_count']!=acceptance['saved_cuda_rng_count']:
            raise ValueError('Saved training CUDA RNG evidence differs from the locked acceptance')
        expected_ue=old_lock['sources'][cfg.source_method(method)]['verification']['ue_sha256']
        if numeric['frozen_state']['ue']!=expected_ue:raise ValueError('Evaluation UE differs from the locked source')
        full=sum(p.numel() for p in chain.codec_model.parameters());ue=sum(p.numel() for p in chain.codec_model.encoder.parameters())
        parameter=dict(arm=method,variant=kind,codec_parameters=full,encoder_parameters=ue,bs_parameters=full-ue,
            ce_parameters=sum(p.numel() for p in chain.ce_model.parameters()),note='Actual stored numel; not MACs or latency')
        parameters.append(parameter);states[method]=state
        methods[method]=dict(path=str(path),sha256=digest,variant=kind,strict_loading=True,migration=False,
            state_sha256=numeric['state_sha256'],saved_cuda_rng_count=numeric['saved_cuda_rng_count'],
            actual_saved_recipe=payload['recipe'],parameters=parameter,additional_training_updates=0,
            training_code=acceptance['training_code'],postprocessing_code=acceptance['postprocessing_code'])
        del payload
    return dict(chains=chains,states=states,methods=methods,parameters=parameters,frozen_files=acceptance['frozen_files'])
