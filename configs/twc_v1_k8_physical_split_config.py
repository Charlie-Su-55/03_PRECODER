"""Standard-library-only protocol and identities for the physical/latent split screen."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
from . import twc_v1_k8_decoder_screen_config as previous

REPO_ROOT=Path(__file__).resolve().parents[1]
NAMESPACE='twc_v1_k8_physical_split_v1'
CHECKPOINT_FORMAT=NAMESPACE
OUTPUT_ROOT=REPO_ROOT/'runs'/NAMESPACE
SOURCE_ACCEPTANCE_PATH=OUTPUT_ROOT/'source_acceptance.json'
SOURCE_ACCEPTANCE_FORMAT='twc_v1_k8_physical_split_source_acceptance_v1'
SELECTION_PATH=OUTPUT_ROOT/'alpha_selection.json'
SELECTION_FORMAT='twc_v1_k8_physical_split_alpha_selection_v1'
ARMS=('fdma_control','fdma_split','g4_control','g4_split')
SOURCE_METHODS=('old_fdma_base2','old_g4_base2','old_fdma_deep4','old_g4_dc2')
METHODS=ARMS+SOURCE_METHODS
SOURCE_COMMIT='b6336de4c80588e8d1ddb57a0607f774182a0f67'
SEED=20261107
SOURCE_LOCAL_UPDATES=6048
SOURCE_CODEC_UPDATES=34192
SOURCE_BS_UPDATES=6048
SOURCE_DATA_ADDRESS=40240
UPDATES=6048
HEAD_ONLY_UPDATES=108
BS_UPDATES=UPDATES-HEAD_ONLY_UPDATES
BATCH_SIZE=4
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-5
GRAD_CLIP_NORM=1.
SAVE_EVERY=108
FIXED_ALPHA=.03
MONITOR_ALPHA=.03
MONITOR_STEPS=(1512,3024,4536,6048)
MONITOR_START_INDEX=14336
MONITOR_BATCHES=16
MONITOR_SAMPLES=64
ALPHA_CALIBRATION_START_INDEX=15360
ALPHA_CALIBRATION_BATCHES=16
ALPHA_CALIBRATION_SAMPLES=64
VALIDATION_START_INDEX=16384
VALIDATION_BATCHES=64
VALIDATION_SAMPLES=256
ALPHAS=previous.ALPHAS
GAMMAS=previous.GAMMAS
PILOT_ENERGIES=previous.PILOT_ENERGIES
ENERGY_RATIOS=previous.ENERGY_RATIOS
PILOT_ENERGY=16.
DOWNLINK_SNR_DB=25.
CDL_MODEL=previous.CDL_MODEL
SPEED_MPS=previous.SPEED_MPS
BASE_DATA_SEED=previous.BASE_DATA_SEED
DATA_SEED_NAMESPACE=previous.DATA_SEED_NAMESPACE
CE_CHECKPOINT=previous.CE_CHECKPOINT
CE_CALIBRATION=previous.CE_CALIBRATION
CE_SHA256=dict(previous.CE_SHA256)

def _method(method):
    if type(method) is not str or method not in METHODS:raise ValueError('A prescribed physical-split method is required')
    return method
def is_new(method):return _method(method) in ARMS
def is_fdma(method):return 'fdma' in _method(method)
def bridge_enabled(method):return _method(method) in ('fdma_split','g4_split')
def source_method(method):
    _method(method)
    return ('old_fdma_base2' if is_fdma(method) else 'old_g4_base2') if method in ARMS else method
def variant(method):
    _method(method)
    return ('split' if bridge_enabled(method) else 'control') if method in ARMS else method.rsplit('_',1)[1]
def normalize_json(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')))
def json_sha256(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
sha256_file=previous.sha256_file

def planned_source_identity(method):
    arm=source_method(method).removeprefix('old_')
    directory=REPO_ROOT/'runs/twc_v1_k8_decoder_screen_v1/training'/arm
    return dict(arm=arm,path=str(directory/'checkpoint_latest.pt'),sha256=None,
        manifest_path=str(directory/'manifest.json'),initialization_path=str(directory/'initialization.json'),
        progress_path=str(directory/'progress.jsonl'),terminal_path=str(directory/'terminal_checkpoint.json'),
        seed=SEED,local_step=SOURCE_LOCAL_UPDATES,source_codec_updates=SOURCE_CODEC_UPDATES,
        source_bs_updates=SOURCE_BS_UPDATES,data_global_step=SOURCE_DATA_ADDRESS,source_implementation_commit=SOURCE_COMMIT)
def acceptance_identity(record):
    if (not isinstance(record,dict) or record.get('format')!=SOURCE_ACCEPTANCE_FORMAT or record.get('status')!='passed'
            or not isinstance(record.get('sources'),dict) or set(record['sources'])!=set(SOURCE_METHODS)):
        raise ValueError('Immutable four-source physical-split acceptance required')
    hashes={}
    for method in SOURCE_METHODS:
        evidence=record['sources'][method]
        if not isinstance(evidence,dict) or not isinstance(evidence.get('identity'),dict):raise ValueError('Source identity evidence required')
        identity=evidence['identity']
        for key,value in planned_source_identity(method).items():
            if key!='sha256' and identity.get(key)!=value:raise ValueError('Accepted historical source '+key+' changed: '+method)
        digest=identity.get('sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Server-verified actual source SHA256 required')
        if not evidence.get('verification') or not evidence.get('frozen_files'):raise ValueError('Source cross-check receipts required')
        hashes[method]=digest
    digest=json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Source acceptance record changed')
    return dict(format=SOURCE_ACCEPTANCE_FORMAT,record_id=digest,source_sha256=hashes)
def source_identity(method,acceptance=None):
    _method(method)
    if acceptance is None:return planned_source_identity(method)
    acceptance_identity(acceptance)
    return normalize_json(acceptance['sources'][source_method(method)]['identity'])
def selection_identity(record):
    selected=record.get('selected_alphas') if isinstance(record,dict) else None
    if (not isinstance(record,dict) or record.get('format')!=SELECTION_FORMAT or record.get('status')!='passed'
            or normalize_json(record.get('grid'))!=list(ALPHAS) or not isinstance(selected,dict) or set(selected)!=set(METHODS)):
        raise ValueError('Independent eight-checkpoint global-alpha lock required')
    for method,alpha in selected.items():
        if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or alpha not in ALPHAS:raise ValueError('Exactly one legal global alpha required: '+method)
    digest=json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Independent alpha lock changed')
    return dict(format=SELECTION_FORMAT,record_id=digest,selected_alphas={method:float(selected[method]) for method in METHODS})

def phase(step):
    if type(step) is not int or not 0<=step<=UPDATES:raise ValueError('Data iteration must lie in0..6048')
    return 'head_only' if step<=HEAD_ONLY_UPDATES else 'head_and_bs'
def update_counts(step):
    phase(step)
    return dict(data_iterations=step,head_updates=step,bs_updates=max(step-HEAD_ONLY_UPDATES,0),ue_updates=0,ce_updates=0)
expected_update_counts=update_counts
def schedule(arm,local_step):
    if arm not in ARMS or type(local_step) is not int or not 1<=local_step<=UPDATES:raise ValueError('Prescribed arm/local iteration1..6048 required')
    n=SOURCE_DATA_ADDRESS+local_step-1
    return dict(local_step=local_step,global_step=n+1,gamma_db=GAMMAS[n%6],pilot_energy_per_ue=PILOT_ENERGIES[(n//6)%3],
        group_count=1 if is_fdma(arm) else 4,energy_ratio=None if is_fdma(arm) else list(ENERGY_RATIOS[(n//2)%2]))
def schedule_counts(arm):
    counts=Counter()
    for step in range(1,UPDATES+1):
        value=schedule(arm,step);counts[(value['group_count'],value['gamma_db'],value['pilot_energy_per_ue'],tuple(value['energy_ratio'] or ()))] += 1
    return [dict(group_count=g,gamma_db=gamma,pilot_energy_per_ue=ep,energy_ratio=list(ratio) if ratio else None,count=count)
        for (g,gamma,ep,ratio),count in sorted(counts.items())]
def training_noise_seed(step):return SEED*10_000_000+schedule(ARMS[0],step)['global_step']
def noise_seed(stage,batch_index,gamma_index,kind):
    sizes={'preflight':1,'monitor':16,'alpha_calibration':16,'evaluation':64}
    offsets={'preflight':149_000_000,'monitor':159_000_000,'alpha_calibration':169_000_000,'evaluation':179_000_000}
    if stage not in sizes or type(batch_index) is not int or not 0<=batch_index<sizes[stage]:raise ValueError('Independent noise stage/batch mismatch')
    if type(gamma_index) is not int or not 0<=gamma_index<6 or kind not in ('pilot','data'):raise ValueError('Noise gamma/kind mismatch')
    return SEED*10_000_000+offsets[stage]+batch_index*1000+gamma_index*10+int(kind=='data')

def recipe(arm,acceptance=None):
    if arm not in ARMS:raise ValueError('A prescribed physical-split training arm required')
    identity=acceptance_identity(acceptance) if acceptance is not None else dict(status='pending',usable_for_training=False)
    return normalize_json(dict(version=NAMESPACE,arm=arm,seed=SEED,source=source_identity(arm,acceptance),source_acceptance=identity,
        source_implementation_commit=SOURCE_COMMIT,source_codec_updates=SOURCE_CODEC_UPDATES,source_bs_updates=SOURCE_BS_UPDATES,
        source_data_address=SOURCE_DATA_ADDRESS,target_update_counts=update_counts(UPDATES),head_only_iterations=HEAD_ONLY_UPDATES,
        UE_CE_frozen=True,UE_condition_embedding_frozen=True,physical_head_present=True,bridge_enabled=bridge_enabled(arm),
        physical_path='x_phy=soft+physical_head(context1.detach()); no output power renormalization',
        reconstruction_path='z_sem+bridge(detached physical features); original context1 spatial attention retained',
        physical_loss='FDMA active normalized-tx NMSE;Hybrid0.5*(key branch NMSE+shared branch NMSE), MU mean then batch mean',
        csi_loss='mean physical DL CSI NMSE only',gradient_boundaries=dict(physical_loss='physical_head only',csi_loss='original BS and active bridge only',
            physical_features_detached_before_bridge=True,rate_loss=False,residual_fit_loss=False,beta_calibration=False),
        batch_size=BATCH_SIZE,optimizers=dict(head='fresh AdamW6048 steps',bs='fresh AdamW5940 steps starting iteration109'),
        learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY,grad_clip_norm=GRAD_CLIP_NORM,schedulers='separate constant LambdaLR',save_every=SAVE_EVERY,
        physical=dict(K=8,Nt=32,Nsc=256,total_feedback_RE=256,total_energy_per_ue=64.,noise_variance='0.25*10**(-gamma/10)'),
        channel_data=dict(base_seed=BASE_DATA_SEED,seed_namespace=DATA_SEED_NAMESPACE,cdl_model=CDL_MODEL,speed=SPEED_MPS,
            role='codec_train',address='40240+local_iteration'),conditions=schedule_counts(arm),
        monitor=dict(role='validation',first_batch=MONITOR_START_INDEX,batches=MONITOR_BATCHES,samples=MONITOR_SAMPLES,steps=MONITOR_STEPS,
            alpha=MONITOR_ALPHA,gamma=GAMMAS,pilot_energy=16.,DL=25.,hybrid_ratios=ENERGY_RATIOS,fixed_channels_and_noise=True,selection=False),
        alpha_calibration=dict(role='validation',first_batch=ALPHA_CALIBRATION_START_INDEX,batches=ALPHA_CALIBRATION_BATCHES,
            samples=ALPHA_CALIBRATION_SAMPLES,grid=ALPHAS,one_global_alpha_per_checkpoint=True,both_G4_ratios_equal_weight=True),
        validation=dict(role='validation',first_batch=VALIDATION_START_INDEX,batches=VALIDATION_BATCHES,samples=VALIDATION_SAMPLES,fixed_alpha=.03,also_locked_tuned=True),
        ce_checkpoint=CE_CHECKPOINT,ce_calibration=CE_CALIBRATION,ce_sha256=CE_SHA256,selection='predeclared terminal6048 data iterations;not best',
        metric_units=dict(dl_nmse='linear',physical_codeword_nmse='linear',semantic_distances='linear latent diagnostic, not physical recovery error',
            received_residual='mean power, unnormalized;not directly comparable across layouts'),test_used=False,training_seed_replications=1))
def fingerprint(arm,acceptance=None):return json_sha256(recipe(arm,acceptance))
def inside_root(path,root=OUTPUT_ROOT):
    target,allowed=Path(path).resolve(),Path(root).resolve()
    if target==allowed or not target.is_relative_to(allowed):raise ValueError('A child of the independent physical-split root required')
    return target
def prepare_output(path,*,resume=False,root=OUTPUT_ROOT):
    target=inside_root(path,root)
    if resume:
        if not target.is_dir() or not (target/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Owned physical-split latest required')
    else:target.mkdir(parents=True,exist_ok=False)
    return target
def validate_resume_metadata(payload,arm,acceptance,*,code_sha256=None):
    if arm not in ARMS:raise ValueError('A prescribed physical-split arm required')
    identity=acceptance_identity(acceptance)
    required={'format','arm','seed','step','phase','update_counts','ue_frozen','source_codec_updates','source_bs_updates','source_data_address','data_global_step',
        'recipe','config_fingerprint','source','source_acceptance','codec_model','head_optimizer','bs_optimizer','head_scheduler','bs_scheduler',
        'rng','data_protocol','ce_identity','code_sha256'}
    if not isinstance(payload,dict) or not required<=set(payload):raise ValueError('Incomplete dual-optimizer physical-split checkpoint')
    if payload['format']!=CHECKPOINT_FORMAT or payload['arm']!=arm or type(payload['seed']) is not int or payload['seed']!=SEED:raise ValueError('Format/arm/seed changed')
    step=payload['step']
    if type(step) is not int or not 0<=step<=UPDATES or step%SAVE_EVERY:raise ValueError('Saved108-iteration boundary required')
    counts=payload['update_counts']
    if (payload['ue_frozen'] is not True or not isinstance(counts,dict) or set(counts)!=set(update_counts(step))
            or any(type(value) is not int for value in counts.values())):
        raise ValueError('Frozen UE and explicit integer optimizer/data counters required')
    expected=dict(phase=phase(step),update_counts=update_counts(step),source_codec_updates=SOURCE_CODEC_UPDATES,source_bs_updates=SOURCE_BS_UPDATES,
        source_data_address=SOURCE_DATA_ADDRESS,data_global_step=SOURCE_DATA_ADDRESS+step,recipe=recipe(arm,acceptance),
        config_fingerprint=fingerprint(arm,acceptance),source=source_identity(arm,acceptance),source_acceptance=identity)
    for key,value in expected.items():
        if normalize_json(payload[key])!=normalize_json(value):raise ValueError('Resume '+key+' changed')
    for key in ('codec_model','head_optimizer','bs_optimizer','head_scheduler','bs_scheduler','data_protocol','ce_identity','code_sha256'):
        if not isinstance(payload[key],dict) or not payload[key]:raise ValueError('Complete resume mapping required: '+key)
    if payload['head_scheduler'].get('last_epoch')!=step or payload['bs_scheduler'].get('last_epoch')!=max(step-HEAD_ONLY_UPDATES,0):raise ValueError('Separate scheduler update counts changed')
    if not isinstance(payload['rng'],dict) or not {'python','numpy','torch_cpu','torch_cuda'}<=set(payload['rng']):raise ValueError('Complete default RNG state required')
    if code_sha256 is not None and payload['code_sha256']!=code_sha256:raise ValueError('Physical-split code changed')
    return step
