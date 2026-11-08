"""Independent identities for K8 matched-layout-RZF NMSE/task-loss continuation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from . import twc_v1_k8_task_loss_screen_config as previous

REPO_ROOT=Path(__file__).resolve().parents[1]
NAMESPACE='twc_v1_k8_rzf_matched_training_v1'
CHECKPOINT_FORMAT=NAMESPACE
OUTPUT_ROOT=REPO_ROOT/'runs'/NAMESPACE
CALIBRATION_PATH=OUTPUT_ROOT/'calibration.json'
CALIBRATION_FORMAT='twc_v1_k8_rzf_matched_beta_calibration_v1'
SELECTION_PATH=OUTPUT_ROOT/'alpha_selection.json'
SELECTION_FORMAT='twc_v1_k8_rzf_matched_alpha_selection_v1'
ARMS=('fdma_nmse','fdma_joint','g4_nmse','g4_joint')
METHODS=ARMS+('source_fdma','source_g4')
SOURCE_ARMS={'fdma_nmse':'fdma_nmse','fdma_joint':'fdma_nmse','g4_nmse':'g4_nmse','g4_joint':'g4_nmse'}
SOURCE_METHODS={'source_fdma':'fdma_nmse','source_g4':'g4_nmse'}
SEED=20261107
SOURCE_COMMIT='596d2a4d640c9001b85f78dec939bf032ff4a024'
DIAGNOSTIC_COMMIT='0e0b6e8d258731cf7de98f77e7fd87a470d6d6be'
INTERNAL_SOURCE_COMMIT='b6dc7ed5453ad0f8bc581b157505e4656932c1b7'
ANCESTOR_SOURCE_COMMIT='8096ed965027a92f6e0738c7653b4c3e75f71b8c'
TRAINING_ALPHA_SELECTION_PATH=REPO_ROOT/'runs/twc_v1_k8_rzf_probe_v1/alpha_selection.json'
TRAINING_ALPHAS={'fdma':.03,'g4':.1}
TRAINING_ALPHA_BY_ARM={arm:TRAINING_ALPHAS['fdma' if arm.startswith('fdma_') else 'g4'] for arm in ARMS}
SOURCE_LOCAL_UPDATES=6048
SOURCE_UPDATES=22096
UPDATES=12096
BATCH_SIZE=4
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-5
GRAD_CLIP_NORM=1.0
SAVE_EVERY=108
WARMUP_UPDATES=756
PROGRESS_VALIDATION_STEPS=(3024,6048,9072,12096)
MONITOR_START_INDEX=8960
MONITOR_BATCHES=16
MONITOR_SAMPLES=64
GAMMAS=previous.GAMMAS
PILOT_ENERGIES=previous.PILOT_ENERGIES
ENERGY_RATIOS=previous.ENERGY_RATIOS
DOWNLINK_SNR_DB=25.
PILOT_ENERGY=16.
CALIBRATION_ROLE='codec_train'
CALIBRATION_START_INDEX=500100
CALIBRATION_BATCHES=12
ALPHAS=(.0001,.0003,.001,.003,.01,.03,.1,.3,1.)
ALPHA_CALIBRATION_ROLE='validation'
ALPHA_CALIBRATION_START_INDEX=6144
ALPHA_CALIBRATION_BATCHES=16
ALPHA_CALIBRATION_SAMPLES=64
VALIDATION_START_INDEX=7168
VALIDATION_BATCHES=64
VALIDATION_SAMPLES=256
CDL_MODEL=previous.CDL_MODEL
SPEED_MPS=previous.SPEED_MPS
BASE_DATA_SEED=previous.BASE_DATA_SEED
DATA_SEED_NAMESPACE=previous.DATA_SEED_NAMESPACE
CE_CHECKPOINT=previous.CE_CHECKPOINT
CE_CALIBRATION=previous.CE_CALIBRATION
CE_SHA256=dict(previous.CE_SHA256)
ORDERED_ARCHITECTURE=previous.ORDERED_ARCHITECTURE
SOURCE_SHA256={
    'fdma_nmse':'5feae0f5c3d6da9386a05d6c4ff1a5d939c54a7a1ebf2ea7065e23fa46dd13be',
    'g4_nmse':'6fa136d9ce5afbc545a85d1646018dbd609df2a9b2a4e2524c42aa7c4fb67d57',
}


def _arm(arm):
    if type(arm) is not str or arm not in ARMS:raise ValueError('A prescribed matched-training arm is required')
    return arm


def is_fdma(method):
    if type(method) is not str or method not in METHODS:raise ValueError('Unknown matched-training method')
    return method in ('fdma_nmse','fdma_joint','source_fdma')


def is_joint(arm):
    return _arm(arm) in ('fdma_joint','g4_joint')


def training_alpha(method):
    return TRAINING_ALPHAS['fdma' if is_fdma(method) else 'g4']


def normalize_json(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')))


def json_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def source_identity(method):
    if method in SOURCE_METHODS:arm=SOURCE_METHODS[method]
    else:arm=SOURCE_ARMS[_arm(method)]
    directory=REPO_ROOT/'runs/twc_v1_k8_task_loss_screen_v1/training'/arm
    return dict(arm=arm,source_arm=arm,seed=SEED,step=SOURCE_LOCAL_UPDATES,cumulative_updates=SOURCE_UPDATES,
        implementation_commit=SOURCE_COMMIT,diagnostic_commit=DIAGNOSTIC_COMMIT,
        internal_source_commit=INTERNAL_SOURCE_COMMIT,ancestor_source_commit=ANCESTOR_SOURCE_COMMIT,
        path=str(directory/'checkpoint_latest.pt'),sha256=SOURCE_SHA256[arm],
        manifest_path=str(directory/'manifest.json'),initialization_path=str(directory/'initialization.json'),
        progress_path=str(directory/'progress.jsonl'))


def calibration_identity(record):
    if not isinstance(record,dict):raise ValueError('A completed immutable new beta calibration record is required')
    beta=record.get('beta_target')
    if (record.get('format')!=CALIBRATION_FORMAT or record.get('status')!='passed' or isinstance(beta,bool)
            or not isinstance(beta,(int,float)) or not math.isfinite(beta) or beta<=0):
        raise ValueError('New matched beta calibration format/status/finite positive beta mismatch')
    digest=json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Immutable new beta calibration changed')
    return dict(format=CALIBRATION_FORMAT,record_id=digest,beta_target=float(beta))


def selection_identity(record):
    if not isinstance(record,dict):raise ValueError('A completed independent final alpha lock is required')
    selected=record.get('selected_alphas')
    if (record.get('format')!=SELECTION_FORMAT or record.get('status')!='passed'
            or normalize_json(record.get('grid'))!=list(ALPHAS)
            or not isinstance(selected,dict) or set(selected)!=set(METHODS)):
        raise ValueError('Final alpha selection format/status/grid/six-checkpoint identity mismatch')
    for method,alpha in selected.items():
        if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or alpha not in ALPHAS:
            raise ValueError('Exactly one candidate alpha per complete checkpoint is required: '+method)
    digest=json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Independent final alpha lock changed')
    return dict(format=SELECTION_FORMAT,record_id=digest,selected_alphas={method:float(selected[method]) for method in METHODS})


def schedule(arm,local_step):
    _arm(arm)
    if type(local_step) is not int or not 1<=local_step<=UPDATES:raise ValueError('Local update must lie in [1,12096]')
    n=SOURCE_UPDATES+local_step-1
    return dict(local_step=local_step,global_step=n+1,gamma_db=GAMMAS[n%6],
        pilot_energy_per_ue=PILOT_ENERGIES[(n//6)%3],group_count=1 if is_fdma(arm) else 4,
        energy_ratio=None if is_fdma(arm) else list(ENERGY_RATIOS[(n//2)%2]))


def schedule_counts(arm):
    counts=Counter()
    for step in range(1,UPDATES+1):
        value=schedule(arm,step)
        counts[(value['group_count'],value['gamma_db'],value['pilot_energy_per_ue'],tuple(value['energy_ratio'] or ()))] += 1
    return [dict(group_count=g,gamma_db=gamma,pilot_energy_per_ue=ep,energy_ratio=list(ratio) if ratio else None,count=count)
        for (g,gamma,ep,ratio),count in sorted(counts.items())]


def training_noise_seed(local_step):
    return SEED*10_000_000+schedule(ARMS[0],local_step)['global_step']


def noise_seed(stage,batch_index,gamma_index,kind):
    sizes={'beta_calibration':12,'monitor':16,'alpha_calibration':16,'evaluation':64}
    offsets={'beta_calibration':69_000_000,'monitor':79_000_000,'alpha_calibration':89_000_000,'evaluation':99_000_000}
    if stage not in sizes or type(batch_index) is not int or not 0<=batch_index<sizes[stage]:
        raise ValueError('Independent diagnostic noise stage/batch index mismatch')
    if type(gamma_index) is not int or not 0<=gamma_index<len(GAMMAS) or kind not in ('pilot','data'):
        raise ValueError('Diagnostic noise gamma/kind mismatch')
    return SEED*10_000_000+offsets[stage]+batch_index*1000+gamma_index*10+int(kind=='data')


def loss_weight(arm,local_step,calibration):
    schedule(arm,local_step);target=calibration_identity(calibration)['beta_target']
    return target*min(local_step/WARMUP_UPDATES,1.) if is_joint(arm) else 0.


def recipe(arm,calibration=None):
    _arm(arm)
    identity=calibration_identity(calibration) if calibration is not None else dict(status='pending',usable_for_training=False)
    return normalize_json(dict(version=NAMESPACE,arm=arm,seed=SEED,source=source_identity(arm),
        architecture=ORDERED_ARCHITECTURE,new_updates=UPDATES,source_updates=SOURCE_UPDATES,batch_size=BATCH_SIZE,
        optimizer='AdamW',learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY,grad_clip_norm=GRAD_CLIP_NORM,
        scheduler='constant LambdaLR',optimizer_initialization='fresh; same source NMSE checkpoint for both losses within each layout',
        loss='mean per-user DL NMSE minus beta times mean sum rate' if is_joint(arm) else 'mean per-user DL NMSE',
        rate_loss_enabled=is_joint(arm),rate_beta_warmup_updates=WARMUP_UPDATES,calibration=identity,beta_target=identity.get('beta_target'),
        training_rzf_alpha=training_alpha(arm),training_alphas=TRAINING_ALPHAS,
        training_alpha_provenance=dict(path=str(TRAINING_ALPHA_SELECTION_PATH),diagnostic_commit=DIAGNOSTIC_COMMIT,
            source_methods=['fdma_nmse','g4_nmse'],expected_alphas=TRAINING_ALPHAS,new_selection_does_not_change_training=True),
        physical=dict(K=8,Nt=32,Nsc=256,total_feedback_RE=256,total_energy_per_ue=64.,noise_variance='0.25*10**(-gamma/10)',
            downlink_snr_db=DOWNLINK_SNR_DB),ce_frozen=True,ce_checkpoint=CE_CHECKPOINT,ce_calibration=CE_CALIBRATION,ce_sha256=CE_SHA256,
        channel_data=dict(base_seed=BASE_DATA_SEED,seed_namespace=DATA_SEED_NAMESPACE,cdl_model=CDL_MODEL,speed=SPEED_MPS,
            training_role='codec_train',training_address='22096+local_step'),
        conditions=dict(gamma=GAMMAS,pilot_energy=PILOT_ENERGIES,key_shared_ratios=ENERGY_RATIOS,
            group_count=1 if is_fdma(arm) else 4,counts=schedule_counts(arm)),
        beta_calibration=dict(role=CALIBRATION_ROLE,first_batch=CALIBRATION_START_INDEX,batches=CALIBRATION_BATCHES,
            two_NMSE_sources_only=True,training_alpha_by_layout=TRAINING_ALPHAS,record=str(CALIBRATION_PATH)),
        monitor=dict(role='validation',first_batch=MONITOR_START_INDEX,batches=MONITOR_BATCHES,samples=MONITOR_SAMPLES,
            steps=PROGRESS_VALIDATION_STEPS,gamma=GAMMAS,pilot_energy=PILOT_ENERGY,hybrid_ratios=ENERGY_RATIOS,
            fixed_channels_and_noise_across_steps=True,checkpoint_selection=False),
        final_alpha_calibration=dict(role=ALPHA_CALIBRATION_ROLE,first_batch=ALPHA_CALIBRATION_START_INDEX,batches=ALPHA_CALIBRATION_BATCHES,
            samples=ALPHA_CALIBRATION_SAMPLES,grid=ALPHAS,one_global_alpha_per_checkpoint=True,record=str(SELECTION_PATH)),
        validation=dict(role='validation',first_batch=VALIDATION_START_INDEX,batches=VALIDATION_BATCHES,samples=VALIDATION_SAMPLES,
            fixed_training_alpha_and_independently_tuned_alpha=True,same_actual_channels_and_noise=True),
        save_every=SAVE_EVERY,selection='predeclared final12096; no best/test selection',test_used=False,
        source_implementation_commit=SOURCE_COMMIT,diagnostic_commit=DIAGNOSTIC_COMMIT,
        internal_source_commit=INTERNAL_SOURCE_COMMIT,ancestor_source_commit=ANCESTOR_SOURCE_COMMIT,
        scope='single-seed matched-RZF task-loss screening; not a final journal result'))


def fingerprint(arm,calibration=None):return json_sha256(recipe(arm,calibration))
sha256_file=previous.sha256_file


def inside_root(path,root=OUTPUT_ROOT):
    target,allowed=Path(path).resolve(),Path(root).resolve()
    if target==allowed or not target.is_relative_to(allowed):raise ValueError('A child of the independent matched-training root is required')
    return target


def prepare_output(path,*,resume=False,root=OUTPUT_ROOT):
    target=inside_root(path,root)
    if resume:
        if not target.is_dir() or not (target/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Existing matched-training checkpoint required')
    else:target.mkdir(parents=True,exist_ok=False)
    return target


def validate_resume_metadata(payload,arm,calibration,*,code_sha256=None):
    _arm(arm);identity=calibration_identity(calibration)
    required={'format','arm','seed','step','cumulative_updates','recipe','config_fingerprint','source',
        'codec_model','optimizer','scheduler','rng','data_protocol','ce_identity','code_sha256','calibration'}
    if not isinstance(payload,dict) or not required<=set(payload):raise ValueError('Incomplete matched-training checkpoint')
    if payload['format']!=CHECKPOINT_FORMAT or payload['arm']!=arm or type(payload['seed']) is not int or payload['seed']!=SEED:
        raise ValueError('Matched-training format/arm/seed mismatch')
    step=payload['step']
    if type(step) is not int or not 0<=step<=UPDATES or step%SAVE_EVERY:raise ValueError('Saved108-step boundary within12096 required')
    if type(payload['cumulative_updates']) is not int or payload['cumulative_updates']!=SOURCE_UPDATES+step:
        raise ValueError('Cumulative count must equal22096+local step')
    for name,value in (('recipe',recipe(arm,calibration)),('source',source_identity(arm)),
                       ('config_fingerprint',fingerprint(arm,calibration)),('calibration',identity)):
        if normalize_json(payload[name])!=normalize_json(value):raise ValueError('Resume '+name+' changed')
    for name in ('codec_model','optimizer','scheduler','data_protocol','ce_identity','code_sha256'):
        if not isinstance(payload[name],dict) or not payload[name]:raise ValueError('Missing complete resume mapping: '+name)
    if not isinstance(payload['rng'],dict) or not {'python','numpy','torch_cpu','torch_cuda'}<=set(payload['rng']):
        raise ValueError('Complete default RNG state required')
    if payload['scheduler'].get('last_epoch')!=step:raise ValueError('Scheduler local step changed')
    if code_sha256 is not None and payload['code_sha256']!=code_sha256:raise ValueError('Matched-training source hashes changed')
    return step
