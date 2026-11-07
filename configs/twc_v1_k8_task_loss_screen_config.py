"""Standard-library-only identities for the isolated K8 task-loss screen."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from . import twc_v1_k8_ordered_encoder_screen_config as previous

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_task_loss_screen_v1'
CHECKPOINT_FORMAT = NAMESPACE
OUTPUT_ROOT = REPO_ROOT/'runs'/NAMESPACE
CALIBRATION_PATH = OUTPUT_ROOT/'calibration.json'
CALIBRATION_FORMAT = 'twc_v1_k8_task_loss_calibration_v1'
ARMS = ('fdma_nmse', 'fdma_joint', 'g4_nmse', 'g4_joint')
SOURCE_ARMS = {'fdma_nmse':'ordered_fdma', 'fdma_joint':'ordered_fdma',
               'g4_nmse':'ordered_g4', 'g4_joint':'ordered_g4'}
METHODS = ARMS+('source_fdma','source_g4')
SEED = 20261107
SOURCE_COMMIT = 'b6dc7ed5453ad0f8bc581b157505e4656932c1b7'
INTERNAL_SOURCE_COMMIT = '8096ed965027a92f6e0738c7653b4c3e75f71b8c'
SOURCE_LOCAL_UPDATES = 3024
SOURCE_UPDATES = 16048
UPDATES = 6048
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
SAVE_EVERY = 108
WARMUP_UPDATES = 756
PROGRESS_VALIDATION_EVERY = 756
PROGRESS_VALIDATION_BATCHES = 4
GAMMAS = previous.GAMMAS
PILOT_ENERGIES = previous.PILOT_ENERGIES
ENERGY_RATIOS = previous.ENERGY_RATIOS
GROUP_COUNTS = (4,)
CDL_MODEL = previous.CDL_MODEL
SPEED_MPS = previous.SPEED_MPS
BASE_DATA_SEED = previous.BASE_DATA_SEED
DATA_SEED_NAMESPACE = previous.DATA_SEED_NAMESPACE
DOWNLINK_SNR_DB = 25.0
VALIDATION_SAMPLES = 256
VALIDATION_BATCHES = 64
VALIDATION_START_INDEX = 3072
VALIDATION_PILOT_ENERGY = 16.0
CALIBRATION_ROLE = 'codec_train'
CALIBRATION_START_INDEX = 500000
CALIBRATION_BATCHES = 12
CE_CHECKPOINT = previous.CE_CHECKPOINT
CE_CALIBRATION = previous.CE_CALIBRATION
CE_SHA256 = dict(previous.CE_SHA256)
ORDERED_ARCHITECTURE = previous.ORDERED_ARCHITECTURE
SOURCE_SHA256 = {
    'ordered_fdma':'25176c8d770775891fa928a60607054b64d91a2049856da626031a7451fd5203',
    'ordered_g4':'1a7ac64e45964043e23af96a69bbdf4bd144352e8cd6708f6dcca15eda4581a4',
}


def _arm(arm):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError('A prescribed K8 task-loss training arm is required')
    return arm


def is_fdma(arm):
    return _arm(arm) in ('fdma_nmse','fdma_joint')


def is_joint(arm):
    return _arm(arm) in ('fdma_joint','g4_joint')


def normalize_json(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')))


def json_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def source_identity(arm):
    source_arm=SOURCE_ARMS[_arm(arm)]
    directory=REPO_ROOT/'runs/twc_v1_k8_ordered_encoder_screen_v2/training'/source_arm
    return dict(arm=source_arm,seed=SEED,step=SOURCE_LOCAL_UPDATES,cumulative_updates=SOURCE_UPDATES,
        implementation_commit=SOURCE_COMMIT,internal_source_commit=INTERNAL_SOURCE_COMMIT,
        path=str(directory/'checkpoint_latest.pt'),sha256=SOURCE_SHA256[source_arm],
        manifest_path=str(directory/'manifest.json'),initialization_path=str(directory/'initialization.json'),
        progress_path=str(directory/'progress.jsonl'))


def architecture(arm):
    return previous.architecture(SOURCE_ARMS[_arm(arm)])


def calibration_identity(record):
    """Bind the exact immutable small record, not a mutable beta-only setting."""
    if not isinstance(record,dict):raise ValueError('A completed immutable calibration record is required')
    beta=record.get('beta_target')
    if (record.get('format')!=CALIBRATION_FORMAT or record.get('status')!='passed'
            or isinstance(beta,bool) or not isinstance(beta,(int,float)) or not math.isfinite(beta) or beta<=0):
        raise ValueError('Calibration format/status/positive finite beta mismatch')
    content={key:value for key,value in record.items() if key!='record_id'}
    digest=json_sha256(content)
    if record.get('record_id')!=digest:raise ValueError('Calibration record identity changed')
    return dict(format=CALIBRATION_FORMAT,record_id=digest,beta_target=float(beta))


def schedule(arm,local_step):
    _arm(arm)
    if type(local_step) is not int or not 1<=local_step<=UPDATES:
        raise ValueError('Local update must lie in [1,6048]')
    n=SOURCE_UPDATES+local_step-1
    return dict(local_step=local_step,global_step=n+1,gamma_db=GAMMAS[n%6],
        pilot_energy_per_ue=PILOT_ENERGIES[(n//6)%3],group_count=1 if is_fdma(arm) else 4,
        energy_ratio=None if is_fdma(arm) else list(ENERGY_RATIOS[(n//2)%2]))


def schedule_counts(arm):
    counts=Counter()
    for step in range(1,UPDATES+1):
        value=schedule(arm,step)
        counts[(value['group_count'],value['gamma_db'],value['pilot_energy_per_ue'],tuple(value['energy_ratio'] or ()))] += 1
    return [dict(group_count=g,gamma_db=gamma,pilot_energy_per_ue=ep,
                 energy_ratio=list(ratio) if ratio else None,count=count)
        for (g,gamma,ep,ratio),count in sorted(counts.items())]


def training_noise_seed(local_step):
    return SEED*10_000_000+schedule(ARMS[0],local_step)['global_step']


def evaluation_noise_seed(batch_index,gamma_index,kind):
    if type(batch_index) is not int or not 0<=batch_index<VALIDATION_BATCHES:
        raise ValueError('Validation batch index must lie in [0,63]')
    if type(gamma_index) is not int or not 0<=gamma_index<len(GAMMAS) or kind not in ('pilot','data'):
        raise ValueError('Validation gamma/noise identity changed')
    return SEED*10_000_000+29_000_000+batch_index*1000+gamma_index*10+int(kind=='data')


def loss_weight(arm,local_step,calibration):
    """One fixed calibrated target; the first756 updates ramp only joint arms."""
    schedule(arm,local_step)
    target=calibration_identity(calibration)['beta_target']
    return target*min(local_step/WARMUP_UPDATES,1.) if is_joint(arm) else 0.


def recipe(arm,calibration=None):
    _arm(arm)
    identity=calibration_identity(calibration) if calibration is not None else dict(status='pending',usable_for_training=False)
    return normalize_json(dict(version=NAMESPACE,arm=arm,seed=SEED,source=source_identity(arm),
        architecture=architecture(arm),new_updates=UPDATES,source_updates=SOURCE_UPDATES,
        batch_size=BATCH_SIZE,learning_rate=LEARNING_RATE,optimizer='AdamW',weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,scheduler='constant LambdaLR',
        optimizer_initialization='fresh, same recipe for all four arms',
        loss='mean per-user DL NMSE minus beta times mean sum rate' if is_joint(arm) else 'mean per-user DL NMSE',
        loss_weight_nmse=1.,rate_loss_enabled=is_joint(arm),rate_beta_warmup_updates=WARMUP_UPDATES,
        calibration=identity,beta_target=identity.get('beta_target'),
        ce_frozen=True,ce_checkpoint=CE_CHECKPOINT,ce_calibration=CE_CALIBRATION,ce_sha256=CE_SHA256,
        physical=dict(K=8,Nt=32,Nsc=256,total_feedback_RE=256,total_energy_per_ue=64.,
            noise_variance='0.25*10**(-gamma/10)',downlink_snr_db=DOWNLINK_SNR_DB,rzf_alpha=.01),
        channel_data=dict(base_seed=BASE_DATA_SEED,seed_namespace=DATA_SEED_NAMESPACE,
            cdl_model=CDL_MODEL,speed=SPEED_MPS,training_role='codec_train',training_address='16048+local_step'),
        conditions=dict(gamma=GAMMAS,pilot_energy=PILOT_ENERGIES,key_shared_ratios=ENERGY_RATIOS,
            group_count=1 if is_fdma(arm) else 4,counts=schedule_counts(arm)),
        calibration_data=dict(role=CALIBRATION_ROLE,first_batch=CALIBRATION_START_INDEX,batches=CALIBRATION_BATCHES,
            used_for_training_updates=False,record=str(CALIBRATION_PATH)),
        save_every=SAVE_EVERY,progress_validation_every=PROGRESS_VALIDATION_EVERY,
        validation=dict(role='validation',first_batch=VALIDATION_START_INDEX,batch_size=BATCH_SIZE,
            batches=VALIDATION_BATCHES,samples=VALIDATION_SAMPLES,pilot_energy=VALIDATION_PILOT_ENERGY,
            gamma=GAMMAS,key_shared_ratios=ENERGY_RATIOS,same_actual_channels_and_noise=True),
        selection='predeclared final6048, no best/test selection',training_seed_replications=1,
        test_used=False,equal_parameter_or_latency_claim=False,scope='single-seed task-loss validation screen'))


def fingerprint(arm,calibration=None):
    return json_sha256(recipe(arm,calibration))


sha256_file=previous.sha256_file


def inside_root(path,root=OUTPUT_ROOT):
    target,allowed=Path(path).resolve(),Path(root).resolve()
    if target==allowed or not target.is_relative_to(allowed):
        raise ValueError('A child of the independent task-loss screen root is required')
    return target


def prepare_output(path,*,resume=False,root=OUTPUT_ROOT):
    target=inside_root(path,root)
    if resume:
        if not target.is_dir() or not (target/'checkpoint_latest.pt').is_file():
            raise FileNotFoundError('Resume requires latest in the existing task-loss arm directory')
    else:target.mkdir(parents=True,exist_ok=False)
    return target


def validate_resume_metadata(payload,arm,calibration,*,code_sha256=None):
    _arm(arm);identity=calibration_identity(calibration)
    required={'format','arm','seed','step','cumulative_updates','recipe','config_fingerprint','source',
        'codec_model','optimizer','scheduler','rng','data_protocol','ce_identity','code_sha256','calibration'}
    if not isinstance(payload,dict) or not required<=set(payload):raise ValueError('Incomplete task-loss checkpoint')
    if payload['format']!=CHECKPOINT_FORMAT or payload['arm']!=arm or type(payload['seed']) is not int or payload['seed']!=SEED:
        raise ValueError('Task-loss format/arm/seed mismatch')
    step=payload['step']
    if type(step) is not int or not 0<=step<=UPDATES or step%SAVE_EVERY:
        raise ValueError('A saved108-step boundary within6048 is required')
    if type(payload['cumulative_updates']) is not int or payload['cumulative_updates']!=SOURCE_UPDATES+step:
        raise ValueError('Cumulative update count must equal16048+local step')
    for name,value in (('recipe',recipe(arm,calibration)),('source',source_identity(arm)),
                       ('config_fingerprint',fingerprint(arm,calibration)),('calibration',identity)):
        if normalize_json(payload[name])!=normalize_json(value):raise ValueError('Resume '+name+' changed')
    for name in ('codec_model','optimizer','scheduler','data_protocol','ce_identity','code_sha256'):
        if not isinstance(payload[name],dict) or not payload[name]:raise ValueError('Missing complete resume mapping: '+name)
    if not isinstance(payload['rng'],dict) or not {'python','numpy','torch_cpu','torch_cuda'}<=set(payload['rng']):
        raise ValueError('Complete default RNG state is required')
    if payload['scheduler'].get('last_epoch')!=step:raise ValueError('Scheduler local step changed')
    if code_sha256 is not None and payload['code_sha256']!=code_sha256:raise ValueError('Task-loss code hashes changed')
    return step
