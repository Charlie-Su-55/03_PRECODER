"""Fixed, standard-library-only plan for the K8 ordered-window continuation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from . import twc_v1_k8_schedule_screen_config as previous

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_ordered_encoder_screen_v2'
ORDERED_ARCHITECTURE = 'twc_v1_k8_ordered_window_encoder_v2'
CHECKPOINT_FORMAT = NAMESPACE
OUTPUT_ROOT = REPO_ROOT/'runs'/NAMESPACE
ARMS = ('mean_fdma_control', 'ordered_fdma', 'mean_g4_control', 'ordered_g4')
SEED = 20261107
SOURCE_COMMIT = '8096ed965027a92f6e0738c7653b4c3e75f71b8c'
SOURCE_LOCAL_UPDATES = 3024
SOURCE_UPDATES = 13024
UPDATES = 3024
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
SAVE_EVERY = 108
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
VALIDATION_START_INDEX = 2048
VALIDATION_PILOT_ENERGY = 16.0
CE_CHECKPOINT = previous.CE_CHECKPOINT
CE_CALIBRATION = previous.CE_CALIBRATION
CE_SHA256 = dict(previous.CE_SHA256)
SOURCE_SHA256 = {
    'fdma_control': 'd0c3537d4e22bd061b387251f22030961829fc231126d1126b0cefde9b8f7d39',
    'specialist_g4': 'ed92ca1c6688ee24f5515d3d281dc012bef27a04e0899ae19f55bf97335d20e8',
}
EXPECTED_PARAMETER_DELTA = 31*128*(8+320)


def _arm(arm):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError('One prescribed ordered-encoder screening arm is required')
    return arm


def is_fdma(arm):
    return _arm(arm) in ('mean_fdma_control', 'ordered_fdma')


def is_ordered(arm):
    return _arm(arm) in ('ordered_fdma', 'ordered_g4')


def normalize_json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')))


def source_identity(arm):
    source_arm = 'fdma_control' if is_fdma(arm) else 'specialist_g4'
    directory = REPO_ROOT/'runs/twc_v1_k8_schedule_screen_v1/training'/source_arm
    return dict(arm=source_arm, seed=SEED, step=SOURCE_LOCAL_UPDATES,
        cumulative_updates=SOURCE_UPDATES, source_commit=SOURCE_COMMIT,
        path=str(directory/'checkpoint_latest.pt'), sha256=SOURCE_SHA256[source_arm],
        manifest_path=str(directory/'manifest.json'), initialization_path=str(directory/'initialization.json'),
        progress_path=str(directory/'progress.jsonl'))


def architecture(arm):
    ordered = is_ordered(arm)
    return dict(architecture=ORDERED_ARCHITECTURE if ordered else 'twc_v1_k8_global_mean_encoder_original',
        encoder='K8OrderedWindowSwinEncoder' if ordered else 'K8SwinEncoder',
        token_grid=[64,8], normalized_token_width=128, window_size=4,
        pooling='original_global_mean_plus_ordered_contrast_residual' if ordered else 'global_token_mean',
        head_input_width=128,
        group_condition='original G embedding only in the unchanged global-mean head path',
        ordered_residual_input_width=3968 if ordered else 0,
        ordered_residual_features='concat(p1-p32,...,p31-p32), fixed row-major windows' if ordered else None,
        ordered_residual_bias=False if ordered else None,
        ordered_residual_initialization='two exactly zero weight matrices' if ordered else None,
        ordered_residual_weight_keys=['encoder.key_window_residual.weight',
            'encoder.shared_window_residual.weight'] if ordered else [],
        parameter_delta=EXPECTED_PARAMETER_DELTA if ordered else 0,
        position_embedding_added=False, decoder_changed=False,
        equivalence='unchanged original heads and G path plus zero contrast residual; all old state copied',
        legacy_v1_checkpoint_loading='rejected')


def schedule(arm, local_step):
    _arm(arm)
    if type(local_step) is not int or not 1<=local_step<=UPDATES:
        raise ValueError('Local update must lie in [1,3024]')
    n = SOURCE_UPDATES+local_step-1
    return dict(local_step=local_step, global_step=n+1, gamma_db=GAMMAS[n%6],
        pilot_energy_per_ue=PILOT_ENERGIES[(n//6)%3], group_count=1 if is_fdma(arm) else 4,
        energy_ratio=None if is_fdma(arm) else list(ENERGY_RATIOS[(n//2)%2]))


def schedule_counts(arm):
    counts=Counter()
    for step in range(1,UPDATES+1):
        r=schedule(arm,step)
        counts[(r['group_count'],r['gamma_db'],r['pilot_energy_per_ue'],tuple(r['energy_ratio'] or ()))] += 1
    return [dict(group_count=g,gamma_db=gamma,pilot_energy_per_ue=ep,
                 energy_ratio=list(ratio) if ratio else None,count=n)
            for (g,gamma,ep,ratio),n in sorted(counts.items())]


def training_noise_seed(local_step):
    return SEED*10_000_000+schedule(ARMS[0],local_step)['global_step']


def evaluation_noise_seed(batch_index,gamma_index,kind):
    if type(batch_index) is not int or not 0<=batch_index<VALIDATION_BATCHES:
        raise ValueError('Validation batch index must lie in [0,63]')
    if type(gamma_index) is not int or not 0<=gamma_index<len(GAMMAS) or kind not in ('pilot','data'):
        raise ValueError('Validation gamma/noise identity changed')
    # Independent of arm/layout/energy; disjoint from previous screening noise.
    return SEED*10_000_000+19_000_000+batch_index*1000+gamma_index*10+int(kind=='data')


def recipe(arm):
    _arm(arm)
    return normalize_json(dict(version=NAMESPACE,arm=arm,seed=SEED,source=source_identity(arm),
        architecture=architecture(arm),new_updates=UPDATES,source_updates=SOURCE_UPDATES,
        batch_size=BATCH_SIZE,learning_rate=LEARNING_RATE,optimizer='AdamW',
        weight_decay=WEIGHT_DECAY,grad_clip_norm=GRAD_CLIP_NORM,scheduler='constant LambdaLR',
        optimizer_initialization='fresh, same recipe for all four arms',
        loss='mean per-user downlink CSI NMSE',loss_weight_nmse=1.0,loss_weight_rate=0.0,
        ce_frozen=True,ce_checkpoint=CE_CHECKPOINT,ce_calibration=CE_CALIBRATION,ce_sha256=CE_SHA256,
        physical=dict(K=8,Nt=32,Nsc=256,total_feedback_RE=256,total_energy_per_ue=64.,
            noise_variance='0.25*10**(-gamma/10)',downlink_snr_db=DOWNLINK_SNR_DB),
        channel_data=dict(base_seed=BASE_DATA_SEED,seed_namespace=DATA_SEED_NAMESPACE,
            cdl_model=CDL_MODEL,speed=SPEED_MPS,training_role='codec_train',
            training_address='13024+local_step'),
        conditions=dict(gamma=GAMMAS,pilot_energy=PILOT_ENERGIES,key_shared_ratios=ENERGY_RATIOS,
            group_count=1 if is_fdma(arm) else 4,counts=schedule_counts(arm)),
        save_every=SAVE_EVERY,progress_validation_every=PROGRESS_VALIDATION_EVERY,
        validation=dict(role='validation',first_batch=VALIDATION_START_INDEX,batch_size=BATCH_SIZE,
            batches=VALIDATION_BATCHES,samples=VALIDATION_SAMPLES,pilot_energy=VALIDATION_PILOT_ENERGY,
            gamma=GAMMAS,key_shared_ratios=ENERGY_RATIOS,same_actual_channels_and_noise=True),
        selection='predeclared final3024, no best/test selection',training_seed_replications=1,
        test_used=False,equal_parameter_or_latency_claim=False,
        scope='ordered-window encoder controlled validation screen; Stage C not started'))


def fingerprint(arm):
    return hashlib.sha256(json.dumps(recipe(arm),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


sha256_file = previous.sha256_file


def inside_root(path,root=OUTPUT_ROOT):
    target,allowed=Path(path).resolve(),Path(root).resolve()
    if target==allowed or not target.is_relative_to(allowed):
        raise ValueError('A child of the independent ordered-screen root is required')
    return target


def prepare_output(path,*,resume=False,root=OUTPUT_ROOT):
    target=inside_root(path,root)
    if resume:
        if not target.is_dir() or not (target/'checkpoint_latest.pt').is_file():
            raise FileNotFoundError('Resume requires latest in the existing new arm directory')
    else:target.mkdir(parents=True,exist_ok=False)
    return target


def validate_resume_metadata(payload,arm,*,code_sha256=None):
    _arm(arm)
    required={'format','arm','seed','step','cumulative_updates','recipe','config_fingerprint','source',
        'codec_model','optimizer','scheduler','rng','data_protocol','ce_identity','code_sha256'}
    if not isinstance(payload,dict) or not required<=set(payload):raise ValueError('Incomplete ordered-screen checkpoint')
    if payload['format']!=CHECKPOINT_FORMAT or payload['arm']!=arm or type(payload['seed']) is not int or payload['seed']!=SEED:
        raise ValueError('Ordered-screen format/arm/seed mismatch')
    step=payload['step']
    if type(step) is not int or not 0<=step<=UPDATES or step%SAVE_EVERY:
        raise ValueError('A saved108-step boundary within3024 is required')
    if type(payload['cumulative_updates']) is not int or payload['cumulative_updates']!=SOURCE_UPDATES+step:
        raise ValueError('Cumulative update count must equal13024+local step')
    for name,value in (('recipe',recipe(arm)),('source',source_identity(arm)),('config_fingerprint',fingerprint(arm))):
        if normalize_json(payload[name])!=normalize_json(value):raise ValueError('Resume '+name+' changed')
    for name in ('codec_model','optimizer','scheduler','data_protocol','ce_identity','code_sha256'):
        if not isinstance(payload[name],dict) or not payload[name]:raise ValueError('Missing complete resume mapping: '+name)
    if not isinstance(payload['rng'],dict) or not {'python','numpy','torch_cpu','torch_cuda'}<=set(payload['rng']):
        raise ValueError('Complete default RNG state is required')
    if payload['scheduler'].get('last_epoch')!=step:raise ValueError('Scheduler local step changed')
    if code_sha256 is not None and payload['code_sha256']!=code_sha256:raise ValueError('New experiment code hashes changed')
    return step
