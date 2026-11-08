"""Standard-library-only identities for the independent BS decoder screen."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
from . import twc_v1_k8_rzf_matched_training_config as previous

REPO_ROOT=Path(__file__).resolve().parents[1]
NAMESPACE='twc_v1_k8_decoder_screen_v1'
CHECKPOINT_FORMAT=NAMESPACE
OUTPUT_ROOT=REPO_ROOT/'runs'/NAMESPACE
SOURCE_ACCEPTANCE_PATH=OUTPUT_ROOT/'source_acceptance.json'
SOURCE_ACCEPTANCE_FORMAT='twc_v1_k8_decoder_source_acceptance_v1'
SELECTION_PATH=OUTPUT_ROOT/'alpha_selection.json'
SELECTION_FORMAT='twc_v1_k8_decoder_alpha_selection_v1'
ARMS=('fdma_base2','fdma_deep4','fdma_dc2','g4_base2','g4_deep4','g4_dc2')
METHODS=ARMS+('source_fdma','source_g4')
SOURCE_METHODS=('source_fdma','source_g4')
SOURCE_COMMIT='ad3901743e1b18dc6829f91b1d43beca4f2c09ce'
SEED=20261107
SOURCE_LOCAL_UPDATES=12096
SOURCE_UPDATES=34192
UPDATES=6048
BATCH_SIZE=4
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-5
GRAD_CLIP_NORM=1.
SAVE_EVERY=108
MONITOR_ALPHA=.03
FIXED_ALPHA=.03
PROGRESS_VALIDATION_STEPS=(1512,3024,4536,6048)
MONITOR_STEPS=PROGRESS_VALIDATION_STEPS
MONITOR_START_INDEX=10240
MONITOR_BATCHES=16
MONITOR_SAMPLES=64
ALPHA_CALIBRATION_START_INDEX=11264
ALPHA_CALIBRATION_BATCHES=16
ALPHA_CALIBRATION_SAMPLES=64
VALIDATION_START_INDEX=12288
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
    if type(method) is not str or method not in METHODS:raise ValueError('A prescribed decoder-screen method is required')
    return method

def is_fdma(method):return _method(method).startswith('fdma_') or method=='source_fdma'
def variant(method):
    _method(method)
    return 'base2' if method.startswith('source_') else method.rsplit('_',1)[1]
def source_method(method):return 'source_fdma' if is_fdma(method) else 'source_g4'
def normalize_json(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')))
def json_sha256(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
sha256_file=previous.sha256_file

def planned_source_identity(method):
    arm='fdma_nmse' if is_fdma(method) else 'g4_nmse'
    directory=REPO_ROOT/'runs/twc_v1_k8_rzf_matched_training_v1/training'/arm
    return dict(arm=arm,path=str(directory/'checkpoint_latest.pt'),sha256=None,
        manifest_path=str(directory/'manifest.json'),initialization_path=str(directory/'initialization.json'),
        progress_path=str(directory/'progress.jsonl'),terminal_path=str(directory/'terminal_checkpoint.json'),
        seed=SEED,local_step=SOURCE_LOCAL_UPDATES,cumulative_updates=SOURCE_UPDATES,source_implementation_commit=SOURCE_COMMIT)

def acceptance_identity(record):
    if (not isinstance(record,dict) or record.get('format')!=SOURCE_ACCEPTANCE_FORMAT or record.get('status')!='passed'
            or not isinstance(record.get('sources'),dict) or set(record['sources'])!=set(SOURCE_METHODS)):
        raise ValueError('Completed two-source decoder acceptance required')
    hashes={}
    for method in SOURCE_METHODS:
        entry=record['sources'][method]
        if not isinstance(entry,dict) or not isinstance(entry.get('identity'),dict):raise ValueError('Complete accepted source identity required')
        identity=entry['identity'];expected=planned_source_identity(method)
        for key,value in expected.items():
            if key!='sha256' and identity.get(key)!=value:raise ValueError('Accepted fixed source '+key+' changed: '+method)
        digest=identity.get('sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Actual server-verified source SHA256 required; no guessed pins')
        if not entry.get('verification') or not entry.get('frozen_files'):raise ValueError('Accepted cross-check evidence missing')
        hashes[method]=digest
    digest=json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Immutable source acceptance changed')
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
        raise ValueError('Independent eight-checkpoint alpha lock required')
    for method,alpha in selected.items():
        if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or alpha not in ALPHAS:raise ValueError('One legal global alpha required: '+method)
    digest=json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Immutable decoder alpha lock changed')
    return dict(format=SELECTION_FORMAT,record_id=digest,selected_alphas={method:float(selected[method]) for method in METHODS})

def schedule(arm,local_step):
    if arm not in ARMS or type(local_step) is not int or not 1<=local_step<=UPDATES:raise ValueError('A prescribed arm and local step1..6048 required')
    n=SOURCE_UPDATES+local_step-1
    return dict(local_step=local_step,global_step=n+1,gamma_db=GAMMAS[n%6],pilot_energy_per_ue=PILOT_ENERGIES[(n//6)%3],
        group_count=1 if is_fdma(arm) else 4,energy_ratio=None if is_fdma(arm) else list(ENERGY_RATIOS[(n//2)%2]))

def schedule_counts(arm):
    counts=Counter()
    for step in range(1,UPDATES+1):
        value=schedule(arm,step);counts[(value['group_count'],value['gamma_db'],value['pilot_energy_per_ue'],tuple(value['energy_ratio'] or ()))] += 1
    return [dict(group_count=g,gamma_db=gamma,pilot_energy_per_ue=ep,energy_ratio=list(ratio) if ratio else None,count=count)
        for (g,gamma,ep,ratio),count in sorted(counts.items())]
def training_noise_seed(local_step):return SEED*10_000_000+schedule(ARMS[0],local_step)['global_step']
def noise_seed(stage,batch_index,gamma_index,kind):
    sizes={'preflight':1,'monitor':16,'alpha_calibration':16,'evaluation':64}
    offsets={'preflight':109_000_000,'monitor':119_000_000,'alpha_calibration':129_000_000,'evaluation':139_000_000}
    if stage not in sizes or type(batch_index) is not int or not 0<=batch_index<sizes[stage]:raise ValueError('Diagnostic stage/batch identity mismatch')
    if type(gamma_index) is not int or not 0<=gamma_index<6 or kind not in ('pilot','data'):raise ValueError('Diagnostic gamma/noise identity mismatch')
    return SEED*10_000_000+offsets[stage]+batch_index*1000+gamma_index*10+int(kind=='data')

def recipe(arm,acceptance=None):
    if arm not in ARMS:raise ValueError('A prescribed decoder training arm is required')
    identity=acceptance_identity(acceptance) if acceptance is not None else dict(status='pending',usable_for_training=False)
    return normalize_json(dict(version=NAMESPACE,arm=arm,seed=SEED,source=source_identity(arm,acceptance),source_acceptance=identity,
        source_implementation_commit=SOURCE_COMMIT,source_codec_updates=SOURCE_UPDATES,new_BS_updates=UPDATES,
        UE_new_updates=0,UE_frozen=True,UE_condition_embedding_frozen=True,CE_frozen=True,
        decoder_variant=variant(arm),all_BS_trainable=True,shared_spatial_decoder=True,
        loss='mean physical downlink CSI NMSE only',sum_rate_loss_weight=0.,codeword_loss_weight=0.,residual_loss_weight=0.,
        batch_size=BATCH_SIZE,optimizer='fresh BS-only AdamW',learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,scheduler='constant LambdaLR',save_every=SAVE_EVERY,
        monitor=dict(steps=PROGRESS_VALIDATION_STEPS,role='validation',first_batch=MONITOR_START_INDEX,batches=MONITOR_BATCHES,
            samples=MONITOR_SAMPLES,gamma=GAMMAS,pilot_energy=PILOT_ENERGY,downlink_snr_db=DOWNLINK_SNR_DB,
            alpha=MONITOR_ALPHA,hybrid_ratios=ENERGY_RATIOS,selection=False,fixed_channels_and_noise=True),
        alpha_calibration=dict(role='validation',first_batch=ALPHA_CALIBRATION_START_INDEX,batches=ALPHA_CALIBRATION_BATCHES,
            samples=ALPHA_CALIBRATION_SAMPLES,grid=ALPHAS,one_global_alpha_per_checkpoint=True,both_G4_ratios_equal_weight=True),
        validation=dict(role='validation',first_batch=VALIDATION_START_INDEX,batches=VALIDATION_BATCHES,samples=VALIDATION_SAMPLES,
            fixed_alpha=FIXED_ALPHA,also_locked_tuned=True),
        physical=dict(K=8,Nt=32,Nsc=256,total_feedback_RE=256,total_energy_per_ue=64.,noise_variance='0.25*10**(-gamma/10)'),
        channel_data=dict(base_seed=BASE_DATA_SEED,seed_namespace=DATA_SEED_NAMESPACE,cdl_model=CDL_MODEL,speed=SPEED_MPS,
            role='codec_train',training_address='34192+local_step'),conditions=schedule_counts(arm),
        ce_checkpoint=CE_CHECKPOINT,ce_calibration=CE_CALIBRATION,ce_sha256=CE_SHA256,
        checkpoint_selection='predeclared final6048; no best/test selection',test_used=False,training_seed_replications=1,
        equal_parameter_MAC_latency_claim=False,scope='composite BS decoder refinement single-seed validation screen'))
def fingerprint(arm,acceptance=None):return json_sha256(recipe(arm,acceptance))

def inside_root(path,root=OUTPUT_ROOT):
    target,allowed=Path(path).resolve(),Path(root).resolve()
    if target==allowed or not target.is_relative_to(allowed):raise ValueError('A child of the independent decoder screen root is required')
    return target
def prepare_output(path,*,resume=False,root=OUTPUT_ROOT):
    target=inside_root(path,root)
    if resume:
        if not target.is_dir() or not (target/'checkpoint_latest.pt').is_file():raise FileNotFoundError('Owned decoder latest checkpoint required')
    else:target.mkdir(parents=True,exist_ok=False)
    return target
def validate_resume_metadata(payload,arm,acceptance,*,code_sha256=None):
    if arm not in ARMS:raise ValueError('A prescribed decoder arm required')
    identity=acceptance_identity(acceptance)
    required={'format','arm','seed','step','new_bs_updates','source_codec_updates','data_global_step','ue_frozen','recipe','config_fingerprint','source','source_acceptance',
        'codec_model','optimizer','scheduler','rng','data_protocol','ce_identity','code_sha256'}
    if not isinstance(payload,dict) or not required<=set(payload):raise ValueError('Incomplete decoder checkpoint')
    if payload['format']!=CHECKPOINT_FORMAT or payload['arm']!=arm or type(payload['seed']) is not int or payload['seed']!=SEED:
        raise ValueError('Decoder format/arm/seed identity mismatch')
    step=payload['step']
    if (type(step) is not int or not 0<=step<=UPDATES or step%SAVE_EVERY
            or type(payload['new_bs_updates']) is not int or payload['new_bs_updates']!=step
            or type(payload['source_codec_updates']) is not int or payload['source_codec_updates']!=SOURCE_UPDATES
            or type(payload['data_global_step']) is not int or payload['data_global_step']!=SOURCE_UPDATES+step
            or payload['ue_frozen'] is not True):
        raise ValueError('Saved BS step/global-address count mismatch')
    for name,value in (('recipe',recipe(arm,acceptance)),('config_fingerprint',fingerprint(arm,acceptance)),
                       ('source',source_identity(arm,acceptance)),('source_acceptance',identity)):
        if normalize_json(payload[name])!=normalize_json(value):raise ValueError('Resume '+name+' changed')
    for name in ('codec_model','optimizer','scheduler','data_protocol','ce_identity','code_sha256'):
        if not isinstance(payload[name],dict) or not payload[name]:raise ValueError('Missing decoder resume mapping: '+name)
    if not isinstance(payload['rng'],dict) or not {'python','numpy','torch_cpu','torch_cuda'}<=set(payload['rng']):raise ValueError('Complete default RNG required')
    if payload['scheduler'].get('last_epoch')!=step:raise ValueError('Decoder scheduler step mismatch')
    if code_sha256 is not None and payload['code_sha256']!=code_sha256:raise ValueError('Decoder experiment code changed')
    return step
