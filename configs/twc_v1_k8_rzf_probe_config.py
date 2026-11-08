"""Independent, standard-library-only plan for a frozen-checkpoint K8 RZF probe."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import twc_v1_k8_task_loss_screen_config as previous

REPO_ROOT=Path(__file__).resolve().parents[1]
NAMESPACE='twc_v1_k8_rzf_probe_v1'
OUTPUT_ROOT=REPO_ROOT/'runs'/NAMESPACE
SELECTION_PATH=OUTPUT_ROOT/'alpha_selection.json'
SELECTION_FORMAT='twc_v1_k8_rzf_alpha_selection_v1'
ARMS=('fdma_nmse','fdma_joint','g4_nmse','g4_joint')
METHODS=ARMS
SEED=20261107
SOURCE_COMMIT='596d2a4d640c9001b85f78dec939bf032ff4a024'
INTERNAL_SOURCE_COMMIT='b6dc7ed5453ad0f8bc581b157505e4656932c1b7'
ANCESTOR_SOURCE_COMMIT='8096ed965027a92f6e0738c7653b4c3e75f71b8c'
SOURCE_LOCAL_UPDATES=6048
SOURCE_UPDATES=22096
BATCH_SIZE=4
ALPHAS=(.0001,.0003,.001,.003,.01,.03,.1,.3,1.0)
FIXED_ALPHA=.01
GAMMAS=previous.GAMMAS
ENERGY_RATIOS=previous.ENERGY_RATIOS
PILOT_ENERGY=16.0
DOWNLINK_SNR_DB=25.0
CALIBRATION_ROLE='validation'
CALIBRATION_START_INDEX=4096
CALIBRATION_BATCHES=16
CALIBRATION_SAMPLES=64
VALIDATION_ROLE='validation'
VALIDATION_START_INDEX=5120
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
    'fdma_joint':'fd1927f6748b5eee35e2efd42e37146b5ab7cc791130bad93b2b8c96966e2926',
    'g4_nmse':'6fa136d9ce5afbc545a85d1646018dbd609df2a9b2a4e2524c42aa7c4fb67d57',
    'g4_joint':'c2fbfd3e1508ea7983b618ec0a863b455dd540e051d6f2e6469779d6951adfe1',
}


def _arm(arm):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError('One prescribed frozen K8 task-loss checkpoint is required')
    return arm


def is_fdma(arm):
    return _arm(arm) in ('fdma_nmse','fdma_joint')


def normalize_json(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')))


def json_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def source_identity(arm):
    _arm(arm)
    directory=REPO_ROOT/'runs/twc_v1_k8_task_loss_screen_v1/training'/arm
    return dict(arm=arm,source_arm=arm,seed=SEED,step=SOURCE_LOCAL_UPDATES,cumulative_updates=SOURCE_UPDATES,
        implementation_commit=SOURCE_COMMIT,internal_source_commit=INTERNAL_SOURCE_COMMIT,
        ancestor_source_commit=ANCESTOR_SOURCE_COMMIT,path=str(directory/'checkpoint_latest.pt'),
        sha256=SOURCE_SHA256[arm],manifest_path=str(directory/'manifest.json'),
        initialization_path=str(directory/'initialization.json'),progress_path=str(directory/'progress.jsonl'))


def selection_identity(record):
    """Lock one global alpha per checkpoint and the complete immutable record."""
    if not isinstance(record,dict):raise ValueError('Completed immutable alpha selection required')
    selected=record.get('selected_alphas')
    if (record.get('format')!=SELECTION_FORMAT or record.get('status')!='passed'
            or normalize_json(record.get('grid'))!=list(ALPHAS)
            or not isinstance(selected,dict) or set(selected)!=set(ARMS)):
        raise ValueError('Alpha selection format/status/grid/four-checkpoint identity changed')
    for arm,value in selected.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value not in ALPHAS:
            raise ValueError('Exactly one candidate-grid global alpha per checkpoint required: '+arm)
    digest=json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Immutable alpha selection record was changed')
    return dict(format=SELECTION_FORMAT,record_id=digest,
        selected_alphas={arm:float(selected[arm]) for arm in ARMS})


def noise_seed(stage,batch_index,gamma_index,kind):
    if stage not in ('calibration','evaluation'):
        raise ValueError('Only the independent calibration/evaluation noise stages are allowed')
    batches=CALIBRATION_BATCHES if stage=='calibration' else VALIDATION_BATCHES
    if type(batch_index) is not int or not 0<=batch_index<batches:
        raise ValueError('Relative noise batch index is outside the declared stage')
    if (type(gamma_index) is not int or not 0<=gamma_index<len(GAMMAS)
            or kind not in ('pilot','data')):
        raise ValueError('Gamma/noise identity changed')
    offset=49_000_000 if stage=='calibration' else 59_000_000
    return SEED*10_000_000+offset+batch_index*1000+gamma_index*10+int(kind=='data')


def calibration_noise_seed(batch_index,gamma_index,kind):
    return noise_seed('calibration',batch_index,gamma_index,kind)


def evaluation_noise_seed(batch_index,gamma_index,kind):
    return noise_seed('evaluation',batch_index,gamma_index,kind)


def recipe(selection=None):
    locked=selection_identity(selection) if selection is not None else dict(status='pending',usable_for_final_validation=False)
    return normalize_json(dict(version=NAMESPACE,seed=SEED,source_checkpoints={arm:source_identity(arm) for arm in ARMS},
        architecture=ORDERED_ARCHITECTURE,optimizer_updates=0,models_always_eval=True,test_used=False,
        precoder='original K8 normalized RZF',alpha_grid=ALPHAS,fixed_alpha=FIXED_ALPHA,
        selection=locked,alpha_scope='one global alpha per checkpoint, never per gamma/sample',
        objective='mean downlink sum rate, equally weighted conditions and complete MU samples',
        condition_weights='FDMA six gamma points; G4 six gamma points times two equally weighted ratios',
        tie_break='exactly equal objective values choose the smaller alpha; no isclose tie rule',
        batch_size=BATCH_SIZE,physical=dict(K=8,Nt=32,Nsc=256,total_feedback_RE=256,
            total_energy_per_ue=64.,pilot_energy_per_ue=PILOT_ENERGY,
            data_energy_per_ue=48.,noise_variance='0.25*10**(-gamma/10)',downlink_snr_db=DOWNLINK_SNR_DB),
        conditions=dict(gamma=GAMMAS,hybrid_group_count=4,key_shared_ratios=ENERGY_RATIOS,
            fdma_evaluated_once_per_gamma=True),
        channel_data=dict(base_seed=BASE_DATA_SEED,seed_namespace=DATA_SEED_NAMESPACE,
            cdl_model=CDL_MODEL,speed=SPEED_MPS),
        calibration=dict(role=CALIBRATION_ROLE,first_batch=CALIBRATION_START_INDEX,batches=CALIBRATION_BATCHES,
            samples=CALIBRATION_SAMPLES,noise_offset=49_000_000,record=str(SELECTION_PATH)),
        validation=dict(role=VALIDATION_ROLE,first_batch=VALIDATION_START_INDEX,batches=VALIDATION_BATCHES,
            samples=VALIDATION_SAMPLES,noise_offset=59_000_000,reselection=False),
        ce_checkpoint=CE_CHECKPOINT,ce_calibration=CE_CALIBRATION,ce_sha256=CE_SHA256,
        same_actual_channels_and_noise=True,H_hat_reused_for_alpha_grid=True,
        source_implementation_commit=SOURCE_COMMIT,internal_source_commit=INTERNAL_SOURCE_COMMIT,
        ancestor_source_commit=ANCESTOR_SOURCE_COMMIT,
        scope='single-seed frozen-checkpoint RZF validation diagnostic; no training or formal test claim'))


def fingerprint(selection=None):
    return json_sha256(recipe(selection))


sha256_file=previous.sha256_file


def inside_root(path,root=OUTPUT_ROOT):
    target,allowed=Path(path).resolve(),Path(root).resolve()
    if target==allowed or not target.is_relative_to(allowed):
        raise ValueError('A child of the independent RZF probe root is required')
    return target


def prepare_output(path,*,resume=False,root=OUTPUT_ROOT):
    target=inside_root(path,root)
    if resume:
        if not target.is_dir() or not (target/'manifest.json').is_file():
            raise FileNotFoundError('Resume requires an owned RZF probe directory and manifest')
    else:target.mkdir(parents=True,exist_ok=False)
    return target
