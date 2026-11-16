"""Independent pilot-only CE continuation; codecs and old outputs stay frozen."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from configs import twc_v1_k8_icc_aligned_config as base
from configs import twc_v1_k8_icc_rate_config as pinned

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = DATA_NAMESPACE = 'twc_v1_k8_ce_refine_v1'
OUTPUT_ROOT = ROOT/'runs'/NAMESPACE
SOURCE_ROOT = base.OUTPUT_ROOT/'training'/'ce'
SOURCE_COMMIT = pinned.SOURCE_COMMIT
SOURCE_CORE = pinned.SOURCE_CORE
SOURCE_CHECKPOINT_SHA256 = pinned.SOURCE_CHECKPOINT_SHA256['ce']
SOURCE_TERMINAL_SHA256 = pinned.SOURCE_TERMINAL_SHA256['ce']
SOURCE_STATE_SHA256 = '661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192'
SOURCE_UPDATES = 5000
TRAJECTORIES = ('nmse', 'geometry')
SEED = 20261129
UPDATES, MICRO_BATCH, ACCUMULATION = 60000, 4, 8
SAVE_EVERY, MONITOR_EVERY = 100, 5000
GAMMAS = (0, 5, 10, 15, 20, 25)
EVAL_GAMMAS = tuple(2.5*i for i in range(11))
MONITOR_BATCHES, CALIBRATION_BATCHES, FINAL_BATCHES = 16, 64, 256
FINAL_START = 4096
MIDPOINT_START, MIDPOINT_BATCHES = 8192, FINAL_BATCHES
MIDPOINT_GAMMAS = (2.5,7.5,12.5,17.5,22.5)
LR, MIN_LR, WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1e-5, 1e-4, 1.
DIRECTION_WEIGHT, GRAM_WEIGHT = 1., 1.
ROLE_RANGES = dict(ce_train=[1, UPDATES*ACCUMULATION], monitor=[0, MONITOR_BATCHES-1],
    calibration=[0, CALIBRATION_BATCHES-1], final=[FINAL_START,FINAL_START+FINAL_BATCHES-1],
    midpoint=[MIDPOINT_START,MIDPOINT_START+MIDPOINT_BATCHES-1])
CORE_FILES = tuple(dict.fromkeys((*SOURCE_CORE['sha256'],
    'configs/twc_v1_k8_icc_rate_config.py', 'run_twc_v1_k8_icc_aligned.py',
    'publish_twc_v1_k8_ul_propagation.py', 'configs/twc_v1_k8_ce_refine_config.py',
    'ce_refine_runtime.py', 'twc_v1_k8_ce_refine.py', 'train_twc_v1_k8_ce_refine.py',
    'evaluate_twc_v1_k8_ce_refine.py', 'run_twc_v1_k8_ce_refine.py',
    'publish_twc_v1_k8_ce_refine.py', 'twc_v1_k8_ce_refine_results.py')))
REPOSITORY = 'Charlie-Su-55/03_PRECODER'
REMOTE_URL = 'https://github.com/'+REPOSITORY+'.git'
RESULT_BRANCH_PREFIX = 'results/'+NAMESPACE+'/'


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def key(trajectory):
    if trajectory not in TRAJECTORIES: raise ValueError('Declared CE trajectory required')
    return trajectory


def gamma_schedule(index):
    if type(index) is not int or not 1 <= index <= UPDATES*ACCUMULATION:
        raise ValueError('Declared CE microbatch index required')
    return tuple(GAMMAS[((index-1)*MICRO_BATCH+i) % len(GAMMAS)] for i in range(MICRO_BATCH))


def gamma_counts(step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError('Complete CE optimizer boundary required')
    q, rest = divmod(step*MICRO_BATCH*ACCUMULATION, len(GAMMAS))
    return {str(g):q+(i<rest) for i,g in enumerate(GAMMAS)}


def micro_index(step, micro):
    if (type(step) is not int or not 1 <= step <= UPDATES or type(micro) is not int
            or not 0 <= micro < ACCUMULATION):
        raise ValueError('CE step and microbatch index required')
    return (step-1)*ACCUMULATION+micro+1


def counters(step):
    counts = gamma_counts(step)
    return dict(optimizer_updates=step, CE_new_updates=step, CE_total_updates=SOURCE_UPDATES+step,
        UE_updates=0, BS_updates=0, FDMA_updates=0, EMA_commits=0,
        microbatches=step*ACCUMULATION, MU_frames=step*ACCUMULATION*MICRO_BATCH,
        UE_links=step*ACCUMULATION*MICRO_BATCH*8, next_micro_index=step*ACCUMULATION+1,
        accumulation_cursor=0, gamma_counts=counts)


def learning_rate(step):
    if type(step) is not int or not 0 <= step <= UPDATES: raise ValueError('LR boundary required')
    return MIN_LR+(LR-MIN_LR)*(1+math.cos(math.pi*step/UPDATES))/2


def noise_seed(role, index, kind='noise'):
    if (role not in ('ce_train','monitor','calibration','final','midpoint') or type(index) is not int
            or index < 0 or kind not in ('noise','polarity')):
        raise ValueError('Explicit CE random role/index/kind required')
    if role == 'ce_train': gamma_schedule(index)
    raw=f'{NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big') % (2**63-1)


def recipe(trajectory=None):
    if trajectory is not None: key(trajectory)
    return plain(dict(namespace=NAMESPACE, trajectories=TRAJECTORIES, trajectory=trajectory,
        source=dict(commit=SOURCE_COMMIT, core=SOURCE_CORE, updates=SOURCE_UPDATES,
            checkpoint_sha256=SOURCE_CHECKPOINT_SHA256, terminal_sha256=SOURCE_TERMINAL_SHA256,
            state_sha256=SOURCE_STATE_SHA256, optimizer_continued=False),
        updates_per_trajectory=UPDATES, total_new_CE_updates=UPDATES*len(TRAJECTORIES),
        trainable=['pilot-only CE U-Net'], UE_updates=0, BS_updates=0, FDMA_updates=0, EMA_commits=0,
        architecture='unchanged ICC-aligned pilot residual U-Net K8',
        optimizer=dict(name='AdamW', lr=LR, min_lr=MIN_LR, scheduler='CosineAnnealingLR',
            T_max=UPDATES, weight_decay=WEIGHT_DECAY, clip=GRAD_CLIP, betas=[.9,.999],eps=1e-8),
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION, seed=SEED,
        objective=dict(nmse=1., direction=DIRECTION_WEIGHT if trajectory=='geometry' else 0.,
            gram=GRAM_WEIGHT if trajectory=='geometry' else 0., coefficients_fixed=True),
        gamma=dict(nodes=GAMMAS, training='balanced per-frame round-robin',
            conditioned=True, feature='(gamma-12.5)/12.5', evaluation=EVAL_GAMMAS,
            off_node_midpoints=[2.5,7.5,12.5,17.5,22.5],
            midpoint_scope='interpolation checks outside new continuation nodes; source trained Uniform[0,25]'),
        physical=dict(K=8,Nt=32,Nsc=256,pilot_RE=64,pilots_per_UE=8,Ep=16.,
            noise='N0=.25*10**(-gamma/10)',rank=8,ridge=.001,pilot_only=True),
        data=dict(namespace=DATA_NAMESPACE, role_ranges=ROLE_RANGES, test_used=False,
            paired_actual_channels_noise_pilots=True),
        monitor=dict(every=MONITOR_EVERY,batches=MONITOR_BATCHES, selection=False),
        calibration=dict(batches=CALIBRATION_BATCHES, nodes=GAMMAS, locked_before_final=True),
        final=dict(batches=FINAL_BATCHES,gammas=EVAL_GAMMAS, samples_per_gamma=FINAL_BATCHES*MICRO_BATCH,
            fixed_terminal=UPDATES, best_checkpoint=False),
        reporting=dict(scope='CE and analytical recovery sensitivity; no codec rate claim',
            source_training_seeds=1, performance_claims_verified=False, test_used=False)))


def fingerprint(trajectory=None):
    return digest(recipe(trajectory))
