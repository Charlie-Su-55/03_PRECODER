"""Hybrid-only direction-assisted rate extension; metadata imports are light."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs import twc_v1_k8_ue_bs_rate_config as previous
from diagnose_twc_v1_k8_rate_headroom import SOURCE_PINS as _SOURCE_PINS, SOURCE_CORE_SHA256

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = DATA_NAMESPACE = TRAIN_DATA_NAMESPACE = 'twc_v1_k8_hybrid_direction_v1'
OUTPUT_ROOT = ROOT/'runs'/NAMESPACE
SEED = TRAIN_SEED = OPTIMIZER_SEED = 20261127
TRAIN_ARM = 'hybrid_g4'
ARMS = TRAIN_ARMS = (TRAIN_ARM,)
EVAL_VARIANTS = ('fdma_frozen', 'hybrid_source', 'hybrid_direction')
GAMMAS = (0, 5, 10, 15, 20, 25)
UPDATES, MICRO_BATCH, ACCUMULATION, SAVE_EVERY = 10000, 4, 8, 100
FINAL_BATCHES = 256
ROLE_RANGES = dict(codec_train=[1, UPDATES*ACCUMULATION], final=[4096, 4096+FINAL_BATCHES-1])
LR = LEARNING_RATE = 1e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1.
ALPHA, DL_SNR, DIRECTION_WEIGHT = .03, 25., 100.
BASE_EMA_UPDATES = 30000
SOURCE_NAMESPACE = previous.NAMESPACE
SOURCE_ROOT = previous.OUTPUT_ROOT
SOURCE_COMMIT = 'cc3ae7148a87f1333d5e789e59f45398e9c6d3e5'
SOURCE_RESULT_COMMIT = 'ab6bb85254018e463c62e0a15f8a6ff4e7b4330f'
REPOSITORY = 'Charlie-Su-55/03_PRECODER'
REMOTE_URL = 'https://github.com/'+REPOSITORY+'.git'
RESULT_BRANCH_PREFIX = RESULTS_BRANCH_PREFIX = 'results/'+NAMESPACE+'/'
LEGACY_CORE_FILES = previous.CORE_FILES
CORE_FILES = (*LEGACY_CORE_FILES, 'diagnose_twc_v1_k8_rate_headroom.py',
    'configs/twc_v1_k8_hybrid_direction_config.py', 'hybrid_direction_runtime.py',
    'twc_v1_k8_hybrid_direction.py', 'train_twc_v1_k8_hybrid_direction.py',
    'evaluate_twc_v1_k8_hybrid_direction.py', 'twc_v1_k8_hybrid_direction_results.py',
    'run_twc_v1_k8_hybrid_direction.py', 'publish_twc_v1_k8_hybrid_direction.py')


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


SOURCE_PINS = plain(_SOURCE_PINS)


def key(arm=TRAIN_ARM):
    if arm != TRAIN_ARM:
        raise ValueError('Only Hybrid G4 may train in this experiment')
    return arm


def gamma_schedule(index):
    if type(index) is not int or not 1 <= index <= UPDATES*ACCUMULATION:
        raise ValueError('Declared Hybrid training micro index required')
    return tuple(GAMMAS[(MICRO_BATCH*(index-1)+j) % len(GAMMAS)] for j in range(MICRO_BATCH))


def gamma_counts(step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError('Complete Hybrid optimizer boundary required')
    whole, rest = divmod(step*ACCUMULATION*MICRO_BATCH, len(GAMMAS))
    return {str(g): whole+(i < rest) for i, g in enumerate(GAMMAS)}


def micro_index(update, micro):
    if (type(update) is not int or not 1 <= update <= UPDATES or type(micro) is not int
            or not 0 <= micro < ACCUMULATION):
        raise ValueError('Declared Hybrid update and accumulation index required')
    return (update-1)*ACCUMULATION+micro+1


def counters(step, base_ema_updates=BASE_EMA_UPDATES):
    gamma = gamma_counts(step)
    if type(base_ema_updates) is not int or base_ema_updates < 0:
        raise ValueError('Explicit inherited EMA counter required')
    return dict(optimizer_updates=step, UE_updates=step, BS_updates=step, CE_updates=0,
        FDMA_updates=0, microbatches=step*ACCUMULATION, MU_frames=step*ACCUMULATION*MICRO_BATCH,
        UE_inputs=step*ACCUMULATION*MICRO_BATCH*8, next_micro_index=step*ACCUMULATION+1,
        accumulation_cursor=0, gamma_counts=gamma, EMA_commits=step,
        base_EMA_updates=base_ema_updates, total_EMA_updates=base_ema_updates+step)


def noise_seed(role, index, kind='noise'):
    if kind not in ('noise', 'polarity') or type(index) is not int:
        raise ValueError('Declared random stream required')
    if role == 'codec_train':
        gamma_schedule(index)
    elif role == 'final':
        if not 0 <= index < FINAL_BATCHES*len(GAMMAS):
            raise ValueError('Bounded fresh final noise index required')
    else:
        raise ValueError('Only training and fixed final roles are permitted')
    raw = f'{NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def recipe(arm=TRAIN_ARM):
    key(arm)
    old = previous.study_recipe()
    return plain(dict(namespace=NAMESPACE, arm=TRAIN_ARM, updates=UPDATES,
        training_arms=ARMS, evaluation_variants=EVAL_VARIANTS,
        source=dict(namespace=SOURCE_NAMESPACE, code_commit=SOURCE_COMMIT,
            audited_result_commit=SOURCE_RESULT_COMMIT, scientific_core_sha256=SOURCE_CORE_SHA256,
            pins=SOURCE_PINS, terminal_updates=10000, base_EMA_updates=BASE_EMA_UPDATES,
            fresh_optimizer=True, source_optimizer_continued=False),
        objective=dict(primary='negative mean complete8UE MU sum rate',
            auxiliary='mean projective channel-direction error per MU/UE/tone',
            direction_weight=DIRECTION_WEIGHT, weight_prespecified=True,
            weight_optimality_claim=False, heldout_weight_tuning=False,
            amplitude_and_phase_invariant=True),
        optimizer=dict(name='AdamW', lr=LR, weight_decay=WEIGHT_DECAY, betas=[.9, .999],
            eps=1e-8, clip=GRAD_CLIP, scheduler='constant LambdaLR', seed=OPTIMIZER_SEED),
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION, effective_batch=MICRO_BATCH*ACCUMULATION,
        trainable=['UE encoder', 'BS decoder'], frozen=['CE', 'CE-error table'],
        prior=dict(policy='one causal historical-power EMA commit per successful update',
            source_counter=BASE_EMA_UPDATES, terminal_counter=BASE_EMA_UPDATES+UPDATES,
            current_frame_power_side_information=False, evaluation_commits=0),
        physical=old['physical'], precision=old['precision'], Fourier_LS=old['Fourier_LS'],
        architecture=dict(unchanged=True, UE_SNR_input=False, UE_UID_input=False,
            user_projection=False, UE_uses_only_own_DL_CSI=True),
        receiver='unchanged original estimated-UL receiver',
        precoder=dict(design='RZF', alpha=ALPHA, downlink_snr_db=DL_SNR,
            per_tone_total_power=1., trainable_alpha=False, direct_W=False),
        gamma=dict(nodes=GAMMAS, balanced_three_update_schedule=True,
            terminal_counts=gamma_counts(UPDATES), unseen_continuous_SNR_claim=False),
        data=dict(namespace=NAMESPACE, seed=SEED, role_ranges=ROLE_RANGES,
            training_fresh_from_parent=True, test_used=False),
        selection=dict(fixed_terminal=UPDATES, intermediate_validation=False, best_checkpoint=False),
        final=dict(batches=FINAL_BATCHES, samples=FINAL_BATCHES*MICRO_BATCH,
            after_all_three_strict_loads=True, paired_channels_noise_pilots=True),
        comparison=dict(FDMA_additional_updates=0, Hybrid_additional_updates=UPDATES,
            equal_compute=False, auxiliary_effect_isolated=False,
            interpretation='Hybrid extension versus frozen previous terminals; extra updates and loss both change'),
        reporting=dict(primary='sum_rate_bps_hz', scopes=['point', 'low_0_5_10', 'all_six'],
            gamma_average_within_MU=True, interval_batch=MICRO_BATCH, training_seeds=1,
            retain_negative_results=True, test_used=False),
        delivery=dict(format='checked scalar CSV/JSON directory; no ZIP',
            explicit_git_upload=True, contains_model_weights=False, contains_channel_tensors=False)))


study_recipe = recipe


def fingerprint(arm=TRAIN_ARM):
    return digest(recipe(arm))


study_fingerprint = fingerprint
