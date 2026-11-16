"""Matched Hybrid-only low-feedback-SNR interference experiment metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs import twc_v1_k8_hybrid_direction_config as previous
from diagnose_twc_v1_k8_rate_headroom import SOURCE_PINS as _OLD_PINS

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = DATA_NAMESPACE = TRAIN_DATA_NAMESPACE = 'twc_v1_k8_low_snr_interference_v1'
OUTPUT_ROOT = ROOT/'runs'/NAMESPACE
SEED = TRAIN_SEED = OPTIMIZER_SEED = 20261128
TRAIN_ARM = 'hybrid_g4'
ARMS = TRAIN_ARMS = (TRAIN_ARM,)
TRAJECTORIES = ('low_snr_control', 'low_snr_interference')
EVAL_VARIANTS = ('fdma_frozen', 'hybrid_source', *TRAJECTORIES)
GAMMAS = (0, 5, 10, 15, 20, 25)
UPDATES, MICRO_BATCH, ACCUMULATION, SAVE_EVERY = 10000, 4, 8, 100
FINAL_BATCHES = 256
ROLE_RANGES = dict(codec_train=[1, UPDATES*ACCUMULATION], final=[4096, 4096+FINAL_BATCHES-1])
LR = LEARNING_RATE = 1e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1.
ALPHA, DL_SNR, DIRECTION_WEIGHT, INTERFERENCE_WEIGHT = .03, 25., 100., 1.
BASE_EMA_UPDATES, FDMA_BASE_EMA_UPDATES = 40000, 30000
SOURCE_NAMESPACE, SOURCE_ROOT = previous.NAMESPACE, previous.OUTPUT_ROOT
SOURCE_COMMIT = '94a796cba82eea07193306243d61ad8b54044466'
SOURCE_RESULT_COMMIT = 'aae777c626781564898f445f7ad659bbb725c3eb'
SOURCE_CORE_SHA256 = '38e880b31f729f6131110c2d88b4df540cc7e8f7c6e2a7c3fd1c7d31e7114fb5'
REPOSITORY = 'Charlie-Su-55/03_PRECODER'
REMOTE_URL = 'https://github.com/'+REPOSITORY+'.git'
RESULT_BRANCH_PREFIX = RESULTS_BRANCH_PREFIX = 'results/'+NAMESPACE+'/'
PARENT_CORE_FILES = previous.CORE_FILES
CORE_FILES = tuple(dict.fromkeys((*PARENT_CORE_FILES,
    'configs/twc_v1_k8_low_snr_config.py', 'low_snr_runtime.py', 'twc_v1_k8_low_snr.py',
    'train_twc_v1_k8_low_snr.py', 'evaluate_twc_v1_k8_low_snr.py',
    'twc_v1_k8_low_snr_results.py', 'run_twc_v1_k8_low_snr.py',
    'publish_twc_v1_k8_low_snr.py', 'twc_v1_k8_rate_headroom.py',
    'twc_v1_k8_ul_propagation_metrics.py', 'twc_v1_k8_ul_diagnostics_metrics.py',
    'ul_diagnostics_runtime.py')))


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


FDMA_PINS = plain(_OLD_PINS['fdma'])
SOURCE_PINS = {
    "checkpoint_sha256": "2260debba08a0dc754cefe53a18975b41e8cfacb3f3613dfbaa1a4498c8c1a97",
    "manifest_sha256": "93b790555848c2494bacc73eab40d2d41ebeead2f9062a27b3b35c3b5e88efbd",
    "terminal_receipt_sha256": "93b790555848c2494bacc73eab40d2d41ebeead2f9062a27b3b35c3b5e88efbd",
    "initialization_sha256": "12736451e4d02c2f41841fb7619996ae5ffc36cbdc78020f14966766dcc8ff88",
    "progress_sha256": "0458442260c13e5e485e19ed932963780da100a71885ba781c66b074434b962a",
    "encoder_state_sha256": "c49265d1ce129f70fb9963c9dbfc51b71e8bbcfc329167afeb7bd4dd45ae4ff4",
    "decoder_state_sha256": "7763c1c203182148e44cb4699aedf8032da33320233ab5b89e62bdbcc931bf6f",
    "state_sha256": "a3ea97e8e557790fc39970a503b561e05e0850eb40b60d659e2d35ac071e4306",
    "statistics_state_sha256": "f2394a1f1af51217a24265bde06d61029b0856fdcb6f70e736a68f7c4afca0c0",
    "CE_state_sha256": "661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192"
}

# The full latest Hybrid terminal pins are bound to the independently audited
# direction-assisted result, not an older rate5000 or joint10000 source.


def key(trajectory):
    if trajectory not in TRAJECTORIES:
        raise ValueError('Only declared Hybrid low-SNR trajectories may train')
    return trajectory


def gamma_schedule(index):
    if type(index) is not int or not 1 <= index <= UPDATES*ACCUMULATION:
        raise ValueError('Declared low-SNR training micro index required')
    if (MICRO_BATCH, ACCUMULATION) != (4, 8):
        raise ValueError('The exact 32-MU low-SNR schedule requires micro4 accumulation8')
    update, micro = (index-1)//ACCUMULATION+1, (index-1) % ACCUMULATION
    result = []
    for within in range(MICRO_BATCH):
        q = MICRO_BATCH*micro+within
        result.append(GAMMAS[q % 3] if q < 24 else GAMMAS[3+(update-1+q-24) % 3])
    return tuple(result)


def gamma_counts(step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError('Complete low-SNR optimizer boundary required')
    full, rest = divmod(step, 3)
    counts = {str(g): 8*step if i < 3 else 8*full for i, g in enumerate(GAMMAS)}
    for update in range(rest):
        for offset in range(8): counts[str(GAMMAS[3+(update+offset) % 3])] += 1
    return counts


def micro_index(update, micro):
    if (type(update) is not int or not 1 <= update <= UPDATES or type(micro) is not int
            or not 0 <= micro < ACCUMULATION):
        raise ValueError('Declared low-SNR update and accumulation index required')
    return (update-1)*ACCUMULATION+micro+1


def counters(step, base_ema_updates=BASE_EMA_UPDATES):
    counts = gamma_counts(step)
    if type(base_ema_updates) is not int or base_ema_updates != BASE_EMA_UPDATES:
        raise ValueError('The actual latest Hybrid EMA40000 must be inherited')
    return dict(optimizer_updates=step, UE_updates=step, BS_updates=step, CE_updates=0,
        FDMA_updates=0, microbatches=step*ACCUMULATION, MU_frames=step*ACCUMULATION*MICRO_BATCH,
        UE_inputs=step*ACCUMULATION*MICRO_BATCH*8, next_micro_index=step*ACCUMULATION+1,
        accumulation_cursor=0, gamma_counts=counts, EMA_commits=step,
        base_EMA_updates=base_ema_updates, total_EMA_updates=base_ema_updates+step)


def noise_seed(role, index, kind='noise'):
    if kind not in ('noise', 'polarity') or type(index) is not int:
        raise ValueError('Declared matched random stream required')
    if role == 'codec_train': gamma_schedule(index)
    elif role == 'final':
        if not 0 <= index < FINAL_BATCHES*len(GAMMAS):
            raise ValueError('Bounded fresh final noise index required')
    else: raise ValueError('Only training and fixed final roles permitted')
    raw = f'{NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def study_recipe():
    old = previous.study_recipe()
    return plain(dict(namespace=NAMESPACE, arm=TRAIN_ARM, trajectories=TRAJECTORIES,
        evaluation_variants=EVAL_VARIANTS, updates_per_trajectory=UPDATES,
        total_new_UE_updates=UPDATES*len(TRAJECTORIES), total_new_BS_updates=UPDATES*len(TRAJECTORIES),
        FDMA_new_updates=0, CE_new_updates=0,
        source=dict(namespace=SOURCE_NAMESPACE, code_commit=SOURCE_COMMIT,
            audited_result_commit=SOURCE_RESULT_COMMIT, scientific_core_sha256=SOURCE_CORE_SHA256,
            pins=SOURCE_PINS, FDMA_pins=FDMA_PINS, terminal_updates=10000,
            base_EMA_updates=BASE_EMA_UPDATES, FDMA_EMA_updates=FDMA_BASE_EMA_UPDATES,
            fresh_optimizer=True, source_optimizer_continued=False),
        objective=dict(primary='negative mean complete8UE MU sum rate',
            direction='mean projective direction error', direction_weight=DIRECTION_WEIGHT,
            interference='mean_tone sum_UE log2(1+I/(S+N_DL))', interference_weight=INTERFERENCE_WEIGHT,
            interference_gamma_max=10, low_mask_renormalized=False,
            amplitude_and_phase_invariant_direction=True, coefficients_prespecified=True,
            optimality_claim=False, heldout_tuning=False, raw_UL_cross_power_loss=False),
        optimizer=dict(name='AdamW', lr=LR, weight_decay=WEIGHT_DECAY, betas=[.9, .999],
            eps=1e-8, clip=GRAD_CLIP, scheduler='constant LambdaLR', seed=OPTIMIZER_SEED),
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION, effective_batch=MICRO_BATCH*ACCUMULATION,
        trainable=['UE encoder', 'BS decoder'], frozen=['CE', 'CE-error table'],
        prior=dict(policy='one causal historical-power EMA commit per successful update',
            source_counter=BASE_EMA_UPDATES, terminal_counter=BASE_EMA_UPDATES+UPDATES,
            current_frame_power_side_information=False, evaluation_commits=0),
        physical=old['physical'], precision=old['precision'], Fourier_LS=old['Fourier_LS'],
        architecture=old['architecture'], receiver='unchanged original estimated-UL receiver',
        precoder=dict(design='RZF', alpha=ALPHA, downlink_snr_db=DL_SNR,
            per_tone_total_power=1., trainable_alpha=False, direct_W=False),
        gamma=dict(nodes=GAMMAS, feedback_low_fraction=.75, feedback_high_fraction=.25,
            exact_each_update_low_counts=[8, 8, 8], high_counts_rotating=[3, 3, 2],
            terminal_counts=gamma_counts(UPDATES), evaluation_equal_six_node_weight=True,
            UE_SNR_input=False, unseen_continuous_SNR_claim=False),
        data=dict(namespace=NAMESPACE, seed=SEED, role_ranges=ROLE_RANGES,
            fresh_from_parent=True, same_actual_inputs_across_trajectories=True, test_used=False),
        selection=dict(fixed_terminal=UPDATES, intermediate_validation=False, best_checkpoint=False),
        final=dict(batches=FINAL_BATCHES, samples=FINAL_BATCHES*MICRO_BATCH,
            after_all_four_strict_loads=True, paired_channels_noise_pilots=True,
            offline_UL_error_decomposition=True, true_UL_used_to_design_filter=False),
        comparison=dict(trajectories_equal_compute_data=True, FDMA_equal_compute=False,
            leakage_effect_isolated_between_new_trajectories=True,
            reweighting_effect_isolated_from_extra_updates=False),
        reporting=dict(primary='sum_rate_bps_hz', primary_scope='low_0_5_10',
            scopes=['point', 'low_0_5_10', 'high_15_20_25', 'all_six'],
            gamma_average_within_MU=True, interval_batch=MICRO_BATCH, source_seeds=1,
            retain_negative_results=True, test_used=False),
        delivery=dict(format='checked scalar CSV/JSON directory; no ZIP',
            explicit_git_upload=True, contains_model_weights=False, contains_channel_tensors=False)))


def recipe(trajectory=None):
    if trajectory is None: return study_recipe()
    key(trajectory)
    return plain(dict(study_recipe(), trajectory=trajectory,
        active_interference_weight=INTERFERENCE_WEIGHT if trajectory == TRAJECTORIES[1] else 0.))


def fingerprint(trajectory=None):
    return digest(recipe(trajectory))


def study_fingerprint():
    return digest(study_recipe())
