"""Pinned, decoder-only rate fine-tuning of the completed ICC-aligned K8 batch.

This is a new experiment. The original source and all old outputs remain read-only.
The channel data namespace is independent while the physical protocol is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs import twc_v1_k8_icc_aligned_config as base

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_icc_rate_v1'
DATA_NAMESPACE = NAMESPACE
OUTPUT_ROOT = ROOT / 'runs' / NAMESPACE
SOURCE_ROOT = base.OUTPUT_ROOT
SOURCE_COMMIT = '9d345b2385d99eec1d047bd9c602638b568c84f8'
ARMS = ('fdma', 'hybrid_g4')
GAMMAS = (0, 5, 10, 15, 20, 25)
SEED = 20261121
UPDATES = 5000
LR = 1e-5
LEARNING_RATE = LR
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.
MICRO_BATCH = 4
ACCUMULATION = 8
SAVE_EVERY = 100
MONITOR_STEPS = (0, 100, 1000, 2500, 5000)
MONITOR_BATCHES = 32
FINAL_BATCHES = 256
ALPHA = .03
FIXED_ALPHA = ALPHA
DL_SNR = 25.
ROLE_RANGES = dict(codec_train=[1, UPDATES * ACCUMULATION], monitor=[0, 31], final=[2048, 2303])

# Pins are from the uploaded, hash-audited completed batch. They are requirements,
# not substitutes for checking the actual server files and strict tensor loading.
SOURCE_CORE = {'schema': 'twc_v1_k8_icc_aligned_v1_core_v1',
 'sha256': {'configs/twc_v1_k8_icc_aligned_config.py': '4ecc52b2f051e65f13881b278b661d5ae198e9c3b294c7c69848d5f634ead9e0',
            'evaluate_twc_v1_k8_icc_aligned.py': '4ec956d385521a44e56dcaf74e056324c0ab9f13b6e4dcca2e1e9a3754ed4757',
            'models/baseline_common.py': '3c1c2f89a309de5c8f8119d8aad10d64efcb3f0bbd783d94efbda3a32e629398',
            'models/swin_precoder.py': '16c67415ccd55af2bb3cb6479c7d8282b3fb3582406f76dd30cb102785e1983c',
            'models/twc_v1_k8_icc_aligned.py': '50442cf3455d366045ae1485678d74d4ea14f2dfe3dccd2bde8a05925a55fcd1',
            'models/twc_v1_k8_joint_decoder.py': '20a8ef8d638cc155afe182b4325a15463aa5444ef0e296739d1e5dd375e53732',
            'physical_split_finish_io.py': '2df47a8d5509fa20cd504a3fcb9a4641dd471daf06dc61a6de5f242c28f1be37',
            'train_twc_v1_k8_icc_aligned.py': '201f60cf980d0b6e5c6fb2978d6772a1a73da06a2934966b49b1b194ed3a8ab7',
            'utils/channel_generator.py': '645987b832f3137273fe113433640bdd4995fb2dfc54d13b13c80c91d7fe251d',
            'utils/sionna_rng.py': 'd670a482399a260b3a930286f15e14ecdbe60252fc83bbb82944c57bcf472861',
            'utils/twc_v1_k8_downlink.py': '6c1effe348a9559874febb29a9a7ef3f0470d72735248e73a172291cbaccf175',
            'utils/twc_v1_k8_icc_aligned_physics.py': 'b959a21d9893793ca4cbbedfea70c716eed3ac36aa1addd2c5841ac50a4f81a9',
            'utils/twc_v1_k8_protocol.py': '68efd01714797f771aa4e165218c1eef1489438a4ec0d01e209ed099e35b58b8',
            'utils/twc_v1_k8_replayable_data.py': 'a8543f6b721b960f99a479841c4ebc8d3789bdb226de43c8b965a9e75d2ba76b'}}
SOURCE_CHECKPOINT_SHA256 = {'ce': '95c1e786b2036ecab81a865408555069ae3ce699620d1afacd14fc11982db494',
 'fdma': '68c8672dc0088da2d8cb24cda5491e1858e60ea9ae49a11911bc86fa9eaaa0f8',
 'hybrid_g4': 'e98df0e507626e39817c3245e7d0b6cefa76a3460563aed88d564c4a8081f155'}
SOURCE_TERMINAL_SHA256 = {'ce': '4567630f64e466ff5e6ffb8e77e20ffa9673a32a98975539d2c09c97f1d76c14',
 'fdma': 'e2be37259e2fac75581dc79c2df0060f3db97904a8af927f4278753f266a0a14',
 'hybrid_g4': '5323ed2e99a718afed3d7ab1b6d4e0d7d31f373c1ed7fbe44cce60d57435e8f4'}
SOURCE_CALIBRATION_SHA256 = 'e9ff6ed64a83f224c9d0d63baa2dd69c783c51b2cec25349c9a843974199fb2b'

CORE_FILES = tuple(dict.fromkeys((*base.CORE_FILES,
    'configs/twc_v1_k8_icc_rate_config.py', 'utils/twc_v1_k8_icc_rate.py',
    'train_twc_v1_k8_icc_rate.py', 'evaluate_twc_v1_k8_icc_rate.py',
    'run_twc_v1_k8_icc_rate.py', 'twc_v1_k8_icc_aligned_results.py',
    'run_twc_v1_k8_icc_aligned.py')))


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('Exactly the paired FDMA and Hybrid G4 arms are allowed')
    original = base.recipe(arm)
    return plain(dict(namespace=NAMESPACE, arm=arm, seed=SEED,
        source=dict(commit=SOURCE_COMMIT, namespace=base.NAMESPACE,
            terminal_updates=base.CODEC_UPDATES, checkpoint_sha256=SOURCE_CHECKPOINT_SHA256[arm],
            CE_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256['ce'],
            CE_calibration_sha256=SOURCE_CALIBRATION_SHA256, core=SOURCE_CORE,
            read_only=True, original_optimizer_continued=False),
        updates=UPDATES, microbatch=MICRO_BATCH, accumulation=ACCUMULATION,
        effective_batch=MICRO_BATCH * ACCUMULATION,
        trainable='BS decoder parameters only',
        frozen=['UE encoder', 'CE parameters and buffers', 'historical power prior and calibration'],
        loss='negative arithmetic mean of complete-MU downlink sum-rate',
        auxiliary_CSI_loss=False, rate_training=True,
        optimizer=dict(name='AdamW', lr=LR, weight_decay=WEIGHT_DECAY,
            betas=[.9,.999], eps=1e-8, clip=GRAD_CLIP, scheduler='constant LambdaLR'),
        precoder=dict(design='original differentiable RZF', alpha=ALPHA,
            per_tone_total_power=1., trainable_alpha=False, direct_W_head=False),
        physical=original['physical'], precision=original['precision'],
        gamma=original['gamma'], Fourier_LS=original['Fourier_LS'],
        CE_calibration=original['CE_calibration'],
        power_prior=dict(source_EMA_commits=base.CODEC_UPDATES, additional_EMA_commits=0,
            update=False, current_frame_power_side_information=False),
        data_namespace=DATA_NAMESPACE, addresses=ROLE_RANGES,
        selection='fixed terminal5000; no validation checkpoint selection',
        monitor=dict(samples=MONITOR_BATCHES * MICRO_BATCH, steps=MONITOR_STEPS,
            gamma=GAMMAS, selection=False),
        final=dict(samples=FINAL_BATCHES * MICRO_BATCH, gamma=GAMMAS,
            DL_SNR=DL_SNR, alpha=ALPHA, test_used=False,
            compare_source_and_finetuned_on_same_new_inputs=True),
        reporting=dict(primary_comparison='interaction_hybrid_gain_minus_fdma_gain',
            primary_metric='sum_rate_bps_hz', primary_scope='all_six',
            secondary_scopes=['low_0_5_10','point'], retain_all_user_rates=True),
        sample_budget=dict(additional_MU_frames_per_arm=UPDATES * ACCUMULATION * MICRO_BATCH,
            unique_paired_additional_MU_frames=UPDATES * ACCUMULATION * MICRO_BATCH,
            additional_UE_inputs_per_arm=UPDATES * ACCUMULATION * MICRO_BATCH * 8,
            new_CE_updates=0, new_UE_updates=0, new_EMA_updates=0)))


def fingerprint(arm):
    return digest(recipe(arm))


def micro_index(update, micro):
    if (type(update) is not int or not 1 <= update <= UPDATES
            or type(micro) is not int or not 0 <= micro < ACCUMULATION):
        raise ValueError('Declared update1..5000 and micro index0..7 required')
    return (update - 1) * ACCUMULATION + micro + 1


def counters(step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError('Invalid additional BS optimizer boundary')
    return dict(optimizer_updates=step, microbatches=step * ACCUMULATION,
        MU_frames=step * ACCUMULATION * MICRO_BATCH,
        UE_inputs=step * ACCUMULATION * MICRO_BATCH * 8,
        next_micro_index=step * ACCUMULATION + 1, accumulation_cursor=0,
        EMA_commits=0)


def noise_seed(role, index, kind='noise'):
    if role not in ROLE_RANGES or type(index) is not int or index < 0:
        raise ValueError('Declared independent data role and nonnegative noise address required')
    if kind not in ('noise', 'gamma', 'polarity'):
        raise ValueError('Declared independent random stream required')
    # No arm term: corresponding arms use actual paired random inputs.
    raw = f'{DATA_NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def validate_resume(payload, arm, *, core, data_protocol, source_identity, frozen_identity):
    required = {'format','arm','step','recipe','fingerprint','core','data_protocol',
        'source_identity','frozen_identity','model','optimizer','scheduler','rng','counters'}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ValueError('Incomplete decoder-only rate checkpoint')
    expected = dict(format=NAMESPACE, arm=arm, recipe=recipe(arm),
        fingerprint=fingerprint(arm), core=core, data_protocol=data_protocol,
        source_identity=source_identity, frozen_identity=frozen_identity)
    for key, value in expected.items():
        if plain(payload[key]) != plain(value):
            raise ValueError('Rate resume identity changed: ' + key)
    step = payload['step']
    if type(step) is not int or step % SAVE_EVERY:
        raise ValueError('Only complete100-update saved boundaries are accepted')
    if payload['counters'] != counters(step):
        raise ValueError('Rate resume counts/cursor/frozen EMA boundary changed')
    scheduler = payload['scheduler']
    if (not isinstance(scheduler, dict) or scheduler.get('last_epoch') != step
            or scheduler.get('_step_count') != step + 1):
        raise ValueError('Rate scheduler must match the complete optimizer boundary')
    if (not isinstance(payload['rng'], dict)
            or set(payload['rng']) != {'python','numpy','torch_cpu','torch_cuda'}):
        raise ValueError('Complete external RNG state required')
    if not payload['model'] or not payload['optimizer']:
        raise ValueError('Decoder and fresh optimizer state are required')
    return step
