"""Paired continuation versus residual-W control from pinned BS-rate terminals.

All earlier source, rate and alpha experiments remain read-only. The new data
namespace separates this study's samples while retaining the physical protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from configs import twc_v1_k8_icc_alpha_config as alphacfg
from configs import twc_v1_k8_icc_rate_config as ratecfg

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_w_residual_v1'
DATA_NAMESPACE = NAMESPACE
OUTPUT_ROOT = ROOT / 'runs' / NAMESPACE
ARMS = ('fdma', 'hybrid_g4')
TRAIN_METHODS = ('rzf_continue', 'w_residual')
POLICIES = ('rate_source', *TRAIN_METHODS)
METHODS = tuple(f'{policy}_{arm}' for policy in POLICIES for arm in ARMS)
GAMMAS = ratecfg.GAMMAS
SEED = 20261121
HEAD_SEED = 20261112
UPDATES = 5000
LR = ratecfg.LR
LEARNING_RATE = LR
WEIGHT_DECAY = ratecfg.WEIGHT_DECAY
GRAD_CLIP = ratecfg.GRAD_CLIP
MICRO_BATCH = ratecfg.MICRO_BATCH
ACCUMULATION = ratecfg.ACCUMULATION
SAVE_EVERY = 100
MONITOR_STEPS = (0, 100, 1000, 2500, UPDATES)
MONITOR_BATCHES = 32
FINAL_BATCHES = 256
FIXED_ALPHA = ratecfg.FIXED_ALPHA
ALPHA = FIXED_ALPHA
DL_SNR = ratecfg.DL_SNR
HEAD_WIDTH = 64
HEAD_LAYERS = 2
HEAD_HEADS = 4
HEAD_DROPOUT = 0.
HEAD_INPUT_FEATURES = 76
HEAD_FF = 128
DELTA_SCALE = 1. / math.sqrt(32 * 8)
ROLE_RANGES = dict(codec_train=[1, UPDATES * ACCUMULATION],
                   monitor=[0, MONITOR_BATCHES - 1], final=[2048, 2303])
PRIMARY_METRIC = 'sum_rate_bps_hz'
PRIMARY_SCOPE = 'all_six'

# These refer to the existing 5000-update rate terminals, never alpha-selected
# weights or a monitor checkpoint. Actual bytes must pass the strict loaders.
SOURCE_ROOT = ratecfg.OUTPUT_ROOT
SOURCE_COMMIT = alphacfg.RATE_COMMIT
SOURCE_CORE = alphacfg.RATE_CORE
SOURCE_CHECKPOINT_SHA256 = alphacfg.RATE_CHECKPOINT_SHA256
SOURCE_TERMINAL_SHA256 = alphacfg.RATE_TERMINAL_SHA256
ORIGINAL_SOURCE_ROOT = ratecfg.SOURCE_ROOT
ORIGINAL_SOURCE_COMMIT = ratecfg.SOURCE_COMMIT
ORIGINAL_SOURCE_CORE = ratecfg.SOURCE_CORE
ORIGINAL_SOURCE_CHECKPOINT_SHA256 = ratecfg.SOURCE_CHECKPOINT_SHA256
ORIGINAL_SOURCE_TERMINAL_SHA256 = ratecfg.SOURCE_TERMINAL_SHA256
SOURCE_CALIBRATION_SHA256 = ratecfg.SOURCE_CALIBRATION_SHA256

# Exact completed alpha core, independently pinned so dirty legacy adapter
# changes cannot silently enter the new study. Never pin the new files here.
LEGACY_CORE = {'schema': 'twc_v1_k8_icc_alpha_v1_core_v1',
 'sha256': {'configs/twc_v1_k8_icc_aligned_config.py': '4ecc52b2f051e65f13881b278b661d5ae198e9c3b294c7c69848d5f634ead9e0',
            'configs/twc_v1_k8_icc_alpha_config.py': '0c2f5e1b8c80473d594d4d6b5e9bb5a96d7170cefe2dccd7e864bc72f769cbf9',
            'configs/twc_v1_k8_icc_rate_config.py': '499c4002cea38ca506a23a89604f7d3c0078de007f43818490031a227f8813e2',
            'evaluate_twc_v1_k8_icc_aligned.py': '4ec956d385521a44e56dcaf74e056324c0ab9f13b6e4dcca2e1e9a3754ed4757',
            'evaluate_twc_v1_k8_icc_alpha.py': '6c098135c6c47120941ff7cd3b1888da228e5d5b763bfeadd4f6b713f7a1217b',
            'evaluate_twc_v1_k8_icc_rate.py': '36e886ced4ec23e14876c3efd4c467c0d09e72d6ade3fe355c0ed8a300bf4fec',
            'icc_alpha_runtime.py': 'a13c98d12aa8ff28b53c8d5503482e38a632ebd5e29457529321ec48c33d7529',
            'models/baseline_common.py': '3c1c2f89a309de5c8f8119d8aad10d64efcb3f0bbd783d94efbda3a32e629398',
            'models/swin_precoder.py': '16c67415ccd55af2bb3cb6479c7d8282b3fb3582406f76dd30cb102785e1983c',
            'models/twc_v1_k8_icc_aligned.py': '50442cf3455d366045ae1485678d74d4ea14f2dfe3dccd2bde8a05925a55fcd1',
            'models/twc_v1_k8_joint_decoder.py': '20a8ef8d638cc155afe182b4325a15463aa5444ef0e296739d1e5dd375e53732',
            'physical_split_finish_io.py': '2df47a8d5509fa20cd504a3fcb9a4641dd471daf06dc61a6de5f242c28f1be37',
            'run_twc_v1_k8_icc_aligned.py': '1c05be2984a93cc3d874743536285427ce481164c86482b19b69e74a894ce00a',
            'run_twc_v1_k8_icc_alpha.py': 'ef14911b589e1a446b6e27cd9a910ebe4cf66fb51843f2c912521b8d0d51a914',
            'run_twc_v1_k8_icc_rate.py': 'ee49b410438f796528c354b1ed58d1484e3d51a58ac451a5ac1c7b99d24fb58f',
            'train_twc_v1_k8_icc_aligned.py': '201f60cf980d0b6e5c6fb2978d6772a1a73da06a2934966b49b1b194ed3a8ab7',
            'train_twc_v1_k8_icc_rate.py': '8e3619a39894422352004ec9b889636901e19fed3b93c7ea8506b1e16dedf1d0',
            'twc_v1_k8_icc_aligned_results.py': '5de1d4e5f675783760648b3140db0716cd12c4882e289c028149931f238854af',
            'twc_v1_k8_icc_alpha_results.py': '6b2ae8cc69c6a81617f99bcda298762cb2d05b50ba82069f1b52c8d8cd1a7cfb',
            'utils/channel_generator.py': '645987b832f3137273fe113433640bdd4995fb2dfc54d13b13c80c91d7fe251d',
            'utils/sionna_rng.py': 'd670a482399a260b3a930286f15e14ecdbe60252fc83bbb82944c57bcf472861',
            'utils/twc_v1_k8_downlink.py': '6c1effe348a9559874febb29a9a7ef3f0470d72735248e73a172291cbaccf175',
            'utils/twc_v1_k8_icc_aligned_physics.py': 'b959a21d9893793ca4cbbedfea70c716eed3ac36aa1addd2c5841ac50a4f81a9',
            'utils/twc_v1_k8_icc_rate.py': '3fa626230dcb6b3aaf0f64931af1ff533973249e19f2639a22369704d3b9d550',
            'utils/twc_v1_k8_protocol.py': '68efd01714797f771aa4e165218c1eef1489438a4ec0d01e209ed099e35b58b8',
            'utils/twc_v1_k8_replayable_data.py': 'a8543f6b721b960f99a479841c4ebc8d3789bdb226de43c8b965a9e75d2ba76b'}}

CORE_FILES = tuple(dict.fromkeys((*alphacfg.CORE_FILES,
    'configs/twc_v1_k8_w_residual_config.py', 'models/twc_v1_k8_w_residual.py',
    'w_residual_runtime.py', 'train_twc_v1_k8_w_residual.py',
    'evaluate_twc_v1_k8_w_residual.py', 'run_twc_v1_k8_w_residual.py',
    'twc_v1_k8_w_residual_results.py')))


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def method_name(policy, arm):
    if policy not in POLICIES or arm not in ARMS:
        raise ValueError('Declared evaluation policy and physical arm required')
    return f'{policy}_{arm}'


def key(arm, method):
    if method not in TRAIN_METHODS or arm not in ARMS:
        raise ValueError('Declared trainable method and physical arm required')
    return method_name(method, arm)


def study_recipe():
    original = ratecfg.recipe('fdma')
    return plain(dict(namespace=NAMESPACE, data_namespace=DATA_NAMESPACE, seed=SEED,
        arms=ARMS, training_methods=TRAIN_METHODS, policies=POLICIES, methods=METHODS,
        source=dict(root=str(SOURCE_ROOT), commit=SOURCE_COMMIT,
            core=SOURCE_CORE, terminal_updates=ratecfg.UPDATES,
            checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            terminal_sha256=SOURCE_TERMINAL_SHA256, read_only=True,
            source_optimizer_continued=False, source_decoder_shared_between_methods=True),
        original_source=dict(root=str(ORIGINAL_SOURCE_ROOT), commit=ORIGINAL_SOURCE_COMMIT,
            core=ORIGINAL_SOURCE_CORE, checkpoint_sha256=ORIGINAL_SOURCE_CHECKPOINT_SHA256,
            terminal_sha256=ORIGINAL_SOURCE_TERMINAL_SHA256,
            CE_calibration_sha256=SOURCE_CALIBRATION_SHA256, read_only=True),
        updates_per_arm_method=UPDATES, microbatch=MICRO_BATCH,
        accumulation=ACCUMULATION, effective_batch=MICRO_BATCH * ACCUMULATION,
        optimizer=dict(name='AdamW', fresh=True, lr=LR, weight_decay=WEIGHT_DECAY,
            betas=[.9, .999], eps=1e-8, clip=GRAD_CLIP, scheduler='constant LambdaLR'),
        loss='negative arithmetic mean of complete-MU downlink sum-rate',
        auxiliary_CSI_loss=False, rate_training=True,
        frozen=['UE encoder parameters and buffers', 'CE parameters and buffers',
                'historical power prior and calibration'],
        base_decoder_trainable_for_both_methods=True,
        head=dict(name='Transformer residual precoder', width=HEAD_WIDTH, layers=HEAD_LAYERS,
            heads=HEAD_HEADS, dropout=HEAD_DROPOUT, seed=HEAD_SEED,
            input_features=HEAD_INPUT_FEATURES, feedforward=HEAD_FF,
            inputs='estimated complex CSI, public gamma, user identity and frequency coordinates',
            shared_initialization_across_arms=True, zero_output_initialization=True,
            delta_scale=DELTA_SCALE, true_channel_input=False),
        precoder=dict(base='original differentiable RZF', alpha=FIXED_ALPHA,
            trainable_alpha=False, residual='normalize_per_tone(W0 + delta_scale * DeltaW)',
            per_tone_total_power=1., pure_direct_W=False),
        physical=original['physical'], precision=original['precision'],
        gamma=original['gamma'], Fourier_LS=original['Fourier_LS'],
        CE_calibration=original['CE_calibration'],
        power_prior=dict(source_EMA_commits=20000, additional_EMA_commits=0,
            update=False, current_frame_power_side_information=False),
        data=dict(namespace=DATA_NAMESPACE, addresses=ROLE_RANGES,
            train_role='codec_train', monitor_role='validation', final_role='validation',
            common_actual_channels_noise_pilots_and_gamma=True,
            train_pairing='both methods and both physical arms', test_used=False),
        selection=f'fixed terminal{UPDATES}; no monitor/final checkpoint selection',
        monitor=dict(samples=MONITOR_BATCHES * MICRO_BATCH, batches=MONITOR_BATCHES,
            steps=MONITOR_STEPS, gamma=GAMMAS, selection=False),
        final=dict(samples=FINAL_BATCHES * MICRO_BATCH, batches=FINAL_BATCHES,
            gamma=GAMMAS, DL_SNR=DL_SNR, methods=METHODS, test_used=False,
            paired_source_and_two_terminal_methods=True),
        reporting=dict(primary_metric=PRIMARY_METRIC, primary_scope=PRIMARY_SCOPE,
            comparison='w_residual minus equally trained rzf_continue within each arm',
            interaction='Hybrid residual gain minus FDMA residual gain',
            primary_comparisons=[f'{arm}_w_residual_minus_rzf_continue' for arm in ARMS],
            interaction_comparison='interaction_w_residual_minus_rzf_continue',
            preserve_all_gamma=True, preserve_negative_results=True,
            paired_unit='complete8UE MU', batch_interval=MICRO_BATCH,
            within_MU_gamma_average=True, retain_all_user_rates=True,
            retain_min_user_rate=True, training_seed_replications=1),
        sample_budget=dict(train_runs=len(ARMS) * len(TRAIN_METHODS),
            additional_MU_frames_per_arm_method=UPDATES * ACCUMULATION * MICRO_BATCH,
            unique_paired_additional_MU_frames=UPDATES * ACCUMULATION * MICRO_BATCH,
            additional_UE_inputs_per_arm_method=UPDATES * ACCUMULATION * MICRO_BATCH * 8,
            new_CE_updates=0, new_UE_updates=0, new_EMA_updates=0)))


def study_fingerprint():
    return digest(study_recipe())


def recipe(arm, method):
    key(arm, method)
    value = study_recipe()
    value.update(arm=arm, method=method, updates=UPDATES,
        trainable=('BS decoder parameters and residual-W head' if method == 'w_residual'
                   else 'BS decoder parameters only'),
        residual_head_enabled=(method == 'w_residual'))
    return plain(value)


def fingerprint(arm, method):
    return digest(recipe(arm, method))


def micro_index(update, micro):
    if (type(update) is not int or not 1 <= update <= UPDATES
            or type(micro) is not int or not 0 <= micro < ACCUMULATION):
        raise ValueError('Declared update and accumulation micro index required')
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
        raise ValueError('Declared independent role and nonnegative noise address required')
    if kind not in ('noise', 'gamma', 'polarity'):
        raise ValueError('Declared independent random stream required')
    # No method/arm term: all four trained systems use actual paired inputs.
    raw = f'{DATA_NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63 - 1)
