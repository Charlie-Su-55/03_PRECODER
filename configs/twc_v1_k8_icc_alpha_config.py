"""Frozen inference-only global RZF-alpha control for the ICC-aligned batch.

This configuration is independent of all source/rate outputs. It declares
selection before final validation and contains no optimizer/training budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs import twc_v1_k8_icc_rate_config as ratecfg

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_icc_alpha_v1'
DATA_NAMESPACE = NAMESPACE
OUTPUT_ROOT = ROOT / 'runs' / NAMESPACE
SOURCE_ROOT = ratecfg.SOURCE_ROOT
RATE_ROOT = ratecfg.OUTPUT_ROOT
RATE_COMMIT = '957b730d56f07fae9c93cb2611f5e460f7566e67'
ARMS = ('fdma', 'hybrid_g4')
GAMMAS = (0, 5, 10, 15, 20, 25)
ALPHA_GRID = (.003, .01, .02, .03, .05, .07, .1, .2, .3)
FIXED_ALPHA = .03
DL_SNR = 25.
MICRO_BATCH = 4
SEED = 20261121
CALIBRATION_BATCHES = 64
FINAL_BATCHES = 256
ROLE_RANGES = dict(calibration=[0,63], final=[2048,2303])
POLICIES = ('source_fixed', 'source_calibrated', 'rate_fixed')
METHODS = tuple(f'{policy}_{arm}' for policy in POLICIES for arm in ARMS)
PRIMARY_METRIC = 'sum_rate_bps_hz'
PRIMARY_SCOPE = 'all_six'

# Audit-derived requirements; actual server bytes and strict tensor loading
# must be checked again. These are never inferred from current final scores.
SOURCE_COMMIT = ratecfg.SOURCE_COMMIT
SOURCE_CORE = ratecfg.SOURCE_CORE
SOURCE_CHECKPOINT_SHA256 = ratecfg.SOURCE_CHECKPOINT_SHA256
SOURCE_TERMINAL_SHA256 = ratecfg.SOURCE_TERMINAL_SHA256
SOURCE_CALIBRATION_SHA256 = ratecfg.SOURCE_CALIBRATION_SHA256

RATE_CORE = {'schema': 'twc_v1_k8_icc_rate_v1_core_v1',
 'sha256': {'configs/twc_v1_k8_icc_aligned_config.py': '4ecc52b2f051e65f13881b278b661d5ae198e9c3b294c7c69848d5f634ead9e0',
            'configs/twc_v1_k8_icc_rate_config.py': '499c4002cea38ca506a23a89604f7d3c0078de007f43818490031a227f8813e2',
            'evaluate_twc_v1_k8_icc_aligned.py': '4ec956d385521a44e56dcaf74e056324c0ab9f13b6e4dcca2e1e9a3754ed4757',
            'evaluate_twc_v1_k8_icc_rate.py': '36e886ced4ec23e14876c3efd4c467c0d09e72d6ade3fe355c0ed8a300bf4fec',
            'models/baseline_common.py': '3c1c2f89a309de5c8f8119d8aad10d64efcb3f0bbd783d94efbda3a32e629398',
            'models/swin_precoder.py': '16c67415ccd55af2bb3cb6479c7d8282b3fb3582406f76dd30cb102785e1983c',
            'models/twc_v1_k8_icc_aligned.py': '50442cf3455d366045ae1485678d74d4ea14f2dfe3dccd2bde8a05925a55fcd1',
            'models/twc_v1_k8_joint_decoder.py': '20a8ef8d638cc155afe182b4325a15463aa5444ef0e296739d1e5dd375e53732',
            'physical_split_finish_io.py': '2df47a8d5509fa20cd504a3fcb9a4641dd471daf06dc61a6de5f242c28f1be37',
            'run_twc_v1_k8_icc_aligned.py': '1c05be2984a93cc3d874743536285427ce481164c86482b19b69e74a894ce00a',
            'run_twc_v1_k8_icc_rate.py': 'ee49b410438f796528c354b1ed58d1484e3d51a58ac451a5ac1c7b99d24fb58f',
            'train_twc_v1_k8_icc_aligned.py': '201f60cf980d0b6e5c6fb2978d6772a1a73da06a2934966b49b1b194ed3a8ab7',
            'train_twc_v1_k8_icc_rate.py': '8e3619a39894422352004ec9b889636901e19fed3b93c7ea8506b1e16dedf1d0',
            'twc_v1_k8_icc_aligned_results.py': '5de1d4e5f675783760648b3140db0716cd12c4882e289c028149931f238854af',
            'utils/channel_generator.py': '645987b832f3137273fe113433640bdd4995fb2dfc54d13b13c80c91d7fe251d',
            'utils/sionna_rng.py': 'd670a482399a260b3a930286f15e14ecdbe60252fc83bbb82944c57bcf472861',
            'utils/twc_v1_k8_downlink.py': '6c1effe348a9559874febb29a9a7ef3f0470d72735248e73a172291cbaccf175',
            'utils/twc_v1_k8_icc_aligned_physics.py': 'b959a21d9893793ca4cbbedfea70c716eed3ac36aa1addd2c5841ac50a4f81a9',
            'utils/twc_v1_k8_icc_rate.py': '3fa626230dcb6b3aaf0f64931af1ff533973249e19f2639a22369704d3b9d550',
            'utils/twc_v1_k8_protocol.py': '68efd01714797f771aa4e165218c1eef1489438a4ec0d01e209ed099e35b58b8',
            'utils/twc_v1_k8_replayable_data.py': 'a8543f6b721b960f99a479841c4ebc8d3789bdb226de43c8b965a9e75d2ba76b'}}
RATE_CHECKPOINT_SHA256 = {'fdma': 'bbe98a72295c5b2985a1c41b59e010fc8d9c01cec830ba85cddc52c116fef96d',
 'hybrid_g4': '26494a2a76100b432f301b6073958dc20f11fbe78fefa2f43676f6e037b5ad30'}
RATE_TERMINAL_SHA256 = {'fdma': '20d1096c7a69408ac7839c5c8dbf9bee4bd6364042fb359670642da9705fab35',
 'hybrid_g4': 'ed9765864eacf19dfdb4c34c678d466cb04ee8b8604c0b293d6ca1be4adf8870'}

CORE_FILES = tuple(dict.fromkeys((*ratecfg.CORE_FILES,
    'configs/twc_v1_k8_icc_alpha_config.py', 'icc_alpha_runtime.py',
    'evaluate_twc_v1_k8_icc_alpha.py', 'run_twc_v1_k8_icc_alpha.py',
    'twc_v1_k8_icc_alpha_results.py')))


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def method_name(policy, arm):
    if policy not in POLICIES or arm not in ARMS:
        raise ValueError('Declared inference policy and physical arm required')
    return f'{policy}_{arm}'


def recipe():
    physical = ratecfg.recipe('fdma')['physical']
    return plain(dict(namespace=NAMESPACE, data_namespace=DATA_NAMESPACE, seed=SEED,
        arms=ARMS, policies=POLICIES, methods=METHODS,
        training=False, optimizer_updates=0, trainable=[], backward=False,
        frozen=['UE encoder', 'BS decoder', 'CE', 'historical power prior and calibration'],
        source=dict(commit=SOURCE_COMMIT, root=str(SOURCE_ROOT), terminal_updates=20000,
            core=SOURCE_CORE, checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            terminal_sha256=SOURCE_TERMINAL_SHA256,
            calibration_sha256=SOURCE_CALIBRATION_SHA256, read_only=True),
        rate_source=dict(commit=RATE_COMMIT, root=str(RATE_ROOT), terminal_updates=5000,
            core=RATE_CORE, checkpoint_sha256=RATE_CHECKPOINT_SHA256,
            terminal_sha256=RATE_TERMINAL_SHA256, read_only=True),
        physical=physical, precision=ratecfg.recipe('fdma')['precision'],
        policies_description=dict(source_fixed='source20k; alpha=.03',
            source_calibrated='same source20k; one global calibration-selected alpha per arm',
            rate_fixed='existing BS-rate5000 terminal; alpha=.03'),
        fixed_alpha=FIXED_ALPHA, alpha_grid=ALPHA_GRID,
        calibration=dict(role='validation', index_range=ROLE_RANGES['calibration'],
            batches=CALIBRATION_BATCHES, samples_per_gamma=CALIBRATION_BATCHES*MICRO_BATCH,
            gamma=GAMMAS, objective='mean complete-MU sum-rate, gamma averaged within MU first',
            one_alpha_per_arm=True, per_gamma_selection=False,
            selection='max six-gamma mean; exact ties prefer fixed alpha, then closest log distance, then lower alpha',
            lock_before_final=True, final_inputs_may_select=False),
        final=dict(role='validation', index_range=ROLE_RANGES['final'],
            batches=FINAL_BATCHES, samples_per_gamma=FINAL_BATCHES*MICRO_BATCH,
            gamma=GAMMAS, methods=METHODS, alpha_locked=True,
            DL_SNR=DL_SNR, per_tone_total_power=1., test_used=False,
            common_actual_channel_noise_and_pilots=True),
        microbatch=MICRO_BATCH, new_CE_updates=0, new_UE_updates=0,
        new_BS_updates=0, new_EMA_updates=0,
        current_frame_power_receiver_input=False,
        reporting=dict(primary_metric=PRIMARY_METRIC, primary_scope=PRIMARY_SCOPE,
            primary_comparisons=[f'{arm}_rate_fixed_minus_source_calibrated' for arm in ARMS],
            interaction_comparison='interaction_rate_gain_minus_calibration_gain',
            preserve_all_gamma=True, preserve_negative_results=True,
            paired_unit='complete8UE MU', batch_interval=MICRO_BATCH,
            within_MU_gamma_average=True, training_seed_replications=1,
            retain_all_user_rates=True, retain_min_user_rate=True,
            interpretation='A scalar-alpha control does not isolate rate loss from extra training; no direct-W claim')))


def fingerprint():
    return digest(recipe())


def noise_seed(role, index, kind='noise'):
    if role not in ROLE_RANGES or type(index) is not int or index < 0:
        raise ValueError('Declared calibration/final role and nonnegative noise address required')
    if kind not in ('noise', 'gamma', 'polarity'):
        raise ValueError('Declared independent random stream required')
    raw = f'{DATA_NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)
