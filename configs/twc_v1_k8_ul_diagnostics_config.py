"""Frozen-codeword, zero-update diagnostics of ICC-aligned UL recovery.

The privileged true-UL condition is an offline sensitivity diagnostic. It is
not a deployable method, an information-theoretic bound or a new training arm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs import twc_v1_k8_icc_alpha_config as alphacfg
from configs import twc_v1_k8_icc_rate_config as ratecfg
from configs import twc_v1_k8_w_residual_config as wcfg

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_ul_diagnostics_v1'
DATA_NAMESPACE = NAMESPACE
OUTPUT_ROOT = ROOT / 'runs' / NAMESPACE
ARMS = ('fdma', 'hybrid_g4')
VARIANTS = ('estimated_ul', 'true_ul_diagnostic')
CONDITIONS = VARIANTS
METHODS = tuple(f'{variant}_{arm}' for variant in VARIANTS for arm in ARMS)
GAMMAS = (0, 5, 10, 15, 20, 25)
SEED = 20261121
MICRO_BATCH = 4
BATCHES = 64
SAMPLES = MICRO_BATCH * BATCHES
ROLE_RANGES = dict(diagnostic=[0, BATCHES - 1])
CHANNEL_ROLE = 'validation'
CALIBRATION_EDGES = (0., .01, .03, .1, .3, 1., 3., 10.)
SOURCE_KIND = 'rate'
SOURCE_ROOT = alphacfg.RATE_ROOT
SOURCE_COMMIT = alphacfg.RATE_COMMIT
SOURCE_CORE = alphacfg.RATE_CORE
SOURCE_CHECKPOINT_SHA256 = alphacfg.RATE_CHECKPOINT_SHA256
SOURCE_TERMINAL_SHA256 = alphacfg.RATE_TERMINAL_SHA256
ORIGINAL_SOURCE_ROOT = ratecfg.SOURCE_ROOT
SOURCE_CALIBRATION_SHA256 = ratecfg.SOURCE_CALIBRATION_SHA256
PINNED_W_COMMIT = '8d7f0c2e2052ffbdfaa3377c1290cfe3aad317c9'
LEGACY_CORE_FILES = wcfg.CORE_FILES

# The completed W package supplied these exact 33 source hashes. The new
# diagnostic checks the old files against this fixed record, not against a
# hash freshly inferred from its own possibly edited checkout.
PINNED_W_CORE = {'schema': 'twc_v1_k8_w_residual_v1_core_v1',
 'sha256': {'configs/twc_v1_k8_icc_aligned_config.py': '4ecc52b2f051e65f13881b278b661d5ae198e9c3b294c7c69848d5f634ead9e0',
            'configs/twc_v1_k8_icc_alpha_config.py': '0c2f5e1b8c80473d594d4d6b5e9bb5a96d7170cefe2dccd7e864bc72f769cbf9',
            'configs/twc_v1_k8_icc_rate_config.py': '499c4002cea38ca506a23a89604f7d3c0078de007f43818490031a227f8813e2',
            'configs/twc_v1_k8_w_residual_config.py': '9bc2e2953ce1ee263dc2038d2186399cee6842fa531f2398ec7a6690b6c20932',
            'evaluate_twc_v1_k8_icc_aligned.py': '4ec956d385521a44e56dcaf74e056324c0ab9f13b6e4dcca2e1e9a3754ed4757',
            'evaluate_twc_v1_k8_icc_alpha.py': '6c098135c6c47120941ff7cd3b1888da228e5d5b763bfeadd4f6b713f7a1217b',
            'evaluate_twc_v1_k8_icc_rate.py': '36e886ced4ec23e14876c3efd4c467c0d09e72d6ade3fe355c0ed8a300bf4fec',
            'evaluate_twc_v1_k8_w_residual.py': 'e9e0b8644cb01e61d12a9855e911f82c4f6ba393074a94ea3fb1d40be28170a9',
            'icc_alpha_runtime.py': 'a13c98d12aa8ff28b53c8d5503482e38a632ebd5e29457529321ec48c33d7529',
            'models/baseline_common.py': '3c1c2f89a309de5c8f8119d8aad10d64efcb3f0bbd783d94efbda3a32e629398',
            'models/swin_precoder.py': '16c67415ccd55af2bb3cb6479c7d8282b3fb3582406f76dd30cb102785e1983c',
            'models/twc_v1_k8_icc_aligned.py': '50442cf3455d366045ae1485678d74d4ea14f2dfe3dccd2bde8a05925a55fcd1',
            'models/twc_v1_k8_joint_decoder.py': '20a8ef8d638cc155afe182b4325a15463aa5444ef0e296739d1e5dd375e53732',
            'models/twc_v1_k8_w_residual.py': '9984dafd903aa35089791024aa61ee1d045e62d27b76ecced4313ab17da78e5a',
            'physical_split_finish_io.py': '2df47a8d5509fa20cd504a3fcb9a4641dd471daf06dc61a6de5f242c28f1be37',
            'run_twc_v1_k8_icc_aligned.py': '1c05be2984a93cc3d874743536285427ce481164c86482b19b69e74a894ce00a',
            'run_twc_v1_k8_icc_alpha.py': 'ef14911b589e1a446b6e27cd9a910ebe4cf66fb51843f2c912521b8d0d51a914',
            'run_twc_v1_k8_icc_rate.py': 'ee49b410438f796528c354b1ed58d1484e3d51a58ac451a5ac1c7b99d24fb58f',
            'run_twc_v1_k8_w_residual.py': '74244b452dfd58470d7654b8692986de21c5083282dee6c85921d1291f897283',
            'train_twc_v1_k8_icc_aligned.py': '201f60cf980d0b6e5c6fb2978d6772a1a73da06a2934966b49b1b194ed3a8ab7',
            'train_twc_v1_k8_icc_rate.py': '8e3619a39894422352004ec9b889636901e19fed3b93c7ea8506b1e16dedf1d0',
            'train_twc_v1_k8_w_residual.py': '634347ac842d0213585cd78b9aaa7b93b0e3dd824e2094f430aff8931d4d8be6',
            'twc_v1_k8_icc_aligned_results.py': '5de1d4e5f675783760648b3140db0716cd12c4882e289c028149931f238854af',
            'twc_v1_k8_icc_alpha_results.py': '6b2ae8cc69c6a81617f99bcda298762cb2d05b50ba82069f1b52c8d8cd1a7cfb',
            'twc_v1_k8_w_residual_results.py': '1e632904fea2bf19de1dd96b3380eb4a19c8319bbc322107a4f23c51e05343c3',
            'utils/channel_generator.py': '645987b832f3137273fe113433640bdd4995fb2dfc54d13b13c80c91d7fe251d',
            'utils/sionna_rng.py': 'd670a482399a260b3a930286f15e14ecdbe60252fc83bbb82944c57bcf472861',
            'utils/twc_v1_k8_downlink.py': '6c1effe348a9559874febb29a9a7ef3f0470d72735248e73a172291cbaccf175',
            'utils/twc_v1_k8_icc_aligned_physics.py': 'b959a21d9893793ca4cbbedfea70c716eed3ac36aa1addd2c5841ac50a4f81a9',
            'utils/twc_v1_k8_icc_rate.py': '3fa626230dcb6b3aaf0f64931af1ff533973249e19f2639a22369704d3b9d550',
            'utils/twc_v1_k8_protocol.py': '68efd01714797f771aa4e165218c1eef1489438a4ec0d01e209ed099e35b58b8',
            'utils/twc_v1_k8_replayable_data.py': 'a8543f6b721b960f99a479841c4ebc8d3789bdb226de43c8b965a9e75d2ba76b',
            'w_residual_runtime.py': '16c2ea430cc6407d89e55da217bc75bd4cb9bb775bdaa242401a608c17cf3539'}}

CORE_FILES = tuple(dict.fromkeys((*LEGACY_CORE_FILES,
    'configs/twc_v1_k8_ul_diagnostics_config.py',
    'twc_v1_k8_ul_diagnostics_metrics.py', 'ul_diagnostics_runtime.py',
    'evaluate_twc_v1_k8_ul_diagnostics.py', 'run_twc_v1_k8_ul_diagnostics.py',
    'twc_v1_k8_ul_diagnostics_results.py')))


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def method_name(variant, arm):
    if variant not in VARIANTS or arm not in ARMS:
        raise ValueError('Declared UL diagnostic variant and physical arm required')
    return f'{variant}_{arm}'


def noise_seed(role, index, kind='noise'):
    if role != 'diagnostic' or type(index) is not int or index < 0:
        raise ValueError('Declared diagnostic role and nonnegative noise index required')
    if kind not in ('noise', 'polarity'):
        raise ValueError('Declared paired noise or pilot-polarity stream required')
    raw = f'{DATA_NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63 - 1)


def recipe():
    original = ratecfg.recipe('fdma')
    return plain(dict(namespace=NAMESPACE, data_namespace=DATA_NAMESPACE, seed=SEED,
        arms=ARMS, variants=VARIANTS, methods=METHODS,
        training=False, optimizer_updates=0, backward=False, trainable=[],
        new_CE_updates=0, new_UE_updates=0, new_BS_updates=0, new_EMA_updates=0,
        frozen=['UE encoder parameters and buffers', 'BS decoder parameters and buffers',
                'CE parameters and buffers', 'historical power prior and calibration'],
        source=dict(kind=SOURCE_KIND, root=str(SOURCE_ROOT), commit=SOURCE_COMMIT,
            terminal_updates=5000, core=SOURCE_CORE,
            checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            terminal_sha256=SOURCE_TERMINAL_SHA256,
            calibration_sha256=SOURCE_CALIBRATION_SHA256, read_only=True),
        legacy_code=dict(commit=PINNED_W_COMMIT, core=PINNED_W_CORE),
        physical=original['physical'], precision=original['precision'],
        Fourier_LS=original['Fourier_LS'], CE_calibration=original['CE_calibration'],
        observation=dict(one_waveform_per_arm=True, same_observation_for_both_variants=True,
            common_actual_channels_noise_and_pilots_across_arms=True,
            measured_current_frame_power_is_receiver_input=False,
            nominal_power='unchanged frozen historical prior'),
        variants_description=dict(
            estimated_ul='normal CE estimate and independently calibrated CE variance',
            true_ul_diagnostic='privileged simulator UL truth and zero CE variance; offline only'),
        data=dict(role=CHANNEL_ROLE, namespace=DATA_NAMESPACE,
            index_range=ROLE_RANGES['diagnostic'], batches=BATCHES,
            samples_per_arm_gamma=SAMPLES, microbatch=MICRO_BATCH,
            gamma=GAMMAS, test_used=False),
        reporting=dict(paired_unit='complete8UE MU', batch_interval=MICRO_BATCH,
            within_MU_gamma_average=True, preserve_all_gamma=True,
            preserve_negative_results=True, training_seed_replications=1,
            true_ul_is_offline_diagnostic=True, performance_upper_bound_claim=False,
            deployable_method_claim=False, downlink_model_forward=False,
            checkpoint_selection=False, test_used=False),
        calibration=dict(bin_lower_edges=CALIBRATION_EDGES, final_bin_upper_edge=None,
            recovery_predicted='modelled per-coordinate error variance / frozen nominal power',
            recovery_actual='squared soft-symbol error / frozen nominal power',
            CE_predicted='calibrated per-antenna complex coefficient error variance',
            CE_actual='mean-antenna squared complex coefficient error',
            reliability_is_exact_posterior=False)))


def fingerprint():
    return digest(recipe())


study_recipe = recipe
study_fingerprint = fingerprint
