"""Independent CE-moment calibration and frozen validation of UL recovery.

Calibration labels may contain simulated truth. Deployment reads only estimated
UL, known pilots, received pilot observations, noise level and frozen history.
All neural networks and historical statistics remain read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs import twc_v1_k8_ul_propagation_config as previous
from configs import twc_v1_k8_icc_rate_config as ratecfg

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_ul_calibration_v1'
DATA_NAMESPACE = NAMESPACE
OUTPUT_ROOT = ROOT / 'runs' / NAMESPACE
ARMS = ('fdma', 'hybrid_g4')
VARIANTS = ('original', 'reliability_only', 'moment_receiver')
GAMMAS = (0, 5, 10, 15, 20, 25)
SEED = 20261123
MICRO_BATCH = 4
CALIBRATION_BATCHES = VALIDATION_BATCHES = 64
CALIBRATION_SAMPLES = CALIBRATION_BATCHES * MICRO_BATCH
VALIDATION_SAMPLES = VALIDATION_BATCHES * MICRO_BATCH
ROLE_RANGES = dict(calibration=[0, CALIBRATION_BATCHES - 1],
                   validation=[0, VALIDATION_BATCHES - 1])
CHANNEL_ROLES = dict(calibration='ce_calibration', validation='validation')
BATCHES_BY_ROLE = dict(calibration=CALIBRATION_BATCHES, validation=VALIDATION_BATCHES)
MODEL_SHRINK_MU = 16.
VARIANCE_FLOOR = 1e-12
PSD_FLOOR = 1e-4
STRATA = 3
ALPHA = FIXED_ALPHA = .03
DL_SNR = 25.
SOURCE_KIND = previous.SOURCE_KIND
SOURCE_ROOT = previous.SOURCE_ROOT
SOURCE_COMMIT = previous.SOURCE_COMMIT
SOURCE_CORE = previous.SOURCE_CORE
SOURCE_CHECKPOINT_SHA256 = previous.SOURCE_CHECKPOINT_SHA256
SOURCE_TERMINAL_SHA256 = previous.SOURCE_TERMINAL_SHA256
ORIGINAL_SOURCE_ROOT = previous.ORIGINAL_SOURCE_ROOT
SOURCE_CALIBRATION_SHA256 = previous.SOURCE_CALIBRATION_SHA256
PINNED_PROPAGATION_COMMIT = '5aa386a8169aa60ea5384f2f5d949e97fa3fde23'
LEGACY_CORE_FILES = previous.CORE_FILES
REPOSITORY = 'Charlie-Su-55/03_PRECODER'
REMOTE_URL = 'https://github.com/' + REPOSITORY + '.git'
RESULT_BRANCH_PREFIX = 'results/' + NAMESPACE + '/'
RESULTS_BRANCH_PREFIX = RESULT_BRANCH_PREFIX

# Literal hashes from main5aa after the completed, audited propagation study.
# They are not inferred from the possibly edited checkout during execution.
PINNED_PROPAGATION_CORE = {'schema': 'twc_v1_k8_ul_propagation_v1_core_v1',
 'sha256': {'configs/twc_v1_k8_icc_aligned_config.py': '4ecc52b2f051e65f13881b278b661d5ae198e9c3b294c7c69848d5f634ead9e0',
            'configs/twc_v1_k8_icc_alpha_config.py': '0c2f5e1b8c80473d594d4d6b5e9bb5a96d7170cefe2dccd7e864bc72f769cbf9',
            'configs/twc_v1_k8_icc_rate_config.py': '499c4002cea38ca506a23a89604f7d3c0078de007f43818490031a227f8813e2',
            'configs/twc_v1_k8_ul_diagnostics_config.py': '7c40e225ff6755f2fc191e27acd2f3d7073b7b2e0f4b24852ff568388b1e5d5e',
            'configs/twc_v1_k8_ul_propagation_config.py': '6460ffa08560553c84a29d9599d6361d5ee0d1fab3a5dd20548e55326c24698b',
            'configs/twc_v1_k8_w_residual_config.py': '9bc2e2953ce1ee263dc2038d2186399cee6842fa531f2398ec7a6690b6c20932',
            'evaluate_twc_v1_k8_icc_aligned.py': '4ec956d385521a44e56dcaf74e056324c0ab9f13b6e4dcca2e1e9a3754ed4757',
            'evaluate_twc_v1_k8_icc_alpha.py': '6c098135c6c47120941ff7cd3b1888da228e5d5b763bfeadd4f6b713f7a1217b',
            'evaluate_twc_v1_k8_icc_rate.py': '36e886ced4ec23e14876c3efd4c467c0d09e72d6ade3fe355c0ed8a300bf4fec',
            'evaluate_twc_v1_k8_ul_diagnostics.py': 'e9ee2a3a4edf5736e6238de1cd2a52de16bb508e221b015e4862a2fc1dc7e3f3',
            'evaluate_twc_v1_k8_ul_propagation.py': '50b691d53ddbfc086a5f86129a3a06b682bc797a792e31416bc4102c06c6aeee',
            'evaluate_twc_v1_k8_w_residual.py': 'e9e0b8644cb01e61d12a9855e911f82c4f6ba393074a94ea3fb1d40be28170a9',
            'icc_alpha_runtime.py': 'a13c98d12aa8ff28b53c8d5503482e38a632ebd5e29457529321ec48c33d7529',
            'models/baseline_common.py': '3c1c2f89a309de5c8f8119d8aad10d64efcb3f0bbd783d94efbda3a32e629398',
            'models/swin_precoder.py': '16c67415ccd55af2bb3cb6479c7d8282b3fb3582406f76dd30cb102785e1983c',
            'models/twc_v1_k8_icc_aligned.py': '50442cf3455d366045ae1485678d74d4ea14f2dfe3dccd2bde8a05925a55fcd1',
            'models/twc_v1_k8_joint_decoder.py': '20a8ef8d638cc155afe182b4325a15463aa5444ef0e296739d1e5dd375e53732',
            'models/twc_v1_k8_w_residual.py': '9984dafd903aa35089791024aa61ee1d045e62d27b76ecced4313ab17da78e5a',
            'physical_split_finish_io.py': '2df47a8d5509fa20cd504a3fcb9a4641dd471daf06dc61a6de5f242c28f1be37',
            'publish_twc_v1_k8_ul_propagation.py': 'd5be5da809ff8c8ba6f7c6e73569a746fed3185a0ff68239d31ae9e4ccb39eb0',
            'run_twc_v1_k8_icc_aligned.py': '1c05be2984a93cc3d874743536285427ce481164c86482b19b69e74a894ce00a',
            'run_twc_v1_k8_icc_alpha.py': 'ef14911b589e1a446b6e27cd9a910ebe4cf66fb51843f2c912521b8d0d51a914',
            'run_twc_v1_k8_icc_rate.py': 'ee49b410438f796528c354b1ed58d1484e3d51a58ac451a5ac1c7b99d24fb58f',
            'run_twc_v1_k8_ul_diagnostics.py': '032501cb3f0c0257f24cb398a9548283634a21aa209d43e52771c4ec4e8000e4',
            'run_twc_v1_k8_ul_propagation.py': '29984d610649f463e14f5d357bace6542656d07e9df3b6f75c13c803211bc423',
            'run_twc_v1_k8_w_residual.py': '74244b452dfd58470d7654b8692986de21c5083282dee6c85921d1291f897283',
            'train_twc_v1_k8_icc_aligned.py': '201f60cf980d0b6e5c6fb2978d6772a1a73da06a2934966b49b1b194ed3a8ab7',
            'train_twc_v1_k8_icc_rate.py': '8e3619a39894422352004ec9b889636901e19fed3b93c7ea8506b1e16dedf1d0',
            'train_twc_v1_k8_w_residual.py': '634347ac842d0213585cd78b9aaa7b93b0e3dd824e2094f430aff8931d4d8be6',
            'twc_v1_k8_icc_aligned_results.py': '5de1d4e5f675783760648b3140db0716cd12c4882e289c028149931f238854af',
            'twc_v1_k8_icc_alpha_results.py': '6b2ae8cc69c6a81617f99bcda298762cb2d05b50ba82069f1b52c8d8cd1a7cfb',
            'twc_v1_k8_ul_diagnostics_metrics.py': '859ba9fa5c204325cff8f2084b128ce6eaf76cc80878d5755b8b5b3113d59b65',
            'twc_v1_k8_ul_diagnostics_results.py': '13aeeb9fca4d2e84d8509929a22f01136c11090ea9d777591adacb5a43fd8f53',
            'twc_v1_k8_ul_propagation_metrics.py': '87d884e1d19a25cc1fd3ef48d956cb2529ede8b6b72293e4e5b20d7e30459de2',
            'twc_v1_k8_ul_propagation_results.py': '0bc7ed273562b5957f877bc68875e2d9211513edb9d462f373bfef292ad317b1',
            'twc_v1_k8_w_residual_results.py': '1e632904fea2bf19de1dd96b3380eb4a19c8319bbc322107a4f23c51e05343c3',
            'ul_diagnostics_runtime.py': 'a23e92f917c715dd98f06fc9a99166ddb2098143fc1dec79fca6ed959e0f83f7',
            'ul_propagation_runtime.py': '29d60c540e52139a4f85a44f54e0f37bf638241f18dd9495a45a6bfa5531ef98',
            'utils/channel_generator.py': '645987b832f3137273fe113433640bdd4995fb2dfc54d13b13c80c91d7fe251d',
            'utils/sionna_rng.py': 'd670a482399a260b3a930286f15e14ecdbe60252fc83bbb82944c57bcf472861',
            'utils/twc_v1_k8_downlink.py': '6c1effe348a9559874febb29a9a7ef3f0470d72735248e73a172291cbaccf175',
            'utils/twc_v1_k8_icc_aligned_physics.py': 'b959a21d9893793ca4cbbedfea70c716eed3ac36aa1addd2c5841ac50a4f81a9',
            'utils/twc_v1_k8_icc_rate.py': '3fa626230dcb6b3aaf0f64931af1ff533973249e19f2639a22369704d3b9d550',
            'utils/twc_v1_k8_protocol.py': '68efd01714797f771aa4e165218c1eef1489438a4ec0d01e209ed099e35b58b8',
            'utils/twc_v1_k8_replayable_data.py': 'a8543f6b721b960f99a479841c4ebc8d3789bdb226de43c8b965a9e75d2ba76b',
            'w_residual_runtime.py': '16c2ea430cc6407d89e55da217bc75bd4cb9bb775bdaa242401a608c17cf3539'}}

CORE_FILES = tuple(dict.fromkeys((*LEGACY_CORE_FILES,
    'configs/twc_v1_k8_ul_calibration_config.py', 'ul_calibration_runtime.py',
    'twc_v1_k8_ul_calibration.py', 'twc_v1_k8_ul_calibration_metrics.py',
    'twc_v1_k8_ul_calibration_results.py', 'evaluate_twc_v1_k8_ul_calibration.py',
    'run_twc_v1_k8_ul_calibration.py', 'publish_twc_v1_k8_ul_calibration.py')))


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def noise_seed(role, index, kind='noise'):
    if (not isinstance(role, str) or role not in BATCHES_BY_ROLE or type(index) is not int
            or not 0 <= index < BATCHES_BY_ROLE[role] * len(GAMMAS)):
        raise ValueError('Declared calibration/validation stream address required')
    if kind not in ('noise', 'polarity'):
        raise ValueError('Declared noise or pilot-polarity stream required')
    raw = f'{DATA_NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63 - 1)


def recipe():
    original = ratecfg.recipe('fdma')
    return plain(dict(namespace=NAMESPACE, data_namespace=DATA_NAMESPACE, seed=SEED,
        arms=ARMS, variants=VARIANTS, training=False, backward=False,
        optimizer_updates=0, new_UE_updates=0, new_CE_updates=0,
        new_BS_updates=0, new_EMA_updates=0, trainable=[],
        frozen=['UE encoder parameters and buffers', 'BS decoder parameters and buffers',
                'CE parameters and buffers', 'historical power prior and CE table'],
        source=dict(kind=SOURCE_KIND, root=str(SOURCE_ROOT), commit=SOURCE_COMMIT,
            terminal_updates=5000, core=SOURCE_CORE,
            checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            terminal_sha256=SOURCE_TERMINAL_SHA256,
            calibration_sha256=SOURCE_CALIBRATION_SHA256, read_only=True),
        legacy_code=dict(commit=PINNED_PROPAGATION_COMMIT, core=PINNED_PROPAGATION_CORE),
        physical=original['physical'], precision=original['precision'],
        Fourier_LS=original['Fourier_LS'], CE_calibration=original['CE_calibration'],
        calibration=dict(role='calibration', channel_role=CHANNEL_ROLES['calibration'],
            source_arm='fdma', common_CE_model_for_both_arms=True,
            labels='simulated true UL minus unchanged pilot-only CE estimate',
            statistical_fit=True, neural_or_optimizer_updates=0,
            parameter_family='common complex relative bias and parallel/perpendicular PSD moments',
            strata=STRATA, quantiles=[1/3, 2/3], quantiles_fit_on_calibration_only=True,
            feature='known-pilot-only residual mean-square divided by known N0',
            feature_is_independent_unbiased_error_measurement=False,
            pilot_fit_degrees_of_freedom=dict(observations=64, coefficients_per_BS_antenna=8*8),
            shrink_MU=MODEL_SHRINK_MU, PSD_floor=PSD_FLOOR,
            variance_floor=VARIANCE_FLOOR, hyperparameter_search=False,
            validation_used_to_fit_or_select=False, lock_before_validation=True),
        deployment=dict(inputs=['estimated UL', 'known pilots', 'received pilot observations',
                'known gamma/noise convention', 'frozen historical prior', 'locked calibration model'],
            true_UL_input=False, true_DL_input=False, true_symbols_input=False,
            realized_frame_power_input=False,
            mean_channel_used_in_all_H_dependent_operations=True),
        receiver=dict(reference='unchanged original receiver',
            reliability_only_keeps_reference_filter_and_soft=True,
            moment_receiver='estimated-only mean and parallel/perpendicular PSD covariance',
            ridge='original RX_RIDGE divided by trace(R)/Nt in whitened precision',
            isotropic_identity_limit_must_reproduce_reference=True,
            historical_nominal_prior_frozen=True,
            calibration_truth_allowed_as_labels=True, validation_truth_scoring_only=True),
        data=dict(namespace=DATA_NAMESPACE, microbatch=MICRO_BATCH, gamma=GAMMAS,
            roles=CHANNEL_ROLES, index_ranges=ROLE_RANGES,
            batches=BATCHES_BY_ROLE,
            calibration_samples=CALIBRATION_SAMPLES,
            validation_samples_per_arm_variant_gamma=VALIDATION_SAMPLES,
            common_actual_channels_noise_pilots_across_arms_variants=True,
            calibration_validation_addresses_disjoint=True,
            noise_streams_include_role=True, test_used=False),
        reporting=dict(paired_unit='complete8UE MU', batch_interval=MICRO_BATCH,
            within_MU_gamma_average=True, primary_scope='all_six',
            secondary_scopes=['low_0_5_10', 'point'], preserve_all_gamma=True,
            preserve_negative_results=True, training_seed_replications=1,
            downlink_model_forward=True, DL_SNR=DL_SNR, fixed_rzf_alpha=ALPHA,
            checkpoint_selection=False, validation_selects_model=False,
            intervals='fixed calibrated model; calibration uncertainty excluded',
            performance_upper_bound_claim=False, test_used=False),
        delivery=dict(format='whitelisted scalar CSV/JSON report directory; no ZIP',
            git_upload_is_explicit=True, result_branch_prefix=RESULT_BRANCH_PREFIX,
            repository=REPOSITORY, repository_is_public=True,
            deleting_branch_guarantees_history_erasure=False,
            contains_model_weights=False, contains_channel_tensors=False,
            contains_credentials_or_machine_environment=False)))


def fingerprint():
    return digest(recipe())


study_recipe = recipe
study_fingerprint = fingerprint
