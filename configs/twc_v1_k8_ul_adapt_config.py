"""Matched-budget BS-only adaptation to original and locked-moment receivers.

The six-node training schedule is exactly balanced every three updates.
Previous sources, model weights, receiver prior and fitted CE calibration
remain read-only; only each trajectory's decoder is trainable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs import twc_v1_k8_ul_calibration_config as previous
from configs import twc_v1_k8_icc_rate_config as ratecfg

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_ul_adapt_v1'
DATA_NAMESPACE = NAMESPACE
OUTPUT_ROOT = ROOT / 'runs' / NAMESPACE
ARMS = ('fdma', 'hybrid_g4')
TRAIN_METHODS = ('original', 'moment_receiver')
METHODS = TRAIN_METHODS
PHASES = ('source', 'final')
GAMMAS = (0, 5, 10, 15, 20, 25)
SEED = 20261124
UPDATES = 10000
LR = LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.
MICRO_BATCH = 4
ACCUMULATION = 8
SAVE_EVERY = 100
MONITOR_STEPS = (0, 100, 1000, 5000, 10000)
MONITOR_BATCHES = 32
FINAL_BATCHES = 256
ALPHA = FIXED_ALPHA = .03
DL_SNR = 25.
ROLE_RANGES = dict(codec_train=[1, UPDATES*ACCUMULATION],
    monitor=[0, MONITOR_BATCHES-1], final=[2048, 2048+FINAL_BATCHES-1])
CALIBRATION_DIR = previous.OUTPUT_ROOT/'calibration'/'attempt_a0dbcecec7b3'
CALIBRATION_MODEL_SHA256 = 'd0477d049f5808d9f9a34c0792767edc5fcdcbe8aa89951b5ff5a91b6d243be1'
CALIBRATION_MANIFEST_SHA256 = '0e86e2a78fcaca74438d194c1c4895a273c3316eb40f65ea832fbc74b742fe48'
MODEL_SHA256 = CALIBRATION_MODEL_SHA256
MANIFEST_SHA256 = CALIBRATION_MANIFEST_SHA256
SOURCE_KIND = previous.SOURCE_KIND
SOURCE_ROOT = previous.SOURCE_ROOT
SOURCE_COMMIT = previous.SOURCE_COMMIT
SOURCE_CORE = previous.SOURCE_CORE
SOURCE_CHECKPOINT_SHA256 = previous.SOURCE_CHECKPOINT_SHA256
SOURCE_TERMINAL_SHA256 = previous.SOURCE_TERMINAL_SHA256
ORIGINAL_SOURCE_ROOT = previous.ORIGINAL_SOURCE_ROOT
SOURCE_CALIBRATION_SHA256 = previous.SOURCE_CALIBRATION_SHA256
PINNED_CALIBRATION_COMMIT = 'ba43b34716b2bb263b5d9c9e8e237d545b70a1f8'
LEGACY_CORE_FILES = previous.CORE_FILES
REPOSITORY = 'Charlie-Su-55/03_PRECODER'
REMOTE_URL = 'https://github.com/' + REPOSITORY + '.git'
RESULT_BRANCH_PREFIX = 'results/' + NAMESPACE + '/'
RESULTS_BRANCH_PREFIX = RESULT_BRANCH_PREFIX

# Literal bytes of the completed zero-shot calibration implementation.
PINNED_CALIBRATION_CORE = {'schema': 'twc_v1_k8_ul_calibration_v1_core_v1',
 'sha256': {'configs/twc_v1_k8_icc_aligned_config.py': '4ecc52b2f051e65f13881b278b661d5ae198e9c3b294c7c69848d5f634ead9e0',
            'configs/twc_v1_k8_icc_alpha_config.py': '0c2f5e1b8c80473d594d4d6b5e9bb5a96d7170cefe2dccd7e864bc72f769cbf9',
            'configs/twc_v1_k8_icc_rate_config.py': '499c4002cea38ca506a23a89604f7d3c0078de007f43818490031a227f8813e2',
            'configs/twc_v1_k8_ul_calibration_config.py': '8c2cb1413b5553be7ac33ecb55d4282f3d9ad3b0437d1c21fc5442d2c8d734d6',
            'configs/twc_v1_k8_ul_diagnostics_config.py': '7c40e225ff6755f2fc191e27acd2f3d7073b7b2e0f4b24852ff568388b1e5d5e',
            'configs/twc_v1_k8_ul_propagation_config.py': '6460ffa08560553c84a29d9599d6361d5ee0d1fab3a5dd20548e55326c24698b',
            'configs/twc_v1_k8_w_residual_config.py': '9bc2e2953ce1ee263dc2038d2186399cee6842fa531f2398ec7a6690b6c20932',
            'evaluate_twc_v1_k8_icc_aligned.py': '4ec956d385521a44e56dcaf74e056324c0ab9f13b6e4dcca2e1e9a3754ed4757',
            'evaluate_twc_v1_k8_icc_alpha.py': '6c098135c6c47120941ff7cd3b1888da228e5d5b763bfeadd4f6b713f7a1217b',
            'evaluate_twc_v1_k8_icc_rate.py': '36e886ced4ec23e14876c3efd4c467c0d09e72d6ade3fe355c0ed8a300bf4fec',
            'evaluate_twc_v1_k8_ul_calibration.py': 'afa0af4ef25cf32c0348e9a9762bc3b044bf4cc7f246c26e20d0866fbb0bf505',
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
            'publish_twc_v1_k8_ul_calibration.py': '2ca550875edff1679dc1013ca1b7aca7bddddfd5b37f79e33d4cc61e68f0b3c9',
            'publish_twc_v1_k8_ul_propagation.py': 'd5be5da809ff8c8ba6f7c6e73569a746fed3185a0ff68239d31ae9e4ccb39eb0',
            'run_twc_v1_k8_icc_aligned.py': '1c05be2984a93cc3d874743536285427ce481164c86482b19b69e74a894ce00a',
            'run_twc_v1_k8_icc_alpha.py': 'ef14911b589e1a446b6e27cd9a910ebe4cf66fb51843f2c912521b8d0d51a914',
            'run_twc_v1_k8_icc_rate.py': 'ee49b410438f796528c354b1ed58d1484e3d51a58ac451a5ac1c7b99d24fb58f',
            'run_twc_v1_k8_ul_calibration.py': '11b3b227b2455d3e0cc094a0ff7b9b38755971df374f617f450c49a1bf3f42a4',
            'run_twc_v1_k8_ul_diagnostics.py': '032501cb3f0c0257f24cb398a9548283634a21aa209d43e52771c4ec4e8000e4',
            'run_twc_v1_k8_ul_propagation.py': '29984d610649f463e14f5d357bace6542656d07e9df3b6f75c13c803211bc423',
            'run_twc_v1_k8_w_residual.py': '74244b452dfd58470d7654b8692986de21c5083282dee6c85921d1291f897283',
            'train_twc_v1_k8_icc_aligned.py': '201f60cf980d0b6e5c6fb2978d6772a1a73da06a2934966b49b1b194ed3a8ab7',
            'train_twc_v1_k8_icc_rate.py': '8e3619a39894422352004ec9b889636901e19fed3b93c7ea8506b1e16dedf1d0',
            'train_twc_v1_k8_w_residual.py': '634347ac842d0213585cd78b9aaa7b93b0e3dd824e2094f430aff8931d4d8be6',
            'twc_v1_k8_icc_aligned_results.py': '5de1d4e5f675783760648b3140db0716cd12c4882e289c028149931f238854af',
            'twc_v1_k8_icc_alpha_results.py': '6b2ae8cc69c6a81617f99bcda298762cb2d05b50ba82069f1b52c8d8cd1a7cfb',
            'twc_v1_k8_ul_calibration.py': '9b53e526b74545a236adeced4d0a58741e19492ec9e40a1f2a3c3cdfd9264dc2',
            'twc_v1_k8_ul_calibration_metrics.py': 'd651580b731b87fc9d74cc4c119293a5e96af56153a36ba9305c2e36635b9411',
            'twc_v1_k8_ul_calibration_results.py': '8bdbdee0b12626a04072f61eca1d1ceb00747f056b45e78880ad408d352b4da1',
            'twc_v1_k8_ul_diagnostics_metrics.py': '859ba9fa5c204325cff8f2084b128ce6eaf76cc80878d5755b8b5b3113d59b65',
            'twc_v1_k8_ul_diagnostics_results.py': '13aeeb9fca4d2e84d8509929a22f01136c11090ea9d777591adacb5a43fd8f53',
            'twc_v1_k8_ul_propagation_metrics.py': '87d884e1d19a25cc1fd3ef48d956cb2529ede8b6b72293e4e5b20d7e30459de2',
            'twc_v1_k8_ul_propagation_results.py': '0bc7ed273562b5957f877bc68875e2d9211513edb9d462f373bfef292ad317b1',
            'twc_v1_k8_w_residual_results.py': '1e632904fea2bf19de1dd96b3380eb4a19c8319bbc322107a4f23c51e05343c3',
            'ul_calibration_runtime.py': 'b581f728c59429a38e5095bca393b0f8fa4b42e6544e5b1b864f637cf0c14692',
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
    'configs/twc_v1_k8_ul_adapt_config.py', 'ul_adapt_runtime.py',
    'train_twc_v1_k8_ul_adapt.py', 'evaluate_twc_v1_k8_ul_adapt.py',
    'twc_v1_k8_ul_adapt_results.py', 'run_twc_v1_k8_ul_adapt.py',
    'publish_twc_v1_k8_ul_adapt.py')))


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def key(arm, method):
    if arm not in ARMS or method not in TRAIN_METHODS:
        raise ValueError('Declared physical arm and adaptation receiver required')
    return method+'_'+arm


def method_name(method, arm):
    return key(arm, method)


def calibration_identity():
    return dict(namespace=previous.NAMESPACE, directory=str(CALIBRATION_DIR),
        model_sha256=CALIBRATION_MODEL_SHA256, manifest_sha256=CALIBRATION_MANIFEST_SHA256,
        refit=False)


def study_recipe():
    original = ratecfg.recipe('fdma')
    return plain(dict(namespace=NAMESPACE, data_namespace=DATA_NAMESPACE, seed=SEED,
        arms=ARMS, training_methods=TRAIN_METHODS, phases=PHASES,
        source=dict(kind=SOURCE_KIND, root=str(SOURCE_ROOT), commit=SOURCE_COMMIT,
            core=SOURCE_CORE, checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            terminal_sha256=SOURCE_TERMINAL_SHA256, terminal_updates=5000,
            original_CE_calibration_sha256=SOURCE_CALIBRATION_SHA256,
            read_only=True, source_optimizer_continued=False,
            source_decoder_shared_between_methods=True),
        legacy_code=dict(commit=PINNED_CALIBRATION_COMMIT, core=PINNED_CALIBRATION_CORE),
        calibration=calibration_identity(), updates_per_arm_method=UPDATES,
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION,
        effective_batch=MICRO_BATCH*ACCUMULATION,
        optimizer=dict(name='AdamW', fresh=True, lr=LR, weight_decay=WEIGHT_DECAY,
            betas=[.9, .999], eps=1e-8, clip=GRAD_CLIP, scheduler='constant LambdaLR'),
        loss='negative arithmetic mean of complete-MU downlink sum-rate',
        auxiliary_CSI_loss=False, rate_training=True, trainable='BS decoder parameters only',
        frozen=['UE encoder parameters and buffers', 'CE parameters and buffers',
                'historical receiver power prior and CE table', 'locked CE-moment calibration'],
        receiver=dict(original='unchanged original receiver',
            moment_receiver='unchanged locked estimated-only CE-moment receiver',
            calibration_refit=False, true_UL_deployment_input=False,
            current_frame_power_side_information=False,
            corrected_mean_channel_used_by_receiver_and_decoder=True),
        precoder=dict(design='original differentiable RZF', alpha=ALPHA,
            per_tone_total_power=1., trainable_alpha=False, direct_W_head=False),
        physical=original['physical'], precision=original['precision'],
        Fourier_LS=original['Fourier_LS'], CE_calibration=original['CE_calibration'],
        gamma=dict(nodes=GAMMAS, conditioning='one decoder per trajectory',
            training_schedule='GAMMAS[(4*(micro_index-1)+within_MU)%6]',
            frames_per_node_every_three_updates=16,
            fixed_prespecified_order=True, continuous_or_unseen_SNR_claim=False,
            terminal_gamma_counts=gamma_counts(UPDATES)),
        data=dict(namespace=DATA_NAMESPACE, addresses=ROLE_RANGES,
            train_role='codec_train', monitor_role='validation', final_role='validation',
            common_actual_channels_noise_pilots_gamma_across_four_trajectories=True,
            no_zero_shot_calibration_validation_samples_reused=True, test_used=False),
        monitor=dict(batches=MONITOR_BATCHES, samples=MONITOR_BATCHES*MICRO_BATCH,
            steps=MONITOR_STEPS, gamma=GAMMAS, selection=False),
        final=dict(batches=FINAL_BATCHES, samples=FINAL_BATCHES*MICRO_BATCH,
            gamma=GAMMAS, phases=PHASES, paired_source_and_all_four_terminals=True,
            generated_only_after_all_four_terminals=True, DL_SNR=DL_SNR,
            fixed_alpha=ALPHA, fixed_terminal_updates=UPDATES, test_used=False),
        reporting=dict(primary_metric='sum_rate_bps_hz', primary_scope='all_six',
            secondary_scopes=['low_0_5_10', 'point'], preserve_all_gamma=True,
            preserve_negative_results=True, checkpoint_selection=False,
            primary_contrast='per-arm final moment_receiver minus matched-budget original',
            paired_unit='complete8UE MU', batch_interval=MICRO_BATCH,
            within_MU_gamma_average=True, retain_all_user_rates=True,
            training_seed_replications=1,
            intervals='fixed trained model and calibration; seed/calibration uncertainty excluded'),
        sample_budget=dict(trajectories=len(ARMS)*len(TRAIN_METHODS),
            additional_updates_per_trajectory=UPDATES,
            additional_MU_frames_per_trajectory=UPDATES*ACCUMULATION*MICRO_BATCH,
            unique_paired_additional_MU_frames=UPDATES*ACCUMULATION*MICRO_BATCH,
            new_UE_updates=0, new_CE_updates=0, new_EMA_updates=0,
            statistical_calibration_refits=0),
        delivery=dict(format='whitelisted scalar CSV/JSON report directory; no ZIP',
            git_upload_is_explicit=True, result_branch_prefix=RESULT_BRANCH_PREFIX,
            repository=REPOSITORY, repository_is_public=True,
            deleting_branch_guarantees_history_erasure=False,
            contains_model_weights=False, contains_channel_tensors=False,
            contains_credentials_or_machine_environment=False)))


def recipe(arm, method):
    key(arm, method)
    value = study_recipe()
    value.update(arm=arm, method=method, updates=UPDATES)
    return plain(value)


def fingerprint(arm, method):
    return digest(recipe(arm, method))


def study_fingerprint():
    return digest(study_recipe())


def micro_index(update, micro):
    if (type(update) is not int or not 1 <= update <= UPDATES
            or type(micro) is not int or not 0 <= micro < ACCUMULATION):
        raise ValueError('Declared optimizer update and accumulation micro index required')
    return (update-1)*ACCUMULATION+micro+1


def gamma_schedule(index):
    if type(index) is not int or not 1 <= index <= UPDATES*ACCUMULATION:
        raise ValueError('Declared1-based training micro index required')
    return tuple(GAMMAS[(MICRO_BATCH*(index-1)+within) % len(GAMMAS)]
        for within in range(MICRO_BATCH))


def gamma_counts(step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError('Complete adaptation optimizer boundary required')
    frames = step*ACCUMULATION*MICRO_BATCH
    quotient, remainder = divmod(frames, len(GAMMAS))
    return {str(gamma): quotient+(index < remainder) for index, gamma in enumerate(GAMMAS)}


def counters(step):
    counts = gamma_counts(step)
    return dict(optimizer_updates=step, microbatches=step*ACCUMULATION,
        MU_frames=step*ACCUMULATION*MICRO_BATCH,
        UE_inputs=step*ACCUMULATION*MICRO_BATCH*8,
        next_micro_index=step*ACCUMULATION+1, accumulation_cursor=0,
        EMA_commits=0, gamma_counts=counts)


def noise_seed(role, index, kind='noise'):
    if not isinstance(role, str) or role not in ROLE_RANGES or type(index) is not int:
        raise ValueError('Declared adaptation noise role/index required')
    if role == 'codec_train':
        allowed = 1 <= index <= UPDATES*ACCUMULATION
    else:
        allowed = 0 <= index < (MONITOR_BATCHES if role == 'monitor' else FINAL_BATCHES)*len(GAMMAS)
    if not allowed or kind not in ('noise', 'polarity'):
        raise ValueError('Declared bounded noise or pilot-polarity stream required')
    raw = f'{DATA_NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def validate_resume(payload, arm, method, *, core, data_protocol, source_identity, frozen_identity):
    required = {'format', 'arm', 'method', 'step', 'recipe', 'fingerprint', 'core',
        'data_protocol', 'source_identity', 'frozen_identity', 'calibration_identity',
        'model', 'optimizer', 'scheduler', 'rng', 'counters'}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ValueError('Incomplete BS adaptation checkpoint')
    expected = dict(format=NAMESPACE, arm=arm, method=method, recipe=recipe(arm, method),
        fingerprint=fingerprint(arm, method), core=core, data_protocol=data_protocol,
        source_identity=source_identity, frozen_identity=frozen_identity,
        calibration_identity=calibration_identity())
    for name, value in expected.items():
        if digest(payload[name]) != digest(value):
            raise ValueError('Adaptation resume identity changed: '+name)
    step = payload['step']
    if type(step) is not int or not 0 <= step <= UPDATES or step % SAVE_EVERY:
        raise ValueError('Only complete saved adaptation boundaries are accepted')
    if digest(payload['counters']) != digest(counters(step)):
        raise ValueError('Adaptation gamma counts/cursor/frozen EMA boundary changed')
    scheduler = payload['scheduler']
    if (not isinstance(scheduler, dict) or type(scheduler.get('last_epoch')) is not int
            or scheduler.get('last_epoch') != step or type(scheduler.get('_step_count')) is not int
            or scheduler.get('_step_count') != step+1):
        raise ValueError('Adaptation scheduler boundary changed')
    if (not isinstance(payload['rng'], dict)
            or set(payload['rng']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}):
        raise ValueError('Complete external RNG state required')
    if (not isinstance(payload['model'], dict) or not payload['model']
            or not isinstance(payload['optimizer'], dict) or not payload['optimizer']):
        raise ValueError('Complete decoder and fresh optimizer state required')
    return step
