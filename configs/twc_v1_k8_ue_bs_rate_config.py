"""Fixed-budget UE+BS rate training with archived BS-only benchmarks.

Training replays the exact completed benchmark stream. All final comparison
views use independent, freshly declared validation addresses and noise.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from configs import twc_v1_k8_ul_adapt_config as previous

ROOT=Path(__file__).resolve().parents[1]
NAMESPACE='twc_v1_k8_ue_bs_rate_v1'
OUTPUT_ROOT=ROOT/'runs'/NAMESPACE
DATA_NAMESPACE=NAMESPACE
SEED=20261125
TRAIN_DATA_NAMESPACE=previous.DATA_NAMESPACE
TRAIN_SEED=previous.SEED
OPTIMIZER_SEED=TRAIN_SEED
ARMS=('fdma','hybrid_g4')
METHODS=EVAL_METHODS=('source','bs_only','joint')
TRAIN_METHODS=('joint',)
GAMMAS=(0,5,10,15,20,25)
UPDATES=10000
MICRO_BATCH=4
ACCUMULATION=8
SAVE_EVERY=100
MONITOR_STEPS=(0,100,1000,5000,10000)
MONITOR_BATCHES=32
FINAL_BATCHES=256
ROLE_RANGES=dict(codec_train=[1,UPDATES*ACCUMULATION],monitor=[0,31],final=[4096,4351])
LR=LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.
ALPHA=FIXED_ALPHA=.03
DL_SNR=25.
SOURCE_KIND=previous.SOURCE_KIND
SOURCE_ROOT=previous.SOURCE_ROOT
SOURCE_COMMIT=previous.SOURCE_COMMIT
SOURCE_CORE=previous.SOURCE_CORE
SOURCE_CHECKPOINT_SHA256=previous.SOURCE_CHECKPOINT_SHA256
SOURCE_TERMINAL_SHA256=previous.SOURCE_TERMINAL_SHA256
SOURCE_CALIBRATION_SHA256=previous.SOURCE_CALIBRATION_SHA256
CALIBRATION_DIR=previous.CALIBRATION_DIR
CALIBRATION_MODEL_SHA256=previous.CALIBRATION_MODEL_SHA256
CALIBRATION_MANIFEST_SHA256=previous.CALIBRATION_MANIFEST_SHA256
CONTROL_ROOT=previous.OUTPUT_ROOT
CONTROL_COMMIT='2986b8502a876b7926dcb6d121194ca80cc4b115'
CONTROL_RESULT_COMMIT='391b68d08db8b2639f12686bdbbad6147fe7f34b'
REPOSITORY='Charlie-Su-55/03_PRECODER'
REMOTE_URL='https://github.com/'+REPOSITORY+'.git'
RESULT_BRANCH_PREFIX=RESULTS_BRANCH_PREFIX='results/'+NAMESPACE+'/'
LEGACY_CORE_FILES=previous.CORE_FILES

CONTROL_CORE={'schema': 'twc_v1_k8_ul_adapt_v1_core_v1',
 'sha256': {'configs/twc_v1_k8_icc_aligned_config.py': '4ecc52b2f051e65f13881b278b661d5ae198e9c3b294c7c69848d5f634ead9e0',
            'configs/twc_v1_k8_icc_alpha_config.py': '0c2f5e1b8c80473d594d4d6b5e9bb5a96d7170cefe2dccd7e864bc72f769cbf9',
            'configs/twc_v1_k8_icc_rate_config.py': '499c4002cea38ca506a23a89604f7d3c0078de007f43818490031a227f8813e2',
            'configs/twc_v1_k8_ul_adapt_config.py': '2cd2facfb0ab73aaf75d2d126ded375191f57ae088ac4a6cd8ee91ae90c3e975',
            'configs/twc_v1_k8_ul_calibration_config.py': '8c2cb1413b5553be7ac33ecb55d4282f3d9ad3b0437d1c21fc5442d2c8d734d6',
            'configs/twc_v1_k8_ul_diagnostics_config.py': '7c40e225ff6755f2fc191e27acd2f3d7073b7b2e0f4b24852ff568388b1e5d5e',
            'configs/twc_v1_k8_ul_propagation_config.py': '6460ffa08560553c84a29d9599d6361d5ee0d1fab3a5dd20548e55326c24698b',
            'configs/twc_v1_k8_w_residual_config.py': '9bc2e2953ce1ee263dc2038d2186399cee6842fa531f2398ec7a6690b6c20932',
            'evaluate_twc_v1_k8_icc_aligned.py': '4ec956d385521a44e56dcaf74e056324c0ab9f13b6e4dcca2e1e9a3754ed4757',
            'evaluate_twc_v1_k8_icc_alpha.py': '6c098135c6c47120941ff7cd3b1888da228e5d5b763bfeadd4f6b713f7a1217b',
            'evaluate_twc_v1_k8_icc_rate.py': '36e886ced4ec23e14876c3efd4c467c0d09e72d6ade3fe355c0ed8a300bf4fec',
            'evaluate_twc_v1_k8_ul_adapt.py': '769ad781d170e0334b8ad2ebbe15e88fa6b7437ce5bbce782f1b6d1243422d6a',
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
            'publish_twc_v1_k8_ul_adapt.py': '42766a571660387bcc32e06d181dee2068c19698c3d9a1eb03d90649bdf5979c',
            'publish_twc_v1_k8_ul_calibration.py': '2ca550875edff1679dc1013ca1b7aca7bddddfd5b37f79e33d4cc61e68f0b3c9',
            'publish_twc_v1_k8_ul_propagation.py': 'd5be5da809ff8c8ba6f7c6e73569a746fed3185a0ff68239d31ae9e4ccb39eb0',
            'run_twc_v1_k8_icc_aligned.py': '1c05be2984a93cc3d874743536285427ce481164c86482b19b69e74a894ce00a',
            'run_twc_v1_k8_icc_alpha.py': 'ef14911b589e1a446b6e27cd9a910ebe4cf66fb51843f2c912521b8d0d51a914',
            'run_twc_v1_k8_icc_rate.py': 'ee49b410438f796528c354b1ed58d1484e3d51a58ac451a5ac1c7b99d24fb58f',
            'run_twc_v1_k8_ul_adapt.py': '149ed0b00a18f492a4d35de26b9bc2440a3ecf71566c640934866315703ce4ef',
            'run_twc_v1_k8_ul_calibration.py': '11b3b227b2455d3e0cc094a0ff7b9b38755971df374f617f450c49a1bf3f42a4',
            'run_twc_v1_k8_ul_diagnostics.py': '032501cb3f0c0257f24cb398a9548283634a21aa209d43e52771c4ec4e8000e4',
            'run_twc_v1_k8_ul_propagation.py': '29984d610649f463e14f5d357bace6542656d07e9df3b6f75c13c803211bc423',
            'run_twc_v1_k8_w_residual.py': '74244b452dfd58470d7654b8692986de21c5083282dee6c85921d1291f897283',
            'train_twc_v1_k8_icc_aligned.py': '201f60cf980d0b6e5c6fb2978d6772a1a73da06a2934966b49b1b194ed3a8ab7',
            'train_twc_v1_k8_icc_rate.py': '8e3619a39894422352004ec9b889636901e19fed3b93c7ea8506b1e16dedf1d0',
            'train_twc_v1_k8_ul_adapt.py': '15a4af3a61eb18af2e435ca51156558bcc9dff94aae824b10627b84ab20dcaae',
            'train_twc_v1_k8_w_residual.py': '634347ac842d0213585cd78b9aaa7b93b0e3dd824e2094f430aff8931d4d8be6',
            'twc_v1_k8_icc_aligned_results.py': '5de1d4e5f675783760648b3140db0716cd12c4882e289c028149931f238854af',
            'twc_v1_k8_icc_alpha_results.py': '6b2ae8cc69c6a81617f99bcda298762cb2d05b50ba82069f1b52c8d8cd1a7cfb',
            'twc_v1_k8_ul_adapt_results.py': '618a1dcaf15a98a23ca67d4b74f3257eb6198902e21d0002a22ae0a775707ade',
            'twc_v1_k8_ul_calibration.py': '9b53e526b74545a236adeced4d0a58741e19492ec9e40a1f2a3c3cdfd9264dc2',
            'twc_v1_k8_ul_calibration_metrics.py': 'd651580b731b87fc9d74cc4c119293a5e96af56153a36ba9305c2e36635b9411',
            'twc_v1_k8_ul_calibration_results.py': '8bdbdee0b12626a04072f61eca1d1ceb00747f056b45e78880ad408d352b4da1',
            'twc_v1_k8_ul_diagnostics_metrics.py': '859ba9fa5c204325cff8f2084b128ce6eaf76cc80878d5755b8b5b3113d59b65',
            'twc_v1_k8_ul_diagnostics_results.py': '13aeeb9fca4d2e84d8509929a22f01136c11090ea9d777591adacb5a43fd8f53',
            'twc_v1_k8_ul_propagation_metrics.py': '87d884e1d19a25cc1fd3ef48d956cb2529ede8b6b72293e4e5b20d7e30459de2',
            'twc_v1_k8_ul_propagation_results.py': '0bc7ed273562b5957f877bc68875e2d9211513edb9d462f373bfef292ad317b1',
            'twc_v1_k8_w_residual_results.py': '1e632904fea2bf19de1dd96b3380eb4a19c8319bbc322107a4f23c51e05343c3',
            'ul_adapt_runtime.py': '928a46687b7ba49fe1aeaab3cee18bc7d730186d30ccfb33626ab2cb64610ec0',
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
PINNED_LEGACY_CORE=CONTROL_CORE

SOURCE_CONTROL_PINS={'fdma': {'CE_state_sha256': '661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192',
          'checkpoint_sha256': '45a970f54d43e71d26c7887246f985e650d78578666686cf70a179e302cdc51e',
          'counters': {'EMA_commits': 0,
                       'MU_frames': 320000,
                       'UE_inputs': 2560000,
                       'accumulation_cursor': 0,
                       'gamma_counts': {'0': 53334,
                                        '10': 53333,
                                        '15': 53333,
                                        '20': 53333,
                                        '25': 53333,
                                        '5': 53334},
                       'microbatches': 80000,
                       'next_micro_index': 80001,
                       'optimizer_updates': 10000},
          'decoder_state_sha256': 'e093bd8d4f015221e29ac7513bd5f848d20117b9eeebe034667153e61a51034d',
          'encoder_state_sha256': 'dc06947d52753640ee754cbaf3202ae4e22b180e02e1c47cc3afdf166a80d814',
          'frozen_identity': {'ce': '661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192',
                              'encoder': 'dc06947d52753640ee754cbaf3202ae4e22b180e02e1c47cc3afdf166a80d814',
                              'statistics': '922e88f255140a1ddd0df0524a971eb515c5cd306cadd701878956ab3f1e1e75'},
          'initialization_sha256': '5f31a01415e261881a422cba44a4754d872e727ff92feefd213673d386801a06',
          'manifest_sha256': '81a9304ac3f4791fb1bf636e6b9a2523913b2f572694608afd56b047065bdf73',
          'progress_sha256': 'b0cc4572c9f942eba32c55412a53b2e18e95d6e0ce08d9ee28ea3a4438377c06',
          'state_sha256': 'f9500dd077314df7d89a4362e48cceb5fc7eb08805f58a896ff70411dcd587ba',
          'statistics_state_sha256': '922e88f255140a1ddd0df0524a971eb515c5cd306cadd701878956ab3f1e1e75',
          'step': 10000,
          'terminal_receipt_sha256': '81a9304ac3f4791fb1bf636e6b9a2523913b2f572694608afd56b047065bdf73',
          'training_trace': {'format': 'twc_v1_k8_ul_adapt_v1_training_trace_v1',
                             'initial_sha256': '9593b72ee5a6f1da20777d5c2912bd19ca950b173742540fce61c60ed8d9d294',
                             'microbatches': 80000,
                             'sha256': '5792558920b6d1ce48b5a6b361f0d487961f6da584cf3f1904d89e4043b3d352'}},
 'hybrid_g4': {'CE_state_sha256': '661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192',
               'checkpoint_sha256': 'b7fe569757d68acebec631f860ed0d2ad8324f3c948a3feb12a9065c081e161d',
               'counters': {'EMA_commits': 0,
                            'MU_frames': 320000,
                            'UE_inputs': 2560000,
                            'accumulation_cursor': 0,
                            'gamma_counts': {'0': 53334,
                                             '10': 53333,
                                             '15': 53333,
                                             '20': 53333,
                                             '25': 53333,
                                             '5': 53334},
                            'microbatches': 80000,
                            'next_micro_index': 80001,
                            'optimizer_updates': 10000},
               'decoder_state_sha256': '3e92ebe9ca24342ff4bfbbc06ca4a0c28d4617fb4b2b34ba94ec5c38d88890fd',
               'encoder_state_sha256': '8917a842088fc40b3b194c21eda6c5a4add42f907d25450b3c6a0ef17c07cbaf',
               'frozen_identity': {'ce': '661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192',
                                   'encoder': '8917a842088fc40b3b194c21eda6c5a4add42f907d25450b3c6a0ef17c07cbaf',
                                   'statistics': 'da687c1275a6c80d5eea2bcacb3340ec1d732be51c5bd2f47fed62267173ce57'},
               'initialization_sha256': 'e1f1234155a851bb60aee9bb10f8d945242b7ae2984d9800dc7452966c16f75b',
               'manifest_sha256': 'dccba8eec9639adab384d9241f2a1618537240fb189770fffd0f8dbd5950dc63',
               'progress_sha256': '398fa63ef74f1db454024cad294bc41c4820c04d9268587a7d0d042b0c2b9a78',
               'state_sha256': 'b6214b4d7d94bb8e138195bb589ba5563203c86c793f539accd773577aa0f1dd',
               'statistics_state_sha256': 'da687c1275a6c80d5eea2bcacb3340ec1d732be51c5bd2f47fed62267173ce57',
               'step': 10000,
               'terminal_receipt_sha256': 'dccba8eec9639adab384d9241f2a1618537240fb189770fffd0f8dbd5950dc63',
               'training_trace': {'format': 'twc_v1_k8_ul_adapt_v1_training_trace_v1',
                                  'initial_sha256': '9593b72ee5a6f1da20777d5c2912bd19ca950b173742540fce61c60ed8d9d294',
                                  'microbatches': 80000,
                                  'sha256': '5792558920b6d1ce48b5a6b361f0d487961f6da584cf3f1904d89e4043b3d352'}}}

TRAINING_TRACE_PIN={'format': 'twc_v1_k8_ul_adapt_v1_training_trace_v1',
 'initial_sha256': '9593b72ee5a6f1da20777d5c2912bd19ca950b173742540fce61c60ed8d9d294',
 'microbatches': 80000,
 'sha256': '5792558920b6d1ce48b5a6b361f0d487961f6da584cf3f1904d89e4043b3d352'}

CONTROL_FILE_SHA256S={arm:{
    'checkpoint_latest.pt':pins['checkpoint_sha256'],
    'manifest.json':pins['manifest_sha256'],
    'terminal_receipt.json':pins['terminal_receipt_sha256'],
    'initialization.json':pins['initialization_sha256'],
    'progress.jsonl':pins['progress_sha256']}
    for arm,pins in SOURCE_CONTROL_PINS.items()}
CONTROL_PINS=SOURCE_CONTROL_PINS
CORE_FILES=(*LEGACY_CORE_FILES,
    'configs/twc_v1_k8_ue_bs_rate_config.py','ue_bs_rate_runtime.py',
    'train_twc_v1_k8_ue_bs_rate.py','evaluate_twc_v1_k8_ue_bs_rate.py',
    'twc_v1_k8_ue_bs_rate_results.py','run_twc_v1_k8_ue_bs_rate.py',
    'publish_twc_v1_k8_ue_bs_rate.py')


def plain(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,
        separators=(',',':')).encode()).hexdigest()


def key(arm,method='joint'):
    if arm not in ARMS or method not in METHODS:
        raise ValueError('Declared UE-BS physical arm and method required')
    return method+'_'+arm


def control_identity(arm):
    key(arm,'bs_only')
    return plain(dict(namespace=previous.NAMESPACE,method='original',
        commit=CONTROL_COMMIT,result_commit=CONTROL_RESULT_COMMIT,
        core=CONTROL_CORE,files=CONTROL_FILE_SHA256S[arm],
        record=SOURCE_CONTROL_PINS[arm],read_only=True))


def calibration_identity():
    return previous.calibration_identity()


def gamma_schedule(index):
    if type(index) is not int or not 1<=index<=UPDATES*ACCUMULATION:
        raise ValueError('Declared joint training micro index required')
    return tuple(GAMMAS[(MICRO_BATCH*(index-1)+within)%len(GAMMAS)]
        for within in range(MICRO_BATCH))


def gamma_counts(step):
    if type(step) is not int or not 0<=step<=UPDATES:
        raise ValueError('Complete joint optimizer boundary required')
    quotient,remainder=divmod(step*ACCUMULATION*MICRO_BATCH,len(GAMMAS))
    return {str(g):quotient+(i<remainder) for i,g in enumerate(GAMMAS)}


def micro_index(update,micro):
    if (type(update) is not int or not 1<=update<=UPDATES or type(micro) is not int
            or not 0<=micro<ACCUMULATION):
        raise ValueError('Declared joint optimizer update and micro index required')
    return (update-1)*ACCUMULATION+micro+1


def counters(step,base_ema_updates=0):
    if type(base_ema_updates) is not int or base_ema_updates<0:
        raise ValueError('Explicit nonnegative inherited EMA counter required')
    value=dict(optimizer_updates=step,microbatches=step*ACCUMULATION,
        MU_frames=step*ACCUMULATION*MICRO_BATCH,UE_inputs=step*ACCUMULATION*MICRO_BATCH*8,
        next_micro_index=step*ACCUMULATION+1,accumulation_cursor=0,gamma_counts=gamma_counts(step))
    value.update(UE_updates=step,BS_updates=step,CE_updates=0,
        EMA_commits=step,base_EMA_updates=base_ema_updates,
        total_EMA_updates=base_ema_updates+step)
    return value


def noise_seed(role,index,kind='noise'):
    if role=='codec_train':
        if type(index) is not int or not 1<=index<=UPDATES*ACCUMULATION:
            raise ValueError('Declared joint training noise index required')
        return previous.noise_seed(role,index,kind)
    if (role not in ('monitor','final') or type(index) is not int
            or not 0<=index<(MONITOR_BATCHES if role=='monitor' else FINAL_BATCHES)*len(GAMMAS)
            or kind not in ('noise','polarity')):
        raise ValueError('Declared fresh bounded evaluation noise stream required')
    raw=f'{DATA_NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**63-1)


def study_recipe():
    old=previous.study_recipe()
    return plain(dict(namespace=NAMESPACE,arms=ARMS,methods=METHODS,
        training_methods=TRAIN_METHODS,updates_per_arm=UPDATES,
        source=dict(kind=SOURCE_KIND,root=str(SOURCE_ROOT),commit=SOURCE_COMMIT,
            core=SOURCE_CORE,checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            terminal_sha256=SOURCE_TERMINAL_SHA256,terminal_updates=5000,
            fresh_optimizer=True,source_optimizer_continued=False),
        archived_controls=dict(namespace=previous.NAMESPACE,commit=CONTROL_COMMIT,
            result_commit=CONTROL_RESULT_COMMIT,core=CONTROL_CORE,
            files=CONTROL_FILE_SHA256S,new_optimizer_updates=0,
            completed_BS_updates_per_arm=10000,
            historical_prior_policy='frozen; identical source UE'),
        loss='negative mean complete-MU downlink sum rate',
        auxiliary_CSI_loss=False,trainable=['UE encoder','BS decoder'],
        frozen=['CE parameters and buffers','independently calibrated CE error table'],
        prior=dict(policy='one causal historical-power EMA commit after each successful effective update',
            source_counter_preserved=True,new_commits_per_arm=UPDATES,
            decay=.99,shrink=.05,all_microbatches_use_previous_prior=True,
            monitor_or_final_commits=0,current_frame_power_side_information=False,
            benchmark_policy_differs=True,
            attribution='joint pipeline with causal prior versus archived BS-only with frozen prior; not UE-gradient-only isolation'),
        optimizer=dict(name='AdamW',fresh=True,lr=LR,weight_decay=WEIGHT_DECAY,
            betas=[.9,.999],eps=1e-8,clip=GRAD_CLIP,scheduler='constant LambdaLR',
            seed=OPTIMIZER_SEED),microbatch=MICRO_BATCH,accumulation=ACCUMULATION,
        effective_batch=MICRO_BATCH*ACCUMULATION,
        physical=old['physical'],precision=old['precision'],
        Fourier_LS=old['Fourier_LS'],CE_calibration=old['CE_calibration'],
        receiver='unchanged original estimated-UL statistical receiver',
        calibration=calibration_identity(),moment_correction_applied=False,
        precoder=dict(design='unchanged differentiable RZF',alpha=ALPHA,
            downlink_snr_db=DL_SNR,per_tone_total_power=1.,trainable_alpha=False,direct_W=False),
        architecture=dict(unchanged=True,UE_SNR_input=False,UE_UID_input=False,
            user_projection=False,UE_uses_only_own_DL_CSI=True),
        gamma=dict(nodes=GAMMAS,training_schedule='GAMMAS[(4*(micro_index-1)+within_MU)%6]',
            terminal_counts=gamma_counts(UPDATES),continuous_unseen_SNR_claim=False),
        data=dict(training_namespace=TRAIN_DATA_NAMESPACE,training_seed=TRAIN_SEED,
            training_addresses=ROLE_RANGES['codec_train'],replay_archived_control=True,
            training_trace=TRAINING_TRACE_PIN,
            evaluation_namespace=DATA_NAMESPACE,evaluation_seed=SEED,
            monitor_addresses=ROLE_RANGES['monitor'],final_addresses=ROLE_RANGES['final'],
            test_used=False),
        monitor=dict(steps=MONITOR_STEPS,batches=MONITOR_BATCHES,
            samples=MONITOR_BATCHES*MICRO_BATCH,gamma=GAMMAS,selection=False),
        final=dict(batches=FINAL_BATCHES,samples=FINAL_BATCHES*MICRO_BATCH,
            methods=METHODS,gamma=GAMMAS,fresh_inputs=True,
            after_both_joint_terminals=True,after_both_control_acceptance=True,
            fixed_terminal_updates=UPDATES,test_used=False),
        reporting=dict(primary_metric='sum_rate_bps_hz',primary_scope='all_six',
            secondary_scopes=['low_0_5_10','point'],retain_negative_results=True,
            training_seed_replications=1,paired_unit='complete8UE MU',
            within_MU_gamma_average=True,batch_interval=MICRO_BATCH,
            uncertainty='fixed models/calibration; training-seed uncertainty excluded'),
        delivery=dict(format='checked scalar CSV/JSON directory; no ZIP',
            explicit_git_upload=True,repository=REPOSITORY,
            model_weights=False,channel_tensors=False,credentials=False)))


def recipe(arm,method='joint'):
    key(arm,method)
    return plain(dict(study_recipe(),arm=arm,method=method,updates=UPDATES))


def fingerprint(arm,method='joint'):
    return digest(recipe(arm,method))


def study_fingerprint():
    return digest(study_recipe())


def validate_resume(payload,arm,method='joint',*,core,data_protocol,
        source_identity,frozen_identity,base_ema_updates):
    if method!='joint':raise ValueError('Only owned joint checkpoints can resume')
    required={'format','arm','method','step','recipe','fingerprint','core',
        'data_protocol','source_identity','frozen_identity','model','statistics',
        'optimizer','scheduler','rng','counters'}
    if not isinstance(payload,dict) or not required<=set(payload):
        raise ValueError('Complete owned joint checkpoint required')
    expected=dict(format=NAMESPACE,arm=arm,method=method,core=core,
        recipe=recipe(arm,method),fingerprint=fingerprint(arm,method),
        data_protocol=data_protocol,source_identity=source_identity,frozen_identity=frozen_identity)
    for field,value in expected.items():
        if digest(payload[field])!=digest(value):
            raise ValueError('Joint resume identity changed: '+field)
    step=payload['step']
    if type(step) is not int or not 0<=step<=UPDATES or step%SAVE_EVERY:
        raise ValueError('Complete owned saved joint boundary required')
    if payload['counters']!=counters(step,base_ema_updates):
        raise ValueError('Joint optimizer/EMA/gamma boundary changed')
    return step
