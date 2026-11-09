"""Independent, preregistered K8 adaptation of ICC ul_unet/round8 mechanisms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k8_icc_aligned_v1'
OUTPUT_ROOT = ROOT / 'runs' / NAMESPACE
ARMS = ('fdma', 'hybrid_g4')
GAMMAS = (0, 5, 10, 15, 20, 25)
SEED = 20261121
INITIAL_CODEC_SEED = 849301
INITIAL_CE_SEED = 859301
MICRO_BATCH = 4
ACCUMULATION = 8
CE_UPDATES = 5000
CODEC_UPDATES = 20000
SMOKE_UPDATES = 100
SAVE_EVERY = 100
MONITOR_STEPS = (100, 1000, 5000, 10000, 15000, 20000)
MONITOR_BATCHES = 32
FINAL_BATCHES = 256
CE_CALIBRATION_BATCHES = 64
FIXED_ALPHA = .03
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.
REFERENCE_COMMIT = 'b8ea3a51b051b7aae22916d99aa52d811edb0850'
DOCUMENT_REFERENCE_COMMIT = 'bfa04279dd55873a61ecdd476c0144a8025cd496'
ROLE_RANGES = dict(ce_train=[1, 40000], codec_train=[1, 160000],
    ce_calibration=[0, 63], monitor=[0, 31], alpha_calibration=[1024, 1039],
    final=[2048, 2303])
CORE_FILES = (
    'configs/twc_v1_k8_icc_aligned_config.py',
    'models/twc_v1_k8_icc_aligned.py',
    'utils/twc_v1_k8_icc_aligned_physics.py',
    'train_twc_v1_k8_icc_aligned.py',
    'evaluate_twc_v1_k8_icc_aligned.py',
    'utils/twc_v1_k8_replayable_data.py', 'utils/sionna_rng.py',
    'utils/channel_generator.py', 'utils/twc_v1_k8_protocol.py',
    'utils/twc_v1_k8_downlink.py', 'models/swin_precoder.py',
    'models/baseline_common.py', 'physical_split_finish_io.py',
    'models/twc_v1_k8_joint_decoder.py',
)
LAUNCH_FILES = ('run_twc_v1_k8_icc_aligned.py', 'twc_v1_k8_icc_aligned_results.py',
    'preflight_twc_v1_k8_icc_aligned.py')


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def recipe(component):
    if component not in ('ce',) + ARMS:
        raise ValueError('Exactly CE, FDMA or fixed G4 is allowed')
    return plain(dict(namespace=NAMESPACE, component=component, seed=SEED,
        initialization_seed=INITIAL_CE_SEED if component == 'ce' else INITIAL_CODEC_SEED,
        updates=CE_UPDATES if component == 'ce' else CODEC_UPDATES,
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION, effective_batch=32,
        physical=dict(K=8, Nt=32, Nsc=256, array=[4, 8], CDL='C',
            pilot_RE=64, pilot_per_UE=8, total_RE=256, Ep=16., Ed=48.,
            frame_energy=64., noise='N0=.25*10**(-gamma/10);E|n|^2=N0',
            fdma_symbols=24, G4_key=4, G4_shared=40, group_members=[[0,1],[2,3],[4,5],[6,7]]),
        optimizer=dict(name='AdamW', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            betas=[.9,.999], eps=1e-8, scheduler='constant LambdaLR', clip=GRAD_CLIP),
        loss='mean per-link physical UL NMSE' if component == 'ce' else 'mean per-link physical DL NMSE',
        auxiliary_loss=False, rate_training=False, precision='FP32/complex64;noAMP;noTF32',
        gamma=dict(distribution='per-MU-frame Uniform[0,25]', evaluation=GAMMAS),
        power_prior=dict(decay=.99, uniform_fraction=.05, total=48.,
            commit='once after successful effective-batch optimizer update;never forward'),
        Fourier_LS=dict(rank=8, unit_column_basis=True, ridge=.001,
            pilot_amplitude='sqrt(16/8)', Gram_diagonal=16/256,
            ridge_to_Gram=.016),
        CE_calibration=dict(nodes=GAMMAS, batches=64, frames_per_node=256,
            interpolation='nonnegative linear in gamma between nodes;noextrapolation'),
        addresses=ROLE_RANGES, data_namespace=NAMESPACE, source=REFERENCE_COMMIT,
        source_weights_loaded=False, CE_frozen_during_codec=True,
        selection='fixed last;5000CE/20000codec;no best selection',
        monitor=dict(samples=128, steps=MONITOR_STEPS, gamma=GAMMAS,
            alpha=FIXED_ALPHA, selection=False),
        final=dict(samples=1024, gamma=GAMMAS, alpha=FIXED_ALPHA,
            DL_SNR=25., per_tone_power=1., test_used=False),
        sample_budget=dict(CE_MU_frames=160000, codec_MU_frames_per_arm=640000,
            UE_inputs_per_codec=5120000, unique_paired_codec_MU_frames=640000,
            ICC_K4_MU_frames_per_codec=1280000, ICC_UE_inputs_per_codec=5120000)))


def fingerprint(component):
    return digest(recipe(component))


def micro_index(update, micro):
    if type(update) is not int or update < 1 or type(micro) is not int or not 0 <= micro < ACCUMULATION:
        raise ValueError('One-based update and micro index0..7 required')
    return (update - 1) * ACCUMULATION + micro + 1


def counters(component, step):
    total = CE_UPDATES if component == 'ce' else CODEC_UPDATES
    recipe(component)
    if type(step) is not int or not 0 <= step <= total:
        raise ValueError('Invalid complete optimizer boundary')
    return dict(optimizer_updates=step, microbatches=step * ACCUMULATION,
        MU_frames=step * ACCUMULATION * MICRO_BATCH,
        UE_inputs=step * ACCUMULATION * MICRO_BATCH * 8,
        next_micro_index=step * ACCUMULATION + 1,
        accumulation_cursor=0, EMA_commits=0 if component == 'ce' else step)


def noise_seed(role, index, kind='noise'):
    if role not in ROLE_RANGES or type(index) is not int or index < 0:
        raise ValueError('Declared independent data role and nonnegative index required')
    if kind not in ('noise', 'gamma', 'polarity'):
        raise ValueError('Declared independent random stream required')
    raw = f'{NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def validate_resume(payload, component, *, core, data_protocol, ce_identity=None):
    required = {'format','component','step','recipe','fingerprint','core','model',
        'optimizer','scheduler','rng','counters','data_protocol','ce_identity','statistics'}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ValueError('Incomplete optimizer-boundary checkpoint')
    expected = dict(format=NAMESPACE, component=component, recipe=recipe(component),
        fingerprint=fingerprint(component), core=core, data_protocol=data_protocol,
        ce_identity=ce_identity)
    for key, value in expected.items():
        if plain(payload[key]) != plain(value):
            raise ValueError('Resume identity changed: ' + key)
    step = payload['step']
    if type(step) is not int or step % SAVE_EVERY:
        raise ValueError('Only complete100-update saved boundaries are accepted')
    if payload['counters'] != counters(component, step):
        raise ValueError('Resume counts/data cursor/EMA boundary changed')
    if payload['scheduler'].get('last_epoch') != step:
        raise ValueError('Scheduler must match optimizer boundary')
    if set(payload['rng']) != {'python','numpy','torch_cpu','torch_cuda'}:
        raise ValueError('Complete external RNG state required')
    if not payload['model'] or not payload['optimizer']:
        raise ValueError('Model and optimizer state are required')
    if (component == 'ce') != (payload['statistics'] is None):
        raise ValueError('Codec EMA state required; CE has no codec EMA')
    return step
