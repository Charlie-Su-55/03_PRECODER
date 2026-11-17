"""Final Hybrid-G4 training and paper evaluation (FDMA frozen, zero FDMA updates).

One continuation from the latest low-SNR interference Hybrid terminal combines:
  * the independently trained pilot-only CE nmse65000 with its locked error table;
  * a public-gamma conditioner at the UE (zero-initialized FiLM on Swin tokens and
    a per-slot transmit gain, so step 0 reproduces the source encoder exactly);
  * a historical power prior kept separately per feedback-gamma node;
  * cosine learning-rate decay and an exponential moving average of weights.
The evaluated model is the fixed terminal EMA, never a monitor-selected checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = DATA_NAMESPACE = 'twc_v1_k8_final_hybrid_v1'
OUTPUT_ROOT = ROOT/'runs'/NAMESPACE
SEED = 20261201
TRAIN_ARM = 'hybrid_g4'
GAMMAS = (0, 5, 10, 15, 20, 25)
MID_GAMMAS = (2.5, 7.5, 12.5, 17.5, 22.5)

UPDATES, MICRO_BATCH, ACCUMULATION, SAVE_EVERY = 20000, 4, 8, 200
LR_BASE, LR_NEW, WARMUP, MIN_FACTOR = 1e-5, 1e-4, 200, .1
WEIGHT_DECAY, GRAD_CLIP, WEIGHT_EMA_DECAY = 1e-4, 1., .999
TRAJECTORY_OBJECTIVE = 'low_snr_interference'  # -rate +100*direction +1*low-gamma DL leakage
ALPHA, DL_SNR = .03, 25.
COND_HIDDEN = 32

SOURCE_TRAJECTORY = 'low_snr_interference'
CE_TRAJECTORY = 'nmse'
CE_CALIBRATION = ROOT/'runs'/'twc_v1_k8_ce_refine_v1'/'reports'/'attempt_9e9d46c5e8f7'/'calibration.json'
CE_CALIBRATION_SHA256 = 'e1d381ea461231ce31a4cd92db221a367c2e9053cc955341d128ee85628e8058'
CE_CALIBRATION_VARIANT = 'nmse65000'

# Validation role only; three disjoint index ranges. Test remains sealed.
SELECTION_BATCHES, FINAL_BATCHES, MID_BATCHES = 64, 256, 64
SELECTION_FIRST, FINAL_FIRST, MID_FIRST = 0, 1024, 2048

CORE_FILES = ('configs/twc_v1_k8_final_config.py', 'final_runtime.py',
    'train_twc_v1_k8_final.py', 'evaluate_twc_v1_k8_final.py', 'twc_v1_k8_final_results.py')


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def gamma_schedule(index):
    """75% low (0/5/10, 8 MU each) and 25% rotating high nodes per 32-MU update."""
    if type(index) is not int or not 1 <= index <= UPDATES*ACCUMULATION:
        raise ValueError('Declared training micro index required')
    update, micro = (index-1)//ACCUMULATION+1, (index-1) % ACCUMULATION
    result = []
    for within in range(MICRO_BATCH):
        q = MICRO_BATCH*micro+within
        result.append(GAMMAS[q % 3] if q < 24 else GAMMAS[3+(update-1+q-24) % 3])
    return tuple(result)


def micro_index(update, micro):
    if not 1 <= update <= UPDATES or not 0 <= micro < ACCUMULATION:
        raise ValueError('Declared update and accumulation index required')
    return (update-1)*ACCUMULATION+micro+1


def lr_factor(step):
    """Multiplier for the LR after ``step`` completed updates (LambdaLR convention)."""
    import math
    if step < WARMUP:
        return (step+1)/WARMUP
    progress = min(1., (step-WARMUP)/max(1, UPDATES-WARMUP))
    return MIN_FACTOR+(1-MIN_FACTOR)*.5*(1+math.cos(math.pi*progress))


def noise_seed(role, index, kind):
    if role not in ('codec_train', 'selection', 'final', 'midpoint') or kind not in ('noise', 'polarity'):
        raise ValueError('Declared random stream required')
    raw = f'{NAMESPACE}:{SEED}:{role}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def recipe():
    return plain(dict(namespace=NAMESPACE, seed=SEED, arm=TRAIN_ARM, updates=UPDATES,
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION, effective_batch=MICRO_BATCH*ACCUMULATION,
        optimizer=dict(name='AdamW', lr_base=LR_BASE, lr_new_modules=LR_NEW, warmup=WARMUP,
            schedule='linear warmup then cosine to MIN_FACTOR', min_factor=MIN_FACTOR,
            weight_decay=WEIGHT_DECAY, clip=GRAD_CLIP, fresh=True),
        weight_ema=WEIGHT_EMA_DECAY, evaluated='terminal weight EMA',
        objective=dict(reuse='twc_v1_k8_low_snr.objective', trajectory=TRAJECTORY_OBJECTIVE),
        source=dict(hybrid=SOURCE_TRAJECTORY, ce=CE_TRAJECTORY,
            ce_calibration_sha256=CE_CALIBRATION_SHA256, ce_calibration_variant=CE_CALIBRATION_VARIANT),
        gamma=dict(nodes=GAMMAS, low_fraction=.75, UE_gamma_input=True, per_node_power_prior=True),
        new_modules=dict(film_tokens=True, per_slot_gain=True, hidden=COND_HIDDEN, zero_initialized=True),
        physical='unchanged K8/Nt32/Nsc256, 64 pilot +192 data RE, Ep16+Ed48, RZF alpha .03, DL 25 dB',
        FDMA_updates=0, CE_updates=0, test_used=False,
        evaluation=dict(role='validation', selection=[SELECTION_FIRST, SELECTION_BATCHES],
            final=[FINAL_FIRST, FINAL_BATCHES], midpoint=[MID_FIRST, MID_BATCHES], mid_gammas=MID_GAMMAS)))


def fingerprint():
    return digest(recipe())
