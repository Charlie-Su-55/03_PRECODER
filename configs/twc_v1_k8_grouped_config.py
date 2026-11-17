"""Group-size study: Hybrid with G=2 (4-user groups) or G=1 (all 8 users share).

Each arm is trained from scratch with one fixed recipe that already contains
the proposed components (gamma-aware UE, per-node power prior, CE nmse65000,
cosine LR, weight EMA): an NMSE warm-up followed by the low-SNR interference
rate objective. The shared CE files are loaded by SHA from
``~/precoder_shared_ce/nmse65000`` so the same run works on every machine,
independent of the repository path. Evaluation reuses the final-model data
seeds and noise streams, so G=1/2/4 and FDMA are scored on identical frames.
"""
from __future__ import annotations

from pathlib import Path

from configs import twc_v1_k8_final_config as final

ROOT = final.ROOT
ARMS = {'hybrid_g2': 2, 'hybrid_g1': 1}
NAMESPACE = 'twc_v1_k8_grouped_v1'
SEED = 20261202
CODEC_INIT_SEED = 849301          # same common initialization as the original arms

NMSE_UPDATES, OBJECTIVE_UPDATES = 5000, 25000
UPDATES = NMSE_UPDATES+OBJECTIVE_UPDATES
MICRO_BATCH, ACCUMULATION, SAVE_EVERY = 4, 8, 250
LR, WARMUP, MIN_FACTOR = 1e-4, 500, .1
WEIGHT_DECAY, GRAD_CLIP, WEIGHT_EMA_DECAY = 1e-4, 1., .999

SHARED_CE_DIR = Path.home()/'precoder_shared_ce'/'nmse65000'
SHARED_CE_SHA256 = {
    'checkpoint_latest.pt': 'a8eded5e4963e87f0488d3455361449c149e41cf2958dcbc65274ccdff54d3d4',
    'manifest.json': '151a502bd4ee96fbde6cc0bd0e8aebe48537b086ed29e385199f6e0e127ea37b',
    'calibration.json': final.CE_CALIBRATION_SHA256,
}

CORE_FILES = ('configs/twc_v1_k8_grouped_config.py', 'grouped_runtime.py', 'train_twc_v1_k8_grouped.py')


def output_root(arm):
    if arm not in ARMS:
        raise ValueError('Declared grouped arm required')
    return ROOT/'runs'/NAMESPACE/arm


def micro_index(update, micro):
    if not 1 <= update <= UPDATES or not 0 <= micro < ACCUMULATION:
        raise ValueError('Declared update and accumulation index required')
    return (update-1)*ACCUMULATION+micro+1


def gamma_schedule(index):
    """Same 75% low / 25% rotating-high node schedule as the final model."""
    update, micro = (index-1)//ACCUMULATION+1, (index-1) % ACCUMULATION
    out = []
    for within in range(MICRO_BATCH):
        q = MICRO_BATCH*micro+within
        out.append(final.GAMMAS[q % 3] if q < 24 else final.GAMMAS[3+(update-1+q-24) % 3])
    return tuple(out)


def lr_factor(step):
    import math
    if step < WARMUP:
        return (step+1)/WARMUP
    progress = min(1., (step-WARMUP)/max(1, UPDATES-WARMUP))
    return MIN_FACTOR+(1-MIN_FACTOR)*.5*(1+math.cos(math.pi*progress))


def recipe(arm):
    return final.plain(dict(namespace=NAMESPACE, arm=arm, group_count=ARMS[arm], seed=SEED,
        codec_init_seed=CODEC_INIT_SEED, updates=dict(nmse=NMSE_UPDATES, objective=OBJECTIVE_UPDATES),
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION,
        optimizer=dict(name='AdamW', lr=LR, warmup=WARMUP, schedule='cosine', min_factor=MIN_FACTOR,
            weight_decay=WEIGHT_DECAY, clip=GRAD_CLIP),
        weight_ema=WEIGHT_EMA_DECAY, objective=dict(stage1='DL NMSE', stage2=final.TRAJECTORY_OBJECTIVE),
        ce=dict(source='nmse65000', sha256=SHARED_CE_SHA256, frozen=True),
        components=dict(gamma_aware_UE=True, per_node_prior=True), FDMA_updates=0, test_used=False))


def fingerprint(arm):
    return final.digest(recipe(arm))


def noise_seed(arm, index, kind):
    import hashlib
    raw = f'{NAMESPACE}:{SEED}:{arm}:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)
