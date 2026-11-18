"""Decoder comparison on the paired Hybrid layout (2 UEs per group), both trained from scratch.

  g4_joint  the 03 joint decoder (same encoder, receiver and recipe as the group-size study)
  g4_detr   a DETR-style direct decoder adapted from the 08 project

Same recipe as the group-size study (NMSE warm-up, then the final objective,
30k updates, CE nmse65000 frozen, gamma-aware UE, per-node prior, cosine LR,
weight EMA, fixed terminal). g4_joint also gives a Hybrid whose number of
updates matches the frozen FDMA (about 3e4). FDMA is not trained.
"""
from __future__ import annotations

from configs.twc_v1_k8_grouped_config import *          # noqa: F401,F403  (shared recipe constants)
from configs import twc_v1_k8_grouped_config as _g
from configs import twc_v1_k8_final_config as final

ARMS = {'g4_joint': 4, 'g4_detr': 4}
DECODERS = {'g4_joint': 'joint', 'g4_detr': 'detr'}
NAMESPACE = 'twc_v1_k8_decoder_cmp_v1'
SEED = 20261203
CORE_FILES = ('configs/twc_v1_k8_decoder_cmp_config.py', 'train_twc_v1_k8_decoder_cmp.py',
              'models/twc_v1_k8_detr_decoder.py', 'grouped_runtime.py', 'train_twc_v1_k8_grouped.py')


def output_root(arm):
    if arm not in ARMS:
        raise ValueError('Declared decoder-comparison arm required')
    return ROOT/'runs'/NAMESPACE/arm


def recipe(arm):
    return final.plain(dict(namespace=NAMESPACE, arm=arm, decoder=DECODERS[arm], group_count=ARMS[arm], seed=SEED,
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
