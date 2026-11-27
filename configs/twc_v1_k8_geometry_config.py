"""Geometry study: FDMA and the paired Hybrid trained from scratch with per-user angular offsets.

Both arms use the group-size recipe unchanged (NMSE warm-up 5k, then the final
objective to 30k updates, CE nmse65000 frozen, gamma-aware UE, per-node prior, cosine
LR, weight EMA, fixed terminal) and the same network and codec initialization; FDMA
uses the 24-RE FDMA layout, the Hybrid the paired layout (4 key + 40 shared REs per
UE, fixed consecutive pairs in training). The only change to the training data: every
codec-training MU of CDL-C draws a range D from TRAIN_BINS and gives every user an
independent angular offset, uniform in [-D, D] DFT bins per array axis, applied as a
phase ramp over the BS array to the user's DL and UL channel (angular reciprocity),
exactly as the rot<D> scenarios of the channel-shift report. Evaluation uses those
pre-registered rot scenarios on the final model's validation frames, whose offsets are
drawn from a different seed and are therefore unseen in training.

This study trains a NEW FDMA model (authorized by the user on 2026-11-27); the frozen
K8 FDMA of the main results is not loaded or changed.
"""
from __future__ import annotations

from configs.twc_v1_k8_grouped_config import *          # noqa: F401,F403  (shared recipe constants)
from configs import twc_v1_k8_final_config as final

ARMS = {'fdma_geo': 'fdma', 'hybrid_geo': 'hybrid_g4'}
NAMESPACE = 'twc_v1_k8_geometry_v1'
SEED = 20261204
TRAIN_BINS = (0, 1, 2, 4)           # per-MU range, uniform choice; offsets uniform in [-D, D] bins per axis
EVAL_ROTATIONS = (0, 1, 2, 4, 8)    # the channel-shift scenarios rot<D>
EVAL_BATCHES = 64                   # 256 MUs per scenario, first frames of the final validation role
CORE_FILES = ('configs/twc_v1_k8_geometry_config.py', 'train_twc_v1_k8_geometry.py',
              'grouped_runtime.py', 'train_twc_v1_k8_grouped.py', 'evaluate_twc_v1_k8_channel_shift.py')


def output_root(arm):
    if arm not in ARMS:
        raise ValueError('Declared geometry arm required')
    return ROOT/'runs'/NAMESPACE/arm


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('Declared geometry arm required')
    return final.plain(dict(namespace=NAMESPACE, arm=arm, layout=ARMS[arm],
        pairing='fixed consecutive (training)' if arm == 'hybrid_geo' else None, seed=SEED,
        codec_init_seed=CODEC_INIT_SEED, network='create_codec(hybrid_g4) with the arm layout geometry',
        geometry=dict(train_bins=list(TRAIN_BINS), offsets='uniform in [-D,D] DFT bins per axis, per user',
                      applied_to='DL and UL channel (same ramp)', role='codec_train only'),
        updates=dict(nmse=NMSE_UPDATES, objective=OBJECTIVE_UPDATES),
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION,
        optimizer=dict(name='AdamW', lr=LR, warmup=WARMUP, schedule='cosine', min_factor=MIN_FACTOR,
            weight_decay=WEIGHT_DECAY, clip=GRAD_CLIP),
        weight_ema=WEIGHT_EMA_DECAY, objective=dict(stage1='DL NMSE', stage2=final.TRAJECTORY_OBJECTIVE),
        ce=dict(source='nmse65000', sha256=SHARED_CE_SHA256, frozen=True),
        components=dict(gamma_aware_UE=True, per_node_prior=True),
        new_FDMA_model_updates=UPDATES if arm == 'fdma_geo' else 0, frozen_K8_FDMA_updates=0,
        test_used=False))


def fingerprint(arm):
    return final.digest(recipe(arm))
