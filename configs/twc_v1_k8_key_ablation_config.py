"""Key-count ablation: paired Hybrid with q in {0, 2, 8} dedicated key REs per UE, from scratch.

Reference arm (not retrained here): g4_joint of twc_v1_k8_decoder_cmp_v1, the paired
Hybrid with q = 4 trained from scratch with the group-size recipe. Every arm here uses
that recipe unchanged (NMSE warm-up 5k, then the final objective to 30k updates, CE
nmse65000 frozen, gamma-aware UE, per-node prior, cosine LR, weight EMA, fixed
terminal), the same codec initialization seed and the same training channel and gamma
stream as g4_joint (DATA_SEED / DATA_NAMESPACE). Only q, and with it the number of
shared REs per pair (192 - 8q)/4, differs; the arm-specific feedback-noise stream is
as in every other arm. Fixed consecutive pairs (0,1),(2,3),(4,5),(6,7). FDMA is not
trained.
"""
from __future__ import annotations

from configs.twc_v1_k8_grouped_config import *          # noqa: F401,F403  (shared recipe constants)
from configs import twc_v1_k8_final_config as final

ARMS = {'q0': 0, 'q2': 2, 'q8': 8}
GROUP_COUNT = 4                     # pairs
NAMESPACE = 'twc_v1_k8_key_ablation_v1'
TEST_NAMESPACE = 'twc_v1_k8_key_ablation_test_v1'
SEED = 20261203                     # as g4_joint (decoder comparison)
DATA_SEED, DATA_NAMESPACE = 20261203, 'twc_v1_k8_decoder_cmp_v1'   # g4_joint's training channels
REFERENCE_ARM = 'g4_joint (q=4), runs/twc_v1_k8_decoder_cmp_v1/g4_joint'
CORE_FILES = ('configs/twc_v1_k8_key_ablation_config.py', 'train_twc_v1_k8_key_ablation.py', 'keyed_runtime.py',
              'grouped_runtime.py', 'train_twc_v1_k8_grouped.py')


def output_root(arm):
    if arm not in ARMS:
        raise ValueError('Declared key-ablation arm required')
    return ROOT/'runs'/NAMESPACE/arm


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('Declared key-ablation arm required')
    q = ARMS[arm]
    return final.plain(dict(namespace=NAMESPACE, arm=arm, keys_per_ue=q, group_count=GROUP_COUNT,
        shared_per_pair=(192-8*q)//GROUP_COUNT, data_dimensions_per_ue=q+(192-8*q)//GROUP_COUNT,
        pairing='fixed consecutive', seed=SEED, codec_init_seed=CODEC_INIT_SEED,
        data=dict(seed=DATA_SEED, namespace=DATA_NAMESPACE, note='same training channels as g4_joint'),
        updates=dict(nmse=NMSE_UPDATES, objective=OBJECTIVE_UPDATES),
        microbatch=MICRO_BATCH, accumulation=ACCUMULATION,
        optimizer=dict(name='AdamW', lr=LR, warmup=WARMUP, schedule='cosine', min_factor=MIN_FACTOR,
            weight_decay=WEIGHT_DECAY, clip=GRAD_CLIP),
        weight_ema=WEIGHT_EMA_DECAY, objective=dict(stage1='DL NMSE', stage2=final.TRAJECTORY_OBJECTIVE),
        ce=dict(source='nmse65000', sha256=SHARED_CE_SHA256, frozen=True),
        components=dict(gamma_aware_UE=True, per_node_prior=True), FDMA_updates=0, test_used=False))


def fingerprint(arm):
    return final.digest(recipe(arm))
