"""K=16 user-load study: CE fine-tune, matched FDMA/Hybrid training from scratch, joint evaluation.

FDMA and Hybrid get the same recipe and the same number of updates. The K8
FDMA is untouched; this K16 FDMA is a new baseline (authorized 2026-11-19).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'twc_v1_k16_v1'
OUTPUT_ROOT = ROOT/'runs'/NAMESPACE
SEED = 20261119
CODEC_INIT_SEED = 849301
GAMMAS = (0, 5, 10, 15, 20, 25)
MID_GAMMAS = (2.5, 7.5, 12.5, 17.5, 22.5)
MICRO_BATCH, ACCUMULATION = 4, 8

# CE: fine-tuned from the K8 CE nmse65000 (per-UE network, same 133 input channels)
CE_UPDATES, CE_SAVE_EVERY, CE_LR, CE_WARMUP = 15000, 500, 1e-4, 200
CALIBRATION_BATCHES = 64
SHARED_CE_DIR = Path.home()/'precoder_shared_ce'/'nmse65000'
SHARED_CE_SHA256 = {
    'checkpoint_latest.pt': 'a8eded5e4963e87f0488d3455361449c149e41cf2958dcbc65274ccdff54d3d4',
    'manifest.json': '151a502bd4ee96fbde6cc0bd0e8aebe48537b086ed29e385199f6e0e127ea37b',
}

# Codecs: same for both arms
ARMS = ('fdma', 'hybrid')
NMSE_UPDATES, OBJECTIVE_UPDATES = 4000, 16000
UPDATES = NMSE_UPDATES+OBJECTIVE_UPDATES
SAVE_EVERY, LR, WARMUP, MIN_FACTOR = 250, 1e-4, 500, .1
WEIGHT_DECAY, GRAD_CLIP, WEIGHT_EMA_DECAY = 1e-4, 1., .999
HYBRID_TRAIN_PAIRING = True       # pairing-aware from the start (greedy least-coupled, independent sounding)

# Evaluation (validation role; test sealed)
SELECTION_FIRST, SELECTION_BATCHES = 0, 32
FINAL_FIRST, FINAL_BATCHES = 1024, 256
MID_FIRST, MID_BATCHES = 2048, 32


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def gamma_schedule(index):
    """Same 75% low / 25% rotating-high node schedule as the K8 final model."""
    update, micro = (index-1)//ACCUMULATION+1, (index-1) % ACCUMULATION
    out = []
    for within in range(MICRO_BATCH):
        q = MICRO_BATCH*micro+within
        out.append(GAMMAS[q % 3] if q < 24 else GAMMAS[3+(update-1+q-24) % 3])
    return tuple(out)


def ce_gamma(index):
    return tuple(GAMMAS[((index-1)*MICRO_BATCH+i) % len(GAMMAS)] for i in range(MICRO_BATCH))


def micro_index(update, micro):
    return (update-1)*ACCUMULATION+micro+1


def cosine(step, total, warmup, floor=MIN_FACTOR):
    if step < warmup:
        return (step+1)/warmup
    progress = min(1., (step-warmup)/max(1, total-warmup))
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))


def ce_recipe():
    return plain(dict(namespace=NAMESPACE, stage='ce', init='K8 CE nmse65000', init_sha256=SHARED_CE_SHA256,
        updates=CE_UPDATES, lr=CE_LR, warmup=CE_WARMUP, microbatch=MICRO_BATCH, accumulation=ACCUMULATION,
        loss='UL NMSE per UE', pilots_per_ue=4, ls_rank=4, calibration_batches=CALIBRATION_BATCHES, seed=SEED))


def codec_recipe(arm):
    return plain(dict(namespace=NAMESPACE, stage='codec', arm=arm, seed=SEED, codec_init_seed=CODEC_INIT_SEED,
        updates=dict(nmse=NMSE_UPDATES, objective=OBJECTIVE_UPDATES), microbatch=MICRO_BATCH,
        accumulation=ACCUMULATION, lr=LR, warmup=WARMUP, min_factor=MIN_FACTOR, weight_decay=WEIGHT_DECAY,
        clip=GRAD_CLIP, weight_ema=WEIGHT_EMA_DECAY, objective='-rate +100*direction +1*(gamma<=10) DL leakage',
        pairing_in_training=(arm == 'hybrid' and HYBRID_TRAIN_PAIRING), gamma_aware_UE=True, per_node_prior=True,
        K=16, dmax=24, test_used=False))
