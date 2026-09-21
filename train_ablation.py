#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone ablation trainer for the Hybrid FDMA-AirComp proposal.

This single file contains:
1. the K=4, Nsc=128, Dtot=128 ablation configuration,
2. the ablation-aware model implementation,
3. the complete training and validation loop,
4. checkpoint and resume handling,
5. the command-line main entry point.

It does not modify or import configs/train_config.py, main_train.py, or
models/adaptive_hybrid.py. It only uses the existing project utilities:
utils.CDLChannelGenerator_Feedback, utils.setup_gpu, and utils.layers.DFTLayer.

Default suite:
    proposal_ablation_k4_n128

Ablation variants:
    no_condition
    no_anchor_injection
    no_cross_user_aggregation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from utils.layers import DFTLayer


# =============================================================================
# Standalone configuration
# =============================================================================

Allocation = Tuple[int, int]  # (D_f per UE, D_a shared)
ModelName = Literal["proposal", "csinet_plus", "swin_cfnet"]
TrainMode = Literal["adaptive", "specialist", "baseline"]


# =============================================================================
# 1. Shared defaults
# =============================================================================

@dataclass(frozen=True)
class PhysicalDefaults:
    ANTENNAS: int = 32
    CARRIER_FREQ: float = 3.5e9
    SPEED: float = 1.0
    TOTAL_POWER: float = 1.0
    NUM_SUBBANDS: int = 4


@dataclass(frozen=True)
class ProposalModelDefaults:
    D_MODEL: int = 256
    NUM_HEADS: int = 8
    NUM_ENCODER_LAYERS: int = 4
    DROPOUT: float = 0.1
    NUM_UNFOLD_LAYERS: int = 5  # Legacy model attribute; paper wording can be GNN refinement layers.
    GNN_AGG_DIM: int = 48

    # Kept only for compatibility with older modules that still query them.
    NUM_DECODER_LAYERS: int = 4
    NUM_ITERATIONS: int = 5


@dataclass(frozen=True)
class BaselineDefaults:
    """Parameters shared by the pure-FDMA SOTA baselines."""

    # Swin-CFNet
    SWIN_NC: int = 32
    SWIN_PATCH_SIZE: int = 4
    SWIN_EMBED_DIMS: Tuple[int, ...] = (128,)
    SWIN_NUM_HEADS: Tuple[int, ...] = (4,)
    SWIN_NUM_BLOCKS: Tuple[int, ...] = (4,)
    SWIN_WINDOW_SIZE: int = 4
    SWIN_MLP_RATIO: float = 4.0

    # CsiNet+
    CSINET_NC: int = 32
    CSINET_ENC_CHANNELS: Tuple[int, ...] = (16, 8, 4, 2)
    CSINET_DEC_CHANNELS: Tuple[int, ...] = (2, 4, 8, 16)


# =============================================================================
# 2. Training recipes
# =============================================================================

@dataclass(frozen=True)
class TrainingRecipe:
    """A reusable optimization recipe.

    Step boundaries are specified as fractions and converted automatically.
    This removes the need to update TOTAL_STEPS, PHASE1_STEPS, STAGE1_STEPS,
    and STAGE2_STEPS independently.
    """

    name: str
    total_steps: int
    batch_size: int
    lr: float
    val_interval: int
    val_batch: int
    warmup_pct: float
    grad_clip: float

    phase1_pct: float
    snr_stage1_pct: float
    snr_stage2_pct: float

    stage1_snr: Tuple[float, float] = (20.0, 25.0)
    stage2_snr: Tuple[float, float] = (10.0, 25.0)
    stage3_snr: Tuple[float, float] = (0.0, 25.0)
    val_snr_list: Tuple[int, ...] = (0, 10, 20, 25)

    dir_mode: Literal["real", "abs"] = "real"
    ph1_w_dir: float = 50.0
    ph1_w_mse: float = 100.0
    ph2_w_dir: float = 5.0
    ph2_w_mse: float = 0.0

    weight_decay: float = 1e-4
    data_cache_size: int = 1 # 30
    log_interval: int = 100
    best_metric: Literal["mean_all", "snr_25"] = "mean_all"

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if self.batch_size <= 0 or self.val_batch <= 0:
            raise ValueError("batch_size and val_batch must be positive.")
        if self.val_interval <= 0 or self.log_interval <= 0:
            raise ValueError("validation and logging intervals must be positive.")
        if not 0.0 < self.warmup_pct < 1.0:
            raise ValueError("warmup_pct must lie strictly between 0 and 1.")
        if not 0.0 < self.phase1_pct < 1.0:
            raise ValueError("phase1_pct must lie strictly between 0 and 1.")
        if not 0.0 < self.snr_stage1_pct < self.snr_stage2_pct < 1.0:
            raise ValueError(
                "Require 0 < snr_stage1_pct < snr_stage2_pct < 1."
            )

    @property
    def phase1_steps(self) -> int:
        return int(round(self.total_steps * self.phase1_pct))

    @property
    def stage1_steps(self) -> int:
        return int(round(self.total_steps * self.snr_stage1_pct))

    @property
    def stage2_steps(self) -> int:
        return int(round(self.total_steps * self.snr_stage2_pct))

TRAINING_RECIPES: Dict[str, TrainingRecipe] = {
    # ============================================================
    # K = 4
    # ============================================================
    "k4_n128_standard": TrainingRecipe(
        name="k4_n128_standard",
        total_steps=50_000,
        batch_size=48,          # 已经验证过的 K4/Nsc128 设置
        lr=1e-4,
        val_interval=500,
        val_batch=64,
        warmup_pct=0.35,
        grad_clip=1.0,
        phase1_pct=0.40,        # 20k
        snr_stage1_pct=0.30,    # 15k
        snr_stage2_pct=0.70,    # 35k
        ph2_w_dir=5.0,
        ph2_w_mse=0.0,
        data_cache_size=1,
        log_interval=100,
        best_metric="mean_all",
    ),

    "k4_n256_standard": TrainingRecipe(
        name="k4_n256_standard",
        total_steps=50_000,
        batch_size=48,          # Nsc 翻倍，先从原 batch 的一半开始
        lr=1e-4,
        val_interval=500,
        val_batch=32,
        warmup_pct=0.35,
        grad_clip=1.0,
        phase1_pct=0.40,
        snr_stage1_pct=0.30,
        snr_stage2_pct=0.70,
        ph2_w_dir=5.0,
        ph2_w_mse=0.0,
        data_cache_size=1,
        log_interval=100,
        best_metric="mean_all",
    ),

    # ============================================================
    # K = 8
    # ============================================================
    "k8_n128_soft_anchor": TrainingRecipe(
        name="k8_n128_soft_anchor",
        total_steps=100_000,
        batch_size=48,          # 你之前已经成功运行过
        lr=1e-4,
        val_interval=1_000,
        val_batch=32,
        warmup_pct=0.20,
        grad_clip=0.5,
        phase1_pct=0.50,        # 50k
        snr_stage1_pct=0.30,    # 30k
        snr_stage2_pct=0.70,    # 70k
        ph2_w_dir=10.0,
        ph2_w_mse=10.0,
        data_cache_size=1,
        log_interval=100,
        best_metric="mean_all",
    ),

    "k8_n256_soft_anchor": TrainingRecipe(
        name="k8_n256_soft_anchor",
        total_steps=100_000,
        batch_size=48,          # 初始保守值，profiling 后可升到 24
        lr=1e-4,
        val_interval=1_000,
        val_batch=32,
        warmup_pct=0.20,
        grad_clip=0.5,
        phase1_pct=0.50,
        snr_stage1_pct=0.30,
        snr_stage2_pct=0.70,
        ph2_w_dir=10.0,
        ph2_w_mse=10.0,
        data_cache_size=1,
        log_interval=100,
        best_metric="mean_all",
    ),

    # ============================================================
    # K = 16
    # ============================================================
    "k16_n128_soft_anchor": TrainingRecipe(
        name="k16_n128_soft_anchor",
        total_steps=100_000,
        batch_size=48,
        lr=1e-4,
        val_interval=1_000,
        val_batch=32,
        warmup_pct=0.20,
        grad_clip=0.5,
        phase1_pct=0.50,
        snr_stage1_pct=0.30,
        snr_stage2_pct=0.70,
        ph2_w_dir=10.0,
        ph2_w_mse=10.0,
        data_cache_size=1,
        log_interval=100,
        best_metric="mean_all",
    ),

    "k16_n256_soft_anchor": TrainingRecipe(
        name="k16_n256_soft_anchor",
        total_steps=100_000,
        batch_size=32,
        lr=1e-4,
        val_interval=1_000,
        val_batch=32,
        warmup_pct=0.20,
        grad_clip=0.5,
        phase1_pct=0.50,
        snr_stage1_pct=0.30,
        snr_stage2_pct=0.70,
        ph2_w_dir=10.0,
        ph2_w_mse=10.0,
        data_cache_size=1,
        log_interval=100,
        best_metric="mean_all",
    ),
}


# =============================================================================
# 3. Allocation generation
# =============================================================================

DEFAULT_AIRCOMP_FRACTIONS: Tuple[Fraction, ...] = tuple(
    Fraction(i, 4) for i in range(5)
)


def build_allocation_grid(
    k_users: int,
    feedback_budget: int,
    num_subbands: int,
    aircomp_fractions: Sequence[Fraction] = DEFAULT_AIRCOMP_FRACTIONS,
) -> Tuple[Allocation, ...]:
    """Generate valid (D_f, D_a) pairs under K*D_f + D_a = D_tot.

    Invalid quarter-grid points are skipped automatically. This matters for
    K=16, D_tot=128, where only (8,0), (4,64), and (0,128) are compatible with
    four subbands.
    """

    if k_users <= 0 or feedback_budget <= 0 or num_subbands <= 0:
        raise ValueError("k_users, feedback_budget, and num_subbands must be positive.")

    allocations = []
    for frac in aircomp_fractions:
        if frac < 0 or frac > 1:
            raise ValueError(f"Invalid AirComp fraction: {frac}")

        d_a_fraction = Fraction(feedback_budget) * frac
        if d_a_fraction.denominator != 1:
            continue
        d_a = int(d_a_fraction)

        fdma_total = feedback_budget - d_a
        if fdma_total % k_users != 0:
            continue
        d_f = fdma_total // k_users

        if d_f % num_subbands != 0 or d_a % num_subbands != 0:
            continue
        allocations.append((d_f, d_a))

    if not allocations:
        raise ValueError(
            f"No valid allocation for K={k_users}, D_tot={feedback_budget}, "
            f"N_sb={num_subbands}."
        )

    # Preserve the user-facing order from pure FDMA to pure AirComp.
    return tuple(allocations)


def validate_allocation(
    allocation: Allocation,
    k_users: int,
    feedback_budget: int,
    num_subbands: int,
) -> None:
    d_f, d_a = allocation
    if d_f < 0 or d_a < 0:
        raise ValueError(f"Allocation must be non-negative: {allocation}")
    if k_users * d_f + d_a != feedback_budget:
        raise ValueError(
            f"Invalid allocation {allocation}: {k_users}*{d_f}+{d_a} "
            f"!= {feedback_budget}."
        )
    if d_f % num_subbands != 0 or d_a % num_subbands != 0:
        raise ValueError(
            f"Allocation {allocation} is incompatible with "
            f"NUM_SUBBANDS={num_subbands}."
        )


def allocation_tag(
    allocation: Allocation,
    k_users: int,
    feedback_budget: int,
) -> str:
    """Readable and unambiguous allocation tag for a specialist checkpoint."""

    d_f, d_a = allocation
    fdma_pct = round(100 * (k_users * d_f) / feedback_budget)
    air_pct = round(100 * d_a / feedback_budget)
    return f"f{fdma_pct:03d}_a{air_pct:03d}_df{d_f}_da{d_a}"


# =============================================================================
# 4. Experiment specification and model-compatible runtime config
# =============================================================================

@dataclass(frozen=True)
class ExperimentSpec:
    model_name: ModelName
    train_mode: TrainMode
    k_users: int
    subcarriers: int
    feedback_budget: int
    recipe_name: str
    seed: int = 42
    allocation: Allocation | None = None
    variant: str = ""


@dataclass
class TrainConfig:
    """Fully resolved configuration for one checkpoint training run.

    Uppercase properties preserve compatibility with the existing
    AdaptiveHybridPrecoder and baseline configuration access patterns.
    """

    spec: ExperimentSpec
    allocations: Tuple[Allocation, ...]
    full_allocation_grid: Tuple[Allocation, ...]
    recipe: TrainingRecipe
    physical: PhysicalDefaults = field(default_factory=PhysicalDefaults)
    proposal: ProposalModelDefaults = field(default_factory=ProposalModelDefaults)
    baselines: BaselineDefaults = field(default_factory=BaselineDefaults)
    output_root: Path = Path("runs")

    def __post_init__(self) -> None:
        if self.spec.recipe_name != self.recipe.name:
            raise ValueError("ExperimentSpec and TrainingRecipe names do not match.")
        for alloc in self.allocations:
            validate_allocation(
                alloc,
                self.spec.k_users,
                self.spec.feedback_budget,
                self.physical.NUM_SUBBANDS,
            )
        if self.spec.train_mode in {"specialist", "baseline"} and len(self.allocations) != 1:
            raise ValueError(f"{self.spec.train_mode} must contain exactly one allocation.")
        if self.spec.model_name in {"csinet_plus", "swin_cfnet"}:
            d_f, d_a = self.allocations[0]
            if d_a != 0:
                raise ValueError("Current SOTA baselines are defined for pure FDMA only.")
            if d_f <= 0:
                raise ValueError("Pure-FDMA baseline requires D_f > 0.")

    # ----- Identity and paths -----
    @property
    def MODEL_NAME(self) -> str:
        return self.spec.model_name

    @property
    def TRAIN_MODE(self) -> str:
        return self.spec.train_mode

    @property
    def SEED(self) -> int:
        return self.spec.seed

    @property
    def EXP_NAME(self) -> str:
        prefix = {
            "proposal": "hybrid_gnn",
            "csinet_plus": "csinet_plus",
            "swin_cfnet": "swin_cfnet",
        }[self.MODEL_NAME]
        base = (
            f"{prefix}_{self.TRAIN_MODE}"
            f"_K{self.K_USERS}_Nsc{self.SUBCARRIERS}_Dfb{self.D_TOT}"
        )
        if self.TRAIN_MODE == "adaptive":
            ratios = "-".join(
                allocation_tag(a, self.K_USERS, self.D_TOT).split("_df", 1)[0]
                for a in self.allocations
            )
            alloc_part = f"grid{len(self.allocations)}_{ratios}"
        else:
            alloc_part = allocation_tag(
                self.allocations[0], self.K_USERS, self.D_TOT
            )
        variant = f"_{self.spec.variant}" if self.spec.variant else ""
        return (
            f"{base}_{alloc_part}_sb{self.NUM_SUBBANDS}"
            f"_{self.recipe.name}{variant}_seed{self.SEED}"
        )

    @property
    def SAVE_DIR(self) -> str:
        return str(self.output_root / "checkpoints" / self.EXP_NAME)

    @property
    def SAVE_PATH(self) -> str:
        return str(Path(self.SAVE_DIR) / "latest.pth")

    @property
    def BEST_PATH(self) -> str:
        return str(Path(self.SAVE_DIR) / "best.pth")

    @property
    def LOG_DIR(self) -> str:
        return str(self.output_root / "logs" / self.EXP_NAME)

    @property
    def CONFIG_PATH(self) -> str:
        return str(Path(self.SAVE_DIR) / "config.json")

    @property
    def METRICS_PATH(self) -> str:
        return str(Path(self.LOG_DIR) / "metrics.jsonl")

    @property
    def STATUS_PATH(self) -> str:
        return str(Path(self.SAVE_DIR) / "status.json")

    # ----- Physical layer -----
    @property
    def K_USERS(self) -> int:
        return self.spec.k_users

    @property
    def ANTENNAS(self) -> int:
        return self.physical.ANTENNAS

    @property
    def SUBCARRIERS(self) -> int:
        return self.spec.subcarriers

    @property
    def D_TOT(self) -> int:
        return self.spec.feedback_budget

    @property
    def CARRIER_FREQ(self) -> float:
        return self.physical.CARRIER_FREQ

    @property
    def SPEED(self) -> float:
        return self.physical.SPEED

    @property
    def TOTAL_POWER(self) -> float:
        return self.physical.TOTAL_POWER

    @property
    def NUM_SUBBANDS(self) -> int:
        return self.physical.NUM_SUBBANDS

    # ----- Resource allocation -----
    @property
    def GAMMA_GRID(self) -> list[Allocation]:
        # Legacy alias used by the current AdaptiveHybridPrecoder.
        return list(self.allocations)

    @property
    def ALLOCATION_GRID(self) -> list[Allocation]:
        return list(self.allocations)

    @property
    def D_F_MAX_PER_USER(self) -> int:
        # Always derive capacity from the complete scenario grid so specialists
        # retain the same maximum-width architecture as the adaptive model.
        return max(d_f for d_f, _ in self.full_allocation_grid)

    @property
    def D_A_MAX_SHARED(self) -> int:
        return max(d_a for _, d_a in self.full_allocation_grid)

    @property
    def DIM_FDMA_PER_USER(self) -> int:
        return self.allocations[0][0]

    @property
    def DIM_AIRCOMP_SHARED(self) -> int:
        return self.allocations[0][1]

    # ----- Proposal model -----
    @property
    def D_MODEL(self) -> int:
        return self.proposal.D_MODEL

    @property
    def NUM_HEADS(self) -> int:
        return self.proposal.NUM_HEADS

    @property
    def NUM_ENCODER_LAYERS(self) -> int:
        return self.proposal.NUM_ENCODER_LAYERS

    @property
    def NUM_DECODER_LAYERS(self) -> int:
        return self.proposal.NUM_DECODER_LAYERS

    @property
    def NUM_ITERATIONS(self) -> int:
        return self.proposal.NUM_ITERATIONS

    @property
    def DROPOUT(self) -> float:
        return self.proposal.DROPOUT

    @property
    def NUM_UNFOLD_LAYERS(self) -> int:
        return self.proposal.NUM_UNFOLD_LAYERS

    @property
    def GNN_AGG_DIM(self) -> int:
        return self.proposal.GNN_AGG_DIM

    # ----- Ablation switches -----
    @property
    def ABLATION_NAME(self) -> str:
        return self.spec.variant or "full"

    @property
    def USE_CONDITION_EMBEDDING(self) -> bool:
        return self.ABLATION_NAME != "no_condition"

    @property
    def USE_ANCHOR_INJECTION(self) -> bool:
        return self.ABLATION_NAME != "no_anchor_injection"

    @property
    def USE_CROSS_USER_AGGREGATION(self) -> bool:
        return self.ABLATION_NAME != "no_cross_user_aggregation"

    # ----- Optimizer / curriculum / loss -----
    @property
    def BATCH_SIZE(self) -> int:
        return self.recipe.batch_size

    @property
    def LR(self) -> float:
        return self.recipe.lr

    @property
    def TOTAL_STEPS(self) -> int:
        return self.recipe.total_steps

    @property
    def VAL_INTERVAL(self) -> int:
        return self.recipe.val_interval

    @property
    def VAL_BATCH(self) -> int:
        return self.recipe.val_batch

    @property
    def WARMUP_PCT(self) -> float:
        return self.recipe.warmup_pct

    @property
    def GRAD_CLIP(self) -> float:
        return self.recipe.grad_clip

    @property
    def WEIGHT_DECAY(self) -> float:
        return self.recipe.weight_decay

    @property
    def DATA_CACHE_SIZE(self) -> int:
        return self.recipe.data_cache_size

    @property
    def LOG_INTERVAL(self) -> int:
        return self.recipe.log_interval

    @property
    def PHASE1_STEPS(self) -> int:
        return self.recipe.phase1_steps

    @property
    def STAGE1_STEPS(self) -> int:
        return self.recipe.stage1_steps

    @property
    def STAGE2_STEPS(self) -> int:
        return self.recipe.stage2_steps

    @property
    def STAGE1_SNR(self) -> Tuple[float, float]:
        return self.recipe.stage1_snr

    @property
    def STAGE2_SNR(self) -> Tuple[float, float]:
        return self.recipe.stage2_snr

    @property
    def STAGE3_SNR(self) -> Tuple[float, float]:
        return self.recipe.stage3_snr

    @property
    def VAL_SNR_LIST(self) -> list[int]:
        return list(self.recipe.val_snr_list)

    @property
    def DIR_MODE(self) -> str:
        return self.recipe.dir_mode

    @property
    def PH1_W_DIR(self) -> float:
        return self.recipe.ph1_w_dir

    @property
    def PH1_W_MSE(self) -> float:
        return self.recipe.ph1_w_mse

    @property
    def PH2_W_DIR(self) -> float:
        return self.recipe.ph2_w_dir

    @property
    def PH2_W_MSE(self) -> float:
        return self.recipe.ph2_w_mse

    @property
    def BEST_METRIC(self) -> str:
        return self.recipe.best_metric

    # ----- Baseline compatibility -----
    @property
    def SWIN_NC(self) -> int:
        return self.baselines.SWIN_NC

    @property
    def SWIN_PATCH_SIZE(self) -> int:
        return self.baselines.SWIN_PATCH_SIZE

    @property
    def SWIN_EMBED_DIMS(self) -> list[int]:
        return list(self.baselines.SWIN_EMBED_DIMS)

    @property
    def SWIN_NUM_HEADS(self) -> list[int]:
        return list(self.baselines.SWIN_NUM_HEADS)

    @property
    def SWIN_NUM_BLOCKS(self) -> list[int]:
        return list(self.baselines.SWIN_NUM_BLOCKS)

    @property
    def SWIN_WINDOW_SIZE(self) -> int:
        return self.baselines.SWIN_WINDOW_SIZE

    @property
    def SWIN_MLP_RATIO(self) -> float:
        return self.baselines.SWIN_MLP_RATIO

    @property
    def SWIN_CODEWORD_DIM(self) -> int:
        return self.DIM_FDMA_PER_USER * 2

    @property
    def CSINET_NC(self) -> int:
        return self.baselines.CSINET_NC

    @property
    def CSINET_ENC_CHANNELS(self) -> list[int]:
        return list(self.baselines.CSINET_ENC_CHANNELS)

    @property
    def CSINET_DEC_CHANNELS(self) -> list[int]:
        return list(self.baselines.CSINET_DEC_CHANNELS)

    def to_dict(self) -> dict:
        """JSON-serializable configuration snapshot."""

        return {
            "experiment": asdict(self.spec),
            "allocations": [list(a) for a in self.allocations],
            "full_allocation_grid": [list(a) for a in self.full_allocation_grid],
            "training_recipe": asdict(self.recipe),
            "physical": asdict(self.physical),
            "proposal_model": asdict(self.proposal),
            "baseline_model": asdict(self.baselines),
            "resolved": {
                "exp_name": self.EXP_NAME,
                "save_dir": self.SAVE_DIR,
                "phase1_steps": self.PHASE1_STEPS,
                "stage1_steps": self.STAGE1_STEPS,
                "stage2_steps": self.STAGE2_STEPS,
                "d_f_max_per_user": self.D_F_MAX_PER_USER,
                "d_a_max_shared": self.D_A_MAX_SHARED,
                "ablation_name": self.ABLATION_NAME,
                "use_condition_embedding": self.USE_CONDITION_EMBEDDING,
                "use_anchor_injection": self.USE_ANCHOR_INJECTION,
                "use_cross_user_aggregation": self.USE_CROSS_USER_AGGREGATION,
            },
        }


# =============================================================================
# 5. Suite definitions
# =============================================================================

SCENARIOS: Tuple[Tuple[int, int, int, str], ...] = (
    # (K, N_sc, D_tot, recipe_name)
    (4, 128, 128, "k4_n128_standard"),
    (4, 256, 256, "k4_n256_standard"),

    (8, 128, 128, "k8_n128_soft_anchor"),
    (8, 256, 256, "k8_n256_soft_anchor"),

    (16, 128, 128, "k16_n128_soft_anchor"),
    (16, 256, 256, "k16_n256_soft_anchor"),
)


def _proposal_adaptive_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            model_name="proposal",
            train_mode="adaptive",
            k_users=k,
            subcarriers=n_sc,
            feedback_budget=d_tot,
            recipe_name=recipe,
        )
        for k, n_sc, d_tot, recipe in SCENARIOS
    ]


def _proposal_specialist_specs() -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    physical = PhysicalDefaults()
    for k, n_sc, d_tot, recipe in SCENARIOS:
        grid = build_allocation_grid(k, d_tot, physical.NUM_SUBBANDS)
        for allocation in grid:
            specs.append(
                ExperimentSpec(
                    model_name="proposal",
                    train_mode="specialist",
                    k_users=k,
                    subcarriers=n_sc,
                    feedback_budget=d_tot,
                    recipe_name=recipe,
                    allocation=allocation,
                )
            )
    return specs


def _proposal_ablation_k4_n128_specs() -> list[ExperimentSpec]:
    """Three minimal ablations for the paper table.

    All runs use the same K=4, Nsc=128, Dtot=128 adaptive grid and the same
    training recipe as the completed full model.
    """
    variants = (
        "no_condition",
        "no_anchor_injection",
        "no_cross_user_aggregation",
    )
    return [
        ExperimentSpec(
            model_name="proposal",
            train_mode="adaptive",
            k_users=4,
            subcarriers=128,
            feedback_budget=128,
            recipe_name="k4_n128_standard",
            variant=variant,
        )
        for variant in variants
    ]


def _baseline_specs(model_name: Literal["csinet_plus", "swin_cfnet"]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for k, n_sc, d_tot, recipe in SCENARIOS:
        pure_fdma = (d_tot // k, 0)
        specs.append(
            ExperimentSpec(
                model_name=model_name,
                train_mode="baseline",
                k_users=k,
                subcarriers=n_sc,
                feedback_budget=d_tot,
                recipe_name=recipe,
                allocation=pure_fdma,
            )
        )
    return specs


EXPERIMENT_SUITES: Mapping[str, Tuple[ExperimentSpec, ...]] = {
    "proposal_adaptive": tuple(_proposal_adaptive_specs()),
    "proposal_specialists": tuple(_proposal_specialist_specs()),
    "proposal_ablation_k4_n128": tuple(_proposal_ablation_k4_n128_specs()),
    "proposal_all": tuple(_proposal_adaptive_specs() + _proposal_specialist_specs()),
    # These two suites expose the correct shared parameters and naming. Their
    # training adapters depend on the actual baseline class/forward interfaces.
    "csinet_baselines": tuple(_baseline_specs("csinet_plus")),
    "swin_baselines": tuple(_baseline_specs("swin_cfnet")),
    "paper_all_declared": tuple(
        _proposal_adaptive_specs()
        + _proposal_specialist_specs()
        + _baseline_specs("csinet_plus")
        + _baseline_specs("swin_cfnet")
    ),
}


def resolve_experiment(spec: ExperimentSpec, output_root: str | Path = "runs") -> TrainConfig:
    physical = PhysicalDefaults()
    full_grid = build_allocation_grid(
        spec.k_users,
        spec.feedback_budget,
        physical.NUM_SUBBANDS,
    )

    if spec.train_mode == "adaptive":
        allocations = full_grid
    else:
        if spec.allocation is None:
            raise ValueError(f"{spec.train_mode} experiment requires one allocation.")
        allocations = (spec.allocation,)

    try:
        recipe = TRAINING_RECIPES[spec.recipe_name]
    except KeyError as exc:
        raise KeyError(f"Unknown training recipe: {spec.recipe_name}") from exc

    return TrainConfig(
        spec=spec,
        allocations=allocations,
        full_allocation_grid=full_grid,
        recipe=recipe,
        output_root=Path(output_root),
    )


def get_suite(name: str, output_root: str | Path = "runs") -> list[TrainConfig]:
    try:
        specs = EXPERIMENT_SUITES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENT_SUITES))
        raise KeyError(f"Unknown suite '{name}'. Available suites: {choices}") from exc
    return [resolve_experiment(spec, output_root=output_root) for spec in specs]


def list_suites() -> Iterable[str]:
    return sorted(EXPERIMENT_SUITES)

# =============================================================================
# Ablation-aware proposal model
# =============================================================================

# ============================================================================
# 条件嵌入: (D_f, D_a, SNR) → [D] 向量
# ============================================================================
class ConditionEmbed(nn.Module):
    def __init__(self, d_model, d_f_max, d_a_max):
        super().__init__()
        self.d_f_max = float(max(d_f_max, 1))
        self.d_a_max = float(max(d_a_max, 1))
        self.net = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, d_f, d_a, snr_db, device):
        s = snr_db.item() if torch.is_tensor(snr_db) else float(snr_db)
        c = torch.tensor(
            [d_f / self.d_f_max, d_a / self.d_a_max, s / 25.0],
            device=device, dtype=torch.float32
        )
        return self.net(c)  # [D]


# ============================================================================
# Encoder (UE 端, 双分支, 最大维度输出 + 前缀截断)
# ============================================================================
class AdaptiveHybridEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.D = cfg.D_MODEL
        self.N_sb = cfg.NUM_SUBBANDS
        self.d_f_sb_max = cfg.D_F_MAX_PER_USER // self.N_sb
        self.d_a_sb_max = cfg.D_A_MAX_SHARED // self.N_sb
        self.ds_factor = 4

        self.dft = DFTLayer(self.Nt, dim=-2)

        # ---------------- FDMA 分支 (角域) ----------------
        self.fdma_input_proj = nn.Linear(self.Nsc * 2, self.D)
        self.fdma_input_norm = nn.LayerNorm(self.D)
        fdma_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=cfg.NUM_HEADS,
            batch_first=True, dropout=cfg.DROPOUT
        )
        self.fdma_backbone = nn.TransformerEncoder(
            fdma_layer, num_layers=cfg.NUM_ENCODER_LAYERS
        )
        self.fdma_attn_pool = nn.Sequential(
            nn.Linear(self.D, self.D // 4), nn.GELU(),
            nn.Linear(self.D // 4, 1)
        )
        self.fdma_sb_query = nn.Parameter(
            torch.randn(1, self.N_sb, self.D) * 0.02
        )
        # 固定最大输出宽度, 前缀截断
        self.fc_fdma = nn.Linear(self.D, self.d_f_sb_max * 2)

        # ---------------- AirComp 分支 (归一化方向) ----------------
        self.aircomp_input_proj = nn.Linear(self.Nt * 2, self.D)
        self.aircomp_input_norm = nn.LayerNorm(self.D)
        ac_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=cfg.NUM_HEADS,
            batch_first=True, dropout=cfg.DROPOUT
        )
        self.aircomp_backbone = nn.TransformerEncoder(
            ac_layer, num_layers=cfg.NUM_ENCODER_LAYERS
        )
        self.sc_pos_emb = nn.Parameter(
            torch.randn(1, self.Nsc, self.D) * 0.02
        )
        self.aircomp_attn_pool = nn.Sequential(
            nn.Linear(self.D, self.D // 4), nn.GELU(),
            nn.Linear(self.D // 4, 1)
        )
        self.fc_aircomp = nn.Linear(self.D, self.d_a_sb_max * 2)

    def forward(self, h_dl_flat, d_f_sb, d_a_sb, cond):
        """
        h_dl_flat: [BK, Nt, Nsc] complex
        d_f_sb / d_a_sb: 当前激活的 per-subband 维度 (int, 可为 0)
        cond: [D] 条件嵌入
        """
        BK = h_dl_flat.shape[0]
        device = h_dl_flat.device
        cond_b = cond.view(1, 1, -1)

        # ---------------- FDMA ----------------
        if d_f_sb > 0:
            h_ang = self.dft(h_dl_flat)
            x = torch.cat([h_ang.real, h_ang.imag], dim=-1)
            x = self.fdma_input_norm(self.fdma_input_proj(x)) + cond_b
            feat = self.fdma_backbone(x)                                   # [BK, Nt, D]
            feat_sb = feat.unsqueeze(1) + self.fdma_sb_query.unsqueeze(2)  # [BK, N_sb, Nt, D]
            w = torch.softmax(self.fdma_attn_pool(feat_sb), dim=2)
            pooled = (feat_sb * w).sum(dim=2)                              # [BK, N_sb, D]
            full = self.fc_fdma(pooled)                                    # [BK, N_sb, 2*d_f_sb_max]
            z = torch.complex(full[..., :self.d_f_sb_max],
                              full[..., self.d_f_sb_max:])                 # [BK, N_sb, d_f_sb_max]
            z_f = z[..., :d_f_sb].reshape(BK, self.N_sb * d_f_sb)          # 前缀截断
        else:
            z_f = torch.zeros(BK, 0, dtype=torch.cfloat, device=device)

        # ---------------- AirComp ----------------
        if d_a_sb > 0:
            eig = h_dl_flat.permute(0, 2, 1)                               # [BK, Nsc, Nt]
            eig = eig / (torch.norm(eig, dim=-1, keepdim=True) + 1e-9)
            eig = eig[:, ::self.ds_factor, :]                              # [BK, Nsc/4, Nt]
            x = torch.cat([eig.real, eig.imag], dim=-1)
            x = self.aircomp_input_norm(self.aircomp_input_proj(x))
            x = x + self.sc_pos_emb[:, ::self.ds_factor, :] + cond_b
            feat = self.aircomp_backbone(x)                                # [BK, Nsc', D]
            n_tok = feat.shape[1]
            feat_sb = feat.reshape(BK, self.N_sb, n_tok // self.N_sb, self.D)
            w = torch.softmax(self.aircomp_attn_pool(feat_sb), dim=2)
            pooled = (feat_sb * w).sum(dim=2)                              # [BK, N_sb, D]
            full = self.fc_aircomp(pooled)                                 # [BK, N_sb, 2*d_a_sb_max]
            z = torch.complex(full[..., :self.d_a_sb_max],
                              full[..., self.d_a_sb_max:])
            z_a = z[..., :d_a_sb].reshape(BK, self.N_sb * d_a_sb)          # 前缀截断
        else:
            z_a = torch.zeros(BK, 0, dtype=torch.cfloat, device=device)

        return z_f, z_a


# ============================================================================
# GNN-style Aggregation (与 V4 完全一致)
# ============================================================================
class GNN_AggregationModule(nn.Module):
    def __init__(self, d_model, d_a=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.d_a = d_a
        self.msg_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.user_query_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.pair_agg = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_a)
        )

    def forward(self, Y_ac_feat, uv_feat):
        B, N_sb, Nt, D = Y_ac_feat.shape
        K = uv_feat.shape[2]

        Y_ac_flat = Y_ac_feat.reshape(B * N_sb, Nt, D)
        uv_flat = uv_feat.reshape(B * N_sb, K, D)

        user_msg, _ = self.user_query_attn(uv_flat, Y_ac_flat, Y_ac_flat)
        msg = self.msg_proj(user_msg)

        Y_i = Y_ac_flat.unsqueeze(2).expand(B * N_sb, Nt, Nt, D)
        Y_j = Y_ac_flat.unsqueeze(1).expand(B * N_sb, Nt, Nt, D)

        msg_sum = msg.sum(dim=1, keepdim=True).unsqueeze(2)
        Y_i = Y_i + msg_sum
        Y_j = Y_j + msg_sum

        A_feat = self.pair_agg(torch.cat([Y_i, Y_j], dim=-1))
        A_feat = A_feat.reshape(B, N_sb, Nt, Nt, self.d_a)
        return A_feat


# ============================================================================
# Unfolded WMMSE Layer (与 V4 完全一致)
# ============================================================================
class UnfoldedWMMSELayer(nn.Module):
    def __init__(self, cfg, d_model, d_a=64, n_heads_solver=8, n_heads_cross=4):
        super().__init__()
        self.K = cfg.K_USERS
        self.Nt = cfg.ANTENNAS
        self.N_sb = cfg.NUM_SUBBANDS
        self.d_model = d_model
        self.d_a = d_a
        self.n_heads = n_heads_solver
        self.use_cross_user_aggregation = bool(
            getattr(cfg, "USE_CROSS_USER_AGGREGATION", True)
        )

        if self.use_cross_user_aggregation:
            self.uv_estimator = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            self.A_estimator = GNN_AggregationModule(
                d_model, d_a=d_a, n_heads=4
            )
            self.A_to_bias = nn.Linear(d_a, self.n_heads)
            self.cross_user_attn = nn.MultiheadAttention(
                d_model, num_heads=n_heads_cross, batch_first=True
            )
        else:
            self.uv_estimator = None
            self.A_estimator = None
            self.A_to_bias = None
            self.cross_user_attn = None

        self.w_solver_attn = nn.MultiheadAttention(
            d_model, num_heads=self.n_heads, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, W_feat, Y_f_feat, Y_ac_feat):
        B, K, Nt, N_sb, D = W_feat.shape
        H = self.n_heads

        if self.use_cross_user_aggregation:
            W_pool = W_feat.mean(dim=2)
            Yf_pool = Y_f_feat.mean(dim=2)
            uv_input = torch.cat([W_pool, Yf_pool], dim=-1)
            uv_feat = self.uv_estimator(uv_input)
            uv_feat = uv_feat.permute(0, 2, 1, 3).contiguous()

            Y_ac_perm = Y_ac_feat.permute(0, 2, 1, 3).contiguous()
            A_feat = self.A_estimator(Y_ac_perm, uv_feat)
            attn_bias = self.A_to_bias(A_feat)
            attn_bias = attn_bias.permute(0, 1, 4, 2, 3).contiguous()

            attn_bias_K = (
                attn_bias.unsqueeze(2)
                .expand(B, N_sb, K, H, Nt, Nt)
                .contiguous()
                .reshape(B * N_sb * K * H, Nt, Nt)
            )
        else:
            attn_bias_K = None

        W_perm = W_feat.permute(0, 3, 1, 2, 4).contiguous()
        W_flat = W_perm.reshape(B * N_sb * K, Nt, D)

        attn_out, _ = self.w_solver_attn(
            W_flat, W_flat, W_flat,
            attn_mask=attn_bias_K
        )
        W_flat = self.norm1(W_flat + attn_out)

        Y_f_perm = Y_f_feat.permute(0, 3, 1, 2, 4).contiguous()
        Y_f_flat = Y_f_perm.reshape(B * N_sb * K, Nt, D)
        W_flat = self.norm2(W_flat + Y_f_flat)

        W_after_solver = W_flat.reshape(B, N_sb, K, Nt, D)

        if self.use_cross_user_aggregation:
            W_for_cross = (
                W_after_solver.permute(0, 1, 3, 2, 4)
                .contiguous()
                .reshape(B * N_sb * Nt, K, D)
            )
            cross_out, _ = self.cross_user_attn(
                W_for_cross, W_for_cross, W_for_cross
            )
            W_after_cross = (
                cross_out.reshape(B, N_sb, Nt, K, D)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
            W_combined = W_after_solver + W_after_cross
        else:
            W_combined = W_after_solver

        W_flat = self.norm_cross(
            W_combined.reshape(B * N_sb * K, Nt, D)
        )

        W_flat = self.norm3(W_flat + self.ffn(W_flat))

        W_out = W_flat.reshape(B, N_sb, K, Nt, D).permute(0, 2, 3, 1, 4).contiguous()
        return W_out


# ============================================================================
# 主模型: 单 checkpoint γ 自适应 Hybrid Precoder
# ============================================================================
class AdaptiveHybridPrecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K_USERS
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.D = cfg.D_MODEL
        self.L = cfg.NUM_UNFOLD_LAYERS
        self.d_a_gnn = cfg.GNN_AGG_DIM
        self.N_sb = cfg.NUM_SUBBANDS

        assert self.Nsc % self.N_sb == 0
        self.sb_size = self.Nsc // self.N_sb
        self.ul_per_sb = self.Nsc // self.N_sb

        self.d_f_sb_max = cfg.D_F_MAX_PER_USER // self.N_sb
        self.d_a_sb_max = cfg.D_A_MAX_SHARED // self.N_sb
        self.n_cfgs = len(cfg.GAMMA_GRID)

        self.use_condition_embedding = bool(
            getattr(cfg, "USE_CONDITION_EMBEDDING", True)
        )
        self.use_anchor_injection = bool(
            getattr(cfg, "USE_ANCHOR_INJECTION", True)
        )
        self.use_cross_user_aggregation = bool(
            getattr(cfg, "USE_CROSS_USER_AGGREGATION", True)
        )

        self.dft = DFTLayer(cfg.ANTENNAS, dim=-2)
        self.encoder = AdaptiveHybridEncoder(cfg)
        if self.use_condition_embedding:
            self.cond_embed = ConditionEmbed(
                self.D, cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED
            )
        else:
            self.cond_embed = None

        # ---- BS 端嵌入层: 固定最大尺寸 + 可学习 padding (JSBANet 式 ψ) ----
        self.fdma_emb = nn.Linear(self.d_f_sb_max * 2, self.D)
        self.aircomp_emb = nn.Linear(self.d_a_sb_max * 2, self.D)
        self.ul_emb = nn.Linear(self.ul_per_sb * 2, self.D)
        self.pad_f = nn.Parameter(torch.zeros(self.d_f_sb_max, 2))
        self.pad_a = nn.Parameter(torch.zeros(self.d_a_sb_max, 2))

        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ANTENNAS, self.D))
        self.user_emb = nn.Parameter(torch.randn(cfg.K_USERS, 1, self.D) * 0.02)
        self.subband_emb = nn.Parameter(torch.randn(1, self.N_sb, self.D) * 0.02)

        cross_sb_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=cfg.NUM_HEADS, batch_first=True, dropout=0.0
        )
        self.cross_subband_attn = nn.TransformerEncoder(cross_sb_layer, num_layers=1)

        self.w_init = nn.Sequential(
            nn.Linear(self.D * 2, self.D), nn.GELU(),
            nn.Linear(self.D, self.D)
        )

        self.layers = nn.ModuleList([
            UnfoldedWMMSELayer(cfg, d_model=self.D, d_a=self.d_a_gnn)
            for _ in range(self.L)
        ])

        # ---- ZF-Residual 双头 (与 V4 一致) ----
        self.h_estimator = nn.Sequential(
            nn.Linear(self.D, self.D // 2), nn.GELU(),
            nn.Linear(self.D // 2, self.sb_size * 2)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.D, self.D // 2), nn.GELU(),
            nn.Linear(self.D // 2, self.sb_size * 2)
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

        # ---- [P2 修复] per-config running power 统计 ----
        self.register_buffer('running_pwr', torch.ones(self.n_cfgs))
        self.pwr_momentum = 0.02

    # ------------------------------------------------------------------
    def add_noise(self, y, snr_db):
        # 注意: sigma 相对该流自身平均接收功率定义 (沿用 V4 约定, 见文件头 TODO)
        sigma = torch.sqrt(torch.mean(torch.abs(y) ** 2) * (10 ** (-snr_db / 10)) / 2)
        return y + (torch.randn_like(y) + 1j * torch.randn_like(y)) * sigma

    # ------------------------------------------------------------------
    def transmit(self, Z_f, Z_a, H_ul, snr_db, d_f, d_a):
        B = H_ul.shape[0]
        device = H_ul.device

        if d_f > 0:
            Y_f = torch.zeros(B, self.K, self.Nt, d_f,
                              dtype=torch.cfloat, device=device)
            for k in range(self.K):
                s = k * d_f
                Y_f[:, k] = H_ul[:, k, :, s:s + d_f] * Z_f[:, k].unsqueeze(1)
            Y_f = self.add_noise(Y_f, snr_db)
        else:
            Y_f = None

        if d_a > 0:
            start = self.K * d_f
            Y_ac = torch.sum(
                H_ul[:, :, :, start:start + d_a] * Z_a.unsqueeze(2),
                dim=1
            )
            Y_ac = self.add_noise(Y_ac, snr_db)
        else:
            Y_ac = None

        return Y_f, Y_ac

    # ------------------------------------------------------------------
    def _pad_chunks(self, Y_sb, pad_param, d_active, d_max):
        """ [..., d_active] complex → [..., d_max], 尾部填可学习 ψ """
        if d_active == d_max:
            return Y_sb
        pad_c = torch.complex(pad_param[:, 0], pad_param[:, 1])[d_active:]
        shape = list(Y_sb.shape[:-1]) + [d_max - d_active]
        pad = pad_c.view(*([1] * (Y_sb.dim() - 1)), -1).expand(shape)
        return torch.cat([Y_sb, pad], dim=-1)

    # ------------------------------------------------------------------
    def zf_solver(self, H_hat):
        B, K, Nt, Nsc = H_hat.shape
        device = H_hat.device

        H_per_sc = H_hat.permute(0, 3, 1, 2)
        H_eff = H_per_sc.conj()

        norm_factor = torch.norm(H_eff, dim=(2, 3), keepdim=True) + 1e-8
        H_eff_norm = H_eff / norm_factor

        H_eff_H = H_eff_norm.conj().transpose(-2, -1)
        H_gram = H_eff_norm @ H_eff_H

        reg = 1e-3 * torch.eye(K, device=device, dtype=torch.cfloat)
        H_gram_reg = H_gram + reg.unsqueeze(0).unsqueeze(0)

        W_zf_per_sc = H_eff_H @ torch.linalg.inv(H_gram_reg)
        return W_zf_per_sc.permute(0, 2, 3, 1)

    # ------------------------------------------------------------------
    def forward(self, H_dl, H_ul, snr_db, d_f, d_a, cfg_idx):
        """
        d_f: 当前每用户 FDMA 维度 (int), d_a: 当前共享 AirComp 维度 (int)
        cfg_idx: 该 (d_f, d_a) 在 cfg.GAMMA_GRID 中的索引 (用于 running power)
        约束: 整 batch 共享同一 (d_f, d_a) — 物理上 AirComp 要求小区级同步 γ
        """
        assert d_f % self.N_sb == 0 and d_a % self.N_sb == 0
        d_f_sb = d_f // self.N_sb
        d_a_sb = d_a // self.N_sb

        B = H_dl.shape[0]
        device = H_dl.device

        if self.use_condition_embedding:
            cond = self.cond_embed(d_f, d_a, snr_db, device)
        else:
            cond = torch.zeros(self.D, device=device, dtype=torch.float32)
        cond_5d = cond.view(1, 1, 1, 1, -1)
        cond_4d = cond.view(1, 1, 1, -1)

        # ---------- 1. UE Encoder ----------
        H_dl_flat = H_dl.reshape(B * self.K, self.Nt, self.Nsc)
        z_f_flat, z_a_flat = self.encoder(H_dl_flat, d_f_sb, d_a_sb, cond)

        # ---------- 2. 功率归一化 (per-config running stats) ----------
        n_sym = z_f_flat.shape[1] + z_a_flat.shape[1]
        energy = (z_f_flat.abs() ** 2).sum() + (z_a_flat.abs() ** 2).sum()
        pwr_batch = energy / (B * self.K * n_sym) + 1e-9

        if self.training:
            with torch.no_grad():
                self.running_pwr[cfg_idx] = (
                    (1 - self.pwr_momentum) * self.running_pwr[cfg_idx]
                    + self.pwr_momentum * pwr_batch.detach()
                )
            scale = torch.sqrt(pwr_batch)
        else:
            scale = torch.sqrt(self.running_pwr[cfg_idx] + 1e-9)

        Z_f = (z_f_flat / scale).reshape(B, self.K, -1)
        Z_a = (z_a_flat / scale).reshape(B, self.K, -1)

        # ---------- 3. 物理层传输 ----------
        Y_f, Y_ac = self.transmit(Z_f, Z_a, H_ul, snr_db, d_f, d_a)

        # ---------- 4. BS 端特征化 ----------
        # ----- FDMA -----
        if d_f > 0:
            Yf_dft = self.dft(Y_f)
            Yf_sb = Yf_dft.reshape(B, self.K, self.Nt, self.N_sb, d_f_sb)
            Yf_sb = self._pad_chunks(Yf_sb, self.pad_f, d_f_sb, self.d_f_sb_max)
            Yf_ri = torch.cat([Yf_sb.real, Yf_sb.imag], dim=-1)
            Y_f_feat = self.fdma_emb(Yf_ri)
        else:
            Y_f_feat = torch.zeros(
                B, self.K, self.Nt, self.N_sb, self.D, device=device
            )

        Y_f_feat = Y_f_feat + self.pos_emb.unsqueeze(1).unsqueeze(3)
        Y_f_feat = Y_f_feat + self.user_emb.unsqueeze(0).unsqueeze(3)
        Y_f_feat = Y_f_feat + self.subband_emb.unsqueeze(0).unsqueeze(0)
        Y_f_feat = Y_f_feat + cond_5d

        # ----- AirComp -----
        if d_a > 0:
            Yac_dft = self.dft(Y_ac)
            Yac_sb = Yac_dft.reshape(B, self.Nt, self.N_sb, d_a_sb)
            Yac_sb = self._pad_chunks(Yac_sb, self.pad_a, d_a_sb, self.d_a_sb_max)
            Yac_ri = torch.cat([Yac_sb.real, Yac_sb.imag], dim=-1)
            Y_ac_feat = self.aircomp_emb(Yac_ri)
            Y_ac_feat = Y_ac_feat + self.pos_emb[:, :, None, :]
            Y_ac_feat = Y_ac_feat + self.subband_emb[:, None, :, :]
            Y_ac_feat = Y_ac_feat + cond_4d
        else:
            # 公平补丁 (运行时分支): 纯 FDMA 时用跨用户均值合成全局上下文
            Y_ac_feat = Y_f_feat.mean(dim=1).contiguous()

        # [BUG 修复] cross-subband attention: 直接 reshape, 不再置乱
        x = Y_ac_feat.reshape(B * self.Nt, self.N_sb, self.D)
        Y_ac_feat = self.cross_subband_attn(x).reshape(
            B, self.Nt, self.N_sb, self.D
        )

        # ----- Uplink 先验 -----
        H_ul_dft = self.dft(H_ul)
        H_ul_r = H_ul_dft.real.reshape(B, self.K, self.Nt, self.N_sb, self.ul_per_sb)
        H_ul_i = H_ul_dft.imag.reshape(B, self.K, self.Nt, self.N_sb, self.ul_per_sb)
        H_ul_feat = self.ul_emb(torch.cat([H_ul_r, H_ul_i], dim=-1))
        H_ul_feat = H_ul_feat + cond_5d

        # ---------- 5. W_feat 初始化 + L 层 Unfolded WMMSE ----------
        W_feat = self.w_init(torch.cat([Y_f_feat, H_ul_feat], dim=-1))

        # Anchor ablation keeps the initial FDMA observation in W_feat, but
        # removes its repeated injection into every refinement layer. This
        # preserves a meaningful pure-FDMA endpoint while isolating the role
        # of iterative anchor guidance.
        if self.use_anchor_injection:
            Y_f_refine = Y_f_feat
        else:
            Y_f_refine = torch.zeros_like(Y_f_feat)

        for layer in self.layers:
            W_feat = layer(W_feat, Y_f_refine, Y_ac_feat)

        # ---------- 6. ZF-Residual 双头 (与 V4 一致) ----------
        h_raw = self.h_estimator(W_feat)
        h_split = h_raw.reshape(B, self.K, self.Nt, self.N_sb, self.sb_size, 2)
        H_hat = torch.complex(h_split[..., 0], h_split[..., 1]) \
                     .reshape(B, self.K, self.Nt, self.Nsc)

        W_zf = self.zf_solver(H_hat)

        if not self.training:
            frob_eval = (W_zf.abs() ** 2).sum(dim=(1, 2), keepdim=True)
            W_eval = W_zf * torch.sqrt(
                self.cfg.TOTAL_POWER / (frob_eval + 1e-9)
            )
            return W_eval

        W_zf_dir = W_zf / (torch.norm(W_zf, dim=(1, 2), keepdim=True) + 1e-8)

        res_raw = self.residual_head(W_feat)
        res_split = res_raw.reshape(B, self.K, self.Nt, self.N_sb, self.sb_size, 2)
        W_residual = torch.complex(res_split[..., 0], res_split[..., 1]) \
                          .reshape(B, self.K, self.Nt, self.Nsc) \
                          .permute(0, 2, 1, 3)

        W_train = W_zf_dir + self.residual_scale * W_residual
        frob_train = (W_train.abs() ** 2).sum(dim=(1, 2), keepdim=True)
        W_train = W_train * torch.sqrt(
            self.cfg.TOTAL_POWER / (frob_train + 1e-9)
        )

        return W_train, H_hat

# =============================================================================
# Training, validation, checkpointing, and main
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu().clone(),
    }

    if torch.cuda.is_available():
        state["cuda"] = [
            cuda_state.cpu().clone()
            for cuda_state in torch.cuda.get_rng_state_all()
        ]

    return state


def _to_cpu_byte_tensor(value: Any) -> torch.Tensor:
    """Convert a serialized RNG state to a contiguous CPU ByteTensor."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.uint8)

    return (
        value.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore Python, NumPy, CPU Torch, and CUDA RNG states safely."""
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    cpu_rng_state = _to_cpu_byte_tensor(state["torch"])
    torch.set_rng_state(cpu_rng_state)

    if torch.cuda.is_available() and "cuda" in state:
        cuda_rng_states = [
            _to_cpu_byte_tensor(cuda_state)
            for cuda_state in state["cuda"]
        ]

        current_device_count = torch.cuda.device_count()

        if len(cuda_rng_states) == current_device_count:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        else:
            # More robust when a checkpoint was saved with a different
            # visible-GPU configuration.
            for device_idx, cuda_state in enumerate(
                cuda_rng_states[:current_device_count]
            ):
                torch.cuda.set_rng_state(cuda_state, device=device_idx)

@contextmanager
def temporary_seed(seed: int):
    """Use a deterministic local seed without changing the training RNG stream."""

    state = capture_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def config_fingerprint(cfg: TrainConfig) -> str:
    payload = json.dumps(
        cfg.to_dict(), sort_keys=True, ensure_ascii=False, default=json_default
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
    os.replace(tmp, path)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def allocation_index_for_step(step: int, n_allocations: int, seed: int) -> int:
    """Balanced deterministic allocation sampling.

    Every block of n_allocations steps visits each allocation exactly once in a
    seed-dependent shuffled order. No sampler state needs to be checkpointed.
    """

    if n_allocations <= 0:
        raise ValueError("n_allocations must be positive.")
    cycle, position = divmod(step - 1, n_allocations)
    order = list(range(n_allocations))
    random.Random(seed + cycle).shuffle(order)
    return order[position]


def get_snr(step: int, cfg: TrainConfig) -> float:
    if step <= cfg.STAGE1_STEPS:
        low, high = cfg.STAGE1_SNR
    elif step <= cfg.STAGE2_STEPS:
        low, high = cfg.STAGE2_SNR
    else:
        low, high = cfg.STAGE3_SNR
    return float(np.random.uniform(low, high))


# =============================================================================
# 2. Losses and validation
# =============================================================================


class SumRateLoss(nn.Module):
    def __init__(self, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        W: torch.Tensor,
        H_dl_true: torch.Tensor,
        noise_power: float | torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        h_eff = torch.einsum("bkns,bnjs->bkjs", H_dl_true.conj(), W)
        power = h_eff.abs().square()
        sig_pwr = torch.diagonal(power, dim1=1, dim2=2).permute(0, 2, 1)
        int_pwr = power.sum(dim=2) - sig_pwr
        sinr = sig_pwr / (int_pwr + noise_power + self.eps)
        rate = torch.log2(1.0 + sinr)
        avg_rate = rate.sum(dim=1).mean()
        return -avg_rate, float(avg_rate.detach().item())


def alignment_losses(
    H_hat: torch.Tensor,
    H_dl: torch.Tensor,
    dir_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-8
    H_hat_n = H_hat / (
        H_hat.abs().square().sum(dim=2, keepdim=True).sqrt() + eps
    )
    H_dl_n = H_dl / (
        H_dl.abs().square().sum(dim=2, keepdim=True).sqrt() + eps
    )
    inner = (H_hat_n * H_dl_n.conj()).sum(dim=2)
    if dir_mode == "abs":
        cos_sim = inner.abs().mean()
    elif dir_mode == "real":
        cos_sim = inner.real.mean()
    else:
        raise ValueError(f"Unsupported DIR_MODE: {dir_mode}")
    dir_loss = 1.0 - cos_sim
    mse_loss = torch.mean(torch.abs(H_hat - H_dl).square())
    return dir_loss, mse_loss, cos_sim


def extract_precoder(model_output: Any) -> torch.Tensor:
    """Support models returning W or tuples whose first element is W."""

    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise RuntimeError("Model returned an empty tuple/list.")
        return model_output[0]
    if not torch.is_tensor(model_output):
        raise TypeError(f"Unsupported model output type: {type(model_output)!r}")
    return model_output


def extract_training_outputs(model_output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(model_output, (tuple, list)) or len(model_output) < 2:
        raise RuntimeError(
            "Proposal training requires model(...) to return at least (W, H_hat)."
        )
    W, H_hat = model_output[0], model_output[1]
    if not torch.is_tensor(W) or not torch.is_tensor(H_hat):
        raise TypeError("W and H_hat must be torch tensors.")
    return W, H_hat


@torch.no_grad()
def validate_proposal(
    model: nn.Module,
    H_dl_val: torch.Tensor,
    H_ul_val: torch.Tensor,
    criterion: SumRateLoss,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[Tuple[int, int], float]:
    """Return {(allocation_index, snr_db): sum_rate}."""

    model.eval()
    table: Dict[Tuple[int, int], float] = {}
    for cfg_idx, (d_f, d_a) in enumerate(cfg.ALLOCATION_GRID):
        for snr_db in cfg.VAL_SNR_LIST:
            # Fix both channel set and feedback-noise realization across steps.
            val_seed = cfg.SEED + 1_000_000 + cfg_idx * 10_000 + int(snr_db) * 10
            with temporary_seed(val_seed):
                snr_t = torch.tensor(float(snr_db), device=device)
                output = model(H_dl_val, H_ul_val, snr_t, d_f, d_a, cfg_idx)
                W = extract_precoder(output)
            _, rate = criterion(W, H_dl_val, 10.0 ** (-float(snr_db) / 10.0))
            table[(cfg_idx, int(snr_db))] = rate
    return table


def validation_metric(
    table: Mapping[Tuple[int, int], float], cfg: TrainConfig
) -> float:
    if cfg.BEST_METRIC == "mean_all":
        return float(np.mean(list(table.values())))
    if cfg.BEST_METRIC == "snr_25":
        values = [value for (_, snr), value in table.items() if snr == 25]
        if not values:
            raise RuntimeError("BEST_METRIC='snr_25' but 25 dB is absent from validation.")
        return float(np.mean(values))
    raise ValueError(f"Unknown BEST_METRIC: {cfg.BEST_METRIC}")


def format_val_table(
    table: Mapping[Tuple[int, int], float], step: int, cfg: TrainConfig
) -> Tuple[str, float]:
    lines = [f"Step {step} | Validation matrix (bps/Hz)"]
    header = "    allocation     " + " | ".join(
        f"{snr:>4}dB" for snr in cfg.VAL_SNR_LIST
    ) + " | mean"
    lines.append(header)

    for cfg_idx, (d_f, d_a) in enumerate(cfg.ALLOCATION_GRID):
        values = [table[(cfg_idx, snr)] for snr in cfg.VAL_SNR_LIST]
        lines.append(
            f"    ({d_f:>3},{d_a:>3})       "
            + " | ".join(f"{value:6.2f}" for value in values)
            + f" | {np.mean(values):5.2f}"
        )

    metric = validation_metric(table, cfg)
    overall = float(np.mean(list(table.values())))
    lines.append(f"    overall mean: {overall:.3f}")
    lines.append(f"    selection metric [{cfg.BEST_METRIC}]: {metric:.3f}")
    return "\n".join(lines), metric


# =============================================================================
# 3. Proposal model and checkpoint handling
# =============================================================================


def build_proposal_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    return AdaptiveHybridPrecoder(cfg).to(device)


def build_channel_generator(cfg: TrainConfig, device: torch.device):
    from utils import CDLChannelGenerator_Feedback

    return CDLChannelGenerator_Feedback(
        Nt=cfg.ANTENNAS,
        Nsc=cfg.SUBCARRIERS,
        speed=cfg.SPEED,
        num_users=cfg.K_USERS,
    ).to(device)


def generate_validation_set(channel_gen, cfg: TrainConfig, device: torch.device):
    # Validation channels are independent of the training RNG and reproducible
    # across fresh runs and resumed runs.
    with temporary_seed(cfg.SEED + 500_000):
        with torch.no_grad():
            H_dl, H_ul = channel_gen.generate_batch_data(cfg.VAL_BATCH)
    return H_dl.to(device), H_ul.to(device)


def generate_training_batch(
    channel_gen,
    cfg: TrainConfig,
    device: torch.device,
    step: int,
):
    # The batch for a given global step is deterministic. This avoids changing
    # the data stream after resume even though the in-memory cache is not saved.
    with temporary_seed(cfg.SEED + 10_000_000 + step):
        with torch.no_grad():
            H_dl, H_ul = channel_gen.generate_batch_data(cfg.BATCH_SIZE)
    return H_dl.to(device), H_ul.to(device)


def checkpoint_payload(
    *,
    cfg: TrainConfig,
    step: int,
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    scheduler: Any | None,
    best_metric: float,
    val_table: Mapping[Tuple[int, int], float] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": step,
        "state": model.state_dict(),
        "best_metric": best_metric,
        "config_fingerprint": config_fingerprint(cfg),
        "config": cfg.to_dict(),
        "allocation_grid": cfg.ALLOCATION_GRID,
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if val_table is not None:
        payload["val_table"] = {
            f"alloc{idx}_snr{snr}": value
            for (idx, snr), value in val_table.items()
        }
    return payload


def load_latest_checkpoint(
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[int, float]:
    path = Path(cfg.SAVE_PATH)
    if not path.exists():
        return 1, float("-inf")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    expected = config_fingerprint(cfg)
    actual = checkpoint.get("config_fingerprint")
    if actual is not None and actual != expected:
        raise RuntimeError(
            "Checkpoint/config mismatch. Refusing unsafe resume.\n"
            f"checkpoint fingerprint: {actual}\n"
            f"current fingerprint:    {expected}\n"
            f"path: {path}"
        )

    model.load_state_dict(checkpoint["state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint.get("rng_state"))

    start_step = int(checkpoint["step"]) + 1
    best_metric = float(
        checkpoint.get("best_metric", checkpoint.get("best_rate", float("-inf")))
    )
    return start_step, best_metric


# =============================================================================
# 4. One resolved experiment
# =============================================================================


@dataclass
class RunOptions:
    resume: bool
    skip_existing: bool
    stop_after: int | None = None


def train_proposal(cfg: TrainConfig, options: RunOptions) -> None:
    torch.set_float32_matmul_precision("high")
    set_seed(cfg.SEED)
    from utils import setup_gpu

    device = setup_gpu()

    save_dir = Path(cfg.SAVE_DIR)
    log_dir = Path(cfg.LOG_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = config_fingerprint(cfg)
    snapshot = cfg.to_dict()
    snapshot["config_fingerprint"] = fingerprint

    config_path = Path(cfg.CONFIG_PATH)
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing_fp = existing.get("config_fingerprint")
        if existing_fp is not None and existing_fp != fingerprint:
            raise RuntimeError(
                f"Existing config.json does not match the resolved run: {config_path}"
            )
    else:
        atomic_write_json(config_path, snapshot)

    status_path = Path(cfg.STATUS_PATH)
    if options.skip_existing and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            print(f"[SKIP] Completed experiment: {cfg.EXP_NAME}")
            return

    atomic_write_json(
        status_path,
        {"state": "running", "experiment": cfg.EXP_NAME, "fingerprint": fingerprint},
    )

    print("=" * 100)
    print(f"[Experiment] {cfg.EXP_NAME}")
    print(f"[Mode]       {cfg.TRAIN_MODE}")
    print(
        f"[Ablation]   {cfg.ABLATION_NAME} | "
        f"condition={cfg.USE_CONDITION_EMBEDDING} | "
        f"anchor_injection={cfg.USE_ANCHOR_INJECTION} | "
        f"cross_user_aggregation={cfg.USE_CROSS_USER_AGGREGATION}"
    )
    print(f"[Scenario]   K={cfg.K_USERS}, Nsc={cfg.SUBCARRIERS}, Dfb={cfg.D_TOT}")
    print(f"[Allocations] {cfg.ALLOCATION_GRID}")
    print(
        f"[Recipe]     {cfg.recipe.name}: steps={cfg.TOTAL_STEPS}, "
        f"batch={cfg.BATCH_SIZE}, lr={cfg.LR:g}, warmup={cfg.WARMUP_PCT:.2f}"
    )
    print(
        f"[Curriculum] phase1={cfg.PHASE1_STEPS}, "
        f"snr_stages=({cfg.STAGE1_STEPS}, {cfg.STAGE2_STEPS}, {cfg.TOTAL_STEPS})"
    )
    print(f"[Checkpoint] {cfg.SAVE_DIR}")

    model = build_proposal_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model]      trainable parameters={n_params / 1e6:.3f} M")

    criterion = SumRateLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.LR,
        total_steps=cfg.TOTAL_STEPS,
        pct_start=cfg.WARMUP_PCT,
        anneal_strategy="cos",
    )

    start_step = 1
    best_metric = float("-inf")
    if options.resume and Path(cfg.SAVE_PATH).exists():
        start_step, best_metric = load_latest_checkpoint(
            cfg, model, optimizer, scheduler, device
        )
        print(
            f"[Resume]     start_step={start_step}, "
            f"best_{cfg.BEST_METRIC}={best_metric:.3f}"
        )
    elif not options.resume and Path(cfg.SAVE_PATH).exists():
        raise FileExistsError(
            f"Checkpoint exists but --no-resume was requested: {cfg.SAVE_PATH}"
        )

    if start_step > cfg.TOTAL_STEPS:
        print(f"[DONE] Latest checkpoint already reached step {cfg.TOTAL_STEPS}.")
        atomic_write_json(
            status_path,
            {
                "state": "completed",
                "experiment": cfg.EXP_NAME,
                "step": cfg.TOTAL_STEPS,
                "best_metric": best_metric,
            },
        )
        return
    
    end_step = cfg.TOTAL_STEPS

    if options.stop_after is not None:
        if options.stop_after <= 0:
            raise ValueError("--stop-after must be a positive integer.")

        end_step = min(
            cfg.TOTAL_STEPS,
            start_step + options.stop_after - 1,
        )

        print(
            f"[Short Run]  Running steps {start_step}--{end_step} "
            f"of {cfg.TOTAL_STEPS}"
        )

    channel_gen = build_channel_generator(cfg, device)
    print("[Validation] Preparing fixed channel set...")
    H_dl_val, H_ul_val = generate_validation_set(channel_gen, cfg, device)
    # Reset CUDA peak-memory statistics before the training loop.
    # The fixed validation tensors remain allocated, so their memory is still
    # included in the current baseline.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Cache is keyed by global step. The deterministic per-step generator keeps
    # resumed runs aligned without storing large tensors inside checkpoints.
    batch_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if step not in batch_cache:
            last = min(cfg.TOTAL_STEPS, step + cfg.DATA_CACHE_SIZE - 1)
            for future_step in range(step, last + 1):
                if future_step not in batch_cache:
                    batch_cache[future_step] = generate_training_batch(
                        channel_gen, cfg, device, future_step
                    )
        batch = batch_cache.pop(step)
        return batch

    progress = tqdm(
        range(start_step, end_step + 1),
        desc=f"K{cfg.K_USERS}-D{cfg.D_TOT}-{cfg.TRAIN_MODE}",
        dynamic_ncols=True,
    )

    for step in progress:
        H_dl_batch, H_ul_batch = get_batch(step)

        cfg_idx = allocation_index_for_step(
            step, len(cfg.ALLOCATION_GRID), cfg.SEED
        )
        d_f, d_a = cfg.ALLOCATION_GRID[cfg_idx]

        model.train()
        optimizer.zero_grad(set_to_none=True)

        snr = get_snr(step, cfg)
        noise_power = 10.0 ** (-snr / 10.0)
        snr_tensor = torch.tensor(snr, device=device, dtype=torch.float32)

        output = model(
            H_dl_batch,
            H_ul_batch,
            snr_tensor,
            d_f,
            d_a,
            cfg_idx,
        )
        # W, H_hat = extract_training_outputs(output)

        # sum_rate_loss, train_rate = criterion(W, H_dl_batch, noise_power)
        # dir_loss, mse_loss, cos_sim = alignment_losses(
        #     H_hat, H_dl_batch, cfg.DIR_MODE
        # )

        # 模型原始训练输出仍保留，用于兼容当前 forward 接口。
        W_residual_train, H_hat = extract_training_outputs(output)

        # ============================================================
        # Deploy-consistent task path
        # 训练时直接优化验证/部署阶段真正使用的 RZF(H_hat)。
        # ============================================================
        proposal_model = (
            model.module
            if hasattr(model, "module")
            else model
        )

        W_deploy = proposal_model.zf_solver(H_hat)

        frob_deploy = (
            W_deploy.abs()
            .square()
            .sum(dim=(1, 2), keepdim=True)
        )

        W_deploy = W_deploy * torch.sqrt(
            cfg.TOTAL_POWER / (frob_deploy + 1e-9)
        )

        # 主 sum-rate loss 只作用于实际部署路径。
        sum_rate_loss, train_rate = criterion(
            W_deploy,
            H_dl_batch,
            noise_power,
        )

        dir_loss, mse_loss, cos_sim = alignment_losses(
            H_hat,
            H_dl_batch,
            cfg.DIR_MODE,
        )        

        if step <= cfg.PHASE1_STEPS:
            w_dir, w_mse = cfg.PH1_W_DIR, cfg.PH1_W_MSE
            phase = "phase1_anchor"
        else:
            w_dir, w_mse = cfg.PH2_W_DIR, cfg.PH2_W_MSE
            phase = "phase2_task"

        loss = sum_rate_loss + w_dir * dir_loss + w_mse * mse_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step={step}, allocation={(d_f, d_a)}, snr={snr:.2f}."
            )

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at step={step}.")
        optimizer.step()
        scheduler.step()

        if step % cfg.LOG_INTERVAL == 0:
            progress.set_postfix(
                {
                    "phase": phase,
                    "alloc": f"({d_f},{d_a})",
                    "SR": f"{train_rate:.2f}",
                    "cos": f"{float(cos_sim.detach()):.3f}",
                    "MSE": f"{float(mse_loss.detach()):.4f}",
                    "SNR": f"{snr:.1f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        if step % cfg.VAL_INTERVAL == 0 or step == end_step:
            val_table = validate_proposal(
                model, H_dl_val, H_ul_val, criterion, cfg, device
            )
            text, current_metric = format_val_table(val_table, step, cfg)
            progress.write(text)

            metrics_record = {
                "step": step,
                "selection_metric_name": cfg.BEST_METRIC,
                "selection_metric": current_metric,
                "best_metric_before_update": best_metric,
                "table": {
                    f"alloc{idx}_snr{snr}": value
                    for (idx, snr), value in val_table.items()
                },
            }
            append_jsonl(cfg.METRICS_PATH, metrics_record)

            if current_metric > best_metric:
                best_metric = current_metric
                progress.write(
                    f"New best {cfg.BEST_METRIC}: {best_metric:.3f}; saving best.pth"
                )
                atomic_torch_save(
                    checkpoint_payload(
                        cfg=cfg,
                        step=step,
                        model=model,
                        optimizer=None,
                        scheduler=None,
                        best_metric=best_metric,
                        val_table=val_table,
                    ),
                    cfg.BEST_PATH,
                )

            atomic_torch_save(
                checkpoint_payload(
                    cfg=cfg,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    val_table=val_table,
                ),
                cfg.SAVE_PATH,
            )

    # Report the true CUDA peak over training, validation, and checkpointing.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )
        peak_reserved_gb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        )

        print(
            f"[GPU Memory] peak allocated={peak_allocated_gb:.2f} GB, "
            f"peak reserved={peak_reserved_gb:.2f} GB"
        )
        
    if end_step < cfg.TOTAL_STEPS:
        atomic_write_json(
            status_path,
            {
                "state": "paused",
                "experiment": cfg.EXP_NAME,
                "step": end_step,
                "total_steps": cfg.TOTAL_STEPS,
                "best_metric_name": cfg.BEST_METRIC,
                "best_metric": best_metric,
                "latest_path": cfg.SAVE_PATH,
            },
        )

        print(
            f"[Paused] {cfg.EXP_NAME} at step {end_step}/"
            f"{cfg.TOTAL_STEPS}"
        )
        return
    
    atomic_write_json(
        status_path,
        {
            "state": "completed",
            "experiment": cfg.EXP_NAME,
            "step": cfg.TOTAL_STEPS,
            "best_metric_name": cfg.BEST_METRIC,
            "best_metric": best_metric,
            "best_path": cfg.BEST_PATH,
            "latest_path": cfg.SAVE_PATH,
        },
    )
    print(f"[Completed] {cfg.EXP_NAME}; best {cfg.BEST_METRIC}={best_metric:.3f}")


# =============================================================================
# 5. Suite filtering and CLI
# =============================================================================


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def filter_runs(
    runs: Sequence[TrainConfig],
    *,
    only_k: set[int] | None,
    only_budget: set[int] | None,
    only_mode: str | None,
    only_model: str | None,
    only_variant: str | None,
    num_shards: int,
    shard_id: int,
) -> list[TrainConfig]:
    filtered = [
        cfg
        for cfg in runs
        if (only_k is None or cfg.K_USERS in only_k)
        and (only_budget is None or cfg.D_TOT in only_budget)
        and (only_mode is None or cfg.TRAIN_MODE == only_mode)
        and (only_model is None or cfg.MODEL_NAME == only_model)
        and (only_variant is None or cfg.ABLATION_NAME == only_variant)
    ]

    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= shard_id < num_shards:
        raise ValueError("Require 0 <= --shard-id < --num-shards.")

    return [cfg for index, cfg in enumerate(filtered) if index % num_shards == shard_id]


def print_run_plan(runs: Sequence[TrainConfig]) -> None:
    if not runs:
        print("No experiments matched the requested filters.")
        return
    print(f"Resolved {len(runs)} experiment(s):")
    for index, cfg in enumerate(runs):
        print(
            f"[{index:02d}] model={cfg.MODEL_NAME:<12} mode={cfg.TRAIN_MODE:<10} "
            f"K={cfg.K_USERS:<2} Nsc={cfg.SUBCARRIERS:<3} Dfb={cfg.D_TOT:<3} "
            f"alloc={cfg.ALLOCATION_GRID}"
        )
        print(f"     {cfg.EXP_NAME}")
        print(
            f"     ablation={cfg.ABLATION_NAME}, "
            f"cond={cfg.USE_CONDITION_EMBEDDING}, "
            f"anchor={cfg.USE_ANCHOR_INJECTION}, "
            f"cross_user={cfg.USE_CROSS_USER_AGGREGATION}"
        )
        print(
            f"     recipe={cfg.recipe.name}, "
            f"steps={cfg.TOTAL_STEPS}, "
            f"batch={cfg.BATCH_SIZE}, "
            f"val_batch={cfg.VAL_BATCH}, "
            f"cache={cfg.DATA_CACHE_SIZE}, "
            f"lr={cfg.LR:g}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train standalone K4-Nsc128-D128 Hybrid FDMA-AirComp ablations."
    )
    parser.add_argument("--suite", type=str, default="proposal_ablation_k4_n128")
    parser.add_argument("--output-root", type=str, default="runs_ablation_deployzf_v1")
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-k", type=str, default=None, help="Example: 4,8")
    parser.add_argument("--only-budget", type=str, default=None, help="Example: 128,256")
    parser.add_argument(
        "--only-mode",
        choices=["adaptive", "specialist", "baseline"],
        default=None,
    )
    parser.add_argument(
        "--only-model",
        choices=["proposal", "csinet_plus", "swin_cfnet"],
        default=None,
    )
    parser.add_argument(
        "--only-variant",
        choices=[
            "full",
            "no_condition",
            "no_anchor_injection",
            "no_cross_user_aggregation",
        ],
        default=None,
        help="Restrict an ablation suite to one variant.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Skip declared baseline runs whose project-specific adapters are not implemented.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Run only this many additional steps, save, and pause.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.list_suites:
        print("Available suites:")
        for name in list_suites():
            print(f"  - {name}")
        return 0

    runs = get_suite(args.suite, output_root=args.output_root)
    runs = filter_runs(
        runs,
        only_k=parse_int_set(args.only_k),
        only_budget=parse_int_set(args.only_budget),
        only_mode=args.only_mode,
        only_model=args.only_model,
        only_variant=args.only_variant,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    print_run_plan(runs)

    if args.dry_run:
        return 0
    if not runs:
        return 0

    options = RunOptions(
        resume=not args.no_resume,
        skip_existing=args.skip_existing,
        stop_after=args.stop_after,
    )
    failures = []

    for index, cfg in enumerate(runs, start=1):
        print(f"\n[Suite] Starting {index}/{len(runs)}: {cfg.EXP_NAME}")
        try:
            if cfg.MODEL_NAME == "proposal":
                train_proposal(cfg, options)
            else:
                message = (
                    f"Training adapter for '{cfg.MODEL_NAME}' is not implemented because "
                    "its constructor and forward/loss interface were not included. "
                    "The resolved baseline config is ready in train_config.py."
                )
                if args.skip_unsupported:
                    print(f"[SKIP] {message}")
                    continue
                raise NotImplementedError(message)
        except Exception as exc:  # Keep an overnight suite alive unless fail-fast is requested.
            failures.append((cfg.EXP_NAME, repr(exc)))
            try:
                atomic_write_json(
                    cfg.STATUS_PATH,
                    {
                        "state": "failed",
                        "experiment": cfg.EXP_NAME,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception:
                pass
            print(f"[FAILED] {cfg.EXP_NAME}: {exc}", file=sys.stderr)
            traceback.print_exc()
            if args.fail_fast:
                break
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        print("\nSuite finished with failures:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        return 1

    print("\nSuite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())