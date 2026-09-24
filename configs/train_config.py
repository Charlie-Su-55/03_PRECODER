#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified experiment definitions for Hybrid FDMA-AirComp training.

Core goals
----------
1. Keep all physical, model, optimization, curriculum, baseline, and naming
   parameters in one place.
2. Generate valid allocation grids automatically from (K, D_tot).
3. Expand one suite into adaptive checkpoints, per-allocation specialists, or
   pure-FDMA baselines without editing source code.
4. Preserve the uppercase attributes expected by the existing model classes.

Typical commands are defined in main_train.py, for example:
    python main_train.py --suite proposal_adaptive
    python main_train.py --suite proposal_specialists
    python main_train.py --suite proposal_all
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
import math
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Sequence, Tuple

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
    # None preserves the historical matched feedback/downlink SNR protocol.
    downlink_snr_db: float | None = None

    def __post_init__(self) -> None:
        if self.downlink_snr_db is not None and not math.isfinite(self.downlink_snr_db):
            raise ValueError("downlink_snr_db must be finite or None.")
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
    # Protocol-aligned pure-AS fine-tuning; deliberately NOT added to old suites.
    # The phase boundaries have no effect on auxiliary weights or SNR ranges here.
    "k4_n256_fb_robust_ft": TrainingRecipe(
        name="k4_n256_fb_robust_ft",
        total_steps=20_000,
        batch_size=48,
        lr=2e-5,
        val_interval=500,
        val_batch=64,
        warmup_pct=0.10,
        grad_clip=0.5,
        phase1_pct=0.40,
        snr_stage1_pct=0.30,
        snr_stage2_pct=0.70,
        stage1_snr=(0.0, 25.0),
        stage2_snr=(0.0, 25.0),
        stage3_snr=(0.0, 25.0),
        val_snr_list=(0, 5, 10, 15, 20, 25),
        downlink_snr_db=25.0,
        ph1_w_dir=0.0,
        ph1_w_mse=0.0,
        ph2_w_dir=0.0,
        ph2_w_mse=0.0,
        weight_decay=1e-5,
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

    "k8_n128_sumrate_ft": TrainingRecipe(
        name="k8_n128_sumrate_ft",

        # Fine-tuning only
        total_steps=20_000,
        batch_size=48,
        lr=2e-5,
        val_interval=500,
        val_batch=64,
        warmup_pct=0.10,
        grad_clip=0.5,

        # These boundaries remain valid, although all auxiliary weights are zero.
        phase1_pct=0.50,
        snr_stage1_pct=0.30,
        snr_stage2_pct=0.70,

        # Retain multi-SNR operation rather than training only at one point.
        stage1_snr=(15.0, 25.0),
        stage2_snr=(5.0, 25.0),
        stage3_snr=(0.0, 25.0),
        val_snr_list=(0, 5, 10, 15, 20, 25),

        # Pure deployment sum-rate optimization
        ph1_w_dir=0.0,
        ph1_w_mse=0.0,
        ph2_w_dir=0.0,
        ph2_w_mse=0.0,

        weight_decay=1e-5,
        data_cache_size=1,
        log_interval=100,

        # Keep scientifically clean selection across the allocation-SNR grid.
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


# Low-load diagnostic: reuse the K=4/Nsc=128 recipe without changing old suites.
TRAINING_RECIPES['k2_n128_screen'] = replace(
    TRAINING_RECIPES['k4_n128_standard'], name='k2_n128_screen'
)


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
    def DOWNLINK_SNR_DB(self) -> float | None:
        return self.recipe.downlink_snr_db

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

        recipe_snapshot = asdict(self.recipe)
        if self.DOWNLINK_SNR_DB is None:
            # Preserve old config.json/checkpoint fingerprints and safe resume.
            # Fixed-DL recipes still serialize the new field explicitly.
            recipe_snapshot.pop("downlink_snr_db")
        return {
            "experiment": asdict(self.spec),
            "allocations": [list(a) for a in self.allocations],
            "full_allocation_grid": [list(a) for a in self.full_allocation_grid],
            "training_recipe": recipe_snapshot,
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

def _proposal_sumrate_ft_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            model_name="proposal",
            train_mode="adaptive",
            k_users=8,
            subcarriers=128,
            feedback_budget=128,
            recipe_name="k8_n128_sumrate_ft",
        )
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
    'proposal_k2_screen': (
        ExperimentSpec(
            model_name='proposal', train_mode='adaptive', k_users=2,
            subcarriers=128, feedback_budget=128, recipe_name='k2_n128_screen',
        ),
    ),
    "proposal_adaptive": tuple(_proposal_adaptive_specs()),
    "proposal_specialists": tuple(_proposal_specialist_specs()),
    "proposal_all": tuple(_proposal_adaptive_specs() + _proposal_specialist_specs()),
    # These two suites expose the correct shared parameters and naming. Their
    # training adapters depend on the actual baseline class/forward interfaces.
    "proposal_sumrate_ft": tuple(_proposal_sumrate_ft_specs()),
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
