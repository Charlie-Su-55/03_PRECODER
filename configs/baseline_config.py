#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified configuration for CsiNet+-based and Swin-based FDMA baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    k_users: int
    subcarriers: int
    feedback_budget: int


SCENARIOS: Dict[str, ScenarioSpec] = {
    "k4_n128_d128": ScenarioSpec("k4_n128_d128", 4, 128, 128),
    "k8_n128_d128": ScenarioSpec("k8_n128_d128", 8, 128, 128),
    "k4_n256_d256": ScenarioSpec("k4_n256_d256", 4, 256, 256),
    "k8_n256_d256": ScenarioSpec("k8_n256_d256", 8, 256, 256),
    "k16_n256_d256": ScenarioSpec("k16_n256_d256", 16, 256, 256),
}


@dataclass
class BaselineConfig:
    MODEL_NAME: str
    SCENARIO_NAME: str
    OBJECTIVE: str
    OUTPUT_ROOT: str

    K_USERS: int
    ANTENNAS: int
    SUBCARRIERS: int
    D_TOT: int
    D_F: int

    CARRIER_FREQ: float
    SPEED: float
    TOTAL_POWER: float
    RZF_REG: float
    SEED: int

    TOTAL_STEPS: int
    BATCH_SIZE: int
    VAL_BATCH: int
    VAL_INTERVAL: int
    LOG_INTERVAL: int
    DATA_CACHE_SIZE: int

    LR: float
    WEIGHT_DECAY: float
    WARMUP_PCT: float
    GRAD_CLIP: float

    SNR_STAGE1_END: int
    SNR_STAGE2_END: int
    STAGE1_SNR: tuple[float, float]
    STAGE2_SNR: tuple[float, float]
    STAGE3_SNR: tuple[float, float]
    VAL_SNR_LIST: tuple[int, ...]

    SHARE_USER_WEIGHTS: bool

    CSINET_NC: int
    CSINET_ENC_CHANNELS: tuple[int, ...]
    CSINET_DEC_CHANNELS: tuple[int, ...]

    SWIN_NC: int
    SWIN_PATCH_SIZE: int
    SWIN_EMBED_DIM: int
    SWIN_NUM_HEADS: int
    SWIN_NUM_BLOCKS: int
    SWIN_WINDOW_SIZE: int
    SWIN_MLP_RATIO: float

    @property
    def DIM_FDMA_PER_USER(self) -> int:
        return self.D_F

    @property
    def DIM_AIRCOMP_SHARED(self) -> int:
        return 0

    @property
    def SWIN_CODEWORD_DIM(self) -> int:
        return 2 * self.D_F

    @property
    def EXP_NAME(self) -> str:
        sharing = "sharedUE" if self.SHARE_USER_WEIGHTS else "independentUE"
        return (
            f"{self.MODEL_NAME}_{self.OBJECTIVE}"
            f"_K{self.K_USERS}_Nsc{self.SUBCARRIERS}"
            f"_Dfb{self.D_TOT}_Df{self.D_F}"
            f"_{sharing}_seed{self.SEED}"
        )

    @property
    def SAVE_DIR(self) -> str:
        return str(Path(self.OUTPUT_ROOT) / "checkpoints" / self.EXP_NAME)

    @property
    def BEST_PATH(self) -> str:
        return str(Path(self.SAVE_DIR) / "best.pth")

    @property
    def SAVE_PATH(self) -> str:
        return str(Path(self.SAVE_DIR) / "latest.pth")

    @property
    def LOG_DIR(self) -> str:
        return str(Path(self.OUTPUT_ROOT) / "logs" / self.EXP_NAME)

    @property
    def METRICS_PATH(self) -> str:
        return str(Path(self.LOG_DIR) / "metrics.jsonl")

    @property
    def STATUS_PATH(self) -> str:
        return str(Path(self.LOG_DIR) / "status.json")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "DIM_FDMA_PER_USER": self.DIM_FDMA_PER_USER,
                "DIM_AIRCOMP_SHARED": self.DIM_AIRCOMP_SHARED,
                "SWIN_CODEWORD_DIM": self.SWIN_CODEWORD_DIM,
                "EXP_NAME": self.EXP_NAME,
                "SAVE_DIR": self.SAVE_DIR,
                "BEST_PATH": self.BEST_PATH,
                "SAVE_PATH": self.SAVE_PATH,
                "LOG_DIR": self.LOG_DIR,
                "METRICS_PATH": self.METRICS_PATH,
                "STATUS_PATH": self.STATUS_PATH,
            }
        )
        return payload


def list_scenarios() -> List[str]:
    return list(SCENARIOS.keys())


def make_config(
    model_name: str,
    scenario_name: str,
    objective: str = "task",
    output_root: str = "runs_baselines",
    seed: int = 42,
) -> BaselineConfig:
    model_name = model_name.lower().strip()
    objective = objective.lower().strip()

    if model_name not in {"csinet", "swin"}:
        raise ValueError(f"Unsupported model: {model_name}")
    if objective not in {"task", "reconstruction"}:
        raise ValueError(f"Unsupported objective: {objective}")
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Available: {', '.join(list_scenarios())}"
        )

    spec = SCENARIOS[scenario_name]
    if spec.feedback_budget % spec.k_users != 0:
        raise ValueError(
            f"D_tot={spec.feedback_budget} is not divisible by K={spec.k_users}."
        )

    d_f = spec.feedback_budget // spec.k_users
    if spec.k_users * d_f > spec.subcarriers:
        raise ValueError(
            f"FDMA allocation uses {spec.k_users*d_f} subcarriers, but Nsc={spec.subcarriers}."
        )

    if spec.k_users == 4:
        total_steps, batch_size, val_batch = 50_000, 48, 64
        val_interval, warmup_pct, grad_clip = 500, 0.35, 1.0
    elif spec.k_users == 8:
        total_steps, batch_size, val_batch = 100_000, 48, 32
        val_interval, warmup_pct, grad_clip = 1_000, 0.20, 0.5
    else:
        total_steps, batch_size, val_batch = 100_000, 32, 32
        val_interval, warmup_pct, grad_clip = 1_000, 0.20, 0.5

    return BaselineConfig(
        MODEL_NAME=model_name,
        SCENARIO_NAME=scenario_name,
        OBJECTIVE=objective,
        OUTPUT_ROOT=output_root,
        K_USERS=spec.k_users,
        ANTENNAS=32,
        SUBCARRIERS=spec.subcarriers,
        D_TOT=spec.feedback_budget,
        D_F=d_f,
        CARRIER_FREQ=3.5e9,
        SPEED=1.0,
        TOTAL_POWER=1.0,
        RZF_REG=1e-3,
        SEED=seed,
        TOTAL_STEPS=total_steps,
        BATCH_SIZE=batch_size,
        VAL_BATCH=val_batch,
        VAL_INTERVAL=val_interval,
        LOG_INTERVAL=100,
        DATA_CACHE_SIZE=1,
        LR=1e-4,
        WEIGHT_DECAY=1e-4,
        WARMUP_PCT=warmup_pct,
        GRAD_CLIP=grad_clip,
        SNR_STAGE1_END=int(round(0.30 * total_steps)),
        SNR_STAGE2_END=int(round(0.70 * total_steps)),
        STAGE1_SNR=(20.0, 25.0),
        STAGE2_SNR=(10.0, 25.0),
        STAGE3_SNR=(0.0, 25.0),
        VAL_SNR_LIST=(0, 10, 20, 25),
        SHARE_USER_WEIGHTS=True,
        CSINET_NC=min(32, spec.subcarriers),
        CSINET_ENC_CHANNELS=(16, 8, 4, 2),
        CSINET_DEC_CHANNELS=(2, 4, 8, 16),
        SWIN_NC=min(32, spec.subcarriers),
        SWIN_PATCH_SIZE=4,
        SWIN_EMBED_DIM=128,
        SWIN_NUM_HEADS=4,
        SWIN_NUM_BLOCKS=4,
        SWIN_WINDOW_SIZE=4,
        SWIN_MLP_RATIO=4.0,
    )