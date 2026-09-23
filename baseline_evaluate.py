#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_evaluate.py
====================

Unified evaluator for the proposed Hybrid FDMA-AirComp model, AI FDMA
baselines, and Perfect-CSI RZF references.

Main features
-------------
1. Select arbitrary K, Nsc, feedback budget, models, allocations, and SNRs.
2. Evaluate every method on the same channel realizations.
3. Separate feedback SNR from downlink SNR.
4. Support one or multiple scenarios in a single command.
5. Discover checkpoints automatically or accept explicit checkpoint paths.
6. Save plot-ready long-form CSV, JSON metadata, and sample-level CSV.GZ.
7. Keep one registry-style entry point for future baselines.

Typical commands
----------------
# Current K=8 comparison, legacy matched feedback/downlink SNR
python baseline_evaluate.py \
  --k 8 --nsc 256 --budget 256 \
  --models proposed,csinet,swin,perfect_rzf \
  --allocations all \
  --feedback-snr 0,10,20,25 \
  --downlink-snr matched \
  --batch-size 8 --num-batches 50 \
  --proposal-root runs_deployzf_v1 \
  --baseline-root runs_baselines \
  --output-dir runs_evaluation/k8_d256

# Pure feedback-SNR sweep with fixed downlink SNR
python baseline_evaluate.py \
  --k 8 --nsc 256 --budget 256 \
  --models proposed,csinet,swin,perfect_rzf \
  --allocations 24,64 \
  --feedback-snr 0,5,10,15,20,25 \
  --downlink-snr 20 \
  --snr-mode cartesian \
  --output-dir runs_evaluation/k8_d256_fixed_dl20

# Evaluate several scenarios in one run
python baseline_evaluate.py \
  --scenario 4,256,256 \
  --scenario 8,256,256 \
  --scenario 16,256,256 \
  --models proposed,csinet,swin,perfect_rzf \
  --feedback-snr 25 \
  --downlink-snr matched \
  --output-dir runs_evaluation/scalability_d256

# Only selected proposed allocations
python baseline_evaluate.py \
  --k 8 --nsc 256 --budget 256 \
  --models proposed \
  --allocations "32,0;24,64;0,256" \
  --feedback-snr 0,10,20,25 \
  --downlink-snr matched

Notes
-----
- "matched" reproduces the current protocol where feedback SNR and downlink
  SNR use the same numerical value.
- A fixed downlink SNR gives a clean feedback-SNR sensitivity experiment.
- The evaluator uses the same channel batch for all selected methods.
- Re-seeding before each forward pass provides reproducible common random
  numbers where model tensor shapes and random call order are compatible.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from utils import CDLChannelGenerator_Feedback, setup_gpu


# =============================================================================
# 1. Constants and lightweight runtime configuration
# =============================================================================

SUPPORTED_MODELS = {
    "proposed",
    "csinet",
    "swin",
    "perfect_rzf",
}

MODEL_ALIASES = {
    "proposal": "proposed",
    "hybrid": "proposed",
    "adaptive": "proposed",
    "csinet+": "csinet",
    "csinet_plus": "csinet",
    "swincfnet": "swin",
    "swin_cfnet": "swin",
    "oracle_rzf": "perfect_rzf",
    "perfect": "perfect_rzf",
}

MODEL_DISPLAY_NAMES = {
    "proposed": "Proposed",
    "csinet": "CsiNet+-based FDMA",
    "swin": "Swin-based FDMA",
    "perfect_rzf": "Perfect-CSI RZF",
}


@dataclass(frozen=True)
class Scenario:
    k_users: int
    subcarriers: int
    feedback_budget: int
    antennas: int = 32
    carrier_freq: float = 3.5e9
    speed: float = 1.0
    total_power: float = 1.0
    num_subbands: int = 4

    def validate(self) -> None:
        if self.k_users <= 0:
            raise ValueError("K must be positive.")
        if self.subcarriers <= 0:
            raise ValueError("Nsc must be positive.")
        if self.feedback_budget <= 0:
            raise ValueError("Feedback budget must be positive.")
        if self.antennas <= 0:
            raise ValueError("Number of antennas must be positive.")
        if self.num_subbands <= 0:
            raise ValueError("Number of subbands must be positive.")
        if self.subcarriers % self.num_subbands != 0:
            raise ValueError(
                f"Nsc={self.subcarriers} must be divisible by "
                f"NUM_SUBBANDS={self.num_subbands}."
            )

    @property
    def scenario_id(self) -> str:
        return (
            f"K{self.k_users}_Nsc{self.subcarriers}"
            f"_Dfb{self.feedback_budget}"
        )


@dataclass
class LoadedMethod:
    model_key: str
    method_key: str
    display_name: str
    model: torch.nn.Module | None
    checkpoint_path: str
    checkpoint_step: int
    checkpoint_kind: str
    allocations: Tuple[Tuple[int, int], ...]
    config: Any
    rzf_lambda: float | None = None


class RuntimeConfig(SimpleNamespace):
    """Attribute-style config consumed by the current model constructors."""


class CompatInt(int):
    """Integer that also supports legacy single-stage access such as value[0]."""

    def __new__(cls, value: Any):
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Cannot convert an empty sequence to CompatInt.")
            value = value[0]
        return super().__new__(cls, int(value))

    def __getitem__(self, index: int) -> int:
        if index != 0:
            raise IndexError(index)
        return int(self)


# =============================================================================
# 2. Parsing and deterministic utilities
# =============================================================================

def normalize_model_name(name: str) -> str:
    key = name.strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{name}'. Available models: "
            f"{', '.join(sorted(SUPPORTED_MODELS))}"
        )
    return key


def parse_csv(text: str | None) -> List[str]:
    if text is None:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_csv(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_scenario(text: str, defaults: argparse.Namespace) -> Scenario:
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "--scenario must be K,Nsc,Dtot, for example 8,256,256."
        )
    scenario = Scenario(
        k_users=int(parts[0]),
        subcarriers=int(parts[1]),
        feedback_budget=int(parts[2]),
        antennas=defaults.antennas,
        carrier_freq=defaults.carrier_freq,
        speed=defaults.speed,
        total_power=defaults.total_power,
        num_subbands=defaults.num_subbands,
    )
    scenario.validate()
    return scenario


def parse_scenarios(args: argparse.Namespace) -> List[Scenario]:
    if args.scenario:
        scenarios = [parse_scenario(text, args) for text in args.scenario]
    else:
        if args.k is None or args.nsc is None or args.budget is None:
            raise ValueError(
                "Provide --k, --nsc, and --budget, or use one or more "
                "--scenario K,Nsc,Dtot arguments."
            )
        scenario = Scenario(
            k_users=args.k,
            subcarriers=args.nsc,
            feedback_budget=args.budget,
            antennas=args.antennas,
            carrier_freq=args.carrier_freq,
            speed=args.speed,
            total_power=args.total_power,
            num_subbands=args.num_subbands,
        )
        scenario.validate()
        scenarios = [scenario]

    unique: Dict[str, Scenario] = {}
    for scenario in scenarios:
        unique[scenario.scenario_id] = scenario
    return list(unique.values())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int(text: str) -> int:
    """Stable small integer independent of Python's randomized hash."""
    value = 0
    for byte in text.encode("utf-8"):
        value = (value * 131 + byte) % 2_000_000_000
    return value


def mean_se_ci95(values: torch.Tensor) -> Tuple[float, float, float, float]:
    values = values.detach().float().cpu().flatten()
    if values.numel() == 0:
        return math.nan, math.nan, math.nan, math.nan

    mean = float(values.mean().item())
    if values.numel() <= 1:
        return mean, 0.0, mean, mean

    se = float(values.std(unbiased=True).item() / math.sqrt(values.numel()))
    half_width = 1.96 * se
    return mean, se, mean - half_width, mean + half_width


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


# =============================================================================
# 3. Allocation and SNR-grid construction
# =============================================================================

def build_quarter_allocation_grid(
    scenario: Scenario,
) -> Tuple[Tuple[int, int], ...]:
    allocations: List[Tuple[int, int]] = []

    for quarter in range(5):
        d_a_numerator = scenario.feedback_budget * quarter
        if d_a_numerator % 4 != 0:
            continue
        d_a = d_a_numerator // 4

        fdma_total = scenario.feedback_budget - d_a
        if fdma_total % scenario.k_users != 0:
            continue
        d_f = fdma_total // scenario.k_users

        if d_f % scenario.num_subbands != 0:
            continue
        if d_a % scenario.num_subbands != 0:
            continue

        allocations.append((d_f, d_a))

    if not allocations:
        raise ValueError(
            f"No valid quarter-grid allocation for {scenario.scenario_id}."
        )
    return tuple(allocations)


def validate_allocation(
    allocation: Tuple[int, int],
    scenario: Scenario,
) -> None:
    d_f, d_a = allocation
    if d_f < 0 or d_a < 0:
        raise ValueError(f"Negative allocation is invalid: {allocation}")
    if scenario.k_users * d_f + d_a != scenario.feedback_budget:
        raise ValueError(
            f"Invalid allocation {allocation}. Expected "
            f"{scenario.k_users}*D_f + D_a = "
            f"{scenario.feedback_budget}."
        )
    if d_f % scenario.num_subbands != 0:
        raise ValueError(
            f"D_f={d_f} is not divisible by "
            f"NUM_SUBBANDS={scenario.num_subbands}."
        )
    if d_a % scenario.num_subbands != 0:
        raise ValueError(
            f"D_a={d_a} is not divisible by "
            f"NUM_SUBBANDS={scenario.num_subbands}."
        )


def parse_allocation_selection(
    text: str,
    available: Sequence[Tuple[int, int]],
    scenario: Scenario,
) -> Tuple[Tuple[int, int], ...]:
    normalized = text.strip().lower()
    available = tuple((int(a), int(b)) for a, b in available)

    if normalized == "all":
        return available
    if normalized in {"pure", "endpoints"}:
        return tuple(
            allocation
            for allocation in available
            if allocation[0] == 0 or allocation[1] == 0
        )
    if normalized == "hybrid":
        return tuple(
            allocation
            for allocation in available
            if allocation[0] > 0 and allocation[1] > 0
        )
    if normalized in {"pure_fdma", "fdma"}:
        matches = [a for a in available if a[1] == 0]
        if not matches:
            raise ValueError("Pure-FDMA allocation is absent.")
        return tuple(matches)
    if normalized in {"pure_aircomp", "aircomp"}:
        matches = [a for a in available if a[0] == 0]
        if not matches:
            raise ValueError("Pure-AirComp allocation is absent.")
        return tuple(matches)

    selected: List[Tuple[int, int]] = []
    for token in text.split(";"):
        parts = [item.strip() for item in token.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "--allocations must be all, pure, hybrid, pure_fdma, "
                "pure_aircomp, or a semicolon-separated list such as "
                "'24,64;0,256'."
            )
        allocation = (int(parts[0]), int(parts[1]))
        validate_allocation(allocation, scenario)
        if allocation not in available:
            raise ValueError(
                f"Allocation {allocation} is not present in checkpoint grid "
                f"{list(available)}."
            )
        selected.append(allocation)

    if not selected:
        raise ValueError("No allocation selected.")
    return tuple(dict.fromkeys(selected))


def build_snr_points(
    feedback_snr_text: str,
    downlink_snr_text: str,
    snr_mode: str,
) -> List[Tuple[float, float]]:
    feedback_values = parse_float_csv(feedback_snr_text)

    if downlink_snr_text.strip().lower() == "matched":
        return [(value, value) for value in feedback_values]

    downlink_values = parse_float_csv(downlink_snr_text)

    if snr_mode == "cartesian":
        return [
            (feedback_snr, downlink_snr)
            for feedback_snr in feedback_values
            for downlink_snr in downlink_values
        ]

    if len(downlink_values) == 1:
        return [
            (feedback_snr, downlink_values[0])
            for feedback_snr in feedback_values
        ]

    if len(feedback_values) != len(downlink_values):
        raise ValueError(
            "For --snr-mode zip, feedback and downlink SNR lists must have "
            "equal lengths, unless the downlink list has one fixed value."
        )

    return list(zip(feedback_values, downlink_values))


# =============================================================================
# 4. Checkpoint discovery and config reconstruction
# =============================================================================

def checkpoint_filename(kind: str) -> str:
    return "best.pth" if kind == "best" else "latest.pth"


def checkpoint_prefixes(
    model_key: str,
    objective: str,
) -> Tuple[str, ...]:
    if model_key == "proposed":
        return (
            "hybrid_gnn_adaptive_",
            "adaptive_",
        )
    if model_key == "csinet":
        return (
            f"csinet_{objective}_",
            f"csinet_plus_{objective}_",
            "csinet_",
            "csinet_plus_",
        )
    if model_key == "swin":
        return (
            f"swin_{objective}_",
            f"swin_cfnet_{objective}_",
            "swin_",
            "swin_cfnet_",
        )
    raise ValueError(model_key)


def discover_checkpoint(
    root: str | Path,
    model_key: str,
    scenario: Scenario,
    checkpoint_kind: str,
    objective: str,
) -> Path:
    root = Path(root)
    checkpoint_root = root / "checkpoints"
    filename = checkpoint_filename(checkpoint_kind)

    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"Checkpoint root does not exist: {checkpoint_root}"
        )

    scenario_token = (
        f"K{scenario.k_users}_Nsc{scenario.subcarriers}"
        f"_Dfb{scenario.feedback_budget}"
    )
    candidates: List[Path] = []

    for prefix in checkpoint_prefixes(model_key, objective):
        pattern = f"{prefix}*{scenario_token}*/{filename}"
        candidates.extend(checkpoint_root.glob(pattern))

    # Preserve unique paths while keeping deterministic order.
    candidates = sorted(set(path.resolve() for path in candidates))

    if not candidates:
        searched = ", ".join(
            f"{prefix}*{scenario_token}*/{filename}"
            for prefix in checkpoint_prefixes(model_key, objective)
        )
        raise FileNotFoundError(
            f"No {model_key} checkpoint found under {checkpoint_root}. "
            f"Searched: {searched}"
        )

    if len(candidates) > 1:
        candidate_text = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            f"Multiple {model_key} checkpoints match {scenario.scenario_id}. "
            f"Use the explicit --{model_key}-checkpoint option.\n"
            f"{candidate_text}"
        )

    return candidates[0]


def flatten_uppercase_config(
    payload: Any,
    result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if result is None:
        result = {}

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(key, str) and key.isupper():
                result[key] = value
            flatten_uppercase_config(value, result)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            flatten_uppercase_config(value, result)

    return result


def checkpoint_config_values(
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    config_payload = checkpoint.get("config")
    if config_payload is not None:
        values.update(flatten_uppercase_config(config_payload))

    allocation_grid = checkpoint.get("allocation_grid")
    if allocation_grid is not None:
        grid = [tuple(map(int, allocation)) for allocation in allocation_grid]
        values["ALLOCATION_GRID"] = grid
        values["GAMMA_GRID"] = grid

    return values


def validate_checkpoint_scenario(
    checkpoint: Mapping[str, Any],
    path: Path,
    scenario: Scenario,
) -> None:
    values = checkpoint_config_values(checkpoint)

    expected = {
        "K_USERS": scenario.k_users,
        "SUBCARRIERS": scenario.subcarriers,
        "D_TOT": scenario.feedback_budget,
    }
    for key, expected_value in expected.items():
        actual = values.get(key)
        if actual is not None and int(actual) != int(expected_value):
            raise RuntimeError(
                f"Checkpoint mismatch for {path}. "
                f"{key}={actual}, requested {expected_value}."
            )


def load_checkpoint_cpu(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if "state" not in checkpoint:
        raise RuntimeError(f"Checkpoint has no 'state' field: {path}")
    return checkpoint


def proposal_runtime_config(
    scenario: Scenario,
    checkpoint: Mapping[str, Any],
) -> RuntimeConfig:
    values = checkpoint_config_values(checkpoint)
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    resolved = config.get("resolved", {})
    if not isinstance(resolved, Mapping):
        resolved = {}

    default_grid = build_quarter_allocation_grid(scenario)
    # This grid determines supported allocations AND running_pwr indices.
    # Do not expand it to the architecture grid for a specialist.
    checkpoint_grid = values.get(
        "ALLOCATION_GRID",
        values.get("GAMMA_GRID", config.get("allocations", default_grid)),
    )
    grid = tuple(tuple(map(int, allocation)) for allocation in checkpoint_grid)
    if not grid:
        raise ValueError("Proposal checkpoint has no supported allocations.")

    # Training retains maximum-width layers even for a single allocation.
    # Prefer saved dimensions, then the saved full grid, then the scenario grid.
    architecture_grid = values.get(
        "FULL_ALLOCATION_GRID", config.get("full_allocation_grid", default_grid)
    )
    architecture_grid = tuple(tuple(map(int, a)) for a in architecture_grid)
    if not architecture_grid:
        raise ValueError("Proposal checkpoint has an empty architecture grid.")
    for allocation in grid + architecture_grid:
        validate_allocation(allocation, scenario)
    dimensions: Dict[str, int] = {}
    for key, saved_key, axis in (
        ("D_F_MAX_PER_USER", "d_f_max_per_user", 0),
        ("D_A_MAX_SHARED", "d_a_max_shared", 1),
    ):
        saved = values.get(key)
        if saved is None:
            saved = resolved.get(saved_key, config.get(saved_key))
        width = int(saved) if saved is not None else max(a[axis] for a in architecture_grid)
        if width < max(a[axis] for a in grid) or width % scenario.num_subbands:
            raise ValueError(f"Invalid proposal architecture dimension {key}={width}.")
        dimensions[key] = width

    defaults: Dict[str, Any] = {
        "K_USERS": scenario.k_users,
        "ANTENNAS": scenario.antennas,
        "SUBCARRIERS": scenario.subcarriers,
        "CARRIER_FREQ": scenario.carrier_freq,
        "SPEED": scenario.speed,
        "TOTAL_POWER": scenario.total_power,
        "D_TOT": scenario.feedback_budget,
        "NUM_SUBBANDS": scenario.num_subbands,
        "D_MODEL": 256,
        "NUM_HEADS": 8,
        "NUM_ENCODER_LAYERS": 4,
        "NUM_DECODER_LAYERS": 4,
        "NUM_ITERATIONS": 5,
        "DROPOUT": 0.1,
        "NUM_UNFOLD_LAYERS": 5,
        "GNN_AGG_DIM": 48,
        "GAMMA_GRID": list(grid),
        "ALLOCATION_GRID": list(grid),
        **dimensions,
    }

    defaults.update(values)

    # The requested scenario remains authoritative after validation.
    defaults.update(
        {
            "K_USERS": scenario.k_users,
            "ANTENNAS": scenario.antennas,
            "SUBCARRIERS": scenario.subcarriers,
            "CARRIER_FREQ": scenario.carrier_freq,
            "SPEED": scenario.speed,
            "TOTAL_POWER": scenario.total_power,
            "D_TOT": scenario.feedback_budget,
            "NUM_SUBBANDS": scenario.num_subbands,
            "GAMMA_GRID": list(grid),
            "ALLOCATION_GRID": list(grid),
            "FULL_ALLOCATION_GRID": list(architecture_grid),
            **dimensions,
        }
    )
    return RuntimeConfig(**defaults)


def baseline_runtime_config(
    scenario: Scenario,
    model_key: str,
    checkpoint: Mapping[str, Any],
    rzf_reg: float,
) -> RuntimeConfig:
    if scenario.feedback_budget % scenario.k_users != 0:
        raise ValueError(
            f"Pure FDMA requires Dtot divisible by K. "
            f"Got Dtot={scenario.feedback_budget}, K={scenario.k_users}."
        )
    d_f = scenario.feedback_budget // scenario.k_users

    values = checkpoint_config_values(checkpoint)
    defaults: Dict[str, Any] = {
        "MODEL_NAME": model_key,
        "K_USERS": scenario.k_users,
        "ANTENNAS": scenario.antennas,
        "SUBCARRIERS": scenario.subcarriers,
        "CARRIER_FREQ": scenario.carrier_freq,
        "SPEED": scenario.speed,
        "TOTAL_POWER": scenario.total_power,
        "D_TOT": scenario.feedback_budget,
        "D_F": d_f,
        "DIM_FDMA_PER_USER": d_f,
        "DIM_AIRCOMP_SHARED": 0,
        "RZF_REG": rzf_reg,
        "SHARE_USER_WEIGHTS": True,
        "CSINET_NC": min(32, scenario.subcarriers),
        "CSINET_ENC_CHANNELS": (16, 8, 4, 2),
        "CSINET_DEC_CHANNELS": (2, 4, 8, 16),
        "SWIN_NC": min(32, scenario.subcarriers),
        "SWIN_PATCH_SIZE": 4,
        "SWIN_EMBED_DIM": 128,
        "SWIN_NUM_HEADS": 4,
        "SWIN_NUM_BLOCKS": 4,
        "SWIN_WINDOW_SIZE": 4,
        "SWIN_MLP_RATIO": 4.0,
        "SWIN_CODEWORD_DIM": 2 * d_f,
    }
    defaults.update(values)

    # Compatibility aliases for both the current scalar config and older
    # single-stage Swin code that accesses the same values through [0].
    embed_source = defaults.get(
        "SWIN_EMBED_DIM",
        defaults.get("SWIN_EMBED_DIMS", 128),
    )
    heads_source = defaults.get("SWIN_NUM_HEADS", 4)
    blocks_source = defaults.get("SWIN_NUM_BLOCKS", 4)

    embed_dim = CompatInt(embed_source)
    num_heads = CompatInt(heads_source)
    num_blocks = CompatInt(blocks_source)

    defaults.update(
        {
            "K_USERS": scenario.k_users,
            "ANTENNAS": scenario.antennas,
            "SUBCARRIERS": scenario.subcarriers,
            "CARRIER_FREQ": scenario.carrier_freq,
            "SPEED": scenario.speed,
            "TOTAL_POWER": scenario.total_power,
            "D_TOT": scenario.feedback_budget,
            "D_F": d_f,
            "DIM_FDMA_PER_USER": d_f,
            "DIM_AIRCOMP_SHARED": 0,
            "RZF_REG": rzf_reg,
            "SWIN_CODEWORD_DIM": 2 * d_f,
            "SWIN_EMBED_DIM": embed_dim,
            "SWIN_EMBED_DIMS": embed_dim,
            "SWIN_NUM_HEADS": num_heads,
            "SWIN_NUM_BLOCKS": num_blocks,
        }
    )
    return RuntimeConfig(**defaults)


# =============================================================================
# 5. Model loading and forward adapters
# =============================================================================

def build_proposed_method(
    scenario: Scenario,
    path: Path,
    checkpoint_kind: str,
    allocation_text: str,
    device: torch.device,
) -> LoadedMethod:
    from models.adaptive_hybrid import AdaptiveHybridPrecoder

    checkpoint = load_checkpoint_cpu(path)
    validate_checkpoint_scenario(checkpoint, path, scenario)
    cfg = proposal_runtime_config(scenario, checkpoint)

    available = tuple(tuple(map(int, a)) for a in cfg.GAMMA_GRID)
    allocations = parse_allocation_selection(
        allocation_text,
        available,
        scenario,
    )

    model = AdaptiveHybridPrecoder(cfg).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()
    print(
        f"[Proposal architecture] D_f_max={cfg.D_F_MAX_PER_USER}, "
        f"D_a_max={cfg.D_A_MAX_SHARED}; "
        f"supported allocations={list(available)}; strict=True"
    )

    return LoadedMethod(
        model_key="proposed",
        method_key="proposed",
        display_name=MODEL_DISPLAY_NAMES["proposed"],
        model=model,
        checkpoint_path=str(path),
        checkpoint_step=int(checkpoint.get("step", -1)),
        checkpoint_kind=checkpoint_kind,
        allocations=allocations,
        config=cfg,
    )


def build_baseline_method(
    model_key: str,
    scenario: Scenario,
    path: Path,
    checkpoint_kind: str,
    device: torch.device,
    rzf_reg: float,
) -> LoadedMethod:
    if model_key == "csinet":
        from models.csinet_plus_precoder import CsiNetPlus_E2E_Precoder
        model_class = CsiNetPlus_E2E_Precoder
    elif model_key == "swin":
        from models.swin_precoder import Swin_E2E_Precoder
        model_class = Swin_E2E_Precoder
    else:
        raise ValueError(model_key)

    checkpoint = load_checkpoint_cpu(path)
    validate_checkpoint_scenario(checkpoint, path, scenario)
    cfg = baseline_runtime_config(
        scenario,
        model_key,
        checkpoint,
        rzf_reg,
    )

    model = model_class(cfg).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()

    allocation = (scenario.feedback_budget // scenario.k_users, 0)
    return LoadedMethod(
        model_key=model_key,
        method_key=model_key,
        display_name=MODEL_DISPLAY_NAMES[model_key],
        model=model,
        checkpoint_path=str(path),
        checkpoint_step=int(checkpoint.get("step", -1)),
        checkpoint_kind=checkpoint_kind,
        allocations=(allocation,),
        config=cfg,
    )


def build_perfect_rzf_methods(
    scenario: Scenario,
    lambdas: Sequence[float],
) -> List[LoadedMethod]:
    methods = []
    for value in lambdas:
        if value < 0:
            raise ValueError("RZF regularization must be non-negative.")
        method_key = f"perfect_rzf_lam{value:.3e}"
        methods.append(
            LoadedMethod(
                model_key="perfect_rzf",
                method_key=method_key,
                display_name=f"Perfect-CSI RZF $\\lambda={value:g}$",
                model=None,
                checkpoint_path="",
                checkpoint_step=-1,
                checkpoint_kind="analytic",
                allocations=((-1, -1),),
                config=RuntimeConfig(
                    TOTAL_POWER=scenario.total_power,
                    RZF_REG=value,
                ),
                rzf_lambda=value,
            )
        )
    return methods


def extract_output(
    model_key: str,
    output: Any,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Normalize the output formats used by the current project models.

    The proposed model returns only the deployed precoder in eval mode, while
    the baseline models return a dictionary. Proposed-channel extraction is
    handled by forward_proposed_with_surrogate().
    """
    if model_key == "proposed":
        if torch.is_tensor(output):
            return output, None
        if isinstance(output, Mapping):
            W = output.get("precoder", output.get("W"))
            H_hat = output.get(
                "channel_surrogate",
                output.get("H_hat"),
            )
            if not torch.is_tensor(W):
                raise RuntimeError(
                    "Proposed mapping output has no tensor precoder."
                )
            return W, H_hat
        if isinstance(output, (tuple, list)) and output:
            W = output[0]
            H_hat = output[1] if len(output) >= 2 else None
            if not torch.is_tensor(W):
                raise RuntimeError(
                    "The first proposed output must be a tensor precoder."
                )
            return W, H_hat
        raise RuntimeError(
            f"Unsupported proposed output type: {type(output)!r}"
        )

    if model_key in {"csinet", "swin"}:
        if not isinstance(output, Mapping):
            raise RuntimeError(
                f"{model_key} must return a mapping with "
                "'precoder' and 'channel_surrogate'."
            )
        return output["precoder"], output["channel_surrogate"]

    raise ValueError(model_key)


def reconstruct_proposed_surrogate(
    model: torch.nn.Module,
    h_raw: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Reconstruct H_hat from the output of model.h_estimator.

    Current AdaptiveHybridPrecoder.eval() returns only W_eval. A temporary
    forward hook captures h_raw without changing the deployed forward path.
    """
    required = ("K", "Nt", "N_sb", "sb_size", "Nsc")
    missing = [
        name for name in required
        if not hasattr(model, name)
    ]
    if missing:
        raise RuntimeError(
            "Cannot reconstruct the proposed channel surrogate. "
            f"Missing model attributes: {missing}"
        )

    h_split = h_raw.reshape(
        batch_size,
        int(model.K),
        int(model.Nt),
        int(model.N_sb),
        int(model.sb_size),
        2,
    )
    H_hat = torch.complex(
        h_split[..., 0],
        h_split[..., 1],
    )
    return H_hat.reshape(
        batch_size,
        int(model.K),
        int(model.Nt),
        int(model.Nsc),
    )


def forward_proposed_with_surrogate(
    model: torch.nn.Module,
    H_dl: torch.Tensor,
    H_ul: torch.Tensor,
    feedback_snr: torch.Tensor,
    d_f: int,
    d_a: int,
    cfg_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the exact deployed proposed path and also recover H_hat for metrics.

    The hook is read-only. It does not change W, model mode, parameters, random
    numbers, or the checkpoint.
    """
    captured: Dict[str, torch.Tensor] = {}

    if not hasattr(model, "h_estimator"):
        raise RuntimeError(
            "The proposed model has no h_estimator module. "
            "Cannot recover H_hat from its eval-only output."
        )

    def capture_h_raw(
        _module: torch.nn.Module,
        _inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        captured["h_raw"] = output

    handle = model.h_estimator.register_forward_hook(
        capture_h_raw
    )
    try:
        output = model(
            H_dl,
            H_ul,
            feedback_snr,
            d_f,
            d_a,
            cfg_idx,
        )
    finally:
        handle.remove()

    W, H_hat = extract_output("proposed", output)

    if H_hat is None:
        h_raw = captured.get("h_raw")
        if h_raw is None:
            raise RuntimeError(
                "The proposed model returned only W, and the evaluator "
                "could not capture h_estimator output."
            )
        H_hat = reconstruct_proposed_surrogate(
            model,
            h_raw,
            H_dl.shape[0],
        )

    if not torch.is_tensor(H_hat):
        raise RuntimeError(
            "Recovered proposed channel surrogate is not a tensor."
        )
    return W, H_hat


def closed_form_rzf_reference(
    H_true: torch.Tensor,
    regularization: float,
    total_power: float,
) -> torch.Tensor:
    """
    Standard per-subcarrier RZF with the same H.conj() @ W convention.

    H_true: [B, K, Nt, Nsc]
    W:      [B, Nt, K, Nsc]
    """
    B, K, _, Nsc = H_true.shape

    H_eff = H_true.conj().permute(0, 3, 1, 2)
    norm = torch.linalg.vector_norm(
        H_eff,
        dim=(2, 3),
        keepdim=True,
    ).clamp_min(1e-8)
    H_eff = H_eff / norm

    H_h = H_eff.conj().transpose(-2, -1)
    gram = H_eff @ H_h

    eye = torch.eye(
        K,
        dtype=H_true.dtype,
        device=H_true.device,
    ).view(1, 1, K, K)

    matrix = gram + float(regularization) * eye
    rhs = eye.expand(B, Nsc, K, K)
    inverse_action = torch.linalg.solve(matrix, rhs)

    W_sc = H_h @ inverse_action
    W = W_sc.permute(0, 2, 3, 1).contiguous()

    power = W.abs().square().sum(dim=(1, 2), keepdim=True)
    return W * torch.sqrt(total_power / (power + 1e-9))


# =============================================================================
# 6. Metrics
# =============================================================================

def per_sample_sum_rate(
    W: torch.Tensor,
    H_dl_true: torch.Tensor,
    downlink_noise_power: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    h_eff = torch.einsum(
        "bkns,bnjs->bkjs",
        H_dl_true.conj(),
        W,
    )
    power = h_eff.abs().square()
    signal = torch.diagonal(
        power,
        dim1=1,
        dim2=2,
    ).permute(0, 2, 1)
    interference = power.sum(dim=2) - signal
    sinr = signal / (
        interference + float(downlink_noise_power) + eps
    )
    rate = torch.log2(1.0 + sinr)
    return rate.sum(dim=1).mean(dim=-1)


def per_sample_nmse_db(
    H_hat: torch.Tensor,
    H_true: torch.Tensor,
) -> torch.Tensor:
    error = (
        H_hat - H_true
    ).abs().square().sum(dim=(1, 2, 3))
    reference = (
        H_true.abs().square().sum(dim=(1, 2, 3))
        .clamp_min(1e-12)
    )
    return 10.0 * torch.log10(
        (error / reference).clamp_min(1e-12)
    )


def per_sample_precoder_power(
    W: torch.Tensor,
) -> torch.Tensor:
    return W.abs().square().sum(dim=(1, 2, 3))


def per_sample_cosine_metrics(
    H_hat: torch.Tensor,
    H_true: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    H_hat_n = H_hat / (
        H_hat.abs().square().sum(dim=2, keepdim=True).sqrt() + eps
    )
    H_true_n = H_true / (
        H_true.abs().square().sum(dim=2, keepdim=True).sqrt() + eps
    )
    inner = (H_hat_n * H_true_n.conj()).sum(dim=2)
    reduce_dims = tuple(range(1, inner.ndim))
    return (
        inner.real.mean(dim=reduce_dims),
        inner.abs().mean(dim=reduce_dims),
    )


def per_sample_mean_log10_condition(
    H: torch.Tensor,
) -> torch.Tensor:
    """
    Mean log10 condition number over subcarriers for each sample.

    H: [B, K, Nt, Nsc]
    """
    matrices = H.permute(0, 3, 1, 2)
    singular_values = torch.linalg.svdvals(matrices)
    condition = (
        singular_values[..., 0]
        / singular_values[..., -1].clamp_min(1e-12)
    )
    return torch.log10(condition.clamp_min(1.0)).mean(dim=1)


# =============================================================================
# 7. Unified evaluation engine
# =============================================================================

def build_channel_generator(
    scenario: Scenario,
    device: torch.device,
):
    generator = CDLChannelGenerator_Feedback(
        Nt=scenario.antennas,
        Nsc=scenario.subcarriers,
        carrier_freq=scenario.carrier_freq,
        speed=scenario.speed,
        num_users=scenario.k_users,
    )
    if hasattr(generator, "to"):
        generator = generator.to(device)
    return generator


def explicit_checkpoint_for(
    model_key: str,
    args: argparse.Namespace,
    scenario_count: int,
) -> Path | None:
    mapping = {
        "proposed": args.proposed_checkpoint,
        "csinet": args.csinet_checkpoint,
        "swin": args.swin_checkpoint,
    }
    value = mapping[model_key]
    if value is None:
        return None
    if scenario_count != 1:
        raise ValueError(
            f"Explicit --{model_key}-checkpoint can only be used when "
            "evaluating one scenario."
        )
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def load_methods_for_scenario(
    scenario: Scenario,
    model_keys: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    scenario_count: int,
) -> List[LoadedMethod]:
    methods: List[LoadedMethod] = []

    for model_key in model_keys:
        if model_key == "perfect_rzf":
            methods.extend(
                build_perfect_rzf_methods(
                    scenario,
                    parse_float_csv(args.rzf_lambdas),
                )
            )
            continue

        explicit = explicit_checkpoint_for(
            model_key,
            args,
            scenario_count,
        )

        if explicit is not None:
            path = explicit
        elif model_key == "proposed":
            path = discover_checkpoint(
                args.proposal_root,
                model_key,
                scenario,
                args.checkpoint,
                args.baseline_objective,
            )
        else:
            path = discover_checkpoint(
                args.baseline_root,
                model_key,
                scenario,
                args.checkpoint,
                args.baseline_objective,
            )

        if model_key == "proposed":
            methods.append(
                build_proposed_method(
                    scenario,
                    path,
                    args.checkpoint,
                    args.allocations,
                    device,
                )
            )
        else:
            methods.append(
                build_baseline_method(
                    model_key,
                    scenario,
                    path,
                    args.checkpoint,
                    device,
                    args.rzf_reg,
                )
            )

    return methods


@torch.no_grad()
def evaluate_scenario(
    scenario: Scenario,
    methods: Sequence[LoadedMethod],
    snr_points: Sequence[Tuple[float, float]],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[dict], List[dict]]:
    generator = build_channel_generator(scenario, device)

    # key -> list[tensor], one tensor per batch
    stores: Dict[Tuple[str, int, int, float, float, str], List[torch.Tensor]] = {}

    def add_metric(
        method_key: str,
        d_f: int,
        d_a: int,
        feedback_snr: float,
        downlink_snr: float,
        metric: str,
        values: torch.Tensor,
    ) -> None:
        key = (
            method_key,
            d_f,
            d_a,
            feedback_snr,
            downlink_snr,
            metric,
        )
        stores.setdefault(key, []).append(values.detach().cpu())

    progress = tqdm(
        range(args.num_batches),
        desc=f"evaluate-{scenario.scenario_id}",
        dynamic_ncols=True,
    )

    for batch_index in progress:
        channel_seed = (
            args.seed
            + stable_int(scenario.scenario_id)
            + batch_index
        )
        set_seed(channel_seed)
        H_dl, H_ul = generator.generate_batch_data(args.batch_size)
        H_dl = H_dl.to(device)
        H_ul = H_ul.to(device)

        perfect_cache: Dict[float, torch.Tensor] = {}

        for feedback_snr, downlink_snr in snr_points:
            downlink_noise_power = 10.0 ** (
                -float(downlink_snr) / 10.0
            )

            for method in methods:
                if method.model_key == "perfect_rzf":
                    assert method.rzf_lambda is not None
                    if method.rzf_lambda not in perfect_cache:
                        perfect_cache[method.rzf_lambda] = (
                            closed_form_rzf_reference(
                                H_dl,
                                regularization=method.rzf_lambda,
                                total_power=scenario.total_power,
                            )
                        )
                    W = perfect_cache[method.rzf_lambda]
                    H_hat = None
                    allocations = ((-1, -1),)
                else:
                    allocations = method.allocations

                for allocation_index, (d_f, d_a) in enumerate(allocations):
                    if method.model_key == "proposed":
                        all_grid = [
                            tuple(map(int, allocation))
                            for allocation in method.config.GAMMA_GRID
                        ]
                        cfg_idx = all_grid.index((d_f, d_a))

                        # Re-seed each method/allocation at the same condition.
                        # Exact common noise is achieved only when random-call
                        # shapes and order are identical.
                        feedback_seed = (
                            args.seed
                            + 100_000_000
                            + batch_index * 100_000
                            + stable_int(f"{feedback_snr:.6f}")
                        )
                        set_seed(feedback_seed)
                        W, H_hat = forward_proposed_with_surrogate(
                            method.model,
                            H_dl,
                            H_ul,
                            torch.tensor(
                                float(feedback_snr),
                                dtype=torch.float32,
                                device=device,
                            ),
                            d_f,
                            d_a,
                            cfg_idx,
                        )

                    elif method.model_key in {"csinet", "swin"}:
                        feedback_seed = (
                            args.seed
                            + 100_000_000
                            + batch_index * 100_000
                            + stable_int(f"{feedback_snr:.6f}")
                        )
                        set_seed(feedback_seed)
                        output = method.model(
                            H_dl,
                            H_ul,
                            torch.tensor(
                                float(feedback_snr),
                                dtype=torch.float32,
                                device=device,
                            ),
                        )
                        W, H_hat = extract_output(
                            method.model_key,
                            output,
                        )

                    elif method.model_key == "perfect_rzf":
                        # W and H_hat were prepared above.
                        pass

                    else:
                        raise ValueError(method.model_key)

                    rate_values = per_sample_sum_rate(
                        W,
                        H_dl,
                        downlink_noise_power,
                    )
                    add_metric(
                        method.method_key,
                        d_f,
                        d_a,
                        feedback_snr,
                        downlink_snr,
                        "sum_rate",
                        rate_values,
                    )

                    power_values = per_sample_precoder_power(W)
                    add_metric(
                        method.method_key,
                        d_f,
                        d_a,
                        feedback_snr,
                        downlink_snr,
                        "precoder_power",
                        power_values,
                    )

                    if H_hat is not None:
                        add_metric(
                            method.method_key,
                            d_f,
                            d_a,
                            feedback_snr,
                            downlink_snr,
                            "nmse_db",
                            per_sample_nmse_db(H_hat, H_dl),
                        )

                        if args.extra_metrics == "full":
                            cos_real, cos_abs = per_sample_cosine_metrics(
                                H_hat,
                                H_dl,
                            )
                            add_metric(
                                method.method_key,
                                d_f,
                                d_a,
                                feedback_snr,
                                downlink_snr,
                                "cos_real",
                                cos_real,
                            )
                            add_metric(
                                method.method_key,
                                d_f,
                                d_a,
                                feedback_snr,
                                downlink_snr,
                                "cos_abs",
                                cos_abs,
                            )
                            add_metric(
                                method.method_key,
                                d_f,
                                d_a,
                                feedback_snr,
                                downlink_snr,
                                "surrogate_log10_condition",
                                per_sample_mean_log10_condition(H_hat),
                            )

    method_by_key = {method.method_key: method for method in methods}
    summary_rows: List[dict] = []
    sample_rows: List[dict] = []

    grouped_conditions = sorted(
        {
            key[:5]
            for key in stores
        },
        key=lambda value: (
            value[3],
            value[4],
            value[0],
            value[1],
            value[2],
        ),
    )

    for (
        method_key,
        d_f,
        d_a,
        feedback_snr,
        downlink_snr,
    ) in grouped_conditions:
        method = method_by_key[method_key]
        metric_values: Dict[str, torch.Tensor] = {}

        for key, batch_values in stores.items():
            if key[:5] != (
                method_key,
                d_f,
                d_a,
                feedback_snr,
                downlink_snr,
            ):
                continue
            metric_values[key[5]] = torch.cat(batch_values)

        rate_values = metric_values["sum_rate"]
        rate_mean, rate_se, rate_ci_low, rate_ci_high = mean_se_ci95(
            rate_values
        )

        row = {
            "scenario_id": scenario.scenario_id,
            "k_users": scenario.k_users,
            "antennas": scenario.antennas,
            "subcarriers": scenario.subcarriers,
            "feedback_budget": scenario.feedback_budget,
            "model_key": method.model_key,
            "method_key": method.method_key,
            "method_label": method.display_name,
            "checkpoint_kind": method.checkpoint_kind,
            "checkpoint_step": method.checkpoint_step,
            "checkpoint_path": method.checkpoint_path,
            "allocation_df": d_f,
            "allocation_da": d_a,
            "fdma_fraction": (
                math.nan
                if d_f < 0
                else scenario.k_users * d_f
                / scenario.feedback_budget
            ),
            "aircomp_fraction": (
                math.nan
                if d_a < 0
                else d_a / scenario.feedback_budget
            ),
            "feedback_snr_db": feedback_snr,
            "downlink_snr_db": downlink_snr,
            "num_samples": int(rate_values.numel()),
            "sum_rate_mean": rate_mean,
            "sum_rate_se": rate_se,
            "sum_rate_ci95_low": rate_ci_low,
            "sum_rate_ci95_high": rate_ci_high,
            "rzf_lambda": (
                method.rzf_lambda
                if method.rzf_lambda is not None
                else getattr(method.config, "RZF_REG", math.nan)
            ),
            "seed": args.seed,
        }

        for metric_name, values in metric_values.items():
            if metric_name == "sum_rate":
                continue
            mean, se, ci_low, ci_high = mean_se_ci95(values)
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_se"] = se
            row[f"{metric_name}_ci95_low"] = ci_low
            row[f"{metric_name}_ci95_high"] = ci_high

        summary_rows.append(row)

        if not args.no_save_samples:
            sample_count = int(rate_values.numel())
            for sample_index in range(sample_count):
                sample_row = {
                    "scenario_id": scenario.scenario_id,
                    "k_users": scenario.k_users,
                    "subcarriers": scenario.subcarriers,
                    "feedback_budget": scenario.feedback_budget,
                    "model_key": method.model_key,
                    "method_key": method.method_key,
                    "method_label": method.display_name,
                    "allocation_df": d_f,
                    "allocation_da": d_a,
                    "feedback_snr_db": feedback_snr,
                    "downlink_snr_db": downlink_snr,
                    "sample_index": sample_index,
                    "sum_rate": float(rate_values[sample_index].item()),
                }
                for metric_name, values in metric_values.items():
                    if metric_name == "sum_rate":
                        continue
                    sample_row[metric_name] = float(
                        values[sample_index].item()
                    )
                sample_rows.append(sample_row)

    if args.include_proposed_best:
        add_proposed_best_rows(summary_rows)

    return summary_rows, sample_rows


def add_proposed_best_rows(summary_rows: List[dict]) -> None:
    groups: Dict[Tuple[str, float, float], List[dict]] = {}

    for row in summary_rows:
        if row["model_key"] != "proposed":
            continue
        key = (
            row["scenario_id"],
            float(row["feedback_snr_db"]),
            float(row["downlink_snr_db"]),
        )
        groups.setdefault(key, []).append(row)

    derived: List[dict] = []
    for rows in groups.values():
        if not rows:
            continue
        best = max(rows, key=lambda row: row["sum_rate_mean"])
        copied = dict(best)
        copied["model_key"] = "proposed_best"
        copied["method_key"] = "proposed_best"
        copied["method_label"] = "Proposed best allocation"
        copied["selected_allocation_df"] = best["allocation_df"]
        copied["selected_allocation_da"] = best["allocation_da"]
        derived.append(copied)

    summary_rows.extend(derived)


# =============================================================================
# 8. Output
# =============================================================================

def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    print("\n" + "=" * 152)
    print("Unified evaluation summary")
    print("=" * 152)
    print(
        f"{'scenario':<20} | {'method':<28} | "
        f"{'allocation':>12} | {'FB SNR':>7} | {'DL SNR':>7} | "
        f"{'rate':>9} | {'SE':>7} | {'Raw representation NMSE (dB)':>28}"
    )
    print("-" * 152)

    for row in summary_rows:
        if row["model_key"] == "proposed_best":
            continue

        allocation = (
            "reference"
            if row["allocation_df"] < 0
            else f"({row['allocation_df']},{row['allocation_da']})"
        )
        nmse = row.get("nmse_db_mean", math.nan)
        nmse_text = (
            ""
            if not math.isfinite(float(nmse))
            else f"{float(nmse):.3f}"
        )

        print(
            f"{row['scenario_id']:<20} | "
            f"{row['method_label']:<28} | "
            f"{allocation:>12} | "
            f"{row['feedback_snr_db']:>7.1f} | "
            f"{row['downlink_snr_db']:>7.1f} | "
            f"{row['sum_rate_mean']:>9.3f} | "
            f"{row['sum_rate_se']:>7.3f} | "
            f"{nmse_text:>28}"
        )


def save_outputs(
    summary_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Scenario],
    model_keys: Sequence[str],
    snr_points: Sequence[Tuple[float, float]],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tag:
        tag = args.tag
    elif len(scenarios) == 1:
        tag = scenarios[0].scenario_id
    else:
        tag = f"multi_{len(scenarios)}scenarios"

    summary_path = output_dir / f"summary_{tag}.csv"
    json_path = output_dir / f"results_{tag}.json"
    metadata_path = output_dir / f"metadata_{tag}.json"
    samples_path = output_dir / f"samples_{tag}.csv.gz"

    write_csv(summary_path, summary_rows)
    if sample_rows and not args.no_save_samples:
        write_gzip_csv(samples_path, sample_rows)

    json_path.write_text(
        json.dumps(
            list(summary_rows),
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    metadata = {
        "tag": tag,
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "k_users": scenario.k_users,
                "antennas": scenario.antennas,
                "subcarriers": scenario.subcarriers,
                "feedback_budget": scenario.feedback_budget,
                "carrier_freq": scenario.carrier_freq,
                "speed": scenario.speed,
                "total_power": scenario.total_power,
                "num_subbands": scenario.num_subbands,
            }
            for scenario in scenarios
        ],
        "models": list(model_keys),
        "snr_points": [
            {
                "feedback_snr_db": feedback_snr,
                "downlink_snr_db": downlink_snr,
            }
            for feedback_snr, downlink_snr in snr_points
        ],
        "checkpoint": args.checkpoint,
        "baseline_objective": args.baseline_objective,
        "proposal_root": args.proposal_root,
        "baseline_root": args.baseline_root,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_samples_per_condition": (
            args.batch_size * args.num_batches
        ),
        "allocations": args.allocations,
        "rzf_lambdas": parse_float_csv(args.rzf_lambdas),
        "rzf_reg": args.rzf_reg,
        "extra_metrics": args.extra_metrics,
        "seed": args.seed,
        "summary_schema": {
            "sum_rate_mean": "Mean downlink sum rate in bps/Hz.",
            "sum_rate_se": "Standard error over channel samples.",
            "sum_rate_ci95_low": "Lower normal-approximation 95% CI.",
            "sum_rate_ci95_high": "Upper normal-approximation 95% CI.",
            "feedback_snr_db": "SNR used inside the feedback channel.",
            "downlink_snr_db": "SNR used in the downlink rate metric.",
            "allocation_df": "Per-user complex FDMA feedback dimension.",
            "allocation_da": "Shared complex AirComp feedback dimension.",
        },
    }
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print("\n[Saved]")
    print(f"  {summary_path}")
    print(f"  {json_path}")
    print(f"  {metadata_path}")
    if sample_rows and not args.no_save_samples:
        print(f"  {samples_path}")


# =============================================================================
# 9. CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluation for proposed, AI baselines, "
            "and Perfect-CSI RZF."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    scenario_group = parser.add_argument_group("Scenario")
    scenario_group.add_argument("--k", type=int, default=None)
    scenario_group.add_argument("--nsc", type=int, default=None)
    scenario_group.add_argument("--budget", type=int, default=None)
    scenario_group.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Repeatable K,Nsc,Dtot triplet. Example: "
            "--scenario 4,256,256 --scenario 8,256,256"
        ),
    )
    scenario_group.add_argument("--antennas", type=int, default=32)
    scenario_group.add_argument("--carrier-freq", type=float, default=3.5e9)
    scenario_group.add_argument("--speed", type=float, default=1.0)
    scenario_group.add_argument("--total-power", type=float, default=1.0)
    scenario_group.add_argument("--num-subbands", type=int, default=4)

    model_group = parser.add_argument_group("Models")
    model_group.add_argument(
        "--models",
        default="proposed,csinet,swin,perfect_rzf",
        help=(
            "Comma-separated selection from proposed, csinet, swin, "
            "perfect_rzf."
        ),
    )
    model_group.add_argument(
        "--allocations",
        default="all",
        help=(
            "Proposed allocation selection: all, pure, hybrid, "
            "pure_fdma, pure_aircomp, or '24,64;0,256'."
        ),
    )
    model_group.add_argument(
        "--include-proposed-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a derived summary row for the best proposed allocation.",
    )
    model_group.add_argument(
        "--rzf-lambdas",
        default="1e-4",
        help="Comma-separated Perfect-CSI RZF regularization values.",
    )
    model_group.add_argument(
        "--rzf-reg",
        type=float,
        default=1e-3,
        help="Fallback deployed RZF regularization for baseline configs.",
    )

    snr_group = parser.add_argument_group("SNR")
    snr_group.add_argument(
        "--feedback-snr",
        default="0,10,20,25",
        help="Comma-separated feedback-SNR values in dB.",
    )
    snr_group.add_argument(
        "--downlink-snr",
        default="matched",
        help=(
            "'matched', one fixed value, or a comma-separated list in dB."
        ),
    )
    snr_group.add_argument(
        "--snr-mode",
        choices=["zip", "cartesian"],
        default="zip",
        help=(
            "Pair feedback and downlink SNR lists by position or evaluate "
            "their Cartesian product."
        ),
    )

    checkpoint_group = parser.add_argument_group("Checkpoints")
    checkpoint_group.add_argument(
        "--checkpoint",
        choices=["best", "latest"],
        default="best",
    )
    checkpoint_group.add_argument(
        "--proposal-root",
        default="runs_deployzf_v1",
    )
    checkpoint_group.add_argument(
        "--baseline-root",
        default="runs_baselines",
    )
    checkpoint_group.add_argument(
        "--baseline-objective",
        choices=["task", "reconstruction"],
        default="task",
    )
    checkpoint_group.add_argument("--proposed-checkpoint", default=None)
    checkpoint_group.add_argument("--csinet-checkpoint", default=None)
    checkpoint_group.add_argument("--swin-checkpoint", default=None)

    evaluation_group = parser.add_argument_group("Evaluation")
    evaluation_group.add_argument("--batch-size", type=int, default=8)
    evaluation_group.add_argument("--num-batches", type=int, default=50)
    evaluation_group.add_argument(
        "--extra-metrics",
        choices=["basic", "full"],
        default="basic",
        help=(
            "basic saves sum rate, NMSE, and power. full also computes "
            "cosine similarity and surrogate condition number."
        ),
    )
    evaluation_group.add_argument("--seed", type=int, default=20260729)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        default="runs_evaluation",
    )
    output_group.add_argument("--tag", default=None)
    output_group.add_argument(
        "--no-save-samples",
        action="store_true",
        help="Do not save sample-level CSV.GZ.",
    )
    output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve scenarios and checkpoints without evaluation.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("Batch size and number of batches must be positive.")

    scenarios = parse_scenarios(args)
    model_keys = [
        normalize_model_name(name)
        for name in parse_csv(args.models)
    ]
    model_keys = list(dict.fromkeys(model_keys))

    snr_points = build_snr_points(
        args.feedback_snr,
        args.downlink_snr,
        args.snr_mode,
    )

    print("=" * 100)
    print("[Unified evaluator]")
    print(f"[Scenarios] {[s.scenario_id for s in scenarios]}")
    print(f"[Models]    {model_keys}")
    print(f"[SNR pairs] {snr_points}")
    print(
        f"[Samples]   {args.num_batches} batches x "
        f"{args.batch_size} = {args.num_batches * args.batch_size}"
    )

    if args.dry_run:
        # Discovery still runs so path/config errors are visible before a long job.
        device = torch.device("cpu")
        for scenario in scenarios:
            print("\n" + "-" * 100)
            print(f"[Scenario] {scenario.scenario_id}")
            for model_key in model_keys:
                if model_key == "perfect_rzf":
                    print(
                        f"  perfect_rzf lambdas="
                        f"{parse_float_csv(args.rzf_lambdas)}"
                    )
                    continue

                explicit = explicit_checkpoint_for(
                    model_key,
                    args,
                    len(scenarios),
                )
                if explicit is not None:
                    path = explicit
                elif model_key == "proposed":
                    path = discover_checkpoint(
                        args.proposal_root,
                        model_key,
                        scenario,
                        args.checkpoint,
                        args.baseline_objective,
                    )
                else:
                    path = discover_checkpoint(
                        args.baseline_root,
                        model_key,
                        scenario,
                        args.checkpoint,
                        args.baseline_objective,
                    )
                print(f"  {model_key}: {path}")
        return 0

    torch.set_float32_matmul_precision("high")
    device = setup_gpu()
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("CUDA is unavailable.")

    all_summary_rows: List[dict] = []
    all_sample_rows: List[dict] = []

    for scenario in scenarios:
        print("\n" + "=" * 100)
        print(f"[Scenario] {scenario.scenario_id}")

        methods = load_methods_for_scenario(
            scenario,
            model_keys,
            args,
            device,
            len(scenarios),
        )

        for method in methods:
            if method.model_key == "proposed":
                print(
                    f"[Loaded] {method.display_name}, "
                    f"step={method.checkpoint_step}, "
                    f"allocations={list(method.allocations)}"
                )
            elif method.model_key == "perfect_rzf":
                print(
                    f"[Loaded] {method.display_name}"
                )
            else:
                print(
                    f"[Loaded] {method.display_name}, "
                    f"step={method.checkpoint_step}"
                )

        summary_rows, sample_rows = evaluate_scenario(
            scenario,
            methods,
            snr_points,
            args,
            device,
        )
        all_summary_rows.extend(summary_rows)
        all_sample_rows.extend(sample_rows)

        for method in methods:
            if method.model is not None:
                del method.model
        torch.cuda.empty_cache()

    print_summary(all_summary_rows)
    save_outputs(
        all_summary_rows,
        all_sample_rows,
        scenarios,
        model_keys,
        snr_points,
        args,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
