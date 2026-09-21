#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count per-UE encoder parameters for:

1. Proposed Hybrid FDMA-AirComp encoder
2. CsiNet+ pure-FDMA encoder
3. Swin-CFNet pure-FDMA encoder

The comparison uses:
    K = 16
    Nsc = 256
    D_tot = 256

Only one UE encoder is counted. BS-side decoders, cross-user processing,
and RZF processing are excluded.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from baseline_evaluate import Scenario, baseline_runtime_config
from configs.train_config import get_suite
from models.adaptive_hybrid import AdaptiveHybridPrecoder
from models.csinet_plus_precoder import CsiNetPlus_E2E_Precoder
from models.swin_precoder import Swin_E2E_Precoder


def count_unique_trainable_parameters(
    modules: Iterable[nn.Module | None],
) -> int:
    """Count unique trainable parameters across several modules."""

    seen: set[int] = set()
    total = 0

    for module in modules:
        if module is None:
            continue

        for parameter in module.parameters():
            parameter_id = id(parameter)

            if parameter.requires_grad and parameter_id not in seen:
                seen.add(parameter_id)
                total += parameter.numel()

    return total


def extract_single_ue_encoder(
    model: nn.Module,
    model_name: str,
) -> nn.Module:
    """
    Extract one logical UE encoder.

    Supports both:
      - model.encoder
      - model.encoders[0]
    """

    for attribute_name in (
        "encoder",
        "ue_encoder",
    ):
        module = getattr(model, attribute_name, None)

        if isinstance(module, nn.Module):
            return module

    for attribute_name in (
        "encoders",
        "ue_encoders",
    ):
        container = getattr(model, attribute_name, None)

        if container is None:
            continue

        if isinstance(container, (nn.ModuleList, list, tuple)):
            if len(container) == 0:
                raise RuntimeError(
                    f"{model_name}.{attribute_name} is empty."
                )

            module = container[0]

            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"{model_name}.{attribute_name}[0] is not an nn.Module."
                )

            return module

    available_children = [
        name for name, _ in model.named_children()
    ]

    raise AttributeError(
        f"Cannot locate the UE encoder in {model_name}. "
        f"Available child modules: {available_children}"
    )


def select_proposal_config():
    """Select the K=16, Nsc=256, Dtot=256 adaptive proposal config."""

    matching_configs = [
        cfg
        for cfg in get_suite(
            "proposal_adaptive",
            output_root="runs_deployzf_v1",
        )
        if cfg.K_USERS == 16
        and cfg.SUBCARRIERS == 256
        and cfg.D_TOT == 256
    ]

    if len(matching_configs) != 1:
        names = [
            cfg.EXP_NAME for cfg in matching_configs
        ]
        raise RuntimeError(
            "Expected exactly one K=16, Nsc=256, Dtot=256 "
            f"proposal config, but found {len(matching_configs)}: {names}"
        )

    return matching_configs[0]


def main() -> None:
    torch.set_float32_matmul_precision("high")

    # ============================================================
    # 1. Proposed model
    # ============================================================

    proposal_cfg = select_proposal_config()
    proposal_model = AdaptiveHybridPrecoder(proposal_cfg)

    proposal_encoder = extract_single_ue_encoder(
        proposal_model,
        "AdaptiveHybridPrecoder",
    )

    proposal_condition = getattr(
        proposal_model,
        "cond_embed",
        None,
    )

    proposal_encoder_params = count_unique_trainable_parameters(
        [proposal_encoder]
    )

    proposal_condition_params = count_unique_trainable_parameters(
        [proposal_condition]
    )

    proposal_total_ue_params = count_unique_trainable_parameters(
        [
            proposal_encoder,
            proposal_condition,
        ]
    )

    # ============================================================
    # 2. Baseline-compatible runtime configs
    # ============================================================

    scenario = Scenario(
        k_users=16,
        subcarriers=256,
        feedback_budget=256,
        antennas=32,
        carrier_freq=3.5e9,
        speed=1.0,
        total_power=1.0,
        num_subbands=4,
    )
    scenario.validate()

    # Empty checkpoint dictionaries are sufficient because parameter
    # counting only requires the architecture configuration.
    csinet_cfg = baseline_runtime_config(
        scenario=scenario,
        model_key="csinet",
        checkpoint={},
        rzf_reg=1e-3,
    )

    swin_cfg = baseline_runtime_config(
        scenario=scenario,
        model_key="swin",
        checkpoint={},
        rzf_reg=1e-3,
    )

    # Confirm that the compatibility aliases are present.
    assert csinet_cfg.D_F == 16
    assert csinet_cfg.DIM_FDMA_PER_USER == 16
    assert swin_cfg.D_F == 16
    assert swin_cfg.DIM_FDMA_PER_USER == 16
    assert swin_cfg.SWIN_CODEWORD_DIM == 32

    # ============================================================
    # 3. Instantiate baselines and extract one UE encoder
    # ============================================================

    csinet_model = CsiNetPlus_E2E_Precoder(csinet_cfg)
    swin_model = Swin_E2E_Precoder(swin_cfg)

    csinet_encoder = extract_single_ue_encoder(
        csinet_model,
        "CsiNetPlus_E2E_Precoder",
    )

    swin_encoder = extract_single_ue_encoder(
        swin_model,
        "Swin_E2E_Precoder",
    )

    csinet_ue_params = count_unique_trainable_parameters(
        [csinet_encoder]
    )

    swin_ue_params = count_unique_trainable_parameters(
        [swin_encoder]
    )

    # ============================================================
    # 4. Report
    # ============================================================

    print()
    print("=" * 78)
    print("Per-UE encoder parameter count")
    print("Scenario: K=16, Nsc=256, Dtot=256, Df=16 for pure FDMA")
    print("BS-side modules are excluded.")
    print("=" * 78)

    rows = [
        (
            "Proposed feedback encoder",
            proposal_encoder_params,
        ),
        (
            "Proposed condition embedding",
            proposal_condition_params,
        ),
        (
            "Proposed total per UE",
            proposal_total_ue_params,
        ),
        (
            "CsiNet+ per-UE encoder",
            csinet_ue_params,
        ),
        (
            "Swin-CFNet per-UE encoder",
            swin_ue_params,
        ),
    ]

    for name, value in rows:
        print(
            f"{name:<36}: "
            f"{value:>12,d} parameters "
            f"({value / 1e6:>9.6f} M)"
        )

    print("-" * 78)
    print(
        "Extracted CsiNet+ encoder class : "
        f"{type(csinet_encoder).__name__}"
    )
    print(
        "Extracted Swin-CFNet encoder class: "
        f"{type(swin_encoder).__name__}"
    )
    print(
        "Extracted proposed encoder class  : "
        f"{type(proposal_encoder).__name__}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()