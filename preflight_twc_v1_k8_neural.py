#!/usr/bin/env python3
"""CPU-only interface smoke for the K8 U-Net and variable-G Swin codec."""
from __future__ import annotations

import json

import torch

from models.twc_v1_k8_neural import K8PilotResidualUNet, K8SwinCSIModel
from utils.twc_v1_k8_protocol import (
    N_ANTENNAS, N_SUBCARRIERS, PILOT_POSITIONS_BY_USER, shared_dimension,
)


def main():
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(914207)
    batch = 1
    shape = (batch, 8, N_ANTENNAS, N_SUBCARRIERS)
    H_initial = torch.complex(torch.randn(shape, generator=generator),
                              torch.randn(shape, generator=generator))
    pilots = torch.zeros(batch, 8, N_SUBCARRIERS, dtype=torch.complex64)
    mask = torch.zeros(8, N_SUBCARRIERS, dtype=torch.bool)
    for user, positions in enumerate(PILOT_POSITIONS_BY_USER):
        mask[user, list(positions)] = True
        pilots[:, user, list(positions)] = 1.0 + 0.0j
    Y = torch.complex(torch.randn(batch, N_ANTENNAS, N_SUBCARRIERS, generator=generator),
                      torch.randn(batch, N_ANTENNAS, N_SUBCARRIERS, generator=generator))

    unet = K8PilotResidualUNet().eval()
    with torch.no_grad():
        H_hat = unet(Y, pilots, H_initial, noise_variance=0.025)
        Y_changed = Y.clone()
        Y_changed[:, :, ~mask.any(0)] = complex(float("nan"), float("nan"))
        H_hat_data_nan = unet(Y_changed, pilots, H_initial, noise_variance=0.025)
    if not torch.equal(H_hat, H_hat_data_nan):
        raise AssertionError("Data-only Y positions affected the pilot U-Net output")

    codec = K8SwinCSIModel().eval()
    codec_results = []
    source = torch.complex(torch.randn(shape, generator=generator),
                           torch.randn(shape, generator=generator))
    with torch.no_grad():
        for group_count in (1, 2, 4):
            key, shared = codec.encode(source, group_count=group_count)
            reconstructed = codec.decode(key, shared, group_count=group_count)
            expected = (batch, 8, N_ANTENNAS, N_SUBCARRIERS)
            if tuple(reconstructed.shape) != expected or not bool(torch.isfinite(reconstructed).all()):
                raise AssertionError(f"Invalid Swin output for G={group_count}")
            if shared.shape[-1] != shared_dimension(group_count):
                raise AssertionError(f"Invalid shared width for G={group_count}")
            codec_results.append({
                "G": group_count,
                "key_shape": list(key.shape),
                "shared_shape": list(shared.shape),
                "reconstructed_shape": list(reconstructed.shape),
            })

    print(json.dumps({
        "status": "K8_NEURAL_CPU_SMOKE_PASS",
        "device": "cpu",
        "K": 8,
        "pilot_RE": int(mask.sum()),
        "pilot_U_Net_shape": list(H_hat.shape),
        "data_only_NaN_isolation": "pass",
        "noise_variance": 0.025,
        "feedback_gamma_dB": 10,
        "swin_by_group": codec_results,
        "training_or_performance_evidence": False,
        "joint_receiver_and_RZF": "not included",
    }, indent=2))


if __name__ == "__main__":
    main()
