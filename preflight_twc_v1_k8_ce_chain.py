#!/usr/bin/env python3
"""CPU-only wiring preflight: real K8 CDL pilots -> LS -> pilot-only U-Net."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import torch

from models.twc_v1_k8_neural import K8PilotResidualUNet

ROOT = Path(__file__).resolve().parent


def _load(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


link = _load("utils/twc_v1_k8_link.py", "_twc_v1_k8_ce_chain_link")
data_module = _load("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_ce_chain_data")
protocol = link.protocol
estimate_ul_ls_interpolated = link.estimate_ul_ls_interpolated
transmit_pilots = link.transmit_pilots
TWCv1K8ReplayableData = data_module.TWCv1K8ReplayableData
EVALUATION_FEEDBACK_GAMMA_DB = protocol.EVALUATION_FEEDBACK_GAMMA_DB
N_ANTENNAS = protocol.N_ANTENNAS
N_SUBCARRIERS = protocol.N_SUBCARRIERS
PILOT_POSITIONS_BY_USER = protocol.PILOT_POSITIONS_BY_USER
feedback_noise_variance = protocol.feedback_noise_variance
if "utils" in sys.modules:
    raise AssertionError("CE chain preflight must not execute the side-effectful utils package initializer")


def standard_noise(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parts = torch.randn((*shape, 2), generator=generator, dtype=torch.float32)
    return torch.view_as_complex(parts.contiguous())


def main():
    torch.set_num_threads(1)
    seed = 20261105
    sizes = {"ce_train": 1, "codec_train": 1, "ce_calibration": 1,
             "validation": 2, "test": 1}
    data = TWCv1K8ReplayableData(base_seed=seed, batch_sizes=sizes, device="cpu", cdl_model="C")
    sample = data.sample("validation", 0)
    batch = sample["H_ul"].shape[0]
    pilot_symbols = torch.zeros(batch, 8, N_SUBCARRIERS, dtype=torch.complex64)
    for user, positions in enumerate(PILOT_POSITIONS_BY_USER):
        pilot_symbols[:, user, list(positions)] = 1.0 + 0.0j
    estimator = K8PilotResidualUNet().eval()
    cases = []
    with torch.no_grad():
        for gamma_index, gamma in enumerate(EVALUATION_FEEDBACK_GAMMA_DB):
            n0 = feedback_noise_variance(gamma)
            noise = standard_noise((batch, N_ANTENNAS, N_SUBCARRIERS),
                                   seed + 50_000 + gamma_index)
            for ep in (8.0, 16.0, 24.0):
                observed = transmit_pilots(sample["H_ul"], ep, gamma, noise)
                initial = estimate_ul_ls_interpolated(observed, ep)
                corrected = estimator(observed, pilot_symbols, initial,
                                      pilot_energy_per_ue=ep, noise_variance=n0)
                if (tuple(initial.shape) != (batch, 8, 32, 256)
                        or tuple(corrected.shape) != (batch, 8, 32, 256)
                        or not bool(torch.isfinite(initial).all() & torch.isfinite(corrected).all())):
                    raise AssertionError(f"Invalid LS/U-Net output at gamma={gamma}, Ep={ep}")
                cases.append(dict(gamma_db=gamma, Ep=ep, N0=n0,
                    observed_shape=list(observed.shape), LS_shape=list(initial.shape),
                    UNet_shape=list(corrected.shape), finite=True))
    print(json.dumps(dict(
        status="K8_REAL_CDL_LS_UNET_CPU_CHAIN_PASS",
        device="cpu", data_id=data.protocol["data_id"],
        validation_pair_fingerprint=sample["sample_fingerprint"],
        validation_frames=batch, pilot_RE=64, cases=len(cases),
        LS_initialization="pilot LS + linear interpolation, selected for first CE run by 32-frame validation design screen",
        U_Net="untrained interface forward only; true H_UL was used to synthesize Y but never passed to LS/U-Net",
        cases_detail=cases,
        optimizer_updates=0, nmse_claim=False,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
