#!/usr/bin/env python3
"""CPU-only smoke path for the TWC 1.0 K8 physical feedback-link prototype."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch

from configs.twc_v1_k8_protocol_config import (
    EVALUATION_FEEDBACK_GAMMA_DB,
    TWCv1K8Config,
    all_groupings,
    candidate_energy_splits,
)


ROOT = Path(__file__).resolve().parent


def load_link():
    name = "_twc_v1_k8_link_preflight"
    path = ROOT / "utils" / "twc_v1_k8_link.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load K8 link module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def qpsk(shape, generator):
    real = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).float() * 2 - 1
    imag = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).float() * 2 - 1
    return torch.complex(real, imag) / (2.0 ** 0.5)


def complex_noise(shape, generator):
    real = torch.randn(shape, generator=generator, dtype=torch.float32)
    imag = torch.randn(shape, generator=generator, dtype=torch.float32)
    return torch.complex(real, imag)


def main():
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(771923)
    link = load_link()
    batch = 2
    H_ul = torch.complex(
        torch.randn((batch, 8, 32, 256), generator=generator),
        torch.randn((batch, 8, 32, 256), generator=generator),
    ) / (2.0 ** 0.5)
    gamma = EVALUATION_FEEDBACK_GAMMA_DB[2]
    # Ep is public/gamma-conditioned before pilots; Ekey/EAS are set after LS.
    pilot_energy = 16.0
    layout = TWCv1K8Config(group_count=2).layout

    pilot_noise = complex_noise((batch, 32, 256), generator)
    pilot_y = link.transmit_pilots(H_ul, pilot_energy, gamma, pilot_noise)
    H_ul_est = link.estimate_ul_ls_interpolated(pilot_y, pilot_energy)
    # A single candidate is fixed for this smoke path only, after pilot reception.
    split = next(split for split in candidate_energy_splits()
                 if split.pilot == pilot_energy and split.key / split.shared == 1.0 / 3.0)

    layout_results = []
    max_data_energy_error = 0.0
    for group_count in (1, 2, 4):
        layout = TWCv1K8Config(group_count=group_count).layout
        key_symbols = qpsk((batch, 8, 4), generator)
        shared_symbols = qpsk((batch, 8, layout.shared_dimension), generator)
        data_noise = complex_noise((batch, 32, 256), generator)
        transmitted = link.transmit_feedback_data(
            H_ul, layout, split, gamma, key_symbols, shared_symbols, data_noise)
        recovered = link.recover_feedback_data(
            transmitted["Y"], H_ul_est, layout, split, gamma)

        expected_data_energy = split.key + split.shared
        energy_error = float((transmitted["actual_data_energy"] - expected_data_energy).abs().max())
        max_data_energy_error = max(max_data_energy_error, energy_error)
        if energy_error > 1e-4:
            raise AssertionError(f"G={group_count} data energy mismatch: {energy_error}")
        for name, tensor in (("Y", transmitted["Y"]), ("X", transmitted["X"]),
                             ("key_symbols_hat", recovered["key_symbols_hat"]),
                             ("shared_symbols_hat", recovered["shared_symbols_hat"])):
            if not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"G={group_count} nonfinite output at {name}")
        layout_results.append({
            "group_count": group_count,
            "groups": [list(group) for group in layout.groups],
            "shared_dimension_per_group": layout.shared_dimension,
            "data_coordinates_per_user": layout.per_user_data_coordinates,
            "total_occupied_re": layout.total_occupied_re,
            "max_data_energy_error": energy_error,
            "recovered_key_shape": list(recovered["key_symbols_hat"].shape),
            "recovered_shared_shape": list(recovered["shared_symbols_hat"].shape),
        })

    summary = {
        "status": "CPU_SMOKE_PASS",
        "device": "cpu",
        "batch": batch,
        "layouts_run": layout_results,
        "energy_candidate_under_test_only": split.to_dict(),
        "feedback_gamma_db": gamma,
        "absolute_N0": transmitted["N0"],
        "pilot_estimator": "LS + linear interpolation placeholder (not U-Net)",
        "max_per_ue_data_energy_error": max_data_energy_error,
        "pilot_estimate_shape": list(H_ul_est.shape),
        "grouping_counts_G1_G2_G4": [len(all_groupings(g)) for g in (1, 2, 4)],
        "candidate_energy_split_count": len(candidate_energy_splits()),
        "downlink_CSI_reconstruction_or_precoding": "not part of this smoke path",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
