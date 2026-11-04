#!/usr/bin/env python3
"""Small CPU CDL screen of pilot-LS initialization methods (not final evidence)."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import torch

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


link = _load("utils/twc_v1_k8_link.py", "_twc_v1_k8_ls_screen_link")
data_module = _load("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_ls_screen_data")
protocol = link.protocol
estimate_ul_ls_fourier = link.estimate_ul_ls_fourier
estimate_ul_ls_interpolated = link.estimate_ul_ls_interpolated
TWCv1K8ReplayableData = data_module.TWCv1K8ReplayableData
EVALUATION_FEEDBACK_GAMMA_DB = protocol.EVALUATION_FEEDBACK_GAMMA_DB
N_ANTENNAS = protocol.N_ANTENNAS
N_SUBCARRIERS = protocol.N_SUBCARRIERS
PILOT_POSITIONS_BY_USER = protocol.PILOT_POSITIONS_BY_USER
feedback_noise_variance = protocol.feedback_noise_variance
if "utils" in sys.modules:
    raise AssertionError("LS screen must not execute the side-effectful utils package initializer")


def noise_for(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parts = torch.randn((*shape, 2), generator=generator, dtype=torch.float32)
    return torch.view_as_complex(parts.contiguous())


def nmse(estimator, H_true):
    per_link_error = (estimator - H_true).abs().square().sum(dim=(2, 3))
    per_link_power = H_true.abs().square().sum(dim=(2, 3)).clamp_min(1e-20)
    per_link = per_link_error / per_link_power
    return float(per_link.mean()), float(per_link.mean().clamp_min(1e-20).log10() * 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path,
        default=Path("docs/evidence/TWC_V1_K8_LS_validation_screen_20261104.csv"))
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("batch must be positive")

    torch.set_num_threads(1)
    sizes = {"ce_train": 1, "codec_train": 1, "ce_calibration": 1,
             "validation": args.batch, "test": 1}
    data = TWCv1K8ReplayableData(base_seed=args.seed, batch_sizes=sizes,
                                 device="cpu", cdl_model="C")
    validation = data.sample("validation", 0)
    H_ul = validation["H_ul"]
    rows = []
    for gamma_index, gamma in enumerate(EVALUATION_FEEDBACK_GAMMA_DB):
        n0 = feedback_noise_variance(gamma)
        standard = noise_for((args.batch, N_ANTENNAS, N_SUBCARRIERS),
                             args.seed + 10_000 + gamma_index)
        for ep in (8.0, 16.0, 24.0):
            observed = torch.zeros_like(standard)
            amplitude = (ep / 8.0) ** 0.5
            for user, positions in enumerate(PILOT_POSITIONS_BY_USER):
                q = torch.tensor(positions, dtype=torch.long)
                observed[:, :, q] = (amplitude * H_ul[:, user, :, q]
                    + (n0 / 2.0) ** 0.5 * standard[:, :, q])
            candidates = [
                ("linear_interpolation", None, None,
                 estimate_ul_ls_interpolated(observed, ep)),
                ("fourier_delay_ls", 4, 1e-3,
                 estimate_ul_ls_fourier(observed, ep, rank=4, ridge=1e-3)),
                ("fourier_delay_ls", 8, 1e-3,
                 estimate_ul_ls_fourier(observed, ep, rank=8, ridge=1e-3)),
            ]
            for method, rank, ridge, estimate in candidates:
                ratio, db = nmse(estimate, H_ul)
                rows.append(dict(data_id=data.protocol["data_id"], split="validation",
                    validation_role="small_cpu_design_screen_not_formal_CE_validation",
                    base_seed=args.seed, sample_index=0, sample_count=args.batch,
                    gamma_db=gamma, N0=n0, Ep_per_UE=ep,
                    pilot_energy_per_RE=ep / 8.0, method=method, delay_rank=rank,
                    normalized_gram_ridge=ridge, mean_per_link_NMSE=ratio,
                    mean_per_link_NMSE_dB=db, optimizer_updates=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = {}
    for row in rows:
        key = (row["method"], row["delay_rank"])
        aggregates.setdefault(key, []).append(row["mean_per_link_NMSE"])
    ranking = [{"method": key[0], "delay_rank": key[1],
                "equal_weight_mean_NMSE_ratio_over_18_conditions": sum(values) / len(values),
                "conditions": len(values)}
               for key, values in aggregates.items()]
    ranking.sort(key=lambda item: item["equal_weight_mean_NMSE_ratio_over_18_conditions"])
    report = dict(status="K8_LS_METHOD_DESIGN_SCREEN_PASS", device="cpu",
        data_protocol=data.protocol, channels=validation["sample_fingerprint"],
        H_ul_shape=list(H_ul.shape), gamma_points=list(EVALUATION_FEEDBACK_GAMMA_DB),
        Ep_candidates=[8, 16, 24], rows=len(rows), optimizer_updates=0,
        methods=ranking, csv_path=str(args.output.resolve()),
        interpretation="Small fixed validation design screen; not formal CE validation, not test, not a performance claim")
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
