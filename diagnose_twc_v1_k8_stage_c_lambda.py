#!/usr/bin/env python3
"""Validation-only gradient-scale diagnostic for Stage-C joint-loss candidates.

Requires a completed 10k warmup. It reads validation-role samples only and
never changes model weights. Candidate multipliers are diagnostics, not an
automatic hyperparameter choice.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_system = load("utils/twc_v1_k8_system.py", "_twc_v1_k8_lambda_system")
_data = load("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_lambda_data")
K8FeedbackChain, protocol = _system.K8FeedbackChain, _system.protocol
TWCv1K8ReplayableData = _data.TWCv1K8ReplayableData


def standard_noise(batch, generator, device):
    z = torch.randn((batch, 32, 256, 2), generator=generator, device=device)
    return torch.view_as_complex(z.contiguous())


def data_for(device, batch_size):
    sizes = {key: batch_size for key in ("ce_train", "codec_train", "ce_calibration", "validation")}
    sizes["test"] = 1
    return TWCv1K8ReplayableData(base_seed=20261106, batch_sizes=sizes,
        device=str(device), cdl_model="C", speed=1.0,
        seed_namespace="twc_v1_k8_codec_warmup_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("fdma", "hybrid"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.validation_batches < 2 or args.batch_size < 1:
        raise ValueError("Use at least two validation batches and a positive batch size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    warmup_path = args.warmup_checkpoint.resolve()
    checkpoint = torch.load(warmup_path, map_location=device, weights_only=False)
    if (checkpoint.get("format") != "twc_v1_k8_codec_warmup_v1"
            or checkpoint.get("step") != 10_000
            or checkpoint.get("arm") != args.arm
            or checkpoint.get("seed") != args.seed):
        raise ValueError("Expected the matching completed Stage-B 10,000-update checkpoint")
    chain = K8FeedbackChain.from_artifacts(ROOT / "runs/twc_v1_k8_ce/seed_20261105/checkpoint_latest.pt",
        ROOT / "runs/twc_v1_k8_ce/seed_20261105/calibration.csv", device=device)
    model = chain.codec_model
    model.load_state_dict(checkpoint["codec_model"], strict=True)
    model.eval()
    data = data_for(device, args.batch_size)
    if data.protocol != checkpoint.get("data_protocol"):
        raise ValueError("Data identity differs from the warmup checkpoint")
    if args.arm == "hybrid":
        layout = protocol.default_hybrid_layout(2)
        split = protocol.EnergySplit(16.0, 12.0, 36.0)
    else:
        layout, split = protocol.FDMALayout(), None

    # R_ref is the mean validation sum-rate for this declared reference condition.
    reference_rates = []
    for bi in range(args.validation_batches):
        sample = data.sample("validation", bi)
        gen = torch.Generator(device=device).manual_seed(args.seed + 7_000_000 + bi)
        with torch.no_grad():
            out = chain(sample["H_dl"].to(device), sample["H_ul"].to(device),
                feedback_gamma_db=10.0, pilot_energy_per_ue=16.0,
                pilot_standard_noise=standard_noise(args.batch_size, gen, device),
                data_standard_noise=standard_noise(args.batch_size, gen, device),
                layout=layout, energy_split=split)
            metric = chain.metrics(out, sample["H_dl"].to(device))
        reference_rates.append(float(metric["sum_rate_bps_hz"].mean()))
    rate_reference = float(np.mean(reference_rates))
    if not np.isfinite(rate_reference) or rate_reference <= 0:
        raise FloatingPointError("Validation rate reference is not positive and finite")

    shared_parameter = model.encoder.patch_embed.proj.weight
    records = []
    for bi in range(args.validation_batches):
        sample = data.sample("validation", bi)
        Hdl, Hul = sample["H_dl"].to(device), sample["H_ul"].to(device)
        gen = torch.Generator(device=device).manual_seed(args.seed + 7_000_000 + bi)
        out = chain(Hdl, Hul, feedback_gamma_db=10.0, pilot_energy_per_ue=16.0,
            pilot_standard_noise=standard_noise(args.batch_size, gen, device),
            data_standard_noise=standard_noise(args.batch_size, gen, device),
            layout=layout, energy_split=split)
        metric = chain.metrics(out, Hdl)
        nmse = metric["dl_nmse_per_user"].mean()
        rate = metric["sum_rate_bps_hz"].mean()
        grad_nmse = torch.autograd.grad(nmse, shared_parameter, retain_graph=True)[0]
        grad_rate = torch.autograd.grad(rate, shared_parameter)[0]
        norm_nmse = float(torch.linalg.vector_norm(grad_nmse).detach())
        norm_rate = float(torch.linalg.vector_norm(grad_rate).detach())
        if not (np.isfinite(norm_nmse) and np.isfinite(norm_rate)
                and norm_nmse > 0 and norm_rate > 0):
            raise FloatingPointError(f"Invalid validation gradient scale at batch {bi}")
        match = norm_nmse * rate_reference / norm_rate
        records.append({"validation_batch": bi,
            "sample_fingerprint": sample["sample_fingerprint"],
            "dl_nmse": float(nmse.detach()), "sum_rate_bps_hz": float(rate.detach()),
            "shared_layer_nmse_gradient_norm": norm_nmse,
            "shared_layer_rate_gradient_norm": norm_rate,
            "lambda_equalizing_gradient_norms": match})
    matched = float(np.median([row["lambda_equalizing_gradient_norms"] for row in records]))
    candidates = []
    for multiplier in (0.1, 0.25, 0.5, 1.0):
        lam = matched * multiplier
        ratios = [lam * row["shared_layer_rate_gradient_norm"] / rate_reference
                  / row["shared_layer_nmse_gradient_norm"] for row in records]
        candidates.append({"multiplier_of_median_gradient_match": multiplier,
            "lambda_candidate": lam,
            "rate_to_nmse_shared_layer_gradient_ratio_by_batch": ratios})
    digest = hashlib.sha256(warmup_path.read_bytes()).hexdigest()
    result = {"status": "VALIDATION_ONLY_GRADIENT_SCALE_DIAGNOSTIC",
        "arm": args.arm, "seed": args.seed, "warmup_step": 10_000,
        "warmup_checkpoint": str(warmup_path), "warmup_checkpoint_sha256": digest,
        "device": str(device), "role": "validation", "test_used": False,
        "optimizer_updates": 0, "reference_condition": {"gamma_db": 10.0,
            "pilot_energy_per_ue": 16.0, "group_count": 2 if args.arm == "hybrid" else 1,
            "energy_split": split.to_dict() if split is not None else None},
        "rate_reference_bps_hz": rate_reference,
        "rate_reference_method": "mean sum-rate over the declared fixed validation batches",
        "shared_parameter": "codec_model.encoder.patch_embed.proj.weight",
        "batch_diagnostics": records, "candidate_lambdas": candidates,
        "selection_note": "Candidates are validation diagnostics only; freeze one before final test. No automatic choice was made."}
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rate_reference_bps_hz": rate_reference,
        "median_gradient_match_lambda": matched, "candidate_lambdas": candidates,
        "test_used": False, "optimizer_updates": 0, "output": str(args.output)},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
