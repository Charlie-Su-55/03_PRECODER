#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""UE-side parameter and MAC complexity evaluation.

Reported MACs cover learned neural operations:
- Linear layers
- Conv2d / ConvTranspose2d
- Multi-head attention projections and QK/AV products
- Swin window-attention QK/AV products
- VQ nearest-codeword search

Fixed transforms such as FFT/DFT, normalization, activations, softmax,
top-k, bit packing, LDPC, and QAM are not included.

One MAC is approximately two FLOPs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

import baseline_evaluate as base_eval
from models.adaptive_hybrid import AdaptiveHybridPrecoder
from models.csinet_plus_precoder import CsiNetPlus_E2E_Precoder
from models.swin_precoder import Swin_E2E_Precoder
from train_tub_vqsscc_baseline import TUBVQSSCC


def unique_parameter_count(modules, trainable_only=False):
    seen = set()
    total = 0
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if trainable_only and not p.requires_grad:
                continue
            total += p.numel()
    return total


def descendants(modules):
    out = {}
    for root in modules:
        if root is None:
            continue
        for module in root.modules():
            out[id(module)] = module
    return list(out.values())


class MACCounter:
    def __init__(self, roots):
        self.roots = [m for m in roots if m is not None]
        self.macs = 0
        self.handles = []

        modules = descendants(self.roots)
        self.mha_out_proj_ids = {
            id(m.out_proj) for m in modules if isinstance(m, nn.MultiheadAttention)
        }

        for module in modules:
            if isinstance(module, nn.MultiheadAttention):
                self.handles.append(module.register_forward_hook(self._mha_hook))
            elif isinstance(module, nn.Linear) and id(module) not in self.mha_out_proj_ids:
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv2d_hook))
            elif isinstance(module, nn.ConvTranspose2d):
                self.handles.append(module.register_forward_hook(self._convtranspose2d_hook))
            elif module.__class__.__name__ == "WindowAttention":
                self.handles.append(module.register_forward_hook(self._window_attention_hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _linear_hook(self, module, inputs, output):
        if not torch.is_tensor(output):
            return
        self.macs += int(output.numel()) * int(module.in_features)

    def _conv2d_hook(self, module, inputs, output):
        if not torch.is_tensor(output):
            return
        kh, kw = module.kernel_size
        kernel_mul = (module.in_channels // module.groups) * kh * kw
        self.macs += int(output.numel()) * int(kernel_mul)

    def _convtranspose2d_hook(self, module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        kh, kw = module.kernel_size
        per_input = (module.out_channels // module.groups) * kh * kw
        self.macs += int(x.numel()) * int(per_input)

    def _mha_hook(self, module, inputs, output):
        if len(inputs) < 3:
            return

        q, k, v = inputs[:3]
        if not all(torch.is_tensor(x) and x.ndim == 3 for x in (q, k, v)):
            return

        if module.batch_first:
            bq, lq, dq = q.shape
            bk, lk, dk = k.shape
            bv, lv, dv = v.shape
        else:
            lq, bq, dq = q.shape
            lk, bk, dk = k.shape
            lv, bv, dv = v.shape

        if not (bq == bk == bv):
            raise RuntimeError("Unexpected MHA batch-size mismatch.")

        e = int(module.embed_dim)
        kdim = int(module.kdim if module.kdim is not None else e)
        vdim = int(module.vdim if module.vdim is not None else e)

        # Q, K, V projections.
        proj = bq * (
            lq * dq * e
            + lk * kdim * e
            + lv * vdim * e
        )

        # QK^T and attention-weighted V.
        attn = 2 * bq * lq * lk * e

        # Output projection.
        out_proj = bq * lq * e * e

        self.macs += int(proj + attn + out_proj)

    def _window_attention_hook(self, module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return

        b, n, c = x.shape

        # q@k^T + attention@v.
        self.macs += int(2 * b * n * n * c)


def run_profile(roots, forward_fn):
    counter = MACCounter(roots)
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        counter.close()
    return counter.macs


def load_proposed(path, scenario, device):
    checkpoint = base_eval.load_checkpoint_cpu(Path(path))
    cfg = base_eval.proposal_runtime_config(scenario, checkpoint)

    model = AdaptiveHybridPrecoder(cfg).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()

    return model, cfg


def load_baseline(path, model_name, scenario, device):
    checkpoint = base_eval.load_checkpoint_cpu(Path(path))
    cfg = base_eval.baseline_runtime_config(
        scenario,
        model_name,
        checkpoint,
        rzf_reg=0.003,
    )

    if model_name == "csinet":
        model = CsiNetPlus_E2E_Precoder(cfg).to(device)
    elif model_name == "swin":
        model = Swin_E2E_Precoder(cfg).to(device)
    else:
        raise ValueError(model_name)

    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()

    return model, cfg


def load_vqsscc(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    model = TUBVQSSCC(
        base=int(cfg["base"]),
        c_lat=int(cfg["c_lat"]),
        num_embeddings=int(cfg["num_embeddings"]),
        keep_k=int(cfg["keep_k"]),
    ).to(device)

    model.load_state_dict(checkpoint["state"], strict=True)
    model.vq.initialized = True
    model.eval()

    return model, cfg


def find_attr(model, names):
    for name in names:
        if hasattr(model, name):
            value = getattr(model, name)
            if isinstance(value, nn.Module):
                return value
    return None


def proposed_result(model, cfg, d_f, d_a, snr_db, device):
    if d_f % cfg.NUM_SUBBANDS != 0 or d_a % cfg.NUM_SUBBANDS != 0:
        raise ValueError("Proposed allocation must divide evenly across subbands.")

    d_f_sb = d_f // cfg.NUM_SUBBANDS
    d_a_sb = d_a // cfg.NUM_SUBBANDS

    h = torch.randn(
        1,
        cfg.ANTENNAS,
        cfg.SUBCARRIERS,
        dtype=torch.complex64,
        device=device,
    )

    roots = [model.encoder, model.cond_embed]

    def forward_fn():
        cond = model.cond_embed(d_f, d_a, snr_db, device)
        model.encoder(h, d_f_sb, d_a_sb, cond)

    macs = run_profile(roots, forward_fn)

    return {
        "method": "Proposed",
        "operating_point": f"({d_f},{d_a})",
        "params": unique_parameter_count(roots),
        "trainable_params": unique_parameter_count(roots, trainable_only=True),
        "macs": macs,
        "note": "dual-branch UE encoder + condition embedding",
    }


def baseline_result(model, cfg, method_name, device):
    encoder = model.encoder
    x = torch.randn(
        1,
        2,
        cfg.CSINET_NC if method_name == "CsiNet+" else cfg.SWIN_NC,
        cfg.ANTENNAS,
        device=device,
    )

    macs = run_profile([encoder], lambda: encoder(x))

    return {
        "method": method_name,
        "operating_point": f"FDMA Df={cfg.D_F}",
        "params": unique_parameter_count([encoder]),
        "trainable_params": unique_parameter_count([encoder], trainable_only=True),
        "macs": macs,
        "note": "single shared per-UE encoder",
    }


def vqsscc_result(model, cfg, device):
    encoder = find_attr(model, ["encoder", "enc"])
    samm = find_attr(model, ["samm", "sam"])
    token_norm = find_attr(model, ["token_norm", "norm_feature", "latent_norm"])
    vq = find_attr(model, ["vq", "quantizer"])

    if encoder is None or samm is None or vq is None:
        available = [name for name, _ in model.named_children()]
        raise RuntimeError(
            "Could not identify VQ-SSCC UE modules. "
            f"Top-level children are: {available}"
        )

    roots = [encoder, samm, vq]
    if token_norm is not None:
        roots.append(token_norm)

    nc = int(cfg["nc"])
    nt = int(cfg["nt"])
    x = torch.randn(1, 2, nc, nt, device=device)

    # Run the normal sparse source path. Hooks are attached only to UE modules,
    # so ADM and decoder operations are not included in the MAC count.
    def forward_fn():
        model.sparse_forward(x, random_mask=False)

    macs = run_profile(roots, forward_fn)

    # Explicit VQ nearest-codeword search:
    # keep_k queries x J codewords x c_lat multiply-accumulates.
    keep_k = int(cfg["keep_k"])
    num_embeddings = int(cfg["num_embeddings"])
    c_lat = int(cfg["c_lat"])
    vq_search_macs = keep_k * num_embeddings * c_lat
    macs += vq_search_macs

    return {
        "method": "VQ-SSCC",
        "operating_point": f"keep={keep_k}",
        "params": unique_parameter_count(roots),
        "trainable_params": unique_parameter_count(roots, trainable_only=True),
        "macs": macs,
        "note": f"encoder + SAMM + VQ; includes {vq_search_macs} VQ-search MACs",
    }


def print_table(rows):
    print()
    print("=" * 112)
    print("UE-side complexity | one UE, one CSI feedback instance")
    print("=" * 112)
    print(
        f"{'Method':<16}"
        f"{'Point':<18}"
        f"{'Params(M)':>12}"
        f"{'Trainable(M)':>15}"
        f"{'MACs(M)':>14}"
        f"{'FLOPs(M)':>14}"
    )
    print("-" * 112)

    for row in rows:
        print(
            f"{row['method']:<16}"
            f"{row['operating_point']:<18}"
            f"{row['params']/1e6:>12.3f}"
            f"{row['trainable_params']/1e6:>15.3f}"
            f"{row['macs']/1e6:>14.3f}"
            f"{2*row['macs']/1e6:>14.3f}"
        )

    print("-" * 112)
    print("MAC definition: dominant learned multiply-accumulate operations.")
    print("1 MAC ~= 2 FLOPs.")
    print("Fixed FFT/DFT, normalization, activations, top-k, bit packing, LDPC and QAM are excluded.")
    print("=" * 112)


def save_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "method",
        "operating_point",
        "params",
        "trainable_params",
        "params_m",
        "trainable_params_m",
        "macs",
        "macs_m",
        "flops",
        "flops_m",
        "note",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            record = dict(row)
            record["params_m"] = row["params"] / 1e6
            record["trainable_params_m"] = row["trainable_params"] / 1e6
            record["macs_m"] = row["macs"] / 1e6
            record["flops"] = 2 * row["macs"]
            record["flops_m"] = 2 * row["macs"] / 1e6
            writer.writerow(record)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--csinet", required=True)
    parser.add_argument("--swin", required=True)
    parser.add_argument("--vqsscc", required=True)
    parser.add_argument(
        "--output",
        default="runs_evaluation/ue_complexity/k8_n128_d128/ue_complexity.csv",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)

    scenario = base_eval.Scenario(
        k_users=8,
        subcarriers=128,
        feedback_budget=128,
        antennas=32,
        carrier_freq=3.5e9,
        speed=1.0,
        total_power=1.0,
        num_subbands=4,
    )

    proposed, proposed_cfg = load_proposed(args.proposed, scenario, device)
    csinet, csinet_cfg = load_baseline(args.csinet, "csinet", scenario, device)
    swin, swin_cfg = load_baseline(args.swin, "swin", scenario, device)
    vqsscc, vq_cfg = load_vqsscc(args.vqsscc, device)

    rows = []

    # Main paper operating point.
    rows.append(proposed_result(proposed, proposed_cfg, 4, 96, 25.0, device))

    # Endpoint diagnostics. Same stored model, different active branch compute.
    rows.append(proposed_result(proposed, proposed_cfg, 16, 0, 25.0, device))
    rows.append(proposed_result(proposed, proposed_cfg, 0, 128, 25.0, device))

    rows.append(baseline_result(csinet, csinet_cfg, "CsiNet+", device))
    rows.append(baseline_result(swin, swin_cfg, "Swin", device))
    rows.append(vqsscc_result(vqsscc, vq_cfg, device))

    print_table(rows)
    save_csv(rows, args.output)

    print()
    print("Sanity checks:")
    print("  Proposed expected UE params ~10.871 M")
    print("  CsiNet+ expected UE params  ~0.068 M")
    print("  Swin expected UE params     ~0.806 M")
    print("  VQ-SSCC expected UE params  ~0.616 M")
    print()
    print("If the parameter counts match these values, the module scoping is correct.")


if __name__ == "__main__":
    main()