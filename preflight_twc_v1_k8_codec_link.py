#!/usr/bin/env python3
"""CPU-only end-to-end K8 feedback-chain preflight; performs no optimizer updates."""
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

system = load("utils/twc_v1_k8_system.py", "_twc_v1_k8_preflight_system")
data_module = load("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_preflight_data")
K8FeedbackChain, protocol = system.K8FeedbackChain, system.protocol
TWCv1K8ReplayableData = data_module.TWCv1K8ReplayableData


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if torch.cuda.is_initialized():
        raise RuntimeError("This preflight must remain CPU-only")
    sizes = {"ce_train": 1, "codec_train": 1, "ce_calibration": 1,
             "validation": 1, "test": 1}
    data = TWCv1K8ReplayableData(base_seed=20261104, batch_sizes=sizes,
        device="cpu", cdl_model="C", seed_namespace="twc_v1_k8_codec_preflight")
    sample = data.sample("codec_train", 1)
    H_dl, H_ul = sample["H_dl"], sample["H_ul"]
    ce_ckpt = ROOT / "runs/twc_v1_k8_ce/seed_20261105/checkpoint_latest.pt"
    calibration = ROOT / "runs/twc_v1_k8_ce/seed_20261105/calibration.csv"
    chain = K8FeedbackChain.from_artifacts(ce_ckpt, calibration, device="cpu",
        swin_kwargs={"embed_dim": 32, "blocks": 1, "heads": 4,
                     "patch_size": 4, "window_size": 4})
    gen = torch.Generator(device="cpu").manual_seed(20261104)
    def noise():
        return torch.view_as_complex(torch.randn((1, 32, 256, 2), generator=gen).contiguous())

    outcomes = []
    for groups in (1, 2, 4):
        layout = protocol.HybridLayout(protocol.enumerate_partitions(groups)[0])
        split = protocol.EnergySplit(16, 12, 36)
        chain.zero_grad(set_to_none=True)
        output = chain(H_dl, H_ul, feedback_gamma_db=10,
            pilot_energy_per_ue=16, pilot_standard_noise=noise(),
            data_standard_noise=noise(), layout=layout, energy_split=split)
        metrics = chain.metrics(output, H_dl)
        loss = metrics["dl_nmse_per_user"].mean()
        loss.backward()
        grad = sum(float(p.grad.detach().abs().sum()) for p in chain.codec_model.parameters()
                   if p.grad is not None)
        if not (torch.isfinite(loss) and torch.isfinite(metrics["sum_rate_bps_hz"]).all()
                and grad > 0):
            raise AssertionError(f"Hybrid G={groups} failed finite forward/gradient acceptance")
        energy_error = float((output["tx"]["actual_data_energy"] - 48).abs().max().detach())
        power_error = float((metrics["rzf_per_subcarrier_power"] - 1).abs().max().detach())
        outcomes.append({"layout": f"Hybrid-G{groups}", "dl_nmse": float(loss.detach()),
            "sum_rate_bps_hz": float(metrics["sum_rate_bps_hz"].mean().detach()),
            "actual_data_energy_max_abs_error": energy_error,
            "rzf_power_max_abs_error": power_error,
            "codec_gradient_l1": grad})

    layout = protocol.FDMALayout()
    chain.zero_grad(set_to_none=True)
    output = chain(H_dl, H_ul, feedback_gamma_db=10,
        pilot_energy_per_ue=16, pilot_standard_noise=noise(),
        data_standard_noise=noise(), layout=layout)
    metrics = chain.metrics(output, H_dl)
    loss = metrics["dl_nmse_per_user"].mean()
    loss.backward()
    grad = sum(float(p.grad.detach().abs().sum()) for p in chain.codec_model.parameters()
               if p.grad is not None)
    if not (torch.isfinite(loss) and torch.isfinite(metrics["sum_rate_bps_hz"]).all()
            and grad > 0):
        raise AssertionError("FDMA failed finite forward/gradient acceptance")
    outcomes.append({"layout": "FDMA", "dl_nmse": float(loss.detach()),
        "sum_rate_bps_hz": float(metrics["sum_rate_bps_hz"].mean().detach()),
        "actual_data_energy_max_abs_error": float((output["tx"]["actual_data_energy"] - 48).abs().max().detach()),
        "rzf_power_max_abs_error": float((metrics["rzf_per_subcarrier_power"] - 1).abs().max().detach()),
        "codec_gradient_l1": grad})

    print(json.dumps({"status": "CPU_CODEC_LINK_PREFLIGHT_PASS",
        "training_updates": 0, "device": "cpu", "sample_role": "codec_train",
        "sample_fingerprint": sample["sample_fingerprint"],
        "frozen_ce_checkpoint": str(ce_ckpt.resolve()),
        "ce_checkpoint_step": 5000, "calibration_csv": str(calibration.resolve()),
        "gamma_db": 10, "pilot_energy_per_ue": 16, "N0": protocol.feedback_noise_variance(10),
        "layouts": outcomes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
