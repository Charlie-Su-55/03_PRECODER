#!/usr/bin/env python3
"""CPU-only end-to-end K8 feedback-chain preflight; performs no optimizer updates."""
from __future__ import annotations

import json
import importlib.util
import inspect
import sys
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-width", action="store_true",
        help="Use frozen width-128/four-block architecture for one full CPU pass")
    parser.add_argument("--output", type=Path,
        help="Optional JSON path for the CPU preflight receipt")
    args = parser.parse_args()
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
    kwargs = None if args.full_width else {"embed_dim": 32, "blocks": 1, "heads": 4,
        "patch_size": 4, "window_size": 4}
    chain = K8FeedbackChain.from_artifacts(ce_ckpt, calibration, device="cpu",
        swin_kwargs=kwargs)
    gen = torch.Generator(device="cpu").manual_seed(20261104)
    def noise():
        return torch.view_as_complex(torch.randn((1, 32, 256, 2), generator=gen).contiguous())

    outcomes = []
    receiver = chain.codec_model.joint_decoder
    hook_counts = {}
    hooked = [
        ("observation_y", receiver.observation_y),
        ("observation_channel", receiver.observation_channel),
        ("query_projection", receiver.query_projection),
        ("joint_0_self", receiver.joint_layers[0].self_attention),
        ("joint_0_observation", receiver.joint_layers[0].cross_attention),
        ("joint_1_self", receiver.joint_layers[1].self_attention),
        ("joint_1_observation", receiver.joint_layers[1].cross_attention),
        ("soft_symbol_residual", receiver.residual_head),
        ("spatial_cross_attention", receiver.spatial_attention),
        ("swin_output", chain.codec_model.decoder.final_proj),
    ] + [(f"swin_block_{i}", block)
         for i, block in enumerate(chain.codec_model.decoder.blocks)]
    handles = [module.register_forward_hook(
        lambda _module, _inputs, _output, name=name: hook_counts.__setitem__(
            name, hook_counts.get(name, 0) + 1)) for name, module in hooked]
    expected_active = {1: 1312, 2: 672, 4: 352}
    sensitivity_case = None

    def assert_receiver_execution(output, expected_layout, expected_groups):
        execution = output["receiver_execution"]
        if not isinstance(execution, dict):
            raise AssertionError("Target receiver did not return execution metadata")
        active = 192 if expected_layout == "FDMA" else expected_active[expected_groups]
        if (execution["active_query_count"] != active
                or execution["observation_memory_shape"] != [1, 256, receiver.width]
                or execution["joint_context_shape"] != [1, 1312, receiver.width]
                or execution["spatial_cross_attention_shape"] != [1, 4096, receiver.width]):
            raise AssertionError(f"Unexpected target receiver execution metadata: {execution}")
        if any(hook_counts.get(name, 0) == 0 for name, _module in hooked):
            missing = [name for name, _module in hooked if hook_counts.get(name, 0) == 0]
            raise AssertionError(f"Required receiver modules did not execute: {missing}")
        return execution

    group_runs = (4,) if args.full_width else (1, 2, 4)
    for groups in group_runs:
        hook_counts.clear()
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
        execution = assert_receiver_execution(output, "Hybrid", groups)
        gradient_groups = {
            "ue_encoder": ("encoder.",),
            "joint_layers": ("joint_decoder.joint_layers.",),
            "symbol_residual": ("joint_decoder.residual_head.",),
            "spatial_cross_attention": ("joint_decoder.spatial_attention.",),
            "swin_decoder": ("decoder.blocks.", "decoder.final_proj."),
        }
        gradient_norms = {name: sum(float(p.grad.detach().abs().sum())
            for key_name, p in chain.codec_model.named_parameters()
            if key_name.startswith(prefixes) and p.grad is not None)
            for name, prefixes in gradient_groups.items()}
        if any(not torch.isfinite(torch.tensor(value)) or value <= 0
               for value in gradient_norms.values()):
            raise AssertionError(f"Missing finite nonzero stage gradient: {gradient_norms}")
        energy_error = float((output["tx"]["actual_data_energy"] - 48).abs().max().detach())
        power_error = float((metrics["rzf_per_subcarrier_power"] - 1).abs().max().detach())
        outcomes.append({"layout": f"Hybrid-G{groups}", "dl_nmse": float(loss.detach()),
            "sum_rate_bps_hz": float(metrics["sum_rate_bps_hz"].mean().detach()),
            "actual_data_energy_max_abs_error": energy_error,
            "rzf_power_max_abs_error": power_error,
            "codec_gradient_l1": grad, "gradient_l1_by_stage": gradient_norms,
            "receiver_execution": execution})
        if groups == (4 if args.full_width else 2):
            sensitivity_case = (output, layout, split, groups)

    if not args.full_width:
        layout = protocol.FDMALayout()
        hook_counts.clear()
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
        fdma_execution = assert_receiver_execution(output, "FDMA", 1)
        outcomes.append({"layout": "FDMA", "dl_nmse": float(loss.detach()),
            "sum_rate_bps_hz": float(metrics["sum_rate_bps_hz"].mean().detach()),
            "actual_data_energy_max_abs_error": float((output["tx"]["actual_data_energy"] - 48).abs().max().detach()),
            "rzf_power_max_abs_error": float((metrics["rzf_per_subcarrier_power"] - 1).abs().max().detach()),
            "codec_gradient_l1": grad, "receiver_execution": fdma_execution})

    signature = inspect.signature(receiver.forward)
    if any("true" in name.lower() or "transmit" in name.lower()
           for name in signature.parameters):
        raise AssertionError(f"Receiver exposes a simulator-truth input: {signature}")
    if sensitivity_case is None:
        raise AssertionError("No receiver output was retained for observation sensitivity")
    output_g2, layout_g2, split_g2, sensitivity_group_count = sensitivity_case
    chain.codec_model.eval()
    soft = output_g2["soft_symbols"]
    common_args = dict(group_count=sensitivity_group_count, H_ul_est=output_g2["H_ul_est"], layout=layout_g2,
        feedback_gamma_db=10, noise_variance=0.025, pilot_energy_per_ue=16,
        key_energy_per_ue=split_g2.key, shared_energy_per_ue=split_g2.shared,
        channel_error_variance=output_g2["channel_error_variance"],
        key_reliability=soft["key_reliability"],
        shared_reliability=soft["shared_reliability"])
    with torch.no_grad():
        baseline, _, _ = chain.codec_model.decode_feedback(soft["key_symbols_hat"],
            soft["shared_symbols_hat"], Y_rx=output_g2["Y_rx"], **common_args)
        changed_y = output_g2["Y_rx"].clone()
        changed_y[:, :, layout_g2.key_positions[0][0]] += 0.125 + 0.25j
        changed, _, _ = chain.codec_model.decode_feedback(soft["key_symbols_hat"],
            soft["shared_symbols_hat"], Y_rx=changed_y, **common_args)
    observation_delta = float((changed - baseline).abs().max())
    if not observation_delta > 0:
        raise AssertionError("Perturbing data-only Y did not affect the BS receiver")
    for handle in handles:
        handle.remove()

    report = {"status": "CPU_CODEC_LINK_PREFLIGHT_PASS",
        "training_updates": 0, "device": "cpu", "full_width": args.full_width,
        "sample_role": "codec_train",
        "sample_fingerprint": sample["sample_fingerprint"],
        "data_observation_sensitivity_max_abs": observation_delta,
        "truth_leakage_signature_check": "pass",
        "independent_performance_evidence": False,
        "frozen_ce_checkpoint": str(ce_ckpt.resolve()),
        "ce_checkpoint_step": 5000, "calibration_csv": str(calibration.resolve()),
        "gamma_db": 10, "pilot_energy_per_ue": 16, "N0": protocol.feedback_noise_variance(10),
        "layouts": outcomes}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
