#!/usr/bin/env python3
"""Explicit server-only synthetic checks for the new five-layout protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import traceback

from as_receiver_batch_support import safe_inside, sha256
from preflight_proposal_as_direct import save_report
from preflight_joint_swin import tensor_digest

ROOT = Path(__file__).resolve().parent
POWER_ATOL = 2e-6
STATIC_PARAMETERS = {
    "legacy": dict(ue=10877090, bs=34933678, full=45810768),
    "ordered": dict(ue=2899584, bs=34933678, full=37833262),
}


def plan():
    from configs.grouped_codec_train_config import ARMS, SEEDS
    return dict(batch_size=48, synthetic_channels=True, gamma_db=10, downlink_snr_db=25,
        seeds=list(SEEDS), arms=list(ARMS), static_parameter_expectation=STATIC_PARAMETERS,
        counts_not_measured_locally=True, power_atol=POWER_ATOL, cuda_required=True,
        precision="FP32, highest matmul precision, TF32 off; no AMP",
        memory="SDPA; all-key query chunks64; nonreentrant checkpointing; no microbatch",
        disposable_optimizer_update="one per arm for allocation/finite checks, no saved weights or official updates",
        historical_checkpoints_loaded=False, automatic_fallback=False)


def required_modules():
    result = ["decoder.codec.fc", "decoder.spatial_position", "decoder.codec.final_proj",
              "decoder.observation_memory", "decoder.query_projection", "decoder.residual_head",
              "decoder.spatial_attention"]
    result += [f"decoder.codec.blocks.{i}" for i in range(4)]
    result += [f"decoder.joint_layers.{i}.{part}" for i in range(2)
               for part in ("self_attention", "cross_attention", "mlp")]
    return result


def _initialization(torch, report):
    from configs.grouped_codec_train_config import ARMS, SEEDS, grouped_codec_config
    from models.grouped_codec_precoder import GroupedCodecPrecoder
    from train_grouped_codec import validate_model_identity
    report["initialization"] = {}
    for seed in SEEDS:
        records = {}
        for arm in ARMS:
            initial_rng = torch.get_rng_state().clone()
            cfg = grouped_codec_config(arm, seed)
            model = GroupedCodecPrecoder(cfg)
            validate_model_identity(cfg, model)
            parameters = dict(model.named_parameters())
            ue = {k: v for k, v in parameters.items() if k.startswith("ue.")}
            bs = {k: v for k, v in parameters.items() if k.startswith("decoder.")}
            if not torch.equal(initial_rng, torch.get_rng_state()):
                raise RuntimeError("Construction perturbed caller CPU RNG")
            model.load_state_dict(model.state_dict(), strict=True)
            records[arm] = dict(ue_sha256=tensor_digest(ue), bs_sha256=tensor_digest(bs),
                strict_self_load=True, caller_rng_unchanged=True)
            del ue, bs, parameters, model
        if len({r["bs_sha256"] for r in records.values()}) != 1:
            raise RuntimeError("Shared BS trainable initialization differs between layouts")
        for family in ("legacy", "ordered"):
            if len({r["ue_sha256"] for a, r in records.items() if a.startswith(family)}) != 1:
                raise RuntimeError("UE-family initialization differs between layouts")
        report["initialization"][str(seed)] = records


def _receiver_boundary(torch, model):
    """Keep recovery fixed; intervention uses only public receiver quantities."""
    model.eval()
    z = torch.randn(1, 4, 256, device="cuda", dtype=torch.complex64, requires_grad=True)
    r = torch.full((1, 4, 256), .5, device="cuda")
    y = torch.randn(1, 32, 256, device="cuda", dtype=torch.complex64)
    h = torch.randn(1, 4, 32, 256, device="cuda", dtype=torch.complex64)
    model.zero_grad(set_to_none=True)
    a, corrected, _ = model.decoder(z, r, y, h, 10.)
    a.square().mean().backward()
    active = model.decoder.active_mask
    inactive_gradient = z.grad[:, ~active]
    if inactive_gradient.numel() and not torch.equal(inactive_gradient, torch.zeros_like(inactive_gradient)):
        raise RuntimeError("Inactive canonical slots provide a gradient backchannel")
    with torch.no_grad():
        reference = model.decoder(z.detach(), r, y, h, 10.)[0]
        b = model.decoder(z.detach(), r, y * 1.1 + .05j, h, 10.)[0]
        changed = z.detach().clone()
        changed[:, ~active] = 91 + 73j
        c = model.decoder(changed, r, y, h, 10.)[0]
    memory_delta = (reference - b).abs().max().item()
    inactive_delta = (reference - c).abs().max().item()
    if not all(torch.isfinite(t).all().item() for t in (a, reference, b, c, z.grad)) or memory_delta <= 0:
        raise RuntimeError("Independent legal observation memory path failed")
    if inactive_delta != 0 or bool((corrected[:, ~active] != 0).any()):
        raise RuntimeError("Inactive canonical slots affect receiver output")
    return dict(public_Y_intervention_max_abs=memory_delta, inactive_output_max_abs=inactive_delta,
        inactive_input_gradient_max_abs=(inactive_gradient.abs().max().item() if inactive_gradient.numel() else 0.),
        corrected_inactive_exact_zero=True, passed=True)


def _one_arm(torch, arm, report):
    from unittest.mock import patch
    from configs.grouped_codec_train_config import grouped_codec_config, protocol
    from models.grouped_codec_precoder import GroupedCodecPrecoder
    from main_train import SumRateLoss, set_seed
    cfg = grouped_codec_config(arm, 20261030)
    set_seed(cfg.SEED)
    model = GroupedCodecPrecoder(cfg).cuda().train()
    counts = dict(ue=sum(p.numel() for p in model.ue_parameters()),
                  bs=sum(p.numel() for p in model.bs_parameters()),
                  full=sum(p.numel() for p in model.parameters()))
    if counts != STATIC_PARAMETERS[arm.split("_")[0]]:
        raise RuntimeError(f"Measured parameter count differs from static derivation: {counts}")
    torch.cuda.reset_peak_memory_stats()
    result = report["arms"][arm] = dict(parameters=counts, batch_size=48, passed=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    hdl = torch.randn(48, 4, 32, 256, device="cuda", dtype=torch.complex64)
    hul = torch.randn_like(hdl)
    eps = protocol.standard_noise(cfg.SEED, "preflight", 0, 48, "cuda")
    modules = dict(model.named_modules())
    calls = {name: 0 for name in required_modules()}
    handles = []
    def count(name):
        def hook(*unused):
            calls[name] += 1
        return hook
    try:
        for name in calls:
            handles.append(modules[name].register_forward_hook(count(name)))
        with patch.object(protocol, "transmit", wraps=protocol.transmit) as physical:
            output = model.forward_with_diagnostics(hdl, hul, 10., standard_noise=eps, H_ul_est=hul)
            result["execution_forward_counts"] = dict(calls)
            if physical.call_count != 1 or any(value < 1 for value in calls.values()):
                raise RuntimeError("Repeated transmission or missing complete Joint decoder path")
            result["physical_transmit_calls"] = physical.call_count
            hat = output["channel_surrogate"]
            if tuple(hat.shape) != (48, 4, 32, 256):
                raise RuntimeError("Not a full frequency-domain reconstruction")
            result["execution_shapes"] = output["diagnostics"]["execution"]
            if result["execution_shapes"]["angular_delay_shape"] != [48, 4, 2, 256, 32]:
                raise RuntimeError("Not a full 256-tap output")
            w = model.zf_solver(hat)
            w = w * torch.sqrt(1. / (w.abs().square().sum((1, 2), keepdim=True) + 1e-9))
            loss, rate = SumRateLoss()(w, hdl, 10 ** (-25 / 10))
            result.update(synthetic_rate_not_experimental=rate, loss=loss.item(),
                power_max_abs=(w.abs().square().sum((1, 2)) - 1).abs().max().item(),
                energy_max_abs=(output["diagnostics"]["energy"] - 64).abs().max().item(),
                N0=output["diagnostics"]["N0"])
            if not all(torch.isfinite(t).all().item() for t in (hat, w, loss)) or result["power_max_abs"] > POWER_ATOL:
                raise RuntimeError("Nonfinite task output or per-subcarrier power violation")
            loss.backward()
        names = required_modules() + (["ue.encoder.aircomp_backbone", "ue.condition"]
            if arm.startswith("legacy") else ["ue.patch_embed", "ue.fc"] + [f"ue.blocks.{i}" for i in range(4)])
        result["gradients"] = {}
        for name in names:
            grads = [p.grad for p in modules[name].parameters() if p.grad is not None]
            norm = sum(g.detach().double().square().sum().item() for g in grads) ** .5
            result["gradients"][name] = dict(tensors=len(grads), norm=norm,
                finite=all(torch.isfinite(g).all().item() for g in grads))
            if not grads or not result["gradients"][name]["finite"] or norm <= 0:
                raise RuntimeError(f"Missing/nonfinite active gradient path: {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()  # Disposable synthetic weights only; never saved or counted as training.
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("Nonfinite disposable AdamW update")
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        result.update(hooks_removed=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del output, hat, w, loss, optimizer, hdl, hul, eps
    model.zero_grad(set_to_none=True)
    result["receiver_boundary"] = _receiver_boundary(torch, model)
    result["passed"] = True
    del model
    torch.cuda.empty_cache()


def run(output):
    from configs.grouped_codec_train_config import ARMS, OUTPUT_ROOT
    output = safe_inside(output, OUTPUT_ROOT / "batches")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, plan=plan(), arms={})
    try:
        import torch
        from preflight_proposal_as_direct import _backend_info
        from train_grouped_codec import SOURCE_FILES, apply_backend
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        report["backend_before"] = _backend_info(torch)
        apply_backend(torch)
        report["backend_training_policy"] = _backend_info(torch)
        report["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["source_sha256"] = {p: sha256(ROOT / p) for p in (*SOURCE_FILES, "preflight_grouped_codec.py")}
        _initialization(torch, report)
        for arm in ARMS:
            _one_arm(torch, arm, report)
        report.update(passed=True, backend_after=_backend_info(torch))
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), automatic_fallback=False)
        raise
    finally:
        save_report(output, report)
        print(json.dumps(dict(passed=report["passed"], report=str(output / "report.json"), error=report.get("error")), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.run:
        if args.output_dir is None:
            parser.error("--run requires a fresh --output-dir below the batch root")
        run(args.output_dir)
    elif args.output_dir is not None:
        parser.error("--output-dir requires --run")
    elif args.dry_run:
        print(json.dumps(plan(), indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
