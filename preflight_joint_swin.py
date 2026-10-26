#!/usr/bin/env python3
"""One-time CUDA structural/memory check; no saved weights or experiment updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from as_receiver_batch_support import safe_inside, sha256
from preflight_proposal_as_direct import save_report

ROOT = Path(__file__).resolve().parent
POWER_ATOL = 2e-6
STATIC_PARAMETERS = {
    "plain_swin256": {"ue_and_condition": 10877090, "bs": 34422944, "full": 45300034},
    "joint_swin256": {"ue_and_condition": 10877090, "bs": 34933038, "full": 45810128},
}


def plan():
    return dict(batch_size=48, synthetic_channels=True, feedback_snr_db=10, downlink_snr_db=25,
                seeds=[20261030, 20261031], arms=list(STATIC_PARAMETERS),
                static_parameter_expectation=STATIC_PARAMETERS, power_atol=POWER_ATOL,
                optimizer_check="one disposable synthetic AdamW update per arm; no checkpoint saved",
                memory="SDPA, all-key query chunks64, nonreentrant activation checkpointing",
                no_microbatch_or_automatic_structure_change=True, cuda_required=True)


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def required_modules(arm):
    names = ["decoder.codec.fc", "decoder.spatial_position", "decoder.codec.final_proj"]
    names += [f"decoder.codec.blocks.{i}" for i in range(4)]
    if arm == "joint_swin256":
        names += ["decoder.observation_memory", "decoder.query_projection", "decoder.residual_head",
                  "decoder.spatial_attention"]
        for i in range(2):
            names += [f"decoder.joint_layers.{i}.{name}" for name in ("self_attention", "cross_attention", "mlp")]
    return names


def _initialization(torch, source, report):
    from configs.joint_swin_train_config import ARMS, SEEDS, joint_swin_config
    from models.adaptive_hybrid_joint_swin import AdaptiveHybridPrecoderJointSwin, COMMON_BS_PREFIXES
    from train_proposal_joint_swin import validate_model_identity
    from main_train import set_seed
    report["initialization"] = {}
    for seed in SEEDS:
        records = {}
        for arm in ARMS:
            set_seed(seed)
            cfg = joint_swin_config(arm, seed)
            model = AdaptiveHybridPrecoderJointSwin(cfg)
            model.initialize_ue_from_full(source)
            validate_model_identity(cfg, model)
            state = model.state_dict()
            common = {k: v for k, v in state.items() if k.startswith(COMMON_BS_PREFIXES)}
            ue = {k: state[k] for k in model.ue_state_keys}
            for name, value in ue.items():
                if not torch.equal(value, source[name]):
                    raise ValueError(f"UE-only source migration changed {name}")
            model.load_state_dict(state, strict=True)
            records[arm] = dict(common_sha256=tensor_digest(common), ue_sha256=tensor_digest(ue),
                                common_tensor_count=len(common),
                                cpu_rng_after_construction_sha256=tensor_digest({"rng": torch.get_rng_state()}),
                                strict_self_load=True)
            del state, common, ue, model
        if records[ARMS[0]] != records[ARMS[1]]:
            raise ValueError(f"Common initialization or RNG mismatch at seed {seed}: {records}")
        report["initialization"][str(seed)] = records


def _memory_intervention(torch, model):
    """Fix recovery outputs; vary only legal receiver observations, in eval mode."""
    model.eval()
    device = next(model.parameters()).device
    z = torch.randn(1, 256, 4, device=device, dtype=torch.complex64)
    r = torch.full((1, 256, 4), .5, device=device)
    y = torch.randn(1, 32, 256, device=device, dtype=torch.complex64)
    h = torch.randn(1, 4, 32, 256, device=device, dtype=torch.complex64)
    with torch.no_grad():
        a = model.decoder(z, r, y, h, 10)[0]
        b = model.decoder(z, r, y * 1.1 + .05j, h, 10)[0]
    maximum = (a - b).abs().max().item()
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item() or maximum <= 0:
        raise RuntimeError("Legal observation memory does not independently affect output")
    return dict(fixed_recovery=True, altered="noisy Y_ac only", max_abs=maximum, passed=True)


def _one_arm(torch, source, arm, report):
    from unittest.mock import patch
    from main_train import set_seed, SumRateLoss
    from configs.joint_swin_train_config import joint_swin_config
    from models.adaptive_hybrid_joint_swin import AdaptiveHybridPrecoderJointSwin
    from train_proposal_joint_swin import optimizer_scheduler_factory
    cfg = joint_swin_config(arm, 20261030)
    set_seed(cfg.SEED)
    model = AdaptiveHybridPrecoderJointSwin(cfg)
    model.initialize_ue_from_full(source)
    counts = dict(ue_and_condition=sum(p.numel() for p in model.ue_parameters()),
                  bs=sum(p.numel() for p in model.bs_parameters()), full=sum(p.numel() for p in model.parameters()))
    if counts != STATIC_PARAMETERS[arm]:
        raise RuntimeError(f"Actual parameters differ from documented derivation: {counts}")
    model.cuda().train()
    torch.cuda.reset_peak_memory_stats()
    result = report["arms"][arm] = dict(parameters=counts, batch_size=48, passed=False)
    optimizer, scheduler = optimizer_scheduler_factory(cfg, model)
    H_dl = torch.randn(48, 4, 32, 256, device="cuda", dtype=torch.complex64)
    H_ul = torch.randn_like(H_dl)
    # A distinct receiver estimate proves the explicit estimator interface is used.
    estimate = H_ul + .01 * torch.randn_like(H_ul)
    modules = dict(model.named_modules())
    calls = {name: 0 for name in required_modules(arm)}
    handles = []
    def hook(name):
        def count(*unused):
            calls[name] += 1
        return count
    try:
        for name in calls:
            handles.append(modules[name].register_forward_hook(hook(name)))
        with patch.object(model, "transmit", wraps=model.transmit) as transmit, \
                patch.object(model, "add_noise", wraps=model.add_noise) as noise:
            output = model.forward_with_diagnostics(H_dl, H_ul, 10., 0, 256, 0, H_ul_est=estimate)
            result["execution_forward_counts"] = dict(calls)
            result["physical_calls"] = dict(transmit=transmit.call_count, noise=noise.call_count)
            if transmit.call_count != 1 or noise.call_count != 1 or any(n < 1 for n in calls.values()):
                raise RuntimeError("Incomplete decoder path or repeated physical transmission/noise")
            H_hat = output["channel_surrogate"]
            if tuple(H_hat.shape) != (48, 4, 32, 256):
                raise RuntimeError("Incorrect full frequency-domain reconstruction shape")
            result["execution_shapes"] = output["diagnostics"]["execution"]
            if result["execution_shapes"]["angular_delay_shape"] != [48, 4, 2, 256, 32]:
                raise RuntimeError("Not a full 256-tap angular-delay output")
            W = model.zf_solver(H_hat)
            W = W * torch.sqrt(cfg.TOTAL_POWER / (W.abs().square().sum((1, 2), keepdim=True) + 1e-9))
            loss, rate = SumRateLoss()(W, H_dl, 10 ** (-25 / 10))
            result.update(loss=loss.item(), synthetic_rate_not_experimental_result=rate,
                          power_max_abs=(W.abs().square().sum((1, 2)) - cfg.TOTAL_POWER).abs().max().item())
            if not all(torch.isfinite(x).all().item() for x in (H_hat, W, loss)) or result["power_max_abs"] > POWER_ATOL:
                raise RuntimeError("Nonfinite output or per-subcarrier power error")
            loss.backward()
        gradients = {}
        for name in required_modules(arm) + ["encoder.aircomp_backbone", "cond_embed"]:
            params = list(modules[name].parameters())
            grads = [p.grad for p in params if p.grad is not None]
            norm = sum(g.detach().double().square().sum().item() for g in grads) ** .5
            gradients[name] = dict(tensors_with_grad=len(grads), norm=norm,
                                   finite=all(torch.isfinite(g).all().item() for g in grads))
            if not grads or not gradients[name]["finite"] or norm <= 0:
                result["gradients"] = gradients
                raise RuntimeError(f"Unreachable/nonfinite gradient path: {name}")
        result["gradients"] = gradients
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise RuntimeError("Nonfinite parameter gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("Nonfinite disposable optimizer update")
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["scratch_optimizer_updates"] = 1
        result["saved_weights"] = False
    finally:
        for handle in handles:
            handle.remove()
        result["hooks_removed"] = True
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if arm == "joint_swin256":
        result["memory_intervention"] = _memory_intervention(torch, model)
    result["passed"] = True
    del output, W, H_hat, loss, optimizer, scheduler, model
    torch.cuda.empty_cache()


def run(output):
    output = safe_inside(output, ROOT / "runs_as_screen/proposal_joint_swin256_v1/batches")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, plan=plan(), arms={}, code_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    try:
        import torch
        from preflight_proposal_as_direct import (
            load_source_record, verify_checkpoint_hash, validate_checkpoint_metadata, _backend_info)
        from train_proposal_joint_swin import SOURCE_FILES
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no laptop CPU fallback")
        report["backend_before"] = _backend_info(torch)
        # Exactly the original trainer's explicit numerical policy; no PASS-only changes.
        torch.set_float32_matmul_precision("high")
        report["backend_training_policy"] = _backend_info(torch)
        report["source_sha256"] = {p: sha256(ROOT / p) for p in (*SOURCE_FILES, "preflight_joint_swin.py")}
        source = load_source_record()
        report["source_checkpoint"] = source
        verify_checkpoint_hash(source["path"], source["checkpoint_sha256"])
        payload = torch.load(source["path"], map_location="cpu", weights_only=False)
        validate_checkpoint_metadata(payload, source)
        _initialization(torch, payload["state"], report)
        for arm in STATIC_PARAMETERS:
            _one_arm(torch, payload["state"], arm, report)
        verify_checkpoint_hash(source["path"], source["checkpoint_sha256"])
        report.update(passed=True, checkpoint_unchanged=True, backend_after=_backend_info(torch))
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      automatic_fallback=False)
        raise
    finally:
        save_report(output, report)
        print(json.dumps({"passed": report["passed"], "report": str(output / "report.json"),
                          "error": report.get("error")}, indent=2))


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
