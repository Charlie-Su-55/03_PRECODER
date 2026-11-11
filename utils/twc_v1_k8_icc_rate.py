"""Rate objective and BS-only update boundaries for the independent K8 screen.

Torch is imported lazily so metadata/CLI checks do not require site packages.
No code here creates channel samples, updates an EMA, or changes the original
ICC-aligned model, physical link, RZF policy, or downlink power constraint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _torch():
    import torch
    return torch


def _downlink():
    """Load the unchanged numerical module without executing utils.__init__."""
    name = "_icc_rate_original_downlink"
    path = ROOT / "utils" / "twc_v1_k8_downlink.py"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("Original downlink module identity collision")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot locate original downlink implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def differentiable_rates(H_true, W, *, downlink_snr_db=25., noise_power=None):
    """Compute true-channel rates while retaining gradients through ``W``.

    H_true is a training/scoring label with shape [B,S,K,N]. W is [B,S,N,K].
    Tiny arbitrary dimensions are supported for real CPU tests. Power is
    reported, not silently normalized: the policy owns its power projection.
    ``interference_to_signal_ratio`` is the per-MU ratio of mean interference
    to mean desired signal over subcarriers/users, not the mean of link ratios.
    """
    torch = _torch()
    if (not torch.is_tensor(H_true) or not torch.is_tensor(W)
            or H_true.ndim != 4 or W.ndim != 4
            or H_true.dtype not in (torch.complex64, torch.complex128)
            or W.dtype != H_true.dtype or W.device != H_true.device
            or any(size < 1 for size in H_true.shape)
            or W.shape != (H_true.shape[0], H_true.shape[1], H_true.shape[3], H_true.shape[2])):
        raise ValueError("Matched finite complex H[B,S,K,N] and W[B,S,N,K] required")
    if not bool(torch.isfinite(H_true).all() & torch.isfinite(W).all()):
        raise ValueError("Finite channels and precoders required")
    if noise_power is None:
        if (isinstance(downlink_snr_db, bool)
                or not isinstance(downlink_snr_db, (float, int))
                or not math.isfinite(downlink_snr_db)):
            raise ValueError("Finite public downlink SNR required")
        try:
            noise_power = 10. ** (-float(downlink_snr_db) / 10.)
        except OverflowError as error:
            raise ValueError("Finite positive noise power required") from error
    if isinstance(noise_power, bool):
        raise ValueError("Finite positive scalar noise power required")
    try:
        noise = torch.as_tensor(noise_power, dtype=H_true.real.dtype, device=W.device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Finite positive scalar noise power required") from error
    if noise.ndim or not bool(torch.isfinite(noise) & (noise > 0)):
        raise ValueError("Finite positive scalar noise power required")
    effective = H_true @ W
    powers = effective.abs().square()
    desired = powers.diagonal(dim1=-2, dim2=-1)
    interference = (powers.sum(dim=-1) - desired).clamp_min(0.)
    sinr = desired / (interference + noise)
    per_link_rate = torch.log2(1. + sinr)
    desired_mean = desired.mean(dim=(1, 2))
    interference_mean = interference.mean(dim=(1, 2))
    result = dict(
        sum_rate_bps_hz=per_link_rate.sum(dim=-1).mean(dim=-1),
        user_rate_bps_hz=per_link_rate.mean(dim=1),
        desired_signal_power_mean=desired_mean,
        interference_power_mean=interference_mean,
        interference_to_signal_ratio=interference_mean / desired_mean.clamp_min(1e-20),
        desired_signal_power=desired,
        interference_power=interference,
        sinr=sinr,
        per_tone_power=W.abs().square().sum(dim=(-2, -1)),
        noise_power=noise,
    )
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("Nonfinite rate or interference diagnostic")
    return result


def rate_objective(H_hat, H_true, *, alpha=.03, downlink_snr_db=25.):
    """Negative mean sum rate through the original fixed-alpha normalized RZF.

    Both channel tensors use the production shape [B,8,32,256]. H_true is used
    only in this loss/scoring function; no receiver or decoder sees it here.
    The auxiliary NMSE weight is exactly zero for this objective-alignment arm.
    """
    torch = _torch()
    if (not torch.is_tensor(H_true) or not torch.is_tensor(H_hat)
            or H_true.shape != H_hat.shape or H_true.dtype != H_hat.dtype
            or H_true.device != H_hat.device):
        raise ValueError("Matched estimated and true production CSI tensors required")
    W, power = _downlink().design_rzf(H_hat, alpha)
    if not bool(((power - 1.).abs() <= 1e-5).all()):
        raise FloatingPointError("Original RZF per-tone total power changed")
    metrics = differentiable_rates(H_true.permute(0, 3, 1, 2), W,
                                   downlink_snr_db=downlink_snr_db)
    # Rate optimization can use H_hat as an effective channel, including scale.
    # Record this rather than silently adding a new CSI normalization constraint.
    metrics['predicted_dl_energy'] = H_hat.abs().square().mean(dim=(1, 2, 3))
    metrics['predicted_to_true_dl_energy_ratio'] = metrics['predicted_dl_energy'] / H_true.abs().square().mean(dim=(1, 2, 3)).clamp_min(1e-20)
    loss = -metrics["sum_rate_bps_hz"].mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite rate objective")
    return loss, metrics


def _frozen_modules(codec, ce, stats):
    if not hasattr(codec, "encoder") or not hasattr(codec, "decoder"):
        raise ValueError("Explicit encoder and decoder modules required")
    return {"encoder": codec.encoder, "ce": ce, "statistics": stats}


def configure_bs_only(codec, ce, stats):
    """Enable only BS decoder gradients, leaving all frozen modules in eval."""
    modules = _frozen_modules(codec, ce, stats)
    codec.eval().requires_grad_(False)
    codec.decoder.train().requires_grad_(True)
    for module in modules.values():
        module.eval().requires_grad_(False)
        for parameter in module.parameters():
            parameter.grad = None
    parameters = tuple(codec.decoder.parameters())
    frozen_ids = {id(p) for module in modules.values() for p in module.parameters()}
    if not parameters or any(id(p) in frozen_ids or not p.requires_grad for p in parameters):
        raise ValueError("Independent trainable BS decoder parameters required")
    return parameters


def _state_hash(module):
    torch = _torch()
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        # A byte view supports every tensor dtype without NumPy dtype mapping.
        value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def frozen_state(codec, ce, stats):
    """Capture all frozen parameter/buffer bytes and validate frozen modes."""
    modules = _frozen_modules(codec, ce, stats)
    for name, module in modules.items():
        if any(child.training for child in module.modules()):
            raise ValueError("Frozen module is in training mode: " + name)
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError("Frozen module has trainable parameters/gradients: " + name)
    return {name: _state_hash(module) for name, module in modules.items()}


def assert_frozen_state(codec, ce, stats, expected):
    """Reject any frozen parameter/buffer, mode, or gradient-boundary change."""
    actual = frozen_state(codec, ce, stats)
    if actual != expected:
        raise ValueError("Frozen encoder/CE/historical-statistics state changed")
