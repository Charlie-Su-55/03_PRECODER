"""Phase/norm-invariant direction supervision with the original MU rate loss."""
from __future__ import annotations

from configs import twc_v1_k8_hybrid_direction_config as cfg


def direction_error(H_hat, H_true):
    """Return one differentiable projective-direction error per complete MU.

    The antenna axis is dimension2. No target amplitude or user phase is
    imposed. Strictly nonzero row norms are required; an undefined direction
    must not silently become a zero-error training label.
    """
    import torch
    if (not torch.is_tensor(H_hat) or not torch.is_tensor(H_true) or H_hat.ndim != 4
            or H_hat.shape != H_true.shape or any(n < 1 for n in H_hat.shape)
            or H_hat.dtype not in (torch.complex64, torch.complex128)
            or H_true.dtype != H_hat.dtype or H_true.device != H_hat.device
            or not bool(torch.isfinite(H_hat).all() & torch.isfinite(H_true).all())):
        raise ValueError('Matching finite complex [MU,UE,Nt,tone] channels required')
    estimate, target = H_hat.to(torch.complex128), H_true.detach().to(torch.complex128)
    estimated_power, target_power = estimate.abs().square().sum(2), target.abs().square().sum(2)
    if not bool((estimated_power > 0).all() & (target_power > 0).all()):
        raise ValueError('Direction undefined for a zero channel row')
    cosine = (estimate.conj()*target).sum(2).abs().square()/(estimated_power*target_power)
    return (1-cosine).clamp(0, 1).mean((1, 2))


def objective(H_hat, H_true):
    """Fixed recipe: -mean sum rate +100*mean projective direction error."""
    import hybrid_direction_runtime as runtime
    loss, rates = runtime.helpers().rate_objective(H_hat, H_true,
        alpha=cfg.ALPHA, downlink_snr_db=cfg.DL_SNR)
    direction = direction_error(H_hat, H_true)
    weighted = cfg.DIRECTION_WEIGHT*direction
    total = loss+weighted.mean()
    return total, dict(rates, direction_error=direction, weighted_direction_loss=weighted,
        rate_loss=-rates['sum_rate_bps_hz'])
