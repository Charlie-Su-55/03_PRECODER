"""Matched low-SNR MU rate objectives with normalized DL interference cost."""
from __future__ import annotations

from configs import twc_v1_k8_low_snr_config as cfg
from twc_v1_k8_hybrid_direction import direction_error


def leakage_penalty(desired, interference, noise):
    """Per-MU mean_tone sum_UE log2(1+I/(S+N)); all gradients retained."""
    import math
    import torch
    if (not torch.is_tensor(desired) or not torch.is_tensor(interference)
            or desired.is_complex() or interference.is_complex() or desired.ndim != 3
            or desired.shape != interference.shape or any(n < 1 for n in desired.shape)
            or desired.dtype != interference.dtype or desired.device != interference.device
            or not desired.is_floating_point() or not bool(torch.isfinite(desired).all())
            or not bool(torch.isfinite(interference).all())
            or not bool((desired >= 0).all() & (interference >= 0).all())):
        raise ValueError('Matching finite nonnegative real [MU,tone,UE] powers required')
    supplied = torch.as_tensor(noise)
    if supplied.is_complex() or supplied.dtype == torch.bool or supplied.requires_grad:
        raise ValueError('Positive finite nonlearned scalar noise required')
    noise = torch.as_tensor(noise, dtype=desired.dtype, device=desired.device)
    if noise.ndim or not bool(torch.isfinite(noise) & (noise > 0)):
        raise ValueError('Positive finite nonlearned scalar noise required')
    result = (torch.log1p(interference/(desired+noise))/math.log(2.)).sum(-1).mean(-1)
    if not bool(torch.isfinite(result).all()): raise FloatingPointError('Nonfinite leakage penalty')
    return result


def objective(H_hat, H_true, gamma_db, trajectory):
    import torch
    import low_snr_runtime as runtime
    cfg.key(trajectory)
    rate_loss, metrics = runtime.helpers().rate_objective(H_hat, H_true,
        alpha=cfg.ALPHA, downlink_snr_db=cfg.DL_SNR)
    supplied = torch.as_tensor(gamma_db)
    if supplied.is_complex() or supplied.dtype == torch.bool or supplied.requires_grad:
        raise ValueError('Nonlearned real public feedback gamma required')
    gamma = torch.as_tensor(gamma_db, dtype=H_hat.real.dtype, device=H_hat.device)
    if gamma.ndim == 0: gamma = gamma.expand(H_hat.shape[0])
    if (gamma.shape != (H_hat.shape[0],) or not bool(torch.isfinite(gamma).all())
            or not bool((gamma[:, None] == gamma.new_tensor(cfg.GAMMAS)[None]).any(-1).all())):
        raise ValueError('One prescribed feedback gamma node per MU required')
    direction = direction_error(H_hat, H_true)
    weighted_direction = cfg.DIRECTION_WEIGHT*direction
    leakage = leakage_penalty(metrics['desired_signal_power'], metrics['interference_power'], metrics['noise_power'])
    low_leakage = (gamma <= 10).to(leakage.dtype)*leakage
    weight = cfg.INTERFERENCE_WEIGHT if trajectory == cfg.TRAJECTORIES[1] else 0.
    weighted_interference = weight*low_leakage
    total = rate_loss+weighted_direction.mean()+weighted_interference.mean()
    if not bool(torch.isfinite(total)): raise FloatingPointError('Nonfinite low-SNR joint objective')
    return total, dict(metrics, direction_error=direction, weighted_direction_loss=weighted_direction,
        rate_loss=-metrics['sum_rate_bps_hz'], dl_leakage_penalty=leakage,
        low_snr_leakage_penalty=low_leakage, weighted_interference_loss=weighted_interference)
