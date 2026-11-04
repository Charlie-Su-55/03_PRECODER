"""K8 downlink CSI metrics and equal-power regularized-ZF precoding."""
from __future__ import annotations

import math

import torch


def downlink_nmse(H_hat, H_true):
    if (not torch.is_tensor(H_hat) or not torch.is_tensor(H_true)
            or H_hat.dtype != torch.complex64 or H_true.dtype != torch.complex64
            or H_hat.shape != H_true.shape or H_hat.ndim != 4
            or tuple(H_hat.shape[1:]) != (8, 32, 256)):
        raise ValueError("H_hat and H_true must both be complex64 [B,8,32,256]")
    numerator = (H_hat - H_true).abs().square().sum(dim=(2, 3))
    denominator = H_true.abs().square().sum(dim=(2, 3)).clamp_min(1e-20)
    return numerator / denominator


def design_rzf(H_dl_est, alpha):
    """Design 8-user RZF from estimated CSI; normalize total power to one/tone."""
    if (not torch.is_tensor(H_dl_est) or H_dl_est.dtype != torch.complex64
            or H_dl_est.ndim != 4 or tuple(H_dl_est.shape[1:]) != (8, 32, 256)
            or not bool(torch.isfinite(H_dl_est).all())):
        raise ValueError("H_dl_est must be finite complex64 [B,8,32,256]")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("RZF alpha must be a finite nonnegative scalar")
    channels = H_dl_est.permute(0, 3, 1, 2)  # [B,Nsc,K,Nt]
    gram = channels @ channels.conj().transpose(-2, -1)
    eye = torch.eye(8, dtype=torch.complex64, device=H_dl_est.device)
    solved = torch.linalg.solve(gram + float(alpha) * eye, channels)
    precoder = solved.conj().transpose(-2, -1)  # [B,Nsc,Nt,K]
    power = precoder.abs().square().sum(dim=(-2, -1), keepdim=True)
    if not bool(torch.isfinite(power).all() & (power > 1e-20).all()):
        raise FloatingPointError("RZF produced a zero or nonfinite precoder")
    precoder = precoder / power.sqrt()
    per_tone_power = precoder.abs().square().sum(dim=(-2, -1))
    return precoder, per_tone_power


def evaluate_sum_rate(precoder, H_dl_true, downlink_snr_db=25.0):
    """Evaluate true-channel spectral efficiency; H_dl_true is metric-only."""
    if (not torch.is_tensor(precoder) or precoder.dtype != torch.complex64
            or precoder.ndim != 4 or tuple(precoder.shape[1:]) != (256, 32, 8)):
        raise ValueError("precoder must be complex64 [B,256,32,8]")
    if (not torch.is_tensor(H_dl_true) or H_dl_true.dtype != torch.complex64
            or H_dl_true.shape[0] != precoder.shape[0]
            or tuple(H_dl_true.shape[1:]) != (8, 32, 256)):
        raise ValueError("H_dl_true must be complex64 [B,8,32,256]")
    if isinstance(downlink_snr_db, bool) or not isinstance(downlink_snr_db, (int, float)) or not math.isfinite(downlink_snr_db):
        raise ValueError("downlink_snr_db must be finite")
    H = H_dl_true.permute(0, 3, 1, 2)  # [B,Nsc,K,Nt]
    effective = H @ precoder
    powers = effective.abs().square()
    desired = powers.diagonal(dim1=-2, dim2=-1)
    interference = (powers.sum(dim=-1) - desired).clamp_min(0.0)
    noise = 10.0 ** (-float(downlink_snr_db) / 10.0)
    sinr = desired / (interference + noise)
    user_rate = torch.log2(1.0 + sinr)
    per_sample_rate = user_rate.sum(dim=-1).mean(dim=-1)
    signal_mean = desired.mean(dim=(1, 2))
    interference_mean = interference.mean(dim=(1, 2))
    if not bool(torch.isfinite(per_sample_rate).all()):
        raise FloatingPointError("Downlink sum-rate evaluation is nonfinite")
    return {"sum_rate_bps_hz": per_sample_rate,
        "user_rate_bps_hz": user_rate.mean(dim=1),
        "desired_signal_power_mean": signal_mean,
        "interference_power_mean": interference_mean,
        "noise_power": noise,
        "sinr": sinr}
