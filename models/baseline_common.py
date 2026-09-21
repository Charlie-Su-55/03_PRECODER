#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common physical-layer operators for all AI FDMA baselines."""

from __future__ import annotations

import torch


def angular_delay_transform(H_dl: torch.Tensor, num_delay_taps: int) -> torch.Tensor:
    """H_dl [B,K,Nt,Nsc] complex -> [B,K,2,Nc,Nt]."""
    H_ang = torch.fft.fft(H_dl, dim=2)
    H_del = torch.fft.ifft(H_ang, dim=3)
    H_trunc = H_del[:, :, :, :num_delay_taps].permute(0, 1, 3, 2)
    return torch.stack([H_trunc.real, H_trunc.imag], dim=2)


def inverse_angular_delay_transform(H_ad: torch.Tensor, num_subcarriers: int) -> torch.Tensor:
    """H_ad [B,K,2,Nc,Nt] -> [B,K,Nt,Nsc] complex."""
    B, K, _, Nc, Nt = H_ad.shape
    H_complex = torch.complex(H_ad[:, :, 0], H_ad[:, :, 1])
    H_pad = torch.zeros(
        B, K, Nt, num_subcarriers,
        dtype=H_complex.dtype, device=H_complex.device,
    )
    H_pad[:, :, :, :Nc] = H_complex.permute(0, 1, 3, 2)
    H_freq = torch.fft.fft(H_pad, dim=3)
    return torch.fft.ifft(H_freq, dim=2)


def normalize_complex_symbols(symbols: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    power = symbols.abs().square().mean(dim=-1, keepdim=True)
    return symbols / torch.sqrt(power + eps)


def add_relative_awgn(received: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
    """Match the proposed model's received-power-relative noise convention."""
    snr = torch.as_tensor(snr_db, dtype=received.real.dtype, device=received.device)
    inv_snr = torch.pow(
        torch.tensor(10.0, device=received.device, dtype=received.real.dtype),
        -snr / 10.0,
    )
    sigma = torch.sqrt(received.abs().square().mean() * inv_snr / 2.0)
    noise = (torch.randn_like(received) + 1j * torch.randn_like(received)) * sigma
    return received + noise


# def fdma_feedback_mrc(
#     codewords_real: torch.Tensor,
#     H_ul: torch.Tensor,
#     snr_db: torch.Tensor | float,
#     d_f: int,
# ) -> torch.Tensor:
#     """Analog FDMA transmission followed by MRC equalization."""
#     _, K, D_real = codewords_real.shape
#     if D_real != 2 * d_f:
#         raise ValueError(f"Expected real codeword dimension {2*d_f}, got {D_real}.")
#     if K * d_f > H_ul.shape[-1]:
#         raise ValueError(
#             f"FDMA requires {K*d_f} subcarriers, but H_ul has {H_ul.shape[-1]}."
#         )

#     symbols = torch.complex(codewords_real[:, :, :d_f], codewords_real[:, :, d_f:])
#     symbols = normalize_complex_symbols(symbols)

#     H_segments = torch.stack(
#         [H_ul[:, k, :, k*d_f:(k+1)*d_f] for k in range(K)], dim=1
#     )
#     received = add_relative_awgn(H_segments * symbols.unsqueeze(2), snr_db)

#     denominator = H_segments.abs().square().sum(dim=2).clamp_min(1e-8)
#     estimated = (H_segments.conj() * received).sum(dim=2) / denominator
#     return torch.cat([estimated.real, estimated.imag], dim=-1)
def fdma_feedback_mrc(
    codewords_real: torch.Tensor,
    H_ul: torch.Tensor,
    snr_db: torch.Tensor | float,
    d_f: int,
    H_ul_est: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Analog FDMA transmission followed by MRC equalization.

    Parameters
    ----------
    codewords_real:
        Real-valued feedback codewords with shape [B,K,2*d_f].

    H_ul:
        True feedback-link channel with shape [B,K,Nt,Nsc].
        This channel is always used for the physical transmission.

    snr_db:
        Feedback-link SNR in dB.

    d_f:
        Number of complex FDMA feedback symbols per UE.

    H_ul_est:
        BS-side estimate of the feedback-link channel,
        shape [B,K,Nt,Nsc].

        If None, perfect feedback-link CSI is assumed:

            H_ul_est = H_ul

        The estimated channel is used only by the MRC receiver,
        while the physical received signal is always generated
        using the true H_ul.

    Returns
    -------
    estimated_real:
        MRC-equalized feedback representation with shape
        [B,K,2*d_f].
    """
    _, K, D_real = codewords_real.shape

    if D_real != 2 * d_f:
        raise ValueError(
            f"Expected real codeword dimension {2*d_f}, "
            f"got {D_real}."
        )

    if K * d_f > H_ul.shape[-1]:
        raise ValueError(
            f"FDMA requires {K*d_f} subcarriers, "
            f"but H_ul has {H_ul.shape[-1]}."
        )

    if H_ul_est is None:
        H_ul_est = H_ul

    if H_ul_est.shape != H_ul.shape:
        raise ValueError(
            "H_ul_est must have the same shape as H_ul: "
            f"H_ul={tuple(H_ul.shape)}, "
            f"H_ul_est={tuple(H_ul_est.shape)}."
        )

    if H_ul_est.device != H_ul.device:
        raise ValueError(
            "H_ul_est and H_ul must be on the same device."
        )

    if H_ul_est.dtype != H_ul.dtype:
        raise TypeError(
            "H_ul_est and H_ul must have the same dtype."
        )

    # --------------------------------------------------------------
    # Convert the real latent representation into d_f complex
    # analog feedback symbols and normalize each UE codeword.
    # --------------------------------------------------------------
    symbols = torch.complex(
        codewords_real[:, :, :d_f],
        codewords_real[:, :, d_f:],
    )
    symbols = normalize_complex_symbols(symbols)

    # --------------------------------------------------------------
    # Extract the orthogonal FDMA resource block assigned to each UE.
    #
    # H_segments_true:
    #     Used by the actual physical feedback transmission.
    #
    # H_segments_est:
    #     Used only by the BS-side MRC receiver.
    #
    # Shape: [B,K,Nt,d_f]
    # --------------------------------------------------------------
    H_segments_true = torch.stack(
        [
            H_ul[:, k, :, k*d_f:(k+1)*d_f]
            for k in range(K)
        ],
        dim=1,
    )

    H_segments_est = torch.stack(
        [
            H_ul_est[:, k, :, k*d_f:(k+1)*d_f]
            for k in range(K)
        ],
        dim=1,
    )

    # --------------------------------------------------------------
    # Physical analog FDMA transmission.
    #
    # IMPORTANT:
    # The received signal always depends on the true channel.
    # Imperfect CSI affects only BS receive combining.
    # --------------------------------------------------------------
    received_clean = (
        H_segments_true
        * symbols.unsqueeze(2)
    )

    received = add_relative_awgn(
        received_clean,
        snr_db,
    )

    # --------------------------------------------------------------
    # MRC using the BS-side channel estimate:
    #
    #       z_hat =
    #           h_hat^H y
    #           ----------
    #           ||h_hat||^2
    #
    # Perfect-CSI legacy behavior is recovered exactly when
    # H_ul_est is None or H_ul_est == H_ul.
    # --------------------------------------------------------------
    denominator = (
        H_segments_est
        .abs()
        .square()
        .sum(dim=2)
        .clamp_min(1e-8)
    )

    estimated = (
        H_segments_est.conj()
        * received
    ).sum(dim=2) / denominator

    return torch.cat(
        [estimated.real, estimated.imag],
        dim=-1,
    )


def normalize_precoder(W: torch.Tensor, total_power: float, eps: float = 1e-9) -> torch.Tensor:
    power = W.abs().square().sum(dim=(1, 2), keepdim=True)
    return W * torch.sqrt(total_power / (power + eps))


def closed_form_rzf(
    H_surrogate: torch.Tensor,
    regularization: float,
    total_power: float,
) -> torch.Tensor:
    """Same channel normalization and RZF convention as the proposed model."""
    B, K, _, Nsc = H_surrogate.shape
    device = H_surrogate.device

    H_eff = H_surrogate.permute(0, 3, 1, 2).conj()
    norm_factor = torch.linalg.vector_norm(
        H_eff, dim=(2, 3), keepdim=True
    ).clamp_min(1e-8)
    H_eff_norm = H_eff / norm_factor

    H_eff_h = H_eff_norm.conj().transpose(-2, -1)
    gram = H_eff_norm @ H_eff_h
    eye = torch.eye(K, dtype=H_surrogate.dtype, device=device).view(1, 1, K, K)
    matrix = gram + float(regularization) * eye
    rhs = eye.expand(B, Nsc, K, K)
    inverse_action = torch.linalg.solve(matrix, rhs)

    W_sc = H_eff_h @ inverse_action
    W = W_sc.permute(0, 2, 3, 1).contiguous()
    return normalize_precoder(W, total_power)