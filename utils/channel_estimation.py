#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pilot-aided channel-estimation utilities.

This module provides a simple and reproducible LS estimator for the
uplink feedback channel used by the CSI-feedback experiments.

Expected channel shape
----------------------
H_ul_true : [B, K, Nt, Nsc] complex

Pilot model
-----------
Each UE transmits an orthogonal unit-modulus pilot sequence of length T_p,
with T_p >= K. By default T_p = K. The same pilot matrix is used on every
subcarrier under the block-fading assumption.

For each subcarrier m,

    Y_p[m] = sum_k h_k[m] p_k^T + N_p[m]

where

    P P^H = T_p I_K.

The LS estimate is

    H_hat = (1 / T_p) Y_p P^H.

The parameter ``pilot_snr_db`` denotes the per-symbol pilot SNR of a
single normalized UE channel coefficient:

    SNR_p = E[|h p|^2] / N0.

For the current normalized channel generator, E[|h|^2] is approximately
one, so unit-power pilots correspond to complex AWGN variance

    N0 = 10^(-SNR_p / 10).

The special value +inf gives perfect channel knowledge and is useful as
the reference condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChannelEstimateResult:
    """Container returned by the pilot-aided LS estimator."""

    H_hat: torch.Tensor
    pilot_observation: torch.Tensor
    pilot_matrix: torch.Tensor
    pilot_length: int
    pilot_snr_db: float
    nmse_linear: float
    nmse_db: float


def _validate_channel(H_ul_true: torch.Tensor) -> None:
    if H_ul_true.ndim != 4:
        raise ValueError(
            "Expected H_ul_true with shape [B,K,Nt,Nsc], "
            f"got shape {tuple(H_ul_true.shape)}."
        )

    if not torch.is_complex(H_ul_true):
        raise TypeError(
            "H_ul_true must be a complex-valued tensor."
        )

    if H_ul_true.shape[1] <= 0:
        raise ValueError("Number of users K must be positive.")


def orthogonal_dft_pilots(
    num_users: int,
    pilot_length: Optional[int] = None,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Construct unit-modulus orthogonal DFT pilots.

    Parameters
    ----------
    num_users:
        Number of UEs K.

    pilot_length:
        Pilot sequence length T_p. Must satisfy T_p >= K.
        Defaults to K.

    device:
        Torch device.

    dtype:
        Complex torch dtype, typically torch.complex64.

    Returns
    -------
    pilots:
        Complex tensor with shape [K, T_p].

        Each entry has unit magnitude and

            pilots @ pilots^H = T_p * I_K.
    """
    if num_users <= 0:
        raise ValueError(
            f"num_users must be positive, got {num_users}."
        )

    if pilot_length is None:
        pilot_length = num_users

    pilot_length = int(pilot_length)

    if pilot_length < num_users:
        raise ValueError(
            f"Orthogonal pilots require pilot_length >= num_users, "
            f"got T_p={pilot_length}, K={num_users}."
        )

    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError(
            "dtype must be torch.complex64 or torch.complex128, "
            f"got {dtype}."
        )

    real_dtype = (
        torch.float32
        if dtype == torch.complex64
        else torch.float64
    )

    k = torch.arange(
        num_users,
        device=device,
        dtype=real_dtype,
    ).view(num_users, 1)

    t = torch.arange(
        pilot_length,
        device=device,
        dtype=real_dtype,
    ).view(1, pilot_length)

    phase = -2.0 * torch.pi * k * t / float(pilot_length)

    pilots = torch.exp(1j * phase).to(dtype=dtype)

    return pilots


def channel_nmse(
    H_hat: torch.Tensor,
    H_true: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute mean sample-wise channel NMSE.

    Both tensors must have shape [B,K,Nt,Nsc].
    """
    if H_hat.shape != H_true.shape:
        raise ValueError(
            f"Shape mismatch: H_hat={tuple(H_hat.shape)}, "
            f"H_true={tuple(H_true.shape)}."
        )

    error_power = (
        H_hat - H_true
    ).abs().square().sum(dim=(1, 2, 3))

    reference_power = (
        H_true.abs()
        .square()
        .sum(dim=(1, 2, 3))
        .clamp_min(eps)
    )

    return (error_power / reference_power).mean()


def channel_nmse_db(
    H_hat: torch.Tensor,
    H_true: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute NMSE in dB from the mean linear NMSE.
    """
    nmse = channel_nmse(
        H_hat,
        H_true,
        eps=eps,
    )

    return 10.0 * torch.log10(
        nmse.clamp_min(eps)
    )


def _complex_awgn(
    shape: torch.Size | tuple[int, ...],
    *,
    noise_variance: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate circularly symmetric complex Gaussian noise.

    ``noise_variance`` is E[|n|^2].
    """
    real_dtype = (
        torch.float32
        if dtype == torch.complex64
        else torch.float64
    )

    std = torch.sqrt(noise_variance / 2.0)

    noise_real = torch.randn(
        shape,
        device=device,
        dtype=real_dtype,
        generator=generator,
    )

    noise_imag = torch.randn(
        shape,
        device=device,
        dtype=real_dtype,
        generator=generator,
    )

    return torch.complex(
        noise_real * std,
        noise_imag * std,
    ).to(dtype=dtype)


@torch.no_grad()
def estimate_feedback_channel_ls(
    H_ul_true: torch.Tensor,
    pilot_snr_db: float,
    pilot_length: Optional[int] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> ChannelEstimateResult:
    """
    Estimate the UL feedback channel using orthogonal pilots and LS.

    Parameters
    ----------
    H_ul_true:
        True feedback-link channel [B,K,Nt,Nsc], complex.

    pilot_snr_db:
        Per-symbol pilot SNR in dB.

        Use ``float("inf")`` for perfect channel knowledge.

    pilot_length:
        Number of pilot symbols T_p. Defaults to K.

    generator:
        Optional torch.Generator used only for pilot AWGN.

    Returns
    -------
    ChannelEstimateResult
        Contains the LS estimate, pilot observation, pilot matrix,
        and achieved estimation NMSE.
    """
    _validate_channel(H_ul_true)

    B, K, Nt, Nsc = H_ul_true.shape

    if pilot_length is None:
        pilot_length = K

    pilot_length = int(pilot_length)

    pilots = orthogonal_dft_pilots(
        K,
        pilot_length,
        device=H_ul_true.device,
        dtype=H_ul_true.dtype,
    )

    # ------------------------------------------------------------------
    # Perfect-CSI reference.
    # ------------------------------------------------------------------
    if pilot_snr_db == float("inf"):
        H_hat = H_ul_true.clone()

        pilot_observation = torch.einsum(
            "bkns,kt->bnst",
            H_ul_true,
            pilots,
        )

        return ChannelEstimateResult(
            H_hat=H_hat,
            pilot_observation=pilot_observation,
            pilot_matrix=pilots,
            pilot_length=pilot_length,
            pilot_snr_db=float("inf"),
            nmse_linear=0.0,
            nmse_db=float("-inf"),
        )

    pilot_snr_db = float(pilot_snr_db)

    if not torch.isfinite(
        torch.tensor(pilot_snr_db)
    ):
        raise ValueError(
            "pilot_snr_db must be finite or +inf."
        )

    # ------------------------------------------------------------------
    # Orthogonal multi-user pilot transmission:
    #
    # H_ul_true : [B,K,Nt,Nsc]
    # pilots    : [K,Tp]
    #
    # Y_clean   : [B,Nt,Nsc,Tp]
    # ------------------------------------------------------------------
    Y_clean = torch.einsum(
        "bkns,kt->bnst",
        H_ul_true,
        pilots,
    )

    # Per-symbol pilot SNR:
    #
    #     SNR_p = 1 / N0
    #
    # under the current normalized-channel convention.
    real_dtype = H_ul_true.real.dtype

    snr_linear = torch.pow(
        torch.tensor(
            10.0,
            device=H_ul_true.device,
            dtype=real_dtype,
        ),
        torch.tensor(
            pilot_snr_db / 10.0,
            device=H_ul_true.device,
            dtype=real_dtype,
        ),
    )

    noise_variance = 1.0 / snr_linear

    noise = _complex_awgn(
        Y_clean.shape,
        noise_variance=noise_variance,
        device=H_ul_true.device,
        dtype=H_ul_true.dtype,
        generator=generator,
    )

    Y_pilot = Y_clean + noise

    # ------------------------------------------------------------------
    # LS estimate:
    #
    #     H_hat = Y P^H / T_p
    #
    # Y_pilot : [B,Nt,Nsc,Tp]
    # pilots  : [K,Tp]
    #
    # output  : [B,K,Nt,Nsc]
    # ------------------------------------------------------------------
    H_hat = torch.einsum(
        "bnst,kt->bkns",
        Y_pilot,
        pilots.conj(),
    ) / float(pilot_length)

    nmse = channel_nmse(
        H_hat,
        H_ul_true,
    )

    nmse_db = 10.0 * torch.log10(
        nmse.clamp_min(1e-12)
    )

    return ChannelEstimateResult(
        H_hat=H_hat,
        pilot_observation=Y_pilot,
        pilot_matrix=pilots,
        pilot_length=pilot_length,
        pilot_snr_db=pilot_snr_db,
        nmse_linear=float(nmse.item()),
        nmse_db=float(nmse_db.item()),
    )