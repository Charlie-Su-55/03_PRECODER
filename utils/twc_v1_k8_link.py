"""CPU/GPU tensor primitives for the TWC 1.0 K8 feedback-link prototype.

The estimator here is deterministic linear interpolation from sparse pilot-LS
samples. It is only a runnable placeholder for the planned LS+U-Net module.
This file does not import the side-effectful ``utils`` package initializer.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_NAME = "_twc_v1_k8_protocol"
_MODULE_PATH = REPO_ROOT / "utils" / "twc_v1_k8_protocol.py"
if _MODULE_NAME in sys.modules:
    protocol = sys.modules[_MODULE_NAME]
    if Path(protocol.__file__).resolve() != _MODULE_PATH.resolve():
        raise RuntimeError("TWC K8 protocol module identity collision")
else:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load TWC K8 protocol module")
    protocol = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = protocol
    try:
        spec.loader.exec_module(protocol)
    except BaseException:
        del sys.modules[_MODULE_NAME]
        raise

EnergySplit = protocol.EnergySplit
HybridLayout = protocol.HybridLayout
FDMALayout = protocol.FDMALayout
K_USERS = protocol.K_USERS
N_ANTENNAS = protocol.N_ANTENNAS
N_SUBCARRIERS = protocol.N_SUBCARRIERS
PILOTS_PER_USER = protocol.PILOTS_PER_USER
TOTAL_RE = protocol.TOTAL_RE


def _validate_complex(value, name, shape=None):
    if not torch.is_tensor(value) or value.dtype != torch.complex64:
        raise TypeError(f"{name} must be a complex64 tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains nonfinite values")


def _validate_split(split):
    if not isinstance(split, EnergySplit):
        raise TypeError("An explicit K8 EnergySplit is required")


def _validate_layout(layout):
    if not isinstance(layout, HybridLayout):
        raise TypeError("An explicit K8 HybridLayout is required")


def _channel_error_variance(value, batch, device):
    value = torch.as_tensor(value, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch, K_USERS)
    elif value.shape == (K_USERS,):
        value = value[None].expand(batch, -1)
    elif value.shape != (batch, K_USERS):
        raise ValueError("channel_error_variance must be scalar, [8], or [B,8]")
    if not bool(torch.isfinite(value).all() & (value >= 0).all()):
        raise ValueError("channel_error_variance must be finite and nonnegative")
    return value


def _n0(feedback_gamma_db):
    return protocol.feedback_noise_variance(feedback_gamma_db)


def transmit_pilots(H_ul, pilot_energy_per_ue, feedback_gamma_db, standard_noise):
    """Send one UE's eight known pilots on each of its disjoint spread REs."""
    if isinstance(pilot_energy_per_ue, bool) or not isinstance(pilot_energy_per_ue, (int, float)):
        raise TypeError("pilot_energy_per_ue must be a real scalar")
    if not math.isfinite(pilot_energy_per_ue) or not 0 < pilot_energy_per_ue < protocol.TOTAL_FRAME_ENERGY_PER_USER:
        raise ValueError("pilot energy must be positive and leave positive data energy")
    if H_ul.ndim != 4 or H_ul.shape[0] < 1:
        raise ValueError("H_ul must have shape [B,8,32,256]")
    batch = H_ul.shape[0]
    _validate_complex(H_ul, "H_ul", (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS))
    _validate_complex(standard_noise, "pilot standard noise", (batch, N_ANTENNAS, N_SUBCARRIERS))
    if H_ul.device != standard_noise.device:
        raise ValueError("H_ul and pilot noise must share a device")
    n0 = _n0(feedback_gamma_db)
    result = torch.zeros_like(standard_noise)
    amplitude = math.sqrt(float(pilot_energy_per_ue) / PILOTS_PER_USER)
    noise_scale = math.sqrt(n0 / 2.0)
    for user, positions in enumerate(protocol.PILOT_POSITIONS_BY_USER):
        q = torch.tensor(positions, device=H_ul.device, dtype=torch.long)
        result[:, :, q] = amplitude * H_ul[:, user, :, q] + noise_scale * standard_noise[:, :, q]
    return result


def estimate_ul_ls_interpolated(pilot_observation, pilot_energy_per_ue):
    """Pilot LS followed by per-antenna linear interpolation over subcarriers."""
    if isinstance(pilot_energy_per_ue, bool) or not isinstance(pilot_energy_per_ue, (int, float)):
        raise TypeError("pilot_energy_per_ue must be a real scalar")
    if not math.isfinite(pilot_energy_per_ue) or not 0 < pilot_energy_per_ue < protocol.TOTAL_FRAME_ENERGY_PER_USER:
        raise ValueError("pilot energy must be positive and leave positive data energy")
    if pilot_observation.ndim != 3 or pilot_observation.shape[0] < 1:
        raise ValueError("Pilot observation must have shape [B,32,256]")
    batch = pilot_observation.shape[0]
    _validate_complex(pilot_observation, "pilot observation", (batch, N_ANTENNAS, N_SUBCARRIERS))
    q_all = torch.arange(N_SUBCARRIERS, device=pilot_observation.device)
    estimate = torch.empty((batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS),
                           device=pilot_observation.device, dtype=torch.complex64)
    for user, positions in enumerate(protocol.PILOT_POSITIONS_BY_USER):
        q_pilot = torch.tensor(positions, device=pilot_observation.device, dtype=torch.long)
        pilot_ls = pilot_observation.index_select(-1, q_pilot) / math.sqrt(
            float(pilot_energy_per_ue) / PILOTS_PER_USER)
        right = torch.searchsorted(q_pilot, q_all).clamp(1, PILOTS_PER_USER - 1)
        left = right - 1
        q0 = q_pilot.index_select(0, left).to(torch.float32)
        q1 = q_pilot.index_select(0, right).to(torch.float32)
        alpha = ((q_all.to(torch.float32) - q0) / (q1 - q0)).clamp(0.0, 1.0)
        low = pilot_ls.index_select(-1, left)
        high = pilot_ls.index_select(-1, right)
        estimate[:, user] = low * (1.0 - alpha) + high * alpha
    if not bool(torch.isfinite(estimate).all()):
        raise FloatingPointError("Interpolated LS channel contains nonfinite values")
    return estimate


def estimate_ul_ls_fourier(pilot_observation, pilot_energy_per_ue, *, rank=4, ridge=1e-3):
    """Regularized pilot-only delay-basis LS; no channel labels are accepted.

    The dictionary is ``exp(-j 2 pi q l / Nsc)`` for l=0..rank-1. Ridge is
    applied to the pilot-normalized Gram matrix ``AᴴA/P``. The eight evenly
    spaced pilot tones can alias delays separated by eight delay bins, so this
    estimator is a candidate to compare, not an assumed improvement.
    """
    if isinstance(pilot_energy_per_ue, bool) or not isinstance(pilot_energy_per_ue, (int, float)):
        raise TypeError("pilot_energy_per_ue must be a real scalar")
    if not math.isfinite(pilot_energy_per_ue) or not 0 < pilot_energy_per_ue < protocol.TOTAL_FRAME_ENERGY_PER_USER:
        raise ValueError("pilot energy must be positive and leave positive data energy")
    if type(rank) is not int or not 1 <= rank <= PILOTS_PER_USER:
        raise ValueError(f"rank must be an integer in [1,{PILOTS_PER_USER}]")
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if pilot_observation.ndim != 3 or pilot_observation.shape[0] < 1:
        raise ValueError("Pilot observation must have shape [B,32,256]")
    batch = pilot_observation.shape[0]
    _validate_complex(pilot_observation, "pilot observation", (batch, N_ANTENNAS, N_SUBCARRIERS))
    estimate = torch.empty((batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS),
                           device=pilot_observation.device, dtype=torch.complex64)
    q_all = torch.arange(N_SUBCARRIERS, device=pilot_observation.device, dtype=torch.float32)
    taps = torch.arange(rank, device=pilot_observation.device, dtype=torch.float32)
    basis_all = torch.exp((-2j * math.pi / N_SUBCARRIERS) * q_all[:, None] * taps[None, :])
    identity = torch.eye(rank, dtype=torch.complex64, device=pilot_observation.device)
    amplitude = math.sqrt(float(pilot_energy_per_ue) / PILOTS_PER_USER)
    for user, positions in enumerate(protocol.PILOT_POSITIONS_BY_USER):
        q = torch.tensor(positions, device=pilot_observation.device, dtype=torch.long)
        basis_pilot = basis_all.index_select(0, q)
        gram = basis_pilot.conj().transpose(0, 1) @ basis_pilot / PILOTS_PER_USER
        rhs = basis_pilot.conj().transpose(0, 1)
        observed = pilot_observation.index_select(-1, q) / amplitude
        coefficients = torch.linalg.solve(gram + float(ridge) * identity,
                                            rhs @ observed.transpose(1, 2) / PILOTS_PER_USER)
        estimate[:, user] = (basis_all @ coefficients).transpose(1, 2)
    if not bool(torch.isfinite(estimate).all()):
        raise FloatingPointError("Fourier pilot-LS channel estimate contains nonfinite values")
    return estimate


def _normalize_branch(raw, target_energy, name):
    if raw.ndim != 3 or raw.shape[0] < 1:
        raise ValueError(f"{name} must have shape [B,8,D]")
    _validate_complex(raw, name, (raw.shape[0], K_USERS, raw.shape[-1]))
    energy = raw.abs().square().sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(energy).all() & (energy > 1e-12).all()):
        raise FloatingPointError(f"{name} contains zero/nonfinite per-UE energy")
    return raw * (target_energy / energy).sqrt()


def transmit_feedback_data(H_ul, layout, split, feedback_gamma_db,
                           key_symbols, shared_symbols, standard_noise):
    """Transmit key and group-shared codewords; shared REs superpose by group."""
    _validate_layout(layout)
    _validate_split(split)
    if H_ul.ndim != 4 or H_ul.shape[0] < 1:
        raise ValueError("H_ul must have shape [B,8,32,256]")
    batch = H_ul.shape[0]
    _validate_complex(H_ul, "H_ul", (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS))
    _validate_complex(key_symbols, "key symbols", (batch, K_USERS, protocol.KEY_DIM_PER_USER))
    _validate_complex(shared_symbols, "shared symbols", (batch, K_USERS, layout.shared_dimension))
    _validate_complex(standard_noise, "data standard noise", (batch, N_ANTENNAS, N_SUBCARRIERS))
    if len({H_ul.device, key_symbols.device, shared_symbols.device, standard_noise.device}) != 1:
        raise ValueError("All physical-link tensors must share a device")

    key_tx = _normalize_branch(key_symbols, split.key, "key symbols")
    shared_tx = _normalize_branch(shared_symbols, split.shared, "shared symbols")
    X = torch.zeros((batch, K_USERS, TOTAL_RE), device=H_ul.device, dtype=torch.complex64)
    for user, positions in enumerate(layout.key_positions):
        q = torch.tensor(positions, device=H_ul.device, dtype=torch.long)
        X[:, user, q] = key_tx[:, user]
    for members, positions in zip(layout.groups, layout.shared_positions_by_group):
        q = torch.tensor(positions, device=H_ul.device, dtype=torch.long)
        for user in members:
            X[:, user, q] = shared_tx[:, user]

    active = torch.zeros(TOTAL_RE, device=H_ul.device, dtype=torch.bool)
    for positions in layout.key_positions + layout.shared_positions_by_group:
        active[torch.tensor(positions, device=H_ul.device, dtype=torch.long)] = True
    Y_clean = (H_ul * X[:, :, None, :]).sum(dim=1)
    Y = Y_clean + math.sqrt(_n0(feedback_gamma_db) / 2.0) * standard_noise * active
    actual = X.abs().square().sum(dim=-1)
    if not bool(torch.isfinite(actual).all() & ((actual - split.key - split.shared).abs() <= 1e-4).all()):
        raise FloatingPointError("Per-UE data energy normalization failed")
    return {
        "Y": Y,
        "X": X,
        "Y_clean": Y_clean,
        "key_symbols_tx": key_tx,
        "shared_symbols_tx": shared_tx,
        "actual_key_energy": key_tx.abs().square().sum(-1),
        "actual_shared_energy": shared_tx.abs().square().sum(-1),
        "actual_data_energy": actual,
        "N0": _n0(feedback_gamma_db),
    }


def recover_feedback_data(Y, H_ul_est, layout, split, feedback_gamma_db,
                          channel_error_variance=0.0):
    """Calibration-aware key and group-shared linear-MMSE soft recovery.

    CE error is approximated as independent isotropic complex error. Its
    measured variance comes from the disjoint calibration role and contributes
    ``sigma_e^2`` per key stream and ``sum_i sigma_e,i^2`` per shared group.
    """
    _validate_layout(layout)
    _validate_split(split)
    if Y.ndim != 3 or Y.shape[0] < 1:
        raise ValueError("Y must have shape [B,32,256]")
    batch = Y.shape[0]
    _validate_complex(Y, "Y", (batch, N_ANTENNAS, N_SUBCARRIERS))
    _validate_complex(H_ul_est, "H_ul_est", (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS))
    if Y.device != H_ul_est.device:
        raise ValueError("Y and H_ul_est must share a device")
    n0 = _n0(feedback_gamma_db)
    ce_var = _channel_error_variance(channel_error_variance, batch, Y.device)
    key_power = split.key / protocol.KEY_DIM_PER_USER
    shared_power = split.shared / layout.shared_dimension
    key_ridge = n0 / key_power + ce_var + 1e-8
    key_estimate = torch.empty((batch, K_USERS, protocol.KEY_DIM_PER_USER),
                               device=Y.device, dtype=torch.complex64)
    key_reliability = torch.empty((batch, K_USERS, protocol.KEY_DIM_PER_USER),
                                  device=Y.device, dtype=torch.float32)
    key_posterior = torch.empty_like(key_reliability)
    for user, positions in enumerate(layout.key_positions):
        q = torch.tensor(positions, device=Y.device, dtype=torch.long)
        h = H_ul_est[:, user].index_select(-1, q)
        y = Y.index_select(-1, q)
        channel_power = h.abs().square().sum(dim=1)
        ridge = key_ridge[:, user, None]
        denominator = channel_power + ridge
        numerator = (h.conj() * y).sum(dim=1)
        key_estimate[:, user] = numerator / denominator
        key_reliability[:, user] = channel_power / denominator
        key_posterior[:, user] = key_power * ridge / denominator

    shared_estimate = torch.empty((batch, K_USERS, layout.shared_dimension),
                                  device=Y.device, dtype=torch.complex64)
    shared_reliability = torch.empty((batch, K_USERS, layout.shared_dimension),
                                     device=Y.device, dtype=torch.float32)
    shared_posterior = torch.empty_like(shared_reliability)
    shared_ridges = []
    for members, positions in zip(layout.groups, layout.shared_positions_by_group):
        users = torch.tensor(members, device=Y.device, dtype=torch.long)
        q = torch.tensor(positions, device=Y.device, dtype=torch.long)
        # [B,g,Nt,a] -> [B,a,Nt,g]
        h = H_ul_est.index_select(1, users).index_select(-1, q).permute(0, 3, 2, 1)
        y = Y.index_select(-1, q).transpose(1, 2).unsqueeze(-1)
        gh = h.conj().transpose(-2, -1)
        identity = torch.eye(len(members), device=Y.device, dtype=torch.complex64)
        ridge = n0 / shared_power + ce_var.index_select(1, users).sum(dim=1) + 1e-8
        shared_ridges.append(ridge)
        system = gh @ h + ridge[:, None, None, None] * identity
        recovered = torch.linalg.solve(system, gh @ y).squeeze(-1)
        inverse = torch.linalg.inv(system)
        posterior_diag = (shared_power * ridge[:, None, None]
                          * inverse.diagonal(dim1=-2, dim2=-1).real).clamp(0.0, shared_power)
        for local_user, user in enumerate(members):
            shared_estimate[:, user] = recovered[:, :, local_user]
            shared_posterior[:, user] = posterior_diag[:, :, local_user]
            shared_reliability[:, user] = (1 - posterior_diag[:, :, local_user] / shared_power).clamp(0, 1)
    if not bool(torch.isfinite(key_estimate).all() & torch.isfinite(shared_estimate).all()
                & torch.isfinite(key_reliability).all() & torch.isfinite(shared_reliability).all()):
        raise FloatingPointError("Recovered data symbols contain nonfinite values")
    return {
        "key_symbols_hat": key_estimate,
        "shared_symbols_hat": shared_estimate,
        "key_ridge": key_ridge,
        "shared_ridge": torch.stack(shared_ridges, dim=1),
        "key_reliability": key_reliability,
        "shared_reliability": shared_reliability,
        "key_posterior_variance": key_posterior,
        "shared_posterior_variance": shared_posterior,
        "channel_error_variance": ce_var,
    }


def transmit_feedback_fdma_data(H_ul, layout, data_energy_per_ue, feedback_gamma_db,
                                data_symbols, standard_noise):
    """Transmit 24 orthogonal data RE per UE for the matched FDMA control."""
    if not isinstance(layout, FDMALayout):
        raise TypeError("An explicit K8 FDMALayout is required")
    if (isinstance(data_energy_per_ue, bool) or not isinstance(data_energy_per_ue, (int, float))
            or not math.isfinite(data_energy_per_ue) or data_energy_per_ue <= 0):
        raise ValueError("FDMA per-UE data energy must be finite and positive")
    if H_ul.ndim != 4 or H_ul.shape[0] < 1:
        raise ValueError("H_ul must have shape [B,8,32,256]")
    batch = H_ul.shape[0]
    _validate_complex(H_ul, "H_ul", (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS))
    _validate_complex(data_symbols, "FDMA data symbols", (batch, K_USERS, layout.data_coordinates_per_user))
    _validate_complex(standard_noise, "FDMA data standard noise", (batch, N_ANTENNAS, N_SUBCARRIERS))
    if len({H_ul.device, data_symbols.device, standard_noise.device}) != 1:
        raise ValueError("All FDMA tensors must share a device")
    data_tx = _normalize_branch(data_symbols, float(data_energy_per_ue), "FDMA data symbols")
    X = torch.zeros((batch, K_USERS, TOTAL_RE), device=H_ul.device, dtype=torch.complex64)
    all_positions = []
    for user, positions in enumerate(layout.data_positions):
        q = torch.tensor(positions, device=H_ul.device, dtype=torch.long)
        X[:, user, q] = data_tx[:, user]
        all_positions.extend(positions)
    active = torch.zeros(TOTAL_RE, device=H_ul.device, dtype=torch.bool)
    active[torch.tensor(all_positions, device=H_ul.device, dtype=torch.long)] = True
    Y_clean = (H_ul * X[:, :, None, :]).sum(dim=1)
    Y = Y_clean + math.sqrt(_n0(feedback_gamma_db) / 2.0) * standard_noise * active
    actual = X.abs().square().sum(dim=-1)
    if not bool(torch.isfinite(actual).all() & ((actual - data_energy_per_ue).abs() <= 1e-4).all()):
        raise FloatingPointError("FDMA per-UE data energy normalization failed")
    return {"Y": Y, "Y_clean": Y_clean, "X": X,
        "data_symbols_tx": data_tx, "actual_data_energy": actual,
        "N0": _n0(feedback_gamma_db)}


def recover_feedback_fdma_data(Y, H_ul_est, layout, data_energy_per_ue,
                                feedback_gamma_db, channel_error_variance=0.0):
    """Calibration-aware per-user MRC/LMMSE soft recovery for FDMA."""
    if not isinstance(layout, FDMALayout):
        raise TypeError("An explicit K8 FDMALayout is required")
    if Y.ndim != 3 or Y.shape[0] < 1:
        raise ValueError("Y must have shape [B,32,256]")
    batch = Y.shape[0]
    _validate_complex(Y, "FDMA observation", (batch, N_ANTENNAS, N_SUBCARRIERS))
    _validate_complex(H_ul_est, "H_ul_est", (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS))
    if Y.device != H_ul_est.device:
        raise ValueError("FDMA observation and estimated UL CSI must share a device")
    ce_var = _channel_error_variance(channel_error_variance, batch, Y.device)
    coordinate_power = float(data_energy_per_ue) / layout.data_coordinates_per_user
    ridge = _n0(feedback_gamma_db) / coordinate_power + ce_var + 1e-8
    estimate = torch.empty((batch, K_USERS, layout.data_coordinates_per_user),
                           dtype=torch.complex64, device=Y.device)
    reliability = torch.empty((batch, K_USERS, layout.data_coordinates_per_user),
                               dtype=torch.float32, device=Y.device)
    posterior = torch.empty_like(reliability)
    for user, positions in enumerate(layout.data_positions):
        q = torch.tensor(positions, device=Y.device, dtype=torch.long)
        h = H_ul_est[:, user].index_select(-1, q)
        y = Y.index_select(-1, q)
        channel_power = h.abs().square().sum(dim=1)
        denominator = channel_power + ridge[:, user, None]
        estimate[:, user] = (h.conj() * y).sum(dim=1) / denominator
        reliability[:, user] = channel_power / denominator
        posterior[:, user] = coordinate_power * ridge[:, user, None] / denominator
    if not bool(torch.isfinite(estimate).all() & torch.isfinite(reliability).all()
                & torch.isfinite(posterior).all()):
        raise FloatingPointError("FDMA recovered symbols contain nonfinite values")
    return {"data_symbols_hat": estimate, "reliability": reliability,
        "posterior_variance": posterior, "ridge": ridge,
        "channel_error_variance": ce_var}


def protocol_metadata():
    return {
        "link": "K8 pilots -> LS+linear interpolation -> key/shared TX -> group MMSE soft recovery",
        "pilot_estimator": "per-UE LS on 8 spread pilot REs, linear complex interpolation over subcarriers",
        "pilot_estimator_is_unet": False,
        "estimator_role": "CPU runnable placeholder pending planned LS+U-Net",
        "feedback_receiver": "scalar key MMSE and group-wise shared MMSE using estimated UL CSI",
        "channel_coherence_assumption": "UL channel unchanged between pilot and feedback-data phases",
        "feedback_energy": "Ep+Ekey+EAS=64 per UE, branch energy normalized per frame",
        "noise": "same absolute N0 at each feedback gamma for all layouts",
        "downlink_csi_reconstruction_and_precoding": "not implemented by this physical-link prototype",
    }
