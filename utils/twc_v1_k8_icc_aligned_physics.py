"""ICC-aligned K8 waveform, rank-eight pilot CE and causal receiver statistics.

Adapted from the archived ICC round8/ul_unet implementation, not its K4
energy convention. Simulator truth is accepted only by ``transmit``. Public
recovery accepts one observation, estimated UL, historical power and calibrated
diagonal CE errors. The latter are modelled features, not calibrated posterior
covariances. No function here optimizes a model or updates an EMA in forward.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn


K, NT, NSC, DMAX = 8, 32, 256, 216
EP, ED, ENERGY = 16.0, 48.0, 64.0
CE_RIDGE = RX_RIDGE = .001
GAMMAS = (0., 5., 10., 15., 20., 25.)


def _protocol():
    # Import the already verified resource-only module without importing the
    # historical utils package (which initializes GPU/Sionna on import).
    name = "_icc_aligned_k8_resource_protocol"
    path = Path(__file__).with_name("twc_v1_k8_protocol.py")
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("ICC resource module identity collision")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def make_layout(arm):
    p = _protocol()
    if arm == "fdma":
        return p.FDMALayout()
    if arm == "hybrid_g4":
        return p.default_hybrid_layout(4)
    raise ValueError("Only fdma and hybrid_g4 are prescribed")


Layout = make_layout


def active_dimension(layout):
    if hasattr(layout, "data_positions"):
        return 24
    if (getattr(layout, "groups", None) == ((0, 1), (2, 3), (4, 5), (6, 7))
            and getattr(layout, "shared_dimension", None) == 40):
        return 44
    raise ValueError("The fixed K8 FDMA or G4 layout is required")


def _finite_complex(value, name, shape=None):
    if (not torch.is_tensor(value) or value.dtype != torch.complex64
            or shape is not None and tuple(value.shape) != tuple(shape)
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + " must be finite complex64 with shape " + str(shape))
    return value


def gamma_vector(gamma_db, batch, device):
    original = torch.as_tensor(gamma_db)
    if torch.is_complex(original) or original.dtype == torch.bool:
        raise TypeError("Public gamma must be real")
    value = torch.as_tensor(gamma_db, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch)
    if (value.shape != (batch,) or value.requires_grad
            or not bool(torch.isfinite(value).all() & (value >= 0).all() & (value <= 25).all())):
        raise ValueError("Public gamma must be finite nonlearned scalar or [B] in [0,25]")
    return value


def noise_variance(gamma_db, batch=None, device=None):
    original = torch.as_tensor(gamma_db)
    count = (original.numel() if original.ndim else 1) if batch is None else batch
    value = gamma_vector(gamma_db, count, device or original.device)
    return .25 * torch.pow(value.new_tensor(10.), -value / 10.)


def pilot_mask(device="cpu"):
    mask = torch.zeros((K, NSC), dtype=torch.bool, device=device)
    for user, positions in enumerate(_protocol().PILOT_POSITIONS_BY_USER):
        mask[user, list(positions)] = True
    return mask


def known_pilots(batch, device="cpu", polarity=None):
    if type(batch) is not int or batch < 1:
        raise ValueError("Positive frame count required")
    if polarity is None:
        polarity = torch.ones((batch, K), dtype=torch.float32, device=device)
    polarity = torch.as_tensor(polarity, device=device)
    if (polarity.shape != (batch, K) or polarity.requires_grad or torch.is_complex(polarity)
            or not bool(((polarity == 1) | (polarity == -1)).all())):
        raise ValueError("Known polarity must be source-independent [B,8] +/-1")
    return (pilot_mask(device)[None] * polarity.float()[..., None] * math.sqrt(EP / 8)).to(torch.complex64)


@dataclass
class Waveform:
    raw: torch.Tensor
    full: torch.Tensor
    active: torch.Tensor
    key: torch.Tensor
    shared: torch.Tensor
    pilot: torch.Tensor
    data: torch.Tensor
    x_grid: torch.Tensor

def build_waveform(raw, layout, polarity=None):
    batch = raw.shape[0] if torch.is_tensor(raw) and raw.ndim else 0
    _finite_complex(raw, "raw codeword", (batch, K, DMAX))
    n = active_dimension(layout)
    selected = raw[..., :n]
    norm = selected.abs().square().sum(-1, keepdim=True)
    if not bool((norm > 1e-12).all()):
        raise FloatingPointError("Near-zero selected codeword; no frame fallback")
    active = selected * (ED / norm).sqrt()
    data = raw.new_zeros((batch, K, NSC))
    if n == 24:
        index = torch.tensor(layout.data_positions, device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active)
        key, shared = active, active[..., :0]
    else:
        index = torch.tensor(layout.key_positions, device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., :4])
        index = torch.tensor([layout.shared_positions_by_group[user // 2] for user in range(K)], device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., 4:])
        key, shared = active[..., :4], active[..., 4:]
    pilots = known_pilots(batch, raw.device, polarity)
    full = torch.cat((active, raw.new_zeros((batch, K, DMAX - n))), -1)
    return Waveform(raw, full, active, key, shared, pilots, data, pilots + data)


def transmit(H_ul_true, x_grid, xi, gamma_db):
    batch = H_ul_true.shape[0]
    _finite_complex(H_ul_true, "simulator-only true UL", (batch, K, NT, NSC))
    _finite_complex(x_grid, "actual transmitted grid", (batch, K, NSC))
    _finite_complex(xi, "one standard complex noise", (batch, NT, NSC))
    if len({H_ul_true.device, x_grid.device, xi.device}) != 1:
        raise ValueError("Simulator tensors must share a device")
    n0 = noise_variance(gamma_db, batch, x_grid.device)
    return (H_ul_true * x_grid[:, :, None]).sum(1) + n0.sqrt()[:, None, None] * xi


def fourier_basis(device="cpu"):
    tone = torch.arange(NSC, dtype=torch.float32, device=device)[:, None]
    tap = torch.arange(8, dtype=torch.float32, device=device)[None]
    return (torch.exp(-2j * math.pi * tone * tap / NSC) / math.sqrt(NSC)).to(torch.complex64)


def fourier_design(pilots):
    batch = pilots.shape[0]
    _finite_complex(pilots, "actual known pilot amplitudes", (batch, K, NSC))
    own = pilot_mask(pilots.device)
    if (bool((pilots[:, ~own] != 0).any())
            or not bool(torch.isclose(pilots[:, own].abs(), pilots.real.new_tensor(math.sqrt(2.)), atol=1e-6, rtol=1e-6).all())):
        raise ValueError("Pilot support and actual energy16/UE must match the paid design")
    basis = fourier_basis(pilots.device)
    union = own.any(0)
    design = (pilots[..., None] * basis[None, None]).permute(0, 2, 1, 3).reshape(batch, NSC, K * 8)
    return design[:, union], basis, union


def fourier_diagnostics(pilots):
    design, basis, _ = fourier_design(pilots)
    gram = design.conj().transpose(-2, -1) @ design
    diagonal = gram.diagonal(dim1=-2, dim2=-1).real
    off = gram - torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1))
    return dict(design_shape=list(design.shape), design_rank=torch.linalg.matrix_rank(design.detach()),
        gram_diagonal=diagonal.detach(), gram_max_off_diagonal=off.abs().amax((-2, -1)).detach(),
        expected_diagonal=EP / NSC, ridge=CE_RIDGE,
        ridge_to_diagonal=CE_RIDGE * NSC / EP,
        basis_gram=basis.conj().T @ basis)


def fourier_ls(Y, pilots):
    """Rank8 unit-norm Fourier ridge CE; data RE are masked before being read."""
    batch = pilots.shape[0]
    if (not torch.is_tensor(Y) or Y.dtype != torch.complex64
            or tuple(Y.shape) != (batch, NT, NSC) or Y.device != pilots.device):
        raise ValueError("One complex64 [B,32,256] public observation required")
    design, basis, union = fourier_design(pilots)
    selected = Y[:, :, union]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Pilot observations must be finite")
    adjoint = design.conj().transpose(-2, -1)
    gram = adjoint @ design + CE_RIDGE * torch.eye(K * 8, dtype=Y.dtype, device=Y.device)
    coefficients = torch.linalg.solve(gram, adjoint @ selected.transpose(-2, -1))
    coefficients = coefficients.reshape(batch, K, 8, NT)
    result = torch.einsum("qr,bkrn->bknq", basis, coefficients)
    return _finite_complex(result, "Fourier initial UL", (batch, K, NT, NSC))


class ReceiverStatistics(nn.Module):
    """Causal, checkpointed power and independently calibrated CE-node errors.

    ``commit_power`` accepts an accumulated sum across all eight microbatches,
    only after the caller has successfully committed one optimizer update.
    Merely reading the prior never changes it. Eval mode forbids commits.
    """
    DECAY, SHRINK = .99, .05

    def __init__(self, layout, ce_errors=None):
        super().__init__()
        self.layout = layout
        n = active_dimension(layout)
        errors = torch.zeros((6, K, NSC)) if ce_errors is None else torch.as_tensor(ce_errors, dtype=torch.float32).detach().cpu().clone()
        if errors.shape == (6, NSC):
            errors = errors[:, None].expand(-1, K, -1).clone()
        if errors.shape != (6, K, NSC) or not bool(torch.isfinite(errors).all() & (errors >= 0).all()):
            raise ValueError("Independent CE node errors must be nonnegative [6,8,256] (or UE-averaged [6,256])")
        self.register_buffer("ce_errors", errors)
        self.register_buffer("ce_calibrated", torch.tensor(ce_errors is not None, dtype=torch.bool))
        self.register_buffer("power_ema", torch.full((K, n), ED / n))
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.int64))

    def nominal_power(self):
        values = self.power_ema.detach()
        if not bool(torch.isfinite(values).all() & (values >= 0).all() & (values.sum(-1) > 1e-12).all()):
            raise FloatingPointError("Invalid stored historical power, no silent reset")
        normalized = ED * values / values.sum(-1, keepdim=True)
        return (1 - self.SHRINK) * normalized + self.SHRINK * (ED / values.shape[-1])

    def ce_variance(self, gamma_db):
        if not bool(self.ce_calibrated):
            raise ValueError("Independent new CE calibration required, no legacy or AWGN-only fallback")
        values = torch.as_tensor(gamma_db)
        batch = values.numel() if values.ndim else 1
        gamma = gamma_vector(gamma_db, batch, self.ce_errors.device)
        lower = torch.div(gamma, 5, rounding_mode="floor").long().clamp(max=4)
        weight = ((gamma - 5 * lower) / 5)[:, None, None]
        result = (1 - weight) * self.ce_errors[lower] + weight * self.ce_errors[lower + 1]
        if not bool(torch.isfinite(result).all() & (result >= 0).all()):
            raise FloatingPointError("Invalid interpolated per-RE CE errors")
        return result

    @torch.no_grad()
    def commit_power(self, power_sum, frames, *, expected_previous_updates):
        if not self.training:
            raise RuntimeError("Historical power is frozen outside TRAIN")
        if type(frames) is not int or frames < 1 or type(expected_previous_updates) is not int:
            raise ValueError("Explicit positive complete-effective-batch frame count and prior update count required")
        if int(self.ema_updates) != expected_previous_updates:
            raise ValueError("Power EMA commit is stale or duplicated")
        values = torch.as_tensor(power_sum, device=self.power_ema.device).detach()
        if (values.dtype != self.power_ema.dtype or values.shape != self.power_ema.shape
                or not bool(torch.isfinite(values).all() & (values >= 0).all())
                or not bool(torch.isclose(values.sum(-1), torch.full_like(values.sum(-1), ED * frames), atol=ED * frames * 2e-5, rtol=1e-5).all())):
            raise ValueError("EMA accepts only accumulated actual jointly-normalized transmitted power")
        self.power_ema.mul_(self.DECAY).add_(values / frames, alpha=1 - self.DECAY)
        self.ema_updates.add_(1)


def recover(Y, H_ul_est, layout, prior, ce_variance, gamma_db):
    """Public scalar/shared receiver, using the corrected UL everywhere.

    The reliability is inverse diagonal modelled codeword MSE including CE
    uncertainty; it is not a measured post-detection SNR or exact posterior.
    """
    batch = Y.shape[0]
    _finite_complex(Y, "one noisy public observation", (batch, NT, NSC))
    _finite_complex(H_ul_est, "corrected public UL estimate", (batch, K, NT, NSC))
    n = active_dimension(layout)
    prior = torch.as_tensor(prior, dtype=torch.float32, device=Y.device).detach()
    ce_variance = torch.as_tensor(ce_variance, dtype=torch.float32, device=Y.device).detach()
    if ce_variance.shape == (1, K, NSC):
        ce_variance = ce_variance.expand(batch, -1, -1)
    if (prior.shape != (K, n) or ce_variance.shape != (batch, K, NSC)
            or not bool(torch.isfinite(prior).all() & (prior > 0).all())
            or not bool(torch.isfinite(ce_variance).all() & (ce_variance >= 0).all())
            or not bool(torch.isclose(prior.sum(-1), torch.full((K,), ED, device=Y.device), atol=2e-4, rtol=1e-5).all())):
        raise ValueError("Positive per-coordinate historical prior summing48 and nonnegative per-RE CE variance required")
    n0 = noise_variance(gamma_db, batch, Y.device)
    exclusive = layout.data_positions if n == 24 else layout.key_positions
    index = torch.tensor(exclusive, device=Y.device)
    d = index.shape[-1]
    h = H_ul_est.gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
    obs = Y[:, None].expand(-1, K, -1, -1).gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
    channel_power = h.abs().square().sum(2)
    if not bool((channel_power > 1e-12).all()):
        raise ValueError("Numerically unidentifiable exclusive data channel")
    pi = prior[:, :d]
    error = ce_variance.gather(-1, index[None].expand(batch, -1, -1))
    equiv = n0[:, None, None] + pi[None] * error
    ridge = RX_RIDGE + equiv / pi[None]
    weights = h.conj() / (channel_power + ridge)[:, :, None]
    z_key = (weights * obs).sum(2)
    v_key = (channel_power / (channel_power + ridge) - 1).square() * pi[None] + equiv * weights.abs().square().sum(2)
    z_shared, v_shared, shared_ridges = [], [], []
    if n == 44:
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            users = torch.tensor(members, device=Y.device)
            tones = torch.tensor(positions, device=Y.device)
            g = H_ul_est.index_select(1, users).index_select(-1, tones).permute(0, 3, 2, 1)
            if not bool((torch.linalg.matrix_rank(g.detach()) == 2).all()):
                raise ValueError("Unidentifiable actual32x2 shared channel; no oracle fallback")
            pi_group = prior.index_select(0, users)[:, 4:].T
            vi = ce_variance.index_select(1, users).index_select(-1, tones).transpose(1, 2)
            effective = n0[:, None] + (pi_group[None] * vi).sum(-1)
            ridge_group = RX_RIDGE + effective[..., None] / pi_group[None]
            gh = g.conj().transpose(-2, -1)
            gram = gh @ g + torch.diag_embed(ridge_group).to(g.dtype)
            w = torch.linalg.solve(gram, gh)
            group_y = Y.index_select(-1, tones).transpose(-2, -1)
            z_shared.append((w @ group_y[..., None]).squeeze(-1).transpose(1, 2))
            deviation = w @ g - torch.eye(2, dtype=g.dtype, device=Y.device)
            v_shared.append(((deviation.abs().square() * pi_group[None, :, None]).sum(-1)
                + effective[..., None] * w.abs().square().sum(-1)).transpose(1, 2))
            shared_ridges.append(ridge_group.transpose(1, 2))
    shared = torch.cat(z_shared, 1) if z_shared else Y.new_empty((batch, K, 0))
    active = torch.cat((z_key, shared), -1)
    variance = torch.cat((v_key, torch.cat(v_shared, 1)), -1) if v_shared else v_key
    if not bool(torch.isfinite(active).all() & torch.isfinite(variance).all() & (variance > 0).all()):
        raise FloatingPointError("Nonfinite statistical recovery or nonpositive variance")
    padding = Y.new_zeros((batch, K, DMAX - n))
    soft = torch.cat((active, padding), -1)
    reliability = torch.cat((variance.reciprocal(), padding.real), -1)
    return dict(soft=soft, z_hat=soft, reliability=reliability, variance=variance,
        key=z_key, shared=shared, H_ul_est=H_ul_est, Y_res=Y,
        prior=prior, ce_variance=ce_variance, noise=n0,
        key_effective_lambda=ridge,
        shared_effective_lambda=torch.cat(shared_ridges, 1) if shared_ridges else Y.real.new_empty((batch, K, 0)),
        csi_mode="pilot_rank8_plus_new_frozen_pilot_only_UNet",
        corrected_UL_used_for_all_H_dependent_operations=True,
        reliability_definition="inverse diagonal modelled recovery MSE, not exact posterior")


def energy_ledger(wave, layout):
    n = active_dimension(layout)
    return dict(pilot=wave.pilot.abs().square().sum(-1), data=wave.data.abs().square().sum(-1),
        key=wave.active[..., :n if n == 24 else 4].abs().square().sum(-1),
        shared=wave.shared.abs().square().sum(-1), total=wave.x_grid.abs().square().sum(-1),
        pilot_data_cross=2 * (wave.pilot.conj() * wave.data).sum(-1).real)
