"""K=16 user-load extension of the K8 frame (protocol, physics, receiver, models, data, objective).

Same frame as K8: 256 RE = 64 pilot + 192 data, per-UE energy 64 = pilot 16 +
data 48, N0 = 0.25*10^(-gamma/10), Nt=32, Nsc=256, RZF alpha=.03, DL 25 dB.
With 16 users each UE gets 4 comb pilots (spacing 64, amplitude 2) and the
pilot-only LS initial estimate has rank 4.

  FDMA    12 dedicated data RE per UE inside its own 16-RE block
  Hybrid  8 pairs; 4 key RE per UE + 16 RE shared within the pair -> 20 dims

Models are the K8 architectures instantiated with K=16 and Dmax=24 from an
isolated copy of models/twc_v1_k8_icc_aligned.py (the K8 module is not touched).
Channels: each K16 sample concatenates two independent K8 CDL-C draws (users
are i.i.d. in the generator) from a K16-specific seed namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
K, NT, NSC, NRE = 16, 32, 256, 256
PILOTS_PER_UE, EP, ED = 4, 16., 48.
DMAX, WIDTH = 24, 128
FDMA_DIM, KEY_DIM, SHARED_DIM = 12, 4, 16
HYBRID_DIM = KEY_DIM+SHARED_DIM
CE_RIDGE = RX_RIDGE = .001
GAMMAS = (0., 5., 10., 15., 20., 25.)


# ------------------------------------------------------------------ protocol --

PILOT_POSITIONS_BY_USER = tuple(tuple(4*u+64*m for m in range(PILOTS_PER_UE)) for u in range(K))
_PILOT_SET = {p for row in PILOT_POSITIONS_BY_USER for p in row}
DATA_POSITIONS = tuple(i for i in range(NRE) if i not in _PILOT_SET)


@dataclass(frozen=True)
class FDMALayout:
    data_positions: tuple = tuple(tuple(i for i in DATA_POSITIONS if 16*u <= i < 16*u+16) for u in range(K))
    name: str = 'fdma'


@dataclass(frozen=True)
class HybridLayout:
    groups: tuple = tuple((2*g, 2*g+1) for g in range(K//2))
    key_positions: tuple = tuple(DATA_POSITIONS[4*u:4*u+4] for u in range(K))
    shared_positions_by_group: tuple = tuple(DATA_POSITIONS[64+16*g:64+16*g+16] for g in range(K//2))
    shared_dimension: int = SHARED_DIM
    name: str = 'hybrid'


def make_layout(arm):
    if arm == 'fdma':
        return FDMALayout()
    if arm == 'hybrid':
        return HybridLayout()
    raise ValueError('K16 arm must be fdma or hybrid')


def active_dimension(layout):
    return FDMA_DIM if hasattr(layout, 'data_positions') else KEY_DIM+layout.shared_dimension


def check_protocol():
    f, h = FDMALayout(), HybridLayout()
    used = sorted(p for row in f.data_positions for p in row)
    assert used == list(DATA_POSITIONS) and all(len(r) == FDMA_DIM for r in f.data_positions)
    used = sorted([p for r in h.key_positions for p in r]+[p for r in h.shared_positions_by_group for p in r])
    assert used == list(DATA_POSITIONS) and len(_PILOT_SET) == 64
    return True


# ------------------------------------------------------------------- physics --

def noise_variance(gamma_db, batch, device):
    value = torch.as_tensor(gamma_db, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch)
    return .25*torch.pow(value.new_tensor(10.), -value/10.)


def pilot_mask(device='cpu'):
    mask = torch.zeros((K, NSC), dtype=torch.bool, device=device)
    for user, positions in enumerate(PILOT_POSITIONS_BY_USER):
        mask[user, list(positions)] = True
    return mask


def known_pilots(batch, device='cpu', polarity=None):
    if polarity is None:
        polarity = torch.ones((batch, K), device=device)
    if polarity.shape != (batch, K):
        raise ValueError('Known polarity [B,16] required')
    return (pilot_mask(device)[None]*polarity.float()[..., None]*math.sqrt(EP/PILOTS_PER_UE)).to(torch.complex64)


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


def group_of(layout):
    owner = [None]*K
    for index, members in enumerate(layout.groups):
        for user in members:
            owner[user] = index
    return owner


def build_waveform(raw, layout, polarity=None):
    batch = raw.shape[0]
    if raw.shape != (batch, K, DMAX) or raw.dtype != torch.complex64:
        raise ValueError('Raw codeword [B,16,24] complex64 required')
    n = active_dimension(layout)
    selected = raw[..., :n]
    norm = selected.abs().square().sum(-1, keepdim=True)
    if not bool((norm > 1e-12).all()):
        raise FloatingPointError('Near-zero selected codeword')
    active = selected*(ED/norm).sqrt()
    data = raw.new_zeros((batch, K, NSC))
    if hasattr(layout, 'data_positions'):
        index = torch.tensor(layout.data_positions, device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active)
        key, shared = active, active[..., :0]
    else:
        index = torch.tensor(layout.key_positions, device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., :KEY_DIM])
        owner = group_of(layout)
        index = torch.tensor([layout.shared_positions_by_group[owner[u]] for u in range(K)], device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., KEY_DIM:])
        key, shared = active[..., :KEY_DIM], active[..., KEY_DIM:]
    pilots = known_pilots(batch, raw.device, polarity)
    full = torch.cat((active, raw.new_zeros((batch, K, DMAX-n))), -1)
    return Waveform(raw, full, active, key, shared, pilots, data, pilots+data)


def transmit(H_ul, x_grid, xi, gamma_db):
    batch = H_ul.shape[0]
    if H_ul.shape != (batch, K, NT, NSC) or x_grid.shape != (batch, K, NSC) or xi.shape != (batch, NT, NSC):
        raise ValueError('K16 transmit shapes')
    n0 = noise_variance(gamma_db, batch, x_grid.device)
    return (H_ul*x_grid[:, :, None]).sum(1)+n0.sqrt()[:, None, None]*xi


def fourier_basis(device='cpu'):
    tone = torch.arange(NSC, dtype=torch.float32, device=device)[:, None]
    tap = torch.arange(PILOTS_PER_UE, dtype=torch.float32, device=device)[None]
    return (torch.exp(-2j*math.pi*tone*tap/NSC)/math.sqrt(NSC)).to(torch.complex64)


def fourier_ls(Y, pilots):
    """Pilot-only rank-4 Fourier ridge LS for all 16 users (data RE never read)."""
    batch = pilots.shape[0]
    own = pilot_mask(pilots.device)
    basis = fourier_basis(pilots.device)
    union = own.any(0)
    design = (pilots[..., None]*basis[None, None]).permute(0, 2, 1, 3).reshape(batch, NSC, K*PILOTS_PER_UE)[:, union]
    adjoint = design.conj().transpose(-2, -1)
    gram = adjoint@design+CE_RIDGE*torch.eye(K*PILOTS_PER_UE, dtype=Y.dtype, device=Y.device)
    coefficients = torch.linalg.solve(gram, adjoint@Y[:, :, union].transpose(-2, -1))
    coefficients = coefficients.reshape(batch, K, PILOTS_PER_UE, NT)
    return torch.einsum('qr,bkrn->bknq', basis, coefficients)


# ------------------------------------------------------------------ receiver --

def recover(Y, H_ul_est, layout, prior, ce_variance, gamma_db):
    """Error-aware LMMSE: per-user on dedicated REs, joint 2-user per shared block."""
    batch = Y.shape[0]
    n = active_dimension(layout)
    prior = torch.as_tensor(prior, dtype=torch.float32, device=Y.device).detach()
    if prior.shape == (K, n):
        prior = prior[None].expand(batch, -1, -1)
    ce_variance = torch.as_tensor(ce_variance, dtype=torch.float32, device=Y.device).detach()
    if prior.shape != (batch, K, n) or ce_variance.shape != (batch, K, NSC) or not bool((prior > 0).all()):
        raise ValueError('Positive per-MU prior [B,16,n] and CE variance [B,16,256] required')
    n0 = noise_variance(gamma_db, batch, Y.device)
    fdma = hasattr(layout, 'data_positions')
    index = torch.tensor(layout.data_positions if fdma else layout.key_positions, device=Y.device)
    d = index.shape[-1]
    h = H_ul_est.gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
    obs = Y[:, None].expand(-1, K, -1, -1).gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
    channel_power = h.abs().square().sum(2)
    pi = prior[:, :, :d]
    error = ce_variance.gather(-1, index[None].expand(batch, -1, -1))
    equiv = n0[:, None, None]+pi*error
    ridge = RX_RIDGE+equiv/pi
    weights = h.conj()/(channel_power+ridge)[:, :, None]
    z_key = (weights*obs).sum(2)
    v_key = (channel_power/(channel_power+ridge)-1).square()*pi+equiv*weights.abs().square().sum(2)
    if fdma:
        active, variance = z_key, v_key
    else:
        z_shared = torch.zeros((batch, K, n-KEY_DIM), dtype=Y.dtype, device=Y.device)
        v_shared = torch.zeros((batch, K, n-KEY_DIM), dtype=torch.float32, device=Y.device)
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            m = len(members)
            users = torch.tensor(members, device=Y.device)
            tones = torch.tensor(positions, device=Y.device)
            g = H_ul_est.index_select(1, users).index_select(-1, tones).permute(0, 3, 2, 1)
            pi_group = prior.index_select(1, users)[:, :, KEY_DIM:].transpose(1, 2)
            vi = ce_variance.index_select(1, users).index_select(-1, tones).transpose(1, 2)
            effective = n0[:, None]+(pi_group*vi).sum(-1)
            ridge_group = RX_RIDGE+effective[..., None]/pi_group
            gh = g.conj().transpose(-2, -1)
            w = torch.linalg.solve(gh@g+torch.diag_embed(ridge_group).to(g.dtype), gh)
            z = (w@Y.index_select(-1, tones).transpose(-2, -1)[..., None]).squeeze(-1).transpose(1, 2)
            deviation = w@g-torch.eye(m, dtype=g.dtype, device=Y.device)
            v = ((deviation.abs().square()*pi_group[:, :, None, :]).sum(-1)
                 +effective[..., None]*w.abs().square().sum(-1)).transpose(1, 2)
            z_shared = z_shared.index_copy(1, users, z)
            v_shared = v_shared.index_copy(1, users, v)
        active = torch.cat((z_key, z_shared), -1)
        variance = torch.cat((v_key, v_shared), -1)
    if not bool(torch.isfinite(active).all() & torch.isfinite(variance).all() & (variance > 0).all()):
        raise FloatingPointError('Nonfinite statistical recovery')
    padding = Y.new_zeros((batch, K, DMAX-n))
    return dict(soft=torch.cat((active, padding), -1),
        reliability=torch.cat((variance.reciprocal(), padding.real), -1), Y_res=Y)


def support_geometry(layout):
    support = torch.full((K, DMAX), -1, dtype=torch.long)
    exclusive = torch.zeros_like(support, dtype=torch.bool)
    n = active_dimension(layout)
    if hasattr(layout, 'data_positions'):
        for user, positions in enumerate(layout.data_positions):
            support[user, :n] = torch.as_tensor(positions); exclusive[user, :n] = True
    else:
        for user, positions in enumerate(layout.key_positions):
            support[user, :KEY_DIM] = torch.as_tensor(positions); exclusive[user, :KEY_DIM] = True
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            for user in members:
                support[user, KEY_DIM:n] = torch.as_tensor(positions)
    return support, support >= 0, exclusive


class NodeStatistics(nn.Module):
    """Per-gamma-node causal power prior and a locked per-RE CE error table [6,16,256]."""
    DECAY, SHRINK = .99, .05

    def __init__(self, layout, ce_errors, power_init):
        super().__init__()
        self.layout = layout
        n = active_dimension(layout)
        errors = torch.as_tensor(ce_errors, dtype=torch.float32).detach().cpu().clone()
        power = torch.as_tensor(power_init, dtype=torch.float32).detach().cpu().clone()
        if errors.shape != (6, K, NSC) or power.shape != (K, n) or not bool((power > 0).all()):
            raise ValueError('CE errors [6,16,256] and positive power [16,n] required')
        self.register_buffer('ce_errors', errors)
        self.register_buffer('power_ema', power[None].repeat(6, 1, 1))
        self.register_buffer('node_updates', torch.zeros(6, dtype=torch.int64))

    def _weights(self, gamma):
        gamma = torch.as_tensor(gamma, dtype=torch.float32, device=self.power_ema.device).reshape(-1)
        lower = torch.div(gamma, 5, rounding_mode='floor').long().clamp(0, 4)
        return lower, (gamma-5*lower)/5

    def ce_variance(self, gamma):
        lower, w = self._weights(gamma)
        w = w[:, None, None]
        return (1-w)*self.ce_errors[lower]+w*self.ce_errors[lower+1]

    def nominal_power(self, gamma):
        lower, w = self._weights(gamma)
        values = self.power_ema.detach()
        normalized = ED*values/values.sum(-1, keepdim=True)
        normalized = (1-self.SHRINK)*normalized+self.SHRINK*(ED/values.shape[-1])
        w = w[:, None, None]
        return (1-w)*normalized[lower]+w*normalized[lower+1]

    @torch.no_grad()
    def commit_power(self, node_power, node_frames):
        if not self.training:
            raise RuntimeError('Historical power is frozen outside training')
        for node in range(6):
            frames = int(node_frames[node])
            if frames:
                self.power_ema[node].mul_(self.DECAY).add_(node_power[node]/frames, alpha=1-self.DECAY)
                self.node_updates[node] += 1


def node_index(gamma):
    nodes = torch.as_tensor(GAMMAS, dtype=torch.float32, device=gamma.device)
    hit = gamma[:, None] == nodes[None]
    if not bool(hit.any(-1).all()):
        raise ValueError('Training gamma must be one of the six nodes')
    return hit.float().argmax(-1)


# -------------------------------------------------------------------- models --

class _Protocol:
    PILOT_POSITIONS_BY_USER = PILOT_POSITIONS_BY_USER


def _ce_features(self, Y, pilots, H_initial, *, gamma_db):
    """K16 copy of the K8 CE features: same 133 channels, pilot amplitude sqrt(16/4)=2."""
    batch = Y.shape[0]
    if Y.dtype != torch.complex64 or tuple(Y.shape) != (batch, NT, NSC):
        raise ValueError('CE requires complex64 Y[B,32,256]')
    if pilots.shape != (batch, K, NSC) or H_initial.shape != (batch, K, NT, NSC):
        raise ValueError('CE K16 pilot/initial shapes')
    own, union = self.pilot_mask, self.pilot_union
    pilot_Y = torch.where(union[None, None], Y, torch.zeros((), dtype=Y.dtype, device=Y.device))
    amplitudes = pilots[:, own].abs()
    if bool((pilots[:, ~own] != 0).any()) or not bool(torch.isclose(amplitudes,
            amplitudes.new_full(amplitudes.shape, math.sqrt(EP/PILOTS_PER_UE)), rtol=1e-6, atol=1e-6).all()):
        raise ValueError('Only paid K16 pilots are allowed')
    gamma = torch.as_tensor(gamma_db, dtype=torch.float32, device=Y.device)
    if gamma.ndim == 0:
        gamma = gamma.expand(batch)
    snr = (gamma-12.5)/12.5
    common = torch.cat((pilot_Y.real, pilot_Y.imag), 1)[:, None].expand(-1, K, -1, -1)
    initial = torch.cat((H_initial.real, H_initial.imag), 2)
    return torch.cat((common, initial, pilots.real[:, :, None], pilots.imag[:, :, None],
        own[None, :, None].expand(batch, -1, -1, -1), union[None, None, None].expand(batch, K, -1, -1),
        snr[:, None, None, None].expand(batch, K, 1, NSC)), 2).reshape(batch*K, 133, NSC)


def models():
    """Isolated K16 instance of the K8 model module (K=16, Dmax=24)."""
    name = '_twc_v1_k16_models'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT/'models'/'twc_v1_k8_icc_aligned.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.K, module.DMAX = K, DMAX
    module.ARCHITECTURE = 'twc_v1_k16_icc_aligned_v1'
    module.ARMS = ('fdma', 'hybrid')
    module._protocol = lambda: _Protocol
    module.public_layout = make_layout
    module.support_geometry = support_geometry
    module.K8ICCAlignedPilotUNet.features = _ce_features
    module.K8ICCAlignedPilotUNet._features = _ce_features
    return module


def gamma_feature(gamma):
    return ((gamma.to(torch.float32)-12.5)/12.5)[:, None]


class GammaAwareEncoder(nn.Module):
    """K16 copy of the final model's gamma-conditioned UE encoder (zero-init FiLM + slot gain)."""
    def __init__(self, source_encoder, hidden=32):
        super().__init__()
        self.base = source_encoder
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 2*WIDTH+DMAX))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, H_dl, gamma):
        from models.baseline_common import angular_delay_transform
        m = models()
        batch = H_dl.shape[0]
        gamma = torch.as_tensor(gamma, dtype=torch.float32, device=H_dl.device)
        if gamma.ndim == 0:
            gamma = gamma.expand(batch)
        scale, shift, log_gain = self.net(gamma_feature(gamma)).split((WIDTH, WIDTH, DMAX), -1)
        b = self.base
        features = angular_delay_transform(H_dl, NSC).reshape(batch*K, 2, NSC, NT)
        tokens, height, width = b.patch_embed(features)
        for block in b.blocks:
            tokens = block(tokens, height, width)
        tokens = b.norm(tokens)
        tokens = tokens*(1+scale.repeat_interleave(K, 0)[:, None])+shift.repeat_interleave(K, 0)[:, None]
        output = b.fc(m.ordered_window_pool(tokens).flatten(1)).reshape(batch, K, 2*DMAX)
        raw = torch.complex(output[..., :DMAX], output[..., DMAX:])
        return raw*torch.exp(log_gain.clamp(-3., 3.))[:, None, :].to(raw.real.dtype)


class Codec(nn.Module):
    def __init__(self, arm, seed, activation_checkpointing=True):
        super().__init__()
        m = models()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            source = m.K8ICCAlignedCodec(arm, activation_checkpointing=activation_checkpointing)
        self.encoder = GammaAwareEncoder(source.encoder)
        self.decoder = source.decoder


def new_ce():
    return models().K8ICCAlignedPilotUNet()


# ---------------------------------------------------------------- objective --

def design_rzf(H_hat, alpha):
    channels = H_hat.permute(0, 3, 1, 2)
    gram = channels@channels.conj().transpose(-2, -1)
    eye = torch.eye(channels.shape[-2], dtype=channels.dtype, device=channels.device)
    precoder = torch.linalg.solve(gram+float(alpha)*eye, channels).conj().transpose(-2, -1)
    power = precoder.abs().square().sum(dim=(-2, -1), keepdim=True)
    return precoder/power.sqrt()


def rates(H_hat, H_dl, alpha=.03, downlink_snr_db=25.):
    W = design_rzf(H_hat, alpha)
    effective = H_dl.permute(0, 3, 1, 2)@W
    powers = effective.abs().square()
    desired = powers.diagonal(dim1=-2, dim2=-1)
    interference = (powers.sum(-1)-desired).clamp_min(0.)
    noise = 10.**(-downlink_snr_db/10.)
    link = torch.log2(1.+desired/(interference+noise))
    return dict(sum_rate_bps_hz=link.sum(-1).mean(-1), user_rate_bps_hz=link.mean(1),
        desired=desired, interference=interference, noise=noise,
        desired_signal_power_mean=desired.mean((1, 2)), interference_power_mean=interference.mean((1, 2)))


def direction_error(H_hat, H_dl):
    from twc_v1_k8_hybrid_direction import direction_error as base
    return base(H_hat, H_dl)


def objective(H_hat, H_dl, gamma, direction_weight=100., interference_weight=1.):
    """Same objective as K8: -rate + 100*direction error + 1*(gamma<=10) DL leakage."""
    r = rates(H_hat, H_dl)
    direction = direction_error(H_hat, H_dl)
    leakage = (torch.log1p(r['interference']/(r['desired']+r['noise']))/math.log(2.)).sum(-1).mean(-1)
    low = (torch.as_tensor(gamma, device=H_hat.device) <= 10).to(leakage.dtype)*leakage
    total = -r['sum_rate_bps_hz'].mean()+direction_weight*direction.mean()+interference_weight*low.mean()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite K16 objective')
    return total, dict(sum_rate_bps_hz=r['sum_rate_bps_hz'], direction_error=direction)


def nmse_loss(H_hat, H_dl):
    err = (H_hat-H_dl).abs().square().sum((2, 3))/H_dl.abs().square().sum((2, 3))
    return err.mean(), dict(dl_nmse_linear=err.mean(1).detach())


# --------------------------------------------------------------- pairing ----

def coupling(H_ul_est):
    channel = H_ul_est.permute(0, 3, 1, 2)
    gram = channel@channel.conj().transpose(-1, -2)
    norm = channel.abs().square().sum(-1).clamp_min(1e-12)
    return (gram.abs().square()/(norm[..., :, None]*norm[..., None, :])).mean(1)


def greedy_pairing(H_ul_est):
    """Permutation [B,16] placing greedily chosen least-coupled pairs on slots (0,1),(2,3),..."""
    c = coupling(H_ul_est).detach().cpu()
    out = []
    for b in range(c.shape[0]):
        cost = c[b].clone()
        cost.fill_diagonal_(float('inf'))
        free, perm = set(range(K)), []
        while free:
            idx = sorted(free)
            sub = cost[idx][:, idx]
            flat = int(torch.argmin(sub))
            i, j = idx[flat//len(idx)], idx[flat % len(idx)]
            perm += [i, j]; free -= {i, j}
        out.append(perm)
    return torch.tensor(out, device=H_ul_est.device)


def slot_pair_coupling(H_ul_est):
    c = coupling(H_ul_est)
    return torch.stack([c[:, 2*g, 2*g+1] for g in range(K//2)], -1).mean(-1)


def permute_users(value, perm):
    index = perm.reshape(perm.shape+(1,)*(value.ndim-2)).expand_as(value)
    return value.gather(1, index)


# ------------------------------------------------------------------- data ----

class K16Data:
    """Each K16 sample concatenates two independent K8 draws of the same role."""
    def __init__(self, namespace, seed, batch, device):
        source_path = ROOT/'utils'/'twc_v1_k8_replayable_data.py'
        name = '_twc_v1_k16_data_source'
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self._base = sys.modules[name].TWCv1K8ReplayableData(base_seed=seed,
            batch_sizes=dict(ce_train=batch, codec_train=batch, ce_calibration=batch, validation=batch, test=1),
            device=str(device), cdl_model='C', speed=1., seed_namespace=namespace)

    def sample(self, role, index):
        if role in ('ce_train', 'codec_train'):
            a, b = self._base.sample(role, 2*index-1), self._base.sample(role, 2*index)
        else:
            a, b = self._base.sample(role, 2*index), self._base.sample(role, 2*index+1)
        return dict(H_dl=torch.cat((a['H_dl'], b['H_dl']), 1), H_ul=torch.cat((a['H_ul'], b['H_ul']), 1))


def random_streams(namespace, seed, role, index, batch, device):
    def s(kind):
        raw = f'{namespace}:{seed}:{role}:{index}:{kind}'.encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)
    gen = torch.Generator(device=device).manual_seed(s('noise'))
    noise = torch.view_as_complex(torch.randn(batch, NT, NSC, 2, generator=gen, device=device).contiguous())/2**.5
    gen = torch.Generator(device=device).manual_seed(s('sounding'))
    sounding = torch.view_as_complex(torch.randn(batch, NT, NSC, 2, generator=gen, device=device).contiguous())/2**.5
    gen = torch.Generator(device=device).manual_seed(s('polarity'))
    polarity = (2*torch.randint(2, (batch, K), generator=gen, device=device)-1).float()
    return noise, sounding, polarity


# ---------------------------------------------------------------- forward ----

def ce_forward(ce, sample, gamma, noise, polarity):
    pilots = known_pilots(sample['H_ul'].shape[0], sample['H_ul'].device, polarity)
    Y = transmit(sample['H_ul'], pilots, noise, gamma)
    return ce(Y, pilots, fourier_ls(Y, pilots), gamma_db=gamma), Y, pilots


def forward(system, sample, gamma, noise, polarity, *, pairing=False, sounding=None, genie_ul=False):
    codec, ce, stats, layout = system['codec'], system['ce'], system['stats'], system['layout']
    H_dl, H_ul = sample['H_dl'], sample['H_ul']
    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=H_ul.device)
    if gamma.ndim == 0:
        gamma = gamma.expand(H_ul.shape[0])
    if pairing:
        with torch.no_grad():
            est0, _, _ = ce_forward(ce, sample, gamma, noise if sounding is None else sounding, polarity)
            perm = greedy_pairing(est0)
        H_dl, H_ul = permute_users(H_dl, perm), permute_users(H_ul, perm)
    raw = codec.encoder(H_dl, gamma)
    wave = build_waveform(raw, layout, polarity)
    Y = transmit(H_ul, wave.x_grid, noise, gamma)
    with torch.no_grad():
        estimated = H_ul if genie_ul else ce(Y, wave.pilot, fourier_ls(Y, wave.pilot), gamma_db=gamma)
        cevar = stats.ce_variance(gamma)
        if genie_ul:
            cevar = torch.zeros_like(cevar)
        prior = stats.nominal_power(gamma)
    recovered = recover(Y, estimated, layout, prior, cevar, gamma)
    out = codec.decoder(Y_rx=recovered['Y_res'], H_ul_est=estimated, soft=recovered['soft'],
        reliability=recovered['reliability'], pilots=wave.pilot, gamma_db=gamma, layout=None)
    return dict(H_hat=out['H_hat'], H_dl=H_dl, wave=wave, estimated=estimated,
        pair_coupling=slot_pair_coupling(estimated))


def sample_metrics(result):
    H_hat, H_dl = result['H_hat'], result['H_dl']
    r = rates(H_hat, H_dl)
    nmse = ((H_hat-H_dl).abs().square().sum((2, 3))/H_dl.abs().square().sum((2, 3))).mean(1)
    wave = result['wave']
    metrics = dict(sum_rate_bps_hz=r['sum_rate_bps_hz'], dl_nmse_linear=nmse,
        direction_error=direction_error(H_hat, H_dl).to(nmse.dtype),
        interference_power_mean=r['interference_power_mean'], desired_signal_power_mean=r['desired_signal_power_mean'],
        min_user_rate_bps_hz=r['user_rate_bps_hz'].min(-1).values,
        key_energy_fraction=wave.key.abs().square().sum(-1).mean(-1)/wave.active.abs().square().sum(-1).mean(-1),
        pair_coupling=result['pair_coupling'])
    return {k: v.detach().float().cpu() for k, v in metrics.items()}


def upper_bound_metrics(H_dl):
    r = rates(H_dl, H_dl)
    return {k: v.detach().float().cpu() for k, v in dict(sum_rate_bps_hz=r['sum_rate_bps_hz'],
        min_user_rate_bps_hz=r['user_rate_bps_hz'].min(-1).values,
        interference_power_mean=r['interference_power_mean'],
        desired_signal_power_mean=r['desired_signal_power_mean']).items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
