"""K8 Hybrid with a variable number q of dedicated key REs per UE (key-count ablation).

Generalizes grouped_runtime (which fixes q = 4) without changing it. Pilots, the
per-UE energy (16 pilot + 48 data), the MAC, CE, gamma-aware encoder, decoder and
error-aware LMMSE receiver are unchanged; only the split of the 192 data REs
moves. Key REs: the first 8q non-pilot REs, q per UE in user order (as in the
original protocol). Shared REs: the remaining 192 - 8q REs in equal contiguous
blocks, one per group. q = 4 with pairs reproduces the default paired layout and
grouped_runtime exactly (CPU-tested); q = 0 is pure pairwise sharing (48 shared
REs per UE), q = 8 gives 8 key + 32 shared REs per UE.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import final_runtime as fr
import grouped_runtime as gr

K, NT, NSC, DMAX, ED = gr.K, gr.NT, gr.NSC, gr.DMAX, gr.ED
DATA_RE = 192
SUPPORTED_KEYS = (0, 2, 4, 8)


def _nonpilot_positions():
    protocol = gr.protocol()
    pilots = {position for row in protocol.PILOT_POSITIONS_BY_USER for position in row}
    return tuple(position for position in range(NSC) if position not in pilots)


@dataclass(frozen=True)
class KeyLayout:
    """Hybrid layout with q key REs per UE and group-shared blocks (fixed consecutive groups)."""
    keys: int
    group_count: int = 4

    def __post_init__(self):
        if type(self.keys) is not int or self.keys not in SUPPORTED_KEYS:
            raise ValueError(f'keys per UE must be one of {SUPPORTED_KEYS}')
        if self.group_count not in (1, 2, 4) or (DATA_RE-K*self.keys) % self.group_count:
            raise ValueError('group_count must divide the shared REs into equal blocks')

    @property
    def groups(self):
        size = K//self.group_count
        return tuple(tuple(range(start, start+size)) for start in range(0, K, size))

    @property
    def users_per_group(self):
        return K//self.group_count

    @property
    def shared_dimension(self):
        return (DATA_RE-K*self.keys)//self.group_count

    @property
    def pilot_positions(self):
        return gr.protocol().PILOT_POSITIONS_BY_USER

    @property
    def key_positions(self):
        nonpilot = _nonpilot_positions()
        return tuple(tuple(nonpilot[u*self.keys:(u+1)*self.keys]) for u in range(K))

    @property
    def shared_positions_by_group(self):
        rest, d = _nonpilot_positions()[K*self.keys:], self.shared_dimension
        return tuple(tuple(rest[g*d:(g+1)*d]) for g in range(self.group_count))

    def to_dict(self):
        return dict(layout='hybrid_keyed', keys_per_ue=self.keys, group_count=self.group_count,
                    users_per_group=self.users_per_group, shared_per_group=self.shared_dimension,
                    data_dimensions_per_ue=self.keys+self.shared_dimension, groups=[list(g) for g in self.groups])


def key_count(layout):
    if hasattr(layout, 'data_positions'):
        return None
    return len(layout.key_positions[0])


def active_dimension(layout):
    if hasattr(layout, 'data_positions'):
        return 24
    return key_count(layout)+layout.shared_dimension


def _key_index(layout, device):
    q = key_count(layout)
    return torch.tensor(layout.key_positions, dtype=torch.long, device=device).reshape(K, q)


def build_waveform(raw, layout, polarity=None):
    if hasattr(layout, 'data_positions'):
        return gr.build_waveform(raw, layout, polarity)
    phy = fr.physics()
    batch, q, n = raw.shape[0], key_count(layout), active_dimension(layout)
    selected = raw[..., :n]
    norm = selected.abs().square().sum(-1, keepdim=True)
    if not bool((norm > 1e-12).all()):
        raise FloatingPointError('Near-zero selected codeword')
    active = selected*(ED/norm).sqrt()
    data = raw.new_zeros((batch, K, NSC))
    if q:
        index = _key_index(layout, raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., :q])
    owner = gr.group_of(layout)
    index = torch.tensor([layout.shared_positions_by_group[owner[u]] for u in range(K)], device=raw.device)
    data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., q:])
    pilots = phy.known_pilots(batch, raw.device, polarity)
    full = torch.cat((active, raw.new_zeros((batch, K, DMAX-n))), -1)
    return phy.Waveform(raw, full, active, active[..., :q], active[..., q:], pilots, data, pilots+data)


def recover(Y, H_ul_est, layout, prior, ce_variance, gamma_db):
    """Error-aware LMMSE: per-user on the q key REs, joint per shared block (as grouped_runtime)."""
    if hasattr(layout, 'data_positions'):
        return gr.recover(Y, H_ul_est, layout, prior, ce_variance, gamma_db)
    phy = fr.physics()
    batch, q, n = Y.shape[0], key_count(layout), active_dimension(layout)
    prior = torch.as_tensor(prior, dtype=torch.float32, device=Y.device).detach()
    if prior.shape == (K, n):
        prior = prior[None].expand(batch, -1, -1)
    ce_variance = torch.as_tensor(ce_variance, dtype=torch.float32, device=Y.device).detach()
    if prior.shape != (batch, K, n) or ce_variance.shape != (batch, K, NSC) or not bool((prior > 0).all()):
        raise ValueError('Positive per-MU prior and per-RE CE variance required')
    n0 = phy.noise_variance(gamma_db, batch, Y.device)
    parts_z, parts_v = [], []
    if q:
        index = _key_index(layout, Y.device)
        h = H_ul_est.gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
        obs = Y[:, None].expand(-1, K, -1, -1).gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
        channel_power = h.abs().square().sum(2)
        pi = prior[:, :, :q]
        error = ce_variance.gather(-1, index[None].expand(batch, -1, -1))
        equiv = n0[:, None, None]+pi*error
        ridge = phy.RX_RIDGE+equiv/pi
        weights = h.conj()/(channel_power+ridge)[:, :, None]
        parts_z.append((weights*obs).sum(2))
        parts_v.append((channel_power/(channel_power+ridge)-1).square()*pi+equiv*weights.abs().square().sum(2))
    z_shared = torch.zeros((batch, K, n-q), dtype=Y.dtype, device=Y.device)
    v_shared = torch.zeros((batch, K, n-q), dtype=torch.float32, device=Y.device)
    for members, positions in zip(layout.groups, layout.shared_positions_by_group):
        m = len(members)
        users = torch.tensor(members, device=Y.device)
        tones = torch.tensor(positions, device=Y.device)
        g = H_ul_est.index_select(1, users).index_select(-1, tones).permute(0, 3, 2, 1)  # [B,T,NT,m]
        pi_group = prior.index_select(1, users)[:, :, q:].transpose(1, 2)               # [B,T,m]
        vi = ce_variance.index_select(1, users).index_select(-1, tones).transpose(1, 2)  # [B,T,m]
        effective = n0[:, None]+(pi_group*vi).sum(-1)
        ridge_group = phy.RX_RIDGE+effective[..., None]/pi_group
        gh = g.conj().transpose(-2, -1)
        w = torch.linalg.solve(gh@g+torch.diag_embed(ridge_group).to(g.dtype), gh)
        group_y = Y.index_select(-1, tones).transpose(-2, -1)
        z = (w@group_y[..., None]).squeeze(-1).transpose(1, 2)
        deviation = w@g-torch.eye(m, dtype=g.dtype, device=Y.device)
        v = ((deviation.abs().square()*pi_group[:, :, None, :]).sum(-1)
             +effective[..., None]*w.abs().square().sum(-1)).transpose(1, 2)
        z_shared = z_shared.index_copy(1, users, z)
        v_shared = v_shared.index_copy(1, users, v)
    active = torch.cat(parts_z+[z_shared], -1)
    variance = torch.cat(parts_v+[v_shared], -1)
    if not bool(torch.isfinite(active).all() & torch.isfinite(variance).all() & (variance > 0).all()):
        raise FloatingPointError('Nonfinite statistical recovery')
    padding = Y.new_zeros((batch, K, DMAX-n))
    return dict(soft=torch.cat((active, padding), -1),
        reliability=torch.cat((variance.reciprocal(), padding.real), -1), Y_res=Y)


def support_geometry(layout):
    if hasattr(layout, 'data_positions'):
        return gr.support_geometry(layout)
    q, n = key_count(layout), active_dimension(layout)
    support = torch.full((K, DMAX), -1, dtype=torch.long)
    exclusive = torch.zeros_like(support, dtype=torch.bool)
    for user, positions in enumerate(layout.key_positions):
        if q:
            support[user, :q] = torch.as_tensor(positions); exclusive[user, :q] = True
    for members, positions in zip(layout.groups, layout.shared_positions_by_group):
        for user in members:
            support[user, q:n] = torch.as_tensor(positions)
    return support, support >= 0, exclusive


def set_geometry(decoder, layout):
    for name, value in zip(('support', 'active', 'exclusive'), support_geometry(layout)):
        setattr(decoder, '_fixed_'+name, value.to(decoder.fc.weight.device))


class NodeStatistics(fr.NodeStatistics):
    """Per-gamma-node power prior for any active dimension (as grouped_runtime.NodeStatistics)."""
    def __init__(self, layout, ce_errors, power_init):
        torch.nn.Module.__init__(self)
        self.layout = layout
        n = active_dimension(layout)
        errors = torch.as_tensor(ce_errors, dtype=torch.float32).detach().cpu().clone()
        power = torch.as_tensor(power_init, dtype=torch.float32).detach().cpu().clone()
        if errors.shape != (6, K, NSC) or power.shape != (K, n) or not bool((power > 0).all()):
            raise ValueError('CE errors [6,8,256] and positive power [8,n] required')
        self.register_buffer('ce_errors', errors)
        self.register_buffer('power_ema', power[None].repeat(6, 1, 1))
        self.register_buffer('node_updates', torch.zeros(6, dtype=torch.int64))


def forward(system, sample, gamma, noise, polarity):
    """grouped_runtime.forward with the q-general waveform and receiver."""
    phy = fr.physics()
    codec, ce, stats, layout = system['codec'], system['ce'], system['stats'], system['layout']
    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=sample['H_ul'].device)
    if gamma.ndim == 0:
        gamma = gamma.expand(sample['H_ul'].shape[0])
    raw = codec.encoder(sample['H_dl'], gamma)
    wave = build_waveform(raw, layout, polarity)
    Y = phy.transmit(sample['H_ul'], wave.x_grid, noise, gamma)
    with torch.no_grad():
        estimated = ce(Y, wave.pilot, phy.fourier_ls(Y, wave.pilot), gamma_db=gamma)
        prior = stats.nominal_power(gamma)
        cevar = stats.ce_variance(gamma)
    recovered = recover(Y, estimated, layout, prior, cevar, gamma)
    out = codec.decoder(Y_rx=recovered['Y_res'], H_ul_est=estimated, soft=recovered['soft'],
        reliability=recovered['reliability'], pilots=wave.pilot, gamma_db=gamma, layout=None)
    return dict(H_hat=out['H_hat'], H_dl=sample['H_dl'], wave=wave, estimated=estimated,
        pair_coupling=fr.slot_pair_coupling(estimated))
