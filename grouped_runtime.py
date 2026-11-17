"""Generic K8 hybrid layouts with groups of m=2, 4 or 8 users.

Reuses the original MAC simulation, pilots, Fourier-LS and CE, and the
final-model gamma-aware encoder. New here: the waveform mapper, the
error-aware LMMSE receiver and the decoder geometry for any group size, and a
SHA-pinned loader for the shared CE files that does not depend on the
repository path.
"""
from __future__ import annotations

import json

import torch
from torch import nn

from configs import twc_v1_k8_grouped_config as cfg
import final_runtime as fr

K, NT, NSC, DMAX, ED = 8, 32, 256, 216, 48.


def protocol():
    return fr.physics()._protocol()


def make_layout(arm):
    if arm == 'fdma':
        return protocol().FDMALayout()
    if arm == 'hybrid_g4':
        return protocol().default_hybrid_layout(4)
    return protocol().default_hybrid_layout(cfg.ARMS[arm])


def active_dimension(layout):
    if hasattr(layout, 'data_positions'):
        return 24
    return 4+layout.shared_dimension


def group_of(layout):
    owner = [None]*K
    for index, members in enumerate(layout.groups):
        for user in members:
            owner[user] = index
    return owner


def build_waveform(raw, layout, polarity=None):
    phy = fr.physics()
    batch = raw.shape[0]
    n = active_dimension(layout)
    selected = raw[..., :n]
    norm = selected.abs().square().sum(-1, keepdim=True)
    if not bool((norm > 1e-12).all()):
        raise FloatingPointError('Near-zero selected codeword')
    active = selected*(ED/norm).sqrt()
    data = raw.new_zeros((batch, K, NSC))
    if n == 24:
        index = torch.tensor(layout.data_positions, device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active)
        key, shared = active, active[..., :0]
    else:
        index = torch.tensor(layout.key_positions, device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., :4])
        owner = group_of(layout)
        index = torch.tensor([layout.shared_positions_by_group[owner[u]] for u in range(K)], device=raw.device)
        data = data.scatter(-1, index[None].expand(batch, -1, -1), active[..., 4:])
        key, shared = active[..., :4], active[..., 4:]
    pilots = phy.known_pilots(batch, raw.device, polarity)
    full = torch.cat((active, raw.new_zeros((batch, K, DMAX-n))), -1)
    return phy.Waveform(raw, full, active, key, shared, pilots, data, pilots+data)


def recover(Y, H_ul_est, layout, prior, ce_variance, gamma_db):
    """Error-aware LMMSE: per-user on key REs, joint m-user per shared block."""
    phy = fr.physics()
    batch = Y.shape[0]
    n = active_dimension(layout)
    prior = torch.as_tensor(prior, dtype=torch.float32, device=Y.device).detach()
    if prior.shape == (K, n):
        prior = prior[None].expand(batch, -1, -1)
    ce_variance = torch.as_tensor(ce_variance, dtype=torch.float32, device=Y.device).detach()
    if prior.shape != (batch, K, n) or ce_variance.shape != (batch, K, NSC) or not bool((prior > 0).all()):
        raise ValueError('Positive per-MU prior and per-RE CE variance required')
    n0 = phy.noise_variance(gamma_db, batch, Y.device)
    exclusive = layout.data_positions if n == 24 else layout.key_positions
    index = torch.tensor(exclusive, device=Y.device)
    d = index.shape[-1]
    h = H_ul_est.gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
    obs = Y[:, None].expand(-1, K, -1, -1).gather(-1, index[None, :, None].expand(batch, -1, NT, -1))
    channel_power = h.abs().square().sum(2)
    pi = prior[:, :, :d]
    error = ce_variance.gather(-1, index[None].expand(batch, -1, -1))
    equiv = n0[:, None, None]+pi*error
    ridge = phy.RX_RIDGE+equiv/pi
    weights = h.conj()/(channel_power+ridge)[:, :, None]
    z_key = (weights*obs).sum(2)
    v_key = (channel_power/(channel_power+ridge)-1).square()*pi+equiv*weights.abs().square().sum(2)
    if n == 24:
        active, variance = z_key, v_key
    else:
        z_shared = torch.zeros((batch, K, n-4), dtype=Y.dtype, device=Y.device)
        v_shared = torch.zeros((batch, K, n-4), dtype=torch.float32, device=Y.device)
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            m = len(members)
            users = torch.tensor(members, device=Y.device)
            tones = torch.tensor(positions, device=Y.device)
            g = H_ul_est.index_select(1, users).index_select(-1, tones).permute(0, 3, 2, 1)  # [B,T,NT,m]
            pi_group = prior.index_select(1, users)[:, :, 4:].transpose(1, 2)               # [B,T,m]
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
    if n == 24:
        for user, positions in enumerate(layout.data_positions):
            support[user, :24] = torch.as_tensor(positions); exclusive[user, :24] = True
    else:
        for user, positions in enumerate(layout.key_positions):
            support[user, :4] = torch.as_tensor(positions); exclusive[user, :4] = True
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            for user in members:
                support[user, 4:n] = torch.as_tensor(positions)
    return support, support >= 0, exclusive


def set_geometry(decoder, layout):
    for name, value in zip(('support', 'active', 'exclusive'), support_geometry(layout)):
        setattr(decoder, '_fixed_'+name, value.to(decoder.fc.weight.device))


class NodeStatistics(fr.NodeStatistics):
    """Per-gamma-node power prior for any active dimension."""
    def __init__(self, layout, ce_errors, power_init):
        nn.Module.__init__(self)
        self.layout = layout
        n = active_dimension(layout)
        errors = torch.as_tensor(ce_errors, dtype=torch.float32).detach().cpu().clone()
        power = torch.as_tensor(power_init, dtype=torch.float32).detach().cpu().clone()
        if errors.shape != (6, K, NSC) or power.shape != (K, n) or not bool((power > 0).all()):
            raise ValueError('CE errors [6,8,256] and positive power [8,n] required')
        self.register_buffer('ce_errors', errors)
        self.register_buffer('power_ema', power[None].repeat(6, 1, 1))
        self.register_buffer('node_updates', torch.zeros(6, dtype=torch.int64))


def _sha(path):
    return fr.sha256(path)


def load_shared_ce(device, directory=None):
    """CE nmse65000 and its locked table, verified by file SHA256, not by path."""
    from models.twc_v1_k8_icc_aligned import K8ICCAlignedPilotUNet
    directory = directory or cfg.SHARED_CE_DIR
    for name, expected in cfg.SHARED_CE_SHA256.items():
        if _sha(directory/name) != expected:
            raise ValueError('Shared CE file differs from the pinned bytes: '+name)
    record = json.loads((directory/'manifest.json').read_text())
    if record.get('status') != 'completed':
        raise ValueError('Shared CE must be a completed terminal')
    payload = torch.load(directory/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    model = K8ICCAlignedPilotUNet()
    model.load_state_dict(payload['model'], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    table = torch.tensor(json.loads((directory/'calibration.json').read_text())['per_RE_error']['nmse65000'],
        dtype=torch.float32)
    return model, table.to(device), dict(sha256=cfg.SHARED_CE_SHA256, step=record.get('step'))


def make_train_data(device):
    source = fr.base().original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4, validation=4, test=1),
        device=str(device), cdl_model='C', speed=1., seed_namespace=cfg.NAMESPACE)


def train_random(arm, index, device):
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(arm, index, 'noise'))
    noise = torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, NT, NSC, 2, generator=gen, device=device).contiguous())/2**.5
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(arm, index, 'polarity'))
    polarity = (2*torch.randint(2, (cfg.MICRO_BATCH, K), generator=gen, device=device)-1).float()
    return noise, polarity


def forward(system, sample, gamma, noise, polarity):
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
