"""Runtime for the final Hybrid-G4 model: gamma-aware UE, per-node prior, pairing.

Everything that is inherited (MAC simulation, Fourier-LS, CE U-Net, receiver
filter, Joint decoder, RZF, rate scoring) is the unchanged original code. New
pieces live here and are additive:

* ``GammaAwareEncoder`` wraps the source encoder. A small MLP of the public
  feedback gamma produces FiLM parameters for the final Swin tokens and a
  positive gain per transmit slot (applied before the unchanged joint energy
  normalization, so it reallocates key/shared energy). Its last layer is zero,
  so the wrapped encoder initially reproduces the source encoder exactly.
* ``NodeStatistics`` keeps the causal historical power prior separately for each
  of the six gamma nodes (linear interpolation in between), with the locked CE
  error table of the chosen CE.
* ``recover_batched`` is the original receiver with a per-MU prior.
* ``best_pairing`` relabels users so that the four shared pairs have the lowest
  estimated-UL spatial coupling. It uses only the pilot-only CE estimate.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
from pathlib import Path

import torch
from torch import nn

from configs import twc_v1_k8_final_config as cfg

K, NT, NSC, DMAX, WIDTH = 8, 32, 256, 216, 128
ED = 48.


@functools.lru_cache(maxsize=None)
def base():
    import low_snr_runtime
    return low_snr_runtime


def physics():
    return base().physics()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    import subprocess
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cfg.ROOT, capture_output=True, text=True).stdout.strip()
    return dict(commit=commit, sha256={name: sha256(cfg.ROOT/name) for name in cfg.CORE_FILES})


# ---------------------------------------------------------------- UE side ----

def gamma_feature(gamma):
    return ((gamma.to(torch.float32)-12.5)/12.5)[:, None]


class GammaConditioner(nn.Module):
    def __init__(self, hidden=cfg.COND_HIDDEN, width=WIDTH, slots=DMAX):
        super().__init__()
        self.width, self.slots = width, slots
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 2*width+slots))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, gamma):
        out = self.net(gamma_feature(gamma))
        scale, shift, log_gain = out.split((self.width, self.width, self.slots), -1)
        return scale, shift, log_gain.clamp(-3., 3.)


class GammaAwareEncoder(nn.Module):
    def __init__(self, source_encoder):
        super().__init__()
        self.base = source_encoder
        self.condition = GammaConditioner()

    def forward(self, H_dl, gamma):
        from models.baseline_common import angular_delay_transform
        from models.twc_v1_k8_icc_aligned import ordered_window_pool, _complex
        batch = H_dl.shape[0]
        _complex(H_dl, (batch, K, NT, NSC), 'UE source')
        gamma = torch.as_tensor(gamma, dtype=torch.float32, device=H_dl.device)
        if gamma.ndim == 0:
            gamma = gamma.expand(batch)
        if gamma.shape != (batch,) or gamma.requires_grad:
            raise ValueError('One public nonlearned gamma per MU required')
        scale, shift, log_gain = self.condition(gamma)
        b = self.base
        features = angular_delay_transform(H_dl, NSC).reshape(batch*K, 2, NSC, NT)
        tokens, height, width = b.patch_embed(features)
        for block in b.blocks:
            tokens = block(tokens, height, width)
        tokens = b.norm(tokens)
        tokens = tokens*(1+scale.repeat_interleave(K, 0)[:, None])+shift.repeat_interleave(K, 0)[:, None]
        output = b.fc(ordered_window_pool(tokens).flatten(1)).reshape(batch, K, 2*DMAX)
        raw = torch.complex(output[..., :DMAX], output[..., DMAX:])
        return raw*torch.exp(log_gain)[:, None, :].to(raw.real.dtype)


class FinalCodec(nn.Module):
    def __init__(self, source_codec):
        super().__init__()
        self.encoder = GammaAwareEncoder(source_codec.encoder)
        self.decoder = source_codec.decoder

    def new_parameters(self):
        return list(self.encoder.condition.parameters())

    def old_parameters(self):
        new = set(map(id, self.new_parameters()))
        return [p for p in self.parameters() if id(p) not in new]


# --------------------------------------------------------------- receiver ----

class NodeStatistics(nn.Module):
    """Per-gamma-node causal power prior plus a locked per-RE CE error table."""
    DECAY, SHRINK = .99, .05

    def __init__(self, layout, ce_errors, power_init):
        super().__init__()
        self.layout = layout
        n = physics().active_dimension(layout)
        errors = torch.as_tensor(ce_errors, dtype=torch.float32).detach().cpu().clone()
        power = torch.as_tensor(power_init, dtype=torch.float32).detach().cpu().clone()
        if errors.shape != (6, K, NSC) or not bool(torch.isfinite(errors).all() & (errors >= 0).all()):
            raise ValueError('Locked CE errors [6,8,256] required')
        if power.shape != (K, n) or not bool((power > 0).all()):
            raise ValueError('Positive source power prior [8,n] required')
        self.register_buffer('ce_errors', errors)
        self.register_buffer('power_ema', power[None].repeat(6, 1, 1))
        self.register_buffer('node_updates', torch.zeros(6, dtype=torch.int64))

    def _weights(self, gamma):
        gamma = torch.as_tensor(gamma, dtype=torch.float32, device=self.power_ema.device).reshape(-1)
        if not bool(torch.isfinite(gamma).all() & (gamma >= 0).all() & (gamma <= 25).all()):
            raise ValueError('Public gamma in [0,25] required')
        lower = torch.div(gamma, 5, rounding_mode='floor').long().clamp(max=4)
        weight = (gamma-5*lower)/5
        return lower, weight

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
        node_frames = torch.as_tensor(node_frames, device=self.power_ema.device)
        for node in range(6):
            frames = int(node_frames[node])
            if frames:
                self.power_ema[node].mul_(self.DECAY).add_(node_power[node]/frames, alpha=1-self.DECAY)
                self.node_updates[node] += 1


def node_index(gamma):
    nodes = torch.as_tensor(cfg.GAMMAS, dtype=torch.float32, device=gamma.device)
    hit = (gamma[:, None] == nodes[None])
    if not bool(hit.any(-1).all()):
        raise ValueError('Training gamma must be one of the six nodes')
    return hit.float().argmax(-1)


def recover_batched(Y, H_ul_est, layout, prior, ce_variance, gamma_db):
    """Original statistical receiver, but with one historical prior per MU [B,8,n]."""
    phy = physics()
    batch = Y.shape[0]
    n = phy.active_dimension(layout)
    prior = torch.as_tensor(prior, dtype=torch.float32, device=Y.device).detach()
    ce_variance = torch.as_tensor(ce_variance, dtype=torch.float32, device=Y.device).detach()
    if prior.shape == (K, n):
        prior = prior[None].expand(batch, -1, -1)
    if (prior.shape != (batch, K, n) or ce_variance.shape != (batch, K, NSC)
            or not bool(torch.isfinite(prior).all() & (prior > 0).all())
            or not bool(torch.isclose(prior.sum(-1), torch.full((batch, K), ED, device=Y.device), atol=2e-4, rtol=1e-5).all())):
        raise ValueError('Positive per-MU prior summing48 and per-RE CE variance required')
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
    z_shared, v_shared = [], []
    if n == 44:
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            users = torch.tensor(members, device=Y.device)
            tones = torch.tensor(positions, device=Y.device)
            g = H_ul_est.index_select(1, users).index_select(-1, tones).permute(0, 3, 2, 1)
            pi_group = prior.index_select(1, users)[:, :, 4:].transpose(1, 2)
            vi = ce_variance.index_select(1, users).index_select(-1, tones).transpose(1, 2)
            effective = n0[:, None]+(pi_group*vi).sum(-1)
            ridge_group = phy.RX_RIDGE+effective[..., None]/pi_group
            gh = g.conj().transpose(-2, -1)
            w = torch.linalg.solve(gh@g+torch.diag_embed(ridge_group).to(g.dtype), gh)
            group_y = Y.index_select(-1, tones).transpose(-2, -1)
            z_shared.append((w@group_y[..., None]).squeeze(-1).transpose(1, 2))
            deviation = w@g-torch.eye(2, dtype=g.dtype, device=Y.device)
            v_shared.append(((deviation.abs().square()*pi_group[:, :, None, :]).sum(-1)
                +effective[..., None]*w.abs().square().sum(-1)).transpose(1, 2))
    active = torch.cat((z_key, torch.cat(z_shared, 1)), -1) if z_shared else z_key
    variance = torch.cat((v_key, torch.cat(v_shared, 1)), -1) if v_shared else v_key
    if not bool(torch.isfinite(active).all() & torch.isfinite(variance).all() & (variance > 0).all()):
        raise FloatingPointError('Nonfinite statistical recovery')
    padding = Y.new_zeros((batch, K, DMAX-n))
    return dict(soft=torch.cat((active, padding), -1),
        reliability=torch.cat((variance.reciprocal(), padding.real), -1), variance=variance,
        H_ul_est=H_ul_est, Y_res=Y)


# ---------------------------------------------------------------- pairing ----

@functools.lru_cache(maxsize=None)
def _pairings():
    def build(rest):
        if not rest:
            yield ()
            return
        first = rest[0]
        for other in rest[1:]:
            remaining = tuple(u for u in rest[1:] if u != other)
            for tail in build(remaining):
                yield ((first, other),)+tail
    return tuple(build(tuple(range(K))))


def coupling(H_ul_est):
    channel = H_ul_est.permute(0, 3, 1, 2)
    gram = channel@channel.conj().transpose(-1, -2)
    norm = channel.abs().square().sum(-1).clamp_min(1e-12)
    return (gram.abs().square()/(norm[..., :, None]*norm[..., None, :])).mean(1)


def best_pairing(H_ul_est):
    """Permutation [B,8]: slots (0,1),(2,3),... receive the least-coupled user pairs."""
    pairs = torch.tensor(_pairings(), device=H_ul_est.device)  # [105,4,2]
    c = coupling(H_ul_est)  # [B,8,8]
    cost = c[:, pairs[..., 0], pairs[..., 1]].sum(-1)  # [B,105]
    choice = cost.argmin(-1)
    return pairs[choice].reshape(-1, K), cost.gather(1, choice[:, None]).squeeze(1)/4


def slot_pair_coupling(H_ul_est):
    c = coupling(H_ul_est)
    return (c[:, 0, 1]+c[:, 2, 3]+c[:, 4, 5]+c[:, 6, 7])/4


def permute_users(value, perm):
    index = perm.reshape(perm.shape+(1,)*(value.ndim-2)).expand_as(value)
    return value.gather(1, index)


# ------------------------------------------------------------ data/random ----

def make_data(device):
    source = base().original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4,
            validation=cfg.MICRO_BATCH, test=1), device=str(device), cdl_model='C', speed=1.,
        seed_namespace=cfg.NAMESPACE)


def batch_random(role, index, device):
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(role, index, 'noise'))
    noise = torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, NT, NSC, 2, generator=gen,
        device=device).contiguous())/2**.5
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed(role, index, 'polarity'))
    polarity = (2*torch.randint(2, (cfg.MICRO_BATCH, K), generator=gen, device=device)-1).float()
    return noise, polarity


# ---------------------------------------------------------------- loading ----

def load_calibration():
    path = cfg.CE_CALIBRATION
    if sha256(path) != cfg.CE_CALIBRATION_SHA256:
        raise ValueError('Locked nmse65000 CE calibration bytes changed')
    record = json.loads(path.read_text())
    table = torch.tensor(record['per_RE_error'][cfg.CE_CALIBRATION_VARIANT], dtype=torch.float32)
    if table.shape != (6, K, NSC):
        raise ValueError('Locked calibration must be [6,8,256]')
    return table, dict(path=str(path), sha256=cfg.CE_CALIBRATION_SHA256, record_id=record.get('record_id'))


def load_sources(device):
    """Return verified source systems. All loaders keep their original strict checks."""
    import train_twc_v1_k8_low_snr as low_snr
    import train_twc_v1_k8_ce_refine as ce_refine
    hybrid, hybrid_stats, ce5000, hybrid_record = low_snr.load_finetuned(cfg.SOURCE_TRAJECTORY, device)
    fdma, fdma_stats, fdma_ce, fdma_record = base().load_source('fdma', device)
    ce65000, ce_record = ce_refine.load_terminal(cfg.CE_TRAJECTORY, device)
    ce65000 = ce65000.to(device).eval().requires_grad_(False)
    table, calibration = load_calibration()
    return dict(hybrid=(hybrid, hybrid_stats, ce5000, hybrid_record),
        fdma=(fdma, fdma_stats, fdma_ce, fdma_record),
        ce65000=(ce65000, ce_record), table65000=(table.to(device), calibration))


def build_final(hybrid_codec, hybrid_stats, table):
    """Independent copy: the source Hybrid stays usable as an unchanged comparator."""
    import copy
    model = FinalCodec(copy.deepcopy(hybrid_codec))
    stats = NodeStatistics(hybrid_stats.layout, table, hybrid_stats.power_ema.detach().cpu())
    return model, stats.to(table.device)


def swapped_statistics(stats, table):
    """Copy of an original ReceiverStatistics with a different locked CE table."""
    phy = physics()
    copy = phy.ReceiverStatistics(stats.layout, table.detach().cpu()).to(table.device)
    copy.power_ema.copy_(stats.power_ema)
    copy.ema_updates.copy_(stats.ema_updates)
    return copy.eval()


# ---------------------------------------------------------------- forward ----

def forward(system, sample, gamma, noise, polarity, *, pairing=False, encoder_gamma=None,
            pooled_prior=False):
    """One complete MU batch. ``system`` = dict(kind, codec, ce, stats).

    kind 'final' uses the gamma-aware encoder and per-node prior; kind 'plain'
    is an unchanged source system (single historical prior, no UE gamma).
    """
    phy = physics()
    codec, ce, stats, kind = system['codec'], system['ce'], system['stats'], system['kind']
    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=sample['H_ul'].device)
    if gamma.ndim == 0:
        gamma = gamma.expand(sample['H_ul'].shape[0])
    H_dl, H_ul = sample['H_dl'], sample['H_ul']
    perm, chosen = None, None
    if pairing:
        pilots = phy.known_pilots(H_ul.shape[0], H_ul.device, polarity)
        Y0 = phy.transmit(H_ul, pilots, noise, gamma)
        with torch.no_grad():
            est0 = ce(Y0, pilots, phy.fourier_ls(Y0, pilots), gamma_db=gamma)
            perm, chosen = best_pairing(est0)
        H_dl, H_ul = permute_users(H_dl, perm), permute_users(H_ul, perm)
    if kind == 'final':
        egamma = gamma if encoder_gamma is None else torch.full_like(gamma, float(encoder_gamma))
        raw = codec.encoder(H_dl, egamma)
    else:
        raw = codec.encoder(H_dl)
    wave = phy.build_waveform(raw, stats.layout, polarity=polarity)
    Y = phy.transmit(H_ul, wave.x_grid, noise, gamma)
    with torch.no_grad():
        estimated = ce(Y, wave.pilot, phy.fourier_ls(Y, wave.pilot), gamma_db=gamma)
        cevar = stats.ce_variance(gamma)
        if kind == 'final':
            prior = stats.nominal_power(gamma)
            if pooled_prior:
                prior = stats.nominal_power(torch.tensor(cfg.GAMMAS, dtype=torch.float32,
                    device=gamma.device)).mean(0, keepdim=True).expand_as(prior)
        else:
            prior = stats.nominal_power()
    recovered = recover_batched(Y, estimated, stats.layout, prior, cevar, gamma)
    out = codec.decoder(Y_rx=recovered['Y_res'], H_ul_est=estimated, soft=recovered['soft'],
        reliability=recovered['reliability'], pilots=wave.pilot, gamma_db=gamma, layout=stats.layout)
    return dict(H_hat=out['H_hat'], H_dl=H_dl, wave=wave, estimated=estimated, gamma=gamma,
        perm=perm, pair_coupling=slot_pair_coupling(estimated))


def sample_metrics(result):
    """Per-MU scalars for reports (true DL is used only for scoring)."""
    from twc_v1_k8_hybrid_direction import direction_error
    H_hat, H_dl = result['H_hat'], result['H_dl']
    _, rates = base().helpers().rate_objective(H_hat, H_dl, alpha=cfg.ALPHA, downlink_snr_db=cfg.DL_SNR)
    nmse = ((H_hat-H_dl).abs().square().sum((2, 3))/H_dl.abs().square().sum((2, 3))).mean(1)
    wave = result['wave']
    key_energy = wave.key.abs().square().sum(-1).mean(-1)
    data_energy = wave.active.abs().square().sum(-1).mean(-1)
    metrics = dict(sum_rate_bps_hz=rates['sum_rate_bps_hz'], dl_nmse_linear=nmse,
        direction_error=direction_error(H_hat, H_dl).to(nmse.dtype),
        interference_power_mean=rates['interference_power_mean'],
        desired_signal_power_mean=rates['desired_signal_power_mean'],
        min_user_rate_bps_hz=rates['user_rate_bps_hz'].min(-1).values,
        key_energy_fraction=key_energy/data_energy,
        pair_coupling=result['pair_coupling'])
    users = {f'user_rate_{k}': rates['user_rate_bps_hz'][:, k] for k in range(K)}
    return {name: value.detach().float().cpu() for name, value in {**metrics, **users}.items()}
