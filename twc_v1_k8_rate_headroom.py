"""Offline rate headroom and fixed-filter interference diagnostics.

Truth is used only for references and scoring. No function updates a model,
prior, receiver filter, or data address. The interference-free water-filled
expression is a mathematical upper bound for the specified total transmit
power; perfect-CSI RZF and the true-direction reference are not upper bounds.
"""
from __future__ import annotations

import math

import torch


POWER_KEYS = (
    'actual_error_power', 'realized_self_power', 'realized_cross_user_power',
    'realized_noise_power', 'realized_self_cross_user_cross',
    'realized_self_noise_cross', 'realized_cross_user_noise_cross',
    'realized_sum_power', 'conditional_self_power',
    'conditional_cross_user_power', 'conditional_noise_power',
    'conditional_total_power',
)


def _complex(value, name):
    if (not torch.is_tensor(value) or value.dtype not in (torch.complex64, torch.complex128)
            or not bool(torch.isfinite(value).all()) or any(size == 0 for size in value.shape)):
        raise ValueError('Finite nonempty complex tensor required: '+name)
    return value.detach()


def _channels(value, name):
    value = _complex(value, name)
    if value.ndim != 4:
        raise ValueError(name+' requires [B,K,Nt,S]')
    return value


def _max_per_mu(value):
    return value.abs().reshape(value.shape[0], -1).amax(1)


def _check(actual, expected, name, atol, rtol):
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(name+' closure failed')
    return _max_per_mu(actual-expected)


@torch.no_grad()
def true_direction_reference(H_hat, H_true):
    """Replace estimated directions by true ones, keeping estimated row norms.

    This removes one error component without selecting a new alpha or stream
    power. A resulting rate difference can have either sign. Zero true rows
    have no defined direction and are rejected rather than silently replaced.
    """
    estimate, truth = _channels(H_hat, 'estimated DL'), _channels(H_true, 'true DL')
    if (estimate.shape != truth.shape or estimate.dtype != truth.dtype
            or estimate.device != truth.device):
        raise ValueError('Matching estimated and true channels required')
    true_norm = torch.linalg.vector_norm(truth.to(torch.complex128), dim=2, keepdim=True)
    estimated_norm = torch.linalg.vector_norm(estimate.to(torch.complex128), dim=2, keepdim=True)
    if not bool((true_norm > 0).all()):
        raise ValueError('True direction undefined for a zero channel row')
    return (truth.to(torch.complex128)*(estimated_norm/true_norm)).to(estimate.dtype)


@torch.no_grad()
def interference_free_upper_bound(H_true, downlink_snr_db=25., *, total_power=1.):
    """Bound sum rate using no interference and optimal per-tone stream powers.

    For any W with ||W||_F^2 <= P, |h_k^H w_k|^2 <= ||h_k||^2 p_k,
    where p_k=||w_k||^2 and sum p_k <= P. Dropping interference then maximizing
    these independent links gives the stated bound. The maximizing powers are
    water filled, not assumed equal across users. Noise follows DL N0=10^-SNR/10.
    """
    truth = _channels(H_true, 'true DL')
    if (isinstance(downlink_snr_db, bool) or isinstance(total_power, bool)
            or not isinstance(downlink_snr_db, (int, float))
            or not isinstance(total_power, (int, float))
            or not math.isfinite(downlink_snr_db) or not math.isfinite(total_power)
            or total_power <= 0):
        raise ValueError('Finite scalar DL SNR and positive total power required')
    try:
        noise_value = 10.**(-downlink_snr_db/10.)
    except OverflowError as exc:
        raise ValueError('DL noise outside float64 range') from exc
    if not math.isfinite(noise_value) or noise_value <= 0:
        raise ValueError('Positive finite DL noise required')
    gains = truth.to(torch.complex128).abs().square().sum(2).permute(0, 2, 1)
    if not bool(torch.isfinite(gains).all()):
        raise ValueError('DL gains outside float64 range')
    noise = gains.new_tensor(noise_value)
    thresholds = torch.where(gains > 0, noise/gains, torch.inf)
    ordered = thresholds.sort(-1).values
    finite = torch.isfinite(ordered)
    count = torch.arange(1, gains.shape[-1]+1, dtype=torch.float64, device=gains.device)
    levels = (total_power+torch.where(finite, ordered, 0.).cumsum(-1))/count
    active_count = ((ordered < levels) & finite).sum(-1, keepdim=True)
    level = levels.gather(-1, (active_count-1).clamp_min(0))
    allocation = torch.where(active_count > 0, (level-thresholds).clamp_min(0), 0.)
    rates = torch.log1p(gains*allocation/noise)/math.log(2.)
    return dict(sum_rate_bps_hz=rates.sum(-1).mean(-1),
        per_tone_sum_rate_bps_hz=rates.sum(-1), user_rate_bps_hz=rates.mean(1),
        power_allocation=allocation, channel_gains=gains, noise_power=noise)


def _nonnegative_broadcast(value, shape, device, name):
    tensor = torch.as_tensor(value, device=device)
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise ValueError('Real nonnegative '+name+' required')
    tensor = tensor.detach().to(torch.float64)
    if not bool(torch.isfinite(tensor).all() & (tensor >= 0).all()):
        raise ValueError('Finite nonnegative '+name+' required')
    try:
        return torch.broadcast_to(tensor, shape)
    except RuntimeError as exc:
        raise ValueError('Broadcastable '+name+' required') from exc


@torch.no_grad()
def fixed_filter_error_decomposition(A, G_true, symbols, noise, *, prior=None,
                                    noise_variance=None, atol=3e-5, rtol=5e-5):
    """Decompose a fixed linear receiver into self error, crosstalk and noise.

    Shapes are A [B,...,K,Nr], G [B,...,Nr,K], z [B,...,K], and actual
    scaled noise [B,...,Nr]. The realized identity retains all signed cross
    terms; independent zero-mean diagonal-prior conditional powers do not.
    Conditional powers require both historical prior and theoretical N0.
    """
    original = [_complex(v, n) for v, n in
                ((A, 'filter'), (G_true, 'true channel'), (symbols, 'symbols'), (noise, 'noise'))]
    a0, g0, z0, n0 = original
    if (a0.ndim < 3 or g0.shape != (*a0.shape[:-2], a0.shape[-1], a0.shape[-2])
            or z0.shape != a0.shape[:-1] or n0.shape != (*a0.shape[:-2], a0.shape[-1])
            or any(v.device != a0.device or v.dtype != a0.dtype for v in original)):
        raise ValueError('Matching fixed-filter block shapes, dtypes and devices required')
    if (prior is None) != (noise_variance is None):
        raise ValueError('Historical prior and noise variance must be supplied together')
    a, g, z, n = [v.to(torch.complex128) for v in original]
    deviation = a@g-torch.eye(a.shape[-2], device=a.device, dtype=a.dtype)
    diagonal = deviation.diagonal(dim1=-2, dim2=-1)
    off_diagonal = deviation-torch.diag_embed(diagonal)
    own = diagonal*z
    cross = (off_diagonal@z.unsqueeze(-1)).squeeze(-1)
    thermal = (a@n.unsqueeze(-1)).squeeze(-1)
    error = (a@((g@z.unsqueeze(-1)).squeeze(-1)+n).unsqueeze(-1)).squeeze(-1)-z
    summed = own+cross+thermal
    values = dict(actual_error_power=error.abs().square(),
        realized_self_power=own.abs().square(), realized_cross_user_power=cross.abs().square(),
        realized_noise_power=thermal.abs().square(),
        realized_self_cross_user_cross=2*(own*cross.conj()).real,
        realized_self_noise_cross=2*(own*thermal.conj()).real,
        realized_cross_user_noise_cross=2*(cross*thermal.conj()).real)
    values['realized_sum_power'] = sum(values[key] for key in POWER_KEYS[1:7])
    checks = dict(error_closure_max_abs=_check(error, summed, 'Fixed-filter error', atol, rtol),
        realized_power_closure_max_abs=_check(values['actual_error_power'],
            values['realized_sum_power'], 'Fixed-filter power', atol, rtol))
    if prior is not None:
        power = _nonnegative_broadcast(prior, z.shape, z.device, 'historical prior')
        variance = _nonnegative_broadcast(noise_variance, z.shape[:-1], z.device, 'noise variance')
        values.update(conditional_self_power=diagonal.abs().square()*power,
            conditional_cross_user_power=(off_diagonal.abs().square()*power.unsqueeze(-2)).sum(-1),
            conditional_noise_power=variance.unsqueeze(-1)*a.abs().square().sum(-1))
        values['conditional_total_power'] = sum(values[key] for key in POWER_KEYS[8:11])
    values.update(error=error, terms=dict(self=own, cross_user=cross, noise=thermal), checks=checks)
    return values


@torch.no_grad()
def receiver_error_decomposition(observation, recovered, H_true, standard_noise, layout, gamma):
    """Score production normal filters without changing their construction.

    Reuse the archived receiver's filter reconstruction and paid waveform.
    Returned coordinates and filters are transient; callers export aggregates.
    The true channel never enters the construction of A.
    """
    import twc_v1_k8_ul_propagation_metrics as propagation
    from ul_diagnostics_runtime import physics

    blocks, filters, sections = propagation._blocks(observation, recovered, layout)
    truth = _complex(observation['wave'].active, 'transmitted active symbols')
    channel = _channels(H_true, 'offline true UL')
    standard = _complex(standard_noise, 'actual standard noise')
    batch, users, dimension = truth.shape
    if (channel.shape != observation['estimated'].shape or standard.shape != observation['Y'].shape
            or observation['prior'].shape != (users, dimension)
            or any(v.device != truth.device for v in (channel, standard, observation['prior']))):
        raise ValueError('Matching production true-channel/noise/prior shapes required')
    active = _complex(recovered['soft'][..., :dimension], 'normal active soft')
    if active.shape != truth.shape:
        raise ValueError('Matching production active soft shape required')
    n0 = physics().noise_variance(gamma, batch, truth.device)
    for name, value in (('prior', observation['prior']), ('Y_res', observation['Y']),
                        ('H_ul_est', observation['estimated']), ('noise', n0)):
        if name not in recovered or not torch.equal(recovered[name], value):
            raise ValueError('Normal recovered input changed: '+name)
    coordinate = {key: torch.empty(truth.shape, dtype=torch.float64, device=truth.device)
                  for key in POWER_KEYS}
    rebuilt = torch.empty_like(truth)
    receives, power_checks = [], []
    for block in blocks:
        gt = propagation._gather_block(channel, block, channel=True)
        z = propagation._gather_block(truth, block)
        if block['kind'] == 'exclusive':
            index, d = block['tones'], block['tones'].shape[-1]
            xi = standard[:, None].expand(-1, users, -1, -1).gather(-1,
                index[None, :, None].expand(batch, users, standard.shape[1], d))
            xi = xi.permute(0, 1, 3, 2).reshape(batch, users*d, standard.shape[1])
        else:
            xi = standard.index_select(-1, block['tones']).transpose(1, 2)
        thermal = xi.to(torch.complex128)*n0.to(torch.float64).sqrt()[:, None, None]
        clean = (gt.to(torch.complex128)@z.to(torch.complex128).unsqueeze(-1)).squeeze(-1)
        receives.append(_check(block['Y'].to(torch.complex128), clean+thermal,
                               'Paid received signal', 3e-5, 5e-5))
        score = fixed_filter_error_decomposition(block['A'].to(torch.complex128),
            gt.to(torch.complex128), z.to(torch.complex128), thermal,
            prior=block['P'][None], noise_variance=n0.to(torch.float64)[:, None])
        pred = (block['A']@block['Y'].unsqueeze(-1)).squeeze(-1)
        propagation._scatter(rebuilt, pred, block)
        power_checks.append(score['checks']['realized_power_closure_max_abs'])
        for key in POWER_KEYS:
            propagation._scatter(coordinate[key], score[key], block)
    actual = (active.to(torch.complex128)-truth.to(torch.complex128)).abs().square()
    _check(actual, coordinate['realized_sum_power'], 'Normal receiver power', 3e-5, 5e-5)
    coordinate['actual_error_power'].copy_(actual)
    return dict(coordinate=coordinate, filters=filters, sections=sections,
        checks=dict(received_closure_max_abs=torch.stack(receives).amax(0),
            soft_closure_max_abs=_check(active, rebuilt, 'Normal receiver soft', 3e-5, 5e-5),
            realized_power_closure_max_abs=torch.maximum(torch.stack(power_checks).amax(0),
                _max_per_mu(actual-coordinate['realized_sum_power']))))
