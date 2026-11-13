"""Read-only scalar diagnostics for frozen-codeword uplink recovery.

Truth is used only to score errors. Neither actual symbol powers nor true UL
channels are substituted into a deployable receiver prior by this module.
Recovery ``variance`` is the old receiver's diagonal *modelled MSE*, and its
``reliability`` is that variance's inverse; neither is an exact posterior.

Calibration bins retain descriptive counts/sums, not independent observations
or confidence intervals. Recovery errors/variances are scaled by historical
nominal coordinate power. CE bins use per-antenna channel coefficient MSE in
the original channel units. The final finite bin lower bound has no upper
bound, represented as ``None`` rather than JSON-incompatible infinity.
"""
from __future__ import annotations

import math

import torch


DEFAULT_CALIBRATION_EDGES = (0., .01, .03, .1, .3, 1., 3., 10.)


def _complex(value, name, *, ndim=None):
    if (not torch.is_tensor(value) or not value.is_complex()
            or ndim is not None and value.ndim != ndim
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + ' must be a finite complex tensor')
    return value.detach()


def _real(value, name):
    if (not torch.is_tensor(value) or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + ' must be a finite real floating tensor')
    return value.detach()


def _section_slices(layout, dimension):
    if hasattr(layout, 'data_positions'):
        if any(len(row) != dimension for row in layout.data_positions):
            raise ValueError('FDMA support does not match the active dimension')
        return {'all': slice(0, dimension)}
    if not hasattr(layout, 'key_positions') or not hasattr(layout, 'groups'):
        raise ValueError('Declared FDMA or pairwise Hybrid layout required')
    key = len(layout.key_positions[0])
    if key < 1 or key >= dimension or any(len(row) != key for row in layout.key_positions):
        raise ValueError('Hybrid key/shared support does not match the active dimension')
    return {'all': slice(0, dimension), 'key': slice(0, key),
            'shared': slice(key, dimension)}


def _prior(value, users, dimension, device):
    value = _real(value, 'historical nominal prior')
    if value.shape != (users, dimension) or value.device != device or not bool((value > 0).all()):
        raise ValueError('Positive historical nominal prior [users,active_dimension] required')
    return value


def _ratio(numerator, denominator, name):
    if not bool((denominator > 0).all()):
        raise ValueError(name + ' requires nonzero truth/normalizing energy')
    return numerator / denominator


def _recovery(wave, recovered, prior, layout):
    truth = _complex(wave.active, 'actual active codeword', ndim=3)
    batch, users, dimension = truth.shape
    if min(batch, users, dimension) < 1:
        raise ValueError('Nonempty active codeword required')
    soft = recovered['soft']
    if (not torch.is_tensor(soft) or soft.ndim != 3 or soft.shape[:2] != truth.shape[:2]
            or soft.shape[-1] < dimension):
        raise ValueError('Recovered soft tensor must include every active coordinate')
    estimate = _complex(soft[..., :dimension], 'active recovered codeword')
    variance = _real(recovered['variance'], 'modelled recovery variance')
    if (variance.shape != truth.shape or variance.device != truth.device
            or estimate.device != truth.device or not bool((variance > 0).all())):
        raise ValueError('Positive active modelled MSE with the codeword shape/device required')
    if 'reliability' in recovered:
        reliability = _real(recovered['reliability'][..., :dimension], 'active reliability')
        if (reliability.shape != truth.shape or reliability.device != truth.device
                or not bool(torch.isclose(reliability * variance,
                    torch.ones_like(variance), atol=2e-5, rtol=2e-5).all())):
            raise ValueError('Reliability must be inverse diagonal modelled recovery MSE')
    nominal = _prior(prior, users, dimension, truth.device)
    sections = _section_slices(layout, dimension)
    return truth, (estimate - truth).abs().square(), variance, nominal, sections


@torch.no_grad()
def recovery_metrics(wave, recovered, prior, layout):
    """Return finite [B] metrics; padded coordinates are never scored.

    Every soft NMSE first divides each user's squared-error energy by that
    user's transmitted energy over the selected coordinates, then averages
    those user ratios inside the MU. This matches the historical evaluator;
    pooling heterogeneous key/shared energies before division is different.

    ``*_actual_to_predicted_mse_ratio`` is the ratio of coordinate-averaged
    nominal-power-scaled actual and predicted errors, rather than the average
    of per-coordinate ratios. The latter is separately named
    ``*_standardized_squared_error_mean``. Neither is a posterior coverage.
    """
    truth, error, variance, nominal, sections = _recovery(wave, recovered, prior, layout)
    result = {}
    for section, selected in sections.items():
        prefix = 'soft' if section == 'all' else 'soft_' + section
        actual, predicted = error[..., selected], variance[..., selected]
        scaled_actual = actual / nominal[None, :, selected]
        scaled_predicted = predicted / nominal[None, :, selected]
        actual_mean = scaled_actual.mean((1, 2))
        predicted_mean = scaled_predicted.mean((1, 2))
        result.update({
            prefix + '_nmse_linear': _ratio(actual.sum(-1),
                truth[..., selected].abs().square().sum(-1), prefix + ' NMSE').mean(1),
            prefix + '_squared_error_mean': actual.mean((1, 2)),
            prefix + '_predicted_variance_mean': predicted.mean((1, 2)),
            prefix + '_nominal_scaled_actual_mse': actual_mean,
            prefix + '_nominal_scaled_predicted_variance': predicted_mean,
            prefix + '_actual_to_predicted_mse_ratio': actual_mean / predicted_mean,
            prefix + '_standardized_squared_error_mean': (actual / predicted).mean((1, 2)),
        })
    if any(value.shape != (truth.shape[0],) or not bool(torch.isfinite(value).all())
           for value in result.values()):
        raise FloatingPointError('Nonfinite per-MU recovery diagnostic')
    return result


def _active_positions(layout, users, tones, device):
    if hasattr(layout, 'data_positions'):
        rows = layout.data_positions
    else:
        if len(layout.key_positions) != users:
            raise ValueError('One key support per user required')
        shared_by_user = {}
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            if len(members) != 2:
                raise ValueError('Two-user shared groups required')
            for user in members:
                if user in shared_by_user:
                    raise ValueError('Shared user appears in multiple groups')
                shared_by_user[user] = positions
        if set(shared_by_user) != set(range(users)):
            raise ValueError('Shared groups must cover every user once')
        rows = [tuple(layout.key_positions[u]) + tuple(shared_by_user[u]) for u in range(users)]
    if len(rows) != users or not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Uniform active support for each user required')
    if any(len(set(row)) != len(row) or any(type(t) is not int or not 0 <= t < tones for t in row)
           for row in rows):
        raise ValueError('Unique in-range active tones required')
    return torch.tensor(rows, dtype=torch.long, device=device)


def _channels(H_est, H_true, ce_variance, layout):
    estimated = _complex(H_est, 'receiver UL channel', ndim=4)
    truth = _complex(H_true, 'scoring-only true UL channel', ndim=4)
    if estimated.shape != truth.shape or estimated.device != truth.device or min(truth.shape) < 1:
        raise ValueError('Matching nonempty channel tensors [B,user,antenna,tone] required')
    batch, users, _, tones = truth.shape
    predicted = _real(ce_variance, 'modelled per-antenna CE variance')
    if predicted.shape == (1, users, tones):
        predicted = predicted.expand(batch, -1, -1)
    if (predicted.shape != (batch, users, tones) or predicted.device != truth.device
            or not bool((predicted >= 0).all())):
        raise ValueError('Nonnegative CE variance [B,user,tone] required')
    error = (estimated - truth).abs().square().mean(2)
    true_energy = truth.abs().square().mean(2)
    positions = _active_positions(layout, users, tones, truth.device)
    selected = positions[None].expand(batch, -1, -1)
    sections = _section_slices(layout, positions.shape[-1])
    return estimated, error, predicted, true_energy, positions, selected, sections


@torch.no_grad()
def channel_metrics(H_est, H_true, ce_variance, layout, prior, *, condition_cap=1e8):
    """Score CE and the per-tone spatial receive geometry without changing it.

    Geometry uses the actual 32-by-2 receive matrix (antenna count is generic
    for CPU fixtures), with columns scaled by sqrt(historical nominal power).
    The receiver itself stays complex64. Analytic two-by-two Gram diagnostics
    use float64/complex128 locally to distinguish near-collinear columns.
    Singular condition values are explicitly capped and separately flagged.
    A CE calibration ratio is deliberately absent: zero predicted and actual
    CE error in the privileged variant has no meaningful zero/zero ratio.
    """
    if not math.isfinite(condition_cap) or condition_cap <= 1:
        raise ValueError('Finite condition cap greater than one required')
    estimated, error, predicted, true_energy, positions, selected, sections = _channels(
        H_est, H_true, ce_variance, layout)
    batch, users, _, _ = estimated.shape
    nominal = _prior(prior, users, positions.shape[-1], estimated.device)
    active_error, active_predicted, active_energy = (
        value.gather(-1, selected) for value in (error, predicted, true_energy))
    result = {}
    subsets = [('all_re', error, predicted, true_energy)]
    subsets += [(section, active_error[..., span], active_predicted[..., span], active_energy[..., span])
                for section, span in sections.items()]
    for section, actual, modelled, energy in subsets:
        prefix = 'ul_ce' if section == 'all' else 'ul_ce_' + section
        result.update({
            prefix + '_nmse_linear': _ratio(actual.sum((1, 2)), energy.sum((1, 2)), prefix + ' NMSE'),
            prefix + '_actual_mse': actual.mean((1, 2)),
            prefix + '_predicted_variance': modelled.mean((1, 2)),
            prefix + '_zero_predicted_variance_fraction': (modelled == 0).float().mean((1, 2)),
        })
    exclusive = layout.data_positions if hasattr(layout, 'data_positions') else layout.key_positions
    exclusive_positions = torch.tensor(exclusive, device=estimated.device)
    power = estimated.abs().square().sum(2).gather(-1, exclusive_positions[None].expand(batch, -1, -1))
    result['exclusive_channel_energy_mean'] = power.mean((1, 2))
    result['exclusive_channel_energy_min'] = power.flatten(1).amin(1)
    if hasattr(layout, 'groups'):
        coherence, conditions, singular, zero_columns = [], [], [], []
        d = len(layout.key_positions[0])
        for members, tones in zip(layout.groups, layout.shared_positions_by_group):
            user = torch.tensor(members, device=estimated.device)
            tone = torch.tensor(tones, device=estimated.device)
            g = estimated.index_select(1, user).index_select(-1, tone).permute(0, 3, 2, 1).to(torch.complex128)
            p = nominal.index_select(0, user)[:, d:].T.to(torch.float64)
            if p.shape != (len(tones), 2):
                raise ValueError('Shared nominal power coordinates must match physical shared tones')
            energy = g.abs().square().sum(-2)
            cross = (g[..., 0].conj() * g[..., 1]).sum(-1)
            coherence.append((cross.abs().square() / (energy[..., 0] * energy[..., 1]).clamp_min(
                torch.finfo(torch.float64).tiny)).clamp(0, 1))
            weighted = g * p.sqrt()[None, :, None]
            a, b = weighted.abs().square().sum(-2).unbind(-1)
            c = (weighted[..., 0].conj() * weighted[..., 1]).sum(-1).abs().square()
            spread = ((a - b).square() + 4 * c).sqrt()
            high, low = (a + b + spread) / 2, ((a + b - spread) / 2).clamp_min(0)
            minimum = (high / condition_cap).clamp_min(torch.finfo(torch.float64).tiny)
            conditions.append(torch.where(high > 0, high / torch.maximum(low, minimum),
                torch.full_like(high, condition_cap)).clamp(1, condition_cap))
            singular.append((low <= high / condition_cap) | (high == 0))
            zero_columns.append((energy <= torch.finfo(torch.float64).tiny).any(-1))
        coherence = torch.cat(coherence, 1)
        conditions = torch.cat(conditions, 1)
        result.update({
            'shared_channel_coherence_squared_mean': coherence.mean(1),
            'shared_channel_coherence_squared_max': coherence.amax(1),
            'shared_weighted_gram_condition_capped_mean': conditions.mean(1),
            'shared_weighted_gram_condition_capped_max': conditions.amax(1),
            'shared_weighted_gram_log10_condition_capped_mean': conditions.log10().mean(1),
            'shared_weighted_gram_near_singular_fraction': torch.cat(singular, 1).float().mean(1),
            'shared_zero_column_fraction': torch.cat(zero_columns, 1).float().mean(1),
        })
    if any(value.shape != (batch,) or not bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError('Nonfinite per-MU channel diagnostic')
    return result


def _bins(actual, predicted, component, edges):
    edges = tuple(float(edge) for edge in edges)
    if (not edges or edges[0] != 0 or any(not math.isfinite(v) or v < 0 for v in edges)
            or any(right <= left for left, right in zip(edges, edges[1:]))):
        raise ValueError('Increasing finite bin lower bounds starting at zero required')
    actual = _real(actual, 'observed calibration error').to(device='cpu', dtype=torch.float64).flatten()
    predicted = _real(predicted, 'predicted calibration variance').to(device='cpu', dtype=torch.float64).flatten()
    if actual.shape != predicted.shape or not bool((actual >= 0).all() & (predicted >= 0).all()):
        raise ValueError('Matching nonnegative calibration errors and variances required')
    index = torch.bucketize(predicted, torch.tensor(edges[1:], dtype=torch.float64), right=True)
    counts = torch.bincount(index, minlength=len(edges))
    actual_sums = torch.bincount(index, weights=actual, minlength=len(edges))
    predicted_sums = torch.bincount(index, weights=predicted, minlength=len(edges))
    return [dict(component=component, bin_index=i, lower=lower,
                 upper=edges[i+1] if i+1 < len(edges) else None,
                 count=int(counts[i]), sum_actual=float(actual_sums[i]),
                 sum_predicted=float(predicted_sums[i])) for i, lower in enumerate(edges)]


@torch.no_grad()
def recovery_calibration_bins(wave, recovered, prior, layout, *, edges=DEFAULT_CALIBRATION_EDGES):
    """Aggregate user-coordinate points; these counts are not independent MU."""
    _, error, variance, nominal, sections = _recovery(wave, recovered, prior, layout)
    output = []
    for section, selected in sections.items():
        output += _bins(error[..., selected] / nominal[None, :, selected],
            variance[..., selected] / nominal[None, :, selected], 'soft_' + section, edges)
    return output


@torch.no_grad()
def channel_calibration_bins(H_est, H_true, ce_variance, layout, *, edges=DEFAULT_CALIBRATION_EDGES):
    """Bin per-antenna CE MSE versus its modelled coefficient variance."""
    _, error, predicted, _, _, selected, sections = _channels(H_est, H_true, ce_variance, layout)
    output = _bins(error, predicted, 'ce_all_re', edges)
    actual, modelled = error.gather(-1, selected), predicted.gather(-1, selected)
    for section, span in sections.items():
        component = 'ce_all_active' if section == 'all' else 'ce_' + section
        output += _bins(actual[..., span], modelled[..., span], component, edges)
    return output
