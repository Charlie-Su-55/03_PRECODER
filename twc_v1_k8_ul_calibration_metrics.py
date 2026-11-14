"""Truth-scored scalar diagnostics for independently calibrated UL recovery.

The deployable receiver never calls this module with a true channel. Channel
truth and transmitted symbols appear only in offline validation scores. Pilot
residuals are observable features, not independent unbiased CE/noise labels.
All exported vectors contain one scalar per complete multiuser sample.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

import twc_v1_k8_ul_diagnostics_metrics as previous


def _channel_inputs(estimated, truth, variance):
    estimated = previous._complex(estimated, 'estimated UL', ndim=4)
    truth = previous._complex(truth, 'scoring-only true UL', ndim=4)
    if (estimated.shape != truth.shape or estimated.device != truth.device
            or min(estimated.shape) < 1 or estimated.shape[2] < 2):
        raise ValueError('Matching nonempty UL [B,user,antenna>=2,tone] tensors required')
    batch, users, _, tones = estimated.shape
    variance = previous._real(variance, 'historical per-antenna CE variance')
    if variance.shape == (1, users, tones):
        variance = variance.expand(batch, -1, -1)
    if (variance.shape != (batch, users, tones) or variance.device != estimated.device
            or not bool((variance >= 0).all())):
        raise ValueError('Nonnegative per-antenna CE variance [B,user,tone] required')
    return estimated, truth, variance


@torch.no_grad()
def pilot_residual_feature(Y, pilots, estimated, gamma):
    """Known-pilot residual/N0, masking data before finiteness inspection.

    The original rank8 LS design has64 coefficients and64 paid pilot RE per
    antenna. Its in-sample residual is not an independent error measurement.
    This U-Net-estimate residual is only an observable candidate feature.
    """
    # Delegate to the exact deployment feature: quantile-boundary membership
    # must not differ because diagnostics used a different numerical precision.
    from twc_v1_k8_ul_calibration import pilot_feature
    return pilot_feature(Y, pilots, estimated, gamma)


@torch.no_grad()
def diagnostic_vectors(estimated, true_ul, ce_variance, Y, pilots, gamma, layout):
    """Score CE direction, spatial structure and estimated shared geometry.

    Directional powers and the error covariance are descriptive within-MU
    statistics over correlated user/tone coordinates. They do not create
    additional independent CI samples. Geometry is unweighted because this
    API accepts no instantaneous power or historical power prior.
    """
    estimated, truth, variance = _channel_inputs(estimated, true_ul, ce_variance)
    batch, users, antennas, tones = estimated.shape
    previous._active_positions(layout, users, tones, estimated.device)
    h, error = estimated.to(torch.complex128), (truth-estimated).to(torch.complex128)
    norm2 = h.abs().square().sum(2)
    if not bool((norm2 > 1e-12).all()):
        raise ValueError('Identifiable estimated channel directions required; no truth fallback')
    inner = (h.conj()*error).sum(2)
    parallel = inner.abs().square()/norm2
    total = error.abs().square().sum(2)
    # Small negative roundoff is excluded only after checking the projection
    # identity; actual signed bias/cross statistics are not clipped.
    raw_perp = total-parallel
    tolerance = 1e-10 * total.clamp_min(1.)
    if bool((raw_perp < -tolerance).any()):
        raise FloatingPointError('Invalid orthogonal CE-error decomposition')
    perpendicular = raw_perp.clamp_min(0)/(antennas-1)
    reference = variance.to(torch.float64).clamp_min(1e-12)
    relative = inner/norm2
    weighted_relative = inner.sum((1, 2))/norm2.sum((1, 2))
    coordinates = error.permute(0, 1, 3, 2).reshape(batch, users*tones, antennas)
    mean = coordinates.mean(1)
    centered = coordinates-mean[:, None]
    covariance = centered.transpose(-2, -1) @ centered.conj() / (users*tones)
    trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(0)
    identity = torch.eye(antennas, dtype=covariance.dtype, device=covariance.device)
    isotropic = trace[:, None, None]/antennas * identity
    norm = torch.linalg.vector_norm(covariance, dim=(-2, -1))
    anisotropy = torch.where(norm > 0,
        torch.linalg.vector_norm(covariance-isotropic, dim=(-2, -1))/norm.clamp_min(1e-300),
        torch.zeros_like(norm))
    eigenvalues = torch.linalg.eigvalsh(covariance).real.clamp_min(0)
    truth_energy = truth.to(torch.complex128).abs().square().sum((1, 2, 3))
    if not bool((truth_energy > 0).all()):
        raise ValueError('Nonzero scoring-only UL energy required')
    result = dict(
        ce_nmse_linear=total.sum((1, 2))/truth_energy,
        ce_actual_mse=total.mean((1, 2))/antennas,
        ce_table_variance_mean=variance.to(torch.float64).mean((1, 2)),
        ce_zero_table_variance_fraction=(variance == 0).to(torch.float64).mean((1, 2)),
        ce_parallel_error_power=parallel.mean((1, 2)),
        ce_perpendicular_error_power_per_antenna=perpendicular.mean((1, 2)),
        ce_parallel_to_table_variance_ratio=(parallel/reference).mean((1, 2)),
        ce_perpendicular_to_table_variance_ratio=(perpendicular/reference).mean((1, 2)),
        ce_relative_projection_bias_real=relative.real.mean((1, 2)),
        ce_relative_projection_bias_imag=relative.imag.mean((1, 2)),
        ce_power_weighted_relative_bias_real=weighted_relative.real,
        ce_power_weighted_relative_bias_imag=weighted_relative.imag,
        ce_spatial_mean_power=mean.abs().square().mean(1),
        ce_spatial_covariance_anisotropy=anisotropy,
        ce_spatial_covariance_max_eigenvalue_fraction=torch.where(trace > 0,
            eigenvalues[:, -1]/trace.clamp_min(1e-300), torch.zeros_like(trace)),
        ce_directional_energy_closure_max_error=(total-parallel-perpendicular*(antennas-1)).abs().flatten(1).amax(1),
        pilot_residual_to_noise_ratio=pilot_residual_feature(Y, pilots, estimated, gamma))
    if hasattr(layout, 'groups'):
        correlations, singular, conditions, grams = [], [], [], []
        groups = tuple(layout.groups)
        supports = tuple(layout.shared_positions_by_group)
        flattened = [user for group in groups for user in group]
        if (len(groups) != len(supports) or sorted(flattened) != list(range(users))
                or any(len(group) != 2 for group in groups)):
            raise ValueError('Pairwise shared groups must cover all users exactly once')
        for members, positions in zip(groups, supports):
            indices = torch.tensor(members, device=h.device)
            selected = torch.tensor(positions, device=h.device)
            if (not len(positions) or len(set(positions)) != len(positions)
                    or any(type(i) is not int or not 0 <= i < tones for i in positions)):
                raise ValueError('Unique in-range shared tones required')
            g = h.index_select(1, indices).index_select(-1, selected).permute(0, 3, 2, 1)
            values = torch.linalg.svdvals(g)
            if (not bool(torch.isfinite(values).all()) or not bool((values[..., -1] > 1e-12).all())
                    or not bool((torch.linalg.matrix_rank(g) == 2).all())):
                raise ValueError('Singular estimated shared matrix; no truth-designed recovery fallback')
            energy = g.abs().square().sum(-2)
            cross = (g[..., 0].conj()*g[..., 1]).sum(-1).abs()
            correlations.append((cross/(energy[..., 0]*energy[..., 1]).sqrt()).clamp(0, 1))
            singular.append(values)
            conditions.append(values[..., 0]/values[..., 1])
            grams.append((values[..., 0]/values[..., 1]).square())
        correlation = torch.cat(correlations, 1)
        singular = torch.cat(singular, 1)
        condition, gram = torch.cat(conditions, 1), torch.cat(grams, 1)
        result.update(
            shared_estimated_normalized_correlation_mean=correlation.mean(1),
            shared_estimated_normalized_correlation_max=correlation.amax(1),
            shared_estimated_min_singular_value_mean=singular[..., 1].mean(1),
            shared_estimated_min_singular_value_min=singular[..., 1].amin(1),
            shared_estimated_max_singular_value_mean=singular[..., 0].mean(1),
            shared_estimated_matrix_condition_mean=condition.mean(1),
            shared_estimated_matrix_condition_max=condition.amax(1),
            shared_estimated_gram_condition_mean=gram.mean(1),
            shared_estimated_gram_condition_max=gram.amax(1))
    if any(value.shape != (batch,) or value.requires_grad or not bool(torch.isfinite(value).all())
           for value in result.values()):
        raise FloatingPointError('Finite detached complete-MU diagnostic scalars required')
    return result


@torch.no_grad()
def recovery_vectors(recovered, truth_active, layout):
    """Score actual soft errors and modelled MSE on the frozen nominal scale.

    Reuse the historical per-user normalization exactly. Only the source of
    recovered soft symbols and uncertainty differs between receiver variants.
    Active key/shared coordinates remain separate; padding never enters scores.
    """
    if not isinstance(recovered, dict) or 'prior' not in recovered:
        raise ValueError('Recovered receiver dictionary and frozen nominal prior required')
    result = previous.recovery_metrics(SimpleNamespace(active=truth_active), recovered,
        recovered['prior'], layout)
    for component in ('soft', 'soft_key', 'soft_shared'):
        name = component+'_nominal_scaled_predicted_variance'
        if name in result:
            result[component+'_nominal_scaled_reported_variance'] = result[name]
            result[component+'_actual_to_reported_mse_ratio'] = result[component+'_actual_to_predicted_mse_ratio']
    return result
