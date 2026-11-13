"""Conditional error propagation through the unchanged normal UL filter.

Only estimated channels and the receiver's actual ridge values build A_est.
True channels, clean symbols and realized noise are used after that step for
offline scoring. Historical nominal powers, rather than realized symbol
powers, remain the diagonal P in every conditional model calculation.
"""
from __future__ import annotations

import torch

import twc_v1_k8_ul_diagnostics_metrics as previous

DEFAULT_CALIBRATION_EDGES = previous.DEFAULT_CALIBRATION_EDGES


def _finite(value, name, *, complex_value=None):
    if (not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
            or complex_value is not None and value.is_complex() != complex_value):
        raise ValueError('Finite '+name+' tensor required')
    return value.detach()


def _max_per_mu(value):
    return value.abs().reshape(value.shape[0], -1).amax(1)


def _check(actual, expected, name, *, atol, rtol):
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError('Normal-filter '+name+' reconstruction failed')
    return _max_per_mu(actual-expected)


def _blocks(observation, recovered, layout):
    """Build normal filters without accepting true channels as an argument."""
    H = _finite(observation['estimated'], 'estimated UL', complex_value=True)
    Y = _finite(observation['Y'], 'received Y', complex_value=True)
    prior = _finite(observation['prior'], 'historical prior', complex_value=False)
    if H.dtype != torch.complex64 or Y.dtype != torch.complex64:
        raise ValueError('Normal receiver channels/Y must retain complex64')
    if H.ndim != 4 or Y.ndim != 3:
        raise ValueError('UL [B,K,Nt,Nsc] and Y [B,Nt,Nsc] required')
    batch, users, antennas, tones = H.shape
    if Y.shape != (batch, antennas, tones) or prior.ndim != 2 or prior.shape[0] != users:
        raise ValueError('Matching receiver shapes required')
    if not bool((prior > 0).all()):
        raise ValueError('Positive historical powers required')
    dimension = prior.shape[-1]
    sections = previous._section_slices(layout, dimension)
    positions = layout.data_positions if hasattr(layout, 'data_positions') else layout.key_positions
    index = torch.tensor(positions, device=H.device)
    exclusive = index.shape[-1]
    h = H.gather(-1, index[None, :, None].expand(batch, users, antennas, exclusive))
    h = h.permute(0, 1, 3, 2)
    y = Y[:, None].expand(-1, users, -1, -1).gather(-1,
        index[None, :, None].expand(batch, users, antennas, exclusive)).permute(0, 1, 3, 2)
    ridge = _finite(recovered['key_effective_lambda'], 'exclusive normal ridge', complex_value=False)
    if ridge.shape != (batch, users, exclusive) or not bool((ridge >= 0).all()):
        raise ValueError('Matching nonnegative exclusive normal ridge required')
    power = h.abs().square().sum(-1)
    if not bool((power > 1e-12).all()):
        raise ValueError('Identifiable exclusive estimated channels required')
    a = h.conj()/(power+ridge)[..., None]
    blocks = [dict(kind='exclusive', users=None, tones=index, start=0, stop=exclusive,
        G=h.reshape(batch, users*exclusive, antennas, 1),
        A=a.reshape(batch, users*exclusive, 1, antennas),
        Y=y.reshape(batch, users*exclusive, antennas),
        P=prior[:, :exclusive].reshape(users*exclusive, 1))]
    shared_filters = []
    if 'shared' in sections:
        shared_ridge = _finite(recovered['shared_effective_lambda'], 'shared normal ridge', complex_value=False)
        if shared_ridge.shape != (batch, users, dimension-exclusive) or not bool((shared_ridge >= 0).all()):
            raise ValueError('Matching nonnegative shared normal ridge required')
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            if len(members) != 2:
                raise ValueError('Pairwise shared receiver required')
            selected_users = torch.tensor(members, device=H.device)
            selected_tones = torch.tensor(positions, device=H.device)
            g = H.index_select(1, selected_users).index_select(-1, selected_tones).permute(0, 3, 2, 1)
            lambdas = shared_ridge.index_select(1, selected_users).transpose(1, 2)
            gh = g.conj().transpose(-2, -1)
            a_group = torch.linalg.solve(gh@g+torch.diag_embed(lambdas).to(g.dtype), gh)
            shared_filters.append(a_group)
            blocks.append(dict(kind='shared', users=selected_users, tones=selected_tones,
                start=exclusive, stop=dimension, G=g, A=a_group,
                Y=Y.index_select(-1, selected_tones).transpose(1, 2),
                P=prior.index_select(0, selected_users)[:, exclusive:].T))
    filters = dict(exclusive=a,
        shared=torch.stack(shared_filters, 1) if shared_filters else H.new_empty((batch, 0, 0, 0, antennas)))
    return blocks, filters, sections


@torch.no_grad()
def reconstruct_normal_filters(observation, recovered, layout):
    """Return filter tensors for hashing; no true-UL argument is accepted."""
    return _blocks(observation, recovered, layout)[1]


def _gather_block(value, block, *, channel=False):
    batch = value.shape[0]
    if block['kind'] == 'exclusive':
        users, d = block['tones'].shape
        if channel:
            antennas = value.shape[2]
            gathered = value.gather(-1, block['tones'][None, :, None].expand(batch, users, antennas, d))
            return gathered.permute(0, 1, 3, 2).reshape(batch, users*d, antennas, 1)
        return value[..., :d].reshape(batch, users*d, 1)
    selected = value.index_select(1, block['users'])
    if channel:
        return selected.index_select(-1, block['tones']).permute(0, 3, 2, 1)
    return selected[..., block['start']:block['stop']].transpose(1, 2)


def _scatter(target, values, block):
    if block['kind'] == 'exclusive':
        users, d = block['tones'].shape
        target[..., :d] = values.reshape(target.shape[0], users, d)
    else:
        target[:, block['users'], block['start']:block['stop']] = values.transpose(1, 2)


@torch.no_grad()
def analyze(observation, recovered, H_true, standard_noise, layout, gamma):
    """Return per-MU vectors, coordinate diagnostics, fixed filters and checks.

    ``coordinate`` and ``filters`` are transient analysis tensors, not result
    export artifacts. The caller may export scalar summaries and filter hashes.
    Conditional models assume zero-mean symbols with covariance diag(prior).
    The realized decomposition makes no such assumption and retains all three
    signed cross terms. Neither conditional truth nor actual frame powers enter
    the deployable filter construction.
    """
    blocks, filters, sections = _blocks(observation, recovered, layout)
    H_true = _finite(H_true, 'offline true UL', complex_value=True)
    noise = _finite(standard_noise, 'actual standard noise', complex_value=True)
    prior = observation['prior']
    wave = observation['wave']
    truth = _finite(wave.active, 'offline transmitted symbols', complex_value=True)
    batch, users, dimension = truth.shape
    if (H_true.shape != observation['estimated'].shape
            or noise.shape != observation['Y'].shape or prior.shape != (users, dimension)):
        raise ValueError('Matching true-channel/noise/codeword shapes required')
    if not all(value.device == truth.device for value in (H_true, noise, prior, observation['Y'])):
        raise ValueError('Diagnostic tensors must share a device')
    # Use the old public noise rule rather than deriving N0 from measured noise.
    from ul_diagnostics_runtime import physics
    n0 = physics().noise_variance(gamma, batch, truth.device)
    ce_variance = _finite(observation['ce_variance'], 'normal calibrated CE variance', complex_value=False)
    if ce_variance.shape == (1, users, H_true.shape[-1]):
        ce_variance = ce_variance.expand(batch, -1, -1)
    if (ce_variance.shape != (batch, users, H_true.shape[-1])
            or not bool((ce_variance >= 0).all())):
        raise ValueError('Matching nonnegative normal CE variance required')
    if ('ce_variance' not in recovered or not torch.equal(recovered['ce_variance'], ce_variance)
            or 'noise' not in recovered or not torch.equal(recovered['noise'], n0)):
        raise ValueError('Normal recovered CE variance/noise input changed')
    actual = recovered['soft'][..., :dimension]
    reported = _finite(recovered['variance'], 'reported normal variance', complex_value=False)
    if actual.shape != truth.shape or reported.shape != truth.shape or not bool((reported > 0).all()):
        raise ValueError('Matching active soft values and positive modelled variance required')
    for name, value in (('prior', prior), ('Y_res', observation['Y']),
                        ('H_ul_est', observation['estimated'])):
        if name not in recovered or not torch.equal(recovered[name], value):
            raise ValueError('Normal recovered input changed: '+name)

    coordinate_names = (
        'actual_error_power', 'reported_variance', 'propagated_variance',
        'realized_bias_power', 'realized_ce_power', 'realized_noise_power',
        'realized_bias_ce_cross', 'realized_bias_noise_cross', 'realized_ce_noise_cross',
        'conditional_bias_power', 'conditional_ce_power', 'conditional_noise_power',
        'conditional_bias_ce_cross', 'reported_ce_power')
    coordinate = {name: torch.empty(truth.shape, dtype=torch.float64, device=truth.device)
                  for name in coordinate_names}
    coordinate['actual_error_power'].copy_((actual-truth).abs().square())
    coordinate['reported_variance'].copy_(reported)
    rebuilt_soft = torch.empty_like(truth)
    rebuilt_variance = torch.empty_like(reported)
    realized_error = torch.empty(truth.shape, dtype=torch.complex128, device=truth.device)
    receive_checks = []
    for block in blocks:
        A, G = block['A'], block['G']
        gt = _gather_block(H_true, block, channel=True)
        z = _gather_block(truth, block)
        if block['kind'] == 'exclusive':
            pred = (A[..., 0, :]*block['Y']).sum(-1)[..., None]
            index = block['tones']
            count = index.shape[-1]
            xi = noise[:, None].expand(-1, users, -1, -1).gather(-1,
                index[None, :, None].expand(batch, users, noise.shape[1], count))
            xi = xi.permute(0, 1, 3, 2).reshape(batch, users*count, noise.shape[1])
            ce = ce_variance.gather(-1, index[None].expand(batch, -1, -1))
            ce = ce.reshape(batch, users*count, 1)
        else:
            pred = (A@block['Y'][..., None]).squeeze(-1)
            xi = noise.index_select(-1, block['tones']).transpose(1, 2)
            ce = ce_variance.index_select(1, block['users']).index_select(-1,
                block['tones']).transpose(1, 2)
        _scatter(rebuilt_soft, pred, block)
        identity = torch.eye(A.shape[-2], dtype=A.dtype, device=A.device)
        deviation = A@G-identity
        normal_bias = (deviation.abs().square()*block['P'][None, :, None]).sum(-1)
        effective_ce = (block['P'][None]*ce).sum(-1)
        row_power = A.abs().square().sum(-1)
        normal_variance = normal_bias+(n0[:, None]+effective_ce)[..., None]*row_power
        _scatter(rebuilt_variance, normal_variance, block)

        # Score with higher precision without modifying the complex64 receiver.
        a, gh, gtrue, symbols = (value.to(torch.complex128) for value in (A, G, gt, z))
        powers = block['P'].to(torch.float64)
        deviation = a@gh-identity.to(torch.complex128)
        ce_deviation = a@(gtrue-gh)
        conditional_deviation = a@gtrue-identity.to(torch.complex128)
        thermal = xi.to(torch.complex128)*n0.to(torch.float64).sqrt()[:, None, None]
        clean = (gtrue@symbols[..., None]).squeeze(-1)
        receive_checks.append(_max_per_mu(block['Y'].to(torch.complex128)-clean-thermal))
        if not torch.allclose(block['Y'].to(torch.complex128), clean+thermal, atol=3e-5, rtol=5e-5):
            raise ValueError('Offline true channel/noise does not reconstruct actual paid Y')
        bias_term = (deviation@symbols[..., None]).squeeze(-1)
        ce_term = (ce_deviation@symbols[..., None]).squeeze(-1)
        noise_term = (a@thermal[..., None]).squeeze(-1)
        _scatter(realized_error, bias_term+ce_term+noise_term, block)
        conditional_bias = (deviation.abs().square()*powers[None, :, None]).sum(-1)
        conditional_ce = (ce_deviation.abs().square()*powers[None, :, None]).sum(-1)
        conditional_noise = n0.to(torch.float64)[:, None, None]*a.abs().square().sum(-1)
        conditional_cross = 2*(deviation*ce_deviation.conj()*powers[None, :, None]).sum(-1).real
        propagated = (conditional_deviation.abs().square()*powers[None, :, None]).sum(-1)+conditional_noise
        values = dict(
            propagated_variance=propagated,
            realized_bias_power=bias_term.abs().square(), realized_ce_power=ce_term.abs().square(),
            realized_noise_power=noise_term.abs().square(),
            realized_bias_ce_cross=2*(bias_term*ce_term.conj()).real,
            realized_bias_noise_cross=2*(bias_term*noise_term.conj()).real,
            realized_ce_noise_cross=2*(ce_term*noise_term.conj()).real,
            conditional_bias_power=conditional_bias, conditional_ce_power=conditional_ce,
            conditional_noise_power=conditional_noise, conditional_bias_ce_cross=conditional_cross,
            reported_ce_power=effective_ce.to(torch.float64)[..., None]*a.abs().square().sum(-1))
        for name, value in values.items():
            _scatter(coordinate[name], value, block)

    realized_sum = sum(coordinate[name] for name in (
        'realized_bias_power', 'realized_ce_power', 'realized_noise_power',
        'realized_bias_ce_cross', 'realized_bias_noise_cross', 'realized_ce_noise_cross'))
    conditional_sum = sum(coordinate[name] for name in (
        'conditional_bias_power', 'conditional_ce_power', 'conditional_noise_power',
        'conditional_bias_ce_cross'))
    checks = dict(
        filter_recovery_max_error=_check(rebuilt_soft, actual, 'A_est Y', atol=3e-5, rtol=5e-5),
        reported_variance_max_error=_check(rebuilt_variance, reported, 'reported variance', atol=1e-7, rtol=5e-5),
        realized_error_decomposition_max_error=_check(realized_error, (actual-truth).to(torch.complex128),
            'realized error', atol=3e-5, rtol=5e-5),
        realized_power_decomposition_max_error=_check(realized_sum, coordinate['actual_error_power'],
            'realized signed powers', atol=2e-7, rtol=5e-4),
        conditional_variance_decomposition_max_error=_check(conditional_sum, coordinate['propagated_variance'],
            'conditional signed powers', atol=1e-10, rtol=1e-10),
        receive_model_max_error=torch.stack(receive_checks).amax(0))
    nominal = prior.to(torch.float64)[None]
    vectors = {}
    for section, selected in sections.items():
        prefix = 'soft' if section == 'all' else 'soft_'+section
        scaled = {name: (value[..., selected]/nominal[..., selected]).mean((1, 2))
                  for name, value in coordinate.items()}
        vectors[prefix+'_nmse_linear'] = (coordinate['actual_error_power'][..., selected].sum(-1)
            / truth[..., selected].abs().square().to(torch.float64).sum(-1)).mean(1)
        vectors[prefix+'_nominal_scaled_actual_mse'] = scaled['actual_error_power']
        vectors[prefix+'_nominal_scaled_reported_variance'] = scaled['reported_variance']
        vectors[prefix+'_nominal_scaled_propagated_variance'] = scaled['propagated_variance']
        vectors[prefix+'_actual_to_reported_mse_ratio'] = scaled['actual_error_power']/scaled['reported_variance']
        vectors[prefix+'_actual_to_propagated_mse_ratio'] = scaled['actual_error_power']/scaled['propagated_variance']
        for name in coordinate_names[3:]:
            vectors[prefix+'_nominal_scaled_'+name] = scaled[name]
    vectors.update(checks)
    if any(not bool(torch.isfinite(value).all()) for value in (*coordinate.values(), *vectors.values())):
        raise FloatingPointError('Nonfinite normal-filter propagation diagnostic')
    if not bool((coordinate['propagated_variance'] > 0).all()):
        raise FloatingPointError('Positive conditional MSE required')
    return dict(vectors=vectors, coordinate=coordinate, filters=filters, checks=checks)


@torch.no_grad()
def calibration_bins(result, prior, layout, *, edges=DEFAULT_CALIBRATION_EDGES):
    """Descriptive normalized-error bins; coordinates do not become MU samples."""
    coordinate = result['coordinate']
    actual = coordinate['actual_error_power']
    nominal = previous._prior(prior, actual.shape[1], actual.shape[2], actual.device)[None]
    sections = previous._section_slices(layout, actual.shape[2])
    rows = []
    for section, selected in sections.items():
        component = 'soft_all' if section == 'all' else 'soft_'+section
        truth = actual[..., selected]/nominal[..., selected]
        for kind in ('reported', 'propagated'):
            predicted = coordinate[kind+'_variance'][..., selected]/nominal[..., selected]
            rows.extend(previous._bins(truth, predicted, kind+'_'+component, edges))
    return rows
