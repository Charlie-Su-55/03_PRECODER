"""Pilot-CE losses and truth-scored geometry; no decoder or true-input receiver.

The CE network receives paid pilots, its LS initial estimate and public gamma.
True UL enters these training labels/offline scores only. All reported vectors
contain one scalar per complete MU, not independent antenna/tone samples.
"""
from __future__ import annotations


def _channels(prediction, truth):
    import torch
    if (not torch.is_tensor(prediction) or not torch.is_tensor(truth)
            or prediction.ndim != 4 or prediction.shape != truth.shape
            or min(prediction.shape) < 1 or prediction.shape[1] < 2
            or prediction.shape[2] < 2 or prediction.dtype not in (torch.complex64, torch.complex128)
            or truth.dtype != prediction.dtype or truth.device != prediction.device
            or not bool(torch.isfinite(prediction).all() & torch.isfinite(truth).all())):
        raise ValueError('Matching finite complex [MU,UE>=2,antenna>=2,tone] channels required')
    truth = truth.detach()
    p2, t2 = prediction.abs().square().sum(2), truth.abs().square().sum(2)
    if not bool((p2 > 1e-20).all() & (t2 > 1e-20).all()):
        raise ValueError('Nonzero predicted and label directions required')
    return prediction, truth, p2, t2


def vectors(prediction, truth):
    """Physical CSI fidelity and normalized inter-user spatial structure."""
    import torch
    pred, label, p2, t2 = _channels(prediction, truth)
    error2 = (pred-label).abs().square()
    link_nmse = error2.sum((2, 3))/label.abs().square().sum((2, 3))
    inner = (pred.conj()*label).sum(2)
    cosine2 = (inner.abs().square()/(p2*t2)).clamp(0, 1)
    u = (pred/p2.sqrt()[:, :, None]).permute(0, 3, 1, 2)
    v = (label/t2.sqrt()[:, :, None]).permute(0, 3, 1, 2)
    gram_difference = u@u.conj().transpose(-2, -1)-v@v.conj().transpose(-2, -1)
    off_diagonal = ~torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device)
    result = dict(ul_nmse_linear=link_nmse.mean(1),
        direction_error=(1-cosine2).mean((1, 2)),
        gram_error=gram_difference.abs().square()[..., off_diagonal].mean((1, 2)),
        energy_ratio=pred.abs().square().sum((1, 2, 3))/label.abs().square().sum((1, 2, 3)),
        ce_mse=error2.mean((1, 2, 3)))
    if any(value.shape != (pred.shape[0],) or not bool(torch.isfinite(value).all())
            for value in result.values()):
        raise FloatingPointError('Nonfinite complete-MU CE score')
    return result


def objective(prediction, truth, trajectory):
    """Fixed weights: NMSE alone, or NMSE+direction+off-diagonal Gram MSE."""
    import torch
    if trajectory not in ('nmse', 'geometry'):
        raise ValueError('Only declared CE continuation trajectories may train')
    from configs import twc_v1_k8_ce_refine_config as cfg
    score = vectors(prediction, truth)
    auxiliary = cfg.DIRECTION_WEIGHT*score['direction_error']+cfg.GRAM_WEIGHT*score['gram_error']
    loss = score['ul_nmse_linear'].mean()
    if trajectory == 'geometry':
        loss = loss+auxiliary.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite CE objective')
    return loss, score


def _layout(layout, users, tones):
    groups, supports = tuple(layout.groups), tuple(layout.shared_positions_by_group)
    if (len(groups) != len(supports) or sorted(user for pair in groups for user in pair) != list(range(users))
            or any(len(pair) != 2 for pair in groups)
            or any(not row or len(set(row)) != len(row) or
                any(type(q) is not int or not 0 <= q < tones for q in row) for row in supports)):
        raise ValueError('Pairwise public shared supports must cover the declared users')
    return groups, supports


def shared_filters(estimated, ce_variance, gamma_db, layout, *, nominal_power=48./44., ridge=.001):
    """Design public filters with NO true channel or current codeword argument.

    The equal nominal power is a declared analytical probe, not a replacement
    for the real trained codec's historical prior. Caller may score these
    frozen filters against true UL, but cannot use true UL to design them.
    """
    import math
    import torch
    if (not torch.is_tensor(estimated) or estimated.ndim != 4 or min(estimated.shape) < 1
            or estimated.dtype not in (torch.complex64, torch.complex128)
            or not bool(torch.isfinite(estimated).all())
            or isinstance(nominal_power, bool) or not isinstance(nominal_power, (int, float))
            or not math.isfinite(nominal_power) or nominal_power <= 0
            or isinstance(ridge, bool) or not isinstance(ridge, (int, float))
            or not math.isfinite(ridge) or ridge <= 0):
        raise ValueError('Finite estimated complex channels and positive fixed probe power/ridge required')
    batch, users, _, tones = estimated.shape
    variance = torch.as_tensor(ce_variance, device=estimated.device, dtype=estimated.real.dtype)
    if variance.shape == (1, users, tones): variance = variance.expand(batch, -1, -1)
    if (variance.shape != (batch, users, tones) or variance.requires_grad
            or not bool(torch.isfinite(variance).all() & (variance >= 0).all())):
        raise ValueError('Nonnegative independently calibrated per-RE CE variance required')
    supplied = torch.as_tensor(gamma_db)
    if supplied.is_complex() or supplied.dtype == torch.bool or supplied.requires_grad:
        raise ValueError('Public nonlearned real feedback gamma required')
    gamma = torch.as_tensor(gamma_db, device=estimated.device, dtype=estimated.real.dtype)
    if gamma.ndim == 0: gamma = gamma.expand(batch)
    if gamma.shape != (batch,) or not bool(torch.isfinite(gamma).all() & (gamma >= 0).all() & (gamma <= 25).all()):
        raise ValueError('One gamma in [0,25] per complete MU required')
    groups, supports = _layout(layout, users, tones)
    n0 = .25*torch.pow(gamma.new_tensor(10.), -gamma/10.)
    filters = []
    for pair, positions in zip(groups, supports):
        ui = torch.tensor(pair, device=estimated.device)
        ti = torch.tensor(positions, device=estimated.device)
        g = estimated.index_select(1, ui).index_select(-1, ti).permute(0, 3, 2, 1)
        v = variance.index_select(1, ui).index_select(-1, ti).transpose(1, 2)
        effective = n0[:, None]+nominal_power*v.sum(-1)
        gh = g.conj().transpose(-2, -1)
        eye = torch.eye(2, dtype=g.dtype, device=g.device)
        a = torch.linalg.solve(gh@g+(ridge+effective/nominal_power)[..., None, None]*eye, gh)
        if not bool(torch.isfinite(a).all()): raise FloatingPointError('Nonfinite public probe filter')
        filters.append(a)
    return filters, n0


def shared_uniform_prior_score(estimated, truth, ce_variance, gamma_db, layout, *, nominal_power=48./44.):
    """Offline conditional MSE of estimated-UL filters under uniform symbols.

    This is neither a measured codec symbol NMSE nor a downlink sum-rate. It
    assumes independent zero-mean symbols with equal nominal power48/44 and
    adds true thermal noise only after scoring the fixed estimated-UL filter.
    """
    import torch
    estimated, truth, _, _ = _channels(estimated, truth)
    filters, n0 = shared_filters(estimated, ce_variance, gamma_db, layout, nominal_power=nominal_power)
    groups, supports = _layout(layout, estimated.shape[1], estimated.shape[-1])
    errors, missing_ranks = [], []
    for a, pair, positions in zip(filters, groups, supports):
        ui = torch.tensor(pair, device=estimated.device)
        ti = torch.tensor(positions, device=estimated.device)
        g_true = truth.index_select(1, ui).index_select(-1, ti).permute(0, 3, 2, 1)
        g_est = estimated.index_select(1, ui).index_select(-1, ti).permute(0, 3, 2, 1)
        deviation = a@g_true-torch.eye(2, dtype=a.dtype, device=a.device)
        mse = nominal_power*deviation.abs().square().sum(-1)+n0[:, None, None]*a.abs().square().sum(-1)
        errors.append(mse.flatten(1))
        missing_ranks.append((torch.linalg.matrix_rank(g_est.detach()) < 2).to(mse.dtype))
    error = torch.cat(errors, 1).mean(1)
    result = dict(shared_uniform_prior_mse=error,
        shared_uniform_prior_scaled_error=error/nominal_power,
        shared_estimated_rank_deficient_fraction=torch.cat(missing_ranks, 1).mean(1))
    if any(not bool(torch.isfinite(v).all()) for v in result.values()):
        raise FloatingPointError('Nonfinite offline recovery probe score')
    return result
