"""Calibration-only CE moments and deployable frozen UL receiver variants.

Truth is accepted only by ``calibration_statistics``. Prediction and recovery
accept measured pilots, estimated channels and a locked calibration model.
The in-sample pilot residual is a feature, never an unbiased error label.
"""
from __future__ import annotations

import json
import math

import torch


def _cfg():
    from configs import twc_v1_k8_ul_calibration_config as cfg
    return cfg


def _physics():
    from train_twc_v1_k8_icc_aligned import physics
    return physics()


def _finite(value, name, *, complex_value=None):
    if (not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
            or complex_value is not None and value.is_complex() != complex_value):
        raise ValueError('Finite '+name+' tensor required')
    return value.detach()


def _channels(estimated, ce_variance):
    estimated = _finite(estimated, 'estimated channel', complex_value=True)
    if estimated.ndim != 4 or estimated.shape[2] < 2 or estimated.dtype != torch.complex64:
        raise ValueError('Complex64 estimated channel [B,K,Nt,Nsc], Nt>=2 required')
    batch, users, _, tones = estimated.shape
    variance = _finite(ce_variance, 'source CE variance', complex_value=False)
    if variance.shape == (1, users, tones):
        variance = variance.expand(batch, -1, -1)
    if (variance.shape != (batch, users, tones) or variance.device != estimated.device
            or not variance.is_floating_point() or not bool((variance >= 0).all())):
        raise ValueError('Matching nonnegative source CE variance required')
    return estimated, variance


def _gamma(gamma, batch, device):
    value = _physics().gamma_vector(gamma, batch, device)
    nodes = value.new_tensor(_cfg().GAMMAS)
    if not bool((value[:, None] == nodes[None]).any(-1).all()):
        raise ValueError('Calibration supports only the six prescribed gamma nodes')
    return value


@torch.no_grad()
def pilot_feature(Y, pilots, estimated, gamma):
    """Known-pilot-only in-sample residual energy divided by public N0."""
    if (not torch.is_tensor(Y) or Y.ndim != 3 or Y.dtype != torch.complex64
            or not torch.is_tensor(pilots) or pilots.ndim != 3 or pilots.dtype != Y.dtype
            or not torch.is_tensor(estimated) or estimated.ndim != 4 or estimated.dtype != Y.dtype):
        raise ValueError('Complex64 public Y, pilots and estimated channel required')
    batch, antennas, tones = Y.shape
    users = pilots.shape[1]
    if (pilots.shape != (batch, users, tones)
            or estimated.shape != (batch, users, antennas, tones)
            or Y.device != pilots.device or Y.device != estimated.device):
        raise ValueError('Matching pilot-feature shapes and devices required')
    _finite(pilots, 'known pilots', complex_value=True)
    union = pilots.abs().ne(0).any(1)
    if not bool(union.any(1).all()) or not bool((union == union[:1]).all()):
        raise ValueError('A common nonempty paid pilot support is required')
    selected = union[0]
    # Mask before finite checks: data-only observations cannot enter this feature.
    received = _finite(Y[..., selected], 'pilot-only received observation', complex_value=True)
    channel = _finite(estimated[..., selected], 'pilot-only estimated channel', complex_value=True)
    known = pilots[..., selected]
    predicted = (channel*known[:, :, None]).sum(1)
    n0 = _physics().noise_variance(gamma, batch, Y.device)
    return ((received-predicted).abs().square().mean((1, 2))/n0).detach()


@torch.no_grad()
def calibration_statistics(estimated, true_ul, ce_variance, feature, gamma_db):
    """Per-MU sufficient statistics; true UL provides calibration labels only."""
    cfg = _cfg()
    estimated, variance = _channels(estimated, ce_variance)
    truth = _finite(true_ul, 'calibration label', complex_value=True)
    if truth.shape != estimated.shape or truth.dtype != estimated.dtype or truth.device != estimated.device:
        raise ValueError('Matching calibration label channel required')
    batch, users, antennas, tones = estimated.shape
    feature = _finite(feature, 'pilot feature', complex_value=False)
    if (feature.shape != (batch,) or not feature.is_floating_point()
            or feature.device != estimated.device or not bool((feature >= 0).all())):
        raise ValueError('One nonnegative feature per calibration MU required')
    gamma = _gamma(gamma_db, batch, estimated.device)
    error = truth-estimated
    h2 = estimated.abs().square().sum(2)
    he = (estimated.conj()*error).sum(2)
    parallel = he.abs().square()/h2.clamp_min(cfg.VARIANCE_FLOOR)
    perpendicular = (error.abs().square().sum(2)-parallel).clamp_min(0)/(antennas-1)
    scale = variance.clamp_min(cfg.VARIANCE_FLOOR)
    tensors = dict(h2_sum=h2, he_real_sum=he.real, he_imag_sum=he.imag,
        parallel_scaled_sum=parallel/scale,
        he_scaled_real_sum=he.real/scale, he_scaled_imag_sum=he.imag/scale,
        h2_scaled_sum=h2/scale, perp_scaled_sum=perpendicular/scale)
    values = {name: value.double().sum((1, 2)).cpu().tolist() for name, value in tensors.items()}
    return [dict(role='calibration', gamma_db=float(gamma[i]), feature=float(feature[i]),
        n_coordinates=users*tones, **{name: values[name][i] for name in values}) for i in range(batch)]


STAT_FIELDS = frozenset(('role', 'gamma_db', 'feature', 'n_coordinates', 'h2_sum',
    'he_real_sum', 'he_imag_sum', 'parallel_scaled_sum', 'he_scaled_real_sum',
    'he_scaled_imag_sum', 'h2_scaled_sum', 'perp_scaled_sum'))
OPTIONAL_IDS = frozenset(('batch_index', 'within_batch', 'sample_index'))
FEATURE_DEFINITION = 'known-pilot in-sample residual energy / N0; not an unbiased CE-error measurement'


def _number(value, *, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or nonnegative and value < 0):
        raise ValueError('Finite calibration scalar required')
    return value


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])


def fit_model(records):
    """Fit one common six-node model exclusively from calibration MU labels.

    Fixed per-gamma feature tertiles and a16-MU identity shrinkage are specified
    before validation. Empty tied-feature strata retain identity moments.
    """
    cfg = _cfg()
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('Nonempty calibration sufficient-statistic records required')
    groups = {float(gamma): [] for gamma in cfg.GAMMAS}
    coordinates, seen = None, set()
    for row in records:
        if (not isinstance(row, dict) or not STAT_FIELDS <= set(row)
                or set(row)-STAT_FIELDS-OPTIONAL_IDS or row['role'] != 'calibration'):
            raise ValueError('Only calibration-role sufficient statistics are accepted')
        if row['gamma_db'] not in groups:
            raise ValueError('All prescribed calibration gamma nodes required')
        for key in STAT_FIELDS-{'role', 'n_coordinates'}:
            _number(row[key], nonnegative=key not in
                ('he_real_sum', 'he_imag_sum', 'he_scaled_real_sum', 'he_scaled_imag_sum'))
        count = row['n_coordinates']
        if type(count) is not int or count < 1 or row['h2_sum'] <= 0 or row['h2_scaled_sum'] <= 0:
            raise ValueError('Positive calibration coordinate count/channel energy required')
        if coordinates is None:
            coordinates = count
        if count != coordinates:
            raise ValueError('Calibration coordinate geometry changed')
        for key in OPTIONAL_IDS & set(row):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('Nonnegative calibration MU identity required')
        if 'sample_index' in row:
            identity = (row['gamma_db'], row['sample_index'])
            if identity in seen:
                raise ValueError('Duplicate calibration MU identity')
            seen.add(identity)
        groups[row['gamma_db']].append(row)
    counts = {len(group) for group in groups.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError('Equal complete calibration samples at all six nodes required')
    nodes = []
    for gamma, group in groups.items():
        thresholds = [_quantile([row['feature'] for row in group], q) for q in (1/3, 2/3)]
        cells = []
        for index in range(3):
            chosen = [row for row in group if sum(row['feature'] > cut for cut in thresholds) == index]
            nmu = len(chosen)
            ncoords = nmu*coordinates
            alpha, parallel, perpendicular = 0j, 1., 1.
            if chosen:
                weight = nmu/(nmu+cfg.MODEL_SHRINK_MU)
                he = complex(sum(row['he_real_sum'] for row in chosen), sum(row['he_imag_sum'] for row in chosen))
                alpha = weight*he/sum(row['h2_sum'] for row in chosen)
                scaled_he = complex(sum(row['he_scaled_real_sum'] for row in chosen),
                    sum(row['he_scaled_imag_sum'] for row in chosen))
                residual_parallel = (sum(row['parallel_scaled_sum'] for row in chosen)
                    -2*(alpha.conjugate()*scaled_he).real
                    +abs(alpha)**2*sum(row['h2_scaled_sum'] for row in chosen))
                parallel = max(cfg.PSD_FLOOR, (1-weight)+weight*max(0., residual_parallel/ncoords))
                perpendicular = max(cfg.PSD_FLOOR, (1-weight)+weight*
                    sum(row['perp_scaled_sum'] for row in chosen)/ncoords)
            cells.append(dict(n_mu=nmu, n_coordinates=ncoords, alpha_real=alpha.real,
                alpha_imag=alpha.imag, parallel=parallel, perpendicular=perpendicular))
        nodes.append(dict(gamma_db=gamma, thresholds=thresholds, strata=cells))
    model = dict(format=cfg.NAMESPACE+'_moments_v1', calibration_role='calibration',
        gammas=list(cfg.GAMMAS), feature_definition=FEATURE_DEFINITION,
        shrink_mu=cfg.MODEL_SHRINK_MU, variance_floor=cfg.VARIANCE_FLOOR,
        psd_floor=cfg.PSD_FLOOR, calibration_samples_per_gamma=next(iter(counts)),
        coordinates_per_mu=coordinates, nodes=nodes)
    return validate_model(model)


def validate_model(model):
    cfg = _cfg()
    expected = {'format', 'calibration_role', 'gammas', 'feature_definition', 'shrink_mu',
        'variance_floor', 'psd_floor', 'calibration_samples_per_gamma', 'coordinates_per_mu', 'nodes'}
    fixed = dict(format=cfg.NAMESPACE+'_moments_v1', calibration_role='calibration',
        gammas=list(cfg.GAMMAS), feature_definition=FEATURE_DEFINITION,
        shrink_mu=cfg.MODEL_SHRINK_MU, variance_floor=cfg.VARIANCE_FLOOR, psd_floor=cfg.PSD_FLOOR)
    if not isinstance(model, dict) or set(model) != expected or any(model[k] != v for k, v in fixed.items()):
        raise ValueError('Locked calibration model identity changed')
    if (not isinstance(model['gammas'], list)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in model['gammas'])
            or any(isinstance(model[key], bool) or not isinstance(model[key], (int, float))
                   for key in ('shrink_mu', 'variance_floor', 'psd_floor'))):
        raise ValueError('Numeric locked calibration identity required')
    for key in ('calibration_samples_per_gamma', 'coordinates_per_mu'):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError('Positive calibration model sample/coordinate count required')
    if not isinstance(model['nodes'], list) or len(model['nodes']) != len(cfg.GAMMAS):
        raise ValueError('Exactly six locked calibration nodes required')
    for gamma, node in zip(cfg.GAMMAS, model['nodes']):
        if (not isinstance(node, dict) or set(node) != {'gamma_db', 'thresholds', 'strata'}
                or isinstance(node['gamma_db'], bool) or not isinstance(node['gamma_db'], (int, float))
                or node['gamma_db'] != gamma):
            raise ValueError('Calibration node identity changed')
        if not isinstance(node['thresholds'], list) or len(node['thresholds']) != 2:
            raise ValueError('Two calibration-only feature cutpoints required')
        for value in node['thresholds']:
            _number(value, nonnegative=True)
        if node['thresholds'] != sorted(node['thresholds']):
            raise ValueError('Ordered calibration-only feature cutpoints required')
        if not isinstance(node['strata'], list) or len(node['strata']) != 3:
            raise ValueError('Exactly three locked calibration strata required')
        count = 0
        for cell in node['strata']:
            if not isinstance(cell, dict) or set(cell) != {'n_mu', 'n_coordinates', 'alpha_real', 'alpha_imag', 'parallel', 'perpendicular'}:
                raise ValueError('Calibration cell schema changed')
            if (type(cell['n_mu']) is not int or cell['n_mu'] < 0
                    or type(cell['n_coordinates']) is not int
                    or cell['n_coordinates'] != cell['n_mu']*model['coordinates_per_mu']):
                raise ValueError('Calibration cell counts changed')
            for key in ('alpha_real', 'alpha_imag', 'parallel', 'perpendicular'):
                _number(cell[key])
            if min(cell['parallel'], cell['perpendicular']) < cfg.PSD_FLOOR:
                raise ValueError('Positive-semidefinite calibration moments required')
            if not cell['n_mu'] and (cell['alpha_real'] != 0 or cell['alpha_imag'] != 0
                    or cell['parallel'] != 1 or cell['perpendicular'] != 1):
                raise ValueError('Empty calibration strata must retain identity moments')
            count += cell['n_mu']
        if count != model['calibration_samples_per_gamma']:
            raise ValueError('Calibration sample count changed')
    return json.loads(json.dumps(model, allow_nan=False))


@torch.no_grad()
def predict_moments(estimated, ce_variance, feature, gamma, model):
    """Predict CE moments from public inputs and frozen calibration only."""
    model = validate_model(model)
    estimated, _ = _channels(estimated, ce_variance)
    batch = estimated.shape[0]
    feature = _finite(feature, 'pilot feature', complex_value=False)
    if (feature.shape != (batch,) or not feature.is_floating_point()
            or feature.device != estimated.device or not bool((feature >= 0).all())):
        raise ValueError('One nonnegative public feature per MU required')
    if model['coordinates_per_mu'] != estimated.shape[1]*estimated.shape[-1]:
        raise ValueError('Calibration channel coordinate geometry changed')
    gamma = _gamma(gamma, batch, estimated.device)
    alpha, parallel, perpendicular, cells = [], [], [], []
    for value, residual in zip(gamma.cpu().tolist(), feature.cpu().tolist()):
        node = model['nodes'][list(model['gammas']).index(value)]
        index = sum(residual > cut for cut in node['thresholds'])
        cell = node['strata'][index]
        alpha.append(complex(cell['alpha_real'], cell['alpha_imag']))
        parallel.append(cell['parallel'])
        perpendicular.append(cell['perpendicular'])
        cells.append(index)
    alpha = estimated.new_tensor(alpha)
    return dict(mean=(1+alpha[:, None, None, None])*estimated,
        parallel=estimated.real.new_tensor(parallel),
        perpendicular=estimated.real.new_tensor(perpendicular),
        cell=torch.tensor(cells, dtype=torch.long, device=estimated.device))


def _block_variance(variance, block):
    batch = variance.shape[0]
    if block['kind'] == 'exclusive':
        users, d = block['tones'].shape
        return variance.gather(-1, block['tones'][None].expand(batch, -1, -1)).reshape(batch, users*d, 1)
    return variance.index_select(1, block['users']).index_select(-1, block['tones']).transpose(1, 2)


def _error_covariance(g, variance, power, n0, parallel, perpendicular):
    antennas = g.shape[-2]
    unit = g/g.abs().square().sum(-2, keepdim=True).clamp_min(_cfg().VARIANCE_FLOOR).sqrt()
    weighted = power[None]*variance
    isotropic = n0[:, None]+(weighted*perpendicular[:, None, None]).sum(-1)
    directional = weighted*(parallel-perpendicular)[:, None, None]
    eye = torch.eye(antennas, dtype=g.dtype, device=g.device)
    covariance = isotropic[..., None, None]*eye + (unit*directional[..., None, :])@unit.mH
    return (covariance+covariance.mH)/2


def _model_variance(a, mean, power, covariance):
    deviation = a@mean-torch.eye(mean.shape[-1], dtype=mean.dtype, device=mean.device)
    bias = (deviation.abs().square()*power[None, :, None]).sum(-1)
    disturbance = ((a@covariance)*a.conj()).sum(-1).real
    return (bias+disturbance).clamp_min(_cfg().VARIANCE_FLOOR)


def _attach(original, active, variance, mean, filters, *, changed_receiver):
    batch, users, dimension = variance.shape
    result = dict(original)
    reliability = original['reliability'].clone()
    reliability.zero_()
    reliability[..., :dimension] = variance.reciprocal()
    result.update(variance=variance, reliability=reliability, filters=filters)
    if changed_receiver:
        soft = original['soft'].clone()
        soft[..., :dimension] = active
        result.update(soft=soft, z_hat=soft, H_ul_est=mean,
            key=active[..., :original['key'].shape[-1]], shared=active[..., original['key'].shape[-1]:])
        result.pop('key_effective_lambda', None)
        result.pop('shared_effective_lambda', None)
    return result


@torch.no_grad()
def recover_variants(observation, layout, gamma, model):
    """Three receivers with no true-channel or current-frame-power input.

    The moment receiver preserves the old RX_RIDGE through a trace-scaled
    precision term; isotropic identity calibration reduces to the old solve.
    """
    from twc_v1_k8_ul_propagation_metrics import _blocks, _gather_block, _scatter
    if not isinstance(observation, dict) or set(observation) != {'Y', 'estimated', 'ce_variance', 'prior', 'pilots'}:
        raise ValueError('Only public Y/estimated/CE variance/prior/pilots may enter recovery')
    phy = _physics()
    # A public-key whitelist alone cannot distinguish known pilots from an
    # accidentally supplied transmitted data waveform under the same key.
    phy.fourier_design(observation['pilots'])
    H, variance = _channels(observation['estimated'], observation['ce_variance'])
    Y, prior = observation['Y'], observation['prior']
    original = phy.recover(Y, H, layout, prior, variance, gamma)
    feature = pilot_feature(Y, observation['pilots'], H, gamma)
    moments = predict_moments(H, variance, feature, gamma, model)
    blocks, filters, _ = _blocks(observation, original, layout)
    original = dict(original, filters=filters)
    actual = torch.zeros_like(original['variance'], dtype=Y.dtype)
    fixed_variance = torch.zeros_like(original['variance'])
    changed_variance = torch.zeros_like(original['variance'])
    changed_filters = {'exclusive': None, 'shared': []}
    n0 = phy.noise_variance(gamma, H.shape[0], H.device)
    for block in blocks:
        mean = _gather_block(moments['mean'], block, channel=True)
        uncertainty = _block_variance(variance, block)
        covariance = _error_covariance(block['G'], uncertainty, block['P'], n0,
            moments['parallel'], moments['perpendicular'])
        fixed = _model_variance(block['A'], mean, block['P'], covariance)
        precision_mean = torch.linalg.solve(covariance, mean)
        adjoint = precision_mean.mH
        rbar = covariance.diagonal(dim1=-2, dim2=-1).real.mean(-1)
        diagonal = block['P'].reciprocal()[None]+phy.RX_RIDGE/rbar[..., None]
        a = torch.linalg.solve(adjoint@mean+torch.diag_embed(diagonal).to(mean.dtype), adjoint)
        corrected = (a@block['Y'][..., None]).squeeze(-1)
        changed = _model_variance(a, mean, block['P'], covariance)
        _scatter(actual, corrected, block)
        _scatter(fixed_variance, fixed, block)
        _scatter(changed_variance, changed, block)
        if block['kind'] == 'exclusive':
            users, d = block['tones'].shape
            changed_filters['exclusive'] = a.reshape(H.shape[0], users, d, H.shape[2])
        else:
            changed_filters['shared'].append(a)
    changed_filters['shared'] = (torch.stack(changed_filters['shared'], 1)
        if changed_filters['shared'] else filters['shared'])
    for value in (actual, fixed_variance, changed_variance):
        _finite(value, 'calibrated recovery')
    return dict(original=original,
        reliability_only=_attach(original, None, fixed_variance, H, filters, changed_receiver=False),
        moment_receiver=_attach(original, actual, changed_variance, moments['mean'], changed_filters,
            changed_receiver=True))
