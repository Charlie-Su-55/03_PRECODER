"""Standard-library calibration selection and paired alpha-control summaries.

One complete eight-user MU sample is the independent observation. Calibration
selects one alpha per physical arm across all six gamma conditions. This module
does not load a checkpoint, generate channels, or accept final results when
selecting alpha.
"""
from __future__ import annotations

import math
import statistics

import twc_v1_k8_icc_aligned_results as tables


CALIBRATION_FIELDS = frozenset((
    'arm', 'alpha', 'gamma_db', 'sample_index', 'batch_index', 'within_batch',
    'sum_rate_bps_hz',
))


def _config():
    from configs import twc_v1_k8_icc_alpha_config as cfg
    return cfg


def _counts(samples, batch_size):
    if (type(samples) is not int or samples < 1 or type(batch_size) is not int
            or batch_size < 1 or samples % batch_size):
        raise ValueError('Positive sample counts and complete physical batches required')


def _finite_number(value, name):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value)):
        raise ValueError('Finite numeric '+name+' required')
    return float(value)


def _sample_identity(row, batch_size):
    for name in ('sample_index', 'batch_index', 'within_batch'):
        if type(row[name]) is not int or row[name] < 0:
            raise ValueError('Nonnegative integer '+name+' required')
    sample = row['sample_index']
    if row['batch_index'] != sample // batch_size or row['within_batch'] != sample % batch_size:
        raise ValueError('Calibration sample/batch identity changed')
    return sample


def validate_calibration_rows(rows, *, samples, batch_size=4):
    """Reject missing/duplicate conditions, labels, alpha values and samples.

    The caller owns provenance and must supply only calibration-address rows.
    Exact numeric schemas cannot prove the original data role without its
    accompanying address/fingerprint record, so unknown metadata is rejected.
    """
    cfg = _config()
    _counts(samples, batch_size)
    grid = tuple(float(alpha) for alpha in cfg.ALPHA_GRID)
    gammas = tuple(float(gamma) for gamma in cfg.GAMMAS)
    if (not grid or len(set(grid)) != len(grid)
            or any(not math.isfinite(alpha) or alpha <= 0 for alpha in grid)):
        raise ValueError('Unique finite positive alpha grid required')
    if len(set(gammas)) != len(gammas) or not gammas:
        raise ValueError('Unique gamma conditions required')
    expected = {(arm, alpha, gamma, i) for arm in cfg.ARMS for alpha in grid
                for gamma in gammas for i in range(samples)}
    copied, index = [], {}
    for source in rows:
        row = dict(source)
        if set(row) != CALIBRATION_FIELDS:
            raise ValueError('Exact calibration-only metric schema required')
        alpha = _finite_number(row['alpha'], 'calibration alpha')
        gamma = _finite_number(row['gamma_db'], 'calibration gamma')
        rate = _finite_number(row['sum_rate_bps_hz'], 'calibration rate')
        if rate < 0:
            raise ValueError('Nonnegative calibration rate required')
        sample = _sample_identity(row, batch_size)
        key = (row['arm'], alpha, gamma, sample)
        if key not in expected:
            raise ValueError('Unexpected calibration arm/alpha/gamma/sample')
        if key in index:
            raise ValueError('Duplicate complete-MU calibration row')
        row['alpha'], row['gamma_db'] = alpha, gamma
        copied.append(row); index[key] = row
    if set(index) != expected:
        raise ValueError('Missing complete-MU calibration observations')
    return copied, index


def _paired_moments(values, batch_size):
    sample = tables.moments(values)
    batches = [statistics.mean(values[start:start+batch_size])
               for start in range(0, len(values), batch_size)]
    batch = tables.moments(batches)
    return dict(mean=sample['mean'], sample_se=sample['se'],
        sample_ci95_low=sample['ci95_low'], sample_ci95_high=sample['ci95_high'],
        batch_se=batch['se'], batch_ci95_low=batch['ci95_low'],
        batch_ci95_high=batch['ci95_high'])


def select_alphas(rows, *, samples, batch_size=4):
    """Select one fixed global alpha for each arm, using calibration rows only.

    Gamma conditions are averaged inside each complete MU before uncertainty.
    Exact mean ties prefer the original fixed alpha, then log-distance to it,
    then the smaller alpha. No final result or per-gamma choice is an input.
    """
    cfg = _config()
    _, index = validate_calibration_rows(rows, samples=samples, batch_size=batch_size)
    fixed = float(cfg.FIXED_ALPHA)
    if fixed not in tuple(float(x) for x in cfg.ALPHA_GRID):
        raise ValueError('Original fixed alpha must be included in calibration grid')
    scores, selected = [], {}
    for arm in cfg.ARMS:
        arm_scores = []
        for value in cfg.ALPHA_GRID:
            alpha = float(value)
            per_mu = [statistics.mean(index[arm,alpha,float(gamma),i]['sum_rate_bps_hz']
                                      for gamma in cfg.GAMMAS) for i in range(samples)]
            score = dict(arm=arm, alpha=alpha, n_samples=samples,
                n_batches=samples//batch_size, batch_size=batch_size,
                **_paired_moments(per_mu, batch_size))
            scores.append(score); arm_scores.append(score)
        best_mean = max(row['mean'] for row in arm_scores)
        ties = [row['alpha'] for row in arm_scores if row['mean'] == best_mean]
        selected[arm] = min(ties, key=lambda alpha: (
            alpha != fixed, abs(math.log(alpha/fixed)), alpha))
    return dict(selected=selected, scores=scores, n_samples=samples,
        n_batches=samples//batch_size, batch_size=batch_size,
        criterion='maximum calibration mean complete-MU sum-rate over all six gamma',
        selection_scope='one global alpha per physical arm; not per gamma',
        tie_rule='exact mean ties: fixed alpha, closest log-distance, lower alpha',
        fixed_alpha=fixed, gamma_db=list(cfg.GAMMAS),
        alpha_grid=list(cfg.ALPHA_GRID), test_used=False)


def validate_final_rows(rows, *, samples, batch_size=4):
    """Validate all six policy/arm methods and consistent optional diagnostics."""
    cfg = _config()
    copied, index = tables.validate_rows(rows, samples=samples, batch_size=batch_size,
                                        arms=cfg.METHODS, gammas=cfg.GAMMAS)
    for method in cfg.METHODS:
        for gamma in cfg.GAMMAS:
            group = [index[method,float(gamma),i] for i in range(samples)]
            metrics = set().union(*(set(row)-tables.ID_FIELDS for row in group))
            for metric in metrics:
                present = [row.get(metric) for row in group]
                if any(value is None for value in present) and not all(value is None for value in present):
                    raise ValueError('Partially missing final diagnostic: '+metric)
    return copied, index


def summarize_rows(rows, *, samples, batch_size=4):
    cfg = _config()
    copied, _ = validate_final_rows(rows, samples=samples, batch_size=batch_size)
    return tables.summarize_rows(copied, samples=samples, batch_size=batch_size, arms=cfg.METHODS)


def _contrasts(cfg):
    method = cfg.method_name
    contrasts = {}
    for policy in cfg.POLICIES:
        contrasts[policy+'_hybrid_minus_fdma'] = (
            (-1., method(policy,'fdma')), (1., method(policy,'hybrid_g4')))
    for arm in cfg.ARMS:
        contrasts[arm+'_source_calibrated_minus_source_fixed'] = (
            (-1., method('source_fixed',arm)), (1., method('source_calibrated',arm)))
        contrasts[arm+'_rate_fixed_minus_source_fixed'] = (
            (-1., method('source_fixed',arm)), (1., method('rate_fixed',arm)))
        contrasts[arm+'_rate_fixed_minus_source_calibrated'] = (
            (-1., method('source_calibrated',arm)), (1., method('rate_fixed',arm)))
    contrasts['interaction_rate_gain_minus_calibration_gain'] = (
        (-1., method('rate_fixed','fdma')), (1., method('rate_fixed','hybrid_g4')),
        (1., method('source_calibrated','fdma')), (-1., method('source_calibrated','hybrid_g4')))
    return contrasts


def paired_comparisons(rows, *, samples, batch_size=4):
    """Compare six methods within MU, before averaging gamma or batch4."""
    cfg = _config()
    _, index = validate_final_rows(rows, samples=samples, batch_size=batch_size)
    gammas = tuple(float(gamma) for gamma in cfg.GAMMAS)
    scopes = [('point', (gamma,)) for gamma in gammas]
    scopes += [('low_0_5_10', gammas[:3]), ('all_six', gammas)]
    output = []
    for scope, selected in scopes:
        for label, terms in _contrasts(cfg).items():
            common = set.intersection(*(set(index[method,gamma,0])-tables.ID_FIELDS
                                        for _,method in terms for gamma in selected))
            for metric in sorted(common):
                if any(index[method,gamma,i].get(metric) is None
                       for _,method in terms for gamma in selected for i in range(samples)):
                    continue
                deltas = [statistics.mean(sum(coef*index[method,gamma,i][metric]
                                             for coef,method in terms)
                                          for gamma in selected) for i in range(samples)]
                output.append(dict(comparison=label, metric=metric, scope=scope,
                    gamma_db=selected[0] if scope=='point' else None,
                    n_samples=samples, n_batches=samples//batch_size, batch_size=batch_size,
                    **_paired_moments(deltas,batch_size)))
    return output
