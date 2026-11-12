"""Paired, CSV-only statistics for the constrained residual-W control.

One observation is one complete eight-user MU channel realization. Replaying
that observation through six policies and six gamma conditions does not add
independent samples. This module loads no weights and imports no ML runtime.
"""
from __future__ import annotations

import statistics

import twc_v1_k8_icc_aligned_results as tables


def _config():
    from configs import twc_v1_k8_w_residual_config as cfg
    return cfg


def validate_final_rows(rows, *, samples, batch_size=4):
    """Require all six methods and complete MU/batch identities.

    Optional diagnostics may be entirely absent for a method/gamma condition,
    such as Hybrid-only shared feedback diagnostics in FDMA. A partly missing
    condition is an error. Required physical metrics cannot all be blank.
    """
    cfg = _config()
    copied, index = tables.validate_rows(rows, samples=samples,
        batch_size=batch_size, arms=cfg.METHODS, gammas=cfg.GAMMAS)
    for row in copied:
        for metric in tables.REQUIRED_METRICS:
            if row[metric] is None:
                raise ValueError('Missing required final metric: '+metric)
    for method in cfg.METHODS:
        for gamma in cfg.GAMMAS:
            group = [index[method,float(gamma),i] for i in range(samples)]
            metrics = set().union(*(set(row)-tables.ID_FIELDS for row in group))
            for metric in metrics:
                values = [row.get(metric) for row in group]
                if any(value is None for value in values) and not all(value is None for value in values):
                    raise ValueError('Partially missing final diagnostic: '+metric)
    return copied, index


def summarize_rows(rows, *, samples, batch_size=4):
    """Retain every available scalar diagnostic and linear-mean NMSE dB."""
    cfg = _config()
    copied, _ = validate_final_rows(rows, samples=samples, batch_size=batch_size)
    return tables.summarize_rows(copied, samples=samples,
                                batch_size=batch_size, arms=cfg.METHODS)


def _contrasts(cfg):
    method = cfg.method_name
    contrasts = {}
    # Primary architecture comparison: the two equally continued rate arms.
    for arm in cfg.ARMS:
        contrasts[arm+'_w_residual_minus_rzf_continue'] = (
            (-1., method('rzf_continue',arm)),
            (1., method('w_residual',arm)))
    contrasts['interaction_w_residual_minus_rzf_continue'] = (
        (-1., method('w_residual','fdma')),
        (1., method('w_residual','hybrid_g4')),
        (1., method('rzf_continue','fdma')),
        (-1., method('rzf_continue','hybrid_g4')))
    # Secondary checks retain both gains from the frozen rate-source baseline.
    for arm in cfg.ARMS:
        contrasts[arm+'_rzf_continue_minus_rate_source'] = (
            (-1., method('rate_source',arm)),
            (1., method('rzf_continue',arm)))
        contrasts[arm+'_w_residual_minus_rate_source'] = (
            (-1., method('rate_source',arm)),
            (1., method('w_residual',arm)))
    for policy in cfg.POLICIES:
        contrasts[policy+'_hybrid_minus_fdma'] = (
            (-1., method(policy,'fdma')),
            (1., method(policy,'hybrid_g4')))
    return contrasts


def _paired_moments(values, batch_size):
    sample = tables.moments(values)
    batches = [statistics.mean(values[start:start+batch_size])
               for start in range(0,len(values),batch_size)]
    batch = tables.moments(batches)
    return dict(mean=sample['mean'], sample_se=sample['se'],
        sample_ci95_low=sample['ci95_low'],
        sample_ci95_high=sample['ci95_high'], batch_se=batch['se'],
        batch_ci95_low=batch['ci95_low'],
        batch_ci95_high=batch['ci95_high'])


def paired_comparisons(rows, *, samples, batch_size=4):
    """Contrast methods within MU, average gamma within MU, then batch MU.

    Intervals condition on the fixed trained weights. They do not estimate
    uncertainty over training seeds or architecture selection. All signed
    results and all six gamma points are retained; no winner is selected here.
    """
    cfg = _config()
    _, index = validate_final_rows(rows, samples=samples, batch_size=batch_size)
    gammas = tuple(float(gamma) for gamma in cfg.GAMMAS)
    scopes = [('point',(gamma,)) for gamma in gammas]
    scopes += [('low_0_5_10',gammas[:3]),('all_six',gammas)]
    output = []
    for scope, selected in scopes:
        for label, terms in _contrasts(cfg).items():
            common = set.intersection(*(set(index[method,gamma,0])-tables.ID_FIELDS
                                        for _,method in terms for gamma in selected))
            for metric in sorted(common):
                if any(index[method,gamma,i].get(metric) is None
                       for _,method in terms for gamma in selected for i in range(samples)):
                    continue
                values = [statistics.mean(sum(coefficient*index[method,gamma,i][metric]
                                              for coefficient,method in terms)
                                          for gamma in selected) for i in range(samples)]
                output.append(dict(comparison=label, metric=metric, scope=scope,
                    gamma_db=selected[0] if scope=='point' else None,
                    n_samples=samples, n_batches=samples//batch_size,
                    batch_size=batch_size, **_paired_moments(values,batch_size)))
    return output
