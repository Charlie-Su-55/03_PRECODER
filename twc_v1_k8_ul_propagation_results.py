"""Complete-MU statistics for fixed-filter conditional error propagation.

No receiver is changed here. A row contains actual, reported and privileged
conditional MSE for one normal receiver and one complete eight-user MU.
Gamma points are repeated views: low/all-gamma summaries average within MU
before either MU or four-MU batch uncertainty is calculated. Calibration
coordinates remain descriptive counts, never independent CI observations.

This module imports no Torch/Sionna and performs no model or network access.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import math
from pathlib import Path
import statistics

import twc_v1_k8_ul_diagnostics_results as previous


ID_FIELDS = frozenset(('arm', 'gamma_db', 'sample_index', 'batch_index', 'within_batch'))
REQUIRED_METRICS = ('soft_nmse_linear', 'soft_nominal_scaled_actual_mse',
                    'soft_nominal_scaled_reported_variance',
                    'soft_nominal_scaled_propagated_variance')
COMPONENTS = ('soft', 'soft_key', 'soft_shared')
CONTRASTS = (
    ('propagated_minus_reported_variance', 'propagated_variance', 'reported_variance'),
    ('actual_minus_reported_variance', 'actual_mse', 'reported_variance'),
    ('actual_minus_propagated_variance', 'actual_mse', 'propagated_variance'),
)


def _config():
    from configs import twc_v1_k8_ul_propagation_config as cfg
    return cfg


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name+' must be a nonnegative integer')
    return value


def _number(value, name, *, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or nonnegative and value < 0):
        raise ValueError('Finite '+('nonnegative ' if nonnegative else '')+'numeric '+name+' required')
    return value


def moments(values):
    """Normal 95% interval; one observation does not identify a standard error."""
    values = list(values)
    if not values:
        raise ValueError('Nonempty observations required')
    for value in values:
        _number(value, 'observation')
    mean = statistics.mean(values)
    se = statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, se=se,
        ci95_low=None if se is None else mean-1.96*se,
        ci95_high=None if se is None else mean+1.96*se)


def _nonnegative_metric(name):
    return (name.endswith('_nmse_linear') or name.endswith('_actual_mse')
            or name.endswith('_variance') or name.endswith('_mse_ratio')
            or name.endswith('_power'))


def validate_rows(rows, *, samples, batch_size=4):
    """Require every declared arm/gamma/MU and complete physical batch identity."""
    cfg = _config()
    if (type(samples) is not int or samples < 1 or type(batch_size) is not int
            or batch_size < 1 or samples % batch_size):
        raise ValueError('Positive counts of complete physical microbatches required')
    copied = [dict(row) for row in rows]
    expected = {(arm, float(gamma), i) for arm in cfg.ARMS
                for gamma in cfg.GAMMAS for i in range(samples)}
    groups, index = defaultdict(list), {}
    for row in copied:
        if not ID_FIELDS.issubset(row) or not set(REQUIRED_METRICS).issubset(row):
            raise ValueError('Incomplete per-MU propagation row')
        arm = row['arm']
        if not isinstance(arm, str) or arm not in cfg.ARMS:
            raise ValueError('Declared physical arm required')
        gamma = float(_number(row['gamma_db'], 'gamma'))
        i = _integer(row['sample_index'], 'sample_index')
        batch = _integer(row['batch_index'], 'batch_index')
        within = _integer(row['within_batch'], 'within_batch')
        key = (arm, gamma, i)
        if key not in expected or batch != i//batch_size or within != i % batch_size:
            raise ValueError('Unexpected propagation condition or MU/batch identity')
        if key in index:
            raise ValueError('Duplicate complete-MU propagation key: '+repr(key))
        for field, value in row.items():
            if not isinstance(field, str) or not field:
                raise ValueError('Nonempty string column names required')
            if field in ID_FIELDS or value is None:
                continue
            _number(value, field, nonnegative=_nonnegative_metric(field))
        if any(row[name] is None for name in REQUIRED_METRICS):
            raise ValueError('Missing required propagation metric')
        index[key] = row
        groups[key[:2]].append(row)
    if set(index) != expected:
        raise ValueError('Missing complete-MU propagation observations')
    for group in groups.values():
        metrics = set().union(*(set(row) for row in group))-ID_FIELDS
        for metric in metrics:
            present = [row.get(metric) is not None for row in group]
            if any(present) and not all(present):
                raise ValueError('Partially missing metric within one propagation condition: '+metric)
    return copied, index


def _scopes(gammas):
    gammas = tuple(float(gamma) for gamma in gammas)
    return ([('point', (gamma,)) for gamma in gammas]
            + [('low_0_5_10', gammas[:3]), ('all_six', gammas)])


def _batch_moments(values, batch_size):
    return moments(statistics.mean(values[first:first+batch_size])
                   for first in range(0, len(values), batch_size))


def summarize_rows(rows, *, samples, batch_size=4):
    """Point and MU-first gamma summaries, retaining signed decomposition terms.

    Ratio fields are means of the already computed per-MU ratios. They are
    not ratios of pooled error/variance means; calibration bins report that
    separately. Only explicitly named linear NMSE fields get a dB view.
    """
    cfg = _config()
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for arm in cfg.ARMS:
        for scope, selected in _scopes(cfg.GAMMAS):
            common = set.intersection(*(set(index[arm, gamma, 0])-ID_FIELDS for gamma in selected))
            result = dict(arm=arm, scope=scope,
                gamma_db=selected[0] if scope == 'point' else None,
                n_samples=samples, n_batches=samples//batch_size,
                batch_size=batch_size, n_gamma_points=len(selected))
            for metric in sorted(common):
                if any(index[arm, gamma, i].get(metric) is None
                       for gamma in selected for i in range(samples)):
                    continue
                values = [statistics.mean(index[arm, gamma, i][metric] for gamma in selected)
                          for i in range(samples)]
                sample, batch = moments(values), _batch_moments(values, batch_size)
                for suffix in ('mean', 'se', 'ci95_low', 'ci95_high'):
                    result[metric+'_'+suffix] = sample[suffix]
                for suffix in ('se', 'ci95_low', 'ci95_high'):
                    result[metric+'_batch_'+suffix] = batch[suffix]
                if metric.endswith('_nmse_linear'):
                    result[metric[:-len('_linear')]+'_db_from_linear_mean'] = (
                        None if sample['mean'] == 0 else 10*math.log10(sample['mean']))
            output.append(result)
    return output


def paired_comparisons(rows, *, samples, batch_size=4):
    """Within-receiver MSE differences, never a competing-method rate ranking."""
    cfg = _config()
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for arm in cfg.ARMS:
        for scope, selected in _scopes(cfg.GAMMAS):
            for component in COMPONENTS:
                for contrast, positive, negative in CONTRASTS:
                    fields = tuple(component+'_nominal_scaled_'+part for part in (positive, negative))
                    if any(index[arm, gamma, i].get(field) is None
                           for gamma in selected for i in range(samples) for field in fields):
                        continue
                    values = [statistics.mean(index[arm, gamma, i][fields[0]]
                                              -index[arm, gamma, i][fields[1]]
                                              for gamma in selected) for i in range(samples)]
                    sample, batch = moments(values), _batch_moments(values, batch_size)
                    output.append(dict(arm=arm, component=component, contrast=contrast,
                        comparison=arm+'_'+component+'_'+contrast, metric='nominal_scaled_mse',
                        scope=scope, gamma_db=selected[0] if scope == 'point' else None,
                        n_samples=samples, n_batches=samples//batch_size,
                        batch_size=batch_size, n_gamma_points=len(selected), mean=sample['mean'],
                        sample_se=sample['se'], sample_ci95_low=sample['ci95_low'],
                        sample_ci95_high=sample['ci95_high'], batch_se=batch['se'],
                        batch_ci95_low=batch['ci95_low'], batch_ci95_high=batch['ci95_high']))
    return output


def aggregate_calibration_bins(rows):
    """Pool descriptive coordinate sums, with no coordinate-level uncertainty.

    Each input uses arm/gamma/component/bin_index/lower/upper/count/
    sum_actual/sum_predicted, optionally batch_index. Name components with
    reported_/propagated_ prefixes to keep the two predictions separate.
    """
    cfg = _config()
    prepared = []
    for original in rows:
        row = dict(original)
        if 'variant' in row:
            raise ValueError('Propagation bins have one fixed normal receiver, not variants')
        if row.get('arm') not in cfg.ARMS or _number(row.get('gamma_db'), 'gamma') not in cfg.GAMMAS:
            raise ValueError('Declared propagation calibration arm/gamma required')
        # Reuse the unchanged, already verified arithmetic; this private
        # adapter label never appears in the propagation export schema.
        prepared.append(dict(row, variant='estimated_ul'))
    result = previous.aggregate_calibration_bins(prepared)
    for row in result:
        row.pop('variant')
    return result


def read_raw_csv(path):
    """Read exported plain/gz CSV; callers explicitly validate its design."""
    opener = gzip.open if str(path).endswith('.gz') else open
    output = []
    with opener(Path(path), 'rt', encoding='utf-8', newline='') as stream:
        for source in csv.DictReader(stream):
            row = {}
            for name, value in source.items():
                if name == 'arm':
                    row[name] = value
                elif value == '':
                    row[name] = None
                elif name in ('sample_index', 'batch_index', 'within_batch'):
                    row[name] = int(value)
                else:
                    row[name] = float(value)
            output.append(row)
    return output
