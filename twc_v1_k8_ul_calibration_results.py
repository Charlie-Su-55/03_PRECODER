"""Complete-MU paired validation statistics for the locked CE calibration.

This module uses only the standard library. Calibration samples are never
admitted to validation summaries. Gamma points are repeated views of one MU:
average them within MU before SE/CI or four-MU physical-batch statistics.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import math
from pathlib import Path
import statistics


REQUIRED_IDS = frozenset(('arm', 'variant', 'gamma_db', 'sample_index', 'batch_index', 'within_batch'))
ID_FIELDS = REQUIRED_IDS | {'role'}
REQUIRED_METRICS = ('soft_nmse_linear', 'soft_nominal_scaled_actual_mse',
    'soft_nominal_scaled_predicted_variance', 'sum_rate_bps_hz', 'dl_nmse_linear',
    'interference_power_mean')


def _config():
    from configs import twc_v1_k8_ul_calibration_config as cfg
    return cfg


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite numeric '+name+' required')
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer '+name+' required')
    return value


def _nonnegative(name):
    return (name in REQUIRED_METRICS or name.endswith(('_nmse_linear', '_actual_mse',
        '_predicted_variance', '_reported_variance', '_mse_ratio', '_squared_error_mean',
        '_predicted_variance_mean', '_bps_hz', '_power_mean', '_max_error')))


def validate_rows(rows, *, samples, batch_size=4):
    cfg = _config()
    if (type(samples) is not int or samples < 1 or type(batch_size) is not int
            or batch_size < 1 or samples % batch_size):
        raise ValueError('Positive complete-MU and physical-batch counts required')
    copied = [dict(row) for row in rows]
    expected = {(arm, variant, float(gamma), i) for arm in cfg.ARMS
        for variant in cfg.VARIANTS for gamma in cfg.GAMMAS for i in range(samples)}
    index, groups = {}, defaultdict(list)
    role_presence = {'role' in row for row in copied}
    if len(role_presence) > 1:
        raise ValueError('Validation role metadata must be consistently present')
    for row in copied:
        if not REQUIRED_IDS.issubset(row) or not set(REQUIRED_METRICS).issubset(row):
            raise ValueError('Complete validation IDs, soft errors and downlink metrics required')
        if row.get('role', 'validation') != 'validation':
            raise ValueError('Calibration/test observations cannot enter validation statistics')
        if row['arm'] not in cfg.ARMS or row['variant'] not in cfg.VARIANTS:
            raise ValueError('Declared physical arm and frozen receiver variant required')
        gamma = float(_number(row['gamma_db'], 'gamma'))
        sample = _integer(row['sample_index'], 'sample_index')
        batch = _integer(row['batch_index'], 'batch_index')
        within = _integer(row['within_batch'], 'within_batch')
        key = (row['arm'], row['variant'], gamma, sample)
        if key not in expected or batch != sample//batch_size or within != sample % batch_size:
            raise ValueError('Unexpected validation sample/condition/batch identity')
        if key in index:
            raise ValueError('Duplicate complete-MU validation key')
        for name, value in row.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Nonempty metric column names required')
            if name in ID_FIELDS or value is None:
                continue
            _number(value, name)
            if _nonnegative(name) and value < 0:
                raise ValueError('Nonnegative validation error/power/rate required')
        if any(row[name] is None for name in REQUIRED_METRICS):
            raise ValueError('Required validation metrics cannot be absent')
        index[key] = row
        groups[row['arm']].append(row)
    if set(index) != expected:
        raise ValueError('Missing complete arm/variant/gamma/MU validation coverage')
    # A metric can be absent for FDMA, e.g. shared condition number. Within
    # one physical arm it must exist for every receiver, gamma and MU so a
    # paired contrast cannot silently omit a failed/missing metric.
    for group in groups.values():
        names = set().union(*(set(row)-ID_FIELDS for row in group))
        for name in names:
            present = [row.get(name) is not None for row in group]
            if any(present) and not all(present):
                raise ValueError('Partially missing metric within a physical arm: '+name)
    return copied, index


def moments(values):
    values = list(values)
    if not values:
        raise ValueError('Nonempty complete-MU observations required')
    for value in values:
        _number(value, 'observation')
    mean = statistics.mean(values)
    se = statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, se=se,
        ci95_low=None if se is None else mean-1.96*se,
        ci95_high=None if se is None else mean+1.96*se)


def _batch_moments(values, batch_size):
    return moments(statistics.mean(values[start:start+batch_size])
        for start in range(0, len(values), batch_size))


def _scopes():
    gammas = tuple(float(gamma) for gamma in _config().GAMMAS)
    return [('point', (gamma,)) for gamma in gammas]+[
        ('low_0_5_10', tuple(gamma for gamma in gammas if gamma in (0., 5., 10.))),
        ('all_six', gammas)]


def _metrics(index, keys):
    rows = [index[key] for key in keys]
    names = set.intersection(*(set(row)-ID_FIELDS for row in rows))
    return sorted(name for name in names if all(row[name] is not None for row in rows))


def summarize_rows(rows, *, samples, batch_size=4):
    """Summarize validation only; ratios remain means of per-MU ratios."""
    cfg = _config()
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for arm in cfg.ARMS:
        for variant in cfg.VARIANTS:
            for scope, selected in _scopes():
                if not selected:
                    continue
                keys = [(arm, variant, gamma, i) for gamma in selected for i in range(samples)]
                result = dict(arm=arm, variant=variant, role='validation', scope=scope,
                    gamma_db=selected[0] if scope == 'point' else None,
                    n_samples=samples, n_batches=samples//batch_size,
                    batch_size=batch_size, n_gamma_points=len(selected))
                for metric in _metrics(index, keys):
                    values = [statistics.mean(index[arm, variant, gamma, i][metric]
                        for gamma in selected) for i in range(samples)]
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


def _contrasts():
    cfg = _config()
    if cfg.ARMS != ('fdma', 'hybrid_g4') or not cfg.VARIANTS or cfg.VARIANTS[0] != 'original':
        raise ValueError('Prescribed FDMA/Hybrid arms and original reference required')
    output = []
    for arm in cfg.ARMS:
        for variant in cfg.VARIANTS[1:]:
            output.append((arm+'_'+variant+'_minus_original', arm, variant,
                ((1., arm, variant), (-1., arm, 'original'))))
    for variant in cfg.VARIANTS:
        output.append((variant+'_hybrid_minus_fdma', 'hybrid_minus_fdma', variant,
            ((1., 'hybrid_g4', variant), (-1., 'fdma', variant))))
    for variant in cfg.VARIANTS[1:]:
        output.append(('interaction_'+variant+'_gain_minus_original', 'interaction', variant,
            ((1., 'hybrid_g4', variant), (-1., 'hybrid_g4', 'original'),
             (-1., 'fdma', variant), (1., 'fdma', 'original'))))
    return output


def paired_comparisons(rows, *, samples, batch_size=4):
    """Paired receiver controls, cross-arm gaps and differential gains.

    All signs are literal positive-minus-negative metric differences. An
    interference/error decrease is therefore negative, a rate gain positive.
    Intervals are conditional on the independently locked calibration fit;
    calibration-fit uncertainty and training-seed variability are excluded.
    """
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for label, arm, variant, terms in _contrasts():
        for scope, selected in _scopes():
            if not selected:
                continue
            keys = [(a, v, gamma, i) for _, a, v in terms
                for gamma in selected for i in range(samples)]
            for metric in _metrics(index, keys):
                values = [statistics.mean(sum(coefficient*index[a, v, gamma, i][metric]
                    for coefficient, a, v in terms) for gamma in selected) for i in range(samples)]
                sample, batch = moments(values), _batch_moments(values, batch_size)
                output.append(dict(comparison=label, arm=arm, variant=variant,
                    metric=metric, role='validation', scope=scope,
                    gamma_db=selected[0] if scope == 'point' else None,
                    n_samples=samples, n_batches=samples//batch_size,
                    batch_size=batch_size, n_gamma_points=len(selected), mean=sample['mean'],
                    sample_se=sample['se'], sample_ci95_low=sample['ci95_low'],
                    sample_ci95_high=sample['ci95_high'], batch_se=batch['se'],
                    batch_ci95_low=batch['ci95_low'], batch_ci95_high=batch['ci95_high']))
    return output


def read_raw_csv(path):
    """Read plain/gzip per-MU observations; validation is an explicit next step."""
    opener = gzip.open if str(path).endswith('.gz') else open
    output = []
    with opener(Path(path), 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Unique nonempty raw CSV columns required')
        for source in reader:
            if None in source:
                raise ValueError('Malformed raw CSV row width')
            row = {}
            for name, value in source.items():
                if name in ('arm', 'variant', 'role'):
                    row[name] = value
                elif value == '':
                    row[name] = None
                elif name in ('sample_index', 'batch_index', 'within_batch'):
                    row[name] = int(value)
                else:
                    row[name] = float(value)
            output.append(row)
    return output
