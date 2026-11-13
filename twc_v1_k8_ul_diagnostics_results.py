"""CSV-only, complete-MU statistics for frozen uplink diagnostics.

The privileged true-UL comparison is a sensitivity diagnostic, not a new
deployable method or a performance bound. Gamma points are repeated views of
one MU realization: paired differences are averaged within MU before either
sample or four-MU batch intervals are calculated. Calibration coordinates
are descriptive counts; they never become independent observations for CI.
This module imports neither Torch nor Sionna and loads no model weights.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import math
from pathlib import Path
import statistics


ID_FIELDS = frozenset(('arm', 'variant', 'gamma_db', 'sample_index',
                       'batch_index', 'within_batch'))
REQUIRED_METRICS = ('soft_nmse_linear',)


def _config():
    from configs import twc_v1_k8_ul_diagnostics_config as cfg
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
    """Normal-approximation 95% interval; one observation cannot estimate SE."""
    values = list(values)
    if not values:
        raise ValueError('Nonempty numeric observations required')
    for value in values:
        _number(value, 'observation')
    mean = statistics.mean(values)
    se = statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, se=se,
        ci95_low=None if se is None else mean-1.96*se,
        ci95_high=None if se is None else mean+1.96*se)


def validate_rows(rows, *, samples, batch_size=4):
    """Require the complete six-gamma design and unambiguous physical MU IDs."""
    cfg = _config()
    if (type(samples) is not int or samples < 1
            or type(batch_size) is not int or batch_size < 1 or samples % batch_size):
        raise ValueError('Positive sample counts comprising complete physical microbatches required')
    copied = [dict(row) for row in rows]
    expected = {(arm, variant, float(gamma), i) for arm in cfg.ARMS
                for variant in cfg.VARIANTS for gamma in cfg.GAMMAS for i in range(samples)}
    index, groups = {}, defaultdict(list)
    for row in copied:
        if not ID_FIELDS.issubset(row) or not set(REQUIRED_METRICS).issubset(row):
            raise ValueError('Incomplete per-MU diagnostic row')
        arm, variant = row['arm'], row['variant']
        if not isinstance(arm, str) or arm not in cfg.ARMS or not isinstance(variant, str) or variant not in cfg.VARIANTS:
            raise ValueError('Declared physical arm and diagnostic variant required')
        gamma = _number(row['gamma_db'], 'gamma')
        i = _integer(row['sample_index'], 'sample_index')
        batch = _integer(row['batch_index'], 'batch_index')
        within = _integer(row['within_batch'], 'within_batch')
        key = (arm, variant, float(gamma), i)
        if key not in expected or batch != i//batch_size or within != i % batch_size:
            raise ValueError('Unexpected diagnostic condition or sample/batch identity')
        if key in index:
            raise ValueError('Duplicate complete-MU diagnostic key: '+repr(key))
        for field, value in row.items():
            if not isinstance(field, str) or not field:
                raise ValueError('Nonempty string column names required')
            if field in ID_FIELDS or value is None:
                continue
            _number(value, field, nonnegative=field.endswith('_nmse_linear'))
        if any(row[name] is None for name in REQUIRED_METRICS):
            raise ValueError('Missing required soft recovery metric')
        index[key] = row
        groups[key[:3]].append(row)
    if set(index) != expected:
        raise ValueError('Missing complete-MU diagnostic observations: '+repr(sorted(expected-set(index))[:3]))
    for group in groups.values():
        metrics = set().union(*(set(row) for row in group))-ID_FIELDS
        for metric in metrics:
            present = [row.get(metric) is not None for row in group]
            if any(present) and not all(present):
                raise ValueError('Partially missing diagnostic within one condition: '+metric)
    return copied, index


def _batch_moments(values, batch_size):
    return moments(statistics.mean(values[start:start+batch_size])
                   for start in range(0, len(values), batch_size))


def summarize_rows(rows, *, samples, batch_size=4):
    """Keep all finite diagnostics and log10(mean linear NMSE), including zero."""
    cfg = _config()
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for arm in cfg.ARMS:
        for variant in cfg.VARIANTS:
            for gamma in cfg.GAMMAS:
                group = [index[arm, variant, float(gamma), i] for i in range(samples)]
                result = dict(arm=arm, variant=variant, gamma_db=float(gamma),
                    n_samples=samples, n_batches=samples//batch_size, batch_size=batch_size)
                metrics = sorted(set().union(*(set(row) for row in group))-ID_FIELDS)
                for metric in metrics:
                    values = [row.get(metric) for row in group]
                    if all(value is None for value in values):
                        continue
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
    """True-UL diagnostic minus normal recovery; no cross-arm rate ranking."""
    cfg = _config()
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    gammas = tuple(float(gamma) for gamma in cfg.GAMMAS)
    scopes = [('point', (gamma,)) for gamma in gammas]
    scopes += [('low_0_5_10', gammas[:3]), ('all_six', gammas)]
    before, after = 'estimated_ul', 'true_ul_diagnostic'
    if tuple(cfg.VARIANTS) != (before, after):
        raise ValueError('The declared estimated/privileged diagnostic contrast changed')
    output = []
    for arm in cfg.ARMS:
        for scope, selected in scopes:
            common = set.intersection(*(set(index[arm, variant, gamma, 0])-ID_FIELDS
                                       for variant in cfg.VARIANTS for gamma in selected))
            for metric in sorted(common):
                if any(index[arm, variant, gamma, i].get(metric) is None
                       for variant in cfg.VARIANTS for gamma in selected for i in range(samples)):
                    continue
                deltas = [statistics.mean(index[arm, after, gamma, i][metric]
                                         -index[arm, before, gamma, i][metric]
                                         for gamma in selected) for i in range(samples)]
                sample, batch = moments(deltas), _batch_moments(deltas, batch_size)
                output.append(dict(arm=arm, variant_from=before, variant_to=after,
                    comparison=arm+'_true_ul_diagnostic_minus_estimated_ul', metric=metric,
                    scope=scope, gamma_db=selected[0] if scope == 'point' else None,
                    n_samples=samples, n_batches=samples//batch_size, batch_size=batch_size,
                    mean=sample['mean'], sample_se=sample['se'],
                    sample_ci95_low=sample['ci95_low'], sample_ci95_high=sample['ci95_high'],
                    batch_se=batch['se'], batch_ci95_low=batch['ci95_low'],
                    batch_ci95_high=batch['ci95_high']))
    return output


def aggregate_calibration_bins(rows):
    """Pool descriptive coordinate counts/sums without coordinate-level CI.

    Input rows carry arm/variant/gamma and the metrics module's component,
    bin_index/lower/upper/count/sum_actual/sum_predicted. Optional batch_index
    makes duplicate per-batch bin exports an error. The final upper boundary
    is None rather than infinity so all output remains finite JSON/CSV.
    """
    cfg = _config()
    required = {'arm', 'variant', 'gamma_db', 'component', 'bin_index',
                'lower', 'upper', 'count', 'sum_actual', 'sum_predicted'}
    grouped, seen = {}, set()
    for source in rows:
        row = dict(source)
        if not required.issubset(row):
            raise ValueError('Incomplete calibration-bin row')
        arm, variant = row['arm'], row['variant']
        if not isinstance(arm, str) or arm not in cfg.ARMS or not isinstance(variant, str) or variant not in cfg.VARIANTS:
            raise ValueError('Declared calibration arm and variant required')
        gamma = float(_number(row['gamma_db'], 'gamma'))
        if gamma not in cfg.GAMMAS:
            raise ValueError('Declared calibration gamma required')
        component = row['component']
        if not isinstance(component, str) or not component:
            raise ValueError('Nonempty calibration component required')
        bin_index = _integer(row['bin_index'], 'bin_index')
        count = _integer(row['count'], 'count')
        lower = _number(row['lower'], 'bin lower', nonnegative=True)
        upper = row['upper']
        if upper is not None and _number(upper, 'bin upper', nonnegative=True) <= lower:
            raise ValueError('Increasing calibration-bin boundaries required')
        actual = _number(row['sum_actual'], 'sum_actual', nonnegative=True)
        predicted = _number(row['sum_predicted'], 'sum_predicted', nonnegative=True)
        if count == 0 and (actual != 0 or predicted != 0):
            raise ValueError('An empty bin cannot contain error/variance sums')
        key = (arm, variant, gamma, component, bin_index)
        if 'batch_index' in row:
            identity = (*key, _integer(row['batch_index'], 'batch_index'))
            if identity in seen:
                raise ValueError('Duplicate calibration bin in one physical microbatch')
            seen.add(identity)
        if key not in grouped:
            grouped[key] = dict(arm=arm, variant=variant, gamma_db=gamma,
                component=component, bin_index=bin_index, lower=lower, upper=upper,
                count=0, sum_actual=0., sum_predicted=0.)
        result = grouped[key]
        if (result['lower'], result['upper']) != (lower, upper):
            raise ValueError('Calibration-bin boundaries changed between batches')
        result['count'] += count
        result['sum_actual'] += actual
        result['sum_predicted'] += predicted
        _number(result['sum_actual'], 'aggregated actual sum', nonnegative=True)
        _number(result['sum_predicted'], 'aggregated predicted sum', nonnegative=True)
    output = []
    for key in sorted(grouped):
        result = grouped[key]
        result['mean_actual'] = result['sum_actual']/result['count'] if result['count'] else None
        result['mean_predicted'] = result['sum_predicted']/result['count'] if result['count'] else None
        result['actual_to_predicted_ratio'] = (result['sum_actual']/result['sum_predicted']
                                              if result['sum_predicted'] else None)
        output.append(result)
    return output


def read_raw_csv(path):
    """Read the exported finite per-MU table; validation remains explicit."""
    opener = gzip.open if str(path).endswith('.gz') else open
    output = []
    with opener(Path(path), 'rt', encoding='utf-8', newline='') as stream:
        for source in csv.DictReader(stream):
            row = {}
            for name, value in source.items():
                if name in ('arm', 'variant'):
                    row[name] = value
                elif value == '':
                    row[name] = None
                elif name in ('sample_index', 'batch_index', 'within_batch'):
                    row[name] = int(value)
                else:
                    row[name] = float(value)
            output.append(row)
    return output
