"""Complete-MU statistics for UE+BS rate training and archived BS controls.

Gamma values are paired views of the same MU. Scope averages are formed
inside each MU before sample and batch4 uncertainty is calculated. The joint
system updates its causal power prior; the archived BS-only control froze
that prior. These contrasts do not isolate the UE gradient alone.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import math
from pathlib import Path
import statistics

REQUIRED_IDS = frozenset(('arm', 'method', 'role', 'gamma_db', 'sample_index',
    'batch_index', 'within_batch'))
ID_FIELDS = REQUIRED_IDS
REQUIRED_METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'soft_nmse_linear',
    'interference_power_mean', 'desired_signal_power_mean')


def _config():
    from configs import twc_v1_k8_ue_bs_rate_config as cfg
    return cfg


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Finite numeric '+name+' required')
    return value


def _nonnegative_metric(name):
    return (name in REQUIRED_METRICS or name == 'interference_to_signal_ratio'
        or '_energy_user_' in name or name.endswith(('_nmse_linear', '_bps_hz',
            '_mse_ratio', '_actual_mse', '_predicted_variance', '_reported_variance',
            '_power_mean', '_max_error', '_energy', '_energy_ratio',
            '_squared_error_mean', '_variance_mean')))


def validate_rows(rows, *, samples, batch_size=4):
    """Require every declared arm/method/gamma/MU exactly once."""
    cfg = _config()
    if (type(samples) is not int or samples < 1 or type(batch_size) is not int
            or batch_size < 1 or samples % batch_size):
        raise ValueError('Positive counts of complete physical microbatches required')
    expected = {(arm, method, float(gamma), i) for arm in cfg.ARMS
        for method in cfg.METHODS for gamma in cfg.GAMMAS for i in range(samples)}
    copied = [dict(row) for row in rows]
    index, groups = {}, defaultdict(list)
    for row in copied:
        if not REQUIRED_IDS.issubset(row) or not set(REQUIRED_METRICS).issubset(row):
            raise ValueError('Complete final IDs and rate/CSI/soft metrics required')
        if row['role'] != 'final':
            raise ValueError('Training/monitor/calibration/test rows cannot enter final statistics')
        if row['arm'] not in cfg.ARMS or row['method'] not in cfg.METHODS:
            raise ValueError('Declared source/BS-only/joint physical view required')
        for field in ('sample_index', 'batch_index', 'within_batch'):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError('Nonnegative integer MU/batch identities required')
        gamma = float(_number(row['gamma_db'], 'gamma'))
        sample, batch, within = (row[n] for n in ('sample_index', 'batch_index', 'within_batch'))
        key = row['arm'], row['method'], gamma, sample
        if (key not in expected or key in index or batch != sample//batch_size
                or within != sample % batch_size):
            raise ValueError('Duplicate, missing or unexpected final condition/MU/batch')
        for name, value in row.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Nonempty scalar metric names required')
            if name in ID_FIELDS or value is None:
                continue
            _number(value, name)
            if _nonnegative_metric(name) and value < 0:
                raise ValueError('Nonnegative errors/powers/energies/rates required')
        if any(row[name] is None for name in REQUIRED_METRICS):
            raise ValueError('Required final metrics cannot be absent')
        index[key] = row
        groups[row['arm']].append(row)
    if set(index) != expected:
        raise ValueError('Incomplete paired arm/method/gamma/MU coverage')
    for group in groups.values():
        names = set().union(*(set(row)-ID_FIELDS for row in group))
        for name in names:
            present = [row.get(name) is not None for row in group]
            if any(present) and not all(present):
                raise ValueError('Partially missing metric within one physical arm: '+name)
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


def _scopes():
    nodes = tuple(float(gamma) for gamma in _config().GAMMAS)
    return [('point', (gamma,)) for gamma in nodes]+[
        ('low_0_5_10', tuple(gamma for gamma in nodes if gamma in (0., 5., 10.))),
        ('all_six', nodes)]


def _metrics(index, keys):
    rows = [index[key] for key in keys]
    names = set.intersection(*(set(row)-ID_FIELDS for row in rows))
    return sorted(name for name in names if all(row[name] is not None for row in rows))


def _statistics(values, batch_size):
    sample = moments(values)
    batch = moments(statistics.mean(values[i:i+batch_size])
        for i in range(0, len(values), batch_size))
    return sample, batch


def summarize_rows(rows, *, samples, batch_size=4):
    cfg = _config()
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for arm in cfg.ARMS:
        for method in cfg.METHODS:
            for scope, selected in _scopes():
                if not selected:
                    continue
                keys = [(arm, method, gamma, i) for gamma in selected for i in range(samples)]
                row = dict(arm=arm, method=method, role='final', scope=scope,
                    gamma_db=selected[0] if scope == 'point' else None,
                    n_samples=samples, n_batches=samples//batch_size, batch_size=batch_size,
                    n_gamma_points=len(selected))
                for metric in _metrics(index, keys):
                    values = [statistics.mean(index[arm, method, gamma, i][metric]
                        for gamma in selected) for i in range(samples)]
                    sample, batch = _statistics(values, batch_size)
                    for suffix in ('mean', 'se', 'ci95_low', 'ci95_high'):
                        row[metric+'_'+suffix] = sample[suffix]
                    for suffix in ('se', 'ci95_low', 'ci95_high'):
                        row[metric+'_batch_'+suffix] = batch[suffix]
                    if metric.endswith('_nmse_linear'):
                        row[metric[:-len('_linear')]+'_db_from_linear_mean'] = (
                            None if sample['mean'] == 0 else 10*math.log10(sample['mean']))
                output.append(row)
    return output


def _contrasts():
    cfg = _config()
    if (tuple(cfg.ARMS) != ('fdma', 'hybrid_g4')
            or tuple(cfg.METHODS) != ('source', 'bs_only', 'joint')):
        raise ValueError('Prespecified source/archived-BS/joint controls required')
    output, gains = [], {}
    for arm in cfg.ARMS:
        for method in ('bs_only', 'joint'):
            terms = ((1., arm, method), (-1., arm, 'source'))
            gains[arm, method] = terms
            output.append((arm+'_'+method+'_minus_source', 'gain', arm, method, terms))
        output.append((arm+'_joint_minus_bs_only', 'matched_terminal', arm,
            'joint_minus_bs_only', ((1., arm, 'joint'), (-1., arm, 'bs_only'))))
    for method in cfg.METHODS:
        output.append((method+'_hybrid_minus_fdma', 'arm_gap', 'hybrid_minus_fdma', method,
            ((1., 'hybrid_g4', method), (-1., 'fdma', method))))
    for method in ('joint', 'bs_only'):
        terms = gains['hybrid_g4', method]+tuple((-c, a, m) for c, a, m in gains['fdma', method])
        output.append((method+'_gain_hybrid_minus_fdma', 'gain_interaction',
            'hybrid_minus_fdma', method, terms))
    output.append(('interaction_joint_minus_bs_only_hybrid_minus_fdma', 'gain_interaction',
        'hybrid_minus_fdma', 'joint_minus_bs_only',
        ((1., 'hybrid_g4', 'joint'), (-1., 'hybrid_g4', 'bs_only'),
         (-1., 'fdma', 'joint'), (1., 'fdma', 'bs_only'))))
    return output


def paired_comparisons(rows, *, samples, batch_size=4):
    _, index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for label, contrast, arm, method, terms in _contrasts():
        for scope, selected in _scopes():
            if not selected:
                continue
            keys = [(a, m, gamma, i) for _, a, m in terms
                for gamma in selected for i in range(samples)]
            for metric in _metrics(index, keys):
                values = [statistics.mean(sum(c*index[a, m, gamma, i][metric]
                    for c, a, m in terms) for gamma in selected) for i in range(samples)]
                sample, batch = _statistics(values, batch_size)
                output.append(dict(comparison=label, contrast=contrast, arm=arm, method=method,
                    metric=metric, role='final', scope=scope,
                    gamma_db=selected[0] if scope == 'point' else None,
                    n_samples=samples, n_batches=samples//batch_size, batch_size=batch_size,
                    n_gamma_points=len(selected), mean=sample['mean'], sample_se=sample['se'],
                    sample_ci95_low=sample['ci95_low'], sample_ci95_high=sample['ci95_high'],
                    batch_se=batch['se'], batch_ci95_low=batch['ci95_low'], batch_ci95_high=batch['ci95_high']))
    return output


def read_raw_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    output = []
    with opener(Path(path), 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or any(not name for name in reader.fieldnames)
                or len(set(reader.fieldnames)) != len(reader.fieldnames)):
            raise ValueError('Unique nonempty raw CSV fields required')
        for source in reader:
            if None in source or any(value is None for value in source.values()):
                raise ValueError('Malformed raw CSV width')
            row = {}
            for name, value in source.items():
                if name in ('arm', 'method', 'role'):
                    row[name] = value
                elif value == '':
                    row[name] = None
                elif name in ('sample_index', 'batch_index', 'within_batch'):
                    row[name] = int(value)
                else:
                    row[name] = float(value)
            output.append(row)
    return output
