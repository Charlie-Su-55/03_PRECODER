"""Complete-MU paired statistics for the independent CE continuation study.

Gamma conditions repeat the same channel MU. Average gamma views inside each
MU before computing uncertainty across physical batches. These intervals
condition on the fixed models and are not training-seed uncertainty.
"""
from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path
import statistics

DEFAULT_VARIANTS = ('source_ce5000', 'nmse65000', 'geometry65000')
GAMMAS = (0., 5., 10., 15., 20., 25.)
MIDPOINTS = (2.5, 7.5, 12.5, 17.5, 22.5)
ID_FIELDS = frozenset(('variant', 'role', 'gamma_db', 'sample_index',
    'batch_index', 'within_batch'))
REQUIRED_METRICS = ('ul_nmse_linear', 'direction_error', 'gram_error',
    'energy_ratio', 'ce_mse')
CONTRASTS = (
    ('geometry_minus_nmse', 'geometry65000', 'nmse65000'),
    ('nmse_minus_source', 'nmse65000', 'source_ce5000'),
    ('geometry_minus_source', 'geometry65000', 'source_ce5000'),
)


def _conditions(variants, gammas, role):
    if role not in ('final', 'midpoint'):
        raise ValueError('Only declared final or midpoint validation roles permitted')
    variants = tuple(variants)
    if (not variants or len(set(variants)) != len(variants)
            or any(not isinstance(v, str) or not v.strip() for v in variants)):
        raise ValueError('Unique nonempty model identifiers required')
    allowed = GAMMAS if role == 'final' else MIDPOINTS
    supplied = allowed if gammas is None else tuple(gammas)
    if (not supplied or any(isinstance(g, bool) or not isinstance(g, (int, float))
            or not math.isfinite(g) or float(g) not in allowed for g in supplied)):
        raise ValueError('Declared finite gamma nodes for this validation role required')
    nodes = tuple(float(g) for g in supplied)
    if len(set(nodes)) != len(nodes):
        raise ValueError('Unique gamma conditions required')
    return variants, nodes


def validate_rows(rows, *, samples, batch_size=4, variants=DEFAULT_VARIANTS,
                  gammas=None, role='final'):
    variants, nodes = _conditions(variants, gammas, role)
    if (type(samples) is not int or type(batch_size) is not int
            or batch_size < 1 or samples < batch_size or samples % batch_size):
        raise ValueError('A whole number of complete physical batches required')
    expected = {(v, g, i) for v in variants for g in nodes for i in range(samples)}
    index, schema = {}, None
    for row in rows:
        if (not isinstance(row, dict) or not ID_FIELDS.issubset(row)
                or not set(REQUIRED_METRICS).issubset(row)):
            raise ValueError('Complete MU identifiers and CE metrics required')
        if (row['role'] != role or not isinstance(row['variant'], str)
                or any(type(row[k]) is not int or row[k] < 0
                    for k in ('sample_index', 'batch_index', 'within_batch'))):
            raise ValueError('Declared validation role and integer MU indices required')
        gamma, i = row['gamma_db'], row['sample_index']
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma):
            raise ValueError('Finite nonboolean gamma required')
        key = row['variant'], float(gamma), i
        if (key not in expected or key in index or row['batch_index'] != i//batch_size
                or row['within_batch'] != i % batch_size):
            raise ValueError('Duplicate or unexpected complete-MU condition')
        current = set(row)-ID_FIELDS
        if schema is None:
            schema = current
        elif current != schema:
            raise ValueError('Identical complete metric schema required across models and MUs')
        for metric in current:
            value = row[metric]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError('Finite complete-MU scalar metrics required')
            if metric in REQUIRED_METRICS and value < 0:
                raise ValueError('Nonnegative CE errors and energy ratio required')
        index[key] = row
    if set(index) != expected:
        raise ValueError('Incomplete paired CE validation data')
    return index


def _scopes(nodes, role):
    output = [('point', (g,)) for g in nodes]
    if role == 'midpoint':
        return output+[('all_midpoints', nodes)]
    if set((0., 5., 10.)).issubset(nodes):
        output.append(('low_0_5_10', (0., 5., 10.)))
    if set((15., 20., 25.)).issubset(nodes):
        output.append(('high_15_20_25', (15., 20., 25.)))
    output.append(('all_six' if set(nodes) == set(GAMMAS) else 'all_nodes', nodes))
    return output


def _moments(values, batch_size):
    n = len(values)
    mean = statistics.mean(values)
    batches = [statistics.mean(values[i:i+batch_size]) for i in range(0, n, batch_size)]
    sample_se = statistics.stdev(values)/math.sqrt(n) if n > 1 else None
    batch_se = statistics.stdev(batches)/math.sqrt(len(batches)) if len(batches) > 1 else None
    return dict(mean=mean, sample_se=sample_se,
        sample_ci95_low=None if sample_se is None else mean-1.96*sample_se,
        sample_ci95_high=None if sample_se is None else mean+1.96*sample_se,
        batch_se=batch_se,
        batch_ci95_low=None if batch_se is None else mean-1.96*batch_se,
        batch_ci95_high=None if batch_se is None else mean+1.96*batch_se)


def summarize_rows(rows, *, samples=1024, batch_size=4, variants=DEFAULT_VARIANTS,
                   gammas=None, role='final'):
    variants, nodes = _conditions(variants, gammas, role)
    index = validate_rows(rows, samples=samples, batch_size=batch_size,
        variants=variants, gammas=nodes, role=role)
    metrics = sorted(set(index[variants[0], nodes[0], 0])-ID_FIELDS)
    output = []
    for variant in variants:
        for scope, selected in _scopes(nodes, role):
            for metric in metrics:
                values = [statistics.mean(index[variant, g, i][metric] for g in selected)
                    for i in range(samples)]
                output.append(dict(variant=variant, role=role, scope=scope,
                    gamma_db=selected[0] if scope == 'point' else None, metric=metric,
                    n_samples=samples, n_batches=samples//batch_size,
                    batch_size=batch_size, n_gamma_points=len(selected),
                    **_moments(values, batch_size)))
    return output


def paired_comparisons(rows, *, samples=1024, batch_size=4, variants=DEFAULT_VARIANTS,
                       gammas=None, role='final'):
    variants, nodes = _conditions(variants, gammas, role)
    index = validate_rows(rows, samples=samples, batch_size=batch_size,
        variants=variants, gammas=nodes, role=role)
    metrics = sorted(set(index[variants[0], nodes[0], 0])-ID_FIELDS)
    output = []
    for label, positive, negative in CONTRASTS:
        if positive not in variants or negative not in variants:
            continue
        for scope, selected in _scopes(nodes, role):
            for metric in metrics:
                values = [statistics.mean(index[positive, g, i][metric]-
                    index[negative, g, i][metric] for g in selected) for i in range(samples)]
                output.append(dict(comparison=label, positive=positive, negative=negative,
                    role=role, scope=scope,
                    gamma_db=selected[0] if scope == 'point' else None, metric=metric,
                    n_samples=samples, n_batches=samples//batch_size,
                    batch_size=batch_size, n_gamma_points=len(selected),
                    **_moments(values, batch_size)))
    return output


def read_raw_csv(path):
    path = Path(path)
    opener = gzip.open if path.name.endswith('.gz') else open
    result = []
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Unique scalar CSV column names required')
        for row in reader:
            if None in row or any(value is None or value == '' for value in row.values()):
                raise ValueError('Complete scalar CSV records required')
            result.append({name: value if name in ('variant', 'role') else
                int(value) if name in ('sample_index', 'batch_index', 'within_batch') else
                float(value) for name, value in row.items()})
    return result
