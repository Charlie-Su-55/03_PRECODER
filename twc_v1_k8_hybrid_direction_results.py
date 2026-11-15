"""Paired complete-MU statistics for a Hybrid-only extension.

The frozen FDMA reference has no extra updates. Comparisons with that model
are deliberately labelled unequal-budget; none isolates the auxiliary loss.
Gamma nodes are repeated views, averaged within MU before batch uncertainty.
"""
from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path
import statistics

ID_FIELDS = frozenset(('variant', 'arm', 'role', 'gamma_db', 'sample_index',
    'batch_index', 'within_batch'))
REQUIRED_METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'soft_nmse_linear',
    'interference_power_mean', 'desired_signal_power_mean', 'direction_error')
CONTRASTS = (
    ('hybrid_direction_minus_hybrid_source', 'hybrid_direction', 'hybrid_source'),
    ('hybrid_direction_minus_fdma_frozen', 'hybrid_direction', 'fdma_frozen'),
    ('hybrid_source_minus_fdma_frozen', 'hybrid_source', 'fdma_frozen'),
)


def _config():
    from configs import twc_v1_k8_hybrid_direction_config as cfg
    return cfg


def physical_arm(variant):
    if variant not in _config().EVAL_VARIANTS:
        raise ValueError('Declared frozen or Hybrid extension variant required')
    return 'fdma' if variant == 'fdma_frozen' else 'hybrid_g4'


def validate_rows(rows, *, samples, batch_size=4):
    cfg = _config()
    if (type(samples) is not int or type(batch_size) is not int or
            batch_size < 1 or samples < batch_size or samples % batch_size):
        raise ValueError('Complete physical batches required')
    index, schemas = {}, {}
    expected = {(v, float(g), i) for v in cfg.EVAL_VARIANTS
        for g in cfg.GAMMAS for i in range(samples)}
    for row in rows:
        if not ID_FIELDS.issubset(row) or not set(REQUIRED_METRICS).issubset(row):
            raise ValueError('Complete final IDs and rate/direction/CSI/error metrics required')
        if (row['role'] != 'final' or row['arm'] != physical_arm(row['variant']) or
                any(type(row[k]) is not int or row[k] < 0
                    for k in ('sample_index', 'batch_index', 'within_batch'))):
            raise ValueError('Only complete final physical MUs may enter statistics')
        g, i = row['gamma_db'], row['sample_index']
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g):
            raise ValueError('Finite gamma node required')
        key = row['variant'], float(g), i
        if (key not in expected or key in index or row['batch_index'] != i//batch_size
                or row['within_batch'] != i % batch_size):
            raise ValueError('Duplicate or unexpected complete-MU condition')
        schema = set(row)-ID_FIELDS
        if row['variant'] in schemas and schema != schemas[row['variant']]:
            raise ValueError('Partially missing metric within a variant')
        schemas[row['variant']] = schema
        for name in schema:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Finite complete-MU scalar required')
            if name in REQUIRED_METRICS and value < 0:
                raise ValueError('Nonnegative rates and errors required')
        index[key] = row
    if set(index) != expected:
        raise ValueError('Incomplete paired six-node final data')
    return index


def _scopes():
    nodes = tuple(float(g) for g in _config().GAMMAS)
    return [('point', (g,)) for g in nodes]+[
        ('low_0_5_10', tuple(g for g in nodes if g in (0., 5., 10.))), ('all_six', nodes)]


def _moments(values, batch_size):
    values = list(values)
    mean = statistics.mean(values)
    blocks = [statistics.mean(values[i:i+batch_size]) for i in range(0, len(values), batch_size)]
    se = statistics.stdev(values)/len(values)**.5 if len(values) > 1 else None
    batch_se = statistics.stdev(blocks)/len(blocks)**.5 if len(blocks) > 1 else None
    return dict(mean=mean, sample_se=se,
        sample_ci95_low=None if se is None else mean-1.96*se,
        sample_ci95_high=None if se is None else mean+1.96*se,
        batch_se=batch_se,
        batch_ci95_low=None if batch_se is None else mean-1.96*batch_se,
        batch_ci95_high=None if batch_se is None else mean+1.96*batch_se)


def summarize_rows(rows, *, samples, batch_size=4):
    index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for variant in _config().EVAL_VARIANTS:
        for scope, gammas in _scopes():
            for metric in sorted(set(index[variant, gammas[0], 0])-ID_FIELDS):
                values = [statistics.mean(index[variant, g, i][metric] for g in gammas)
                    for i in range(samples)]
                output.append(dict(variant=variant, arm=physical_arm(variant), role='final',
                    scope=scope, gamma_db=gammas[0] if scope == 'point' else None,
                    metric=metric, n_samples=samples, n_batches=samples//batch_size,
                    batch_size=batch_size, n_gamma_points=len(gammas),
                    **_moments(values, batch_size)))
    return output


def paired_comparisons(rows, *, samples, batch_size=4):
    index = validate_rows(rows, samples=samples, batch_size=batch_size)
    output = []
    for label, positive, negative in CONTRASTS:
        common = (set(index[positive, float(_config().GAMMAS[0]), 0]) &
            set(index[negative, float(_config().GAMMAS[0]), 0]))-ID_FIELDS
        for scope, gammas in _scopes():
            for metric in sorted(common):
                values = [statistics.mean(index[positive, g, i][metric]-
                    index[negative, g, i][metric] for g in gammas) for i in range(samples)]
                output.append(dict(comparison=label, positive=positive, negative=negative,
                    role='final', scope=scope, gamma_db=gammas[0] if scope == 'point' else None,
                    metric=metric, n_samples=samples, n_batches=samples//batch_size,
                    batch_size=batch_size, n_gamma_points=len(gammas),
                    **_moments(values, batch_size)))
    return output


def read_raw_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    output = []
    with opener(Path(path), 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Unique scalar CSV columns required')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed scalar row')
            parsed = {}
            for name, value in row.items():
                if not value: continue  # Hybrid-only soft components are absent in FDMA.
                parsed[name] = (value if name in ('variant', 'arm', 'role') else
                    int(value) if name in ('sample_index', 'batch_index', 'within_batch') else float(value))
            output.append(parsed)
    return output
