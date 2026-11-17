"""Paired complete-MU statistics for the final Hybrid evaluation.

Gamma views of one MU are averaged within the MU before uncertainty over
batches of four (same convention as all earlier K8 reports). The adaptive
switch and the FDMA baseline choice are decided on the separate selection
set and only applied to the final set.
"""
from __future__ import annotations

import statistics

ID_FIELDS = frozenset(('variant', 'role', 'gamma_db', 'sample_index', 'batch_index'))
CONTRASTS = (
    ('final_minus_fdma', 'hybrid_final', 'fdma_baseline'),
    ('adaptive_minus_fdma', 'adaptive_switch', 'fdma_baseline'),
    ('final_minus_source', 'hybrid_final', 'hybrid_source'),
    ('source_minus_fdma', 'hybrid_source', 'fdma_baseline'),
    ('paired_minus_final', 'hybrid_final_paired', 'hybrid_final'),
    ('paired_minus_fdma', 'hybrid_final_paired', 'fdma_baseline'),
    ('final_minus_ablate_ce5000', 'hybrid_final', 'hybrid_final_ce5000'),
    ('final_minus_ablate_fixed_gamma', 'hybrid_final', 'hybrid_final_fixed_gamma'),
    ('final_minus_ablate_pooled_prior', 'hybrid_final', 'hybrid_final_pooled_prior'),
    ('fdma_ce65000_minus_ce5000', 'fdma_ce65000', 'fdma_ce5000'),
)


def moments(values, batch_size=4):
    values = list(values)
    mean = statistics.mean(values)
    blocks = [statistics.mean(values[i:i+batch_size]) for i in range(0, len(values), batch_size)]
    se = statistics.stdev(blocks)/len(blocks)**.5 if len(blocks) > 1 else None
    return dict(mean=mean, batch_se=se,
        batch_ci95_low=None if se is None else mean-1.96*se,
        batch_ci95_high=None if se is None else mean+1.96*se)


def index_rows(rows):
    table = {}
    for row in rows:
        key = row['variant'], row['role'], float(row['gamma_db']), int(row['sample_index'])
        if key in table:
            raise ValueError('Duplicate complete-MU row')
        table[key] = row
    return table


def scopes(gammas):
    gammas = tuple(float(g) for g in gammas)
    out = [('point', (g,)) for g in gammas]
    low, high = tuple(g for g in gammas if g <= 10), tuple(g for g in gammas if g >= 15)
    if low and high:
        out += [('low_0_5_10', low), ('high_15_20_25', high), ('all_six', gammas)]
    return out


def _samples(table, variant, role, gammas):
    ids = sorted({k[3] for k in table if k[0] == variant and k[1] == role and k[2] == gammas[0]})
    for g in gammas:
        if sorted({k[3] for k in table if k[0] == variant and k[1] == role and k[2] == g}) != ids:
            raise ValueError('Incomplete paired gamma grid for '+variant)
    return ids


def summarize(rows, role, gammas, metrics):
    table = index_rows(rows)
    out = []
    for variant in sorted({k[0] for k in table if k[1] == role}):
        for scope, selected in scopes(gammas):
            ids = _samples(table, variant, role, selected)
            for metric in metrics:
                if any(metric not in table[variant, role, g, ids[0]] for g in selected):
                    continue
                values = [statistics.mean(float(table[variant, role, g, i][metric]) for g in selected) for i in ids]
                out.append(dict(variant=variant, role=role, scope=scope,
                    gamma_db=selected[0] if scope == 'point' else None, metric=metric,
                    n_samples=len(ids), **moments(values)))
    return out


def compare(rows, role, gammas, metrics, contrasts=CONTRASTS):
    table = index_rows(rows)
    present = {k[0] for k in table if k[1] == role}
    out = []
    for label, positive, negative in contrasts:
        if positive not in present or negative not in present:
            continue
        for scope, selected in scopes(gammas):
            ids = _samples(table, positive, role, selected)
            if ids != _samples(table, negative, role, selected):
                raise ValueError('Contrast variants must share the same complete MUs')
            for metric in metrics:
                if any(metric not in table[v, role, g, ids[0]] for v in (positive, negative) for g in selected):
                    continue
                values = [statistics.mean(float(table[positive, role, g, i][metric])
                    -float(table[negative, role, g, i][metric]) for g in selected) for i in ids]
                out.append(dict(comparison=label, positive=positive, negative=negative, role=role,
                    scope=scope, gamma_db=selected[0] if scope == 'point' else None, metric=metric,
                    n_samples=len(ids), **moments(values)))
    return out


def choose_fdma_baseline(selection_rows):
    """FDMA variant with the higher all-six mean sum rate on the selection set."""
    best = {}
    for row in selection_rows:
        if row['variant'] in ('fdma_ce5000', 'fdma_ce65000'):
            best.setdefault(row['variant'], []).append(float(row['sum_rate_bps_hz']))
    if set(best) != {'fdma_ce5000', 'fdma_ce65000'}:
        raise ValueError('Both FDMA variants required on the selection set')
    means = {k: statistics.mean(v) for k, v in best.items()}
    return max(sorted(means), key=means.get), means


def choose_switch(selection_rows, fdma_variant, hybrid_variant='hybrid_final'):
    """Per gamma node: use Hybrid where its selection-set mean rate is at least FDMA's."""
    sums = {}
    for row in selection_rows:
        if row['variant'] in (fdma_variant, hybrid_variant):
            sums.setdefault((row['variant'], float(row['gamma_db'])), []).append(float(row['sum_rate_bps_hz']))
    decision = {}
    for (variant, gamma) in sums:
        if variant == hybrid_variant:
            h, f = statistics.mean(sums[hybrid_variant, gamma]), statistics.mean(sums[fdma_variant, gamma])
            decision[gamma] = dict(mode='hybrid' if h >= f else 'fdma', hybrid_mean=h, fdma_mean=f)
    return dict(sorted(decision.items()))


def materialize(rows, role, fdma_variant, decision):
    """Add 'fdma_baseline' and 'adaptive_switch' rows (copies; values unchanged)."""
    extra = []
    for row in rows:
        if row['role'] != role:
            continue
        if row['variant'] == fdma_variant:
            extra.append(dict(row, variant='fdma_baseline'))
        gamma = float(row['gamma_db'])
        mode = decision.get(gamma, {}).get('mode')
        if mode == 'fdma' and row['variant'] == fdma_variant or mode == 'hybrid' and row['variant'] == 'hybrid_final':
            extra.append(dict(row, variant='adaptive_switch'))
    return extra
