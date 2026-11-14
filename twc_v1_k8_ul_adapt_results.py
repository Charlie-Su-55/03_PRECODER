"""Complete-MU, fixed-terminal statistics for matched receiver adaptation.

Only the standard library is imported. Gamma points are paired views of the
same MU; scope averages precede MU and batch4 confidence intervals.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import math
from pathlib import Path
import statistics

REQUIRED_IDS = frozenset(('phase', 'arm', 'method', 'gamma_db', 'sample_index',
    'batch_index', 'within_batch'))
ID_FIELDS = REQUIRED_IDS | {'role'}
REQUIRED_METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'soft_nmse_linear',
    'interference_power_mean', 'desired_signal_power_mean')


def _config():
    from configs import twc_v1_k8_ul_adapt_config as cfg
    return cfg


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Finite numeric '+name+' required')
    return value


def validate_rows(rows, *, samples, batch_size=4):
    cfg = _config()
    if (type(samples) is not int or samples < 1 or type(batch_size) is not int
            or batch_size < 1 or samples % batch_size):
        raise ValueError('Positive counts of complete physical microbatches required')
    expected = {(phase, arm, method, float(gamma), i) for phase in cfg.PHASES
        for arm in cfg.ARMS for method in cfg.TRAIN_METHODS
        for gamma in cfg.GAMMAS for i in range(samples)}
    copied = [dict(row) for row in rows]
    index, groups = {}, defaultdict(list)
    if len({'role' in row for row in copied}) > 1:
        raise ValueError('Final role metadata must be consistently present')
    for row in copied:
        if not REQUIRED_IDS.issubset(row) or not set(REQUIRED_METRICS).issubset(row):
            raise ValueError('Complete final IDs and rate/CSI/soft metrics required')
        if row.get('role', 'final') != 'final':
            raise ValueError('Training/monitor/calibration/test rows cannot enter final statistics')
        if (row['phase'] not in cfg.PHASES or row['arm'] not in cfg.ARMS
                or row['method'] not in cfg.TRAIN_METHODS):
            raise ValueError('Declared source/final physical arm and receiver method required')
        for field in ('sample_index', 'batch_index', 'within_batch'):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError('Nonnegative integer MU/batch identities required')
        gamma = float(_number(row['gamma_db'], 'gamma'))
        sample, batch, within = (row[n] for n in ('sample_index','batch_index','within_batch'))
        key = row['phase'], row['arm'], row['method'], gamma, sample
        if (key not in expected or key in index or batch != sample//batch_size
                or within != sample % batch_size):
            raise ValueError('Duplicate, missing or unexpected final condition/MU/batch')
        for name, value in row.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Nonempty scalar metric names required')
            if name in ID_FIELDS or value is None:
                continue
            _number(value, name)
            if (name in REQUIRED_METRICS or name.endswith(('_nmse_linear', '_bps_hz',
                    '_mse_ratio', '_actual_mse', '_predicted_variance', '_reported_variance',
                    '_power_mean', '_max_error'))) and value < 0:
                raise ValueError('Nonnegative errors/powers/rates required')
        if any(row[name] is None for name in REQUIRED_METRICS):
            raise ValueError('Required final metrics cannot be absent')
        index[key] = row
        groups[row['arm']].append(row)
    if set(index) != expected:
        raise ValueError('Incomplete paired phase/arm/method/gamma/MU coverage')
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
    se = statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
    return dict(n=len(values), mean=mean, se=se,
        ci95_low=None if se is None else mean-1.96*se,
        ci95_high=None if se is None else mean+1.96*se)


def _scopes():
    nodes = tuple(float(gamma) for gamma in _config().GAMMAS)
    return [('point',(gamma,)) for gamma in nodes]+[
        ('low_0_5_10',tuple(gamma for gamma in nodes if gamma in (0.,5.,10.))),
        ('all_six',nodes)]


def _metrics(index, keys):
    rows = [index[key] for key in keys]
    names = set.intersection(*(set(row)-ID_FIELDS for row in rows))
    return sorted(name for name in names if all(row[name] is not None for row in rows))


def _statistics(values, batch_size):
    sample = moments(values)
    batch = moments(statistics.mean(values[i:i+batch_size])
        for i in range(0,len(values),batch_size))
    return sample,batch


def summarize_rows(rows, *, samples, batch_size=4):
    cfg = _config()
    _, index = validate_rows(rows,samples=samples,batch_size=batch_size)
    output=[]
    for phase in cfg.PHASES:
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                for scope,selected in _scopes():
                    if not selected: continue
                    keys=[(phase,arm,method,gamma,i) for gamma in selected for i in range(samples)]
                    row=dict(phase=phase,arm=arm,method=method,role='final',scope=scope,
                        gamma_db=selected[0] if scope=='point' else None,
                        n_samples=samples,n_batches=samples//batch_size,batch_size=batch_size,
                        n_gamma_points=len(selected))
                    for metric in _metrics(index,keys):
                        values=[statistics.mean(index[phase,arm,method,gamma,i][metric]
                            for gamma in selected) for i in range(samples)]
                        sample,batch=_statistics(values,batch_size)
                        for suffix in ('mean','se','ci95_low','ci95_high'):
                            row[metric+'_'+suffix]=sample[suffix]
                        for suffix in ('se','ci95_low','ci95_high'):
                            row[metric+'_batch_'+suffix]=batch[suffix]
                        if metric.endswith('_nmse_linear'):
                            row[metric[:-len('_linear')]+'_db_from_linear_mean']=(
                                None if sample['mean']==0 else 10*math.log10(sample['mean']))
                    output.append(row)
    return output


def _contrasts():
    cfg=_config()
    if (cfg.PHASES!=('source','final') or cfg.ARMS!=('fdma','hybrid_g4')
            or cfg.TRAIN_METHODS!=('original','moment_receiver')):
        raise ValueError('Prespecified paired source/final receiver controls required')
    output=[]
    gain={}
    for arm in cfg.ARMS:
        for method in cfg.TRAIN_METHODS:
            terms=((1.,'final',arm,method),(-1.,'source',arm,method))
            gain[arm,method]=terms
            output.append((arm+'_'+method+'_adapt_gain','final_minus_source',arm,method,terms))
    for phase in cfg.PHASES:
        for arm in cfg.ARMS:
            output.append((phase+'_'+arm+'_moment_minus_original',phase,arm,'moment_minus_original',
                ((1.,phase,arm,'moment_receiver'),(-1.,phase,arm,'original'))))
        for method in cfg.TRAIN_METHODS:
            output.append((phase+'_'+method+'_hybrid_minus_fdma',phase,'hybrid_minus_fdma',method,
                ((1.,phase,'hybrid_g4',method),(-1.,phase,'fdma',method))))
    interactions={}
    for arm in cfg.ARMS:
        terms=gain[arm,'moment_receiver']+tuple((-c,p,a,m) for c,p,a,m in gain[arm,'original'])
        interactions[arm]=terms
        output.append((arm+'_moment_gain_minus_original_gain','gain_interaction',arm,
            'moment_minus_original',terms))
    for method in cfg.TRAIN_METHODS:
        terms=gain['hybrid_g4',method]+tuple((-c,p,a,m) for c,p,a,m in gain['fdma',method])
        output.append((method+'_hybrid_gain_minus_fdma_gain','gain_interaction',
            'hybrid_minus_fdma',method,terms))
    terms=interactions['hybrid_g4']+tuple((-c,p,a,m) for c,p,a,m in interactions['fdma'])
    output.append(('interaction_moment_adaptation_hybrid_minus_fdma','three_way_interaction',
        'hybrid_minus_fdma','moment_minus_original',terms))
    return output


def paired_comparisons(rows, *, samples, batch_size=4):
    _,index=validate_rows(rows,samples=samples,batch_size=batch_size)
    output=[]
    for label,phase,arm,method,terms in _contrasts():
        for scope,selected in _scopes():
            if not selected: continue
            keys=[(p,a,m,gamma,i) for _,p,a,m in terms for gamma in selected for i in range(samples)]
            for metric in _metrics(index,keys):
                values=[statistics.mean(sum(c*index[p,a,m,gamma,i][metric]
                    for c,p,a,m in terms) for gamma in selected) for i in range(samples)]
                sample,batch=_statistics(values,batch_size)
                output.append(dict(comparison=label,phase=phase,arm=arm,method=method,
                    metric=metric,role='final',scope=scope,
                    gamma_db=selected[0] if scope=='point' else None,
                    n_samples=samples,n_batches=samples//batch_size,batch_size=batch_size,
                    n_gamma_points=len(selected),mean=sample['mean'],sample_se=sample['se'],
                    sample_ci95_low=sample['ci95_low'],sample_ci95_high=sample['ci95_high'],
                    batch_se=batch['se'],batch_ci95_low=batch['ci95_low'],batch_ci95_high=batch['ci95_high']))
    return output


def read_raw_csv(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    output=[]
    with opener(Path(path),'rt',encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):
            raise ValueError('Unique nonempty raw CSV fields required')
        for source in reader:
            if None in source: raise ValueError('Malformed raw CSV width')
            row={}
            for name,value in source.items():
                if name in ('phase','arm','method','role'): row[name]=value
                elif value=='': row[name]=None
                elif name in ('sample_index','batch_index','within_batch'): row[name]=int(value)
                else: row[name]=float(value)
            output.append(row)
    return output
