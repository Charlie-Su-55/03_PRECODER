#!/usr/bin/env python3
"""Build a small scalar digest of a completed report directory and push it.

The full report stays on the server. The digest keeps every file's SHA256,
the export manifest, priority-metric summary/comparison/curve rows with their
batch intervals, an independent recomputation of point means from
samples.csv.gz, and compacted JSON metadata. It never reads weights.

Pushing uses Git plumbing (hash-object/mktree/commit-tree) and creates one new
orphan branch ``results-digest/<namespace>/<export_id>``; the working tree,
index, HEAD and main are untouched. An existing remote branch is never
overwritten. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'results_digest_v1'
PRIORITY = {
    'sum_rate_bps_hz', 'dl_nmse_linear', 'soft_nmse_linear', 'direction_error',
    'interference_power_mean', 'desired_signal_power_mean', 'dl_leakage_penalty',
    'ul_all_nominal_scaled_actual_error_power',
    'ul_nmse_linear', 'gram_error', 'energy_ratio', 'ce_mse',
    'min_user_rate_bps_hz', 'key_energy_fraction', 'pair_coupling', 'low_gamma_sum_rate', 'loss',
    'true_ul_coupling', 'sum_rate_dl10', 'sum_rate_dl15', 'sum_rate_dl20', 'sum_rate_dl30'}
KEEP_COLUMNS = ('variant', 'comparison', 'trajectory', 'step', 'CE_total_updates', 'role',
    'scope', 'gamma_db', 'metric', 'n_samples', 'mean', 'batch_ci95_low', 'batch_ci95_high')
ID_COLUMNS = {'variant', 'arm', 'role', 'gamma_db', 'sample_index', 'batch_index', 'within_batch'}
IP = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
DEFAULT_BUDGET_KIB = 24


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    if not math.isfinite(value):
        return text
    if value == int(value) and abs(value) < 1e9:
        return int(value)
    return float(f'{value:.7g}')


def clean_string(text):
    text = IP.sub('<ip>', text)
    return text if len(text) <= 200 else text[:120] + '…sha256=' + hashlib.sha256(text.encode()).hexdigest()[:16]


def compact(value, depth=0):
    """Shrink arbitrary JSON: long numeric arrays become stats, deep nesting is cut."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return number(value)
    if isinstance(value, str):
        return clean_string(value)
    if isinstance(value, list):
        flat = []
        def walk(item):
            if isinstance(item, list):
                for x in item: walk(x)
            else:
                flat.append(item)
        walk(value)
        if len(flat) > 12 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in flat):
            finite = [float(x) for x in flat if math.isfinite(float(x))]
            if not finite:
                return dict(n=len(flat))
            return dict(n=len(flat), mean=number(sum(finite)/len(finite)),
                min=number(min(finite)), max=number(max(finite)))
        if len(value) > 12:
            return dict(n=len(value), first=[compact(x, depth+1) for x in value[:2]])
        return [compact(x, depth+1) for x in value]
    if isinstance(value, dict):
        if depth >= 4:
            return dict(keys=sorted(map(str, value))[:20], n_keys=len(value))
        return {str(k): compact(v, depth+1) for k, v in value.items()}
    return clean_string(str(value))


def read_csv(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def table(rows, *, priority_only):
    if not rows:
        return dict(columns=[], rows=[])
    columns = [c for c in KEEP_COLUMNS if c in rows[0]]
    for extra in ('positive', 'negative'):
        if extra in rows[0] and extra not in columns:
            columns.insert(1, extra)
    selected = [r for r in rows if ('metric' not in r) or ((r['metric'] in PRIORITY) == priority_only)]
    if not priority_only:  # non-priority metrics: aggregate scopes only
        selected = [r for r in selected if r.get('scope') not in ('point',)]
    return dict(columns=columns, rows=[[number(r.get(c, '')) if r.get(c, '') != '' else None for c in columns]
                                       for r in selected])


def recompute_points(samples_path, summary_rows):
    """Independent check: per (variant, role, gamma, metric) means from raw samples."""
    sums, counts = {}, {}
    for row in read_csv(samples_path):
        for metric in PRIORITY & set(row):
            if row[metric] == '':
                continue
            key = (row['variant'], row.get('role', ''), float(row['gamma_db']), metric)
            sums[key] = sums.get(key, 0.) + float(row[metric])
            counts[key] = counts.get(key, 0) + 1
    published = {}
    for r in summary_rows:
        if r.get('scope') == 'point' and r.get('metric') in PRIORITY:
            published[(r['variant'], r.get('role', ''), float(r['gamma_db']), r['metric'])] = float(r['mean'])
    worst, matched = 0., 0
    for key, total in sums.items():
        if key in published:
            mean = total / counts[key]
            worst = max(worst, abs(mean - published[key]) / max(1e-12, abs(published[key])))
            matched += 1
    return dict(raw_rows_per_condition=sorted(set(counts.values())), conditions=len(sums),
        matched_summary_points=matched, max_relative_mean_difference=number(worst))


def build(report):
    report = Path(report).resolve()
    if not report.is_dir():
        raise SystemExit('STOP: report directory not found: ' + str(report))
    parts = report.parts
    namespace = parts[parts.index('runs') + 1] if 'runs' in parts else report.parent.name
    export_id = report.name.removeprefix('attempt_')
    files = {p.name: dict(sha256=sha256(p), bytes=p.stat().st_size)
             for p in sorted(report.iterdir()) if p.is_file()}
    tiers = [dict(format=FORMAT, namespace=namespace, export_id=export_id, files=files,
                  priority_metrics=sorted(PRIORITY))]
    tier1 = tiers[0]
    if (report/'export_manifest.json').is_file():
        tier1['export_manifest'] = compact(json.loads((report/'export_manifest.json').read_text()))
    summary = read_csv(report/'summary.csv') if (report/'summary.csv').is_file() else []
    for name in ('summary.csv', 'comparisons.csv', 'curve_summary.csv'):
        if (report/name).is_file():
            rows = summary if name == 'summary.csv' else read_csv(report/name)
            tier1[name.removesuffix('.csv')] = table(rows, priority_only=True)
    if (report/'samples.csv.gz').is_file() and summary:
        tier1['audit_recomputed_from_samples'] = recompute_points(report/'samples.csv.gz', summary)
    tier2 = {}
    for name in ('provenance.json', 'calibration.json'):
        if (report/name).is_file():
            tier2[name.removesuffix('.json')] = compact(json.loads((report/name).read_text()))
    tier3 = {}
    for name in ('summary.csv', 'comparisons.csv'):
        if (report/name).is_file():
            rows = summary if name == 'summary.csv' else read_csv(report/name)
            tier3[name.removesuffix('.csv') + '_other_metrics_aggregate_scopes'] = table(rows, priority_only=False)
    return namespace, export_id, tier1, [('metadata', tier2), ('other_metrics', tier3)]


def encode(document):
    return (json.dumps(document, ensure_ascii=False, separators=(',', ':'), sort_keys=True) + '\n').encode()


def assemble(tier1, optional, budget):
    document = dict(tier1, included_tiers=['priority'], omitted_tiers=[])
    for label, content in optional:
        trial = dict(document, **content)
        trial['included_tiers'] = document['included_tiers'] + [label]
        if len(zlib.compress(encode(trial), 9)) <= budget:
            document = trial
        else:
            document['omitted_tiers'] = document['omitted_tiers'] + [label]
    data = encode(document)
    return data, len(zlib.compress(data, 9))


def git(*args, data=None, env=None):
    result = subprocess.run(['git', *args], cwd=ROOT, input=data, capture_output=True, env=env)
    if result.returncode:
        raise SystemExit('STOP: git ' + args[0] + ' failed: ' + result.stderr.decode(errors='replace').strip()[-400:])
    return result.stdout.decode().strip()


def push(namespace, export_id, data, tag=None):
    if tag is not None and not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,30}', tag):
        raise SystemExit('STOP: --tag must be lowercase letters, digits and dashes')
    ref = f'refs/heads/results-digest/{namespace}/{export_id}' + (f'-{tag}' if tag else '')
    if git('ls-remote', 'origin', ref):
        raise SystemExit('STOP: remote branch already exists; nothing overwritten: ' + ref)
    blob = git('hash-object', '-w', '--stdin', data=data)
    tree = git('mktree', data=f'100644 blob {blob}\tdigest.json\n'.encode())
    env = dict(os.environ, GIT_AUTHOR_NAME='03_PRECODER results digest',
               GIT_AUTHOR_EMAIL='results-digest@users.noreply.github.com',
               GIT_COMMITTER_NAME='03_PRECODER results digest',
               GIT_COMMITTER_EMAIL='results-digest@users.noreply.github.com')
    commit = git('commit-tree', tree, '-m', f'Results digest {namespace} {export_id}', env=env)
    result = subprocess.run(['git', 'push', 'origin', f'{commit}:{ref}'], cwd=ROOT)
    if result.returncode:
        raise SystemExit('STOP: push failed; local report untouched. Commit ' + commit)
    print(f'PUSHED {ref} commit={commit}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report_dir')
    parser.add_argument('--budget-kib', type=float, default=DEFAULT_BUDGET_KIB,
                        help='compressed size limit for optional tiers (default %(default)s)')
    parser.add_argument('--push', action='store_true', help='push; default only prints the plan')
    parser.add_argument('--write', type=Path, help='also write digest.json to this path')
    parser.add_argument('--tag', help='suffix for a second digest of the same report (new branch <export_id>-<tag>)')
    args = parser.parse_args(argv)
    namespace, export_id, tier1, optional = build(args.report_dir)
    data, packed = assemble(tier1, optional, int(args.budget_kib * 1024))
    document = json.loads(data)
    print(f'namespace={namespace} export_id={export_id} raw={len(data)/1024:.1f}KiB '
          f'compressed≈{packed/1024:.1f}KiB included={document["included_tiers"]} '
          f'omitted={document["omitted_tiers"]}')
    audit = document.get('audit_recomputed_from_samples')
    if audit:
        print('audit:', json.dumps(audit))
    if args.write:
        args.write.write_bytes(data)
    if args.push:
        push(namespace, export_id, data, args.tag)
    return 0


if __name__ == '__main__':
    sys.exit(main())
