#!/usr/bin/env python3
"""Export per-batch means of selected variants from an existing report (CPU, read-only source).

    python -B tools/export_batch_means.py runs/twc_v1_k8_test_v1/reports/attempt_388f04558e96 \
        --variants fdma_baseline --tag fdma

GitHub rejects pushes from the servers above roughly 60-170 KiB, so export one or two
variants per call (about 28 KiB compressed per variant with three metrics).

Reads samples.csv.gz of the source report and writes a NEW report directory
runs/<source namespace>_batch_means/reports/attempt_<source id>[-<tag>]/ with batch_means.csv,
provenance.json (source path and samples SHA256) and export_manifest.json. The source report
is never modified; an existing output directory stops the run. No torch, no GPU.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('fdma_baseline', 'proposed', 'hybrid_final', 'hybrid_paired', 'ft_paired')
METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'min_user_rate_bps_hz')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def batch_means(samples_path, variants=VARIANTS, metrics=METRICS, batch_size=4):
    acc = {}
    with gzip.open(samples_path, 'rt', newline='') as stream:
        for r in csv.DictReader(stream):
            if r['variant'] not in variants:
                continue
            for m in metrics:
                if r.get(m, '') != '':
                    acc.setdefault((r['variant'], r['role'], float(r['gamma_db']), int(r['batch_index']), m),
                                   []).append(float(r[m]))
    out = []
    for (v, role, g, b, m), x in sorted(acc.items()):
        if len(x) != batch_size:
            raise SystemExit('STOP: incomplete batch in source samples')
        out.append(dict(variant=v, role=role, gamma_db=g, batch_index=b, metric=m, n_samples=len(x),
                        mean=repr(statistics.mean(x))))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source_report')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS))
    parser.add_argument('--metrics', nargs='+', default=list(METRICS))
    parser.add_argument('--tag', help='lowercase suffix for the output directory and digest id')
    args = parser.parse_args(argv)
    if args.tag is not None and not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,30}', args.tag):
        raise SystemExit('STOP: --tag must be lowercase letters, digits and dashes')
    src = Path(args.source_report).resolve()
    samples = src/'samples.csv.gz'
    if not samples.is_file():
        raise SystemExit('STOP: samples.csv.gz not found in ' + str(src))
    namespace = src.parts[src.parts.index('runs')+1] + '_batch_means'
    export_id = src.name.removeprefix('attempt_') + (f'-{args.tag}' if args.tag else '')
    out = ROOT/'runs'/namespace/'reports'/('attempt_'+export_id)
    if out.exists():
        raise SystemExit('STOP: output exists, nothing overwritten: ' + str(out))
    rows = batch_means(samples, tuple(args.variants), tuple(args.metrics))
    if not rows:
        raise SystemExit('STOP: no rows for the requested variants/metrics')
    out.mkdir(parents=True)
    with open(out/'batch_means.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out/'provenance.json').write_text(json.dumps(dict(namespace=namespace, source_report=str(src.relative_to(ROOT)),
        source_samples_sha256=sha256(samples), variants=sorted({r['variant'] for r in rows}), metrics=list(args.metrics),
        batch_size=4, contains_model_weights=False, contains_channel_tensors=False), indent=2, sort_keys=True)+'\n')
    files = {p.name: sha256(p) for p in sorted(out.iterdir())}
    (out/'export_manifest.json').write_text(json.dumps(dict(namespace=namespace, export_id=export_id, files=files),
        indent=2, sort_keys=True)+'\n')
    print(f'rows={len(rows)} variants={sorted({r["variant"] for r in rows})}')
    print('REPORT', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
