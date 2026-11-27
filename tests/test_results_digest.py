"""CPU tests for tools/results_digest.py (standard library only)."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('results_digest', ROOT/'tools'/'results_digest.py')
digest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(digest)


def _write(path, rows, gz=False):
    opener = gzip.open if gz else open
    with opener(path, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ResultsDigestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.report = Path(self.tmp.name)/'runs'/'ns_v1'/'reports'/'attempt_abc123'
        self.report.mkdir(parents=True)
        samples = []
        for variant in ('a', 'b'):
            for gamma in (0., 5.):
                for i in range(8):
                    samples.append(dict(variant=variant, role='final', gamma_db=gamma, sample_index=i,
                        sum_rate_bps_hz=10+i+gamma, other_metric=i))
        _write(self.report/'samples.csv.gz', samples, gz=True)
        summary = []
        for variant in ('a', 'b'):
            for gamma in (0., 5.):
                mean = sum(10+i+gamma for i in range(8))/8
                summary.append(dict(variant=variant, role='final', scope='point', gamma_db=gamma,
                    metric='sum_rate_bps_hz', n_samples=8, mean=mean, sample_se=0,
                    batch_ci95_low=mean-1, batch_ci95_high=mean+1))
                summary.append(dict(variant=variant, role='final', scope='point', gamma_db=gamma,
                    metric='other_metric', n_samples=8, mean=3.5, sample_se=0,
                    batch_ci95_low=3, batch_ci95_high=4))
            summary.append(dict(variant=variant, role='final', scope='all_six', gamma_db='',
                metric='other_metric', n_samples=8, mean=3.5, sample_se=0, batch_ci95_low=3, batch_ci95_high=4))
        _write(self.report/'summary.csv', summary)
        (self.report/'provenance.json').write_text(json.dumps(
            dict(host='10.20.30.40', table=list(range(100)), note='ok')))
        (self.report/'export_manifest.json').write_text(json.dumps(dict(files={'summary.csv': '0'*64})))

    def tearDown(self):
        self.tmp.cleanup()

    def test_skip_and_match_filters(self):
        import re
        _write(self.report/'batch_means.csv', [dict(variant='a', role='final', gamma_db=0., batch_index=0,
            metric='sum_rate_bps_hz', mean=1.)])
        _, _, tier1, _ = digest.build(self.report, ('batch_means.csv',), re.compile('^b$'))
        self.assertNotIn('batch_means', tier1)
        self.assertIn('batch_means.csv', tier1['files'])
        self.assertEqual({r[0] for r in tier1['summary']['rows']}, {'b'})
        self.assertEqual(tier1['audit_recomputed_from_samples']['conditions'], 2)
        self.assertEqual(tier1['filter'], dict(skipped_files=['batch_means.csv'], match='^b$'))
        _, _, plain, _ = digest.build(self.report)
        self.assertIn('batch_means', plain)
        self.assertNotIn('filter', plain)
        with self.assertRaises(SystemExit):
            digest.main([str(self.report), '--match', 'b', '--push'])

    def test_digest_contents_audit_and_redaction(self):
        namespace, export_id, tier1, optional = digest.build(self.report)
        self.assertEqual((namespace, export_id), ('ns_v1', 'abc123'))
        data, packed = digest.assemble(tier1, optional, 24*1024)
        document = json.loads(data)
        self.assertEqual(set(document['files']), {'samples.csv.gz', 'summary.csv', 'provenance.json', 'export_manifest.json'})
        self.assertEqual(document['files']['summary.csv']['sha256'], digest.sha256(self.report/'summary.csv'))
        metrics = {row[document['summary']['columns'].index('metric')] for row in document['summary']['rows']}
        self.assertEqual(metrics, {'sum_rate_bps_hz'})
        audit = document['audit_recomputed_from_samples']
        self.assertEqual(audit['matched_summary_points'], 4)
        self.assertLess(audit['max_relative_mean_difference'], 1e-12)
        self.assertEqual(document['provenance']['host'], '<ip>')
        self.assertEqual(document['provenance']['table']['n'], 100)
        other = document['summary_other_metrics_aggregate_scopes']
        self.assertTrue(all(row[other['columns'].index('scope')] != 'point' for row in other['rows']))
        self.assertEqual(packed, len(zlib.compress(data, 9)))

    def test_budget_drops_optional_tiers_not_priority(self):
        _, _, tier1, optional = digest.build(self.report)
        document = json.loads(digest.assemble(tier1, optional, 1)[0])
        self.assertEqual(document['included_tiers'], ['priority'])
        self.assertEqual(document['omitted_tiers'], ['metadata', 'other_metrics'])
        self.assertIn('summary', document)

    def test_push_creates_orphan_branch_and_never_overwrites(self):
        work = Path(self.tmp.name)/'work'
        bare = Path(self.tmp.name)/'remote.git'
        subprocess.run(['git', 'init', '-q', '--bare', str(bare)], check=True)
        subprocess.run(['git', 'init', '-q', str(work)], check=True)
        subprocess.run(['git', '-C', str(work), 'remote', 'add', 'origin', str(bare)], check=True)
        original = digest.ROOT
        digest.ROOT = work
        try:
            digest.push('ns_v1', 'abc123', b'{"x":1}\n')
            with self.assertRaises(SystemExit):
                digest.push('ns_v1', 'abc123', b'{"x":2}\n')
        finally:
            digest.ROOT = original
        shown = subprocess.run(['git', '-C', str(bare), 'show', 'results-digest/ns_v1/abc123:digest.json'],
            capture_output=True, check=True).stdout
        self.assertEqual(shown, b'{"x":1}\n')
        self.assertNotEqual(subprocess.run(['git', '-C', str(work), 'rev-parse', '--verify', '-q', 'HEAD']).returncode, 0)


if __name__ == '__main__':
    unittest.main()
