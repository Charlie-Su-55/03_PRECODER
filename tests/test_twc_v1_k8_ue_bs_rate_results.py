"""Independent complete-MU arithmetic checks for joint rate training results."""
import copy
import csv
import gzip
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import twc_v1_k8_ue_bs_rate_results as results

CFG = SimpleNamespace(ARMS=('fdma', 'hybrid_g4'),
    METHODS=('source', 'bs_only', 'joint'), GAMMAS=(0, 5, 10, 15, 20, 25))


def fixture_rows(samples=8):
    rows = []
    for arm in CFG.ARMS:
        a = int(arm == 'hybrid_g4')
        for method in CFG.METHODS:
            shift = {'source': 0., 'bs_only': 3.+5*a, 'joint': 7.+11*a}[method]
            for gamma in CFG.GAMMAS:
                for i in range(samples):
                    metrics = {name: .1 for name in results.REQUIRED_METRICS}
                    metrics.update(sum_rate_bps_hz=10+a+shift+.02*gamma+.1*i,
                        dl_nmse_linear=.01+.001*i, diagnostic_signed=-.2+.01*i,
                        paired_noise=(i-3.5)*(gamma-12.5))
                    rows.append(dict(role='final', arm=arm, method=method,
                        gamma_db=float(gamma), sample_index=i, batch_index=i//4,
                        within_batch=i%4, **metrics))
    return rows


class JointRateStatisticsTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(results, '_config', return_value=CFG)
        patch.start()
        self.addCleanup(patch.stop)
        self.rows = fixture_rows()

    def test_complete_design_and_48_summaries(self):
        copied, index = results.validate_rows(self.rows, samples=8)
        self.assertEqual(len(index), 2*3*6*8)
        self.assertIsNot(copied[0], self.rows[0])
        summary = results.summarize_rows(self.rows, samples=8)
        self.assertEqual(len(summary), 48)
        self.assertTrue(all(r['n_samples'] == 8 and r['n_batches'] == 2 for r in summary))
        row = next(r for r in summary if r['arm'] == 'fdma'
            and r['method'] == 'source' and r['scope'] == 'all_six')
        values = [10+.02*12.5+.1*i for i in range(8)]
        expected = sum(values)/len(values)
        se = math.sqrt(sum((x-expected)**2 for x in values)/(len(values)-1))/math.sqrt(len(values))
        self.assertAlmostEqual(row['sum_rate_bps_hz_mean'], expected)
        self.assertAlmostEqual(row['sum_rate_bps_hz_se'], se)
        self.assertAlmostEqual(row['sum_rate_bps_hz_ci95_low'], expected-1.96*se)
        self.assertAlmostEqual(row['dl_nmse_db_from_linear_mean'], 10*math.log10(.0135))
        self.assertNotIn('phase', row)

    def test_gamma_views_average_within_mu_before_uncertainty(self):
        row = next(r for r in results.summarize_rows(self.rows, samples=8)
            if r['arm'] == 'fdma' and r['method'] == 'source' and r['scope'] == 'all_six')
        self.assertEqual(row['n_gamma_points'], 6)
        self.assertAlmostEqual(row['paired_noise_mean'], 0)
        self.assertAlmostEqual(row['paired_noise_se'], 0)
        self.assertAlmostEqual(row['paired_noise_batch_se'], 0)
        self.assertAlmostEqual(row['sum_rate_bps_hz_batch_se'], .2)
        self.assertAlmostEqual(row['sum_rate_bps_hz_batch_ci95_high'],
            row['sum_rate_bps_hz_mean']+1.96*.2)

    def test_12_prespecified_matched_gains_and_difference_in_differences(self):
        comparisons = results.paired_comparisons(self.rows, samples=8)
        actual = {r['comparison']: r for r in comparisons
            if r['scope'] == 'all_six' and r['metric'] == 'sum_rate_bps_hz'}
        expected = {
            'fdma_bs_only_minus_source': 3., 'fdma_joint_minus_source': 7.,
            'hybrid_g4_bs_only_minus_source': 8., 'hybrid_g4_joint_minus_source': 18.,
            'fdma_joint_minus_bs_only': 4., 'hybrid_g4_joint_minus_bs_only': 10.,
            'source_hybrid_minus_fdma': 1., 'bs_only_hybrid_minus_fdma': 6.,
            'joint_hybrid_minus_fdma': 12., 'joint_gain_hybrid_minus_fdma': 11.,
            'bs_only_gain_hybrid_minus_fdma': 5.,
            'interaction_joint_minus_bs_only_hybrid_minus_fdma': 6.,
        }
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name]['mean'], value, msg=name)
            self.assertAlmostEqual(actual[name]['batch_se'], 0, msg=name)
            self.assertNotIn('phase', actual[name])

    def test_paired_contrast_uses_within_mu_covariance(self):
        # Large common fluctuations cancel exactly in a paired difference.
        for row in self.rows:
            row['sum_rate_bps_hz'] += (row['sample_index'] % 4)*4
        difference = next(r for r in results.paired_comparisons(self.rows, samples=8)
            if r['comparison'] == 'fdma_joint_minus_bs_only'
            and r['metric'] == 'sum_rate_bps_hz' and r['scope'] == 'all_six')
        self.assertAlmostEqual(difference['mean'], 4.)
        self.assertAlmostEqual(difference['sample_se'], 0.)
        self.assertAlmostEqual(difference['batch_se'], 0.)

    def test_negative_training_and_matched_terminal_results_are_retained(self):
        for row in self.rows:
            if row['method'] == 'joint':
                row['sum_rate_bps_hz'] -= 10
        difference = next(r for r in results.paired_comparisons(self.rows, samples=8)
            if r['comparison'] == 'fdma_joint_minus_bs_only'
            and r['metric'] == 'sum_rate_bps_hz' and r['scope'] == 'all_six')
        self.assertAlmostEqual(difference['mean'], -6.)
        self.assertLess(difference['batch_ci95_high'], 0.)

    def test_missing_duplicate_wrong_batch_gamma_and_method_are_rejected(self):
        for mutate in ('missing', 'duplicate', 'batch', 'within', 'gamma', 'method', 'arm', 'phase'):
            rows = copy.deepcopy(self.rows)
            if mutate == 'missing':
                rows.pop()
            elif mutate == 'duplicate':
                rows.append(dict(rows[0]))
            elif mutate == 'batch':
                rows[0]['batch_index'] = 1
            elif mutate == 'within':
                rows[0]['within_batch'] = 3
            elif mutate == 'gamma':
                rows[0]['gamma_db'] = 2.5
            elif mutate == 'method':
                rows[0]['method'] = 'moment_receiver'
            elif mutate == 'arm':
                rows[0]['arm'] = 'pure_shared'
            else:
                rows[0]['phase'] = 'final'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                results.validate_rows(rows, samples=8)

    def test_nonfinal_or_missing_roles_are_rejected(self):
        for role in ('codec_train', 'monitor', 'calibration', 'test', 'validation'):
            rows = copy.deepcopy(self.rows)
            rows[0]['role'] = role
            with self.subTest(role=role), self.assertRaises(ValueError):
                results.validate_rows(rows, samples=8)
        rows = copy.deepcopy(self.rows)
        rows[0].pop('role')
        with self.assertRaises(ValueError):
            results.validate_rows(rows, samples=8)

    def test_nonfinite_boolean_negative_and_missing_required_metrics_rejected(self):
        for value in (math.nan, math.inf, -1., True, None):
            rows = copy.deepcopy(self.rows)
            rows[0]['sum_rate_bps_hz'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                results.validate_rows(rows, samples=8)
        for field in ('sample_index', 'batch_index', 'within_batch', 'gamma_db'):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                results.validate_rows(rows, samples=8)
        rows = copy.deepcopy(self.rows)
        rows[0].pop('soft_nmse_linear')
        with self.assertRaises(ValueError):
            results.validate_rows(rows, samples=8)

    def test_optional_hybrid_metrics_cannot_disappear_in_a_method(self):
        for row in self.rows:
            row['shared_diagnostic'] = 1. if row['arm'] == 'hybrid_g4' else None
        results.validate_rows(self.rows, samples=8)
        next(row for row in self.rows if row['arm'] == 'hybrid_g4')['shared_diagnostic'] = None
        with self.assertRaisesRegex(ValueError, 'Partially missing'):
            results.validate_rows(self.rows, samples=8)

    def test_zero_nmse_db_and_physical_nonnegativity(self):
        for row in self.rows:
            row['dl_nmse_linear'] = 0.
            row['data_energy'] = 48.
        summary = results.summarize_rows(self.rows, samples=8)
        self.assertTrue(all(row['dl_nmse_db_from_linear_mean'] is None for row in summary))
        self.rows[0]['data_energy'] = -1.
        with self.assertRaises(ValueError):
            results.validate_rows(self.rows, samples=8)

    def test_gzip_csv_roundtrip_and_malformed_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.csv.gz'
            with gzip.open(path, 'wt', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            self.assertEqual(results.read_raw_csv(path), self.rows)
            for raw in ('arm,arm\nfdma,fdma\n', ',method\nfdma,joint\n',
                        'arm,method\nfdma\n', 'arm,method\nfdma,joint,extra\n'):
                with gzip.open(path, 'wt') as stream:
                    stream.write(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    results.read_raw_csv(path)

    def test_single_mu_uncertainty_and_invalid_effective_batches(self):
        value = results.moments([2.])
        self.assertEqual(value['mean'], 2.)
        self.assertIsNone(value['se'])
        self.assertIsNone(value['ci95_low'])
        for samples, batch in ((0, 4), (7, 4), (8, 0), (True, 4), (8, True)):
            with self.subTest(samples=samples, batch=batch), self.assertRaises(ValueError):
                results.validate_rows(self.rows, samples=samples, batch_size=batch)


if __name__ == '__main__':
    unittest.main()
