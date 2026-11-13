"""Analytic, standard-library checks of paired uplink diagnostic statistics."""
import copy
import csv
import gzip
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import unittest

import twc_v1_k8_ul_diagnostics_results as results


ARMS = ('fdma', 'hybrid_g4')
VARIANTS = ('estimated_ul', 'true_ul_diagnostic')
GAMMAS = (0., 5., 10., 15., 20., 25.)


def diagnostic_rows(samples=8):
    rows = []
    for arm in ARMS:
        for variant in VARIANTS:
            for gi, gamma in enumerate(GAMMAS):
                for i in range(samples):
                    normal = 100.+.1*gi+.01*i
                    actual = normal-(i+1)*(gi+1) if variant == 'true_ul_diagnostic' else normal
                    rows.append(dict(arm=arm, variant=variant, gamma_db=gamma,
                        sample_index=i, batch_index=i//4, within_batch=i % 4,
                        soft_nmse_linear=actual,
                        ul_ce_nmse_linear=0. if variant == 'true_ul_diagnostic' else .1,
                        soft_shared_nmse_linear=None if arm == 'fdma' else actual+1,
                        geometry_near_singular_fraction=0.))
    return rows


def find(rows, arm='fdma', metric='soft_nmse_linear', scope='all_six', gamma=None):
    return next(row for row in rows if row['arm'] == arm and row['metric'] == metric
                and row['scope'] == scope and row['gamma_db'] == gamma)


def calibration_row(**kwargs):
    value = dict(arm='hybrid_g4', variant='estimated_ul', gamma_db=0.,
        component='recovery_soft_shared', bin_index=2, lower=.03, upper=.1,
        batch_index=0, count=2, sum_actual=4., sum_predicted=2.)
    value.update(kwargs)
    return value


class UplinkDiagnosticStatistics(unittest.TestCase):
    def test_complete_mu_gamma_average_precedes_batch_interval(self):
        output = results.paired_comparisons(diagnostic_rows(), samples=8)
        # Six repeated gamma observations give delta_i=-3.5*(i+1), not48
        # independent observations. The two batch means are -8.75,-22.75.
        for arm in ARMS:
            actual = find(output, arm)
            self.assertAlmostEqual(actual['mean'], -15.75, places=12)
            self.assertAlmostEqual(actual['batch_se'], 7., places=12)
            self.assertAlmostEqual(actual['batch_ci95_low'], -29.47, places=12)
            self.assertAlmostEqual(actual['batch_ci95_high'], -2.03, places=12)
            expected_se = statistics.stdev([-3.5*(i+1) for i in range(8)])/math.sqrt(8)
            self.assertAlmostEqual(actual['sample_se'], expected_se, places=12)
            self.assertEqual((actual['n_samples'], actual['n_batches'], actual['batch_size']), (8, 2, 4))
            low = find(output, arm, scope='low_0_5_10')
            self.assertAlmostEqual(low['mean'], -9., places=12)
            self.assertAlmostEqual(low['batch_se'], 4., places=12)
            points = [r for r in output if r['arm'] == arm and r['scope'] == 'point'
                      and r['metric'] == 'soft_nmse_linear']
            self.assertEqual([r['gamma_db'] for r in points], list(GAMMAS))
        self.assertEqual({r['comparison'] for r in output},
            {a+'_true_ul_diagnostic_minus_estimated_ul' for a in ARMS})

    def test_signed_worsening_and_optional_components_are_not_pruned(self):
        rows = diagnostic_rows()
        for row in rows:
            if row['variant'] == 'true_ul_diagnostic':
                row['soft_nmse_linear'] += 50.
        output = results.paired_comparisons(rows, samples=8)
        self.assertGreater(find(output)['mean'], 0.)
        self.assertFalse(any(r['arm'] == 'fdma' and r['metric'] == 'soft_shared_nmse_linear' for r in output))
        self.assertIsNotNone(find(output, 'hybrid_g4', 'soft_shared_nmse_linear'))

    def test_summary_uses_linear_mean_and_zero_error_is_finite_missing_db(self):
        rows = diagnostic_rows()
        for row in rows:
            row['soft_nmse_linear'] = .01 if row['sample_index'] % 2 == 0 else 1.
        summary = results.summarize_rows(rows, samples=8)
        self.assertEqual(len(summary), 24)
        for row in summary:
            self.assertAlmostEqual(row['soft_nmse_linear_mean'], .505, places=12)
            self.assertAlmostEqual(row['soft_nmse_db_from_linear_mean'], 10*math.log10(.505), places=12)
            self.assertAlmostEqual(row['soft_nmse_linear_batch_se'], 0., places=12)
            if row['variant'] == 'true_ul_diagnostic':
                self.assertEqual(row['ul_ce_nmse_linear_mean'], 0.)
                self.assertIsNone(row['ul_ce_nmse_db_from_linear_mean'])
            else:
                self.assertAlmostEqual(row['ul_ce_nmse_db_from_linear_mean'], -10., places=12)
        json.dumps(summary, allow_nan=False)

    def test_single_batch_has_no_fabricated_batch_uncertainty_and_order_is_irrelevant(self):
        rows = diagnostic_rows(4)
        expected = results.paired_comparisons(rows, samples=4)
        self.assertTrue(all(row['batch_se'] is None and row['batch_ci95_low'] is None
                            and row['batch_ci95_high'] is None for row in expected))
        random.Random(52).shuffle(rows)
        self.assertEqual(results.paired_comparisons(rows, samples=4), expected)

    def test_missing_duplicate_unknown_and_mismatched_mu_identity_fail(self):
        rows = diagnostic_rows()
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)
        for field, value in [('arm', 'unknown'), ('variant', 'deployed_oracle'),
                ('gamma_db', 30.), ('gamma_db', True), ('sample_index', True),
                ('sample_index', 8), ('batch_index', 1), ('within_batch', 4)]:
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)
        for samples, batch_size in [(0, 4), (7, 4), (True, 4), (8, 0), (8, True)]:
            with self.subTest(samples=samples, batch_size=batch_size), self.assertRaises(ValueError):
                results.validate_rows(rows, samples=samples, batch_size=batch_size)

    def test_missing_required_partial_optional_and_nonfinite_metrics_fail(self):
        rows = diagnostic_rows()
        for metric, value in [('soft_nmse_linear', None), ('soft_nmse_linear', -1.),
                ('soft_nmse_linear', math.nan), ('soft_nmse_linear', math.inf),
                ('soft_nmse_linear', True), ('geometry_near_singular_fraction', 'bad')]:
            bad = copy.deepcopy(rows)
            bad[0][metric] = value
            with self.subTest(metric=metric, value=value), self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)
        bad = copy.deepcopy(rows)
        bad[0]['new_optional'] = 1.
        with self.assertRaisesRegex(ValueError, 'Partially missing'):
            results.validate_rows(bad, samples=8)
        bad = copy.deepcopy(rows)
        del bad[0]['soft_nmse_linear']
        with self.assertRaisesRegex(ValueError, 'Incomplete per-MU'):
            results.validate_rows(bad, samples=8)

    def test_raw_gzip_csv_roundtrip_preserves_variant_identity_and_blank_diagnostics(self):
        rows = diagnostic_rows()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'samples.csv.gz'
            with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            read = results.read_raw_csv(path)
        self.assertEqual(read, rows)
        results.validate_rows(read, samples=8)

    def test_calibration_pools_counts_and_sums_without_coordinate_confidence_intervals(self):
        rows = [calibration_row(), calibration_row(batch_index=1, count=3,
                    sum_actual=8., sum_predicted=4.),
                calibration_row(component='channel_ce_shared', upper=None),
                calibration_row(bin_index=0, lower=0., upper=.01, count=0,
                                sum_actual=0., sum_predicted=0.)]
        pooled = results.aggregate_calibration_bins(rows)
        result = next(row for row in pooled if row['component'] == 'recovery_soft_shared' and row['bin_index'] == 2)
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['sum_actual'], 12.)
        self.assertEqual(result['sum_predicted'], 6.)
        self.assertEqual(result['mean_actual'], 2.4)
        self.assertEqual(result['mean_predicted'], 1.2)
        self.assertEqual(result['actual_to_predicted_ratio'], 2.)
        empty = next(row for row in pooled if row['bin_index'] == 0)
        self.assertIsNone(empty['mean_actual'])
        self.assertIsNone(empty['mean_predicted'])
        self.assertIsNone(empty['actual_to_predicted_ratio'])
        self.assertFalse(any('ci95' in key or key.endswith('_se') for row in pooled for key in row))
        self.assertEqual(results.aggregate_calibration_bins(reversed(rows)), pooled)
        json.dumps(pooled, allow_nan=False)

    def test_calibration_duplicates_boundary_drift_and_invalid_sums_are_rejected(self):
        for bad in ([calibration_row(), calibration_row()],
                    [calibration_row(), calibration_row(batch_index=1, lower=.04)]):
            with self.assertRaises(ValueError):
                results.aggregate_calibration_bins(bad)
        for field, value in [('count', -1), ('count', True), ('count', .5),
                ('sum_actual', -1.), ('sum_predicted', math.inf), ('upper', math.inf),
                ('upper', .02), ('lower', -.1), ('gamma_db', 30.),
                ('component', ''), ('batch_index', True)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                results.aggregate_calibration_bins([calibration_row(**{field:value})])
        with self.assertRaises(ValueError):
            results.aggregate_calibration_bins([calibration_row(count=0)])
        zero_prediction = results.aggregate_calibration_bins([
            calibration_row(sum_predicted=0.)])[0]
        self.assertIsNone(zero_prediction['actual_to_predicted_ratio'])
        self.assertEqual(zero_prediction['sum_actual'], 4.)

    def test_module_and_configuration_import_without_site_packages(self):
        code = ("import sys; import twc_v1_k8_ul_diagnostics_results as r; "
                "r._config(); assert 'torch' not in sys.modules; assert 'sionna' not in sys.modules")
        subprocess.run([sys.executable, '-B', '-S', '-c', code],
            cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
