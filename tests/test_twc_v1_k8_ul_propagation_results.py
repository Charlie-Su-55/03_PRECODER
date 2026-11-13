"""Hand-computed, standard-library propagation-statistics checks."""
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
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import twc_v1_k8_ul_propagation_results as results


ARMS = ('fdma', 'hybrid_g4')
GAMMAS = (0., 5., 10., 15., 20., 25.)


def propagation_rows(samples=8):
    rows = []
    for arm in ARMS:
        for gi, gamma in enumerate(GAMMAS):
            for i in range(samples):
                # The repeated-gamma paired delta is (i+1)*(gi+1).
                actual, reported = 100.+i, 1.+.01*gi
                propagated = reported+(i+1)*(gi+1)
                row = dict(arm=arm, gamma_db=gamma, sample_index=i,
                    batch_index=i//4, within_batch=i % 4, soft_nmse_linear=.01 if i % 2 == 0 else 1.,
                    soft_nominal_scaled_actual_mse=actual,
                    soft_nominal_scaled_reported_variance=reported,
                    soft_nominal_scaled_propagated_variance=propagated,
                    soft_actual_to_reported_mse_ratio=actual/reported,
                    soft_actual_to_propagated_mse_ratio=actual/propagated,
                    soft_nominal_scaled_realized_bias_ce_cross=-(i+1)*(gi+1),
                    soft_nominal_scaled_conditional_bias_ce_cross=-.5)
                for prefix in ('soft_key', 'soft_shared'):
                    for suffix, value in [('actual_mse', actual), ('reported_variance', reported),
                                          ('propagated_variance', propagated)]:
                        row[prefix+'_nominal_scaled_'+suffix] = None if arm == 'fdma' else value
                rows.append(row)
    return rows


def find(rows, *, arm='fdma', component='soft', contrast='propagated_minus_reported_variance',
         scope='all_six', gamma=None):
    return next(row for row in rows if row['arm'] == arm and row['component'] == component
                and row['contrast'] == contrast and row['scope'] == scope and row['gamma_db'] == gamma)


def bin_row(**changes):
    result = dict(arm='hybrid_g4', gamma_db=0., component='propagated_soft_shared',
        bin_index=2, lower=.03, upper=.1, batch_index=0,
        count=2, sum_actual=4., sum_predicted=2.)
    result.update(changes)
    return result


class PropagationStatisticsTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(results, '_config', return_value=SimpleNamespace(ARMS=ARMS, GAMMAS=GAMMAS))
        patch.start()
        self.addCleanup(patch.stop)

    def test_complete_mu_difference_and_gamma_average_precede_batch_interval(self):
        output = results.paired_comparisons(propagation_rows(), samples=8)
        for arm in ARMS:
            actual = find(output, arm=arm)
            self.assertAlmostEqual(actual['mean'], 15.75)
            self.assertAlmostEqual(actual['batch_se'], 7.)
            self.assertAlmostEqual(actual['batch_ci95_low'], 2.03)
            self.assertAlmostEqual(actual['batch_ci95_high'], 29.47)
            expected = statistics.stdev([3.5*(i+1) for i in range(8)])/math.sqrt(8)
            self.assertAlmostEqual(actual['sample_se'], expected)
            self.assertEqual((actual['n_samples'], actual['n_batches'], actual['n_gamma_points']), (8, 2, 6))
            low = find(output, arm=arm, scope='low_0_5_10')
            self.assertAlmostEqual(low['mean'], 9.)
            self.assertAlmostEqual(low['batch_se'], 4.)
        self.assertFalse(any(row['arm'] == 'fdma' and row['component'] != 'soft' for row in output))
        self.assertEqual({row['metric'] for row in output}, {'nominal_scaled_mse'})
        self.assertEqual({row['contrast'] for row in output}, {row[0] for row in results.CONTRASTS})

    def test_summary_scopes_use_same_independent_mu_and_preserve_negative_cross(self):
        summary = results.summarize_rows(propagation_rows(), samples=8)
        self.assertEqual(len(summary), 16)
        for arm in ARMS:
            points = [row for row in summary if row['arm'] == arm and row['scope'] == 'point']
            self.assertEqual([row['gamma_db'] for row in points], list(GAMMAS))
            whole = next(row for row in summary if row['arm'] == arm and row['scope'] == 'all_six')
            self.assertEqual((whole['n_samples'], whole['n_batches'], whole['n_gamma_points']), (8, 2, 6))
            self.assertIsNone(whole['gamma_db'])
            self.assertAlmostEqual(whole['soft_nominal_scaled_realized_bias_ce_cross_mean'], -15.75)
            self.assertAlmostEqual(whole['soft_nominal_scaled_realized_bias_ce_cross_batch_se'], 7.)
            self.assertAlmostEqual(whole['soft_nominal_scaled_conditional_bias_ce_cross_mean'], -.5)

    def test_linear_mean_nmse_and_zero_error_have_no_negative_infinite_db(self):
        rows = propagation_rows()
        for row in rows:
            row['zero_nmse_linear'] = 0.
        summary = results.summarize_rows(rows, samples=8)
        for row in summary:
            self.assertAlmostEqual(row['soft_nmse_linear_mean'], .505)
            self.assertAlmostEqual(row['soft_nmse_db_from_linear_mean'], 10*math.log10(.505))
            self.assertIsNone(row['zero_nmse_db_from_linear_mean'])
            self.assertEqual(row['soft_nmse_linear_batch_se'], 0.)
        json.dumps(summary, allow_nan=False)

    def test_ratio_summary_is_mean_per_mu_ratio_not_ratio_of_means(self):
        rows = propagation_rows()
        for row in rows:
            i = row['sample_index']
            row['soft_nominal_scaled_actual_mse'] = float(i+1)
            row['soft_nominal_scaled_reported_variance'] = float((i+1)**2)
            row['soft_actual_to_reported_mse_ratio'] = 1./(i+1)
        summary = results.summarize_rows(rows, samples=8)
        row = next(row for row in summary if row['arm'] == 'fdma' and row['scope'] == 'all_six')
        ratio = row['soft_actual_to_reported_mse_ratio_mean']
        self.assertAlmostEqual(ratio, statistics.mean([1./(i+1) for i in range(8)]))
        self.assertNotAlmostEqual(ratio, row['soft_nominal_scaled_actual_mse_mean']/row['soft_nominal_scaled_reported_variance_mean'])

    def test_signed_negative_calibration_differences_are_retained(self):
        rows = propagation_rows()
        for row in rows:
            row['soft_nominal_scaled_propagated_variance'] = .1
        output = results.paired_comparisons(rows, samples=8)
        actual = find(output)
        self.assertLess(actual['mean'], 0.)
        self.assertLess(actual['batch_ci95_high'], 0.)

    def test_single_batch_uncertainty_is_missing_and_order_does_not_matter(self):
        rows = propagation_rows(4)
        expected = results.paired_comparisons(rows, samples=4)
        self.assertTrue(all(row['batch_se'] is None and row['batch_ci95_low'] is None for row in expected))
        random.Random(71).shuffle(rows)
        self.assertEqual(results.paired_comparisons(rows, samples=4), expected)

    def test_missing_duplicate_bad_roles_and_mu_identities_are_rejected(self):
        rows = propagation_rows()
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)
        for field, value in [('arm', 'unknown'), ('gamma_db', 30.), ('gamma_db', True),
                ('sample_index', True), ('sample_index', 8), ('batch_index', 1), ('within_batch', 4)]:
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)
        for samples, batch in [(0, 4), (True, 4), (7, 4), (8, True), (8, 0)]:
            with self.assertRaises(ValueError):
                results.validate_rows(rows, samples=samples, batch_size=batch)

    def test_missing_required_partial_optional_negative_power_and_nonfinite_fail(self):
        rows = propagation_rows()
        for metric, value in [('soft_nmse_linear', None), ('soft_nmse_linear', -1.),
                ('soft_nmse_linear', math.nan), ('soft_nominal_scaled_actual_mse', -1.),
                ('soft_nominal_scaled_reported_variance', math.inf),
                ('soft_nominal_scaled_realized_ce_power', -1.), ('debug_scalar', 'bad')]:
            bad = copy.deepcopy(rows)
            bad[0][metric] = value
            with self.subTest(metric=metric, value=value), self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)
        bad = copy.deepcopy(rows)
        bad[0]['new_optional'] = 1.
        with self.assertRaisesRegex(ValueError, 'Partially missing'):
            results.validate_rows(bad, samples=8)
        bad = copy.deepcopy(rows)
        del bad[0]['soft_nominal_scaled_propagated_variance']
        with self.assertRaisesRegex(ValueError, 'Incomplete per-MU'):
            results.validate_rows(bad, samples=8)

    def test_plain_and_gzip_csv_roundtrip_preserve_raw_ids_and_optional_blanks(self):
        rows = propagation_rows()
        with tempfile.TemporaryDirectory() as temp:
            for name in ('samples.csv', 'samples.csv.gz'):
                path = Path(temp)/name
                opener = gzip.open if name.endswith('.gz') else open
                with opener(path, 'wt', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                restored = results.read_raw_csv(path)
                self.assertEqual(restored, rows)
                results.validate_rows(restored, samples=8)

    def test_bins_pool_reported_and_propagated_separately_without_coordinate_ci(self):
        rows = [bin_row(), bin_row(batch_index=1, count=3, sum_actual=8., sum_predicted=4.),
                bin_row(component='reported_soft_shared'),
                bin_row(bin_index=0, lower=0., upper=.01, count=0, sum_actual=0., sum_predicted=0.)]
        pooled = results.aggregate_calibration_bins(rows)
        actual = next(row for row in pooled if row['component'] == 'propagated_soft_shared' and row['bin_index'] == 2)
        self.assertEqual((actual['count'], actual['sum_actual'], actual['sum_predicted']), (5, 12., 6.))
        self.assertEqual(actual['actual_to_predicted_ratio'], 2.)
        self.assertEqual(actual['mean_actual'], 2.4)
        empty = next(row for row in pooled if row['bin_index'] == 0)
        self.assertIsNone(empty['actual_to_predicted_ratio'])
        self.assertFalse(any('variant' in row or any('ci95' in key or key.endswith('_se') for key in row) for row in pooled))
        self.assertEqual(results.aggregate_calibration_bins(reversed(rows)), pooled)
        json.dumps(pooled, allow_nan=False)

    def test_bin_duplicates_bad_boundaries_bad_sums_and_variants_fail(self):
        for bad in ([bin_row(), bin_row()], [bin_row(), bin_row(batch_index=1, lower=.04)],
                    [bin_row(variant='estimated_ul')]):
            with self.assertRaises(ValueError):
                results.aggregate_calibration_bins(bad)
        for field, value in [('count', True), ('count', -1), ('sum_actual', -1.),
                ('sum_predicted', math.inf), ('upper', math.inf), ('upper', .02),
                ('arm', 'unknown'), ('gamma_db', True), ('gamma_db', 30.)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                results.aggregate_calibration_bins([bin_row(**{field:value})])
        zero = results.aggregate_calibration_bins([bin_row(sum_predicted=0.)])[0]
        self.assertIsNone(zero['actual_to_predicted_ratio'])

    def test_import_and_moments_work_without_site_packages_or_config_import(self):
        code = ('import sys; import twc_v1_k8_ul_propagation_results as r; '
                'assert r.moments([1.,3.])["mean"]==2.; '
                'assert "torch" not in sys.modules; assert "sionna" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-S', '-c', code],
            cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
