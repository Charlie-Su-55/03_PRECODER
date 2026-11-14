"""Independent analytic complete-MU and paired-statistics checks."""
import copy
import csv
import gzip
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import twc_v1_k8_ul_calibration_results as results


CFG = SimpleNamespace(ARMS=('fdma', 'hybrid_g4'),
    VARIANTS=('original', 'reliability_only', 'moment_receiver'), GAMMAS=(0, 5, 10, 15, 20, 25))


def raw_rows(samples=8, batch_size=4):
    rows = []
    for arm in CFG.ARMS:
        for variant in CFG.VARIANTS:
            for gamma in CFG.GAMMAS:
                for i in range(samples):
                    gain = {('fdma', 'original'): 0., ('fdma', 'reliability_only'): .03,
                        ('fdma', 'moment_receiver'): .04, ('hybrid_g4', 'original'): 0.,
                        ('hybrid_g4', 'reliability_only'): .05, ('hybrid_g4', 'moment_receiver'): .01}[arm, variant]
                    error = 1.+gamma*.01+i*.1-(.05 if variant == 'moment_receiver' else 0.)
                    row = dict(role='validation', arm=arm, variant=variant, gamma_db=float(gamma),
                        sample_index=i, batch_index=i//batch_size, within_batch=i%batch_size,
                        soft_nmse_linear=error, soft_nominal_scaled_actual_mse=error,
                        soft_nominal_scaled_predicted_variance=.5,
                        sum_rate_bps_hz=10.+gamma*.1+i*.2-(2. if arm == 'hybrid_g4' else 0.)+gain*(i+1),
                        dl_nmse_linear=.01+gamma*.001,
                        interference_power_mean=.1+gamma*.001,
                        ce_relative_projection_bias_real=-.25+gamma*.001+i*.02)
                    if arm == 'hybrid_g4':
                        row['shared_estimated_matrix_condition_mean'] = 1.+i*.1
                    rows.append(row)
    return rows


class CalibrationResultsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = patch.object(results, '_config', return_value=CFG)
        self.cfg.start()
        self.addCleanup(self.cfg.stop)
        self.rows = raw_rows()

    def test_gamma_average_precedes_mu_and_physical_batch_statistics(self):
        summary = results.summarize_rows(self.rows, samples=8)
        self.assertEqual(len(summary), 48)
        all_six = next(row for row in summary if row['arm']=='fdma'
            and row['variant']=='original' and row['scope']=='all_six')
        self.assertEqual((all_six['n_samples'], all_six['n_batches'], all_six['n_gamma_points']), (8, 2, 6))
        self.assertAlmostEqual(all_six['sum_rate_bps_hz_mean'], 11.95)
        self.assertAlmostEqual(all_six['sum_rate_bps_hz_se'], math.sqrt(.03))
        self.assertAlmostEqual(all_six['sum_rate_bps_hz_batch_se'], .4)
        self.assertAlmostEqual(all_six['sum_rate_bps_hz_batch_ci95_low'], 11.95-1.96*.4)
        point = next(row for row in summary if row['arm']=='fdma'
            and row['variant']=='original' and row['scope']=='point' and row['gamma_db']==0.)
        self.assertAlmostEqual(point['sum_rate_bps_hz_se'], all_six['sum_rate_bps_hz_se'])

    def test_paired_controls_use_same_mu_and_keep_covariance(self):
        comparisons = results.paired_comparisons(self.rows, samples=8)
        row = next(r for r in comparisons if r['comparison']=='fdma_reliability_only_minus_original'
            and r['scope']=='all_six' and r['metric']=='sum_rate_bps_hz')
        self.assertAlmostEqual(row['mean'], .135)
        self.assertAlmostEqual(row['sample_se'], .03*math.sqrt(6)/math.sqrt(8))
        self.assertAlmostEqual(row['batch_se'], .06)
        gap = next(r for r in comparisons if r['comparison']=='original_hybrid_minus_fdma'
            and r['scope']=='all_six' and r['metric']=='sum_rate_bps_hz')
        self.assertAlmostEqual(gap['mean'], -2.)
        self.assertLess(gap['sample_se'], 1e-14)

    def test_differential_gain_is_paired_and_negative_results_are_retained(self):
        comparisons = results.paired_comparisons(self.rows, samples=8)
        row = next(r for r in comparisons if r['comparison']=='interaction_moment_receiver_gain_minus_original'
            and r['scope']=='all_six' and r['metric']=='sum_rate_bps_hz')
        self.assertAlmostEqual(row['mean'], -.135)
        self.assertAlmostEqual(row['batch_se'], .06)
        improvement = next(r for r in comparisons if r['comparison']=='fdma_moment_receiver_minus_original'
            and r['scope']=='all_six' and r['metric']=='soft_nmse_linear')
        self.assertAlmostEqual(improvement['mean'], -.05)

    def test_shared_geometry_is_only_compared_within_hybrid(self):
        comparisons = results.paired_comparisons(self.rows, samples=8)
        geometry = [r for r in comparisons if r['metric']=='shared_estimated_matrix_condition_mean']
        self.assertEqual(len(geometry), 16)
        self.assertTrue(all(r['arm']=='hybrid_g4' for r in geometry))
        self.assertTrue(all(r['mean']==0. for r in geometry))

    def test_linear_nmse_db_view_is_from_mean_linear_and_signed_bias_remains(self):
        summary = results.summarize_rows(self.rows, samples=8)
        row = next(r for r in summary if r['arm']=='fdma' and r['variant']=='original'
            and r['scope']=='all_six')
        self.assertAlmostEqual(row['soft_nmse_db_from_linear_mean'],
            10*math.log10(1.+.125+.35))
        self.assertLess(row['ce_relative_projection_bias_real_mean'], 0)

    def test_missing_duplicate_and_bad_batch_identity_are_rejected(self):
        bad_batch = copy.deepcopy(self.rows)
        bad_batch[0]['batch_index'] = 1
        for rows in (self.rows[:-1], self.rows+[self.rows[0]], bad_batch):
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                results.validate_rows(rows, samples=8)

    def test_calibration_test_and_unknown_conditions_cannot_enter_validation(self):
        changes = [('role', 'calibration'), ('role', 'test'), ('variant', 'oracle'),
                   ('arm', 'extra'), ('gamma_db', 30.)]
        for name, value in changes:
            rows = copy.deepcopy(self.rows)
            rows[0][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                results.summarize_rows(rows, samples=8)

    def test_partial_metric_absence_across_variants_or_gamma_is_rejected(self):
        for name in ('soft_nmse_linear', 'shared_estimated_matrix_condition_mean'):
            rows = copy.deepcopy(self.rows)
            target = next(r for r in rows if r['arm']=='hybrid_g4'
                and r['variant']=='moment_receiver' and r['gamma_db']==25.)
            target[name] = None
            with self.subTest(name=name), self.assertRaises(ValueError):
                results.paired_comparisons(rows, samples=8)

    def test_nonfinite_boolean_and_negative_power_are_refused(self):
        for name, value in (('sum_rate_bps_hz', float('inf')), ('dl_nmse_linear', float('nan')),
                            ('soft_nmse_linear', True), ('interference_power_mean', -.1)):
            rows = copy.deepcopy(self.rows)
            rows[0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                results.validate_rows(rows, samples=8)

    def test_input_order_does_not_change_statistics_or_original_rows(self):
        before = copy.deepcopy(self.rows)
        normal = results.summarize_rows(self.rows, samples=8)
        reverse = results.summarize_rows(list(reversed(self.rows)), samples=8)
        self.assertEqual(normal, reverse)
        self.assertEqual(self.rows, before)

    def test_plain_and_gzip_raw_csv_roundtrip_preserve_id_types_and_missing_geometry(self):
        fields = sorted(set().union(*(set(row) for row in self.rows)))
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.csv', '.csv.gz'):
                path = Path(directory)/('samples'+suffix)
                opener = gzip.open if suffix.endswith('.gz') else open
                with opener(path, 'wt', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(self.rows)
                loaded = results.read_raw_csv(path)
                results.validate_rows(loaded, samples=8)
                self.assertEqual(results.summarize_rows(loaded, samples=8),
                    results.summarize_rows(self.rows, samples=8))
                self.assertIsInstance(loaded[0]['sample_index'], int)
                self.assertIsNone(loaded[0]['shared_estimated_matrix_condition_mean'])

    def test_duplicate_csv_columns_and_single_incomplete_batch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.csv'
            path.write_text('arm,arm\nfdma,hybrid_g4\n')
            with self.assertRaises(ValueError):
                results.read_raw_csv(path)
        for samples, batch_size in ((3, 4), (0, 4), (8, 0), (True, 4)):
            with self.subTest(samples=samples, batch_size=batch_size), self.assertRaises(ValueError):
                results.validate_rows([], samples=samples, batch_size=batch_size)

    def test_results_import_does_not_import_torch_sionna_or_touch_data(self):
        code = ('import sys; import twc_v1_k8_ul_calibration_results; '
                'assert "torch" not in sys.modules; assert "sionna" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-S', '-c', code],
            cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
