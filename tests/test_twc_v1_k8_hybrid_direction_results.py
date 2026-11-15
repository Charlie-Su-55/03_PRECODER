"""Independent small-data checks for paired, unequal-budget comparisons."""
import copy
import gzip
from pathlib import Path
import tempfile
import unittest

import twc_v1_k8_hybrid_direction_results as results


class CompleteMUStatisticsTests(unittest.TestCase):
    def rows(self):
        rows = []
        cfg = results._config()
        for variant in cfg.EVAL_VARIANTS:
            for gamma in cfg.GAMMAS:
                for sample in range(8):
                    batch, within = divmod(sample, 4)
                    baseline = 10+gamma*.25+batch*.5+within*.01
                    offset = 1+2*batch if variant == 'hybrid_direction' else .2 if variant == 'hybrid_source' else 0
                    rows.append(dict(variant=variant, arm=results.physical_arm(variant), role='final',
                        gamma_db=float(gamma), sample_index=sample, batch_index=batch,
                        within_batch=within, signed_term=-sample*.01,
                        **{name: baseline+offset for name in results.REQUIRED_METRICS}))
        return rows

    def test_paired_gamma_average_precedes_batch_uncertainty(self):
        rows = self.rows()
        output = results.paired_comparisons(rows, samples=8)
        row = next(r for r in output if r['comparison'] == 'hybrid_direction_minus_fdma_frozen'
            and r['scope'] == 'all_six' and r['metric'] == 'sum_rate_bps_hz')
        self.assertEqual(row['n_samples'], 8)
        self.assertEqual(row['n_batches'], 2)
        self.assertEqual(row['n_gamma_points'], 6)
        self.assertAlmostEqual(row['mean'], 2.)
        self.assertAlmostEqual(row['batch_se'], 1.)
        self.assertAlmostEqual(row['batch_ci95_low'], .04)
        self.assertAlmostEqual(row['batch_ci95_high'], 3.96)
        source = next(r for r in output if r['comparison'] == 'hybrid_direction_minus_hybrid_source'
            and r['scope'] == 'all_six' and r['metric'] == 'sum_rate_bps_hz')
        self.assertAlmostEqual(source['mean'], 1.8)
        summaries = results.summarize_rows(rows, samples=8)
        nominal = next(r for r in summaries if r['variant'] == 'fdma_frozen'
            and r['scope'] == 'all_six' and r['metric'] == 'sum_rate_bps_hz')
        self.assertAlmostEqual(nominal['mean'], 13.39)
        self.assertAlmostEqual(nominal['batch_se'], .25)
        self.assertEqual(len([r for r in output if r['comparison'] == 'hybrid_direction_minus_fdma_frozen'
            and r['scope'] == 'point' and r['metric'] == 'sum_rate_bps_hz']), 6)

    def test_incomplete_duplicate_nonfinite_partial_or_wrong_role_data_are_rejected(self):
        rows = self.rows()
        cases = [rows[:-1], rows+[rows[0]], copy.deepcopy(rows), copy.deepcopy(rows), copy.deepcopy(rows)]
        cases[2][0]['sum_rate_bps_hz'] = float('inf')
        cases[3][0].pop('signed_term')
        cases[4][0]['role'] = 'test'
        for bad in cases:
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)

    def test_hybrid_only_fields_remain_absent_in_cross_arm_comparisons(self):
        rows = self.rows()
        for row in rows:
            if row['arm'] == 'hybrid_g4': row['shared_soft_error'] = .03
        output = results.paired_comparisons(rows, samples=8)
        self.assertTrue(any(r['metric'] == 'shared_soft_error' and
            r['comparison'] == 'hybrid_direction_minus_hybrid_source' for r in output))
        self.assertFalse(any(r['metric'] == 'shared_soft_error' and
            r['comparison'] == 'hybrid_direction_minus_fdma_frozen' for r in output))

    def test_real_gzip_scalar_roundtrip_preserves_missing_physical_components(self):
        from evaluate_twc_v1_k8_hybrid_direction import _csv
        rows = self.rows()
        for row in rows:
            if row['arm'] == 'hybrid_g4': row['shared_soft_error'] = .03
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'samples.csv.gz'
            _csv(path, rows)
            reread = results.read_raw_csv(path)
            self.assertEqual(rows, reread)
            results.validate_rows(reread, samples=8)


if __name__ == '__main__': unittest.main()
