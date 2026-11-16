"""Small independent checks of CE pairing and repeated-gamma statistics."""
import copy
import csv
import gzip
from pathlib import Path
import tempfile
import unittest

import twc_v1_k8_ce_refine_results as results


class CEStatisticsTests(unittest.TestCase):
    def rows(self, *, role='final', samples=8):
        gammas = results.GAMMAS if role == 'final' else results.MIDPOINTS
        rows = []
        for variant in results.DEFAULT_VARIANTS:
            for gamma in gammas:
                for i in range(samples):
                    batch, within = divmod(i, 4)
                    shared = 100.+20.*batch+within*.1+gamma
                    offset = (2.+2.*batch if variant == 'geometry65000' else
                        1. if variant == 'nmse65000' else 0.)
                    rows.append(dict(variant=variant, role=role, gamma_db=gamma,
                        sample_index=i, batch_index=batch, within_batch=within,
                        signed_bias=-.1*i,
                        **{name: shared+offset for name in results.REQUIRED_METRICS}))
        return rows

    def select(self, rows, *, comparison=None, variant=None, scope='all_six', metric='ul_nmse_linear'):
        return next(row for row in rows if row.get('comparison') == comparison
            and row.get('variant') == variant and row['scope'] == scope and row['metric'] == metric)

    def test_paired_complete_mu_differences_cancel_common_channel_variation(self):
        rows = self.rows()
        compared = results.paired_comparisons(rows, samples=8)
        point = self.select(compared, comparison='geometry_minus_source')
        # The two batch differences are exactly2 and4, not variations of
        # the channel-dependent100/120 baseline shared by the fixed models.
        self.assertEqual(point['mean'], 3.)
        self.assertEqual(point['batch_se'], 1.)
        self.assertAlmostEqual(point['batch_ci95_low'], 1.04)
        self.assertAlmostEqual(point['batch_ci95_high'], 4.96)
        self.assertEqual(point['n_samples'], 8)
        self.assertEqual(point['n_batches'], 2)
        self.assertEqual(point['n_gamma_points'], 6)
        control = self.select(compared, comparison='nmse_minus_source')
        self.assertEqual(control['mean'], 1.)
        self.assertEqual(control['batch_se'], 0.)
        method = self.select(compared, comparison='geometry_minus_nmse')
        self.assertEqual(method['mean'], 2.)
        self.assertEqual(method['batch_se'], 1.)

    def test_gamma_views_average_within_mu_before_uncertainty(self):
        rows = self.rows()
        for row in rows:
            # Opposite gamma effects cancel exactly in the six-view MU
            # average. Treating48 views as independent would give nonzeroSE.
            value = 50.+(row['gamma_db']-12.5)*(row['sample_index']-3.5)
            row['signed_bias'] = value
        summary = results.summarize_rows(rows, samples=8)
        pooled = self.select(summary, variant='source_ce5000', metric='signed_bias')
        self.assertEqual(pooled['mean'], 50.)
        self.assertEqual(pooled['sample_se'], 0.)
        self.assertEqual(pooled['batch_se'], 0.)
        self.assertEqual(pooled['n_samples'], 8)
        low = self.select(summary, variant='source_ce5000', scope='low_0_5_10')
        high = self.select(summary, variant='source_ce5000', scope='high_15_20_25')
        self.assertEqual(low['n_gamma_points'], 3)
        self.assertAlmostEqual(high['mean']-low['mean'], 15.)

    def test_midpoints_have_own_role_and_scope_without_inflating_sample_count(self):
        rows = self.rows(role='midpoint')
        output = results.paired_comparisons(rows, samples=8, role='midpoint')
        pooled = self.select(output, comparison='geometry_minus_source', scope='all_midpoints')
        self.assertEqual(pooled['mean'], 3.)
        self.assertEqual(pooled['batch_se'], 1.)
        self.assertEqual(pooled['n_samples'], 8)
        self.assertEqual(pooled['n_gamma_points'], 5)
        self.assertEqual({r['scope'] for r in output}, {'point', 'all_midpoints'})
        with self.assertRaises(ValueError):
            results.validate_rows(rows, samples=8)

    def test_rejects_missing_duplicate_wrong_index_nonfinite_and_partial_schema(self):
        rows = self.rows()
        bads = [rows[:-1], rows+[rows[0]]]
        changes = [('gamma_db', 2.5), ('sample_index', True), ('batch_index', 1),
            ('within_batch', 4), ('role', 'test'), ('ul_nmse_linear', -1.),
            ('gram_error', float('nan')), ('signed_bias', float('inf')),
            ('energy_ratio', True)]
        for key, value in changes:
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            bads.append(bad)
        partial = copy.deepcopy(rows); del partial[0]['signed_bias']; bads.append(partial)
        extra = copy.deepcopy(rows); extra[0]['extra_metric'] = 1.; bads.append(extra)
        for bad in bads:
            with self.subTest(last_change=bad[0]), self.assertRaises(ValueError):
                results.validate_rows(bad, samples=8)
        for kwargs in (dict(samples=True), dict(samples=7), dict(samples=8, batch_size=0),
                dict(samples=8, gammas=(0, 0)), dict(samples=8, variants=('x', 'x'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                results.validate_rows(rows, **kwargs)

    def test_one_batch_has_mean_but_no_batch_confidence_interval(self):
        summary = results.summarize_rows(self.rows(samples=4), samples=4)
        point = self.select(summary, variant='source_ce5000')
        self.assertIsNone(point['batch_se'])
        self.assertIsNone(point['batch_ci95_low'])
        self.assertIsNone(point['batch_ci95_high'])
        self.assertIsNotNone(point['sample_se'])

    def test_real_compressed_csv_roundtrip_and_duplicate_header_failure(self):
        rows = self.rows()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'samples.csv.gz'
            with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            loaded = results.read_raw_csv(path)
            self.assertEqual(rows, loaded)
            results.validate_rows(loaded, samples=8)
            with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
                stream.write('variant,variant\nx,x\n')
            with self.assertRaisesRegex(ValueError, 'Unique'):
                results.read_raw_csv(path)


if __name__ == '__main__':
    unittest.main()
