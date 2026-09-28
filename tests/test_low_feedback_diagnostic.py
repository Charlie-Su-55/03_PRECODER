"""Standard-library-only checks; no model, torch import, checkpoint or evaluation."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_low_feedback_snr as diagnostic
from train_proposal_specialist import specialist_config
from train_proposal_snr_robust import robust_config
from feedback_link_diagnostics import FeedbackLinkProbe
from types import SimpleNamespace


class DiagnosticProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = diagnostic.diagnostic_plan(diagnostic.build_parser().parse_args([]))

    def test_exactly_eight_fixed_configs_and_distinct_paths(self):
        self.assertEqual(len(self.plan), 8)
        self.assertNotIn('universal', [e['method_key'] for e in self.plan])
        self.assertEqual([e['allocations'] for e in self.plan], [
            ((64, 0),), ((48, 64),), ((32, 128),), ((16, 192),),
            ((0, 256),), ((0, 256),), ((64, 0),), ((64, 0),),
        ])
        self.assertEqual(len({e['path'] for e in self.plan}), 8)
        self.assertTrue(all(str(e['path']).endswith('best.pth') for e in self.plan))

    def test_not_a_formal_test_seed_and_no_torch_import(self):
        self.assertNotEqual(diagnostic.SEED, 20260921)
        self.assertNotEqual(diagnostic.SEED, 500042)
        self.assertGreater(abs(diagnostic.SEED-20260921), diagnostic.SAMPLE_COUNT)
        self.assertEqual(diagnostic.BATCH_SIZE * 32, diagnostic.SAMPLE_COUNT)
        self.assertNotIn('torch', sys.modules)

    def fixture_samples(self):
        return [
            dict(method_key=e['method_key'], sample_index=i, feedback_snr_db=snr,
                 sum_rate=(10 if e['method_key']=='swin' else 5) + 2*i + snr/10)
            for e in self.plan for i in range(2) for snr in diagnostic.FB_SNRS
        ]

    def test_channel_paired_low_snr_se_and_negative_gains(self):
        rows = diagnostic.low_snr_rows(self.fixture_samples(), self.plan, 2)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertAlmostEqual(row['low_snr_rate_se'], 1.0)
            self.assertAlmostEqual(row['delta_vs_swin_se'], 0.0)
        specialist = rows[0]
        self.assertAlmostEqual(specialist['low_snr_rate_mean'], 6.5)
        self.assertAlmostEqual(specialist['delta_vs_swin_mean'], -5)
        self.assertEqual(specialist['dataset_role'], diagnostic.ROLE)

    def test_missing_sample_rejected(self):
        samples = self.fixture_samples()
        samples.pop(0)
        with self.assertRaises(KeyError):
            diagnostic.low_snr_rows(samples, self.plan, 2)

    def test_duplicate_sample_rejected(self):
        samples = self.fixture_samples()
        samples.append(samples[0])
        with self.assertRaises(ValueError):
            diagnostic.low_snr_rows(samples, self.plan, 2)

    def test_nonfinite_rates_rejected(self):
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                diagnostic.rate_statistics([0, value])

    def test_actual_training_recipe_checks(self):
        for e in self.plan:
            if e['method_key'] == 'robust_specialist':
                config = robust_config().to_dict()
            elif e['model_key'] == 'proposed':
                config = specialist_config(e['allocations'][0], 'unused_test_root').to_dict()
            else:
                config = dict(OBJECTIVE='task', SEED=42)
            self.assertTrue(diagnostic.validate_training_metadata(e, config))
            bad = copy.deepcopy(config)
            if e['model_key'] == 'proposed':
                bad['experiment']['train_mode'] = 'adaptive'
            else:
                bad['OBJECTIVE'] = 'nmse'
            with self.assertRaises(RuntimeError):
                diagnostic.validate_training_metadata(e, bad)

    def test_robust_auxiliary_loss_rejected(self):
        config = robust_config().to_dict()
        config['training_recipe']['ph2_w_mse'] = 1.0
        with self.assertRaises(RuntimeError):
            diagnostic.validate_training_metadata(self.plan[5], config)

    def test_transparency_comparison_detects_changed_rates(self):
        row = dict(method_key='x', allocation_df=0, allocation_da=256,
                   feedback_snr_db=0, sample_index=0, sum_rate=1.0,
                   nmse_db=-2.0, precoder_power=256.0)
        diagnostic.compare_probe_results([row], [dict(row)])
        with self.assertRaises(RuntimeError):
            diagnostic.compare_probe_results([row], [dict(row, sum_rate=2.0)])

    def test_physical_coverage_requires_only_active_branches(self):
        methods = [SimpleNamespace(method_key=e['method_key'], allocations=e['allocations']) for e in self.plan]
        probe = FeedbackLinkProbe(methods, {})
        for e in self.plan:
            for branch, width in zip(('FDMA', 'AS'), e['allocations'][0]):
                if width:
                    for snr in diagnostic.FB_SNRS:
                        probe.rows.append(dict(method_key=e['method_key'], branch=branch,
                                               feedback_snr_db=snr, batch_index=0))
        self.assertEqual(len(probe.rows), 66)
        probe.verify_coverage(1, diagnostic.FB_SNRS)
        probe.rows.pop()
        with self.assertRaises(RuntimeError):
            probe.verify_coverage(1, diagnostic.FB_SNRS)


if __name__ == '__main__':
    unittest.main()
