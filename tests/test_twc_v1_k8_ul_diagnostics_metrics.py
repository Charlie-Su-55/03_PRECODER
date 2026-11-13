"""True small CPU tensor checks for read-only uplink mechanism diagnostics."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


def module(name, relative):
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / relative
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


@unittest.skipIf(torch is None, 'Actual PyTorch is unavailable; no stubs')
class MechanismMetricsTests(unittest.TestCase):
    def setUp(self):
        self.m = module('_ul_mechanism_metrics_test', 'twc_v1_k8_ul_diagnostics_metrics.py')
        self.layout = SimpleNamespace(key_positions=((0,), (1,)),
            groups=((0, 1),), shared_positions_by_group=((2, 3),))
        self.prior = torch.tensor([[1., 2., 4.], [4., 8., 16.]])

    def recovery(self):
        truth = torch.ones((2, 2, 3), dtype=torch.complex64)
        actual = torch.tensor([[[1., 2., 3.], [4., 5., 6.]],
                               [[2., 3., 4.], [5., 6., 7.]]])
        variance = torch.tensor([[[.5, 1., 2.], [2., 4., 8.]]]).expand(2, -1, -1).clone()
        # Padding is deliberately unusable; only the real active support is scored.
        soft = torch.cat((truth + actual.sqrt().to(torch.complex64),
            torch.full((2, 2, 4), complex(float('nan'), 0), dtype=torch.complex64)), -1)
        return SimpleNamespace(active=truth), dict(soft=soft, variance=variance,
            reliability=torch.cat((variance.reciprocal(), torch.zeros((2, 2, 4))), -1)), actual

    def channels(self, collinear=False):
        h = torch.zeros((2, 2, 3, 4), dtype=torch.complex64)
        h[:, 0, 0] = 1j
        h[:, 1, 0 if collinear else 1] = -1
        return h

    def test_recovery_uses_historical_coordinate_power_not_current_truth_power(self):
        wave, recovered, actual = self.recovery()
        metrics = self.m.recovery_metrics(wave, recovered, self.prior, self.layout)
        expected_actual = (actual / self.prior).mean((1, 2))
        expected_predicted = (recovered['variance'] / self.prior).mean((1, 2))
        torch.testing.assert_close(metrics['soft_nominal_scaled_actual_mse'], expected_actual)
        torch.testing.assert_close(metrics['soft_nominal_scaled_predicted_variance'], expected_predicted)
        torch.testing.assert_close(metrics['soft_actual_to_predicted_mse_ratio'], expected_actual / expected_predicted)
        torch.testing.assert_close(metrics['soft_nmse_linear'], actual.mean((1, 2)))
        torch.testing.assert_close(metrics['soft_key_nmse_linear'], actual[..., :1].mean((1, 2)))
        torch.testing.assert_close(metrics['soft_shared_nmse_linear'], actual[..., 1:].mean((1, 2)))

    def test_calibration_bins_keep_exact_point_sums_and_finite_json(self):
        wave, recovered, actual = self.recovery()
        rows = self.m.recovery_calibration_bins(wave, recovered, self.prior, self.layout)
        selected = [row for row in rows if row['component'] == 'soft_all']
        self.assertEqual(sum(row['count'] for row in selected), 12)
        self.assertAlmostEqual(sum(row['sum_actual'] for row in selected),
            float((actual / self.prior).double().sum()), places=5)
        self.assertAlmostEqual(sum(row['sum_predicted'] for row in selected), 6.)
        self.assertIsNone(selected[-1]['upper'])
        json.dumps(rows, allow_nan=False)

    def test_soft_nmse_averages_user_ratios_before_pooling_heterogeneous_energy(self):
        truth = torch.tensor([[[1., 1., 1.], [3., 2., 2.]]], dtype=torch.complex64)
        error = torch.tensor([[[1., 2., 2.], [1., 2., 2.]]])
        variance = torch.ones((1, 2, 3))
        recovered = dict(soft=truth + error.sqrt(), variance=variance,
            reliability=variance.reciprocal())
        values = self.m.recovery_metrics(SimpleNamespace(active=truth),
            recovered, self.prior, self.layout)
        # Per-user key energies are 1 and 9; errors are 1 each. A pooled
        # energy ratio would be 2/10 instead of the required (1+1/9)/2.
        torch.testing.assert_close(values['soft_key_nmse_linear'], torch.tensor([5 / 9]))
        self.assertNotAlmostEqual(float(values['soft_key_nmse_linear'][0]), 2 / 10)
        # Shared energies are 2 and 8, with error energy 4 per user.
        torch.testing.assert_close(values['soft_shared_nmse_linear'], torch.tensor([1.25]))
        self.assertNotAlmostEqual(float(values['soft_shared_nmse_linear'][0]), 8 / 10)
        torch.testing.assert_close(values['soft_nmse_linear'], torch.tensor([(5 / 3 + 5 / 17) / 2]))
        # Nominal-power-scaled calibration retains coordinate averaging.
        torch.testing.assert_close(values['soft_nominal_scaled_actual_mse'],
            (error / self.prior).mean((1, 2)))

    def test_bin_boundaries_are_lower_inclusive_and_have_overflow(self):
        rows = self.m._bins(torch.tensor([1., 2., 3., 4.]),
            torch.tensor([0., .5, 1., 9.]), 'known', (0., .5, 1.))
        self.assertEqual([row['count'] for row in rows], [1, 1, 2])
        self.assertEqual(rows[-1]['sum_actual'], 7.)
        for invalid in ((.1, 1.), (0., 0.), (0., float('inf'))):
            with self.assertRaises(ValueError):
                self.m._bins(torch.ones(1), torch.ones(1), 'bad', invalid)

    def test_orthogonal_receive_geometry_and_power_weighted_condition(self):
        h = self.channels()
        metrics = self.m.channel_metrics(h, h.clone(), torch.zeros((2, 2, 4)), self.layout, self.prior)
        torch.testing.assert_close(metrics['shared_channel_coherence_squared_mean'], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(metrics['shared_weighted_gram_condition_capped_mean'], torch.full((2,), 4., dtype=torch.float64))
        self.assertTrue(torch.equal(metrics['shared_weighted_gram_near_singular_fraction'], torch.zeros(2)))
        self.assertTrue(torch.equal(metrics['ul_ce_nmse_linear'], torch.zeros(2)))
        self.assertTrue(torch.equal(metrics['ul_ce_zero_predicted_variance_fraction'], torch.ones(2)))

    def test_collinear_receive_geometry_is_explicitly_capped_and_flagged(self):
        h = self.channels(collinear=True)
        metrics = self.m.channel_metrics(h, h, torch.zeros((2, 2, 4)), self.layout,
            self.prior, condition_cap=1e6)
        torch.testing.assert_close(metrics['shared_channel_coherence_squared_mean'], torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(metrics['shared_weighted_gram_condition_capped_mean'], torch.full((2,), 1e6, dtype=torch.float64))
        self.assertTrue(torch.equal(metrics['shared_weighted_gram_near_singular_fraction'], torch.ones(2)))

    def test_complex_spatial_coherence_uses_hermitian_inner_product(self):
        h = torch.zeros((2, 2, 3, 4), dtype=torch.complex64)
        scale = 2 ** -.5
        h[:, 0, 0], h[:, 0, 1] = scale, 1j * scale
        h[:, 1, 0], h[:, 1, 1] = 1j * scale, scale
        metrics = self.m.channel_metrics(h, h, torch.zeros((2, 2, 4)), self.layout, self.prior)
        torch.testing.assert_close(metrics['shared_channel_coherence_squared_mean'],
            torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(metrics['shared_weighted_gram_condition_capped_mean'],
            torch.full((2,), 4., dtype=torch.float64))

    def test_actual_ce_error_is_per_antenna_and_uses_correct_tone_support(self):
        truth = self.channels()
        estimate = truth.clone()
        estimate[:, 0, :, 2:] += 1j
        estimate[:, 1, :, 2:] += 2
        modelled = torch.full((2, 2, 4), .3)
        metrics = self.m.channel_metrics(estimate, truth, modelled, self.layout, self.prior)
        torch.testing.assert_close(metrics['ul_ce_key_actual_mse'], torch.zeros(2))
        torch.testing.assert_close(metrics['ul_ce_shared_actual_mse'], torch.full((2,), 2.5))
        torch.testing.assert_close(metrics['ul_ce_shared_predicted_variance'], torch.full((2,), .3))
        rows = self.m.channel_calibration_bins(estimate, truth, modelled, self.layout)
        shared = [row for row in rows if row['component'] == 'ce_shared']
        self.assertEqual(sum(row['count'] for row in shared), 8)
        self.assertAlmostEqual(sum(row['sum_actual'] for row in shared), 20.)

    def test_true_channel_zero_error_bins_have_no_infinite_ratio(self):
        truth = self.channels()
        rows = self.m.channel_calibration_bins(truth, truth, torch.zeros((2, 2, 4)), self.layout)
        self.assertTrue(all(row['sum_actual'] == row['sum_predicted'] == 0 for row in rows))
        json.dumps(rows, allow_nan=False)

    def test_metrics_do_not_change_inputs_or_autograd_flags(self):
        wave, recovered, _ = self.recovery()
        wave.active.requires_grad_()
        before = wave.active.detach().clone(), self.prior.clone(), recovered['variance'].clone()
        metrics = self.m.recovery_metrics(wave, recovered, self.prior, self.layout)
        self.assertTrue(wave.active.requires_grad)
        self.assertTrue(torch.equal(wave.active, before[0]))
        self.assertTrue(torch.equal(self.prior, before[1]))
        self.assertTrue(torch.equal(recovered['variance'], before[2]))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        h = self.channels()
        h_before = h.clone()
        self.m.channel_metrics(h, h, torch.zeros((2, 2, 4)), self.layout, self.prior)
        self.assertTrue(torch.equal(h, h_before))

    def test_invalid_variance_or_inverse_reliability_is_rejected(self):
        wave, recovered, _ = self.recovery()
        recovered['reliability'][..., 0] *= 2
        with self.assertRaisesRegex(ValueError, 'inverse diagonal'):
            self.m.recovery_metrics(wave, recovered, self.prior, self.layout)
        recovered.pop('reliability')
        recovered['variance'][0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'Positive active'):
            self.m.recovery_metrics(wave, recovered, self.prior, self.layout)

    def test_fdma_reports_only_its_active_coordinates(self):
        wave, recovered, _ = self.recovery()
        layout = SimpleNamespace(data_positions=((0, 1, 2), (1, 2, 3)))
        metrics = self.m.recovery_metrics(wave, recovered, self.prior, layout)
        self.assertIn('soft_nmse_linear', metrics)
        self.assertNotIn('soft_shared_nmse_linear', metrics)
        h = self.channels()
        channel = self.m.channel_metrics(h, h, torch.zeros((2, 2, 4)), layout, self.prior)
        self.assertNotIn('shared_channel_coherence_squared_mean', channel)

    def test_real_existing_receiver_variance_is_scored_without_mocking(self):
        physics = module('_ul_diagnostic_existing_physics', 'utils/twc_v1_k8_icc_aligned_physics.py')
        generator = torch.Generator().manual_seed(721)
        raw = torch.complex(torch.randn((1, 8, 216), generator=generator),
                            torch.randn((1, 8, 216), generator=generator))
        h = torch.zeros((1, 8, 32, 256), dtype=torch.complex64)
        for user in range(8):
            h[:, user, user] = 1j if user % 2 else 1
        xi = torch.zeros((1, 32, 256), dtype=torch.complex64)
        for arm in ('fdma', 'hybrid_g4'):
            layout = physics.make_layout(arm)
            wave = physics.build_waveform(raw, layout)
            statistics = physics.ReceiverStatistics(layout, torch.zeros((6, 8, 256)))
            prior = statistics.nominal_power()
            ce = statistics.ce_variance(25.)
            observation = physics.transmit(h, wave.x_grid, xi, 25.)
            before = {name: value.clone() for name, value in statistics.state_dict().items()}
            recovered = physics.recover(observation, h, layout, prior, ce, 25.)
            metrics = self.m.recovery_metrics(wave, recovered, prior, layout)
            self.assertLess(float(metrics['soft_nmse_linear'][0]), 1e-5)
            channel = self.m.channel_metrics(h, h, ce, layout, prior)
            self.assertEqual(float(channel['ul_ce_nmse_linear'][0]), 0.)
            self.assertTrue(all(torch.equal(value, before[name])
                                for name, value in statistics.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
