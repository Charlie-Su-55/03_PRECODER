"""Real CPU tensor checks for CE directions, geometry and soft-score semantics."""
import inspect
import math
from types import SimpleNamespace
import unittest

import torch

import twc_v1_k8_ul_calibration_metrics as metrics


class CalibrationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.layout = SimpleNamespace(key_positions=((0,), (1,)), groups=((0, 1),),
            shared_positions_by_group=((2, 3),))
        self.h = torch.zeros((2, 2, 3, 4), dtype=torch.complex64)
        self.h[:, 0, 0] = 1.
        self.h[:, 1, 1] = 2.
        self.variance = torch.ones((2, 2, 4))
        self.pilots = torch.zeros((2, 2, 4), dtype=torch.complex64)
        self.pilots[:, 0, 0] = math.sqrt(2)
        self.pilots[:, 1, 1] = math.sqrt(2)
        self.Y = (self.h*self.pilots[:, :, None]).sum(1)

    def diagnostic(self, h=None, truth=None, variance=None, Y=None, pilots=None, gamma=0.):
        return metrics.diagnostic_vectors(self.h if h is None else h,
            self.h if truth is None else truth, self.variance if variance is None else variance,
            self.Y if Y is None else Y, self.pilots if pilots is None else pilots,
            gamma, self.layout)

    def test_parallel_complex_bias_and_signed_projection_are_exact(self):
        alpha = complex(-.2, .3)
        vectors = self.diagnostic(truth=self.h*(1+alpha))
        expected_power = torch.full((2,), .13*(1.+4.)/2, dtype=torch.float64)
        torch.testing.assert_close(vectors['ce_parallel_error_power'], expected_power,
            rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(vectors['ce_perpendicular_error_power_per_antenna'],
            torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(vectors['ce_relative_projection_bias_real'],
            torch.full((2,), -.2, dtype=torch.float64), atol=2e-8, rtol=1e-6)
        torch.testing.assert_close(vectors['ce_power_weighted_relative_bias_imag'],
            torch.full((2,), .3, dtype=torch.float64), atol=2e-8, rtol=1e-6)
        self.assertLess(float(vectors['ce_directional_energy_closure_max_error'].max()), 1e-14)

    def test_orthogonal_error_and_rank_one_centered_covariance_have_known_values(self):
        error = torch.zeros_like(self.h)
        error[:, :, 2] = torch.tensor([1., -1., 1., -1.])
        vectors = self.diagnostic(truth=self.h+error)
        torch.testing.assert_close(vectors['ce_parallel_error_power'], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(vectors['ce_perpendicular_error_power_per_antenna'],
            torch.full((2,), .5, dtype=torch.float64))
        torch.testing.assert_close(vectors['ce_spatial_covariance_anisotropy'],
            torch.full((2,), math.sqrt(2/3), dtype=torch.float64))
        torch.testing.assert_close(vectors['ce_spatial_covariance_max_eigenvalue_fraction'],
            torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(vectors['ce_spatial_mean_power'], torch.zeros(2, dtype=torch.float64))

    def test_zero_ce_error_has_finite_zero_covariance_diagnostics(self):
        vectors = self.diagnostic()
        for name in ('ce_actual_mse', 'ce_spatial_covariance_anisotropy',
                     'ce_spatial_covariance_max_eigenvalue_fraction', 'pilot_residual_to_noise_ratio'):
            torch.testing.assert_close(vectors[name], torch.zeros_like(vectors[name]))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in vectors.values()))

    def test_shared_singular_values_and_gram_condition_are_distinguished(self):
        vectors = self.diagnostic()
        expected = dict(shared_estimated_normalized_correlation_mean=0.,
            shared_estimated_min_singular_value_mean=1.,
            shared_estimated_max_singular_value_mean=2.,
            shared_estimated_matrix_condition_mean=2., shared_estimated_gram_condition_mean=4.)
        for name, value in expected.items():
            torch.testing.assert_close(vectors[name], torch.full((2,), value, dtype=torch.float64))

    def test_near_collinearity_is_reported_without_a_condition_cap(self):
        h = self.h.clone()
        h[:, 1] = h[:, 0]
        h[:, 1, 1] = 1e-4
        vectors = self.diagnostic(h=h, truth=h)
        self.assertGreater(float(vectors['shared_estimated_gram_condition_mean'].min()), 1e8)
        self.assertGreater(float(vectors['shared_estimated_normalized_correlation_mean'].min()), .999)

    def test_singular_shared_geometry_refuses_truth_designed_fallback(self):
        h = self.h.clone()
        h[:, 1] = h[:, 0]
        with self.assertRaisesRegex(ValueError, 'Singular estimated shared matrix'):
            self.diagnostic(h=h, truth=self.h)

    def test_pilot_feature_masks_data_before_finiteness_and_uses_public_noise(self):
        Y = self.Y.clone()
        Y[:, :, 0] += 1.
        expected = torch.tensor([2., 20.], dtype=torch.float64)
        feature = metrics.pilot_residual_feature(Y, self.pilots, self.h, torch.tensor([0., 10.]))
        torch.testing.assert_close(feature, expected.to(feature.dtype))
        Y[:, :, 2:] = complex(float('nan'), float('inf'))
        torch.testing.assert_close(metrics.pilot_residual_feature(Y, self.pilots, self.h,
            torch.tensor([0., 10.])), expected.to(feature.dtype))
        estimated = self.h.clone()
        estimated[..., 2:] = complex(float('nan'), 0)
        torch.testing.assert_close(metrics.pilot_residual_feature(Y, self.pilots, estimated,
            torch.tensor([0., 10.])), expected.to(feature.dtype))
        Y[:, 0, 0] = complex(float('nan'), 0)
        with self.assertRaisesRegex(ValueError, '[Ff]inite'):
            metrics.pilot_residual_feature(Y, self.pilots, self.h, 0.)
        self.assertNotIn('true_ul', inspect.signature(metrics.pilot_residual_feature).parameters)

    def test_fdma_has_no_invented_shared_geometry(self):
        fdma = SimpleNamespace(data_positions=((0, 2), (1, 3)))
        result = metrics.diagnostic_vectors(self.h, self.h, self.variance, self.Y,
            self.pilots, 0., fdma)
        self.assertFalse(any(name.startswith('shared_') for name in result))

    def test_invalid_ce_inputs_and_missing_pilots_are_refused(self):
        cases = [dict(variance=-self.variance), dict(variance=self.variance[:, :, :2]),
                 dict(h=self.h[:, :, :1]), dict(gamma=True), dict(pilots=torch.zeros_like(self.pilots))]
        for changes in cases:
            with self.subTest(changes=tuple(changes)), self.assertRaises((ValueError, TypeError)):
                self.diagnostic(**changes)
        zero = self.h.clone()
        zero[:, 0] = 0
        with self.assertRaisesRegex(ValueError, 'Identifiable estimated channel'):
            self.diagnostic(h=zero)

    def test_scoring_is_detached_and_never_changes_original_tensors(self):
        inputs = (self.h, self.variance, self.Y, self.pilots)
        before = [value.clone() for value in inputs]
        h = self.h.clone().requires_grad_()
        vectors = self.diagnostic(h=h, truth=h.detach()*1.1)
        self.assertTrue(all(value.shape == (2,) and not value.requires_grad for value in vectors.values()))
        for original, saved in zip(inputs, before):
            torch.testing.assert_close(original, saved, rtol=0, atol=0)

    def recovery(self):
        truth = torch.tensor([[[1., 1., 1.], [3., 2., 2.]]], dtype=torch.complex64)
        error = torch.tensor([[[1., 2., 2.], [1., 2., 2.]]])
        prior = torch.tensor([[1., 2., 4.], [4., 8., 16.]])
        variance = torch.ones((1, 2, 3))
        soft = torch.cat((truth+error.sqrt(),
            torch.full((1, 2, 2), complex(float('nan'), 0))), -1)
        recovered = dict(soft=soft, variance=variance, prior=prior,
            reliability=torch.cat((variance.reciprocal(), torch.zeros((1, 2, 2))), -1))
        return recovered, truth, error

    def test_recovery_preserves_user_nmse_and_historical_power_normalization(self):
        recovered, truth, error = self.recovery()
        values = metrics.recovery_vectors(recovered, truth, self.layout)
        torch.testing.assert_close(values['soft_key_nmse_linear'], torch.tensor([5/9]))
        torch.testing.assert_close(values['soft_shared_nmse_linear'], torch.tensor([1.25]))
        torch.testing.assert_close(values['soft_nominal_scaled_actual_mse'],
            (error/recovered['prior']).mean((1, 2)))
        torch.testing.assert_close(values['soft_nominal_scaled_reported_variance'],
            values['soft_nominal_scaled_predicted_variance'])
        torch.testing.assert_close(values['soft_actual_to_reported_mse_ratio'],
            values['soft_actual_to_predicted_mse_ratio'])

    def test_recovery_rejects_mutated_precision_or_current_power_as_bad_prior(self):
        recovered, truth, _ = self.recovery()
        recovered['reliability'][..., :3] *= 2
        with self.assertRaisesRegex(ValueError, 'Reliability'):
            metrics.recovery_vectors(recovered, truth, self.layout)
        recovered, truth, _ = self.recovery()
        recovered['prior'][0, 0] = 0
        with self.assertRaises(ValueError):
            metrics.recovery_vectors(recovered, truth, self.layout)

    def test_production_k8_nt32_nsc256_shapes_with_real_paid_pilots(self):
        from models.twc_v1_k8_icc_aligned import public_layout
        from ul_diagnostics_runtime import physics
        generator = torch.Generator().manual_seed(321)
        h = torch.complex(torch.randn(1, 8, 32, 256, generator=generator),
            torch.randn(1, 8, 32, 256, generator=generator))/math.sqrt(32)
        error = .02*torch.complex(torch.randn(h.shape, generator=generator),
            torch.randn(h.shape, generator=generator))
        pilots = physics().known_pilots(1)
        Y = ((h+error)*pilots[:, :, None]).sum(1)
        for arm in ('fdma', 'hybrid_g4'):
            with self.subTest(arm=arm):
                values = metrics.diagnostic_vectors(h, h+error,
                    torch.full((1, 8, 256), .001), Y, pilots,
                    torch.tensor([10.]), public_layout(arm))
                self.assertTrue(all(value.shape==(1,) and bool(torch.isfinite(value).all())
                    for value in values.values()))
                self.assertEqual('shared_estimated_matrix_condition_mean' in values,
                    arm=='hybrid_g4')


if __name__ == '__main__':
    unittest.main()
