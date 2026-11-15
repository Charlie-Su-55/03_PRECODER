"""Actual CPU projective-direction invariance and rate/auxiliary gradients."""
import unittest

import torch

import twc_v1_k8_hybrid_direction as math_tools
import hybrid_direction_runtime as runtime


class HybridDirectionMathTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(112701)

    def test_direction_is_per_mu_phase_and_row_norm_invariant(self):
        truth = torch.randn(3, 4, 5, 7, dtype=torch.complex128, generator=self.gen)
        estimate = torch.randn(truth.shape, dtype=truth.dtype, generator=self.gen)
        expected = math_tools.direction_error(estimate, truth)
        amp = torch.rand(3, 4, 1, 7, generator=self.gen, dtype=torch.float64)+.1
        phase = torch.exp(1j*torch.rand(3, 4, 1, 7, generator=self.gen, dtype=torch.float64))
        torch.testing.assert_close(math_tools.direction_error(estimate*amp*phase, truth), expected,
            atol=2e-15, rtol=2e-15)
        self.assertEqual(expected.shape, (3,))
        torch.testing.assert_close(math_tools.direction_error(truth*amp*phase, truth),
            torch.zeros(3, dtype=torch.float64), atol=5e-16, rtol=0)
        original = estimate.clone(); math_tools.direction_error(estimate, truth)
        self.assertTrue(torch.equal(estimate, original))

    def test_actual_total_objective_matches_rate_plus_fixed_direction_and_gradient(self):
        truth = torch.randn(2, 8, 32, 256, dtype=torch.complex64, generator=self.gen)
        estimate = (truth+.15*torch.randn(truth.shape, dtype=truth.dtype, generator=self.gen)).requires_grad_()
        loss, metrics = math_tools.objective(estimate, truth)
        base, rate_metrics = runtime.helpers().rate_objective(estimate, truth, alpha=.03, downlink_snr_db=25.)
        direction = math_tools.direction_error(estimate, truth)
        torch.testing.assert_close(loss, base+100*direction.mean())
        torch.testing.assert_close(metrics['sum_rate_bps_hz'], rate_metrics['sum_rate_bps_hz'])
        torch.testing.assert_close(metrics['weighted_direction_loss'], 100*direction)
        g_rate = torch.autograd.grad(base, estimate, retain_graph=True)[0]
        g_total = torch.autograd.grad(loss, estimate)[0]
        self.assertTrue(bool(torch.isfinite(g_total).all()))
        self.assertGreater(float((g_total-g_rate).abs().sum()), 1e-4)

    def test_direction_gradient_is_tangent_to_complex_row_scaling(self):
        truth = torch.randn(1, 2, 3, 5, dtype=torch.complex128, generator=self.gen)
        estimate = torch.randn(truth.shape, dtype=truth.dtype, generator=self.gen, requires_grad=True)
        gradient = torch.autograd.grad(math_tools.direction_error(estimate, truth).sum(), estimate)[0]
        radial = (gradient.conj()*estimate).sum(2)
        self.assertLess(float(radial.detach().abs().max()), 2e-16)

    def test_bad_or_undefined_directions_are_rejected(self):
        truth = torch.ones(1, 2, 3, 5, dtype=torch.complex64)
        for estimate in (truth.real, truth[..., :2], truth*complex('nan'), truth*0):
            with self.assertRaises(ValueError): math_tools.direction_error(estimate, truth)
        with self.assertRaises(ValueError): math_tools.direction_error(truth, truth*0)


if __name__ == '__main__': unittest.main()
