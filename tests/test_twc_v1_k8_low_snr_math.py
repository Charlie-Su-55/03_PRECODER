"""Real CPU leakage mathematics, masking, units and differentiable RZF tests."""
import unittest

import torch

from configs import twc_v1_k8_low_snr_config as cfg
import low_snr_runtime as runtime
import twc_v1_k8_low_snr as math_tools


class LowSNRMathTests(unittest.TestCase):
    def setUp(self): self.gen = torch.Generator().manual_seed(112811)

    def test_leakage_matches_closed_form_and_retains_desired_signal_gradient(self):
        desired = (torch.rand(2, 5, 3, dtype=torch.float64, generator=self.gen)+.1).requires_grad_()
        interference = (torch.rand(2, 5, 3, dtype=torch.float64, generator=self.gen)*.2).requires_grad_()
        noise = .0031622776601683794
        leakage = math_tools.leakage_penalty(desired, interference, noise)
        expected = torch.log2(1+interference/(desired+noise)).sum(-1).mean(-1)
        torch.testing.assert_close(leakage, expected, atol=1e-15, rtol=1e-15)
        rate = torch.log2(1+desired/(interference+noise)).sum(-1).mean(-1)
        combined = -rate+leakage
        closed = torch.log2((interference+noise)/(desired+noise)).sum(-1).mean(-1)
        torch.testing.assert_close(combined, closed, atol=3e-15, rtol=1e-15)
        grad_s, grad_i = torch.autograd.grad(leakage.sum(), (desired, interference))
        self.assertTrue(bool((grad_s < 0).all() & (grad_i > 0).all()))
        zero = math_tools.leakage_penalty(desired.detach(), interference.detach()*0, noise)
        torch.testing.assert_close(zero, torch.zeros_like(zero))

    def test_actual_rzf_control_and_interference_match_exact_masked_objective(self):
        truth = torch.randn(2, 8, 32, 256, dtype=torch.complex64, generator=self.gen)
        estimate = (truth+.15*torch.randn(truth.shape, dtype=truth.dtype, generator=self.gen)).requires_grad_()
        gamma = torch.tensor([0., 25.])
        control, base = math_tools.objective(estimate, truth, gamma, cfg.TRAJECTORIES[0])
        experiment, extra = math_tools.objective(estimate, truth, gamma, cfg.TRAJECTORIES[1])
        self.assertGreater(float(extra['dl_leakage_penalty'].detach().min()), 0)
        torch.testing.assert_close(base['weighted_interference_loss'], torch.zeros(2))
        torch.testing.assert_close(extra['low_snr_leakage_penalty'],
            extra['dl_leakage_penalty']*torch.tensor([1., 0.]))
        torch.testing.assert_close(experiment-control, extra['dl_leakage_penalty'][0].double()/2,
            atol=2e-6, rtol=2e-6)
        rate, _ = runtime.helpers().rate_objective(estimate, truth, alpha=.03, downlink_snr_db=25.)
        torch.testing.assert_close(control, rate+100*math_tools.direction_error(estimate, truth).mean())
        g_control = torch.autograd.grad(control, estimate, retain_graph=True)[0]
        g_experiment = torch.autograd.grad(experiment, estimate, retain_graph=True)[0]
        self.assertGreater(float((g_control[0]-g_experiment[0]).abs().sum()), 1e-4)
        torch.testing.assert_close(g_control[1], g_experiment[1])
        g_rate = torch.view_as_real(torch.autograd.grad(rate, estimate, retain_graph=True)[0]).reshape(-1)
        g_leak = torch.view_as_real(torch.autograd.grad(extra['dl_leakage_penalty'].mean(), estimate)[0]).reshape(-1)
        scale = torch.dot(g_rate, g_leak)/torch.dot(g_rate, g_rate)
        self.assertGreater(float((g_leak-scale*g_rate).norm()), 1e-4)

    def test_all_high_gamma_trajectories_have_identical_loss_and_gradients(self):
        truth = torch.randn(1, 8, 32, 256, dtype=torch.complex64, generator=self.gen)
        estimate = (truth+.1*torch.randn(truth.shape, dtype=truth.dtype, generator=self.gen)).requires_grad_()
        control, _ = math_tools.objective(estimate, truth, 15., cfg.TRAJECTORIES[0])
        experiment, _ = math_tools.objective(estimate, truth, 15., cfg.TRAJECTORIES[1])
        torch.testing.assert_close(control, experiment)
        torch.testing.assert_close(torch.autograd.grad(control, estimate, retain_graph=True)[0],
            torch.autograd.grad(experiment, estimate)[0])

    def test_invalid_shapes_powers_noise_gamma_or_trajectory_are_refused(self):
        powers = torch.ones(2, 5, 3)
        for desired, interference, noise in (
                (powers, powers[..., :2], .1), (powers, -powers, .1),
                (powers.to(torch.complex64), powers, .1), (powers, powers, 0),
                (powers, powers, torch.tensor(.1+0j)), (powers, powers, True),
                (powers, powers, torch.tensor(.1, requires_grad=True))):
            with self.assertRaises(ValueError): math_tools.leakage_penalty(desired, interference, noise)
        truth = torch.randn(1, 8, 32, 256, dtype=torch.complex64, generator=self.gen)
        for gamma, trajectory in ((False, cfg.TRAJECTORIES[0]), (3., cfg.TRAJECTORIES[0]),
                (0., 'fdma'), (torch.tensor([0.], requires_grad=True), cfg.TRAJECTORIES[1])):
            with self.assertRaises(ValueError): math_tools.objective(truth, truth, gamma, trajectory)


if __name__ == '__main__': unittest.main()
