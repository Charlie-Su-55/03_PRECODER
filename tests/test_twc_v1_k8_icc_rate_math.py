"""Real CPU rates/RZF gradients and BS-only optimizer boundaries; no CDL."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "_test_icc_rate_math", ROOT / "utils" / "twc_v1_k8_icc_rate.py")
rate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rate)

try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


@unittest.skipIf(torch is None or np is None, "Real PyTorch/NumPy unavailable; no fake dependency")
class RealRateMath(unittest.TestCase):
    def test_cross_user_interference_and_gradient_match_closed_form(self):
        theta = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
        one = theta.new_tensor(1.)
        raw = torch.stack((one, theta, theta, one)).reshape(1, 1, 2, 2)
        W = (raw / raw.square().sum().sqrt()).to(torch.complex128)
        H = torch.eye(2, dtype=torch.complex128).reshape(1, 1, 2, 2)
        metrics = rate.differentiable_rates(H, W, noise_power=.05)
        desired = 1. / (2. * (1. + theta.square()))
        interference = theta.square() / (2. * (1. + theta.square()))
        expected = 2. * torch.log2(1. + desired / (interference + .05))
        torch.testing.assert_close(metrics["sum_rate_bps_hz"][0], expected)
        torch.testing.assert_close(metrics["desired_signal_power_mean"][0], desired)
        torch.testing.assert_close(metrics["interference_power_mean"][0], interference)
        torch.testing.assert_close(metrics["interference_to_signal_ratio"][0], theta.square())
        torch.testing.assert_close(metrics["per_tone_power"], torch.ones(1, 1, dtype=torch.float64))
        loss = -metrics["sum_rate_bps_hz"].mean()
        loss.backward()
        self.assertGreater(float(theta.grad), 0.)

        def scalar_loss(value):
            return -2. * math.log2(1. + 1. / (value ** 2 + .1 * (1. + value ** 2)))
        eps = 1e-6
        expected_grad = (scalar_loss(.4 + eps) - scalar_loss(.4 - eps)) / (2. * eps)
        self.assertAlmostEqual(float(theta.grad), expected_grad, places=6)

    def test_rate_objective_keeps_original_rzf_and_backpropagates(self):
        generator = torch.Generator(device="cpu").manual_seed(37160)
        shape = (1, 8, 32, 256)
        truth = torch.complex(torch.randn(shape, generator=generator),
                              torch.randn(shape, generator=generator)) / 32 ** .5
        estimate = (truth + .12 * torch.complex(torch.randn(shape, generator=generator),
                                               torch.randn(shape, generator=generator))).requires_grad_(True)
        loss, metrics = rate.rate_objective(estimate, truth)
        W, power = rate._downlink().design_rzf(estimate, .03)
        original = rate._downlink().evaluate_sum_rate(W, truth, 25.)
        for key in ("sum_rate_bps_hz", "user_rate_bps_hz",
                    "desired_signal_power_mean", "interference_power_mean"):
            torch.testing.assert_close(metrics[key], original[key])
        torch.testing.assert_close(loss, -original["sum_rate_bps_hz"].mean())
        self.assertLess(float((power - 1.).abs().max().detach()), 1e-6)
        loss.backward()
        self.assertIsNotNone(estimate.grad)
        self.assertTrue(bool(torch.isfinite(estimate.grad).all()))
        self.assertGreater(float(estimate.grad.abs().sum()), 0.)
        self.assertIsNone(truth.grad)

    def test_public_snr_and_explicit_noise_agree(self):
        H = torch.eye(2, dtype=torch.complex64)[None, None].expand(3, 4, 2, 2)
        W = H / 2 ** .5
        a = rate.differentiable_rates(H, W, downlink_snr_db=25.)
        b = rate.differentiable_rates(H, W, noise_power=10 ** -2.5)
        for key in a:
            torch.testing.assert_close(a[key], b[key])
        self.assertEqual(tuple(a["sum_rate_bps_hz"].shape), (3,))
        self.assertEqual(tuple(a["user_rate_bps_hz"].shape), (3, 2))

    def test_invalid_inputs_are_rejected_before_training(self):
        H = torch.eye(2, dtype=torch.complex64)[None, None]
        W = H / 2 ** .5
        for noise in (0., -1., float("nan"), float("inf"), True, [1., 2.]):
            with self.subTest(noise=noise), self.assertRaises(ValueError):
                rate.differentiable_rates(H, W, noise_power=noise)
        for snr in (True, float("inf"), float("nan"), -100000.):
            with self.subTest(snr=snr), self.assertRaises(ValueError):
                rate.differentiable_rates(H, W, downlink_snr_db=snr)
        for bad in (W.real, W.to(torch.complex128), W[..., :1], W * complex(float("nan"), 0.)):
            with self.assertRaises(ValueError):
                rate.differentiable_rates(H, bad)
        with self.assertRaises(ValueError):
            rate.rate_objective(torch.zeros(1, 8, 32, 256, dtype=torch.complex64),
                                torch.zeros(1, 8, 32, 255, dtype=torch.complex64))


@unittest.skipIf(torch is None or np is None, "Real PyTorch/NumPy unavailable; no fake dependency")
class RealBSOnlyBoundary(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(2981)
        codec = torch.nn.Module()
        codec.encoder = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.4))
        codec.decoder = torch.nn.Linear(2, 4)
        with torch.no_grad():
            codec.decoder.weight.zero_()
            codec.decoder.bias.copy_(torch.tensor([1., .4, .4, 1.]))
        ce = torch.nn.Linear(2, 2)
        stats = torch.nn.Module()
        stats.register_buffer("historical_power", torch.ones(2))
        stats.register_buffer("ema_updates", torch.tensor(20000))
        parameters = rate.configure_bs_only(codec, ce, stats)
        return codec, ce, stats, parameters

    def test_actual_bs_optimizer_improves_rate_without_frozen_gradients_or_updates(self):
        codec, ce, stats, parameters = self.fixture()
        expected = rate.frozen_state(codec, ce, stats)
        decoder_before = {key: value.clone() for key, value in codec.decoder.state_dict().items()}
        optimizer = torch.optim.SGD(parameters, lr=.025)
        H = torch.eye(2, dtype=torch.complex64)[None, None].expand(4, 1, 2, 2)
        public = torch.ones(4, 2)

        def objective():
            observations = codec.encoder(public) + ce(public)
            raw = codec.decoder(observations).reshape(4, 1, 2, 2)
            W = (raw / raw.square().sum(dim=(-2, -1), keepdim=True).sqrt()).to(torch.complex64)
            return -rate.differentiable_rates(H, W, noise_power=.05)["sum_rate_bps_hz"].mean()

        initial_rate = float(-objective().detach())
        for _ in range(5):
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in parameters))
            self.assertTrue(all(p.grad is None for p in codec.encoder.parameters()))
            self.assertTrue(all(p.grad is None for p in ce.parameters()))
            optimizer.step()
            rate.assert_frozen_state(codec, ce, stats, expected)
        self.assertGreater(float(-objective().detach()), initial_rate)
        self.assertTrue(any(not torch.equal(value, codec.decoder.state_dict()[key])
                            for key, value in decoder_before.items()))
        self.assertEqual(int(stats.ema_updates), 20000)
        self.assertFalse(codec.encoder.training)
        self.assertFalse(ce.training)
        self.assertFalse(stats.training)
        self.assertTrue(codec.decoder.training)

    def test_mutated_frozen_buffer_parameter_and_mode_are_rejected(self):
        for change in ("buffer", "parameter", "mode", "gradient", "requires_grad"):
            with self.subTest(change=change):
                codec, ce, stats, _ = self.fixture()
                before = rate.frozen_state(codec, ce, stats)
                if change == "buffer":
                    stats.ema_updates.add_(1)
                elif change == "parameter":
                    with torch.no_grad():
                        ce.weight.add_(1.)
                elif change == "mode":
                    codec.encoder.train()
                elif change == "gradient":
                    ce.weight.grad = torch.zeros_like(ce.weight)
                else:
                    ce.weight.requires_grad_(True)
                with self.assertRaises(ValueError):
                    rate.assert_frozen_state(codec, ce, stats, before)

    def test_decoder_cannot_alias_encoder_parameters(self):
        codec, ce, stats, _ = self.fixture()
        codec.decoder = codec.encoder
        with self.assertRaisesRegex(ValueError, "Independent trainable"):
            rate.configure_bs_only(codec, ce, stats)


if __name__ == "__main__":
    unittest.main()
