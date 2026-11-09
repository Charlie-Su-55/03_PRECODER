"""Small CPU components only: no codec, Sionna, CDL or checkpoint assets."""
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None


def source():
    name = "_icc_physics_component_tests"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "utils/twc_v1_k8_icc_aligned_physics.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@unittest.skipIf(torch is None, "Real PyTorch unavailable; no dependency stub")
class PhysicsComponents(unittest.TestCase):
    def setUp(self):
        self.p = source()
        self.generator = torch.Generator().manual_seed(27)

    def complex(self, shape):
        return torch.complex(torch.randn(shape, generator=self.generator),
                             torch.randn(shape, generator=self.generator)) / 2**.5

    def test_resource_energy_joint_normalization_and_single_noise(self):
        raw = self.complex((2, 8, 216))
        h = self.complex((2, 8, 32, 256))
        xi = self.complex((2, 32, 256))
        gamma = torch.tensor([0., 25.])
        for arm in ("fdma", "hybrid_g4"):
            layout = self.p.make_layout(arm)
            wave = self.p.build_waveform(raw, layout)
            ledger = self.p.energy_ledger(wave, layout)
            torch.testing.assert_close(ledger["pilot"], torch.full((2, 8), 16.))
            torch.testing.assert_close(ledger["data"], torch.full((2, 8), 48.))
            torch.testing.assert_close(ledger["total"], torch.full((2, 8), 64.))
            self.assertTrue(torch.equal(ledger["pilot_data_cross"], torch.zeros((2, 8))))
            self.assertEqual(int((wave.x_grid.abs().sum((0, 1)) > 0).sum()), 256)
            expected = (h * wave.x_grid[:, :, None]).sum(1) + self.p.noise_variance(gamma)[:, None, None].sqrt() * xi
            torch.testing.assert_close(self.p.transmit(h, wave.x_grid, xi, gamma), expected, rtol=0, atol=0)
        # Joint vector normalization preserves the raw key/shared ratio; it
        # does not force either legacy energy split.
        wave = self.p.build_waveform(raw, self.p.make_layout("hybrid_g4"))
        torch.testing.assert_close(wave.key.abs().square().sum(-1) / wave.shared.abs().square().sum(-1),
            raw[..., :4].abs().square().sum(-1) / raw[..., 4:44].abs().square().sum(-1))

    def test_fourier_rank_gram_scale_and_ridge_shrink(self):
        pilots = self.p.known_pilots(1)
        info = self.p.fourier_diagnostics(pilots)
        self.assertEqual(info["design_rank"].tolist(), [64])
        torch.testing.assert_close(info["gram_diagonal"], torch.full((1, 64), 16 / 256), rtol=1e-6, atol=1e-7)
        self.assertLess(float(info["gram_max_off_diagonal"].max()), 2e-7)
        torch.testing.assert_close(info["basis_gram"], torch.eye(8, dtype=torch.complex64), rtol=1e-6, atol=1e-6)
        self.assertEqual(info["ridge_to_diagonal"], .016)
        coefficients = self.complex((1, 8, 8, 32))
        h = torch.einsum("qr,bkrn->bknq", self.p.fourier_basis(), coefficients)
        y = (h * pilots[:, :, None]).sum(1)
        estimated = self.p.fourier_ls(y, pilots)
        torch.testing.assert_close(estimated, h / 1.016, rtol=2e-5, atol=2e-6)

    def test_fourier_reads_only_pilot_support(self):
        pilots = self.p.known_pilots(1)
        y = self.complex((1, 32, 256))
        a = self.p.fourier_ls(y, pilots)
        y[..., ~self.p.pilot_mask().any(0)] = complex(float("nan"), float("nan"))
        torch.testing.assert_close(self.p.fourier_ls(y, pilots), a, rtol=0, atol=0)

    def test_causal_complete_batch_commit_freeze_and_strict_roundtrip(self):
        layout = self.p.make_layout("hybrid_g4")
        stats = self.p.ReceiverStatistics(layout, torch.ones((6, 8, 256)))
        before = {key: value.clone() for key, value in stats.state_dict().items()}
        prior = stats.nominal_power()
        self.assertEqual(int(stats.ema_updates), 0)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in stats.state_dict().items()))
        waves = [self.p.build_waveform(self.complex((2, 8, 216)), layout) for _ in range(2)]
        powers = sum(wave.active.abs().square().sum(0) for wave in waves)
        stats.commit_power(powers, 4, expected_previous_updates=0)
        torch.testing.assert_close(stats.power_ema, .99 * before["power_ema"] + .01 * powers / 4)
        torch.testing.assert_close(stats.nominal_power().sum(-1), torch.full((8,), 48.))
        self.assertFalse(torch.equal(prior, stats.nominal_power()))
        with self.assertRaisesRegex(ValueError, "stale or duplicated"):
            stats.commit_power(powers, 4, expected_previous_updates=0)
        stats.eval()
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            stats.commit_power(powers, 4, expected_previous_updates=1)
        clone = self.p.ReceiverStatistics(layout)
        clone.load_state_dict(stats.state_dict(), strict=True)
        for key, value in stats.state_dict().items():
            self.assertTrue(torch.equal(value, clone.state_dict()[key]))
        with self.assertRaises(RuntimeError):
            self.p.ReceiverStatistics(self.p.make_layout("fdma")).load_state_dict(stats.state_dict(), strict=True)

    def test_ce_nodes_interpolation_and_no_extrapolation(self):
        errors = torch.arange(6.).reshape(6, 1, 1).expand(-1, 8, 256).clone()
        stats = self.p.ReceiverStatistics(self.p.make_layout("fdma"), errors)
        actual = stats.ce_variance(torch.tensor([0., 2.5, 7.5, 25.]))
        torch.testing.assert_close(actual[:, 0, 0], torch.tensor([0., .5, 1.5, 5.]))
        for gamma in (-.1, 25.1, float("nan")):
            with self.assertRaises(ValueError):
                stats.ce_variance(gamma)
        with self.assertRaisesRegex(ValueError, "Independent new CE"):
            self.p.ReceiverStatistics(self.p.make_layout("fdma")).ce_variance(10.)

    def test_scalar_recovery_uses_coordinate_prior_and_ce_error(self):
        layout = self.p.make_layout("fdma")
        h = torch.ones((1, 8, 32, 256), dtype=torch.complex64)
        y = torch.ones((1, 32, 256), dtype=torch.complex64) * (2 + 3j)
        prior = torch.arange(1., 25.)[None].expand(8, -1).clone()
        prior = 48 * prior / prior.sum(-1, keepdim=True)
        ce = torch.full((1, 8, 256), .02)
        out = self.p.recover(y, h, layout, prior, ce, 0.)
        ridge = .001 + (.25 + prior * .02) / prior
        torch.testing.assert_close(out["soft"][..., :24], (32 * (2 + 3j) / (32 + ridge))[None])
        expected_var = (32 / (32 + ridge) - 1).square() * prior + (.25 + prior * .02) * 32 / (32 + ridge).square()
        torch.testing.assert_close(out["variance"], expected_var[None])
        self.assertTrue(torch.equal(out["soft"][..., 24:], torch.zeros((1, 8, 192), dtype=torch.complex64)))

    def test_shared_recovery_conjugate_axes_and_finite_gradient(self):
        layout = self.p.make_layout("hybrid_g4")
        # Distinct antenna direction for each user gives known orthonormal
        # 32x2 shared channels with complex phases, not an oracle receiver input.
        h = torch.zeros((1, 8, 32, 256), dtype=torch.complex64)
        for user in range(8):
            h[:, user, user] = torch.exp(torch.tensor(1j * (user + .5)))
        raw = self.complex((1, 8, 216)).requires_grad_()
        wave = self.p.build_waveform(raw, layout)
        y = self.p.transmit(h, wave.x_grid, torch.zeros((1, 32, 256), dtype=torch.complex64), 25.)
        stats = self.p.ReceiverStatistics(layout, torch.zeros((6, 8, 256)))
        out = self.p.recover(y, h, layout, stats.nominal_power(), stats.ce_variance(25.), 25.)
        ridge = .001 + float(self.p.noise_variance(25.)[0]) / (48 / 44)
        torch.testing.assert_close(out["soft"][..., :44], wave.active / (1 + ridge), rtol=2e-5, atol=2e-6)
        out["soft"].abs().square().mean().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertGreater(float(raw.grad.abs().sum()), 0.)

    def test_malformed_prior_and_unpaid_power_rejected(self):
        layout = self.p.make_layout("fdma")
        stats = self.p.ReceiverStatistics(layout)
        with self.assertRaisesRegex(ValueError, "actual jointly-normalized"):
            stats.commit_power(torch.ones((8, 24)), 4, expected_previous_updates=0)
        with self.assertRaises(FloatingPointError):
            self.p.build_waveform(torch.zeros((1, 8, 216), dtype=torch.complex64), layout)


if __name__ == "__main__":
    unittest.main()
