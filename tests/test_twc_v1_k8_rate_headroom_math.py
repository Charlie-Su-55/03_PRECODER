"""Real CPU tensor checks for diagnostic identities and the rate bound."""
import itertools
import math
import unittest

import torch

import twc_v1_k8_rate_headroom as headroom


def complex_randn(*shape, seed=19, dtype=torch.complex128):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=generator, dtype=torch.float64),
                         torch.randn(shape, generator=generator, dtype=torch.float64)).to(dtype)/math.sqrt(2)


class DirectionReferenceTests(unittest.TestCase):
    def test_true_directions_preserve_estimated_norms_without_mutation_or_gradient(self):
        estimated = complex_randn(2, 3, 4, 5).requires_grad_()
        truth = complex_randn(2, 3, 4, 5, seed=20).requires_grad_()
        before = estimated.detach().clone(), truth.detach().clone()
        result = headroom.true_direction_reference(estimated, truth)
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(result.norm(dim=2), estimated.norm(dim=2))
        torch.testing.assert_close(result/result.norm(dim=2, keepdim=True),
                                   truth/truth.norm(dim=2, keepdim=True))
        self.assertTrue(torch.equal(estimated, before[0]) and torch.equal(truth, before[1]))

    def test_zero_estimate_is_zero_and_zero_true_direction_is_refused(self):
        true = complex_randn(1, 2, 3, 4)
        torch.testing.assert_close(headroom.true_direction_reference(torch.zeros_like(true), true),
                                   torch.zeros_like(true))
        true[:, 0, :, 0] = 0
        with self.assertRaisesRegex(ValueError, 'undefined'):
            headroom.true_direction_reference(torch.ones_like(true), true)

    def test_shape_and_dtype_mismatch_are_refused(self):
        true = complex_randn(1, 2, 3, 4)
        for other in (true.real, true.to(torch.complex64), true[..., :2]):
            with self.assertRaises(ValueError):
                headroom.true_direction_reference(other, true)


class InterferenceFreeBoundTests(unittest.TestCase):
    def test_water_filling_matches_exhaustive_two_user_power_search(self):
        channel = torch.tensor([[[[1.]], [[math.sqrt(.2)]]]], dtype=torch.complex128)
        result = headroom.interference_free_upper_bound(channel, 0.)
        grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
        brute = torch.log2(1+grid)+torch.log2(1+.2*(1-grid))
        self.assertAlmostEqual(result['sum_rate_bps_hz'].item(), brute.max().item(), places=7)
        torch.testing.assert_close(result['power_allocation'], torch.tensor([[[1., 0.]]], dtype=torch.float64))

    def test_active_streams_have_common_water_level_and_total_power(self):
        result = headroom.interference_free_upper_bound(complex_randn(2, 4, 3, 5), 10., total_power=.7)
        p, gain = result['power_allocation'], result['channel_gains']
        torch.testing.assert_close(p.sum(-1), torch.full((2, 5), .7, dtype=torch.float64))
        levels = p+result['noise_power']/gain
        for i in range(2):
            for tone in range(5):
                active_levels = levels[i, tone][p[i, tone] > 0]
                torch.testing.assert_close(active_levels, active_levels[0].expand_as(active_levels))
        torch.testing.assert_close(result['sum_rate_bps_hz'], result['user_rate_bps_hz'].sum(-1))

    def test_zero_gains_use_no_power_and_have_zero_rate(self):
        result = headroom.interference_free_upper_bound(torch.zeros((2, 3, 4, 5), dtype=torch.complex64))
        self.assertTrue(torch.equal(result['power_allocation'], torch.zeros_like(result['power_allocation'])))
        self.assertTrue(torch.equal(result['sum_rate_bps_hz'], torch.zeros(2, dtype=torch.float64)))

    def test_bounds_random_unequal_power_linear_precoders(self):
        channel = complex_randn(6, 3, 4, 5)
        bound = headroom.interference_free_upper_bound(channel, 7.)
        for seed in range(5):
            w = complex_randn(6, 5, 4, 3, seed=seed)
            w = w/w.abs().square().sum((-2, -1), keepdim=True).sqrt()
            effective = channel.permute(0, 3, 1, 2).conj()@w
            signal = effective.diagonal(dim1=-2, dim2=-1).abs().square()
            interference = (effective.abs().square().sum(-1)-signal).clamp_min(0)
            rate = torch.log2(1+signal/(interference+bound['noise_power'])).sum(-1).mean(-1)
            self.assertTrue(bool((rate <= bound['sum_rate_bps_hz']+1e-12).all()))

    def test_equal_power_is_not_substituted_for_the_mathematical_bound(self):
        channel = torch.tensor([[[[1.]], [[0.]]]], dtype=torch.complex128)
        result = headroom.interference_free_upper_bound(channel, 0.)
        self.assertGreater(result['sum_rate_bps_hz'].item(), math.log2(1+.5))
        self.assertAlmostEqual(result['sum_rate_bps_hz'].item(), 1.)

    def test_invalid_power_snr_or_channel_are_refused(self):
        channel = complex_randn(1, 2, 3, 4)
        for kwargs in ({'total_power': 0}, {'total_power': -1}, {'downlink_snr_db': float('nan')},
                       {'downlink_snr_db': 10000.}, {'downlink_snr_db': True}):
            with self.assertRaises(ValueError):
                headroom.interference_free_upper_bound(channel, **kwargs)
        with self.assertRaises(ValueError):
            headroom.interference_free_upper_bound(channel.real)


class FixedFilterDecompositionTests(unittest.TestCase):
    def test_signed_cross_terms_explain_exact_cancellation(self):
        a = torch.eye(2, dtype=torch.complex128)[None]
        g = torch.tensor([[[.8, .4], [.3, .6]]], dtype=torch.complex128)
        z = torch.ones((1, 2), dtype=torch.complex128)
        n = torch.tensor([[-.2, .1]], dtype=torch.complex128)
        result = headroom.fixed_filter_error_decomposition(a, g, z, n, prior=torch.ones(2), noise_variance=.1)
        torch.testing.assert_close(result['error'], torch.zeros_like(z), atol=1e-15, rtol=0)
        self.assertTrue(bool((result['realized_self_power'] > 0).all()))
        self.assertLess(result['realized_self_cross_user_cross'].sum().item(), 0)
        torch.testing.assert_close(result['realized_sum_power'], torch.zeros_like(z.real), atol=1e-15, rtol=0)

    def test_conditional_powers_match_exact_independent_qpsk_ensemble(self):
        qpsk = torch.tensor([1., -1., 1j, -1j], dtype=torch.complex128)
        indices = torch.tensor(list(itertools.product(range(4), repeat=4)))
        prior = torch.tensor([.3, 1.7], dtype=torch.float64)
        z, n = qpsk[indices[:, :2]]*prior.sqrt(), qpsk[indices[:, 2:]]*math.sqrt(.2)
        a = torch.tensor([[[.8+.1j, -.3j], [.2, .9-.2j]]], dtype=torch.complex128).expand(len(indices), -1, -1)
        g = torch.tensor([[[1.+.2j, .4j], [.3-.1j, .9]]], dtype=torch.complex128).expand(len(indices), -1, -1)
        result = headroom.fixed_filter_error_decomposition(a, g, z, n, prior=prior, noise_variance=.2)
        torch.testing.assert_close(result['actual_error_power'].mean(0), result['conditional_total_power'][0])
        for key in headroom.POWER_KEYS[4:7]:
            torch.testing.assert_close(result[key].mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0)

    def test_scalar_block_has_no_cross_user_error_and_does_not_mutate(self):
        a, g = complex_randn(2, 3, 1, 4), complex_randn(2, 3, 4, 1)
        z, n = complex_randn(2, 3, 1), complex_randn(2, 3, 4)
        copies = [value.clone() for value in (a, g, z, n)]
        result = headroom.fixed_filter_error_decomposition(a, g, z, n)
        self.assertTrue(torch.equal(result['realized_cross_user_power'], torch.zeros_like(z.real)))
        self.assertNotIn('conditional_total_power', result)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip((a, g, z, n), copies)))

    def test_bad_shapes_powers_and_partial_conditional_inputs_are_refused(self):
        a, g = complex_randn(1, 2, 3), complex_randn(1, 3, 2)
        z, n = complex_randn(1, 2), complex_randn(1, 3)
        for args, kwargs in (((a, g, z.real, n), {}), ((a, g, z, n[:, :2]), {}),
                             ((a, g, z, n), {'prior': torch.ones(2)}),
                             ((a, g, z, n), {'prior': torch.tensor([-1., 2.]), 'noise_variance': .1})):
            with self.assertRaises(ValueError):
                headroom.fixed_filter_error_decomposition(*args, **kwargs)


class ProductionReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ul_diagnostics_runtime import physics
        cls.physics = physics()

    def fixture(self, arm):
        phy = self.physics
        layout = phy.make_layout(arm)
        wave = phy.build_waveform(complex_randn(1, 8, 216, dtype=torch.complex64), layout)
        true = complex_randn(1, 8, 32, 256, seed=26, dtype=torch.complex64)
        estimated = true+.05*complex_randn(1, 8, 32, 256, seed=27, dtype=torch.complex64)
        noise = complex_randn(1, 32, 256, seed=28, dtype=torch.complex64)
        gamma = torch.tensor([10.])
        prior = torch.full((8, phy.active_dimension(layout)), 48./phy.active_dimension(layout))
        ce = torch.full((1, 8, 256), .05**2)
        received = phy.transmit(true, wave.x_grid, noise, gamma)
        recovered = phy.recover(received, estimated, layout, prior, ce, gamma)
        observation = dict(Y=received, estimated=estimated, prior=prior, wave=wave, ce_variance=ce)
        return observation, recovered, true, noise, layout, gamma

    def test_real_production_fdma_and_shared_filters_close_without_input_changes(self):
        for arm in ('fdma', 'hybrid_g4'):
            with self.subTest(arm=arm):
                inputs = self.fixture(arm)
                observed, recovered, true, noise, _, _ = inputs
                before = [v.clone() for v in (observed['Y'], observed['estimated'], observed['prior'],
                                              recovered['soft'], true, noise)]
                result = headroom.receiver_error_decomposition(*inputs)
                values = result['coordinate']
                self.assertEqual(set(values), set(headroom.POWER_KEYS))
                for check in result['checks'].values():
                    self.assertLess(check.max().item(), 3e-5)
                torch.testing.assert_close(values['actual_error_power'], values['realized_sum_power'], atol=3e-5, rtol=5e-5)
                if arm == 'fdma':
                    self.assertEqual(values['realized_cross_user_power'].max().item(), 0.)
                else:
                    self.assertGreater(values['realized_cross_user_power'][..., 4:].mean().item(), 0.)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before,
                    (observed['Y'], observed['estimated'], observed['prior'], recovered['soft'], true, noise))))

    def test_wrong_paid_noise_and_changed_soft_are_refused(self):
        observed, recovered, true, noise, layout, gamma = self.fixture('hybrid_g4')
        with self.assertRaisesRegex(ValueError, 'received signal'):
            headroom.receiver_error_decomposition(observed, recovered, true, -noise, layout, gamma)
        changed = dict(recovered, soft=recovered['soft']+.1)
        with self.assertRaisesRegex(ValueError, 'power|soft'):
            headroom.receiver_error_decomposition(observed, changed, true, noise, layout, gamma)


if __name__ == '__main__':
    unittest.main()
