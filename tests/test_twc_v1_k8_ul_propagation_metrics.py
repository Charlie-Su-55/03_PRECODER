"""Actual CPU receiver/filter identities, conditional models and signed terms."""
import copy
import hashlib
import unittest

try:
    import torch
    import twc_v1_k8_ul_propagation_metrics as metrics
    import ul_diagnostics_runtime as old
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Real PyTorch unavailable; no fake tensor arithmetic')
class ConditionalPropagationTests(unittest.TestCase):
    def fixture(self, arm='hybrid_g4', *, perfect=False, biased=False, zero_noise=False):
        phy = old.physics()
        generator = torch.Generator().manual_seed(991)
        layout = phy.make_layout(arm)
        prior = torch.linspace(.2, 2., phy.active_dimension(layout)).repeat(8, 1)
        prior = 48*prior/prior.sum(-1, keepdim=True)
        raw = torch.randn(4, 8, 216, dtype=torch.complex64, generator=generator)
        polarity = torch.ones(4, 8)
        wave = phy.build_waveform(raw, layout, polarity)
        if biased:
            H = torch.zeros(4, 8, 32, 256, dtype=torch.complex64)
            for user in range(8):
                H[:, user, user] = 32**.5
            estimated = H*.85
        else:
            H = torch.randn(4, 8, 32, 256, dtype=torch.complex64, generator=generator)
            estimated = H if perfect else H*.94+.08*torch.randn(H.shape, dtype=H.dtype, generator=generator)
        gamma = torch.tensor([0., 5., 15., 25.])
        noise = torch.zeros(4, 32, 256, dtype=torch.complex64) if zero_noise else torch.randn(
            4, 32, 256, dtype=torch.complex64, generator=generator)
        ce_variance = torch.full((4, 8, 256), 0. if perfect else .02)
        Y = phy.transmit(H, wave.x_grid, noise, gamma)
        observation = dict(wave=wave, Y=Y, estimated=estimated, prior=prior, ce_variance=ce_variance)
        recovered = phy.recover(Y, estimated, layout, prior, ce_variance, gamma)
        return observation, recovered, H, noise, layout, gamma

    def test_actual_both_receivers_reconstruct_fixed_filter_and_signed_error(self):
        for arm in ('fdma', 'hybrid_g4'):
            args = self.fixture(arm)
            result = metrics.analyze(*args)
            self.assertEqual(result['vectors']['soft_nmse_linear'].shape, (4,))
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in result['vectors'].values()))
            self.assertLess(float(result['checks']['filter_recovery_max_error'].max()), 3e-5)
            self.assertLess(float(result['checks']['reported_variance_max_error'].max()), 1e-7)
            self.assertLess(float(result['checks']['realized_error_decomposition_max_error'].max()), 3e-5)
            self.assertLess(float(result['checks']['conditional_variance_decomposition_max_error'].max()), 1e-10)
            if arm == 'fdma':
                self.assertFalse(any(name.startswith('soft_shared_') for name in result['vectors']))
            else:
                self.assertIn('soft_shared_nominal_scaled_propagated_variance', result['vectors'])

    def test_true_channel_never_changes_the_normal_filter(self):
        observation, recovered, H, noise, layout, gamma = self.fixture()
        first = metrics.analyze(observation, recovered, H, noise, layout, gamma)
        altered_truth = H*1.15
        changed_observation = dict(observation, Y=old.physics().transmit(
            altered_truth, observation['wave'].x_grid, noise, gamma))
        changed_recovered = old.physics().recover(changed_observation['Y'], observation['estimated'],
            layout, observation['prior'], observation['ce_variance'], gamma)
        second = metrics.analyze(changed_observation, changed_recovered, altered_truth, noise, layout, gamma)
        self.assertTrue(all(torch.equal(first['filters'][name], second['filters'][name])
            for name in first['filters']))
        self.assertFalse(torch.allclose(first['coordinate']['propagated_variance'],
            second['coordinate']['propagated_variance']))

    def test_perfect_ce_makes_propagated_and_reported_models_agree(self):
        for arm in ('fdma', 'hybrid_g4'):
            result = metrics.analyze(*self.fixture(arm, perfect=True))
            c = result['coordinate']
            self.assertTrue(torch.allclose(c['reported_variance'], c['propagated_variance'], atol=1e-7, rtol=5e-5))
            for name in ('realized_ce_power', 'conditional_ce_power', 'conditional_bias_ce_cross', 'reported_ce_power'):
                self.assertTrue(bool((c[name] == 0).all()))

    def test_rank_deficient_true_channel_is_scored_without_oracle_filter_solve(self):
        observation, _, H, noise, layout, gamma = self.fixture(biased=True)
        estimated = H.clone()
        true = H.clone()
        for first, second in layout.groups:
            true[:, second] = true[:, first]
        observation = dict(observation, estimated=estimated,
            Y=old.physics().transmit(true, observation['wave'].x_grid, noise, gamma))
        recovered = old.physics().recover(observation['Y'], estimated, layout,
            observation['prior'], observation['ce_variance'], gamma)
        result = metrics.analyze(observation, recovered, true, noise, layout, gamma)
        self.assertTrue(bool(torch.isfinite(result['coordinate']['propagated_variance']).all()))
        before = metrics.reconstruct_normal_filters(observation, recovered, layout)
        self.assertTrue(all(torch.equal(before[name], result['filters'][name]) for name in before))
        with self.assertRaisesRegex(ValueError, 'Unidentifiable'):
            old.physics().recover(observation['Y'], true, layout,
                observation['prior'], torch.zeros_like(observation['ce_variance']), gamma)

    def test_negative_cross_terms_are_retained_not_clamped(self):
        for arm in ('fdma', 'hybrid_g4'):
            result = metrics.analyze(*self.fixture(arm, biased=True, zero_noise=True))
            c = result['coordinate']
            self.assertTrue(bool((c['realized_bias_ce_cross'] < 0).all()))
            self.assertTrue(bool((c['conditional_bias_ce_cross'] < 0).all()))
            self.assertTrue(bool((c['realized_noise_power'] == 0).all()))
            positive_only = c['realized_bias_power']+c['realized_ce_power']
            self.assertTrue(bool((positive_only > c['actual_error_power']).all()))

    def test_heterogeneous_prior_scales_diagnostics_without_current_power_substitution(self):
        args = self.fixture()
        observation = args[0]
        result = metrics.analyze(*args)
        actual = result['coordinate']['actual_error_power']
        expected = (actual/observation['prior'][None]).mean((1, 2))
        self.assertTrue(torch.equal(result['vectors']['soft_nominal_scaled_actual_mse'], expected))
        frame_power = observation['wave'].active.abs().square()
        self.assertFalse(torch.allclose(frame_power, observation['prior'][None].expand_as(frame_power)))
        self.assertFalse(torch.allclose(expected, (actual/frame_power).mean((1, 2))))

    def test_exact_two_user_conditional_formula_with_cross_user_channel_mix(self):
        observation, _, H, noise, layout, gamma = self.fixture(biased=True)
        estimated = H.clone()
        true = H.clone()
        for first, second in layout.groups:
            true[:, first] = H[:, first]+.2*H[:, second]
            true[:, second] = H[:, second]-.1*H[:, first]
        observation = dict(observation, estimated=estimated,
            Y=old.physics().transmit(true, observation['wave'].x_grid, noise, gamma))
        recovered = old.physics().recover(observation['Y'], estimated, layout,
            observation['prior'], observation['ce_variance'], gamma)
        result = metrics.analyze(observation, recovered, true, noise, layout, gamma)
        ridge = recovered['shared_effective_lambda'][:, :2]
        gain = 32/(32+ridge.to(torch.float64))
        prior = observation['prior'][:2, 4:].to(torch.float64)
        n0 = old.physics().noise_variance(gamma, 4, 'cpu').to(torch.float64)
        thermal = n0[:, None, None]*32/(32+ridge.to(torch.float64)).square()
        expected0 = (gain[:, 0]-1).square()*prior[0]+(.1*gain[:, 0]).square()*prior[1]+thermal[:, 0]
        expected1 = (.2*gain[:, 1]).square()*prior[0]+(gain[:, 1]-1).square()*prior[1]+thermal[:, 1]
        actual = result['coordinate']['propagated_variance'][:, :2, 4:]
        self.assertTrue(torch.allclose(actual[:, 0], expected0, atol=1e-7, rtol=2e-5))
        self.assertTrue(torch.allclose(actual[:, 1], expected1, atol=1e-7, rtol=2e-5))

    def test_changed_noise_ridge_or_reported_variance_is_rejected(self):
        args = self.fixture()
        observation, recovered, H, noise, layout, gamma = args
        with self.assertRaisesRegex(ValueError, 'paid Y'):
            metrics.analyze(observation, recovered, H, noise+.1, layout, gamma)
        changed = copy.deepcopy(recovered)
        changed['key_effective_lambda'] += .7
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            metrics.analyze(observation, changed, H, noise, layout, gamma)
        changed = copy.deepcopy(recovered)
        changed['variance'] *= 1.5
        with self.assertRaisesRegex(ValueError, 'reported variance'):
            metrics.analyze(observation, changed, H, noise, layout, gamma)
        changed_observation = dict(observation, estimated=observation['estimated'].to(torch.complex128))
        with self.assertRaisesRegex(ValueError, 'complex64'):
            metrics.reconstruct_normal_filters(changed_observation, recovered, layout)
        changed_observation = dict(observation, ce_variance=observation['ce_variance']*.5)
        with self.assertRaisesRegex(ValueError, 'CE variance/noise'):
            metrics.analyze(changed_observation, recovered, H, noise, layout, gamma)

    def test_single_batch_ce_variance_broadcast_matches_actual_receiver_input(self):
        observation, recovered, H, noise, layout, gamma = self.fixture()
        broadcast = dict(observation, ce_variance=observation['ce_variance'][:1])
        result = metrics.analyze(broadcast, recovered, H, noise, layout, gamma)
        self.assertTrue(bool(torch.isfinite(result['coordinate']['propagated_variance']).all()))

    def test_calibration_bins_have_exact_counts_sums_and_no_coordinate_ci(self):
        args = self.fixture()
        result = metrics.analyze(*args)
        rows = metrics.calibration_bins(result, args[0]['prior'], args[4])
        for kind in ('reported', 'propagated'):
            for name, start, stop in (('all', 0, 44), ('key', 0, 4), ('shared', 4, 44)):
                selected = [r for r in rows if r['component'] == kind+'_soft_'+name]
                self.assertEqual(sum(r['count'] for r in selected), 4*8*(stop-start))
                expected = (result['coordinate']['actual_error_power'][..., start:stop]
                    /args[0]['prior'][None, :, start:stop]).sum()
                self.assertAlmostEqual(sum(r['sum_actual'] for r in selected), float(expected), places=9)
                self.assertFalse(any('ci' in key for r in selected for key in r))

    def test_input_tensor_bytes_and_flags_remain_unchanged(self):
        args = self.fixture()
        observation, recovered, H, noise, _, _ = args
        values = [observation['Y'], observation['estimated'], observation['prior'],
            observation['ce_variance'], observation['wave'].active, recovered['soft'],
            recovered['variance'], H, noise]
        def digest(value):
            return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        before = [(digest(v), v.requires_grad) for v in values]
        metrics.analyze(*args)
        self.assertEqual(before, [(digest(v), v.requires_grad) for v in values])


if __name__ == '__main__':
    unittest.main()
