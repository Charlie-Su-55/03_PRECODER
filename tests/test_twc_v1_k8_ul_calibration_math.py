"""Real CPU calibration/receiver arithmetic and deployment input boundaries."""
import copy
import inspect
import unittest

import torch

from configs import twc_v1_k8_ul_calibration_config as cfg
import twc_v1_k8_ul_calibration as calibrated


def identity_model(coordinates=8*256, samples=3):
    records = []
    for gamma in cfg.GAMMAS:
        for i in range(samples):
            records.append(dict(role='calibration', gamma_db=gamma, feature=float(i),
                sample_index=i, n_coordinates=coordinates,
                h2_sum=float(coordinates), he_real_sum=0., he_imag_sum=0.,
                parallel_scaled_sum=float(coordinates), he_scaled_real_sum=0.,
                he_scaled_imag_sum=0., h2_scaled_sum=float(coordinates),
                perp_scaled_sum=float(coordinates)))
    return calibrated.fit_model(records), records


def changed_model(coordinates=8*256):
    model, _ = identity_model(coordinates)
    for node in model['nodes']:
        for cell in node['strata']:
            cell.update(alpha_real=.08, alpha_imag=.03, parallel=2.3, perpendicular=.8)
    return calibrated.validate_model(model)


class CalibrationMathTests(unittest.TestCase):
    def fixture(self, arm='hybrid_g4'):
        phy = calibrated._physics()
        generator = torch.Generator().manual_seed(102)
        layout = phy.make_layout(arm)
        batch = 2
        raw = torch.randn(batch, 8, 216, dtype=torch.complex64, generator=generator)
        wave = phy.build_waveform(raw, layout, torch.ones(batch, 8))
        true = torch.randn(batch, 8, 32, 256, dtype=torch.complex64, generator=generator)
        estimate = .96*true+.03*torch.randn(true.shape, dtype=true.dtype, generator=generator)
        prior = torch.linspace(.3, 2.5, phy.active_dimension(layout)).repeat(8, 1)
        prior = 48*prior/prior.sum(-1, keepdim=True)
        gamma = torch.tensor([0., 25.])
        xi = torch.randn(batch, 32, 256, dtype=torch.complex64, generator=generator)
        public = dict(Y=phy.transmit(true, wave.x_grid, xi, gamma), estimated=estimate,
            prior=prior, ce_variance=torch.full((batch, 8, 256), .02), pilots=wave.pilot)
        return public, layout, gamma, true, wave

    def test_pilot_feature_masks_data_before_reading_or_finite_checks(self):
        public, _, gamma, _, _ = self.fixture()
        feature = calibrated.pilot_feature(public['Y'], public['pilots'], public['estimated'], gamma)
        own = public['pilots'].abs().ne(0).any(1)[0]
        y, h = public['Y'].clone(), public['estimated'].clone()
        y[..., ~own] = complex(float('nan'), 0)
        h[..., ~own] = complex(float('inf'), 0)
        changed = calibrated.pilot_feature(y, public['pilots'], h, gamma)
        self.assertTrue(torch.equal(feature, changed))
        y[..., own] = complex(float('nan'), 0)
        with self.assertRaises(ValueError):
            calibrated.pilot_feature(y, public['pilots'], h, gamma)

    def test_pilot_feature_matches_direct_known_pilot_formula(self):
        public, _, gamma, _, _ = self.fixture()
        own = public['pilots'].abs().ne(0).any(1)[0]
        residual = public['Y'][..., own]-(public['estimated'][..., own]*public['pilots'][..., own][:, :, None]).sum(1)
        expected = residual.abs().square().mean((1, 2))/calibrated._physics().noise_variance(gamma, 2, 'cpu')
        self.assertTrue(torch.equal(expected, calibrated.pilot_feature(
            public['Y'], public['pilots'], public['estimated'], gamma)))

    def test_sufficient_statistics_reproduce_real_complex_projection(self):
        h = torch.zeros(3, 2, 3, 4, dtype=torch.complex64)
        h[:, :, 0] = 2
        error = (.2+.1j)*h
        error[:, :, 1] = .3j
        v = torch.full((3, 2, 4), .05)
        rows = calibrated.calibration_statistics(h, h+error, v,
            torch.tensor([1., 2., 3.]), torch.full((3,), 10.))
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row['role'], 'calibration')
            self.assertEqual(row['n_coordinates'], 8)
            self.assertAlmostEqual(row['h2_sum'], 32.)
            self.assertAlmostEqual(row['he_real_sum'], 6.4, places=5)
            self.assertAlmostEqual(row['he_imag_sum'], 3.2, places=5)
            self.assertAlmostEqual(row['parallel_scaled_sum'], 32., places=4)
            self.assertAlmostEqual(row['perp_scaled_sum'], 7.2, places=4)

    def test_fit_complex_bias_and_centered_parallel_power_closed_form(self):
        model, records = identity_model(coordinates=1, samples=3)
        for row in records:
            row.update(h2_sum=4., he_real_sum=.8, he_imag_sum=.4,
                h2_scaled_sum=80., he_scaled_real_sum=16., he_scaled_imag_sum=8.,
                parallel_scaled_sum=4., perp_scaled_sum=.9)
        model = calibrated.fit_model(records)
        weight = 1/17
        alpha = weight*(.2+.1j)
        expected_parallel = 1-weight+weight*(4-2*(alpha.conjugate()*(16+8j)).real+abs(alpha)**2*80)
        for node in model['nodes']:
            for cell in node['strata']:
                self.assertAlmostEqual(cell['alpha_real'], alpha.real)
                self.assertAlmostEqual(cell['alpha_imag'], alpha.imag)
                self.assertAlmostEqual(cell['parallel'], expected_parallel)
                self.assertAlmostEqual(cell['perpendicular'], 1-weight+weight*.9)

    def test_tied_feature_empty_strata_are_fixed_identity_without_validation_fallback(self):
        _, records = identity_model(samples=3)
        for row in records:
            row['feature'] = 7.
        model = calibrated.fit_model(records)
        for node in model['nodes']:
            self.assertEqual(node['thresholds'], [7., 7.])
            self.assertEqual([cell['n_mu'] for cell in node['strata']], [3, 0, 0])
            self.assertTrue(all(cell['parallel'] == 1 and cell['perpendicular'] == 1 for cell in node['strata']))

    def test_fit_refuses_validation_labels_duplicates_partial_nodes_and_bad_scalars(self):
        _, original = identity_model()
        mutations = [lambda r: r[0].update(role='validation'),
            lambda r: r.append(copy.deepcopy(r[0])),
            lambda r: r.pop(), lambda r: r[0].update(feature=float('nan')),
            lambda r: r[0].update(secret_future_field=1),
            lambda r: r[0].update(n_coordinates=True)]
        for mutate in mutations:
            records = copy.deepcopy(original)
            mutate(records)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                calibrated.fit_model(records)

    def test_locked_model_rejects_schema_psd_nonfinite_identity_and_boolean_gamma_changes(self):
        original, _ = identity_model()
        mutations = [lambda m: m.update(calibration_role='validation'),
            lambda m: m.update(unregistered_field=0),
            lambda m: m.update(shrink_mu=0.),
            lambda m: m['gammas'].__setitem__(0, False),
            lambda m: m['nodes'][0].update(gamma_db=False),
            lambda m: m['nodes'][0]['strata'][0].update(parallel=-1.),
            lambda m: m['nodes'][0]['strata'][0].update(alpha_real=float('inf')),
            lambda m: m['nodes'][0].update(thresholds=[3., 2.])]
        for mutate in mutations:
            model = copy.deepcopy(original)
            mutate(model)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                calibrated.validate_model(model)

    def test_predictions_are_truth_free_and_keep_tensor_inputs_unchanged(self):
        public, _, gamma, _, _ = self.fixture()
        model = changed_model()
        original = {key: value.clone() for key, value in public.items()}
        feature = calibrated.pilot_feature(public['Y'], public['pilots'], public['estimated'], gamma)
        moments = calibrated.predict_moments(public['estimated'], public['ce_variance'], feature, gamma, model)
        self.assertTrue(torch.allclose(moments['mean'], public['estimated']*(1.08+.03j)))
        self.assertTrue(bool((moments['parallel'] == 2.3).all()))
        self.assertFalse(any('true' in name for name in inspect.signature(calibrated.predict_moments).parameters))
        for key in public:
            self.assertTrue(torch.equal(public[key], original[key]))
        with self.assertRaises(TypeError):
            calibrated.predict_moments(public['estimated'], public['ce_variance'], feature, gamma, model, true_ul=public['estimated'])

    def test_predictions_require_float_features_registered_gamma_and_same_geometry(self):
        public, _, gamma, _, _ = self.fixture()
        model, _ = identity_model()
        for feature, values, used_model in ((torch.ones(2, dtype=torch.long), gamma, model),
                (torch.ones(2), torch.tensor([1., 5.]), model),
                (torch.ones(2), gamma, identity_model(coordinates=1)[0])):
            with self.assertRaises(ValueError):
                calibrated.predict_moments(public['estimated'], public['ce_variance'], feature, values, used_model)

    def test_isotropic_identity_calibration_recovers_original_receiver_for_both_arms(self):
        for arm in ('fdma', 'hybrid_g4'):
            public, layout, gamma, _, _ = self.fixture(arm)
            model, _ = identity_model()
            expected = calibrated._physics().recover(public['Y'], public['estimated'], layout,
                public['prior'], public['ce_variance'], gamma)
            variants = calibrated.recover_variants(public, layout, gamma, model)
            for name in ('soft', 'variance', 'reliability', 'H_ul_est'):
                self.assertTrue(torch.equal(variants['original'][name], expected[name]), (arm, name))
                self.assertTrue(torch.allclose(variants['moment_receiver'][name], expected[name],
                    atol=3e-5, rtol=3e-5), (arm, name))
            self.assertTrue(torch.equal(variants['reliability_only']['soft'], expected['soft']))
            self.assertTrue(torch.allclose(variants['reliability_only']['variance'], expected['variance'],
                atol=1e-7, rtol=3e-5))

    def test_reliability_only_keeps_soft_channel_and_filter_bytes(self):
        public, layout, gamma, _, _ = self.fixture()
        variants = calibrated.recover_variants(public, layout, gamma, changed_model())
        original, reliability = variants['original'], variants['reliability_only']
        for name in ('soft', 'z_hat', 'Y_res', 'H_ul_est', 'prior'):
            self.assertIs(original[name], reliability[name])
        self.assertFalse(torch.allclose(original['variance'], reliability['variance']))
        for key in original['filters']:
            self.assertTrue(torch.equal(original['filters'][key], reliability['filters'][key]))
        moment = variants['moment_receiver']
        self.assertFalse(torch.allclose(original['soft'], moment['soft']))
        self.assertTrue(torch.allclose(moment['H_ul_est'], public['estimated']*(1.08+.03j)))
        self.assertNotIn('key_effective_lambda', moment)

    def test_recovery_rejects_truth_waveform_and_current_power_side_channels(self):
        public, layout, gamma, true, wave = self.fixture()
        model, _ = identity_model()
        for key, value in (('true_ul', true), ('wave', wave), ('current_power', wave.active.abs().square())):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Only public'):
                calibrated.recover_variants(dict(public, **{key:value}), layout, gamma, model)

    def test_recovery_rejects_data_waveform_smuggled_into_known_pilot_key(self):
        public, layout, gamma, _, wave = self.fixture()
        model, _ = identity_model()
        with self.assertRaisesRegex(ValueError, 'Pilot support'):
            calibrated.recover_variants(dict(public, pilots=wave.x_grid), layout, gamma, model)

    def test_anisotropic_covariance_is_psd_even_when_parallel_is_smaller(self):
        generator = torch.Generator().manual_seed(10)
        g = torch.randn(2, 3, 5, 2, dtype=torch.complex64, generator=generator)
        v = torch.full((2, 3, 2), .1)
        prior = torch.tensor([[.4, .8]]).expand(3, -1)
        covariance = calibrated._error_covariance(g, v, prior, torch.tensor([.2, .05]),
            torch.tensor([.1, 3.]), torch.tensor([2., .2]))
        self.assertTrue(torch.allclose(covariance, covariance.mH, atol=1e-7))
        self.assertTrue(bool((torch.linalg.eigvalsh(covariance) > 0).all()))

    def test_source_tensors_and_locked_model_never_mutate_during_recovery(self):
        public, layout, gamma, _, _ = self.fixture()
        tensors = {key: value.clone() for key, value in public.items()}
        model = changed_model()
        original_model = copy.deepcopy(model)
        result = calibrated.recover_variants(public, layout, gamma, model)
        self.assertEqual(model, original_model)
        for name, tensor in tensors.items():
            self.assertTrue(torch.equal(tensor, public[name]))
        for receiver in result.values():
            self.assertFalse(receiver['soft'].requires_grad)
            self.assertTrue(bool((receiver['variance'] > 0).all()))
            self.assertTrue(bool((receiver['reliability'][..., 44:] == 0).all()))


if __name__ == '__main__':
    unittest.main()
