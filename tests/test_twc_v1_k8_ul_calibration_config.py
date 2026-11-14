"""Pinned protocol, disjoint role addresses and calibration-only fitting scope."""
import hashlib
import json
import subprocess
import sys
import unittest

from configs import twc_v1_k8_icc_aligned_config as aligned
from configs import twc_v1_k8_icc_alpha_config as alpha
from configs import twc_v1_k8_icc_rate_config as rate
from configs import twc_v1_k8_ul_diagnostics_config as diagnostic
from configs import twc_v1_k8_ul_propagation_config as propagation
from configs import twc_v1_k8_w_residual_config as residual
from configs import twc_v1_k8_ul_calibration_config as cfg
import ul_calibration_runtime as runtime


class ULCalibrationConfigurationTests(unittest.TestCase):
    def test_statistical_calibration_is_explicit_and_neural_budget_stays_zero(self):
        recipe = cfg.recipe()
        self.assertFalse(recipe['training'])
        self.assertFalse(recipe['backward'])
        self.assertEqual(recipe['trainable'], [])
        for name in ('optimizer_updates', 'new_UE_updates', 'new_CE_updates',
                     'new_BS_updates', 'new_EMA_updates'):
            self.assertEqual(recipe[name], 0)
        fit = recipe['calibration']
        self.assertTrue(fit['statistical_fit'])
        self.assertTrue(fit['common_CE_model_for_both_arms'])
        self.assertEqual(fit['source_arm'], 'fdma')
        self.assertTrue(fit['quantiles_fit_on_calibration_only'])
        self.assertTrue(fit['lock_before_validation'])
        self.assertFalse(fit['validation_used_to_fit_or_select'])
        self.assertFalse(fit['hyperparameter_search'])
        self.assertFalse(fit['feature_is_independent_unbiased_error_measurement'])
        self.assertEqual(fit['pilot_fit_degrees_of_freedom'],
            dict(observations=64, coefficients_per_BS_antenna=64))
        self.assertEqual((cfg.STRATA, cfg.MODEL_SHRINK_MU), (3, 16.))

    def test_paid_protocol_model_pins_and_fixed_precoder_are_unchanged(self):
        recipe = cfg.recipe()
        for name in ('physical', 'precision', 'Fourier_LS', 'CE_calibration'):
            self.assertEqual(recipe[name], rate.recipe('fdma')[name])
        for name in ('SOURCE_KIND', 'SOURCE_ROOT', 'SOURCE_COMMIT', 'SOURCE_CORE',
                     'SOURCE_CHECKPOINT_SHA256', 'SOURCE_TERMINAL_SHA256',
                     'ORIGINAL_SOURCE_ROOT', 'SOURCE_CALIBRATION_SHA256'):
            self.assertEqual(getattr(cfg, name), getattr(propagation, name))
        self.assertEqual((cfg.ALPHA, cfg.DL_SNR), (.03, 25.))
        self.assertEqual(cfg.ARMS, ('fdma', 'hybrid_g4'))
        self.assertEqual(cfg.VARIANTS, ('original', 'reliability_only', 'moment_receiver'))
        self.assertEqual(cfg.GAMMAS, (0, 5, 10, 15, 20, 25))
        deploy = recipe['deployment']
        for key in ('true_UL_input', 'true_DL_input', 'true_symbols_input',
                    'realized_frame_power_input'):
            self.assertFalse(deploy[key])
        self.assertTrue(recipe['receiver']['reliability_only_keeps_reference_filter_and_soft'])
        self.assertTrue(recipe['receiver']['isotropic_identity_limit_must_reproduce_reference'])
        self.assertFalse(recipe['reporting']['validation_selects_model'])
        self.assertFalse(recipe['reporting']['test_used'])

    def test_all_46_legacy_source_bytes_are_pinned_literally(self):
        self.assertEqual(cfg.PINNED_PROPAGATION_COMMIT,
            '5aa386a8169aa60ea5384f2f5d949e97fa3fde23')
        self.assertEqual(cfg.PINNED_PROPAGATION_CORE['schema'], propagation.NAMESPACE+'_core_v1')
        self.assertEqual(cfg.LEGACY_CORE_FILES, propagation.CORE_FILES)
        self.assertEqual(cfg.CORE_FILES[:46], propagation.CORE_FILES)
        self.assertEqual(len(cfg.CORE_FILES), 54)
        self.assertEqual(len(set(cfg.CORE_FILES)), 54)
        self.assertEqual(set(cfg.PINNED_PROPAGATION_CORE['sha256']), set(propagation.CORE_FILES))
        for name, expected in cfg.PINNED_PROPAGATION_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT/name).read_bytes()).hexdigest(), expected, name)

    def test_actual_channel_roles_are_disjoint_and_prior_addresses_are_not_reused(self):
        self.assertEqual((cfg.MICRO_BATCH, cfg.CALIBRATION_BATCHES, cfg.VALIDATION_BATCHES), (4, 64, 64))
        self.assertEqual((cfg.CALIBRATION_SAMPLES, cfg.VALIDATION_SAMPLES), (256, 256))
        self.assertEqual(cfg.ROLE_RANGES, dict(calibration=[0, 63], validation=[0, 63]))
        self.assertEqual(cfg.CHANNEL_ROLES, dict(calibration='ce_calibration', validation='validation'))
        self.assertEqual(cfg.SEED, 20261123)
        self.assertEqual(cfg.OUTPUT_ROOT, cfg.ROOT/'runs'/cfg.NAMESPACE)
        for device in ('cpu', 'cuda:0'):
            data = runtime.make_data(device)
            cal = [data.address('ce_calibration', i) for i in range(64)]
            val = [data.address('validation', i) for i in range(64)]
            self.assertNotEqual(cal[0]['role_tag'], val[0]['role_tag'])
            seeds = {row['sionna_seed'] for row in cal+val}
            self.assertEqual(len(seeds), 128)
            source = runtime.previous.previous.legacy.rate.original.load_local(
                'utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
            for previous in (aligned, alpha, rate, residual, diagnostic, propagation):
                old = source.TWCv1K8ReplayableData(base_seed=previous.SEED,
                    batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
                    device=device, seed_namespace=getattr(previous, 'DATA_NAMESPACE', previous.NAMESPACE))
                prior = {old.address(role, i)['sionna_seed']
                    for role in ('ce_calibration', 'validation') for i in range(256)}
                self.assertTrue(seeds.isdisjoint(prior), (device, previous.NAMESPACE))
                self.assertNotEqual(data.protocol['data_id'], old.protocol['data_id'])

    def test_role_specific_noise_streams_are_reproducible_bounded_and_disjoint(self):
        streams = {role: {cfg.noise_seed(role, i, kind)
            for i in range(64*6) for kind in ('noise', 'polarity')}
            for role in cfg.BATCHES_BY_ROLE}
        self.assertEqual(len(streams['calibration']), 768)
        self.assertEqual(len(streams['validation']), 768)
        self.assertTrue(streams['calibration'].isdisjoint(streams['validation']))
        self.assertEqual(cfg.noise_seed('calibration', 5), cfg.noise_seed('calibration', 5))
        prior = {propagation.noise_seed('diagnostic', i, kind)
            for i in range(64*6) for kind in ('noise', 'polarity')}
        self.assertTrue((streams['calibration']|streams['validation']).isdisjoint(prior))
        for args in (('test', 0), ('ce_calibration', 0), ([], 0),
                     ('calibration', -1), ('validation', True), ('validation', 384),
                     ('validation', 0, 'gamma')):
            with self.assertRaises(ValueError):
                cfg.noise_seed(*args)

    def test_stdlib_import_and_explicit_public_directory_delivery(self):
        recipe = cfg.recipe()
        self.assertEqual(cfg.fingerprint(), cfg.digest(recipe))
        self.assertEqual(cfg.study_recipe(), recipe)
        self.assertEqual(cfg.study_fingerprint(), cfg.fingerprint())
        recipe['calibration']['validation_used_to_fit_or_select'] = True
        self.assertNotEqual(cfg.digest(recipe), cfg.fingerprint())
        delivery = cfg.recipe()['delivery']
        self.assertIn('no ZIP', delivery['format'])
        self.assertTrue(delivery['git_upload_is_explicit'])
        self.assertTrue(delivery['repository_is_public'])
        self.assertFalse(delivery['deleting_branch_guarantees_history_erasure'])
        for field in ('contains_model_weights', 'contains_channel_tensors',
                      'contains_credentials_or_machine_environment'):
            self.assertFalse(delivery[field])
        self.assertEqual(cfg.RESULT_BRANCH_PREFIX, 'results/'+cfg.NAMESPACE+'/')
        command = ('import sys; import ul_calibration_runtime as r; '
            'r.make_data("cpu"); assert "torch" not in sys.modules; '
            'assert "sionna" not in sys.modules; print(r.cfg.NAMESPACE)')
        actual = subprocess.check_output([sys.executable, '-B', '-S', '-c', command],
            cwd=cfg.ROOT, text=True).strip()
        self.assertEqual(actual, cfg.NAMESPACE)
        json.dumps(cfg.recipe(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
