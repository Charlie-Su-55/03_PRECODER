"""Budget, source identity and independent replay addresses without ML imports."""
import hashlib
import importlib.util
import math
import unittest

from configs import twc_v1_k8_icc_aligned_config as alignedcfg
from configs import twc_v1_k8_icc_alpha_config as alphacfg
from configs import twc_v1_k8_icc_rate_config as ratecfg
from configs import twc_v1_k8_w_residual_config as wcfg
from configs import twc_v1_k8_ul_diagnostics_config as cfg


class ULDiagnosticConfigurationTests(unittest.TestCase):
    def test_no_updates_or_deployment_claim_and_unchanged_paid_protocol(self):
        recipe = cfg.recipe()
        self.assertFalse(recipe['training'])
        self.assertFalse(recipe['backward'])
        self.assertEqual(recipe['trainable'], [])
        for name in ('optimizer_updates', 'new_CE_updates', 'new_UE_updates',
                     'new_BS_updates', 'new_EMA_updates'):
            self.assertEqual(recipe[name], 0)
        for name in ('physical', 'precision', 'Fourier_LS', 'CE_calibration'):
            self.assertEqual(recipe[name], ratecfg.recipe('fdma')[name])
        self.assertEqual(recipe['physical']['total_RE'], 256)
        self.assertEqual(recipe['physical']['Ep'], 16)
        self.assertEqual(recipe['physical']['Ed'], 48)
        report = recipe['reporting']
        for name in ('downlink_model_forward', 'performance_upper_bound_claim',
                     'deployable_method_claim', 'checkpoint_selection', 'test_used'):
            self.assertFalse(report[name])
        self.assertTrue(report['true_ul_is_offline_diagnostic'])
        self.assertFalse(recipe['observation']['measured_current_frame_power_is_receiver_input'])
        self.assertFalse(recipe['calibration']['reliability_is_exact_posterior'])

    def test_complete_fixed_budget_and_same_observation_contrasts(self):
        self.assertEqual(cfg.ARMS, ('fdma', 'hybrid_g4'))
        self.assertEqual(cfg.VARIANTS, ('estimated_ul', 'true_ul_diagnostic'))
        self.assertEqual(cfg.GAMMAS, (0, 5, 10, 15, 20, 25))
        self.assertEqual((cfg.MICRO_BATCH, cfg.BATCHES, cfg.SAMPLES), (4, 64, 256))
        self.assertEqual(cfg.ROLE_RANGES, dict(diagnostic=[0, 63]))
        self.assertEqual(cfg.CHANNEL_ROLE, 'validation')
        expected = tuple(f'{variant}_{arm}' for variant in cfg.VARIANTS for arm in cfg.ARMS)
        self.assertEqual(cfg.METHODS, expected)
        for variant in cfg.VARIANTS:
            for arm in cfg.ARMS:
                self.assertEqual(cfg.method_name(variant, arm), f'{variant}_{arm}')
        for variant, arm in (('oracle', 'fdma'), ('estimated_ul', 'ce'), ('trained', 'hybrid_g4')):
            with self.assertRaises(ValueError):
                cfg.method_name(variant, arm)
        observation = cfg.recipe()['observation']
        self.assertTrue(observation['same_observation_for_both_variants'])
        self.assertTrue(observation['common_actual_channels_noise_and_pilots_across_arms'])

    def test_pinned_source_bytes_and_legacy_33_core_are_unchanged(self):
        self.assertEqual(cfg.SOURCE_KIND, 'rate')
        self.assertEqual(cfg.SOURCE_ROOT, alphacfg.RATE_ROOT)
        self.assertEqual(cfg.SOURCE_COMMIT, alphacfg.RATE_COMMIT)
        self.assertEqual(cfg.SOURCE_CORE, alphacfg.RATE_CORE)
        self.assertEqual(cfg.SOURCE_CHECKPOINT_SHA256, alphacfg.RATE_CHECKPOINT_SHA256)
        self.assertEqual(cfg.SOURCE_TERMINAL_SHA256, alphacfg.RATE_TERMINAL_SHA256)
        self.assertEqual(cfg.LEGACY_CORE_FILES, wcfg.CORE_FILES)
        self.assertEqual(cfg.CORE_FILES[:33], wcfg.CORE_FILES)
        self.assertEqual(len(cfg.CORE_FILES), 39)
        self.assertEqual(len(set(cfg.CORE_FILES)), 39)
        self.assertEqual(set(cfg.PINNED_W_CORE['sha256']), set(wcfg.CORE_FILES))
        self.assertEqual(cfg.PINNED_W_CORE['schema'], wcfg.NAMESPACE+'_core_v1')
        self.assertEqual(cfg.PINNED_W_COMMIT, '8d7f0c2e2052ffbdfaa3377c1290cfe3aad317c9')
        for name, expected in cfg.PINNED_W_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT/name).read_bytes()).hexdigest(), expected, name)
        previous_outputs = (alignedcfg.OUTPUT_ROOT, ratecfg.OUTPUT_ROOT,
                            alphacfg.OUTPUT_ROOT, wcfg.OUTPUT_ROOT)
        self.assertNotIn(cfg.OUTPUT_ROOT, previous_outputs)
        self.assertEqual(cfg.OUTPUT_ROOT, cfg.ROOT/'runs'/cfg.NAMESPACE)

    def test_diagnostic_actual_validation_seed_addresses_do_not_reuse_old_experiments(self):
        spec = importlib.util.spec_from_file_location('_ul_diagnostic_data_metadata_test',
            cfg.ROOT/'utils/twc_v1_k8_replayable_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def data(namespace, device):
            return module.TWCv1K8ReplayableData(base_seed=cfg.SEED,
                batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
                device=device, seed_namespace=namespace)
        def old_ranges(previous):
            return [previous.ROLE_RANGES[role] for role in
                    ('monitor', 'alpha_calibration', 'calibration', 'final')
                    if role in previous.ROLE_RANGES]
        for device in ('cpu', 'cuda:0'):
            fresh = data(cfg.DATA_NAMESPACE, device)
            actual = {fresh.address('validation', i)['sionna_seed'] for i in range(cfg.BATCHES)}
            self.assertEqual(len(actual), cfg.BATCHES)
            for previous in (alignedcfg, ratecfg, alphacfg, wcfg):
                old = data(getattr(previous, 'DATA_NAMESPACE', previous.NAMESPACE), device)
                used = {old.address('validation', i)['sionna_seed']
                        for lo, hi in old_ranges(previous) for i in range(lo, hi+1)}
                self.assertTrue(actual.isdisjoint(used), (device, previous.NAMESPACE))

    def test_paired_noise_streams_are_independent_and_reject_training_roles(self):
        self.assertEqual(cfg.SEED, 20261121)
        self.assertEqual(cfg.DATA_NAMESPACE, cfg.NAMESPACE)
        seeds = {cfg.noise_seed('diagnostic', i, kind)
                 for i in range(cfg.BATCHES*len(cfg.GAMMAS)) for kind in ('noise', 'polarity')}
        self.assertEqual(len(seeds), cfg.BATCHES*len(cfg.GAMMAS)*2)
        self.assertEqual(cfg.noise_seed('diagnostic', 0), cfg.noise_seed('diagnostic', 0))
        for previous in (ratecfg, alphacfg, wcfg):
            old = {previous.noise_seed('final', i, kind)
                   for i in range(256*10) for kind in ('noise', 'polarity')}
            self.assertTrue(seeds.isdisjoint(old), previous.NAMESPACE)
        for role, index, kind in (('test', 0, 'noise'), ('codec_train', 0, 'noise'),
                                  ('diagnostic', -1, 'noise'), ('diagnostic', True, 'noise'),
                                  ('diagnostic', 0, 'gamma')):
            with self.assertRaises(ValueError):
                cfg.noise_seed(role, index, kind)

    def test_finite_calibration_edges_and_signed_summary_identity(self):
        self.assertEqual(cfg.CALIBRATION_EDGES, (0., .01, .03, .1, .3, 1., 3., 10.))
        self.assertTrue(all(math.isfinite(v) for v in cfg.CALIBRATION_EDGES))
        self.assertEqual(list(cfg.CALIBRATION_EDGES), sorted(set(cfg.CALIBRATION_EDGES)))
        self.assertIsNone(cfg.recipe()['calibration']['final_bin_upper_edge'])
        self.assertEqual(cfg.fingerprint(), cfg.digest(cfg.recipe()))
        self.assertEqual(cfg.study_recipe(), cfg.recipe())
        self.assertEqual(cfg.study_fingerprint(), cfg.fingerprint())
        recipe = cfg.recipe()
        recipe['reporting']['preserve_negative_results'] = False
        self.assertNotEqual(cfg.digest(recipe), cfg.fingerprint())


if __name__ == '__main__':
    unittest.main()
