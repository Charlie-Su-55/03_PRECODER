"""No-ML checks of frozen-filter scope, old pins and fresh replay addresses."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest

from configs import twc_v1_k8_icc_aligned_config as alignedcfg
from configs import twc_v1_k8_icc_alpha_config as alphacfg
from configs import twc_v1_k8_icc_rate_config as ratecfg
from configs import twc_v1_k8_w_residual_config as wcfg
from configs import twc_v1_k8_ul_diagnostics_config as ulcfg
from configs import twc_v1_k8_ul_propagation_config as cfg


class ULPropagationConfigurationTests(unittest.TestCase):
    def test_zero_updates_fixed_filter_and_unchanged_paid_protocol(self):
        recipe = cfg.recipe()
        self.assertFalse(recipe['training'])
        self.assertFalse(recipe['backward'])
        self.assertEqual(recipe['trainable'], [])
        for name in ('optimizer_updates', 'new_UE_updates', 'new_CE_updates',
                     'new_BS_updates', 'new_EMA_updates'):
            self.assertEqual(recipe[name], 0)
        for name in ('physical', 'precision', 'Fourier_LS', 'CE_calibration'):
            self.assertEqual(recipe[name], ratecfg.recipe('fdma')[name])
        self.assertEqual(recipe['physical']['total_RE'], 256)
        self.assertEqual(recipe['physical']['Ep'], 16)
        self.assertEqual(recipe['physical']['Ed'], 48)
        receiver = recipe['receiver']
        self.assertFalse(receiver['true_UL_used_to_design_filter'])
        self.assertFalse(receiver['true_UL_used_to_recompute_soft'])
        self.assertFalse(receiver['current_frame_power_is_receiver_input'])
        self.assertTrue(receiver['historical_nominal_prior_frozen'])
        self.assertTrue(receiver['check_A_times_Y_equals_original_soft'])
        self.assertTrue(receiver['check_reported_variance_equals_original'])
        self.assertTrue(recipe['scoring']['retain_signed_cross_terms'])
        self.assertTrue(recipe['scoring']['true_UL_for_offline_scoring_only'])
        for name in ('downlink_model_forward', 'performance_upper_bound_claim',
                     'deployable_method_claim', 'checkpoint_selection', 'test_used'):
            self.assertFalse(recipe['reporting'][name])

    def test_fixed_six_gamma_budget_and_separate_output(self):
        self.assertEqual(cfg.ARMS, ('fdma', 'hybrid_g4'))
        self.assertEqual(cfg.GAMMAS, (0, 5, 10, 15, 20, 25))
        self.assertEqual((cfg.MICRO_BATCH, cfg.BATCHES, cfg.SAMPLES), (4, 64, 256))
        self.assertEqual(cfg.ROLE_RANGES, dict(diagnostic=[0, 63]))
        self.assertEqual(cfg.CHANNEL_ROLE, 'validation')
        self.assertEqual(cfg.MSE_VIEWS, ('reported', 'conditional_true_ul', 'empirical'))
        self.assertEqual(cfg.OUTPUT_ROOT, cfg.ROOT/'runs'/cfg.NAMESPACE)
        self.assertNotIn(cfg.OUTPUT_ROOT,
            [p.OUTPUT_ROOT for p in (alignedcfg, ratecfg, alphacfg, wcfg, ulcfg)])
        self.assertTrue(cfg.recipe()['data']['same_observation_filter_soft_for_all_MSE_views'])

    def test_actual_39_legacy_hashes_and_source_pins_remain_unchanged(self):
        self.assertEqual(cfg.SOURCE_KIND, 'rate')
        for name in ('SOURCE_ROOT', 'SOURCE_COMMIT', 'SOURCE_CORE',
                     'SOURCE_CHECKPOINT_SHA256', 'SOURCE_TERMINAL_SHA256',
                     'ORIGINAL_SOURCE_ROOT', 'SOURCE_CALIBRATION_SHA256'):
            self.assertEqual(getattr(cfg, name), getattr(ulcfg, name))
        self.assertEqual(cfg.LEGACY_CORE_FILES, ulcfg.CORE_FILES)
        self.assertEqual(cfg.CORE_FILES[:39], ulcfg.CORE_FILES)
        self.assertEqual(len(cfg.CORE_FILES), 46)
        self.assertEqual(len(set(cfg.CORE_FILES)), 46)
        self.assertEqual(set(cfg.PINNED_UL_CORE['sha256']), set(ulcfg.CORE_FILES))
        self.assertEqual(cfg.PINNED_UL_CORE['schema'], ulcfg.NAMESPACE+'_core_v1')
        self.assertEqual(cfg.PINNED_UL_COMMIT, '40c931f6be702951b14a9e84e8e8eccb80d49b67')
        for name, expected in cfg.PINNED_UL_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT/name).read_bytes()).hexdigest(), expected, name)

    def test_actual_validation_addresses_are_disjoint_from_five_prior_experiments(self):
        spec = importlib.util.spec_from_file_location('_ul_propagation_metadata_test',
            cfg.ROOT/'utils/twc_v1_k8_replayable_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def data(namespace, device):
            return module.TWCv1K8ReplayableData(base_seed=cfg.SEED,
                batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
                device=device, seed_namespace=namespace)
        for device in ('cpu', 'cuda:0'):
            fresh = data(cfg.DATA_NAMESPACE, device)
            seeds = {fresh.address('validation', i)['sionna_seed'] for i in range(cfg.BATCHES)}
            self.assertEqual(len(seeds), cfg.BATCHES)
            for previous in (alignedcfg, ratecfg, alphacfg, wcfg, ulcfg):
                old = data(getattr(previous, 'DATA_NAMESPACE', previous.NAMESPACE), device)
                intervals = [previous.ROLE_RANGES[role] for role in
                    ('monitor', 'alpha_calibration', 'calibration', 'final', 'diagnostic')
                    if role in previous.ROLE_RANGES]
                used = {old.address('validation', i)['sionna_seed']
                        for lo, hi in intervals for i in range(lo, hi+1)}
                self.assertTrue(seeds.isdisjoint(used), (device, previous.NAMESPACE))

    def test_new_paired_random_streams_do_not_reuse_previous_noise(self):
        self.assertEqual(cfg.SEED, 20261121)
        self.assertEqual(cfg.DATA_NAMESPACE, cfg.NAMESPACE)
        fresh = {cfg.noise_seed('diagnostic', i, kind)
            for i in range(cfg.BATCHES*len(cfg.GAMMAS)) for kind in ('noise', 'polarity')}
        self.assertEqual(len(fresh), cfg.BATCHES*len(cfg.GAMMAS)*2)
        for previous in (ratecfg, alphacfg, wcfg, ulcfg):
            role = 'diagnostic' if previous is ulcfg else 'final'
            used = {previous.noise_seed(role, i, kind)
                    for i in range(256*10) for kind in ('noise', 'polarity')}
            self.assertTrue(fresh.isdisjoint(used), previous.NAMESPACE)
        for role, index, kind in (('test', 0, 'noise'), ('codec_train', 0, 'noise'),
                                  ('diagnostic', -1, 'noise'), ('diagnostic', True, 'noise'),
                                  ('diagnostic', 0, 'gamma')):
            with self.assertRaises(ValueError):
                cfg.noise_seed(role, index, kind)

    def test_public_scalar_git_delivery_is_explicit_and_deletion_is_not_erasure(self):
        delivery = cfg.recipe()['delivery']
        self.assertIn('no ZIP', delivery['format'])
        self.assertTrue(delivery['git_upload_is_explicit'])
        self.assertTrue(delivery['repository_is_public'])
        self.assertFalse(delivery['deleting_branch_guarantees_history_erasure'])
        for name in ('contains_model_weights', 'contains_channel_tensors',
                     'contains_credentials_or_machine_environment'):
            self.assertFalse(delivery[name])
        self.assertEqual(cfg.RESULT_BRANCH_PREFIX, 'results/'+cfg.NAMESPACE+'/')
        self.assertEqual(cfg.RESULTS_BRANCH_PREFIX, cfg.RESULT_BRANCH_PREFIX)
        self.assertEqual(cfg.REMOTE_URL, 'https://github.com/Charlie-Su-55/03_PRECODER.git')

    def test_recipe_fingerprint_and_stdlib_import(self):
        self.assertEqual(cfg.fingerprint(), cfg.digest(cfg.recipe()))
        self.assertEqual(cfg.study_recipe(), cfg.recipe())
        self.assertEqual(cfg.study_fingerprint(), cfg.fingerprint())
        recipe = cfg.recipe()
        recipe['receiver']['true_UL_used_to_design_filter'] = True
        self.assertNotEqual(cfg.digest(recipe), cfg.fingerprint())
        command = 'from configs import twc_v1_k8_ul_propagation_config as c; print(c.NAMESPACE)'
        actual = subprocess.check_output([sys.executable, '-B', '-S', '-c', command],
            cwd=cfg.ROOT, text=True).strip()
        self.assertEqual(actual, cfg.NAMESPACE)
        json.dumps(cfg.recipe(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
