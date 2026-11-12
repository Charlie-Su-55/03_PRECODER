"""Standard-library checks of residual-W fairness, provenance and data addresses."""
import hashlib
import importlib.util
import math
import unittest

from configs import twc_v1_k8_icc_aligned_config as alignedcfg
from configs import twc_v1_k8_icc_alpha_config as alphacfg
from configs import twc_v1_k8_icc_rate_config as ratecfg
from configs import twc_v1_k8_w_residual_config as cfg


class ResidualWConfigurationTests(unittest.TestCase):
    def test_four_training_runs_and_six_final_methods_are_unambiguous(self):
        self.assertEqual(cfg.ARMS, ('fdma', 'hybrid_g4'))
        self.assertEqual(cfg.TRAIN_METHODS, ('rzf_continue', 'w_residual'))
        self.assertEqual(cfg.POLICIES, ('rate_source', 'rzf_continue', 'w_residual'))
        self.assertEqual(cfg.METHODS, tuple(f'{p}_{a}' for p in cfg.POLICIES for a in cfg.ARMS))
        self.assertEqual(len(set(cfg.METHODS)), 6)
        self.assertEqual({cfg.key(a, m) for a in cfg.ARMS for m in cfg.TRAIN_METHODS},
                         set(cfg.METHODS) - {cfg.method_name('rate_source', a) for a in cfg.ARMS})
        for arm in cfg.ARMS:
            for policy in cfg.POLICIES:
                self.assertEqual(cfg.method_name(policy, arm), f'{policy}_{arm}')
        for arm, method in (('ce', 'w_residual'), ('fdma', 'rate_source'), ('fdma', 'direct_W')):
            with self.assertRaises(ValueError):
                cfg.key(arm, method)
            with self.assertRaises(ValueError):
                cfg.recipe(arm, method)
        with self.assertRaises(ValueError):
            cfg.method_name('oracle', 'fdma')

    def test_equal_fresh_optimizer_budget_and_frozen_feedback(self):
        study = cfg.study_recipe()
        self.assertEqual(cfg.UPDATES, 5000)
        self.assertEqual(study['sample_budget']['train_runs'], 4)
        self.assertEqual(study['sample_budget']['additional_MU_frames_per_arm_method'], 160000)
        self.assertEqual(study['sample_budget']['unique_paired_additional_MU_frames'], 160000)
        self.assertEqual(study['sample_budget']['additional_UE_inputs_per_arm_method'], 1280000)
        optimizer = dict(name='AdamW', fresh=True, lr=1e-5, weight_decay=1e-4,
                         betas=[.9, .999], eps=1e-8, clip=1., scheduler='constant LambdaLR')
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                recipe = cfg.recipe(arm, method)
                self.assertEqual(recipe['optimizer'], optimizer)
                self.assertEqual(recipe['updates'], 5000)
                self.assertEqual(recipe['microbatch'], 4)
                self.assertEqual(recipe['accumulation'], 8)
                self.assertEqual(recipe['effective_batch'], 32)
                self.assertTrue(recipe['base_decoder_trainable_for_both_methods'])
                self.assertFalse(recipe['source']['source_optimizer_continued'])
                self.assertTrue(recipe['source']['source_decoder_shared_between_methods'])
                self.assertEqual(recipe['residual_head_enabled'], method == 'w_residual')
                self.assertFalse(recipe['auxiliary_CSI_loss'])
                self.assertFalse(recipe['power_prior']['update'])
                self.assertEqual(recipe['power_prior']['source_EMA_commits'], 20000)
                for name in ('new_CE_updates', 'new_UE_updates', 'new_EMA_updates'):
                    self.assertEqual(recipe['sample_budget'][name], 0)
                self.assertEqual(recipe['frozen'], study['frozen'])
        self.assertEqual(cfg.WEIGHT_DECAY, ratecfg.WEIGHT_DECAY)

    def test_physical_protocol_and_rzf_base_are_unchanged(self):
        study = cfg.study_recipe()
        old = ratecfg.recipe('fdma')
        for field in ('physical', 'precision', 'gamma', 'Fourier_LS', 'CE_calibration'):
            self.assertEqual(study[field], old[field])
        self.assertEqual(cfg.GAMMAS, (0, 5, 10, 15, 20, 25))
        self.assertEqual(cfg.FIXED_ALPHA, .03)
        self.assertEqual(study['precoder']['alpha'], .03)
        self.assertEqual(study['precoder']['per_tone_total_power'], 1.)
        self.assertFalse(study['precoder']['trainable_alpha'])
        self.assertFalse(study['precoder']['pure_direct_W'])
        self.assertEqual(study['final']['DL_SNR'], 25.)
        self.assertFalse(study['final']['test_used'])
        self.assertEqual(study['monitor']['samples'], 128)
        self.assertEqual(study['final']['samples'], 1024)
        self.assertFalse(study['monitor']['selection'])

    def test_head_is_shared_zero_initialized_residual_not_pure_direct_w(self):
        head = cfg.study_recipe()['head']
        self.assertEqual((head['width'], head['layers'], head['heads'], head['dropout']), (64, 2, 4, 0.))
        self.assertEqual((head['input_features'], head['feedforward']), (76, 128))
        self.assertEqual(head['seed'], 20261112)
        self.assertTrue(head['shared_initialization_across_arms'])
        self.assertTrue(head['zero_output_initialization'])
        self.assertFalse(head['true_channel_input'])
        self.assertEqual(head['delta_scale'], 1. / math.sqrt(32 * 8))

    def test_counts_and_complete_microbatch_addresses(self):
        self.assertEqual(cfg.counters(5000), dict(optimizer_updates=5000, microbatches=40000,
            MU_frames=160000, UE_inputs=1280000, next_micro_index=40001,
            accumulation_cursor=0, EMA_commits=0))
        self.assertEqual(cfg.counters(0)['next_micro_index'], 1)
        self.assertEqual(cfg.micro_index(1, 0), 1)
        self.assertEqual(cfg.micro_index(5000, 7), 40000)
        for step in (-1, 5001, True, 100.):
            with self.assertRaises(ValueError):
                cfg.counters(step)
        for update, micro in ((0, 0), (5001, 0), (1, 8), (True, 0), (1, False)):
            with self.assertRaises(ValueError):
                cfg.micro_index(update, micro)

    def test_actual_seed_sets_are_disjoint_from_all_three_previous_experiments(self):
        path = cfg.ROOT / 'utils/twc_v1_k8_replayable_data.py'
        spec = importlib.util.spec_from_file_location('_w_residual_protocol_metadata_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def data(namespace, device):
            return module.TWCv1K8ReplayableData(base_seed=cfg.SEED,
                batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
                device=device, seed_namespace=namespace)
        def seeds(protocol, role, ranges):
            size = 1 << protocol['address']['index_bits']
            tag = protocol['roles'][role] << protocol['address']['index_bits']
            offset = protocol['role_offsets'][role]
            return {tag | ((offset + index) % size) for lo, hi in ranges for index in range(lo, hi + 1)}
        for device in ('cpu', 'cuda:0'):
            fresh = data(cfg.DATA_NAMESPACE, device)
            protocol = fresh.protocol
            train = seeds(protocol, 'codec_train', [cfg.ROLE_RANGES['codec_train']])
            monitor = seeds(protocol, 'validation', [cfg.ROLE_RANGES['monitor']])
            final = seeds(protocol, 'validation', [cfg.ROLE_RANGES['final']])
            self.assertEqual(len(train), 40000)
            self.assertEqual(len(monitor), 32)
            self.assertEqual(len(final), 256)
            self.assertTrue(monitor.isdisjoint(final))
            for role, ranges, actual in (('codec_train', [cfg.ROLE_RANGES['codec_train']], train),
                    ('validation', [cfg.ROLE_RANGES['monitor'], cfg.ROLE_RANGES['final']], monitor | final)):
                for lo, hi in ranges:
                    for index in (lo, hi):
                        self.assertIn(fresh.address(role, index)['sionna_seed'], actual)
            for oldcfg in (alignedcfg, ratecfg, alphacfg):
                previous = data(oldcfg.DATA_NAMESPACE if hasattr(oldcfg, 'DATA_NAMESPACE') else oldcfg.NAMESPACE,
                                device).protocol
                old_training = ([oldcfg.ROLE_RANGES['codec_train']]
                                if 'codec_train' in oldcfg.ROLE_RANGES else [])
                self.assertTrue(train.isdisjoint(seeds(previous, 'codec_train', old_training)))
                old_validation = [oldcfg.ROLE_RANGES[role] for role in
                                  ('monitor', 'alpha_calibration', 'calibration', 'final')
                                  if role in oldcfg.ROLE_RANGES]
                seen = seeds(previous, 'validation', old_validation)
                self.assertTrue(monitor.isdisjoint(seen), (device, oldcfg.NAMESPACE))
                self.assertTrue(final.isdisjoint(seen), (device, oldcfg.NAMESPACE))

    def test_noise_streams_are_role_separated_without_method_or_arm_terms(self):
        self.assertEqual(cfg.DATA_NAMESPACE, cfg.NAMESPACE)
        self.assertEqual(cfg.SEED, 20261121)
        self.assertEqual(cfg.ROLE_RANGES, dict(codec_train=[1, 40000], monitor=[0, 31], final=[2048, 2303]))
        for kind in ('noise', 'gamma', 'polarity'):
            self.assertEqual(cfg.noise_seed('codec_train', 1, kind), cfg.noise_seed('codec_train', 1, kind))
            self.assertNotEqual(cfg.noise_seed('monitor', 0, kind), cfg.noise_seed('final', 0, kind))
            for previous in (ratecfg, alphacfg):
                self.assertNotEqual(cfg.noise_seed('final', 2048, kind), previous.noise_seed('final', 2048, kind))
        for role, index, kind in (('test', 0, 'noise'), ('final', True, 'noise'),
                                  ('codec_train', -1, 'noise'), ('final', 2048, 'unknown')):
            with self.assertRaises(ValueError):
                cfg.noise_seed(role, index, kind)

    def test_sources_and_all_legacy_core_files_are_retained(self):
        self.assertEqual(cfg.SOURCE_ROOT, ratecfg.OUTPUT_ROOT)
        self.assertEqual(cfg.SOURCE_COMMIT, alphacfg.RATE_COMMIT)
        self.assertEqual(cfg.SOURCE_CORE, alphacfg.RATE_CORE)
        self.assertEqual(cfg.SOURCE_CHECKPOINT_SHA256, alphacfg.RATE_CHECKPOINT_SHA256)
        self.assertEqual(cfg.SOURCE_TERMINAL_SHA256, alphacfg.RATE_TERMINAL_SHA256)
        self.assertEqual(cfg.ORIGINAL_SOURCE_CORE, ratecfg.SOURCE_CORE)
        self.assertEqual(cfg.ORIGINAL_SOURCE_CHECKPOINT_SHA256, ratecfg.SOURCE_CHECKPOINT_SHA256)
        self.assertEqual(cfg.SOURCE_CALIBRATION_SHA256, ratecfg.SOURCE_CALIBRATION_SHA256)
        self.assertEqual(cfg.LEGACY_CORE['schema'], alphacfg.NAMESPACE + '_core_v1')
        self.assertEqual(set(cfg.LEGACY_CORE['sha256']), set(alphacfg.CORE_FILES))
        self.assertEqual({name: cfg.LEGACY_CORE['sha256'][name] for name in cfg.SOURCE_CORE['sha256']},
                         cfg.SOURCE_CORE['sha256'])
        for name, expected in cfg.LEGACY_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(cfg.CORE_FILES[:len(alphacfg.CORE_FILES)], alphacfg.CORE_FILES)
        self.assertEqual(len(cfg.CORE_FILES), 33)
        self.assertEqual(len(set(cfg.CORE_FILES)), 33)
        self.assertNotIn(cfg.OUTPUT_ROOT, (cfg.SOURCE_ROOT, cfg.ORIGINAL_SOURCE_ROOT, alphacfg.OUTPUT_ROOT))

    def test_fingerprints_preserve_declared_controls_and_signed_results(self):
        study = cfg.study_recipe()
        self.assertEqual(cfg.study_fingerprint(), cfg.digest(study))
        fingerprints = set()
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                self.assertEqual(cfg.fingerprint(arm, method), cfg.digest(cfg.recipe(arm, method)))
                fingerprints.add(cfg.fingerprint(arm, method))
        self.assertEqual(len(fingerprints), 4)
        for key in ('preserve_all_gamma', 'preserve_negative_results', 'within_MU_gamma_average',
                    'retain_all_user_rates', 'retain_min_user_rate'):
            self.assertTrue(study['reporting'][key])
        self.assertEqual(study['reporting']['primary_scope'], 'all_six')
        self.assertEqual(study['reporting']['primary_metric'], 'sum_rate_bps_hz')
        self.assertEqual(study['reporting']['primary_comparisons'],
            ['fdma_w_residual_minus_rzf_continue', 'hybrid_g4_w_residual_minus_rzf_continue'])
        self.assertEqual(study['reporting']['interaction_comparison'],
                         'interaction_w_residual_minus_rzf_continue')
        self.assertEqual(study['reporting']['training_seed_replications'], 1)
        study['updates_per_arm_method'] += 100
        self.assertNotEqual(cfg.digest(study), cfg.study_fingerprint())


if __name__ == '__main__':
    unittest.main()
