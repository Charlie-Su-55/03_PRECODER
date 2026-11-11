"""Standard-library identity and accounting checks; no tensor/model execution."""
import copy
import hashlib
from pathlib import Path
import unittest

from configs import twc_v1_k8_icc_aligned_config as source
from configs import twc_v1_k8_icc_rate_config as cfg


class RateConfigurationTests(unittest.TestCase):
    def test_physical_protocol_is_unchanged_and_only_bs_is_trainable(self):
        for arm in cfg.ARMS:
            recipe = cfg.recipe(arm)
            old = source.recipe(arm)
            for field in ('physical', 'gamma', 'Fourier_LS', 'CE_calibration', 'precision'):
                self.assertEqual(recipe[field], old[field])
            self.assertEqual(recipe['loss'], 'negative arithmetic mean of complete-MU downlink sum-rate')
            self.assertFalse(recipe['auxiliary_CSI_loss'])
            self.assertTrue(recipe['rate_training'])
            self.assertEqual(recipe['trainable'], 'BS decoder parameters only')
            self.assertEqual(recipe['precoder']['alpha'], source.FIXED_ALPHA)
            self.assertFalse(recipe['precoder']['trainable_alpha'])
            self.assertFalse(recipe['precoder']['direct_W_head'])
            self.assertFalse(recipe['power_prior']['update'])
            self.assertEqual(recipe['power_prior']['source_EMA_commits'], 20000)
            self.assertEqual(recipe['power_prior']['additional_EMA_commits'], 0)
            self.assertTrue(recipe['final']['compare_source_and_finetuned_on_same_new_inputs'])
            self.assertFalse(recipe['final']['test_used'])
        with self.assertRaises(ValueError):
            cfg.recipe('ce')

    def test_fresh_budget_counts_do_not_continue_old_optimizer(self):
        self.assertEqual(cfg.UPDATES, 5000)
        self.assertEqual(cfg.LR, 1e-5)
        counts = cfg.counters(5000)
        self.assertEqual(counts, dict(optimizer_updates=5000, microbatches=40000,
            MU_frames=160000, UE_inputs=1280000, next_micro_index=40001,
            accumulation_cursor=0, EMA_commits=0))
        self.assertEqual(cfg.micro_index(1, 0), 1)
        self.assertEqual(cfg.micro_index(5000, 7), 40000)
        for value in (-1, 5001, True):
            with self.assertRaises(ValueError):
                cfg.counters(value)
        for update, micro in ((0, 0), (5001, 0), (1, 8), (True, 0)):
            with self.assertRaises(ValueError):
                cfg.micro_index(update, micro)
        for arm in cfg.ARMS:
            self.assertFalse(cfg.recipe(arm)['source']['original_optimizer_continued'])
            self.assertEqual(cfg.recipe(arm)['sample_budget']['new_UE_updates'], 0)
            self.assertEqual(cfg.recipe(arm)['sample_budget']['new_CE_updates'], 0)

    def test_independent_addresses_preserve_pairing_and_seal_test(self):
        import train_twc_v1_k8_icc_rate as runtime
        import train_twc_v1_k8_icc_aligned as old_runtime
        self.assertNotEqual(cfg.NAMESPACE, source.NAMESPACE)
        self.assertEqual(cfg.DATA_NAMESPACE, cfg.NAMESPACE)
        self.assertNotEqual(cfg.OUTPUT_ROOT, source.OUTPUT_ROOT)
        self.assertEqual(cfg.SOURCE_ROOT, source.OUTPUT_ROOT)
        self.assertEqual(cfg.ROLE_RANGES['codec_train'], [1, 40000])
        self.assertEqual(cfg.ROLE_RANGES['monitor'], [0, 31])
        self.assertEqual(cfg.ROLE_RANGES['final'], [2048, 2303])
        self.assertLess(cfg.ROLE_RANGES['monitor'][1], cfg.ROLE_RANGES['final'][0])
        self.assertNotEqual(cfg.noise_seed('codec_train', 1), source.noise_seed('codec_train', 1))
        self.assertNotEqual(cfg.noise_seed('monitor', 0), cfg.noise_seed('final', 0))
        self.assertNotEqual(cfg.noise_seed('codec_train', 1, 'noise'), cfg.noise_seed('codec_train', 1, 'gamma'))
        # Check actual Sionna seed intervals, not merely distinct namespaces.
        for device in ('cpu','cuda:0'):
            fresh=runtime.make_data(device).protocol
            old=old_runtime.make_data(device).protocol
            size=1<<fresh['address']['index_bits']
            new_train={(fresh['role_offsets']['codec_train']+i)%size for i in range(1,40001)}
            old_train={(old['role_offsets']['codec_train']+i)%size for i in range(1,160001)}
            self.assertTrue(new_train.isdisjoint(old_train))
            new_val={(fresh['role_offsets']['validation']+i)%size
                     for lo,hi in (cfg.ROLE_RANGES['monitor'],cfg.ROLE_RANGES['final']) for i in range(lo,hi+1)}
            old_val={(old['role_offsets']['validation']+i)%size
                     for lo,hi in (source.ROLE_RANGES['monitor'],source.ROLE_RANGES['alpha_calibration'],source.ROLE_RANGES['final']) for i in range(lo,hi+1)}
            self.assertTrue(new_val.isdisjoint(old_val))
        self.assertEqual(cfg.noise_seed('codec_train', 801), cfg.noise_seed('codec_train', 801))
        for role in ('test', 'ce_train', 'alpha_calibration'):
            with self.assertRaises(ValueError):
                cfg.noise_seed(role, 0)

    def test_original_core_bytes_and_completed_batch_pins(self):
        self.assertEqual(len(cfg.SOURCE_CORE['sha256']), 14)
        self.assertEqual(set(cfg.SOURCE_CORE['sha256']), set(source.CORE_FILES))
        for name, expected in cfg.SOURCE_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(set(cfg.SOURCE_CHECKPOINT_SHA256), {'ce', *cfg.ARMS})
        self.assertEqual(set(cfg.SOURCE_TERMINAL_SHA256), {'ce', *cfg.ARMS})
        self.assertEqual(cfg.SOURCE_COMMIT, '9d345b2385d99eec1d047bd9c602638b568c84f8')
        for value in (*cfg.SOURCE_CHECKPOINT_SHA256.values(), *cfg.SOURCE_TERMINAL_SHA256.values(),
                      cfg.SOURCE_CALIBRATION_SHA256):
            self.assertEqual(len(value), 64)
            int(value, 16)
        self.assertEqual(len(cfg.CORE_FILES), len(set(cfg.CORE_FILES)))
        self.assertIn('utils/twc_v1_k8_icc_rate.py', cfg.CORE_FILES)
        self.assertIn('twc_v1_k8_icc_aligned_results.py', cfg.CORE_FILES)


class RateResumeIdentityTests(unittest.TestCase):
    @staticmethod
    def payload(arm='fdma', step=100):
        # Metadata-only schema fixture; tensor serialization/update is covered
        # by the separate real CPU tests, never implied by these placeholders.
        return dict(format=cfg.NAMESPACE, arm=arm, step=step,
            recipe=cfg.recipe(arm), fingerprint=cfg.fingerprint(arm),
            core={'schema': 'test', 'sha256': {'test': 'a' * 64}},
            data_protocol={'seed_namespace': cfg.DATA_NAMESPACE},
            source_identity={'checkpoint_sha256': cfg.SOURCE_CHECKPOINT_SHA256[arm]},
            frozen_identity={'encoder': 'b' * 64, 'ce': 'c' * 64, 'prior': 'd' * 64},
            model={'metadata_fixture_only': 1}, optimizer={'metadata_fixture_only': 1},
            scheduler={'last_epoch': step, '_step_count': step + 1},
            rng={'python': [], 'numpy': [], 'torch_cpu': [], 'torch_cuda': []},
            counters=cfg.counters(step))

    def validate(self, value):
        original = self.payload(value['arm'])
        return cfg.validate_resume(value, value['arm'], core=original['core'],
            data_protocol=original['data_protocol'], source_identity=original['source_identity'],
            frozen_identity=original['frozen_identity'])

    def test_only_owned_complete_boundaries_are_accepted(self):
        for arm in cfg.ARMS:
            self.assertEqual(self.validate(self.payload(arm)), 100)
            self.assertEqual(self.validate(self.payload(arm, 5000)), 5000)
        for key in ('format', 'recipe', 'fingerprint', 'core', 'data_protocol',
                    'source_identity', 'frozen_identity'):
            value = self.payload(); value[key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.validate(value)
        for key, replacement in (('step', 101), ('step', True), ('step', 5100),
                ('counters', {}), ('scheduler', {'last_epoch': 100, '_step_count': 100}),
                ('rng', {'torch_cpu': []}), ('model', {}), ('optimizer', {})):
            value = self.payload(); value[key] = replacement
            with self.assertRaises(ValueError):
                self.validate(value)
        for key in ('optimizer', 'scheduler', 'frozen_identity', 'source_identity'):
            value = self.payload(); del value[key]
            with self.assertRaises(ValueError):
                self.validate(value)

    def test_optimizer_cursor_cannot_update_frozen_statistics(self):
        value = self.payload()
        value['counters']['EMA_commits'] = 100
        with self.assertRaisesRegex(ValueError, 'counts/cursor/frozen EMA'):
            self.validate(value)
        value = self.payload()
        value['counters']['next_micro_index'] += 1
        with self.assertRaises(ValueError):
            self.validate(value)
        changed = copy.deepcopy(self.payload())
        changed['recipe']['optimizer']['lr'] = 1e-4
        with self.assertRaisesRegex(ValueError, 'recipe'):
            self.validate(changed)


if __name__ == '__main__':
    unittest.main()
