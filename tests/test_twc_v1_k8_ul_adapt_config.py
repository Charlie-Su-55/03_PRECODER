"""Prespecified matched budget, exact six-node exposure and identity locks."""
from collections import Counter
import copy
import hashlib
import subprocess
import sys
import unittest

from configs import twc_v1_k8_ul_adapt_config as cfg
from configs import twc_v1_k8_ul_calibration_config as calibration
from configs import twc_v1_k8_icc_rate_config as rate
import ul_adapt_runtime as runtime


class ULAdaptConfigurationTests(unittest.TestCase):
    def test_only_decoder_changes_under_matched_10000_update_budget(self):
        study = cfg.study_recipe()
        self.assertEqual((cfg.UPDATES, cfg.MICRO_BATCH, cfg.ACCUMULATION), (10000, 4, 8))
        self.assertEqual(cfg.TRAIN_METHODS, ('original', 'moment_receiver'))
        self.assertEqual(cfg.METHODS, cfg.TRAIN_METHODS)
        self.assertEqual(cfg.PHASES, ('source', 'final'))
        self.assertEqual(study['sample_budget']['trajectories'], 4)
        self.assertEqual(study['sample_budget']['additional_MU_frames_per_trajectory'], 320000)
        self.assertEqual(study['sample_budget']['unique_paired_additional_MU_frames'], 320000)
        for field in ('new_UE_updates', 'new_CE_updates', 'new_EMA_updates', 'statistical_calibration_refits'):
            self.assertEqual(study['sample_budget'][field], 0)
        self.assertEqual(study['trainable'], 'BS decoder parameters only')
        self.assertFalse(study['source']['source_optimizer_continued'])
        self.assertFalse(study['receiver']['true_UL_deployment_input'])
        self.assertFalse(study['reporting']['checkpoint_selection'])
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                self.assertEqual(cfg.key(arm, method), cfg.method_name(method, arm))
                self.assertEqual(cfg.recipe(arm, method)['updates'], 10000)
                self.assertEqual(cfg.fingerprint(arm, method), cfg.digest(cfg.recipe(arm, method)))

    def test_balanced_training_stream_matches_all_saved_gamma_counters(self):
        observed = Counter({str(gamma):0 for gamma in cfg.GAMMAS})
        for step in range(1, cfg.UPDATES+1):
            for micro in range(cfg.ACCUMULATION):
                index = cfg.micro_index(step, micro)
                expected = tuple(cfg.GAMMAS[(4*(index-1)+j) % 6] for j in range(4))
                self.assertEqual(cfg.gamma_schedule(index), expected)
                observed.update(str(gamma) for gamma in expected)
            if step % 3 == 0:
                self.assertEqual(set(observed.values()), {step//3*16})
            if step % cfg.SAVE_EVERY == 0:
                self.assertEqual(dict(observed), cfg.counters(step)['gamma_counts'])
        self.assertEqual(dict(observed), {'0':53334, '5':53334, '10':53333,
            '15':53333, '20':53333, '25':53333})
        self.assertEqual(cfg.counters(0), dict(optimizer_updates=0, microbatches=0,
            MU_frames=0, UE_inputs=0, next_micro_index=1, accumulation_cursor=0,
            EMA_commits=0, gamma_counts={str(g):0 for g in cfg.GAMMAS}))
        for call, args in ((cfg.micro_index, (0, 0)), (cfg.micro_index, (1, 8)),
                (cfg.gamma_schedule, (True,)), (cfg.gamma_schedule, (80001,)),
                (cfg.counters, (True,)), (cfg.counters, (10001,))):
            with self.assertRaises(ValueError):
                call(*args)

    def test_all_54_previous_source_pins_and_locked_calibration_are_fixed(self):
        self.assertEqual(len(cfg.CORE_FILES), 61)
        self.assertEqual(len(set(cfg.CORE_FILES)), 61)
        self.assertEqual(cfg.CORE_FILES[:54], calibration.CORE_FILES)
        self.assertEqual(set(cfg.PINNED_CALIBRATION_CORE['sha256']), set(calibration.CORE_FILES))
        for name, digest in cfg.PINNED_CALIBRATION_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT/name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(cfg.CALIBRATION_DIR,
            calibration.OUTPUT_ROOT/'calibration'/'attempt_a0dbcecec7b3')
        self.assertEqual(cfg.CALIBRATION_MODEL_SHA256,
            'd0477d049f5808d9f9a34c0792767edc5fcdcbe8aa89951b5ff5a91b6d243be1')
        self.assertEqual(cfg.CALIBRATION_MANIFEST_SHA256,
            '0e86e2a78fcaca74438d194c1c4895a273c3316eb40f65ea832fbc74b742fe48')
        for field in ('SOURCE_KIND', 'SOURCE_ROOT', 'SOURCE_COMMIT', 'SOURCE_CORE',
                'SOURCE_CHECKPOINT_SHA256', 'SOURCE_TERMINAL_SHA256', 'ORIGINAL_SOURCE_ROOT',
                'SOURCE_CALIBRATION_SHA256'):
            self.assertEqual(getattr(cfg, field), getattr(calibration, field))
        for name in ('physical', 'precision', 'Fourier_LS', 'CE_calibration'):
            self.assertEqual(cfg.study_recipe()[name], rate.recipe('fdma')[name])
        self.assertEqual((cfg.ALPHA, cfg.DL_SNR), (.03, 25.))

    def test_train_monitor_final_addresses_are_separate_from_zero_shot_data(self):
        self.assertEqual(cfg.ROLE_RANGES,
            dict(codec_train=[1,80000], monitor=[0,31], final=[2048,2303]))
        self.assertEqual(cfg.MONITOR_STEPS, (0,100,1000,5000,10000))
        self.assertTrue(cfg.study_recipe()['final']['generated_only_after_all_four_terminals'])
        for device in ('cpu', 'cuda:0'):
            data = runtime.make_data(device)
            old = runtime.previous.make_data(device)
            self.assertNotEqual(data.protocol['data_id'], old.protocol['data_id'])
            train = {data.address('codec_train', i)['sionna_seed'] for i in range(1,80001)}
            monitor = {data.address('validation', i)['sionna_seed'] for i in range(32)}
            final = {data.address('validation', i)['sionna_seed'] for i in range(2048,2304)}
            self.assertEqual(len(train), 80000)
            self.assertTrue(train.isdisjoint(monitor|final))
            self.assertTrue(monitor.isdisjoint(final))
            original = {old.address(role, i)['sionna_seed']
                for role in ('ce_calibration', 'validation') for i in range(64)}
            self.assertTrue((train|monitor|final).isdisjoint(original))

    def test_random_streams_are_role_bounded_and_have_no_method_or_arm_term(self):
        self.assertEqual(cfg.noise_seed('codec_train', 1), cfg.noise_seed('codec_train', 1))
        self.assertNotEqual(cfg.noise_seed('codec_train', 1), cfg.noise_seed('monitor', 1))
        self.assertNotEqual(cfg.noise_seed('final', 0), cfg.noise_seed('final', 0, 'polarity'))
        for args in (('test',0), ([],0), ('codec_train',0), ('codec_train',80001),
                ('monitor',192), ('final',1536), ('final',True), ('final',0,'gamma')):
            with self.assertRaises(ValueError):
                cfg.noise_seed(*args)

    def test_resume_refuses_budget_method_calibration_cursor_and_type_changes(self):
        identities = dict(core={'schema':'fixture'}, data_protocol={'fixture':True},
            source_identity={'source':'pinned'}, frozen_identity={'frozen':'pinned'})
        payload = dict(format=cfg.NAMESPACE, arm='fdma', method='original', step=100,
            recipe=cfg.recipe('fdma','original'), fingerprint=cfg.fingerprint('fdma','original'),
            calibration_identity=cfg.calibration_identity(), **identities,
            counters=cfg.counters(100), model={'w':[1.]}, optimizer={'state':{}},
            scheduler=dict(last_epoch=100, _step_count=101),
            rng=dict(python=[],numpy=[],torch_cpu=[],torch_cuda=[]))
        self.assertEqual(cfg.validate_resume(payload,'fdma','original',**identities), 100)
        for mutation in ('method','budget','calibration','gamma_counts','cursor','EMA','boolEMA','step','scheduler','rng','model'):
            bad = copy.deepcopy(payload)
            if mutation == 'method': bad['method'] = 'moment_receiver'
            elif mutation == 'budget': bad['recipe']['updates'] += 1
            elif mutation == 'calibration': bad['calibration_identity']['model_sha256'] = '0'*64
            elif mutation == 'gamma_counts': bad['counters']['gamma_counts']['0'] += 1
            elif mutation == 'cursor': bad['counters']['accumulation_cursor'] = 1
            elif mutation == 'EMA': bad['counters']['EMA_commits'] = 1
            elif mutation == 'boolEMA': bad['counters']['EMA_commits'] = False
            elif mutation == 'step': bad['step'] = 101
            elif mutation == 'scheduler': bad['scheduler']['_step_count'] -= 1
            elif mutation == 'rng': bad['rng'].pop('torch_cuda')
            else: bad['model'] = {}
            with self.assertRaises(ValueError, msg=mutation):
                cfg.validate_resume(bad,'fdma','original',**identities)

    def test_stdlib_dry_configuration_and_explicit_scalar_delivery(self):
        self.assertEqual(cfg.study_fingerprint(), cfg.digest(cfg.study_recipe()))
        delivery = cfg.study_recipe()['delivery']
        self.assertIn('no ZIP', delivery['format'])
        self.assertTrue(delivery['git_upload_is_explicit'])
        for flag in ('contains_model_weights','contains_channel_tensors',
                'contains_credentials_or_machine_environment','deleting_branch_guarantees_history_erasure'):
            self.assertFalse(delivery[flag])
        code = ('import sys; import ul_adapt_runtime as r; r.make_data("cpu"); '
            'assert "torch" not in sys.modules; assert "sionna" not in sys.modules; print(r.cfg.NAMESPACE)')
        output = subprocess.check_output([sys.executable,'-B','-S','-c',code], cwd=cfg.ROOT, text=True)
        self.assertEqual(output.strip(), cfg.NAMESPACE)


if __name__ == '__main__':
    unittest.main()
