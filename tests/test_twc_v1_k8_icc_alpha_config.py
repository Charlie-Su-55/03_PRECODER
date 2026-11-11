"""Standard-library protocol and provenance checks for inference-only alpha control."""
import hashlib
import importlib.util
import math
from pathlib import Path
import unittest

from configs import twc_v1_k8_icc_alpha_config as cfg
from configs import twc_v1_k8_icc_rate_config as ratecfg


class AlphaConfigurationTests(unittest.TestCase):
    def test_three_policies_share_physical_protocol_and_no_updates(self):
        recipe=cfg.recipe()
        self.assertEqual(recipe['physical'],ratecfg.recipe('fdma')['physical'])
        self.assertEqual(recipe['precision'],ratecfg.recipe('fdma')['precision'])
        self.assertFalse(recipe['training'])
        self.assertFalse(recipe['backward'])
        self.assertEqual(recipe['optimizer_updates'],0)
        self.assertEqual(recipe['trainable'],[])
        for key in ('new_CE_updates','new_UE_updates','new_BS_updates','new_EMA_updates'):
            self.assertEqual(recipe[key],0)
        self.assertFalse(recipe['current_frame_power_receiver_input'])
        self.assertEqual(recipe['final']['per_tone_total_power'],1.)
        self.assertEqual(recipe['final']['DL_SNR'],25.)
        self.assertFalse(recipe['final']['test_used'])
        self.assertEqual(cfg.POLICIES,('source_fixed','source_calibrated','rate_fixed'))
        self.assertEqual(set(cfg.METHODS),{f'{p}_{a}' for p in cfg.POLICIES for a in cfg.ARMS})
        self.assertEqual(len(cfg.METHODS),6)
        for p in cfg.POLICIES:
            for a in cfg.ARMS:self.assertEqual(cfg.method_name(p,a),f'{p}_{a}')
        for p,a in (('direct_W','fdma'),('source_fixed','ce')):
            with self.assertRaises(ValueError):cfg.method_name(p,a)

    def test_calibration_grid_is_common_global_and_locked_before_final(self):
        recipe=cfg.recipe();cal=recipe['calibration'];final=recipe['final']
        self.assertEqual(cfg.ALPHA_GRID,(.003,.01,.02,.03,.05,.07,.1,.2,.3))
        self.assertEqual(tuple(sorted(set(cfg.ALPHA_GRID))),cfg.ALPHA_GRID)
        self.assertTrue(all(math.isfinite(v) and v>0 for v in cfg.ALPHA_GRID))
        self.assertIn(cfg.FIXED_ALPHA,cfg.ALPHA_GRID)
        self.assertEqual(cfg.FIXED_ALPHA,.03)
        self.assertTrue(cal['one_alpha_per_arm'])
        self.assertFalse(cal['per_gamma_selection'])
        self.assertTrue(cal['lock_before_final'])
        self.assertFalse(cal['final_inputs_may_select'])
        self.assertIn('exact ties prefer fixed alpha',cal['selection'])
        self.assertEqual(cal['gamma'],list(cfg.GAMMAS))
        self.assertEqual(final['gamma'],list(cfg.GAMMAS))
        self.assertEqual(cal['samples_per_gamma'],256)
        self.assertEqual(final['samples_per_gamma'],1024)
        self.assertEqual(cal['role'],'validation')
        self.assertEqual(final['role'],'validation')
        self.assertTrue(final['alpha_locked'])
        self.assertTrue(final['common_actual_channel_noise_and_pilots'])

    def test_actual_seed_intervals_do_not_reuse_previous_validation(self):
        path=cfg.ROOT/'utils/twc_v1_k8_replayable_data.py'
        spec=importlib.util.spec_from_file_location('_alpha_protocol_metadata_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def protocol(namespace,device):
            return module.TWCv1K8ReplayableData(base_seed=cfg.SEED,
                batch_sizes=dict(ce_train=4,codec_train=4,ce_calibration=4,validation=4,test=1),
                device=device,seed_namespace=namespace).protocol
        for device in ('cpu','cuda:0'):
            fresh=protocol(cfg.DATA_NAMESPACE,device)
            size=1<<fresh['address']['index_bits']
            def seeds(p,ranges):
                return {(p['role_offsets']['validation']+i)%size for lo,hi in ranges for i in range(lo,hi+1)}
            cal=seeds(fresh,[cfg.ROLE_RANGES['calibration']])
            final=seeds(fresh,[cfg.ROLE_RANGES['final']])
            self.assertEqual(len(cal),256//cfg.MICRO_BATCH)
            self.assertEqual(len(final),1024//cfg.MICRO_BATCH)
            self.assertTrue(cal.isdisjoint(final))
            for namespace in (ratecfg.DATA_NAMESPACE,'twc_v1_k8_icc_aligned_v1'):
                old=protocol(namespace,device)
                seen=seeds(old,[(0,63),(1024,1039),(2048,2303)])
                self.assertTrue(cal.isdisjoint(seen))
                self.assertTrue(final.isdisjoint(seen))
        self.assertNotEqual(cfg.DATA_NAMESPACE,ratecfg.DATA_NAMESPACE)
        self.assertNotEqual(cfg.OUTPUT_ROOT,cfg.RATE_ROOT)
        self.assertNotEqual(cfg.OUTPUT_ROOT,cfg.SOURCE_ROOT)

    def test_random_inputs_are_role_separated_and_arm_independent(self):
        for kind in ('noise','gamma','polarity'):
            self.assertEqual(cfg.noise_seed('final',7,kind),cfg.noise_seed('final',7,kind))
            self.assertNotEqual(cfg.noise_seed('calibration',7,kind),cfg.noise_seed('final',7,kind))
            self.assertNotEqual(cfg.noise_seed('final',7,kind),ratecfg.noise_seed('final',7,kind))
        self.assertNotEqual(cfg.noise_seed('final',7,'noise'),cfg.noise_seed('final',7,'polarity'))
        for role,index,kind in (('test',0,'noise'),('final',True,'noise'),('final',-1,'noise'),('final',0,'unknown')):
            with self.assertRaises(ValueError):cfg.noise_seed(role,index,kind)

    def test_completed_rate_and_source_provenance_are_pinned(self):
        self.assertEqual(cfg.RATE_COMMIT,'957b730d56f07fae9c93cb2611f5e460f7566e67')
        self.assertEqual(cfg.SOURCE_COMMIT,'9d345b2385d99eec1d047bd9c602638b568c84f8')
        self.assertEqual(len(cfg.RATE_CORE['sha256']),21)
        self.assertEqual(set(cfg.RATE_CORE['sha256']),set(ratecfg.CORE_FILES))
        for name,digest in cfg.RATE_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT/name).read_bytes()).hexdigest(),digest,name)
        self.assertEqual(cfg.SOURCE_CORE,ratecfg.SOURCE_CORE)
        self.assertEqual(cfg.SOURCE_CHECKPOINT_SHA256,ratecfg.SOURCE_CHECKPOINT_SHA256)
        self.assertEqual(cfg.SOURCE_TERMINAL_SHA256,ratecfg.SOURCE_TERMINAL_SHA256)
        self.assertEqual(cfg.SOURCE_CALIBRATION_SHA256,ratecfg.SOURCE_CALIBRATION_SHA256)
        self.assertEqual(set(cfg.RATE_CHECKPOINT_SHA256),set(cfg.ARMS))
        self.assertEqual(set(cfg.RATE_TERMINAL_SHA256),set(cfg.ARMS))
        for digest in (*cfg.RATE_CHECKPOINT_SHA256.values(),*cfg.RATE_TERMINAL_SHA256.values()):
            self.assertEqual(len(digest),64);int(digest,16)
        self.assertEqual(len(cfg.CORE_FILES),len(set(cfg.CORE_FILES)))
        for name in ('icc_alpha_runtime.py','evaluate_twc_v1_k8_icc_alpha.py',
                     'run_twc_v1_k8_icc_alpha.py','twc_v1_k8_icc_alpha_results.py'):
            self.assertIn(name,cfg.CORE_FILES)

    def test_recipe_hash_is_deterministic_and_preserves_all_results(self):
        recipe=cfg.recipe()
        self.assertEqual(cfg.fingerprint(),cfg.digest(recipe))
        self.assertEqual(cfg.fingerprint(),cfg.fingerprint())
        self.assertEqual(recipe['reporting']['primary_metric'],'sum_rate_bps_hz')
        self.assertEqual(recipe['reporting']['primary_scope'],'all_six')
        self.assertEqual(recipe['reporting']['primary_comparisons'],[
            'fdma_rate_fixed_minus_source_calibrated','hybrid_g4_rate_fixed_minus_source_calibrated'])
        self.assertEqual(recipe['reporting']['interaction_comparison'],'interaction_rate_gain_minus_calibration_gain')
        self.assertTrue(recipe['reporting']['preserve_all_gamma'])
        self.assertTrue(recipe['reporting']['preserve_negative_results'])
        self.assertTrue(recipe['reporting']['within_MU_gamma_average'])
        self.assertEqual(recipe['reporting']['paired_unit'],'complete8UE MU')
        self.assertEqual(recipe['reporting']['batch_interval'],4)
        self.assertEqual(recipe['reporting']['training_seed_replications'],1)
        recipe['alpha_grid']=[.03]
        self.assertNotEqual(cfg.digest(recipe),cfg.fingerprint())


if __name__=='__main__':unittest.main()
