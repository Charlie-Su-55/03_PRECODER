"""Declared CE-only budgets, causal addresses, and exact SNR accounting."""
import unittest
from configs import twc_v1_k8_ce_refine_config as cfg


class CERecipeTests(unittest.TestCase):
    def test_fixed_budget_is_CE_only_and_source_is_not_retrained(self):
        self.assertEqual(cfg.TRAJECTORIES,('nmse','geometry'))
        self.assertEqual(cfg.UPDATES,60000)
        self.assertEqual(cfg.SOURCE_UPDATES,5000)
        self.assertEqual(cfg.SOURCE_CHECKPOINT_SHA256,'95c1e786b2036ecab81a865408555069ae3ce699620d1afacd14fc11982db494')
        counts=cfg.counters(cfg.UPDATES)
        self.assertEqual(counts['CE_total_updates'],65000)
        self.assertEqual(counts['CE_new_updates'],60000)
        self.assertEqual(counts['MU_frames'],1920000)
        self.assertEqual(counts['UE_links'],15360000)
        for k in ('UE_updates','BS_updates','FDMA_updates','EMA_commits'):
            self.assertEqual(counts[k],0)
        for trajectory in cfg.TRAJECTORIES:
            recipe=cfg.recipe(trajectory)
            self.assertEqual(recipe['trainable'],['pilot-only CE U-Net'])
            self.assertFalse(recipe['data']['test_used'])
            self.assertEqual(recipe['total_new_CE_updates'],120000)

    def test_balanced_actual_per_frame_schedule_and_counters_match(self):
        for step in (0,1,2,3,7,17):
            actual={str(g):0 for g in cfg.GAMMAS}
            for update in range(1,step+1):
                for micro in range(cfg.ACCUMULATION):
                    for gamma in cfg.gamma_schedule(cfg.micro_index(update,micro)):
                        actual[str(gamma)]+=1
            self.assertEqual(actual,cfg.gamma_counts(step))
        self.assertEqual(set(cfg.gamma_counts(cfg.UPDATES).values()),{320000})
        self.assertEqual(cfg.micro_index(60000,7),480000)
        for value in (False,-1,60001,1.0):
            with self.assertRaises(ValueError):cfg.counters(value)
        for value in (0,480001,True):
            with self.assertRaises(ValueError):cfg.gamma_schedule(value)

    def test_six_nodes_and_off_node_midpoints_have_disjoint_validation_addresses(self):
        self.assertEqual(cfg.EVAL_GAMMAS,tuple(2.5*i for i in range(11)))
        self.assertEqual(cfg.FINAL_BATCHES*cfg.MICRO_BATCH,1024)
        ranges=[cfg.ROLE_RANGES[k] for k in ('monitor','final','midpoint')]
        for a,b in ((ranges[0],ranges[1]),(ranges[0],ranges[2]),(ranges[1],ranges[2])):
            self.assertTrue(a[1]<b[0] or b[1]<a[0])
        self.assertEqual(cfg.MIDPOINT_GAMMAS,(2.5,7.5,12.5,17.5,22.5))

    def test_noise_address_has_no_trajectory_and_test_role_is_forbidden(self):
        seeds={cfg.noise_seed(role,1,kind) for role in ('ce_train','monitor','calibration','final','midpoint')
            for kind in ('noise','polarity')}
        self.assertEqual(len(seeds),10)
        self.assertEqual(cfg.noise_seed('ce_train',1),cfg.noise_seed('ce_train',1))
        with self.assertRaises(ValueError):cfg.noise_seed('test',0)
        with self.assertRaises(ValueError):cfg.key('fdma')

    def test_cosine_has_fixed_endpoints_and_loss_fingerprints_differ(self):
        self.assertAlmostEqual(cfg.learning_rate(0),1e-4)
        self.assertAlmostEqual(cfg.learning_rate(cfg.UPDATES),1e-5)
        self.assertGreater(cfg.learning_rate(1),cfg.learning_rate(cfg.UPDATES-1))
        self.assertNotEqual(cfg.fingerprint('nmse'),cfg.fingerprint('geometry'))
        self.assertEqual(cfg.recipe('nmse')['objective']['direction'],0.)
        self.assertEqual(cfg.recipe('geometry')['objective']['direction'],1.)
        self.assertIn('twc_v1_k8_ce_refine_results.py',cfg.CORE_FILES)
        self.assertEqual(len(cfg.CORE_FILES),len(set(cfg.CORE_FILES)))


if __name__=='__main__':unittest.main()
