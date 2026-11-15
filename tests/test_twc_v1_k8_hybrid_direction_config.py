"""Independent source, budget and fresh-data recipe checks without training."""
import json
import subprocess
import sys
import unittest

from configs import twc_v1_k8_hybrid_direction_config as cfg
from configs import twc_v1_k8_ue_bs_rate_config as parent


class HybridDirectionConfigTests(unittest.TestCase):
    def test_hybrid_only_budget_and_unequal_frozen_comparison(self):
        r = cfg.recipe()
        self.assertEqual((cfg.TRAIN_ARM, cfg.UPDATES), ('hybrid_g4', 10000))
        self.assertEqual((cfg.MICRO_BATCH, cfg.ACCUMULATION), (4, 8))
        self.assertEqual((cfg.LR, cfg.WEIGHT_DECAY, cfg.GRAD_CLIP), (1e-5, 1e-4, 1.))
        self.assertEqual((cfg.ALPHA, cfg.DL_SNR, cfg.DIRECTION_WEIGHT), (.03, 25., 100.))
        self.assertEqual(r['physical'], parent.study_recipe()['physical'])
        self.assertEqual(r['comparison']['FDMA_additional_updates'], 0)
        self.assertFalse(r['comparison']['equal_compute'])
        self.assertFalse(r['comparison']['auxiliary_effect_isolated'])
        self.assertFalse(r['selection']['best_checkpoint'])
        self.assertFalse(r['architecture']['user_projection'])
        with self.assertRaises(ValueError): cfg.key('fdma')
        with self.assertRaises(ValueError): cfg.recipe('fdma')
        end = cfg.counters(cfg.UPDATES)
        self.assertEqual((end['base_EMA_updates'], end['total_EMA_updates']), (30000, 40000))
        self.assertEqual((end['FDMA_updates'], end['UE_updates']), (0, 10000))

    def test_new_metadata_imports_without_tensor_packages(self):
        code = ('import json; from configs import twc_v1_k8_hybrid_direction_config as c; '
            'print(json.dumps(c.recipe())); import sys; '
            'assert "torch" not in sys.modules; assert "sionna" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-S', '-c', code], cwd=cfg.ROOT,
            text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)['namespace'], cfg.NAMESPACE)

    def test_balanced_schedule_and_distinct_fresh_random_roles(self):
        counts = {str(g): 0 for g in cfg.GAMMAS}
        for index in range(1, cfg.UPDATES*cfg.ACCUMULATION+1):
            for gamma in cfg.gamma_schedule(index): counts[str(gamma)] += 1
        self.assertEqual(counts, cfg.gamma_counts(cfg.UPDATES))
        self.assertEqual(sum(counts.values()), 320000)
        for first in (0, 3, 9996):
            group = {g: 0 for g in cfg.GAMMAS}
            for step in range(first+1, first+4):
                for micro in range(cfg.ACCUMULATION):
                    for gamma in cfg.gamma_schedule(cfg.micro_index(step, micro)): group[gamma] += 1
            self.assertEqual(set(group.values()), {16})
        self.assertNotEqual(cfg.NAMESPACE, parent.TRAIN_DATA_NAMESPACE)
        seeds = {cfg.noise_seed(role, index, kind) for role in ('codec_train', 'final')
            for index in (1, 2, 3) for kind in ('noise', 'polarity')}
        self.assertEqual(len(seeds), 12)
        for role, index in (('test', 1), ('monitor', 0), ('final', cfg.FINAL_BATCHES*6), ('codec_train', 0)):
            with self.assertRaises(ValueError): cfg.noise_seed(role, index)


if __name__ == '__main__': unittest.main()
