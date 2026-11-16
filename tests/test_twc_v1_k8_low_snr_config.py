"""Independent count/budget/identity checks for the matched low-SNR recipe."""
import collections
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from configs import twc_v1_k8_low_snr_config as cfg


class LowSNRConfigTests(unittest.TestCase):
    def test_exact_75_25_counts_every_update_and_rotating_high_nodes(self):
        accumulated = collections.Counter()
        for step in range(1, 31):
            values = [g for m in range(8) for g in cfg.gamma_schedule(cfg.micro_index(step, m))]
            count = collections.Counter(values)
            self.assertEqual(len(values), 32)
            self.assertEqual([count[g] for g in cfg.GAMMAS[:3]], [8, 8, 8])
            self.assertEqual(sorted(count[g] for g in cfg.GAMMAS[3:]), [2, 3, 3])
            accumulated.update(values)
            self.assertEqual(cfg.gamma_counts(step), {str(g): accumulated[g] for g in cfg.GAMMAS})
        self.assertEqual(cfg.gamma_counts(cfg.UPDATES),
            {'0': 80000, '5': 80000, '10': 80000, '15': 26667, '20': 26667, '25': 26666})
        self.assertEqual(sum(cfg.gamma_counts(cfg.UPDATES).values()), 320000)

    def test_budget_source_and_matched_random_stream_identity_are_explicit(self):
        recipe = cfg.study_recipe()
        self.assertEqual(recipe['total_new_UE_updates'], 20000)
        self.assertEqual(recipe['FDMA_new_updates'], 0)
        self.assertEqual(cfg.counters(10000)['total_EMA_updates'], 50000)
        self.assertEqual(cfg.recipe('low_snr_control')['active_interference_weight'], 0)
        self.assertEqual(cfg.recipe('low_snr_interference')['active_interference_weight'], 1)
        self.assertNotEqual(cfg.fingerprint(cfg.TRAJECTORIES[0]), cfg.fingerprint(cfg.TRAJECTORIES[1]))
        self.assertEqual(cfg.study_fingerprint(), cfg.fingerprint())
        self.assertEqual(cfg.SOURCE_PINS['checkpoint_sha256'],
            '2260debba08a0dc754cefe53a18975b41e8cfacb3f3613dfbaa1a4498c8c1a97')
        for bad in ('fdma', 'hybrid_g4', 'test'):
            with self.assertRaises(ValueError): cfg.key(bad)
        from configs import twc_v1_k8_hybrid_direction_config as old
        self.assertNotEqual(cfg.noise_seed('codec_train', 1), old.noise_seed('codec_train', 1))
        self.assertNotEqual(cfg.noise_seed('codec_train', 1), cfg.noise_seed('final', 1))
        self.assertNotEqual(cfg.noise_seed('codec_train', 1), cfg.noise_seed('codec_train', 1, 'polarity'))
        self.assertEqual(len(cfg.CORE_FILES), len(set(cfg.CORE_FILES)))
        self.assertTrue(set(cfg.PARENT_CORE_FILES) <= set(cfg.CORE_FILES))

    def test_metadata_loads_without_site_packages_or_writes(self):
        script = ('from configs import twc_v1_k8_low_snr_config as c; '
            'import json; print(json.dumps(c.study_recipe()))')
        result = subprocess.run([sys.executable, '-B', '-S', '-c', script],
            cwd=cfg.ROOT, capture_output=True, text=True, check=True)
        recipe = json.loads(result.stdout)
        self.assertEqual(recipe['gamma']['feedback_low_fraction'], .75)
        self.assertFalse(recipe['data']['test_used'])


if __name__ == '__main__': unittest.main()
