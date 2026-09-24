"""Server-side regression checks; no real model, channel generator or training.

Run from 03_PRECODER:
    python -m unittest discover -s tests -p test_snr_protocol.py -v
"""
from contextlib import nullcontext
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from configs.train_config import (
    EXPERIMENT_SUITES, TRAINING_RECIPES, ExperimentSpec, resolve_experiment,
)
from main_train import (
    format_val_table, get_downlink_snr, get_snr, validate_proposal, validation_metric,
)
from train_proposal_snr_robust import RECIPE, robust_config


def legacy_config(recipe_name='k4_n256_standard'):
    return resolve_experiment(ExperimentSpec(
        model_name='proposal', train_mode='specialist',
        k_users=4, subcarriers=256, feedback_budget=256,
        recipe_name=recipe_name, allocation=(0, 256), seed=42,
    ))


class SNRProtocolTests(unittest.TestCase):
    def test_legacy_none_keeps_snapshot_schema_and_matched_snr(self):
        for name, recipe in TRAINING_RECIPES.items():
            if name == RECIPE:
                continue
            with self.subTest(recipe=name):
                cfg = legacy_config(name)
                self.assertIsNone(cfg.DOWNLINK_SNR_DB)
                expected = asdict(recipe)
                expected.pop('downlink_snr_db')
                self.assertEqual(cfg.to_dict()['training_recipe'], expected)
                self.assertNotIn('downlink_snr_db', cfg.to_dict()['training_recipe'])
                for snr in (0, 5.125, 10, 15, 20, 25):
                    self.assertEqual(get_downlink_snr(snr, cfg), snr)
                    self.assertEqual(10.0 ** (-get_downlink_snr(snr, cfg) / 10.0), 10.0 ** (-snr / 10.0))

    def test_existing_suites_do_not_launch_new_recipe(self):
        for specs in EXPERIMENT_SUITES.values():
            self.assertTrue(all(spec.recipe_name != RECIPE for spec in specs))

    def test_robust_recipe_and_capacity(self):
        cfg = robust_config()
        expected = {
            'TOTAL_STEPS': 20000, 'BATCH_SIZE': 48, 'LR': 2e-5,
            'VAL_INTERVAL': 500, 'VAL_BATCH': 64, 'WARMUP_PCT': 0.10,
            'GRAD_CLIP': 0.5, 'DOWNLINK_SNR_DB': 25.0, 'WEIGHT_DECAY': 1e-5,
            'PH1_W_DIR': 0.0, 'PH1_W_MSE': 0.0, 'PH2_W_DIR': 0.0, 'PH2_W_MSE': 0.0,
            'SEED': 42, 'BEST_METRIC': 'mean_all', 'NUM_SUBBANDS': 4,
            'D_F_MAX_PER_USER': 64, 'D_A_MAX_SHARED': 256,
        }
        for name, value in expected.items():
            self.assertEqual(getattr(cfg, name), value)
        self.assertEqual(cfg.ALLOCATION_GRID, [(0, 256)])
        self.assertEqual(cfg.VAL_SNR_LIST, [0, 5, 10, 15, 20, 25])
        self.assertEqual(cfg.to_dict()['training_recipe']['downlink_snr_db'], 25.0)
        for snr in (0, 5.125, 10, 15, 20, 25):
            self.assertEqual(get_downlink_snr(snr, cfg), 25.0)

    def test_every_robust_stage_samples_uniform_0_25(self):
        cfg = robust_config()
        for step in (1, cfg.STAGE1_STEPS + 1, cfg.STAGE2_STEPS + 1, cfg.TOTAL_STEPS):
            with patch('main_train.np.random.uniform', return_value=12.345) as sample:
                self.assertEqual(get_snr(step, cfg), 12.345)
                sample.assert_called_once_with(0.0, 25.0)

    def check_validation_routing(self, cfg):
        feedback, noise, seeds = [], [], []
        class Model:
            def eval(self):
                pass
            def __call__(self, h_dl, h_ul, snr, df, da, idx):
                feedback.append((float(snr), df, da, idx))
                return torch.zeros(1)
        def criterion(w, h, noise_power):
            noise.append(noise_power)
            return 0.0, float(len(noise))
        def seeded(seed):
            seeds.append(seed)
            return nullcontext()
        with patch('main_train.temporary_seed', side_effect=seeded):
            table = validate_proposal(
                Model(), torch.zeros(1), torch.zeros(1), criterion, cfg, torch.device('cpu'),
            )
        self.assertEqual(feedback, [(float(snr), 0, 256, 0) for snr in cfg.VAL_SNR_LIST])
        self.assertEqual(seeds, [cfg.SEED + 1_000_000 + int(snr) * 10 for snr in cfg.VAL_SNR_LIST])
        self.assertEqual(noise, [
            10.0 ** (-float(snr if cfg.DOWNLINK_SNR_DB is None else 25.0) / 10.0)
            for snr in cfg.VAL_SNR_LIST
        ])
        self.assertEqual(set(table), {(0, snr) for snr in cfg.VAL_SNR_LIST})
        self.assertEqual(validation_metric(table, cfg), sum(table.values()) / len(table))
        return table

    def test_legacy_validation_keeps_noise_and_seeds(self):
        self.check_validation_routing(legacy_config())

    def test_robust_validation_feedback_grid_fixed_dl_and_mean(self):
        cfg = robust_config()
        table = self.check_validation_routing(cfg)
        text, metric = format_val_table(table, 500, cfg)
        self.assertIn('fixed DL SNR = 25 dB', text)
        self.assertIn('[0, 5, 10, 15, 20, 25]', text)
        self.assertEqual(metric, 3.5)


if __name__ == '__main__':
    unittest.main()
