"""F configuration/resume/CLI checks; no large model or real channel execution."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.grouped_codec_fdma_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, fdma_config, validate_config, make_channel_data, protocol,
)
from configs.grouped_codec_train_config import (
    ARMS as OLD_ARMS, grouped_codec_config, validate_config as validate_old_config,
)
from configs.train_config import TRAINING_RECIPES
from cli_contracts import assert_lightweight_cli
from test_train_grouped_codec import resume_metadata
import train_grouped_codec as old_entry
import train_grouped_codec_fdma as entry


class ConfigTests(unittest.TestCase):
    def test_real_f_identity_and_same_fixed_training_recipe(self):
        self.assertEqual(ARMS, ('legacy_fdma',))
        for seed in SEEDS:
            cfg = fdma_config(seed)
            validate_config(cfg)
            old = grouped_codec_config('legacy_all4', seed)
            self.assertEqual(cfg.arm, 'legacy_fdma')
            self.assertEqual(cfg.to_dict()['arm'], 'legacy_fdma')
            self.assertEqual(cfg.to_dict()['schema'], 'grouped-codec-fdma-config-v1')
            self.assertEqual(cfg.to_dict()['recipe'], old.to_dict()['recipe'])
            self.assertEqual((cfg.BATCH_SIZE, cfg.TOTAL_STEPS, cfg.VAL_BATCH, cfg.VAL_INTERVAL),
                             (48, 20000, 64, 500))
            self.assertEqual((cfg.LR, cfg.WEIGHT_DECAY, cfg.WARMUP_PCT, cfg.GRAD_CLIP),
                             (1e-4, 1e-5, .1, .5))
            self.assertEqual(cfg.VAL_SNR_LIST, [0, 5, 10, 15, 20, 25])
            self.assertEqual(cfg.DOWNLINK_SNR_DB, 25.)
            self.assertEqual([cfg.PH1_W_DIR, cfg.PH1_W_MSE, cfg.PH2_W_DIR, cfg.PH2_W_MSE], [0.] * 4)
            self.assertFalse(hasattr(cfg, 'ALLOCATION_GRID'))
            self.assertIsNone(cfg.to_dict()['initialization']['historical_checkpoint'])
            self.assertEqual(Path(cfg.SAVE_PATH).parents[2], OUTPUT_ROOT)
            self.assertNotEqual(entry.fingerprint(cfg), entry.fingerprint(old))
            with self.assertRaises(ValueError):
                validate_old_config(cfg)

    def test_fdma_is_64_active_slots_and_interleaved_physical_support(self):
        cfg = fdma_config(SEEDS[0])
        layout = cfg.layout_spec
        self.assertEqual(layout.to_dict(), protocol.LayoutSpec('fdma').to_dict())
        self.assertEqual((layout.active_per_user, layout.users_per_resource, layout.occupied_resources), (64, 1, 256))
        self.assertEqual(layout.active_slots, tuple(sb * 64 + j for sb in range(4) for j in range(16)))
        self.assertEqual(layout.supports, tuple(tuple(range(k, 256, 4)) for k in range(4)))
        for k, row in enumerate(layout.canonical_to_physical):
            self.assertEqual([row[s] for s in layout.active_slots], list(layout.supports[k]))
            self.assertEqual(sum(q >= 0 for q in row), 64)

    def test_shared_physical_random_and_channel_protocols_not_f_specific(self):
        for seed in SEEDS:
            cfg = fdma_config(seed)
            for arm in OLD_ARMS:
                old = grouped_codec_config(arm, seed)
                self.assertEqual(cfg.channel_data_protocol, old.channel_data_protocol)
                self.assertEqual(cfg.to_dict()['execution_protocol'], old.to_dict()['execution_protocol'])
            self.assertEqual(make_channel_data(cfg).protocol, cfg.channel_data_protocol)
        obj = entry.FDMAExecutionProtocol(fdma_config(SEEDS[0]))
        self.assertIs(type(obj).training, old_entry.GroupedExecutionProtocol.training)
        self.assertIs(type(obj).validate, old_entry.GroupedExecutionProtocol.validate)
        self.assertIs(type(obj).format_validation, old_entry.GroupedExecutionProtocol.format_validation)
        self.assertEqual(obj.descriptor, obj.cfg.to_dict()['execution_protocol'])
        self.assertEqual(protocol.PROTOCOL, 'grouped_feedback_energy64_absolute_noise_v1')
        self.assertEqual(protocol.noise_variance(0), .25)

    def test_legacy_as_encoder_and_bs_metadata_are_not_changed(self):
        cfg = fdma_config(SEEDS[0])
        old = grouped_codec_config('legacy_all4', SEEDS[0])
        metadata = cfg.architecture_metadata
        self.assertEqual(metadata['architecture'], 'grouped_codec_fdma_v1')
        self.assertEqual(metadata['model_class'], 'GroupedCodecFDMAPrecoder')
        self.assertEqual(metadata['arm'], 'legacy_fdma')
        self.assertEqual(metadata['ue'], old.architecture_metadata['ue'])
        self.assertEqual(metadata['bs'], old.architecture_metadata['bs'])
        self.assertEqual(metadata['protocol'], old.architecture_metadata['protocol'])
        self.assertTrue(metadata['ue']['stored_original_fdma_branch'])
        self.assertFalse(metadata['ue']['fdma_branch_executed'])
        self.assertEqual(metadata['layout_spec'], cfg.layout_spec.to_dict())

    def test_no_registry_mutation_and_changed_config_rejected(self):
        registry = deepcopy(TRAINING_RECIPES)
        old_arms = tuple(protocol.ARMS)
        cfg = fdma_config(SEEDS[0])
        self.assertEqual(TRAINING_RECIPES, registry)
        self.assertEqual(protocol.ARMS, old_arms)
        self.assertNotIn(cfg.arm, protocol.ARMS)
        for kind in ('arm', 'recipe', 'channel'):
            changed = deepcopy(cfg)
            if kind == 'arm':
                changed.arm = 'ordered_fdma'
            elif kind == 'recipe':
                changed.recipe = replace(changed.recipe, lr=2e-5)
            else:
                changed.channel_data_protocol['base_seed'] += 1
            self.assertNotEqual(entry.fingerprint(cfg), entry.fingerprint(changed))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_config(changed)
        with self.assertRaises(ValueError):
            fdma_config(42)
        with self.assertRaises(ValueError):
            validate_config(grouped_codec_config('legacy_all4', SEEDS[0]))


class ResumeTests(unittest.TestCase):
    def test_smoke_is_part_of_20k_plan_and_completed_only_explicit(self):
        for seed in SEEDS:
            cfg = fdma_config(seed)
            entry.validate_resume_payload(resume_metadata(cfg, 500), cfg)
            entry.validate_resume_payload(resume_metadata(cfg, 20000), cfg, allow_completed=True)
            with self.assertRaises(ValueError):
                entry.validate_resume_payload(resume_metadata(cfg, 20000), cfg)

    def test_old_arms_other_seed_and_missing_resume_state_rejected(self):
        cfg = fdma_config(SEEDS[0])
        for other in (fdma_config(SEEDS[1]), grouped_codec_config('legacy_all4', SEEDS[0]),
                      grouped_codec_config('ordered_fdma', SEEDS[0])):
            with self.assertRaises(ValueError):
                entry.validate_resume_payload(resume_metadata(other), cfg)
        for key in ('config', 'config_fingerprint', 'layout_spec', 'execution_protocol',
                    'state', 'optimizer', 'scheduler', 'rng_state'):
            payload = resume_metadata(cfg)
            payload.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_resume_payload(payload, cfg)
        payload = resume_metadata(cfg)
        payload['allocation_grid'] = [(64, 0)]
        with self.assertRaises(ValueError):
            entry.validate_resume_payload(payload, cfg)

    def test_schedule_lr_and_selection_integrity(self):
        cfg = fdma_config(SEEDS[0])
        for change in ('last_epoch', 'total_steps', 'max_lr', 'missing_gamma', 'metric'):
            payload = resume_metadata(cfg)
            if change == 'last_epoch':
                payload['scheduler']['last_epoch'] = 0
            elif change == 'total_steps':
                payload['scheduler']['total_steps'] = 500
            elif change == 'max_lr':
                payload['optimizer']['param_groups'][0]['max_lr'] = 2e-5
            elif change == 'missing_gamma':
                payload['val_table'].pop('layout0_gamma0')
            else:
                payload['best_metric'] = 1.
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.validate_resume_payload(payload, cfg)
        latest, best = resume_metadata(cfg, 20000), resume_metadata(cfg, 19500)
        entry.validate_best_metadata(best, latest, cfg)
        best['step'] = 0
        with self.assertRaises(ValueError):
            entry.validate_best_metadata(best, latest, cfg)
        best = resume_metadata(cfg, 19500)
        best['val_table']['layout0_gamma0'] = 0
        with self.assertRaises(ValueError):
            entry.validate_best_metadata(best, latest, cfg)

    def test_output_protection_and_exact_resume_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'F'
            cfg = fdma_config(SEEDS[0], root)
            entry.check_run_state(cfg, resume=False, allowed_root=root)
            self.assertFalse(root.exists())
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                entry.check_run_state(cfg, resume=False, allowed_root=root)
            with self.assertRaises(FileNotFoundError):
                entry.check_run_state(cfg, resume=True, allowed_root=root)
            Path(cfg.LOG_DIR).mkdir(parents=True)
            for path in (cfg.SAVE_PATH, cfg.BEST_PATH, cfg.METRICS_PATH):
                Path(path).touch()  # Existence sentinels; never opened as weights.
            Path(cfg.CONFIG_PATH).write_text(json.dumps(dict(cfg.to_dict(),
                config_fingerprint=entry.fingerprint(cfg))), encoding='utf-8')
            Path(cfg.STATUS_PATH).write_text(json.dumps(dict(
                state='paused', experiment=cfg.EXP_NAME, step=500)), encoding='utf-8')
            entry.check_run_state(cfg, resume=True, allowed_root=root)
            Path(cfg.CONFIG_PATH).write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                entry.check_run_state(cfg, resume=True, allowed_root=root)

    def test_final_step_status_recovery_does_not_run_updates(self):
        cfg = fdma_config(SEEDS[0])
        latest, best = resume_metadata(cfg, 20000), resume_metadata(cfg, 19500)
        previous = dict(state='running', experiment=cfg.EXP_NAME)
        status = entry.completed_resume_status(cfg, latest, best, previous)
        self.assertEqual((status['state'], status['step']), ('completed', 20000))
        self.assertEqual(status['best_metric_name'], 'mean_all')
        self.assertEqual(status['additional_optimizer_updates'], 0)
        for changed in (dict(previous, state='completed'), dict(previous, experiment='old_A'),
                        dict(previous, state='paused', step=500)):
            with self.assertRaises(ValueError):
                entry.completed_resume_status(cfg, latest, best, changed)
        with self.assertRaises(ValueError):
            entry.completed_resume_status(cfg, resume_metadata(cfg, 19500), best, previous)
        tree = ast.parse((ROOT / 'train_grouped_codec_fdma.py').read_text(encoding='utf-8'))
        branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == 'finished')
        calls = [ast.unparse(n.func) for n in ast.walk(branch) if isinstance(n, ast.Call)]
        for forbidden in ('train_proposal', 'make_model_factory', 'optimizer_scheduler_factory'):
            self.assertNotIn(forbidden, calls)
        self.assertTrue(any(isinstance(n, ast.Return) for n in branch.body))


class StaticIntegrationTests(unittest.TestCase):
    def test_numerical_forward_inherited_and_actual_legacy_init_streams(self):
        tree = ast.parse((ROOT / 'models/grouped_codec_fdma.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertEqual([ast.unparse(b) for b in cls.bases], ['GroupedCodecPrecoder'])
        self.assertEqual([n.name for n in cls.body if isinstance(n, ast.FunctionDef)],
                         ['__init__', 'architecture_metadata'])
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
        functions = [ast.unparse(n.func) for n in calls]
        self.assertIn('LegacyUE', functions)
        self.assertIn('GroupedJointSwinDecoder', functions)
        addresses = [tuple(ast.unparse(a) for a in n.args) for n in calls
                     if ast.unparse(n.func) == 'protocol.RandomAddress']
        self.assertEqual(addresses, [('cfg.SEED', "'init'", '0', "'ue_legacy'"),
                                    ('cfg.SEED', "'init'", '0', "'bs'")])
        self.assertNotIn('OrderedWindowUE', functions)
        self.assertNotIn('GroupedCodecPrecoder.__init__', functions)

    def test_optional_trainer_contract_accepts_true_f_not_old_allocation(self):
        tree = ast.parse((ROOT / 'main_train.py').read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_validate_execution_protocol')
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'binding', 'exec'), scope)
        cfg = fdma_config(SEEDS[0])
        scope[node.name](cfg, entry.FDMAExecutionProtocol(cfg))
        with self.assertRaises(ValueError):
            scope[node.name](cfg, None)
        with self.assertRaises(ValueError):
            scope[node.name](grouped_codec_config('ordered_fdma', SEEDS[0]), entry.FDMAExecutionProtocol(cfg))
        source = (ROOT / 'train_grouped_codec_fdma.py').read_text(encoding='utf-8')
        for forbidden in ('loss.backward(', 'optimizer.step(', 'scheduler.step(', 'strict=False',
                          'initialize_ue_from_full', 'migrate_original_state_dict'):
            self.assertNotIn(forbidden, source)
        self.assertNotIn('--checkpoint', entry.build_parser()._option_string_actions)
        self.assertIs(entry.validate_model_identity, old_entry.validate_model_identity)
        self.assertIs(entry.validate_model_state, old_entry.validate_model_state)
        self.assertIs(entry.make_model_factory, old_entry.make_model_factory)

    def test_original_grouped_sources_and_default_trainer_unchanged(self):
        for name in ('configs/grouped_codec_train_config.py', 'models/grouped_codec_precoder.py',
                     'train_grouped_codec.py', 'utils/grouped_feedback_protocol.py', 'main_train.py'):
            before = subprocess.check_output(['git', 'show', f'd985ce4:{name}'], cwd=ROOT)
            after = (ROOT / name).read_bytes()
            self.assertEqual(before.replace(b'\r\n', b'\n'), after.replace(b'\r\n', b'\n'), name)

    def test_two_dry_runs_are_readonly_stdlib_no_checkpoint(self):
        calls = [['--seed', str(seed), '--dry-run'] for seed in SEEDS]
        assert_lightweight_cli(self, 'train_grouped_codec_fdma', calls,
            "assert len(plans)==2; assert all(p['config']['arm']=='legacy_fdma' and "
            "p['scheduler_total_steps']==20000 and p['historical_checkpoint'] is None for p in plans)")


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,
    'Installed PyTorch unavailable; no dependencies installed or fabricated.')
class SmallTensorTests(unittest.TestCase):
    def test_fdma_active_as_prefix_normalization_without_any_model(self):
        import torch
        layout = fdma_config(SEEDS[0]).layout_spec
        raw = torch.ones((1, 4, 256), dtype=torch.complex64)
        physical, canonical, energy = protocol.normalize_and_map(raw, layout)
        self.assertEqual(tuple(physical.shape), (1, 4, 256))
        self.assertTrue(torch.equal(energy, torch.full((1, 4), 64.)))
        for k in range(4):
            self.assertEqual(torch.nonzero(physical[0, k]).flatten().tolist(), list(layout.supports[k]))
            self.assertEqual(torch.nonzero(canonical[0, k]).flatten().tolist(), list(layout.active_slots))

    def test_private_noise_and_single_optimizer_group_without_model_or_update(self):
        import torch
        cfg = fdma_config(SEEDS[0])
        before = torch.get_rng_state().clone()
        a = protocol.standard_noise(cfg.SEED, 'train', 501, 1)
        b = old_entry.protocol.standard_noise(cfg.SEED, 'train', 501, 1)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        owner = SimpleNamespace(parameters=lambda: iter([torch.tensor([1.], requires_grad=True)]))
        optimizer, scheduler = entry.optimizer_scheduler_factory(cfg, owner)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['max_lr'], 1e-4)
        self.assertEqual(scheduler.total_steps, 20000)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(optimizer.state, {})


if __name__ == '__main__':
    unittest.main()
