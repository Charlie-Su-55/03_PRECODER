"""Lightweight protocol and tiny CPU state tests; no checkpoint/GPU evaluation."""
import copy
import contextlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_low_feedback_snr as diagnostic
from train_proposal_specialist import specialist_config
from train_proposal_snr_robust import robust_config
from feedback_link_diagnostics import FeedbackLinkProbe
from types import SimpleNamespace


class DiagnosticProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = diagnostic.diagnostic_plan(diagnostic.build_parser().parse_args([]))

    def test_exactly_eight_fixed_configs_and_distinct_paths(self):
        self.assertEqual(len(self.plan), 8)
        self.assertNotIn('universal', [e['method_key'] for e in self.plan])
        self.assertEqual([e['allocations'] for e in self.plan], [
            ((64, 0),), ((48, 64),), ((32, 128),), ((16, 192),),
            ((0, 256),), ((0, 256),), ((64, 0),), ((64, 0),),
        ])
        self.assertEqual(len({e['path'] for e in self.plan}), 8)
        self.assertTrue(all(str(e['path']).endswith('best.pth') for e in self.plan))

    def test_not_a_formal_test_seed_and_no_torch_import(self):
        self.assertNotEqual(diagnostic.SEED, 20260921)
        self.assertNotEqual(diagnostic.SEED, 500042)
        self.assertGreater(abs(diagnostic.SEED-20260921), diagnostic.SAMPLE_COUNT)
        self.assertEqual(diagnostic.BATCH_SIZE * 32, diagnostic.SAMPLE_COUNT)
        subprocess.run([sys.executable, '-c',
                        'import sys; import diagnose_low_feedback_snr; assert "torch" not in sys.modules'],
                       cwd=Path(__file__).resolve().parents[1], check=True)

    def fixture_samples(self):
        return [
            dict(method_key=e['method_key'], sample_index=i, feedback_snr_db=snr,
                 sum_rate=(10 if e['method_key']=='swin' else 5) + 2*i + snr/10)
            for e in self.plan for i in range(2) for snr in diagnostic.FB_SNRS
        ]

    def test_channel_paired_low_snr_se_and_negative_gains(self):
        rows = diagnostic.low_snr_rows(self.fixture_samples(), self.plan, 2)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertAlmostEqual(row['low_snr_rate_se'], 1.0)
            self.assertAlmostEqual(row['delta_vs_swin_se'], 0.0)
        specialist = rows[0]
        self.assertAlmostEqual(specialist['low_snr_rate_mean'], 6.5)
        self.assertAlmostEqual(specialist['delta_vs_swin_mean'], -5)
        self.assertEqual(specialist['dataset_role'], diagnostic.ROLE)

    def test_missing_sample_rejected(self):
        samples = self.fixture_samples()
        samples.pop(0)
        with self.assertRaises(KeyError):
            diagnostic.low_snr_rows(samples, self.plan, 2)

    def test_duplicate_sample_rejected(self):
        samples = self.fixture_samples()
        samples.append(samples[0])
        with self.assertRaises(ValueError):
            diagnostic.low_snr_rows(samples, self.plan, 2)

    def test_nonfinite_rates_rejected(self):
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                diagnostic.rate_statistics([0, value])

    def test_actual_training_recipe_checks(self):
        for e in self.plan:
            if e['method_key'] == 'robust_specialist':
                config = robust_config().to_dict()
            elif e['model_key'] == 'proposed':
                config = specialist_config(e['allocations'][0], 'unused_test_root').to_dict()
            else:
                config = dict(OBJECTIVE='task', SEED=42)
            self.assertTrue(diagnostic.validate_training_metadata(e, config))
            bad = copy.deepcopy(config)
            if e['model_key'] == 'proposed':
                bad['experiment']['train_mode'] = 'adaptive'
            else:
                bad['OBJECTIVE'] = 'nmse'
            with self.assertRaises(RuntimeError):
                diagnostic.validate_training_metadata(e, bad)

    def test_robust_auxiliary_loss_rejected(self):
        config = robust_config().to_dict()
        config['training_recipe']['ph2_w_mse'] = 1.0
        with self.assertRaises(RuntimeError):
            diagnostic.validate_training_metadata(self.plan[5], config)

    def test_transparency_comparison_detects_changed_rates(self):
        row = dict(method_key='x', allocation_df=0, allocation_da=256,
                   scenario_id='K4_Nsc256_Dfb256', downlink_snr_db=25,
                   feedback_snr_db=0, sample_index=0, sum_rate=1.0,
                   nmse_db=-2.0, precoder_power=256.0)
        self.assertTrue(diagnostic.compare_probe_results([row], [dict(row)])['passed'])
        report = diagnostic.compare_probe_results([row], [dict(row, sum_rate=2.0)])
        self.assertFalse(report['passed'])
        self.assertEqual(report['fields']['sum_rate']['first_failure']['key']['sample_index'], 0)

    def test_physical_coverage_requires_only_active_branches(self):
        methods = [SimpleNamespace(method_key=e['method_key'], allocations=e['allocations']) for e in self.plan]
        probe = FeedbackLinkProbe(methods, {})
        for e in self.plan:
            for branch, width in zip(('FDMA', 'AS'), e['allocations'][0]):
                if width:
                    for snr in diagnostic.FB_SNRS:
                        probe.rows.append(dict(method_key=e['method_key'], branch=branch,
                                               feedback_snr_db=snr, batch_index=0))
        self.assertEqual(len(probe.rows), 66)
        probe.verify_coverage(1, diagnostic.FB_SNRS)
        probe.rows.pop()
        with self.assertRaises(RuntimeError):
            probe.verify_coverage(1, diagnostic.FB_SNRS)




def sample_row(index=0, value=1.0):
    return dict(scenario_id='K4_Nsc256_Dfb256', method_key='toy', allocation_df=0, allocation_da=256,
                feedback_snr_db=0, downlink_snr_db=25, sample_index=index, sum_rate=value,
                nmse_db=-2.0, precoder_power=256.0)


class PairingTests(unittest.TestCase):
    def test_row_order_is_irrelevant_and_values_include_abc(self):
        a, b = [sample_row(0), sample_row(1)], [sample_row(1), sample_row(0)]
        self.assertTrue(diagnostic.compare_probe_results(a, b, labels=('A', 'B'))['passed'])
        c = [sample_row(0, 3), sample_row(1)]
        report = diagnostic.compare_probe_results(a, c, all_runs=dict(A=a, B=b, C=c))
        first = report['fields']['sum_rate']['first_failure']
        self.assertEqual(first['values'], dict(A=1.0, B=1.0, C=3))
        self.assertEqual(first['absolute_difference'], 2)

    def test_missing_duplicate_nonfinite_and_dl_key(self):
        for right in ([], [sample_row(), sample_row()], [sample_row(value=math.nan)],
                      [dict(sample_row(), downlink_snr_db=20)]):
            with self.subTest(right=right):
                self.assertFalse(diagnostic.compare_probe_results([sample_row()], right)['passed'])

    def test_expected_grid_detects_missing_in_both_runs(self):
        expected = {tuple(sample_row(i)[k] for k in diagnostic.SAMPLE_KEY_FIELDS) for i in (0, 1)}
        report = diagnostic.compare_probe_results([sample_row()], [sample_row()], expected_keys=expected)
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['missing']['A']), 1)

    def test_tolerance_not_relaxed(self):
        report = diagnostic.compare_probe_results([sample_row()], [sample_row(value=1.000002)])
        self.assertFalse(report['passed'])
        self.assertEqual((report['rel_tol'], report['abs_tol']), (1e-6, 1e-6))


class CPUStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.torch, cls.np = torch, np

    def make_model(self):
        torch = self.torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))
                self.register_buffer('persistent', torch.tensor([2.0]))
                self.register_buffer('temporary', torch.tensor([3.0]), persistent=False)
                self.cached_value = {'x': torch.tensor([4.0])}
            def add_noise(self, value, snr):
                return value
            def transmit(self, *args):
                return args
        model = Tiny().eval()
        return SimpleNamespace(method_key='toy', model_key='proposed', model=model,
                               allocations=((0, 256),), config=SimpleNamespace(GAMMA_GRID=((0, 256),)))

    def test_nonpersistent_buffer_parameter_mode_and_input_are_reported_and_restored(self):
        from feedback_link_diagnostics import ModelMemorySnapshot
        method = self.make_model()
        channels = (self.torch.ones(8, 1), self.torch.zeros(8, 1))
        snapshot = ModelMemorySnapshot([method], channels)
        self.assertNotIn('temporary', method.model.state_dict())
        with self.torch.no_grad():
            method.model.temporary.add_(1)
            method.model.weight.add_(2)
            method.model.cached_value['x'].add_(1)
            channels[0].add_(1)
        method.model.train()
        report = snapshot.report()
        for field in ('parameters', 'buffers', 'modes', 'extra_caches', 'inputs'):
            self.assertTrue(report[field], field)
        self.assertTrue(snapshot.restore()['passed'])
        self.assertFalse(report['passed'])  # Restore cannot erase previously recorded evidence.

    def test_rng_types_and_independent_generators(self):
        from feedback_link_diagnostics import RNGSnapshot, rng_comparison
        method = self.make_model()
        method.model.independent_torch = self.torch.Generator().manual_seed(12)
        ev = SimpleNamespace(independent_numpy=self.np.random.default_rng(13),
                             independent_python=random.Random(14),
                             independent_legacy=self.np.random.RandomState(15))
        initial = RNGSnapshot([method], ev)
        with patch.object(self.torch.cuda, 'is_available', return_value=False):
            # CPU draws only; no real model or GPU work.
            random.random()
            self.np.random.random()
            self.torch.rand(1)
            self.torch.rand(1, generator=method.model.independent_torch)
            ev.independent_numpy.random()
            ev.independent_python.random()
            ev.independent_legacy.random()
        changed = RNGSnapshot([method], ev)
        report = rng_comparison(initial.fingerprints(), changed.fingerprints())
        for field in ('python', 'numpy', 'torch_cpu', 'independent'):
            self.assertFalse(report['equal'][field], field)
        self.assertTrue(initial.restore())

    def test_opaque_cache_does_not_silently_pass_state_verification(self):
        from feedback_link_diagnostics import ModelMemorySnapshot
        method = self.make_model()
        method.model.cached_opaque = object()
        snapshot = ModelMemorySnapshot([method], (self.torch.ones(8, 1), self.torch.zeros(8, 1)))
        report = snapshot.report()
        self.assertFalse(report['passed'])
        self.assertFalse(report['coverage_complete'])
        self.assertTrue(any('cached_opaque' in p for p in report['unverified_cache_paths']))

    def test_all_cuda_rng_states_are_compared_and_restored_without_gpu(self):
        from feedback_link_diagnostics import RNGSnapshot, rng_comparison
        torch = self.torch
        states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
        def setter(saved):
            states[:] = [v.clone() for v in saved]
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_rng_state_all', side_effect=lambda: [v.clone() for v in states]), \
                patch.object(torch.cuda, 'set_rng_state_all', side_effect=setter):
            initial = RNGSnapshot([], SimpleNamespace())
            states[1][0] = 8
            changed = RNGSnapshot([], SimpleNamespace())
            self.assertFalse(rng_comparison(initial.fingerprints(), changed.fingerprints())['equal']['cuda'])
            self.assertEqual(len(initial.cuda), 2)
            self.assertTrue(initial.restore())

    def test_probe_exception_cleanup_and_partial_enter_cleanup(self):
        from feedback_link_diagnostics import FeedbackLinkProbe, ProbeInstallationSnapshot
        method = self.make_model()
        original = ProbeInstallationSnapshot([method])
        probe = FeedbackLinkProbe([method], {})
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with probe:
                self.assertTrue(method.model._forward_pre_hooks)
                raise RuntimeError('test exception')
        self.assertTrue(probe.cleanup_report['passed'])
        self.assertTrue(original.report()['passed'])
        second = self.make_model()
        second.method_key = 'second'
        with patch.object(second.model, 'register_forward_pre_hook', side_effect=RuntimeError('enter failed')):
            probe = FeedbackLinkProbe([method, second], {})
            with self.assertRaisesRegex(RuntimeError, 'enter failed'):
                probe.__enter__()
        self.assertTrue(probe.cleanup_report['passed'])
        self.assertTrue(original.report()['passed'])

    def run_toy_replays(self, mode):
        """Exercise the real orchestration with scalar fake-evaluator output."""
        torch, np = self.torch, self.np
        method = self.make_model()
        channels = (torch.ones(8, 1), torch.zeros(8, 1))
        control = dict(probed=False, call=0, starts=[])
        class TinyProbe:
            def __init__(self, *_):
                self.rows = []
                self.context = None
                self.cleanup_report = dict(passed=True)
            def __enter__(self):
                control['probed'] = True
                return self
            def __exit__(self, *_):
                control['probed'] = False
            def verify_coverage(self, *_):
                pass
        def seed(value):
            random.seed(value)
            np.random.seed(value)
            torch.manual_seed(value)
        def evaluate(scenario, methods, snrs, args, device):
            control['call'] += 1
            generator = ev.build_channel_generator(scenario, device)
            control['starts'].append(generator.batch)
            generator.generate_batch_data(8)
            value = random.random() + np.random.random() + float(torch.rand(1))
            if mode == 'plain_failure' and control['call'] == 2:
                value += 0.01
            if mode == 'probe_failure' and control['probed']:
                value += 0.01
                random.random()  # Must be reported even though metrics also fail.
            if mode == 'buffer_effect' and control['call'] == 1:
                methods[0].model.temporary.add_(1)
            if mode == 'probe_exception' and control['probed']:
                raise RuntimeError('synthetic operator error')
            rows = [dict(sample_row(i, value), feedback_snr_db=float(snr))
                    for snr, _ in snrs for i in range(8)]
            return [], rows
        ev = SimpleNamespace(set_seed=seed, evaluate_scenario=evaluate,
                             build_channel_generator=lambda *_: None)
        scenario = SimpleNamespace(scenario_id='K4_Nsc256_Dfb256')
        args = SimpleNamespace(num_batches=1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch('feedback_link_diagnostics.FeedbackLinkProbe', TinyProbe), \
                    contextlib.redirect_stdout(io.StringIO()):
                if mode == 'pass':
                    diagnostic.run_preflight(ev, scenario, [method], [(float(x), 25.) for x in diagnostic.FB_SNRS],
                                             args, torch.device('cpu'), channels, output)
                else:
                    with self.assertRaises(RuntimeError):
                        diagnostic.run_preflight(ev, scenario, [method], [(float(x), 25.) for x in diagnostic.FB_SNRS],
                                                 args, torch.device('cpu'), channels, output)
            report = json.loads((output / 'probe_transparency_report.json').read_text())
        self.assertEqual(control['call'], 3)
        self.assertEqual(control['starts'], [0, 0, 0])
        return report

    def test_plain_control_and_probe_pass(self):
        report = self.run_toy_replays('pass')
        self.assertTrue(report['plain_vs_plain_passed'])
        self.assertTrue(report['plain_vs_probed_passed'])

    def test_plain_failure_is_not_attributed_to_probe(self):
        report = self.run_toy_replays('plain_failure')
        self.assertEqual(report['status'], 'baseline replay not reproducible')

    def test_metric_failure_still_saves_rng_state_cleanup_and_abc(self):
        report = self.run_toy_replays('probe_failure')
        self.assertTrue(report['plain_vs_plain_passed'])
        self.assertFalse(report['plain_vs_probed_passed'])
        self.assertFalse(report['rng_comparisons']['plain_vs_probed']['equal']['python'])
        self.assertIn('state_changes', report['runs']['C'])
        self.assertIn('probe_cleanup', report['runs']['C'])
        first = report['comparisons']['plain_vs_probed']['fields']['sum_rate']['first_failure']
        self.assertEqual(set(first['values']), {'A', 'B', 'C'})
        self.assertEqual(set(first['key']), set(diagnostic.SAMPLE_KEY_FIELDS))

    def test_buffer_side_effect_is_reported_even_after_restore(self):
        report = self.run_toy_replays('buffer_effect')
        self.assertFalse(report['passed'])
        self.assertTrue(report['runs']['A']['state_changes']['buffers'])
        self.assertTrue(report['runs']['B']['initial_restore']['passed'])

    def test_runtime_exception_also_saves_complete_report(self):
        report = self.run_toy_replays('probe_exception')
        self.assertTrue(report['plain_vs_plain_passed'])
        self.assertFalse(report['plain_vs_probed_passed'])
        self.assertEqual(report['runs']['C']['error']['message'], 'synthetic operator error')
        for field in ('rng_end', 'state_changes', 'batch_inputs', 'probe_cleanup'):
            self.assertIn(field, report['runs']['C'])


if __name__ == '__main__':
    unittest.main()
