"""Independent power-stage config, provenance and two-clock resume contracts."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from configs.grouped_hybrid_power_train_config import (
    ARMS, SEEDS, SOURCE_BEST_SHA256, SOURCE_PRIOR_OFFSET, power_config,
    source_arm, source_identity, protocol, validate_config)
from configs.grouped_hybrid_anchor_train_config import anchor_config
from configs.train_config import TRAINING_RECIPES
from test_train_grouped_codec import resume_metadata as schema_metadata
from cli_contracts import assert_lightweight_cli
import train_grouped_hybrid_power as entry


def resume_metadata(cfg, step=500):
    """Schema sentinels only, not a numerical checkpoint or server evidence."""
    value = schema_metadata(cfg, step)
    value['state'].update({'power_prior.pi_nodes': 'schema only',
        'power_prior.update_count': SOURCE_PRIOR_OFFSET + step,
        'power_prior.node_update_mass': 'schema only'})
    if step == 0: value['optimizer']['state'] = {}
    return value


class PowerConfigTests(unittest.TestCase):
    def test_eight_configs_equal_recipe_shared_new_addresses_unchanged_old_registry(self):
        before = deepcopy(TRAINING_RECIPES)
        for seed in SEEDS:
            cfgs = [power_config(arm, seed) for arm in ARMS]
            for cfg in cfgs:
                validate_config(cfg)
                self.assertEqual((cfg.TOTAL_STEPS, cfg.BATCH_SIZE, cfg.VAL_BATCH, cfg.VAL_INTERVAL), (20000, 48, 256, 500))
                self.assertEqual((cfg.LR, cfg.WEIGHT_DECAY, cfg.WARMUP_PCT, cfg.GRAD_CLIP), (1e-4, 1e-5, .1, .5))
                self.assertEqual([cfg.PH1_W_DIR, cfg.PH1_W_MSE, cfg.PH2_W_DIR, cfg.PH2_W_MSE], [0, 0, 0, 0])
                self.assertEqual(cfg.VAL_SNR_LIST, [0, 5, 10, 15, 20, 25])
                self.assertEqual(cfg.DOWNLINK_SNR_DB, 25.)
                self.assertTrue(cfg.to_dict()['screening_protocol']['source_step_zero_competes'])
                source = source_identity(cfg.arm, seed)
                self.assertEqual(source['best_sha256'], SOURCE_BEST_SHA256[(source_arm(cfg.arm), seed)])
                self.assertEqual(source['prior_update_count'], 20000)
            self.assertTrue(all(c.channel_data_protocol == cfgs[0].channel_data_protocol for c in cfgs))
            self.assertTrue(all(c.to_dict()['execution_protocol'] == cfgs[0].to_dict()['execution_protocol'] for c in cfgs))
            old = anchor_config('fdma_control', seed)
            self.assertNotEqual(cfgs[0].channel_data_protocol, old.channel_data_protocol)
            self.assertEqual(len({entry.fingerprint(c) for c in cfgs}), 4)
        self.assertEqual(before, TRAINING_RECIPES)

    def test_real_source_identities_share_h32_and_reject_wrong_arm_or_unsaved_mutation(self):
        for seed in SEEDS:
            self.assertEqual(source_identity('h32_control', seed), source_identity('h32_gamma_power', seed))
            self.assertEqual(source_identity('h32_static_power', seed), source_identity('h32_control', seed))
            self.assertIn('grouped_hybrid_anchor_v1', source_identity('fdma_control', seed)['best_path'])
        cfg = power_config('h32_control', SEEDS[0])
        for mutate in ('recipe', 'channel', 'extra'):
            other = deepcopy(cfg)
            if mutate == 'recipe': other.recipe = replace(other.recipe, lr=2e-5)
            elif mutate == 'channel': other.channel_data_protocol['base_seed'] += 1
            else: other.unserialized_option = True
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): validate_config(other)
        with self.assertRaises(ValueError): power_config('hybrid_f32_a64', SEEDS[0])

    def test_cli_dry_run_is_stdlib_only_no_weight_reads_or_outputs(self):
        assert_lightweight_cli(self, 'train_grouped_hybrid_power', [
            ['--arm', arm, '--seed', '20261030', '--dry-run'] for arm in ARMS],
            "assert all(p['cumulative_prior_counts']==[20000,20500,40000] for p in plans)")

    def test_output_namespaces_do_not_overwrite_source_or_existing_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = power_config('h32_static_power', SEEDS[0], root)
            entry.check_run_state(cfg, resume=False, allowed_root=root)
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError): entry.check_run_state(cfg, resume=False, allowed_root=root)
            with self.assertRaises(FileNotFoundError): entry.check_run_state(cfg, resume=True, allowed_root=root)
            self.assertFalse(Path(cfg.LOG_DIR).exists())


class PowerResumeTests(unittest.TestCase):
    def test_stepzero_500_and_completed_have_distinct_optimizer_and_prior_clocks(self):
        for arm in ARMS:
            cfg = power_config(arm, SEEDS[0])
            for step, prior in ((0, 20000), (500, 20500), (20000, 40000)):
                payload = resume_metadata(cfg, step)
                entry.validate_resume_payload(payload, cfg, allow_completed=True)
                self.assertEqual(payload['scheduler']['last_epoch'], step)
                self.assertEqual(payload['state']['power_prior.update_count'], prior)
            with self.assertRaises(ValueError): entry.validate_resume_payload(resume_metadata(cfg, 20000), cfg)

    def test_reject_prior_reset_offset_doublecount_old_budget_and_partial_resume(self):
        cfg = power_config('h32_gamma_power', SEEDS[0])
        for bad in ('reset', 'double', 'old20k', 'scheduler_restart', 'short_schedule', 'missing_rng', 'missing_state', 'wrong_fingerprint', 'cross_arm'):
            payload = resume_metadata(cfg)
            target = cfg
            if bad == 'reset': payload['state']['power_prior.update_count'] = 500
            elif bad == 'double': payload['state']['power_prior.update_count'] = 40500
            elif bad == 'old20k': payload['step'] = 20500
            elif bad == 'scheduler_restart': payload['scheduler']['last_epoch'] = 0
            elif bad == 'short_schedule': payload['scheduler']['total_steps'] = 500
            elif bad == 'missing_rng': payload['rng_state'].pop('cuda')
            elif bad == 'missing_state': payload['state'].pop('power_prior.node_update_mass')
            elif bad == 'wrong_fingerprint': payload['config_fingerprint'] = '0' * 64
            else: target = power_config('h32_static_power', SEEDS[0])
            with self.subTest(bad=bad), self.assertRaises(ValueError): entry.validate_resume_payload(payload, target)

    def test_stepzero_can_remain_best_and_bad_selection_is_rejected(self):
        cfg = power_config('h32_control', SEEDS[0])
        latest, best = resume_metadata(cfg, 20000), resume_metadata(cfg, 0)
        entry.validate_best_metadata(best, latest, cfg)
        latest['best_metric'] = 13.
        with self.assertRaises(ValueError): entry.validate_best_metadata(best, latest, cfg)

    def test_actual_resume_source_comparison_accepts_json_roundtrip_not_changed_identity(self):
        cfg = power_config('h32_gamma_power', SEEDS[0])
        source = source_identity(cfg.arm, cfg.SEED)
        current = dict(source=source, source_sha256={'model.py': 'a' * 64})
        initial = json.loads(json.dumps(dict(current, config=cfg.to_dict(), config_fingerprint=entry.fingerprint(cfg))))
        entry.validate_initialization_provenance(cfg, initial, current)
        for field in ('best_sha256', 'training_seed', 'best_path', 'configuration', 'code'):
            changed = deepcopy(current)
            if field == 'configuration': changed['source']['complete_source_configuration']['recipe']['stage1_snr'] = [1., 25.]
            elif field == 'code': changed['source_sha256']['model.py'] = 'b' * 64
            else: changed['source'][field] = 'changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                entry.validate_initialization_provenance(cfg, initial, changed)

    def test_source_loader_rejects_unpinned_old_F_before_payload_access(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            entry.verify_source_payloads('h32_control', SEEDS[0], {}, {}, {}, {'best_sha256': '0' * 64})


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Real Torch absent; tiny prior lifecycle not run')
class PowerLifecycleTests(unittest.TestCase):
    def test_actual_execution_and_lifecycle_500_to501_keep_local_random_address(self):
        import torch
        cfg = power_config('h32_gamma_power', SEEDS[0])
        prior = protocol.make_power_prior(cfg.layout_spec)
        prior.update_count.fill_(20500)
        prior.node_update_mass.fill_(20500 / 6)
        prior.validate_state(20500)
        captured = {}
        class TinyModel:
            power_prior = prior
            def training_with_statistics(self, H_dl, H_ul, gamma, *, global_step, standard_noise):
                captured.update(step=global_step, gamma=gamma, noise=standard_noise)
                raw = torch.ones(2, 4, 256, dtype=torch.complex64)
                _, canonical, _ = protocol.normalize_and_map(raw, cfg.layout_spec)
                return None, None, prior.pending(canonical, gamma, global_step)
            def post_optimizer_commit(self, stats, global_step):
                if stats['step'] != global_step: raise ValueError('counter mismatch')
                prior.commit(stats)
        model = TinyModel()
        execution, lifecycle = entry.PowerExecutionProtocol(cfg), entry.PowerLifecycle(cfg)
        channels = torch.zeros(2, 4, 32, 256, dtype=torch.complex64)
        output, gamma = execution.training(model, channels, channels, 501, 'cpu')
        self.assertEqual(captured['step'], 20501)
        self.assertEqual(gamma, protocol.sample_gamma(SEEDS[0], 501))
        self.assertNotEqual(gamma, protocol.sample_gamma(SEEDS[0], 20501))
        self.assertTrue(torch.equal(captured['noise'], protocol.standard_noise(SEEDS[0], 'train', 501, 2)))
        self.assertEqual(prior.update_count.item(), 20500)  # forward never commits
        lifecycle.after_optimizer_step(model, output, 501)
        lifecycle.validate_model_step(model, 501)
        self.assertEqual(prior.update_count.item(), 20501)
        with self.assertRaises(ValueError): lifecycle.after_optimizer_step(model, output, 501)
        with self.assertRaises(ValueError): execution.training(model, channels, channels, 501, 'cpu')

    def test_lifecycle_counts_validate_inherited_stepzero_and_do_not_reset_prior(self):
        import torch
        cfg = power_config('fdma_control', SEEDS[0])
        prior = protocol.make_power_prior(cfg.layout_spec)
        prior.update_count.fill_(20000)
        prior.node_update_mass[0] = 20000
        model = SimpleNamespace(power_prior=prior)
        lifecycle = entry.PowerLifecycle(cfg)
        before = {name: tensor.clone() for name, tensor in prior.state_dict().items()}
        lifecycle.validate_model_step(model, 0)
        for name, tensor in prior.state_dict().items(): self.assertTrue(torch.equal(tensor, before[name]))
        with self.assertRaises(ValueError): lifecycle.validate_model_step(model, 500)


class PowerEvaluationTests(unittest.TestCase):
    """Scalar synthetic observations; never simulated checkpoint/model evidence."""
    @staticmethod
    def samples(count=16, difference=None):
        import evaluate_grouped_hybrid_power as ev
        result=[]
        for method in ev.METHODS:
            for gamma in ev.GAMMAS:
                for index in range(count):
                    row=ev.sample_identity(method,gamma,index,SEEDS[0])
                    row.update({name:(None if name in ev.OPTIONAL_METRICS and not ev._available(method,name) else 1.)
                                for name in ev.METRICS})
                    df,da=ev.LAYOUTS[method];exclusive=64. if da==0 else 32.;shared=64.-exclusive
                    row.update({f'energy_ue{k}':64. for k in range(4)})
                    row.update(exclusive_energy_mean_per_ue=exclusive,shared_energy_mean_per_ue=shared,
                        exclusive_energy_per_active_symbol=exclusive/df,
                        shared_energy_per_active_symbol=None if not da else shared/da,
                        precoder_power=256.,precoder_power_max_error=0.,g_log_energy_ratio=0.,
                        empirical_noise_power=.25*10**(-gamma/10),
                        sum_rate=10.+gamma/10.+index/100.+(difference(method,gamma,index) if difference else 0.))
                    result.append(row)
        return result

    def test_six_methods_six_nodes_all_eight_signed_pairs_keep_negatives(self):
        import evaluate_grouped_hybrid_power as ev
        samples=self.samples(difference=lambda m,g,i:-.75 if m=='h32_gamma_power' else 0.)
        summary,_=ev.summarize_samples(samples,16,SEEDS[0])
        pairs,raw=ev.paired_rows(summary,samples,16,SEEDS[0])
        self.assertEqual((len(summary),len(pairs),len(raw)),(36,64,1024))
        self.assertEqual({r['contrast_id'] for r in pairs},set(ev.CONTRASTS))
        self.assertEqual({r['method_key'] for r in summary},set(ev.METHODS))
        self.assertTrue(all(math.isclose(r['delta_mean'],-.75,abs_tol=1e-12)
                            for r in pairs if r['contrast_id']=='gamma_minus_H32'))
        self.assertTrue(all(r['delta']<0 for r in raw if r['contrast_id']=='gamma_minus_H32'))

    def test_low_gamma_uses_within_channel_covariance_not_average_point_standard_errors(self):
        import evaluate_grouped_hybrid_power as ev
        def delta(method,gamma,index):
            if method!='h32_gamma_power':return 0.
            return .4+({0:1.,5:-1.}.get(gamma,0.))*(index-7.5)
        samples=self.samples(difference=delta)
        summary,_=ev.summarize_samples(samples,16,SEEDS[0]);pairs,_=ev.paired_rows(summary,samples,16,SEEDS[0])
        selected=[r for r in pairs if r['contrast_id']=='gamma_minus_H32']
        low=next(r for r in selected if r['estimand']=='low_gamma_mean')
        nodes=[r for r in selected if r['estimand']=='per_gamma' and r['gamma_db'] in (0,5,10)]
        self.assertAlmostEqual(low['delta_mean'],.4,places=12)
        self.assertLess(low['delta_se'],1e-12)
        self.assertGreater(sum(r['delta_se'] for r in nodes)/3,.1)
        self.assertLess(low['batch_delta_se'],1e-12)

    def test_batch_interval_uses_eight_channel_clusters_and_preflight_ci_is_not_zero(self):
        import evaluate_grouped_hybrid_power as ev
        samples=self.samples(difference=lambda m,g,i:float(i//8) if m=='h32_gamma_power' else 0.)
        summary,_=ev.summarize_samples(samples,16,SEEDS[0]);pairs,_=ev.paired_rows(summary,samples,16,SEEDS[0])
        low=next(r for r in pairs if r['contrast_id']=='gamma_minus_H32' and r['estimand']=='low_gamma_mean')
        self.assertAlmostEqual(low['delta_mean'],.5)
        self.assertAlmostEqual(low['batch_delta_mean'],.5)
        self.assertAlmostEqual(low['batch_delta_se'],.5)
        self.assertGreater(low['batch_delta_se'],low['delta_se'])
        single=[r for r in samples if r['sample_index']<8]
        summary,_=ev.summarize_samples(single,8,SEEDS[0]);pairs,_=ev.paired_rows(summary,single,8,SEEDS[0])
        for row in pairs:
            self.assertFalse(row['batch_ci_estimable'])
            self.assertIsNone(row['batch_delta_se'])
            self.assertIsNone(row['batch_delta_ci95_low'])
            self.assertIsNone(row['batch_delta_ci95_high'])

    def test_rejects_missing_duplicate_nonfinite_wrong_identity_energy_and_gate(self):
        import evaluate_grouped_hybrid_power as ev
        original=self.samples()
        cases={
            'missing':lambda rows:rows.pop(),
            'duplicate':lambda rows:rows.append(deepcopy(rows[0])),
            'nonfinite':lambda rows:rows[0].update(sum_rate=float('nan')),
            'wrong_seed':lambda rows:rows[0].update(seed=20261228),
            'wrong_batch':lambda rows:rows[0].update(batch_index=1),
            'wrong_layout':lambda rows:rows[0].update(Df_per_user=32),
            'wrong_units':lambda rows:rows[0].update(feedback_snr_db=0),
            'energy':lambda rows:rows[0].update(energy_ue0=65),
            'branch_energy':lambda rows:rows[0].update(shared_energy_mean_per_ue=1),
            'symbol_energy':lambda rows:rows[0].update(exclusive_energy_per_active_symbol=2),
            'noise':lambda rows:rows[0].update(empirical_noise_power=2),
            'raw_nmse_nonfinite':lambda rows:rows[0].update(nmse_db=float('inf')),
            'nominal_power':lambda rows:rows[0].update(precoder_power=1),
            'control_gate':lambda rows:rows[0].update(g_log_energy_ratio=.1),
            'empty_shared':lambda rows:rows[0].update(shared_energy_per_active_symbol=0),
        }
        for name,mutate in cases.items():
            changed=deepcopy(original);mutate(changed)
            with self.subTest(case=name),self.assertRaises(ValueError):ev.summarize_samples(changed,16,SEEDS[0])
        for field,value in (('g_log_energy_ratio',2.),('g_log_energy_ratio',.1)):
            changed=deepcopy(original)
            next(r for r in changed if r['method_key']=='h32_static_power').update({field:value})
            with self.subTest(case='bounded_or_channel_dependent_gate',value=value),self.assertRaises(ValueError):
                ev.summarize_samples(changed,16,SEEDS[0])

    def test_summary_must_match_raw_and_cannot_omit_an_unfavorable_method(self):
        import evaluate_grouped_hybrid_power as ev
        samples=self.samples();summary,_=ev.summarize_samples(samples,16,SEEDS[0])
        for kind in ('different_mean','missing_method','duplicate_condition'):
            changed=deepcopy(summary)
            if kind=='different_mean':changed[0]['sum_rate_mean']+=.1
            elif kind=='missing_method':changed=[r for r in changed if r['method_key']!='h32_gamma_power']
            else:changed[-1]=deepcopy(changed[0])
            with self.subTest(kind=kind),self.assertRaises(ValueError):ev.paired_rows(changed,samples,16,SEEDS[0])

    def test_result_rule_uses_continued_fdma_and_retains_gamma_vs_static_comparison(self):
        import evaluate_grouped_hybrid_power as ev
        from grouped_hybrid_power_results import interpret
        offsets={'h32_gamma_power':.7,'h32_static_power':.8,'h32_control':.2,
                 'frozen_fdma_source':5.,'frozen_h32_source':5.}
        samples=self.samples(1600,lambda m,g,i:offsets.get(m,0.))
        summary,_=ev.summarize_samples(samples,1600,SEEDS[0]);pairs,_=ev.paired_rows(summary,samples,1600,SEEDS[0])
        result=interpret(summary,pairs,SEEDS[0])
        self.assertTrue(result['engineering_criteria']['gamma_minus_FDMA'])
        self.assertTrue(result['engineering_criteria']['static_minus_FDMA'])
        self.assertFalse(result['engineering_criteria']['H32_minus_FDMA'])
        adaptive=result['comparisons']['gamma_minus_static']['low_gamma']
        self.assertAlmostEqual(adaptive['delta_mean'],-.1,places=12)
        self.assertLess(result['comparisons']['FDMA_minus_frozen']['low_gamma']['delta_mean'],0)
        self.assertEqual(set(result['engineering_criteria']),set(ev.PRIMARY))
        self.assertFalse(result['training_seeds_pooled'])

    def test_hash_pairing_requires_all_actual_input_digests(self):
        import evaluate_grouped_hybrid_power as ev
        records=[dict(batch_index=i,sample_index_start=8*i,sample_index_stop=8*(i+1),
            H_dl_sha256='a'*64,H_ul_sha256='b'*64,standard_noise_sha256='c'*64) for i in range(200)]
        manifest=dict(protocol=dict(num_batches=200),shared_batch_inputs=records)
        self.assertEqual(ev.verify_shared_inputs(manifest,deepcopy(records)),records)
        bad=deepcopy(manifest);bad['shared_batch_inputs'][100]['standard_noise_sha256']='d'*64
        with self.assertRaises(ValueError):ev.verify_shared_inputs(bad,records)
        bad=deepcopy(manifest);bad['shared_batch_inputs'].pop()
        with self.assertRaises(ValueError):ev.verify_shared_inputs(bad)
        bad=deepcopy(manifest);bad['shared_batch_inputs'][0]['H_dl_sha256']='not-a-hash'
        with self.assertRaises(ValueError):ev.verify_shared_inputs(bad)

    def test_evaluator_dry_run_is_lightweight_and_has_no_weight_or_output_side_effect(self):
        assert_lightweight_cli(self,'evaluate_grouped_hybrid_power',[
            ['--seed',str(seed),'--dry-run'] for seed in SEEDS],
            "assert all(p['protocol']['num_samples']==1600 and len(p['checkpoints'])==6 and p['outputs_not_created'] for p in plans)")


class PowerControllerTests(unittest.TestCase):
    def test_summary_reads_nested_training_trends_without_loading_weights(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout
        from io import StringIO
        import run_grouped_hybrid_power_batch as ctl
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch_root=root/'batches';folder=batch_root/'batch_fixture'
            folder.mkdir(parents=True)
            (root/'last_batch.json').write_text(json.dumps(dict(directory=str(folder))),encoding='utf-8')
            trend=dict(best_step=20000,best_mean_all=1.25,last_five_validation_points=[],
                last_5000_updates=dict(mean_change=.01),convergence_established=False)
            summary=dict(status='running',stages={},results=dict(training_trends=dict(
                per_seed={str(SEEDS[0]):{'h32_gamma_power':trend}},source_files={})))
            (folder/'batch_summary.json').write_text(json.dumps(summary),encoding='utf-8')
            stream=StringIO()
            with patch.multiple(ctl,OUTPUT_ROOT=root,BATCH_ROOT=batch_root),patch.object(
                    ctl,'power_config',side_effect=lambda arm,seed:power_config(arm,seed,root)),redirect_stdout(stream):
                self.assertEqual(ctl.print_summary(),0)
            rendered=stream.getvalue()
            self.assertIn('training_trend ',rendered)
            self.assertIn('"convergence_established": false',rendered)
            self.assertIn('"weights_read": false',rendered)
            self.assertIn('unknown (not yet evaluated)',rendered)

    def fixture(self, root):
        import run_grouped_hybrid_power_batch as ctl
        from configs.grouped_hybrid_anchor_train_config import ARMS as old_arms
        folder = root / 'old_batch'; folder.mkdir()
        evidence = folder / 'evidence.json'; evidence.write_text('{}\n', encoding='utf-8')
        package = folder / 'package.zip'; package.write_bytes(b'schema-only package fixture')
        names = {'environment', 'sources', 'cuda_preflight', 'results'} | {
            f'train_{a}_{s}' for s in SEEDS for a in old_arms} | {f'evaluate_{s}' for s in SEEDS}
        summary = dict(status='completed', stages={n: dict(status='completed',
            evidence_sha256={str(evidence): ctl.sha256(evidence)}, result={}) for n in names},
            training={}, result_package=str(package), result_package_sha256=ctl.sha256(package))
        for seed in SEEDS:
            summary['training'][str(seed)] = {}
            for arm in old_arms:
                row = dict(arm=arm, training_seed=seed, completed_updates=20000, best_step=20000,
                    best_prior_update_count=20000, latest_prior_update_count=20000)
                summary['training'][str(seed)][arm] = row
                summary['stages'][f'train_{arm}_{seed}']['result']['report'] = deepcopy(row)
        (folder / 'batch_summary.json').write_text(json.dumps(summary), encoding='utf-8')
        return folder, summary, evidence

    def test_completed_old_batch_read_only_and_all_incomplete_variants_rejected(self):
        from unittest.mock import patch
        import run_grouped_hybrid_power_batch as ctl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder, summary, evidence = self.fixture(root)
            before = {str(p): p.read_bytes() for p in folder.iterdir()}
            with patch.multiple(ctl, ROOT=root, OLD_BATCH=folder):
                frozen = ctl.inspect_old_batch(summary, folder)
                self.assertEqual(frozen[str(evidence)], ctl.sha256(evidence))
                for kind in ('unfinished', 'missing_stage', 'empty_evidence', 'wrong_count', 'different_stage_record'):
                    bad = deepcopy(summary)
                    if kind == 'unfinished': bad['status'] = 'failed'
                    elif kind == 'missing_stage': bad['stages'].pop('cuda_preflight')
                    elif kind == 'empty_evidence': bad['stages']['environment']['evidence_sha256'] = {}
                    elif kind == 'wrong_count': bad['training'][str(SEEDS[0])]['fdma_control']['best_prior_update_count'] = 500
                    else: bad['stages'][f'train_fdma_control_{SEEDS[0]}']['result']['report']['best_step'] = 500
                    with self.subTest(kind=kind), self.assertRaises(ValueError): ctl.inspect_old_batch(bad, folder)
                self.assertEqual(before, {str(p): p.read_bytes() for p in folder.iterdir()})
                evidence.write_text('{"changed":true}', encoding='utf-8')
                with self.assertRaises(ValueError): ctl.inspect_old_batch(summary, folder)

    def test_completed_stage_reuse_never_reruns_and_tampered_evidence_stops(self):
        from unittest.mock import Mock
        import run_grouped_hybrid_power_batch as ctl
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); evidence = folder / 'checked.json'; evidence.write_text('{}', encoding='utf-8')
            result = {'passed': True}; summary = {'stages': {'cuda_preflight': dict(status='completed',
                evidence_sha256={str(evidence): ctl.sha256(evidence)}, result=result)}}
            action = Mock(side_effect=AssertionError('must not rerun completed stage'))
            self.assertEqual(ctl._run_stage(summary, folder, 'cuda_preflight', action), result)
            action.assert_not_called(); self.assertFalse((folder / 'batch_summary.json').exists())
            evidence.write_text('changed', encoding='utf-8')
            with self.assertRaises(RuntimeError): ctl._run_stage(summary, folder, 'cuda_preflight', action)
            action.assert_not_called()

    def test_train_controller_500_pause_then_only_explicit_resume_no_extra_budget(self):
        from unittest.mock import patch
        import run_grouped_hybrid_power_batch as ctl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = power_config('h32_control', SEEDS[0], root)
            commands = []
            def child(folder, label, command):
                commands.append(command)
                Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=True)
                Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=True)
                if command[-2:] == ['--stop-after', '500']:
                    self.assertFalse(Path(cfg.SAVE_PATH).exists())
                    Path(cfg.SAVE_PATH).write_bytes(b'not a real checkpoint; subprocess control fixture')
                    state, step = 'paused', 500
                else:
                    self.assertEqual(command[-2:], ['--resume', '--train'])
                    self.assertEqual(json.loads(Path(cfg.STATUS_PATH).read_text())['step'], 500)
                    state, step = 'completed', 20000
                Path(cfg.STATUS_PATH).write_text(json.dumps(dict(state=state, step=step)), encoding='utf-8')
                return str(root / (label + '.log'))
            def record(*args, **kwargs): return {'report': {'passed': True}, '_evidence': []}
            with patch.object(ctl, 'power_config', return_value=cfg), patch.object(ctl, '_child', side_effect=child), patch.object(ctl, 'stage_child', side_effect=record):
                ctl.train_one(root, 'h32_control', SEEDS[0])
                self.assertEqual(len(commands), 2)
                self.assertEqual(commands[0][-2:], ['--stop-after', '500'])
                self.assertEqual(commands[1][-2:], ['--resume', '--train'])
                ctl.train_one(root, 'h32_control', SEEDS[0])
                self.assertEqual(len(commands), 2)
                Path(cfg.STATUS_PATH).write_text(json.dumps(dict(state='paused', step=500)), encoding='utf-8')
                ctl.train_one(root, 'h32_control', SEEDS[0])
                self.assertEqual(len(commands), 3)
                self.assertEqual(commands[-1][-2:], ['--resume', '--train'])

    def test_unowned_output_and_failed_smoke_do_not_trigger_continue(self):
        from unittest.mock import patch, Mock
        import run_grouped_hybrid_power_batch as ctl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = power_config('h32_static_power', SEEDS[0], root)
            Path(cfg.LOG_DIR).mkdir(parents=True)
            child, stage = Mock(), Mock()
            with patch.object(ctl, 'power_config', return_value=cfg), patch.object(ctl, '_child', child), patch.object(ctl, 'stage_child', stage):
                with self.assertRaises(ValueError): ctl.train_one(root, cfg.arm, cfg.SEED)
            child.assert_not_called(); stage.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = power_config('h32_static_power', SEEDS[0], root)
            child = Mock(side_effect=RuntimeError('synthetic failed technical smoke')); stage = Mock()
            with patch.object(ctl, 'power_config', return_value=cfg), patch.object(ctl, '_child', child), patch.object(ctl, 'stage_child', stage):
                with self.assertRaises(RuntimeError): ctl.train_one(root, cfg.arm, cfg.SEED)
            self.assertEqual(child.call_count, 1); stage.assert_not_called()


if __name__ == '__main__': unittest.main()
