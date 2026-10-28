"""Lightweight hybrid config/lifecycle/checkpoint tests; no large model or channels."""
from __future__ import annotations
import ast
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,Path(__file__).resolve().parent):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from configs.grouped_hybrid_anchor_train_config import (
    ARMS,SEEDS,anchor_config,validate_config,protocol,make_channel_data,SOURCE_BEST_SHA256)
from configs.grouped_codec_fdma_train_config import fdma_config
from configs.train_config import TRAINING_RECIPES
from test_train_grouped_codec import resume_metadata as old_resume_metadata
from cli_contracts import assert_lightweight_cli
import train_grouped_hybrid_anchor as entry


def resume_metadata(cfg,step=500):
    """Schema sentinels, not fabricated tensors or real checkpoint evidence."""
    payload=old_resume_metadata(cfg,step)
    payload['state'].update({'power_prior.pi_nodes':'schema only',
        'power_prior.update_count':step,'power_prior.node_update_mass':'schema only'})
    if step==0:payload['optimizer']['state']={}
    return payload


def trainer_lifecycle_helpers():
    """Execute the real dependency-free helper definitions, without fake Torch/Sionna."""
    tree=ast.parse((ROOT/'main_train.py').read_text(encoding='utf-8'))
    names={'_validate_lifecycle_protocol','_commit_after_optimizer'}
    module=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
    scope={};exec(compile(module,'main_train.py','exec'),scope)
    trainer=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='train_proposal')
    loop=next(node for node in trainer.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='step')
    index=next(i for i,node in enumerate(loop.body) if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name)
        and node.value.func.value.id=='optimizer' and node.value.func.attr=='step')
    definition=ast.parse('def actual_update(optimizer,scheduler,lifecycle_protocol,model,output,step):\n    pass').body[0]
    definition.body=loop.body[index:index+3]
    module=ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[]))
    exec(compile(module,'main_train.py','exec'),scope)
    return scope


class HybridConfigTests(unittest.TestCase):
    def test_three_fixed_layouts_six_runs_same_new_addresses_and_unchanged_old_registry(self):
        before=deepcopy(TRAINING_RECIPES)
        for seed in SEEDS:
            protocols=[];executions=[]
            for arm in ARMS:
                cfg=anchor_config(arm,seed);validate_config(cfg)
                self.assertEqual((cfg.TOTAL_STEPS,cfg.BATCH_SIZE,cfg.VAL_BATCH,cfg.VAL_INTERVAL),(20000,48,256,500))
                self.assertEqual((cfg.LR,cfg.WEIGHT_DECAY,cfg.WARMUP_PCT,cfg.GRAD_CLIP),(1e-4,1e-5,.1,.5))
                self.assertEqual(cfg.VAL_SNR_LIST,[0,5,10,15,20,25]);self.assertEqual(cfg.DOWNLINK_SNR_DB,25)
                self.assertEqual([cfg.PH1_W_DIR,cfg.PH1_W_MSE,cfg.PH2_W_DIR,cfg.PH2_W_MSE],[0,0,0,0])
                self.assertFalse(hasattr(cfg,'ALLOCATION_GRID'))
                self.assertEqual(cfg.to_dict()['source_identity']['best_sha256'],SOURCE_BEST_SHA256[seed])
                self.assertTrue(cfg.to_dict()['screening_protocol']['source_step_zero_competes'])
                protocols.append(cfg.channel_data_protocol);executions.append(cfg.to_dict()['execution_protocol'])
            self.assertTrue(all(p==protocols[0] for p in protocols));self.assertTrue(all(p==executions[0] for p in executions))
            self.assertNotEqual(protocols[0],fdma_config(seed).channel_data_protocol)
        self.assertEqual(before,TRAINING_RECIPES)

    def test_namespace_changes_data_init_gamma_noise_and_fingerprint_locked(self):
        from configs.grouped_codec_train_config import protocol as old
        cfg=anchor_config(ARMS[0],SEEDS[0])
        self.assertNotEqual(protocol.namespace_seed(SEEDS[0],'channel'),SEEDS[0])
        self.assertNotEqual(protocol.namespace_seed(SEEDS[0],'initialization'),SEEDS[0])
        for step in (1,500,501,20000):self.assertNotEqual(protocol.sample_gamma(SEEDS[0],step),old.sample_gamma(SEEDS[0],step))
        for mutate in ('recipe','channel','extra'):
            other=deepcopy(cfg)
            if mutate=='recipe':other.recipe=replace(other.recipe,lr=2e-5)
            elif mutate=='channel':other.channel_data_protocol['base_seed']+=1
            else:other.unsaved_override=True
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):validate_config(other)
        with self.assertRaises(ValueError):anchor_config('legacy_fdma',SEEDS[0])

    def test_one_guarded_stdlib_dry_run_no_weights_or_outputs(self):
        assert_lightweight_cli(self,'train_grouped_hybrid_anchor',[
            ['--arm','hybrid_f16_a96','--seed','20261030','--dry-run']],
            "assert plans[0]['config']['lifecycle_protocol']['initial_validation'] is True\n"
            "assert plans[0]['config']['recipe']['total_steps']==20000")


class HybridCheckpointTests(unittest.TestCase):
    def test_step0_500_and20k_full_state_with_matching_committed_count(self):
        for arm in ARMS:
            cfg=anchor_config(arm,SEEDS[0])
            for step in (0,500,20000):entry.validate_resume_payload(resume_metadata(cfg,step),cfg,allow_completed=True)
            latest=resume_metadata(cfg,20000);best=resume_metadata(cfg,0)
            entry.validate_best_metadata(best,latest,cfg)
            with self.assertRaises(ValueError):entry.validate_resume_payload(latest,cfg)

    def test_missing_stats_wrong_count_cross_arm_and_reset_scheduler_rejected(self):
        cfg=anchor_config(ARMS[1],SEEDS[0])
        for issue in ('count','prior','scheduler','rng','optimizer','crossarm','step0_moments'):
            payload=resume_metadata(cfg,0 if issue=='step0_moments' else 500)
            if issue=='count':payload['state']['power_prior.update_count']=499
            elif issue=='prior':payload['state'].pop('power_prior.pi_nodes')
            elif issue=='scheduler':payload['scheduler']['last_epoch']=0
            elif issue=='rng':payload['rng_state'].pop('cuda')
            elif issue=='optimizer':payload['optimizer']['state']={}
            elif issue=='crossarm':payload['config']['arm']='fdma_control'
            else:payload['optimizer']['state']={0:{'schema':'unexpected initial state'}}
            with self.subTest(issue=issue),self.assertRaises((ValueError,KeyError)):entry.validate_resume_payload(payload,cfg)

    def test_pinned_source_identity_and_completed_source_recipe(self):
        from train_grouped_codec import fingerprint
        seed=SEEDS[0];cfg=fdma_config(seed);latest=old_resume_metadata(cfg,20000);best=deepcopy(latest);best['step']=19500
        sidecars=dict(config=dict(cfg.to_dict(),config_fingerprint=fingerprint(cfg)),status=dict(
            state='completed',step=20000,experiment=cfg.EXP_NAME,best_metric_name='mean_all',best_metric=12.))
        hashes=dict(best_sha256=SOURCE_BEST_SHA256[seed],latest_sha256='1'*64,config_sha256='2'*64,status_sha256='3'*64)
        record=entry.verify_source_payloads(seed,best,latest,sidecars,hashes)
        self.assertEqual(record['best_step'],19500);self.assertEqual(record['latest_step'],20000)
        with self.assertRaises(ValueError):entry.verify_source_payloads(seed,best,latest,sidecars,dict(hashes,best_sha256='0'*64))
        latest['step']=500
        with self.assertRaises(ValueError):entry.verify_source_payloads(seed,best,latest,sidecars,hashes)

    def test_output_protection_and_incomplete_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new';cfg=anchor_config(ARMS[0],SEEDS[0],root)
            entry.check_run_state(cfg,resume=False,allowed_root=root)
            Path(cfg.SAVE_DIR).mkdir(parents=True)
            with self.assertRaises(FileExistsError):entry.check_run_state(cfg,resume=False,allowed_root=root)
            with self.assertRaises(FileNotFoundError):entry.check_run_state(cfg,resume=True,allowed_root=root)


class HybridLifecycleTests(unittest.TestCase):
    def test_default_none_unchanged_and_binding_rejected_before_outputs(self):
        helpers=trainer_lifecycle_helpers();validate=helpers['_validate_lifecycle_protocol'];commit=helpers['_commit_after_optimizer']
        old=SimpleNamespace(to_dict=lambda:{})
        validate(old,None);commit(None,None,None,1)
        cfg=anchor_config(ARMS[0],SEEDS[0]);life=entry.AnchorLifecycle(cfg)
        validate(cfg,life)
        with self.assertRaises(ValueError):validate(cfg,None)
        with self.assertRaises(ValueError):validate(old,life)
        with self.assertRaises(ValueError):validate(anchor_config(ARMS[1],SEEDS[0]),life)

    def test_actual_trainer_update_order_and_failed_optimizer_never_commits(self):
        helpers=trainer_lifecycle_helpers();events=[]
        life=SimpleNamespace(after_optimizer_step=lambda *args:events.append('commit'),
            validate_model_step=lambda *args:events.append('count'))
        optimizer=SimpleNamespace(step=lambda:events.append('optimizer'))
        scheduler=SimpleNamespace(step=lambda:events.append('scheduler'))
        helpers['actual_update'](optimizer,scheduler,life,None,None,1)
        self.assertEqual(events,['optimizer','commit','count','scheduler'])
        events.clear();helpers['actual_update'](optimizer,scheduler,None,None,None,1)
        self.assertEqual(events,['optimizer','scheduler'])
        def fail():events.append('failed');raise RuntimeError('synthetic optimizer failure')
        events.clear();optimizer.step=fail
        with self.assertRaises(RuntimeError):helpers['actual_update'](optimizer,scheduler,life,None,None,1)
        self.assertEqual(events,['failed'])

    def test_callback_uses_only_returned_pending_statistics(self):
        cfg=anchor_config(ARMS[0],SEEDS[0]);life=entry.AnchorLifecycle(cfg);calls=[]
        model=SimpleNamespace(post_optimizer_commit=lambda stats,step:calls.append((stats,step)),
            power_prior=SimpleNamespace(validate_state=lambda step:calls.append(step)))
        pending={'detached':'schema'};life.after_optimizer_step(model,('W','Hhat',pending),500);life.validate_model_step(model,500)
        self.assertEqual(calls,[(pending,500),500])
        with self.assertRaises(ValueError):life.after_optimizer_step(model,('W','Hhat'),500)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Real Torch unavailable; tiny prior resume check not executed')
    def test_small_real_prior_checkpoint_state_accepts_nonuniform_but_rejects_corrupt(self):
        import torch
        cfg=anchor_config(ARMS[1],SEEDS[0]);prior=protocol.make_power_prior(cfg.layout_spec)
        raw=torch.ones(2,4,256,dtype=torch.complex64);raw[:,:,0]*=3
        _,canonical,_=protocol.normalize_and_map(raw,cfg.layout_spec)
        prior.commit(prior.pending(canonical,7.5,1))
        holder=torch.nn.Module();holder.add_module('power_prior',protocol.make_power_prior(cfg.layout_spec))
        state={'power_prior.'+key:value.clone() for key,value in prior.state_dict().items()}
        entry.validate_model_state(torch,holder,state,step=1)
        changed=deepcopy(state);changed['power_prior.pi_nodes'][0,0,0]=-1
        with self.assertRaises(ValueError):entry.validate_model_state(torch,holder,changed,step=1)
        changed=deepcopy(state);changed['power_prior.node_update_mass'].zero_()
        with self.assertRaises(ValueError):entry.validate_model_state(torch,holder,changed,step=1)




# Synthetic result-schema/statistics fixtures below are NOT real evaluation or
# checkpoint evidence. No model construction, channel generation or local plots.
import csv
import gzip
import hashlib
import math
import statistics
from unittest import mock
import evaluate_grouped_hybrid_anchor as evaluation
import grouped_hybrid_anchor_results as report


def synthetic_anchor_samples(count=16,seed=SEEDS[0]):
    samples=[]
    for method in evaluation.METHODS:
        df,da=evaluation.LAYOUTS[method]
        for gamma in evaluation.GAMMAS:
            for index in range(count):
                baseline=10+.02*index+.1*gamma
                control=baseline+.2+.005*index*(gamma/5-2)
                difference=(.8 if gamma<=10 else -.05)+.02*(index%8-3.5)+.03*(index//8)
                rate={"frozen_F":baseline,"fdma_control":control,
                    "hybrid_f16_a96":control+difference,
                    "hybrid_f32_a64":control+.2-.02*gamma-.01*(index%8-3.5)+.01*(index//8)}[method]
                row=evaluation.sample_identity(method,gamma,index,seed)
                row.update({name:1. for name in evaluation.COMMON_METRICS})
                row.update(sum_rate=rate,nmse_db=-3.,precoder_power=256.,precoder_power_max_error=0.,
                    exclusive_energy_mean_per_ue=64*df/(df+da),shared_energy_mean_per_ue=64*da/(df+da),
                    empirical_noise_power=row["noise_variance_N0"]*(1+.001*(index%8)),
                    recovered_mse_active=.2,corrected_mse_active=.1,recovered_mse_exclusive=.2,
                    downlink_desired_mean=1.,downlink_interference_mean=.02)
                row.update({f"energy_ue{k}":64. for k in range(4)})
                for name in evaluation.OPTIONAL_METRICS:
                    row[name]=.15 if evaluation._available(method,name) else None
                samples.append(row)
    return samples


def synthetic_formal_tables(seed):
    """Formal-sized table schema sentinel; the actual pairing test uses16 rows."""
    summary,_=evaluation.summarize_samples(synthetic_anchor_samples(seed=seed),16,seed)
    paired,_=evaluation.paired_rows(summary,synthetic_anchor_samples(seed=seed),16,seed)
    for row in summary+paired:
        row.update(num_samples=1600,num_batches=200,batch_size=8)
    for row in summary:row["preflight"]=False
    return summary,paired


def _synthetic_csv(path,records):
    with path.open("w",encoding="utf-8",newline="")as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def synthetic_report_inputs(root):
    """Schema-only completed manifests in a TemporaryDirectory, not model evidence."""
    from as_receiver_batch_support import sha256
    evaluations,training,sources={},{},{}
    for seed in SEEDS:
        training[str(seed)]={}
        for arm_index,arm in enumerate(ARMS):
            path=root/f"metrics_{seed}_{arm}.jsonl"
            points=[]
            for step in range(0,20001,500):
                value=12. if arm_index==0 else 10.+step/10000
                table={f"layout0_gamma{g}":value+.01*g for g in evaluation.GAMMAS}
                points.append(dict(event="validation",step=step,selection_metric_name="mean_all",
                    selection_metric=sum(table.values())/6,table=table,
                    gamma_db=list(evaluation.GAMMAS),downlink_snr_db=25))
            path.write_text("\n".join(json.dumps(p)for p in points)+"\n",encoding="utf-8")
            training[str(seed)][arm]=dict(completed_updates=20000,best_step=0 if arm_index==0 else 20000,
                best_sha256=hashlib.sha256(f"synthetic-best-{seed}-{arm}".encode()).hexdigest(),
                latest_sha256=hashlib.sha256(f"synthetic-latest-{seed}-{arm}".encode()).hexdigest(),
                metrics_path=str(path),metrics_sha256=sha256(path))
        pinned=dict(best_step=19500,latest_step=20000,best_sha256=SOURCE_BEST_SHA256[seed],
            latest_sha256="1"*64,config_sha256="2"*64,status_sha256="3"*64)
        pinned.update({name+"_path":str(root/f"not_created_source_{seed}_{name}")
                       for name in ("best","latest","config","status")})
        sources[str(seed)]=pinned
        directory=root/f"evaluation_{seed}";directory.mkdir()
        evaluations[str(seed)]=str(directory)
        summary,paired=synthetic_formal_tables(seed)
        _synthetic_csv(directory/f"summary_{evaluation.TAG}.csv",summary)
        _synthetic_csv(directory/f"paired_deltas_{evaluation.TAG}.csv",paired)
        parameters=[dict(method_key=m,training_seed=seed,full_parameters=120 if m=="frozen_F" else 130,
            ue_parameters=20,bs_parameters=100 if m=="frozen_F" else 110)for m in evaluation.METHODS]
        _synthetic_csv(directory/f"parameters_{evaluation.TAG}.csv",parameters)
        for name in ("samples","paired_samples"):
            with gzip.open(directory/f"{name}_{evaluation.TAG}.csv.gz","wt",encoding="utf-8")as stream:
                stream.write("fixture_kind\nsynthetic_schema_only\n")
        for name in ("metadata","results"):
            (directory/f"{name}_{evaluation.TAG}.json").write_text(json.dumps({"synthetic":True}),encoding="utf-8")
        methods={}
        for arm in ARMS:
            record=training[str(seed)][arm]
            methods[arm]=dict(selected_best_step=record["best_step"],prior_update_count=record["best_step"],
                latest_step=20000,strict_loading=True,best_sha256=record["best_sha256"],latest_sha256=record["latest_sha256"])
        methods["frozen_F"]=dict(selected_best_step=19500,prior_update_count=None,
            strict_loading=True,best_sha256=SOURCE_BEST_SHA256[seed])
        inputs=[dict(batch_index=i,sample_index_start=8*i,sample_index_stop=8*(i+1),
            **{key:hashlib.sha256(f"synthetic-{key}-{i}".encode()).hexdigest()
               for key in ("H_dl_sha256","H_ul_sha256","standard_noise_sha256")})for i in range(200)]
        snapshots={method:dict(buffers={},training={"":False})for method in evaluation.METHODS}
        manifest=dict(status="completed",results_verified=True,protocol=evaluation.protocol(seed),
            freeze_checks=dict(passed=True,errors=[],loaded_methods=list(evaluation.METHODS)),
            shared_channel_noise_inputs_unchanged=True,samples_hashed_pairing=True,prior_frozen=True,
            model_buffers_before=snapshots,model_buffers_after=deepcopy(snapshots),
            input_sha256_before={"synthetic-not-a-weight":"0"*64},input_sha256_after={"synthetic-not-a-weight":"0"*64},
            methods=methods,source_F=pinned,shared_batch_inputs=inputs,
            output_sha256=evaluation.artifact_hashes(directory))
        (directory/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    return evaluations,training,sources


class HybridEvaluationTests(unittest.TestCase):
    def test_new_test_namespace_and_exact_explicit_four_method_plan(self):
        first,second=(evaluation.protocol(seed)for seed in SEEDS)
        self.assertEqual(first["channel_data_protocol"],second["channel_data_protocol"])
        self.assertEqual(first["noise_address"],second["noise_address"])
        self.assertTrue(first["namespace_check"]["actual_channel_seed_sets_disjoint"])
        self.assertEqual(first["seed"],20261228)
        self.assertEqual([p["method_key"]for p in evaluation.checkpoint_plan(SEEDS[0])],list(evaluation.METHODS))
        self.assertEqual(evaluation.checkpoint_plan(SEEDS[0])[-1]["expected_best_sha256"],SOURCE_BEST_SHA256[SEEDS[0]])
        self.assertEqual(first["layout_by_method"]["hybrid_f16_a96"],dict(Df_per_user=16,Da_per_group=96))

    def test_all_conditions_pairing_correlated_crossgamma_and_batch_intervals(self):
        samples=synthetic_anchor_samples()
        summary,indexed=evaluation.summarize_samples(samples,16,SEEDS[0])
        paired,raw=evaluation.paired_rows(summary,list(reversed(samples)),16,SEEDS[0])
        self.assertEqual((len(summary),len(paired),len(raw)),(24,32,512))
        row=next(r for r in paired if r["contrast_id"]=="H16_minus_Fnew"and r["estimand"]=="low_gamma_mean")
        values=[statistics.fmean(indexed["hybrid_f16_a96",g][i]["sum_rate"]-
            indexed["fdma_control",g][i]["sum_rate"]for g in (0,5,10))for i in range(16)]
        batches=[statistics.fmean(values[:8]),statistics.fmean(values[8:])]
        self.assertEqual(row["delta_mean"],statistics.fmean(values))
        self.assertAlmostEqual(row["delta_se"],statistics.stdev(values)/4)
        self.assertAlmostEqual(row["batch_delta_se"],statistics.stdev(batches)/math.sqrt(2))
        self.assertNotEqual(row["delta_se"],row["batch_delta_se"])
        self.assertEqual(evaluation.CONTRASTS["Fnew_minus_Ffrozen"],{"fdma_control":1,"frozen_F":-1})

    def test_absent_frozen_proxy_and_empty_shared_support_are_NA_not_fake_zero(self):
        samples=synthetic_anchor_samples()
        summary,_=evaluation.summarize_samples(samples,16,SEEDS[0])
        frozen=next(r for r in summary if r["method_key"]=="frozen_F")
        self.assertIsNone(frozen["proxy_mse_active_mean"]);self.assertIsNone(frozen["anchor_rms_mean_mean"])
        fdma=next(r for r in summary if r["method_key"]=="fdma_control")
        self.assertIsNone(fdma["recovered_mse_shared_mean"]);self.assertIsNotNone(fdma["proxy_mse_active_mean"])
        bad=deepcopy(samples);next(r for r in bad if r["method_key"]=="frozen_F")["anchor_rms_mean"]=0.
        with self.assertRaises(ValueError):evaluation.summarize_samples(bad,16,SEEDS[0])

    def test_duplicate_missing_mixed_seed_nonfinite_wrong_energy_noise_and_layout_rejected(self):
        for issue in ("duplicate","missing","seed","nonfinite","energy","noise","layout","legacyfield"):
            data=synthetic_anchor_samples()
            if issue=="duplicate":data.append(deepcopy(data[0]))
            elif issue=="missing":data.pop()
            elif issue=="seed":data[0]["seed"]=20261220
            elif issue=="nonfinite":data[0]["sum_rate"]=float("nan")
            elif issue=="energy":data[0]["shared_energy_mean_per_ue"]=64.
            elif issue=="noise":data[0]["empirical_noise_power"]*=2
            elif issue=="layout":data[0]["Da_per_group"]=96
            else:data[0]["allocation_da"]=0
            with self.subTest(issue=issue),self.assertRaises(ValueError):evaluation.summarize_samples(data,16,SEEDS[0])

    def test_summary_raw_mismatch_and_single_batch_CI_not_invented(self):
        samples=synthetic_anchor_samples(8);summary,_=evaluation.summarize_samples(samples,8,SEEDS[0])
        paired,_=evaluation.paired_rows(summary,samples,8,SEEDS[0])
        self.assertTrue(all(r["batch_delta_se"]is None and not r["batch_ci_estimable"]for r in paired))
        changed=deepcopy(summary);changed[0]["sum_rate_mean"]+=1
        with self.assertRaises(ValueError):evaluation.paired_rows(changed,samples,8,SEEDS[0])


    @unittest.skipUnless(importlib.util.find_spec("torch")is not None,
                         "Real Torch unavailable; tiny diagnostic-axis check not executed")
    def test_small_tensor_diagnostics_route_energy_proxy_NA_and_downlink_axes(self):
        import torch
        for method in evaluation.METHODS:
            layout=protocol.LayoutSpec("fdma_control"if method=="frozen_F"else method)
            geometry=protocol.tensor_geometry(layout)
            active=geometry["active_mask"];exclusive=geometry["exclusive_mask"];shared=geometry["shared_mask"]
            truth=(active.to(torch.float32)*math.sqrt(64/layout.active_per_user)).to(torch.complex64)[None]
            energy=truth.abs().square().sum(-1)
            noise=torch.zeros(1,32,256,dtype=torch.complex64)
            diagnostic=dict(z_hat=truth*.9,z_corrected=truth*.95,true_z=truth,active_mask=active,
                energy=energy,noise_power=torch.zeros(1),N0=.25,gamma_db=0.)
            if method!="frozen_F":
                prior=active.float()*(64/layout.active_per_user)
                proxy=active.float()[None]*.2
                diagnostic.update(exclusive_mask=exclusive,shared_mask=shared,
                    energy_exclusive=truth.abs().square().masked_fill(~exclusive[None],0).sum(-1),
                    energy_shared=truth.abs().square().masked_fill(~shared[None],0).sum(-1),
                    error_proxy=proxy,prior_power=prior,error_feature=torch.log1p(proxy/prior.clamp_min(1e-8)),
                    anchor_rms=torch.full((1,4),.001),prior_update_count=0)
            values=evaluation.extract_diagnostics(torch,diagnostic,noise,0,method)
            self.assertAlmostEqual(float(values["exclusive_energy_mean_per_ue"]+
                                         values["shared_energy_mean_per_ue"]),64,places=4)
            if method=="frozen_F":
                self.assertIsNone(values["proxy_mse_active"]);self.assertIsNone(values["anchor_rms_mean"])
            else:self.assertAlmostEqual(float(values["proxy_mse_active"]),.2,places=6)
            if layout.da==0:self.assertIsNone(values["recovered_mse_shared"])
        H=torch.ones(1,4,32,256,dtype=torch.complex64)/math.sqrt(32)
        W=torch.ones(1,32,4,256,dtype=torch.complex64)/math.sqrt(128)
        powers=evaluation.downlink_diagnostics(torch,W,H)
        self.assertAlmostEqual(float(powers["downlink_desired_mean"]),.25,places=5)
        self.assertAlmostEqual(float(powers["downlink_interference_mean"]),.75,places=5)


class HybridReportTests(unittest.TestCase):
    def test_formal_interpretation_retains_negative_values_and_both_engineering_rules(self):
        summary,paired=synthetic_formal_tables(SEEDS[0]);original=deepcopy((summary,paired))
        result=report.interpret(summary,paired,SEEDS[0])
        self.assertEqual(set(result["engineering_criteria"]),set(report.PRIMARY))
        self.assertFalse(result["training_seeds_pooled"])
        self.assertEqual((summary,paired),original)
        self.assertLess(result["comparisons"]["H32_minus_Fnew"]["per_gamma"][-1]["delta_mean"],0)
        row=next(r for r in paired if r["contrast_id"]=="H16_minus_Fnew"and r["estimand"]=="low_gamma_mean")
        row["coefficients"]={"hybrid_f16_a96":-1,"fdma_control":1}
        with self.assertRaises(ValueError):report.interpret(summary,paired,SEEDS[0])

    def test_report_rejects_missing_points_pooling_fakeNA_and_uncertainty(self):
        for issue in ("missing","pool","fakeNA","CI","wrongn"):
            summary,paired=synthetic_formal_tables(SEEDS[0])
            if issue=="missing":summary.pop()
            elif issue=="pool":paired[0]["training_seeds_pooled"]=True
            elif issue=="fakeNA":next(r for r in summary if r["method_key"]=="frozen_F")["proxy_mse_active_mean"]=0.
            elif issue=="CI":paired[0]["batch_delta_ci95_low"]+=1.
            else:paired[0]["num_samples"]=6400
            with self.subTest(issue=issue),self.assertRaises(ValueError):report.interpret(summary,paired,SEEDS[0])

    def test_six_logs_include_stepzero_best_and_last5000_trend_not_convergence(self):
        with tempfile.TemporaryDirectory()as directory:
            _,training,_=synthetic_report_inputs(Path(directory))
            curves=report.training_curves(training)
            self.assertEqual(len(curves["source_files"]),6)
            for seed in SEEDS:
                item=curves["per_seed"][str(seed)]["fdma_control"]
                self.assertEqual(item["best_step"],0);self.assertEqual(len(item["points"]),41)
                self.assertEqual(item["last_5000_updates"]["num_points"],11)
                self.assertEqual(len(item["last_five_validation_points"]),5)
                self.assertFalse(item["convergence_established"])
            changed=deepcopy(training);changed[str(SEEDS[0])]["fdma_control"]["best_step"]=500
            with self.assertRaises(ValueError):report.training_curves(changed)
            changed=deepcopy(training);changed[str(SEEDS[0])]["fdma_control"]["metrics_sha256"]="0"*64
            with self.assertRaises(ValueError):report.training_curves(changed)

    def test_small_report_roundtrip_no_plot_weights_or_raw_export(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);evaluations,training,sources=synthetic_report_inputs(root)
            original={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for record in training.values()for item in record.values()for path in [item["metrics_path"]]}
            with mock.patch.object(report,"_figures",side_effect=AssertionError("Local plots forbidden")):
                result=report.organize_results(root,evaluations,training,sources,make_plots=False)
            output=Path(result["export_directory"])
            self.assertEqual(len(result["per_seed"]),2)
            self.assertTrue(all(Path(path).suffix in (".csv",".json",".md")for path in result["files"]))
            self.assertFalse(any(output.glob("*.gz")));self.assertFalse(any(output.glob("*.pth")))
            self.assertIn("不证明收敛",(output/"results_draft_zh.md").read_text(encoding="utf-8"))
            self.assertIn("NA",(output/"results_draft_en.md").read_text(encoding="utf-8"))
            for seed in SEEDS:
                name=f"summary_{evaluation.TAG}.csv"
                self.assertEqual((Path(evaluations[str(seed)])/name).read_bytes(),(output/f"seed{seed}_{name}").read_bytes())
            for path,digest in original.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest)

    def test_report_rejects_prior_drift_crossseed_input_mismatch_and_wrong_pinned_source(self):
        for issue in ("prior","inputs","source","capacity"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);evaluations,training,sources=synthetic_report_inputs(root)
                path=Path(evaluations[str(SEEDS[1])])/"manifest.json"
                manifest=json.loads(path.read_text())
                if issue=="prior":manifest["methods"]["fdma_control"]["prior_update_count"]=500
                elif issue=="inputs":manifest["shared_batch_inputs"][0]["H_dl_sha256"]="f"*64
                elif issue=="source":sources[str(SEEDS[1])]["best_sha256"]="0"*64
                else:
                    parameters=Path(evaluations[str(SEEDS[1])])/f"parameters_{evaluation.TAG}.csv"
                    with parameters.open(newline="")as stream:values=list(csv.DictReader(stream))
                    values[0]["bs_parameters"]="111";values[0]["full_parameters"]="131"
                    _synthetic_csv(parameters,values);manifest["output_sha256"]=evaluation.artifact_hashes(parameters.parent)
                path.write_text(json.dumps(manifest),encoding="utf-8")
                with self.assertRaises(ValueError):report.organize_results(root,evaluations,training,sources,make_plots=False)
                self.assertFalse(list(root.glob("export_*")))


class ControllerContractTests(unittest.TestCase):
    def test_planning_family_is_stdlib_only_no_weights_or_outputs(self):
        for module,arguments in (
            ('preflight_grouped_hybrid_anchor',['--dry-run']),
            ('evaluate_grouped_hybrid_anchor',['--seed','20261030','--dry-run']),
            ('run_grouped_hybrid_anchor_batch',['--dry-run'])):
            with self.subTest(module=module):
                assert_lightweight_cli(self,module,[arguments])

    def test_controller_includes_500_in_budget_and_does_not_retrain_completed_arm(self):
        from unittest.mock import patch
        import run_grouped_hybrid_anchor_batch as controller
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=SimpleNamespace(SAVE_PATH=str(root/'weights/latest.file'),BEST_PATH=str(root/'weights/best.file'),
                SAVE_DIR=str(root/'weights'),LOG_DIR=str(root/'logs'),STATUS_PATH=str(root/'logs/status.json'),
                CONFIG_PATH=str(root/'logs/config.json'),METRICS_PATH=str(root/'logs/metrics.jsonl'))
            calls=[]
            def child(folder,label,command):
                calls.append(command)
                Path(cfg.SAVE_DIR).mkdir(exist_ok=True);Path(cfg.LOG_DIR).mkdir(exist_ok=True)
                Path(cfg.SAVE_PATH).write_text('synthetic stage marker, not weights',encoding='utf-8')
                Path(cfg.STATUS_PATH).write_text(json.dumps({'state':'paused' if '--stop-after' in command else 'completed'}),encoding='utf-8')
                return str(root/(label+'.log'))
            with patch.object(controller,'anchor_config',return_value=cfg),patch.object(controller,'_child',side_effect=child),\
                    patch.object(controller,'stage_child',side_effect=lambda *a:dict(report={'completed_updates':20000},_evidence=[])):
                result=controller.train_one(root,ARMS[0],SEEDS[0])
                self.assertEqual(len(calls),2)
                self.assertEqual(calls[0][-2:],['--stop-after','500'])
                self.assertEqual(calls[1][-2:],['--resume','--train'])
                self.assertIn(Path(cfg.LOG_DIR)/'initialization.json',result['_evidence'])
                controller.train_one(root,ARMS[0],SEEDS[0]);self.assertEqual(len(calls),2)
                Path(cfg.SAVE_PATH).unlink()
                with self.assertRaises(ValueError):controller.train_one(root,ARMS[0],SEEDS[0])
                self.assertEqual(len(calls),2)

    def test_small_package_excludes_weights_raw_logs_and_completed_evidence_is_rechecked(self):
        from unittest.mock import patch
        import zipfile
        import run_grouped_hybrid_anchor_batch as controller
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'batch';folder.mkdir()
            for name in ('checkpoint.pth','samples.csv.gz','terminal.log','allowed.csv'):
                (folder/name).write_text('synthetic fixture',encoding='utf-8')
            summary=dict(status='completed',stages={},results={})
            with patch.object(controller,'BATCH_ROOT',root):controller.package_small(folder,summary)
            with zipfile.ZipFile(summary['result_package']) as archive:
                self.assertEqual(set(archive.namelist()),{'batch_summary.json','allowed.csv'})
            evidence=folder/'allowed.csv';count=[]
            def action():count.append(1);return dict(value=1,_evidence=[evidence])
            controller._run_stage(summary,folder,'new_stage',action)
            controller._run_stage(summary,folder,'new_stage',action)
            self.assertEqual(count,[1])
            evidence.write_text('changed fixture',encoding='utf-8')
            with self.assertRaises(RuntimeError):controller._run_stage(summary,folder,'new_stage',action)
            self.assertEqual(count,[1])

    def test_progress_retains_step_zero_best_without_reading_weights(self):
        import run_grouped_hybrid_anchor_batch as controller
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=SimpleNamespace(EXP_NAME='fixture',STATUS_PATH=str(root/'status.json'),METRICS_PATH=str(root/'metrics.jsonl'))
            Path(cfg.STATUS_PATH).write_text(json.dumps(dict(experiment='fixture',state='paused',step=500)),encoding='utf-8')
            values=[dict(step=0,selection_metric_name='mean_all',selection_metric=9.),
                dict(step=500,selection_metric_name='mean_all',selection_metric=8.)]
            Path(cfg.METRICS_PATH).write_text('\n'.join(map(json.dumps,values)),encoding='utf-8')
            report=controller.training_progress(ARMS[0],SEEDS[0],cfg=cfg)
            self.assertEqual((report['best_step'],report['latest_step']),(0,500))
            self.assertFalse(report['weights_read']);self.assertIsNotNone(report['last_record_utc'])


if __name__=='__main__':unittest.main()
