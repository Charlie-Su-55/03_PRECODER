"""Synthetic metadata/statistics and tiny dual-optimizer checks, never a full codec."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from configs import twc_v1_k8_physical_split_config as cfg
import train_twc_v1_k8_physical_split as trainer
import evaluate_twc_v1_k8_physical_split as evaluation
import run_twc_v1_k8_physical_split as controller

DEPENDENCIES=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None
if DEPENDENCIES:
    import torch
    import numpy as np


def acceptance_fixture():
    record=dict(format=cfg.SOURCE_ACCEPTANCE_FORMAT,status='passed',sources={})
    for method in cfg.SOURCE_METHODS:
        identity=cfg.planned_source_identity(method);identity['sha256']='a'*64
        record['sources'][method]=dict(identity=identity,verification=dict(passed=True),frozen_files={identity['path']:'a'*64})
    record['record_id']=cfg.json_sha256(record)
    return record


def metadata_fixture(arm='g4_split',step=108):
    acceptance=acceptance_fixture()
    value=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=step,phase=cfg.phase(step),update_counts=cfg.update_counts(step),
        source_codec_updates=34192,source_bs_updates=6048,source_data_address=40240,data_global_step=40240+step,ue_frozen=True,
        source=cfg.source_identity(arm,acceptance),source_acceptance=cfg.acceptance_identity(acceptance),
        recipe=cfg.recipe(arm,acceptance),config_fingerprint=cfg.fingerprint(arm,acceptance),codec_model={'schema_only':True},
        head_optimizer={'schema_only':True},bs_optimizer={'schema_only':True},head_scheduler={'last_epoch':step},
        bs_scheduler={'last_epoch':max(step-108,0)},rng={k:'schema only' for k in ('python','numpy','torch_cpu','torch_cuda')},
        ce_identity={'schema_only':True},data_protocol={'schema_only':True},code_sha256={'new.py':'b'*64})
    return value


class ProtocolTests(unittest.TestCase):
    def test_budgets_data_and_sources_are_separate_and_unknown_shas_not_guessed(self):
        for arm in cfg.ARMS:
            self.assertEqual(cfg.update_counts(108),dict(data_iterations=108,head_updates=108,bs_updates=0,ue_updates=0,ce_updates=0))
            self.assertEqual(cfg.update_counts(109)['bs_updates'],1)
            self.assertEqual(cfg.update_counts(6048)['bs_updates'],5940)
            self.assertEqual((cfg.schedule(arm,1)['global_step'],cfg.schedule(arm,6048)['global_step']),(40241,46288))
            self.assertEqual(sum(row['count'] for row in cfg.schedule_counts(arm)),6048)
            self.assertIsNone(cfg.source_identity(arm)['sha256'])
        self.assertEqual(cfg.source_identity('fdma_control'),cfg.source_identity('fdma_split'))
        self.assertEqual(cfg.source_identity('g4_control'),cfg.source_identity('g4_split'))
        self.assertEqual((cfg.MONITOR_START_INDEX,cfg.ALPHA_CALIBRATION_START_INDEX,cfg.VALIDATION_START_INDEX),(14336,15360,16384))

    def test_json_resume_requires_both_optimizers_and_correct_phase_counters(self):
        record=acceptance_fixture()
        for step in (108,216,6048):
            payload=json.loads(json.dumps(metadata_fixture(step=step)))
            self.assertEqual(cfg.validate_resume_metadata(payload,'g4_split',record),step)
            for label,mutate in (
                ('wrongBS',lambda p:p['update_counts'].update(bs_updates=step)),
                ('oldtotal',lambda p:p.update(source_data_address=34192)),
                ('resethead',lambda p:p['head_scheduler'].update(last_epoch=0)),
                ('resetBS',lambda p:p['bs_scheduler'].update(last_epoch=9)),
                ('missingoptimizer',lambda p:p.pop('bs_optimizer')),
                ('wrongphase',lambda p:p.update(phase='unexpected')),
                ('crossarm',lambda p:p.update(arm='fdma_split'))):
                bad=deepcopy(payload);mutate(bad)
                with self.subTest(issue=label,step=step),self.assertRaises(ValueError):cfg.validate_resume_metadata(bad,'g4_split',record)

    def test_source_lock_rejects_real_path_seed_sha_or_architecture_changes(self):
        accepted=acceptance_fixture();cfg.acceptance_identity(json.loads(json.dumps(accepted)))
        for field,value in (('path','wrong.pt'),('seed',1),('arm','fdma_nmse'),('data_global_step',34192),('sha256',None)):
            bad=deepcopy(accepted);bad['sources']['old_fdma_base2']['identity'][field]=value
            bad['record_id']=cfg.json_sha256({k:v for k,v in bad.items() if k!='record_id'})
            with self.subTest(field=field),self.assertRaises(ValueError):cfg.acceptance_identity(bad)
        bad=deepcopy(accepted);bad['sources']['old_fdma_base2']['identity']['sha256']='c'*64
        with self.assertRaises(ValueError):cfg.acceptance_identity(bad)

    def test_archived_source_validation_does_not_bind_historical_records_to_new_head(self):
        old=trainer.old;oldcfg=old.cfg;arm='fdma_base2'
        historical=dict(format=oldcfg.SOURCE_ACCEPTANCE_FORMAT,status='passed',sources={})
        for method in oldcfg.SOURCE_METHODS:
            identity=oldcfg.planned_source_identity(method);identity['sha256']='d'*64
            historical['sources'][method]=dict(identity=identity,verification={'schema_fixture':True},frozen_files={identity['path']:'d'*64})
        historical['record_id']=oldcfg.json_sha256(historical)
        archived={'oldfile.py':'e'*64};code=dict(commit=cfg.SOURCE_COMMIT,sha256=archived)
        payload=dict(format=oldcfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=6048,new_bs_updates=6048,source_codec_updates=34192,
            data_global_step=40240,ue_frozen=True,recipe=oldcfg.recipe(arm,historical),config_fingerprint=oldcfg.fingerprint(arm,historical),
            source=oldcfg.source_identity(arm,historical),source_acceptance=oldcfg.acceptance_identity(historical),
            codec_model={'schema_only':True},optimizer={'schema_only':True},scheduler={'last_epoch':6048},
            rng={k:'schema only' for k in ('python','numpy','torch_cpu','torch_cuda')},data_protocol={'schema_only':True},
            ce_identity={'schema_only':True},code_sha256=archived,actual_condition_counts=old.expected_counts(arm,6048),
            training_input_history='f'*64,layout_transmit_history='f'*64,monitor_records={str(s):{'schema_only':True} for s in oldcfg.MONITOR_STEPS})
        manifest=dict(format=oldcfg.CHECKPOINT_FORMAT,status='completed',arm=arm,seed=cfg.SEED,step=6048,new_bs_updates=6048,
            source_codec_updates=34192,data_global_step=40240,ue_frozen=True,code=code,recipe=payload['recipe'],
            config_fingerprint=payload['config_fingerprint'],source=payload['source'],source_acceptance=payload['source_acceptance'],
            checkpoint_sha256='a'*64,ce_identity=payload['ce_identity'],actual_condition_counts=payload['actual_condition_counts'],
            training_input_history='f'*64,layout_transmit_history='f'*64)
        initial={k:manifest[k] for k in ('code','source','recipe','source_acceptance','ce_identity')}
        initial.update(new_bs_optimizer=True,migration=dict(strict=True,source_values_preserved=True))
        path=str(oldcfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')
        terminal=dict(path=path,sha256='a'*64,new_bs_updates=6048,source_codec_updates=34192,data_global_step=40240)
        events=[dict(step=s,data_global_step=34192+s,condition=oldcfg.schedule(arm,s)) for s in range(108,6049,108)]
        batch=dict(status='completed',code=code,training={a:dict(checkpoint_path=path,checkpoint_sha256='a'*64,step=6048) for a in oldcfg.ARMS})
        args=[payload,manifest,initial,terminal,events,batch,dict(status='completed'),arm,historical,archived,'a'*64]
        self.assertTrue(trainer.validate_archived_source(*json.loads(json.dumps(args)))['passed'])
        for label,index,mutate in (
            ('unfinished',5,lambda p:p.update(status='running')),
            ('newHEADforced',1,lambda p:p['code'].update(commit='c'*40)),
            ('wrongSHA',3,lambda p:p.update(sha256='b'*64)),
            ('missinglog',4,lambda p:p.pop()),
            ('resetphase',0,lambda p:p.update(new_bs_updates=0))):
            bad=deepcopy(args);mutate(bad[index])
            with self.subTest(issue=label),self.assertRaises(ValueError):trainer.validate_archived_source(*bad)

    def test_immutable_records_output_guard_and_cleanup_not_part_of_git_scope(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory);path=root/'receipt.json'
            with patch.object(cfg,'inside_root',lambda p:cfg.previous.inside_root(p,root=root)):
                trainer.write_once_json(path,{'passed':True});before=path.read_bytes()
                with self.assertRaises(FileExistsError):trainer.write_once_json(path,{'passed':False})
                self.assertEqual(path.read_bytes(),before)
                with self.assertRaises(ValueError):trainer.write_once_json(root.parent/'outside.json',{})
        self.assertFalse(any(p.startswith('cleanup_reports/') for p in trainer.NEW_FILES))

    def test_controller_108_pause_and_resume_are_one_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory);commands=[]
            def child(folder,label,command):
                commands.append(command);target=root/'training/fdma_control';target.mkdir(parents=True,exist_ok=True)
                (target/'checkpoint_latest.pt').write_bytes(b'CLI fixture, not numerical weights')
                return root/(label+'.log')
            with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(controller,'_child',child),patch.object(
                controller,'inspect_training',return_value=dict(step=108,status='paused',update_counts=cfg.update_counts(108))),patch.object(
                controller,'verify_training',return_value=dict(report={'status':'completed'},_evidence=[])):
                controller.run_arm(root,'fdma_control',acceptance_fixture())
            self.assertEqual([command[-2:] for command in commands],[['--stop-after','108'],['--resume','--train']])


def raw_fixture(count=8):
    result=[]
    for method,mode,g,policy,gamma in evaluation.conditions():
        for i in range(count):
            batch=i//4;rate=10.+batch
            if method.endswith('split'):rate+=(-1. if batch==0 else -3.)
            step,globalstep=evaluation.method_steps(method)
            row={key:1. for key in evaluation.METRICS+evaluation.USER_RATES+evaluation.DIAGNOSTICS}
            row.update(arm=method,rzf_mode=mode,group_count=g,energy_policy=policy,gamma_db=gamma,alpha=.03 if mode=='fixed' else .1,
                seed=cfg.SEED,pilot_energy_per_ue=16.,downlink_snr_db=25.,sample_index=i,batch_index=batch,
                channel_fingerprint=str(batch)*64,checkpoint_step=step,data_global_step=globalstep,
                source_codec_updates=34192,source_bs_updates=6048,source_data_address=40240,ue_frozen=True,
                sum_rate_bps_hz=rate,dl_nmse_linear=1.,data_energy_per_ue=48.,frame_energy_per_ue=64.,rzf_power_mean=1.,
                energy_max_abs_error=0.,rzf_power_max_abs_error=0.)
            counts=evaluation.method_updates(method);row['update_counts']=json.dumps(counts,sort_keys=True);row.update(counts)
            for key in evaluation.USER_RATES:row[key]=rate/8.
            result.append(row)
    return result


class EvaluationTests(unittest.TestCase):
    def test_all_conditions_signed_paired_and_batch_intervals(self):
        rows=raw_fixture();summary,indexed=evaluation.summarize_samples(rows,count=8)
        self.assertEqual(len(summary),144)
        paired=evaluation.paired_statistics(rows,count=8);self.assertEqual(len(paired),1600)
        row=next(r for r in paired if r['contrast']=='fdma_split_minus_control_fdma_fixed'
            and r['metric']=='sum_rate_bps_hz' and r['estimand']=='low_0_5_10')
        self.assertEqual(row['sample_mean'],-2.);self.assertGreater(row['batch_se'],row['sample_se'])
        self.assertAlmostEqual(row['batch_ci95_low'],evaluation.statistics_for([-1.,-3.])['ci95_low'])
        for bad in (rows[:-1],rows+[deepcopy(rows[0])]):
            with self.assertRaises(ValueError):evaluation.summarize_samples(bad,count=8)
        bad=deepcopy(rows);bad[8]['dl_nmse_linear']=3.
        with self.assertRaises(ValueError):evaluation.summarize_samples(bad,count=8)

    def test_alpha_is_global_and_tie_breaking_is_prespecified(self):
        rows=[dict(arm=m,group_count=g,energy_policy=p,gamma_db=gamma,alpha=a,num_samples=64,sum_rate_bps_hz_mean=1.)
            for m,g,p,gamma in evaluation.base_conditions() for a in cfg.ALPHAS]
        selected=evaluation.select_global_alphas(rows)
        self.assertEqual(set(selected['selected_alphas'].values()),{.0001})
        with self.assertRaises(ValueError):evaluation.select_global_alphas(rows[:-1])

    def test_summary_contains_curves_core_ci_and_failure_tail_within_bound(self):
        rows=raw_fixture();summary,_=evaluation.summarize_samples(rows,count=8)
        diagnostics=[dict(arm=arm,energy_policy=policy,metric=metric,estimand='all_six',sample_mean=.01)
            for arm in cfg.ARMS for policy in (('fdma',) if cfg.is_fdma(arm) else ('hybrid_1_3','hybrid_1_1'))
            for metric in ('physical_soft_codeword_nmse_linear_all','physical_head_codeword_nmse_linear_all')]
        result=dict(summary=summary,paired=evaluation.paired_statistics(rows,count=8),receiver_diagnostics=diagnostics)
        state=dict(status='failed',error='example failure',updated='now',result_package='exact.zip',summary_path='summary.json')
        lines=controller.summary_lines(state,{},result,None,['traceback line','line2','line3','line4','final failure detail'])
        text='\n'.join(lines)
        self.assertLessEqual(len(lines),40);self.assertLessEqual(len(text.encode()),8000)
        for method in cfg.METHODS:self.assertIn(method+'/',text)
        self.assertIn('g4_split_minus_control_hybrid_1_3_fixed',text)
        self.assertIn('final failure detail',text);self.assertIn('exact.zip',text)

    def test_stdlib_dry_runs_do_not_load_dependencies_or_create_result_root(self):
        before=cfg.OUTPUT_ROOT.exists()
        for role in ('train','evaluate','run'):
            source=("import sys,runpy\nsys.argv=['x','--dry-run']\ntry:\n"
                " runpy.run_path('"+role+"_twc_v1_k8_physical_split.py',run_name='__main__')\n"
                "except SystemExit as e:\n assert e.code in (None,0)\n"
                "assert not any(m in sys.modules for m in ('torch','numpy','sionna'))\n")
            result=subprocess.run([sys.executable,'-B','-S','-c',source],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(cfg.OUTPUT_ROOT.exists(),before)


@unittest.skipUnless(DEPENDENCIES,'Real PyTorch/NumPy absent; tiny optimizer serialization check not executed')
class DualOptimizerTests(unittest.TestCase):
    def test_actual_108_to_109_restore_has_head108_bs0_without_reinitializing_weights(self):
        from models.twc_v1_k8_physical_split import make_physical_modules,physical_split_values
        class Codec(torch.nn.Module):
            def __init__(self,variant):
                super().__init__();self.encoder=torch.nn.Linear(2,2);self.decoder=torch.nn.Linear(2,2)
                self.joint_decoder=torch.nn.Module();self.joint_decoder.base_decoder=self.decoder
                self.joint_decoder.variant=variant
                self.joint_decoder.physical_head,self.joint_decoder.physical_bridge=make_physical_modules(2)
        def chain_for(variant):
            model=torch.nn.Module();model.codec_model=Codec(variant);model.ce_model=torch.nn.Linear(1,1)
            return model
        def step_once(chain,opts,sched,step):
            trainer.configure_phase(chain,step);model=chain.codec_model
            x=torch.tensor([[[[.2,.5],[.7,-.1]]],[[[.1,-.4],[.3,.2]]]])
            context=model.decoder(model.encoder(x));z=torch.complex(context[...,0],context[...,1])
            soft=torch.complex(x[...,0],x[...,1])
            values=physical_split_values(model.joint_decoder.physical_head,model.joint_decoder.physical_bridge,
                soft,context,z,torch.ones_like(soft.real),torch.ones_like(soft.real,dtype=torch.bool),
                lambda s:dict(correction=torch.full_like(s,.1+.1j),residual_energy=torch.ones(s.shape[0])),
                bridge_enabled=model.joint_decoder.variant=='split')
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            (values['x_phy']-(soft+.2-.1j)).abs().square().mean().backward()
            if step>108:(values['z_recon']-(soft+.1+.2j)).abs().square().mean().backward()
            opts['head'].step();sched['head'].step()
            if step>108:opts['bs'].step();sched['bs'].step()
        for variant in ('control','split'):
            torch.manual_seed(41);chain=chain_for(variant);arm='fdma_'+variant
            groups=trainer.parameter_groups(chain)
            opts={key:torch.optim.AdamW([p for n,p in values],lr=2e-4,weight_decay=1e-5) for key,values in groups.items()}
            sched={key:torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda _:1.) for key,opt in opts.items()}
            original_bs=deepcopy(chain.codec_model.decoder.state_dict())
            for step in range(1,109):step_once(chain,opts,sched,step)
            for name,value in original_bs.items():self.assertTrue(torch.equal(value,chain.codec_model.decoder.state_dict()[name]))
            payload=metadata_fixture(arm,108);payload.update(codec_model=deepcopy(chain.codec_model.state_dict()),
                rng=trainer.capture_rng(torch,np),optimizer_parameter_names={key:[n for n,p in values] for key,values in groups.items()})
            for key in opts:payload[key+'_optimizer']=deepcopy(opts[key].state_dict());payload[key+'_scheduler']=deepcopy(sched[key].state_dict())
            import io
            buffer=io.BytesIO();torch.save(payload,buffer);buffer.seek(0)
            payload=torch.load(buffer,map_location='cpu',weights_only=False)
            for key in opts:trainer.check_optimizer_tensors(torch,payload,key,groups[key],cfg.update_counts(108)[key+'_updates'])
            restored=chain_for(variant);restored.codec_model.load_state_dict(payload['codec_model'],strict=True)
            restored.ce_model.load_state_dict(chain.ce_model.state_dict(),strict=True)
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path=Path(directory)/'source_acceptance.json';path.write_text(json.dumps(acceptance_fixture()),encoding='utf-8')
                payload.update(source_acceptance_sha256=trainer.sha256(path),
                    actual_condition_counts=trainer.expected_counts(arm,108),monitor_records={},
                    training_input_history='a'*64,layout_transmit_history='b'*64,
                    frozen_state=trainer.frozen_part_digest(torch,restored))
                with patch.object(cfg,'SOURCE_ACCEPTANCE_PATH',path),patch.object(
                        trainer,'source_code_record',return_value={'sha256':payload['code_sha256']}):
                    trainer.validate_runtime_resume(torch,restored,payload,arm,acceptance_fixture(),payload['data_protocol'],payload['ce_identity'])
                    bad=deepcopy(payload);bad['bs_optimizer']['state']=deepcopy(payload['head_optimizer']['state'])
                    with self.assertRaises(ValueError):
                        trainer.validate_runtime_resume(torch,restored,bad,arm,acceptance_fixture(),payload['data_protocol'],payload['ce_identity'])
            new_groups=trainer.parameter_groups(restored)
            new_opts={key:torch.optim.AdamW([p for n,p in values],lr=2e-4,weight_decay=1e-5) for key,values in new_groups.items()}
            new_sched={key:torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda _:1.) for key,opt in new_opts.items()}
            for key in opts:
                new_opts[key].load_state_dict(payload[key+'_optimizer']);new_sched[key].load_state_dict(payload[key+'_scheduler'])
            self.assertEqual((new_sched['head'].last_epoch,new_sched['bs'].last_epoch),(108,0))
            self.assertFalse(new_opts['bs'].state)
            for name,value in payload['codec_model'].items():self.assertTrue(torch.equal(restored.codec_model.state_dict()[name],value))
            step_once(restored,new_opts,new_sched,109)
            post=dict(payload,step=109,optimizer_parameter_names={key:[n for n,p in values] for key,values in new_groups.items()})
            for key in opts:post[key+'_optimizer']=new_opts[key].state_dict();post[key+'_scheduler']=new_sched[key].state_dict()
            for key in opts:trainer.check_optimizer_tensors(torch,post,key,new_groups[key],cfg.update_counts(109)[key+'_updates'])
            self.assertEqual((new_sched['head'].last_epoch,new_sched['bs'].last_epoch),(109,1))
            self.assertTrue(new_opts['bs'].state)

    def test_monitor_restores_real_modes_buffers_rng_on_success_and_exception(self):
        from types import SimpleNamespace
        import random
        import models.twc_v1_k8_physical_split as model_module
        class Chain(torch.nn.Module):
            def __init__(self):
                super().__init__();self.codec_model=torch.nn.Module()
                self.codec_model.encoder=torch.nn.Linear(1,1);self.codec_model.decoder=torch.nn.Linear(1,1)
                self.ce_model=torch.nn.Linear(1,1)
            def metrics(self,out,hdl,**kwargs):
                rate=hdl[:,0].abs()+1.
                return dict(dl_nmse_per_user=rate[:,None].expand(-1,8),sum_rate_bps_hz=rate,
                    desired_signal_power_mean=rate,interference_power_mean=rate,rzf_per_subcarrier_power=torch.ones(4,2))
        chain=Chain().train();chain.ce_model.requires_grad_(False).eval();chain.codec_model.encoder.requires_grad_(False).eval()
        class Data:
            def sample(self,role,index):return dict(H_dl=torch.arange(4.)[:,None],H_ul=torch.ones(4,1),sample_fingerprint='a'*64)
        warmup=SimpleNamespace(standard_noise=lambda n,generator,device:torch.randn(n,2,generator=generator,device=device))
        def run(*args):
            torch.rand(1);np.random.rand();random.random()
            return dict(energy={'actual_data_per_ue':torch.full((4,8),48.)})
        for fail in (False,True):
            before=trainer.capture_rng(torch,np);modes={n:m.training for n,m in chain.named_modules()};state=deepcopy(chain.state_dict())
            def failing(*args):run(*args);raise RuntimeError('injected physical monitor failure')
            with patch.object(trainer,'run_forward',failing if fail else run),patch.object(
                model_module,'physical_diagnostics',return_value={'physical_soft_codeword_nmse_linear_all':torch.ones(4)}):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'injected'):trainer.monitor(torch,np,warmup,chain,Data(),'g4_split',1512)
                else:
                    report=trainer.monitor(torch,np,warmup,chain,Data(),'g4_split',1512)
                    self.assertEqual((len(report['summary']),len(report['all_six'])),(12,2))
            after=trainer.capture_rng(torch,np)
            self.assertEqual(before['python'],after['python'])
            self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
            self.assertEqual(before['numpy'][2:],after['numpy'][2:])
            self.assertTrue(torch.equal(before['torch_cpu'],after['torch_cpu']))
            self.assertEqual(modes,{n:m.training for n,m in chain.named_modules()})
            for n,v in state.items():self.assertTrue(torch.equal(v,chain.state_dict()[n]))


if __name__=='__main__':unittest.main()
