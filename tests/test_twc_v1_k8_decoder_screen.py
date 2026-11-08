"""Small decoder-component/physical algebra checks; no full codec or CDL."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
TORCH_PRESENT=importlib.util.find_spec('torch') is not None
IMPORT_ERROR=None
if TORCH_PRESENT:
    import torch
    try:
        from models import twc_v1_k8_decoder_refinement as refinement
        from models.twc_v1_k8_joint_decoder import _JointLayer,K8JointTransformerSwinReceiver
        from models.twc_v1_k8_neural import _protocol as protocol
        residual=refinement.residual_ops
    except ModuleNotFoundError as error:IMPORT_ERROR=str(error)
else:IMPORT_ERROR='PyTorch absent; small numerical component checks not executed'


@unittest.skipIf(IMPORT_ERROR is not None,IMPORT_ERROR or '')
class PhysicalResidualTests(unittest.TestCase):
    def test_complex_solve_uses_conjugate_and_correct_receive_user_axes(self):
        G=torch.tensor([[1+1j,2-.5j],[.2+2j,-1+.3j],[2-.1j,.5+1j]],dtype=torch.complex128)
        r=torch.tensor([[1-.3j],[.2+.5j],[-.7+.8j]],dtype=torch.complex128)
        ridge=.2
        actual=residual.solve_matched_residual(G,r,ridge)
        expected=torch.linalg.solve(G.conj().T@G+ridge*torch.eye(2,dtype=G.dtype),G.conj().T@r).squeeze(-1)
        wrong=torch.linalg.solve(G.T@G+ridge*torch.eye(2,dtype=G.dtype),G.T@r).squeeze(-1)
        self.assertTrue(torch.allclose(actual,expected,rtol=1e-12,atol=1e-12))
        self.assertGreater(float((actual-wrong).abs().max()),.01)

    def _physical(self):
        resource=torch.tensor([[1,2],[2,3]])
        active=torch.ones(2,2,dtype=torch.bool)
        kind=torch.ones(2,2,dtype=torch.long)
        H=torch.tensor([[[[1+.2j,1+1j,2-.5j,.3+.2j],[.5-.1j,.2+2j,1+.3j,2-.1j]],
                         [[.2+.1j,1-.2j,.5+1j,-1+.3j],[1+.5j,.5+.2j,-.3+.7j,.8-1j]]]],dtype=torch.complex64)
        symbols=torch.tensor([[[2+3j,-1+.5j],[.3-2j,1.2+.8j]]],dtype=torch.complex64)
        prediction=torch.zeros(1,2,4,dtype=torch.complex64)
        for user in range(2):
            for slot in range(2):prediction[:,:,resource[user,slot]]+=H[:,user,:,resource[user,slot]]*symbols[:,user,slot,None]
        return symbols,prediction,H,resource,kind,active

    def test_physical_normalized_scale_no_renormalization_and_pilot_exclusion(self):
        x,Y,H,q,kind,active=self._physical()
        Y[:,:,0]=10000+3000j
        actual=residual.data_received_residual(x,Y,H,q,active,pilot_positions=((0,),))
        self.assertEqual(float(actual['energy'].max()),0.)
        self.assertEqual(float(actual['residual'].abs().max()),0.)
        Y[:,:,2]+=torch.tensor([[.5+.2j,-.3+.1j]])
        first=residual.data_received_residual(x,Y,H,q,active,pilot_positions=((0,),))
        Y[:,:,0]*=7
        second=residual.data_received_residual(x,Y,H,q,active,pilot_positions=((0,),))
        self.assertTrue(torch.equal(first['residual'],second['residual']))
        self.assertTrue(torch.equal(first['energy'],second['energy']))
        bad=q.clone();bad[0,0]=0
        with self.assertRaises(ValueError):residual.data_received_residual(x,Y,H,bad,active,pilot_positions=((0,),))

    def test_matched_residual_uses_original_nominal_power_ce_ridge_and_remains_differentiable(self):
        x,Y,H,q,kind,active=self._physical()
        perturbation=torch.tensor([[[0,.2+.3j,.7-.1j,-.3+.2j],[0,.1-.4j,.3+.8j,.2-.1j]]],dtype=torch.complex64)
        Y=Y+perturbation
        ce=torch.tensor([[.1,.2]])
        actual=residual.regularized_matched_residual(x,Y,H,q,kind,active,noise_variance=.25,
            key_energy_per_ue=0.,shared_energy_per_ue=8.,channel_error_variance=ce,pilot_positions=((0,),))
        expected=torch.zeros_like(x)
        for position in (1,2,3):
            members=[user for user in range(2) if position in q[user].tolist()]
            G=H[:,members,:,position].transpose(-2,-1)
            r=perturbation[:,:,position,None]
            ridge=.25/(8./2)+ce[:,members].sum(1)+1e-8
            value=torch.linalg.solve(G.conj().transpose(-2,-1)@G+ridge[:,None,None]*torch.eye(len(members)),
                G.conj().transpose(-2,-1)@r).squeeze(-1)
            for local,user in enumerate(members):expected[:,user,q[user].tolist().index(position)]=value[:,local]
        self.assertTrue(torch.allclose(actual['correction'],expected,rtol=1e-5,atol=1e-6))
        differentiable=x.detach().clone().requires_grad_(True)
        result=residual.regularized_matched_residual(differentiable,Y,H,q,kind,active,noise_variance=.25,
            key_energy_per_ue=0.,shared_energy_per_ue=8.,channel_error_variance=ce,pilot_positions=((0,),))
        result['correction'].abs().square().sum().backward()
        self.assertTrue(bool(torch.isfinite(differentiable.grad).all()))
        self.assertGreater(float(differentiable.grad.abs().sum()),0.)

    def test_layout_maps_and_local_attention_do_not_treat_fdma_minus_one_as_a_group(self):
        for layout in (protocol.FDMALayout(),protocol.default_hybrid_layout(4)):
            mapped=residual.packed_layout_map(layout)
            original=K8JointTransformerSwinReceiver._slot_map(layout)
            for left,right in zip(mapped[:4],original[:4]):self.assertTrue(torch.equal(left,right))
            self.assertEqual(mapped[4:],original[4:])
        resource,kind,group,active,_,_=residual.packed_layout_map(protocol.FDMALayout())
        self.assertTrue(bool((group[active]==-1).all()))
        self.assertTrue(bool((kind[:,:24]==2).all()))
        users=torch.arange(8)[:,None].expand(-1,164).flatten()
        permitted=residual.local_attention_allowed(users,resource.flatten(),active.flatten(),start=0,end=24)
        self.assertEqual(int(permitted[0].sum()),24)
        self.assertFalse(bool(permitted[:,:][...,164:].any()))
        resource,kind,_,active,_,_=residual.packed_layout_map(protocol.default_hybrid_layout(4))
        allowed=residual.local_attention_allowed(users,resource.flatten(),active.flatten(),start=4,end=5)[0]
        expected=(users==0)|((resource.flatten()==resource[0,4])&active.flatten())
        expected&=active.flatten()
        self.assertTrue(torch.equal(allowed,expected))
        self.assertEqual(int(((users!=0)&allowed).sum()),1)
        key_allowed=residual.local_attention_allowed(users,resource.flatten(),active.flatten(),start=0,end=1)[0]
        self.assertEqual(int(((users!=0)&key_allowed).sum()),0)


@unittest.skipIf(IMPORT_ERROR is not None,IMPORT_ERROR or '')
class NewModuleAndMigrationTests(unittest.TestCase):
    def _layer(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            return _JointLayer(width=8,query_chunk=3,activation_checkpoint=False)

    def test_added_deep_layer_is_identity_and_zero_outputs_receive_gradients(self):
        layer=refinement.initialize_identity_joint_layer(self._layer())
        generator=torch.Generator().manual_seed(21)
        x=torch.randn(1,5,8,generator=generator,requires_grad=True)
        memory=torch.randn(1,3,8,generator=generator)
        active=torch.ones(1,5,dtype=torch.bool)
        result=layer(x,memory,active,None,None)
        self.assertTrue(torch.equal(result,x))
        result.square().sum().backward()
        for output in (layer.self_attention.out,layer.cross_attention.out,layer.mlp[-1]):
            self.assertTrue(bool(torch.isfinite(output.weight.grad).all()))
            self.assertGreater(float(output.weight.grad.abs().sum()),0.)

    def test_actual_dc_modules_add514_parameters_and_keep_zero_head_in_graph(self):
        projection,head=refinement.make_dc_residual_modules(128)
        self.assertIsNone(projection.bias)
        self.assertEqual(sum(p.numel() for module in (projection,head) for p in module.parameters()),514)
        self.assertEqual(float(head.weight.abs().sum()+head.bias.abs().sum()),0.)
        context=torch.arange(256,dtype=torch.float32).reshape(1,2,128)/100
        direction=projection(torch.tensor([[[.1,.2],[.3,-.4]]]))
        x1=torch.tensor([[1+.3j,-.2+2j]],dtype=torch.complex64)
        output=head(context+direction)
        x2=x1+torch.complex(output[...,0],output[...,1])
        self.assertTrue(torch.equal(x2,x1))
        x2.abs().square().sum().backward()
        self.assertTrue(bool(torch.isfinite(head.weight.grad).all()))
        self.assertGreater(float(head.weight.grad.abs().sum()),0.)
        self.assertIsNotNone(projection.weight.grad)
        self.assertEqual(float(projection.weight.grad.abs().sum()),0.)

    def _codec(self,variant):
        class TinyReceiver(torch.nn.Module):
            def __init__(inner,base):
                super().__init__();inner.base_decoder=base;inner.variant=variant
                inner.joint_layers=torch.nn.ModuleList([self._layer() for _ in range(4 if variant=='deep4' else 2)])
                if variant=='deep4':
                    for layer in inner.joint_layers[2:]:refinement.initialize_identity_joint_layer(layer)
                if variant=='dc2':inner.residual_projection,inner.second_residual_head=refinement.make_dc_residual_modules(128)
        class TinyCodec(torch.nn.Module):
            def __init__(inner):
                super().__init__();inner.encoder=torch.nn.Linear(2,2)
                inner.decoder=torch.nn.Linear(3,2)
                inner.decoder.register_buffer('persistent_marker',torch.tensor([7.]))
                inner.joint_decoder=TinyReceiver(inner.decoder)
        return TinyCodec()

    def test_strict_migration_copies_old_aliases_and_adds_only_declared_parameters(self):
        source=self._codec('base2');state=deepcopy(source.state_dict())
        for variant in refinement.VARIANTS:
            target=self._codec(variant)
            self.assertIs(target.decoder,target.joint_decoder.base_decoder)
            record=refinement.migrate_source_state(target,state)
            self.assertTrue(record['strict']);self.assertTrue(record['source_values_preserved'])
            self.assertEqual(set(record['added_keys']),set(refinement.added_state_keys(variant)))
            for key,value in state.items():self.assertTrue(torch.equal(target.state_dict()[key],value),key)
            if variant=='dc2':self.assertEqual(record['new_parameters'],514)
            if variant=='deep4':self.assertEqual(len(record['added_keys']),56)
        for key,value in source.state_dict().items():self.assertTrue(torch.equal(state[key],value))

    def test_migration_rejects_bad_old_state_and_normal_resume_does_not_reset_new_head(self):
        source=self._codec('base2');state=deepcopy(source.state_dict())
        target=self._codec('dc2')
        for issue in ('missing','extra','shape','dtype','nonfinite','conflicting alias'):
            bad=deepcopy(state)
            if issue=='missing':bad.pop('encoder.weight')
            elif issue=='extra':bad['unknown.weight']=torch.ones(1)
            elif issue=='shape':bad['encoder.weight']=torch.zeros(8)
            elif issue=='dtype':bad['encoder.weight']=bad['encoder.weight'].double()
            elif issue=='nonfinite':bad['encoder.weight'].fill_(float('nan'))
            else:bad['joint_decoder.base_decoder.weight']=bad['joint_decoder.base_decoder.weight']+1
            with self.subTest(issue=issue),self.assertRaises(ValueError):refinement.migrate_source_state(target,bad)
        refinement.migrate_source_state(target,state)
        with torch.no_grad():target.joint_decoder.second_residual_head.weight.fill_(.25)
        checkpoint=deepcopy(target.state_dict());restored=self._codec('dc2')
        restored.load_state_dict(checkpoint,strict=True)
        self.assertTrue(torch.equal(restored.joint_decoder.second_residual_head.weight,target.joint_decoder.second_residual_head.weight))
        with self.assertRaises(ValueError):refinement.migrate_source_state(restored,checkpoint)
        missing=deepcopy(checkpoint);missing.pop('joint_decoder.second_residual_head.bias')
        with self.assertRaises(RuntimeError):restored.load_state_dict(missing,strict=True)

    def test_local_attention_padding_is_finite_and_cannot_update_inactive_queries(self):
        layer=self._layer()
        users=torch.tensor([0,0,1,2])
        resources=torch.tensor([5,6,5,0])
        active=torch.tensor([True,True,True,False])
        def bias(start,end):
            allowed=residual.local_attention_allowed(users,resources,active,start=start,end=end)
            return torch.zeros(1,4,end-start,4).masked_fill(~allowed[None,None],-torch.inf)
        x=torch.arange(32,dtype=torch.float32).reshape(1,4,8)/10
        x=x*active[None,:,None]
        output=layer(x,torch.ones(1,3,8),active[None],bias,None)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(float(output[:,3].abs().max()),0.)


@unittest.skipIf(IMPORT_ERROR is not None,IMPORT_ERROR or '')
class OfflineMetricAndFreezeTests(unittest.TestCase):
    def _output(self,layout):
        resource,kind,_,active,layout_kind,_=residual.packed_layout_map(layout)
        truth=torch.zeros(1,8,164,dtype=torch.complex64)
        if layout_kind=='fdma':
            data=torch.full((1,8,24),2.**.5,dtype=torch.complex64)
            truth[...,:24]=data
            tx=dict(data_symbols_tx=data,raw_encoder_symbols=torch.full_like(data,99.))
            soft={'data_symbols_hat':.5*data}
        else:
            key=torch.full((1,8,4),3.**.5,dtype=torch.complex64)
            shared=torch.full((1,8,layout.shared_dimension),.9**.5,dtype=torch.complex64)
            truth[...,:4]=key;truth[...,4:4+layout.shared_dimension]=shared
            tx=dict(key_symbols_tx=key,shared_symbols_tx=shared,raw_encoder_symbols=torch.full_like(shared,99.))
            soft=dict(key_symbols_hat=.5*key,shared_symbols_hat=.5*shared)
        H=torch.ones(1,8,2,256,dtype=torch.complex64)
        Y=torch.zeros(1,2,256,dtype=torch.complex64)
        for user in range(8):
            for slot in torch.nonzero(active[user],as_tuple=False).flatten().tolist():
                q=int(resource[user,slot]);Y[:,:,q]+=H[:,user,:,q]*truth[:,user,slot,None]
        return dict(layout=layout,tx=tx,soft_symbols=soft,corrected_symbols=.75*truth,
            receiver_execution={},Y_rx=Y,H_ul_est=H),truth

    def test_offline_metrics_use_actual_normalized_tx_and_source_fallback_not_raw_heads(self):
        for layout in (protocol.FDMALayout(),protocol.default_hybrid_layout(4)):
            output,truth=self._output(layout)
            metrics=residual.offline_decoder_diagnostics(output)
            types=('all','fdma') if hasattr(layout,'data_positions') else ('all','key','shared')
            for name in types:
                self.assertTrue(torch.allclose(metrics['stage_soft_codeword_nmse_'+name],torch.tensor([.25]),atol=1e-6))
                self.assertTrue(torch.allclose(metrics['stage_round1_codeword_nmse_'+name],torch.tensor([.0625]),atol=1e-6))
                self.assertTrue(torch.equal(metrics['stage_round1_codeword_nmse_'+name],metrics['stage_final_codeword_nmse_'+name]))
            self.assertFalse(any(key.endswith('_key') or key.endswith('_shared') for key in metrics) if hasattr(layout,'data_positions') else False)
            output['receiver_execution']['stages']=dict(soft=.5*truth,x1=.75*truth,x2=truth,
                received_residual_energy_x1=metrics['received_residual_energy_x1'],
                received_residual_energy_final=torch.zeros(1))
            final=residual.offline_decoder_diagnostics(output)
            self.assertEqual(float(final['stage_final_codeword_nmse_all']),0.)
            self.assertEqual(float(final['received_residual_energy_final']),0.)

    def test_real_freeze_helper_excludes_full_encoder_condition_and_ce_without_disabling_bs_graph(self):
        import train_twc_v1_k8_decoder_screen as trainer
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.projection=torch.nn.Linear(2,2);self.group_embedding=torch.nn.Embedding(3,2)
                self.register_buffer('running_pwr',torch.tensor([1.]))
            def forward(self,x):return self.projection(x)+self.group_embedding(torch.tensor(1))
        class Codec(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=Encoder();self.decoder=torch.nn.Linear(2,2)
                self.joint_decoder=torch.nn.Module();self.joint_decoder.base_decoder=self.decoder
                self.joint_decoder.residual_head=torch.nn.Linear(2,2)
        chain=torch.nn.Module();chain.codec_model=Codec();chain.ce_model=torch.nn.Linear(1,1)
        chain.train()
        before=deepcopy(chain.state_dict())
        parameters=trainer.freeze_training_parts(chain)
        self.assertFalse(chain.codec_model.encoder.training);self.assertFalse(chain.ce_model.training)
        self.assertTrue(chain.codec_model.decoder.training)
        self.assertTrue(all(not name.startswith('encoder.') for name,_ in parameters))
        self.assertTrue(all(not p.requires_grad for p in chain.codec_model.encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in chain.ce_model.parameters()))
        x=chain.codec_model.encoder(torch.tensor([[.4,.7],[1.,-.2]]))
        result=chain.codec_model.joint_decoder.residual_head(chain.codec_model.decoder(x))
        result.square().sum().backward()
        for _,parameter in parameters:
            self.assertIsNotNone(parameter.grad);self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertTrue(all(p.grad is None for p in chain.codec_model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in chain.ce_model.parameters()))
        for name,value in before.items():self.assertTrue(torch.equal(chain.state_dict()[name],value),name)


def acceptance_fixture():
    from configs import twc_v1_k8_decoder_screen_config as cfg
    record=dict(format=cfg.SOURCE_ACCEPTANCE_FORMAT,status='passed',sources={})
    for method in cfg.SOURCE_METHODS:
        identity=cfg.planned_source_identity(method);identity['sha256']='a'*64
        record['sources'][method]=dict(identity=identity,verification={'schema_fixture':True},
            frozen_files={identity['path']:identity['sha256']})
    record['record_id']=cfg.json_sha256(record)
    return record


class SourceAndResumeMetadataTests(unittest.TestCase):
    def test_unknown_source_sha_is_not_guessed_and_lock_rejects_changes(self):
        from configs import twc_v1_k8_decoder_screen_config as cfg
        import json
        for method in cfg.METHODS:self.assertIsNone(cfg.source_identity(method)['sha256'])
        accepted=acceptance_fixture()
        self.assertEqual(cfg.acceptance_identity(json.loads(json.dumps(accepted)))['source_sha256']['source_fdma'],'a'*64)
        for key,value in (('sha256','not-a-digest'),('seed',0),('path','other.pt'),
                          ('arm','fdma_joint'),('source_implementation_commit','b'*40)):
            changed=deepcopy(accepted);changed['sources']['source_fdma']['identity'][key]=value
            changed['record_id']=cfg.json_sha256({k:v for k,v in changed.items() if k!='record_id'})
            with self.subTest(field=key),self.assertRaises(ValueError):cfg.acceptance_identity(changed)
        changed=deepcopy(accepted);changed['sources']['source_fdma']['identity']['sha256']='b'*64
        with self.assertRaises(ValueError):cfg.acceptance_identity(changed)

    def test_bs_only_budget_addresses_conditions_and_resume_are_strict(self):
        from configs import twc_v1_k8_decoder_screen_config as cfg
        import json
        acceptance=acceptance_fixture()
        for arm in cfg.ARMS:
            recipe=cfg.recipe(arm,acceptance)
            self.assertEqual((cfg.schedule(arm,1)['global_step'],cfg.schedule(arm,6048)['global_step']),(34193,40240))
            self.assertEqual(sum(row['count'] for row in cfg.schedule_counts(arm)),6048)
            self.assertTrue(recipe['UE_frozen']);self.assertEqual(recipe['UE_new_updates'],0)
            self.assertFalse(recipe['rate_loss_enabled'] if 'rate_loss_enabled' in recipe else recipe['sum_rate_loss_weight']!=0)
            payload=dict(format=cfg.CHECKPOINT_FORMAT,arm=arm,seed=cfg.SEED,step=108,new_bs_updates=108,
                source_codec_updates=34192,data_global_step=34300,ue_frozen=True,recipe=recipe,
                config_fingerprint=cfg.fingerprint(arm,acceptance),source=cfg.source_identity(arm,acceptance),
                source_acceptance=cfg.acceptance_identity(acceptance),codec_model={'schema_only':True},
                optimizer={'schema_only':True},scheduler={'last_epoch':108},
                rng={key:'schema only' for key in ('python','numpy','torch_cpu','torch_cuda')},
                data_protocol={'schema_only':True},ce_identity={'schema_only':True},code_sha256={'new.py':'c'*64})
            payload=json.loads(json.dumps(payload))
            self.assertEqual(cfg.validate_resume_metadata(payload,arm,acceptance,code_sha256=payload['code_sha256']),108)
            for field,value in (('new_bs_updates',0),('data_global_step',34192),('ue_frozen',False),
                                 ('source_codec_updates',22096),('arm','wrong')):
                changed=deepcopy(payload);changed[field]=value
                with self.subTest(arm=arm,field=field),self.assertRaises(ValueError):cfg.validate_resume_metadata(changed,arm,acceptance)
            changed=deepcopy(payload);changed['scheduler']['last_epoch']=0
            with self.assertRaises(ValueError):cfg.validate_resume_metadata(changed,arm,acceptance)
        self.assertEqual(cfg.MONITOR_STEPS,(1512,3024,4536,6048))
        self.assertEqual(cfg.MONITOR_ALPHA,.03)

    def test_actual_source_acceptance_validator_survives_json_and_rejects_lineage_changes(self):
        import json
        import train_twc_v1_k8_decoder_screen as training
        old=training.previous.cfg;arm='fdma_nmse'
        beta=dict(format=old.CALIBRATION_FORMAT,status='passed',beta_target=1.,
            state_freeze={m:{'codec_model_sha256':'d'*64} for m in ('source_fdma','source_g4')})
        beta['record_id']=old.json_sha256(beta)
        archived={'old.py':'e'*64};code=dict(commit=training.cfg.SOURCE_COMMIT,sha256=archived)
        source=old.source_identity(arm);recipe=old.recipe(arm,beta);counts=training.previous.expected_counts(arm,12096)
        payload=dict(format=old.CHECKPOINT_FORMAT,arm=arm,seed=old.SEED,step=12096,cumulative_updates=34192,
            recipe=recipe,config_fingerprint=old.fingerprint(arm,beta),source=source,calibration=old.calibration_identity(beta),
            calibration_record=beta,codec_model={'schema only':True},optimizer={'schema only':True},
            scheduler={'last_epoch':12096},rng={k:'schema only' for k in ('python','numpy','torch_cpu','torch_cuda')},
            data_protocol={'schema only':True},ce_identity={'schema only':True},code_sha256=archived,
            actual_condition_counts=counts,training_input_history='f'*64,training_input_history_steps=12096)
        manifest=dict(format=old.CHECKPOINT_FORMAT,arm=arm,seed=old.SEED,step=12096,cumulative_updates=34192,status='completed',
            source=source,recipe=recipe,config_fingerprint=old.fingerprint(arm,beta),code=code,
            calibration=old.calibration_identity(beta),checkpoint_sha256='a'*64,training_alpha=.03,
            actual_condition_counts=counts,training_input_history='f'*64,ce_identity=payload['ce_identity'])
        initial=dict(source=source,recipe=recipe,code=code,calibration=old.calibration_identity(beta),
            ce_identity=payload['ce_identity'],migration=False,new_optimizer=True,strict_loading=True,initialized_state_sha256='d'*64)
        path=str(old.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt')
        terminal=dict(local_step=12096,cumulative_updates=34192,sha256='a'*64,path=path)
        events=[dict(step=s,cumulative_updates=22096+s,condition=old.schedule(arm,s),standard_noise_seed=old.training_noise_seed(s))
            for s in range(108,12097,108)]
        batch=dict(status='completed',code=code,training={a:dict(checkpoint_path=path,checkpoint_sha256='a'*64,step=12096,cumulative_updates=34192) for a in old.ARMS})
        stage=dict(status='completed')
        arguments=[payload,manifest,initial,terminal,events,batch,stage,arm,beta,archived,'a'*64]
        roundtrip=json.loads(json.dumps(arguments))
        self.assertTrue(training.validate_source_records(*roundtrip)['passed'])
        for label,index,mutate in (
            ('source seed',0,lambda r:r.update(seed=0)),
            ('source SNR',0,lambda r:r['recipe']['conditions'].update(gamma=[0])),
            ('source checkpoint path',3,lambda r:r.update(path=path+'.wrong')),
            ('source weight SHA',1,lambda r:r.update(checkpoint_sha256='b'*64)),
            ('source code',2,lambda r:r['code'].update(commit='c'*40)),
            ('source unfinished batch',5,lambda r:r.update(status='running')),
            ('source missing prior count',4,lambda r:r.pop())):
            changed=deepcopy(arguments);mutate(changed[index])
            with self.subTest(issue=label),self.assertRaises(ValueError):training.validate_source_records(*changed)

    def test_output_namespace_and_immutable_small_records(self):
        import tempfile
        import train_twc_v1_k8_decoder_screen as training
        from configs import twc_v1_k8_decoder_screen_config as cfg
        self.assertNotEqual(cfg.OUTPUT_ROOT,training.previous.cfg.OUTPUT_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(cfg,'inside_root',lambda p: cfg.previous.inside_root(p,root=root)):
                path=root/'locked.json';training.write_once_json(path,{'x':1})
                before=path.read_bytes()
                with self.assertRaises(FileExistsError):training.write_once_json(path,{'x':2})
                self.assertEqual(path.read_bytes(),before)
                with self.assertRaises(ValueError):training.write_once_json(root.parent/'outside.json',{})

    def test_completed_stage_requires_real_evidence_and_changed_bytes_are_rejected(self):
        import tempfile
        import run_twc_v1_k8_decoder_screen as controller
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/'report.json';path.write_text('{"passed":true}',encoding='utf-8')
            summary={'stages':{'cuda':{'status':'completed','evidence_sha256':{str(path):controller.sha256(path)}}}}
            controller.verify_completed_stages(summary)
            path.write_text('{"passed":false}',encoding='utf-8')
            with self.assertRaises(ValueError):controller.verify_completed_stages(summary)
        with self.assertRaises(ValueError):controller.verify_completed_stages({'stages':{'cuda':{'status':'completed'}}})

    def test_controller_first108_is_in_budget_and_explicit_resume_continues(self):
        import tempfile
        import run_twc_v1_k8_decoder_screen as controller
        from configs import twc_v1_k8_decoder_screen_config as cfg
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'batch';folder.mkdir()
            commands=[]
            def child(_folder,label,command):
                commands.append(command)
                (root/'training/fdma_base2').mkdir(parents=True,exist_ok=True)
                (root/'training/fdma_base2/checkpoint_latest.pt').write_bytes(b'no numerical weights: CLI fixture')
                return folder/(label+'.log')
            with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(controller,'_child',child),patch.object(
                    controller,'inspect_training',return_value={'step':108,'status':'paused'}),patch.object(
                    controller,'verify_training',return_value={'report':{'status':'completed'},'_evidence':[]}):
                controller.run_arm(folder,'fdma_base2',acceptance_fixture())
            self.assertEqual(commands[0][-2:],['--stop-after','108'])
            self.assertEqual(commands[1][-2:],['--resume','--train'])
            self.assertEqual(len(commands),2)


def raw_statistics_fixture(count=8):
    from configs import twc_v1_k8_decoder_screen_config as cfg
    import evaluate_twc_v1_k8_decoder_screen as ev
    rows=[]
    for method,mode,g,policy,gamma in ev.conditions():
        for i in range(count):
            batch=i//4;base=10.+batch
            delta=0. if method.startswith('source_') or method.endswith('base2') else (
                (-1. if batch==0 else -3.) if method.endswith('deep4') else (1. if batch==0 else 2.))
            rate=base+delta+(.3 if not cfg.is_fdma(method) else 0.)
            step,global_step=ev.method_steps(method)
            row={name:1. for name in ev.METRICS+ev.USER_RATES+ev.DIAGNOSTICS}
            row.update(arm=method,rzf_mode=mode,group_count=g,energy_policy=policy,gamma_db=gamma,
                sample_index=i,batch_index=batch,seed=cfg.SEED,checkpoint_step=step,data_global_step=global_step,
                source_codec_updates=34192,new_bs_updates=0 if method.startswith('source_') else 6048,
                ue_frozen=True,pilot_energy_per_ue=16.,downlink_snr_db=25.,channel_fingerprint=str(batch)*64,
                alpha=.03 if mode=='fixed' else .1,sum_rate_bps_hz=rate,dl_nmse=1.,
                data_energy_per_ue=48.,frame_energy_per_ue=64.,rzf_power_mean=1.,
                energy_max_abs_error=0.,rzf_power_max_abs_error=0.)
            for name in ev.USER_RATES:row[name]=rate/8.
            rows.append(row)
    return rows


class PairedAndSummaryTests(unittest.TestCase):
    def test_signed_sample_and_batch_pairing_and_cross_snr_average(self):
        import evaluate_twc_v1_k8_decoder_screen as ev
        rows=raw_statistics_fixture()
        summary,indexed=ev.summarize_samples(rows,count=8)
        self.assertEqual(len(summary),144)
        paired=ev.paired_statistics(rows,count=8)
        self.assertEqual(len(paired),1920)
        result=next(r for r in paired if r['contrast']=='fdma_deep4_minus_base2_fdma_fixed'
            and r['metric']=='sum_rate_bps_hz' and r['estimand']=='low_0_5_10')
        self.assertEqual(result['sample_mean'],-2.)
        self.assertGreater(result['batch_se'],result['sample_se'])
        expected=ev.statistics_for([-1.,-3.])
        self.assertAlmostEqual(result['batch_ci95_low'],expected['ci95_low'])
        self.assertEqual(result['num_samples'],8);self.assertEqual(result['num_batches'],2)

    def test_duplicate_missing_invariant_nonfinite_and_changed_pairing_are_rejected(self):
        import evaluate_twc_v1_k8_decoder_screen as ev
        rows=raw_statistics_fixture()
        for label,changed in (('duplicate',rows+[deepcopy(rows[0])]),('missing',rows[:-1])):
            with self.subTest(issue=label),self.assertRaises(ValueError):ev.summarize_samples(changed,count=8)
        for label,field,value in (('NMSE depends alpha','dl_nmse',3.),('nonfinite','sum_rate_bps_hz',float('nan')),
                                  ('unpaired','channel_fingerprint','b'*64)):
            changed=deepcopy(rows);changed[8][field]=value
            with self.subTest(issue=label),self.assertRaises(ValueError):ev.summarize_samples(changed,count=8)

    def test_global_alpha_tie_smallest_and_one_fixed_value_for_both_layouts(self):
        from configs import twc_v1_k8_decoder_screen_config as cfg
        import evaluate_twc_v1_k8_decoder_screen as ev
        rows=[dict(arm=m,group_count=g,energy_policy=p,gamma_db=gamma,alpha=a,
            num_samples=64,sum_rate_bps_hz_mean=1.) for m,g,p,gamma in ev.base_conditions() for a in cfg.ALPHAS]
        result=ev.select_global_alphas(rows)
        self.assertEqual(set(result['selected_alphas'].values()),{.0001})
        with self.assertRaises(ValueError):ev.select_global_alphas(rows[:-1])
        self.assertEqual({r['alpha'] for r in raw_statistics_fixture() if r['rzf_mode']=='fixed'},{.03})

    def test_default_summary_has_all_six_rates_core_structure_ci_and_size_bound(self):
        import run_twc_v1_k8_decoder_screen as controller
        import evaluate_twc_v1_k8_decoder_screen as ev
        from configs import twc_v1_k8_decoder_screen_config as cfg
        rows=raw_statistics_fixture();summary,_=ev.summarize_samples(rows,count=8)
        result=dict(summary=summary,paired=ev.paired_statistics(rows,count=8))
        state=dict(status='completed',code={'commit':'f'*40},result_package='exact-package.zip',summary_path='batch_summary.json',stages={})
        lines=controller.summary_lines(state,{a:{'step':6048,'status':'completed'} for a in cfg.ARMS},result)
        text='\n'.join(lines)
        self.assertLessEqual(len(lines),40);self.assertLessEqual(len(text.encode()),8000)
        for method in cfg.METHODS:self.assertIn(method+'/',text)
        self.assertIn('g4_dc2_minus_base2_hybrid_1_3_fixed',text)
        self.assertIn('fdma_deep4_minus_base2_fdma_fixed',text)
        self.assertIn('exact-package.zip',text)
        for line in lines:
            if ' fixed=' in line:self.assertEqual(len(line.split('fixed=',1)[1].split(' ')[0].split(',')),6)

    def test_dry_run_uses_standard_library_and_creates_no_output(self):
        import subprocess
        from configs import twc_v1_k8_decoder_screen_config as cfg
        before=cfg.OUTPUT_ROOT.exists()
        for name in ('train','evaluate','run'):
            code=("import sys,runpy\nsys.argv=['"+name+"','--dry-run']\n"
                "try:\n runpy.run_path('"+name+"_twc_v1_k8_decoder_screen.py',run_name='__main__')\n"
                "except SystemExit as exc:\n assert exc.code in (0,None),exc.code\n"
                "assert not any(x in sys.modules for x in ('torch','numpy','sionna'))\n")
            # A fresh subprocess cannot hide a previously imported heavy module.
            result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('false',result.stdout)
        self.assertEqual(cfg.OUTPUT_ROOT.exists(),before)


@unittest.skipIf(IMPORT_ERROR is not None or importlib.util.find_spec('numpy') is None,
                 IMPORT_ERROR or 'NumPy absent; monitor RNG unit check not executed')
class MonitorStateTests(unittest.TestCase):
    def test_monitor_actual_finally_restores_modes_buffers_and_default_rng_even_on_failure(self):
        import numpy as np
        import random
        from types import SimpleNamespace
        import train_twc_v1_k8_decoder_screen as trainer
        class Chain(torch.nn.Module):
            def __init__(self):
                super().__init__();self.codec_model=torch.nn.Module();self.codec_model.encoder=torch.nn.Linear(1,1)
                self.codec_model.decoder=torch.nn.Linear(1,1);self.ce_model=torch.nn.Linear(1,1)
            def metrics(self,out,hdl,**kwargs):
                rate=hdl[:,0].abs()+1.
                return dict(dl_nmse_per_user=rate[:,None].expand(-1,8),
                    sum_rate_bps_hz=rate,desired_signal_power_mean=rate,interference_power_mean=rate,
                    rzf_per_subcarrier_power=torch.ones(4,2))
        chain=Chain().train();trainer.freeze_training_parts(chain)
        class Data:
            def sample(self,role,index):
                return dict(H_dl=torch.arange(4.)[:,None],H_ul=torch.ones(4,1),sample_fingerprint='a'*64)
        warmup=SimpleNamespace(standard_noise=lambda n,generator,device:torch.randn(n,2,generator=generator,device=device))
        def run(*args):
            torch.rand(1);np.random.rand();random.random()
            return dict(energy={'actual_data_per_ue':torch.full((4,8),48.)})
        for should_fail in (False,True):
            before=trainer.capture_rng(torch,np);modes={n:m.training for n,m in chain.named_modules()}
            state=deepcopy(chain.state_dict())
            def failing(*args):run(*args);raise RuntimeError('injected monitor failure')
            with patch.object(trainer.previous.task.previous.old,'_run',failing if should_fail else run):
                if should_fail:
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        trainer.monitor(torch,np,warmup,chain,Data(),'g4_base2',1512)
                else:
                    report=trainer.monitor(torch,np,warmup,chain,Data(),'g4_base2',1512)
                    self.assertEqual((len(report['summary']),len(report['all_six'])),(12,2))
            after=trainer.capture_rng(torch,np)
            self.assertEqual(before['python'],after['python'])
            self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
            self.assertEqual(before['numpy'][2:],after['numpy'][2:])
            self.assertTrue(torch.equal(before['torch_cpu'],after['torch_cpu']))
            self.assertEqual(modes,{n:m.training for n,m in chain.named_modules()})
            for n,v in state.items():self.assertTrue(torch.equal(chain.state_dict()[n],v))


if __name__=='__main__':unittest.main()
