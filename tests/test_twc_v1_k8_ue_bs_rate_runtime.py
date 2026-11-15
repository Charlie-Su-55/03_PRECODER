"""Real CPU waveform/recovery gradients and causal-prior boundaries."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn

from configs import twc_v1_k8_ue_bs_rate_config as cfg
import ue_bs_rate_runtime as runtime
import ul_adapt_runtime as old


class JointRuntimeTests(unittest.TestCase):
    def setUp(self):
        gen=torch.Generator().manual_seed(7711)
        self.codes=torch.randn(8,216,dtype=torch.complex64,generator=gen)
        self.templates=torch.randn(3,8,32,256,dtype=torch.complex64,generator=gen)
        self.sample={name:torch.randn(4,8,32,256,dtype=torch.complex64,generator=gen)
            for name in ('H_dl','H_ul')}

    def modules(self,arm='hybrid_g4'):
        codes,templates=self.codes,self.templates
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__();self.codes=nn.Parameter(codes.clone())
            def forward(self,H):
                return self.codes[None].expand(H.shape[0],-1,-1)
        class CE(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.96))
            def forward(self,Y,pilots,initial,*,gamma_db):return self.scale*initial
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__();self.mix=nn.Parameter(torch.tensor([.1,.2]))
                self.register_buffer('templates',templates.clone())
            def forward(self,*,Y_rx,H_ul_est,soft,reliability,pilots,gamma_db,layout,prior,ce_error):
                return dict(H_hat=self.templates[0][None]+
                    self.mix[0]*soft[...,0,None,None]*self.templates[1][None]+
                    self.mix[1]*soft[...,1,None,None]*self.templates[2][None])
        codec=nn.Module();codec.encoder=Encoder();codec.decoder=Decoder()
        ce=CE();phy=runtime.physics()
        stats=phy.ReceiverStatistics(phy.make_layout(arm),ce_errors=torch.full((6,8,256),.02))
        stats.ema_updates.fill_(20000)
        for module in (codec,stats,ce):module.eval().requires_grad_(False)
        return codec,stats,ce

    def test_real_rate_gradient_reaches_ue_and_bs_but_not_ce_or_prior(self):
        for arm in cfg.ARMS:
            codec,stats,ce=self.modules(arm)
            parameters=runtime.configure_joint(codec,ce,stats)
            self.assertEqual(set(map(id,parameters)),set(map(id,codec.parameters())))
            frozen=runtime.frozen_state(codec,ce,stats)
            prior=copy.deepcopy(stats.state_dict())
            out=runtime.forward_codec(codec,stats,ce,self.sample,None,('codec_train',1,0))
            self.assertTrue(out['waveform'].active.requires_grad)
            self.assertTrue(out['soft'].requires_grad)
            self.assertFalse(out['H_ul_est'].requires_grad)
            loss,_=runtime.helpers().rate_objective(out['H_hat'],self.sample['H_dl'],alpha=.03,downlink_snr_db=25.)
            loss.backward()
            for module in (codec.encoder,codec.decoder):
                gradients=[p.grad for p in module.parameters()]
                self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients),1e-7)
            self.assertTrue(all(p.grad is None for p in ce.parameters()))
            for key,value in prior.items():self.assertTrue(torch.equal(value,stats.state_dict()[key]))
            runtime.assert_frozen_state(codec,ce,stats,frozen)

    def test_eight_microbatches_use_one_prior_and_commit_only_after_optimizer(self):
        codec,stats,ce=self.modules()
        parameters=runtime.configure_joint(codec,ce,stats)
        frozen=runtime.frozen_state(codec,ce,stats)
        before=stats.power_ema.clone()
        optimizer=torch.optim.AdamW(parameters,lr=1e-5)
        power=None;prior_hashes=[]
        for index in range(1,9):
            out=runtime.forward_codec(codec,stats,ce,self.sample,None,('codec_train',index,0))
            prior_hashes.append(out['input_hashes']['prior'])
            observed=out['waveform'].active.detach().abs().square().sum(0)
            power=observed if power is None else power+observed
            (out['H_hat'].abs().square().mean()/8).backward()
            self.assertEqual(int(stats.ema_updates),20000)
            self.assertTrue(torch.equal(stats.power_ema,before))
        self.assertEqual(len(set(prior_hashes)),1)
        optimizer.step()
        stats.commit_power(power,32,expected_previous_updates=20000)
        self.assertEqual(int(stats.ema_updates),20001)
        expected=before*.99+(power/32)*.01
        torch.testing.assert_close(stats.power_ema,expected,rtol=2e-7,atol=1e-7)
        runtime.assert_frozen_state(codec,ce,stats,frozen)
        with self.assertRaises(ValueError):stats.commit_power(power,32,expected_previous_updates=20000)

    def test_changed_ue_changes_data_but_keeps_paid_pilot_ce_identical(self):
        codec,stats,ce=self.modules()
        runtime.configure_joint(codec,ce,stats)
        first=runtime.forward_codec(codec,stats,ce,self.sample,None,('codec_train',1,0))
        with torch.no_grad():codec.encoder.codes[:,0].add_(.2+.3j)
        second=runtime.forward_codec(codec,stats,ce,self.sample,None,('codec_train',1,0))
        self.assertNotEqual(first['input_hashes']['Y_rx'],second['input_hashes']['Y_rx'])
        for name in ('standard_noise','pilot_polarity','gamma','estimated_UL','CE_variance','prior'):
            self.assertEqual(first['input_hashes'][name],second['input_hashes'][name],name)
        self.assertTrue(torch.equal(first['waveform'].pilot,second['waveform'].pilot))

    def test_source_and_bs_eval_are_same_frozen_radio_and_never_commit(self):
        codec,stats,ce=self.modules('fdma')
        before=runtime.state_digest(stats)
        with torch.inference_mode():
            source=runtime.forward_codec(codec,stats,ce,self.sample,10.,('final',0,2),method='source')
            control=runtime.forward_codec(codec,stats,ce,self.sample,10.,('final',0,2),method='bs_only')
        self.assertEqual(source['input_hashes'],control['input_hashes'])
        self.assertEqual(runtime.state_digest(stats),before)
        with self.assertRaises(RuntimeError):
            stats.commit_power(source['waveform'].active.abs().square().sum(0),4,expected_previous_updates=20000)

    def test_training_noise_replay_and_fresh_eval_leave_default_rng_unchanged(self):
        state=torch.random.get_rng_state().clone()
        for index in (1,8,80000):
            actual=runtime.batch_random('codec_train',index,'cpu')
            expected=old.batch_random('codec_train',index,'cpu')
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
        fresh=runtime.batch_random('final',0,'cpu',gamma=0.)
        oldfinal=old.batch_random('final',0,'cpu',gamma=0.)
        self.assertFalse(torch.equal(fresh[1],oldfinal[1]))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        train=runtime.make_data('cpu',role='codec_train')
        final=runtime.make_data('cpu',role='final')
        monitor=runtime.make_data('cpu',role='monitor')
        self.assertEqual(train.protocol,old.make_data('cpu').protocol)
        self.assertNotEqual(train.protocol['data_id'],final.protocol['data_id'])
        self.assertEqual(final.protocol,monitor.protocol)
        self.assertNotEqual(final.address('validation',4096),monitor.address('validation',0))

    def test_ce_scope_or_table_mutations_are_rejected(self):
        codec,stats,ce=self.modules();runtime.configure_joint(codec,ce,stats)
        frozen=runtime.frozen_state(codec,ce,stats)
        stats.ce_errors.add_(.01)
        with self.assertRaises(ValueError):runtime.assert_frozen_state(codec,ce,stats,frozen)
        ce.train()
        with self.assertRaises(ValueError):runtime.forward_codec(codec,stats,ce,self.sample,None,('codec_train',1,0))
        ce.eval().requires_grad_(True)
        with self.assertRaises(ValueError):runtime.frozen_state(codec,ce,stats)

    def test_owned_atomic_json_failure_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'joint'
            with mock.patch.object(runtime,'OUTPUT_ROOT',output):
                target=output/'training/a.json'
                runtime.atomic_json(target,{'old':1})
                before=target.read_bytes()
                with mock.patch.object(runtime.os,'replace',side_effect=OSError('real replace refusal')):
                    with self.assertRaises(OSError):runtime.atomic_json(target,{'new':2})
                self.assertEqual(target.read_bytes(),before)
                self.assertEqual(list(target.parent.iterdir()),[target])
                with self.assertRaises(FileExistsError):runtime.atomic_json(target,{'new':2},exclusive=True)
                with self.assertRaises(ValueError):runtime.atomic_json(output.parent/'outside.json',{})

    def test_production_cpu_source_cannot_bypass_historical_device_identity(self):
        for function,args in ((runtime.load_initial,('fdma',)),(runtime.load_control,('fdma',))):
            with self.assertRaisesRegex(ValueError,'canonical cuda:0'):function(*args,device='cpu')

    def test_control_file_acceptance_binds_every_artifact_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'training/original_fdma'
            directory.mkdir(parents=True)
            files={}
            for name in cfg.CONTROL_FILE_SHA256S['fdma']:
                (directory/name).write_bytes(('public synthetic fixture '+name).encode())
                files[name]=runtime.sha256(directory/name)
            pins=cfg.SOURCE_CONTROL_PINS['fdma']
            terminal={key:copy.deepcopy(pins[key]) for key in ('checkpoint_sha256','counters','training_trace')}
            with mock.patch.object(cfg,'CONTROL_ROOT',root), \
                    mock.patch.object(cfg,'CONTROL_FILE_SHA256S',{'fdma':files}), \
                    mock.patch.object(runtime.benchmark,'verify_terminal_record',return_value=terminal):
                accepted=runtime.verify_control_files('fdma')
                self.assertEqual(len(accepted['frozen_files']),5)
                self.assertEqual(accepted['training_trace'],cfg.TRAINING_TRACE_PIN)
                (directory/'checkpoint_latest.pt').write_bytes(b'changed fixture')
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    runtime.verify_control_files('fdma')

    def test_original_source_binding_is_explicit_and_not_a_null_fallback(self):
        base={'arm':'fdma','checkpoint_sha256':'a'*64}
        initial={'method':'joint','base_source_identity':base}
        terminal={'source_identity':initial}
        self.assertEqual(runtime.stable_source_identity(initial),base)
        self.assertEqual(runtime.stable_source_identity(terminal),base)
        for bad in ({},{'source_identity':None},{'source_identity':{}}):
            with self.assertRaises(ValueError):runtime.stable_source_identity(bad)


if __name__=='__main__':unittest.main()
