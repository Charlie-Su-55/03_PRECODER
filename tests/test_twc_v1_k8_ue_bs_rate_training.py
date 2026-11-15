"""Actual CPU UE/MAC/recovery/BS/RZF gradients and causal checkpoint replay."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import train_twc_v1_k8_ue_bs_rate as train
from evaluate_twc_v1_k8_icc_aligned import _tensor_hash


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.angle = torch.nn.Parameter(torch.tensor(.17))
        self.register_buffer('shape', torch.cos(torch.arange(216).float()/5.))

    def forward(self, h):
        real = (1.+self.angle*self.shape)[None,None].expand(h.shape[0],8,-1)
        return torch.complex(real, .07*real.flip(-1))


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coupling = torch.nn.Parameter(torch.tensor(.22))
        self.snr_slope = torch.nn.Parameter(torch.tensor(.03))
        base = torch.zeros(1,8,32,256,dtype=torch.complex64)
        base[:,torch.arange(8),torch.arange(8),:] = 1.
        mix = torch.zeros_like(base)
        mix[:,0,1,:] = 1.
        mix[:,1,0,:] = 1.
        self.register_buffer('base',base)
        self.register_buffer('mix',mix)

    def forward(self, soft, gamma):
        observed = soft[...,0].real.mean(-1)
        gain = self.coupling + .06*observed + self.snr_slope*(gamma/25.-.5)
        return self.base+self.mix*gain[:,None,None,None]


class JointTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = train.capture_rng()
        self.addCleanup(train.restore_rng,rng)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'joint'
        self.core = dict(schema=train.cfg.NAMESPACE+'_core_v1',sha256={'CPU_fixture':'a'*64})
        self.patches = []
        for patcher in (
            mock.patch.object(train.cfg,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.runtime,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.cfg,'UPDATES',4),
            mock.patch.object(train.cfg,'SAVE_EVERY',2),
            mock.patch.object(train.cfg,'ACCUMULATION',2),
            mock.patch.object(train.cfg,'MONITOR_BATCHES',1),
            mock.patch.object(train.cfg,'MONITOR_STEPS',(0,2,4)),
            mock.patch.object(train.cfg,'ROLE_RANGES',dict(codec_train=[1,8],monitor=[0,0],final=[4096,4351])),
            mock.patch.object(train.runtime,'core_identity',return_value=self.core),
            mock.patch.object(train.runtime,'source_identity',return_value=dict(commit='b'*40,sha256=self.core['sha256'])),
            mock.patch.object(train.runtime,'guard_core',side_effect=lambda value:self.assertEqual(value,self.core))):
            patcher.start()
            self.addCleanup(patcher.stop)
        torch.manual_seed(379)
        codec = torch.nn.Module()
        codec.encoder,codec.decoder = TinyEncoder(),TinyDecoder()
        phy = train.runtime.physics()
        stats = phy.ReceiverStatistics(phy.make_layout('fdma'),torch.full((6,8,256),.002))
        stats.ema_updates.fill_(20000)
        ce = torch.nn.Linear(2,2)
        for module in (codec,stats,ce):
            module.eval().requires_grad_(False)
        self.base = codec,stats,ce
        self.source_path = self.root/'explicit_CPU_fixture.pt'
        torch.save(dict(model=codec.state_dict(),statistics=stats.state_dict()),self.source_path)
        self.source_bytes = self.source_path.read_bytes()
        self.source = dict(arm='fdma',base_ema_updates=20000,
            base_encoder_state_sha256=train.state_digest(codec.encoder),
            base_decoder_state_sha256=train.state_digest(codec.decoder),
            base_statistics_state_sha256=train.state_digest(stats),
            frozen_files=[dict(path=str(self.source_path),sha256=train.sha256(self.source_path))])
        self.loaded,self.requests,self.forward_calls = [],[],[]
        outer = self
        class Data:
            def __init__(self, role):
                self.protocol = dict(fixture='actual_CPU_MAC_RZF',stream='train' if role=='codec_train' else 'monitor')
            def address(self,role,index):
                return dict(namespace=self.protocol['stream'],role=role,index=index)
            def sample(self,role,index):
                if role not in ('codec_train','validation'):
                    raise AssertionError('Unexpected data role')
                outer.requests.append((self.protocol['stream'],role,index))
                h = outer.base[0].decoder.base.expand(4,-1,-1,-1).clone()
                h[:,0,1,:] = .01*(index%7)
                ul = outer.base[0].decoder.base.expand_as(h).clone()
                ul[:,torch.arange(8),torch.arange(8),:] *= complex(1.,.003*(index%5))
                return dict(H_dl=h,H_ul=ul,address=self.address(role,index),
                    sample_fingerprint=train.cfg.digest([_tensor_hash(h),_tensor_hash(ul)]))
        self.data = {role:Data(role) for role in ('codec_train','monitor')}
        patcher = mock.patch.object(train.runtime,'verify_files',side_effect=self.verify_files)
        patcher.start()
        self.addCleanup(patcher.stop)

    def verify_files(self, record):
        for item in record['frozen_files']:
            if train.sha256(item['path']) != item['sha256']:
                raise ValueError('Fixture source bytes changed')

    def loader(self, arm, method, device):
        self.assertEqual((arm,method,str(device)),('fdma','joint','cpu'))
        result = copy.deepcopy(self.base)
        self.loaded.append(result)
        return (*result,copy.deepcopy(self.source))

    def data_factory(self, device, role='final'):
        self.assertEqual(str(device),'cpu')
        return self.data[role]

    def forward(self, codec, stats, ce, sample, gamma_db, noise_key, *, method):
        phy = train.runtime.physics()
        role,index,gi = noise_key
        self.forward_calls.append((role,index,int(stats.ema_updates),train.state_digest(stats)))
        if role=='codec_train':
            self.assertTrue(codec.encoder.training and codec.decoder.training and stats.training)
            self.assertTrue(torch.is_grad_enabled())
        else:
            self.assertFalse(codec.encoder.training or codec.decoder.training or stats.training)
            self.assertFalse(torch.is_grad_enabled())
        self.assertFalse(ce.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in ce.parameters()))
        random.random();np.random.rand();torch.rand(1)
        ri = index if role=='codec_train' else index*len(train.cfg.GAMMAS)+gi
        gamma,noise,polarity = train.runtime.batch_random(role,ri,'cpu',gamma=gamma_db)
        raw = codec.encoder(sample['H_dl'])
        wave = phy.build_waveform(raw,stats.layout,polarity=polarity)
        Y = phy.transmit(sample['H_ul'],wave.x_grid,noise,gamma)
        # This declared fixture uses a synthetic known estimated channel;
        # the production forward's pilot-only CE path is tested separately.
        estimated = sample['H_ul'].detach().clone()
        prior,cevar = stats.nominal_power(),stats.ce_variance(gamma)
        recovered = phy.recover(Y,estimated,stats.layout,prior,cevar,gamma)
        hhat = codec.decoder(recovered['soft'],gamma)
        return dict(H_hat=hhat,soft=recovered['soft'],reliability=recovered['reliability'],
            H_ul_est=estimated,source_estimated_UL=estimated,recovered=recovered,
            waveform=wave,layout=stats.layout,gamma_db=gamma,
            input_hashes=dict(standard_noise=_tensor_hash(noise),pilot_polarity=_tensor_hash(polarity),
                gamma=_tensor_hash(gamma),wave_active=_tensor_hash(wave.active),Y_rx=_tensor_hash(Y),
                estimated_UL=_tensor_hash(estimated),CE_variance=_tensor_hash(cevar),prior=_tensor_hash(prior)))

    def run_train(self, target=2, resume=False, **kwargs):
        return train.train('fdma',target,resume,require_cuda=False,loader=self.loader,
            data_factory=self.data_factory,forward=self.forward,**kwargs)

    def directory(self):
        return self.output/'training'/train.cfg.key('fdma')

    def payload(self):
        return torch.load(self.directory()/'checkpoint_latest.pt',map_location='cpu',weights_only=False)

    def verify(self,payload,manifest,**kwargs):
        codec,stats,ce = self.loaded[-1]
        return train.verify_saved(self.directory(),payload,manifest,arm='fdma',data=self.data['codec_train'],
            core=self.core,source=self.source,frozen=train.runtime.frozen_state(codec,ce,stats),
            allow_fixture=True,**kwargs)

    def zero_boundary(self):
        def stop(*args,**kwargs):
            raise RuntimeError('intentional monitor0 stop')
        with self.assertRaisesRegex(RuntimeError,'monitor0'):
            self.run_train(monitor=stop)
        return train.read_json(self.directory()/'manifest.json')

    def test_real_joint_gradients_and_causal_full_state_resume(self):
        first = self.run_train()
        saved = self.payload()
        self.assertEqual((first['step'],first['status']),(2,'paused_boundary'))
        self.assertFalse(first['production_evidence'])
        self.assertEqual(saved['counters'],train.cfg.counters(2,20000))
        self.assertEqual(int(saved['statistics']['ema_updates']),20002)
        self.assertNotEqual(saved['model']['encoder.angle'],self.base[0].encoder.angle)
        self.assertNotEqual(saved['model']['decoder.coupling'],self.base[0].decoder.coupling)
        self.assertEqual(self.verify(saved,first),2)
        self.assertEqual(len(saved['optimizer']['state']),3)
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'],1e-5)
        self.assertTrue(torch.equal(saved['statistics']['ce_errors'],self.base[1].ce_errors))
        self.assertEqual(train.state_digest(self.loaded[-1][2]),train.state_digest(self.base[2]))
        final = self.run_train(target=4,resume=True)
        resumed = self.payload()
        self.assertEqual(self.verify(resumed,final,require_final=True),4)
        self.assertEqual(int(resumed['statistics']['ema_updates']),20004)
        self.assertEqual(self.source_path.read_bytes(),self.source_bytes)
        requests = [i for stream,role,i in self.requests if role=='codec_train']
        self.assertEqual(requests,list(range(1,9)))
        calls = [c for c in self.forward_calls if c[0]=='codec_train']
        for i in range(0,len(calls),2):
            self.assertEqual(calls[i][2:],calls[i+1][2:])
        before_continuous = self.root/'continuous'
        with mock.patch.object(train.cfg,'OUTPUT_ROOT',before_continuous),mock.patch.object(train.runtime,'OUTPUT_ROOT',before_continuous):
            self.run_train(target=4)
            continuous = self.payload()
        for name in ('model','statistics'):
            self.assertEqual(train.state_digest(resumed[name]),train.state_digest(continuous[name]))
        self.assertEqual(resumed['training_trace'],continuous['training_trace'])
        self.assertTrue(torch.equal(resumed['rng']['torch_cpu'],continuous['rng']['torch_cpu']))
        self.assertEqual(resumed['rng']['python'],continuous['rng']['python'])
        for key,record in resumed['optimizer']['state'].items():
            for field,value in record.items():
                self.assertTrue(torch.equal(value,continuous['optimizer']['state'][key][field]))

    def test_monitoring_never_changes_models_prior_ce_or_rng(self):
        self.run_train()
        for step in (0,2):
            receipt=train.read_json(self.directory()/'monitor'/f'step_{step:05d}_receipt.json')
            self.assertFalse(receipt['selection'])
            self.assertFalse(receipt['production_evidence'])
            self.assertEqual(receipt['data_protocol']['stream'],'monitor')
            self.assertEqual(receipt['training_data_protocol']['stream'],'train')
            self.assertTrue(all(train.sha256(name)==digest for name,digest in receipt['files'].items()))

    def test_step_zero_failure_has_fresh_optimizer_and_preserves_source_prior(self):
        manifest=self.zero_boundary()
        payload=self.payload()
        self.assertEqual(payload['optimizer']['state'],{})
        self.assertEqual(int(payload['statistics']['ema_updates']),20000)
        self.assertEqual(self.verify(payload,manifest),0)
        self.assertEqual(self.requests,[])
        self.run_train(resume=True)
        self.assertEqual(int(self.payload()['statistics']['ema_updates']),20002)

    def test_fixture_cannot_be_production_loaded(self):
        self.run_train(target=4)
        with mock.patch.object(train.runtime,'load_initial',side_effect=AssertionError('must not load source')):
            with self.assertRaisesRegex(ValueError,'fixture'):
                train.load_finetuned('fdma','cpu')

    def test_resume_rejects_changed_recipe_prior_counter_model_optimizer_rng_and_trace(self):
        manifest=self.run_train()
        payload=self.payload()
        mutations=[
            lambda p:p.update(method='bs_only'),
            lambda p:p.update(counters=dict(p['counters'],EMA_commits=999)),
            lambda p:p['statistics']['ema_updates'].fill_(20001),
            lambda p:p['statistics']['power_ema'].fill_(float('nan')),
            lambda p:p['model'].update({'encoder.angle':torch.zeros(2)}),
            lambda p:p['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
            lambda p:p['scheduler'].update(last_epoch=999),
            lambda p:p['rng'].update(torch_cpu=torch.tensor([1],dtype=torch.uint8)),
            lambda p:p['training_trace'].update(sha256='0'*64),
        ]
        for index,mutate in enumerate(mutations):
            changed=copy.deepcopy(payload);mutate(changed)
            with self.subTest(mutation=index),self.assertRaises(ValueError):
                self.verify(changed,manifest)

    def test_actual_torch_serialization_failure_and_replace_failure_preserve_checkpoint(self):
        self.zero_boundary()
        directory=self.directory()
        before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
        codec,stats,ce=self.loaded[-1]
        params=train.runtime.configure_joint(codec,ce,stats)
        optimizer=torch.optim.AdamW(params,lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        kwargs=dict(arm='fdma',step=0,data=self.data['codec_train'],core=self.core,
            frozen=train.runtime.frozen_state(codec,ce,stats),elapsed=0.,measured=0.,production_evidence=False)
        with self.assertRaises((AttributeError,TypeError)):
            train.save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,
                source=dict(self.source,unpicklable=lambda:None),**kwargs)
        self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()})
        with mock.patch.object(train.original.os,'replace',side_effect=OSError('intentional replace failure')):
            with self.assertRaises(OSError):
                train.save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,source=self.source,**kwargs)
        self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()})
        self.assertEqual(list(directory.glob('checkpoint_*.pt')),[directory/'checkpoint_latest.pt'])

    def test_invalid_gradient_or_power_never_commits_ema(self):
        self.zero_boundary()
        codec,stats,ce=self.loaded[-1]
        params=train.runtime.configure_joint(codec,ce,stats)
        optimizer=torch.optim.AdamW(params,lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        before=train.state_digest(codec),train.state_digest(stats)
        for p in params:p.grad=torch.full_like(p,float('nan'))
        with self.assertRaises(FloatingPointError):
            train.commit_update(codec,optimizer,scheduler,stats=stats,power_sum=torch.full((8,24),16.),
                frames=8,previous_updates=0,base_ema_updates=20000)
        for p in params:p.grad=torch.ones_like(p)
        with self.assertRaises(ValueError):
            train.commit_update(codec,optimizer,scheduler,stats=stats,power_sum=torch.zeros(8,24),
                frames=8,previous_updates=0,base_ema_updates=20000)
        self.assertEqual(before,(train.state_digest(codec),train.state_digest(stats)))
        self.assertEqual(optimizer.state,{})

    def test_nonfinite_optimizer_result_refuses_ema_commit(self):
        self.zero_boundary()
        codec,stats,ce=self.loaded[-1]
        params=train.runtime.configure_joint(codec,ce,stats)
        optimizer=torch.optim.AdamW(params,lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        for p in params:p.grad=torch.ones_like(p)
        actual=optimizer.step
        def bad_step(*args,**kwargs):
            actual(*args,**kwargs)
            with torch.no_grad():params[0].fill_(float('nan'))
        with mock.patch.object(optimizer,'step',side_effect=bad_step):
            with self.assertRaises(FloatingPointError):
                train.commit_update(codec,optimizer,scheduler,stats=stats,power_sum=torch.full((8,24),16.),
                    frames=8,previous_updates=0,base_ema_updates=20000)
        self.assertEqual(int(stats.ema_updates),20000)

    def test_source_encoder_or_prior_change_refused_before_output(self):
        def wrong(arm,method,device):
            codec,stats,ce,source=self.loader(arm,method,device)
            with torch.no_grad():codec.encoder.angle.add_(1)
            return codec,stats,ce,source
        with self.assertRaisesRegex(ValueError,'encoder'):
            train.train('fdma',2,require_cuda=False,loader=wrong,data_factory=self.data_factory,forward=self.forward)
        self.assertFalse(self.output.exists())

    def test_optimizer_seed_matches_replayed_benchmark_recipe(self):
        with mock.patch.object(torch,'manual_seed',wraps=torch.manual_seed) as seeded:
            self.zero_boundary()
        seeded.assert_any_call(train.cfg.OPTIMIZER_SEED)
        self.assertNotEqual(train.cfg.OPTIMIZER_SEED,train.cfg.SEED)

    def test_failed_serialization_after_update_resumes_prior_from_saved_boundary(self):
        self.zero_boundary()
        initial_bytes=(self.directory()/'checkpoint_latest.pt').read_bytes()
        original_save=train.original.atomic_torch_save
        def serialization_failure(payload,target):
            if payload['step']==2:
                payload=dict(payload,deliberately_unpicklable=lambda:None)
            return original_save(payload,target)
        with mock.patch.object(train.original,'atomic_torch_save',side_effect=serialization_failure):
            with self.assertRaises((AttributeError,TypeError)):
                self.run_train(resume=True)
        self.assertEqual((self.directory()/'checkpoint_latest.pt').read_bytes(),initial_bytes)
        self.assertEqual(int(self.payload()['statistics']['ema_updates']),20000)
        final=self.run_train(resume=True)
        self.assertEqual(final['step'],2)
        self.assertEqual(int(self.payload()['statistics']['ema_updates']),20002)
        progress=[json.loads(line) for line in (self.directory()/'progress.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in progress],[2,2])
        self.assertEqual(progress[0]['training_trace'],progress[1]['training_trace'])

    def test_real_100_boundary_receipt_survives_200_terminal_replacement(self):
        with mock.patch.object(train.cfg,'UPDATES',200),mock.patch.object(train.cfg,'SAVE_EVERY',100),\
                mock.patch.object(train.cfg,'MONITOR_STEPS',()),\
                mock.patch.object(train.cfg,'ROLE_RANGES',dict(codec_train=[1,400],monitor=[0,0],final=[4096,4351])):
            first=self.run_train(target=100)
            receipt=self.directory()/'boundary_00100.json'
            original_bytes=receipt.read_bytes()
            original_hash=first['checkpoint_sha256']
            terminal=self.run_train(target=200,resume=True)
            self.assertEqual(receipt.read_bytes(),original_bytes)
            self.assertEqual(terminal['step'],200)
            self.assertEqual(int(self.payload()['statistics']['ema_updates']),20200)
            self.assertNotEqual(terminal['checkpoint_sha256'],original_hash)
            self.assertEqual(train.read_json(receipt)['checkpoint_sha256'],original_hash)
            with self.assertRaises(ValueError):
                train.verify_boundary(self.directory(),'fdma',core=self.core)


if __name__=='__main__':
    unittest.main()
