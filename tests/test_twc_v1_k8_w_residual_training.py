"""Real CPU MU rates, fresh AdamW, checkpoint boundaries and strict resume."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import train_twc_v1_k8_w_residual as train

try:
    import numpy as np
    import torch
except ImportError:
    np=torch=None


if torch is not None:
    class TinyCSI(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coupling=torch.nn.Parameter(torch.tensor(.4))
            base=torch.zeros(1,8,32,256,dtype=torch.complex64)
            base[:,torch.arange(8),torch.arange(8),:]=1.
            mix=torch.zeros_like(base);mix[:,0,1,:]=1.;mix[:,1,0,:]=1.
            self.register_buffer('base',base);self.register_buffer('mix',mix)

        def forward(self,batch):
            return (self.base+self.coupling*self.mix).expand(batch,-1,-1,-1)


    class TinyPrecoder(torch.nn.Module):
        """Small head, using the real full-8UE RZF and power projection."""
        def __init__(self,base,method):
            super().__init__();self.base_decoder=base;self.method=method
            if method=='w_residual':self.head=torch.nn.Parameter(torch.tensor(0.))

        def forward(self,batch):
            model=train.runtime.model_module()
            h=self.base_decoder(batch)
            w,_=model._rate_module()._downlink().design_rzf(h,.03)
            if self.method=='w_residual':
                direction=torch.zeros_like(w);direction[:,:,1,0]=1.;direction[:,:,0,1]=1.
                delta=self.head*direction
                actual=model.normalize_precoder(w+delta)
            else:delta=torch.zeros_like(w);actual=w
            return dict(H_hat=h,W=actual,W_base=w,delta_W=delta)


    class TinyStatistics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('historical_power',torch.full((8,24),2.))
            self.register_buffer('ema_updates',torch.tensor(20000))

        def nominal_power(self):return self.historical_power


@unittest.skipIf(torch is None or np is None,'Real PyTorch/NumPy required; no mock ML runtime')
class ResidualWTrainingTests(unittest.TestCase):
    def setUp(self):
        saved=train.capture_rng();self.addCleanup(train.restore_rng,saved)
        threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.output=self.root/'w_residual'
        self.core=dict(schema=train.cfg.NAMESPACE+'_core_v1',sha256={'CPU_fixture':'a'*64})
        patches=[mock.patch.object(train.cfg,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.runtime,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.cfg,'UPDATES',200),mock.patch.object(train.cfg,'MICRO_BATCH',1),
            mock.patch.object(train.cfg,'ACCUMULATION',1),mock.patch.object(train.cfg,'MONITOR_BATCHES',1),
            mock.patch.object(train.cfg,'MONITOR_STEPS',(0,100,200)),
            mock.patch.object(train.cfg,'ROLE_RANGES',dict(codec_train=[1,200],monitor=[0,0],final=[2048,2303])),
            mock.patch.object(train.runtime,'core_identity',return_value=self.core),
            mock.patch.object(train.runtime,'source_identity',return_value=dict(commit='b'*40,sha256=self.core['sha256']))]
        for patcher in patches:patcher.start();self.addCleanup(patcher.stop)
        torch.manual_seed(826)
        codec=torch.nn.Module();codec.encoder=torch.nn.Linear(2,3);codec.decoder=TinyCSI()
        # Source moments exist, but neither continuation may inherit them.
        optimizer=torch.optim.AdamW(codec.decoder.parameters(),lr=1e-4)
        codec.decoder.coupling.square().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        stats,ce=TinyStatistics(),torch.nn.Linear(2,2)
        for module in (codec,stats,ce):
            module.eval().requires_grad_(False)
            for p in module.parameters():p.grad=None
        self.base=(codec,stats,ce)
        self.source_path=self.root/'pinned_rate_teacher.pt'
        torch.save(dict(model=codec.state_dict(),optimizer=optimizer.state_dict()),self.source_path)
        self.source_bytes=self.source_path.read_bytes()
        self.source=dict(arm='fdma',source_commit=train.cfg.SOURCE_COMMIT,
            base_decoder_state_sha256=train.state_digest(codec.decoder),
            frozen_files=[dict(path=str(self.source_path),sha256=train.sha256(self.source_path))])
        self.loaded=[];self.calls=[];self.requests=[]
        parent=self

        class Data:
            protocol={'fixture':'real_CPU_full_8UE_32antenna_256tone_no_CDL'}

            def address(self,role,index):return dict(namespace='fixture',role=role,index=index)

            def sample(self,role,index):
                if role not in ('codec_train','validation'):raise AssertionError('Test/unrelated role sampled')
                parent.requests.append((role,index))
                truth=parent.base[0].decoder.base.clone()
                return dict(H_dl=truth,H_ul=truth.clone(),address=self.address(role,index),
                    sample_fingerprint=train.cfg.digest([role,index]))

        self.data=Data()

    def loader(self,arm,method,device):
        self.assertEqual((arm,str(device)),('fdma','cpu'))
        codec,stats,ce=copy.deepcopy(self.base)
        codec.decoder=TinyPrecoder(codec.decoder,method)
        codec.eval().requires_grad_(False)
        self.loaded.append((codec,stats,ce))
        return codec,stats,ce,copy.deepcopy(self.source)

    def forward(self,codec,stats,ce,sample,gamma_db,noise_key):
        role=noise_key[0];self.calls.append((role,codec.decoder.training,torch.is_grad_enabled()))
        self.assertFalse(codec.encoder.training);self.assertFalse(stats.training);self.assertFalse(ce.training)
        self.assertTrue(all(not p.requires_grad for p in codec.encoder.parameters()))
        self.assertEqual(codec.decoder.training,role=='codec_train')
        self.assertEqual(torch.is_grad_enabled(),role=='codec_train')
        # Exercise monitor restoration without changing the deterministic data.
        random.random();np.random.rand();torch.rand(1)
        active=torch.full((1,8,24),2**.5,dtype=torch.complex64)
        grid=torch.zeros(1,8,256,dtype=torch.complex64);grid[...,:32]=2**.5
        noise=torch.zeros(1,32,256,dtype=torch.complex64);polarity=torch.ones(1,8)
        gamma=torch.full((1,),12.5 if gamma_db is None else float(gamma_db))
        return dict(codec.decoder(1),waveform=SimpleNamespace(active=active,x_grid=grid),
            soft=active*(1.+.01j),H_ul_est=sample['H_ul'],input_hashes=dict(
                standard_noise=train.state_digest({'noise':noise}),
                pilot_polarity=train.state_digest({'polarity':polarity}),
                gamma=train.state_digest({'gamma':gamma})))

    def run_train(self,method='w_residual',target=100,resume=False,**kwargs):
        return train.train('fdma',method,target,resume,require_cuda=False,
            loader=self.loader,data_factory=lambda _:self.data,forward=self.forward,**kwargs)

    def directory(self,method='w_residual'):
        return self.output/'training'/train.cfg.key('fdma',method)

    def payload(self,method='w_residual'):
        return torch.load(self.directory(method)/'checkpoint_latest.pt',map_location='cpu',weights_only=False)

    def verify(self,payload,manifest,method='w_residual',**kwargs):
        codec,stats,ce=self.loaded[-1]
        frozen=train.helpers().frozen_state(codec,ce,stats)
        return train.verify_saved(self.directory(method),payload,manifest,arm='fdma',method=method,
            data=self.data,core=self.core,source=self.source,frozen=frozen,**kwargs)

    def stop_at_zero(self):
        def failing(*args,**kwargs):raise RuntimeError('simulated monitor0 failure')
        with self.assertRaisesRegex(RuntimeError,'monitor0'):
            self.run_train(monitor=failing)
        return train.read_json(self.directory()/'manifest.json')

    def fresh_optimizer(self):
        codec,stats,ce=self.loaded[-1]
        parameters=train.helpers().configure_bs_only(codec,ce,stats)
        optimizer=torch.optim.AdamW(parameters,lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY,
                                   betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        return codec,stats,ce,optimizer,scheduler

    def save_zero(self,**overrides):
        codec,stats,ce,optimizer,scheduler=self.fresh_optimizer()
        kwargs=dict(arm='fdma',method='w_residual',step=0,data=self.data,core=self.core,
            source=self.source,frozen=train.helpers().frozen_state(codec,ce,stats),elapsed=0.,measured=0.)
        kwargs.update(overrides)
        return train.save_checkpoint(self.directory(),codec,stats,ce,optimizer,scheduler,**kwargs)

    def test_real_100_update_train_roundtrip_resume_and_both_bs_gradients(self):
        first=self.run_train();payload=self.payload();directory=self.directory()
        self.assertEqual((first['step'],first['status']),(100,'paused_boundary'))
        self.assertEqual(first['recipe_fingerprint'],train.cfg.fingerprint('fdma','w_residual'))
        self.assertEqual(payload['counters'],train.cfg.counters(100))
        self.assertEqual(payload['scheduler']['last_epoch'],100)
        self.assertEqual(payload['scheduler']['_step_count'],101)
        self.assertEqual(len(payload['optimizer']['state']),2)
        self.assertTrue(all(float(s['step'])==100 for s in payload['optimizer']['state'].values()))
        self.assertNotIn('encoder.weight',payload['model'])
        self.assertNotEqual(float(payload['model']['base_decoder.coupling']),float(self.base[0].decoder.coupling))
        self.assertNotEqual(float(payload['model']['head']),0.)
        self.assertEqual(self.verify(payload,first),100)
        for field,value in [('step',torch.tensor(99.)),('exp_avg',torch.zeros(2)),
                            ('exp_avg_sq',torch.full_like(next(iter(payload['optimizer']['state'].values()))['exp_avg_sq'],float('nan')))]:
            changed=copy.deepcopy(payload)
            identifier=next(iter(changed['optimizer']['state']))
            changed['optimizer']['state'][identifier][field]=value
            with self.subTest(moment=field),self.assertRaises(ValueError):self.verify(changed,first)
        with self.assertRaisesRegex(ValueError,'Fixed residual-W terminal'):
            self.verify(payload,first,require_final=True)
        for step in (0,100):
            receipt=train.read_json(directory/'monitor'/f'step_{step:05d}_receipt.json')
            self.assertEqual((receipt['status'],receipt['selection']),('passed',False))
            self.assertTrue(receipt['files'])
            self.assertTrue(all(train.sha256(name)==digest for name,digest in receipt['files'].items()))
        self.assertTrue((directory/'boundary_00100.json').is_file())
        final=self.run_train(target=200,resume=True);restored=self.payload()
        self.assertEqual((final['step'],final['status']),(200,'completed'))
        self.assertEqual(self.verify(restored,final,require_final=True),200)
        self.assertEqual(train.read_json(directory/'terminal_receipt.json'),final)
        terminal_path=directory/'terminal_receipt.json';terminal_bytes=terminal_path.read_bytes()
        train.atomic_json(terminal_path,dict(final,changed=True))
        with self.assertRaisesRegex(ValueError,'Fixed residual-W terminal'):
            self.verify(restored,final,require_final=True)
        terminal_path.write_bytes(terminal_bytes)
        self.assertEqual([i for role,i in self.requests if role=='codec_train'],list(range(1,201)))
        progress=[__import__('json').loads(x) for x in (directory/'progress.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in progress],[100,200])
        for record in progress:
            self.assertEqual(len(record['input_fingerprints']),1)
            item=record['input_fingerprints'][0]
            self.assertEqual(set(item),{'micro_index','address','channel','standard_noise','pilot_polarity','gamma'})
            self.assertEqual(item['micro_index'],record['step'])
            self.assertEqual(item['address'],self.data.address('codec_train',record['step']))
        codec,stats,ce=self.loaded[-1]
        expected=train.helpers().frozen_state(self.base[0],self.base[2],self.base[1])
        train.helpers().assert_frozen_state(codec,ce,stats,expected)
        self.assertEqual(self.source_path.read_bytes(),self.source_bytes)
        # A resumed optimizer/RNG boundary must reproduce the uninterrupted
        # path, not merely produce another checkpoint that can be opened.
        other=self.root/'uninterrupted'
        with mock.patch.object(train.cfg,'OUTPUT_ROOT',other),mock.patch.object(train.runtime,'OUTPUT_ROOT',other):
            self.run_train(target=200)
            continuous=self.payload()
        self.assertEqual(train.state_digest(continuous['model']),train.state_digest(restored['model']))
        for identifier,state in restored['optimizer']['state'].items():
            for name in ('step','exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(state[name],continuous['optimizer']['state'][identifier][name]))

    def test_rzf_continuation_also_uses_fresh_optimizer_and_actual_complete_mu_rate(self):
        result=self.run_train(method='rzf_continue')
        payload=self.payload('rzf_continue')
        self.assertEqual(result['step'],100)
        self.assertEqual(len(payload['optimizer']['state']),1)
        self.assertFalse(any('head' in name for name in payload['model']))
        self.assertNotEqual(float(payload['model']['base_decoder.coupling']),float(self.base[0].decoder.coupling))
        self.assertTrue(any(role=='monitor' and not training and not grad for role,training,grad in self.calls))
        self.assertEqual(self.source_path.read_bytes(),self.source_bytes)

    def test_monitor0_failure_keeps_real_fresh_checkpoint_and_resume_starts_at_one(self):
        manifest=self.stop_at_zero();payload=self.payload()
        self.assertEqual((payload['step'],payload['optimizer']['state']),(0,{}))
        self.assertEqual(payload['scheduler']['last_epoch'],0)
        self.assertEqual(payload['scheduler']['_step_count'],1)
        self.assertEqual(self.verify(payload,manifest),0)
        self.assertFalse(self.requests)
        result=self.run_train(resume=True)
        self.assertEqual(result['step'],100)
        self.assertEqual([i for role,i in self.requests if role=='codec_train'],list(range(1,101)))
        self.assertEqual(self.source_path.read_bytes(),self.source_bytes)

    def test_real_serialization_failure_preserves_checkpoint_manifest_and_temp_cleanup(self):
        self.stop_at_zero();directory=self.directory()
        before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
        invalid=dict(self.source,unpicklable=lambda:None)
        with self.assertRaises((AttributeError,TypeError)):
            self.save_zero(source=invalid)
        self.assertEqual({p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()},before)
        self.assertFalse(any(p.name.startswith('checkpoint_') and p.name!='checkpoint_latest.pt'
                             for p in directory.iterdir()))
        self.assertEqual(self.source_path.read_bytes(),self.source_bytes)

    def test_checkpoint_schema_source_data_code_model_optimizer_and_rng_tampering_fail(self):
        manifest=self.stop_at_zero();good=self.payload()
        changes=[('format','another_run'),('method','rzf_continue'),('core',{'changed':True}),
            ('data_protocol',{'changed':True}),('source_identity',{'changed':True}),
            ('frozen_identity',{'changed':True}),('step',True),('step',1),
            ('counters',dict(good['counters'],EMA_commits=1))]
        for key,value in changes:
            payload=copy.deepcopy(good);payload[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.verify(payload,manifest)
        bad=copy.deepcopy(good);bad['unknown_extension']=1
        with self.assertRaisesRegex(ValueError,'schema'):self.verify(bad,manifest)
        bad=copy.deepcopy(good);bad['optimizer']['param_groups'][0]['lr']*=2
        with self.assertRaisesRegex(ValueError,'optimizer recipe'):self.verify(bad,manifest)
        bad=copy.deepcopy(good);bad['scheduler']['_step_count']=2
        with self.assertRaisesRegex(ValueError,'scheduler'):self.verify(bad,manifest)
        bad=copy.deepcopy(good);bad['model']['head']=torch.ones(2)
        with self.assertRaisesRegex(ValueError,'schema'):self.verify(bad,manifest)
        bad=copy.deepcopy(good);bad['training_code']['sha256']={'changed':True}
        with self.assertRaisesRegex(ValueError,'code identity'):self.verify(bad,manifest)
        bad=copy.deepcopy(good);bad['rng']['python']='bad state'
        with self.assertRaisesRegex(ValueError,'RNG'):self.verify(bad,manifest)
        initial=self.directory()/'initialization.json';original=initial.read_bytes();initial.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError,'initialization_sha256'):self.verify(good,manifest)
        initial.write_bytes(original)
        target=self.directory()/'checkpoint_latest.pt';target.write_bytes(target.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint bytes'):self.verify(good,manifest)

    def test_source_core_and_frozen_mutations_stop_before_checkpoint_replacement(self):
        self.stop_at_zero();before=(self.directory()/'checkpoint_latest.pt').read_bytes()
        self.loaded[-1][1].ema_updates.add_(1)
        frozen=train.helpers().frozen_state(self.base[0],self.base[2],self.base[1])
        with self.assertRaisesRegex(ValueError,'Frozen encoder'):
            self.save_zero(frozen=frozen)
        self.loaded[-1][1].ema_updates.sub_(1)
        self.source_path.write_bytes(self.source_bytes+b'changed')
        with self.assertRaisesRegex(ValueError,'source artifact changed'):self.save_zero()
        self.source_path.write_bytes(self.source_bytes)
        with mock.patch.object(train.runtime,'core_identity',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'core changed'):self.save_zero()
        self.assertEqual((self.directory()/'checkpoint_latest.pt').read_bytes(),before)

    def test_missing_nonfinite_zero_gradients_and_changed_optimizer_stop_before_update(self):
        self.stop_at_zero();codec,stats,ce,optimizer,scheduler=self.fresh_optimizer()
        before=train.state_digest(codec.decoder)
        for p in codec.decoder.parameters():p.grad=torch.full_like(p,float('nan'))
        with self.assertRaises(FloatingPointError):train.commit_update(codec,optimizer,scheduler,previous_updates=0)
        for p in codec.decoder.parameters():p.grad=None
        with self.assertRaises(FloatingPointError):train.commit_update(codec,optimizer,scheduler,previous_updates=0)
        for p in codec.decoder.parameters():p.grad=torch.zeros_like(p)
        with self.assertRaisesRegex(ValueError,'Zero BS'):train.commit_update(codec,optimizer,scheduler,previous_updates=0)
        self.assertEqual(train.state_digest(codec.decoder),before)
        self.assertFalse(optimizer.state);self.assertEqual(scheduler.last_epoch,0)
        optimizer.param_groups[0]['weight_decay']*=2
        with self.assertRaisesRegex(ValueError,'optimizer recipe'):train.optimizer_check(optimizer,scheduler,0)

    def test_actual_monitor_restores_default_rng_modes_and_only_uses_validation(self):
        self.stop_at_zero();codec,stats,ce=self.loaded[-1]
        before=train.capture_rng();modes={name:m.training for name,m in codec.named_modules()}
        train.monitor_if_missing('fdma','w_residual',0,codec,stats,ce,self.data,self.directory(),forward=self.forward)
        after=train.capture_rng()
        self.assertEqual(before['python'],after['python'])
        self.assertEqual(before['numpy'][0],after['numpy'][0]);self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
        self.assertEqual(before['numpy'][2:],after['numpy'][2:]);self.assertTrue(torch.equal(before['torch_cpu'],after['torch_cpu']))
        self.assertEqual(modes,{name:m.training for name,m in codec.named_modules()})
        self.assertEqual(self.requests,[('validation',0)])
        receipt=train.read_json(self.directory()/'monitor/step_00000_receipt.json')
        evidence=next(Path(name) for name in receipt['files'] if name.endswith('summary.csv'))
        import csv
        with evidence.open() as stream:summary=list(csv.DictReader(stream))
        self.assertEqual(len(summary),6)
        self.assertTrue(all(int(row['n_samples'])==1 and int(row['n_batches'])==1 for row in summary))

    def test_wrong_actual_training_address_and_invalid_stop_budget_are_refused(self):
        for target in (0,99,101,201,True):
            with self.subTest(target=target),self.assertRaises(ValueError):self.run_train(target=target)
        original=self.data.address
        # The data returns a correct address; a changed expected identity cannot pass.
        sample=self.data.sample
        def wrong(role,index):
            value=sample(role,index)
            if role=='codec_train':value['address']=original('validation',index)
            return value
        self.data.sample=wrong
        with self.assertRaisesRegex(ValueError,'training data address changed'):
            self.run_train(monitor=lambda *a,**k:None)
        self.assertEqual(self.payload()['step'],0)


if __name__=='__main__':unittest.main()
