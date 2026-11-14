"""Real CPU complete-MU rates, fresh AdamW and exact saved-boundary replay."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import train_twc_v1_k8_ul_adapt as train
from evaluate_twc_v1_k8_icc_aligned import _tensor_hash


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coupling=torch.nn.Parameter(torch.tensor(.35))
        self.snr_slope=torch.nn.Parameter(torch.tensor(.03))
        base=torch.zeros(1,8,32,256,dtype=torch.complex64)
        base[:,torch.arange(8),torch.arange(8),:]=1.
        mix=torch.zeros_like(base);mix[:,0,1,:]=1.;mix[:,1,0,:]=1.
        self.register_buffer('base',base);self.register_buffer('mix',mix)

    def forward(self,gamma):
        gain=self.coupling+self.snr_slope*(gamma/25.-.5)
        return self.base+self.mix*gain[:,None,None,None]


class TinyStatistics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('historical_power',torch.full((8,24),2.))
        self.register_buffer('ema_updates',torch.tensor(20000))

    def nominal_power(self):return self.historical_power


class AdaptationTrainingTests(unittest.TestCase):
    def setUp(self):
        saved=train.capture_rng();self.addCleanup(train.restore_rng,saved)
        threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.output=self.root/'adapt'
        self.core=dict(schema=train.cfg.NAMESPACE+'_core_v1',sha256={'CPU_fixture':'a'*64})
        for patcher in (
            mock.patch.object(train.cfg,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.runtime,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.cfg,'UPDATES',4),
            mock.patch.object(train.cfg,'SAVE_EVERY',2),
            mock.patch.object(train.cfg,'ACCUMULATION',2),
            mock.patch.object(train.cfg,'MONITOR_BATCHES',1),
            mock.patch.object(train.cfg,'MONITOR_STEPS',(0,2,4)),
            mock.patch.object(train.cfg,'ROLE_RANGES',dict(codec_train=[1,8],monitor=[0,0],final=[2048,2303])),
            mock.patch.object(train.runtime,'core_identity',return_value=self.core),
            mock.patch.object(train.runtime,'source_identity',return_value=dict(commit='b'*40,sha256=self.core['sha256'])),
            mock.patch.object(train.runtime,'guard_core',side_effect=lambda value:self.assertEqual(value,self.core))):
            patcher.start();self.addCleanup(patcher.stop)
        torch.manual_seed(739)
        codec=torch.nn.Module();codec.encoder=torch.nn.Linear(2,3);codec.decoder=TinyDecoder()
        old_optimizer=torch.optim.AdamW(codec.decoder.parameters(),lr=1e-4)
        (codec.decoder.coupling.square()+codec.decoder.snr_slope.square()).backward()
        old_optimizer.step();old_optimizer.zero_grad(set_to_none=True)
        stats,ce=TinyStatistics(),torch.nn.Linear(2,2)
        for module in (codec,stats,ce):module.eval().requires_grad_(False)
        self.base=codec,stats,ce
        self.source_path=self.root/'explicit_cpu_fixture.pt'
        torch.save(dict(model=codec.state_dict(),optimizer=old_optimizer.state_dict()),self.source_path)
        self.source_bytes=self.source_path.read_bytes()
        self.source=dict(arm='fdma',base_decoder_state_sha256=train.state_digest(codec.decoder),
            calibration_model_sha256=train.cfg.CALIBRATION_MODEL_SHA256,
            frozen_files=[dict(path=str(self.source_path),sha256=train.sha256(self.source_path))])
        self.loaded=[];self.calls=[];self.requests=[]
        outer=self

        class Data:
            protocol={'fixture':'real CPU8UE32antenna256tone; no production evidence'}
            def address(self,role,index):return dict(namespace='fixture',role=role,index=index)
            def sample(self,role,index):
                if role not in ('codec_train','validation'):raise AssertionError('Unrelated data role read')
                outer.requests.append((role,index))
                h=outer.base[0].decoder.base.expand(4,-1,-1,-1).clone()
                return dict(H_dl=h,H_ul=h.clone(),address=self.address(role,index),
                    sample_fingerprint=train.cfg.digest([role,index]))
        self.data=Data()
        # This seam accepts an explicit synthetic file, not a real source
        # checkpoint. The actual serialization, neural gradients and rate
        # objective are never mocked; fixtures remain nonproduction.
        patcher=mock.patch.object(train.runtime,'verify_files',side_effect=self.check_files)
        patcher.start();self.addCleanup(patcher.stop)

    def check_files(self,source):
        for item in source['frozen_files']:
            if train.sha256(item['path'])!=item['sha256']:
                raise ValueError('Fixture source bytes changed')
        if source.get('calibration_model_sha256')!=train.cfg.CALIBRATION_MODEL_SHA256:
            raise ValueError('Calibration identity changed')

    def loader(self,arm,method,device):
        self.assertEqual((arm,str(device)),('fdma','cpu'))
        self.assertIn(method,train.cfg.TRAIN_METHODS)
        modules=copy.deepcopy(self.base);self.loaded.append(modules)
        return (*modules,copy.deepcopy(self.source))

    def forward(self,codec,stats,ce,sample,gamma_db,noise_key,*,method,calibration=None):
        role,index,gi=noise_key
        self.calls.append((role,index,method,codec.decoder.training,torch.is_grad_enabled()))
        self.assertFalse(codec.encoder.training or ce.training or stats.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in codec.encoder.parameters()))
        self.assertEqual(codec.decoder.training,role=='codec_train')
        self.assertEqual(torch.is_grad_enabled(),role=='codec_train')
        # Monitor draws exercise exact global RNG restoration on resume.
        random.random();np.random.rand();torch.rand(1)
        gamma=torch.tensor(train.cfg.gamma_schedule(index),dtype=torch.float32) if gamma_db is None else torch.full((4,),float(gamma_db))
        active=torch.full((4,8,24),2**.5,dtype=torch.complex64)
        grid=torch.zeros(4,8,256,dtype=torch.complex64);grid[...,:32]=2**.5
        noise=torch.zeros(4,32,256,dtype=torch.complex64);polarity=torch.ones(4,8)
        h=codec.decoder(gamma)
        if method=='moment_receiver':h=h+.01*codec.decoder.mix
        return dict(H_hat=h,soft=active*(1.+.01j),H_ul_est=sample['H_ul'],
            reliability=torch.ones(4,8,216),waveform=SimpleNamespace(active=active,x_grid=grid),
            gamma_db=gamma,input_hashes=dict(standard_noise=_tensor_hash(noise),
                pilot_polarity=_tensor_hash(polarity),gamma=_tensor_hash(gamma)))

    def run_train(self,method='moment_receiver',target=2,resume=False,**kwargs):
        return train.train('fdma',method,target,resume,require_cuda=False,
            loader=self.loader,data_factory=lambda _:self.data,forward=self.forward,**kwargs)

    def directory(self,method='moment_receiver'):
        return self.output/'training'/train.cfg.key('fdma',method)

    def payload(self,method='moment_receiver'):
        return torch.load(self.directory(method)/'checkpoint_latest.pt',map_location='cpu',weights_only=False)

    def verify(self,payload,manifest,method='moment_receiver',**kwargs):
        codec,stats,ce=self.loaded[-1]
        return train.verify_saved(self.directory(method),payload,manifest,arm='fdma',method=method,
            data=self.data,core=self.core,source=self.source,
            frozen=train.helpers().frozen_state(codec,ce,stats),allow_fixture=True,**kwargs)

    def zero_boundary(self):
        def fail(*args,**kwargs):raise RuntimeError('intentional monitor0 stop')
        with self.assertRaisesRegex(RuntimeError,'monitor0'):
            self.run_train(monitor=fail)
        return train.read_json(self.directory()/'manifest.json')

    def test_real_rate_gradients_exact_resume_and_frozen_sources(self):
        first=self.run_train();payload=self.payload()
        self.assertEqual((first['step'],first['status']),(2,'paused_boundary'))
        self.assertFalse(first['production_evidence'])
        self.assertEqual(payload['evidence_kind'],'test_fixture')
        self.assertEqual(payload['calibration_identity'],train.cfg.calibration_identity())
        self.assertEqual(payload['counters'],train.cfg.counters(2))
        self.assertEqual(payload['training_trace']['microbatches'],4)
        self.assertEqual(len(payload['optimizer']['state']),2)
        self.assertTrue(all(float(s['step'])==2 for s in payload['optimizer']['state'].values()))
        self.assertNotEqual(float(payload['model']['coupling']),float(self.base[0].decoder.coupling))
        self.assertEqual(self.verify(payload,first),2)
        final=self.run_train(target=4,resume=True);resumed=self.payload()
        self.assertEqual((final['step'],final['status']),(4,'completed'))
        self.assertEqual(self.verify(resumed,final,require_final=True),4)
        self.assertEqual(train.read_json(self.directory()/'terminal_receipt.json'),final)
        progress=[json.loads(line) for line in (self.directory()/'progress.jsonl').read_text().splitlines()]
        for row in progress:
            trace=row['training_trace_before_update']
            for entry in row['input_fingerprints']:
                trace=train.advance_training_trace(trace,entry,self.data.protocol)
            self.assertEqual(trace,row['training_trace'])
        self.assertEqual([i for role,i in self.requests if role=='codec_train'],list(range(1,9)))
        for step in (0,2,4):
            receipt=train.read_json(self.directory()/'monitor'/f'step_{step:05d}_receipt.json')
            self.assertFalse(receipt['selection'])
            self.assertFalse(receipt['production_evidence'])
            self.assertTrue(all(train.sha256(p)==s for p,s in receipt['files'].items()))
        for modules in self.loaded:
            train.helpers().assert_frozen_state(modules[0],modules[2],modules[1],
                train.helpers().frozen_state(self.base[0],self.base[2],self.base[1]))
        self.assertEqual(self.source_path.read_bytes(),self.source_bytes)
        other=self.root/'uninterrupted'
        with mock.patch.object(train.cfg,'OUTPUT_ROOT',other),mock.patch.object(train.runtime,'OUTPUT_ROOT',other):
            self.run_train(target=4)
            continuous=self.payload()
        self.assertEqual(train.state_digest(resumed['model']),train.state_digest(continuous['model']))
        self.assertEqual(resumed['training_trace'],continuous['training_trace'])
        for identifier,state in resumed['optimizer']['state'].items():
            for field in ('step','exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(state[field],continuous['optimizer']['state'][identifier][field]))

    def test_original_control_has_identical_initial_decoder_and_matching_budget(self):
        moment=self.run_train();original=self.run_train('original')
        self.assertEqual(moment['counters'],original['counters'])
        self.assertEqual(moment['training_trace'],original['training_trace'])
        for method in train.cfg.TRAIN_METHODS:
            init=train.read_json(self.directory(method)/'initialization.json')
            self.assertEqual(init['initial_decoder_sha256'],self.source['base_decoder_state_sha256'])
        self.assertEqual(self.source_path.read_bytes(),self.source_bytes)

    def test_step_zero_failure_is_resumable_with_fresh_optimizer(self):
        manifest=self.zero_boundary();payload=self.payload()
        self.assertEqual((payload['step'],payload['optimizer']['state']),(0,{}))
        self.assertEqual(self.verify(payload,manifest),0)
        self.assertEqual(self.requests,[])
        self.run_train(resume=True)
        self.assertEqual([i for role,i in self.requests if role=='codec_train'],[1,2,3,4])

    def test_fixture_terminal_is_refused_before_any_real_source_load(self):
        self.run_train(target=4)
        with mock.patch.object(train.runtime,'load_initial',side_effect=AssertionError('must not load production source')):
            with self.assertRaisesRegex(ValueError,'production'):
                train.load_finetuned('fdma','moment_receiver','cpu')

    def test_resume_rejects_changed_method_calibration_counts_optimizer_and_rng(self):
        manifest=self.run_train();payload=self.payload()
        changes=[('method','original'),('calibration_identity',{'model_sha256':'f'*64}),
            ('counters',dict(payload['counters'],EMA_commits=1)),
            ('rng',dict(payload['rng'],torch_cpu=torch.tensor([1],dtype=torch.uint8)))]
        for key,value in changes:
            changed=copy.deepcopy(payload);changed[key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):self.verify(changed,manifest)
        for field,value in [('step',torch.tensor(99.)),('exp_avg',torch.zeros(2)),
                ('exp_avg_sq',torch.tensor(float('nan')))]:
            changed=copy.deepcopy(payload)
            changed['optimizer']['state'][0][field]=value
            with self.subTest(moment=field),self.assertRaises(ValueError):self.verify(changed,manifest)
        changed=copy.deepcopy(payload);changed['scheduler']['last_epoch']=1
        with self.assertRaises(ValueError):self.verify(changed,manifest)

    def test_real_serialization_failure_preserves_existing_checkpoint_and_receipts(self):
        self.zero_boundary();directory=self.directory()
        before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
        codec,stats,ce=self.loaded[-1]
        parameters=train.helpers().configure_bs_only(codec,ce,stats)
        optimizer=torch.optim.AdamW(parameters,lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        invalid=dict(self.source,unpicklable=lambda:None)
        with self.assertRaises((AttributeError,TypeError)):
            train.save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,arm='fdma',method='moment_receiver',step=0,
                data=self.data,core=self.core,source=invalid,frozen=train.helpers().frozen_state(codec,ce,stats),
                elapsed=0.,measured=0.,production_evidence=False)
        self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()})
        self.assertEqual(list(directory.glob('checkpoint_*.pt')), [directory/'checkpoint_latest.pt'])

    def test_nonfinite_gradient_and_wrong_optimizer_are_refused_without_update(self):
        self.zero_boundary();codec,stats,ce=self.loaded[-1]
        parameters=train.helpers().configure_bs_only(codec,ce,stats)
        optimizer=torch.optim.AdamW(parameters,lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        before=train.state_digest(codec.decoder)
        for parameter in parameters:parameter.grad=torch.full_like(parameter,float('nan'))
        with self.assertRaises(FloatingPointError):train.commit_update(codec,optimizer,scheduler,previous_updates=0)
        self.assertEqual(train.state_digest(codec.decoder),before)
        for parameter in parameters:parameter.grad=torch.ones_like(parameter)
        optimizer.param_groups[0]['params']=list(parameters[:1])
        with self.assertRaises(ValueError):train.commit_update(codec,optimizer,scheduler,previous_updates=0)

    def test_source_initial_decoder_change_is_rejected_before_output_creation(self):
        original=self.loader
        def wrong(arm,method,device):
            codec,stats,ce,source=original(arm,method,device)
            with torch.no_grad():codec.decoder.coupling.add_(.1)
            return codec,stats,ce,source
        with self.assertRaisesRegex(ValueError,'exact source decoder'):
            train.train('fdma','original',2,require_cuda=False,loader=wrong,
                data_factory=lambda _:self.data,forward=self.forward)
        self.assertFalse(self.output.exists())

    def test_real_100_boundary_receipt_survives_terminal_checkpoint_replacement(self):
        with mock.patch.object(train.cfg,'UPDATES',200),mock.patch.object(train.cfg,'SAVE_EVERY',100),\
                mock.patch.object(train.cfg,'ACCUMULATION',1),\
                mock.patch.object(train.cfg,'ROLE_RANGES',dict(codec_train=[1,200],monitor=[0,0],final=[2048,2303])):
            quiet=lambda *args,**kwargs:None
            self.run_train(target=100,monitor=quiet)
            boundary=self.directory()/'boundary_00100.json'
            before=boundary.read_bytes()
            hundred=train.read_json(boundary)
            self.assertEqual(hundred['step'],100)
            self.assertEqual(hundred['calibration_identity'],train.cfg.calibration_identity())
            self.run_train(target=200,resume=True,monitor=quiet)
            self.assertEqual(boundary.read_bytes(),before)
            self.assertNotEqual(hundred['checkpoint_sha256'],train.sha256(self.directory()/'checkpoint_latest.pt'))
            self.assertEqual(train.read_json(self.directory()/'terminal_receipt.json')['step'],200)

    def test_actual_gamma_skew_is_rejected_before_optimizer_update(self):
        original=self.forward
        def wrong(*args,**kwargs):
            out=original(*args,**kwargs)
            if kwargs['noise_key'][0]=='codec_train':out['gamma_db']=torch.zeros_like(out['gamma_db'])
            return out
        with self.assertRaisesRegex(ValueError,'balanced schedule'):
            train.train('fdma','original',2,require_cuda=False,loader=self.loader,
                data_factory=lambda _:self.data,forward=wrong,monitor=lambda *a,**k:None)
        payload=self.payload('original')
        self.assertEqual(payload['step'],0)
        self.assertEqual(payload['optimizer']['state'],{})

    def test_terminal_validation_does_not_initialize_cuda_or_change_global_rng(self):
        manifest=self.run_train();payload=self.payload()
        before=train.capture_rng()
        with mock.patch.object(torch.cuda,'get_rng_state_all',side_effect=AssertionError('must not touch CUDA')):
            self.assertEqual(self.verify(payload,manifest),2)
        after=train.capture_rng()
        self.assertEqual(before['python'],after['python'])
        self.assertTrue(torch.equal(before['torch_cpu'],after['torch_cpu']))
        self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))

    def test_production_terminal_routes_through_actual_source_device(self):
        def strict_source(arm,method,device):
            if str(device)!='cuda:0':
                raise ValueError('Historical source data protocol changed')
            return object(),object(),object(),{'production_evidence':True}
        with mock.patch.object(train.runtime,'configure_cuda',return_value=torch.device('cuda:0')) as configure,\
                mock.patch.object(train,'load_finetuned',side_effect=strict_source) as load:
            record=train.verify_terminal('fdma','moment_receiver')
        self.assertTrue(record['production_evidence'])
        configure.assert_called_once_with()
        self.assertEqual(str(load.call_args.args[2]),'cuda:0')

    def test_real_replace_failure_preserves_saved_checkpoint(self):
        self.zero_boundary();directory=self.directory()
        before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
        codec,stats,ce=self.loaded[-1]
        parameters=train.helpers().configure_bs_only(codec,ce,stats)
        optimizer=torch.optim.AdamW(parameters,lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY,betas=(.9,.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        with mock.patch('os.replace',side_effect=OSError('intentional atomic replacement failure')):
            with self.assertRaises(OSError):
                train.save_checkpoint(directory,codec,stats,ce,optimizer,scheduler,arm='fdma',method='moment_receiver',step=0,
                    data=self.data,core=self.core,source=self.source,frozen=train.helpers().frozen_state(codec,ce,stats),
                    elapsed=0.,measured=0.,production_evidence=False)
        self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()})

    def test_rolling_trace_covers_actual_inputs_and_rejects_cursor_corruption(self):
        protocol=self.data.protocol
        initial=train.initial_training_trace(protocol)
        first=dict(micro_index=1,address=self.data.address('codec_train',1),
            channel='a'*64,standard_noise='b'*64,pilot_polarity='c'*64,gamma='d'*64,
            wave_active='0'*64)
        actual=train.advance_training_trace(initial,first,protocol)
        self.assertEqual(actual['microbatches'],1)
        self.assertNotEqual(actual['sha256'],initial['sha256'])
        excluded=train.advance_training_trace(initial,dict(first,wave_active='f'*64),protocol)
        self.assertEqual(excluded,actual)
        for field in ('channel','standard_noise','pilot_polarity','gamma'):
            changed=train.advance_training_trace(initial,dict(first,**{field:'e'*64}),protocol)
            self.assertNotEqual(changed['sha256'],actual['sha256'])
        for changed in (dict(first,micro_index=2),dict(first,gamma='not a tensor hash'),
                dict(first,address=self.data.address('validation',1))):
            with self.assertRaises(ValueError):train.advance_training_trace(initial,changed,protocol)
        with self.assertRaises(ValueError):train.advance_training_trace(actual,first,protocol)
        with self.assertRaises(ValueError):train.validate_training_trace(dict(actual,microbatches=2),protocol,microbatches=1)
        with self.assertRaises(ValueError):train.validate_training_trace(dict(actual,sha256=initial['sha256']),protocol,microbatches=1)

    def test_bool_cannot_impersonate_integer_in_saved_schema_or_manifest(self):
        manifest=self.zero_boundary();payload=self.payload()
        changed=copy.deepcopy(payload)
        changed['model_schema']['base']['shape'][0]=True
        with self.assertRaisesRegex(ValueError,'schema'):
            self.verify(changed,manifest)
        changed_manifest=copy.deepcopy(manifest)
        changed_manifest['counters']['EMA_commits']=False
        train.atomic_json(self.directory()/'manifest.json',changed_manifest)
        with self.assertRaisesRegex(ValueError,'manifest identity'):
            self.verify(payload,changed_manifest)


if __name__=='__main__':unittest.main()
