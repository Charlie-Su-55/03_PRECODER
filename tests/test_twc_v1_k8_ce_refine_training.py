"""Small real CPU CE optimizer/save/replay tests; no production weights or Sionna."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
import train_twc_v1_k8_ce_refine as train


class TinyCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.tensor(.8))
        self.bias=torch.nn.Parameter(torch.tensor(.07))
        self.snr_gain=torch.nn.Parameter(torch.tensor(.015))
    def forward(self,observed,gamma):
        return observed*(self.scale+self.snr_gain*(gamma/25.)[:,None,None,None])+self.bias*(1+.3j)


class _SerializationFailure:
    def __reduce__(self):raise RuntimeError('intentional CE serialization failure')


class CEContinuationCPU(unittest.TestCase):
    def setUp(self):
        rng=train.capture_rng();self.addCleanup(train.restore_rng,rng)
        threads=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,threads)
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.output=self.root/'study'
        self.core=dict(schema=train.cfg.NAMESPACE+'_core_v1',sha256={'CPU_fixture':'a'*64})
        for patcher in (
            mock.patch.object(train.cfg,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.cfg,'UPDATES',4),mock.patch.object(train.cfg,'SAVE_EVERY',2),
            mock.patch.object(train.cfg,'MONITOR_EVERY',2),
            mock.patch.object(train.runtime,'core_identity',return_value=self.core),
            mock.patch.object(train.runtime,'source_identity',return_value=dict(commit='b'*40,sha256=self.core['sha256'])),
            mock.patch.object(train.runtime,'guard_core',side_effect=lambda core:self.assertEqual(core,self.core))):
            patcher.start();self.addCleanup(patcher.stop)
        self.base=TinyCE()
        self.source_path=self.root/'explicit_CPU_CE_source.pt'
        torch.save(self.base.state_dict(),self.source_path)
        self.source=dict(fixture=True,source_updates=5000,source_state_sha256=train.state_digest(self.base),
            frozen_files=[dict(path=str(self.source_path),sha256=train.sha256(self.source_path))])
        outer=self
        class Data:
            protocol=dict(fixture='tiny_pilot_CE',execution={'device':'cpu'})
            def address(self,role,index):return dict(namespace='tiny_fixture',role=role,index=index)
            def sample(self,role,index):
                if role!='ce_train':raise ValueError('No test role permitted')
                gen=torch.Generator().manual_seed(1000+index)
                h=torch.complex(torch.randn(4,2,3,5,generator=gen),torch.randn(4,2,3,5,generator=gen))
                return dict(H_ul=h,address=self.address(role,index),sample_fingerprint=train.state_digest({'H_ul':h}))
        self.data=Data()
        def loader(device):return copy.deepcopy(self.base).to(device),copy.deepcopy(self.source)
        self.loader=loader
        def forward(model,sample,gamma,noise,polarity):
            # Tiny orthogonal-pilot observation, with a separate noise draw already
            # addressed by the production random helper. Model never sees truth.
            perturb=.001*(random.random()+float(np.random.random())+float(torch.rand(())))
            n=noise[:,:3,:10].reshape(4,3,2,5).permute(0,2,1,3)
            observed=sample['H_ul']+(.25*10**(-gamma/10.)).sqrt()[:,None,None,None]*n+perturb
            pilots=torch.ones(4,2,5,dtype=torch.complex64)
            return model(observed,gamma),observed,observed,pilots
        self.forward=forward
        def monitor(model,data,trajectory,step,output_dir):
            random.random();np.random.random();torch.rand(2)
        self.monitor=monitor

    def run_train(self,trajectory='nmse',**kwargs):
        return train.train(trajectory,require_cuda=False,loader=self.loader,
            data_factory=lambda device:self.data,forward=self.forward,monitor=self.monitor,**kwargs)

    def payload(self,trajectory='nmse'):
        return torch.load(self.output/'training'/trajectory/'checkpoint_latest.pt',weights_only=False,map_location='cpu')

    def test_real_CPU_split_resume_exactly_matches_uninterrupted(self):
        self.run_train(stop_after=4)
        reference=self.payload();source_bytes=self.source_path.read_bytes()
        old=self.output;self.output=self.root/'split'
        train.cfg.OUTPUT_ROOT=self.output
        first=self.run_train(stop_after=2)
        self.assertEqual(first['counters']['CE_total_updates'],5002)
        self.run_train(stop_after=4,resume=True)
        actual=self.payload()
        self.assertEqual(actual['training_trace'],reference['training_trace'])
        self.assertEqual(train.state_digest(actual['model']),train.state_digest(reference['model']))
        for key in ('optimizer','scheduler'):
            if key=='optimizer':
                for pid,values in actual[key]['state'].items():
                    for name,value in values.items():torch.testing.assert_close(value,reference[key]['state'][pid][name],rtol=0,atol=0)
            else:self.assertEqual(actual[key],reference[key])
        self.assertEqual(source_bytes,self.source_path.read_bytes())
        self.assertFalse(actual['production_evidence'])
        self.assertTrue(actual['fixture'])
        self.assertEqual(actual['counters']['UE_updates'],0)
        self.assertEqual(actual['counters']['BS_updates'],0)
        self.assertEqual(actual['counters']['EMA_commits'],0)
        self.assertTrue((self.output/'training/nmse/checkpoint_step_00000.pt').is_file())
        self.assertTrue((self.output/'training/nmse/checkpoint_step_00002.pt').is_file())
        self.assertTrue((self.output/'training/nmse/checkpoint_step_00004.pt').is_file())
        self.assertEqual(set(actual['rng']),{'python','numpy','torch_cpu','torch_cuda'})

    def test_objectives_have_identical_actual_input_trace_but_different_weights(self):
        self.run_train('nmse',stop_after=4)
        self.run_train('geometry',stop_after=4)
        a,b=self.payload('nmse'),self.payload('geometry')
        self.assertEqual(a['training_trace'],b['training_trace'])
        self.assertNotEqual(train.state_digest(a['model']),train.state_digest(b['model']))

    def test_resume_rejects_changed_optimizer_rng_and_provenance(self):
        self.run_train(stop_after=2)
        directory=self.output/'training/nmse'
        payload=self.payload();record=json.loads((directory/'manifest.json').read_text())
        for field,value in [('fingerprint','changed'),('source_identity',{}),('rng',{}),('counters',{}),('fixture',False)]:
            changed=copy.deepcopy(payload);changed[field]=value
            with self.assertRaises(ValueError):
                train.verify_saved(directory,changed,record,trajectory='nmse',data=self.data,
                    core=self.core,source=self.source,allow_fixture=True)
        payload['optimizer']['param_groups'][0]['lr']=.5
        train.runtime.original.atomic_torch_save(payload,directory/'checkpoint_latest.pt')
        record['checkpoint_sha256']=train.sha256(directory/'checkpoint_latest.pt')
        (directory/'manifest.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'optimizer recipe'):
            self.run_train(stop_after=4,resume=True)

    def test_real_failed_save_preserves_checkpoint_and_cleans_temporary_file(self):
        self.run_train(stop_after=2)
        directory=self.output/'training/nmse';before=(directory/'checkpoint_latest.pt').read_bytes()
        names={p.name for p in directory.iterdir()}
        model=copy.deepcopy(self.base)
        optimizer=torch.optim.AdamW(model.parameters(),lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=train.cfg.UPDATES,eta_min=train.cfg.MIN_LR)
        source=copy.deepcopy(self.source);source['cannot_pickle']=_SerializationFailure()
        with self.assertRaisesRegex(RuntimeError,'intentional CE serialization'):
            train.save_checkpoint(directory,model,optimizer,scheduler,trajectory='nmse',step=0,
                data=self.data,core=self.core,source=source,trace=train.initial_trace(self.data.protocol),fixture=True)
        self.assertEqual(before,(directory/'checkpoint_latest.pt').read_bytes())
        self.assertEqual(names,{p.name for p in directory.iterdir()})

    def test_nonfinite_gradient_does_not_commit_optimizer_or_save(self):
        model=copy.deepcopy(self.base)
        optimizer=torch.optim.AdamW(model.parameters(),lr=train.cfg.LR,weight_decay=train.cfg.WEIGHT_DECAY)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=train.cfg.UPDATES,eta_min=train.cfg.MIN_LR)
        before=train.state_digest(model)
        for p in model.parameters():p.grad=torch.full_like(p,float('nan'))
        with self.assertRaises(FloatingPointError):train.commit_update(model,optimizer,scheduler,0)
        self.assertEqual(before,train.state_digest(model));self.assertEqual(scheduler.last_epoch,0)
        self.assertEqual(len(optimizer.state),0)

    def test_source_change_and_implicit_resume_are_refused(self):
        self.run_train(stop_after=2)
        with self.assertRaises(FileExistsError):self.run_train(stop_after=4)
        self.source_path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source file changed'):
            self.run_train(stop_after=4,resume=True)

    def test_fixture_is_refused_as_production_terminal(self):
        self.run_train(stop_after=4)
        p=self.payload();directory=self.output/'training/nmse'
        record=json.loads((directory/'manifest.json').read_text())
        with self.assertRaisesRegex(ValueError,'not production'):
            train.verify_saved(directory,p,record,trajectory='nmse',data=self.data,
                core=self.core,source=self.source,allow_fixture=False)


if __name__=='__main__':unittest.main()
