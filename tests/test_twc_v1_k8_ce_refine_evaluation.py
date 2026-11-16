"""Tiny real-CPU end-to-end export; no CUDA, Sionna or production weights."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from configs import twc_v1_k8_ce_refine_config as cfg
import ce_refine_runtime as runtime
import evaluate_twc_v1_k8_ce_refine as evaluate
import twc_v1_k8_ce_refine_results as statistics


class TinyCE(torch.nn.Module):
    def __init__(self,scale):
        super().__init__();self.gain=torch.nn.Parameter(torch.tensor(float(scale)))
    def forward(self,initial,gamma):
        # Deliberately consume default RNG to verify monitor/evaluation isolation.
        return self.gain*initial+torch.rand((),device=initial.device)*0


class TinyData:
    protocol={'namespace':cfg.NAMESPACE,'fixture':True,'batch_size':4}
    def __init__(self):self.calls=[]
    def address(self,role,index):return dict(data_id='fixture',role=role,index=index)
    def sample(self,role,index):
        self.calls.append((role,index))
        seed=int.from_bytes(hashlib.sha256(f'{role}:{index}'.encode()).digest()[:4],'big')
        gen=torch.Generator().manual_seed(seed)
        h=torch.complex(torch.randn(4,4,5,6,generator=gen),torch.randn(4,4,5,6,generator=gen))
        return dict(H_ul=h,H_dl=h.clone(),address=self.address(role,index),
            sample_fingerprint=hashlib.sha256(h.numpy().tobytes()).hexdigest())


def tiny_random(role,index,device,*,gamma=None):
    gen=torch.Generator().manual_seed(cfg.noise_seed(role,index))
    noise=torch.complex(torch.randn(4,4,5,6,generator=gen),torch.randn(4,4,5,6,generator=gen))/2**.5
    polarity=2*torch.randint(2,(4,4),generator=gen)-1
    return torch.full((4,),float(gamma)),noise,polarity


def tiny_forward(model,sample,gamma,noise,polarity):
    initial=sample['H_ul']+.12*noise
    y=initial.sum(1)
    pilots=polarity[...,None].expand(-1,-1,6).to(torch.complex64)
    return model(initial,gamma),initial,y,pilots


class CEReadOnlyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.data=TinyData()
        self.models={name:(TinyCE(scale),{'fixture':True}) for name,scale in
            zip(evaluate.VARIANTS,(1.03,1.01,.99))}
        self.layout=SimpleNamespace(groups=((0,1),(2,3)),
            shared_positions_by_group=((0,2,4),(1,3,5)))
        self.noise=mock.patch.object(runtime,'batch_random',side_effect=tiny_random);self.noise.start()
    def tearDown(self):self.noise.stop();self.tmp.cleanup()

    def test_complete_fixture_calibration_then_six_nodes_and_five_midpoints(self):
        before={k:runtime.original.state_digest(v[0]) for k,v in self.models.items()}
        rng=torch.get_rng_state().clone()
        result=evaluate.run_evaluation(self.root/'result',require_cuda=False,models=self.models,
            data=self.data,forward=tiny_forward,final_batches=2,calibration_batches=2,
            midpoint_batches=2,probe_layout=self.layout)
        self.assertFalse(result['production_evidence'])
        self.assertEqual(result['raw_rows'],3*11*8)
        self.assertEqual(set(result['files']),set(evaluate.PUBLIC_FILES))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,(model,_) in self.models.items():
            self.assertEqual(runtime.original.state_digest(model),before[name])
            self.assertTrue(model.training);self.assertTrue(model.gain.requires_grad)
        path=self.root/'result'
        provenance=json.loads((path/'provenance.json').read_text())
        self.assertFalse(provenance['production_evidence'])
        self.assertTrue(provenance['all_models_verified_before_data'])
        self.assertTrue(provenance['paired_inputs_verified'])
        self.assertEqual(provenance['freeze_before'],provenance['freeze_after'])
        self.assertEqual((provenance['UE_updates'],provenance['BS_updates'],provenance['FDMA_updates']), (0,0,0))
        cal=json.loads((path/'calibration.json').read_text());record_id=cal.pop('record_id')
        self.assertEqual(record_id,cfg.digest(cal))
        self.assertEqual(cal['samples_per_node'],8)
        raw=statistics.read_raw_csv(path/'samples.csv.gz')
        self.assertEqual(len(raw),264)
        for role in ('final','midpoint'):
            rows=[r for r in raw if r['role']==role]
            statistics.validate_rows(rows,samples=8,role=role)
        self.assertEqual(self.data.calls[:2],[('ce_calibration',0),('ce_calibration',1)])
        self.assertIn(('validation',cfg.FINAL_START),self.data.calls)
        self.assertIn(('validation',cfg.MIDPOINT_START),self.data.calls)
        self.assertFalse(any(role=='test' for role,_ in self.data.calls))
        for name,digest in result['files'].items():self.assertEqual(evaluate._sha(path/name),digest)

    def test_missing_fixed_model_refused_before_any_channel_generation(self):
        models=dict(self.models);models.pop('geometry65000')
        with self.assertRaises(ValueError):evaluate.run_evaluation(self.root/'bad',require_cuda=False,
            models=models,data=self.data,forward=tiny_forward,final_batches=1,
            calibration_batches=1,midpoint_batches=1,probe_layout=self.layout)
        self.assertEqual(self.data.calls,[])

    def test_monitor_restores_rng_modes_and_receipt_rejects_tampering_or_fixture_promotion(self):
        model=TinyCE(1.)
        before=runtime.original.state_digest(model);rng=torch.get_rng_state().clone()
        directory=self.root/'monitor'
        record=evaluate.monitor_model(model,self.data,'nmse',0,directory,
            production_evidence=False,forward=tiny_forward,batches=2)
        self.assertFalse(record['production_evidence'])
        self.assertEqual(record['model_state_sha256'],before)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training and model.gain.requires_grad)
        evaluate.verify_monitor(directory,trajectory='nmse',step=0,production_required=False)
        with self.assertRaises(ValueError):evaluate.verify_monitor(directory,production_required=True)
        with (directory/'summary.csv').open('a') as stream:stream.write('changed\n')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            evaluate.verify_monitor(directory,production_required=False)

    def test_same_observation_required_across_variants(self):
        def changed(model,sample,gamma,noise,polarity):
            pred,initial,y,pilot=tiny_forward(model,sample,gamma,noise,polarity)
            if float(model.gain.detach())<1.:y=y+1
            return pred,initial,y,pilot
        with self.assertRaisesRegex(ValueError,'identical physical'):
            evaluate.run_evaluation(self.root/'changed',require_cuda=False,models=self.models,
                data=self.data,forward=changed,final_batches=1,calibration_batches=1,
                midpoint_batches=1,probe_layout=self.layout)
        for model,_ in self.models.values():self.assertTrue(model.training and model.gain.requires_grad)

    def test_actual_channel_and_noise_overlap_are_refused_across_roles(self):
        first=dict(role='calibration',batch_index=0,gamma_db=0.,UL_channel='a'*64,standard_noise='b'*64)
        second=dict(role='final',batch_index=0,gamma_db=0.,UL_channel='c'*64,standard_noise='d'*64)
        self.assertEqual(evaluate._verify_disjoint_inputs([first,second])['cross_role_UL_collisions'],0)
        with self.assertRaisesRegex(ValueError,'UL channels overlap'):
            evaluate._verify_disjoint_inputs([first,dict(second,UL_channel=first['UL_channel'])])
        with self.assertRaisesRegex(ValueError,'noise realization reused'):
            evaluate._verify_disjoint_inputs([first,dict(second,standard_noise=first['standard_noise'])])

    def test_public_record_keeps_source_binding_but_omits_private_paths(self):
        private=dict(path='/private/model',frozen_files=[{'path':'/private/secret'}],
            source_identity=dict(checkpoint_sha256='a'*64,source_state_sha256='b'*64,source_commit='c'*40),
            counters=cfg.counters(cfg.UPDATES),model_state_sha256='d'*64,step=cfg.UPDATES)
        public=evaluate._public_record(private)
        self.assertNotIn('path',public);self.assertNotIn('frozen_files',public)
        self.assertEqual(public['source_checkpoint_sha256'],'a'*64)
        self.assertEqual(public['CE_total_updates'],65000)


if __name__=='__main__':unittest.main()
