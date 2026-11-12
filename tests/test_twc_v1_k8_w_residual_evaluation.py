"""Real CPU scoring of transmitted W and frozen six-method paired evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from configs import twc_v1_k8_w_residual_config as cfg
import evaluate_twc_v1_k8_w_residual as evaluation

try:
    import torch
    import numpy as np
except ImportError:
    torch=np=None


def fingerprints(batches=1):
    result=[]
    for method in cfg.METHODS:
        for bi in range(batches):
            for gamma in cfg.GAMMAS:
                digest=hashlib.sha256(f'{bi}:{gamma}'.encode()).hexdigest()
                result.append(dict(method=method,batch_index=bi,gamma_db=float(gamma),
                    fingerprint={name:digest for name in ('H_dl','H_ul','standard_noise','pilot_polarity')}))
    return result


class InputEvidenceTests(unittest.TestCase):
    def test_all_six_methods_require_complete_equal_actual_input_hashes(self):
        rows=fingerprints()
        self.assertEqual(len(evaluation.fingerprint_map(rows,batches=1)),len(cfg.METHODS)*len(cfg.GAMMAS))
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            evaluation.fingerprint_map(rows[:-1],batches=1)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            evaluation.fingerprint_map(rows+[rows[0]],batches=1)
        changed=copy.deepcopy(rows);changed[-1]['fingerprint']['standard_noise']='f'*64
        with self.assertRaisesRegex(ValueError,'different actual inputs'):
            evaluation.fingerprint_map(changed,batches=1)

    def test_wrong_phase_path_cannot_create_outputs_or_load_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);runtime=SimpleNamespace(inside=lambda path:Path(path).resolve())
            loader=mock.Mock(side_effect=AssertionError('Wrong phase must not load a model'))
            with mock.patch.object(cfg,'OUTPUT_ROOT',root),mock.patch.object(evaluation,'_runtime',return_value=runtime):
                for phase,path in (('preflight',root/'preflight'),('preflight',root/'evaluation/final/attempt'),
                                   ('final',root/'evaluation/final'),('final',root/'preflight/attempt'),
                                   ('final',root/'other/attempt')):
                    with self.subTest(phase=phase,path=path):
                        with self.assertRaisesRegex(ValueError,'phase subtree'):
                            evaluation._checked_output(path,phase)
                        self.assertFalse(path.exists())
                with self.assertRaisesRegex(ValueError,'phase subtree'):
                    evaluation.evaluate_final(root/'other/attempt',require_cuda=False,loader=loader)
                loader.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])

    def test_actual_sample_address_is_phase_checked_without_sampling_test(self):
        import w_residual_runtime as runtime
        metadata=runtime.make_data('cpu');seen=[]
        class Data:
            def sample(self,role,index):
                seen.append((role,index));return dict(address=metadata.address('validation',0))
        with self.assertRaisesRegex(ValueError,'declared phase'):
            evaluation._sample(Data(),metadata,'final',0)
        self.assertEqual(seen,[('validation',2048)])
        self.assertEqual(evaluation._sample(Data(),metadata,'monitor',0)['address'],metadata.address('validation',0))
        with self.assertRaisesRegex(ValueError,'outside'):
            evaluation._sample(Data(),metadata,'final',256)

    def test_preflight_cli_delegates_with_cuda_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'preflight/attempt_cli'
            preflight=mock.Mock()
            runtime=SimpleNamespace(inside=lambda path:Path(path).resolve(),preflight_sources=preflight)
            with mock.patch.object(cfg,'OUTPUT_ROOT',root),mock.patch.object(evaluation,'_runtime',return_value=runtime):
                self.assertEqual(evaluation.main(['--run','--phase','preflight','--output-dir',str(output)]),0)
            preflight.assert_called_once_with(output,require_cuda=True)
            self.assertFalse(output.exists())

    def test_production_final_requires_training_pairing_before_outputs_or_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'evaluation/final/attempt_missing_pairing'
            pairing=mock.Mock(side_effect=ValueError('Actual training-input pairing evidence missing'))
            loader=mock.Mock(side_effect=AssertionError('Unpaired training must not load a model'))
            cuda=mock.Mock(side_effect=AssertionError('Pairing must be checked before CUDA setup'))
            runtime=SimpleNamespace(inside=lambda path:Path(path).resolve(),
                training_pairing_evidence=pairing,load_model=loader,configure_cuda=cuda)
            with mock.patch.object(cfg,'OUTPUT_ROOT',root),mock.patch.object(evaluation,'_runtime',return_value=runtime):
                with self.assertRaisesRegex(ValueError,'training-input pairing evidence missing'):
                    evaluation.evaluate_final(output)
            pairing.assert_called_once_with();loader.assert_not_called();cuda.assert_not_called()
            self.assertFalse(output.exists());self.assertEqual(list(root.iterdir()),[])


@unittest.skipIf(torch is None or np is None,'Real PyTorch/NumPy unavailable; no fake dependency')
class FrozenDirectWCPU(unittest.TestCase):
    def setUp(self):
        import w_residual_runtime as real_runtime
        from train_twc_v1_k8_icc_aligned import load_local
        self.downlink=load_local('utils/twc_v1_k8_downlink.py','_w_residual_cpu_test_downlink')
        original_rng=torch.get_rng_state().clone();self.addCleanup(torch.set_rng_state,original_rng)
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.metadata=real_runtime.make_data('cpu')
        self.source=self.root/'source_bytes.bin';self.source.write_bytes(b'preserved source checkpoint fixture')
        self.source_before=self.source.read_bytes();self.core=dict(schema='real_CPU_fixture',sha256={'fixture':'a'*64})
        self.loaded=[];self.seen_models=[];self.samples=[];self.forward_calls=[]
        def inside(path):
            path=Path(path).resolve()
            if path==self.root or not path.is_relative_to(self.root):raise ValueError('Owned child required')
            return path
        def atomic(path,value,exclusive=False):
            with inside(path).open('x' if exclusive else 'w') as stream:
                json.dump(value,stream,sort_keys=True,allow_nan=False)
        def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        def verify(record):
            for item in record['frozen_files']:
                if sha(item['path'])!=item['sha256']:raise ValueError('Source bytes changed')
        self.runtime=SimpleNamespace(OUTPUT_ROOT=self.root,inside=inside,atomic_json=atomic,sha256=sha,
            core_identity=lambda:copy.deepcopy(self.core),guard_core=lambda core:self.assertEqual(core,self.core),
            make_data=lambda device:self.metadata,verify_files=verify)
        for patch in (mock.patch.object(evaluation,'_runtime',return_value=self.runtime),
                      mock.patch.object(cfg,'OUTPUT_ROOT',self.root),mock.patch.object(cfg,'FINAL_BATCHES',1)):
            patch.start();self.addCleanup(patch.stop)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(492011)
            self.templates={}
            for arm in cfg.ARMS:
                codec=torch.nn.Module();codec.encoder=torch.nn.Linear(2,2);codec.bs=torch.nn.Linear(2,2)
                codec.residual_head=torch.nn.Linear(2,1);codec.register_buffer('counter',torch.tensor(0))
                ce=torch.nn.Linear(2,2);stats=torch.nn.Module()
                stats.register_buffer('historical',torch.ones(8,24 if arm=='fdma' else 44))
                stats.register_buffer('ema_updates',torch.tensor(20000))
                self.templates[arm]=(codec,stats,ce)

    def loader(self,policy,arm,device):
        self.assertEqual(str(device),'cpu');self.loaded.append((policy,arm))
        codec,stats,ce=copy.deepcopy(self.templates[arm]);codec.policy=policy;codec.arm=arm
        before={name:module_state(module) for name,module in (('codec',codec),('stats',stats),('ce',ce))}
        self.seen_models.append((codec,stats,ce,before))
        record=dict(policy=policy,arm=arm,frozen_files=[dict(path=str(self.source),sha256=self.runtime.sha256(self.source))])
        return codec,stats,ce,record

    def data(self,device):
        outer=self
        class Data:
            protocol=outer.metadata.protocol
            def sample(self,role,index):
                outer.samples.append((role,index));generator=torch.Generator().manual_seed(5501+index)
                identity=torch.eye(8,32,dtype=torch.complex64)[None,:,:,None].expand(4,8,32,256).clone()
                perturb=torch.complex(torch.randn(identity.shape,generator=generator),
                                      torch.randn(identity.shape,generator=generator))*.02
                truth=identity+perturb
                return dict(H_dl=truth,H_ul=truth*1.2,address=outer.metadata.address(role,index))
        return Data()

    def forward(self,codec,stats,ce,sample,*,gamma_db,noise_key):
        self.assertFalse(torch.is_grad_enabled())
        for module in (codec,stats,ce):self.assertFalse(module.training)
        self.forward_calls.append((codec.policy,codec.arm,gamma_db,noise_key))
        features=codec.bs(codec.encoder(torch.ones(4,2)))
        estimate=sample['H_dl'].clone()+features[:,0,None,None,None]*.003
        base,_=self.downlink.design_rzf(estimate,alpha=.03)
        delta=torch.zeros_like(base)
        if codec.policy=='w_residual':
            scale=codec.residual_head(features).tanh()[:,0,None,None,None]*.05
            delta=scale*(base.roll(1,dims=-1)-base)
        W=base+delta;W=W/W.abs().square().sum(dim=(-2,-1),keepdim=True).sqrt()
        dimensions=24 if codec.arm=='fdma' else 44
        active=torch.full((4,8,dimensions),complex((48/dimensions)**.5,0),dtype=torch.complex64)
        grid=torch.zeros(4,8,256,dtype=torch.complex64)
        grid[...,:dimensions]=active;grid[...,dimensions:dimensions+8]=2**.5
        digest=hashlib.sha256(repr(noise_key).encode()).hexdigest()
        return dict(H_hat=estimate,W=W,W_base=base,delta_W=delta,
            waveform=SimpleNamespace(active=active,x_grid=grid),soft=active,H_ul_est=sample['H_ul'],
            input_hashes=dict(standard_noise=digest,pilot_polarity=digest))

    def assert_unchanged(self):
        self.assertEqual(self.source.read_bytes(),self.source_before)
        for codec,stats,ce,before in self.seen_models:
            for name,module in (('codec',codec),('stats',stats),('ce',ce)):
                self.assertEqual(module_state(module),before[name]);self.assertTrue(module.training)
                self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_real_six_policy_pipeline_preserves_weights_rng_and_actual_pairing(self):
        rng=torch.get_rng_state().clone();output=self.root/'evaluation/final/attempt_fixture'
        manifest=evaluation.evaluate_final(output,require_cuda=False,loader=self.loader,
            data_factory=self.data,forward=self.forward)
        self.assertEqual(manifest['status'],'passed');self.assertTrue(manifest['transmitted_precoder_scored'])
        self.assertEqual(len(self.loaded),len(cfg.METHODS))
        self.assertEqual(len(self.forward_calls),len(cfg.METHODS)*len(cfg.GAMMAS))
        self.assertEqual(set(self.samples),{('validation',2048)})
        rows=read_rows(output/'samples.csv.gz')
        self.assertEqual(len(rows),len(cfg.METHODS)*len(cfg.GAMMAS)*cfg.MICRO_BATCH)
        self.assertEqual({row['arm'] for row in rows},set(cfg.METHODS))
        self.assertEqual(set(manifest['source_records']),set(cfg.METHODS))
        self.assertEqual(set(manifest['files']),{'samples.csv.gz','summary.csv','comparisons.csv','input_fingerprints.json','training_pairing.json'})
        self.assertFalse(manifest['training_pairing_verified'])
        self.assertEqual(json.loads((output/'training_pairing.json').read_text())['status'],'test_fixture')
        for name,digest in manifest['files'].items():self.assertEqual(self.runtime.sha256(output/name),digest)
        evidence=json.loads((output/'input_fingerprints.json').read_text())
        self.assertEqual(len(evidence),len(cfg.METHODS)*len(cfg.GAMMAS))
        self.assertTrue(all(row['address']==self.metadata.address('validation',2048) for row in evidence))
        self.assertTrue((output/'comparisons.csv').stat().st_size)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assert_unchanged()

    def test_score_uses_actual_W_and_reports_all_users_and_two_residual_norms(self):
        codec,stats,ce,_=self.loader('w_residual','hybrid_g4','cpu');sample=self.data('cpu').sample('validation',2048)
        with torch.no_grad():
            codec.eval();stats.eval();ce.eval()
            out=self.forward(codec,stats,ce,sample,gamma_db=10.,noise_key=('final',0,2))
            # Deliberately permute destination users. Recomputing RZF from the
            # unchanged H_hat would erase this real interference degradation.
            out['W']=out['W_base'].roll(1,dims=-1);out['delta_W']=out['W']-out['W_base']
            vectors=evaluation.score_out(out,sample['H_dl'])
            actual=self.downlink.evaluate_sum_rate(out['W'],sample['H_dl'],downlink_snr_db=cfg.DL_SNR)
            replacement=self.downlink.evaluate_sum_rate(out['W_base'],sample['H_dl'],downlink_snr_db=cfg.DL_SNR)
        torch.testing.assert_close(torch.tensor(vectors['sum_rate_bps_hz']),actual['sum_rate_bps_hz'])
        self.assertGreater(float((replacement['sum_rate_bps_hz']-actual['sum_rate_bps_hz']).min()),1.)
        for i in range(4):
            self.assertAlmostEqual(sum(vectors[f'user_{u}_rate_bps_hz'][i] for u in range(8)),vectors['sum_rate_bps_hz'][i],places=5)
            self.assertAlmostEqual(min(vectors[f'user_{u}_rate_bps_hz'][i] for u in range(8)),vectors['min_user_rate_bps_hz'][i],places=6)
            self.assertGreater(vectors['delta_w_norm'][i],0.)
            self.assertAlmostEqual(vectors['delta_w_norm'][i],vectors['effective_delta_w_norm'][i],places=6)
            self.assertLess(vectors['precoder_power_max_error'][i],1e-5)
            self.assertAlmostEqual(vectors['data_energy'][i],48.,places=4)
            self.assertAlmostEqual(vectors['frame_energy'][i],64.,places=4)

    def test_bad_precoder_power_or_paid_feedback_energy_is_rejected(self):
        codec,stats,ce,_=self.loader('rzf_continue','fdma','cpu');sample=self.data('cpu').sample('validation',2048)
        with torch.no_grad():
            codec.eval();stats.eval();ce.eval()
            out=self.forward(codec,stats,ce,sample,gamma_db=0.,noise_key=('final',0,0))
            out['W'].mul_(2.)
            with self.assertRaisesRegex(FloatingPointError,'precoder per-tone'):
                evaluation.score_out(out,sample['H_dl'])
            out['W']=out['W_base'].clone();out['waveform'].active.mul_(2.)
            with self.assertRaisesRegex(FloatingPointError,'data energy'):
                evaluation.score_out(out,sample['H_dl'])

    def test_noise_mismatch_across_methods_fails_instead_of_claiming_pairing(self):
        def mismatched(codec,*args,**kwargs):
            out=self.forward(codec,*args,**kwargs)
            if codec.policy=='w_residual':out['input_hashes']['standard_noise']='f'*64
            return out
        output=self.root/'evaluation/final/attempt_unpaired'
        with self.assertRaisesRegex(ValueError,'different actual inputs'):
            evaluation.evaluate_final(output,require_cuda=False,loader=self.loader,
                data_factory=self.data,forward=mismatched)
        self.assertFalse((output/'summary.csv').exists())
        self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'failed')
        self.assert_unchanged()

    def test_offline_scorer_cannot_mutate_transmitted_precoder(self):
        def changed(out,truth):
            vectors=evaluation.score_out(out,truth);out['W'].add_(.001);return vectors
        output=self.root/'evaluation/final/attempt_mutated'
        with self.assertRaisesRegex(RuntimeError,'cached BS output'):
            evaluation.evaluate_final(output,require_cuda=False,loader=self.loader,
                data_factory=self.data,forward=self.forward,scorer=changed)
        self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'failed')
        self.assert_unchanged()


def module_state(module):
    from evaluate_twc_v1_k8_icc_aligned import _module_state
    return _module_state(module)


def read_rows(path):
    import twc_v1_k8_icc_aligned_results as tables
    return tables.read_raw_csv(path)


if __name__=='__main__':unittest.main()
