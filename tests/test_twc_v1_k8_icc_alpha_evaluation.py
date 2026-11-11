"""Actual frozen CPU inference/scoring and calibration/final separation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from configs import twc_v1_k8_icc_alpha_config as cfg
import evaluate_twc_v1_k8_icc_alpha as evaluation

try:
    import torch
    import numpy as np
except ImportError:
    torch=np=None


def input_fingerprints(batches=1,kinds=('source',)):
    result=[]
    for kind in kinds:
        for arm in cfg.ARMS:
            for bi in range(batches):
                for gamma in cfg.GAMMAS:
                    value=hashlib.sha256(f'{bi}:{gamma}'.encode()).hexdigest()
                    result.append(dict(kind=kind,arm=arm,batch_index=bi,gamma_db=float(gamma),
                        fingerprint={key:value for key in ('H_dl','H_ul','standard_noise','pilot_polarity')}))
    return result


class InputEvidenceTests(unittest.TestCase):
    def test_phase_path_rejected_before_creating_output_or_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            runtime=SimpleNamespace(inside=lambda path:Path(path).resolve())
            loader=mock.Mock(side_effect=AssertionError('Wrong-phase output must not load models'))
            with mock.patch.object(cfg,'OUTPUT_ROOT',root),mock.patch.object(evaluation,'_runtime',return_value=runtime):
                for phase,path in (('calibration',root/'unexpected/attempt'),
                                   ('calibration',root/'evaluation/final/attempt'),
                                   ('calibration',root/'calibration'),
                                   ('final',root/'calibration/attempt'),
                                   ('final',root/'evaluation/final')):
                    with self.subTest(phase=phase,path=path):
                        with self.assertRaisesRegex(ValueError,'phase subtree'):
                            evaluation._setup(path,False,phase=phase)
                        self.assertFalse(path.exists())
                with self.assertRaisesRegex(ValueError,'phase subtree'):
                    evaluation.calibrate(root/'unexpected/attempt',require_cuda=False,loader=loader)
                loader.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])

    def test_pairing_requires_actual_hashes_and_complete_conditions(self):
        rows=input_fingerprints(kinds=('source','rate'))
        found=evaluation.fingerprint_map(rows,batches=1,kinds=('source','rate'))
        self.assertEqual(len(found),len(cfg.ARMS)*len(cfg.GAMMAS)*2)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            evaluation.fingerprint_map(rows[:-1],batches=1,kinds=('source','rate'))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            evaluation.fingerprint_map(rows+[rows[0]],batches=1,kinds=('source','rate'))
        bad=copy.deepcopy(rows);bad[-1]['fingerprint']['standard_noise']='f'*64
        with self.assertRaisesRegex(ValueError,'different actual inputs'):
            evaluation.fingerprint_map(bad,batches=1,kinds=('source','rate'))

    def test_actual_sample_address_must_match_calibration_or_final_phase(self):
        import icc_alpha_runtime as runtime
        metadata=runtime.make_data('cpu')
        seen=[]
        class Data:
            def sample(self,role,index):
                seen.append((role,index))
                return dict(address=metadata.address('validation',2048))
        with self.assertRaisesRegex(ValueError,'declared phase'):
            evaluation._sample(Data(),metadata,'calibration',0)
        self.assertEqual(seen,[('validation',0)])
        got=evaluation._sample(Data(),metadata,'final',0)
        self.assertEqual(got['address'],metadata.address('validation',2048))


@unittest.skipIf(torch is None or np is None,'Real PyTorch/NumPy unavailable; no fake dependency')
class FrozenCPUInference(unittest.TestCase):
    def setUp(self):
        import icc_alpha_runtime as real_runtime
        self.original_rng=torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state,self.original_rng)
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.metadata=real_runtime.make_data('cpu')
        self.core={'schema':'real_CPU_fixture','sha256':{'fixture':'a'*64}}
        self.source=self.root/'source_bytes.bin';self.source.write_bytes(b'never rewritten')
        self.source_before=self.source.read_bytes()
        self.seen_models=[];self.forward_calls=[];self.samples=[];self.loaded=[]
        self.selection=None
        def inside(path):
            path=Path(path).resolve()
            if path==self.root or not path.is_relative_to(self.root):raise ValueError('owned child required')
            return path
        def atomic(path,value,exclusive=False):
            path=inside(path)
            with path.open('x' if exclusive else 'w') as f:json.dump(value,f,sort_keys=True,allow_nan=False)
        def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        def verify(record):
            for item in record['frozen_files']:
                if sha(item['path'])!=item['sha256']:raise ValueError('source bytes changed')
        def load_selection(path):
            if self.selection is None:raise ValueError('locked calibration required')
            return copy.deepcopy(self.selection)
        self.runtime=SimpleNamespace(OUTPUT_ROOT=self.root,inside=inside,atomic_json=atomic,sha256=sha,
            core_identity=lambda:copy.deepcopy(self.core),guard_core=lambda core:self.assertEqual(core,self.core),
            make_data=lambda device:self.metadata,verify_files=verify,load_selection=load_selection)
        for patch in (mock.patch.object(evaluation,'_runtime',return_value=self.runtime),
                      mock.patch.object(cfg,'OUTPUT_ROOT',self.root),
                      mock.patch.object(cfg,'CALIBRATION_BATCHES',1),
                      mock.patch.object(cfg,'FINAL_BATCHES',1)):
            patch.start();self.addCleanup(patch.stop)
        self.templates={}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(74105)
            for arm in cfg.ARMS:
                codec=torch.nn.Module();codec.encoder=torch.nn.Linear(2,2);codec.decoder=torch.nn.Linear(2,2)
                ce=torch.nn.Linear(2,2);stats=torch.nn.Module()
                stats.register_buffer('historical',torch.ones(8,24 if arm=='fdma' else 44))
                stats.register_buffer('ema_updates',torch.tensor(20000))
                self.templates[arm]=(codec,stats,ce)

    def loader(self,kind,arm,device):
        self.assertEqual(str(device),'cpu');self.loaded.append((kind,arm))
        codec,stats,ce=copy.deepcopy(self.templates[arm]);codec.arm=arm;codec.kind=kind
        before={name:evaluation_state(module) for name,module in [('codec',codec),('stats',stats),('ce',ce)]}
        self.seen_models.append((codec,stats,ce,before))
        record=dict(kind=kind,arm=arm,frozen_files=[dict(path=str(self.source),
                    sha256=self.runtime.sha256(self.source))])
        return codec,stats,ce,record

    def data(self,device):
        outer=self
        class Data:
            protocol=outer.metadata.protocol
            def sample(self,role,index):
                outer.samples.append((role,index))
                generator=torch.Generator().manual_seed(12340+index)
                identity=torch.eye(8,32,dtype=torch.complex64)[None,:,:,None].expand(4,8,32,256).clone()
                perturb=torch.complex(torch.randn(identity.shape,generator=generator),
                                      torch.randn(identity.shape,generator=generator))*.025
                truth=identity+perturb
                return dict(H_dl=truth,H_ul=truth*1.2,address=outer.metadata.address(role,index))
        return Data()

    def forward(self,codec,stats,ce,sample,*,gamma_db,noise_key):
        self.assertFalse(torch.is_grad_enabled())
        for module in (codec,stats,ce):self.assertFalse(module.training)
        self.forward_calls.append((codec.kind,codec.arm,gamma_db,noise_key))
        features=codec.decoder(codec.encoder(torch.ones(4,2)))
        # Tiny actual modules produce a full production-shape estimate, allowing
        # the unchanged complex RZF and rate scorer to run on CPU.
        estimate=sample['H_dl'].clone()+features[:,0,None,None,None]*.003
        dimensions=24 if codec.arm=='fdma' else 44
        active=torch.full((4,8,dimensions),complex((48/dimensions)**.5,0),dtype=torch.complex64)
        grid=torch.zeros(4,8,256,dtype=torch.complex64)
        grid[...,:dimensions]=active;grid[...,dimensions:dimensions+8]=2**.5
        digest=hashlib.sha256(repr(noise_key).encode()).hexdigest()
        return dict(H_hat=estimate,waveform=SimpleNamespace(active=active,x_grid=grid),soft=active,
                    H_ul_est=sample['H_ul'],input_hashes=dict(standard_noise=digest,pilot_polarity=digest))

    def assert_unchanged(self):
        self.assertEqual(self.source.read_bytes(),self.source_before)
        for codec,stats,ce,before in self.seen_models:
            for name,module in [('codec',codec),('stats',stats),('ce',ce)]:
                self.assertEqual(evaluation_state(module),before[name])
                self.assertTrue(module.training)
                self.assertTrue(all(p.grad is None for p in module.parameters()))

    def calibrate(self):
        output=self.root/'calibration/attempt_fixture'
        result=evaluation.calibrate(output,require_cuda=False,loader=self.loader,
            data_factory=self.data,forward=self.forward)
        self.selection=json.loads((output/'selection_candidate.json').read_text())
        self.selection['record_id']='e'*64
        return output,result

    def test_real_cached_calibration_and_all_six_policy_final_scoring(self):
        rng=torch.get_rng_state().clone()
        output,manifest=self.calibrate()
        self.assertEqual(manifest['status'],'passed')
        self.assertEqual(len(self.forward_calls),len(cfg.ARMS)*len(cfg.GAMMAS))
        self.assertEqual({index for role,index in self.samples},{0})
        self.assertTrue((output/'samples.csv.gz').is_file())
        self.assertEqual(set(self.selection['selected']),set(cfg.ARMS))
        self.assertTrue(all(value in cfg.ALPHA_GRID for value in self.selection['selected'].values()))
        self.samples.clear();self.forward_calls.clear()
        final=self.root/'evaluation/final/attempt_fixture'
        result=evaluation.evaluate_final(final,self.root/'selection.json',require_cuda=False,
            loader=self.loader,data_factory=self.data,forward=self.forward)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['selection_record_id'],'e'*64)
        self.assertEqual(len(self.forward_calls),2*len(cfg.ARMS)*len(cfg.GAMMAS))
        self.assertEqual({index for role,index in self.samples},{2048})
        rows=tables_rows(final/'samples.csv.gz')
        self.assertEqual(len(rows),len(cfg.METHODS)*len(cfg.GAMMAS)*cfg.MICRO_BATCH)
        self.assertEqual({row['arm'] for row in rows},set(cfg.METHODS))
        self.assertTrue((final/'comparisons.csv').stat().st_size)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assert_unchanged()

    def test_final_requires_valid_lock_before_creating_outputs(self):
        final=self.root/'evaluation/final/attempt_no_lock'
        with self.assertRaisesRegex(ValueError,'locked calibration required'):
            evaluation.evaluate_final(final,self.root/'missing.json',require_cuda=False,
                loader=self.loader,data_factory=self.data,forward=self.forward)
        self.assertFalse(final.exists());self.assertFalse(self.loaded)

    def test_alpha_scoring_cannot_mutate_shared_cached_prediction(self):
        def changed(out,truth,alpha):
            vectors=evaluation.score_out(out,truth,alpha)
            out['H_hat'].add_(.001)
            return vectors
        with self.assertRaisesRegex(RuntimeError,'mutated the cached CSI'):
            evaluation.calibrate(self.root/'calibration/attempt_bad_cache',require_cuda=False,loader=self.loader,
                data_factory=self.data,forward=self.forward,scorer=changed)
        manifest=json.loads((self.root/'calibration/attempt_bad_cache/manifest.json').read_text())
        self.assertEqual(manifest['status'],'failed')
        self.assert_unchanged()

    def test_production_score_checks_energy_and_keeps_selected_alpha(self):
        codec,stats,ce,_=self.loader('source','hybrid_g4','cpu')
        sample=self.data('cpu').sample('validation',0)
        with torch.no_grad():
            codec.eval();stats.eval();ce.eval()
            out=self.forward(codec,stats,ce,sample,gamma_db=10.,noise_key=('calibration',0,2))
            low=evaluation.score_out(out,sample['H_dl'],.003)
            high=evaluation.score_out(out,sample['H_dl'],.3)
        self.assertNotEqual(low['sum_rate_bps_hz'],high['sum_rate_bps_hz'])
        for values in (low,high):
            for i in range(4):
                self.assertAlmostEqual(sum(values[f'user_{u}_rate_bps_hz'][i] for u in range(8)),values['sum_rate_bps_hz'][i],places=4)
                self.assertAlmostEqual(values['data_energy'][i],48.,places=4)
                self.assertAlmostEqual(values['frame_energy'][i],64.,places=4)
                self.assertLess(values['rzf_power_max_error'][i],1e-5)
        out['waveform'].active.mul_(2)
        with self.assertRaisesRegex(FloatingPointError,'data energy'):
            evaluation.score_out(out,sample['H_dl'],.03)


def evaluation_state(module):
    from evaluate_twc_v1_k8_icc_aligned import _module_state
    return _module_state(module)


def tables_rows(path):
    import twc_v1_k8_icc_aligned_results as tables
    return tables.read_raw_csv(path)


if __name__=='__main__':unittest.main()
