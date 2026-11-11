"""Paired four-policy statistics and actual tiny frozen tensor evaluation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import evaluate_twc_v1_k8_icc_rate as evaluation
import twc_v1_k8_icc_aligned_results as tables

try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


def rows(final=False, samples=8):
    output=[]
    for arm in tables.ARMS:
        for gamma in tables.GAMMAS:
            for i in range(samples):
                rate=10+.5*i+.1*gamma
                if arm=='hybrid_g4':rate+=-2+.2*i
                if final:rate+=(.2+.01*i if arm=='fdma' else .7-.02*i)
                output.append(dict(arm=arm,gamma_db=gamma,sample_index=i,batch_index=i//4,within_batch=i%4,
                    dl_nmse_linear=.1+.001*i,sum_rate_bps_hz=rate,
                    desired_signal_power_mean=1.,interference_power_mean=.1,rzf_power_max_error=1e-7,
                    interference_to_signal_ratio=.1,min_user_rate_bps_hz=rate/8,
                    **{f'user_{u}_rate_bps_hz':rate/8 for u in range(8)}))
    return output


def fingerprints(arms=tables.ARMS,batches=2):
    result=[]
    for arm in arms:
        for bi in range(batches):
            for gamma in tables.GAMMAS:
                digest=hashlib.sha256(f'{bi}:{gamma}'.encode()).hexdigest()
                result.append(dict(arm=arm,batch_index=bi,gamma_db=gamma,
                    fingerprint={key:digest for key in ('H_dl','H_ul','standard_noise','pilot_polarity')}))
    return result


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cfg=SimpleNamespace(ARMS=tables.ARMS,GAMMAS=tables.GAMMAS,MICRO_BATCH=4,FINAL_BATCHES=2,
                                 NAMESPACE='tiny_rate',ALPHA=.03,DL_SNR=25.)
        def inside(path):
            p=Path(path).resolve()
            if p==self.root or not p.is_relative_to(self.root):raise ValueError('owned child required')
            return p
        def atomic_json(path,value,exclusive=False):
            path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x' if exclusive else 'w') as f:json.dump(value,f)
        self.runtime=SimpleNamespace(inside=inside,atomic_json=atomic_json,core_identity=lambda:{'tiny':1})
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(evaluation,'_config',return_value=self.cfg).start()
        mock.patch.object(evaluation,'_runtime',return_value=self.runtime).start()

    def write_phase(self,name,final=False):
        path=self.root/name;path.mkdir()
        table=tables.write_result_tables(path,rows(final),samples=8,batch_size=4)
        self.runtime.atomic_json(path/'input_fingerprints.json',fingerprints(),exclusive=True)
        files=dict(table['files']);files['input_fingerprints.json']=tables.file_sha256(path/'input_fingerprints.json')
        self.runtime.atomic_json(path/'manifest.json',dict(status='passed',mode='final',phase=name,
            arms=list(tables.ARMS),core={'tiny':1},batch_size=4,n_batches=2,data_protocol={'fresh':True},files=files),exclusive=True)
        return path

    def test_four_policy_interaction_is_paired_within_mu_before_gamma_and_batch(self):
        compared=evaluation.comparison_rows(rows(),rows(True),samples=8)
        record=next(x for x in compared if x['comparison']=='interaction_hybrid_gain_minus_fdma_gain'
                    and x['scope']=='all_six' and x['metric']=='sum_rate_bps_hz')
        deltas=[.5-.03*i for i in range(8)]
        grouped=[statistics.mean(deltas[i:i+4]) for i in (0,4)]
        self.assertAlmostEqual(record['mean'],statistics.mean(deltas),places=12)
        self.assertAlmostEqual(record['sample_se'],statistics.stdev(deltas)/math.sqrt(8),places=12)
        self.assertAlmostEqual(record['batch_se'],statistics.stdev(grouped)/math.sqrt(2),places=12)
        self.assertEqual(record['n_samples'],8)
        self.assertEqual(record['n_batches'],2)
        self.assertTrue(any(x['metric']=='user_7_rate_bps_hz' for x in compared))
        self.assertTrue(any(x['metric']=='interference_to_signal_ratio' for x in compared))
        # A relative improvement need not make Hybrid win in absolute rate.
        terminal=next(x for x in compared if x['comparison']=='rate_hybrid_minus_fdma'
                      and x['scope']=='all_six' and x['metric']=='sum_rate_bps_hz')
        self.assertGreater(record['mean'],0.)
        self.assertLess(terminal['mean'],0.)

    def test_missing_duplicate_and_nonfinite_complete_mu_rows_are_rejected(self):
        for broken in (rows()[:-1],rows()+[rows()[0]]):
            with self.assertRaises(ValueError):evaluation.comparison_rows(broken,rows(True),samples=8)
        bad=rows(True);bad[0]['sum_rate_bps_hz']=math.nan
        with self.assertRaises(ValueError):evaluation.comparison_rows(rows(),bad,samples=8)

    def test_actual_pairing_hashes_not_only_seeds_are_required(self):
        valid=fingerprints();evaluation.fingerprint_map(valid,batches=2)
        bad=copy.deepcopy(valid);bad[-1]['fingerprint']['standard_noise']='f'*64
        with self.assertRaisesRegex(ValueError,'identical actual inputs'):
            evaluation.fingerprint_map(bad,batches=2)
        with self.assertRaisesRegex(ValueError,'Missing'):
            evaluation.fingerprint_map(valid[:-1],batches=2)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            evaluation.fingerprint_map(valid+[valid[0]],batches=2)

    def test_comparison_verifies_artifacts_and_source_final_actual_inputs(self):
        source=self.write_phase('baseline');final=self.write_phase('final',True)
        output=self.root/'comparison'
        manifest=evaluation.compare_evaluations(source,final,output)
        self.assertEqual(manifest['status'],'passed')
        self.assertTrue((output/'comparisons.csv').is_file())
        (final/'summary.csv').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            evaluation.compare_evaluations(source,final,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_cross_phase_mismatch_refuses_even_when_each_phase_is_paired(self):
        source=self.write_phase('baseline');final=self.write_phase('final',True)
        entries=fingerprints()
        for item in entries:item['fingerprint']['standard_noise']='f'*64
        (final/'input_fingerprints.json').write_text(json.dumps(entries))
        manifest=json.loads((final/'manifest.json').read_text())
        manifest['files']['input_fingerprints.json']=tables.file_sha256(final/'input_fingerprints.json')
        (final/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'different actual channel/noise'):
            evaluation.compare_evaluations(source,final,self.root/'bad')


@unittest.skipIf(torch is None or np is None,'Real PyTorch/NumPy unavailable; no fake dependency')
class TinyFrozenEvaluation(unittest.TestCase):
    def test_production_rzf_scoring_keeps_energy_and_per_user_metrics(self):
        truth=torch.eye(8,32,dtype=torch.complex64)[None,:,:,None].expand(4,8,32,256).clone()
        for dimensions in (24,44):
            with self.subTest(active_coordinates=dimensions):
                active=torch.full((4,8,dimensions),complex((48/dimensions)**.5,0),dtype=torch.complex64)
                grid=torch.zeros(4,8,256,dtype=torch.complex64)
                grid[...,:dimensions]=active
                grid[...,dimensions:dimensions+8]=2**.5
                waveform=SimpleNamespace(active=active,x_grid=grid)
                scored=evaluation.metric_vectors(dict(H_hat=truth,waveform=waveform,soft=active,
                                                       H_ul_est=truth),truth)
                for i in range(4):
                    self.assertAlmostEqual(scored['data_energy'][i],48.,places=4)
                    self.assertAlmostEqual(scored['frame_energy'][i],64.,places=4)
                    self.assertLess(scored['rzf_power_max_error'][i],1e-5)
                    self.assertAlmostEqual(scored['interference_to_signal_ratio'][i],0.,places=6)
                    self.assertAlmostEqual(sum(scored[f'user_{u}_rate_bps_hz'][i] for u in range(8)),
                                           scored['sum_rate_bps_hz'][i],places=4)
                    self.assertGreater(scored['min_user_rate_bps_hz'][i],0.)
                self.assertEqual('soft_shared_nmse_linear' in scored,dimensions==44)

    def test_real_cpu_tensors_modes_rng_and_gradients_remain_frozen(self):
        class Codec(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(2,2);self.decoder=torch.nn.Linear(2,2)
        class Stats(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('power',torch.full((8,24),2.))
            def nominal_power(self):return self.power
        class Data:
            protocol={'tiny_cpu_only':True}
            def sample(self,role,index):
                if role!='validation':raise AssertionError('test/train sampled')
                return {'H_dl':torch.ones(4,2),'H_ul':torch.ones(4,2)*2}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);codec=Codec().train();ce=torch.nn.Linear(2,2).train();stats=Stats().train()
            before={name:evaluation._module_state(module) for name,module in [('codec',codec),('ce',ce),('stats',stats)]}
            rng=torch.get_rng_state().clone()
            cfg=SimpleNamespace(ARMS=tables.ARMS,GAMMAS=tables.GAMMAS,MICRO_BATCH=4,MONITOR_BATCHES=1,
                FINAL_BATCHES=1,ROLE_RANGES={'monitor':[0,0],'final':[2048,2048]},NAMESPACE='tiny',ALPHA=.03,DL_SNR=25.)
            def inside(path):
                p=Path(path).resolve();self.assertTrue(p.is_relative_to(root));return p
            def atomic_json(path,value,exclusive=False):
                with Path(path).open('x' if exclusive else 'w') as f:json.dump(value,f)
            runtime=SimpleNamespace(inside=inside,atomic_json=atomic_json,core_identity=lambda:{'core':'tiny'},
                guard_core=lambda value:self.assertEqual(value,{'core':'tiny'}),source_identity=lambda:{'tiny':True})
            def forward(model,stat,estimator,sample,**kwargs):
                self.assertFalse(torch.is_grad_enabled());self.assertFalse(model.training)
                return {'score':model.decoder(model.encoder(sample['H_dl'])).square().sum(1),
                        'input_hashes':{'standard_noise':'a'*64,'pilot_polarity':'b'*64}}
            def scorer(out,truth):
                v=out['score'].detach().tolist()
                return dict(dl_nmse_linear=[x+.01 for x in v],sum_rate_bps_hz=[x+1 for x in v],
                    desired_signal_power_mean=[1.]*4,interference_power_mean=[.1]*4,rzf_power_max_error=[0.]*4,
                    **{f'user_{u}_rate_bps_hz':[(x+1)/8 for x in v] for u in range(8)})
            with mock.patch.object(evaluation,'_config',return_value=cfg),mock.patch.object(evaluation,'_runtime',return_value=runtime):
                result=evaluation.evaluate_models({'fdma':(codec,stats)},ce,Data(),mode='monitor',
                    output_dir=root/'monitor',forward=forward,scorer=scorer,require_cuda=False)
            self.assertEqual(result['status'],'passed');self.assertTrue(codec.training and ce.training and stats.training)
            for name,module in [('codec',codec),('ce',ce),('stats',stats)]:
                self.assertEqual(evaluation._module_state(module),before[name])
                self.assertTrue(all(p.grad is None for p in module.parameters()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
