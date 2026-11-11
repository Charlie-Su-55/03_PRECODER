"""Calibration-only choices and six-policy paired statistics; no dependencies."""
import copy
import inspect
import math
import random
import statistics
from types import SimpleNamespace
import unittest
from unittest import mock

import twc_v1_k8_icc_alpha_results as results


ARMS = ('fdma', 'hybrid_g4')
POLICIES = ('source_fixed', 'source_calibrated', 'rate_fixed')
GRID = (.003, .01, .02, .03, .05, .07, .1, .2, .3)
GAMMAS = (0., 5., 10., 15., 20., 25.)


def method_name(policy, arm):
    return policy+'_'+arm


def configuration():
    return SimpleNamespace(ARMS=ARMS, POLICIES=POLICIES, ALPHA_GRID=GRID,
        FIXED_ALPHA=.03, GAMMAS=GAMMAS,
        METHODS=tuple(method_name(policy,arm) for policy in POLICIES for arm in ARMS),
        method_name=method_name)


def calibration_rows(samples=8):
    rows=[]
    for arm,target in (('fdma',.05),('hybrid_g4',.07)):
        for alpha in GRID:
            for gamma in GAMMAS:
                for i in range(samples):
                    score=100+.5*i+.02*gamma-4*math.log(alpha/target)**2
                    # The gamma0 winner differs from the global six-point one.
                    if gamma==0 and alpha==.003:score+=50
                    rows.append(dict(arm=arm,alpha=alpha,gamma_db=gamma,
                        sample_index=i,batch_index=i//4,within_batch=i%4,
                        sum_rate_bps_hz=score))
    return rows


def final_rows(samples=8):
    rows=[]
    for policy in POLICIES:
        for arm in ARMS:
            for gamma in GAMMAS:
                for i in range(samples):
                    base=10+.2*i+.02*gamma
                    if arm=='hybrid_g4':base+=-1+.1*i+.003*gamma
                    gain=0.
                    if policy=='source_calibrated':
                        gain=(.25+.01*i+.002*gamma if arm=='fdma' else .45-.015*i+.001*gamma)
                    elif policy=='rate_fixed':
                        gain=(.55+.02*i+.003*gamma if arm=='fdma' else .9-.01*i+.004*gamma)
                    rate=base+gain
                    rows.append(dict(arm=method_name(policy,arm),gamma_db=gamma,
                        sample_index=i,batch_index=i//4,within_batch=i%4,
                        sum_rate_bps_hz=rate,dl_nmse_linear=.01+.001*i,
                        desired_signal_power_mean=1.,interference_power_mean=.1/(1+gain),
                        rzf_power_max_error=1e-7,min_user_rate_bps_hz=rate/12,
                        interference_to_signal_ratio=.1/(1+gain),predicted_dl_energy=.8,
                        soft_key_nmse_linear=(None if arm=='fdma' else .02),
                        soft_shared_nmse_linear=(None if arm=='fdma' else .05),
                        **{f'user_{u}_rate_bps_hz':rate/8 for u in range(8)}))
    return rows


class AlphaCalibrationStatistics(unittest.TestCase):
    def setUp(self):
        patcher=mock.patch.object(results,'_config',return_value=configuration())
        patcher.start();self.addCleanup(patcher.stop)

    def test_one_alpha_per_arm_maximizes_complete_mu_six_gamma_mean(self):
        rows=calibration_rows()
        chosen=results.select_alphas(rows,samples=8)
        self.assertEqual(chosen['selected'],{'fdma':.05,'hybrid_g4':.07})
        self.assertFalse(chosen['test_used'])
        self.assertEqual(chosen['n_samples'],8)
        self.assertEqual(chosen['n_batches'],2)
        self.assertEqual(len(chosen['scores']),2*len(GRID))
        for arm in ARMS:
            point0={alpha:statistics.mean(r['sum_rate_bps_hz'] for r in rows
                if r['arm']==arm and r['alpha']==alpha and r['gamma_db']==0.) for alpha in GRID}
            self.assertEqual(max(point0,key=point0.get),.003)
            selected=next(x for x in chosen['scores'] if x['arm']==arm and x['alpha']==chosen['selected'][arm])
            mu=[100+.5*i+.02*statistics.mean(GAMMAS) for i in range(8)]
            self.assertAlmostEqual(selected['mean'],statistics.mean(mu),places=12)
            self.assertAlmostEqual(selected['sample_se'],statistics.stdev(mu)/math.sqrt(8),places=12)
            self.assertAlmostEqual(selected['batch_se'],statistics.stdev([
                statistics.mean(mu[:4]),statistics.mean(mu[4:])])/math.sqrt(2),places=12)

    def test_ties_prefer_fixed_then_nearest_log_distance_and_ignore_row_order(self):
        rows=calibration_rows()
        for row in rows:row['sum_rate_bps_hz']=10.+row['sample_index']
        self.assertEqual(results.select_alphas(rows,samples=8)['selected'],dict.fromkeys(ARMS,.03))
        for row in rows:
            row['sum_rate_bps_hz']+=(1. if row['alpha'] in (.02,.05) else 0.)
        reference=results.select_alphas(rows,samples=8)
        self.assertEqual(reference['selected'],dict.fromkeys(ARMS,.02))
        shuffled=copy.deepcopy(rows);random.Random(741).shuffle(shuffled)
        self.assertEqual(results.select_alphas(shuffled,samples=8),reference)

    def test_selection_only_accepts_calibration_schema_and_no_final_argument(self):
        self.assertNotIn('final',inspect.signature(results.select_alphas).parameters)
        with self.assertRaises(TypeError):
            results.select_alphas(calibration_rows(),samples=8,final_rows=final_rows())
        for change in ({'role':'test'},{'policy':'rate_fixed'},{'future_rate':1000.}):
            rows=calibration_rows();rows[0].update(change)
            with self.assertRaisesRegex(ValueError,'calibration-only'):
                results.select_alphas(rows,samples=8)
        rows=calibration_rows();rows[0]['arm']='rate_fixed_fdma'
        with self.assertRaisesRegex(ValueError,'Unexpected calibration'):
            results.select_alphas(rows,samples=8)

    def test_incomplete_duplicate_nonnumeric_or_wrong_address_calibration_is_rejected(self):
        good=calibration_rows()
        for rows in (good[:-1],good+[good[0]]):
            with self.assertRaises(ValueError):results.select_alphas(rows,samples=8)
        for field,value in (('alpha',0.),('alpha',.04),('alpha',True),('gamma_db',30.),
                ('sum_rate_bps_hz',math.nan),('sum_rate_bps_hz',-1.),
                ('sum_rate_bps_hz',True),('sample_index',True),('batch_index',99),('within_batch',4)):
            rows=copy.deepcopy(good);rows[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                results.select_alphas(rows,samples=8)
        for samples,batch_size in ((0,4),(7,4),(True,4),(8,0)):
            with self.assertRaises(ValueError):
                results.select_alphas(good,samples=samples,batch_size=batch_size)


class AlphaFinalStatistics(unittest.TestCase):
    def setUp(self):
        patcher=mock.patch.object(results,'_config',return_value=configuration())
        patcher.start();self.addCleanup(patcher.stop)

    def test_all_contrasts_preserve_mu_then_gamma_and_batch_pairing(self):
        output=results.paired_comparisons(final_rows(),samples=8)
        self.assertEqual(len({r['comparison'] for r in output}),10)
        actual=next(r for r in output if r['comparison']=='interaction_rate_gain_minus_calibration_gain'
                    and r['scope']=='all_six' and r['metric']=='sum_rate_bps_hz')
        deltas=[.15-.005*i+.002*statistics.mean(GAMMAS) for i in range(8)]
        grouped=[statistics.mean(deltas[:4]),statistics.mean(deltas[4:])]
        self.assertAlmostEqual(actual['mean'],statistics.mean(deltas),places=12)
        self.assertAlmostEqual(actual['sample_se'],statistics.stdev(deltas)/math.sqrt(8),places=12)
        self.assertAlmostEqual(actual['batch_se'],statistics.stdev(grouped)/math.sqrt(2),places=12)
        self.assertAlmostEqual(actual['batch_ci95_low'],statistics.mean(deltas)-1.96*actual['batch_se'],places=12)
        self.assertEqual(actual['n_samples'],8)
        self.assertEqual(actual['n_batches'],2)
        self.assertEqual(actual['batch_size'],4)
        self.assertGreater(actual['mean'],0.)
        # A positive residual interaction does not imply that Hybrid wins.
        final=next(r for r in output if r['comparison']=='rate_fixed_hybrid_minus_fdma'
                   and r['scope']=='all_six' and r['metric']=='sum_rate_bps_hz')
        self.assertLess(final['mean'],0.)
        self.assertEqual({r['scope'] for r in output},{'point','low_0_5_10','all_six'})

    def test_each_declared_gain_matches_its_policies_and_retains_users(self):
        rows=final_rows();output=results.paired_comparisons(rows,samples=8)
        indexed={(r['arm'],r['gamma_db'],r['sample_index']):r for r in rows}
        for arm in ARMS:
            for left,right,label in (
                ('source_calibrated','source_fixed','source_calibrated_minus_source_fixed'),
                ('rate_fixed','source_fixed','rate_fixed_minus_source_fixed'),
                ('rate_fixed','source_calibrated','rate_fixed_minus_source_calibrated')):
                record=next(r for r in output if r['comparison']==arm+'_'+label
                            and r['scope']=='low_0_5_10' and r['metric']=='sum_rate_bps_hz')
                expected=[statistics.mean(indexed[method_name(left,arm),g,i]['sum_rate_bps_hz']
                    -indexed[method_name(right,arm),g,i]['sum_rate_bps_hz'] for g in GAMMAS[:3]) for i in range(8)]
                self.assertAlmostEqual(record['mean'],statistics.mean(expected),places=12)
        for metric in ('user_0_rate_bps_hz','user_7_rate_bps_hz','min_user_rate_bps_hz',
                       'interference_to_signal_ratio','predicted_dl_energy','rzf_power_max_error'):
            self.assertTrue(any(r['metric']==metric for r in output),metric)
        self.assertTrue(any(r['metric']=='soft_shared_nmse_linear' and
                            r['comparison']=='hybrid_g4_rate_fixed_minus_source_calibrated' for r in output))
        self.assertFalse(any(r['metric']=='soft_shared_nmse_linear' and
                             r['comparison']=='rate_fixed_hybrid_minus_fdma' for r in output))

    def test_final_missing_duplicate_bad_identity_and_partial_metrics_fail(self):
        rows=final_rows()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):results.paired_comparisons(bad,samples=8)
        for field,value in (('arm','fdma'),('within_batch',4),('sum_rate_bps_hz',math.inf),
                            ('desired_signal_power_mean',-1.),('gamma_db',True)):
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                results.paired_comparisons(bad,samples=8)
        bad=copy.deepcopy(rows);del bad[0]['user_7_rate_bps_hz']
        with self.assertRaisesRegex(ValueError,'Partially missing'):
            results.paired_comparisons(bad,samples=8)

    def test_summaries_reuse_complete_six_method_validation_and_linear_nmse(self):
        rows=final_rows();summary=results.summarize_rows(rows,samples=8)
        self.assertEqual(len(summary),6*6)
        sample=next(r for r in summary if r['arm']=='source_fixed_fdma' and r['gamma_db']==0.)
        self.assertEqual((sample['n_samples'],sample['n_batches'],sample['batch_size']),(8,2,4))
        self.assertAlmostEqual(sample['dl_nmse_db_from_linear_mean'],
                               10*math.log10(statistics.mean(.01+.001*i for i in range(8))))
        self.assertAlmostEqual(sample['sum_rate_bps_hz_mean'],statistics.mean(10+.2*i for i in range(8)))


if __name__=='__main__':unittest.main()
