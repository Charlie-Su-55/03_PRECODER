"""Standard-library checks of complete-MU residual-W statistical controls."""
import copy
import math
import random
import statistics
from types import SimpleNamespace
import unittest
from unittest import mock

import twc_v1_k8_w_residual_results as results


ARMS = ('fdma','hybrid_g4')
POLICIES = ('rate_source','rzf_continue','w_residual')
GAMMAS = (0.,5.,10.,15.,20.,25.)


def method_name(policy,arm):
    return policy+'_'+arm


def configuration():
    return SimpleNamespace(ARMS=ARMS, POLICIES=POLICIES, GAMMAS=GAMMAS,
        METHODS=tuple(method_name(p,a) for p in POLICIES for a in ARMS),
        method_name=method_name)


def final_rows(samples=8):
    rows=[]
    for policy in POLICIES:
        for arm in ARMS:
            for gamma in GAMMAS:
                for i in range(samples):
                    base=12.+.3*i+.02*gamma
                    if arm=='hybrid_g4':base+=-1.5+.1*i+.004*gamma
                    continued=(.4+.02*i+.002*gamma if arm=='fdma'
                               else .7-.01*i+.004*gamma)
                    residual=(.1+.03*i-.001*gamma if arm=='fdma'
                              else .25+.02*i+.003*gamma)
                    gain=(0. if policy=='rate_source' else continued)
                    if policy=='w_residual':gain+=residual
                    rate=base+gain
                    rows.append(dict(arm=method_name(policy,arm),gamma_db=gamma,
                        sample_index=i,batch_index=i//4,within_batch=i%4,
                        sum_rate_bps_hz=rate,dl_nmse_linear=.02+.001*i,
                        desired_signal_power_mean=1.+.01*gain,
                        interference_power_mean=.1/(1+gain),
                        rzf_power_max_error=1e-7,
                        min_user_rate_bps_hz=rate/8-.035*(1+.1*i),
                        interference_to_signal_ratio=.1/(1+gain),
                        predicted_dl_energy=.8,data_energy=48.,frame_energy=64.,
                        residual_relative_norm=(.005 if policy=='w_residual' else 0.),
                        soft_shared_nmse_linear=(None if arm=='fdma' else .05),
                        **{f'user_{u}_rate_bps_hz':rate/8+.01*(u-3.5)*(1+.1*i)
                           for u in range(8)}))
    return rows


def find(rows,label,scope='all_six',metric='sum_rate_bps_hz',gamma=None):
    return next(r for r in rows if r['comparison']==label and r['scope']==scope
                and r['metric']==metric and r['gamma_db']==gamma)


class ResidualWStatistics(unittest.TestCase):
    def setUp(self):
        patcher=mock.patch.object(results,'_config',return_value=configuration())
        patcher.start();self.addCleanup(patcher.stop)

    def test_primary_means_and_interaction_have_hand_derived_mu_uncertainty(self):
        output=results.paired_comparisons(final_rows(),samples=8)
        # All-six mean gamma is12.5. These expressions are per complete MU,
        # rather than treating the48 method/gamma replicas as independent.
        expected={
            'fdma_w_residual_minus_rzf_continue':[.0875+.03*i for i in range(8)],
            'hybrid_g4_w_residual_minus_rzf_continue':[.2875+.02*i for i in range(8)],
            'interaction_w_residual_minus_rzf_continue':[.20-.01*i for i in range(8)],
        }
        for label,mu in expected.items():
            actual=find(output,label)
            batches=[statistics.mean(mu[:4]),statistics.mean(mu[4:])]
            se=statistics.stdev(mu)/math.sqrt(8)
            bse=statistics.stdev(batches)/math.sqrt(2)
            self.assertAlmostEqual(actual['mean'],statistics.mean(mu),places=12)
            self.assertAlmostEqual(actual['sample_se'],se,places=12)
            self.assertAlmostEqual(actual['batch_se'],bse,places=12)
            self.assertAlmostEqual(actual['sample_ci95_low'],statistics.mean(mu)-1.96*se,places=12)
            self.assertAlmostEqual(actual['batch_ci95_high'],statistics.mean(mu)+1.96*bse,places=12)
            self.assertEqual((actual['n_samples'],actual['n_batches'],actual['batch_size']),(8,2,4))
        self.assertGreater(find(output,'interaction_w_residual_minus_rzf_continue')['mean'],0.)
        self.assertLess(find(output,'w_residual_hybrid_minus_fdma')['mean'],0.)

    def test_all_ten_contrasts_and_low_gamma_signed_results_are_retained(self):
        output=results.paired_comparisons(final_rows(),samples=8)
        labels={r['comparison'] for r in output}
        self.assertEqual(labels,{
            *(a+'_w_residual_minus_rzf_continue' for a in ARMS),
            'interaction_w_residual_minus_rzf_continue',
            *(a+'_rzf_continue_minus_rate_source' for a in ARMS),
            *(a+'_w_residual_minus_rate_source' for a in ARMS),
            *(p+'_hybrid_minus_fdma' for p in POLICIES)})
        low=find(output,'interaction_w_residual_minus_rzf_continue','low_0_5_10')
        self.assertAlmostEqual(low['mean'],.17-.01*3.5,places=12)
        for label in labels:
            point=[r for r in output if r['comparison']==label
                   and r['scope']=='point' and r['metric']=='sum_rate_bps_hz']
            self.assertEqual({r['gamma_db'] for r in point},set(GAMMAS))
            self.assertEqual(len(point),6)
        # Explicit secondary effects from this fixture, across all six gamma.
        for arm,continued,residual in [('fdma',.495,.1925),('hybrid_g4',.715,.3575)]:
            self.assertAlmostEqual(find(output,arm+'_rzf_continue_minus_rate_source')['mean'],continued,places=12)
            self.assertAlmostEqual(find(output,arm+'_w_residual_minus_rate_source')['mean'],continued+residual,places=12)
        for policy,delta in [('rate_source',-1.1),('rzf_continue',-.88),('w_residual',-.715)]:
            self.assertAlmostEqual(find(output,policy+'_hybrid_minus_fdma')['mean'],delta,places=12)

    def test_all_users_weakest_user_and_common_physical_diagnostics_survive(self):
        output=results.paired_comparisons(final_rows(),samples=8)
        primary='hybrid_g4_w_residual_minus_rzf_continue'
        for metric in [*(f'user_{i}_rate_bps_hz' for i in range(8)),
                'min_user_rate_bps_hz','interference_to_signal_ratio',
                'predicted_dl_energy','rzf_power_max_error','residual_relative_norm',
                'data_energy','frame_energy']:
            self.assertIsNotNone(find(output,primary,metric=metric))
        self.assertIsNotNone(find(output,primary,metric='soft_shared_nmse_linear'))
        self.assertFalse(any(r['metric']=='soft_shared_nmse_linear' and
                             r['comparison']=='w_residual_hybrid_minus_fdma' for r in output))
        gain=find(output,primary)['mean']
        self.assertAlmostEqual(sum(find(output,primary,metric=f'user_{i}_rate_bps_hz')['mean']
                                   for i in range(8)),gain,places=12)

    def test_row_order_cannot_change_selection_or_sample_count(self):
        rows=final_rows();expected=results.paired_comparisons(rows,samples=8)
        random.Random(713).shuffle(rows)
        self.assertEqual(results.paired_comparisons(rows,samples=8),expected)
        self.assertTrue(all(r['n_samples']==8 and r['n_batches']==2 for r in expected))

    def test_missing_duplicate_unknown_method_and_wrong_batch_identity_fail(self):
        rows=final_rows()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):results.paired_comparisons(bad,samples=8)
        for field,value in [('arm','fdma'),('arm','winner_fdma'),('gamma_db',30.),
                            ('gamma_db',True),('sample_index',True),('sample_index',8),
                            ('batch_index',1),('within_batch',4)]:
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                results.paired_comparisons(bad,samples=8)
        for samples,batch_size in [(0,4),(7,4),(True,4),(8,0),(8,True)]:
            with self.subTest(samples=samples,batch_size=batch_size),self.assertRaises(ValueError):
                results.paired_comparisons(rows,samples=samples,batch_size=batch_size)

    def test_nonfinite_boolean_or_nonnumeric_diagnostics_fail(self):
        rows=final_rows()
        for metric,value in [('sum_rate_bps_hz',math.nan),('sum_rate_bps_hz',math.inf),
                ('sum_rate_bps_hz',True),('predicted_dl_energy','selected'),
                ('interference_power_mean',-1.),('role','test')]:
            bad=copy.deepcopy(rows);bad[0][metric]=value
            with self.subTest(metric=metric,value=value),self.assertRaises(ValueError):
                results.paired_comparisons(bad,samples=8)

    def test_optional_partial_conditions_and_missing_required_rates_fail(self):
        rows=final_rows()
        for missing in (None,'delete'):
            bad=copy.deepcopy(rows)
            if missing is None:bad[0]['user_7_rate_bps_hz']=None
            else:del bad[0]['user_7_rate_bps_hz']
            with self.assertRaisesRegex(ValueError,'Partially missing'):
                results.paired_comparisons(bad,samples=8)
        for required in results.tables.REQUIRED_METRICS:
            bad=copy.deepcopy(rows)
            for row in bad:
                if row['arm']=='rate_source_fdma' and row['gamma_db']==0.:row[required]=None
            with self.subTest(required=required),self.assertRaisesRegex(ValueError,'required final metric'):
                results.paired_comparisons(bad,samples=8)
        bad=copy.deepcopy(rows);del bad[0]['dl_nmse_linear']
        with self.assertRaises(ValueError):results.paired_comparisons(bad,samples=8)

    def test_summary_uses_linear_nmse_mean_and_preserves_six_methods(self):
        summary=results.summarize_rows(final_rows(),samples=8)
        self.assertEqual(len(summary),6*6)
        self.assertEqual({r['arm'] for r in summary},set(configuration().METHODS))
        one=next(r for r in summary if r['arm']=='rate_source_fdma' and r['gamma_db']==0.)
        self.assertEqual((one['n_samples'],one['n_batches'],one['batch_size']),(8,2,4))
        self.assertAlmostEqual(one['sum_rate_bps_hz_mean'],13.05,places=12)
        self.assertAlmostEqual(one['dl_nmse_db_from_linear_mean'],10*math.log10(.0235),places=12)
        self.assertIn('min_user_rate_bps_hz_mean',one)

    def test_one_physical_batch_has_no_batch_interval(self):
        one=find(results.paired_comparisons(final_rows(samples=4),samples=4),
                 'fdma_w_residual_minus_rzf_continue')
        self.assertEqual((one['n_samples'],one['n_batches']),(4,1))
        self.assertIsNone(one['batch_se'])
        self.assertIsNone(one['batch_ci95_low'])
        self.assertIsNone(one['batch_ci95_high'])
        self.assertIsNotNone(one['sample_se'])


if __name__=='__main__':unittest.main()
