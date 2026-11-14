"""Independent complete-MU arithmetic checks for paired adaptation results."""
import copy
import csv
import gzip
import io
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from configs import twc_v1_k8_ul_adapt_config as cfg
import twc_v1_k8_ul_adapt_results as results


def fixture_rows(samples=8):
    rows=[]
    for phase in cfg.PHASES:
        p=int(phase=='final')
        for arm in cfg.ARMS:
            a=int(arm=='hybrid_g4')
            for method in cfg.TRAIN_METHODS:
                m=int(method=='moment_receiver')
                for gamma in cfg.GAMMAS:
                    for i in range(samples):
                        base=10+a+2*m+3*p+4*p*m+5*p*a+6*p*m*a
                        rate=base+.02*gamma+.1*i
                        metrics={name:.1 for name in results.REQUIRED_METRICS}
                        metrics.update(sum_rate_bps_hz=rate,dl_nmse_linear=.01+.001*i,
                            diagnostic_signed=-.2+.01*i,paired_noise=(i-3.5)*(gamma-12.5))
                        rows.append(dict(role='final',phase=phase,arm=arm,method=method,
                            gamma_db=float(gamma),sample_index=i,batch_index=i//4,within_batch=i%4,
                            **metrics))
    return rows


class PairedAdaptationStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.rows=fixture_rows()

    def test_complete_design_and_64_summaries(self):
        copied,index=results.validate_rows(self.rows,samples=8)
        self.assertEqual(len(index),2*2*2*6*8)
        self.assertIsNot(copied[0],self.rows[0])
        summary=results.summarize_rows(self.rows,samples=8)
        self.assertEqual(len(summary),64)
        self.assertTrue(all(r['n_samples']==8 and r['n_batches']==2 for r in summary))
        row=next(r for r in summary if r['phase']=='source' and r['arm']=='fdma'
            and r['method']=='original' and r['scope']=='all_six')
        values=[10+.02*12.5+.1*i for i in range(8)]
        expected=sum(values)/len(values)
        se=math.sqrt(sum((x-expected)**2 for x in values)/(len(values)-1))/math.sqrt(len(values))
        self.assertAlmostEqual(row['sum_rate_bps_hz_mean'],expected)
        self.assertAlmostEqual(row['sum_rate_bps_hz_se'],se)
        self.assertAlmostEqual(row['sum_rate_bps_hz_ci95_low'],expected-1.96*se)
        self.assertAlmostEqual(row['dl_nmse_db_from_linear_mean'],10*math.log10(.0135))

    def test_gamma_views_average_within_mu_before_uncertainty(self):
        row=next(r for r in results.summarize_rows(self.rows,samples=8)
            if r['phase']=='source' and r['arm']=='fdma' and r['method']=='original'
            and r['scope']=='all_six')
        self.assertEqual(row['n_gamma_points'],6)
        self.assertAlmostEqual(row['paired_noise_mean'],0)
        self.assertAlmostEqual(row['paired_noise_se'],0)
        self.assertAlmostEqual(row['paired_noise_batch_se'],0)
        self.assertAlmostEqual(row['sum_rate_bps_hz_batch_se'],.2)

    def test_matched_gains_and_difference_in_differences(self):
        comparisons=results.paired_comparisons(self.rows,samples=8)
        actual={r['comparison']:r for r in comparisons
            if r['scope']=='all_six' and r['metric']=='sum_rate_bps_hz'}
        expected={
            'fdma_original_adapt_gain':3.,'fdma_moment_receiver_adapt_gain':7.,
            'hybrid_g4_original_adapt_gain':8.,'hybrid_g4_moment_receiver_adapt_gain':18.,
            'fdma_moment_gain_minus_original_gain':4.,
            'hybrid_g4_moment_gain_minus_original_gain':10.,
            'interaction_moment_adaptation_hybrid_minus_fdma':6.,
            'source_original_hybrid_minus_fdma':1.,
            'final_original_hybrid_minus_fdma':6.,
            'final_moment_receiver_hybrid_minus_fdma':12.,
        }
        self.assertEqual(len(actual),17)
        for name,value in expected.items():
            self.assertAlmostEqual(actual[name]['mean'],value,name)
            self.assertAlmostEqual(actual[name]['batch_se'],0,name)

    def test_negative_receiver_and_adaptation_results_are_not_filtered(self):
        for row in self.rows:
            if row['phase']=='final' and row['method']=='moment_receiver':
                row['sum_rate_bps_hz']-=18
        comparison=next(r for r in results.paired_comparisons(self.rows,samples=8)
            if r['comparison']=='fdma_moment_receiver_adapt_gain'
            and r['metric']=='sum_rate_bps_hz' and r['scope']=='all_six')
        self.assertAlmostEqual(comparison['mean'],-11)
        self.assertLess(comparison['batch_ci95_high'],0)

    def test_missing_duplicate_and_wrong_batch_or_gamma_are_rejected(self):
        for mutate in ('missing','duplicate','batch','within','gamma','phase','method'):
            rows=copy.deepcopy(self.rows)
            if mutate=='missing':rows.pop()
            elif mutate=='duplicate':rows.append(dict(rows[0]))
            elif mutate=='batch':rows[0]['batch_index']=1
            elif mutate=='within':rows[0]['within_batch']=3
            elif mutate=='gamma':rows[0]['gamma_db']=2.5
            elif mutate=='phase':rows[0]['phase']='monitor'
            else:rows[0]['method']='reliability_only'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                results.validate_rows(rows,samples=8)

    def test_training_monitor_calibration_and_test_roles_are_rejected(self):
        for role in ('codec_train','monitor','calibration','test'):
            rows=copy.deepcopy(self.rows);rows[0]['role']=role
            with self.subTest(role=role),self.assertRaises(ValueError):
                results.validate_rows(rows,samples=8)
        rows=copy.deepcopy(self.rows);rows[0].pop('role')
        with self.assertRaisesRegex(ValueError,'consistently'):
            results.validate_rows(rows,samples=8)

    def test_nonfinite_boolean_and_negative_required_metrics_rejected(self):
        for value in (math.nan,math.inf,-1.,True,None):
            rows=copy.deepcopy(self.rows);rows[0]['sum_rate_bps_hz']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                results.validate_rows(rows,samples=8)
        rows=copy.deepcopy(self.rows);rows[0]['sample_index']=False
        with self.assertRaises(ValueError):results.validate_rows(rows,samples=8)

    def test_optional_hybrid_metrics_must_not_disappear_in_a_control(self):
        for row in self.rows:
            row['shared_diagnostic']=1. if row['arm']=='hybrid_g4' else None
        results.validate_rows(self.rows,samples=8)
        next(row for row in self.rows if row['arm']=='hybrid_g4')['shared_diagnostic']=None
        with self.assertRaisesRegex(ValueError,'Partially missing'):
            results.validate_rows(self.rows,samples=8)

    def test_gzip_csv_roundtrip_and_duplicate_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'samples.csv.gz'
            with gzip.open(path,'wt',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=sorted(self.rows[0]))
                writer.writeheader();writer.writerows(self.rows)
            decoded=results.read_raw_csv(path)
            self.assertEqual(decoded,self.rows)
            with gzip.open(path,'wt') as stream:stream.write('arm,arm\nfdma,fdma\n')
            with self.assertRaises(ValueError):results.read_raw_csv(path)

    def test_single_mu_has_no_identifiable_standard_error(self):
        value=results.moments([2.])
        self.assertEqual(value['mean'],2.)
        self.assertIsNone(value['se']);self.assertIsNone(value['ci95_low'])
        for samples,batch in ((0,4),(7,4),(8,0),(True,4)):
            with self.assertRaises(ValueError):results.validate_rows(self.rows,samples=samples,batch_size=batch)


if __name__=='__main__':unittest.main()
