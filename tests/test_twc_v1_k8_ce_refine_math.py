"""Real small CPU autograd and independently evaluated CE geometry/probes."""
import copy
from types import SimpleNamespace
import unittest

import torch
import twc_v1_k8_ce_refine as math


class CELossTests(unittest.TestCase):
    def setUp(self):
        gen=torch.Generator().manual_seed(921)
        self.target=torch.complex(torch.randn(3,4,5,7,generator=gen),torch.randn(3,4,5,7,generator=gen))
        self.pred=(self.target+.2*torch.complex(torch.randn(3,4,5,7,generator=gen),torch.randn(3,4,5,7,generator=gen))).requires_grad_()

    def test_exact_channel_zero_errors_and_unit_energy(self):
        score=math.vectors(self.target,self.target)
        torch.testing.assert_close(score['ul_nmse_linear'],torch.zeros(3))
        torch.testing.assert_close(score['direction_error'],torch.zeros(3),atol=2e-7,rtol=0)
        torch.testing.assert_close(score['gram_error'],torch.zeros(3))
        torch.testing.assert_close(score['energy_ratio'],torch.ones(3))

    def test_real_cpu_gradients_for_both_objectives_and_frozen_labels(self):
        label=self.target.clone().requires_grad_()
        for trajectory in ('nmse','geometry'):
            pred=self.pred.detach().clone().requires_grad_()
            loss,score=math.objective(pred,label,trajectory)
            expected=score['ul_nmse_linear'].mean()
            if trajectory=='geometry':expected=expected+score['direction_error'].mean()+score['gram_error'].mean()
            torch.testing.assert_close(loss,expected)
            loss.backward()
            self.assertIsNone(label.grad)
            self.assertTrue(bool(torch.isfinite(pred.grad).all()))
            self.assertGreater(float(pred.grad.abs().sum()),0.)

    def test_nmse_matches_direct_per_link_reference_not_global_energy_ratio(self):
        score=math.vectors(self.pred,self.target)
        ref=[]
        for b in range(3):
            links=[]
            for k in range(4):
                links.append(float(((self.pred[b,k]-self.target[b,k]).abs().square().sum()/self.target[b,k].abs().square().sum()).detach()))
            ref.append(sum(links)/4)
        torch.testing.assert_close(score['ul_nmse_linear'],torch.tensor(ref))

    def test_direction_is_phase_scale_invariant_but_physical_nmse_is_not(self):
        scaled=self.target*(2j)
        score=math.vectors(scaled,self.target)
        self.assertTrue(bool((score['ul_nmse_linear']>1).all()))
        torch.testing.assert_close(score['direction_error'],torch.zeros(3),atol=2e-7,rtol=0)
        torch.testing.assert_close(score['gram_error'],torch.zeros(3),atol=2e-13,rtol=0)

    def test_gram_detects_relative_user_phase_invisible_to_projective_direction(self):
        pred=self.target.clone();pred[:,0]*=1j
        score=math.vectors(pred,self.target)
        torch.testing.assert_close(score['direction_error'],torch.zeros(3),atol=2e-7,rtol=0)
        self.assertGreater(float(score['gram_error'].min()),0.)

    def test_invalid_or_undefined_geometry_rejected(self):
        for trajectory in ('fdma','other'):
            with self.assertRaises(ValueError):math.objective(self.pred,self.target,trajectory)
        bad=self.target.clone();bad[:,0]=0
        with self.assertRaises(ValueError):math.vectors(bad,self.target)
        bad=self.pred.detach().clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):math.vectors(bad,self.target)


class PublicRecoveryProbeTests(unittest.TestCase):
    def setUp(self):
        gen=torch.Generator().manual_seed(753)
        self.true=torch.complex(torch.randn(2,4,5,6,generator=gen),torch.randn(2,4,5,6,generator=gen))
        self.est=self.true+.1*(torch.randn(2,4,5,6,generator=gen)+1j*torch.randn(2,4,5,6,generator=gen))
        self.variance=torch.full((2,4,6),.01)
        self.layout=SimpleNamespace(groups=((0,1),(2,3)),shared_positions_by_group=((0,2,4),(1,3,5)))

    def test_probe_matches_direct_conditional_mse_and_filters_ignore_truth(self):
        p=48/44
        filters,n0=math.shared_filters(self.est,self.variance,10.,self.layout)
        score=math.shared_uniform_prior_score(self.est,self.true,self.variance,10.,self.layout)
        reference=[]
        for b in range(2):
            all_rows=[]
            for a,pair,tones in zip(filters,self.layout.groups,self.layout.shared_positions_by_group):
                for j,t in enumerate(tones):
                    g=self.true[b,list(pair),:,t].T
                    error=a[b,j]@g-torch.eye(2)
                    row=p*(error.abs().square()).sum(-1)+n0[b]*(a[b,j].abs().square()).sum(-1)
                    all_rows.extend(row.tolist())
            reference.append(sum(all_rows)/len(all_rows))
        torch.testing.assert_close(score['shared_uniform_prior_mse'],torch.tensor(reference),rtol=2e-6,atol=1e-8)
        # Truth affects only the score: public design has no truth argument.
        other=math.shared_uniform_prior_score(self.est,self.true*2,self.variance,10.,self.layout)
        after,_=math.shared_filters(self.est,self.variance,10.,self.layout)
        for first,last in zip(filters,after):torch.testing.assert_close(first,last,rtol=0,atol=0)
        self.assertFalse(torch.allclose(score['shared_uniform_prior_mse'],other['shared_uniform_prior_mse']))

    def test_zero_calibration_and_true_channel_remain_finite_without_oracle_recovery(self):
        score=math.shared_uniform_prior_score(self.true,self.true,torch.zeros_like(self.variance),25.,self.layout)
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in score.values()))
        self.assertTrue(bool((score['shared_uniform_prior_mse']>0).all()))
        self.assertEqual(float(score['shared_estimated_rank_deficient_fraction'].sum()),0.)

    def test_learned_gamma_and_current_frame_variance_are_refused(self):
        with self.assertRaises(ValueError):math.shared_filters(self.est,self.variance,torch.tensor(10.,requires_grad=True),self.layout)
        with self.assertRaises(ValueError):math.shared_filters(self.est,self.variance.requires_grad_(),10.,self.layout)


if __name__=='__main__':unittest.main()
