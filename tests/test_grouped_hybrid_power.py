"""Small real CPU tensors only; full source/CUDA acceptance is explicit server work."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from configs.grouped_hybrid_power_train_config import protocol
from configs.grouped_hybrid_anchor_train_config import protocol as old


class ProtocolTests(unittest.TestCase):
    def test_new_identity_same_physical_geometry_without_old_global_mutation(self):
        old_metadata=deepcopy(old.protocol_metadata())
        for arm in protocol.ARMS:
            layout=protocol.LayoutSpec(arm);reference=old.LayoutSpec(protocol.source_arm(arm))
            self.assertEqual(layout.arm,arm)
            self.assertEqual(layout.to_dict()['arm'],arm)
            for key in ('df','da','active_per_user','active_slots','exclusive_slots','shared_slots',
                        'supports','canonical_to_physical','physical_to_canonical','receive_groups'):
                self.assertEqual(getattr(layout,key),getattr(reference,key))
            self.assertEqual(4*layout.df+2*layout.da,256)
            self.assertEqual(protocol.architecture_metadata(arm)['bs'],old.architecture_metadata(protocol.source_arm(arm))['bs'])
        self.assertEqual(old_metadata,old.protocol_metadata())
        with self.assertRaises(ValueError):protocol.LayoutSpec('hybrid_f16_a96')

    def test_new_addresses_separate_physical_trials_but_not_training_arms(self):
        before=random.getstate()
        values=[protocol.sample_gamma(20261030,i) for i in (1,500,501,20000)]
        self.assertEqual(before,random.getstate())
        self.assertEqual(values,[protocol.sample_gamma(20261030,i) for i in (1,500,501,20000)])
        self.assertNotEqual(values,[old.sample_gamma(20261030,i) for i in (1,500,501,20000)])
        for address in (protocol.RandomAddress(20261030,'train',500,'standard_noise'),
                        protocol.RandomAddress(20270108,'test',0,'standard_noise')):
            self.assertEqual(address.to_dict()['namespace'],'grouped_hybrid_power_v1')
            self.assertNotIn('arm',address.to_dict())
        self.assertEqual(protocol.noise_variance(0),.25)
        self.assertEqual(protocol.noise_variance(25),old.noise_variance(25))
        with self.assertRaises(ValueError):protocol.namespace_seed(True,'channel')


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Real PyTorch unavailable; not installed or substituted')
class SmallGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch

    def raw(self):
        t=self.t;g=t.Generator().manual_seed(41)
        return t.view_as_complex(t.randn(2,4,256,2,generator=g))

    def test_gate_count_zero_initialization_and_rng_isolation(self):
        t=self.t
        for arm,count in zip(protocol.ARMS,(0,0,1,6)):
            before=t.get_rng_state().clone();gate=protocol.make_power_gate(arm)
            self.assertTrue(t.equal(before,t.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in gate.parameters()),count)
            self.assertEqual(set(gate.state_dict()),{'theta'} if count else set())
            raw=self.raw();gated,value=gate(raw,7.5,protocol.LayoutSpec(arm))
            self.assertTrue(t.equal(gated,raw));self.assertEqual(value.item(),0)
            if count:
                with self.assertRaises(RuntimeError):gate.load_state_dict({},strict=True)
                other=protocol.make_power_gate(arm);other.load_state_dict(gate.state_dict(),strict=True)
                self.assertTrue(t.equal(other.theta,gate.theta))
        with self.assertRaises(RuntimeError):
            protocol.make_power_gate('h32_gamma_power').load_state_dict({'theta':t.zeros(1)},strict=True)

    def test_interpolation_is_before_tanh_and_constant_is_gamma_independent(self):
        t=self.t;theta=t.tensor([-1.,-.3,.2,.5,.9,1.4],requires_grad=True)
        for gamma,index in zip(protocol.NODES,range(6)):
            self.assertTrue(t.equal(protocol.gate_value(theta,gamma),math.log(4)*theta[index].tanh()))
        value=protocol.gate_value(theta,7.5)
        self.assertTrue(t.equal(value,math.log(4)*((theta[1]+theta[2])/2).tanh()))
        value.backward()
        self.assertTrue(t.isfinite(theta.grad).all())
        self.assertEqual(t.count_nonzero(theta.grad).item(),2)
        self.assertGreater(theta.grad[1].item(),0)
        single=t.tensor([.3])
        self.assertEqual(len({protocol.gate_value(single,g).item() for g in protocol.NODES}),1)
        for gamma in (-1,26,float('nan')):
            with self.assertRaises(ValueError):protocol.gate_value(theta,gamma)

    def test_joint_energy_ratio_bounds_and_both_directions(self):
        t=self.t;layout=protocol.LayoutSpec('h32_static_power');raw=self.raw()
        geometry=protocol.tensor_geometry(layout)
        _,base,_=protocol.normalize_and_map(raw,layout)
        def ratio(z):
            p=z.abs().square()
            return p.masked_fill(~geometry['exclusive_mask'][None],0).sum(-1)/p.masked_fill(~geometry['shared_mask'][None],0).sum(-1)
        for theta_value in (-20.,-1.,0.,1.,20.):
            g=protocol.gate_value(t.tensor([theta_value]),10)
            gated=protocol.gate_raw_symbols(raw,layout,g)
            self.assertTrue(t.equal(gated[:,~geometry['exclusive_mask']],raw[:,~geometry['exclusive_mask']]))
            X,z,energy=protocol.normalize_and_map(gated,layout)
            self.assertTrue(t.allclose(energy,t.full_like(energy,64),rtol=0,atol=1e-4))
            self.assertTrue(t.allclose(ratio(z)/ratio(base),g.exp().expand_as(energy),rtol=1e-6,atol=1e-6))
            self.assertGreaterEqual(g.exp().item(),.25);self.assertLessEqual(g.exp().item(),4.)
            self.assertEqual(t.count_nonzero(X.masked_select(~geometry['physical_mask'][None])).item(),0)

    def test_zero_gate_actual_small_physical_and_receiver_path_exact(self):
        t=self.t;raw=self.raw();generator=t.Generator().manual_seed(76)
        H=t.view_as_complex(t.randn(2,4,32,256,2,generator=generator))
        for arm in protocol.ARMS:
            layout=protocol.LayoutSpec(arm);reference=old.LayoutSpec(protocol.source_arm(arm))
            gate=protocol.make_power_gate(arm)
            eps=protocol.standard_noise(20261030,'preflight',0,2)
            prior=protocol.make_power_prior(layout).read(10)
            for gamma in (0,12.5,25):
                gated,_=gate(raw,gamma,layout)
                a,az,ae=protocol.normalize_and_map(gated,layout)
                b,bz,be=old.normalize_and_map(raw,reference)
                self.assertTrue(t.equal(a,b) and t.equal(az,bz) and t.equal(ae,be))
                aY=protocol.transmit(H,a,layout,N0=protocol.noise_variance(gamma),standard_noise=eps)
                bY=old.transmit(H,b,reference,N0=old.noise_variance(gamma),standard_noise=eps)
                self.assertTrue(t.equal(aY,bY))
                ra=protocol.recover(aY,H,layout,N0=protocol.noise_variance(gamma),prior_power=prior)
                rb=old.recover(bY,H,reference,N0=old.noise_variance(gamma),prior_power=prior)
                self.assertTrue(all(t.equal(ra[k],rb[k]) for k in ra))

    def test_zero_theta_gradient_matches_analytic_joint_energy_derivative(self):
        t=self.t;layout=protocol.LayoutSpec('h32_gamma_power')
        raw=t.ones(1,4,256,dtype=t.complex64,requires_grad=True)
        gate=protocol.make_power_gate('h32_gamma_power')
        gated,_=gate(raw,7.5,layout);_,z,_=protocol.normalize_and_map(gated,layout)
        mask=protocol.tensor_geometry(layout)['exclusive_mask']
        energy=z.abs().square().masked_fill(~mask[None],0).sum(-1).mean()
        energy.backward()
        derivative=64*32*64/(96**2)*math.log(4)
        expected=t.tensor([0,derivative/2,derivative/2,0,0,0])
        self.assertTrue(t.allclose(gate.theta.grad,expected,rtol=1e-6,atol=1e-6))
        self.assertTrue(t.isfinite(raw.grad).all())
        self.assertGreater(raw.grad.abs().sum().item(),0)

    def test_cumulative_prior_and_strict_500_to501_restore_use_gated_symbols(self):
        t=self.t;layout=protocol.LayoutSpec('h32_static_power');gate=protocol.make_power_gate(layout.arm)
        # Synthetic history fixture, not a claim that server source was inspected.
        prior=protocol.make_power_prior(layout)
        prior.update_count.fill_(20499);prior.node_update_mass[0]=20499
        prior.validate_state(20499)
        with t.no_grad():gate.theta.fill_(.5)
        gated,_=gate(self.raw(),7.5,layout);_,z,_=protocol.normalize_and_map(gated,layout)
        original={k:v.clone() for k,v in prior.state_dict().items()}
        pending=prior.pending(z,7.5,20500)
        self.assertTrue(all(t.equal(original[k],v) for k,v in prior.state_dict().items()))
        self.assertTrue(t.equal(pending['power_mean'],z.detach().abs().square().mean(0)))
        with self.assertRaises(ValueError):prior.pending(z,7.5,500)
        prior.commit(pending);prior.validate_state(20500)
        restored=protocol.make_power_prior(layout);restored.load_state_dict(prior.state_dict(),strict=True)
        for model in (prior,restored):model.commit(model.pending(z,25,20501))
        self.assertTrue(all(t.equal(v,restored.state_dict()[k]) for k,v in prior.state_dict().items()))
        restored.eval()
        with self.assertRaises(ValueError):restored.commit(restored.pending(z,0,20502))
        restored.validate_state(20501)

    def test_nonfinite_gate_and_incorrect_shape_fail_closed(self):
        t=self.t
        for theta in (t.tensor([float('nan')]),t.zeros(2),t.zeros(1,dtype=t.float64)):
            with self.assertRaises((ValueError,FloatingPointError)):protocol.gate_value(theta,10)


class ServerBoundaryTests(unittest.TestCase):
    @unittest.skip('Real source checkpoints, full model equivalence and physical batch48 CUDA run only in explicit server preflight; not run on laptop')
    def test_actual_checkpoint_zero_gate_and_cuda_batch48(self):
        pass


if __name__=='__main__':unittest.main()
