"""Small real CPU protocol/EMA/anchor tests; full model checks explicitly server-only."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import importlib.metadata
import math
import os
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from configs.grouped_hybrid_anchor_train_config import protocol
from configs.grouped_codec_train_config import protocol as old_protocol


class MappingTests(unittest.TestCase):
    def test_exact_budget_bidirectional_maps_and_group_occupancy(self):
        for arm in protocol.ARMS:
            layout=protocol.LayoutSpec(arm)
            self.assertEqual(4*layout.df+2*layout.da,256)
            self.assertEqual(len(layout.exclusive_slots),layout.df)
            self.assertEqual(len(layout.shared_slots),layout.da)
            self.assertEqual(len(layout.active_slots),layout.df+layout.da)
            occupancy={q:tuple(k for k,s in enumerate(layout.supports) if q in s) for q in range(256)}
            self.assertEqual(sum(len(x)==1 for x in occupancy.values()),4*layout.df)
            self.assertEqual(sum(len(x)==2 for x in occupancy.values()),2*layout.da)
            self.assertTrue(all(len({k//2 for k in users})==1 for users in occupancy.values()))
            for k in range(4):
                for slot,q in enumerate(layout.canonical_to_physical[k]):
                    if q>=0:
                        self.assertEqual(layout.physical_to_canonical[k][q],slot)
                        self.assertEqual(len(occupancy[q]),1 if slot in layout.exclusive_slots else 2)
            for sb in range(4):
                active=[q for q in layout.active_slots if q//64==sb]
                self.assertEqual(active,list(range(sb*64,sb*64+(layout.df+layout.da)//4)))

    def test_control_endpoint_matches_existing_F_exactly(self):
        new=protocol.LayoutSpec('fdma_control');old=old_protocol.LayoutSpec('fdma')
        self.assertEqual(new.active_slots,old.active_slots)
        self.assertEqual(new.supports,old.supports)
        self.assertEqual(new.canonical_to_physical,old.canonical_to_physical)
        self.assertEqual(new.shared_slots,())

    def test_new_random_namespaces_ignore_arm_and_external_rng(self):
        before=random.getstate()
        values=[protocol.sample_gamma(20261030,i) for i in (1,500,501,20000)]
        self.assertEqual(before,random.getstate())
        self.assertNotEqual(values,[old_protocol.sample_gamma(20261030,i) for i in (1,500,501,20000)])
        seeds=[protocol.namespace_seed(20261030,x) for x in ('channel','initialization','channel_test','channel_validation')]
        self.assertEqual(len(set(seeds)),4)
        self.assertTrue(all(0<=s<2**32 for s in seeds))
        self.assertTrue(set(seeds).isdisjoint({20261030,20261031,20261210,20261220}))
        for arm in protocol.ARMS:
            self.assertNotIn(arm,str(protocol.RandomAddress(20261030,'train',500,'standard_noise').to_dict()))

    def test_interpolation_endpoints_and_fixed_physical_definition(self):
        self.assertEqual(protocol.interpolation(0),((0,1.),))
        self.assertEqual(protocol.interpolation(25),((5,1.),))
        self.assertEqual(protocol.interpolation(7.5),((1,.5),(2,.5)))
        for gamma in protocol.NODES:
            self.assertEqual(protocol.noise_variance(gamma),old_protocol.noise_variance(gamma))
        for gamma in (-1,26,float('nan'),True,'10'):
            with self.assertRaises(ValueError):protocol.noise_variance(gamma)
        with self.assertRaises(ValueError):protocol.LayoutSpec('legacy_group2')
        old=old_protocol.architecture_metadata('legacy_all4')['bs']
        new=protocol.architecture_metadata('fdma_control')['bs']
        self.assertIn('log1p(error_proxy/pi.clamp_min(1e-8))',new['memory_features'])
        self.assertNotIn('log1p_reliability_proxy',new['memory_features'])
        self.assertIn('historical per-user coordinate power',new['public_reliability'])
        self.assertIn('not calibrated posterior variance or precision',new['public_reliability'])
        for key in old.keys()-{'memory_features','public_reliability'}:
            self.assertEqual(new[key],old[key])
        self.assertIn('log1p_reliability_proxy',old_protocol.architecture_metadata('legacy_all4')['bs']['memory_features'])


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Real PyTorch unavailable; no dependency substitution')
class SmallTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch

    def raw(self,batch=2):
        t=self.torch
        g=t.Generator().manual_seed(73)
        return t.view_as_complex(t.randn(batch,4,256,2,generator=g))

    def test_joint_normalization_exclusive_plus_shared_is_64_not_128(self):
        t=self.torch
        for arm in protocol.ARMS:
            layout=protocol.LayoutSpec(arm)
            X,z,E=protocol.normalize_and_map(self.raw(),layout)
            geometry=protocol.tensor_geometry(layout)
            self.assertTrue(t.allclose(E,t.full_like(E,64),rtol=0,atol=1e-4))
            exclusive=z.abs().square().masked_fill(~geometry['exclusive_mask'][None],0).sum(-1)
            shared=z.abs().square().masked_fill(~geometry['shared_mask'][None],0).sum(-1)
            self.assertTrue(t.allclose(exclusive+shared,E,rtol=0,atol=1e-5))
            self.assertEqual(t.count_nonzero(X.masked_select(~geometry['physical_mask'][None])).item(),0)
            if layout.da:
                self.assertTrue((exclusive>0).all() and (shared>0).all())
                self.assertTrue((exclusive<64).all() and (shared<64).all())
        with self.assertRaises(FloatingPointError):
            protocol.normalize_and_map(t.zeros(1,4,256,dtype=t.complex64),protocol.LayoutSpec('fdma_control'))

    def test_one_full_mac_and_one_absolute_noise_tensor(self):
        t=self.torch;layout=protocol.LayoutSpec('hybrid_f16_a96')
        X,_,_=protocol.normalize_and_map(self.raw(),layout)
        H=t.ones(2,4,32,256,dtype=t.complex64)
        before=t.get_rng_state().clone()
        eps=protocol.standard_noise(20261030,'preflight',0,2)
        self.assertTrue(t.equal(before,t.get_rng_state()))
        self.assertTrue(t.equal(eps,protocol.standard_noise(20261030,'preflight',0,2)))
        N0=protocol.noise_variance(10)
        actual=protocol.transmit(H,X,layout,N0=N0,standard_noise=eps)
        expected=(H*X[:,:,None]).sum(1)+math.sqrt(N0/2)*eps
        self.assertTrue(t.equal(actual,expected))
        corrupted=X.clone();corrupted[0,0,2]=1
        with self.assertRaises(ValueError):protocol.transmit(H,corrupted,layout,N0=N0,standard_noise=eps)

    def test_receiver_matches_direct_full_error_formula_with_nonuniform_user_pi(self):
        t=self.torch;layout=protocol.LayoutSpec('hybrid_f32_a64')
        g=t.Generator().manual_seed(18)
        H=t.view_as_complex(t.randn(1,4,32,256,2,generator=g))
        Y=t.view_as_complex(t.randn(1,32,256,2,generator=g))
        prior=protocol.make_power_prior(layout)
        pi=prior.read(10)
        for k in range(4):
            values=t.linspace(.1+k,.8+2*k,layout.active_per_user)
            pi[k,list(layout.active_slots)]=64*values/values.sum()
        N0=protocol.noise_variance(10)
        out=protocol.recover(Y,H,layout,N0=N0,prior_power=pi)
        for users,resources in layout.receive_groups:
            q=resources[0];slots=[layout.physical_to_canonical[k][q] for k in users]
            G=H[0,list(users),:,q].T
            power=t.stack([pi[k,s] for k,s in zip(users,slots)])
            I=t.eye(len(users),dtype=t.complex64)
            L=t.linalg.solve(G.mH@G+N0*t.diag(1/power)+protocol.RIDGE*I,G.mH)
            expected=L@Y[0,:,q]
            M=L@G-I
            covariance=M@t.diag(power.to(t.complex64))@M.mH+N0*L@L.mH
            for j,(k,s) in enumerate(zip(users,slots)):
                self.assertTrue(t.allclose(out['z_hat'][0,k,s],expected[j],rtol=1e-5,atol=1e-6))
                self.assertTrue(t.allclose(out['error_proxy'][0,k,s],covariance[j,j].real,rtol=1e-5,atol=1e-7))
        mask=protocol.tensor_geometry(layout)['active_mask']
        self.assertEqual(t.count_nonzero(out['error_proxy'][:,~mask]).item(),0)
        self.assertTrue(t.allclose(out['error_feature'],t.log1p(out['error_proxy']/pi.clamp_min(1e-8)[None]),rtol=0,atol=0))

    def test_fdma_uniform_prior_recovery_matches_previous_linear_receiver(self):
        t=self.torch;layout=protocol.LayoutSpec('fdma_control');old=old_protocol.LayoutSpec('fdma')
        g=t.Generator().manual_seed(24)
        H=t.view_as_complex(t.randn(1,4,32,256,2,generator=g))
        Y=t.view_as_complex(t.randn(1,32,256,2,generator=g))
        N0=.25
        a=protocol.recover(Y,H,layout,N0=N0,prior_power=protocol.make_power_prior(layout).read(0))
        b=old_protocol.recover(Y,H,old,N0=N0)
        self.assertTrue(t.allclose(a['z_hat'],b['z_hat'],rtol=1e-5,atol=1e-6))

    def test_forward_read_pending_and_eval_never_update_prior(self):
        t=self.torch;layout=protocol.LayoutSpec('hybrid_f16_a96');prior=protocol.make_power_prior(layout)
        before={k:v.clone() for k,v in prior.state_dict().items()}
        _,z,_=protocol.normalize_and_map(self.raw(),layout)
        stats=prior.pending(z,7.5,1)
        prior.read(0);prior.read(7.5);prior.read(25)
        self.assertTrue(all(t.equal(before[k],v) for k,v in prior.state_dict().items()))
        self.assertFalse(stats['power_mean'].requires_grad)
        prior.eval()
        with self.assertRaises(ValueError):prior.commit(stats)
        self.assertTrue(all(t.equal(before[k],v) for k,v in prior.state_dict().items()))

    def test_weighted_ema_shrink_and_exact_update_count(self):
        t=self.torch;layout=protocol.LayoutSpec('hybrid_f16_a96');prior=protocol.make_power_prior(layout)
        _,z,_=protocol.normalize_and_map(self.raw(),layout)
        old=prior.pi_nodes.clone();stats=prior.pending(z,7.5,1)
        target=.95*stats['power_mean']+.05*prior.uniform
        expected=.995*old[1]+.005*target
        expected=expected*(64/expected.sum(-1,keepdim=True))
        prior.commit(stats)
        self.assertTrue(t.equal(prior.pi_nodes[1],expected))
        self.assertTrue(t.equal(prior.pi_nodes[2],expected))
        self.assertTrue(t.equal(prior.pi_nodes[[0,3,4,5]],old[[0,3,4,5]]))
        self.assertEqual(prior.update_count.item(),1)
        self.assertEqual(prior.node_update_mass.tolist(),[0,.5,.5,0,0,0])
        prior.validate_state(expected_step=1)
        with self.assertRaises(ValueError):prior.commit(stats)
        with self.assertRaises(ValueError):prior.validate_state(expected_step=0)

    def test_reject_raw_wrong_user_energy_stale_or_other_layout_statistics(self):
        t=self.torch;layout=protocol.LayoutSpec('hybrid_f16_a96');prior=protocol.make_power_prior(layout)
        _,z,_=protocol.normalize_and_map(self.raw(),layout)
        clean=prior.pending(z,10,1)
        for kind in ('raw','other_layout','next_step','read_count','negative'):
            bad=deepcopy(clean)
            if kind=='raw':bad['power_mean']*=2
            elif kind=='other_layout':bad['layout']=protocol.LayoutSpec('fdma_control').to_dict()
            elif kind=='next_step':bad['step']=2
            elif kind=='read_count':bad['read_count']=1
            else:bad['power_mean'][0,0]=-1
            with self.subTest(kind=kind),self.assertRaises(ValueError):prior.commit(bad)
        self.assertEqual(prior.update_count.item(),0)

    def test_prior_strict_pause_resume_retains_nodes_counts_and_next_update(self):
        t=self.torch;layout=protocol.LayoutSpec('hybrid_f32_a64');a=protocol.make_power_prior(layout)
        _,z,_=protocol.normalize_and_map(self.raw(),layout)
        a.commit(a.pending(z,0,1))
        b=protocol.make_power_prior(layout);b.load_state_dict(a.state_dict(),strict=True)
        b.validate_state(1)
        a.commit(a.pending(z,25,2));b.commit(b.pending(z,25,2))
        self.assertTrue(all(t.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        missing=dict(a.state_dict());missing.pop('update_count')
        with self.assertRaises(RuntimeError):b.load_state_dict(missing,strict=True)
        b.update_count.fill_(3)
        with self.assertRaises(ValueError):b.validate_state()

    def test_actual_anchor_function_depends_on_exclusive_inputs_and_has_gradients(self):
        t=self.torch;layout=protocol.LayoutSpec('hybrid_f16_a96')
        mask=protocol.tensor_geometry(layout)['exclusive_mask']
        g=t.Generator().manual_seed(91)
        z=t.view_as_complex(t.randn(1,4,256,2,generator=g)).requires_grad_()
        error=t.rand(1,4,256,generator=g,requires_grad=True)
        coords=t.randn(4,256,12,generator=g)
        first=t.nn.Linear(9,128);last=t.nn.Linear(128,128)
        t.nn.init.normal_(first.weight,std=.02);t.nn.init.normal_(last.weight,std=.001)
        t.nn.init.zeros_(first.bias);t.nn.init.zeros_(last.bias)
        output=protocol.anchor_context(z,error,coords,mask,first,last)
        changed=z.detach().clone();changed[:,0,list(layout.exclusive_slots)]+=.3+.1j
        other=protocol.anchor_context(changed,error,coords,mask,first,last)
        self.assertGreater((other[:,0]-output[:,0]).abs().max().item(),0)
        self.assertTrue(t.equal(other[:,1:],output[:,1:]))
        changed=z.detach().clone();changed[:,:,list(layout.shared_slots)]+=100+100j
        self.assertTrue(t.equal(protocol.anchor_context(changed,error,coords,mask,first,last),output))
        output.square().sum().backward()
        self.assertTrue(t.isfinite(z.grad).all() and t.isfinite(error.grad).all())
        self.assertGreater(z.grad[:,mask].abs().sum().item(),0)
        self.assertEqual(z.grad[:,~mask].abs().sum().item(),0)
        self.assertGreater(error.grad[:,mask].abs().sum().item(),0)
        for parameter in list(first.parameters())+list(last.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(t.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(),0)


class ExplicitServerModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('GROUPED_HYBRID_MODEL_TEST_DEVICE')!='cuda':
            raise unittest.SkipTest('Full model migration/CUDA tests require explicit server-only opt-in; no laptop large models')
        try:
            if importlib.metadata.version('sionna')!='2.0.1':raise unittest.SkipTest('Actual Sionna2.0.1 required')
            import torch
        except (ImportError,importlib.metadata.PackageNotFoundError) as error:
            raise unittest.SkipTest('Actual dependencies unavailable; not substituted') from error
        if not torch.cuda.is_available():raise unittest.SkipTest('Actual CUDA unavailable')

    def test_strict_full_source_migration_only_initializes_anchor_and_priors(self):
        import torch
        from configs.grouped_codec_fdma_train_config import fdma_config
        from configs.grouped_hybrid_anchor_train_config import anchor_config
        from models.grouped_codec_fdma import GroupedCodecFDMAPrecoder
        from models.grouped_hybrid_anchor import GroupedHybridAnchorPrecoder,migrate_legacy_fdma_state_dict
        source=GroupedCodecFDMAPrecoder(fdma_config(20261030)).state_dict()
        model=GroupedHybridAnchorPrecoder(anchor_config('hybrid_f16_a96',20261030))
        report=migrate_legacy_fdma_state_dict(model,source)
        self.assertTrue(report['strict_loaded'])
        self.assertEqual(set(report['copied_keys']),set(source))
        self.assertTrue(all(torch.equal(value,model.state_dict()[key]) for key,value in source.items()))
        self.assertEqual(sum(p.numel() for n,p in model.named_parameters() if n.startswith('decoder.anchor.')),17792)
        bad=dict(source);bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):migrate_legacy_fdma_state_dict(model,bad)
        with self.assertRaises(RuntimeError):model.load_state_dict(source,strict=True)


if __name__=='__main__':unittest.main()
