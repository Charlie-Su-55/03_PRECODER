"""Small CPU components only: never build/run the complete ICC-aligned codec."""
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

DEPENDENCY_ERROR = None
if importlib.util.find_spec("torch") is None:
    DEPENDENCY_ERROR = "PyTorch absent: small numerical model components not executed"
else:
    try:
        import torch
        from models import twc_v1_k8_icc_aligned as model
    except ModuleNotFoundError as exc:
        DEPENDENCY_ERROR = str(exc)


@unittest.skipIf(DEPENDENCY_ERROR is not None, DEPENDENCY_ERROR or "")
class ModelComponentTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator(device="cpu").manual_seed(31)

    def test_window_pool_keeps_all_32_actual_window_positions(self):
        grid = torch.arange(64*8*2,dtype=torch.float32).reshape(1,64,8,2)
        pooled = model.ordered_window_pool(grid.reshape(1,512,2))
        expected = torch.stack([grid[:,r:r+4,c:c+4].mean((1,2))
            for r in range(0,64,4) for c in range(0,8,4)],1)
        self.assertTrue(torch.equal(pooled,expected))
        self.assertEqual(pooled.flatten(1).shape,(1,64))
        self.assertGreater(float((pooled[:,0]-pooled[:,-1]).abs().max()),0.)
        with self.assertRaises(ValueError):
            model.ordered_window_pool(torch.zeros(1,511,2))

    def test_public_geometry_retains_216_slots_and_physical_mapping(self):
        fdma = model.public_layout("fdma")
        hybrid = model.public_layout("hybrid_g4")
        f,fa,fe = model.support_geometry(fdma)
        h,ha,he = model.support_geometry(hybrid)
        self.assertEqual(tuple(f.shape),(8,216))
        self.assertTrue(torch.equal(fa.sum(-1),torch.full((8,),24)))
        self.assertTrue(torch.equal(ha.sum(-1),torch.full((8,),44)))
        self.assertTrue(torch.equal(he.sum(-1),torch.full((8,),4)))
        self.assertTrue(torch.equal(fe,fa))
        self.assertEqual(len(set(f[fa].tolist())),192)
        self.assertEqual(len(set(h[he].tolist())),32)
        self.assertEqual(len(set(h[ha&~he].tolist())),160)
        self.assertTrue(torch.equal(h[0,4:44],h[1,4:44]))
        self.assertFalse(torch.equal(h[0,4:44],h[2,4:44]))

    def _ce_inputs(self):
        ce = model.K8ICCAlignedPilotUNet()
        pilots = torch.zeros(1,8,256,dtype=torch.complex64)
        pilots[:,ce.pilot_mask] = 2.**.5
        observed = torch.randn(1,32,256,dtype=torch.complex64,generator=self.generator)
        initial = torch.randn(1,8,32,256,dtype=torch.complex64,generator=self.generator)
        return ce,pilots,observed,initial

    def test_new_ce_pilot_features_ignore_all_data_values_and_broadcast_gamma(self):
        ce,pilots,Y,initial = self._ce_inputs()
        reference = ce.features(Y,pilots,initial,gamma_db=torch.tensor([5.]))
        poisoned = Y.clone()
        poisoned[:,:,~ce.pilot_union] = complex(float("nan"),float("inf"))
        result = ce.features(poisoned,pilots,initial,gamma_db=5.)
        self.assertEqual(tuple(reference.shape),(8,133,256))
        self.assertTrue(torch.equal(reference,result))
        self.assertTrue(torch.allclose(result[:,-1],torch.full_like(result[:,-1],-.6),rtol=0,atol=1e-7))
        altered = Y.clone()
        altered[:,:,ce.pilot_union] += 1
        self.assertFalse(torch.equal(reference,ce.features(altered,pilots,initial,gamma_db=5.)))
        with self.assertRaises(ValueError):
            ce.features(Y,pilots/2,initial,gamma_db=5.)
        with self.assertRaises(ValueError):
            ce.features(Y,pilots,initial,gamma_db=26.)

    def test_ce_feature_gradient_cannot_read_data_RE(self):
        ce,pilots,Y,initial = self._ce_inputs()
        Y.requires_grad_(True)
        features = ce.features(Y,pilots,initial,gamma_db=10.)
        features[:,:64].square().mean().backward()
        self.assertTrue(torch.isfinite(Y.grad).all())
        self.assertTrue(torch.equal(Y.grad[:,:,~ce.pilot_union],torch.zeros_like(Y.grad[:,:,~ce.pilot_union])))
        self.assertGreater(float(Y.grad[:,:,ce.pilot_union].abs().sum()),0.)

    def test_small_joint_attention_has_memory_and_all_query_gradient_paths(self):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(11)
            layer = model._ICCJointLayer(width=8,heads=2,query_chunk=2,activation_checkpointing=False)
        query = torch.randn(1,4,8,generator=self.generator,requires_grad=True)
        memory = torch.randn(1,3,8,generator=self.generator,requires_grad=True)
        active = torch.ones(1,4,dtype=torch.bool)
        output = layer(query,memory,active)
        # One user/query's scored output depends on other queries and memory.
        output[:,0].square().sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all() & torch.isfinite(memory.grad).all())
        self.assertGreater(float(query.grad[:,1:].abs().sum()),0.)
        self.assertGreater(float(memory.grad.abs().sum()),0.)
        changed = layer(query.detach(),memory.detach()+torch.arange(8).float(),active)
        self.assertGreater(float((output.detach()-changed).abs().max()),0.)

    def test_tiny_joint_copy_strict_loading_and_independent_storage(self):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(91)
            a = model._ICCJointLayer(width=8,heads=2,activation_checkpointing=False)
        b = deepcopy(a)
        b.load_state_dict(a.state_dict(),strict=True)
        for (ak,av),(bk,bv) in zip(a.state_dict().items(),b.state_dict().items()):
            self.assertEqual(ak,bk)
            self.assertTrue(torch.equal(av,bv))
            self.assertNotEqual(av.data_ptr(),bv.data_ptr())
        bad = deepcopy(a.state_dict())
        bad.pop(next(iter(bad)))
        with self.assertRaises(RuntimeError):
            b.load_state_dict(bad,strict=True)

    def test_receiver_callable_has_no_truth_or_actual_frame_energy_input(self):
        parameters = set(inspect.signature(model.K8ICCAlignedJointDecoder.forward).parameters)
        self.assertTrue({"Y_rx","H_ul_est","soft","reliability","pilots","gamma_db"} <= parameters)
        self.assertTrue(parameters.isdisjoint({"H_dl","H_ul","tx","true_codeword","key_energy","shared_energy","noise_free_Y"}))


if __name__ == "__main__":
    unittest.main()
