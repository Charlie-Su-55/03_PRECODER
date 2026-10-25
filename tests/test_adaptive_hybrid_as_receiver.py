"""The current ASReceiver batch's focused checks, not the old ASDirect suite.

Only synthetic inputs. No checkpoint, CDL sampling or optimizer. CUDA requested
by the controller is mandatory; laptop dependency absence is an explicit skip.
"""
import ast
import copy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / "models/adaptive_hybrid.py"
RECEIVER = ROOT / "models/adaptive_hybrid_as_receiver.py"
ATOL = RTOL = 1e-6


def function(path, cls, name):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    container = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    node = copy.deepcopy(next(node for node in container.body if isinstance(node, ast.FunctionDef) and node.name == name))
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body.pop(0)
    return node


class SourceBoundaryTests(unittest.TestCase):
    def test_full_source_unchanged_and_only_stage1_insertion(self):
        self.assertEqual(hashlib.sha256(ORIGINAL.read_text(encoding="utf-8").encode()).hexdigest(),
                         "67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb")
        old = function(ORIGINAL, "AdaptiveHybridPrecoder", "forward")
        new = function(RECEIVER, "AdaptiveHybridPrecoderASReceiver", "forward")
        additions = [node for node in new.body if isinstance(node, ast.If) and ast.unparse(node.test) == "self.receiver_enabled and d_a > 0"]
        self.assertEqual(len(additions), 1)
        calls = [node for node in ast.walk(additions[0]) if isinstance(node, ast.Call)]
        self.assertEqual(len(calls), 1)
        self.assertEqual([ast.unparse(arg) for arg in calls[0].args],
                         ["W_feat", "Y_ac", "H_ul_est", "snr_db", "d_f", "d_a"])
        new.body.remove(additions[0])
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_receiver_information_boundary(self):
        source = ast.parse(RECEIVER.read_text(encoding="utf-8-sig"))
        recover = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "regularized_recovery")
        self.assertEqual([arg.arg for arg in recover.args.args],
                         ["Y_ac", "H_ul_est", "snr_db", "d_f", "d_a", "lambda_floor", "solve_dtype"])
        names = {node.id for node in ast.walk(recover) if isinstance(node, ast.Name)}
        self.assertFalse(names & {"H_dl", "Z_a", "Z_f", "pwr_batch", "running_pwr", "H_ul"})
        self.assertNotIn("register_forward_hook", ast.unparse(source))


class NumericalReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("AS_RECEIVER_TEST_DEVICE", "cpu")
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "2.0.1":
            reason = f"Genuine Sionna2.0.1 missing (found {version}); no model checks executed"
            if requested == "cuda":
                raise RuntimeError(reason)
            raise unittest.SkipTest(reason)
        import torch
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Controller requested CUDA checks; CPU fallback forbidden")
        from models.adaptive_hybrid import AdaptiveHybridPrecoder
        from models.adaptive_hybrid_as_receiver import (AdaptiveHybridPrecoderASReceiver, ASReceiverV1,
                                                       regularized_recovery, migrate_original_state_dict)
        cls.torch, cls.device = torch, torch.device(requested)
        cls.Original, cls.Model, cls.Branch = AdaptiveHybridPrecoder, AdaptiveHybridPrecoderASReceiver, ASReceiverV1
        cls.recover, cls.migrate = staticmethod(regularized_recovery), staticmethod(migrate_original_state_dict)

    def setUp(self):
        torch = self.torch
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(20261029)
        if self.cuda_rng is not None:
            torch.cuda.manual_seed_all(20261029)
        self.cfg = SimpleNamespace(K_USERS=2, ANTENNAS=4, SUBCARRIERS=16, NUM_SUBBANDS=4,
            D_MODEL=256, NUM_HEADS=8, NUM_ENCODER_LAYERS=1, DROPOUT=0.1,
            NUM_UNFOLD_LAYERS=1, GNN_AGG_DIM=48, D_F_MAX_PER_USER=8,
            D_A_MAX_SHARED=16, TOTAL_POWER=1.0, GAMMA_GRID=[(8, 0), (4, 8), (0, 16)])

    def tearDown(self):
        self.torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            self.torch.cuda.set_rng_state_all(self.cuda_rng)

    def complex(self, shape):
        torch = self.torch
        return torch.complex(torch.randn(shape, device=self.device), torch.randn(shape, device=self.device)) / 2**0.5

    def assert_close(self, actual, expected):
        error = (actual-expected).abs()
        print(f"receiver check max_abs={float(error.max()):.9g}, max_rel={float((error/expected.abs().clamp_min(1e-12)).max()):.9g}, atol=rtol=1e-6")
        self.torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)

    def test_known_complex_linear_problem_and_physical_slice(self):
        torch = self.torch
        g = torch.tensor([[1+1j, .2-.3j], [.4+.1j, 2-1j], [1-.5j, -.2+.7j]], dtype=torch.complex64, device=self.device)
        z = torch.tensor([.3+.7j, -.8+.2j], dtype=torch.complex64, device=self.device)
        h = self.complex((1, 2, 3, 8))
        h[:, :, :, 4:8] = g.T[None, :, :, None]
        y = (g @ z)[None, :, None].expand(1, 3, 4).clone()
        observed = self.recover(y, h, 10., 2, 4)
        lam = max(float(y.abs().square().mean()) / 11., 1e-6)
        gc, yc = g.detach().cpu().tolist(), y[0, :, 0].detach().cpu().tolist()
        a = sum(row[0].conjugate()*row[0] for row in gc)+lam
        b = sum(row[0].conjugate()*row[1] for row in gc)
        c = sum(row[1].conjugate()*row[0] for row in gc)
        d = sum(row[1].conjugate()*row[1] for row in gc)+lam
        u = sum(row[0].conjugate()*v for row,v in zip(gc,yc))
        v = sum(row[1].conjugate()*value for row,value in zip(gc,yc))
        determinant = a*d-b*c
        expected = torch.tensor([(d*u-b*v)/determinant, (a*v-c*u)/determinant], dtype=torch.complex64, device=self.device)
        self.assert_close(observed["z_hat"], expected[None, None].expand(1, 4, 2))
        expected_r = torch.tensor([1-(lam*d/determinant).real, 1-(lam*a/determinant).real], device=self.device).clamp(0, 1)
        self.assert_close(observed["reliability"], expected_r[None, None].expand(1,4,2))
        h[:, :, :, :4] *= 100
        self.assert_close(self.recover(y,h,10.,2,4)["z_hat"], observed["z_hat"])

    def test_migration_zero_projection_three_paths_and_strict_restore(self):
        torch = self.torch
        old, new = self.Original(self.cfg).to(self.device), self.Model(self.cfg).to(self.device)
        self.migrate(new, old.state_dict())
        self.assertEqual(set(new.state_dict())-set(old.state_dict()), set(new.added_state_keys))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]))
        h_dl, h_ul = self.complex((1,2,4,16)), self.complex((1,2,4,16))
        # Training returns H_hat directly; restore RNG and both running buffers.
        for index, (df, da) in enumerate(self.cfg.GAMMA_GRID):
            old.train(); new.train()
            cpu = torch.get_rng_state(); cuda = torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None
            with torch.no_grad():
                _, h0 = old(h_dl,h_ul,10.,df,da,index,H_ul_est=h_ul)
                torch.set_rng_state(cpu)
                if cuda is not None: torch.cuda.set_rng_state_all(cuda)
                _, h1 = new(h_dl,h_ul,10.,df,da,index,H_ul_est=h_ul)
                self.assert_close(h1,h0)
                w0,w1=old.zf_solver(h0),new.zf_solver(h1)
                w0=w0/(w0.abs().square().sum((1,2),keepdim=True)+1e-9).sqrt()
                w1=w1/(w1.abs().square().sum((1,2),keepdim=True)+1e-9).sqrt()
                self.assert_close(w1,w0)
                self.assert_close(w1.abs().square().sum((1,2)),torch.ones((1,16),device=self.device))
        # A normal receiver restore must not fill missing new keys or re-zero.
        saved = new.state_dict()
        missing = dict(saved); missing.pop(new.added_state_keys[0])
        with self.assertRaises(RuntimeError): new.load_state_dict(missing,strict=True)
        bad = dict(old.state_dict()); bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError): self.migrate(new,bad)
        bad = dict(old.state_dict()); key='running_pwr'; bad[key]=bad[key].double()
        with self.assertRaises(TypeError): self.migrate(new,bad)

    def test_output_projection_then_upstream_gradients_and_fdma_bypass(self):
        torch = self.torch
        branch=self.Branch(self.cfg).to(self.device)
        query=self.complex((1,2,4,4,256)).real
        h=self.complex((1,2,4,16)); y=self.complex((1,4,16)); target=torch.randn_like(query)
        residual=branch(query,y,h,10.,0,16)
        self.assertEqual(int(torch.count_nonzero(residual)),0)
        (residual*target).sum().backward()
        grad=branch.output_proj.weight.grad
        self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.norm()),0)
        # Synthetic technical gradient probe only; no optimizer/checkpoint/training run.
        with torch.no_grad(): branch.output_proj.weight.add_(grad,alpha=-1e-5)
        branch.zero_grad(set_to_none=True)
        (branch(query,y,h,10.,0,16)*target).sum().backward()
        for name in ('query_proj.weight','token_proj.weight','attention.in_proj_weight','position_embedding'):
            gradient=dict(branch.named_parameters())[name].grad
            self.assertIsNotNone(gradient); self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()),0,name)
        self.assertTrue(torch.equal(branch(query,None,None,10.,8,0),torch.zeros_like(query)))

    def test_deployed_task_loss_reaches_receiver_projection(self):
        torch=self.torch
        from main_train import SumRateLoss
        model=self.Model(self.cfg).to(self.device).train()
        h_dl,h_ul=self.complex((1,2,4,16)),self.complex((1,2,4,16))
        _,h_hat=model(h_dl,h_ul,10.,0,16,2)
        w=model.zf_solver(h_hat)
        w=w*torch.sqrt(1.0/(w.abs().square().sum((1,2),keepdim=True)+1e-9))
        loss,_=SumRateLoss()(w,h_dl,10.0**(-25/10))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad=model.as_receiver.output_proj.weight.grad
        self.assertIsNotNone(grad); self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()),0)

    def test_prefix_order_and_nonpersistent_inference_switch(self):
        torch=self.torch
        branch=self.Branch(self.cfg).to(self.device)
        with torch.no_grad():
            branch.token_proj.weight.zero_(); branch.token_proj.bias.zero_(); branch.position_embedding.zero_()
            branch.token_proj.weight[:3,:]=torch.eye(3,device=self.device)
        z=torch.arange(16,device=self.device,dtype=torch.float32).reshape(1,8,2).to(torch.complex64)
        r=torch.full_like(z.real,.5)
        token=branch._codeword_tokens(z,r)
        self.assertEqual(tuple(token.shape),(1,2,4,2,128))
        for user in range(2):
            for sb in range(4):
                self.assert_close(token[0,user,sb,:,0],z[0,sb*2:(sb+1)*2,user].real)
        model=self.Model(self.cfg).to(self.device).eval()
        self.assertNotIn('receiver_enabled',model.state_dict())
        with torch.no_grad(): model.as_receiver.output_proj.bias.fill_(.2)
        h_dl,h_ul=self.complex((1,2,4,16)),self.complex((1,2,4,16))
        before=torch.get_rng_state(); cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None
        with torch.no_grad():
            enabled=model(h_dl,h_ul,10.,8,0,0)
            model.receiver_enabled=False
            torch.set_rng_state(before)
            if cuda is not None: torch.cuda.set_rng_state_all(cuda)
            disabled=model(h_dl,h_ul,10.,8,0,0)
        self.assert_close(enabled,disabled)


if __name__ == "__main__":
    unittest.main()
