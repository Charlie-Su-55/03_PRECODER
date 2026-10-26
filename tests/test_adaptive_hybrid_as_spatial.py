"""Focused ASSpatial checks: static locally, genuine-runtime synthetic CUDA on server.

No checkpoint files, channel generation or training loop. Missing genuine
Sionna 2.0.1 skips numerical checks locally; explicitly requested CUDA must fail
rather than silently skip or fall back. No dependency stubs or backend changes.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / "models/adaptive_hybrid_as_spatial.py"
FULL_PATH = ROOT / "models/adaptive_hybrid.py"
FULL_LF_SHA256 = "67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb"


def module_tree():
    return ast.parse(MODEL_PATH.read_text(encoding="utf-8-sig"))


def class_node(name):
    return next(node for node in module_tree().body
                if isinstance(node, ast.ClassDef) and node.name == name)


def method_node(cls, name):
    return next(node for node in class_node(cls).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


class StaticSpatialTests(unittest.TestCase):
    def test_protected_full_source_unchanged(self):
        normalized = FULL_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(normalized).hexdigest(), FULL_LF_SHA256)

    def test_independent_model_no_old_bs_or_hooks(self):
        cls = class_node("AdaptiveHybridPrecoderASSpatial")
        self.assertEqual([ast.unparse(base) for base in cls.bases], ["nn.Module"])
        text = ast.unparse(module_tree())
        for forbidden in ("self.w_init", "self.layers", "self.h_estimator",
                          "self.residual_head", "self.residual_scale",
                          "self.cross_subband_attn", "register_forward_hook"):
            self.assertNotIn(forbidden, text)
        calls = [ast.unparse(node.func) for node in ast.walk(cls) if isinstance(node, ast.Call)]
        self.assertNotIn("AdaptiveHybridPrecoder", calls)
        assignments = {target.id: ast.unparse(node.value)
                       for node in cls.body if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        for method in ("add_noise", "transmit", "zf_solver"):
            self.assertEqual(assignments[method], "AdaptiveHybridPrecoder." + method)

    def test_explicit_spatial_mapping_and_existing_four_blocks(self):
        init = method_node("ASSpatialDecoder", "__init__")
        lists = [node for node in ast.walk(init) if isinstance(node, ast.ListComp)
                 and isinstance(node.elt, ast.Call)]
        blocks = next(node for node in lists if ast.unparse(node.elt.func) == "SwinTransformerBlock")
        self.assertEqual([ast.literal_eval(arg) for arg in blocks.elt.args], [128, 4, 4, 4.0])
        self.assertEqual(ast.unparse(blocks.generators[0].iter), "range(4)")
        forward = method_node("ASSpatialDecoder", "forward")
        self.assertEqual([arg.arg for arg in forward.args.args],
                         ["self", "z_hat", "reliability", "snr_db"])
        names = {node.id for node in ast.walk(forward) if isinstance(node, ast.Name)}
        self.assertFalse(names & {"H_dl", "H_ul", "Z_a", "true_z_a", "pwr_batch"})
        self.assertIn("self.spatial_expansion(features)", ast.unparse(forward))
        self.assertIn("inverse_angular_delay_transform(H_ad, self.Nsc)",
                      ast.unparse(method_node("AdaptiveHybridPrecoderASSpatial", "_forward")))

    def test_diagnostic_truth_is_explicit_detached_after_output(self):
        forward = method_node("AdaptiveHybridPrecoderASSpatial", "_forward")
        diagnostic = next(node for node in forward.body if isinstance(node, ast.If)
                          and ast.unparse(node.test) == "with_diagnostics")
        index = forward.body.index(diagnostic)
        preceding = ast.unparse(ast.Module(body=forward.body[:index], type_ignores=[]))
        self.assertIn("W = W_zf *", preceding)
        payload = ast.unparse(diagnostic)
        self.assertIn("'true_z_a': Z_a.detach()", payload)
        self.assertNotIn("self.diagnostic", ast.unparse(module_tree()))
        repr_method = method_node("AdaptiveHybridPrecoderASSpatial", "forward_with_representation")
        self.assertNotIn("with_diagnostics=True", ast.unparse(repr_method))


class NumericalSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("AS_SPATIAL_TEST_DEVICE", "cpu")
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError("AS_SPATIAL_TEST_DEVICE must be cpu or a CUDA device.")
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "2.0.1":
            reason = f"Genuine Sionna 2.0.1 unavailable (found {version}); no model execution"
            if requested.startswith("cuda"):
                raise RuntimeError(reason)
            raise unittest.SkipTest(reason)
        import torch
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly required; CPU fallback forbidden")
        from models.adaptive_hybrid import AdaptiveHybridPrecoder
        from models.adaptive_hybrid_as_spatial import (
            AdaptiveHybridPrecoderASSpatial, ASSpatialDecoder, initialize_ue_from_full,
        )
        from models.baseline_common import angular_delay_transform, inverse_angular_delay_transform
        from models.swin_precoder import SwinTransformerBlock
        cls.torch, cls.device = torch, torch.device(requested)
        cls.Model, cls.Decoder, cls.Full = AdaptiveHybridPrecoderASSpatial, ASSpatialDecoder, AdaptiveHybridPrecoder
        cls.initialize = staticmethod(initialize_ue_from_full)
        cls.forward_transform = staticmethod(angular_delay_transform)
        cls.inverse_transform = staticmethod(inverse_angular_delay_transform)
        cls.SwinBlock = SwinTransformerBlock
        cls.original_threads = torch.get_num_threads()
        if requested == "cpu":
            torch.set_num_threads(1)
        cls.class_rng = torch.get_rng_state().clone()
        torch.manual_seed(125)
        cls.full = cls.Full(cls.config("joint"))
        with torch.no_grad():
            cls.full.running_pwr.fill_(2.25)
        cls.source = cls.full.state_dict()

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_rng_state(cls.class_rng)
        cls.torch.set_num_threads(cls.original_threads)
        del cls.source, cls.full

    @staticmethod
    def config(arm="joint"):
        return SimpleNamespace(K_USERS=4, ANTENNAS=32, SUBCARRIERS=256,
            NUM_SUBBANDS=4, D_MODEL=256, NUM_HEADS=8, NUM_ENCODER_LAYERS=4,
            DROPOUT=0.1, NUM_UNFOLD_LAYERS=5, GNN_AGG_DIM=48,
            D_F_MAX_PER_USER=64, D_A_MAX_SHARED=256, TOTAL_POWER=1.0,
            GAMMA_GRID=[(0, 256)], arm=arm)

    def setUp(self):
        torch = self.torch
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        torch.manual_seed(412)
        self.model = self.Model(self.config()).to(self.device)
        self.initialize(self.model, self.source)
        generator = torch.Generator(device="cpu").manual_seed(907)
        shape = (1, 4, 32, 256)
        def channel():
            return torch.complex(torch.randn(shape, generator=generator),
                                 torch.randn(shape, generator=generator)).to(self.device) / 2**0.5
        self.h_dl, self.h_ul = channel(), channel()
        self.h_est = (0.91 + 0.07j) * self.h_ul + 0.03 * channel()
        self.args = (self.h_dl, self.h_ul, 10.0, 0, 256, 0)

    def tearDown(self):
        self.torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            self.torch.cuda.set_rng_state_all(self.cuda_rng)
        del self.model, self.h_dl, self.h_ul, self.h_est

    def rng(self):
        return (self.torch.get_rng_state().clone(),
                self.torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])

    def restore(self, state):
        self.torch.set_rng_state(state[0])
        if state[1]:
            self.torch.cuda.set_rng_state_all(state[1])

    def equal(self, actual, expected):
        error = float((actual - expected).abs().max()) if actual.numel() else 0.0
        self.assertTrue(self.torch.equal(actual, expected),
                        f"not exact; max_abs={error}, device={self.device}")

    def finite_power(self, W, H_hat):
        self.assertEqual(tuple(W.shape), (1, 32, 4, 256))
        self.assertEqual(tuple(H_hat.shape), (1, 4, 32, 256))
        self.assertTrue(bool(self.torch.isfinite(W).all()))
        self.assertTrue(bool(self.torch.isfinite(H_hat).all()))
        self.torch.testing.assert_close(W.abs().square().sum((1, 2)),
                                       self.torch.ones((1, 256), device=self.device),
                                       rtol=0, atol=2e-6)

    def test_strict_ue_only_initialization_and_groups(self):
        torch = self.torch
        fresh = self.Model(self.config()).to(self.device)
        before = {name: value.detach().clone() for name, value in fresh.decoder.state_dict().items()}
        result = fresh.initialize_ue_from_full(self.source)
        self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))
        for name in fresh.ue_state_keys:
            self.equal(fresh.state_dict()[name].cpu(), self.source[name])
        for name, value in fresh.decoder.state_dict().items():
            self.equal(value, before[name])
        ue, bs = {id(p) for p in fresh.ue_parameters()}, {id(p) for p in fresh.bs_parameters()}
        self.assertFalse(ue & bs)
        self.assertEqual(ue | bs, {id(p) for p in fresh.parameters()})
        self.assertTrue(all(isinstance(block, self.SwinBlock) for block in fresh.decoder.blocks))
        self.assertEqual(len(fresh.decoder.blocks), 4)
        self.assertEqual(tuple(fresh.decoder.spatial_expansion.weight.shape), (8192, 769))
        self.assertIs(type(fresh).transmit, self.Full.transmit)
        self.assertIs(type(fresh).add_noise, self.Full.add_noise)
        self.assertIs(type(fresh).zf_solver, self.Full.zf_solver)

    def test_invalid_migrations_do_not_change_state(self):
        torch = self.torch
        key = next(name for name in self.model.ue_state_keys if name.endswith(".weight"))
        before = self.model.state_dict()[key].clone()
        cases = []
        missing = dict(self.source)
        del missing["running_pwr"]
        cases.append(missing)
        extra = dict(self.source)
        extra["encoder.unrecognized"] = torch.ones(1)
        cases.append(extra)
        wrong_shape = dict(self.source)
        wrong_shape[key] = self.source[key].flatten()[:1]
        cases.append(wrong_shape)
        wrong_dtype = dict(self.source)
        wrong_dtype[key] = self.source[key].double()
        cases.append(wrong_dtype)
        wrong_value = dict(self.source)
        wrong_value[key] = None
        cases.append(wrong_value)
        for state in cases:
            with self.subTest(keys=len(state)):
                with self.assertRaises((ValueError, TypeError)):
                    self.model.initialize_ue_from_full(state)
                self.equal(self.model.state_dict()[key], before)

    def test_frozen_ue_modes_power_and_task_gradients(self):
        torch = self.torch
        self.model.set_ue_frozen(True).train()
        self.assertFalse(self.model.encoder.training)
        self.assertFalse(self.model.cond_embed.training)
        self.assertTrue(self.model.decoder.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.ue_parameters()))
        power_before = self.model.running_pwr.clone()
        W, H_hat = self.model(*self.args, H_ul_est=self.h_est)
        self.finite_power(W, H_hat)
        self.task_loss(W).backward()
        self.equal(self.model.running_pwr, power_before)
        self.assertTrue(all(p.grad is None for p in self.model.ue_parameters()))
        for name in ("spatial_expansion.weight", "blocks.0.attn.qkv.weight",
                     "blocks.3.mlp.0.weight", "final_proj.weight"):
            grad = dict(self.model.decoder.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)
        self.model.eval().train()
        self.assertFalse(self.model.encoder.training)

    def task_loss(self, W):
        torch = self.torch
        effective = torch.einsum("bknt,bnlt->bklt", self.h_dl.conj(), W)
        power = effective.abs().square()
        desired = power.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
        interference = power.sum(dim=2) - desired
        return -torch.log2(1.0 + desired / (interference + 10**(-25/10))).sum(dim=1).mean()

    def test_joint_task_gradients_and_original_power_update(self):
        torch = self.torch
        self.model.train()
        before = self.model.running_pwr.clone()
        initial = self.rng()
        with torch.no_grad():
            cond = self.model.cond_embed(0, 256, 10.0, self.device)
            zf, za = self.model.encoder(self.h_dl.reshape(4, 32, 256), 0, 64, cond)
            expected_batch = (zf.abs().square().sum() + za.abs().square().sum()) / (4 * 256) + 1e-9
        self.restore(initial)
        W, H_hat = self.model(*self.args, H_ul_est=self.h_est)
        self.finite_power(W, H_hat)
        self.equal(self.model.running_pwr, 0.98 * before + 0.02 * expected_batch)
        self.task_loss(W).backward()
        for parameter in (self.model.encoder.fc_aircomp.weight,
                          self.model.cond_embed.net[0].weight,
                          self.model.decoder.spatial_expansion.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        self.assertIsNone(self.model.encoder.fc_fdma.weight.grad)

    def test_plain_representation_and_diagnostics_are_identical(self):
        torch = self.torch
        self.model.eval()
        before = {name: value.clone() for name, value in self.model.named_buffers()}
        initial = self.rng()
        with torch.no_grad():
            plain = self.model(*self.args, H_ul_est=self.h_est)
            after_plain = self.rng()
            self.restore(initial)
            W, H_hat = self.model.forward_with_representation(*self.args, H_ul_est=self.h_est)
            self.restore(initial)
            output = self.model.forward_with_diagnostics(*self.args, H_ul_est=self.h_est)
        self.equal(plain, W)
        self.equal(W, output["precoder"])
        self.equal(H_hat, output["channel_surrogate"])
        after_diagnostic = self.rng()
        for actual, expected in zip((after_diagnostic[0], *after_diagnostic[1]),
                                    (after_plain[0], *after_plain[1])):
            self.equal(actual, expected)
        for name, value in self.model.named_buffers():
            self.equal(value, before[name])
        self.assertFalse(self.model.training)
        diagnostic = output["diagnostics"]
        self.assertEqual(tuple(diagnostic["z_hat"].shape), (1, 256, 4))
        self.assertEqual(tuple(diagnostic["true_z_a"].shape), (1, 4, 256))
        for value in diagnostic.values():
            if torch.is_tensor(value):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
        self.assertFalse(any("diagnostic" in name for name in vars(self.model)))

    def test_frozen_physical_symbols_match_full_ue(self):
        torch = self.torch
        self.model.set_ue_frozen(True).train()
        # The source Full UE modules are copied as a reference, without executing
        # any old BS decoder. Frozen mode must use archived inference power.
        encoder = deepcopy(self.full.encoder).to(self.device).eval()
        condition = deepcopy(self.full.cond_embed).to(self.device).eval()
        initial = self.rng()
        with torch.no_grad():
            cond = condition(0, 256, 10.0, self.device)
            _, symbols = encoder(self.h_dl.reshape(4, 32, 256), 0, 64, cond)
            expected = (symbols / torch.sqrt(self.model.running_pwr[0] + 1e-9)).reshape(1, 4, 256)
            self.restore(initial)
            output = self.model.forward_with_diagnostics(*self.args, H_ul_est=self.h_est)
        self.equal(output["diagnostics"]["true_z_a"], expected)
        self.equal(self.model.running_pwr.cpu(), self.source["running_pwr"])

    def test_inverse_angular_delay_axes_and_physical_estimate_boundary(self):
        torch = self.torch
        angular_delay = torch.randn((1, 4, 2, 32, 32), device=self.device)
        reconstructed = self.inverse_transform(angular_delay, 256)
        restored = self.forward_transform(reconstructed, 32)
        torch.testing.assert_close(restored, angular_delay, rtol=2e-5, atol=2e-6)
        self.model.eval()
        initial = self.rng()
        with torch.no_grad():
            omitted = self.model.forward_with_diagnostics(*self.args)
            self.restore(initial)
            explicit = self.model.forward_with_diagnostics(*self.args, H_ul_est=self.h_ul)
            self.restore(initial)
            estimated = self.model.forward_with_diagnostics(*self.args, H_ul_est=self.h_est)
        self.equal(omitted["precoder"], explicit["precoder"])
        self.equal(omitted["channel_surrogate"], explicit["channel_surrogate"])
        self.equal(explicit["diagnostics"]["true_z_a"], estimated["diagnostics"]["true_z_a"])
        self.assertFalse(torch.equal(explicit["diagnostics"]["z_hat"], estimated["diagnostics"]["z_hat"]))

    def test_strict_resume_and_fixed_configuration(self):
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            self.model.decoder.final_proj.bias.add_(0.15)
        fresh = self.Model(self.config()).to(self.device).eval()
        fresh.load_state_dict(self.model.state_dict(), strict=True)
        for name, value in fresh.state_dict().items():
            self.equal(value, self.model.state_dict()[name])
        bad = dict(self.model.state_dict())
        bad.pop("decoder.final_proj.bias")
        with self.assertRaises(RuntimeError):
            fresh.load_state_dict(bad, strict=True)
        with self.assertRaises(ValueError):
            self.model(self.h_dl, self.h_ul, 10.0, 32, 128, 0)
        bad_cfg = self.config()
        bad_cfg.GAMMA_GRID = [(0, 256), (64, 0)]
        with self.assertRaises(ValueError):
            self.Model(bad_cfg)
        metadata = self.model.architecture_metadata
        json.dumps(metadata, allow_nan=False)
        self.assertEqual(metadata["decoder"]["spatial_expansion"], [769, 8192])
        self.assertFalse(metadata["decoder"]["shifted_windows"])
        self.assertFalse(metadata["old_stage1_or_refinement_or_head"])


if __name__ == "__main__":
    unittest.main()
