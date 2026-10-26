"""Static checks plus explicitly opted-in, genuine-runtime small tensor checks.

No complete Joint+Swin model is constructed here. Its real B=48 integration and
checkpoint checks belong to the batch controller's CUDA preflight. Numerical
unit tests require JOINT_SWIN_TEST_DEVICE=cuda (or cpu explicitly); no dependency
stubs, backend changes, real checkpoint loads, channel generation or training loop.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / "models/adaptive_hybrid_joint_swin.py"
FULL_SHA = "67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb"
SWIN_AST_SHA = {
    "SwinTransformerBlock": "cd430d51b32d4e982965c0c09abaeb07e248f5049bec6c318cdaed793791f4f5",
    "WindowAttention": "bbe061928a4b0cc48052affa3f151a71f31720c48f6930d1f1e0d7f67baf5b37",
    "SwinCFNetDecoder": "d44c0576ce4280ab2e26de40e447d605ad81c2166a4417d9105fe5cb02b9b4bb",
}


def tree():
    return ast.parse(MODEL_PATH.read_text(encoding="utf-8-sig"))


def cls(name):
    return next(node for node in tree().body
                if isinstance(node, ast.ClassDef) and node.name == name)


def method(class_name, name):
    return next(node for node in cls(class_name).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def constant(name):
    return next(ast.literal_eval(node.value) for node in tree().body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets))


class StaticJointSwinTests(unittest.TestCase):
    def test_original_full_source_unchanged(self):
        raw = (ROOT / "models/adaptive_hybrid.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), FULL_SHA)

    def test_existing_swin_ast_matches_audited_pinned_08(self):
        parsed = ast.parse((ROOT / "models/swin_precoder.py").read_text(encoding="utf-8-sig"))
        for name, expected in SWIN_AST_SHA.items():
            with self.subTest(name=name):
                node = next(node for node in parsed.body
                            if isinstance(node, ast.ClassDef) and node.name == name)
                normalized = ast.dump(node, include_attributes=False)
                self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(), expected)

    def test_independent_bs_reuses_only_explicit_original_operators(self):
        model = cls("AdaptiveHybridPrecoderJointSwin")
        self.assertEqual([ast.unparse(base) for base in model.bases], ["nn.Module"])
        calls = {ast.unparse(node.func) for node in ast.walk(model) if isinstance(node, ast.Call)}
        self.assertFalse(calls & {"AdaptiveHybridPrecoder", "AdaptiveHybridPrecoderASSpatial"})
        assignments = {target.id: ast.unparse(node.value)
                       for node in model.body if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        for name in ("add_noise", "transmit", "zf_solver"):
            self.assertEqual(assignments[name], "AdaptiveHybridPrecoder." + name)
        for name in ("ue_state_keys", "initialize_ue_from_full", "ue_parameters", "bs_parameters"):
            self.assertEqual(assignments[name], "AdaptiveHybridPrecoderASSpatial." + name)
        source = ast.unparse(tree())
        for forbidden in ("self.w_init", "self.h_estimator", "self.cross_subband_attn",
                          "self.refinement", "self.as_receiver", "register_forward_hook",
                          "icc08", "sys.modules", "torch.backends"):
            self.assertNotIn(forbidden, source)

    def test_fixed_full_resolution_and_no_truncation(self):
        settings = constant("SETTINGS")
        self.assertEqual(settings["codeword_fc"], [512, 65536])
        self.assertEqual(settings["patch_grid"], [64, 8])
        self.assertEqual(settings["full_delay_taps"], 256)
        self.assertEqual(settings["swin_blocks"], 4)
        self.assertEqual(settings["spatial_cross_queries"], 2048)
        self.assertEqual(settings["spatial_cross_keys"], 1024)
        self.assertFalse(settings["shifted_windows"])
        init = ast.unparse(method("JointSwin256Decoder", "__init__"))
        for actual in ("SWIN_NC=256", "SWIN_CODEWORD_DIM=512", "SWIN_NUM_BLOCKS=4"):
            self.assertIn(actual, init)
        forward = ast.unparse(method("JointSwin256Decoder", "forward"))
        self.assertIn("torch.cat((corrected.real, corrected.imag), dim=-1)", forward)
        self.assertIn("x.reshape(batch, 4, 2, 256, 32)", forward)
        self.assertNotIn("F.pad", forward)

    def test_common_initialization_precedes_rng_isolated_joint_extras(self):
        init = method("JointSwin256Decoder", "__init__")
        joint = next(node for node in init.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "arm == 'joint_swin256'")
        common = ast.unparse(ast.Module(body=init.body[:init.body.index(joint)], type_ignores=[]))
        self.assertIn("self.codec = SwinCFNetDecoder", common)
        self.assertIn("self.spatial_position = nn.Linear(10, 128)", common)
        joint_source = ast.unparse(joint)
        self.assertIn("with torch.random.fork_rng(devices=[]):", joint_source)
        for name in ("observation_memory", "joint_layers", "residual_head", "spatial_attention"):
            self.assertIn("self." + name + " =", joint_source)
            self.assertNotIn("self." + name + " =", common)
        self.assertEqual(constant("COMMON_BS_PREFIXES"),
                         ("decoder.codec.", "decoder.spatial_position."))

    def test_joint_structure_bias_and_small_nonzero_initialization(self):
        settings = constant("SETTINGS")
        self.assertEqual((settings["width"], settings["heads"], settings["joint_layers"],
                          settings["joint_ffn_dim"]), (128, 4, 2, 256))
        self.assertEqual(settings["query_features"], 15)
        self.assertEqual(settings["residual_weight_std"], 0.001)
        self.assertEqual(settings["head_bias_initial_value"], 0.02)
        self.assertFalse(settings["residual_gate"])
        self.assertFalse(settings["receive_power_normalization"])
        layer = ast.unparse(method("JointLayer", "forward"))
        self.assertLess(layer.index("self.self_attention("), layer.index("self.cross_attention("))
        self.assertLess(layer.index("self.cross_attention("), layer.index("self.mlp("))
        init = ast.unparse(method("JointSwin256Decoder", "__init__"))
        self.assertIn("nn.init.normal_(self.residual_head.weight, std=0.001)", init)
        self.assertIn("nn.init.normal_(self.spatial_attention.out.weight, std=0.001)", init)
        self.assertNotIn("zeros_(self.residual_head.weight)", init)

    def test_chunked_attention_keeps_all_keys_and_checkpoints_each_chunk(self):
        forward = ast.unparse(method("ChunkedAttention", "forward"))
        self.assertIn("key, value = (self._split(self.k(memory)), self._split(self.v(memory)))",
                      forward)
        self.assertIn("range(0, query.shape[1], self.query_chunk)", forward)
        self.assertIn("query[:, start:start + self.query_chunk], key, value", forward)
        self.assertNotIn("memory[:,", forward)
        self.assertNotIn("key[:,", forward)
        chunk = ast.unparse(method("ChunkedAttention", "_chunk"))
        self.assertIn("F.scaled_dot_product_attention", chunk)
        self.assertIn("dropout_p=0.0", chunk)
        self.assertIn("self.q(query)", chunk)
        self.assertIn("self.out(attended)", chunk)
        runner = next(node for node in tree().body
                      if isinstance(node, ast.FunctionDef) and node.name == "_run")
        self.assertIn("use_reentrant=False", ast.unparse(runner))
        self.assertIn("query_chunk_size", constant("SETTINGS"))
        self.assertEqual(constant("SETTINGS")["query_chunk_size"], 64)

    def test_receiver_input_boundary_and_original_pure_as_path(self):
        names = [arg.arg for arg in method("JointSwin256Decoder", "forward").args.args]
        self.assertEqual(names, ["self", "z_hat", "reliability", "Y_ac", "H_ul_est", "snr_db"])
        source = ast.unparse(cls("ObservationMemory"))
        for forbidden in ("pilot", "N0", "noiseless", "H_dl", "true_z", "pwr_batch"):
            self.assertNotIn(forbidden, source)
        self.assertIn("torch.log1p(reliability)", source)
        forward = ast.unparse(method("AdaptiveHybridPrecoderJointSwin", "_forward"))
        self.assertEqual(forward.count("self.transmit("), 1)
        self.assertIn("regularized_recovery(Y_ac, H_ul_est", forward)
        self.assertIn("inverse_angular_delay_transform(H_ad, self.Nsc)", forward)
        self.assertIn("self.zf_solver(H_hat)", forward)
        self.assertIn("if self.training:", forward)
        self.assertNotIn("set_ue_frozen", ast.unparse(cls("AdaptiveHybridPrecoderJointSwin")))

    def test_diagnostics_after_deployment_and_no_fake_execution_flags(self):
        forward = method("AdaptiveHybridPrecoderJointSwin", "_forward")
        diagnostic = next(node for node in forward.body if isinstance(node, ast.If)
                          and ast.unparse(node.test) == "with_diagnostics")
        earlier = ast.unparse(ast.Module(body=forward.body[:forward.body.index(diagnostic)],
                                        type_ignores=[]))
        self.assertIn("W = W_zf *", earlier)
        self.assertIn("'true_z_a': Z_a.detach()", ast.unparse(diagnostic))
        self.assertIn("'z_corrected': corrected.detach()", ast.unparse(diagnostic))
        decoder = ast.unparse(method("JointSwin256Decoder", "forward"))
        self.assertNotIn("'joint_attention_executed': True", decoder)
        self.assertNotIn("'spatial_cross_attention_executed': True", decoder)
        self.assertNotIn("self.diagnostic", ast.unparse(tree()))


class NumericalJointSwinTests(unittest.TestCase):
    """Small genuine-runtime tensors only; no 45M-parameter model constructor."""

    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("JOINT_SWIN_TEST_DEVICE")
        if requested is None:
            raise unittest.SkipTest("Numerical checks require explicit JOINT_SWIN_TEST_DEVICE; no local model execution")
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError("JOINT_SWIN_TEST_DEVICE must be cpu or a CUDA device")
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "2.0.1":
            raise RuntimeError(f"Explicit numerical checks require genuine Sionna 2.0.1, found {version}")
        import torch
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly required; CPU fallback forbidden")
        from models.adaptive_hybrid_joint_swin import ChunkedAttention, ObservationMemory, channel_coupling
        from models.baseline_common import angular_delay_transform, inverse_angular_delay_transform
        cls.torch, cls.device = torch, torch.device(requested)
        cls.Attention, cls.Memory = ChunkedAttention, ObservationMemory
        cls.coupling = staticmethod(channel_coupling)
        cls.forward_transform = staticmethod(angular_delay_transform)
        cls.inverse_transform = staticmethod(inverse_angular_delay_transform)

    def setUp(self):
        torch = self.torch
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
        torch.manual_seed(409)

    def tearDown(self):
        self.torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng:
            self.torch.cuda.set_rng_state_all(self.cuda_rng)

    def reference(self, module, query, memory, bias=None):
        torch = self.torch
        q, k, v = (module._split(module.q(query)), module._split(module.k(memory)),
                   module._split(module.v(memory)))
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                             dropout_p=0.0, is_causal=False)
        return module.out(y.transpose(1, 2).reshape(query.shape[0], query.shape[1], module.width))

    def compare_attention(self, kind):
        torch = self.torch
        actual_module = self.Attention(width=16, heads=4, query_chunk=3, symbols=4,
                                       activation_checkpoint=True).to(self.device).train()
        reference_module = deepcopy(actual_module)
        query = torch.randn(2, 8, 16, device=self.device, requires_grad=True)
        memory = torch.randn(2, 4 if kind == "observation" else 8, 16,
                             device=self.device, requires_grad=True)
        reference_query, reference_memory = query.detach().clone().requires_grad_(), memory.detach().clone().requires_grad_()
        coupling = torch.rand(2, 2, 2, device=self.device, requires_grad=True)
        reference_coupling = coupling.detach().clone().requires_grad_()
        overlap = torch.full((4,), 0.02, device=self.device, requires_grad=True)
        coupling_weight = torch.full((4,), 0.02, device=self.device, requires_grad=True)
        resource = torch.full((4,), 0.02, device=self.device, requires_grad=True)
        reference_weights = [value.detach().clone().requires_grad_()
                             for value in (overlap, coupling_weight, resource)]
        qindices = torch.arange(8, device=self.device)
        kindices = torch.arange(memory.shape[1], device=self.device)
        same = (qindices.remainder(4)[:, None] == kindices.remainder(4)[None]).float()
        bias = None
        if kind == "self":
            user_coupling = reference_coupling[:, qindices // 4][:, :, kindices // 4]
            bias = (same[None, None] * reference_weights[0][None, :, None, None]
                    + user_coupling[:, None] * reference_weights[1][None, :, None, None])
        elif kind == "observation":
            bias = same[None, None] * reference_weights[2][None, :, None, None]
        actual = actual_module(query, memory, coupling, overlap, coupling_weight, resource, kind)
        expected = self.reference(reference_module, reference_query, reference_memory, bias)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        weights = torch.linspace(-1, 1, actual.numel(), device=self.device).reshape_as(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for value, ref in ((query, reference_query), (memory, reference_memory),
                           *zip(actual_module.parameters(), reference_module.parameters())):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            torch.testing.assert_close(value.grad, ref.grad, rtol=3e-5, atol=4e-6)
        if kind == "self":
            for value, ref in ((coupling, reference_coupling),
                               (overlap, reference_weights[0]),
                               (coupling_weight, reference_weights[1])):
                self.assertIsNotNone(value.grad)
                torch.testing.assert_close(value.grad, ref.grad, rtol=3e-5, atol=4e-6)
        elif kind == "observation":
            torch.testing.assert_close(resource.grad, reference_weights[2].grad, rtol=3e-5, atol=4e-6)

    def test_all_key_chunked_attention_output_and_gradients(self):
        self.compare_attention("none")

    def test_self_resource_and_channel_bias_output_and_gradients(self):
        self.compare_attention("self")

    def test_observation_resource_bias_output_and_gradients(self):
        self.compare_attention("observation")

    def test_known_complex_channel_coupling(self):
        torch = self.torch
        channel = torch.zeros(1, 2, 2, 3, dtype=torch.complex64, device=self.device)
        channel[:, 0, 0, :] = torch.tensor([1j, 2, -3j], device=self.device)
        channel[:, 1, 1, :] = torch.tensor([2, -1j, 1], device=self.device)
        torch.testing.assert_close(self.coupling(channel), torch.eye(2, device=self.device)[None],
                                   rtol=0, atol=0)
        # The pinned 08 formula floors each norm before multiplication, not
        # their product; otherwise weak but well-conditioned channels differ.
        torch.testing.assert_close(self.coupling(channel * 1e-4),
                                   torch.eye(2, device=self.device)[None],
                                   rtol=2e-6, atol=2e-6)
        channel[:, 1] = channel[:, 0] * (0.5 + 0.75j)
        torch.testing.assert_close(self.coupling(channel), torch.ones(1, 2, 2, device=self.device),
                                   rtol=2e-6, atol=2e-6)

    def test_observation_memory_uses_received_values_and_estimated_channel(self):
        torch = self.torch
        memory = self.Memory().to(self.device).eval()
        received = torch.randn(1, 32, 256, dtype=torch.complex64, device=self.device)
        estimate = torch.randn(1, 4, 32, 256, dtype=torch.complex64, device=self.device)
        reliability = torch.rand(1, 256, 4, device=self.device)
        coordinates = torch.randn(256, 3, device=self.device)
        snr = torch.full((1, 1, 1), 0.4, device=self.device)
        with torch.no_grad():
            original = memory(received, estimate, reliability, coordinates, snr)
            changed_y = memory(received + (0.25 + 0.5j), estimate, reliability, coordinates, snr)
            changed_h = memory(received, estimate * (0.6 + 0.4j), reliability, coordinates, snr)
        self.assertEqual(tuple(original.shape), (1, 256, 128))
        self.assertTrue(bool(torch.isfinite(original).all()))
        self.assertFalse(torch.equal(original, changed_y))
        self.assertFalse(torch.equal(original, changed_h))

    def test_full_256_delay_transform_roundtrip(self):
        torch = self.torch
        angular_delay = torch.randn(1, 4, 2, 256, 32, device=self.device)
        restored = self.forward_transform(self.inverse_transform(angular_delay, 256), 256)
        torch.testing.assert_close(restored, angular_delay, rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
