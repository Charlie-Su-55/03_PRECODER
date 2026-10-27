"""Focused grouped-model static checks and genuine-runtime small-module checks.

No test constructs GroupedCodecPrecoder or the 34.9M-parameter full BS. The
controller owns actual five-arm initialization, batch48 execution and restoration.
Missing Sionna skips model-module tests locally; explicit CUDA requests fail.
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
MODEL = ROOT / "models/grouped_codec_precoder.py"
FULL_LF_SHA = "67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb"


def tree():
    return ast.parse(MODEL.read_text(encoding="utf-8-sig"))


def cls(name):
    return next(node for node in tree().body if isinstance(node, ast.ClassDef) and node.name == name)


def method(name, function):
    return next(node for node in cls(name).body if isinstance(node, ast.FunctionDef) and node.name == function)


class StaticGroupedModelTests(unittest.TestCase):
    def test_original_full_unchanged_and_new_physical_path_explicit(self):
        raw = (ROOT / "models/adaptive_hybrid.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), FULL_LF_SHA)
        model = cls("GroupedCodecPrecoder")
        self.assertEqual([ast.unparse(base) for base in model.bases], ["nn.Module"])
        source = ast.unparse(model)
        for forbidden in ("self.running_pwr", "self.add_noise", "self.transmit",
                          "AdaptiveHybridPrecoder(cfg)", "load_state_dict(", "torch.load(",
                          "ALLOCATION_GRID", "GAMMA_GRID"):
            self.assertNotIn(forbidden, source)
        forward = ast.unparse(method("GroupedCodecPrecoder", "_forward"))
        self.assertEqual(forward.count("protocol.transmit("), 1)
        self.assertEqual(forward.count("protocol.recover("), 1)
        self.assertIn("zf_solver = AdaptiveHybridPrecoder.zf_solver", source)
        args = method("GroupedCodecPrecoder", "forward_with_representation").args
        self.assertEqual([arg.arg for arg in args.args], ["self", "H_dl", "H_ul", "gamma_db"])
        self.assertIn("standard_noise", [arg.arg for arg in args.kwonlyargs])

    def test_original_as_encoder_kept_and_ordered_full_window_pooling(self):
        legacy = ast.unparse(cls("LegacyUE"))
        self.assertIn("self.encoder = AdaptiveHybridEncoder(cfg)", legacy)
        self.assertIn("self.condition = ConditionEmbed(256, 64, 256)", legacy)
        self.assertIn("H_dl.reshape(batch * 4, 32, 256), 0, 64, cond", legacy)
        ordered = ast.unparse(cls("OrderedWindowUE"))
        self.assertIn("angular_delay_transform(H_dl, 256)", ordered)
        self.assertIn("nn.Linear(4096, 512)", ordered)
        self.assertIn("SwinTransformerBlock(128, 4, 4, 4.0) for _ in range(4)", ordered)
        self.assertIn("torch.complex(real[:, :256], real[:, 256:])", ordered)
        pool = next(node for node in tree().body
                    if isinstance(node, ast.FunctionDef) and node.name == "ordered_window_pool")
        self.assertIn("16, 4, 2, 4, 128).mean((2, 4))", ast.unparse(pool))
        self.assertNotIn("mean(dim=1)", ordered)

    def test_true_resource_bias_and_all_inactive_context_masks(self):
        chunk = ast.unparse(method("MappedAttention", "_mapped_chunk"))
        self.assertIn("query_q = support[query_indices]", chunk)
        self.assertIn("query_q[:, None] == key_q[None, :]", chunk)
        self.assertNotIn("remainder(", chunk)
        self.assertIn("kind in ('self', 'spatial')", chunk)
        self.assertIn("masked_fill(~active[None, None, None, :], -torch.inf)", chunk)
        attention = ast.unparse(method("MappedAttention", "forward"))
        self.assertIn("query[:, start:start + self.query_chunk], key, value", attention)
        self.assertNotIn("memory[:,", attention)
        self.assertIn("self.activation_checkpoint and self.training", attention)
        layer = method("MappedJointLayer", "forward")
        masks = [node for node in ast.walk(layer) if isinstance(node, ast.BinOp)
                 and isinstance(node.op, ast.Mult) and ast.unparse(node.right) == "mask"]
        self.assertEqual(len(masks), 3)

    def test_bs_masks_initial_query_residual_and_fc_input(self):
        forward = ast.unparse(method("GroupedJointSwinDecoder", "forward"))
        self.assertIn("soft = z_hat.masked_fill(~mask, 0)", forward)
        self.assertIn("self.query_projection(features) * mask[..., None]", forward)
        self.assertIn("corrected = (soft + correction).masked_fill(~mask, 0)", forward)
        self.assertLess(forward.index("corrected ="), forward.index("self.codec.fc("))
        self.assertIn("torch.cat((corrected.real, corrected.imag), dim=-1)", forward)
        self.assertIn("reshape(batch, 4, 2, 256, 32)", forward)
        self.assertIn("SWIN_NC=256", ast.unparse(method("GroupedJointSwinDecoder", "__init__")))
        memory = ast.unparse(cls("GroupedObservationMemory"))
        self.assertIn("self.activity = nn.Linear(4, 128)", memory)
        self.assertIn("physical_mask.T.to(received.dtype)", memory)

    def test_independent_initialization_addresses_no_historical_state(self):
        init = ast.unparse(method("GroupedCodecPrecoder", "__init__"))
        self.assertEqual(init.count("torch.random.fork_rng(devices=[])"), 2)
        self.assertEqual(init.count("torch.random.default_generator.manual_seed("), 2)
        self.assertNotIn("torch.manual_seed(", init)
        self.assertIn("RandomAddress(cfg.SEED, 'init', 0, 'bs')", init)
        self.assertIn("RandomAddress(cfg.SEED, 'init', 0, 'ue_' + family)", init)
        source = ast.unparse(tree())
        for forbidden in ("sys.modules", "icc08", "torch.backends", "torch.load"):
            self.assertNotIn(forbidden, source)
        inactive = ast.unparse(method("GroupedCodecPrecoder", "inactive_parameter_names"))
        self.assertIn("'ue.encoder.fdma_'", inactive)
        self.assertIn("'ue.encoder.fc_fdma.'", inactive)

    def test_decoder_receiver_boundary_and_post_prediction_detached_truth(self):
        decoder = method("GroupedJointSwinDecoder", "forward")
        self.assertEqual([arg.arg for arg in decoder.args.args],
                         ["self", "z_hat", "reliability", "Y", "H_ul_est", "gamma_db"])
        forbidden = {"H_dl", "X", "standard_noise", "canonical_truth", "raw"}
        self.assertFalse({node.id for node in ast.walk(decoder) if isinstance(node, ast.Name)} & forbidden)
        forward = method("GroupedCodecPrecoder", "_forward")
        diagnostics = next(node for node in forward.body if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "diagnostics")
        earlier = ast.unparse(ast.Module(body=forward.body[:forward.body.index(diagnostics)], type_ignores=[]))
        self.assertIn("W = W_rzf *", earlier)
        self.assertIn("'true_z': canonical_truth.detach()", ast.unparse(diagnostics))
        self.assertNotIn("self.diagnostic", ast.unparse(tree()))


class NumericalGroupedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("GROUPED_CODEC_TEST_DEVICE", "cpu")
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError("GROUPED_CODEC_TEST_DEVICE must be cpu or CUDA")
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "2.0.1":
            reason = f"Genuine Sionna 2.0.1 unavailable (found {version}); no model import/execution"
            if requested.startswith("cuda"):
                raise RuntimeError(reason)
            raise unittest.SkipTest(reason)
        import torch
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly required; no fallback")
        from configs.grouped_codec_train_config import protocol
        from models.grouped_codec_precoder import MappedAttention, MappedJointLayer, ordered_window_pool
        cls.torch, cls.device = torch, torch.device(requested)
        cls.protocol = protocol
        cls.Attention, cls.Layer = MappedAttention, MappedJointLayer
        cls.pool = staticmethod(ordered_window_pool)
        cls.old_threads = torch.get_num_threads()
        if requested == "cpu":
            torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch = self.torch
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
        torch.manual_seed(933)

    def tearDown(self):
        self.torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng:
            self.torch.cuda.set_rng_state_all(self.cuda_rng)

    def geometry(self, layout="group2"):
        return self.protocol.tensor_geometry(self.protocol.LayoutSpec(layout), self.device)

    def test_ordered_window_values_and_gradients(self):
        torch = self.torch
        values = torch.arange(512, device=self.device, dtype=torch.float32).reshape(1, 512, 1).expand(1, 512, 128).clone().requires_grad_()
        output = self.pool(values)
        expected = []
        for row in range(16):
            for col in range(2):
                window = [(row * 4 + i) * 8 + col * 4 + j for i in range(4) for j in range(4)]
                expected.append(sum(window) / 16)
        torch.testing.assert_close(output[0, :, 0], torch.tensor(expected, device=self.device),
                                   rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, 1 / 16), rtol=0, atol=0)

    def test_true_q_bias_matches_dense_reference_and_inactive_keys_have_zero_gradient(self):
        torch = self.torch
        geometry = self.geometry()
        support, active = geometry["support_sc"].flatten(), geometry["active_mask"].flatten()
        module = self.Attention(width=16, heads=4, query_chunk=64).to(self.device).train()
        reference = deepcopy(module)
        query = torch.randn(1, 1024, 16, device=self.device, requires_grad=True)
        memory = torch.randn(1, 1024, 16, device=self.device, requires_grad=True)
        q_ref, m_ref = query.detach().clone().requires_grad_(), memory.detach().clone().requires_grad_()
        coupling = torch.rand(1, 4, 4, device=self.device)
        overlap = torch.full((4,), 0.02, device=self.device)
        coupling_weight = overlap.clone()
        resource = overlap.clone()
        output = module(query, memory, support, active, coupling, overlap, coupling_weight, resource, "self")
        qi = torch.arange(1024, device=self.device) // 256
        bias = ((support[:, None] == support[None]).float()[None, None] * overlap[None, :, None, None]
                + coupling[:, qi][:, :, qi][:, None] * coupling_weight[None, :, None, None])
        bias = bias.masked_fill(~active[None, None, None], -torch.inf)
        q, k, v = reference._split(reference.q(q_ref)), reference._split(reference.k(m_ref)), reference._split(reference.v(m_ref))
        dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                                 dropout_p=0.0, is_causal=False)
        expected = reference.out(dense.transpose(1, 2).reshape_as(output))
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=3e-6)
        weights = torch.randn_like(output) * active[None, :, None]
        (output * weights).sum().backward()
        (expected * weights).sum().backward()
        self.assertTrue(bool((memory.grad[:, ~active] == 0).all()))
        self.assertTrue(bool((query.grad[:, ~active] == 0).all()))
        torch.testing.assert_close(memory.grad, m_ref.grad, rtol=3e-5, atol=4e-6)
        for actual, wanted in zip(module.parameters(), reference.parameters()):
            self.assertIsNotNone(actual.grad)
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=3e-5, atol=8e-6)

    def test_spatial_cross_attention_ignores_inactive_context_values(self):
        torch = self.torch
        geometry = self.geometry("fdma")
        support, active = geometry["support_sc"].flatten(), geometry["active_mask"].flatten()
        module = self.Attention(width=16, heads=4, query_chunk=4).to(self.device).eval()
        query = torch.randn(1, 13, 16, device=self.device)
        memory = torch.randn(1, 1024, 16, device=self.device, requires_grad=True)
        first = module(query, memory, support, active, kind="spatial")
        perturbed = memory.detach().clone()
        perturbed[:, ~active] += 100
        second = module(query, perturbed, support, active, kind="spatial")
        self.assertTrue(torch.equal(first, second))
        first.square().sum().backward()
        self.assertTrue(bool((memory.grad[:, ~active] == 0).all()))
        self.assertGreater(float(memory.grad[:, active].abs().sum()), 0)

    def test_each_joint_layer_update_masks_inactive_queries_and_gradients(self):
        torch = self.torch
        geometry = self.geometry()
        support, active = geometry["support_sc"].flatten(), geometry["active_mask"].flatten()
        layer = self.Layer().to(self.device).train()
        x = torch.randn(1, 1024, 128, device=self.device, requires_grad=True)
        memory = torch.randn(1, 256, 128, device=self.device, requires_grad=True)
        coupling = torch.rand(1, 4, 4, device=self.device)
        weight = torch.full((4,), .02, device=self.device)
        output = layer(x, memory, support, active, coupling, weight, weight, weight)
        self.assertTrue(bool((output[:, ~active] == 0).all()))
        (output * torch.randn_like(output)).sum().backward()
        self.assertTrue(bool((x.grad[:, ~active] == 0).all()))
        for part in (layer.self_attention, layer.cross_attention, layer.mlp):
            grads = [p.grad for p in part.parameters()]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)


if __name__ == "__main__":
    unittest.main()
