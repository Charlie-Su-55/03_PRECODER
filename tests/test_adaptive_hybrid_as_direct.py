"""CPU synthetic contracts for the isolated AS-direct model.

Run from 03_PRECODER:
    python -X utf8 -B -m unittest discover -s tests -p test_adaptive_hybrid_as_direct.py -v

No real weights, datasets, training/evaluation entry points, optimizer steps,
or channel generators are used. Numerical checks use the normal model import;
missing PyTorch/Sionna is reported as a skip, never bypassed or mocked. Static
source checks remain available without those dependencies. CPU thread count
and RNG state are restored; precision and attention backends are not changed.
"""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_PATH = ROOT / "models/adaptive_hybrid.py"
DIRECT_PATH = ROOT / "models/adaptive_hybrid_as_direct.py"
ORIGINAL_SOURCE_LF_SHA256 = (
    "67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb"
)
# Fixed before any numerical run; never relaxed in response to a failure.
REPLAY_RTOL = 1e-6
REPLAY_ATOL = 1e-6

MODEL_IMPORT_ERROR = None
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    MODEL_IMPORT_ERROR = "PyTorch is not installed; CPU model checks were not run."

if torch is not None:
    try:
        from models.adaptive_hybrid import (
            AdaptiveHybridPrecoder,
            GNN_AggregationModule,
            UnfoldedWMMSELayer,
        )
        from models.adaptive_hybrid_as_direct import (
            AdaptiveHybridPrecoderASDirect,
            _ASDirectAggregation,
            _ASDirectRefinement,
            migrate_original_state_dict,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"sionna", "sionna.phy"}:
            raise
        MODEL_IMPORT_ERROR = (
            f"Normal model import requires missing {exc.name!r} through utils.__init__; "
            "CPU model checks were not run (no dependency bypass)."
        )


def source_class(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name)


def method(class_node, name):
    return next(node for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def without_docstring(function):
    function = deepcopy(function)
    if (function.body and isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant)
            and isinstance(function.body[0].value.value, str)):
        function.body.pop(0)
    return function


def synthetic_config(*, singleton=False):
    return SimpleNamespace(
        K_USERS=2, ANTENNAS=4, SUBCARRIERS=32, NUM_SUBBANDS=4,
        D_MODEL=256, NUM_HEADS=8, NUM_ENCODER_LAYERS=4,
        NUM_UNFOLD_LAYERS=5, GNN_AGG_DIM=48, DROPOUT=0.1,
        D_F_MAX_PER_USER=16, D_A_MAX_SHARED=32, TOTAL_POWER=1.0,
        GAMMA_GRID=((0, 32),) if singleton else ((16, 0), (8, 16), (0, 32)),
    )


def seed_cpu(seed):
    torch.random.default_generator.manual_seed(seed)


def build_model(*, direct=False, seed=191, cfg=None):
    seed_cpu(seed)
    cfg = synthetic_config() if cfg is None else cfg
    model_class = AdaptiveHybridPrecoderASDirect if direct else AdaptiveHybridPrecoder
    return model_class(cfg).cpu()


def synthetic_channels(cfg, batch_size=1):
    generator = torch.Generator(device="cpu").manual_seed(731)
    shape = (batch_size, cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS)
    h_dl = torch.randn(shape, dtype=torch.complex64, generator=generator)
    h_ul = torch.randn(shape, dtype=torch.complex64, generator=generator)
    error = torch.randn(shape, dtype=torch.complex64, generator=generator)
    return h_dl, h_ul, (0.9 + 0.1j) * h_ul + 0.05 * error


def deployed_precoder(model, h_hat):
    precoder = model.zf_solver(h_hat)
    power = precoder.abs().square().sum(dim=(1, 2), keepdim=True)
    return precoder * torch.sqrt(model.cfg.TOTAL_POWER / (power + 1e-9))


class ASDirectSourceTests(unittest.TestCase):
    def test_original_source_matches_pre_change_hash(self):
        source = ORIGINAL_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(source).hexdigest(), ORIGINAL_SOURCE_LF_SHA256)

    def test_top_forward_changes_only_as_activity_argument(self):
        old_class = source_class(ORIGINAL_PATH, "AdaptiveHybridPrecoder")
        new_class = source_class(DIRECT_PATH, "AdaptiveHybridPrecoderASDirect")
        self.assertEqual([ast.unparse(base) for base in new_class.bases],
                         ["AdaptiveHybridPrecoder"])
        old_forward = without_docstring(method(old_class, "forward"))
        new_forward = without_docstring(method(new_class, "forward"))
        layer_calls = [node for node in ast.walk(new_forward)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == "layer"]
        self.assertEqual(len(layer_calls), 1)
        self.assertEqual(len(layer_calls[0].keywords), 1)
        keyword = layer_calls[0].keywords[0]
        self.assertEqual(keyword.arg, "as_active")
        self.assertEqual(ast.dump(keyword.value), ast.dump(ast.parse("d_a > 0", mode="eval").body))
        layer_calls[0].keywords.clear()
        self.assertEqual(ast.dump(new_forward), ast.dump(old_forward))

    def test_aggregation_changes_only_return_of_existing_message(self):
        original = without_docstring(method(source_class(
            ORIGINAL_PATH, "GNN_AggregationModule"), "forward"))
        direct = without_docstring(method(source_class(
            DIRECT_PATH, "_ASDirectAggregation"), "forward"))
        self.assertIsInstance(direct.body[-1], ast.Return)
        self.assertEqual(ast.dump(direct.body[-1].value), ast.dump(ast.parse(
            "A_feat, msg.reshape(B, N_sb, K, D)", mode="eval").body))
        direct.body[-1] = deepcopy(original.body[-1])
        self.assertEqual(ast.dump(direct), ast.dump(original))

    def test_refinement_changes_only_message_unpack_and_final_residual(self):
        original = without_docstring(method(source_class(
            ORIGINAL_PATH, "UnfoldedWMMSELayer"), "forward"))
        direct = without_docstring(method(source_class(
            DIRECT_PATH, "_ASDirectRefinement"), "forward"))
        self.assertEqual([argument.arg for argument in direct.args.kwonlyargs], ["as_active"])
        self.assertEqual(direct.args.kw_defaults, [None])
        direct.args.kwonlyargs.clear()
        direct.args.kw_defaults.clear()
        aggregation = [node for node in direct.body if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Attribute)
                       and node.value.func.attr == "A_estimator"]
        self.assertEqual(len(aggregation), 1)
        self.assertEqual(ast.dump(aggregation[0].targets[0]), ast.dump(ast.parse(
            "A_feat, msg = value").body[0].targets[0]))
        aggregation[0].targets = [ast.Name(id="A_feat", ctx=ast.Store())]
        residual = direct.body[-2]
        self.assertEqual(ast.dump(residual), ast.dump(ast.parse(
            "if as_active:\n"
            "    W_out = W_out + torch.tanh(self.alpha) * msg.permute(0, 2, 1, 3).unsqueeze(2)\n"
        ).body[0]))
        direct.body.pop(-2)
        self.assertEqual(ast.dump(direct), ast.dump(original))

    def test_no_hooks_forward_caches_or_global_monkey_patches(self):
        tree = ast.parse(DIRECT_PATH.read_text(encoding="utf-8"))
        prohibited = {
            "register_forward_hook", "register_forward_pre_hook", "register_backward_hook",
            "register_full_backward_hook", "register_full_backward_pre_hook",
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "__class__",
        }
        for node in ast.walk(tree):
            self.assertNotIsInstance(node, (ast.Global, ast.Nonlocal))
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, prohibited)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval"})
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                for target in ast.walk(node):
                    if isinstance(target, ast.Attribute) and isinstance(target.ctx, ast.Store):
                        self.fail("A forward must not store module/global state or cache messages.")
        imported_names = {
            item.asname or item.name.split(".")[0]
            for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            for item in node.names
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                base = node.value
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    self.assertNotIn(base.id, imported_names, "Do not modify imported modules/classes.")

    def test_original_physical_methods_and_strict_loader_are_inherited(self):
        new_class = source_class(DIRECT_PATH, "AdaptiveHybridPrecoderASDirect")
        names = {node.name for node in new_class.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"transmit", "add_noise", "_pad_chunks", "zf_solver",
                         "load_state_dict", "state_dict"}.isdisjoint(names))
        tree = ast.parse(DIRECT_PATH.read_text(encoding="utf-8"))
        migration = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == "migrate_original_state_dict")
        loads = [node for node in ast.walk(migration) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict"]
        self.assertEqual(len(loads), 1)
        self.assertEqual([ast.literal_eval(item.value) for item in loads[0].keywords
                          if item.arg == "strict"], [True])


@unittest.skipIf(MODEL_IMPORT_ERROR is not None, MODEL_IMPORT_ERROR or "")
class ASDirectNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_rng = torch.get_rng_state().clone()

    def tearDown(self):
        torch.set_rng_state(self.previous_rng)

    def assertTensorExact(self, actual, expected, context=""):
        self.assertEqual(actual.shape, expected.shape, context)
        self.assertEqual(actual.dtype, expected.dtype, context)
        self.assertTrue(torch.equal(actual, expected), context)

    def assertPrecoder(self, value, cfg, batch_size):
        self.assertEqual(tuple(value.shape),
                         (batch_size, cfg.ANTENNAS, cfg.K_USERS, cfg.SUBCARRIERS))
        self.assertEqual(value.device.type, "cpu")
        self.assertTrue(value.is_complex())
        self.assertTrue(torch.isfinite(value).all().item())
        power = value.abs().square().sum(dim=(1, 2))
        torch.testing.assert_close(power, torch.full_like(power, cfg.TOTAL_POWER),
                                   rtol=REPLAY_RTOL, atol=REPLAY_ATOL)

    def test_constructor_adds_only_five_zero_scalar_gates_without_rng_change(self):
        original = build_model()
        expected_rng = torch.get_rng_state().clone()
        direct = build_model(direct=True)
        self.assertTensorExact(torch.get_rng_state(), expected_rng, "constructor RNG")
        expected_gates = tuple(f"layers.{index}.alpha" for index in range(5))
        self.assertEqual(direct.gate_state_keys, expected_gates)
        old_state, new_state = original.state_dict(), direct.state_dict()
        self.assertEqual(set(new_state) - set(old_state), set(expected_gates))
        self.assertEqual(set(old_state) - set(new_state), set())
        old_parameters = dict(original.named_parameters())
        new_parameters = dict(direct.named_parameters())
        self.assertEqual(set(new_parameters) - set(old_parameters), set(expected_gates))
        self.assertEqual(sum(p.numel() for p in direct.parameters())
                         - sum(p.numel() for p in original.parameters()), 5)
        for name, value in old_state.items():
            self.assertTensorExact(new_state[name], value, name)
        old_buffers, new_buffers = dict(original.named_buffers()), dict(direct.named_buffers())
        self.assertEqual(set(old_buffers), set(new_buffers))
        for name in old_buffers:
            self.assertTensorExact(new_buffers[name], old_buffers[name], name)
        gates = [new_parameters[name] for name in expected_gates]
        self.assertEqual(len({gate.data_ptr() for gate in gates}), 5)
        for gate in gates:
            self.assertEqual(tuple(gate.shape), ())
            self.assertTrue(gate.requires_grad)
            self.assertEqual(gate.item(), 0.0)
        self.assertEqual(direct.architecture, "as_direct_v1")
        self.assertEqual(direct.architecture_metadata["architecture"], "as_direct_v1")
        json.dumps(direct.architecture_metadata)
        for name in ("transmit", "add_noise", "_pad_chunks", "zf_solver", "load_state_dict"):
            self.assertIs(getattr(type(direct), name), getattr(type(original), name))
        for module in direct.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._backward_hooks)

    def test_explicit_migration_copies_all_original_values_and_resets_gates(self):
        original = build_model(seed=17)
        direct = build_model(direct=True, seed=93)
        with torch.no_grad():
            for layer in direct.layers:
                layer.alpha.fill_(0.75)
        source = original.state_dict()
        source_before = {key: value.clone() for key, value in source.items()}
        result = migrate_original_state_dict(direct, source)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(set(source), set(source_before))
        for name, value in source.items():
            self.assertTensorExact(value, source_before[name], "source " + name)
            self.assertTensorExact(direct.state_dict()[name], value, "destination " + name)
        for layer in direct.layers:
            self.assertEqual(layer.alpha.item(), 0.0)
        with torch.no_grad():
            for index, layer in enumerate(direct.layers):
                layer.alpha.fill_(0.1 * (index + 1))
        restored = build_model(direct=True, seed=257)
        result = restored.load_state_dict(direct.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for name, value in direct.state_dict().items():
            self.assertTensorExact(restored.state_dict()[name], value, "AS-direct roundtrip " + name)

    def test_invalid_migration_is_rejected_before_destination_changes(self):
        original = build_model(seed=17)
        direct = build_model(direct=True, seed=93)
        source = original.state_dict()
        before = {key: value.clone() for key, value in direct.state_dict().items()}
        malformed = {}
        malformed["missing_buffer"] = {key: value for key, value in source.items()
                                       if key != "running_pwr"}
        malformed["missing_parameter"] = {key: value for key, value in source.items()
                                          if key != "pad_f"}
        malformed["unexpected_key"] = {**source, "unexpected": torch.zeros(())}
        malformed["already_has_gate"] = {**source, "layers.0.alpha": torch.zeros(())}
        malformed["wrong_shape"] = {**source, "running_pwr": source["running_pwr"][:1]}
        malformed["wrong_dtype"] = {**source, "pad_f": source["pad_f"].double()}
        for label, candidate in malformed.items():
            with self.subTest(case=label):
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    migrate_original_state_dict(direct, candidate)
                for name, value in direct.state_dict().items():
                    self.assertTensorExact(value, before[name], label + ": " + name)

    def test_plain_strict_load_rejects_original_state_missing_gates(self):
        original = build_model()
        direct = build_model(direct=True)
        with self.assertRaisesRegex(RuntimeError, r"layers\.0\.alpha"):
            direct.load_state_dict(original.state_dict(), strict=True)
        missing_one_gate = {key: value for key, value in direct.state_dict().items()
                            if key != "layers.2.alpha"}
        with self.assertRaisesRegex(RuntimeError, r"layers\.2\.alpha"):
            direct.load_state_dict(missing_one_gate, strict=True)

    def test_zero_gate_replays_hhat_deployed_rzf_and_eval_with_identical_rng(self):
        cfg = synthetic_config()
        original = build_model(cfg=cfg)
        direct = build_model(direct=True, seed=999, cfg=cfg)
        initial_state = {key: value.clone() for key, value in original.state_dict().items()}
        h_dl, h_ul, h_ul_est = synthetic_channels(cfg)
        maxima = {}

        def compare(label, actual, expected):
            difference = (actual - expected).abs()
            absolute = difference.max().item()
            relative = (difference / expected.abs().clamp_min(1e-12)).max().item()
            previous = maxima.setdefault(label, {"max_abs": 0.0, "max_rel": 0.0})
            previous["max_abs"] = max(previous["max_abs"], absolute)
            previous["max_rel"] = max(previous["max_rel"], relative)
            torch.testing.assert_close(actual, expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)

        for training in (True, False):
            for estimate_name, estimate in (("default", None), ("estimated", h_ul_est)):
                for index, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
                    with self.subTest(training=training, estimate=estimate_name,
                                      allocation=(d_f, d_a)), torch.no_grad():
                        original.load_state_dict(initial_state, strict=True)
                        migrate_original_state_dict(direct, initial_state)
                        original.train(training)
                        direct.train(training)
                        seed_cpu(8200 + index)
                        start = torch.get_rng_state().clone()
                        expected = original(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=estimate)
                        end = torch.get_rng_state().clone()
                        torch.set_rng_state(start)
                        actual = direct(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=estimate)
                        self.assertTensorExact(torch.get_rng_state(), end, "forward RNG")
                        self.assertTensorExact(direct.running_pwr, original.running_pwr, "power buffer")
                        if training:
                            compare("W_train", actual[0], expected[0])
                            compare("H_hat", actual[1], expected[1])
                            compare("deployed_RZF", deployed_precoder(direct, actual[1]),
                                    deployed_precoder(original, expected[1]))
                            self.assertPrecoder(actual[0], cfg, 1)
                        else:
                            compare("W_eval", actual, expected)
                            self.assertPrecoder(actual, cfg, 1)
        print("AS-direct zero-gate replay (rtol=atol=1e-6): " + json.dumps(maxima, sort_keys=True))

    def test_endpoints_and_hybrid_are_finite_for_two_sample_batch(self):
        cfg = synthetic_config()
        model = build_model(direct=True, cfg=cfg).eval()
        h_dl, h_ul, estimate = synthetic_channels(cfg, batch_size=2)
        with torch.no_grad():
            for layer in model.layers:
                layer.alpha.fill_(0.2)
            for index, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
                with self.subTest(allocation=(d_f, d_a)):
                    self.assertPrecoder(model(h_dl, h_ul, 10.0, d_f, d_a, index,
                                              H_ul_est=estimate), cfg, 2)

    def test_pure_fdma_ignores_nonzero_direct_gates(self):
        cfg = synthetic_config()
        original = build_model(cfg=cfg)
        direct = build_model(direct=True, cfg=cfg)
        h_dl, h_ul, estimate = synthetic_channels(cfg)
        with torch.no_grad():
            for index, layer in enumerate(direct.layers):
                layer.alpha.fill_(0.25 * (index + 1))
            for training in (True, False):
                with self.subTest(training=training):
                    original.train(training)
                    direct.train(training)
                    seed_cpu(927)
                    start = torch.get_rng_state().clone()
                    expected = original(h_dl, h_ul, 10.0, 16, 0, 0, H_ul_est=estimate)
                    end = torch.get_rng_state().clone()
                    torch.set_rng_state(start)
                    actual = direct(h_dl, h_ul, 10.0, 16, 0, 0, H_ul_est=estimate)
                    if training:
                        for left, right in zip(actual, expected):
                            self.assertTensorExact(left, right, "pure FDMA training")
                    else:
                        self.assertTensorExact(actual, expected, "pure FDMA inference")
                    self.assertTensorExact(torch.get_rng_state(), end, "pure FDMA RNG")
                    self.assertTensorExact(direct.running_pwr, original.running_pwr, "pure FDMA power")

    def test_default_ul_csi_matches_explicit_true_csi_and_estimate_is_used(self):
        cfg = synthetic_config()
        model = build_model(direct=True, cfg=cfg).eval()
        h_dl, h_ul, estimate = synthetic_channels(cfg)
        with torch.no_grad():
            seed_cpu(207)
            start = torch.get_rng_state().clone()
            implicit = model(h_dl, h_ul, 10.0, 8, 16, 1, H_ul_est=None)
            torch.set_rng_state(start)
            explicit = model(h_dl, h_ul, 10.0, 8, 16, 1, H_ul_est=h_ul)
            torch.set_rng_state(start)
            estimated = model(h_dl, h_ul, 10.0, 8, 16, 1, H_ul_est=estimate)
            self.assertTensorExact(implicit, explicit, "default vs explicit true UL CSI")
            self.assertPrecoder(estimated, cfg, 1)
            self.assertFalse(torch.equal(explicit, estimated), "BS-side estimate was ignored")

    def test_singleton_gamma_grid_keeps_full_feedback_capacity(self):
        full_cfg, singleton_cfg = synthetic_config(), synthetic_config(singleton=True)
        full = build_model(direct=True, cfg=full_cfg)
        singleton = build_model(direct=True, cfg=singleton_cfg)
        self.assertEqual(singleton.running_pwr.shape, (1,))
        self.assertEqual(singleton_cfg.GAMMA_GRID, ((0, 32),))
        self.assertEqual(singleton.encoder.fc_fdma.out_features, 2 * 16 // 4)
        self.assertEqual(singleton.encoder.fc_aircomp.out_features, 2 * 32 // 4)
        self.assertEqual(singleton.fdma_emb.in_features, 2 * 16 // 4)
        self.assertEqual(singleton.aircomp_emb.in_features, 2 * 32 // 4)
        for name, value in full.state_dict().items():
            if name != "running_pwr":
                self.assertTensorExact(singleton.state_dict()[name], value, name)
        original = build_model(cfg=singleton_cfg)
        migrate_original_state_dict(singleton, original.state_dict())
        singleton.eval()
        h_dl, h_ul, _ = synthetic_channels(singleton_cfg)
        with torch.no_grad():
            self.assertPrecoder(singleton(h_dl, h_ul, 10.0, 0, 32, 0), singleton_cfg, 1)

    def test_aggregation_exposes_existing_per_user_message_and_preserves_pair_features(self):
        seed_cpu(341)
        original = GNN_AggregationModule(256, d_a=48, n_heads=4)
        direct = _ASDirectAggregation(deepcopy(original))
        y_ac = torch.randn(2, 4, 4, 256)
        uv = torch.randn(2, 4, 2, 256)
        with torch.no_grad():
            expected_pair = original(y_ac, uv)
            actual_pair, msg = direct(y_ac, uv)
            flat_y, flat_uv = y_ac.reshape(8, 4, 256), uv.reshape(8, 2, 256)
            user_msg, _ = original.user_query_attn(flat_uv, flat_y, flat_y)
            expected_msg = original.msg_proj(user_msg).reshape(2, 4, 2, 256)
        self.assertTensorExact(actual_pair, expected_pair, "original pair aggregation")
        self.assertTensorExact(msg, expected_msg, "existing per-user message")
        self.assertEqual(set(direct.state_dict()), set(original.state_dict()))

    def test_isolated_direct_route_has_gate_gradient_at_zero_and_message_gradient_when_open(self):
        cfg = synthetic_config()
        seed_cpu(571)
        layer = _ASDirectRefinement(UnfoldedWMMSELayer(cfg, cfg.D_MODEL, d_a=cfg.GNN_AGG_DIM))
        w_feat = torch.randn(2, 2, 4, 4, 256)
        y_f = torch.randn_like(w_feat)
        y_ac = torch.randn(2, 4, 4, 256)
        probe = torch.randn_like(w_feat)
        uv = layer.uv_estimator(torch.cat([w_feat.mean(2), y_f.mean(2)], dim=-1))
        uv = uv.permute(0, 2, 1, 3).contiguous()
        _, msg = layer.A_estimator(y_ac.permute(0, 2, 1, 3).contiguous(), uv)
        direct_message = msg.permute(0, 2, 1, 3).unsqueeze(2)
        active = layer(w_feat, y_f, y_ac, as_active=True)
        inactive = layer(w_feat, y_f, y_ac, as_active=False)
        self.assertTensorExact(active, inactive, "zero-gate isolated output")
        alpha_grad, = torch.autograd.grad((active * probe).sum(), (layer.alpha,))
        expected_grad = (direct_message.detach() * probe).sum()
        self.assertTrue(torch.isfinite(alpha_grad).item())
        self.assertGreater(alpha_grad.abs().item(), 1e-8)
        torch.testing.assert_close(alpha_grad, expected_grad, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        inactive_grad, = torch.autograd.grad((inactive * probe).sum(), (layer.alpha,), allow_unused=True)
        self.assertIsNone(inactive_grad, "Pure FDMA must bypass the direct gate entirely.")
        with torch.no_grad():
            layer.alpha.fill_(0.3)
        active = layer(w_feat, y_f, y_ac, as_active=True)
        inactive = layer(w_feat, y_f, y_ac, as_active=False)
        delta = active - inactive
        expected_delta = torch.tanh(layer.alpha.detach()) * direct_message.detach()
        torch.testing.assert_close(delta, expected_delta.expand_as(delta),
                                   rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        gradients = torch.autograd.grad((delta * probe).sum(), tuple(layer.A_estimator.msg_proj.parameters()))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))

    def test_explicit_as_activity_keeps_route_active_even_for_zero_context(self):
        cfg = synthetic_config()
        seed_cpu(663)
        layer = _ASDirectRefinement(UnfoldedWMMSELayer(cfg, cfg.D_MODEL, d_a=cfg.GNN_AGG_DIM))
        w_feat = torch.randn(1, 2, 4, 4, 256)
        y_f = torch.randn_like(w_feat)
        y_ac = torch.zeros(1, 4, 4, 256)
        with torch.no_grad():
            layer.alpha.fill_(0.4)
            layer.A_estimator.msg_proj[-1].weight.zero_()
            layer.A_estimator.msg_proj[-1].bias.fill_(0.5)
            inactive = layer(w_feat, y_f, y_ac, as_active=False)
            active = layer(w_feat, y_f, y_ac, as_active=True)
            expected = inactive + torch.tanh(layer.alpha) * 0.5
        torch.testing.assert_close(active, expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        self.assertFalse(torch.equal(active, inactive), "AS activity must not depend on context magnitude.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
