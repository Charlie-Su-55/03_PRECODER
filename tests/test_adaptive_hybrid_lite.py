"""CPU-only synthetic checks for the parallel UE-FFN Lite model.

Run from 03_PRECODER:
    python -X utf8 -B -m unittest discover -s tests -p test_adaptive_hybrid_lite.py -v

No trainer, channel generation, real checkpoints, optimizer steps, or experiment
data are used. The existing model's normal import transitively requires
Sionna through utils.__init__; unavailable dependencies are explicitly reported
as skipped CPU checks, without mocking or bypassing that import. Physical
dimensions are small; D=256, four UE layers per branch, and
five BS refinement layers are retained. The parameter-count test constructs
the actual K4/Nt32/Nsc256 architecture, without forwarding it. Tests temporarily
use one CPU thread and restore both the thread count and CPU RNG state. No
precision, deterministic-algorithm, attention-backend, or noise settings change.
"""

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        from models.adaptive_hybrid import AdaptiveHybridPrecoder
        from models.adaptive_hybrid_lite import AdaptiveHybridPrecoderLite
    except ModuleNotFoundError as exc:
        if exc.name not in {"sionna", "sionna.phy"}:
            raise
        MODEL_IMPORT_ERROR = (
            f"Normal model import requires missing {exc.name!r} through utils.__init__; "
            "CPU model checks were not run (no dependency bypass)."
        )


# Canonical LF source hash, portable to Git checkouts on Windows and Linux.
# The separate pre-change raw Windows CRLF audit hash is
# efaad80d29d238326bcdd8a7e8e4212705a00d7a2edba1bff8bfb565a836ad25.
ORIGINAL_SOURCE_LF_SHA256 = (
    "67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb"
)
BRANCHES = ("fdma_backbone", "aircomp_backbone")


def synthetic_config(*, real_dimensions=False):
    if real_dimensions:
        users, antennas, subcarriers = 4, 32, 256
        allocations = ((64, 0), (48, 64), (32, 128), (16, 192), (0, 256))
    else:
        users, antennas, subcarriers = 2, 4, 32
        allocations = ((16, 0), (8, 16), (0, 32))
    return SimpleNamespace(
        K_USERS=users,
        ANTENNAS=antennas,
        SUBCARRIERS=subcarriers,
        NUM_SUBBANDS=4,
        D_MODEL=256,
        NUM_HEADS=8,
        NUM_ENCODER_LAYERS=4,
        NUM_UNFOLD_LAYERS=5,
        GNN_AGG_DIM=48,
        DROPOUT=0.1,
        D_F_MAX_PER_USER=subcarriers // users,
        D_A_MAX_SHARED=subcarriers,
        GAMMA_GRID=allocations,
        TOTAL_POWER=1.0,
    )


def seed_cpu(seed):
    # Unlike torch.manual_seed(), this does not seed CUDA generators.
    torch.random.default_generator.manual_seed(seed)


def build_model(*, width=None, seed=42, cfg=None):
    seed_cpu(seed)
    cfg = synthetic_config() if cfg is None else cfg
    if width is None:
        return AdaptiveHybridPrecoder(cfg).cpu()
    return AdaptiveHybridPrecoderLite(cfg, ue_ffn_dim=width).cpu()


def ue_ffn_keys(layers=4):
    return {
        f"encoder.{branch}.layers.{index}.{linear}.{tensor}"
        for branch in BRANCHES
        for index in range(layers)
        for linear in ("linear1", "linear2")
        for tensor in ("weight", "bias")
    }


def synthetic_channels(cfg, batch_size=2):
    generator = torch.Generator(device="cpu").manual_seed(731)
    shape = (batch_size, cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS)
    h_dl = torch.randn(shape, dtype=torch.complex64, generator=generator)
    h_ul = torch.randn(shape, dtype=torch.complex64, generator=generator)
    error = torch.randn(shape, dtype=torch.complex64, generator=generator)
    h_ul_est = (0.9 + 0.1j) * h_ul + 0.05 * error
    return h_dl, h_ul, h_ul_est


def task_loss_from_representation(model, h_hat, h_dl):
    """Only the deploy-consistent path from main_train.py:746-778.

    The sum-rate expression mirrors SumRateLoss at main_train.py:218-237;
    the legacy W_train returned by forward is intentionally not an input.
    """
    precoder = model.zf_solver(h_hat)
    power = precoder.abs().square().sum(dim=(1, 2), keepdim=True)
    precoder = precoder * torch.sqrt(model.cfg.TOTAL_POWER / (power + 1e-9))
    effective = torch.einsum("bkns,bnjs->bkjs", h_dl.conj(), precoder)
    received_power = effective.abs().square()
    signal = torch.diagonal(received_power, dim1=1, dim2=2).permute(0, 2, 1)
    interference = received_power.sum(dim=2) - signal
    sinr = signal / (interference + 10.0 ** (-25.0 / 10.0) + 1e-10)
    return -torch.log2(1.0 + sinr).sum(dim=1).mean(), precoder


class ProtectedOriginalSourceTests(unittest.TestCase):
    def test_original_source_matches_recorded_pre_change_hash(self):
        source = (ROOT / "models/adaptive_hybrid.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(source).hexdigest(), ORIGINAL_SOURCE_LF_SHA256)

    def test_expected_parameter_reduction_formula(self):
        # Two branches, four layers, each FFN has two D x F weights and an F bias.
        self.assertEqual(2 * 4 * (2048 - 1024) * (2 * 256 + 1), 4_202_496)

    def test_lite_statically_inherits_the_original_execution_paths(self):
        tree = ast.parse((ROOT / "models/adaptive_hybrid_lite.py").read_text(encoding="utf-8"))
        lite_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                          and node.name == "AdaptiveHybridPrecoderLite")
        self.assertEqual([ast.unparse(base) for base in lite_class.bases], ["AdaptiveHybridPrecoder"])
        methods = {node.name for node in lite_class.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"forward", "transmit", "add_noise", "zf_solver",
                         "load_state_dict", "state_dict"}.isdisjoint(methods))
        init = next(node for node in lite_class.body if isinstance(node, ast.FunctionDef)
                    and node.name == "__init__")
        self.assertEqual(init.args.args[-1].arg, "ue_ffn_dim")
        self.assertEqual(ast.literal_eval(init.args.defaults[-1]), 1024)
        calls = [node for node in ast.walk(init) if isinstance(node, ast.Call)]
        replacements = [node for node in calls if isinstance(node.func, ast.Name)
                        and node.func.id == "_replace_backbone_ffn"]
        self.assertEqual(len(replacements), 2)
        self.assertEqual({ast.unparse(node.args[0]) for node in replacements}, {
            "self.encoder.fdma_backbone", "self.encoder.aircomp_backbone",
        })


@unittest.skipIf(MODEL_IMPORT_ERROR is not None, MODEL_IMPORT_ERROR or "")
class AdaptiveHybridLiteTests(unittest.TestCase):
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
        if not torch.equal(actual, expected):
            max_error = (actual - expected).abs().max().item()
            self.fail(f"{context}: exact equality failed; max_abs_error={max_error}")

    def assertOutputExact(self, actual, expected, context):
        self.assertEqual(type(actual), type(expected), context)
        if isinstance(actual, tuple):
            self.assertEqual(len(actual), len(expected))
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assertTensorExact(left, right, f"{context}, output={index}")
        else:
            self.assertTensorExact(actual, expected, context)

    def assertPrecoder(self, precoder, cfg, batch_size):
        self.assertEqual(
            tuple(precoder.shape),
            (batch_size, cfg.ANTENNAS, cfg.K_USERS, cfg.SUBCARRIERS),
        )
        self.assertEqual(precoder.device.type, "cpu")
        self.assertTrue(precoder.is_complex())
        self.assertTrue(torch.isfinite(precoder).all().item())
        # This is the physical per-subcarrier constraint, not a replay tolerance.
        power = precoder.abs().square().sum(dim=(1, 2))
        torch.testing.assert_close(
            power, torch.full_like(power, cfg.TOTAL_POWER), rtol=0, atol=2e-6
        )

    def test_real_architecture_parameter_reduction(self):
        cfg = synthetic_config(real_dimensions=True)
        original = build_model(cfg=cfg)
        lite = build_model(width=1024, cfg=cfg)

        def counts(model):
            encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
            condition = sum(parameter.numel() for parameter in model.cond_embed.parameters())
            full = sum(parameter.numel() for parameter in model.parameters())
            return {"encoder": encoder, "condition": condition, "ue": encoder + condition,
                    "bs": full - encoder - condition, "full": full}

        before, after = counts(original), counts(lite)
        expected = 2 * 4 * (2048 - 1024) * (2 * 256 + 1)
        self.assertEqual(expected, 4_202_496)
        self.assertEqual(before["encoder"] - after["encoder"], expected)
        self.assertEqual(before["ue"] - after["ue"], expected)
        self.assertEqual(before["full"] - after["full"], expected)
        self.assertEqual(before["condition"], after["condition"])
        self.assertEqual(before["bs"], after["bs"])
        print("CPU constructor-only parameter counts: " + json.dumps({
            "original": before, "ueffn1024": after, "reduction": expected,
        }, sort_keys=True))

    def test_only_ue_ffn_widths_change_and_layer_attributes_are_preserved(self):
        original = build_model()
        lite = build_model(width=1024)
        self.assertIs(AdaptiveHybridPrecoderLite.forward, AdaptiveHybridPrecoder.forward)
        self.assertIs(AdaptiveHybridPrecoderLite.transmit, AdaptiveHybridPrecoder.transmit)
        self.assertIs(AdaptiveHybridPrecoderLite.add_noise, AdaptiveHybridPrecoder.add_noise)
        self.assertIs(AdaptiveHybridPrecoderLite.zf_solver, AdaptiveHybridPrecoder.zf_solver)
        for branch in BRANCHES:
            old_layers = getattr(original.encoder, branch).layers
            new_layers = getattr(lite.encoder, branch).layers
            self.assertEqual(len(old_layers), 4)
            self.assertEqual(len(new_layers), 4)
            for old, new in zip(old_layers, new_layers):
                self.assertEqual(old.linear1.out_features, 2048)
                self.assertEqual((new.linear1.in_features, new.linear1.out_features), (256, 1024))
                self.assertEqual((new.linear2.in_features, new.linear2.out_features), (1024, 256))
                self.assertIs(new.activation, old.activation)
                self.assertEqual(new.activation.__name__, "relu")
                self.assertFalse(new.norm_first)
                self.assertTrue(new.self_attn.batch_first)
                self.assertEqual(new.self_attn.num_heads, 8)
                self.assertEqual(new.self_attn.dropout, old.self_attn.dropout)
                for name in ("dropout", "dropout1", "dropout2"):
                    self.assertEqual(getattr(new, name).p, getattr(old, name).p)
                for name in ("norm1", "norm2"):
                    self.assertEqual(getattr(new, name).eps, getattr(old, name).eps)
        cross_subband = lite.cross_subband_attn.layers[0]
        self.assertEqual(cross_subband.linear1.out_features, 2048)
        self.assertEqual(cross_subband.self_attn.num_heads, 8)
        self.assertFalse(cross_subband.norm_first)
        self.assertEqual(cross_subband.activation.__name__, "relu")
        self.assertEqual(cross_subband.dropout.p, 0.0)
        self.assertEqual(len(lite.layers), 5)
        for layer in lite.layers:
            self.assertEqual(layer.ffn[0].out_features, 512)
            self.assertEqual(layer.ffn[2].in_features, 512)
            self.assertIsInstance(layer.ffn[1], torch.nn.GELU)
            self.assertEqual(layer.w_solver_attn.num_heads, 8)
            self.assertEqual(layer.cross_user_attn.num_heads, 4)
            self.assertEqual(layer.A_estimator.user_query_attn.num_heads, 4)

    def test_branch_ffn_templates_are_cloned_without_parameter_sharing(self):
        model = build_model(width=1024)
        for branch in BRANCHES:
            layers = getattr(model.encoder, branch).layers
            for name in ("linear1", "linear2"):
                for tensor_name in ("weight", "bias"):
                    parameters = [getattr(getattr(layer, name), tensor_name) for layer in layers]
                    self.assertEqual(len({id(parameter) for parameter in parameters}), 4)
                    self.assertEqual(len({parameter.data_ptr() for parameter in parameters}), 4)
                    for parameter in parameters[1:]:
                        self.assertTensorExact(parameter, parameters[0], f"{branch}.{name}.{tensor_name}")
        self.assertFalse(torch.equal(
            model.encoder.fdma_backbone.layers[0].linear1.weight,
            model.encoder.aircomp_backbone.layers[0].linear1.weight,
        ))

    def test_non_ffn_state_is_identical_and_shape_changes_are_limited(self):
        original = build_model().state_dict()
        changed_keys = ue_ffn_keys()
        expected_shape_changes = {key for key in changed_keys if not key.endswith("linear2.bias")}
        for width in (1024,):
            with self.subTest(width=width):
                narrower = build_model(width=width).state_dict()
                self.assertEqual(set(narrower), set(original))
                shape_changes = {key for key in original if original[key].shape != narrower[key].shape}
                self.assertEqual(shape_changes, expected_shape_changes)
                for key in original.keys() - changed_keys:
                    self.assertTensorExact(narrower[key], original[key], key)
                for branch in BRANCHES:
                    key = f"encoder.{branch}.layers.0.linear1.weight"
                    # A new random FFN must not be a sliced original FFN.
                    self.assertFalse(torch.equal(narrower[key], original[key][:width]))

    def test_2048_initialization_is_an_exact_state_and_rng_noop(self):
        original = build_model(seed=191)
        expected_rng = torch.get_rng_state().clone()
        lite = build_model(width=2048, seed=191)
        self.assertTensorExact(torch.get_rng_state(), expected_rng, "2048 constructor RNG")
        old_state, new_state = original.state_dict(), lite.state_dict()
        self.assertEqual(set(old_state), set(new_state))
        for key in old_state:
            self.assertTensorExact(new_state[key], old_state[key], key)

    def test_2048_strict_loading_and_train_eval_outputs_are_exact(self):
        cfg = synthetic_config()
        original = build_model(cfg=cfg)
        lite = build_model(width=2048, seed=999, cfg=cfg)
        loaded = lite.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        loaded_back = original.load_state_dict(lite.state_dict(), strict=True)
        self.assertEqual(loaded_back.missing_keys, [])
        self.assertEqual(loaded_back.unexpected_keys, [])
        h_dl, h_ul, h_ul_est = synthetic_channels(cfg, batch_size=1)
        for training in (False, True):
            original.train(training)
            lite.train(training)
            for index, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
                context = f"training={training}, allocation={(d_f, d_a)}, cpu_threads=1"
                with self.subTest(training=training, allocation=(d_f, d_a)), torch.no_grad():
                    seed_cpu(8200 + index)
                    start = torch.get_rng_state().clone()
                    expected = original(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=h_ul_est)
                    end = torch.get_rng_state().clone()
                    torch.set_rng_state(start)
                    actual = lite(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=h_ul_est)
                    self.assertOutputExact(actual, expected, context)
                    self.assertTensorExact(torch.get_rng_state(), end, context + ", RNG")
                    self.assertTensorExact(lite.running_pwr, original.running_pwr, context + ", power")

    def test_eval_endpoints_and_hybrid_have_finite_normalized_precoders(self):
        cfg = synthetic_config()
        model = build_model(width=1024, cfg=cfg).eval()
        for batch_size in (1, 2):
            h_dl, h_ul, _ = synthetic_channels(cfg, batch_size=batch_size)
            for index, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
                with self.subTest(batch_size=batch_size, allocation=(d_f, d_a)), torch.no_grad():
                    output = model(h_dl, h_ul, 10.0, d_f, d_a, index)
                    self.assertPrecoder(output, cfg, batch_size)

    def test_available_ul_csi_default_explicit_and_nontrivial_estimates(self):
        cfg = synthetic_config()
        model = build_model(width=1024, cfg=cfg).eval()
        h_dl, h_ul, h_ul_est = synthetic_channels(cfg)
        for index, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
            with self.subTest(allocation=(d_f, d_a)), torch.no_grad():
                seed_cpu(127)
                start = torch.get_rng_state().clone()
                implicit = model(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=None)
                torch.set_rng_state(start)
                explicit = model(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=h_ul)
                self.assertTensorExact(implicit, explicit, "None vs explicit true UL CSI")
                torch.set_rng_state(start)
                estimated = model(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=h_ul_est)
                self.assertPrecoder(estimated, cfg, h_dl.shape[0])
                self.assertFalse(torch.equal(estimated, explicit), "Nontrivial BS UL CSI was ignored")

    def test_deploy_task_loss_backward_reaches_only_active_ue_ffns(self):
        cfg = synthetic_config()
        model = build_model(width=1024, cfg=cfg).train()
        h_dl, h_ul, h_ul_est = synthetic_channels(cfg, batch_size=1)
        for index, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
            with self.subTest(allocation=(d_f, d_a)):
                model.zero_grad(set_to_none=True)
                output = model(h_dl, h_ul, 10.0, d_f, d_a, index, H_ul_est=h_ul_est)
                self.assertIsInstance(output, tuple)
                self.assertEqual(len(output), 2)
                legacy_precoder, h_hat = output
                self.assertPrecoder(legacy_precoder, cfg, h_dl.shape[0])
                self.assertEqual(tuple(h_hat.shape), tuple(h_dl.shape))
                self.assertTrue(torch.isfinite(h_hat).all().item())
                loss, deployed = task_loss_from_representation(model, h_hat, h_dl)
                self.assertPrecoder(deployed, cfg, h_dl.shape[0])
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                for branch, active in (("fdma_backbone", d_f > 0), ("aircomp_backbone", d_a > 0)):
                    for layer in getattr(model.encoder, branch).layers:
                        for linear in (layer.linear1, layer.linear2):
                            for parameter in linear.parameters():
                                if active:
                                    self.assertIsNotNone(parameter.grad, f"Missing {branch} FFN gradient")
                                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                                    self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
                                else:
                                    self.assertIsNone(parameter.grad)
                for parameter in model.h_estimator.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertTrue(all(parameter.grad is None for parameter in model.residual_head.parameters()))
                self.assertIsNone(model.residual_scale.grad)

    def test_lite1024_strict_synthetic_state_roundtrip(self):
        cfg = synthetic_config()
        source = build_model(width=1024, cfg=cfg).eval()
        restored = build_model(width=1024, seed=89, cfg=cfg).eval()
        state = {key: value.detach().clone() for key, value in source.state_dict().items()}
        loaded = restored.load_state_dict(state, strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        for key, value in restored.state_dict().items():
            self.assertTensorExact(value, state[key], key)
        h_dl, h_ul, _ = synthetic_channels(cfg, batch_size=1)
        with torch.no_grad():
            seed_cpu(124)
            rng = torch.get_rng_state().clone()
            expected = source(h_dl, h_ul, 10.0, 8, 16, 1)
            torch.set_rng_state(rng)
            actual = restored(h_dl, h_ul, 10.0, 8, 16, 1)
        self.assertTensorExact(actual, expected, "1024 strict roundtrip output")

    def test_original_and_lite_width_mismatch_fail_strict_loading(self):
        original = build_model()
        lite = build_model(width=1024)
        with self.assertRaisesRegex(RuntimeError, "size mismatch.*linear1"):
            lite.load_state_dict(original.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, "size mismatch.*linear1"):
            original.load_state_dict(lite.state_dict(), strict=True)

    def test_architecture_metadata_is_json_serializable_and_not_model_state(self):
        model = build_model(width=1024)
        metadata = model.architecture_metadata
        self.assertEqual(metadata, {
            "architecture": "ueffn1024", "ue_ffn_dim": 1024,
            "ue_layers": {"fdma": 4, "as": 4}, "D_MODEL": 256,
        })
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        self.assertEqual(model.experiment_name, "FDMA-AS-Lite1024")
        self.assertNotIn("_extra_state", model.state_dict())

    def test_invalid_widths_fail_before_consuming_rng(self):
        for width, exception in ((0, ValueError), (-1, ValueError), (True, TypeError),
                                 (1024.0, TypeError), ("1024", TypeError)):
            with self.subTest(width=width):
                before = torch.get_rng_state().clone()
                with self.assertRaises(exception):
                    AdaptiveHybridPrecoderLite(synthetic_config(), ue_ffn_dim=width)
                self.assertTensorExact(torch.get_rng_state(), before, "invalid-width RNG")


if __name__ == "__main__":
    unittest.main(verbosity=2)
