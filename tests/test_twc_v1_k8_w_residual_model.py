"""Real CPU complex residual-W/RZF checks; no CUDA, Sionna or full decoder."""
import copy
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from models import twc_v1_k8_w_residual as residual
except ImportError:
    torch = residual = None


@unittest.skipIf(torch is None, "Real PyTorch unavailable; no fake dependency")
class ResidualWModelTests(unittest.TestCase):
    def setUp(self):
        self.saved_rng = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, self.saved_rng)
        self.saved_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.saved_threads)

    def channel(self, seed=401):
        generator = torch.Generator().manual_seed(seed)
        shape = (1, 8, 32, 256)
        return torch.complex(torch.randn(shape, generator=generator),
                             torch.randn(shape, generator=generator)) / 32 ** .5

    def base(self, estimate=None):
        estimate = self.channel() if estimate is None else estimate
        class TinyDecoder(torch.nn.Module):
            arm = "fdma"
            def __init__(self, value):
                super().__init__()
                self.estimate = torch.nn.Parameter(value.clone())
            def forward(self, **kwargs):
                return dict(H_hat=self.estimate, diagnostics={"tiny_cpu": True})
        return TinyDecoder(estimate)

    @staticmethod
    def inputs(gamma=12.5):
        # This fixture tests the new wrapper/head, not the unchanged full
        # decoder's production input validators.
        return dict(Y_rx=None, H_ul_est=None, soft=None, reliability=None,
                    pilots=None, gamma_db=gamma)

    def test_zero_head_matches_original_precoder_rates_and_total_power(self):
        base = self.base()
        original_weights = {k: v.clone() for k, v in base.state_dict().items()}
        control = residual.PrecodingDecoder(copy.deepcopy(base), "rzf_continue")
        model = residual.PrecodingDecoder(base, "w_residual", head_seed=517)
        a, b = control(**self.inputs()), model(**self.inputs())
        self.assertEqual(tuple(b["W"].shape), (1, 256, 32, 8))
        self.assertTrue(torch.equal(b["delta_W"], torch.zeros_like(b["delta_W"])))
        self.assertTrue(torch.equal(a["W"], b["W_base"]))
        torch.testing.assert_close(a["W"], b["W"], rtol=2e-6, atol=2e-7)
        truth = self.channel(402)
        a_loss, a_metrics = residual.precoding_objective(a["W"], truth)
        b_loss, b_metrics = residual.precoding_objective(b["W"], truth)
        torch.testing.assert_close(a_loss, b_loss, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(a_metrics["user_rate_bps_hz"], b_metrics["user_rate_bps_hz"], rtol=2e-6, atol=2e-6)
        self.assertLess(float((b_metrics["per_tone_power"] - 1.).abs().max().detach()), 1e-6)
        self.assertTrue(b["diagnostics"]["tiny_cpu"])
        for key, value in original_weights.items():
            self.assertTrue(torch.equal(base.state_dict()[key], value))

    def test_constructor_rng_isolated_deterministic_and_only_last_map_zero(self):
        base = self.base()
        before = torch.get_rng_state().clone()
        a = residual.PrecodingDecoder(copy.deepcopy(base), "w_residual", head_seed=81)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        b = residual.PrecodingDecoder(copy.deepcopy(base), "w_residual", head_seed=81)
        c = residual.PrecodingDecoder(copy.deepcopy(base), "w_residual", head_seed=82)
        for key, value in a.residual_head.state_dict().items():
            self.assertTrue(torch.equal(value, b.residual_head.state_dict()[key]))
        self.assertFalse(torch.equal(a.residual_head.input_projection.weight,
                                     c.residual_head.input_projection.weight))
        self.assertGreater(float(a.residual_head.input_projection.weight.detach().abs().sum()), 0.)
        self.assertEqual(float(a.residual_head.final_projection.weight.detach().abs().sum()), 0.)
        self.assertEqual(float(a.residual_head.final_projection.bias.detach().abs().sum()), 0.)

    def test_complex_rate_gradient_learns_head_without_initial_hidden_gradient(self):
        model = residual.PrecodingDecoder(self.base(), "w_residual", head_seed=83)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
        initial_base = model.base_decoder.estimate.detach().clone()
        result = model(**self.inputs(7.))
        loss, _ = residual.precoding_objective(result["W"], self.channel(403))
        loss.backward()
        head = model.residual_head
        self.assertGreater(float(head.final_projection.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.base_decoder.estimate.grad.abs().sum()), 0.)
        self.assertTrue(bool(torch.isfinite(model.base_decoder.estimate.grad).all()))
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            if not name.startswith("final_projection."):
                self.assertEqual(float(parameter.grad.abs().sum()), 0., name)
        optimizer.step()
        self.assertFalse(torch.equal(model.base_decoder.estimate, initial_base))
        changed = model(**self.inputs(7.))
        self.assertGreater(float(changed["delta_W"].detach().abs().sum()), 0.)
        self.assertLess(float((changed["W"].abs().square().sum((-2, -1)) - 1.).abs().max().detach()), 1e-6)
        optimizer.zero_grad(set_to_none=True)
        residual.precoding_objective(changed["W"], self.channel(403))[0].backward()
        self.assertGreater(float(head.input_projection.weight.grad.abs().sum()), 0.)

    def test_attention_couples_users_without_coupling_tones(self):
        model = residual.PrecodingDecoder(self.base(), "w_residual", head_seed=84)
        with torch.no_grad():
            torch.nn.init.normal_(model.residual_head.final_projection.weight, std=.02)
        channel = self.channel().requires_grad_(True)
        delta = model.residual_head(channel, gamma_db=5.)
        # One user's beam depends on another user's token at the same tone.
        delta[0, 12, 0, 0].real.backward()
        self.assertGreater(float(channel.grad[0, 1:, :, 12].abs().sum()), 0.)
        self.assertEqual(float(channel.grad[..., :12].abs().sum()), 0.)
        self.assertEqual(float(channel.grad[..., 13:].abs().sum()), 0.)

    def test_normalizer_preserves_column_power_ratios(self):
        raw = torch.ones((1, 256, 32, 8), dtype=torch.complex64)
        raw[..., 0] *= 3.
        W = residual.normalize_precoder(raw)
        columns = W.abs().square().sum(-2)
        torch.testing.assert_close(columns[..., 0] / columns[..., 1], torch.full((1, 256), 9.))
        torch.testing.assert_close(W.abs().square().sum((-2, -1)), torch.ones(1, 256))

    def test_public_gamma_batch_broadcast_and_signal_are_retained(self):
        head = residual.PrecodingDecoder(self.base(), "w_residual").residual_head
        with torch.no_grad():
            torch.nn.init.normal_(head.final_projection.weight, std=.02)
        repeated = self.channel().expand(2, -1, -1, -1).contiguous()
        scalar = head(repeated, gamma_db=10.)
        vector = head(repeated, gamma_db=torch.tensor([10., 10.]))
        torch.testing.assert_close(scalar, vector)
        torch.testing.assert_close(vector[0], vector[1])
        changed = head(repeated, gamma_db=torch.tensor([0., 25.]))
        self.assertGreater(float((changed[0] - changed[1]).detach().abs().sum()), 0.)

    def test_invalid_inputs_and_nonfinite_or_zero_candidates_fail(self):
        for method in ("other", None):
            with self.assertRaises(ValueError):
                residual.PrecodingDecoder(self.base(), method)
        for seed in (True, -1, 2**32, 1.2):
            with self.assertRaises(ValueError):
                residual.PrecodingDecoder(self.base(), "w_residual", head_seed=seed)
        model = residual.PrecodingDecoder(self.base(), "w_residual")
        for gamma in (True, complex(1., 0.), -1., 26., float("nan"), float("inf"),
                      [1., 2.], [[1.]], torch.tensor(1., requires_grad=True)):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                model(**self.inputs(gamma))
        for estimate in (self.channel().real, self.channel().to(torch.complex128),
                         self.channel()[..., :255], torch.zeros_like(self.channel()),
                         self.channel() * complex(float("nan"), 0.)):
            with self.assertRaises((ValueError, FloatingPointError)):
                residual.PrecodingDecoder(self.base(estimate), "rzf_continue")(**self.inputs())
        good = torch.ones((1, 256, 32, 8), dtype=torch.complex64)
        for bad in (good.real, good[..., :7], good.to(torch.complex128),
                    torch.zeros_like(good), good * complex(float("inf"), 0.)):
            with self.assertRaises((ValueError, FloatingPointError)):
                residual.normalize_precoder(bad)
        with self.assertRaisesRegex(FloatingPointError, "total power"):
            residual.precoding_objective(good, self.channel())

    def test_metadata_and_forward_boundary_expose_no_truth_inputs(self):
        control = residual.PrecodingDecoder(self.base(), "rzf_continue")
        model = residual.PrecodingDecoder(self.base(), "w_residual")
        self.assertEqual(control.head_parameter_count(), 0)
        metadata = model.metadata()
        self.assertEqual(metadata["head_parameters"], model.head_parameter_count())
        self.assertGreater(metadata["head_parameters"], 0)
        self.assertEqual(metadata["total_parameters"], metadata["head_parameters"] + metadata["base_decoder_parameters"])
        self.assertEqual(metadata["residual_scale"], 1 / 16)
        self.assertFalse(metadata["rzf_output_space_expansion_claim"])
        parameters = inspect.signature(model.forward).parameters
        self.assertNotIn("H_true", parameters)
        self.assertNotIn("H_dl", parameters)
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
        with self.assertRaises(TypeError):
            model(**self.inputs(), H_true=self.channel())


if __name__ == "__main__":
    unittest.main()
