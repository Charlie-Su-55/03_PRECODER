"""Small synthetic grouped-PHY tests: no Sionna, real channels or model construction.

Default tensor checks use genuine CPU Torch. GROUPED_CODEC_TEST_DEVICE=cuda
explicitly requires CUDA; unavailable dependencies are never substituted.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "utils/grouped_feedback_protocol.py"
MODULE_NAME = "_grouped_feedback_protocol_v1"
if MODULE_NAME in sys.modules:
    protocol = sys.modules[MODULE_NAME]
    if Path(protocol.__file__).resolve() != PROTOCOL_PATH.resolve():
        raise RuntimeError("Wrong actual protocol module path")
else:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, PROTOCOL_PATH)
    protocol = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = protocol
    spec.loader.exec_module(protocol)


class LayoutAndAddressTests(unittest.TestCase):
    def test_all_layout_supports_and_canonical_maps(self):
        for arm in protocol.ARMS:
            with self.subTest(arm=arm):
                layout = protocol.layout_for_arm(arm)
                active = layout.active_per_user
                self.assertEqual(layout.occupied_resources, 256)
                self.assertEqual(len(layout.active_slots), active)
                self.assertEqual(set().union(*(set(row) for row in layout.supports)), set(range(256)))
                self.assertEqual({sum(q in row for row in layout.supports) for q in range(256)},
                                 {layout.users_per_resource})
                for k, (support, mapping) in enumerate(zip(layout.supports, layout.canonical_to_physical)):
                    self.assertEqual(len(set(support)), active)
                    self.assertEqual(tuple(mapping[slot] for slot in layout.active_slots), support)
                    self.assertTrue(all(q == -1 for slot, q in enumerate(mapping)
                                        if slot not in layout.active_slots))
                    self.assertEqual(sum(q >= 0 for q in mapping), active)
                    if layout.layout == "fdma":
                        self.assertEqual(support, tuple(4 * j + k for j in range(64)))
                    if layout.layout == "group2":
                        self.assertEqual(support, tuple(q for q in range(256) if q % 4 // 2 == k // 2))
                self.assertEqual(layout.active_slots[active // 4], 64)
                json.dumps(protocol.architecture_metadata(arm), allow_nan=False)

    def test_protocol_identity_n0_and_invalid_inputs(self):
        metadata = protocol.protocol_metadata()
        self.assertEqual(metadata["energy_per_frame_per_ue"], 64)
        self.assertFalse(metadata["historical_protocol_compatible"])
        self.assertFalse(metadata["power_ema"])
        self.assertEqual(protocol.noise_variance(0), 0.25)
        for gamma in (0, 5, 10, 15, 20, 25):
            N0 = protocol.noise_variance(gamma)
            self.assertAlmostEqual(10 * math.log10(64 / (256 * N0)), gamma, places=12)
        for value in (True, -1, 26, float("nan"), float("inf"), "10"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                protocol.noise_variance(value)
        with self.assertRaises(ValueError):
            protocol.LayoutSpec("hybrid")
        with self.assertRaises(ValueError):
            protocol.layout_for_arm("swin")

    def test_random_addresses_gamma_replay_and_namespace_separation(self):
        before = random.getstate()
        values = [protocol.sample_gamma(20261030, step) for step in (0, 1, 500, 501, 20000)]
        self.assertTrue(all(0 <= x <= 25 for x in values))
        self.assertEqual(values, [protocol.sample_gamma(20261030, step) for step in (0, 1, 500, 501, 20000)])
        self.assertEqual(before, random.getstate())
        addresses = [protocol.RandomAddress(20261030, split, 1, stream).seed_value
                     for split in ("init", "train", "validation", "test", "preflight")
                     for stream in ("gamma", "standard_noise", "ue_legacy", "ue_ordered", "bs")]
        self.assertEqual(len(set(addresses)), len(addresses))
        self.assertNotEqual(protocol.RandomAddress(20261030, "train", 500, "standard_noise").seed_value,
                            protocol.RandomAddress(20261030, "train", 501, "standard_noise").seed_value)
        for args in ((True, "train", 1, "gamma"), (1, "unknown", 1, "gamma"),
                     (1, "train", -1, "gamma"), (1, "train", 1, "arm")):
            with self.assertRaises(ValueError):
                protocol.RandomAddress(*args)

    def test_public_receiver_boundary_and_lazy_stdlib_import(self):
        self.assertEqual(tuple(inspect.signature(protocol.recover).parameters),
                         ("Y", "H_ul_est", "layout", "N0"))
        parsed = ast.parse(PROTOCOL_PATH.read_text(encoding="utf-8-sig"))
        top_imports = [ast.unparse(node) for node in parsed.body
                       if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("torch" in node or "sionna" in node for node in top_imports))
        code = ("import runpy,sys; runpy.run_path(" + repr(str(PROTOCOL_PATH)) + "); "
                "assert 'torch' not in sys.modules and 'sionna' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-S", "-c", code],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class NumericalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("GROUPED_CODEC_TEST_DEVICE", "cpu")
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError("GROUPED_CODEC_TEST_DEVICE must be cpu or CUDA")
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch" or requested != "cpu":
                raise
            raise unittest.SkipTest("Genuine Torch is unavailable; no replacement or installation")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested; no CPU fallback")
        cls.torch, cls.device = torch, torch.device(requested)
        cls.old_threads = torch.get_num_threads()
        if requested == "cpu":
            torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.cpu_rng = self.torch.get_rng_state().clone()
        self.cuda_rng = self.torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
        self.generator = self.torch.Generator(device="cpu").manual_seed(821)

    def tearDown(self):
        self.torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng:
            self.torch.cuda.set_rng_state_all(self.cuda_rng)

    def raw(self, batch=2):
        torch = self.torch
        parts = torch.randn((batch, 4, 256, 2), generator=self.generator)
        return torch.view_as_complex(parts).to(self.device)

    def channel(self, batch=2):
        torch = self.torch
        h = torch.zeros(batch, 4, 32, 256, dtype=torch.complex64, device=self.device)
        frequency = 1 + torch.arange(256, device=self.device) / 1024
        for user in range(4):
            h[:, user, user, :] = complex(user + 1, 0.3 * (user + 1)) * frequency
        return h

    def test_per_frame_energy_canonical_scatter_and_inactive_gradients(self):
        torch = self.torch
        for name in ("all4", "group2", "fdma"):
            with self.subTest(layout=name):
                layout = protocol.LayoutSpec(name)
                raw = self.raw().requires_grad_()
                X, canonical, energy = protocol.normalize_and_map(raw, layout)
                geometry = protocol.tensor_geometry(layout, self.device)
                torch.testing.assert_close(energy, torch.full_like(energy, 64),
                                           rtol=0, atol=protocol.ENERGY_ATOL)
                self.assertTrue(bool((canonical[:, ~geometry["active_mask"]] == 0).all()))
                self.assertTrue(bool((X[:, ~geometry["physical_mask"]] == 0).all()))
                for k in range(4):
                    self.assertTrue(torch.equal(canonical[:, k, geometry["active_slots"]],
                                                X[:, k, geometry["physical_supports"][k]]))
                X.real.sum().backward()
                self.assertTrue(bool(torch.isfinite(raw.grad).all()))
                self.assertTrue(bool((raw.grad[:, ~geometry["active_mask"]] == 0).all()))
                self.assertGreater(float(raw.grad[:, geometry["active_mask"]].abs().sum()), 0)

    def test_zero_near_zero_and_nonfinite_codewords_fail_stop(self):
        torch = self.torch
        layout = protocol.LayoutSpec("group2")
        for value in (0.0, 1e-10):
            with self.assertRaisesRegex(FloatingPointError, "fail-stop evidence"):
                protocol.normalize_and_map(torch.full((1, 4, 256), value, dtype=torch.complex64,
                                                      device=self.device), layout)
        for value in (complex(float("nan"), 0), complex(float("inf"), 0)):
            raw = self.raw(1)
            raw[0, 2, 5] = value
            with self.assertRaises(FloatingPointError):
                protocol.normalize_and_map(raw, layout)

    def test_known_complex_1_2_4_user_recovery_and_canonical_axes(self):
        torch = self.torch
        h, raw = self.channel(), self.raw()
        N0 = protocol.noise_variance(10)
        eps = torch.zeros((2, 32, 256), dtype=torch.complex64, device=self.device)
        for name in ("all4", "group2", "fdma"):
            with self.subTest(layout=name):
                layout = protocol.LayoutSpec(name)
                X, truth, _ = protocol.normalize_and_map(raw, layout)
                Y = protocol.transmit(h, X, layout, N0=N0, standard_noise=eps)
                result = protocol.recover(Y, h, layout, N0=N0)
                ridge = N0 / (64 / layout.active_per_user) + 1e-6
                geometry = protocol.tensor_geometry(layout, self.device)
                expected, expected_rel = torch.zeros_like(truth), torch.zeros_like(truth.real)
                for user in range(4):
                    power = h[:, user, user, geometry["physical_supports"][user]].abs().square()
                    gain = power / (power + ridge)
                    expected[:, user, geometry["active_slots"]] = truth[:, user, geometry["active_slots"]] * gain
                    expected_rel[:, user, geometry["active_slots"]] = gain
                torch.testing.assert_close(result["z_hat"], expected, rtol=3e-6, atol=8e-7)
                torch.testing.assert_close(result["reliability"], expected_rel, rtol=2e-6, atol=3e-7)
                self.assertAlmostEqual(result["lambda_rx"].item(), ridge, places=7)
                self.assertTrue(bool((result["z_hat"][:, ~geometry["active_mask"]] == 0).all()))

    def test_full_grid_noise_is_absolute_single_and_common_across_layouts(self):
        torch = self.torch
        eps = protocol.standard_noise(20261210, "test", 3, 2, self.device)
        N0 = protocol.noise_variance(5)
        expected_noise = math.sqrt(N0 / 2) * eps
        for name in ("all4", "group2", "fdma"):
            layout = protocol.LayoutSpec(name)
            X, _, _ = protocol.normalize_and_map(self.raw(), layout)
            h = self.channel()
            zero_h = torch.zeros_like(h)
            Y_noise = protocol.transmit(zero_h, X, layout, N0=N0, standard_noise=eps)
            self.assertTrue(torch.equal(Y_noise, expected_noise))
            Y = protocol.transmit(h, X, layout, N0=N0, standard_noise=eps)
            clean = (h * X[:, :, None]).sum(1)
            self.assertTrue(torch.equal(Y, clean + expected_noise))
            # Scaling received signal cannot rescale the absolute noise.
            self.assertTrue(torch.equal(protocol.transmit(h * 10, X, layout, N0=N0, standard_noise=eps),
                                        (h * 10 * X[:, :, None]).sum(1) + expected_noise))

    def test_private_noise_replay_ignores_global_rng_and_preserves_it(self):
        torch = self.torch
        before = torch.get_rng_state().clone()
        first = protocol.standard_noise(20261030, "train", 501, 2, self.device)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.rand(19)
        after_unrelated = torch.get_rng_state().clone()
        second = protocol.standard_noise(20261030, "train", 501, 2, self.device)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(after_unrelated, torch.get_rng_state()))
        self.assertFalse(torch.equal(first, protocol.standard_noise(20261030, "validation", 501, 2, self.device)))
        self.assertFalse(torch.equal(first, protocol.standard_noise(20261030, "train", 502, 2, self.device)))
        self.assertAlmostEqual(first.real.var().item(), 1.0, delta=0.05)
        self.assertAlmostEqual(first.imag.var().item(), 1.0, delta=0.05)
        torch.set_rng_state(before)

    def test_physical_input_and_public_noise_errors_are_rejected(self):
        torch = self.torch
        layout = protocol.LayoutSpec("fdma")
        X, _, _ = protocol.normalize_and_map(self.raw(1), layout)
        h = self.channel(1)
        eps = protocol.standard_noise(2, "preflight", 0, 1, self.device)
        bad = X.clone()
        bad[0, 0, 1] = 1
        with self.assertRaises(ValueError):
            protocol.transmit(h, bad, layout, N0=0.25, standard_noise=eps)
        with self.assertRaises(ValueError):
            protocol.transmit(h, X, layout, N0=0.25, standard_noise=eps.requires_grad_())
        for value in (0, float("nan"), 1, -0.1, True):
            with self.assertRaises((TypeError, ValueError)):
                protocol.recover(torch.zeros_like(eps), h, layout, N0=value)


if __name__ == "__main__":
    unittest.main()
