"""Metadata and tiny actual optimizer-path tests; no full codec or CDL."""
import copy
import io
import random
import unittest

from configs import twc_v1_k8_icc_aligned_config as cfg
import train_twc_v1_k8_icc_aligned as runtime

try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


class BoundaryMetadata(unittest.TestCase):
    @staticmethod
    def payload(component="fdma", step=100):
        # Schema fixture only. Actual tensor/optimizer roundtrip is tested with
        # real small CPU objects below, never represented as server evidence.
        return dict(format=cfg.NAMESPACE, component=component, step=step,
            recipe=cfg.recipe(component), fingerprint=cfg.fingerprint(component),
            core={"schema": "test", "sha256": {"core": "a" * 64}},
            model={"synthetic_schema_only": 1}, optimizer={"synthetic_schema_only": 1},
            scheduler={"last_epoch": step},
            rng={"python": [], "numpy": [], "torch_cpu": [], "torch_cuda": []},
            counters=cfg.counters(component, step), data_protocol={"new_namespace": cfg.NAMESPACE},
            ce_identity=None if component == "ce" else {"checkpoint_sha256": "b" * 64},
            statistics=None if component == "ce" else {"synthetic_schema_only": 1},
            training_code={"commit": "c" * 40})

    def validate(self, payload):
        return cfg.validate_resume(payload, payload["component"], core=self.payload(payload["component"])["core"],
            data_protocol={"new_namespace": cfg.NAMESPACE},
            ce_identity=None if payload["component"] == "ce" else {"checkpoint_sha256": "b" * 64})

    def test_actual_budget_micro_address_and_split_accounting(self):
        self.assertEqual(cfg.counters("ce", 5000)["MU_frames"], 160000)
        for arm in cfg.ARMS:
            counts = cfg.counters(arm, 20000)
            self.assertEqual(counts["MU_frames"], 640000)
            self.assertEqual(counts["UE_inputs"], 5120000)
            self.assertEqual(counts["microbatches"], 160000)
            self.assertEqual(counts["EMA_commits"], 20000)
            self.assertEqual(cfg.micro_index(101, 0), 801)
            self.assertEqual(cfg.micro_index(20000, 7), 160000)
        self.assertNotEqual(cfg.noise_seed("ce_train", 1), cfg.noise_seed("codec_train", 1))
        self.assertNotEqual(cfg.noise_seed("monitor", 1), cfg.noise_seed("final", 1))
        self.assertNotEqual(cfg.noise_seed("final", 1, "noise"), cfg.noise_seed("final", 1, "gamma"))
        # No arm appears in the random address; replay gives the two arms the
        # same actual channels/standard noise, rather than only a shared label.
        self.assertEqual(cfg.noise_seed("codec_train", 800), cfg.noise_seed("codec_train", 800))

    def test_actual_resume_validator_rejects_identity_and_partial_boundaries(self):
        for component in ("ce",) + cfg.ARMS:
            self.assertEqual(self.validate(self.payload(component)), 100)
        for field in ("fingerprint", "core", "data_protocol", "ce_identity", "recipe"):
            value = self.payload()
            value[field] = "changed"
            with self.assertRaisesRegex(ValueError, "identity changed"):
                self.validate(value)
        for field, replacement in (("step", 101), ("step", True),
                ("counters", {}), ("scheduler", {"last_epoch": 99}),
                ("rng", {"torch_cpu": []}), ("statistics", None)):
            value = self.payload()
            value[field] = replacement
            with self.assertRaises(ValueError):
                self.validate(value)
        for missing in ("optimizer", "statistics", "rng", "model"):
            value = self.payload(); del value[missing]
            with self.assertRaises(ValueError):
                self.validate(value)


@unittest.skipIf(torch is None or np is None, "Real PyTorch/NumPy unavailable; no fake dependency")
class TinyProductionPath(unittest.TestCase):
    def setUp(self):
        self.external = runtime.capture_rng()

    def tearDown(self):
        runtime.restore_rng(self.external)

    def optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY, betas=(.9, .999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        return optimizer, scheduler

    def test_eight_micro_losses_one_actual_update_and_causal_commit(self):
        torch.manual_seed(370)
        model = torch.nn.Linear(2, 1)
        reference = copy.deepcopy(model)
        optimizer, scheduler = self.optimizer(model)
        ref_optimizer, ref_scheduler = self.optimizer(reference)
        phy = runtime.physics()
        stats = phy.ReceiverStatistics(phy.Layout("fdma"), torch.zeros((6, 8, 256)))
        old_prior = stats.nominal_power().clone()
        x = torch.randn((32, 2)); target = torch.randn((32, 1))
        power_sum = torch.zeros((8, 24))
        for micro in range(8):
            loss = (model(x[micro * 4:(micro + 1) * 4]) - target[micro * 4:(micro + 1) * 4]).square().mean()
            (loss / 8).backward()
            raw = torch.complex(torch.randn((4, 8, 216)), torch.randn((4, 8, 216)))
            power_sum += phy.build_waveform(raw, stats.layout).active.detach().abs().square().sum(0)
            self.assertTrue(torch.equal(stats.nominal_power(), old_prior))
            self.assertEqual(int(stats.ema_updates), 0)
            self.assertEqual(len(optimizer.state), 0)
        (reference(x) - target).square().mean().backward()
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-6, atol=1e-7)
        norm = runtime.commit_update(model, optimizer, scheduler, stats=stats,
            power_sum=power_sum, frames=32, previous_updates=0)
        runtime.commit_update(reference, ref_optimizer, ref_scheduler, previous_updates=0)
        self.assertTrue(bool(torch.isfinite(norm)) and float(norm) > 0)
        self.assertEqual(int(stats.ema_updates), 1)
        self.assertEqual(scheduler.last_epoch, 1)
        runtime.optimizer_check(optimizer, scheduler, 1)
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)

    def test_tiny_optimizer_ema_rng_serialization_and_strict_restore(self):
        torch.manual_seed(475)
        phy = runtime.physics(); layout = phy.Layout("fdma")
        model = torch.nn.Linear(2, 1)
        optimizer, scheduler = self.optimizer(model)
        stats = phy.ReceiverStatistics(layout, torch.zeros((6, 8, 256)))
        model(torch.tensor([[1., 2.]])).square().mean().backward()
        runtime.commit_update(model, optimizer, scheduler, stats=stats,
            power_sum=torch.full((8, 24), 64.), frames=32, previous_updates=0)
        blob = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), statistics=stats.state_dict(), rng=runtime.capture_rng()), blob)
        blob.seek(0)
        value = torch.load(blob, map_location="cpu", weights_only=False)
        expected_random = (random.random(), float(np.random.random()), torch.rand(3))
        clone = torch.nn.Linear(2, 1)
        clone_stats = phy.ReceiverStatistics(layout)
        clone_optimizer, clone_scheduler = self.optimizer(clone)
        runtime.strict_load(clone, value["model"])
        runtime.strict_load(clone_stats, value["statistics"])
        clone_optimizer.load_state_dict(value["optimizer"])
        clone_scheduler.load_state_dict(value["scheduler"])
        runtime.optimizer_check(clone_optimizer, clone_scheduler, 1)
        runtime.restore_rng(value["rng"])
        actual_random = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(actual_random[:2], expected_random[:2])
        self.assertTrue(torch.equal(actual_random[2], expected_random[2]))
        self.assertEqual(runtime.state_digest(clone), runtime.state_digest(model))
        self.assertEqual(runtime.state_digest(clone_stats), runtime.state_digest(stats))
        wrong = dict(value["model"]); wrong["weight"] = wrong["weight"].double()
        with self.assertRaisesRegex(ValueError, "dtype"):
            runtime.strict_load(clone, wrong)

    def test_nonfinite_gradient_stops_before_optimizer_and_prior(self):
        model = torch.nn.Linear(2, 1)
        optimizer, scheduler = self.optimizer(model)
        phy = runtime.physics()
        stats = phy.ReceiverStatistics(phy.Layout("fdma"), torch.zeros((6, 8, 256)))
        initial = runtime.state_digest(model)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaises(FloatingPointError):
            runtime.commit_update(model, optimizer, scheduler, stats=stats,
                power_sum=torch.full((8, 24), 64.), frames=32, previous_updates=0)
        self.assertEqual(runtime.state_digest(model), initial)
        self.assertEqual(int(stats.ema_updates), 0)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
