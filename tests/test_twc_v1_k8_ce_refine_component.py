"""Actual pilot LS and CE U-Net backward on synthetic CPU channels only."""
import unittest

import torch
from models.twc_v1_k8_icc_aligned import K8ICCAlignedPilotUNet
import ce_refine_runtime as runtime
from twc_v1_k8_ce_refine import objective


class ActualPilotCEComponent(unittest.TestCase):
    def test_both_losses_reach_actual_ce_parameters_through_paid_pilot_interface(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1901)
            model = K8ICCAlignedPilotUNet().train()
            channels = torch.view_as_complex(torch.randn(4, 8, 32, 256, 2).contiguous())/2**.5
        gamma, noise, polarity = runtime.batch_random('ce_train', 1, torch.device('cpu'))
        before = runtime.state_digest(model)
        for trajectory in ('nmse', 'geometry'):
            model.zero_grad(set_to_none=True)
            prediction, initial, observation, pilots = runtime.pilot_forward(model,
                {'H_ul': channels}, gamma, noise, polarity)
            self.assertEqual(prediction.shape, channels.shape)
            self.assertEqual(prediction.dtype, torch.complex64)
            self.assertEqual(tuple(observation.shape), (4, 32, 256))
            torch.testing.assert_close(pilots.abs().square().sum(-1), torch.full((4, 8), 16.))
            loss, scores = objective(prediction, channels, trajectory)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters()]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
            self.assertEqual(runtime.state_digest(model), before)
            self.assertIsNone(channels.grad)
            self.assertTrue(all(v.shape == (4,) for v in scores.values()))


if __name__ == '__main__':
    unittest.main()
