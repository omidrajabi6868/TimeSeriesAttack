import unittest

import torch
from torch import nn

from Attacks.ImageAttacks.CostFunction import FeatureExtractor, PSPUAPObjective
from Attacks.ImageAttacks.PSPUAP import PSPTransformSampler


class PSPUAPTests(unittest.TestCase):
    def test_objective_supports_both_activation_directions(self):
        minimizing_feature = torch.tensor([[[[2.0]]]], requires_grad=True)
        maximizing_feature = torch.tensor([[[[2.0]]]], requires_grad=True)

        PSPUAPObjective(maximize_activations=False)(
            outputs=[minimizing_feature]
        ).backward()
        PSPUAPObjective(maximize_activations=True)(
            outputs=[maximizing_feature]
        ).backward()

        self.assertGreater(minimizing_feature.grad.item(), 0.0)
        self.assertLess(maximizing_feature.grad.item(), 0.0)

    def test_sampler_uses_requested_copy_count_and_keeps_delta_gradient(self):
        torch.manual_seed(0)
        delta = torch.zeros(1, 3, 16, 16, requires_grad=True)
        sampler = PSPTransformSampler((16, 16), input_transform=False)

        semantic_prior, semantic_delta = sampler.sample(delta, num_copies=3)

        self.assertEqual(semantic_prior.shape, (3, 3, 16, 16))
        self.assertEqual(semantic_delta.shape, (3, 3, 16, 16))
        self.assertFalse(semantic_prior.requires_grad)
        self.assertTrue(semantic_delta.requires_grad)
        semantic_delta.sum().backward()
        self.assertIsNotNone(delta.grad)

    def test_binary_psp_weights_are_finite(self):
        objective = PSPUAPObjective(re_weight=True)
        clean = torch.tensor([[0.2], [-0.3]])
        adversarial = torch.tensor([[0.5], [0.1]])

        weights = objective.compute_psp_weights(clean, adversarial)

        self.assertEqual(weights.shape, (2, 1))
        self.assertTrue(torch.isfinite(weights).all())

    def test_feature_capture_can_be_suspended_and_resumed(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
        extractor = FeatureExtractor(model, n_last_layers=0)
        inputs = torch.randn(2, 3, 8, 8)

        with extractor.suspend_capture():
            model(inputs)
        self.assertEqual(extractor.activations, [])

        model(inputs)
        self.assertEqual(len(extractor.activations), 1)
        extractor.remove()


if __name__ == "__main__":
    unittest.main()
