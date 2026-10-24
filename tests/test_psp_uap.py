import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Attacks.ImageAttacks.CostFunction import FeatureExtractor, PSPUAPObjective
from Attacks.ImageAttacks.PSPUAP import PSPTransformSampler


class PSPUAPTests(unittest.TestCase):
    def test_inception_profile_uses_published_layer_fraction_and_temperature(self):
        from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack

        class TinyInceptionV3(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 2, 1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(2, 1)

            def forward(self, inputs):
                return self.fc(self.pool(self.conv(inputs)).flatten(1))

        attack = AdversarialAttack(TinyInceptionV3(), device='cpu', use_multi_gpu=False)
        attack._build_cost_function('psp_uap')

        # The extractor now hooks only the selected first 20%, so the
        # objective consumes all already-filtered activations.
        self.assertEqual(attack.cost_function.p_rate, 1.0)
        self.assertEqual(attack.cost_function.temperature, 3.0)
        self.assertEqual(attack.feature_profiles['model']['p_rate'], 0.2)
        self.assertEqual(attack.feature_profiles['model']['selected_count'], 1)
        attack._remove_feature_extractor()

    def test_attack_builds_activation_maximizing_psp_objective(self):
        from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack

        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(144, 1))
        attack = AdversarialAttack(model, device='cpu', use_multi_gpu=False)

        attack._build_cost_function('psp_uap')

        self.assertTrue(attack.cost_function.maximize_activations)
        attack._remove_feature_extractor()

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

    def test_targeted_psp_mode_records_hybrid_objective_and_keeps_image_size(self):
        from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack

        class TinyBinaryModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 2, 1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(2, 1)

            def forward(self, inputs):
                return self.fc(self.pool(self.conv(inputs)).flatten(1))

        inputs = torch.zeros(2, 3, 8, 12)
        targets = torch.zeros(2, 1)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        attack = AdversarialAttack(TinyBinaryModel(), device='cpu', use_multi_gpu=False)

        result = attack.learn_universal_trigger(
            data_loader=loader,
            validation_loader=loader,
            trigger_box={'x': 0, 'y': 0, 'width': 12, 'height': 8},
            patch_update_method='targeted_psp_uap',
            steps=1,
            learning_rate=0.01,
            epsilon=0.03,
            optimize_mask=False,
            progressive_resize=False,
            randomize_training_location=False,
            log_interval=0,
            trigger_preview_interval=0,
            psp_num_copies=2,
        )

        self.assertEqual(result['patch_update_method'], 'targeted_psp_uap')
        self.assertEqual(result['psp_uap']['training_mode'], 'targeted_hybrid')
        self.assertEqual(result['psp_uap']['image_size'], [12, 8])
        self.assertGreater(result['history'][0]['targeted_training_samples'], 0)
        self.assertGreater(result['history'][0]['targeted_loss'], 0)


if __name__ == "__main__":
    unittest.main()
