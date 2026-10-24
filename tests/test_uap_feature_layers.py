import pytest
import torch
from torch import nn
from torchvision import models

from Attacks.ImageAttacks.CostFunction import FeaturBaseObjective
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


@pytest.mark.parametrize(
    ("builder", "classifier_name", "expected_rate"),
    [
        (models.resnet18, "fc", 0.65),
        (models.resnet34, "fc", 0.65),
        (models.resnet50, "fc", 0.70),
        (models.resnet101, "fc", 0.65),
        (models.alexnet, "classifier.6", 1.00),
        (models.mobilenet_v3_small, "classifier.3", 0.90),
        (models.efficientnet_b0, "classifier.1", 0.90),
        (models.inception_v3, "fc", 0.20),
        (models.swin_t, "head", 0.20),
    ],
)
def test_feature_uap_layer_selection_supports_every_image_model(
    builder, classifier_name, expected_rate
):
    model = builder(weights=None)
    attack = AdversarialAttack(model, device="cpu", use_multi_gpu=False)

    attack._build_cost_function("fg_uap")

    profile = attack.feature_profiles['model']
    assert attack.feature_extractor.layer_names
    assert profile['p_rate'] == expected_rate
    assert len(attack.feature_extractor.layer_names) == profile['selected_count']
    assert classifier_name not in attack.feature_extractor.layer_names
    attack._remove_feature_extractor()


@pytest.mark.parametrize(
    ("builder", "classifier_name", "expected_layer_type"),
    [
        (models.resnet18, "fc", nn.Conv2d),
        (models.resnet34, "fc", nn.Conv2d),
        (models.resnet50, "fc", nn.Conv2d),
        (models.resnet101, "fc", nn.Conv2d),
        (models.alexnet, "classifier.6", nn.Conv2d),
        (models.mobilenet_v3_small, "classifier.3", nn.Conv2d),
        (models.efficientnet_b0, "classifier.1", nn.Conv2d),
        (models.inception_v3, "fc", nn.Conv2d),
        (models.swin_t, "head", "SwinTransformerBlock"),
    ],
)
def test_gd_uap_layer_selection_supports_every_image_model(
    builder, classifier_name, expected_layer_type
):
    model = builder(weights=None)
    attack = AdversarialAttack(model, device="cpu", use_multi_gpu=False)

    attack._build_cost_function("gd_uap")

    profile = attack.feature_profiles['model']
    assert attack.feature_extractor.layer_names
    assert len(attack.feature_extractor.layer_names) == profile['selected_count']
    assert classifier_name not in attack.feature_extractor.layer_names
    selected_modules = dict(model.named_modules())
    assert all(
        (
            type(selected_modules[name]).__name__ == expected_layer_type
            if isinstance(expected_layer_type, str)
            else isinstance(selected_modules[name], expected_layer_type)
        )
        for name in attack.feature_extractor.layer_names
    )
    attack._remove_feature_extractor()


def test_gd_uap_maximizes_instead_of_suppressing_activations():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(16, 1))
    attack = AdversarialAttack(model, device='cpu', use_multi_gpu=False)

    attack._build_cost_function('gd_uap')

    assert attack.cost_function.maximize_activations
    attack._remove_feature_extractor()


def test_activation_maximization_has_negative_descent_gradient():
    activation = torch.tensor([[[[2.0]]]], requires_grad=True)

    FeaturBaseObjective(maximize_activations=True)(outputs=[activation]).backward()

    assert activation.grad.item() < 0.0


def test_feature_layer_names_work_through_data_parallel_wrapper():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.Linear(16, 1))
    attack = AdversarialAttack(
        torch.nn.DataParallel(model), device="cpu", use_multi_gpu=False
    )

    attack._build_cost_function("fg_uap")

    assert attack.feature_extractor.layer_names == ["module.0"]
    attack._remove_feature_extractor()
