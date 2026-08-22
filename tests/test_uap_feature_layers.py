import pytest
import torch
from torch import nn
from torchvision import models

from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


@pytest.mark.parametrize(
    ("builder", "classifier_name"),
    [
        (models.resnet18, "fc"),
        (models.resnet34, "fc"),
        (models.resnet50, "fc"),
        (models.resnet101, "fc"),
        (models.alexnet, "classifier.6"),
        (models.mobilenet_v3_small, "classifier.3"),
        (models.efficientnet_b0, "classifier.1"),
        (models.swin_t, "head"),
    ],
)
def test_feature_uap_layer_selection_supports_every_image_model(builder, classifier_name):
    model = builder(weights=None)
    attack = AdversarialAttack(model, device="cpu", use_multi_gpu=False)

    attack._build_cost_function("fg_uap")

    assert attack.feature_extractor.layer_names
    assert len(attack.feature_extractor.layer_names) <= 4
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
        (models.swin_t, "head", nn.Linear),
    ],
)
def test_gd_uap_layer_selection_supports_every_image_model(
    builder, classifier_name, expected_layer_type
):
    model = builder(weights=None)
    attack = AdversarialAttack(model, device="cpu", use_multi_gpu=False)

    attack._build_cost_function("gd_uap")

    assert attack.feature_extractor.layer_names
    assert classifier_name not in attack.feature_extractor.layer_names
    selected_modules = dict(model.named_modules())
    assert all(
        isinstance(selected_modules[name], expected_layer_type)
        for name in attack.feature_extractor.layer_names
    )
    attack._remove_feature_extractor()


def test_feature_layer_names_work_through_data_parallel_wrapper():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.Linear(16, 1))
    attack = AdversarialAttack(
        torch.nn.DataParallel(model), device="cpu", use_multi_gpu=False
    )

    attack._build_cost_function("fg_uap")

    assert attack.feature_extractor.layer_names == ["module.0"]
    attack._remove_feature_extractor()
