from unittest.mock import MagicMock, patch

import pytest
import torch

from Network import ClassificationModels
from Tasks.ImageClassification import ClassificationBase


@pytest.mark.parametrize(
    ('wrapper', 'builder_name', 'weights', 'classifier_attribute', 'classifier_index'),
    [
        (
            ClassificationModels.MobileNetV3Small,
            'mobilenet_v3_small',
            ClassificationModels.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
            'classifier',
            3,
        ),
        (
            ClassificationModels.EfficientNetB0,
            'efficientnet_b0',
            ClassificationModels.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            'classifier',
            1,
        ),
        (
            ClassificationModels.SwinT,
            'swin_t',
            ClassificationModels.models.Swin_T_Weights.IMAGENET1K_V1,
            'head',
            None,
        ),
    ],
)
def test_new_models_use_imagenet_weights_and_replace_classifier(
    wrapper, builder_name, weights, classifier_attribute, classifier_index
):
    model = MagicMock()
    if classifier_index is None:
        model.head.in_features = 32
    else:
        classifier = [MagicMock() for _ in range(classifier_index + 1)]
        classifier[classifier_index].in_features = 32
        model.classifier = classifier

    with patch.object(ClassificationModels.models, builder_name, return_value=model) as builder:
        wrapped = wrapper(num_classes=1)

    builder.assert_called_once_with(weights=weights)
    output_layer = getattr(wrapped.model, classifier_attribute)
    if classifier_index is not None:
        output_layer = output_layer[classifier_index]
    assert isinstance(output_layer, torch.nn.Linear)
    assert output_layer.in_features == 32
    assert output_layer.out_features == 1


@pytest.mark.parametrize('model_name', ['MobileNetV3Small', 'EfficientNetB0', 'SwinT'])
def test_classification_base_builds_new_binary_models(model_name):
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch(
        f'Network.ClassificationModels.{model_name}',
        return_value=MagicMock(model=mock_model),
    ) as wrapper:
        classification = ClassificationBase(model_name, device='cpu')
        result = classification._build_model()

    wrapper.assert_called_once_with(1)
    assert result is mock_model
