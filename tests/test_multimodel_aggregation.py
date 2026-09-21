import pytest
import torch
from torch import nn

from Attacks.ImageAttacks.Aggregation import aggregate, register_aggregator
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


class ScaleModel(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)

    def forward(self, inputs):
        return inputs.flatten(1).mean(dim=1, keepdim=True) * self.scale


def test_builtin_aggregation_strategies():
    values = [torch.tensor(1.0), torch.tensor(3.0)]
    assert aggregate(values, 'mean').item() == 2.0
    assert aggregate(values, 'sum').item() == 4.0
    assert aggregate(values, 'max').item() == 3.0
    assert aggregate(values, 'min').item() == 1.0
    assert aggregate(values, 'weighted_mean', [3, 1]).item() == 1.5


def test_custom_aggregation_strategy_can_be_registered():
    register_aggregator('test_first', lambda values, weights: values[0], replace=True)
    assert aggregate([torch.tensor(2.0), torch.tensor(9.0)], 'test_first').item() == 2.0


def test_multimodel_loss_backpropagates_to_shared_input():
    attack = AdversarialAttack(
        {'small': ScaleModel(1), 'large': ScaleModel(3)},
        device='cpu',
        aggregation='weighted_mean',
        model_weights=[1, 3],
    )
    attack._build_cost_function('classification')
    inputs = torch.ones(2, 1, 2, 2, requires_grad=True)
    loss, outputs = attack._classification_loss(inputs, target_label=0)
    loss.backward()

    assert outputs.shape == (2, 1)
    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad).item() > 0


def test_model_weights_must_match_ensemble():
    with pytest.raises(ValueError, match='number of model weights'):
        AdversarialAttack(
            [ScaleModel(1), ScaleModel(2)],
            device='cpu',
            model_weights=[1],
        )
