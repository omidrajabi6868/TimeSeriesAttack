import pytest
import torch
from torch import nn

from Attacks.ImageAttacks.Aggregation import aggregate, register_aggregator
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack
from imageattack import _default_output_dir, build_parser


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


def test_default_output_directories_separate_single_and_multi_model_runs():
    parser = build_parser()
    single_args = parser.parse_args(['--no-multimodel-optimization', '--model-name', 'AlexNet'])
    multi_args = parser.parse_args([
        '--multimodel-optimization',
        '--ensemble-models', 'AlexNet', 'ResNet34',
        '--aggregation', 'max',
    ])

    single_path = _default_output_dir(single_args)
    multi_path = _default_output_dir(multi_args)
    assert single_path.startswith('backups/single_model/AlexNet/')
    assert multi_path.startswith('backups/multi_model/AlexNet__ResNet34/max/')
    assert single_path != multi_path


def test_multimodel_optimization_is_a_boolean_flag():
    parser = build_parser()

    assert parser.parse_args([]).multimodel_optimization is False
    assert parser.parse_args(['--multimodel-optimization']).multimodel_optimization is True
    assert parser.parse_args(['--no-multimodel-optimization']).multimodel_optimization is False


def test_ensemble_metadata_records_reproducibility_settings():
    attack = AdversarialAttack(
        {'small': ScaleModel(1), 'large': ScaleModel(2)},
        device='cpu',
        aggregation='weighted_mean',
        model_weights=[1, 2],
    )
    assert attack._ensemble_metadata() == {
        'multi_model': True,
        'model_names': ['small', 'large'],
        'aggregation': 'weighted_mean',
        'model_weights': [1, 2],
        'gpu_ids': None,
    }
