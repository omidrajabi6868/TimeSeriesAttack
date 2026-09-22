import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Attacks.ImageAttacks.Aggregation import aggregate, register_aggregator
import Attacks.ImageAttacks.ImageAdversarialAttack as attack_module
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack
from imageattack import _default_output_dir, build_parser


class ScaleModel(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)

    def forward(self, inputs):
        return inputs.flatten(1).mean(dim=1, keepdim=True) * self.scale


class TinyConvModel(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.head = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.conv.weight, float(scale))
        nn.init.constant_(self.head.weight, 0.5)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        features = torch.relu(self.conv(inputs)).mean(dim=(2, 3))
        return self.head(features)


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


def _tiny_image_loader():
    inputs = torch.zeros(2, 3, 8, 8)
    targets = torch.zeros(2, 1)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def test_multimodel_psp_uap_aggregates_each_models_feature_loss():
    models = {'first': TinyConvModel(0.5), 'second': TinyConvModel(1.0)}
    attack = AdversarialAttack(
        models,
        device='cpu',
        use_multi_gpu=False,
        aggregation='weighted_mean',
        model_weights=[1, 2],
    )
    loader = _tiny_image_loader()

    result = attack.learn_universal_trigger(
        data_loader=loader,
        trigger_box={'x': 0, 'y': 0, 'width': 8, 'height': 8},
        validation_loader=loader,
        steps=1,
        learning_rate=0.01,
        optimize_mask=False,
        patch_update_method='psp_uap',
        epsilon=0.03,
        log_interval=0,
        trigger_preview_interval=0,
        progressive_resize=False,
        randomize_training_location=False,
        psp_num_copies=2,
    )

    assert result['ensemble']['multi_model'] is True
    assert result['ensemble']['model_names'] == ['first', 'second']
    assert result['ensemble']['aggregation'] == 'weighted_mean'
    assert all(model.forward_calls >= 2 for model in models.values())


def test_multimodel_robust_uap_aggregates_each_models_classification_loss(monkeypatch):
    class TinyRobustConfig:
        def __init__(self, alpha=0.01):
            self.psi = 0.2
            self.phi = 0.2
            self.gamma = 0.7
            self.zeta = 0.8
            self.alpha = alpha
            self.max_inner_steps = 1
            self.max_batch_size = 2
            self.norm = 'linf'
            self.num_transform_samples = 1

    monkeypatch.setattr(attack_module, 'RobustUAPConfig', TinyRobustConfig)
    monkeypatch.setattr(attack_module, 'estimate_robustness', lambda **kwargs: 0.0)

    models = {'first': TinyConvModel(0.5), 'second': TinyConvModel(1.0)}
    attack = AdversarialAttack(
        models,
        device='cpu',
        use_multi_gpu=False,
        aggregation='mean',
    )
    loader = _tiny_image_loader()

    result = attack.learn_universal_trigger(
        data_loader=loader,
        trigger_box={'x': 0, 'y': 0, 'width': 8, 'height': 8},
        validation_loader=loader,
        steps=1,
        learning_rate=0.01,
        optimize_mask=False,
        patch_update_method='robust_uap',
        epsilon=0.03,
        log_interval=0,
        trigger_preview_interval=0,
        progressive_resize=False,
        randomize_training_location=False,
    )

    assert result['ensemble']['multi_model'] is True
    assert result['ensemble']['model_names'] == ['first', 'second']
    assert result['ensemble']['aggregation'] == 'mean'
    assert all(model.forward_calls > 0 for model in models.values())
