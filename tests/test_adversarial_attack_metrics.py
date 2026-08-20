import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


class IdentityLogitModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.shape[0], -1)[:, :1]


@pytest.mark.parametrize(
    ('stored_softness', 'expected'),
    [
        (0.15, 0.15),
        ({'initial_edge_softness': 0.2, 'final_edge_softness': 0.1}, 0.1),
        ({
            'initial_edge_softness': 0.2,
            'final_edge_softness': 0.1,
            'selected_edge_softness': 0.05,
        }, 0.05),
        ({'selected_edge_softness': None, 'final_edge_softness': 0.1}, 0.1),
        ({}, 0.0),
        (None, 0.0),
    ],
)
def test_normalize_edge_softness_supports_saved_trigger_metadata(stored_softness, expected):
    assert AdversarialAttack._normalize_edge_softness(stored_softness) == expected


def test_attack_metrics_use_source_class_but_classification_uses_full_loader():
    inputs = torch.tensor([-1.0, -1.0, 1.0, -1.0]).view(-1, 1, 1, 1)
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0]).view(-1, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    attack = AdversarialAttack(
        model=IdentityLogitModel(), device='cpu', use_multi_gpu=False
    )
    attack._inject_trigger = lambda selected_inputs, *args, **kwargs: torch.ones_like(
        selected_inputs
    )

    metrics = attack.evaluate_attack_success(
        test_loader=loader,
        trigger_box=[],
        target_label=1.0,
        source_filter='bad',
    )

    assert metrics['samples_evaluated'] == 4
    assert metrics['attacked_samples_evaluated'] == 2
    assert metrics['attack_success_rate'] == 100.0
    assert metrics['prediction_change_rate'] == 100.0
    assert metrics['before_attack_metrics']['accuracy'] == 75.0
    assert metrics['before_attack_metrics']['recall'] == 50.0
    assert metrics['after_attack_metrics']['accuracy'] == 75.0
    assert metrics['after_attack_metrics']['precision'] == pytest.approx(2 / 3 * 100)
    assert metrics['after_attack_metrics']['recall'] == 100.0
    assert metrics['before_attack_metrics']['samples'] == 4
    assert metrics['after_attack_metrics']['samples'] == 4
    assert metrics['classification_metrics_scope'] == 'all'
    assert metrics['attack_metrics_scope'] == 'bad'
