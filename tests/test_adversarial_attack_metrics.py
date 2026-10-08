import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


class IdentityLogitModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.shape[0], -1)[:, :1]


class MeanLogitModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs.mean(dim=(1, 2, 3)).unsqueeze(1) - 0.1


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


def test_trigger_learning_saves_periodic_checkpoint(tmp_path):
    inputs = torch.zeros(2, 1, 2, 2)
    targets = torch.zeros(2, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    attack = AdversarialAttack(
        model=IdentityLogitModel(), device='cpu', use_multi_gpu=False
    )
    checkpoint_path = tmp_path / 'saved_trigger_checkpoint'

    attack.learn_universal_trigger(
        data_loader=loader,
        trigger_box={'x': 0, 'y': 0, 'width': 2, 'height': 2},
        validation_loader=None,
        steps=2,
        learning_rate=0.01,
        optimize_mask=False,
        patch_update_method='adam',
        epsilon=0.03,
        log_interval=0,
        trigger_preview_interval=0,
        checkpoint_interval=1,
        checkpoint_path=checkpoint_path,
        progressive_resize=False,
        randomize_training_location=False,
    )

    checkpoint = attack.load_trigger(checkpoint_path)
    assert checkpoint['selection'] == 'latest_checkpoint'
    assert checkpoint['selected_step'] == 2
    assert len(checkpoint['history']) == 2
    assert checkpoint['patch_update_method'] == 'adam'


def test_universal_trigger_uses_resized_binary_spatial_mask(tmp_path):
    mask_path = tmp_path / 'mask.png'
    # Exercise 0/1 encoding: the right half permits perturbation, the left half is protected.
    Image.fromarray(torch.tensor([[0, 1]], dtype=torch.uint8).numpy()).save(mask_path)
    images = torch.zeros(1, 3, 2, 4)
    loader = DataLoader(
        TensorDataset(images, torch.zeros(1, 1)),
        batch_size=1,
    )
    attack = AdversarialAttack(MeanLogitModel(), device='cpu', use_multi_gpu=False)

    result = attack.learn_universal_trigger(
        data_loader=loader,
        trigger_box={'x': 0, 'y': 0, 'width': 4, 'height': 2},
        validation_loader=None,
        steps=1,
        learning_rate=0.1,
        optimize_mask=False,
        patch_update_method='adam',
        epsilon=0.5,
        log_interval=0,
        trigger_preview_interval=0,
        trigger_preview_dir=tmp_path,
        progressive_resize=False,
        randomize_training_location=False,
        perturbation_mask_path=mask_path,
    )

    assert result['mask'].shape == (1, 3, 2, 4)
    assert torch.count_nonzero(result['mask'][:, :, :, :2]) == 0
    assert torch.all(result['mask'][:, :, :, 2:] == 1)
    poisoned = attack._inject_trigger(
        images,
        result['trigger_boxes'],
        trigger_patch=result['patch'],
        trigger_mask=result['mask'],
        edge_softness=0.0,
        how_to_attach='blend',
    )
    assert torch.equal(poisoned[:, :, :, :2], images[:, :, :, :2])
    assert result['perturbation_mask']['perturbation_mask_source_size'] == [2, 1]
    assert result['perturbation_mask']['perturbation_mask_resolved_size'] == [4, 2]
    assert (tmp_path / 'resolved_perturbation_mask.png').exists()
