import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack, TransformSampler


class MeanThresholdModel(torch.nn.Module):
    def __init__(self, bias=-0.2):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))

    def forward(self, inputs):
        return inputs.mean(dim=(1, 2, 3)).unsqueeze(1) + self.bias


def test_image_specific_generation_saves_independent_artifacts(tmp_path):
    attack = AdversarialAttack(MeanThresholdModel(), device='cpu', use_multi_gpu=False)
    images = torch.zeros(2, 3, 2, 2)
    labels = torch.tensor([0.0, 1.0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    summary = attack.learn_image_specific_trigger(
        loader,
        {'x': 0, 'y': 0, 'width': 2, 'height': 2},
        target_label=1.0,
        source_filter='bad',
        steps=10,
        learning_rate=0.1,
        epsilon=0.5,
        patch_update_method='pgd',
        output_dir=tmp_path,
        split_name='validation',
        log_interval=0,
    )

    assert summary['samples_seen'] == 2
    assert summary['eligible_samples'] == 1
    assert summary['successful_attacks'] == 1
    assert summary['skipped_samples'] == 1
    assert summary['artifacts_saved'] == 1

    manifest_path = tmp_path / 'validation' / 'manifest.jsonl'
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]['artifact'] == 'artifacts/00000000.pt'
    assert records[1]['artifact'] is None
    assert records[1]['status'] == 'skipped'

    attacked = torch.load(tmp_path / 'validation' / records[0]['artifact'], map_location='cpu')
    assert attacked['linf'] <= 0.5 + 1e-6
    assert not torch.equal(attacked['adversarial_image'], attacked['original_image'])
    assert len(list((tmp_path / 'validation' / 'artifacts').glob('*.pt'))) == 1

    metrics = attack.evaluate_image_specific_artifacts(
        tmp_path, {'source': attack.model}, split_name='validation'
    )
    assert metrics['models']['source']['eligible_samples'] == 1


def test_transfer_evaluation_reuses_saved_adversarial_images(tmp_path):
    source = MeanThresholdModel(-0.2)
    attack = AdversarialAttack(source, device='cpu', use_multi_gpu=False)
    loader = DataLoader(
        TensorDataset(torch.zeros(1, 3, 2, 2), torch.tensor([0.0])),
        batch_size=1,
    )
    attack.learn_image_specific_trigger(
        loader,
        {'x': 0, 'y': 0, 'width': 2, 'height': 2},
        steps=10,
        learning_rate=0.1,
        epsilon=0.5,
        output_dir=tmp_path,
        split_name='test',
        log_interval=0,
    )

    metrics = attack.evaluate_image_specific_artifacts(
        tmp_path,
        {'source': source, 'black_box': MeanThresholdModel(-0.4)},
        split_name='test',
    )

    assert metrics['models']['source']['conditional_transfer_asr'] == 100.0
    assert metrics['models']['black_box']['eligible_samples'] == 1
    assert (tmp_path / 'test' / 'transfer_summary.json').exists()


@pytest.mark.parametrize('method', ['fgsm', 'ifgsm', 'mi_fgsm', 'pgd', 'adam', 'deepfool', 'hp'])
def test_supported_image_specific_method_names(method):
    attack = AdversarialAttack(MeanThresholdModel(), device='cpu', use_multi_gpu=False)
    result = attack.optimize_image_specific_trigger(
        torch.zeros(1, 3, 2, 2),
        torch.tensor(0.0),
        {'x': 0, 'y': 0, 'width': 2, 'height': 2},
        steps=10,
        learning_rate=0.1,
        epsilon=0.5,
        bandwidth=0,
        patch_update_method=method,
    )

    assert result['attack_method'] == method
    assert result['linf'] <= 0.5 + 1e-6
    assert result['success']


@pytest.mark.parametrize(
    'method',
    ['deepfool_uap', 'gd_uap', 'gap_uap', 'hp_uap', 'fg_uap', 'robust_uap', 'psp_uap'],
)
def test_universal_method_names_are_rejected_for_image_specific_attacks(method):
    attack = AdversarialAttack(MeanThresholdModel(), device='cpu', use_multi_gpu=False)
    with pytest.raises(ValueError, match='Universal-only methods'):
        attack.optimize_image_specific_trigger(
            torch.zeros(1, 3, 2, 2),
            torch.tensor(0.0),
            {'x': 0, 'y': 0, 'width': 2, 'height': 2},
            patch_update_method=method,
        )


def test_robust_image_specific_attack_uses_per_image_eot(monkeypatch):
    monkeypatch.setattr(
        TransformSampler,
        'sample',
        lambda self, count: [lambda inputs: inputs for _ in range(count)],
    )
    attack = AdversarialAttack(MeanThresholdModel(), device='cpu', use_multi_gpu=False)
    result = attack.optimize_image_specific_trigger(
        torch.zeros(1, 3, 2, 2),
        torch.tensor(0.0),
        {'x': 0, 'y': 0, 'width': 2, 'height': 2},
        steps=10,
        learning_rate=0.1,
        epsilon=0.5,
        eot_samples=3,
        patch_update_method='robust',
    )

    assert result['attack_method'] == 'robust'
    assert result['success']
