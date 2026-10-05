import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


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
        patch_update_method='pgd_sign',
        output_dir=tmp_path,
        split_name='validation',
        log_interval=0,
    )

    assert summary['samples_seen'] == 2
    assert summary['eligible_samples'] == 1
    assert summary['successful_attacks'] == 1

    manifest_path = tmp_path / 'validation' / 'manifest.jsonl'
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]['artifact'] != records[1]['artifact']
    assert records[1]['status'] == 'skipped'

    attacked = torch.load(tmp_path / 'validation' / records[0]['artifact'], map_location='cpu')
    skipped = torch.load(tmp_path / 'validation' / records[1]['artifact'], map_location='cpu')
    assert attacked['linf'] <= 0.5 + 1e-6
    assert not torch.equal(attacked['adversarial_image'], attacked['original_image'])
    assert torch.equal(skipped['adversarial_image'], skipped['original_image'])


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
