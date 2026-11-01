import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from Attacks.ImageAttacks.GradCAM import compute_gradcam
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


class TinySpatialBinaryModel(torch.nn.Module):
    def __init__(self, bias=-0.2):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 3.0)

    def forward(self, inputs):
        return self.pool(self.conv(inputs)).flatten(1) + self.bias


def test_gradcam_supports_one_logit_binary_predictions():
    model = TinySpatialBinaryModel()
    result = compute_gradcam(model, torch.full((1, 3, 4, 6), 0.1))

    assert result.heatmap.shape == (4, 6)
    assert result.prediction == 0
    assert result.target_class == 0
    assert result.target_layer == 'conv'
    assert torch.isfinite(result.heatmap).all()


def test_universal_gradcam_saves_clean_and_adversarial_pairs(tmp_path):
    model = TinySpatialBinaryModel()
    attack = AdversarialAttack(model, device='cpu', use_multi_gpu=False)
    images = torch.full((1, 3, 4, 6), 0.1)
    loader = DataLoader(TensorDataset(images, torch.tensor([0.0])), batch_size=1)

    summary = attack.save_universal_gradcam_visualizations(
        loader,
        tmp_path,
        trigger_box={'x': 0, 'y': 0, 'width': 6, 'height': 4},
        trigger_patch=torch.full((1, 3, 4, 6), 0.3),
        trigger_mask=torch.ones(1, 3, 4, 6),
        target_label=1.0,
        source_filter='bad',
        edge_softness=0.0,
        max_examples=1,
        models={
            'TinyModel': model,
            'UnsuccessfulBlackBox': TinySpatialBinaryModel(bias=-1.0),
        },
    )

    assert summary['visualizations_saved'] == 1
    record = summary['records'][0]
    assert record['clean_prediction'] == 0
    assert record['adversarial_prediction'] == 1
    assert record['layout'] == 'clean|clean_gradcam|adversarial|adversarial_gradcam'
    assert summary['records'][1]['model'] == 'UnsuccessfulBlackBox'
    assert summary['records'][1]['status'] == 'skipped_unsuccessful'
    for key in ('clean_gradcam', 'adversarial_gradcam', 'comparison'):
        assert Path(record[key]).exists()
    assert (tmp_path / 'gradcam_summary.json').exists()


def test_universal_gradcam_does_not_fall_back_to_unsuccessful_samples(tmp_path):
    model = TinySpatialBinaryModel()
    attack = AdversarialAttack(model, device='cpu', use_multi_gpu=False)
    images = torch.full((1, 3, 4, 6), 0.1)
    loader = DataLoader(TensorDataset(images, torch.tensor([0.0])), batch_size=1)

    summary = attack.save_universal_gradcam_visualizations(
        loader,
        tmp_path,
        trigger_box={'x': 0, 'y': 0, 'width': 6, 'height': 4},
        trigger_patch=torch.zeros(1, 3, 4, 6),
        target_label=1.0,
        source_filter='bad',
        max_examples=1,
        models={'TinyModel': model},
    )

    assert summary['examples_requested'] == 0
    assert summary['visualizations_saved'] == 0
    assert summary['records'] == []


def test_image_specific_gradcam_reuses_saved_attack_artifacts(tmp_path):
    model = TinySpatialBinaryModel()
    attack = AdversarialAttack(model, device='cpu', use_multi_gpu=False)
    split_dir = tmp_path / 'test'
    artifact_dir = split_dir / 'artifacts'
    artifact_dir.mkdir(parents=True)
    clean = torch.full((1, 3, 4, 6), 0.1)
    adversarial = torch.full((1, 3, 4, 6), 0.4)
    torch.save({
        'original_image': clean,
        'adversarial_image': adversarial,
        'true_label': 0.0,
        'target_label': 1.0,
        'success': True,
    }, artifact_dir / '00000000.pt')
    manifest = {
        'sample_id': 'sample-0',
        'eligible': True,
        'success': True,
        'artifact': 'artifacts/00000000.pt',
    }
    (split_dir / 'manifest.jsonl').write_text(json.dumps(manifest) + '\n')

    summary = attack.save_image_specific_gradcam_visualizations(
        tmp_path,
        models={'TinyModel': model},
        split_name='test',
        max_examples=1,
    )

    assert summary['visualizations_saved'] == 1
    assert summary['records'][0]['sample_id'] == 'sample-0'
    assert (split_dir / 'gradcam' / 'TinyModel').is_dir()
    assert (split_dir / 'gradcam' / 'gradcam_summary.json').exists()
