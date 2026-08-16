import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torchvision.transforms as transforms


@dataclass(frozen=True)
class RobustUAPConfig:
    """Hyperparameters from the RobustUAP robustness-estimation loop."""

    psi: float = 0.25
    phi: float = 0.1
    gamma: float = 0.7
    zeta: float = 0.8
    alpha: float = 0.01
    max_inner_steps: int = 5
    max_batch_size: int = 128
    norm: str = 'linf'

    @property
    def num_transform_samples(self):
        return max(1, int(math.ceil((1.0 / (2.0 * self.psi ** 2)) * math.log(2.0 / self.phi))))


class TransformSampler:
    def __init__(
        self,
        height,
        width,
        p_brightness=0.5,
        p_contrast=0.5,
        p_affine=0.8,
    ):
        self.height = height
        self.width = width

        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_affine = p_affine

    def sample(self, n):

        sampled = []

        for _ in range(n):

            ops = []

            # -------------------------------
            # Brightness
            # -------------------------------
            if random.random() < self.p_brightness:
                ops.append(
                    transforms.ColorJitter(
                        brightness=0.2
                    )
                )

            # -------------------------------
            # Contrast
            # -------------------------------
            if random.random() < self.p_contrast:
                ops.append(
                    transforms.ColorJitter(
                        contrast=0.2
                    )
                )

            # -------------------------------
            # Rotation
            # Scaling
            # Translation
            # Shearing
            # -------------------------------
            if random.random() < self.p_affine:
                ops.append(
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                        shear=2,
                    )
                )

            # -------------------------------
            # Identity
            # -------------------------------
            if len(ops) == 0:
                ops.append(transforms.Lambda(lambda x: x))

            # -------------------------------
            # Keep input size fixed
            # -------------------------------
            ops.append(
                transforms.Resize(
                    (self.height, self.width)
                )
            )

            sampled.append(
                transforms.Compose(ops)
            )

        return sampled


def project_lp_ball(perturbation, epsilon, norm='linf'):
    """Project a perturbation into an Lp ball."""
    norm = str(norm).lower()
    epsilon = float(epsilon)
    if norm in ('linf', 'inf', 'l_inf'):
        return torch.clamp(perturbation, -epsilon, epsilon)
    if norm in ('l2', '2'):
        flat = perturbation.reshape(perturbation.shape[0], -1) if perturbation.dim() > 1 else perturbation.reshape(1, -1)
        norms = torch.linalg.vector_norm(flat, ord=2, dim=1).clamp_min(1e-12)
        scales = torch.clamp(epsilon / norms, max=1.0)
        return (flat * scales.unsqueeze(1)).reshape_as(perturbation)
    raise ValueError(f'Unsupported RobustUAP norm: {norm}')


def _targeted_binary_success(outputs, target_label):
    predictions = (outputs > 0).float().view(-1)
    targets = torch.full_like(predictions, float(target_label))
    return predictions.eq(targets).float().mean().item() if predictions.numel() else 0.0


def estimate_robustness(
    model,
    inject_trigger: Callable,
    inputs,
    trigger_boxes,
    universal_patch,
    sampler: TransformSampler,
    num_transform_samples,
    gamma,
    target_label,
    edge_softness,
    how_to_attach,
    max_batch_size=None,
):
    """Estimate how often transformed UAP neighbors remain successful."""
    robust_successes = 0
    batch_size = inputs.shape[0]
    max_batch_size = batch_size if max_batch_size is None else max(1, int(max_batch_size))
    with torch.no_grad():
        for augmentation in sampler.sample(num_transform_samples):
            transformed_patch = augmentation(universal_patch)
            successes = 0.0
            total = 0
            for batch_start in range(0, batch_size, max_batch_size):
                input_batch = inputs[batch_start:batch_start + max_batch_size]
                poisoned_inputs = inject_trigger(
                    input_batch,
                    trigger_boxes,
                    trigger_patch=transformed_patch,
                    trigger_mask=None,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach,
                )
                outputs = model(poisoned_inputs)
                predictions = (outputs > 0).float().view(-1)
                targets = torch.full_like(predictions, float(target_label))
                successes += predictions.eq(targets).float().sum().item()
                total += int(predictions.numel())
            if total > 0 and successes / float(total) >= float(gamma):
                robust_successes += 1
    return robust_successes / float(max(1, int(num_transform_samples)))
