"""Architecture-aware Grad-CAM utilities for attack visualizations."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class GradCAMResult:
    heatmap: torch.Tensor
    prediction: int
    confidence: float
    target_class: int
    target_layer: str


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _tensor_output(output):
    if torch.is_tensor(output):
        return output
    if hasattr(output, 'logits') and torch.is_tensor(output.logits):
        return output.logits
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value
    if isinstance(output, dict):
        for value in output.values():
            if torch.is_tensor(value):
                return value
    raise TypeError('Grad-CAM requires a tensor classification output.')


def _module_name(model, target):
    for name, module in model.named_modules():
        if module is target:
            return name or model.__class__.__name__
    return target.__class__.__name__


def resolve_target_layer(model):
    """Return a semantically late spatial layer and whether it is channel-last."""
    model = _unwrap_model(model)
    class_name = model.__class__.__name__.lower()

    if 'swin' in class_name and hasattr(model, 'features'):
        target = model.features[-1]
        return _module_name(model, target), target, True
    if hasattr(model, 'Mixed_7c'):
        target = model.Mixed_7c
        return _module_name(model, target), target, False
    if hasattr(model, 'layer4'):
        target = model.layer4[-1]
        return _module_name(model, target), target, False
    if 'alexnet' in class_name:
        convolutions = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        if convolutions:
            name, target = convolutions[-1]
            return name, target, False
    if hasattr(model, 'features') and len(model.features) > 0:
        target = model.features[-1]
        return _module_name(model, target), target, False

    convolutions = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    if convolutions:
        name, target = convolutions[-1]
        return name, target, False
    raise ValueError(
        f'Could not find a spatial Grad-CAM target layer in {model.__class__.__name__}.'
    )


def _prediction_score(logits, target_class=None):
    logits = _tensor_output(logits)
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    logits = logits.reshape(logits.shape[0], -1)
    if logits.shape[0] != 1:
        raise ValueError('Grad-CAM currently expects one image at a time.')

    if logits.shape[1] == 1:
        probability_one = torch.sigmoid(logits[0, 0])
        prediction = int(probability_one >= 0.5)
        selected_class = prediction if target_class is None else int(target_class)
        if selected_class not in (0, 1):
            raise ValueError('A one-logit classifier supports Grad-CAM classes 0 and 1 only.')
        score = logits[0, 0] if selected_class == 1 else -logits[0, 0]
        confidence = probability_one if prediction == 1 else 1.0 - probability_one
    else:
        probabilities = torch.softmax(logits[0], dim=0)
        prediction = int(torch.argmax(probabilities).item())
        selected_class = prediction if target_class is None else int(target_class)
        if not 0 <= selected_class < logits.shape[1]:
            raise ValueError(
                f'Grad-CAM target class {selected_class} is outside the model output range.'
            )
        score = logits[0, selected_class]
        confidence = probabilities[prediction]
    return score, prediction, float(confidence.detach().item()), selected_class


def compute_gradcam(model, image, target_class=None):
    """Compute Grad-CAM for one image and the requested or predicted class."""
    base_model = _unwrap_model(model)
    target_name, target_layer, channel_last = resolve_target_layer(base_model)
    try:
        device = next(base_model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    activation_holder = {}
    gradient_holder = {}

    def capture_activation(_module, _inputs, output):
        activation = _tensor_output(output)
        activation_holder['value'] = activation
        if activation.requires_grad:
            activation.register_hook(
                lambda gradient: gradient_holder.__setitem__('value', gradient)
            )

    handle = target_layer.register_forward_hook(capture_activation)
    was_training = base_model.training
    base_model.eval()
    base_model.zero_grad(set_to_none=True)
    try:
        model_input = image.detach().to(device)
        if model_input.ndim == 3:
            model_input = model_input.unsqueeze(0)
        if model_input.shape[0] != 1:
            raise ValueError('Grad-CAM currently expects one image at a time.')
        model_input.requires_grad_(True)
        with torch.enable_grad():
            logits = base_model(model_input)
            score, prediction, confidence, selected_class = _prediction_score(
                logits, target_class=target_class
            )
            score.backward()

        activation = activation_holder.get('value')
        gradient = gradient_holder.get('value')
        if activation is None or gradient is None:
            raise RuntimeError(
                f'Grad-CAM did not capture activations and gradients from {target_name}.'
            )
        if activation.ndim != 4 or gradient.ndim != 4:
            raise ValueError(
                f'Grad-CAM target {target_name} must produce a 4-D spatial tensor; '
                f'got activation {tuple(activation.shape)}.'
            )

        if channel_last:
            weights = gradient.mean(dim=(1, 2), keepdim=True)
            cam = (weights * activation).sum(dim=-1)
        else:
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1)
        cam = torch.relu(cam).unsqueeze(1)
        cam = F.interpolate(
            cam,
            size=tuple(model_input.shape[-2:]),
            mode='bilinear',
            align_corners=False,
        )[0, 0]
        cam = cam - cam.min()
        maximum = cam.max()
        if torch.isfinite(maximum) and maximum.item() > 1e-12:
            cam = cam / maximum
        else:
            cam = torch.zeros_like(cam)
        return GradCAMResult(
            heatmap=cam.detach().cpu(),
            prediction=prediction,
            confidence=confidence,
            target_class=selected_class,
            target_layer=target_name,
        )
    finally:
        handle.remove()
        base_model.zero_grad(set_to_none=True)
        base_model.train(was_training)


def heatmap_to_pil(heatmap):
    """Render a normalized heatmap using a dependency-free jet-style colormap."""
    values = torch.as_tensor(heatmap).detach().cpu().float().clamp(0.0, 1.0).numpy()
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode='RGB')


def overlay_heatmap(image, heatmap, alpha=0.45):
    """Overlay Grad-CAM while leaving zero-importance pixels unchanged."""
    resampling = getattr(Image, 'Resampling', Image)
    base = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    heatmap_image = heatmap_to_pil(heatmap).resize(image.size, resampling.BILINEAR)
    colors = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    importance = torch.as_tensor(heatmap).detach().cpu().float().clamp(0.0, 1.0).numpy()
    importance_image = Image.fromarray((importance * 255.0).astype(np.uint8), mode='L')
    importance_image = importance_image.resize(image.size, resampling.BILINEAR)
    importance = np.asarray(importance_image, dtype=np.float32)[..., None] / 255.0
    blend = float(alpha) * importance
    overlay = base * (1.0 - blend) + colors * blend
    return Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGB')
