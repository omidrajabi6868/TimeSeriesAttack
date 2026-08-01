"""Shared visualization and metrics helpers for image defense evaluations."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack


def tensor_to_pil_image(image_tensor):
    image_uint8 = image_tensor.detach().cpu().clamp(0.0, 1.0).mul(255.0).byte()
    image_uint8 = image_uint8.permute(1, 2, 0).contiguous()
    height, width = image_uint8.shape[:2]
    return Image.frombytes('RGB', (width, height), bytes(image_uint8.view(-1).tolist()))


def difference_image(before, after, scale=5.0):
    return (after.detach().cpu() - before.detach().cpu()).abs().mul(float(scale)).clamp(0.0, 1.0)


def trigger_coverage_ratio(trigger_boxes, image_height=None, image_width=None):
    if image_height is None or image_width is None:
        return None
    image_area = float(image_height * image_width)
    if image_area <= 0:
        return None
    covered_area = 0.0
    for box in AdversarialAttack._normalize_trigger_boxes(trigger_boxes):
        covered_area += max(0, int(box['width'])) * max(0, int(box['height']))
    return min(1.0, covered_area / image_area)


def collect_defense_examples(example_list, selection_mask, max_examples, clean_inputs,
                             defended_clean_inputs, poisoned_inputs, defended_poisoned_inputs,
                             targets, clean_preds, defended_clean_preds,
                             poisoned_preds, defended_poisoned_preds, defended,
                             clean_key, adversarial_key, clean_pred_key,
                             adversarial_pred_key):
    if len(example_list) >= max_examples:
        return
    for idx in torch.where(selection_mask.cpu())[0].tolist():
        if len(example_list) >= max_examples:
            break
        example_list.append({
            'clean': clean_inputs[idx],
            clean_key: defended_clean_inputs[idx],
            'adversarial': poisoned_inputs[idx],
            adversarial_key: defended_poisoned_inputs[idx],
            'target': float(targets.view(-1)[idx].item()),
            'clean_pred': float(clean_preds.view(-1)[idx].item()),
            clean_pred_key: float(defended_clean_preds.view(-1)[idx].item()),
            'adversarial_pred': float(poisoned_preds.view(-1)[idx].item()),
            adversarial_pred_key: float(defended_poisoned_preds.view(-1)[idx].item()),
            'defended': bool(defended),
        })


def save_defense_examples(examples, output_dir, example_source, filename_prefix, panels_builder):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for idx, example in enumerate(examples, start=1):
        save_path = output_path / f'{filename_prefix}_{example_source}_{idx:02d}.png'
        save_defense_comparison(example, save_path, panels_builder(example))
        saved_paths.append(str(save_path))
    return {
        'output_dir': str(output_path),
        'selection': example_source,
        'saved_images': len(saved_paths),
        'paths': saved_paths,
    }


def save_defense_comparison(example, save_path, panels):
    images = [tensor_to_pil_image(tensor) for _, tensor, _ in panels]
    widths, heights = zip(*(image.size for image in images))
    label_height = 42
    canvas = Image.new('RGB', (sum(widths), max(heights) + label_height), color='white')
    draw = ImageDraw.Draw(canvas)
    x_offset = 0
    for (title, _, pred), image in zip(panels, images):
        canvas.paste(image, (x_offset, label_height))
        pred_text = '' if pred is None else f'\ntrue={example["target"]:.0f}, pred={pred:.0f}'
        draw.text((x_offset + 4, 4), f'{title}{pred_text}', fill='black')
        x_offset += image.size[0]
    canvas.save(save_path)
