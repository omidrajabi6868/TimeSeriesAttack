import numpy as np
import torch
from typing import Callable, Optional
from pathlib import Path
from Network import ClassificationModels


class AdversarialAttack:
    def __init__(self, 
                model: Callable,
                device: Optional[str] = None
                ):
                    
        self.model = model
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self._build_cost_function()
    
    def _build_cost_function(self):
        self.cost_function = torch.nn.BCEWithLogitsLoss()
        return self.cost_function

    @staticmethod
    def save_trigger(trigger, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        patch = trigger['patch']
        if not torch.is_tensor(patch):
            patch = torch.tensor(patch, dtype=torch.float32)
        torch.save(
            {
                'patch': patch.detach().cpu(),
                'trigger_box': trigger.get('trigger_box'),
                'trigger_boxes': trigger.get('trigger_boxes'),
                'target_label': float(trigger.get('target_label', 0.0)),
                'source_filter': trigger.get('source_filter', 'bad'),
                'epsilon': float(trigger.get('epsilon', 0.0)),
                'history': trigger.get('history', []),
            },
            output_path,
        )
        return str(output_path)

    @staticmethod
    def load_trigger(trigger_path, map_location='cpu'):
        trigger_path = Path(trigger_path)
        if not trigger_path.exists():
            raise FileNotFoundError(f'Trigger file not found: {trigger_path}')
        trigger_payload = torch.load(trigger_path, map_location=map_location)
        return {
            'patch': trigger_payload['patch'],
            'trigger_box': trigger_payload.get('trigger_box'),
            'trigger_boxes': trigger_payload.get('trigger_boxes'),
            'target_label': float(trigger_payload.get('target_label', 0.0)),
            'source_filter': trigger_payload.get('source_filter', 'bad'),
            'epsilon': float(trigger_payload.get('epsilon', 0.0)),
            'history': trigger_payload.get('history', []),
            'path': str(trigger_path),
        }

    def learn_universal_trigger(self,
                                data_loader,
                                trigger_box,
                                target_label=0.0,
                                source_filter='bad',
                                validation_loader=None,
                                steps=100,
                                learning_rate=0.1,
                                epsilon=0.08,
                                log_interval=1):
        self.model.eval()

        trigger_boxes = self._normalize_trigger_boxes(trigger_box)
        base_box = trigger_boxes[0]
        width = int(base_box['width'])
        height = int(base_box['height'])
        for candidate_box in trigger_boxes[1:]:
            if int(candidate_box['width']) != width or int(candidate_box['height']) != height:
                raise ValueError('All trigger_boxes must have identical width/height for universal trigger learning.')

        channels = 3
        trigger_delta = torch.zeros((len(trigger_boxes), channels, height, width), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([trigger_delta], lr=learning_rate)
        target_tensor_base = torch.tensor(target_label, dtype=torch.float32, device=self.device)

        history = []
        best_patch = None
        best_step = 0
        best_val_asr = float('-inf')

        for step_idx in range(steps):
            step_losses = []
            step_samples = 0

            batch_counter = 0
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)
                flat_targets = targets.view(-1)

                if source_filter == 'bad':
                    source_mask = (flat_targets == 0)
                elif source_filter == 'good':
                    source_mask = (flat_targets == 1)
                else:
                    source_mask = torch.ones(targets.shape[0], dtype=torch.bool, device=self.device)

                if source_mask.sum().item() == 0:
                    continue

                selected_inputs = inputs[source_mask].clone()
                poisoned_inputs = self._inject_trigger(selected_inputs, trigger_boxes, trigger_patch=trigger_delta)

                outputs = self.model(poisoned_inputs)
                target_tensor = target_tensor_base.unsqueeze(0).expand(outputs.shape[0], -1)
                loss = self.cost_function(outputs, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_losses.append(float(loss.item()))
                step_samples += int(outputs.shape[0])
                batch_counter += 1

            step_loss = float(np.mean(step_losses)) if step_losses else 0.0
            step_history = {
                'step': step_idx + 1,
                'loss': step_loss,
                'samples': step_samples,
            }

            if validation_loader is not None:
                val_metrics = self.evaluate_attack_success(
                    test_loader=validation_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=trigger_delta.detach(),
                    target_label=target_label,
                    source_only_bad=(source_filter == 'bad'),
                )
                val_asr = float(val_metrics['attack_success_rate'])
                step_history['validation_asr'] = val_asr
                if val_asr > best_val_asr:
                    best_val_asr = val_asr
                    best_patch = trigger_delta.detach().cpu().clone()
                    best_step = step_idx + 1

            history.append(step_history)

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                val_log = ''
                if validation_loader is not None:
                    val_log = f', val_asr={step_history.get("validation_asr", 0.0):.4f}'
                print(
                    f'[Trigger Learning] step={step_idx + 1}/{steps}, '
                    f'loss={step_loss:.6f}, samples={step_samples}'
                    f'{val_log}'
                )

        if best_patch is not None:
            learned_patch = best_patch
            selected_step = best_step
        else:
            learned_patch = trigger_delta.detach().cpu()
            selected_step = steps

        return {
            'patch': learned_patch,
            'history': history,
            'trigger_box': base_box,
            'trigger_boxes': trigger_boxes,
            'target_label': float(target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
            'selection': 'best_validation_asr' if validation_loader is not None else 'last_step',
            'selected_step': int(selected_step),
            'best_validation_asr': None if validation_loader is None else float(best_val_asr),
        }

    def evaluate_attack_success(self,
                                 test_loader,
                                 trigger_box,
                                 trigger_value=(1.0, 1.0, 1.0),
                                 trigger_patch=None,
                                 target_label=0.0,
                                 source_only_bad=False):
        self.model.eval()
        target_tensor = torch.tensor(target_label, dtype=torch.float32, device=self.device).view(1, -1)

        total = 0
        attack_success = 0
        clean_correct = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)
                flat_targets = targets.view(-1)

                if source_only_bad:
                    source_mask = (flat_targets == 0)
                else:
                    source_mask = torch.ones(targets.shape[0], dtype=torch.bool, device=self.device)

                if source_mask.sum().item() == 0:
                    continue

                source_inputs = inputs[source_mask]
                source_targets = targets[source_mask]

                clean_outputs = self.model(source_inputs)
                clean_preds = (clean_outputs > 0).float().view(-1)
                clean_targets = source_targets.view(-1)
                clean_correct += int((clean_preds == clean_targets).sum().item())

                poisoned_inputs = self._inject_trigger(
                    source_inputs.clone(),
                    trigger_box,
                    trigger_value=trigger_value,
                    trigger_patch=trigger_patch,
                )
                poisoned_outputs = self.model(poisoned_inputs)
                poisoned_preds = (poisoned_outputs > 0).float()

                expanded_target = target_tensor.expand(poisoned_preds.shape[0], -1)
                attack_success += int((poisoned_preds == expanded_target).sum().item())
                total += int(poisoned_preds.shape[0])

        return {
            'samples_evaluated': total,
            'clean_source_accuracy': (clean_correct / total) * 100 if total else 0.0,
            'attack_success_rate': (attack_success / total) * 100 if total else 0.0,
            'target_label': float(target_label),
            'trigger_box': trigger_box,
        }

    @staticmethod
    def _normalize_trigger_boxes(trigger_box):
        if isinstance(trigger_box, list):
            if len(trigger_box) == 0:
                raise ValueError('trigger_box list cannot be empty.')
            return trigger_box
        return [trigger_box]

    @staticmethod
    def _build_blend_mask(height, width, channels, device, dtype, edge_softness=0.2):
        softness = float(max(0.0, min(edge_softness, 0.49)))
        if softness <= 0.0:
            return torch.ones((1, channels, height, width), device=device, dtype=dtype)

        y_coords = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        y_dist = torch.minimum(y_coords, 1.0 - y_coords)
        x_dist = torch.minimum(x_coords, 1.0 - x_coords)
        y_weights = torch.clamp(y_dist / softness, 0.0, 1.0)
        x_weights = torch.clamp(x_dist / softness, 0.0, 1.0)
        y_weights = 0.5 - 0.5 * torch.cos(np.pi * y_weights)
        x_weights = 0.5 - 0.5 * torch.cos(np.pi * x_weights)
        mask_2d = torch.outer(y_weights, x_weights)
        return mask_2d.view(1, 1, height, width).expand(1, channels, height, width)

    @staticmethod
    def _inject_trigger(inputs, trigger_box, trigger_value=(1.0, 1.0, 1.0), trigger_patch=None):
        trigger_boxes = AdversarialAttack._normalize_trigger_boxes(trigger_box)

        _, channels, input_h, input_w = inputs.shape

        if trigger_patch is not None:
            if not torch.is_tensor(trigger_patch):
                trigger_patch = torch.tensor(trigger_patch, dtype=inputs.dtype, device=inputs.device)
            trigger_patch = trigger_patch.to(device=inputs.device, dtype=inputs.dtype)

            if trigger_patch.dim() == 3:
                patch_bank = trigger_patch.unsqueeze(0)
            elif trigger_patch.dim() == 4:
                patch_bank = trigger_patch
            else:
                raise ValueError('trigger_patch must be CHW or NCHW tensor-like.')
        else:
            patch_bank = None

        for idx, box in enumerate(trigger_boxes):
            x = int(box['x'])
            y = int(box['y'])
            width = int(box['width'])
            height = int(box['height'])

            if x < 0 or y < 0 or x + width > input_w or y + height > input_h:
                raise ValueError('trigger_box is out of image bounds.')

            region = inputs[:, :, y:y + height, x:x + width]
            blend_mask = AdversarialAttack._build_blend_mask(
                height=height,
                width=width,
                channels=channels,
                device=inputs.device,
                dtype=inputs.dtype,
            )

            if patch_bank is not None:
                if patch_bank.shape[1] != channels or patch_bank.shape[2] != height or patch_bank.shape[3] != width:
                    raise ValueError('trigger_patch shape must match (C, height, width) from trigger_box.')

                if patch_bank.shape[0] == len(trigger_boxes):
                    patch = patch_bank[idx].unsqueeze(0).expand(inputs.shape[0], -1, -1, -1)
                elif patch_bank.shape[0] == 1:
                    patch = patch_bank.expand(inputs.shape[0], -1, -1, -1)
                elif patch_bank.shape[0] == inputs.shape[0]:
                    patch = patch_bank
                else:
                    raise ValueError(
                        'trigger_patch batch dimension must be 1, match input batch size, '
                        'or match number of trigger boxes.'
                    )

                patched_region = torch.clamp(region + patch, 0.0, 1.0)
                blended_region = region * (1.0 - blend_mask) + patched_region * blend_mask
            else:
                trigger_tensor = torch.tensor(trigger_value, dtype=inputs.dtype, device=inputs.device).view(channels, 1, 1)
                if trigger_tensor.shape[0] != channels:
                    raise ValueError('trigger_value channel count must match input channels.')
                target_region = trigger_tensor.unsqueeze(0).expand(inputs.shape[0], -1, height, width)
                blended_region = region * (1.0 - blend_mask) + target_region * blend_mask

            inputs[:, :, y:y + height, x:x + width] = blended_region
        return inputs
