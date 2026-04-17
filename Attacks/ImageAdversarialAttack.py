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
        self.device = torch.device(self.device)
        self.model = self.model.to(self.device)
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
                'mask': (
                    trigger.get('mask').detach().cpu()
                    if trigger.get('mask') is not None and torch.is_tensor(trigger.get('mask'))
                    else trigger.get('mask')
                ),
                'trigger_box': trigger.get('trigger_box'),
                'trigger_boxes': trigger.get('trigger_boxes'),
                'target_label': float(trigger.get('target_label', 0.0)),
                'source_filter': trigger.get('source_filter', 'bad'),
                'epsilon': float(trigger.get('epsilon', 0.0)),
                'softness': trigger.get('softness', {}),
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
            'mask': trigger_payload.get('mask'),
            'trigger_box': trigger_payload.get('trigger_box'),
            'trigger_boxes': trigger_payload.get('trigger_boxes'),
            'target_label': float(trigger_payload.get('target_label', 0.0)),
            'source_filter': trigger_payload.get('source_filter', 'bad'),
            'epsilon': float(trigger_payload.get('epsilon', 0.0)),
            'softness': trigger_payload.get('softness', {}),
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
                                mask_learning_rate=0.02,
                                epsilon=1.0,
                                optimize_mask=True,
                                initial_edge_softness=0.30,
                                min_edge_softness=0.05,
                                softness_decay=0.85,
                                softness_patience=8,
                                asr_hardening_threshold=0.70,
                                mask_l1_weight=0.01,
                                patch_l2_weight=0.0005,
                                softness_alignment_weight=0.05,
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
        trigger_delta = torch.randn((len(trigger_boxes), channels, height, width), device=self.device, requires_grad=True)
        
        patch_optimizer = torch.optim.Adam([trigger_delta], lr=learning_rate)
        current_softness = float(max(min_edge_softness, initial_edge_softness))

        learned_mask = None
        mask_logits = None
        mask_optimizer = None
        base_mask = None
        mask_training_active = bool(optimize_mask)
        if optimize_mask:
            base_mask = self._build_blend_mask(
                height=height,
                width=width,
                channels=channels,
                device=self.device,
                dtype=trigger_delta.dtype,
                edge_softness=current_softness,
            ).expand(len(trigger_boxes), -1, -1, -1)
            # Learn an opacity gain over the smooth base mask.
            # Start from a neutral gain (sigmoid(0)=0.5) so the patch receives
            # usable gradients from the first step. Initializing near zero
            # opacity can stall updates and lead to early ASR plateaus.
            mask_logits = torch.zeros_like(base_mask, device=self.device).requires_grad_(True)
            mask_optimizer = torch.optim.Adam([mask_logits], lr=mask_learning_rate)

        history = []
        best_patch = None
        best_mask = None
        best_step = 0
        best_val_asr = float('-inf')
        no_improve_steps = 0

        for step_idx in range(steps):
            step_losses = []
            step_samples = 0
            previous_patch = trigger_delta.detach().clone()

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
                blend_mask = (
                    self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits)
                    if mask_logits is not None else None
                )
                poisoned_inputs = self._inject_trigger(
                    selected_inputs,
                    trigger_boxes,
                    trigger_patch=trigger_delta,
                    trigger_mask=blend_mask,
                    edge_softness=current_softness,
                )

                outputs = self.model(poisoned_inputs)
                target_tensor = torch.full_like(outputs, float(target_label))
                attack_loss = self.cost_function(outputs, target_tensor)
                patch_reg = patch_l2_weight * torch.mean(trigger_delta ** 2)

                if mask_logits is not None:
                    base_mask = self._build_blend_mask(
                        height=height,
                        width=width,
                        channels=channels,
                        device=self.device,
                        dtype=trigger_delta.dtype,
                        edge_softness=current_softness,
                    ).expand(len(trigger_boxes), -1, -1, -1)
                    mask_values = self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits)
                    mask_growth = torch.relu(mask_values - base_mask)
                    mask_reg = mask_l1_weight * torch.mean(mask_growth)
                    softness_reg = softness_alignment_weight * torch.mean((mask_values - base_mask) ** 2)
                else:
                    mask_reg = torch.tensor(0.0, device=self.device)
                    softness_reg = torch.tensor(0.0, device=self.device)
                loss = attack_loss + patch_reg + mask_reg + softness_reg

                patch_optimizer.zero_grad()
                if mask_optimizer is not None:
                    mask_optimizer.zero_grad()
                loss.backward()
                patch_optimizer.step()
                if epsilon is not None and epsilon > 0:
                    with torch.no_grad():
                        trigger_delta.data.clamp_(-float(epsilon), float(epsilon))
                if mask_optimizer is not None and mask_training_active:
                    mask_optimizer.step()
                    with torch.no_grad():
                        mask_logits.data.clamp_(-8.0, 8.0)

                step_losses.append(float(loss.item()))
                step_samples += int(outputs.shape[0])

            step_loss = float(np.mean(step_losses)) if step_losses else 0.0
            patch_update_l2 = float(torch.norm((trigger_delta.detach() - previous_patch).reshape(-1), p=2).item())
            step_history = {
                'step': step_idx + 1,
                'loss': step_loss,
                'samples': step_samples,
                'patch_update_l2': patch_update_l2,
            }

            if validation_loader is not None:
                val_metrics = self.evaluate_attack_success(
                    test_loader=validation_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=trigger_delta.detach(),
                    trigger_mask=(
                        self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach())
                        if mask_logits is not None else None
                    ),
                    target_label=target_label,
                    source_filter=source_filter,
                    edge_softness=current_softness,
                )
                val_asr = float(val_metrics['attack_success_rate'])
                step_history['validation_asr'] = val_asr
                step_history['edge_softness'] = current_softness
                if val_asr > best_val_asr:
                    best_val_asr = val_asr
                    best_patch = trigger_delta.detach().cpu().clone()
                    best_mask = (
                        self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).cpu().clone()
                        if mask_logits is not None else None
                    )
                    best_step = step_idx + 1
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                threshold_ratio = self._normalize_asr_threshold(asr_hardening_threshold)
                val_asr_ratio = val_asr / 100.0

                if (
                    val_asr_ratio < threshold_ratio
                    and no_improve_steps >= softness_patience
                ):
                    new_softness = max(min_edge_softness, current_softness * softness_decay)
                    if new_softness < current_softness:
                        current_softness = new_softness
                    no_improve_steps = 0

                if val_asr_ratio >= threshold_ratio:
                    mask_training_active = False

            history.append(step_history)

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                val_log = ''
                if validation_loader is not None:
                    val_log = f', val_asr={step_history.get("validation_asr", 0.0):.4f}'
                print(
                    f'[Trigger Learning] step={step_idx + 1}/{steps}, '
                    f'loss={step_loss:.6f}, samples={step_samples}, '
                    f'patch_update_l2={patch_update_l2:.6f}'
                    f'{val_log}'
                )
                if step_samples == 0:
                    print(
                        '[Trigger Learning] warning: no samples matched source_filter '
                        f'"{source_filter}" at this step.'
                    )

        if best_patch is not None:
            learned_patch = best_patch
            learned_mask = best_mask
            selected_step = best_step
        else:
            learned_patch = trigger_delta.detach().cpu()
            learned_mask = (
                self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).cpu()
                if mask_logits is not None else None
            )
            selected_step = steps

        return {
            'patch': learned_patch,
            'mask': learned_mask,
            'history': history,
            'trigger_box': base_box,
            'trigger_boxes': trigger_boxes,
            'target_label': float(target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
            'softness': {
                'initial_edge_softness': float(initial_edge_softness),
                'final_edge_softness': float(current_softness),
                'min_edge_softness': float(min_edge_softness),
                'softness_decay': float(softness_decay),
                'softness_patience': int(softness_patience),
                'asr_hardening_threshold': float(asr_hardening_threshold),
            },
            'selection': 'best_validation_asr' if validation_loader is not None else 'last_step',
            'selected_step': int(selected_step),
            'best_validation_asr': None if validation_loader is None else float(best_val_asr),
        }

    def evaluate_attack_success(self,
                                 test_loader,
                                 trigger_box,
                                 trigger_value=(1.0, 1.0, 1.0),
                                 trigger_patch=None,
                                 trigger_mask=None,
                                 target_label=0.0,
                                 source_only_bad=False,
                                 source_filter=None,
                                 edge_softness=0.2):
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

                if source_filter is None:
                    source_filter = 'bad' if source_only_bad else 'all'
                if source_filter == 'bad':
                    source_mask = (flat_targets == 0)
                elif source_filter == 'good':
                    source_mask = (flat_targets == 1)
                elif source_filter == 'all':
                    source_mask = torch.ones(targets.shape[0], dtype=torch.bool, device=self.device)
                else:
                    raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")

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
                    trigger_mask=trigger_mask,
                    edge_softness=edge_softness,
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
            'source_filter': source_filter if source_filter is not None else ('bad' if source_only_bad else 'all'),
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
    def _compose_trigger_mask(base_mask=None, mask_logits=None):
        if mask_logits is None:
            return base_mask
        if base_mask is None:
            return torch.sigmoid(mask_logits)
        gain = torch.sigmoid(mask_logits)
        return torch.clamp(base_mask + (1.0 - base_mask) * gain, 0.0, 1.0)

    @staticmethod
    def _normalize_asr_threshold(asr_hardening_threshold):
        threshold = float(asr_hardening_threshold)
        # ASR metrics are tracked in percentage [0, 100]. Accept user thresholds
        # in either ratio [0, 1] or percentage [0, 100].
        if threshold > 1.0:
            threshold = threshold / 100.0
        return max(0.0, min(1.0, threshold))

    @staticmethod
    def _inject_trigger(
        inputs,
        trigger_box,
        trigger_value=(1.0, 1.0, 1.0),
        trigger_patch=None,
        trigger_mask=None,
        edge_softness=0.2,
    ):
        trigger_boxes = AdversarialAttack._normalize_trigger_boxes(trigger_box)
        poisoned_inputs = inputs.clone()

        _, channels, input_h, input_w = poisoned_inputs.shape

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
        mask_bank = None
        if trigger_mask is not None:
            if not torch.is_tensor(trigger_mask):
                trigger_mask = torch.tensor(trigger_mask, dtype=inputs.dtype, device=inputs.device)
            trigger_mask = trigger_mask.to(device=inputs.device, dtype=inputs.dtype)
            if trigger_mask.dim() == 3:
                mask_bank = trigger_mask.unsqueeze(0)
            elif trigger_mask.dim() == 4:
                mask_bank = trigger_mask
            else:
                raise ValueError('trigger_mask must be CHW or NCHW tensor-like.')

        for idx, box in enumerate(trigger_boxes):
            x = int(box['x'])
            y = int(box['y'])
            width = int(box['width'])
            height = int(box['height'])

            if x < 0 or y < 0 or x + width > input_w or y + height > input_h:
                raise ValueError('trigger_box is out of image bounds.')

            region = poisoned_inputs[:, :, y:y + height, x:x + width].clone()
            if mask_bank is not None:
                if mask_bank.shape[1] != channels or mask_bank.shape[2] != height or mask_bank.shape[3] != width:
                    raise ValueError('trigger_mask shape must match (C, height, width) from trigger_box.')
                if mask_bank.shape[0] == len(trigger_boxes):
                    blend_mask = mask_bank[idx].unsqueeze(0)
                elif mask_bank.shape[0] == 1:
                    blend_mask = mask_bank
                else:
                    raise ValueError('trigger_mask batch dimension must be 1 or match number of trigger boxes.')
                blend_mask = torch.clamp(blend_mask, 0.0, 1.0).expand(poisoned_inputs.shape[0], -1, -1, -1)
            else:
                blend_mask = AdversarialAttack._build_blend_mask(
                    height=height,
                    width=width,
                    channels=channels,
                    device=poisoned_inputs.device,
                    dtype=poisoned_inputs.dtype,
                    edge_softness=edge_softness,
                ).expand(poisoned_inputs.shape[0], -1, -1, -1)

            if patch_bank is not None:
                if patch_bank.shape[1] != channels or patch_bank.shape[2] != height or patch_bank.shape[3] != width:
                    raise ValueError('trigger_patch shape must match (C, height, width) from trigger_box.')

                if patch_bank.shape[0] == len(trigger_boxes):
                    patch = patch_bank[idx].unsqueeze(0).expand(poisoned_inputs.shape[0], -1, -1, -1)
                elif patch_bank.shape[0] == 1:
                    patch = patch_bank.expand(poisoned_inputs.shape[0], -1, -1, -1)
                elif patch_bank.shape[0] == poisoned_inputs.shape[0]:
                    patch = patch_bank
                else:
                    raise ValueError(
                        'trigger_patch batch dimension must be 1, match input batch size, '
                        'or match number of trigger boxes.'
                    )

                patched_region = torch.clamp(region + patch, 0.0, 1.0)
                blended_region = region * (1.0 - blend_mask) + patched_region * blend_mask
            else:
                trigger_tensor = torch.tensor(
                    trigger_value,
                    dtype=poisoned_inputs.dtype,
                    device=poisoned_inputs.device,
                ).view(channels, 1, 1)
                if trigger_tensor.shape[0] != channels:
                    raise ValueError('trigger_value channel count must match input channels.')
                target_region = trigger_tensor.unsqueeze(0).expand(poisoned_inputs.shape[0], -1, height, width)
                blended_region = region * (1.0 - blend_mask) + target_region * blend_mask

            poisoned_inputs[:, :, y:y + height, x:x + width] = blended_region
        return poisoned_inputs
