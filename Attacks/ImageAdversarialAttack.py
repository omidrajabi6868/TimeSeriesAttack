import numpy as np
import torch
from typing import Callable, Optional, Sequence
from pathlib import Path
from Network import ClassificationModels


class AdversarialAttack:
    def __init__(self, 
                model: Callable,
                device: Optional[str] = None,
                use_multi_gpu: bool = True,
                gpu_ids: Optional[Sequence[int]] = None,
                ):
                    
        self.model = model
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.device = torch.device(self.device)
        self.model = self.model.to(self.device)
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        if (
            self.use_multi_gpu
            and self.device.type == 'cuda'
            and torch.cuda.device_count() > 1
            and not isinstance(self.model, torch.nn.DataParallel)
        ):
            if self.gpu_ids is None:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            if len(self.gpu_ids) > 1:
                print(f'Using DataParallel for adversarial attack on GPUs: {self.gpu_ids}')
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
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
                'progressive_shrink': trigger.get('progressive_shrink', {}),
                'progressive_resize': trigger.get('progressive_resize', {}),
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
            'progressive_shrink': trigger_payload.get('progressive_shrink', {}),
            'progressive_resize': trigger_payload.get('progressive_resize', {}),
            'history': trigger_payload.get('history', []),
            'path': str(trigger_path),
        }

    def learn_universal_trigger(self,
                                data_loader,
                                trigger_box,
                                target_label=1.0,
                                source_filter='bad',
                                validation_loader=None,
                                report_training_asr=True,
                                steps=100,
                                learning_rate=0.1,
                                mask_learning_rate=0.02,
                                optimize_mask=True,
                                initial_edge_softness=0.30,
                                min_edge_softness=0.05,
                                softness_decay=0.85,
                                softness_patience=8,
                                asr_hardening_threshold=70.0,
                                mask_l1_weight=0.01,
                                patch_l2_weight=0.0005,
                                softness_alignment_weight=0.05,
                                patch_update_method='momentum_sign',
                                momentum_decay=1.0,
                                gradient_norm_epsilon=1e-12,
                                log_interval=1,
                                progressive_resize=None,
                                progressive_shrink=True,
                                patch_shrink_factor=0.85,
                                min_patch_size=(16, 16),
                                randomize_training_location=True,
                                patch_growth_factor=None,
                                min_steps_per_patch_size=10,
                                size_patience=None):
        self.model.eval()
        progressive_resize_enabled = bool(progressive_shrink if progressive_resize is None else progressive_resize)

        validation_trigger_boxes = self._normalize_trigger_boxes(trigger_box)
        validation_anchor_boxes = [dict(box) for box in validation_trigger_boxes]
        base_box = validation_anchor_boxes[0]
        width = int(base_box['width'])
        height = int(base_box['height'])
        for candidate_box in validation_anchor_boxes[1:]:
            if int(candidate_box['width']) != width or int(candidate_box['height']) != height:
                raise ValueError('All trigger_boxes must have identical width/height for universal trigger learning.')

        channels = 3
        full_patch_size = self._infer_full_patch_size(data_loader, fallback=(width, height))
        max_width, max_height = self._normalize_patch_size(full_patch_size)
        min_width, min_height = self._normalize_patch_size(min_patch_size)
        min_width = min(min_width, max_width)
        min_height = min(min_height, max_height)
        patch_shrink_factor = float(patch_shrink_factor)
        if not 0.0 < patch_shrink_factor < 1.0:
            raise ValueError('patch_shrink_factor must be between 0 and 1.')
        if patch_growth_factor is None:
            patch_growth_factor = 1.0 / patch_shrink_factor
        patch_growth_factor = float(patch_growth_factor)
        if patch_growth_factor <= 1.0:
            raise ValueError('patch_growth_factor must be greater than 1.')
        min_steps_per_patch_size = max(1, int(min_steps_per_patch_size))
        size_patience = int(size_patience) if size_patience is not None else int(softness_patience)
        size_patience = max(1, size_patience)
        if progressive_resize_enabled:
            width, height = min_width, min_height

        trigger_boxes = self._resize_trigger_boxes(validation_anchor_boxes, width, height, full_patch_size)
        trigger_delta = torch.zeros((len(trigger_boxes), channels, height, width), device=self.device)
        trigger_delta.requires_grad_()
        patch_update_method = str(patch_update_method).lower()
        if patch_update_method in ('mi_fgsm', 'mifgsm', 'momentum'):
            patch_update_method = 'momentum_sign'
        elif patch_update_method in ('iterative_fgsm', 'ifgsm', 'sign'):
            patch_update_method = 'pgd_sign'
        elif patch_update_method == 'pgd':
            patch_update_method = 'pgd_sign'

        valid_patch_update_methods = {'adam', 'pgd_sign', 'momentum_sign'}
        if patch_update_method not in valid_patch_update_methods:
            raise ValueError(
                'patch_update_method must be one of: '
                f'{sorted(valid_patch_update_methods)}.'
            )

        patch_optimizer = None
        if patch_update_method == 'adam':
            patch_optimizer = torch.optim.Adam([trigger_delta], lr=learning_rate)

        patch_momentum = torch.zeros_like(trigger_delta, device=self.device)
        alpha = float(learning_rate)
        mu = float(momentum_decay)
        grad_norm_epsilon = float(gradient_norm_epsilon)

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
            mask_logits = torch.zeros_like(base_mask, device=self.device).requires_grad_(True)
            mask_optimizer = torch.optim.Adam([mask_logits], lr=mask_learning_rate)

        history = []
        best_patch = None
        best_mask = None
        best_trigger_boxes = [dict(box) for box in trigger_boxes]
        best_step = 0
        best_val_asr = float('-inf')
        first_success_patch = None
        first_success_mask = None
        first_success_boxes = None
        first_success_step = 0
        first_success_asr = float('-inf')
        resize_events = []
        no_improve_steps = 0
        size_step_count = 0
        size_no_improve_steps = 0
        best_size_asr = float('-inf')

        for step_idx in range(steps):
            stop_training = False
            size_step_count += 1
            step_losses = []
            step_samples = 0
            previous_patch = torch.tanh(trigger_delta).detach().clone()

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
                bounded_trigger_patch = torch.tanh(trigger_delta)
                training_trigger_boxes = (
                    self._random_trigger_boxes(
                        batch_size=selected_inputs.shape[0],
                        patch_width=width,
                        patch_height=height,
                        image_width=selected_inputs.shape[-1],
                        image_height=selected_inputs.shape[-2],
                    )
                    if randomize_training_location else trigger_boxes
                )
                training_patch = bounded_trigger_patch
                training_mask = blend_mask
                if randomize_training_location:
                    training_patch = bounded_trigger_patch.mean(dim=0, keepdim=True)
                    training_mask = blend_mask.mean(dim=0, keepdim=True) if blend_mask is not None else None

                poisoned_inputs = self._inject_trigger(
                    selected_inputs,
                    training_trigger_boxes,
                    trigger_patch=training_patch,
                    trigger_mask=training_mask,
                    edge_softness=current_softness,
                )

                outputs = self.model(poisoned_inputs)
                target_tensor = torch.full_like(outputs, float(target_label))
                attack_loss = self.cost_function(outputs, target_tensor)
                patch_reg = patch_l2_weight * torch.mean(bounded_trigger_patch ** 2)

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

                if patch_optimizer is not None:
                    patch_optimizer.zero_grad()
                elif trigger_delta.grad is not None:
                    trigger_delta.grad.zero_()
                if mask_optimizer is not None:
                    mask_optimizer.zero_grad()
                loss.backward()

                if patch_update_method == 'adam':
                    patch_optimizer.step()
                else:
                    with torch.no_grad():
                        patch_grad = trigger_delta.grad
                        if patch_grad is not None:
                            if patch_update_method == 'pgd_sign':
                                trigger_delta.add_(-alpha * patch_grad.sign())
                            elif patch_update_method == 'momentum_sign':
                                grad_l1_norm = patch_grad.norm(p=1)
                                if torch.isfinite(grad_l1_norm) and grad_l1_norm.item() > grad_norm_epsilon:
                                    # Targeted trigger learning minimizes the BCE objective. Use the
                                    # negative loss gradient as the ascent objective so the patch update
                                    # follows: g = mu * g + grad / grad.norm(p=1),
                                    # delta += alpha * sign(g), patch = tanh(delta).
                                    normalized_grad = -patch_grad / torch.clamp(
                                        grad_l1_norm,
                                        min=grad_norm_epsilon,
                                    )
                                    patch_momentum.mul_(mu).add_(normalized_grad)
                                    trigger_delta.add_(alpha * patch_momentum.sign())
                if trigger_delta.grad is not None:
                    trigger_delta.grad.zero_()

                if mask_optimizer is not None and mask_training_active:
                    mask_optimizer.step()

                step_losses.append(float(loss.item()))
                step_samples += int(outputs.shape[0])

            step_loss = float(np.mean(step_losses)) if step_losses else 0.0
            patch_update_l2 = float(torch.norm(
                (torch.tanh(trigger_delta).detach() - previous_patch).reshape(-1),
                p=2,
            ).item())
            step_history = {
                'step': step_idx + 1,
                'loss': step_loss,
                'samples': step_samples,
                'patch_update_l2': patch_update_l2,
                'patch_update_method': patch_update_method,
            }

            if report_training_asr:
                train_metrics = self.evaluate_attack_success(
                    test_loader=data_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=torch.tanh(trigger_delta).detach(),
                    trigger_mask=(
                        self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach())
                        if mask_logits is not None else None
                    ),
                    target_label=target_label,
                    source_filter=source_filter,
                    edge_softness=current_softness,
                )
                step_history['training_asr'] = float(train_metrics['attack_success_rate'])

            if validation_loader is not None:

                val_metrics = self.evaluate_attack_success(
                    test_loader=validation_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=torch.tanh(trigger_delta).detach(),
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
                    best_patch = torch.tanh(trigger_delta).detach().cpu().clone()
                    best_mask = (
                        self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).cpu().clone()
                        if mask_logits is not None else None
                    )
                    best_trigger_boxes = [dict(box) for box in trigger_boxes]
                    best_step = step_idx + 1
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                if val_asr > best_size_asr:
                    best_size_asr = val_asr
                    size_no_improve_steps = 0
                else:
                    size_no_improve_steps += 1

                if (
                    val_asr < asr_hardening_threshold
                    and no_improve_steps >= softness_patience
                ):
                    new_softness = max(min_edge_softness, current_softness * softness_decay)
                    if new_softness < current_softness:
                        current_softness = new_softness
                    no_improve_steps = 0

                size_optimized_enough = size_step_count >= min_steps_per_patch_size
                if val_asr >= asr_hardening_threshold and size_optimized_enough:
                    first_success_patch = torch.tanh(trigger_delta).detach().cpu().clone()
                    first_success_mask = (
                        self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).cpu().clone()
                        if mask_logits is not None else None
                    )
                    first_success_boxes = [dict(box) for box in trigger_boxes]
                    first_success_step = step_idx + 1
                    first_success_asr = val_asr
                    step_history['size_decision'] = 'accepted'
                    stop_training = True

                should_grow = (
                    progressive_resize_enabled
                    and validation_loader is not None
                    and size_optimized_enough
                    and val_asr < asr_hardening_threshold
                    and size_no_improve_steps >= size_patience
                )
                if should_grow:
                    next_width = min(int(round(width * patch_growth_factor)), max_width)
                    next_height = min(int(round(height * patch_growth_factor)), max_height)
                    if next_width <= width and width < max_width:
                        next_width = width + 1
                    if next_height <= height and height < max_height:
                        next_height = height + 1
                    can_grow = next_width > width or next_height > height
                    if can_grow:
                        resize_events.append({
                            'step': step_idx + 1,
                            'from_size': (int(width), int(height)),
                            'to_size': (int(next_width), int(next_height)),
                            'validation_asr': val_asr,
                            'decision': 'grow',
                        })
                        resized_patch = torch.nn.functional.interpolate(
                            torch.tanh(trigger_delta).detach(),
                            size=(next_height, next_width),
                            mode='bilinear',
                            align_corners=False,
                        )
                        trigger_delta = torch.atanh(torch.clamp(resized_patch, -0.999999, 0.999999)).detach()
                        trigger_delta.requires_grad_()
                        width, height = next_width, next_height
                        trigger_boxes = self._resize_trigger_boxes(
                            validation_anchor_boxes,
                            width,
                            height,
                            full_patch_size,
                        )
                        patch_momentum = torch.zeros_like(trigger_delta, device=self.device)
                        if patch_update_method == 'adam':
                            patch_optimizer = torch.optim.Adam([trigger_delta], lr=learning_rate)
                        if optimize_mask:
                            base_mask = self._build_blend_mask(
                                height=height,
                                width=width,
                                channels=channels,
                                device=self.device,
                                dtype=trigger_delta.dtype,
                                edge_softness=current_softness,
                            ).expand(len(trigger_boxes), -1, -1, -1)
                            mask_logits = torch.zeros_like(base_mask, device=self.device).requires_grad_(True)
                            mask_optimizer = torch.optim.Adam([mask_logits], lr=mask_learning_rate)
                            mask_training_active = True
                        else:
                            base_mask = None
                            mask_logits = None
                            mask_optimizer = None
                        no_improve_steps = 0
                        size_step_count = 0
                        size_no_improve_steps = 0
                        best_size_asr = float('-inf')
                        step_history['grow_to_size'] = (int(width), int(height))
                    else:
                        step_history['size_decision'] = 'max_size_reached'

            history.append(step_history)

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                val_log = ''
                train_log = ''
                if report_training_asr:
                    train_log = f', train_asr={step_history.get("training_asr", 0.0):.4f}'
                if validation_loader is not None:
                    val_log = f', val_asr={step_history.get("validation_asr", 0.0):.4f}'
                print(
                    f'[Trigger Learning] step={step_idx + 1}/{steps}, '
                    f'loss={step_loss:.6f}, samples={step_samples}, '
                    f'patch_update_l2={patch_update_l2:.6f}'
                    f'{train_log}{val_log}'
                )
                if step_samples == 0:
                    print(
                        '[Trigger Learning] warning: no samples matched source_filter '
                        f'"{source_filter}" at this step.'
                    )

            if stop_training:
                break

        if progressive_resize_enabled and first_success_patch is not None:
            learned_patch = first_success_patch
            learned_mask = first_success_mask
            trigger_boxes = first_success_boxes
            selected_step = first_success_step
            best_val_asr = first_success_asr
        elif best_patch is not None:
            learned_patch = best_patch
            learned_mask = best_mask
            trigger_boxes = best_trigger_boxes
            selected_step = best_step
        else:
            learned_patch = torch.tanh(trigger_delta).detach().cpu()
            learned_mask = (
                self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).cpu()
                if mask_logits is not None else None
            )
            selected_step = steps

        return {
            'patch': learned_patch,
            'mask': learned_mask,
            'history': history,
            'trigger_box': trigger_boxes[0],
            'trigger_boxes': trigger_boxes,
            'target_label': float(target_label),
            'source_filter': source_filter,
            'patch_update_method': patch_update_method,
            'softness': {
                'initial_edge_softness': float(initial_edge_softness),
                'final_edge_softness': float(current_softness),
                'min_edge_softness': float(min_edge_softness),
                'softness_decay': float(softness_decay),
                'softness_patience': int(softness_patience),
                'asr_hardening_threshold': float(asr_hardening_threshold),
            },
            'progressive_shrink': {
                'enabled': bool(progressive_resize_enabled),
                'randomize_training_location': bool(randomize_training_location),
                'direction': 'grow',
                'initial_patch_size': (int(min_width), int(min_height)),
                'final_patch_size': (int(trigger_boxes[0]['width']), int(trigger_boxes[0]['height'])),
                'max_patch_size': (int(max_width), int(max_height)),
                'patch_growth_factor': float(patch_growth_factor),
                'min_steps_per_patch_size': int(min_steps_per_patch_size),
                'size_patience': int(size_patience),
                'asr_threshold': float(asr_hardening_threshold),
                'events': resize_events,
            },
            'progressive_resize': {
                'enabled': bool(progressive_resize_enabled),
                'direction': 'grow',
                'randomize_training_location': bool(randomize_training_location),
                'initial_patch_size': (int(min_width), int(min_height)),
                'final_patch_size': (int(trigger_boxes[0]['width']), int(trigger_boxes[0]['height'])),
                'max_patch_size': (int(max_width), int(max_height)),
                'patch_growth_factor': float(patch_growth_factor),
                'min_steps_per_patch_size': int(min_steps_per_patch_size),
                'size_patience': int(size_patience),
                'asr_threshold': float(asr_hardening_threshold),
                'events': resize_events,
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
        clean_correct_and_not_target = 0

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
                clean_correct_and_not_target += int((clean_preds != float(target_label)).sum().item())

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
            'clean_not_target_count': clean_correct_and_not_target,
            'conditional_attack_success_rate': (
                (attack_success / clean_correct_and_not_target) * 100
                if clean_correct_and_not_target else 0.0
            ),
            'target_label': float(target_label),
            'trigger_box': trigger_box,
            'source_filter': source_filter if source_filter is not None else ('bad' if source_only_bad else 'all'),
        }


    @staticmethod
    def _normalize_patch_size(patch_size):
        if isinstance(patch_size, int):
            return int(patch_size), int(patch_size)
        if len(patch_size) != 2:
            raise ValueError('patch_size must be an int or a (width, height) pair.')
        width, height = int(patch_size[0]), int(patch_size[1])
        if width <= 0 or height <= 0:
            raise ValueError('patch_size values must be positive integers.')
        return width, height

    def _infer_full_patch_size(self, data_loader, fallback):
        dataset = getattr(data_loader, 'dataset', None)
        image_size = getattr(dataset, 'image_size', None)
        if image_size is not None:
            return self._normalize_patch_size(image_size)
        try:
            sample_inputs, _ = next(iter(data_loader))
            return int(sample_inputs.shape[-1]), int(sample_inputs.shape[-2])
        except StopIteration:
            return self._normalize_patch_size(fallback)

    @staticmethod
    def _resize_trigger_boxes(anchor_boxes, width, height, image_size):
        image_width, image_height = int(image_size[0]), int(image_size[1])
        resized_boxes = []
        for box in anchor_boxes:
            anchor_w = int(box['width'])
            anchor_h = int(box['height'])
            center_x = int(box['x']) + anchor_w / 2.0
            center_y = int(box['y']) + anchor_h / 2.0
            x = int(round(center_x - width / 2.0))
            y = int(round(center_y - height / 2.0))
            x = max(0, min(x, image_width - int(width)))
            y = max(0, min(y, image_height - int(height)))
            resized_box = dict(box)
            resized_box.update({
                'x': int(x),
                'y': int(y),
                'width': int(width),
                'height': int(height),
            })
            resized_boxes.append(resized_box)
        return resized_boxes

    @staticmethod
    def _random_trigger_boxes(batch_size, patch_width, patch_height, image_width, image_height):
        if patch_width > image_width or patch_height > image_height:
            raise ValueError('Patch size cannot exceed image dimensions.')
        max_x = int(image_width - patch_width)
        max_y = int(image_height - patch_height)
        xs = torch.randint(0, max_x + 1, (batch_size,)).tolist() if max_x > 0 else [0] * batch_size
        ys = torch.randint(0, max_y + 1, (batch_size,)).tolist() if max_y > 0 else [0] * batch_size
        return [
            {'x': int(x), 'y': int(y), 'width': int(patch_width), 'height': int(patch_height)}
            for x, y in zip(xs, ys)
        ]

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
    def _inject_trigger(
        inputs,
        trigger_box,
        trigger_value=(1.0, 1.0, 1.0),
        trigger_patch=None,
        trigger_mask=None,
        edge_softness=0.2):
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

        per_sample_boxes = len(trigger_boxes) == poisoned_inputs.shape[0]
        if per_sample_boxes and patch_bank is not None:
            patch_count_valid = patch_bank.shape[0] in (1, poisoned_inputs.shape[0])
            mask_count_valid = mask_bank is None or mask_bank.shape[0] in (1, poisoned_inputs.shape[0])
            if patch_count_valid and mask_count_valid:
                for sample_idx, box in enumerate(trigger_boxes):
                    x = int(box['x'])
                    y = int(box['y'])
                    width = int(box['width'])
                    height = int(box['height'])

                    if x < 0 or y < 0 or x + width > input_w or y + height > input_h:
                        raise ValueError('trigger_box is out of image bounds.')

                    region = poisoned_inputs[sample_idx:sample_idx + 1, :, y:y + height, x:x + width].clone()
                    if patch_bank.shape[1] != channels or patch_bank.shape[2] != height or patch_bank.shape[3] != width:
                        raise ValueError('trigger_patch shape must match (C, height, width) from trigger_box.')
                    patch = patch_bank[0 if patch_bank.shape[0] == 1 else sample_idx].unsqueeze(0)

                    if mask_bank is not None:
                        if mask_bank.shape[1] != channels or mask_bank.shape[2] != height or mask_bank.shape[3] != width:
                            raise ValueError('trigger_mask shape must match (C, height, width) from trigger_box.')
                        blend_mask = mask_bank[0 if mask_bank.shape[0] == 1 else sample_idx].unsqueeze(0)
                        blend_mask = torch.clamp(blend_mask, 0.0, 1.0)
                    else:
                        blend_mask = AdversarialAttack._build_blend_mask(
                            height=height,
                            width=width,
                            channels=channels,
                            device=poisoned_inputs.device,
                            dtype=poisoned_inputs.dtype,
                            edge_softness=edge_softness,
                        )

                    poisoned_inputs[sample_idx:sample_idx + 1, :, y:y + height, x:x + width] = torch.clamp(
                        region + patch * blend_mask,
                        0.0,
                        1.0,
                    )
                return poisoned_inputs

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
                blended_region = torch.clamp(region + patch * blend_mask, 0.0, 1.0)
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
