import torch


from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack
from pathlib import Path
from typing import Callable, Optional, Sequence

class PatchAttck(AdversarialAttack):
    def __init__(self,
                patch_size: int, 
                model: Callable,
                device: Optional[str] = None,
                use_multi_gpu: bool = True,
                gpu_ids: Optional[Sequence[int]] = None):
        
        self.patch_size = patch_size
        super().__init__(model)
        pass

    def learn_fixed_size_patch(self,
                        dataset, 
                        data_loader, 
                        val_loader, 
                        target_label, 
                        source_filter, 
                        steps,
                        learning_rate,
                        asr_hardening_threshold,
                        log_interval=1,
                        trigger_preview_interval=10,
                        trigger_preview_dir='backups/fixed_size_adversarial_trigger_previews',
                        trigger_preview_loader=None,
                        trigger_preview_max_images=5,
                        randomize_training_location=True,
                        how_to_attach='blend',
                        patch_count=1,
                        ):
        natural_trigger = dataset.find_natural_trigger_candidates(
            window_size=self.patch_size,
            stride=8,
            max_samples_per_group=1000,
            top_k=10)

        print('Natural trigger candidates (bad vs good):')
        for candidate in natural_trigger['top_candidates']:
            print(candidate)

        requested_patch_count = max(1, patch_count)
        selected_trigger_boxes = self._select_non_overlapping_boxes(
            natural_trigger['top_candidates'],
            max_count=requested_patch_count,
        )

        self.model.eval()

        trigger_boxes = self._normalize_trigger_boxes(selected_trigger_boxes)
        validation_anchor_boxes = [dict(box) for box in trigger_boxes]
        base_box = validation_anchor_boxes[0]

        width = int(base_box['width'])
        height = int(base_box['height'])
        for candidate_box in validation_anchor_boxes[1:]:
            if int(candidate_box['width']) != width or int(candidate_box['height']) != height:
                raise ValueError('All trigger_boxes must have identical width/height for universal trigger learning.')

        channels = 3
        trigger_delta = torch.zeros((len(validation_anchor_boxes), channels, height, width), device=self.device)
        trigger_delta.requires_grad_()

        patch_optimizer = torch.optim.Adam([trigger_delta], lr=learning_rate)

        preview_records = []
        preview_interval = int(trigger_preview_interval) if trigger_preview_interval is not None else 0
        preview_max_images = (
            max(0, int(trigger_preview_max_images))
            if trigger_preview_max_images is not None else 0
        )
        preview_output_dir = Path(trigger_preview_dir) if trigger_preview_dir is not None else None
        if preview_interval > 0 and preview_output_dir is not None:
            preview_output_dir.mkdir(parents=True, exist_ok=True)
        preview_data_loader = trigger_preview_loader or val_loader or data_loader

        history = []
        best_patch = None
        best_trigger_boxes = [dict(box) for box in trigger_boxes]
        best_step = 0
        best_val_loss = float('inf')
        best_val_asr = float('-inf')

        for step_idx in range(steps):
            step_losses = []
            step_attack_losses = []

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

                poisoned_inputs = self._inject_trigger(
                    selected_inputs,
                    training_trigger_boxes,
                    trigger_patch=training_patch,
                    edge_softness=0.0,
                    how_to_attach=how_to_attach, # ['blend, 'replace']
                )

                outputs = self.model(poisoned_inputs)
                target_tensor = torch.full_like(outputs, float(target_label))
                attack_loss = self.cost_function(outputs, target_tensor)
                loss = attack_loss 

                if patch_optimizer is not None:
                    patch_optimizer.zero_grad()
                elif trigger_delta.grad is not None:
                    trigger_delta.grad.zero_()

                loss.backward()
                patch_optimizer.step()


                batch_samples = int(outputs.shape[0])
                step_losses.append(float(loss.item()) * batch_samples)
                step_attack_losses.append(float(attack_loss.item()) * batch_samples)
                step_samples += batch_samples

            step_loss = (sum(step_losses) / step_samples) if step_samples else 0.0
            step_attack_loss = (sum(step_attack_losses) / step_samples) if step_samples else 0.0
          
            current_patch_for_metrics = torch.tanh(trigger_delta).detach()
            patch_update_l2 = float(torch.norm(
                (current_patch_for_metrics - previous_patch).reshape(-1),
                p=2,
            ).item())
            patch_l1_norm = float(torch.norm(current_patch_for_metrics.reshape(-1), p=1).item())
            patch_l2_norm = float(torch.norm(current_patch_for_metrics.reshape(-1), p=2).item())
            patch_linf_norm = float(torch.norm(current_patch_for_metrics.reshape(-1), p=float('inf')).item())
           
            step_history = {
                'step': step_idx + 1,
                'loss': step_loss,
                'attack_loss': step_attack_loss,
                'samples': step_samples,
                'patch_update_l2': patch_update_l2,
                'patch_l1_norm': patch_l1_norm,
                'patch_l2_norm': patch_l2_norm,
                'patch_linf_norm': patch_linf_norm,
            }

            if val_loader is not None:

                current_patch = torch.tanh(trigger_delta).detach()

                val_metrics = self.evaluate_attack_success(
                    test_loader=val_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=current_patch,
                    target_label=target_label,
                    source_filter=source_filter,
                    edge_softness=0.0,
                    how_to_attach=how_to_attach
                )
                val_loss_metrics = self.evaluate_trigger_loss(
                    data_loader=val_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=current_patch,
                    target_label=target_label,
                    source_filter=source_filter,
                    edge_softness=0.0,
                    how_to_attach=how_to_attach
                )
                val_asr = float(val_metrics['attack_success_rate'])
                val_loss = float(val_loss_metrics['loss'])
                step_history['validation_asr'] = val_asr
                step_history['validation_loss'] = val_loss
                step_history['validation_samples'] = int(val_loss_metrics['samples_evaluated'])

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_asr = val_asr
                    best_patch = current_patch.cpu().clone()
                    best_trigger_boxes = [dict(box) for box in trigger_boxes]
                    best_step = step_idx + 1


            if (
                preview_interval > 0
                and preview_output_dir is not None
                and (step_idx + 1) % preview_interval == 0
            ):
                step_preview_records = self._save_trigger_preview(
                    data_loader=preview_data_loader,
                    output_dir=preview_output_dir,
                    step=step_idx + 1,
                    trigger_box=trigger_boxes,
                    trigger_patch=current_patch_for_metrics,
                    trigger_mask=None,
                    target_label=target_label,
                    source_filter=source_filter,
                    edge_softness=0.0,
                    max_images=preview_max_images,
                    how_to_attach=how_to_attach
                )
                if step_preview_records:
                    preview_records.extend(step_preview_records)
                    step_history['trigger_previews'] = step_preview_records

            history.append(step_history)

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                val_log = ''
                train_log = ''
                if val_loader is not None:
                    val_log = (
                        f', val_loss={step_history.get("validation_loss", 0.0):.6f}'
                        f', val_asr={step_history.get("validation_asr", 0.0):.4f}'
                    )
                print(
                    f'[Trigger Learning] step={step_idx + 1}/{steps}, '
                    f'loss={step_loss:.6f}, attack_loss={step_attack_loss:.6f}, '
                    f'patch_update_l2={patch_update_l2:.6f}, '
                    f'patch_l1_norm={patch_l1_norm:.6f}, patch_l2_norm={patch_l2_norm:.6f}, '
                    f'patch_linf_norm={patch_linf_norm:.6f},'
                    f'{train_log}{val_log}'
                )
                if step_samples == 0:
                    print(
                        '[Trigger Learning] warning: no samples matched source_filter '
                        f'"{source_filter}" at this step.'
                    )

        selection = 'last_step'
        if best_patch is not None:
            learned_patch = best_patch
            trigger_boxes = best_trigger_boxes
            selected_step = best_step
            selection = 'best_validation_loss'
        else:
            learned_patch = torch.tanh(trigger_delta).detach().cpu()
            selected_step = steps
            selection = 'last_step'

        output_path = Path(f'{preview_output_dir}/saved_trigger')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        history_path = self._default_trigger_history_path(output_path)

        if not torch.is_tensor(learned_patch):
            learned_patch = torch.tensor(learned_patch, dtype=torch.float32)

        torch.save(
            {
                'patch': learned_patch.detach().cpu(),
                'trigger_box': trigger_boxes[0],
                'trigger_boxes': trigger_boxes,
                'target_label': float(target_label),
                'source_filter': source_filter,
                'selection': selection,
                'selected_step': int(selected_step),
                'best_validation_loss': None if val_loader is None else float(best_val_loss),
                'best_validation_asr': None if val_loader is None else float(best_val_asr),
                'history_path': str(history_path),
            },
            output_path,
        )

        history_payload = {
            'history': history,
            'selection': selection,
            'selected_step': selected_step,
            'best_validation_loss': best_validation_loss,
            'best_validation_asr': best_validation_asr,
            'patch_path': str(output_path),
        }
        with open(history_path, 'w', encoding='utf-8') as history_file:
            json.dump(self._json_safe(history_payload), history_file, indent=2)

        dataset.save_trigger_visualizations(
                trigger_analysis=natural_trigger,
                output_dir=f'{preview_output_dir}/trigger_visualization',
                num_examples=20,
                trigger_box=trigger_boxes,
                trigger_delta=learned_trigger['patch'],
                model=self.model,
                target_label=target_label,
                source_filter=source_filter,
                only_successful_poisoned=True,
            )
        print('Saved trigger visualizations to trigger_visualization/')
    
        return {
            'patch': learned_patch,
            'history': history,
            'trigger_box': trigger_boxes[0],
            'trigger_boxes': trigger_boxes,
            'target_label': float(target_label),
            'source_filter': source_filter,
            'trigger_previews': preview_records,
            'selection': selection,
            'selected_step': int(selected_step),
            'best_validation_loss': None if val_loader is None else float(best_val_loss),
            'best_validation_asr': None if val_loader is None else float(best_val_asr),
        }

    def learn_fixed_size_patch_with_mask_optimization(self, 
                                                    dataset,
                                                    data_loader,
                                                    val_loader,
                                                    target_label, 
                                                    source_filter, 
                                                    steps,
                                                    learning_rate,
                                                    mask_learning_rate,
                                                    mask_l1_weight,
                                                    patch_l2_weight,
                                                    trigger_preview_dir,
                                                    trigger_preview_loader,
                                                    trigger_preview_max_images,
                                                    how_to_attach,
                                                    patch_count):
       
        natural_trigger = dataset.find_natural_trigger_candidates(
            window_size=self.patch_size,
            stride=8,
            max_samples_per_group=1000,
            top_k=10)

        print('Natural trigger candidates (bad vs good):')
        for candidate in natural_trigger['top_candidates']:
            print(candidate)

        requested_patch_count = max(1, patch_count)
        selected_trigger_boxes = self._select_non_overlapping_boxes(
            natural_trigger['top_candidates'],
            max_count=requested_patch_count,
        )

        return self.learn_universal_trigger(
            data_loader,
            selected_trigger_boxes,
            target_label=target_label,
            source_filter=source_filter,
            validation_loader=val_loader,
            report_training_asr=False,
            steps=steps,
            learning_rate=learning_rate,
            mask_learning_rate=mask_learning_rate,
            optimize_mask=True,
            initial_edge_softness=0.0,
            min_edge_softness=0.0,
            softness_decay=0.0,
            softness_patience=0,
            asr_hardening_threshold=70.0,
            mask_l1_weight=mask_l1_weight,
            patch_l2_weight=patch_l2_weight,
            softness_alignment_weight=0.0,
            patch_update_method='adam',
            log_interval=1,
            trigger_preview_interval=10,
            trigger_preview_dir=trigger_preview_dir,
            trigger_preview_loader=trigger_preview_loader,
            trigger_preview_max_images=trigger_preview_max_images,
            progressive_resize=False,
            randomize_training_location=False,
            enable_compression_phase=False,
            how_to_attach=how_to_attach)

    @staticmethod
    def _boxes_overlap(box_a, box_b):
        ax1, ay1 = int(box_a['x']), int(box_a['y'])
        ax2, ay2 = ax1 + int(box_a['width']), ay1 + int(box_a['height'])
        bx1, by1 = int(box_b['x']), int(box_b['y'])
        bx2, by2 = bx1 + int(box_b['width']), by1 + int(box_b['height'])
        return (ax1 < bx2) and (ax2 > bx1) and (ay1 < by2) and (ay2 > by1)

    @staticmethod
    def _select_non_overlapping_boxes(candidates, max_count):
        selected = []
        for candidate in candidates:
            overlaps_existing = any(self._boxes_overlap(candidate, chosen) for chosen in selected)
            if overlaps_existing:
                continue
            selected.append(candidate)
            if len(selected) >= max_count:
                break
        return selected
    