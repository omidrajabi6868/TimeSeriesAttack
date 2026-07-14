from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image, ImageDraw
from .InputPurification import FeatureDistillation
from .DiffusionPurification import DiffusionPurifier
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack

class Defender:
    def __init__(self,
                model,
                dataset,
                val_loader,
                calibration_loader=None,
                device: Optional[str] = None,
                use_multi_gpu: bool = True,
                gpu_ids: Optional[Sequence[int]] = None):

        self.dataset = dataset
        self.val_loader = val_loader
        self.calibration_loader = calibration_loader if calibration_loader is not None else val_loader
        self.model = model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None

        self.model = self._prepare_data_parallel_model(
            self.model,
            model_name="classifier",
        )
        return

    def _unwrap_data_parallel(self, model):
        """Return the underlying module and its DataParallel device IDs.

        Loading code may already wrap checkpoints in DataParallel. Wrapping an
        existing DataParallel module again creates nested replicas, which can
        feed cuda:1 inputs to inner modules whose parameters still live on
        cuda:0. Unwrapping first guarantees there is only one DataParallel layer.
        """
        device_ids = None
        while isinstance(model, torch.nn.DataParallel):
            if device_ids is None:
                device_ids = list(model.device_ids)
            model = model.module
        return model, device_ids

    def _prepare_data_parallel_model(self, model, model_name="model"):
        model, existing_device_ids = self._unwrap_data_parallel(model)
        if self.gpu_ids is None and existing_device_ids:
            self.gpu_ids = existing_device_ids

        model = model.to(self.device)
        if (
            self.use_multi_gpu
            and self.device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            dp_gpu_ids = self.gpu_ids if self.gpu_ids is not None else list(range(torch.cuda.device_count()))
            if len(dp_gpu_ids) > 1:
                print(f"Using DataParallel for {model_name} on GPUs: {dp_gpu_ids}")
                model = torch.nn.DataParallel(model, device_ids=dp_gpu_ids)
        return model

    def feature_distillation(self,
                            trigger_path,
                            source_filter='bad',
                            how_to_attach='blend',
                            block=8, QS=50.0, preserve_ratio=0.5, fd_batch_size=32, fd_max_blocks_per_chunk=65536,
                            save_examples_dir=None, max_saved_examples=5):

        learned_trigger = AdversarialAttack.load_trigger(trigger_path)
        target_label = float(learned_trigger['target_label'])

        std_map = FeatureDistillation.compute_dct_statistics(
            self.calibration_loader,
            block=block,
            max_blocks_per_chunk=fd_max_blocks_per_chunk,
            output_device=self.device,
        )

        fd = FeatureDistillation(std_map=std_map, block=block, quality=QS, preserve_ratio=preserve_ratio).to(self.device)
        fd.eval()

        total = 0
        attack_success = 0
        asr_after_defend = 0
        clean_correct = 0
        fd_correct = 0
        clean_correct_and_not_target = 0
        conditional_attack_success = 0
        conditional_asr_after_defend = 0
        clean_prediction_changes_after_fd = 0
        poisoned_prediction_changes_after_fd = 0
        clean_fd_abs_diff_sum = 0.0
        clean_fd_pixel_count = 0
        poisoned_fd_abs_diff_sum = 0.0
        poisoned_fd_pixel_count = 0
        trigger_region_fd_abs_diff_sum = 0.0
        trigger_region_fd_pixel_count = 0
        clean_fd_max_abs_input_change = 0.0
        poisoned_fd_max_abs_input_change = 0.0
        clean_output_abs_diff_sum = 0.0
        clean_output_count = 0
        poisoned_output_abs_diff_sum = 0.0
        poisoned_output_count = 0
        successful_defense_examples = []
        unsuccessful_defense_examples = []
        max_saved_examples = int(max_saved_examples)

        for inputs, targets in self.val_loader:
            self.model.eval()
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

            source_inputs = inputs[source_mask]
            source_targets = targets[source_mask]

            with torch.no_grad():
                clean_outputs = self.model(source_inputs)
            clean_preds = (clean_outputs > 0).float().view(-1)
            clean_targets = source_targets.view(-1)
            clean_correct += int((clean_preds == clean_targets).sum().item())
            eligible_mask = (clean_preds == clean_targets) & (clean_preds != target_label)
            clean_correct_and_not_target += int(eligible_mask.sum().item())

            fd_clean_preds = []
            fd_clean_input_batches = []
            with torch.no_grad():
                for start in range(0, source_inputs.shape[0], fd_batch_size):
                    end = min(start + fd_batch_size, source_inputs.shape[0])
                    fd_clean_inputs = fd(source_inputs[start:end].clone())
                    fd_clean_input_batches.append(fd_clean_inputs.detach().cpu())
                    clean_input_diff = (fd_clean_inputs - source_inputs[start:end]).abs()
                    clean_fd_abs_diff_sum += float(
                        clean_input_diff.sum().item()
                    )
                    clean_fd_max_abs_input_change = max(
                        clean_fd_max_abs_input_change,
                        float(clean_input_diff.max().item()),
                    )
                    clean_fd_pixel_count += int(fd_clean_inputs.numel())
                    fd_outputs = self.model(fd_clean_inputs)
                    clean_output_abs_diff_sum += float(
                        (fd_outputs.view(-1) - clean_outputs[start:end].view(-1)).abs().sum().item()
                    )
                    clean_output_count += int(fd_outputs.numel())
                    fd_clean_preds.append((fd_outputs > 0).float().view(-1))

            fd_preds = torch.cat(fd_clean_preds, dim=0)
            fd_clean_all = torch.cat(fd_clean_input_batches, dim=0)
            fd_correct += int((fd_preds == clean_targets).sum().item())
            clean_prediction_changes_after_fd += int((fd_preds != clean_preds).sum().item())

            poisoned_inputs = AdversarialAttack._inject_trigger(
                source_inputs.clone(),
                learned_trigger['trigger_boxes'],
                trigger_value=None,
                trigger_patch=learned_trigger['patch'],
                trigger_mask=learned_trigger['mask'],
                edge_softness=learned_trigger['softness'],
                how_to_attach=how_to_attach
            )

            with torch.no_grad():
                poisoned_outputs = self.model(poisoned_inputs.clone())
            poisoned_preds = (poisoned_outputs > 0).float().view(-1)

            attack_success += int((poisoned_preds == target_label).sum().item())

            fd_poisoned_preds_by_batch = []
            fd_poisoned_input_batches = []
            with torch.no_grad():
                for start in range(0, poisoned_inputs.shape[0], fd_batch_size):
                    end = min(start + fd_batch_size, poisoned_inputs.shape[0])
                    fd_poisoned_inputs = fd(poisoned_inputs[start:end].clone())
                    fd_poisoned_input_batches.append(fd_poisoned_inputs.detach().cpu())
                    poisoned_input_diff = (fd_poisoned_inputs - poisoned_inputs[start:end]).abs()
                    poisoned_fd_abs_diff_sum += float(
                        poisoned_input_diff.sum().item()
                    )
                    poisoned_fd_max_abs_input_change = max(
                        poisoned_fd_max_abs_input_change,
                        float(poisoned_input_diff.max().item()),
                    )
                    poisoned_fd_pixel_count += int(fd_poisoned_inputs.numel())

                    for box in AdversarialAttack._normalize_trigger_boxes(learned_trigger['trigger_boxes']):
                        x = int(box['x'])
                        y = int(box['y'])
                        width = int(box['width'])
                        height = int(box['height'])
                        before_region = poisoned_inputs[start:end, :, y:y + height, x:x + width]
                        after_region = fd_poisoned_inputs[:, :, y:y + height, x:x + width]
                        trigger_region_fd_abs_diff_sum += float(
                            (after_region - before_region).abs().sum().item()
                        )
                        trigger_region_fd_pixel_count += int(after_region.numel())

                    fd_outputs = self.model(fd_poisoned_inputs)
                    poisoned_output_abs_diff_sum += float(
                        (fd_outputs.view(-1) - poisoned_outputs[start:end].view(-1)).abs().sum().item()
                    )
                    poisoned_output_count += int(fd_outputs.numel())
                    fd_poisoned_preds_by_batch.append((fd_outputs > 0).float().view(-1))

            fd_poisoned_preds = torch.cat(fd_poisoned_preds_by_batch, dim=0)
            fd_poisoned_all = torch.cat(fd_poisoned_input_batches, dim=0)
            asr_after_defend += int((fd_poisoned_preds == target_label).sum().item())
            poisoned_prediction_changes_after_fd += int((fd_poisoned_preds != poisoned_preds).sum().item())
            conditional_attack_success += int((poisoned_preds[eligible_mask] == target_label).sum().item())
            conditional_asr_after_defend += int((fd_poisoned_preds[eligible_mask] == target_label).sum().item())

            if save_examples_dir is not None and max_saved_examples > 0:
                successful_mask = (poisoned_preds == target_label) & (fd_poisoned_preds != target_label)
                unsuccessful_mask = (poisoned_preds == target_label) & (fd_poisoned_preds == target_label)
                self._collect_fd_examples(
                    successful_defense_examples,
                    successful_mask,
                    max_saved_examples,
                    source_inputs.detach().cpu(),
                    fd_clean_all,
                    poisoned_inputs.detach().cpu(),
                    fd_poisoned_all,
                    clean_targets.detach().cpu(),
                    clean_preds.detach().cpu(),
                    fd_preds.detach().cpu(),
                    poisoned_preds.detach().cpu(),
                    fd_poisoned_preds.detach().cpu(),
                    defended=True,
                )
                if len(successful_defense_examples) < max_saved_examples:
                    self._collect_fd_examples(
                        unsuccessful_defense_examples,
                        unsuccessful_mask,
                        max_saved_examples,
                        source_inputs.detach().cpu(),
                        fd_clean_all,
                        poisoned_inputs.detach().cpu(),
                        fd_poisoned_all,
                        clean_targets.detach().cpu(),
                        clean_preds.detach().cpu(),
                        fd_preds.detach().cpu(),
                        poisoned_preds.detach().cpu(),
                        fd_poisoned_preds.detach().cpu(),
                        defended=False,
                    )
            total += int(poisoned_preds.shape[0])

        saved_example_info = None
        if save_examples_dir is not None and max_saved_examples > 0:
            examples_to_save = successful_defense_examples[:max_saved_examples]
            example_source = 'successful_defenses'
            if not examples_to_save:
                examples_to_save = unsuccessful_defense_examples[:max_saved_examples]
                example_source = 'unsuccessful_defenses'
            saved_example_info = self._save_fd_examples(
                examples_to_save,
                save_examples_dir,
                example_source,
            )

        clean_source_accuracy = (clean_correct / total) * 100 if total else 0.0
        clean_fd_accuracy = (fd_correct / total) * 100 if total else 0.0
        attack_success_rate = (attack_success / total) * 100 if total else 0.0
        defended_attack_success_rate = (asr_after_defend / total) * 100 if total else 0.0
        conditional_attack_success_rate = (
            (conditional_attack_success / clean_correct_and_not_target) * 100
            if clean_correct_and_not_target else 0.0
        )
        conditional_defended_attack_success_rate = (
            (conditional_asr_after_defend / clean_correct_and_not_target) * 100
            if clean_correct_and_not_target else 0.0
        )

        return {
            'samples_evaluated': total,
            'clean_source_accuracy': clean_source_accuracy,
            'clean_fd_accuracy': clean_fd_accuracy,
            'clean_accuracy_change_after_fd': clean_fd_accuracy - clean_source_accuracy,
            'clean_prediction_changes_after_fd': clean_prediction_changes_after_fd,
            'clean_fd_mean_abs_input_change': (
                clean_fd_abs_diff_sum / clean_fd_pixel_count
                if clean_fd_pixel_count else 0.0
            ),
            'clean_fd_max_abs_input_change': clean_fd_max_abs_input_change,
            'clean_fd_mean_abs_output_change': (
                clean_output_abs_diff_sum / clean_output_count
                if clean_output_count else 0.0
            ),
            'attack_success_rate': attack_success_rate,
            'asr_after_defend': defended_attack_success_rate,
            'asr_reduction_after_defend': attack_success_rate - defended_attack_success_rate,
            'poisoned_prediction_changes_after_fd': poisoned_prediction_changes_after_fd,
            'poisoned_fd_mean_abs_input_change': (
                poisoned_fd_abs_diff_sum / poisoned_fd_pixel_count
                if poisoned_fd_pixel_count else 0.0
            ),
            'poisoned_fd_max_abs_input_change': poisoned_fd_max_abs_input_change,
            'poisoned_fd_mean_abs_output_change': (
                poisoned_output_abs_diff_sum / poisoned_output_count
                if poisoned_output_count else 0.0
            ),
            'trigger_region_fd_mean_abs_input_change': (
                trigger_region_fd_abs_diff_sum / trigger_region_fd_pixel_count
                if trigger_region_fd_pixel_count else 0.0
            ),
            'clean_not_target_count': clean_correct_and_not_target,
            'conditional_attack_success_rate': conditional_attack_success_rate,
            'conditional_asr_after_defend': conditional_defended_attack_success_rate,
            'conditional_asr_reduction_after_defend': (
                conditional_attack_success_rate - conditional_defended_attack_success_rate
            ),
            'fd_quality': fd.quality,
            'fd_preserve_ratio': fd.preserve_ratio,
            'fd_preserved_coefficients': int(fd.accuracy_sensitive_mask.sum().item()),
            'fd_total_coefficients': int(fd.accuracy_sensitive_mask.numel()),
            'fd_quantization_table': fd.quantization_table.detach().cpu().tolist(),
            'target_label': target_label,
            'trigger_box': learned_trigger['trigger_boxes'],
            'trigger_coverage_ratio': self._trigger_coverage_ratio(
                learned_trigger['trigger_boxes'],
                image_height=self.dataset.image_size[1] if getattr(self.dataset, 'image_size', None) else None,
                image_width=self.dataset.image_size[0] if getattr(self.dataset, 'image_size', None) else None,
            ),
            'saved_feature_distillation_examples': saved_example_info,
        }


    def diffusion_purification(self,
                            trigger_path,
                            diffusion_checkpoint_path,
                            source_filter='bad',
                            how_to_attach='blend',
                            diffusion_step=100,
                            reverse_steps=None,
                            stochastic=True,
                            dp_batch_size=16,
                            save_examples_dir=None,
                            max_saved_examples=5):
        """Evaluate a DiffPure-style defense using a trained diffusion checkpoint.

        The purifier first applies the forward diffusion process to each clean or
        poisoned input at ``diffusion_step`` and then runs the learned reverse
        denoising process before the classifier sees the image.
        """
        learned_trigger = AdversarialAttack.load_trigger(trigger_path)
        target_label = float(learned_trigger['target_label'])
        purifier = DiffusionPurifier.from_checkpoint(diffusion_checkpoint_path, map_location=self.device).to(self.device)
        purifier.model = self._prepare_data_parallel_model(
            purifier.model,
            model_name="diffusion purifier",
        )
        purifier.eval()

        total = 0
        attack_success = 0
        asr_after_defend = 0
        clean_correct = 0
        purified_clean_correct = 0
        clean_correct_and_not_target = 0
        conditional_attack_success = 0
        conditional_asr_after_defend = 0
        clean_prediction_changes_after_dp = 0
        poisoned_prediction_changes_after_dp = 0
        clean_dp_abs_diff_sum = 0.0
        clean_dp_pixel_count = 0
        poisoned_dp_abs_diff_sum = 0.0
        poisoned_dp_pixel_count = 0
        trigger_region_dp_abs_diff_sum = 0.0
        trigger_region_dp_pixel_count = 0
        successful_defense_examples = []
        unsuccessful_defense_examples = []
        max_saved_examples = int(max_saved_examples)

        for inputs, targets in self.val_loader:
            self.model.eval()
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

            source_inputs = inputs[source_mask]
            source_targets = targets[source_mask]
            clean_targets = source_targets.view(-1)

            with torch.no_grad():
                clean_outputs = self.model(source_inputs)
            clean_preds = (clean_outputs > 0).float().view(-1)
            clean_correct += int((clean_preds == clean_targets).sum().item())
            eligible_mask = (clean_preds == clean_targets) & (clean_preds != target_label)
            clean_correct_and_not_target += int(eligible_mask.sum().item())

            purified_clean_preds = []
            purified_clean_batches = []
            with torch.no_grad():
                for start in range(0, source_inputs.shape[0], dp_batch_size):
                    end = min(start + dp_batch_size, source_inputs.shape[0])
                    purified_clean = purifier.purify(
                        source_inputs[start:end].clone(),
                        diffusion_step=diffusion_step,
                        reverse_steps=reverse_steps,
                        stochastic=stochastic,
                    )
                    purified_clean_batches.append(purified_clean.detach().cpu())
                    clean_diff = (purified_clean - source_inputs[start:end]).abs()
                    clean_dp_abs_diff_sum += float(clean_diff.sum().item())
                    clean_dp_pixel_count += int(purified_clean.numel())
                    purified_clean_outputs = self.model(purified_clean)
                    purified_clean_preds.append((purified_clean_outputs > 0).float().view(-1))

            dp_clean_preds = torch.cat(purified_clean_preds, dim=0)
            purified_clean_all = torch.cat(purified_clean_batches, dim=0)
            purified_clean_correct += int((dp_clean_preds == clean_targets).sum().item())
            clean_prediction_changes_after_dp += int((dp_clean_preds != clean_preds).sum().item())

            poisoned_inputs = AdversarialAttack._inject_trigger(
                source_inputs.clone(),
                learned_trigger['trigger_boxes'],
                trigger_value=None,
                trigger_patch=learned_trigger['patch'],
                trigger_mask=learned_trigger['mask'],
                edge_softness=learned_trigger['softness'],
                how_to_attach=how_to_attach
            )

            with torch.no_grad():
                poisoned_outputs = self.model(poisoned_inputs.clone())
            poisoned_preds = (poisoned_outputs > 0).float().view(-1)
            attack_success += int((poisoned_preds == target_label).sum().item())

            purified_poisoned_preds = []
            purified_poisoned_batches = []
            with torch.no_grad():
                for start in range(0, poisoned_inputs.shape[0], dp_batch_size):
                    end = min(start + dp_batch_size, poisoned_inputs.shape[0])
                    purified_poisoned = purifier.purify(
                        poisoned_inputs[start:end].clone(),
                        diffusion_step=diffusion_step,
                        reverse_steps=reverse_steps,
                        stochastic=stochastic,
                    )
                    purified_poisoned_batches.append(purified_poisoned.detach().cpu())
                    poisoned_diff = (purified_poisoned - poisoned_inputs[start:end]).abs()
                    poisoned_dp_abs_diff_sum += float(poisoned_diff.sum().item())
                    poisoned_dp_pixel_count += int(purified_poisoned.numel())
                    for box in AdversarialAttack._normalize_trigger_boxes(learned_trigger['trigger_boxes']):
                        x = int(box['x'])
                        y = int(box['y'])
                        width = int(box['width'])
                        height = int(box['height'])
                        before_region = poisoned_inputs[start:end, :, y:y + height, x:x + width]
                        after_region = purified_poisoned[:, :, y:y + height, x:x + width]
                        trigger_region_dp_abs_diff_sum += float((after_region - before_region).abs().sum().item())
                        trigger_region_dp_pixel_count += int(after_region.numel())
                    purified_poisoned_outputs = self.model(purified_poisoned)
                    purified_poisoned_preds.append((purified_poisoned_outputs > 0).float().view(-1))

            dp_poisoned_preds = torch.cat(purified_poisoned_preds, dim=0)
            purified_poisoned_all = torch.cat(purified_poisoned_batches, dim=0)
            asr_after_defend += int((dp_poisoned_preds == target_label).sum().item())
            poisoned_prediction_changes_after_dp += int((dp_poisoned_preds != poisoned_preds).sum().item())
            conditional_attack_success += int((poisoned_preds[eligible_mask] == target_label).sum().item())
            conditional_asr_after_defend += int((dp_poisoned_preds[eligible_mask] == target_label).sum().item())

            if save_examples_dir is not None and max_saved_examples > 0:
                successful_mask = (poisoned_preds == target_label) & (dp_poisoned_preds != target_label)
                unsuccessful_mask = (poisoned_preds == target_label) & (dp_poisoned_preds == target_label)
                self._collect_dp_examples(
                    successful_defense_examples,
                    successful_mask,
                    max_saved_examples,
                    source_inputs.detach().cpu(),
                    purified_clean_all,
                    poisoned_inputs.detach().cpu(),
                    purified_poisoned_all,
                    clean_targets.detach().cpu(),
                    clean_preds.detach().cpu(),
                    dp_clean_preds.detach().cpu(),
                    poisoned_preds.detach().cpu(),
                    dp_poisoned_preds.detach().cpu(),
                    defended=True,
                )
                if len(successful_defense_examples) < max_saved_examples:
                    self._collect_dp_examples(
                        unsuccessful_defense_examples,
                        unsuccessful_mask,
                        max_saved_examples,
                        source_inputs.detach().cpu(),
                        purified_clean_all,
                        poisoned_inputs.detach().cpu(),
                        purified_poisoned_all,
                        clean_targets.detach().cpu(),
                        clean_preds.detach().cpu(),
                        dp_clean_preds.detach().cpu(),
                        poisoned_preds.detach().cpu(),
                        dp_poisoned_preds.detach().cpu(),
                        defended=False,
                    )
            total += int(poisoned_preds.shape[0])

        saved_example_info = None
        if save_examples_dir is not None and max_saved_examples > 0:
            examples_to_save = successful_defense_examples[:max_saved_examples]
            example_source = 'successful_defenses'
            if not examples_to_save:
                examples_to_save = unsuccessful_defense_examples[:max_saved_examples]
                example_source = 'unsuccessful_defenses'
            saved_example_info = self._save_dp_examples(
                examples_to_save,
                save_examples_dir,
                example_source,
            )

        clean_source_accuracy = (clean_correct / total) * 100 if total else 0.0
        clean_dp_accuracy = (purified_clean_correct / total) * 100 if total else 0.0
        attack_success_rate = (attack_success / total) * 100 if total else 0.0
        defended_attack_success_rate = (asr_after_defend / total) * 100 if total else 0.0
        conditional_attack_success_rate = ((conditional_attack_success / clean_correct_and_not_target) * 100 if clean_correct_and_not_target else 0.0)
        conditional_defended_attack_success_rate = ((conditional_asr_after_defend / clean_correct_and_not_target) * 100 if clean_correct_and_not_target else 0.0)

        return {
            'samples_evaluated': total,
            'clean_source_accuracy': clean_source_accuracy,
            'clean_dp_accuracy': clean_dp_accuracy,
            'clean_accuracy_change_after_dp': clean_dp_accuracy - clean_source_accuracy,
            'clean_prediction_changes_after_dp': clean_prediction_changes_after_dp,
            'clean_dp_mean_abs_input_change': (clean_dp_abs_diff_sum / clean_dp_pixel_count if clean_dp_pixel_count else 0.0),
            'attack_success_rate': attack_success_rate,
            'asr_after_defend': defended_attack_success_rate,
            'asr_reduction_after_defend': attack_success_rate - defended_attack_success_rate,
            'poisoned_prediction_changes_after_dp': poisoned_prediction_changes_after_dp,
            'poisoned_dp_mean_abs_input_change': (poisoned_dp_abs_diff_sum / poisoned_dp_pixel_count if poisoned_dp_pixel_count else 0.0),
            'trigger_region_dp_mean_abs_input_change': (trigger_region_dp_abs_diff_sum / trigger_region_dp_pixel_count if trigger_region_dp_pixel_count else 0.0),
            'clean_not_target_count': clean_correct_and_not_target,
            'conditional_attack_success_rate': conditional_attack_success_rate,
            'conditional_asr_after_defend': conditional_defended_attack_success_rate,
            'conditional_asr_reduction_after_defend': conditional_attack_success_rate - conditional_defended_attack_success_rate,
            'diffusion_checkpoint_path': str(diffusion_checkpoint_path),
            'diffusion_step': int(diffusion_step),
            'reverse_steps': reverse_steps,
            'stochastic_reverse_process': bool(stochastic),
            'target_label': target_label,
            'trigger_box': learned_trigger['trigger_boxes'],
            'trigger_coverage_ratio': self._trigger_coverage_ratio(
                learned_trigger['trigger_boxes'],
                image_height=self.dataset.image_size[1] if getattr(self.dataset, 'image_size', None) else None,
                image_width=self.dataset.image_size[0] if getattr(self.dataset, 'image_size', None) else None,
            ),
            'saved_diffusion_purification_examples': saved_example_info,
        }

    @staticmethod
    def _collect_dp_examples(example_list, selection_mask, max_examples, clean_inputs,
                             dp_clean_inputs, poisoned_inputs, dp_poisoned_inputs,
                             targets, clean_preds, dp_clean_preds,
                             poisoned_preds, dp_poisoned_preds, defended):
        if len(example_list) >= max_examples:
            return
        for idx in torch.where(selection_mask.cpu())[0].tolist():
            if len(example_list) >= max_examples:
                break
            example_list.append({
                'clean': clean_inputs[idx],
                'clean_dp': dp_clean_inputs[idx],
                'adversarial': poisoned_inputs[idx],
                'adversarial_dp': dp_poisoned_inputs[idx],
                'target': float(targets.view(-1)[idx].item()),
                'clean_pred': float(clean_preds.view(-1)[idx].item()),
                'clean_dp_pred': float(dp_clean_preds.view(-1)[idx].item()),
                'adversarial_pred': float(poisoned_preds.view(-1)[idx].item()),
                'adversarial_dp_pred': float(dp_poisoned_preds.view(-1)[idx].item()),
                'defended': bool(defended),
            })

    @staticmethod
    def _save_dp_examples(examples, output_dir, example_source):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for idx, example in enumerate(examples, start=1):
            save_path = output_path / f'diffusion_purification_{example_source}_{idx:02d}.png'
            Defender._save_dp_comparison(example, save_path)
            saved_paths.append(str(save_path))
        return {
            'output_dir': str(output_path),
            'selection': example_source,
            'saved_images': len(saved_paths),
            'paths': saved_paths,
        }

    @staticmethod
    def _save_dp_comparison(example, save_path):
        panels = [
            ('Clean', example['clean'], example['clean_pred']),
            ('Clean + DP', example['clean_dp'], example['clean_dp_pred']),
            ('|Clean DP diff| x5', Defender._difference_image(example['clean'], example['clean_dp'], scale=5.0), None),
            ('Adversarial', example['adversarial'], example['adversarial_pred']),
            ('Adversarial + DP', example['adversarial_dp'], example['adversarial_dp_pred']),
            ('|Adv DP diff| x5', Defender._difference_image(example['adversarial'], example['adversarial_dp'], scale=5.0), None),
        ]
        images = [Defender._tensor_to_pil_image(tensor) for _, tensor, _ in panels]
        widths, heights = zip(*(image.size for image in images))
        label_height = 42
        canvas = Image.new('RGB', (sum(widths), max(heights) + label_height), color='white')
        draw = ImageDraw.Draw(canvas)
        x_offset = 0
        for (title, _, pred), image in zip(panels, images):
            canvas.paste(image, (x_offset, label_height))
            pred_text = '' if pred is None else f'\ntrue={example["target"]:.0f}, pred={pred:.0f}'
            draw.text(
                (x_offset + 4, 4),
                f'{title}{pred_text}',
                fill='black',
            )
            x_offset += image.size[0]
        canvas.save(save_path)

    @staticmethod
    def _collect_fd_examples(example_list, selection_mask, max_examples, clean_inputs,
                             fd_clean_inputs, poisoned_inputs, fd_poisoned_inputs,
                             targets, clean_preds, fd_clean_preds,
                             poisoned_preds, fd_poisoned_preds, defended):
        if len(example_list) >= max_examples:
            return
        for idx in torch.where(selection_mask.cpu())[0].tolist():
            if len(example_list) >= max_examples:
                break
            example_list.append({
                'clean': clean_inputs[idx],
                'clean_fd': fd_clean_inputs[idx],
                'adversarial': poisoned_inputs[idx],
                'adversarial_fd': fd_poisoned_inputs[idx],
                'target': float(targets.view(-1)[idx].item()),
                'clean_pred': float(clean_preds.view(-1)[idx].item()),
                'clean_fd_pred': float(fd_clean_preds.view(-1)[idx].item()),
                'adversarial_pred': float(poisoned_preds.view(-1)[idx].item()),
                'adversarial_fd_pred': float(fd_poisoned_preds.view(-1)[idx].item()),
                'defended': bool(defended),
            })

    @staticmethod
    def _save_fd_examples(examples, output_dir, example_source):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for idx, example in enumerate(examples, start=1):
            save_path = output_path / f'feature_distillation_{example_source}_{idx:02d}.png'
            Defender._save_fd_comparison(example, save_path)
            saved_paths.append(str(save_path))
        return {
            'output_dir': str(output_path),
            'selection': example_source,
            'saved_images': len(saved_paths),
            'paths': saved_paths,
        }

    @staticmethod
    def _save_fd_comparison(example, save_path):
        panels = [
            ('Clean', example['clean'], example['clean_pred']),
            ('Clean + FD', example['clean_fd'], example['clean_fd_pred']),
            ('|Clean FD diff| x5', Defender._difference_image(example['clean'], example['clean_fd'], scale=5.0), None),
            ('Adversarial', example['adversarial'], example['adversarial_pred']),
            ('Adversarial + FD', example['adversarial_fd'], example['adversarial_fd_pred']),
            ('|Adv FD diff| x5', Defender._difference_image(example['adversarial'], example['adversarial_fd'], scale=5.0), None),
        ]
        images = [Defender._tensor_to_pil_image(tensor) for _, tensor, _ in panels]
        widths, heights = zip(*(image.size for image in images))
        label_height = 42
        canvas = Image.new('RGB', (sum(widths), max(heights) + label_height), color='white')
        draw = ImageDraw.Draw(canvas)
        x_offset = 0
        for (title, _, pred), image in zip(panels, images):
            canvas.paste(image, (x_offset, label_height))
            pred_text = '' if pred is None else f'\ntrue={example["target"]:.0f}, pred={pred:.0f}'
            draw.text(
                (x_offset + 4, 4),
                f'{title}{pred_text}',
                fill='black',
            )
            x_offset += image.size[0]
        canvas.save(save_path)

    @staticmethod
    def _tensor_to_pil_image(image_tensor):
        image_uint8 = image_tensor.detach().cpu().clamp(0.0, 1.0).mul(255.0).byte()
        image_uint8 = image_uint8.permute(1, 2, 0).contiguous()
        height, width = image_uint8.shape[:2]
        return Image.frombytes('RGB', (width, height), bytes(image_uint8.view(-1).tolist()))

    @staticmethod
    def _difference_image(before, after, scale=5.0):
        return (after.detach().cpu() - before.detach().cpu()).abs().mul(float(scale)).clamp(0.0, 1.0)

    @staticmethod
    def _trigger_coverage_ratio(trigger_boxes, image_height=None, image_width=None):
        if image_height is None or image_width is None:
            return None
        image_area = float(image_height * image_width)
        if image_area <= 0:
            return None
        covered_area = 0.0
        for box in AdversarialAttack._normalize_trigger_boxes(trigger_boxes):
            covered_area += max(0, int(box['width'])) * max(0, int(box['height']))
        return min(1.0, covered_area / image_area)
