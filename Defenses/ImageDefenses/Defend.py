import time
from typing import Callable, Optional, Sequence, Tuple, TypeVar

import torch.nn.functional as F
import torch
from .InputPurification import FeatureDistillation
from .DiffusionPurification import DiffusionPurifier
from .FeatureSqueezing import JointFeatureSqueezingDetector, BitDepthReduction, MedianSmoothing, NonLocalMeansSmoothing
from .defense_visualization import trigger_coverage_ratio
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack

_TimedResult = TypeVar("_TimedResult")


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

    def _timed_inference(self, operation: Callable[[], _TimedResult]) -> Tuple[_TimedResult, float]:
        """Run an inference operation and return its wall-clock duration.

        CUDA kernels are asynchronous, so synchronizing immediately before and
        after the operation is necessary for the reported duration to represent
        the work performed by the defender rather than kernel launch time.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        result = operation()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return result, time.perf_counter() - start

    @staticmethod
    def _runtime_metrics(total_seconds, sample_count, prefix):
        seconds_per_image = total_seconds / sample_count if sample_count else 0.0
        return {
            f'{prefix}_runtime_seconds': total_seconds,
            f'{prefix}_runtime_seconds_per_image': seconds_per_image,
            f'{prefix}_throughput_images_per_second': (
                sample_count / total_seconds if total_seconds else 0.0
            ),
        }

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
        clean_fd_runtime = 0.0
        poisoned_fd_runtime = 0.0

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
                    (fd_clean_inputs, fd_outputs), elapsed = self._timed_inference(
                        lambda start=start, end=end: self._feature_distillation_inference(
                            fd, source_inputs[start:end], self.model
                        )
                    )
                    clean_fd_runtime += elapsed
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
                    (fd_poisoned_inputs, fd_outputs), elapsed = self._timed_inference(
                        lambda start=start, end=end: self._feature_distillation_inference(
                            fd, poisoned_inputs[start:end], self.model
                        )
                    )
                    poisoned_fd_runtime += elapsed
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
                FeatureDistillation.collect_examples(
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
                    FeatureDistillation.collect_examples(
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
            saved_example_info = FeatureDistillation.save_examples(
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

        result = {
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
            'fd_timing_batch_size': int(fd_batch_size),
            'fd_quantization_table': fd.quantization_table.detach().cpu().tolist(),
            'target_label': target_label,
            'trigger_box': learned_trigger['trigger_boxes'],
            'trigger_coverage_ratio': trigger_coverage_ratio(
                learned_trigger['trigger_boxes'],
                image_height=self.dataset.image_size[1] if getattr(self.dataset, 'image_size', None) else None,
                image_width=self.dataset.image_size[0] if getattr(self.dataset, 'image_size', None) else None,
            ),
            'saved_feature_distillation_examples': saved_example_info,
        }
        result.update(self._runtime_metrics(clean_fd_runtime, total, 'clean_fd'))
        result.update(self._runtime_metrics(poisoned_fd_runtime, total, 'poisoned_fd'))
        result.update(self._runtime_metrics(clean_fd_runtime + poisoned_fd_runtime, total * 2, 'fd'))
        return result

    @staticmethod
    def _feature_distillation_inference(fd, inputs, model):
        defended_inputs = fd(inputs.clone())
        return defended_inputs, model(defended_inputs)

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
        clean_dp_runtime = 0.0
        poisoned_dp_runtime = 0.0

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
                    (purified_clean, purified_clean_outputs), elapsed = self._timed_inference(
                        lambda start=start, end=end: self._diffusion_inference(
                            purifier, source_inputs[start:end], self.model,
                            diffusion_step, reverse_steps, stochastic,
                        )
                    )
                    clean_dp_runtime += elapsed
                    purified_clean_batches.append(purified_clean.detach().cpu())
                    clean_diff = (purified_clean - source_inputs[start:end]).abs()
                    clean_dp_abs_diff_sum += float(clean_diff.sum().item())
                    clean_dp_pixel_count += int(purified_clean.numel())
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
                    (purified_poisoned, purified_poisoned_outputs), elapsed = self._timed_inference(
                        lambda start=start, end=end: self._diffusion_inference(
                            purifier, poisoned_inputs[start:end], self.model,
                            diffusion_step, reverse_steps, stochastic,
                        )
                    )
                    poisoned_dp_runtime += elapsed
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
                DiffusionPurifier.collect_examples(
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
                    DiffusionPurifier.collect_examples(
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
            saved_example_info = DiffusionPurifier.save_examples(
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

        result = {
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
            'dp_timing_batch_size': int(dp_batch_size),
            'target_label': target_label,
            'trigger_box': learned_trigger['trigger_boxes'],
            'trigger_coverage_ratio': trigger_coverage_ratio(
                learned_trigger['trigger_boxes'],
                image_height=self.dataset.image_size[1] if getattr(self.dataset, 'image_size', None) else None,
                image_width=self.dataset.image_size[0] if getattr(self.dataset, 'image_size', None) else None,
            ),
            'saved_diffusion_purification_examples': saved_example_info,
        }
        result.update(self._runtime_metrics(clean_dp_runtime, total, 'clean_dp'))
        result.update(self._runtime_metrics(poisoned_dp_runtime, total, 'poisoned_dp'))
        result.update(self._runtime_metrics(clean_dp_runtime + poisoned_dp_runtime, total * 2, 'dp'))
        return result

    @staticmethod
    def _diffusion_inference(purifier, inputs, model, diffusion_step, reverse_steps, stochastic):
        defended_inputs = purifier.purify(
            inputs.clone(),
            diffusion_step=diffusion_step,
            reverse_steps=reverse_steps,
            stochastic=stochastic,
        )
        return defended_inputs, model(defended_inputs)

    def feature_squeezing(self,
                        trigger_path,
                        source_filter='bad',
                        how_to_attach='blend',
                        sqz_threshold=0.08,
                        save_examples_dir=None,
                        max_saved_examples=5):

        learned_trigger = AdversarialAttack.load_trigger(trigger_path)
        target_label = float(learned_trigger['target_label'])

        print("\n--- Initializing Detector & Running Evaluation ---")

        squeezers = [
            BitDepthReduction(bit_depth=1),
            MedianSmoothing(kernel_size=2),
            NonLocalMeansSmoothing()
        ]
        self.model.eval()
        detector = JointFeatureSqueezingDetector(model=self.model, squeezers=squeezers, threshold=sqz_threshold).to(self.device)

        detector.eval()

        total_clean, false_positives = 0, 0
        total_adv, true_positives = 0, 0
        clean_detection_runtime = 0.0
        adversarial_detection_runtime = 0.0

        for inputs, targets in self.val_loader:
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
                out_clean, elapsed = self._timed_inference(lambda: detector(source_inputs))
            clean_detection_runtime += elapsed
            false_positives += out_clean["is_adversarial"].sum().item()
            total_clean += source_inputs.size(0)

            adv_images = AdversarialAttack._inject_trigger(
                source_inputs.clone(),
                learned_trigger['trigger_boxes'],
                trigger_value=None,
                trigger_patch=learned_trigger['patch'],
                trigger_mask=learned_trigger['mask'],
                edge_softness=learned_trigger['softness'],
                how_to_attach=how_to_attach
            )

            # Only test detection on attacks that successfully fooled the model
            with torch.no_grad():
                adv_preds = (self.model(adv_images) > 0).float().view(-1)
                successful_attacks = (adv_preds == target_label)

            if successful_attacks.sum() > 0:
                valid_adv_images = adv_images[successful_attacks]
                out_adv, elapsed = self._timed_inference(lambda: detector(valid_adv_images))
                adversarial_detection_runtime += elapsed
                true_positives += out_adv["is_adversarial"].sum().item()
                total_adv += valid_adv_images.size(0)
        
        # Calculate final percentages
        fpr = (false_positives / total_clean) * 100.0 if total_clean > 0 else 0.0
        detection_rate = (true_positives / total_adv) * 100.0 if total_adv > 0 else 0.0

        print("-" * 50)
        print(f"Total Clean Test Samples Analyzed : {total_clean}")
        print(f"Successful Adversarial Examples   : {total_adv}")
        print("-" * 50)
        print(f"False Positive Rate (FPR)         : {fpr:.2f}%  (Target: Low)")
        print(f"Adversarial Detection Rate (TPR)  : {detection_rate:.2f}%  (Target: High)")
        print("-" * 50)

        result = {
            'total_clean_samples': total_clean,
            'successful_adversarial_samples': total_adv,
            'false_positive_rate': fpr,
            'adversarial_detection_rate': detection_rate,
        }
        result.update(self._runtime_metrics(clean_detection_runtime, total_clean, 'clean_detection'))
        result.update(self._runtime_metrics(adversarial_detection_runtime, total_adv, 'adversarial_detection'))
        result.update(self._runtime_metrics(
            clean_detection_runtime + adversarial_detection_runtime,
            total_clean + total_adv,
            'detection',
        ))
        return result
