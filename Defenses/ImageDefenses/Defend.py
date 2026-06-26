import torch
from typing import Optional, Sequence
from .InputPurification import FeatureDistillation
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

        self.model = self.model.to(self.device)
        if (
            self.use_multi_gpu
            and self.device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            if self.gpu_ids is None:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            if len(self.gpu_ids) > 1:
                print(f"Using DataParallel on GPUs: {self.gpu_ids}")
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        return

    def feature_distillation(self,
                            trigger_path,
                            source_filter='bad',
                            how_to_attach='blend',
                            block=8, QS=50.0, preserve_ratio=0.5, fd_batch_size=32, fd_max_blocks_per_chunk=65536):

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
            with torch.no_grad():
                for start in range(0, source_inputs.shape[0], fd_batch_size):
                    end = min(start + fd_batch_size, source_inputs.shape[0])
                    fd_clean_inputs = fd(source_inputs[start:end].clone())
                    clean_fd_abs_diff_sum += float(
                        (fd_clean_inputs - source_inputs[start:end]).abs().sum().item()
                    )
                    clean_fd_pixel_count += int(fd_clean_inputs.numel())
                    fd_outputs = self.model(fd_clean_inputs)
                    fd_clean_preds.append((fd_outputs > 0).float().view(-1))

            fd_preds = torch.cat(fd_clean_preds, dim=0)
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
            with torch.no_grad():
                for start in range(0, poisoned_inputs.shape[0], fd_batch_size):
                    end = min(start + fd_batch_size, poisoned_inputs.shape[0])
                    fd_poisoned_inputs = fd(poisoned_inputs[start:end].clone())
                    poisoned_fd_abs_diff_sum += float(
                        (fd_poisoned_inputs - poisoned_inputs[start:end]).abs().sum().item()
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
                    fd_poisoned_preds_by_batch.append((fd_outputs > 0).float().view(-1))

            fd_poisoned_preds = torch.cat(fd_poisoned_preds_by_batch, dim=0)
            asr_after_defend += int((fd_poisoned_preds == target_label).sum().item())
            poisoned_prediction_changes_after_fd += int((fd_poisoned_preds != poisoned_preds).sum().item())
            conditional_attack_success += int((poisoned_preds[eligible_mask] == target_label).sum().item())
            conditional_asr_after_defend += int((fd_poisoned_preds[eligible_mask] == target_label).sum().item())
            total += int(poisoned_preds.shape[0])

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
            'attack_success_rate': attack_success_rate,
            'asr_after_defend': defended_attack_success_rate,
            'asr_reduction_after_defend': attack_success_rate - defended_attack_success_rate,
            'poisoned_prediction_changes_after_fd': poisoned_prediction_changes_after_fd,
            'poisoned_fd_mean_abs_input_change': (
                poisoned_fd_abs_diff_sum / poisoned_fd_pixel_count
                if poisoned_fd_pixel_count else 0.0
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
        }
