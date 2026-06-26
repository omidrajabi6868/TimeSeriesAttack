import torch
import math
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

    def compute_dct_statistics(self, block=8, max_blocks_per_chunk=65536):
        """
        Compute the standard deviation of each DCT coefficient over the
        entire validation/training loader.

        Returns
        -------
        std_map : Tensor (8,8)
            Standard deviation of every DCT coefficient.
        """

        self.model.eval()

        # Keep calibration statistics on CPU by default. The DCT statistics pass
        # can touch every image in the calibration set and does not need GPU
        # memory. Moving full 608x256 batches to CUDA here can easily cause OOM.
        device = torch.device('cpu')

        # ------------------------------------------------------------------
        # DCT matrix
        # ------------------------------------------------------------------
        T = torch.zeros((block, block), device=device)

        for k in range(block):
            for n in range(block):

                if k == 0:
                    alpha = 1.0 / math.sqrt(block)
                else:
                    alpha = math.sqrt(2.0 / block)

                T[k, n] = alpha * math.cos(
                    math.pi * (2 * n + 1) * k / (2 * block)
                )

        count = 0
        coeff_sum = torch.zeros((block, block), device=device)
        coeff_squares_sum = torch.zeros((block, block), device=device)

        with torch.no_grad():

            for images, _ in self.calibration_loader:

                images = images.to(device, non_blocking=False)

                # ----------------------------------------------------------
                # paper assumes pixel values in [0,255]
                # ----------------------------------------------------------
                images = images * 255.0

                B, C, H, W = images.shape

                # ----------------------------------------------------------
                # split into 8x8 blocks
                # ----------------------------------------------------------
                pad_h = (block - H % block) % block
                pad_w = (block - W % block) % block
                if pad_h or pad_w:
                    images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')

                blocks = images.unfold(2, block, block).unfold(3, block, block)

                # (B,C,Hb,Wb,block,block)
                blocks = blocks.contiguous().view(-1, block, block)

                # ----------------------------------------------------------
                # DCT
                # ----------------------------------------------------------
                for start in range(0, blocks.shape[0], max_blocks_per_chunk):
                    end = min(start + max_blocks_per_chunk, blocks.shape[0])
                    dct = T @ blocks[start:end] @ T.t()
                    coeff_sum += dct.sum(dim=0)
                    coeff_squares_sum += dct.square().sum(dim=0)
                    count += dct.shape[0]

        if count < 2:
            raise ValueError('At least two DCT blocks are required to compute standard deviation.')

        # --------------------------------------------------------------
        # unbiased std of every coefficient without storing all blocks
        # --------------------------------------------------------------
        variance = (coeff_squares_sum - coeff_sum.square() / count) / (count - 1)
        std_map = variance.clamp_min(0.0).sqrt()

        return std_map.to(self.device)

    def _predict_with_feature_distillation(self, fd, inputs, fd_batch_size):
        preds = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], fd_batch_size):
                end = min(start + fd_batch_size, inputs.shape[0])
                fd_inputs = fd(inputs[start:end].clone())
                outputs = self.model(fd_inputs)
                preds.append((outputs > 0).float().view(-1).cpu())
        return torch.cat(preds, dim=0).to(self.device)

    def _predict_with_feature_distillation(self, fd, inputs, fd_batch_size):
        preds = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], fd_batch_size):
                end = min(start + fd_batch_size, inputs.shape[0])
                fd_inputs = fd(inputs[start:end].clone())
                outputs = self.model(fd_inputs)
                preds.append((outputs > 0).float().view(-1).cpu())
        return torch.cat(preds, dim=0).to(self.device)

    def feature_distillation(self,
                            trigger_path, 
                            source_filter='bad', 
                            how_to_attach='blend',
                            block=8, QS=50.0, preserve_ratio=0.5, fd_batch_size=32):

        learned_trigger = AdversarialAttack.load_trigger(trigger_path)
        target_label = float(learned_trigger['target_label'])

        std_map = self.compute_dct_statistics(block=block)

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

            fd_preds = self._predict_with_feature_distillation(fd, source_inputs, fd_batch_size)
            clean_targets = source_targets.view(-1)
            fd_correct += int((fd_preds == clean_targets).sum().item())

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

            fd_poisoned_preds = self._predict_with_feature_distillation(fd, poisoned_inputs, fd_batch_size)
            asr_after_defend += int((fd_poisoned_preds == target_label).sum().item())
            conditional_attack_success += int((poisoned_preds[eligible_mask] == target_label).sum().item())
            conditional_asr_after_defend += int((fd_poisoned_preds[eligible_mask] == target_label).sum().item())
            total += int(poisoned_preds.shape[0])

        return {
            'samples_evaluated': total,
            'clean_source_accuracy': (clean_correct / total) * 100 if total else 0.0,
            'clean_fd_accuracy': (fd_correct / total) * 100 if total else 0.0,
            'attack_success_rate': (attack_success / total) * 100 if total else 0.0,
            'asr_after_defend':(asr_after_defend / total) * 100 if total else 0.0,
            'clean_not_target_count': clean_correct_and_not_target,
            'conditional_attack_success_rate': (
                (conditional_attack_success / clean_correct_and_not_target) * 100
                if clean_correct_and_not_target else 0.0
            ),
            'conditional_asr_after_defend': (
                (conditional_asr_after_defend / clean_correct_and_not_target) * 100
                if clean_correct_and_not_target else 0.0
            ),
            'target_label': target_label,
            'trigger_box': learned_trigger['trigger_boxes'],
        }
            








            

    