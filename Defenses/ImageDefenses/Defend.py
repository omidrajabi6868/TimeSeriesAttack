import torch
import math
from typing import Callable, Optional, Sequence
from .InputPurification import FeatureDistillation
from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack

class Defender:
    def __init__(self, 
                model, 
                dataset, 
                val_loader, 
                device: Optional[str] = None,
                use_multi_gpu: bool = True,
                gpu_ids: Optional[Sequence[int]] = None):

        self.dataset = dataset
        self.val_loader = val_loader
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

    def compute_dct_statistics(self, block=8):
        """
        Compute the standard deviation of each DCT coefficient over the
        entire validation/training loader.

        Returns
        -------
        std_map : Tensor (8,8)
            Standard deviation of every DCT coefficient.
        """

        self.model.eval()

        device = self.device

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

        coeffs = []

        with torch.no_grad():

            for images, _ in self.val_loader:

                images = images.to(device)

                # ----------------------------------------------------------
                # paper assumes pixel values in [0,255]
                # ----------------------------------------------------------
                images = images * 255.0

                B, C, H, W = images.shape

                # ----------------------------------------------------------
                # split into 8x8 blocks
                # ----------------------------------------------------------
                blocks = images.unfold(2, 8, 8).unfold(3, 8, 8)

                # (B,C,Hb,Wb,8,8)
                blocks = blocks.contiguous().view(-1, 8, 8)

                # ----------------------------------------------------------
                # DCT
                # ----------------------------------------------------------
                dct = T @ blocks @ T.t()

                coeffs.append(dct)

        coeffs = torch.cat(coeffs, dim=0)

        # --------------------------------------------------------------
        # std of every coefficient
        # --------------------------------------------------------------
        std_map = coeffs.std(dim=0)

        return std_map

    def feature_distillation(self,
                            trigger_path, 
                            source_filter='bad', 
                            how_to_attach='blend',
                            block=8, QS=30.0):

        learned_trigger = AdversarialAttack.load_trigger(trigger_path)
        target_tensor = torch.tensor(learned_trigger['target_label'], dtype=torch.float32, device=self.device).view(1, -1)

        std_map = self.compute_dct_statistics(block=block)

        fd = FeatureDistillation(std_map=std_map, block=block, QS=QS)

        total = 0
        attack_success = 0
        asr_after_defend = 0
        clean_correct = 0
        fd_correct = 0
        clean_correct_and_not_target = 0

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

            clean_outputs = self.model(source_inputs)
            clean_preds = (clean_outputs > 0).float().view(-1)
            clean_targets = source_targets.view(-1)
            clean_correct += int((clean_preds == clean_targets).sum().item())
            clean_correct_and_not_target += ((clean_preds == clean_targets) & (clean_preds != target_tensor.view(-1))).sum().item()

            fd_inputs = fd(source_inputs.clone())
            fd_outputs = self.model(fd_inputs)
            fd_preds = (fd_outputs > 0).float().view(-1)
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

            poisoned_outputs = self.model(poisoned_inputs.clone())
            poisoned_preds = (poisoned_outputs > 0).float()

            expanded_target = target_tensor.expand(poisoned_preds.shape[0], -1)
            attack_success += int((poisoned_preds == expanded_target).sum().item())

            fd_poisoned_inputs = fd(poisoned_inputs)
            fd_poisoned_outputs = self.model(fd_poisoned_inputs)
            fd_poisoned_preds = (fd_poisoned_outputs > 0).float()
            asr_after_defend += int((fd_poisoned_preds == expanded_target).sum().item())
            total += int(poisoned_preds.shape[0])

        return {
            'samples_evaluated': total,
            'clean_source_accuracy': (clean_correct / total) * 100 if total else 0.0,
            'clean_fd_accuracy': (fd_correct / total) * 100 if total else 0.0,
            'attack_success_rate': (attack_success / total) * 100 if total else 0.0,
            'asr_after_defend':(asr_after_defend / total) * 100 if total else 0.0,
            'clean_not_target_count': clean_correct_and_not_target,
            'conditional_attack_success_rate': (
                (attack_success / clean_correct_and_not_target) * 100
                if clean_correct_and_not_target else 0.0
            ),
            'target_label': float(learned_trigger['target_label']),
            'trigger_box': learned_trigger['trigger_boxes'],
        }
            








            

    