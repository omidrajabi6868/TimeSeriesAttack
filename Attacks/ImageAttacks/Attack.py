import torch


from Attacks.ImageAttacks.ImageAdversarialAttack import AdversarialAttack
from pathlib import Path
from typing import Callable, Optional, Sequence

class Attck(AdversarialAttack):
    def __init__(self,
                patch_size: int, 
                model: Callable,
                device: Optional[str] = None,
                use_multi_gpu: bool = True,
                gpu_ids: Optional[Sequence[int]] = None):
        
        self.patch_size = patch_size
        super().__init__(model)
        pass

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
                                                    patch_count, 
                                                    patch_update_method,
                                                    gradient_norm_epsilon,
                                                    epsilon=1.0):
       
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
            initial_edge_softness=0.5,
            min_edge_softness=0.005,
            softness_decay=0.8,
            softness_patience=5,
            asr_hardening_threshold=80.0,
            mask_l1_weight=mask_l1_weight,
            patch_l2_weight=patch_l2_weight,
            softness_alignment_weight=1,
            patch_update_method=patch_update_method,
            gradient_norm_epsilon=gradient_norm_epsilon,
            epsilon=epsilon,
            log_interval=1,
            trigger_preview_interval=10,
            trigger_preview_dir=trigger_preview_dir,
            trigger_preview_loader=trigger_preview_loader,
            trigger_preview_max_images=trigger_preview_max_images,
            progressive_resize=False,
            randomize_training_location=False,
            enable_compression_phase=False,
            how_to_attach=how_to_attach)