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
                gpu_ids: Optional[Sequence[int]] = None,
                aggregation: str = 'mean',
                model_weights: Optional[Sequence[float]] = None):
        
        self.patch_size = patch_size
        super().__init__(model, device=device, use_multi_gpu=use_multi_gpu,
                         gpu_ids=gpu_ids, aggregation=aggregation, model_weights=model_weights)

    def learn_fixed_size_patch(self, 
                            dataset,
                            data_loader,
                            val_loader,
                            target_label, 
                            source_filter, 
                            steps,
                            learning_rate,
                            mask_learning_rate,
                            optimize_mask,
                            mask_l1_weight,
                            patch_l2_weight,
                            trigger_preview_dir,
                            trigger_preview_loader,
                            trigger_preview_max_images,
                            how_to_attach,
                            patch_count, 
                            patch_update_method,
                            epsilon,
                            bandwidth,
                            checkpoint_interval=None,
                            checkpoint_path=None):

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
            optimize_mask=optimize_mask,
            initial_edge_softness=0.0,
            min_edge_softness=0.0,
            softness_decay=0.0,
            softness_patience=0,
            asr_hardening_threshold=80.0,
            mask_l1_weight=mask_l1_weight,
            patch_l2_weight=patch_l2_weight,
            softness_alignment_weight=1,
            patch_update_method=patch_update_method,
            epsilon=epsilon,
            log_interval=5,
            trigger_preview_interval=10,
            trigger_preview_dir=trigger_preview_dir,
            trigger_preview_loader=trigger_preview_loader,
            trigger_preview_max_images=trigger_preview_max_images,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            progressive_resize=False,
            randomize_training_location=False,
            enable_compression_phase=False,
            how_to_attach=how_to_attach,
            bandwidth=bandwidth)

    def learn_image_specific_patch(self,
                                dataset,
                                data_loader,
                                target_label, 
                                source_filter, 
                                steps,
                                learning_rate,
                                mask_learning_rate,
                                optimize_mask,
                                mask_l1_weight,
                                patch_l2_weight,
                                trigger_preview_dir,
                                trigger_preview_loader,
                                trigger_preview_max_images,
                                how_to_attach,
                                patch_update_method,
                                epsilon,
                                bandwidth,
                                eot_samples,
                                output_dir,
                                split_name,
                                visualization_examples=0,
                                perturbation_mask_path=None,
                                checkpoint_interval=None,
                                checkpoint_path=None):
                                
        patch_width, patch_height = self._normalize_patch_size(self.patch_size)
        image_width, image_height = self._normalize_patch_size(dataset.image_size)
        if patch_width > image_width or patch_height > image_height:
            raise ValueError('patch_size cannot exceed image_size.')
        # A fixed centered location avoids using validation/test labels to
        # choose an attack location. Only the patch values are image-specific.
        selected_trigger_boxes = [{
            'x': (image_width - patch_width) // 2,
            'y': (image_height - patch_height) // 2,
            'width': patch_width,
            'height': patch_height,
        }]

        return self.learn_image_specific_trigger(
            data_loader,
            selected_trigger_boxes,
            target_label=target_label,
            source_filter=source_filter,
            steps=steps,
            learning_rate=learning_rate,
            mask_learning_rate=mask_learning_rate,
            optimize_mask=optimize_mask,
            initial_edge_softness=0.0,
            min_edge_softness=0.0,
            softness_decay=0.0,
            mask_l1_weight=mask_l1_weight,
            patch_l2_weight=patch_l2_weight,
            patch_update_method=patch_update_method,
            epsilon=epsilon,
            bandwidth=bandwidth,
            eot_samples=eot_samples,
            log_interval=5,
            how_to_attach=how_to_attach,
            output_dir=output_dir,
            split_name=split_name,
            visualization_examples=visualization_examples,
            perturbation_mask_path=perturbation_mask_path)
