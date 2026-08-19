import json
import math
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from typing import Callable, Optional, Sequence
from pathlib import Path
from Network import ClassificationModels
from Attacks.ImageAttacks.CostFunction import ClassificationObjective, FeaturBaseObjective, FeatureExtractor, PSPUAPObjective
from Network.UNet import ParameterRender
from Attacks.ImageAttacks.Fourier import FourierFilter
from Attacks.ImageAttacks.RobustUAP import RobustUAPConfig, TransformSampler, estimate_robustness, project_lp_ball
from Attacks.ImageAttacks.PSPUAP import PSPTransformSampler 
from torch.utils.data import TensorDataset, DataLoader


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
    
    def _remove_feature_extractor(self):
        feature_extractor = getattr(self, 'feature_extractor', None)
        if feature_extractor is not None:
            feature_extractor.clear()
            feature_extractor.remove()
            self.feature_extractor = None

    def _build_cost_function(self, name):
        self._remove_feature_extractor()
        if name == 'classification':
            self.cost_function = ClassificationObjective()
        elif name == 'gd_uap':
            self.feature_extractor = FeatureExtractor(self.model, n_last_layers=100, layer_types=(torch.nn.Conv2d,))
            self.cost_function = FeaturBaseObjective(self.feature_extractor)
        elif name == 'fg_uap':
            self.feature_extractor = FeatureExtractor(self.model, n_last_layers=4, layer_types=(torch.nn.Linear,),exclude_last_layers=1)
            self.cost_function = FeaturBaseObjective(self.feature_extractor)
        elif name == 'psp_uap':
            self.feature_extractor = FeatureExtractor(self.model, n_last_layers=0, layer_types=(torch.nn.Conv2d,))
            self.cost_function = PSPUAPObjective(feature_extractor=self.feature_extractor, p_active=True, re_weight=True)
        else:
            assert name not in ['classification', 'gd_uap', 'fg_uap'], "This cost is not defined."

    @staticmethod
    def _default_trigger_history_path(output_path):
        output_path = Path(output_path)
        return output_path.with_name(f'{output_path.stem}_history.json')

    @staticmethod
    def _json_safe(value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): AdversarialAttack._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AdversarialAttack._json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    @staticmethod
    def save_trigger(trigger, output_path, history_path=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        history_path = (
            Path(history_path)
            if history_path is not None else AdversarialAttack._default_trigger_history_path(output_path)
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)

        patch = trigger['patch']
        if not torch.is_tensor(patch):
            patch = torch.tensor(patch, dtype=torch.float32)
        mask = trigger.get('mask')
        torch.save(
            {
                'patch': patch.detach().cpu(),
                'mask': (
                    mask.detach().cpu()
                    if mask is not None and torch.is_tensor(mask)
                    else mask
                ),
                'trigger_box': trigger.get('trigger_box'),
                'trigger_boxes': trigger.get('trigger_boxes'),
                'target_label': float(trigger.get('target_label', 0.0)),
                'source_filter': trigger.get('source_filter', 'bad'),
                'epsilon': float(trigger.get('epsilon', 0.0)),
                'effective_epsilon': float(trigger.get('effective_epsilon', trigger.get('epsilon', 0.0))),
                'patch_norms': trigger.get('patch_norms', {}),
                'softness': trigger.get('softness', {}),
                'progressive_resize': trigger.get('progressive_resize', {}),
                'patch_update_method': trigger.get('patch_update_method'),
                'how_to_attach': trigger.get('how_to_attach', 'blend'),
                'selection': trigger.get('selection'),
                'selected_step': trigger.get('selected_step'),
                'selected_validation_asr': trigger.get('selected_validation_asr'),
                'best_validation_loss': trigger.get('best_validation_loss'),
                'best_validation_asr': trigger.get('best_validation_asr'),
                'smallest_success_validation_loss': trigger.get('smallest_success_validation_loss'),
                'smallest_success_validation_asr': trigger.get('smallest_success_validation_asr'),
                'smallest_success_patch_area': trigger.get('smallest_success_patch_area'),
                'trigger_previews': trigger.get('trigger_previews', []),
                'history_path': str(history_path),
            },
            output_path,
        )

        history_payload = {
            'history': trigger.get('history', []),
            'selection': trigger.get('selection'),
            'selected_step': trigger.get('selected_step'),
            'selected_validation_asr': trigger.get('selected_validation_asr'),
            'best_validation_loss': trigger.get('best_validation_loss'),
            'best_validation_asr': trigger.get('best_validation_asr'),
            'smallest_success_validation_loss': trigger.get('smallest_success_validation_loss'),
            'smallest_success_validation_asr': trigger.get('smallest_success_validation_asr'),
            'smallest_success_patch_area': trigger.get('smallest_success_patch_area'),
            'trigger_previews': trigger.get('trigger_previews', []),
            'patch_path': str(output_path),
        }
        with open(history_path, 'w', encoding='utf-8') as history_file:
            json.dump(AdversarialAttack._json_safe(history_payload), history_file, indent=2)
        trigger['path'] = str(output_path)
        trigger['history_path'] = str(history_path)
        return str(output_path)

    @staticmethod
    def load_trigger(trigger_path, map_location='cpu', history_path=None):
        trigger_path = Path(trigger_path)
        if not trigger_path.exists():
            raise FileNotFoundError(f'Trigger file not found: {trigger_path}')
        trigger_payload = torch.load(trigger_path, map_location=map_location)

        resolved_history_path = history_path or trigger_payload.get('history_path')
        if resolved_history_path is None:
            default_history_path = AdversarialAttack._default_trigger_history_path(trigger_path)
            resolved_history_path = str(default_history_path) if default_history_path.exists() else None
        history = trigger_payload.get('history', [])
        if resolved_history_path is not None and Path(resolved_history_path).exists():
            with open(resolved_history_path, 'r', encoding='utf-8') as history_file:
                history_payload = json.load(history_file)
            history = history_payload.get('history', history)

        return {
            'patch': trigger_payload['patch'],
            'mask': trigger_payload.get('mask'),
            'trigger_box': trigger_payload.get('trigger_box'),
            'trigger_boxes': trigger_payload.get('trigger_boxes'),
            'target_label': float(trigger_payload.get('target_label', 0.0)),
            'source_filter': trigger_payload.get('source_filter', 'bad'),
            'epsilon': float(trigger_payload.get('epsilon', 0.0)),
            'effective_epsilon': float(trigger_payload.get('effective_epsilon', trigger_payload.get('epsilon', 0.0))),
            'patch_norms': trigger_payload.get('patch_norms', {}),
            'softness': trigger_payload.get('softness', {}),
            'progressive_resize': trigger_payload.get('progressive_resize', {}),
            'patch_update_method': trigger_payload.get('patch_update_method'),
            'how_to_attach': trigger_payload.get('how_to_attach', 'blend'),
            'selection': trigger_payload.get('selection'),
            'selected_step': trigger_payload.get('selected_step'),
            'selected_validation_asr': trigger_payload.get('selected_validation_asr'),
            'best_validation_loss': trigger_payload.get('best_validation_loss'),
            'best_validation_asr': trigger_payload.get('best_validation_asr'),
            'smallest_success_validation_loss': trigger_payload.get('smallest_success_validation_loss'),
            'smallest_success_validation_asr': trigger_payload.get('smallest_success_validation_asr'),
            'smallest_success_patch_area': trigger_payload.get('smallest_success_patch_area'),
            'trigger_previews': trigger_payload.get('trigger_previews', []),
            'history': history,
            'path': str(trigger_path),
            'history_path': resolved_history_path,
        }

    def learn_universal_trigger(self,
                                data_loader,
                                trigger_box,
                                target_label=1.0,
                                source_filter='bad',
                                validation_loader=None,
                                report_training_asr=False,
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
                                epsilon=1.0,
                                log_interval=1,
                                trigger_preview_interval=10,
                                trigger_preview_dir='backups/adversarial_trigger_previews',
                                trigger_preview_loader=None,
                                trigger_preview_max_images=5,
                                progressive_resize=True,
                                progressive_resize_direction='grow',
                                min_patch_size=(16, 16),
                                randomize_training_location=True,
                                patch_growth_factor=None,
                                patch_shrink_factor=None,
                                patch_recovery_growth_factor=None,
                                min_steps_per_patch_size=10,
                                size_patience=None,
                                resize_hysteresis=2.0,
                                compression_asr_threshold=None,
                                enable_compression_phase=True,
                                how_to_attach='blend',
                                bandwidth=60,
                                psp_num_copies=128):
        self.model.eval()
        progressive_resize_enabled = bool(progressive_resize)

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
        progressive_resize_direction = str(progressive_resize_direction).lower()
        if progressive_resize_direction in ('increase', 'growing'):
            progressive_resize_direction = 'grow'
        elif progressive_resize_direction in ('decrease', 'shrinking', 'minimize'):
            progressive_resize_direction = 'shrink'
        if progressive_resize_direction not in {'grow', 'shrink'}:
            raise ValueError("progressive_resize_direction must be either 'grow' or 'shrink'.")
        if patch_growth_factor is None:
            patch_growth_factor = 1.25
        patch_growth_factor = float(patch_growth_factor)
        if patch_growth_factor <= 1.0:
            raise ValueError('patch_growth_factor must be greater than 1.')
        patch_shrink_factor = patch_growth_factor if patch_shrink_factor is None else float(patch_shrink_factor)
        if patch_shrink_factor <= 1.0:
            raise ValueError('patch_shrink_factor must be greater than 1.')
        if patch_recovery_growth_factor is None:
            patch_recovery_growth_factor = (
                math.sqrt(patch_shrink_factor)
                if progressive_resize_direction == 'shrink' else patch_growth_factor
            )
        patch_recovery_growth_factor = float(patch_recovery_growth_factor)
        if patch_recovery_growth_factor <= 1.0:
            raise ValueError('patch_recovery_growth_factor must be greater than 1.')
        if (
            progressive_resize_direction == 'shrink'
            and patch_recovery_growth_factor >= patch_shrink_factor
        ):
            raise ValueError(
                'patch_recovery_growth_factor must be less than patch_shrink_factor '
                'when progressive_resize_direction is shrink.'
            )
        min_steps_per_patch_size = max(1, int(min_steps_per_patch_size))
        size_patience = int(size_patience) if size_patience is not None else int(softness_patience)
        size_patience = max(1, size_patience)
        resize_hysteresis = max(0.0, float(resize_hysteresis))
        shrink_asr_threshold = min(100.0, float(asr_hardening_threshold) + resize_hysteresis)
        grow_asr_threshold = max(0.0, float(asr_hardening_threshold) - resize_hysteresis)
        if compression_asr_threshold is None:
            compression_asr_threshold = asr_hardening_threshold
        compression_asr_threshold = min(100.0, max(0.0, float(compression_asr_threshold)))
        compression_phase_active = (
            bool(enable_compression_phase)
            and progressive_resize_enabled
            and progressive_resize_direction == 'shrink'
        )
        if progressive_resize_enabled and progressive_resize_direction == 'grow':
            width, height = min_width, min_height
        elif progressive_resize_enabled and progressive_resize_direction == 'shrink':
            width, height = max_width, max_height
        initial_width, initial_height = int(width), int(height)
        current_softness = float(max(min_edge_softness, initial_edge_softness))

        trigger_boxes = self._resize_trigger_boxes(validation_anchor_boxes, width, height, full_patch_size)
        trigger_delta = torch.randn((len(trigger_boxes), channels, height, width), device=self.device)
        trigger_delta.requires_grad_()
        patch_update_method = str(patch_update_method).lower()
        if patch_update_method in ('mi_fgsm', 'mifgsm', 'momentum'):
            patch_update_method = 'momentum_sign'
        elif patch_update_method in ('iterative_fgsm', 'ifgsm', 'sign'):
            patch_update_method = 'pgd_sign'
        elif patch_update_method == 'pgd':
            patch_update_method = 'pgd_sign'
        elif patch_update_method in ('uap', 'deepfool', 'deepfool_uap'):
            patch_update_method = 'deepfool_uap'
        elif patch_update_method in ('gd_uap', 'gd'):
            patch_update_method = 'gd_uap'
        elif patch_update_method in ('gap_uap', 'gap'):
            patch_update_method = 'gap_uap'
        elif patch_update_method in ('hp_uap', 'hp'):
            patch_update_method = 'hp_uap'
        elif patch_update_method in ('fg_uap', 'fg'):
            patch_update_method = 'fg_uap'
        elif patch_update_method in ('robust', 'robust_uap'):
            patch_update_method = 'robust_uap'
        elif patch_update_method in ('psp', 'psp_uap'):
            patch_update_method = 'psp_uap'

        valid_patch_update_methods = {'adam', 'pgd_sign', 'momentum_sign', 'deepfool_uap', 'gd_uap', 'gap_uap', 'hp_uap', 'fg_uap', 'robust_uap', 'psp_uap'}
        if patch_update_method not in valid_patch_update_methods:
            raise ValueError(
                'patch_update_method must be one of: '
                f'{sorted(valid_patch_update_methods)}.'
            )

        epsilon = float(epsilon)
        if epsilon < 0 or epsilon > 1:
            raise ValueError('epsilon must be between 0 and 1 for normalized input-space perturbations.')

        if patch_update_method == 'deepfool_uap':
            if optimize_mask:
                raise ValueError(
                    'DeepFool UAP follows the original additive-perturbation algorithm and does not '
                    'optimize masks. Set optimize_mask=False.'
                )
            if randomize_training_location:
                raise ValueError(
                    'DeepFool UAP requires a fixed universal perturbation location. Set '
                    'randomize_training_location=False.'
                )
            if progressive_resize_enabled:
                raise ValueError(
                    'DeepFool UAP does not use the training-loop progressive resize heuristic. Set '
                    'progressive_resize=False.'
                )
            return self._learn_deepfool_uap_trigger(
                data_loader=data_loader,
                validation_loader=validation_loader,
                trigger_boxes=trigger_boxes,
                target_label=target_label,
                source_filter=source_filter,
                steps=steps,
                epsilon=epsilon,
                log_interval=log_interval,
                trigger_preview_interval=trigger_preview_interval,
                trigger_preview_dir=trigger_preview_dir,
                trigger_preview_loader=trigger_preview_loader,
                trigger_preview_max_images=trigger_preview_max_images,
                edge_softness=current_softness,
                how_to_attach=how_to_attach,
                overshoot=0.02,
                max_deepfool_iter=50,
            )

        patch_optimizer = None
        if patch_update_method in ('adam', 'hp_uap', 'fg_uap', 'gd_uap', 'robust_uap', 'psp_uap'):
            patch_optimizer = torch.optim.Adam([trigger_delta], lr=learning_rate)
            if patch_update_method == 'hp_uap':
                hp_filtering = FourierFilter(mode='high_pass', bandwidth=bandwidth)
        
        if patch_update_method == 'gap_uap':
            generator = ParameterRender().to(self.device)
            # Keep a direct, trainable path from the latent perturbation to the
            # rendered perturbation.  Optimizing only the deep renderer made the
            # targeted BCE gradient pass through every U-Net block before it
            # could change the patch and commonly left GAP-UAP near its random
            # initialization.  The residual parameterization is still rendered
            # by GAP, while guaranteeing a well-conditioned identity path.
            patch_optimizer = torch.optim.Adam(
                [trigger_delta, *generator.parameters()],
                lr=learning_rate,
            )

            def render_gap_patch():
                return epsilon * torch.tanh(trigger_delta + generator(trigger_delta))

        if patch_update_method == 'robust_uap':
            return self._learn_robust_uap_trigger(
                data_loader=data_loader,
                validation_loader=validation_loader,
                trigger_boxes=trigger_boxes,
                trigger_delta=trigger_delta,
                target_label=target_label,
                source_filter=source_filter,
                steps=steps,
                epsilon=epsilon,
                log_interval=log_interval,
                trigger_preview_interval=trigger_preview_interval,
                trigger_preview_dir=trigger_preview_dir,
                trigger_preview_loader=trigger_preview_loader,
                trigger_preview_max_images=trigger_preview_max_images,
                edge_softness=current_softness,
                how_to_attach=how_to_attach,
                patch_optimizer=patch_optimizer
            )

        if patch_update_method == 'psp_uap':
            return self._learn_psp_uap_trigger(
                validation_loader=validation_loader,
                trigger_boxes=trigger_boxes,
                trigger_delta=trigger_delta,
                target_label=target_label,
                source_filter=source_filter,
                steps=steps,
                epsilon=epsilon,
                log_interval=log_interval,
                trigger_preview_interval=trigger_preview_interval,
                trigger_preview_dir=trigger_preview_dir,
                trigger_preview_loader=trigger_preview_loader,
                trigger_preview_max_images=trigger_preview_max_images,
                edge_softness=current_softness,
                how_to_attach=how_to_attach,
                patch_optimizer=patch_optimizer,
                num_copies=psp_num_copies,
            )

        patch_momentum = torch.zeros_like(trigger_delta, device=self.device)
        alpha = float(learning_rate)
        mu = float(momentum_decay)
        grad_norm_epsilon = float(gradient_norm_epsilon)

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

        preview_records = []
        preview_interval = int(trigger_preview_interval) if trigger_preview_interval is not None else 0
        preview_max_images = (
            max(0, int(trigger_preview_max_images))
            if trigger_preview_max_images is not None else 0
        )
        preview_output_dir = Path(trigger_preview_dir) if trigger_preview_dir is not None else None
        if preview_interval > 0 and preview_output_dir is not None:
            preview_output_dir.mkdir(parents=True, exist_ok=True)
        preview_data_loader = trigger_preview_loader or validation_loader or data_loader

        history = []
        best_patch = None
        best_mask = None
        best_trigger_boxes = [dict(box) for box in trigger_boxes]
        best_step = 0
        best_val_loss = float('inf')
        best_val_asr = float('-inf')
        best_softness = None
        smallest_success_patch = None
        smallest_success_mask = None
        smallest_success_boxes = None
        smallest_success_step = 0
        smallest_success_asr = float('-inf')
        smallest_success_val_loss = float('inf')
        smallest_success_area = float('inf')
        smallest_success_softness = None
        resize_events = []
        no_improve_steps = 0
        size_step_count = 0
        size_no_improve_steps = 0
        best_size_asr = float('-inf')

        if patch_update_method in ('adam', 'pgd_sign', 'momentum_sign', 'deepfool_uap', 'gap_uap', 'hp_uap'):
            self._build_cost_function('classification')
        elif patch_update_method == 'gd_uap':
            self._build_cost_function('gd_uap')
        elif patch_update_method == 'fg_uap':
            self._build_cost_function('fg_uap')

        for step_idx in range(steps):
            size_step_count += 1
            step_losses = []
            step_attack_losses = []
            step_patch_reg_losses = []
            step_mask_reg_losses = []
            step_softness_reg_losses = []
            step_samples = 0
            if patch_update_method == 'gap_uap':
                generator.eval()
                with torch.no_grad():
                    previous_patch = render_gap_patch().detach().clone()
            else:
                previous_patch = (epsilon * torch.tanh(trigger_delta)).detach().clone()

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
                if patch_update_method == 'gap_uap':
                    generator.train()
                    bounded_trigger_patch = render_gap_patch()
                elif patch_update_method == 'hp_uap':
                    bounded_trigger_patch = epsilon * torch.tanh(hp_filtering(trigger_delta))
                else:
                    bounded_trigger_patch = epsilon * torch.tanh(trigger_delta)

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

                target_tensor = None
                if patch_update_method == 'gd_uap':
                    self.feature_extractor.clear()
                elif patch_update_method == 'fg_uap':
                    self.feature_extractor.clear()
                    with torch.no_grad():
                        self.model(selected_inputs)
                    target_tensor = self.cost_function.detach_targets(self.feature_extractor.activations)

                poisoned_inputs = self._inject_trigger(
                    selected_inputs,
                    training_trigger_boxes,
                    trigger_patch=training_patch,
                    trigger_mask=training_mask,
                    edge_softness=current_softness,
                    how_to_attach=how_to_attach
                )

                feature_extractor = getattr(self, 'feature_extractor', None)
                if feature_extractor is not None:
                    feature_extractor.clear()

                model_outputs = self.model(poisoned_inputs)
                if patch_update_method not in ('gd_uap', 'fg_uap'):
                    target_tensor = torch.full_like(model_outputs, float(target_label))
                objective_outputs = (
                    feature_extractor.activations
                    if patch_update_method in ('gd_uap', 'fg_uap') else model_outputs
                )

                attack_loss = self.cost_function(outputs=objective_outputs, targets=target_tensor).to(self.device)

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

                if patch_update_method in ('adam', 'gap_uap', 'hp_uap', 'gd_uap', 'fg_uap'):
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

                batch_samples = int(model_outputs.shape[0])
                step_losses.append(float(loss.item()) * batch_samples)
                step_attack_losses.append(float(attack_loss.item()) * batch_samples)
                step_patch_reg_losses.append(float(patch_reg.item()) * batch_samples)
                step_mask_reg_losses.append(float(mask_reg.item()) * batch_samples)
                step_softness_reg_losses.append(float(softness_reg.item()) * batch_samples)
                step_samples += batch_samples

            step_loss = (sum(step_losses) / step_samples) if step_samples else 0.0
            step_attack_loss = (sum(step_attack_losses) / step_samples) if step_samples else 0.0
            step_patch_reg_loss = (sum(step_patch_reg_losses) / step_samples) if step_samples else 0.0
            step_mask_reg_loss = (sum(step_mask_reg_losses) / step_samples) if step_samples else 0.0
            step_softness_reg_loss = (sum(step_softness_reg_losses) / step_samples) if step_samples else 0.0
            if patch_update_method == 'gap_uap':
                # Materialize GAP in evaluation mode so BatchNorm uses the same
                # running statistics for validation, selection, and export.
                generator.eval()
                with torch.no_grad():
                    current_patch_for_metrics = render_gap_patch().detach()
            elif patch_update_method == 'hp_uap':
                current_patch_for_metrics = (epsilon * torch.tanh(hp_filtering(trigger_delta))).detach()
            else:
                current_patch_for_metrics = (epsilon * torch.tanh(trigger_delta)).detach()

            current_mask_for_metrics = (
                self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).detach()
                if mask_logits is not None else None
            )
            patch_update_l2 = float(torch.norm(
                (current_patch_for_metrics - previous_patch).reshape(-1),
                p=2,
            ).item())
            patch_l1_norm = float(torch.norm(current_patch_for_metrics.reshape(-1), p=1).item())
            patch_l2_norm = float(torch.norm(current_patch_for_metrics.reshape(-1), p=2).item())
            patch_linf_norm = float(torch.norm(current_patch_for_metrics.reshape(-1), p=float('inf')).item())
            if current_mask_for_metrics is not None:
                mask_l1_norm = float(torch.norm(current_mask_for_metrics.reshape(-1), p=1).item())
                mask_l2_norm = float(torch.norm(current_mask_for_metrics.reshape(-1), p=2).item())
                mask_linf_norm = float(torch.norm(current_mask_for_metrics.reshape(-1), p=float('inf')).item())
                mask_mean = float(current_mask_for_metrics.mean().item())
                effective_patch_for_metrics = current_patch_for_metrics * current_mask_for_metrics
            else:
                mask_l1_norm = 0.0
                mask_l2_norm = 0.0
                mask_linf_norm = 0.0
                mask_mean = 0.0
                effective_patch_for_metrics = current_patch_for_metrics
            effective_patch_linf_norm = float(torch.norm(
                effective_patch_for_metrics.reshape(-1),
                p=float('inf'),
            ).item())
            step_history = {
                'step': step_idx + 1,
                'loss': step_loss,
                'attack_loss': step_attack_loss,
                'patch_regularization_loss': step_patch_reg_loss,
                'mask_regularization_loss': step_mask_reg_loss,
                'softness_alignment_loss': step_softness_reg_loss,
                'samples': step_samples,
                'patch_update_l2': patch_update_l2,
                'patch_l1_norm': patch_l1_norm,
                'patch_l2_norm': patch_l2_norm,
                'patch_linf_norm': patch_linf_norm,
                'effective_patch_linf_norm': effective_patch_linf_norm,
                'inferred_epsilon': effective_patch_linf_norm,
                'mask_l1_norm': mask_l1_norm,
                'mask_l2_norm': mask_l2_norm,
                'mask_linf_norm': mask_linf_norm,
                'mask_mean': mask_mean,
                'patch_update_method': patch_update_method,
            }

            if report_training_asr:
                train_metrics = self.evaluate_attack_success(
                    test_loader=data_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=current_patch_for_metrics,
                    trigger_mask=(
                        self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach())
                        if mask_logits is not None else None
                    ),
                    target_label=target_label,
                    source_filter=source_filter,
                    edge_softness=current_softness,
                    how_to_attach=how_to_attach
                )
                step_history['training_asr'] = float(train_metrics['attack_success_rate'])
            with torch.no_grad():
                if validation_loader is not None:

                    current_patch = current_patch_for_metrics

                    current_mask = (
                        self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach())
                        if mask_logits is not None else None
                    )
                    val_metrics = self.evaluate_attack_success(
                        test_loader=validation_loader,
                        trigger_box=trigger_boxes,
                        trigger_patch=current_patch,
                        trigger_mask=current_mask,
                        target_label=target_label,
                        source_filter=source_filter,
                        edge_softness=current_softness,
                        how_to_attach=how_to_attach
                    )
                    val_loss_metrics = self.evaluate_trigger_loss(
                        data_loader=validation_loader,
                        trigger_box=trigger_boxes,
                        trigger_patch=current_patch,
                        trigger_mask=current_mask,
                        target_label=target_label,
                        source_filter=source_filter,
                        edge_softness=current_softness,
                        mask_l1_weight=mask_l1_weight,
                        patch_l2_weight=patch_l2_weight,
                        softness_alignment_weight=softness_alignment_weight,
                        how_to_attach=how_to_attach,
                        use_clean_feature_targets=(patch_update_method == 'fg_uap'),
                    )
                    val_asr = float(val_metrics['attack_success_rate'])
                    val_loss = float(val_loss_metrics['loss'])
                    step_history['validation_asr'] = val_asr
                    step_history['validation_loss'] = val_loss
                    step_history['validation_attack_loss'] = float(val_loss_metrics['attack_loss'])
                    step_history['validation_patch_regularization_loss'] = float(
                        val_loss_metrics['patch_regularization_loss']
                    )
                    step_history['validation_mask_regularization_loss'] = float(
                        val_loss_metrics['mask_regularization_loss']
                    )
                    step_history['validation_softness_alignment_loss'] = float(
                        val_loss_metrics['softness_alignment_loss']
                    )
                    step_history['validation_samples'] = int(val_loss_metrics['samples_evaluated'])
                    step_history['edge_softness'] = current_softness
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_asr = val_asr
                        best_patch = current_patch.cpu().clone()
                        best_mask = current_mask.cpu().clone() if current_mask is not None else None
                        best_trigger_boxes = [dict(box) for box in trigger_boxes]
                        best_step = step_idx + 1
                        best_softness = float(current_softness)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1

                    if val_asr > best_size_asr:
                        best_size_asr = val_asr
                        size_no_improve_steps = 0
                    else:
                        size_no_improve_steps += 1

                    if (
                        val_asr < grow_asr_threshold
                        and no_improve_steps >= softness_patience
                    ):
                        new_softness = max(min_edge_softness, current_softness * softness_decay)
                        if new_softness < current_softness:
                            current_softness = new_softness
                        no_improve_steps = 0

                    size_optimized_enough = size_step_count >= min_steps_per_patch_size
                    if val_asr >= asr_hardening_threshold and size_optimized_enough:
                        current_area = int(width) * int(height)
                        is_smaller_success = current_area < smallest_success_area
                        is_better_tie = (
                            current_area == smallest_success_area
                            and (
                                val_asr > smallest_success_asr
                                or (
                                    val_asr == smallest_success_asr
                                    and val_loss < smallest_success_val_loss
                                )
                            )
                        )
                        if is_smaller_success or is_better_tie:
                            # Save exactly the tensors that produced `val_asr`.
                            # Reconstructing from trigger_delta is not equivalent
                            # for methods that render/filter it (GAP-UAP and
                            # HP-UAP), and caused the returned trigger to differ
                            # from the one accepted during validation.
                            smallest_success_patch = current_patch.detach().cpu().clone()
                            smallest_success_mask = (
                                current_mask.detach().cpu().clone()
                                if current_mask is not None else None
                            )
                            smallest_success_boxes = [dict(box) for box in trigger_boxes]
                            smallest_success_step = step_idx + 1
                            smallest_success_asr = val_asr
                            smallest_success_val_loss = val_loss
                            smallest_success_area = current_area
                            smallest_success_softness = float(current_softness)
                        step_history['size_decision'] = 'accepted'

                    if (
                        bool(enable_compression_phase)
                        and progressive_resize_enabled
                        and val_asr >= compression_asr_threshold
                    ):
                        compression_phase_active = True
                    step_history['compression_phase_active'] = bool(compression_phase_active)

                    resize_decision = None
                    next_width = width
                    next_height = height
                    size_limit_decision = None
                    if progressive_resize_enabled and validation_loader is not None and size_optimized_enough:
                        if compression_phase_active and val_asr >= shrink_asr_threshold:
                            resize_decision = 'compress_shrink'
                            next_width = max(int(round(width / patch_shrink_factor)), min_width)
                            next_height = max(int(round(height / patch_shrink_factor)), min_height)
                            if next_width >= width and width > min_width:
                                next_width = width - 1
                            if next_height >= height and height > min_height:
                                next_height = height - 1
                            size_limit_decision = 'min_size_reached'
                        elif progressive_resize_direction == 'grow' and val_asr <= grow_asr_threshold:
                            if size_no_improve_steps >= size_patience:
                                resize_decision = 'grow'
                                next_width = min(int(round(width * patch_growth_factor)), max_width)
                                next_height = min(int(round(height * patch_growth_factor)), max_height)
                                if next_width <= width and width < max_width:
                                    next_width = width + 1
                                if next_height <= height and height < max_height:
                                    next_height = height + 1
                                size_limit_decision = 'max_size_reached'
                        elif progressive_resize_direction == 'shrink':
                            if val_asr >= shrink_asr_threshold:
                                resize_decision = 'shrink'
                                next_width = max(int(round(width / patch_shrink_factor)), min_width)
                                next_height = max(int(round(height / patch_shrink_factor)), min_height)
                                if next_width >= width and width > min_width:
                                    next_width = width - 1
                                if next_height >= height and height > min_height:
                                    next_height = height - 1
                                size_limit_decision = 'min_size_reached'
                            elif val_asr <= grow_asr_threshold and size_no_improve_steps >= size_patience:
                                # Hysteresis keeps shrink/recover decisions from flapping when
                                # validation ASR hovers near the hardening threshold.
                                resize_decision = 'recover_grow'
                                next_width = min(int(round(width * patch_recovery_growth_factor)), max_width)
                                next_height = min(int(round(height * patch_recovery_growth_factor)), max_height)
                                if next_width <= width and width < max_width:
                                    next_width = width + 1
                                if next_height <= height and height < max_height:
                                    next_height = height + 1
                                size_limit_decision = 'max_size_reached'

                    if resize_decision is not None:
                        can_resize = next_width != width or next_height != height
                        if can_resize:
                            resize_events.append({
                                'step': step_idx + 1,
                                'from_size': (int(width), int(height)),
                                'to_size': (int(next_width), int(next_height)),
                                'validation_asr': val_asr,
                                'decision': resize_decision,
                                'compression_phase_active': bool(compression_phase_active),
                            })
                            resized_patch = torch.nn.functional.interpolate(
                                current_patch_for_metrics,
                                size=(next_height, next_width),
                                mode='bilinear',
                                align_corners=False,
                            )
                            if epsilon > 0:
                                resized_unscaled_patch = resized_patch / epsilon
                            else:
                                resized_unscaled_patch = torch.zeros_like(resized_patch)
                            trigger_delta = torch.atanh(torch.clamp(resized_unscaled_patch, -0.999999, 0.999999)).detach()
                            trigger_delta.requires_grad_()
                            width, height = next_width, next_height
                            print(
                                'adversarial_patch_size_change: '
                                f'step={step_idx + 1}, decision={resize_decision}, '
                                f'patch_size=({width}, {height})'
                            )
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
                                previous_mask_logits = mask_logits.detach() if mask_logits is not None else None
                                base_mask = self._build_blend_mask(
                                    height=height,
                                    width=width,
                                    channels=channels,
                                    device=self.device,
                                    dtype=trigger_delta.dtype,
                                    edge_softness=current_softness,
                                ).expand(len(trigger_boxes), -1, -1, -1)
                                if previous_mask_logits is not None:
                                    mask_logits = torch.nn.functional.interpolate(
                                        previous_mask_logits,
                                        size=(height, width),
                                        mode='bilinear',
                                        align_corners=False,
                                    ).detach()
                                    if mask_logits.shape[0] != len(trigger_boxes):
                                        mask_logits = mask_logits[:1].expand(len(trigger_boxes), -1, -1, -1).clone()
                                else:
                                    mask_logits = torch.zeros_like(base_mask, device=self.device)
                                mask_logits = mask_logits.to(device=self.device, dtype=base_mask.dtype).requires_grad_(True)
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
                            current_mask_for_metrics = (
                                self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).detach()
                                if mask_logits is not None else None
                            )
                            step_history['resize_decision'] = resize_decision
                            step_history['resize_to_size'] = (int(width), int(height))
                        else:
                            step_history['size_decision'] = size_limit_decision


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
                        trigger_mask=current_mask_for_metrics,
                        target_label=target_label,
                        source_filter=source_filter,
                        edge_softness=current_softness,
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
                    if report_training_asr:
                        train_log = f', train_asr={step_history.get("training_asr", 0.0):.4f}'
                    if validation_loader is not None:
                        val_log = (
                            f', val_loss={step_history.get("validation_loss", 0.0):.6f}'
                            f', val_asr={step_history.get("validation_asr", 0.0):.4f}'
                        )
                    print(
                        f'[Trigger Learning] step={step_idx + 1}/{steps}, '
                        f'loss={step_loss:.6f}, attack_loss={step_attack_loss:.6f}, '
                        f'patch_reg={step_patch_reg_loss:.6f}, mask_reg={step_mask_reg_loss:.6f}, '
                        f'softness_reg={step_softness_reg_loss:.6f}, samples={step_samples}, '
                        f'patch_update_l2={patch_update_l2:.6f}, '
                        f'patch_l1_norm={patch_l1_norm:.6f}, patch_l2_norm={patch_l2_norm:.6f}, '
                        f'patch_linf_norm={patch_linf_norm:.6f}, mask_l1_norm={mask_l1_norm:.6f}, '
                        f'mask_l2_norm={mask_l2_norm:.6f}, mask_linf_norm={mask_linf_norm:.6f}, '
                        f'mask_mean={mask_mean:.6f}'
                        f'{train_log}{val_log}'
                    )
                    if step_samples == 0:
                        print(
                            '[Trigger Learning] warning: no samples matched source_filter '
                            f'"{source_filter}" at this step.'
                        )

        selection = 'last_step'
        if smallest_success_patch is not None:
            learned_patch = smallest_success_patch
            learned_mask = smallest_success_mask
            trigger_boxes = smallest_success_boxes
            selected_step = smallest_success_step
            selection = 'smallest_successful_patch'
            selected_validation_asr = smallest_success_asr
            selected_edge_softness = smallest_success_softness
        elif best_patch is not None:
            learned_patch = best_patch
            learned_mask = best_mask
            trigger_boxes = best_trigger_boxes
            selected_step = best_step
            selection = 'best_validation_loss'
            selected_validation_asr = best_val_asr
            selected_edge_softness = best_softness
        else:
            learned_patch = current_patch_for_metrics.cpu()
            learned_mask = (
                self._compose_trigger_mask(base_mask=base_mask, mask_logits=mask_logits.detach()).cpu()
                if mask_logits is not None else None
            )
            selected_step = steps
            selection = 'last_step'
            selected_validation_asr = None
            selected_edge_softness = float(current_softness)

        learned_patch_l1_norm = float(torch.norm(learned_patch.reshape(-1), p=1).item())
        learned_patch_l2_norm = float(torch.norm(learned_patch.reshape(-1), p=2).item())
        learned_patch_linf_norm = float(torch.norm(learned_patch.reshape(-1), p=float('inf')).item())
        learned_effective_patch = learned_patch * learned_mask if learned_mask is not None else learned_patch
        learned_effective_epsilon = float(torch.norm(
            learned_effective_patch.reshape(-1),
            p=float('inf'),
        ).item())

        self._remove_feature_extractor()

        return {
            'patch': learned_patch,
            'mask': learned_mask,
            'history': history,
            'trigger_box': trigger_boxes[0],
            'trigger_boxes': trigger_boxes,
            'target_label': float(target_label),
            'source_filter': source_filter,
            'patch_update_method': patch_update_method,
            'how_to_attach': how_to_attach,
            'epsilon': learned_patch_linf_norm,
            'effective_epsilon': learned_effective_epsilon,
            'patch_norms': {
                'l1': learned_patch_l1_norm,
                'l2': learned_patch_l2_norm,
                'linf': learned_patch_linf_norm,
                'effective_linf': learned_effective_epsilon,
            },
            'softness': {
                'initial_edge_softness': float(initial_edge_softness),
                'final_edge_softness': float(current_softness),
                'min_edge_softness': float(min_edge_softness),
                'softness_decay': float(softness_decay),
                'softness_patience': int(softness_patience),
                'asr_hardening_threshold': float(asr_hardening_threshold),
                'selected_edge_softness': selected_edge_softness,
            },
            'progressive_resize': {
                'enabled': bool(progressive_resize_enabled),
                'direction': progressive_resize_direction,
                'randomize_training_location': bool(randomize_training_location),
                'initial_patch_size': (int(initial_width), int(initial_height)),
                'final_patch_size': (int(trigger_boxes[0]['width']), int(trigger_boxes[0]['height'])),
                'min_patch_size': (int(min_width), int(min_height)),
                'max_patch_size': (int(max_width), int(max_height)),
                'patch_growth_factor': float(patch_growth_factor),
                'patch_shrink_factor': float(patch_shrink_factor),
                'patch_recovery_growth_factor': float(patch_recovery_growth_factor),
                'min_steps_per_patch_size': int(min_steps_per_patch_size),
                'size_patience': int(size_patience),
                'asr_threshold': float(asr_hardening_threshold),
                'shrink_asr_threshold': float(shrink_asr_threshold),
                'grow_asr_threshold': float(grow_asr_threshold),
                'resize_hysteresis': float(resize_hysteresis),
                'compression_asr_threshold': float(compression_asr_threshold),
                'compression_phase_active': bool(compression_phase_active),
                'events': resize_events,
            },
            'trigger_previews': preview_records,
            'selection': selection,
            'selected_step': int(selected_step),
            'selected_validation_asr': (
                None if selected_validation_asr is None else float(selected_validation_asr)
            ),
            'best_validation_loss': None if validation_loader is None else float(best_val_loss),
            'best_validation_asr': None if validation_loader is None else float(best_val_asr),
            'smallest_success_validation_loss': (
                None if smallest_success_patch is None else float(smallest_success_val_loss)
            ),
            'smallest_success_validation_asr': (
                None if smallest_success_patch is None else float(smallest_success_asr)
            ),
            'smallest_success_patch_area': (
                None if smallest_success_patch is None else int(smallest_success_area)
            ),
        }

    def _learn_deepfool_uap_trigger(self,
                                    data_loader,
                                    validation_loader,
                                    trigger_boxes,
                                    target_label,
                                    source_filter,
                                    steps,
                                    epsilon,
                                    log_interval,
                                    trigger_preview_interval,
                                    trigger_preview_dir,
                                    trigger_preview_loader,
                                    trigger_preview_max_images,
                                    edge_softness,
                                    how_to_attach,
                                    overshoot=0.02,
                                    max_deepfool_iter=50):
        """Learn a targeted universal perturbation with a DeepFool-style loop.

        This keeps the Moosavi-Dezfooli UAP structure: repeatedly scan the
        training set, compute a minimal per-image projection only for samples
        not yet satisfying the attack objective, add it to the shared universal
        perturbation, and project the universal perturbation back to the lp
        ball. In this repository's binary-classification setting, the
        projection is targeted: it moves the single logit across the decision
        boundary toward ``target_label``.
        """
        if len(trigger_boxes) < 1:
            raise ValueError('DeepFool UAP requires at least one trigger box.')
        target_class = int(target_label)
        if float(target_label) not in (0.0, 1.0):
            raise ValueError('Targeted DeepFool UAP currently supports binary target_label values 0 or 1.')
        channels = 3
        height = int(trigger_boxes[0]['height'])
        width = int(trigger_boxes[0]['width'])
        for box in trigger_boxes[1:]:
            if int(box['width']) != width or int(box['height']) != height:
                raise ValueError('All trigger boxes must share size for DeepFool UAP.')

        universal_patch = torch.zeros(
            (len(trigger_boxes), channels, height, width),
            device=self.device,
        )
        history = []
        preview_records = []
        preview_interval = int(trigger_preview_interval) if trigger_preview_interval is not None else 0
        preview_max_images = (
            max(0, int(trigger_preview_max_images))
            if trigger_preview_max_images is not None else 0
        )
        preview_output_dir = Path(trigger_preview_dir) if trigger_preview_dir is not None else None
        if preview_interval > 0 and preview_output_dir is not None:
            preview_output_dir.mkdir(parents=True, exist_ok=True)
        preview_data_loader = trigger_preview_loader or validation_loader or data_loader

        best_patch = None
        best_step = 0
        best_target_asr = float('-inf')
        best_val_asr = float('-inf')
        best_val_loss = float('inf')

        for step_idx in range(int(steps)):
            changed_samples = 0
            considered_samples = 0
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)
                flat_targets = targets.view(-1)
                if source_filter == 'bad':
                    source_mask = (flat_targets == 0)
                elif source_filter == 'good':
                    source_mask = (flat_targets == 1)
                elif source_filter == 'all':
                    source_mask = torch.ones(targets.shape[0], dtype=torch.bool, device=self.device)
                else:
                    raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")

                for image in inputs[source_mask]:
                    considered_samples += 1
                    perturbed = self._inject_trigger(
                        image.unsqueeze(0),
                        trigger_boxes,
                        trigger_patch=universal_patch.detach(),
                        trigger_mask=None,
                        edge_softness=edge_softness,
                        how_to_attach=how_to_attach,
                    )
                    if self._binary_prediction(perturbed) == target_class:
                        continue
                    sample_patch = self._deepfool_binary_targeted_patch_update(
                        image=image,
                        trigger_boxes=trigger_boxes,
                        universal_patch=universal_patch,
                        target_label=target_class,
                        edge_softness=edge_softness,
                        how_to_attach=how_to_attach,
                        overshoot=overshoot,
                        max_iter=max_deepfool_iter,
                    )
                    if sample_patch is not None:
                        universal_patch = torch.clamp(universal_patch + sample_patch, -epsilon, epsilon).detach()
                        changed_samples += 1

            target_metrics = self.evaluate_attack_success(
                test_loader=validation_loader or data_loader,
                trigger_box=trigger_boxes,
                trigger_patch=universal_patch.detach(),
                trigger_mask=None,
                target_label=target_class,
                source_filter=source_filter,
                edge_softness=edge_softness,
                how_to_attach=how_to_attach,
            )
            val_metrics = None
            val_loss_metrics = None
            if validation_loader is not None:
                val_metrics = self.evaluate_attack_success(
                    test_loader=validation_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=universal_patch.detach(),
                    trigger_mask=None,
                    target_label=target_class,
                    source_filter=source_filter,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach,
                )
                val_loss_metrics = self.evaluate_trigger_loss(
                    data_loader=validation_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=universal_patch.detach(),
                    trigger_mask=None,
                    target_label=target_class,
                    source_filter=source_filter,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach,
                )

            target_asr = float(target_metrics['attack_success_rate'])
            if target_asr > best_target_asr:
                best_target_asr = target_asr
                best_patch = universal_patch.detach().cpu().clone()
                best_step = step_idx + 1
                if val_metrics is not None:
                    best_val_asr = float(val_metrics['attack_success_rate'])
                    best_val_loss = float(val_loss_metrics['loss'])

            step_history = {
                'step': step_idx + 1,
                'patch_update_method': 'deepfool_uap',
                'samples': considered_samples,
                'deepfool_updates': changed_samples,
                'targeted_attack_success_rate': target_asr,
                'patch_linf_norm': float(torch.norm(universal_patch.reshape(-1), p=float('inf')).item()),
                'effective_patch_linf_norm': float(torch.norm(universal_patch.reshape(-1), p=float('inf')).item()),
            }
            if val_metrics is not None:
                step_history['validation_asr'] = float(val_metrics['attack_success_rate'])
                step_history['validation_loss'] = float(val_loss_metrics['loss'])
            history.append(step_history)

            if preview_interval > 0 and preview_output_dir is not None and (step_idx + 1) % preview_interval == 0:
                step_preview_records = self._save_trigger_preview(
                    data_loader=preview_data_loader,
                    output_dir=preview_output_dir,
                    step=step_idx + 1,
                    trigger_box=trigger_boxes,
                    trigger_patch=universal_patch.detach(),
                    trigger_mask=None,
                    target_label=target_class,
                    source_filter=source_filter,
                    edge_softness=edge_softness,
                    max_images=preview_max_images,
                    how_to_attach=how_to_attach,
                )
                preview_records.extend(step_preview_records)

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                print(
                    f'[Targeted DeepFool UAP] pass={step_idx + 1}/{steps}, samples={considered_samples}, '
                    f'updates={changed_samples}, target_asr={target_asr:.4f}'
                )

        learned_patch = best_patch if best_patch is not None else universal_patch.detach().cpu()
        learned_patch_l1_norm = float(torch.norm(learned_patch.reshape(-1), p=1).item())
        learned_patch_l2_norm = float(torch.norm(learned_patch.reshape(-1), p=2).item())
        learned_patch_linf_norm = float(torch.norm(learned_patch.reshape(-1), p=float('inf')).item())
        return {
            'patch': learned_patch,
            'mask': None,
            'history': history,
            'trigger_box': trigger_boxes[0],
            'trigger_boxes': trigger_boxes,
            'target_label': float(target_class),
            'source_filter': source_filter,
            'patch_update_method': 'deepfool_uap',
            'how_to_attach': how_to_attach,
            'epsilon': learned_patch_linf_norm,
            'effective_epsilon': learned_patch_linf_norm,
            'patch_norms': {'l1': learned_patch_l1_norm, 'l2': learned_patch_l2_norm, 'linf': learned_patch_linf_norm, 'effective_linf': learned_patch_linf_norm},
            'softness': {'initial_edge_softness': float(edge_softness), 'final_edge_softness': float(edge_softness)},
            'progressive_resize': {'enabled': False, 'events': []},
            'trigger_previews': preview_records,
            'selection': 'best_targeted_attack_success_rate',
            'selected_step': int(best_step or steps),
            'best_validation_loss': None if validation_loader is None else float(best_val_loss),
            'best_validation_asr': None if validation_loader is None else float(best_val_asr),
            'smallest_success_validation_loss': None,
            'smallest_success_validation_asr': None,
            'smallest_success_patch_area': None,
        }

    def _binary_prediction(self, inputs):
        with torch.no_grad():
            return int((self.model(inputs) > 0).float().view(-1)[0].item())

    def _deepfool_binary_targeted_patch_update(self,
                                               image,
                                               trigger_boxes,
                                               universal_patch,
                                               target_label,
                                               edge_softness,
                                               how_to_attach,
                                               overshoot,
                                               max_iter,
                                               min_norm=1e-12):
        patch_update = torch.zeros_like(universal_patch, requires_grad=True)
        for _ in range(int(max_iter)):
            candidate_patch = universal_patch.detach() + patch_update
            perturbed = self._inject_trigger(
                image.unsqueeze(0),
                trigger_boxes,
                trigger_patch=candidate_patch,
                trigger_mask=None,
                edge_softness=edge_softness,
                how_to_attach=how_to_attach,
            )
            output = self.model(perturbed).view(-1)[0]
            pred = int((output > 0).float().item())
            target = int(target_label)
            if pred == target:
                return ((1.0 + float(overshoot)) * patch_update).detach()
            if patch_update.grad is not None:
                patch_update.grad.zero_()
            self.model.zero_grad(set_to_none=True)
            output.backward()
            grad = patch_update.grad.detach()
            grad_norm_sq = torch.sum(grad * grad)
            if not torch.isfinite(grad_norm_sq) or grad_norm_sq.item() <= min_norm:
                return None
            signed_distance = output.detach() / torch.sqrt(grad_norm_sq)
            step = (torch.abs(signed_distance) + min_norm) * grad / torch.sqrt(grad_norm_sq)
            with torch.no_grad():
                direction = 1.0 if int(target_label) == 1 else -1.0
                patch_update.add_(direction * step)
        return ((1.0 + float(overshoot)) * patch_update).detach()

    def _learn_robust_uap_trigger(self,
                                    data_loader,
                                    validation_loader,
                                    trigger_boxes,
                                    trigger_delta,
                                    target_label,
                                    source_filter,
                                    steps,
                                    epsilon,
                                    log_interval,
                                    trigger_preview_interval,
                                    trigger_preview_dir,
                                    trigger_preview_loader,
                                    trigger_preview_max_images,
                                    edge_softness,
                                    how_to_attach,
                                    patch_optimizer):
        target_class = int(target_label)
        if float(target_label) not in (0.0, 1.0):
            raise ValueError('Targeted RobustUAP currently supports binary target_label values 0 or 1.')
        height = int(trigger_boxes[0]['height'])
        width = int(trigger_boxes[0]['width'])
        for box in trigger_boxes[1:]:
            if int(box['width']) != width or int(box['height']) != height:
                raise ValueError('All trigger boxes must share size for RobustUAP.')

        robust_config = RobustUAPConfig(
            alpha=float(getattr(patch_optimizer, 'param_groups', [{'lr': 0.01}])[0].get('lr', 0.01))
            if patch_optimizer is not None else 0.01,
        )
        sampler = TransformSampler(height=height, width=width)
        num_transform_samples = robust_config.num_transform_samples

        history = []
        preview_records = []
        preview_interval = int(trigger_preview_interval) if trigger_preview_interval is not None else 0
        preview_max_images = (
            max(0, int(trigger_preview_max_images))
            if trigger_preview_max_images is not None else 0
        )
        preview_output_dir = Path(trigger_preview_dir) if trigger_preview_dir is not None else None
        if preview_interval > 0 and preview_output_dir is not None:
            preview_output_dir.mkdir(parents=True, exist_ok=True)
        preview_data_loader = trigger_preview_loader or validation_loader or data_loader

        self._build_cost_function('classification')

        universal_patch = project_lp_ball(epsilon * torch.tanh(trigger_delta.detach()), epsilon, robust_config.norm).detach()
        best_patch = None
        best_step = 0
        best_target_asr = float('-inf')
        best_val_asr = float('-inf')
        best_val_loss = float('inf')

        for step_idx in range(int(steps)):
            step_losses = []
            step_samples = 0
            inner_updates = 0
            batch_robustness_values = []

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
                batch_robustness = estimate_robustness(
                    model=self.model,
                    inject_trigger=self._inject_trigger,
                    inputs=selected_inputs,
                    trigger_boxes=trigger_boxes,
                    universal_patch=universal_patch,
                    sampler=sampler,
                    num_transform_samples=num_transform_samples,
                    gamma=robust_config.gamma,
                    target_label=target_class,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach,
                    max_batch_size=robust_config.max_batch_size,
                )
                batch_robustness_values.append(float(batch_robustness))

                # Keep a default no-op update for batches that are already robust or
                # configurations that request zero inner optimization steps. This
                # prevents the robust UAP path from referencing ``patch_update``
                # before it has been initialized.
                patch_update = torch.zeros_like(universal_patch, device=self.device)
                loss = None
                if batch_robustness >= robust_config.zeta:
                    continue

                for _ in range(int(robust_config.max_inner_steps)):
                    patch_update = patch_update.detach().requires_grad_(True)
                    candidate_patch = project_lp_ball(universal_patch.detach() + patch_update, epsilon, robust_config.norm)
                    loss_value = 0.0
                    self.model.zero_grad(set_to_none=True)
                    for augmentation in sampler.sample(num_transform_samples):
                        for batch_start in range(0, selected_inputs.shape[0], robust_config.max_batch_size):
                            candidate_patch = project_lp_ball(
                                universal_patch.detach() + patch_update,
                                epsilon,
                                robust_config.norm,
                            )
                            transformed_patch = augmentation(candidate_patch)
                            input_batch = selected_inputs[batch_start:batch_start + robust_config.max_batch_size]
                            poisoned_inputs = self._inject_trigger(
                                input_batch,
                                trigger_boxes,
                                trigger_patch=transformed_patch,
                                trigger_mask=None,
                                edge_softness=edge_softness,
                                how_to_attach=how_to_attach,
                            )
                            objective_outputs = self.model(poisoned_inputs)
                            target_tensor = torch.full_like(objective_outputs, float(target_label))
                            batch_loss = self.cost_function(
                                outputs=objective_outputs,
                                targets=target_tensor,
                            ).to(self.device)
                            batch_weight = float(input_batch.shape[0]) / float(selected_inputs.shape[0])
                            scaled_loss = batch_loss * batch_weight / float(num_transform_samples)
                            scaled_loss.backward()
                            loss_value += float(scaled_loss.detach().item())
                    loss = torch.tensor(loss_value, device=self.device)
                    if patch_update.grad is None:
                        break
                    with torch.no_grad():
                        signed_step = robust_config.alpha * patch_update.grad.sign()
                        patch_update = project_lp_ball(
                            patch_update - signed_step,
                            epsilon,
                            robust_config.norm,
                        )
                        candidate_patch = project_lp_ball(universal_patch.detach() + patch_update, epsilon, robust_config.norm)
                    inner_updates += 1

                    batch_robustness = estimate_robustness(
                        model=self.model,
                        inject_trigger=self._inject_trigger,
                        inputs=selected_inputs,
                        trigger_boxes=trigger_boxes,
                        universal_patch=candidate_patch.detach(),
                        sampler=sampler,
                        num_transform_samples=num_transform_samples,
                        gamma=robust_config.gamma,
                        target_label=target_class,
                        edge_softness=edge_softness,
                        how_to_attach=how_to_attach,
                        max_batch_size=robust_config.max_batch_size,
                    )
                    batch_robustness_values.append(float(batch_robustness))
                    if batch_robustness >= robust_config.zeta:
                        break

                with torch.no_grad():
                    universal_patch = project_lp_ball(universal_patch + patch_update.detach(), epsilon, robust_config.norm).detach()

                batch_samples = int(selected_inputs.shape[0])
                if loss is not None:
                    step_losses.append(float(loss.item()) * batch_samples)
                step_samples += batch_samples

            target_metrics = self.evaluate_attack_success(
                test_loader=validation_loader or data_loader,
                trigger_box=trigger_boxes,
                trigger_patch=universal_patch,
                trigger_mask=None,
                target_label=target_class,
                source_filter=source_filter,
                edge_softness=edge_softness,
                how_to_attach=how_to_attach,
            )
            val_metrics = None
            val_loss_metrics = None
            if validation_loader is not None:
                val_metrics = self.evaluate_attack_success(
                    test_loader=validation_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=universal_patch,
                    trigger_mask=None,
                    target_label=target_class,
                    source_filter=source_filter,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach,
                )
                val_loss_metrics = self.evaluate_trigger_loss(
                    data_loader=validation_loader,
                    trigger_box=trigger_boxes,
                    trigger_patch=universal_patch,
                    trigger_mask=None,
                    target_label=target_class,
                    source_filter=source_filter,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach,
                )

            target_asr = float(target_metrics['attack_success_rate'])
            if target_asr > best_target_asr:
                best_target_asr = target_asr
                best_patch = universal_patch.detach().cpu().clone()
                best_step = step_idx + 1
                if val_metrics is not None:
                    best_val_asr = float(val_metrics['attack_success_rate'])
                    best_val_loss = float(val_loss_metrics['loss'])

            step_loss = (sum(step_losses) / step_samples) if step_samples else 0.0
            step_history = {
                'step': step_idx + 1,
                'patch_update_method': 'robust_uap',
                'targeted_attack_success_rate': target_asr,
                'loss': step_loss,
                'robustness': (sum(batch_robustness_values) / len(batch_robustness_values)) if batch_robustness_values else 0.0,
                'robustness_gamma': float(robust_config.gamma),
                'robustness_zeta': float(robust_config.zeta),
                'robustness_psi': float(robust_config.psi),
                'robustness_phi': float(robust_config.phi),
                'transform_samples': int(num_transform_samples),
                'inner_updates': int(inner_updates),
                'patch_linf_norm': float(torch.norm(universal_patch.reshape(-1), p=float('inf')).item()),
                'effective_patch_linf_norm': float(torch.norm(universal_patch.reshape(-1), p=float('inf')).item()),
            }
            if val_metrics is not None:
                step_history['validation_asr'] = float(val_metrics['attack_success_rate'])
                step_history['validation_loss'] = float(val_loss_metrics['loss'])
            history.append(step_history)

            if preview_interval > 0 and preview_output_dir is not None and (step_idx + 1) % preview_interval == 0:
                step_preview_records = self._save_trigger_preview(
                    data_loader=preview_data_loader,
                    output_dir=preview_output_dir,
                    step=step_idx + 1,
                    trigger_box=trigger_boxes,
                    trigger_patch=universal_patch.detach(),
                    trigger_mask=None,
                    target_label=target_class,
                    source_filter=source_filter,
                    edge_softness=edge_softness,
                    max_images=preview_max_images,
                    how_to_attach=how_to_attach,
                )
                preview_records.extend(step_preview_records)

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                print(
                    f'[Targeted Robust UAP] '
                    f'target_asr={target_asr:.4f}, '
                    f'robustness={step_history["robustness"]:.4f}, '
                    f'inner_updates={inner_updates}'
                )

        learned_patch = best_patch if best_patch is not None else universal_patch.detach().cpu()
        learned_patch_l1_norm = float(torch.norm(learned_patch.reshape(-1), p=1).item())
        learned_patch_l2_norm = float(torch.norm(learned_patch.reshape(-1), p=2).item())
        learned_patch_linf_norm = float(torch.norm(learned_patch.reshape(-1), p=float('inf')).item())
        return {
            'patch': learned_patch,
            'mask': None,
            'history': history,
            'trigger_box': trigger_boxes[0],
            'trigger_boxes': trigger_boxes,
            'target_label': float(target_class),
            'source_filter': source_filter,
            'patch_update_method': 'robust_uap',
            'how_to_attach': how_to_attach,
            'epsilon': learned_patch_linf_norm,
            'effective_epsilon': learned_patch_linf_norm,
            'patch_norms': {'l1': learned_patch_l1_norm, 'l2': learned_patch_l2_norm, 'linf': learned_patch_linf_norm, 'effective_linf': learned_patch_linf_norm},
            'softness': {'initial_edge_softness': float(edge_softness), 'final_edge_softness': float(edge_softness)},
            'progressive_resize': {'enabled': False, 'events': []},
            'robust_uap': {
                'gamma': float(robust_config.gamma),
                'zeta': float(robust_config.zeta),
                'psi': float(robust_config.psi),
                'phi': float(robust_config.phi),
                'alpha': float(robust_config.alpha),
                'max_inner_steps': int(robust_config.max_inner_steps),
                'max_batch_size': int(robust_config.max_batch_size),
                'transform_samples': int(num_transform_samples),
                'norm': robust_config.norm,
            },
            'trigger_previews': preview_records,
            'selection': 'best_targeted_attack_success_rate',
            'selected_step': int(best_step or steps),
            'best_validation_loss': None if validation_loader is None else float(best_val_loss),
            'best_validation_asr': None if validation_loader is None else float(best_val_asr),
            'smallest_success_validation_loss': None,
            'smallest_success_validation_asr': None,
            'smallest_success_patch_area': None,
        }

    def _learn_psp_uap_trigger(self,
                                validation_loader,
                                trigger_boxes,
                                trigger_delta,
                                target_label,
                                source_filter,
                                steps,
                                epsilon,
                                log_interval,
                                trigger_preview_interval,
                                trigger_preview_dir,
                                trigger_preview_loader,
                                trigger_preview_max_images,
                                edge_softness,
                                how_to_attach,
                                patch_optimizer,
                                num_copies=10,
                            ):
        target_class = int(target_label)

        if float(target_label) not in (0.0, 1.0):
            raise ValueError(
                "Targeted PSP-UAP currently supports binary "
                "target_label values 0 or 1."
            )
        height = int(trigger_boxes[0]["height"])
        width = int(trigger_boxes[0]["width"])

        for box in trigger_boxes[1:]:
            if (int(box["width"]) != width or int(box["height"]) != height):
                raise ValueError("All trigger boxes must share size for PSP-UAP.")

        history = []
        preview_records = []

        preview_interval = (int(trigger_preview_interval) if trigger_preview_interval is not None else 0)

        preview_max_images = (
            max(0, int(trigger_preview_max_images))
            if trigger_preview_max_images is not None
            else 0
        )

        preview_output_dir = (
            Path(trigger_preview_dir)
            if trigger_preview_dir is not None
            else None
        )

        if (preview_interval > 0 and preview_output_dir is not None):
            preview_output_dir.mkdir(parents=True, exist_ok=True)

        preview_data_loader = (trigger_preview_loader or validation_loader)

        self._build_cost_function("psp_uap")
        psp_sampler = PSPTransformSampler(image_size=(height, width), device=self.device)

        num_copies = int(num_copies)
        if num_copies <= 0:
            raise ValueError("psp_num_copies must be a positive integer.")

        if not torch.is_tensor(trigger_delta):
            raise TypeError(
                "trigger_delta must be a torch.Tensor."
            )

        if not trigger_delta.requires_grad:
            trigger_delta.requires_grad_(True)

        best_patch = None
        best_step = 0

        best_target_asr = float("-inf")
        best_val_asr = float("-inf")
        best_val_loss = float("inf")

        for step_idx in range(int(steps)):

            step_losses = []
            step_samples = 0

            universal_patch = (epsilon*torch.tanh(trigger_delta))

            semantic_prior, semantic_delta = (
                psp_sampler.sample(
                    delta=universal_patch,
                    num_copies=num_copies,
                    input_transform=True,
                )
            )

            semantic_prior = semantic_prior.to(self.device)
            semantic_delta = semantic_delta.to(self.device)

            # Clean samples provide only the KL teacher logits. Disabling the
            # hooks avoids retaining a second copy of every convolution map.
            with torch.no_grad(), self.feature_extractor.suspend_capture():
                semantic_logits = self.model(semantic_prior)

            self.feature_extractor.clear()

            delta_logits = self.model(semantic_delta)

            adv_features = self.feature_extractor.activations

            batch_loss = self.cost_function(
                outputs=adv_features,
                semantic_logits=semantic_logits,
                delta_logits=delta_logits,
            )

            patch_optimizer.zero_grad(set_to_none=True)

            batch_loss.backward()

            patch_optimizer.step()
            self.feature_extractor.clear()

            with torch.no_grad():
                universal_patch = (epsilon*torch.tanh(trigger_delta))

            batch_loss_value = float(batch_loss.detach().item())

            step_losses.append(batch_loss_value)

            step_samples += int(semantic_prior.shape[0])

            
            step_loss = (sum(step_losses)/ len(step_losses) if step_losses else 0.0)

            patch_linf = float(universal_patch.detach().reshape(-1).abs().max().item())

            step_history = {
                "step": step_idx + 1,
                "patch_update_method": "psp_uap",
                "loss": step_loss,
                "patch_linf_norm": patch_linf,
                "effective_patch_linf_norm": patch_linf,
            }

            history.append(step_history)

            if (preview_interval > 0 and preview_output_dir is not None and (step_idx + 1) % preview_interval == 0):

                step_preview_records = (
                    self._save_trigger_preview(
                        data_loader=preview_data_loader,
                        output_dir=preview_output_dir,
                        step=step_idx + 1,
                        trigger_box=trigger_boxes,
                        trigger_patch=universal_patch.detach(),
                        trigger_mask=None,
                        target_label=target_class,
                        source_filter=source_filter,
                        edge_softness=edge_softness,
                        max_images=preview_max_images,
                        how_to_attach=how_to_attach,
                    )
                )

                preview_records.extend(step_preview_records)

            if (log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0):

                target_metrics = (
                    self.evaluate_attack_success(
                        test_loader=validation_loader,
                        trigger_box=trigger_boxes,
                        trigger_patch=universal_patch.detach(),
                        trigger_mask=None,
                        target_label=target_class,
                        source_filter=source_filter,
                        edge_softness=edge_softness,
                        how_to_attach=how_to_attach,
                    )
                )

                val_metrics = None


                if validation_loader is not None:
                    val_metrics = (
                        self.evaluate_attack_success(
                            test_loader=validation_loader,
                            trigger_box=trigger_boxes,
                            trigger_patch=universal_patch.detach(),
                            trigger_mask=None,
                            target_label=target_class,
                            source_filter=source_filter,
                            edge_softness=edge_softness,
                            how_to_attach=how_to_attach,
                        )
                    )

                target_asr = float(target_metrics["attack_success_rate"])

                if target_asr > best_target_asr:
                    best_target_asr = target_asr
                    best_patch = (universal_patch.detach().cpu().clone())
                    best_step = step_idx + 1
                    if val_metrics is not None:
                        best_val_asr = float(
                            val_metrics["attack_success_rate"])
                print(
                    "[Targeted PSP-UAP] "
                    f"step={step_idx + 1} "
                    f"loss={step_loss:.6f} "
                    f"target_asr={target_asr:.4f} "
                    f"linf={patch_linf:.6f}"
                )

        if best_patch is not None:
            learned_patch = best_patch
        else:
            with torch.no_grad():
                learned_patch = (epsilon* torch.tanh(trigger_delta)).detach().cpu()

        learned_patch_l1_norm = float(torch.norm(learned_patch.reshape(-1),p=1,).item())
        learned_patch_l2_norm = float(torch.norm(learned_patch.reshape(-1), p=2,).item())
        learned_patch_linf_norm = float(learned_patch.reshape(-1).abs().max().item())

        # Do not leave hooks alive after optimization; later inference would
        # otherwise retain feature maps that no caller consumes.
        self._remove_feature_extractor()

        return {
            "patch": learned_patch,
            "mask": None,
            "history": history,
            "trigger_box": trigger_boxes[0],
            "trigger_boxes": trigger_boxes,
            "target_label": float(target_class),
            "source_filter": source_filter,
            "patch_update_method": "psp_uap",
            "how_to_attach": how_to_attach,
            "epsilon": learned_patch_linf_norm,
            "effective_epsilon": learned_patch_linf_norm,
            "patch_norms": {
                "l1": learned_patch_l1_norm,
                "l2": learned_patch_l2_norm,
                "linf": learned_patch_linf_norm,
                "effective_linf": learned_patch_linf_norm,
            },
            "softness": {
                "initial_edge_softness": float(edge_softness),
                "final_edge_softness": float(edge_softness),
            },
            "progressive_resize": {
                "enabled": False,
                "events": [],
            },
            "trigger_previews": preview_records,
            "selection": ("best_targeted_attack_success_rate"),
            "selected_step": int(best_step or steps),
            "best_validation_asr": (None if validation_loader is None else float(best_val_asr)),
            "smallest_success_validation_loss": None,
            "smallest_success_validation_asr": None,
            "smallest_success_patch_area": None,
        }

    @staticmethod
    def _image_tensor_to_pil(image_tensor, scale_from_signed=False):
        image_tensor = image_tensor.detach().cpu().float()
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        elif image_tensor.shape[0] > 3:
            image_tensor = image_tensor[:3]

        if scale_from_signed:
            image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        image_array = (image_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(image_array)

    def _save_trigger_preview(
        self,
        data_loader,
        output_dir,
        step,
        trigger_box,
        trigger_patch,
        trigger_mask,
        target_label,
        source_filter,
        edge_softness,
        max_images=5,
        how_to_attach='blend'
    ):
        if data_loader is None or max_images <= 0:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trigger_boxes = self._normalize_trigger_boxes(trigger_box)
        target_value = float(target_label)
        max_images = int(max_images)
        successful_examples = []
        fallback_examples = []

        self.model.eval()
        with torch.no_grad():
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

                source_inputs = inputs[source_mask].clone()
                source_targets = targets[source_mask].view(-1)
                poisoned_inputs = self._inject_trigger(
                    source_inputs.clone(),
                    trigger_boxes,
                    trigger_patch=trigger_patch,
                    trigger_mask=trigger_mask,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach

                )
                poisoned_outputs = self.model(poisoned_inputs)
                poisoned_preds = (poisoned_outputs > 0).float().view(-1)
                success_mask = (poisoned_preds == target_value)

                for candidate_idx in torch.nonzero(success_mask, as_tuple=False).view(-1).tolist():
                    successful_examples.append({
                        'clean': source_inputs[candidate_idx].detach().cpu(),
                        'poisoned': poisoned_inputs[candidate_idx].detach().cpu(),
                        'source_label': float(source_targets[candidate_idx].detach().cpu().item()),
                        'predicted_label': float(poisoned_preds[candidate_idx].detach().cpu().item()),
                        'success': True,
                    })
                    if len(successful_examples) >= max_images:
                        break

                if len(fallback_examples) < max_images:
                    for candidate_idx in range(int(source_inputs.shape[0])):
                        if bool(success_mask[candidate_idx].detach().cpu().item()):
                            continue
                        fallback_examples.append({
                            'clean': source_inputs[candidate_idx].detach().cpu(),
                            'poisoned': poisoned_inputs[candidate_idx].detach().cpu(),
                            'source_label': float(source_targets[candidate_idx].detach().cpu().item()),
                            'predicted_label': float(poisoned_preds[candidate_idx].detach().cpu().item()),
                            'success': False,
                        })
                        if len(fallback_examples) >= max_images:
                            break

                if len(successful_examples) >= max_images:
                    break

        selected_examples = successful_examples[:max_images]
        if len(selected_examples) < max_images:
            selected_examples.extend(fallback_examples[:max_images - len(selected_examples)])
        if not selected_examples:
            return []

        patch_preview = trigger_patch.detach().cpu()
        if patch_preview.dim() == 4:
            patch_preview = patch_preview[0]
        mask_preview = trigger_mask.detach().cpu() if trigger_mask is not None else None
        if mask_preview is not None and mask_preview.dim() == 4:
            mask_preview = mask_preview[0]

        patch_image = self._image_tensor_to_pil(patch_preview, scale_from_signed=True)
        if mask_preview is None:
            mask_image = Image.new('RGB', patch_image.size, color=(0, 0, 0))
        else:
            mask_image = self._image_tensor_to_pil(mask_preview)

        records = []
        nearest_resample = getattr(Image, 'Resampling', Image).NEAREST
        for preview_idx, example in enumerate(selected_examples, start=1):
            clean_image = self._image_tensor_to_pil(example['clean'])
            poisoned_image = self._image_tensor_to_pil(example['poisoned'])
            panel_width, panel_height = clean_image.size
            resized_patch_image = patch_image.resize((panel_width, panel_height), nearest_resample)
            resized_mask_image = mask_image.resize((panel_width, panel_height), nearest_resample)
            poisoned_image = poisoned_image.resize((panel_width, panel_height), nearest_resample)

            preview_image = Image.new('RGB', (panel_width * 4, panel_height))
            preview_image.paste(clean_image, (0, 0))
            preview_image.paste(poisoned_image, (panel_width, 0))
            preview_image.paste(resized_patch_image, (panel_width * 2, 0))
            preview_image.paste(resized_mask_image, (panel_width * 3, 0))

            status = 'success' if example['success'] else 'candidate'
            filename = f'trigger_preview_step_{int(step):04d}_{preview_idx:02d}_{status}.png'
            preview_path = output_dir / filename
            preview_image.save(preview_path)

            records.append({
                'step': int(step),
                'path': str(preview_path),
                'success': bool(example['success']),
                'source_label': example['source_label'],
                'predicted_label': example['predicted_label'],
                'target_label': target_value,
                'preview_index': int(preview_idx),
                'max_images': int(max_images),
                'layout': 'clean|poisoned|patch|mask',
            })

        return records

    def evaluate_trigger_loss(self,
                              data_loader,
                              trigger_box,
                              trigger_patch=None,
                              trigger_mask=None,
                              target_label=0.0,
                              source_filter='all',
                              edge_softness=0.2,
                              mask_l1_weight=0.0,
                              patch_l2_weight=0.0,
                              softness_alignment_weight=0.0,
                              how_to_attach='blend',
                              use_clean_feature_targets=False):
        self.model.eval()
        losses = []
        attack_losses = []
        total = 0

        patch_reg = 0.0
        mask_reg = 0.0
        softness_reg = 0.0
        if trigger_patch is not None:
            patch_tensor = trigger_patch.to(device=self.device, dtype=torch.float32)
            patch_reg = float(patch_l2_weight) * float(torch.mean(patch_tensor ** 2).item())

            if trigger_mask is not None:
                mask_tensor = trigger_mask.to(device=self.device, dtype=patch_tensor.dtype)
                if patch_tensor.dim() == 3:
                    _, patch_height, patch_width = patch_tensor.shape
                elif patch_tensor.dim() == 4:
                    _, _, patch_height, patch_width = patch_tensor.shape
                else:
                    raise ValueError('trigger_patch must be CHW or NCHW tensor-like.')
                channels = int(mask_tensor.shape[-3])
                base_mask = self._build_blend_mask(
                    height=int(patch_height),
                    width=int(patch_width),
                    channels=channels,
                    device=self.device,
                    dtype=mask_tensor.dtype,
                    edge_softness=edge_softness,
                )
                if mask_tensor.dim() == 4:
                    base_mask = base_mask.expand(mask_tensor.shape[0], -1, -1, -1)
                mask_growth = torch.relu(mask_tensor - base_mask)
                mask_reg = float(mask_l1_weight) * float(torch.mean(mask_growth).item())
                softness_reg = float(softness_alignment_weight) * float(
                    torch.mean((mask_tensor - base_mask) ** 2).item()
                )

        regularization_loss = patch_reg + mask_reg + softness_reg
        cost_function = getattr(self, 'cost_function', None)
        if cost_function is None:
            cost_function = ClassificationObjective()
        use_feature_objective = isinstance(cost_function, FeaturBaseObjective)

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)
                flat_targets = targets.view(-1)

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

                selected_inputs = inputs[source_mask].clone()
                poisoned_inputs = self._inject_trigger(
                    selected_inputs,
                    trigger_box,
                    trigger_patch=trigger_patch,
                    trigger_mask=trigger_mask,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach
                )
                feature_extractor = getattr(self, 'feature_extractor', None)
                if feature_extractor is None and use_feature_objective:
                    feature_extractor = getattr(cost_function, 'feature_extractor', None)
                if use_feature_objective and feature_extractor is None:
                    raise RuntimeError('Feature objective evaluation requires a feature extractor.')
                if feature_extractor is not None:
                    feature_extractor.clear()
                if use_feature_objective and use_clean_feature_targets:
                    self.model(selected_inputs)
                    target_tensor = cost_function.detach_targets(feature_extractor.activations)
                    feature_extractor.clear()
                else:
                    target_tensor = None
                model_outputs = self.model(poisoned_inputs)
                objective_outputs = feature_extractor.activations if use_feature_objective else model_outputs
                if not use_feature_objective:
                    target_tensor = torch.full_like(model_outputs, float(target_label))
                attack_loss = cost_function(objective_outputs, target_tensor).to(self.device)
                loss = float(attack_loss.item()) + regularization_loss
                batch_size = int(model_outputs.shape[0])
                losses.append(loss * batch_size)
                attack_losses.append(float(attack_loss.item()) * batch_size)
                total += batch_size

        attack_loss = (sum(attack_losses) / total) if total else float('inf')
        return {
            'loss': (sum(losses) / total) if total else float('inf'),
            'attack_loss': attack_loss,
            'patch_regularization_loss': patch_reg,
            'mask_regularization_loss': mask_reg,
            'softness_alignment_loss': softness_reg,
            'samples_evaluated': total,
            'target_label': float(target_label),
            'trigger_box': trigger_box,
            'source_filter': source_filter,
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
                                 edge_softness=0.2,
                                 how_to_attach='blend'):
        self.model.eval()
        target_tensor = torch.tensor(target_label, dtype=torch.float32, device=self.device).view(1, -1)

        total = 0
        attack_success = 0
        clean_correct = 0
        clean_correct_and_not_target = 0
        conditional_attack_success = 0

        effective_source_filter = source_filter
        if effective_source_filter is None:
            effective_source_filter = 'bad' if source_only_bad else 'all'
        if effective_source_filter not in {'bad', 'good', 'all'}:
            raise ValueError("source_filter must be one of: 'bad', 'good', 'all'.")

        feature_extractor = getattr(self, 'feature_extractor', None)
        capture_context = (
            feature_extractor.suspend_capture()
            if feature_extractor is not None
            else nullcontext()
        )

        # ASR evaluation needs logits only. Feature hooks can otherwise retain
        # an entire loader's convolution maps between optimization steps.
        with torch.no_grad(), capture_context:
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)
                flat_targets = targets.view(-1)

                if effective_source_filter == 'bad':
                    source_mask = (flat_targets == 0)
                elif effective_source_filter == 'good':
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
                eligible = (
                    (clean_preds == clean_targets)
                    & (clean_targets != float(target_label))
                )
                clean_correct_and_not_target += int(eligible.sum().item())

                poisoned_inputs = self._inject_trigger(
                    source_inputs.clone(),
                    trigger_box,
                    trigger_value=trigger_value,
                    trigger_patch=trigger_patch,
                    trigger_mask=trigger_mask,
                    edge_softness=edge_softness,
                    how_to_attach=how_to_attach
                )
                poisoned_outputs = self.model(poisoned_inputs)
                poisoned_preds = (poisoned_outputs > 0).float()

                expanded_target = target_tensor.expand(poisoned_preds.shape[0], -1)
                successful = (poisoned_preds == expanded_target).view(-1)
                attack_success += int(successful.sum().item())
                conditional_attack_success += int((successful & eligible).sum().item())
                total += int(poisoned_preds.shape[0])

        return {
            'samples_evaluated': total,
            'clean_source_accuracy': (clean_correct / total) * 100 if total else 0.0,
            'attack_success_rate': (attack_success / total) * 100 if total else 0.0,
            'clean_not_target_count': clean_correct_and_not_target,
            'conditional_attack_success_rate': (
                (conditional_attack_success / clean_correct_and_not_target) * 100
                if clean_correct_and_not_target else 0.0
            ),
            'target_label': float(target_label),
            'trigger_box': trigger_box,
            'source_filter': effective_source_filter,
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
        
    @staticmethod
    def _inject_trigger(
        inputs,
        trigger_box,
        trigger_value=(1.0, 1.0, 1.0),
        trigger_patch=None,
        trigger_mask=None,
        edge_softness=0.2,
        how_to_attach='blend'):
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

        # per_sample_boxes = len(trigger_boxes) == poisoned_inputs.shape[0]
        # if per_sample_boxes and patch_bank is not None:
        #     patch_count_valid = patch_bank.shape[0] in (1, poisoned_inputs.shape[0])
        #     mask_count_valid = mask_bank is None or mask_bank.shape[0] in (1, poisoned_inputs.shape[0])
        #     if patch_count_valid and mask_count_valid:
        #         for sample_idx, box in enumerate(trigger_boxes):
        #             x = int(box['x'])
        #             y = int(box['y'])
        #             width = int(box['width'])
        #             height = int(box['height'])

        #             if x < 0 or y < 0 or x + width > input_w or y + height > input_h:
        #                 raise ValueError('trigger_box is out of image bounds.')

        #             region = poisoned_inputs[sample_idx:sample_idx + 1, :, y:y + height, x:x + width].clone()
        #             if patch_bank.shape[1] != channels or patch_bank.shape[2] != height or patch_bank.shape[3] != width:
        #                 raise ValueError('trigger_patch shape must match (C, height, width) from trigger_box.')
        #             patch = patch_bank[0 if patch_bank.shape[0] == 1 else sample_idx].unsqueeze(0)

        #             if mask_bank is not None:
        #                 if mask_bank.shape[1] != channels or mask_bank.shape[2] != height or mask_bank.shape[3] != width:
        #                     raise ValueError('trigger_mask shape must match (C, height, width) from trigger_box.')
        #                 blend_mask = mask_bank[0 if mask_bank.shape[0] == 1 else sample_idx].unsqueeze(0)
        #                 blend_mask = torch.clamp(blend_mask, 0.0, 1.0)
        #             else:
        #                 blend_mask = AdversarialAttack._build_blend_mask(
        #                     height=height,
        #                     width=width,
        #                     channels=channels,
        #                     device=poisoned_inputs.device,
        #                     dtype=poisoned_inputs.dtype,
        #                     edge_softness=edge_softness,
        #                 )

        #             poisoned_inputs[sample_idx:sample_idx + 1, :, y:y + height, x:x + width] = torch.clamp(
        #                 region + patch * blend_mask,
        #                 0.0,
        #                 1.0,
        #             )
        #         return poisoned_inputs

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
                if how_to_attach == 'replace':
                    blended_region = torch.clamp(patch*blend_mask, 0, 1)
                else:
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
                blended_region = target_region

            poisoned_inputs[:, :, y:y + height, x:x + width] = blended_region
        return poisoned_inputs
