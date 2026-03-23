import torch
from typing import Callable, Optional
from Network import ClassificationModels


class AdversarialAttack:
    def __init__(self, 
                model: Callable,
                device: Optional[str] = None
                ):
                    
        self.model = model
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self._build_cost_function()
    
    def _build_cost_function(self):
        self.cost_function = torch.nn.BCEWithLogitsLoss()
        return self.cost_function

    def learn_universal_trigger(self,
                                data_loader,
                                trigger_box,
                                target_label=(0.0, 0.0),
                                source_filter='bad',
                                steps=100,
                                learning_rate=0.1,
                                epsilon=0.08,
                                log_interval=1):
        self.model.eval()

        x = int(trigger_box['x'])
        y = int(trigger_box['y'])
        width = int(trigger_box['width'])
        height = int(trigger_box['height'])

        channels = 3
        trigger_delta = torch.zeros((1, channels, height, width), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([trigger_delta], lr=learning_rate)
        target_tensor_base = torch.tensor(target_label, dtype=torch.float32, device=self.device)

        history = []

        for step_idx in range(steps):
            step_losses = []
            step_samples = 0

            batch_counter = 0
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)

                if source_filter == 'bad':
                    source_mask = (targets == 0).any(dim=1)
                elif source_filter == 'good':
                    source_mask = (targets[:, 0] == 1) & (targets[:, 1] == 1)
                else:
                    source_mask = torch.ones(targets.shape[0], dtype=torch.bool, device=self.device)

                if source_mask.sum().item() == 0:
                    continue

                selected_inputs = inputs[source_mask].clone()
                poisoned_inputs = self._inject_trigger(selected_inputs, trigger_box, trigger_patch=trigger_delta)

                outputs = self.model(poisoned_inputs)
                target_tensor = target_tensor_base.unsqueeze(0).expand(outputs.shape[0], -1)
                loss = self.cost_function(outputs, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_losses.append(float(loss.item()))
                step_samples += int(outputs.shape[0])
                batch_counter += 1

            step_loss = float(np.mean(step_losses)) if step_losses else 0.0
            history.append({
                'step': step_idx + 1,
                'loss': step_loss,
                'samples': step_samples,
            })

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                print(
                    f'[Trigger Learning] step={step_idx + 1}/{steps}, '
                    f'loss={step_loss:.6f}, samples={step_samples}'
                )

        learned_patch = trigger_delta.detach().squeeze(0).cpu()
        return {
            'patch': learned_patch,
            'history': history,
            'trigger_box': trigger_box,
            'target_label': tuple(float(v) for v in target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
        }

    def evaluate_attack_success(self,
                                 test_loader,
                                 trigger_box,
                                 trigger_value=(1.0, 1.0, 1.0),
                                 trigger_patch=None,
                                 target_label=(0.0, 0.0),
                                 source_only_bad=False):
        self.model.eval()
        target_tensor = torch.tensor(target_label, dtype=torch.float32, device=self.device).view(1, -1)

        total = 0
        attack_success = 0
        clean_correct = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)

                if source_only_bad:
                    source_mask = (targets == 0).any(dim=1)
                else:
                    source_mask = torch.ones(targets.shape[0], dtype=torch.bool, device=self.device)

                if source_mask.sum().item() == 0:
                    continue

                source_inputs = inputs[source_mask]
                source_targets = targets[source_mask]

                clean_outputs = self.model(source_inputs)
                clean_preds = (clean_outputs > 0).float()
                clean_correct += int((clean_preds == source_targets).all(dim=1).sum().item())

                poisoned_inputs = self._inject_trigger(
                    source_inputs.clone(),
                    trigger_box,
                    trigger_value=trigger_value,
                    trigger_patch=trigger_patch,
                )
                poisoned_outputs = self.model(poisoned_inputs)
                poisoned_preds = (poisoned_outputs > 0).float()

                expanded_target = target_tensor.expand(poisoned_preds.shape[0], -1)
                attack_success += int((poisoned_preds == expanded_target).all(dim=1).sum().item())
                total += int(poisoned_preds.shape[0])

        return {
            'samples_evaluated': total,
            'clean_source_accuracy': (clean_correct / total) * 100 if total else 0.0,
            'attack_success_rate': (attack_success / total) * 100 if total else 0.0,
            'target_label': tuple(float(v) for v in target_label),
            'trigger_box': trigger_box,
        }

    @staticmethod
    def _inject_trigger(inputs, trigger_box, trigger_value=(1.0, 1.0, 1.0), trigger_patch=None):
        x = int(trigger_box['x'])
        y = int(trigger_box['y'])
        width = int(trigger_box['width'])
        height = int(trigger_box['height'])

        _, channels, input_h, input_w = inputs.shape
        if x < 0 or y < 0 or x + width > input_w or y + height > input_h:
            raise ValueError('trigger_box is out of image bounds.')

        if trigger_patch is not None:
            if not torch.is_tensor(trigger_patch):
                trigger_patch = torch.tensor(trigger_patch, dtype=inputs.dtype, device=inputs.device)
            trigger_patch = trigger_patch.to(device=inputs.device, dtype=inputs.dtype)

            if trigger_patch.dim() == 3:
                patch = trigger_patch.unsqueeze(0)
            elif trigger_patch.dim() == 4:
                patch = trigger_patch
            else:
                raise ValueError('trigger_patch must be CHW or NCHW tensor-like.')

            if patch.shape[1] != channels or patch.shape[2] != height or patch.shape[3] != width:
                raise ValueError('trigger_patch shape must match (C, height, width) from trigger_box.')

            if patch.shape[0] == 1:
                patch = patch.expand(inputs.shape[0], -1, -1, -1)
            elif patch.shape[0] != inputs.shape[0]:
                raise ValueError('trigger_patch batch dimension must be 1 or equal to input batch size.')

            patched_region = torch.clamp(inputs[:, :, y:y + height, x:x + width] + patch, 0.0, 1.0)
            inputs[:, :, y:y + height, x:x + width] = patched_region
            return inputs

        trigger_tensor = torch.tensor(trigger_value, dtype=inputs.dtype, device=inputs.device).view(channels, 1, 1)
        if trigger_tensor.shape[0] != channels:
            raise ValueError('trigger_value channel count must match input channels.')

        inputs[:, :, y:y + height, x:x + width] = trigger_tensor
        return inputs