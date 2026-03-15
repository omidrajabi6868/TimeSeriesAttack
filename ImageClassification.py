import json
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

import torch
import numpy as np

from Network import ClassificationModels


class ClassificationBase:
    def __init__(
        self,
        model_name: str,
        optimizer_name: str = 'Adam',
        checkpoint_dir: str = 'checkpoints',
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = None
        self.cost_function = None
        self.optimizer = None

    def _build_model(self):
        if self.model_name == 'ResNet18':
            self.model = ClassificationModels.ResNet('18', 2).model
        elif self.model_name == 'ResNet34':
            self.model = ClassificationModels.ResNet('34', 2).model
        elif self.model_name == 'ResNet50':
            self.model = ClassificationModels.ResNet('50', 2).model
        elif self.model_name == 'ResNet101':
            self.model = ClassificationModels.ResNet('101', 2).model
        else:
            raise ValueError(f'Unsupported model_name: {self.model_name}')

        self.model = self.model.to(self.device)
        return self.model

    def _build_cost_function(self):
        self.cost_function = torch.nn.BCEWithLogitsLoss()
        return self.cost_function

    def _build_optimization_algorithm(self, params, learning_rate: float):
        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(params=params, lr=learning_rate)
        elif self.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=params, lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f'Unsupported optimizer_name: {self.optimizer_name}')

        return self.optimizer

    def _save_checkpoint(self, checkpoint_path: Path, epoch: int, best_val_loss: float, history: dict):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'model_name': self.model_name,
                'optimizer_name': self.optimizer_name,
                'history': history,
            },
            checkpoint_path,
        )

    def _save_history_json(self, history: dict):
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as file:
            json.dump(history, file, indent=2)

    def _plot_history(self, history: dict):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not installed; skipping training history plot generation.')
            return

        epochs = history.get('epoch', [])
        if not epochs:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(epochs, history.get('train_loss', []), label='Train Loss')
        axes[0].plot(epochs, history.get('val_loss', []), label='Validation Loss')
        axes[0].set_title('Loss vs Epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, history.get('train_accuracy', []), label='Train Accuracy')
        axes[1].plot(epochs, history.get('val_accuracy', []), label='Validation Accuracy')
        axes[1].set_title('Accuracy vs Epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        plot_path = self.checkpoint_dir / 'training_curves.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        if self.model is None:
            self._build_model()
        if self.cost_function is None:
            self._build_cost_function()
        if self.optimizer is None:
            self._build_optimization_algorithm(self.model.parameters(), learning_rate=1e-3)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history')

        return start_epoch, best_val_loss, history

    def train_model(
        self,
        train_loader: Callable,
        val_loader: Callable,
        learning_rate: float = 1e-3,
        epoch_num: int = 10,
        resume_from: Optional[str] = None,
    ):
        self._build_model()
        self._build_cost_function()
        self._build_optimization_algorithm(self.model.parameters(), learning_rate)

        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
        }

        start_epoch = 0
        best_val_loss = float('inf')

        if resume_from is not None:
            start_epoch, best_val_loss, loaded_history = self.load_checkpoint(resume_from, load_optimizer=True)
            if loaded_history is not None:
                history = loaded_history
        print('Training starts ...')
        for epoch in range(start_epoch, epoch_num):
            train_loss = []
            self.model.train()
            total_num = 0
            correct_pairs = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)

                outputs = self.model(inputs)
                loss = self.cost_function(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                preds = (outputs > 0).float()

                correct_pairs += (preds == targets).all(dim=1).sum().item()
                total_num += int(inputs.shape[0])
            
            avg_train_loss = mean(train_loss) if train_loss else 0.0
            train_acc = (correct_pairs / total_num) * 100 if total_num else 0.0
            print(f'Epoch {epoch + 1}: train_loss={avg_train_loss:.5f}, train_accuracy={train_acc:.2f}')


            with torch.no_grad():
                metrics = self.evaluate_model(val_loader)
                val_loss = metrics['loss']
                val_acc = metrics['accuracy']
                print(f'val_loss={val_loss:.5f}, val_accuracy={val_acc:.2f}')

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)

            self._save_history_json(history)
            self._plot_history(history)

            last_ckpt_path = self.checkpoint_dir / 'last_checkpoint.pth'
            self._save_checkpoint(last_ckpt_path, epoch, best_val_loss, history)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = self.checkpoint_dir / 'best_checkpoint.pth'
                self._save_checkpoint(best_ckpt_path, epoch, best_val_loss, history)

        return self.model, history

    def evaluate_model(self, test_loader):
        self.model.eval()
        losses = []
        total_num = 0
        correct_pairs = 0
        gg_total = 0
        gg_correct = 0
        others_total = 0
        others_correct = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.float().to(self.device)

            outputs = self.model(inputs)
            loss = self.cost_function(outputs, targets)
            losses.append(loss.item())

            preds = (outputs > 0).float()

            correct_pairs += (preds == targets).all(dim=1).sum().item()
            total_num += int(inputs.shape[0])

            per_sample_correct = (preds == targets).all(dim=1)
            good_good_mask = (targets[:, 0] == 1) & (targets[:, 1] == 1)
            others_mask = ~good_good_mask

            gg_total += int(good_good_mask.sum().item())
            gg_correct += int(per_sample_correct[good_good_mask].sum().item())
            others_total += int(others_mask.sum().item())
            others_correct += int(per_sample_correct[others_mask].sum().item())

        accuracy = (correct_pairs / total_num) * 100 if total_num else 0.0
        return {
            'loss': mean(losses) if losses else 0.0,
            'accuracy': accuracy,
            'good_good_accuracy': (gg_correct / gg_total) * 100 if gg_total else 0.0,
            'others_accuracy': (others_correct / others_total) * 100 if others_total else 0.0,
            'good_good_count': gg_total,
            'others_count': others_total,
        }

    def learn_universal_trigger(self,
                                data_loader,
                                trigger_box,
                                target_label=(0.0, 0.0),
                                source_filter='bad',
                                steps=100,
                                learning_rate=0.1,
                                epsilon=0.08,
                                max_batches_per_step=2):
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

        for _ in range(steps):
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

                with torch.no_grad():
                    trigger_delta.clamp_(-epsilon, epsilon)

                step_losses.append(float(loss.item()))
                step_samples += int(outputs.shape[0])
                batch_counter += 1
                if max_batches_per_step is not None and batch_counter >= max_batches_per_step:
                    break

            history.append({
                'loss': float(np.mean(step_losses)) if step_losses else 0.0,
                'samples': step_samples,
            })

        learned_patch = trigger_delta.detach().squeeze(0).cpu()
        return {
            'patch': learned_patch,
            'history': history,
            'trigger_box': trigger_box,
            'target_label': tuple(float(v) for v in target_label),
            'source_filter': source_filter,
            'epsilon': float(epsilon),
        }

    def evaluate_backdoor_success(self,
                                 test_loader,
                                 trigger_box,
                                 trigger_value=(1.0, 1.0, 1.0),
                                 trigger_patch=None,
                                 target_label=(0.0, 0.0),
                                 source_only_bad=True):
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