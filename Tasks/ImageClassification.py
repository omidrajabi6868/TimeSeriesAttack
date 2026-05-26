import json
from pathlib import Path
from statistics import mean
from typing import Callable, Optional, Sequence

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

from Network import ClassificationModels


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


class ClassificationBase:
    def __init__(
        self,
        model_name: str,
        optimizer_name: str = 'Adam',
        checkpoint_dir: str = 'checkpoints',
        device: Optional[str] = None,
        use_multi_gpu: bool = True,
        gpu_ids: Optional[Sequence[int]] = None):
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
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None

    def _build_model(self):
        if self.model_name == 'ResNet18':
            self.model = ClassificationModels.ResNet('18', 1).model
        elif self.model_name == 'ResNet34':
            self.model = ClassificationModels.ResNet('34', 1).model
        elif self.model_name == 'ResNet50':
            self.model = ClassificationModels.ResNet('50', 1).model
        elif self.model_name == 'ResNet101':
            self.model = ClassificationModels.ResNet('101', 1).model
        elif self.model_name == "AlexNet":
            self.model = ClassificationModels.AlexNet('', 1).model
        else:
            raise ValueError(f'Unsupported model_name: {self.model_name}')

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
        return self.model

    def _build_cost_function(self):
        self.cost_function = torch.nn.BCEWithLogitsLoss()
        return self.cost_function

    def _build_weighted_cost_function(self, pos_weight: Optional[float]):
        if pos_weight is None:
            return self._build_cost_function()
        pos_weight_tensor = torch.tensor([float(pos_weight)], device=self.device)
        self.cost_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        return self.cost_function

    @staticmethod
    def build_weighted_sampler_from_labels(labels):
        labels_np = np.array(labels, dtype=np.int64)
        class_counts = np.bincount(labels_np, minlength=2)
        if np.any(class_counts == 0):
            return None
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels_np]
        sample_weights_t = torch.from_numpy(sample_weights.astype(np.float64))
        return WeightedRandomSampler(weights=sample_weights_t, num_samples=len(sample_weights_t), replacement=True)

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
                'model_state_dict': self._stateful_model().state_dict(),
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

    def _stateful_model(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        if self.model is None:
            self._build_model()
        if self.cost_function is None:
            self._build_cost_function()
        if self.optimizer is None:
            self._build_optimization_algorithm(self.model.parameters(), learning_rate=1e-3)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_state = _strip_module_prefix(checkpoint['model_state_dict'])
        self._stateful_model().load_state_dict(checkpoint_state)

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
        resume: bool = False,
        resume_from: Optional[str] = None,
        pos_weight: Optional[float] = None,
        noise_probability_check: bool = False,
        noise_regularization_weight: float = 0.0,
        input_shape: Sequence[int] = (3, 256, 608),
    ):
        self._build_model()
        self._build_weighted_cost_function(pos_weight)
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

        if resume_from is not None and resume == True:
            start_epoch, best_val_loss, loaded_history = self.load_checkpoint(resume_from, load_optimizer=True)
            if loaded_history is not None:
                history = loaded_history
        print('Training starts ...')
        for epoch in range(start_epoch, epoch_num):
            train_loss = []
            self.model.train()
            total_num = 0
            correct = 0
            noise_probabilities = []
            noise_reg_losses = []
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().unsqueeze(-1).to(self.device)
                

                outputs = self.model(inputs)
                classification_loss = self.cost_function(outputs, targets)
                loss = classification_loss

                if noise_regularization_weight > 0.0:
                    noise_reg_loss = self._noise_regularization_loss(
                        batch_size=int(inputs.shape[0]),
                        input_shape=input_shape,
                    )
                    noise_reg_losses.append(float(noise_reg_loss.detach().cpu().item()))
                    loss = classification_loss + (noise_regularization_weight * noise_reg_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                preds = (outputs > 0).float()

                correct += (preds == targets).sum().item()
                total_num += int(inputs.shape[0])

                if noise_probability_check:
                    noise_stats = self._noise_probability_check(
                        batch_size=int(inputs.shape[0]),
                        batches=1,
                        input_shape=input_shape,
                    )
                    noise_probabilities.append(noise_stats['positive_probability_mean'])
            
            avg_train_loss = mean(train_loss) if train_loss else 0.0
            train_acc = (correct / total_num) * 100 if total_num else 0.0
            print(f'Epoch {epoch + 1}: train_loss={avg_train_loss:.5f}, train_accuracy={train_acc:.2f}')

            if noise_probability_check:
                pos_mean = float(np.mean(noise_probabilities)) if noise_probabilities else 0.0
                noise_stats = {
                    'positive_probability_mean': pos_mean,
                    'negative_probability_mean': 1.0 - pos_mean,
                    'deviation_from_0_5': abs(pos_mean - 0.5),
                }
                print(
                    'noise_probability_check: '
                    f'positive={noise_stats["positive_probability_mean"]:.4f}, '
                    f'negative={noise_stats["negative_probability_mean"]:.4f}, '
                    f'deviation_from_0.5={noise_stats["deviation_from_0_5"]:.4f}'
                )

            if noise_regularization_weight > 0.0:
                noise_reg_mean = float(np.mean(noise_reg_losses)) if noise_reg_losses else 0.0
                print(
                    'noise_regularization: '
                    f'weight={noise_regularization_weight:.6f}, '
                    f'mean_loss={noise_reg_mean:.6f}'
                )


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

    def _noise_probability_check(self, batch_size: int = 32, batches: int = 5, input_shape: Sequence[int] = (3, 256, 608)):
        if batch_size <= 0 or batches <= 0:
            raise ValueError('batch_size and batches must be positive integers.')

        state_before = self.model.training
        self.model.eval()
        probs = []

        with torch.no_grad():
            for _ in range(batches):
                noise = torch.randn(batch_size, *input_shape, device=self.device)
                outputs = self.model(noise)
                prob_pos = torch.sigmoid(outputs).detach().cpu().view(-1)
                probs.append(prob_pos)

        if state_before:
            self.model.train()

        all_probs = torch.cat(probs)
        pos_mean = float(all_probs.mean().item())
        neg_mean = float((1.0 - all_probs).mean().item())
        deviation = float(abs(pos_mean - 0.5))
        return {
            'positive_probability_mean': pos_mean,
            'negative_probability_mean': neg_mean,
            'deviation_from_0_5': deviation,
        }

    def _noise_regularization_loss(self, batch_size: int, input_shape: Sequence[int]):
        if batch_size <= 0:
            raise ValueError('batch_size must be a positive integer.')

        noise = torch.randn(batch_size, *input_shape, device=self.device)
        noise_logits = self.model(noise)
        noise_probabilities = torch.sigmoid(noise_logits)
        target = torch.full_like(noise_probabilities, 0.5)
        return torch.nn.functional.mse_loss(noise_probabilities, target)

    def evaluate_model(self, test_loader):
        self.model.eval()
        losses = []
        total_num = 0
        correct = 0
        g_total = 0
        g_correct = 0
        bad_total = 0
        bad_correct = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.float().unsqueeze(-1).to(self.device)

            outputs = self.model(inputs)
            loss = self.cost_function(outputs, targets)
            losses.append(loss.item())

            preds = (outputs > 0).float()

            correct += (preds == targets).sum().item()
            total_num += int(inputs.shape[0])

            per_sample_correct = (preds == targets)
            good_mask = (targets == 1)
            bad_mask = ~good_mask

            g_total += int(good_mask.sum().item())
            g_correct += int(per_sample_correct[good_mask].sum().item())
            bad_total += int(bad_mask.sum().item())
            bad_correct += int(per_sample_correct[bad_mask].sum().item())

        accuracy = (correct / total_num) * 100 if total_num else 0.0
        return {
            'loss': mean(losses) if losses else 0.0,
            'accuracy': accuracy,
            'good_accuracy': (g_correct / g_total) * 100 if g_total else 0.0,
            'bad_accuracy': (bad_correct / bad_total) * 100 if bad_total else 0.0,
            'good_count': g_total,
            'bad_count': bad_total,
        }
