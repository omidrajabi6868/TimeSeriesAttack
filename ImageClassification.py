import json
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

import torch

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
        if self.model_name == 'ResNet50':
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
        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.float().to(self.device)

            outputs = self.model(inputs)
            loss = self.cost_function(outputs, targets)
            losses.append(loss.item())

            preds = (outputs > 0).float()

            correct_pairs += (preds == targets).all(dim=1).sum().item()
            total_num += int(inputs.shape[0])

        accuracy = (correct_pairs / total_num) * 100 if total_num else 0.0
        return {'loss': mean(losses) if losses else 0.0, 'accuracy': accuracy}