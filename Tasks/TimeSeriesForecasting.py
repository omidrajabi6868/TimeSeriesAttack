from pathlib import Path
import torch.nn as nn
import torch

from typing import Callable, Optional, Sequence
from .TimeSeriesModels.PatchTSTModel import PatchTST

class ForecastBase:
    def __init__(
        self,
        model_name: str,
        optimizer_name: str = 'Adam',
        checkpoint_dir: str = 'checkpoints',
        input_len: int = 96,
        output_len: int = 96,
        num_vars: int = 6,
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
        self.input_len = input_len
        self.pred_len = output_len,
        self.num_vars = num_vars,

    def _build_model(self):
        if self.model_name == "PatchTST":
            self.model = PatchTST(input_len=self.input_len, pred_len=self.pred_len, num_vars=self.num_vars)
        return self.model

    def _build_cost_function(self):
        self.cost_function = nn.MSELoss()
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
            'model_state_dict': self._stateful_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'model_name': self.model_name,
            'optimizer_name': self.optimizer_name,
            'history': history,
        },
        checkpoint_path)

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
        resume_from: Optional[str] = None):
            
        self._build_model()
        self._build_cost_function()
        self._build_optimization_algorithm(self.model.parameters(), learning_rate)

        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
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
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.float().unsqueeze(-1).to(self.device)
                

                outputs = self.model(inputs)
                loss = self.cost_function(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
            
            avg_train_loss = mean(train_loss) if train_loss else 0.0
            print(f'Epoch {epoch + 1}: train_loss={avg_train_loss:.5f}')


            with torch.no_grad():
                metrics = self.evaluate_model(val_loader)
                val_loss = metrics['loss']
                print(f'val_loss={val_loss:.5f}')

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)

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
        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.float().unsqueeze(-1).to(self.device)

            outputs = self.model(inputs)
            loss = self.cost_function(outputs, targets)
            losses.append(loss.item())

        return {
            'loss': mean(losses) if losses else 0.0,
            'accuracy': accuracy,
        }












