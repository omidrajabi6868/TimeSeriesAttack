from Network import ClassificationModels
from typing import Callable

import torch


class ClassificationBase:
    def __init__(self, model_name: str, optimizer_name: str='Adam'):
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        return

    def build_model():
        if self.model_name == 'ResNet50':
            self.model = ClassificationModels.ResNet('50', 2).model
        return self.model

    def build_cost_function():
        self.cost_function = torch.nn.BCEWithLogitsLoss()
        return self.cost_function

    def build_optimization_algorithm(params):
        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(params=params)

        return self.optimizer


    def train_model(self, train_loader: Callable, val_loader: Callable,  learning_rate: float=1e-3, epoch_num=10):

        build_cost_function()
        build_optimization_algorithm(self.model.parameters())

        for epoch in range(epoch_num):
            train_loss = []
            self.model.train()
            total_num = 0
            correct_pairs = 0
            for inputs, targets in train_loader:
                outputs = self.model(inputs)
                loss = self.cost_function(outputs, targets)

                self.optimizer.zeros()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                preds = (outputs > 0).float()

                # Strict Accuracy (Both must match)
                correct_pairs += (preds == targets).all(dim=1).sum()
                total_num += int(inputs.shape[0])
            
            train_acc = (correct_pairs/total_num)*100

            print(f'Epoch {epoch+1}: {mean(train_loss)}, train_accuracy: {train_acc}')


            with torch.no_grad():
                val_loss, val_acc = self.evaluate_model(self.model, val_loader).values()
                print(f'val_loss: {val_loss}, val_accuracy: {val_acc}')
            
        return self.model

    @staticmethod
    def evaluate_model(model, test_loader):
        model.eval()
        losses = []
        total_num = 0
        correct_pairs = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            targets = targets.float()
            loss = self.cost_function(outputs, targets)
            losses.append(loss.item())

            preds = (outputs > 0).float()

            # Strict Accuracy (Both must match)
            correct_pairs += (preds == targets).all(dim=1).sum()
            total_num += int(inputs.shape[0])

        accuracy = (correct_pairs/total_num)*100

        return {'loss': mean(losses), 'accuracy': accuracy}