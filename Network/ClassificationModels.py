import torch
from torchvision import models
from typing import List

class ResNet:
    def __init__(self, name: str ='50', num_classes: int=2):
        
        if name == '18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif name == '34':
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif name == '50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif name == '101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError('This model has not been implemented yet.')

        self.num_classes = num_classes
        self.name = name
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

class AlexNet:
    def __init__(self, name: str ='', num_classes: int=2):
        
        self.model = models.alexnet.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

        self.num_classes = num_classes
        self.name = name
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)