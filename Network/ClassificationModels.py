import torch
from torchvision import models
from typing import List

class ResNet:
    def __init__(self, name: str ='50', num_classes: int=2):
        if name =='50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
        elif name == '101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            return print('This model has not been implemented yet.')
        self.num_classes = num_classes
        self.name = name
        self.model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        pass
        