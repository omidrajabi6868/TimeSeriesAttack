import torch
from torchvision import models


class ResNet:
    def __init__(self, name: str ='18', num_classes: int=2):
        
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
        
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

        self.num_classes = num_classes
        self.name = name
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(in_features, num_classes)


class MobileNetV3Small:
    """A compact MobileNetV3 initialized with ImageNet weights."""

    def __init__(self, num_classes: int = 2):
        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.num_classes = num_classes
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = torch.nn.Linear(in_features, num_classes)


class EfficientNetB0:
    """The smallest EfficientNet variant initialized with ImageNet weights."""

    def __init__(self, num_classes: int = 2):
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.num_classes = num_classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, num_classes)


class SwinT:
    """A compact hierarchical vision transformer with ImageNet weights."""

    def __init__(self, num_classes: int = 2):
        self.model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.num_classes = num_classes
        in_features = self.model.head.in_features
        self.model.head = torch.nn.Linear(in_features, num_classes)
