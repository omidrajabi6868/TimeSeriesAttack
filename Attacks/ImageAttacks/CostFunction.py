import torch
import torch.nn as nn

class AdversarialObjective(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ClassificationObjective(AdversarialObjective):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)


class FeaturBaseObjective(AdversarialObjective):
    def __init__(self, feature_extractor, eps=1e-8):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.eps = eps

    def forward(self, outputs=None, targets=None):
        features = self.feature_extractor.activations
        loss = 0.0
        for feat in features:
            norm = torch.norm(feat, p=2)
            loss -= torch.log(norm + self.eps)
        return loss

class FeatureExtractor:
    def __init__(self, model, n_last_layers=4):
        self.activations = []
        self.hooks = []

        layers = [
            m for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ]

        for layer in layers[-n_last_layers:]:
            self.hooks.append(
                layer.register_forward_hook(self._hook)
            )

    def _hook(self, module, inp, out):
        self.activations.append(out)

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()