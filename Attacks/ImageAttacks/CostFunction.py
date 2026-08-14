import torch
import torch.nn as nn
import torch.nn.functional as F

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
    @staticmethod
    def detach_targets(features):
        """Return a stable, non-differentiable snapshot of hooked features."""
        if features is None:
            return None
        if torch.is_tensor(features):
            return features.detach().clone()
        return [
            feature.detach().clone() if torch.is_tensor(feature) else feature
            for feature in features
        ]

    def __init__(self, feature_extractor=None, eps=1e-8):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.eps = eps

    def forward(self, outputs=None, targets=None):
        adv_features = outputs
        if adv_features is None and self.feature_extractor is not None:
            adv_features = self.feature_extractor.activations
        if torch.is_tensor(adv_features):
            adv_features = [adv_features]
        if not adv_features:
            raise RuntimeError(
                'Feature-based objective did not receive any model activations. '
                'Pass hooked features as outputs or run the hooked model before computing this loss.'
            )

        loss_device = None
        if torch.is_tensor(targets):
            loss_device = targets.device
        else:
            for feat in adv_features:
                if torch.is_tensor(feat):
                    loss_device = feat.device
                    break
        if loss_device is None:
            raise RuntimeError('Feature-based objective received no tensor activations.')

        if targets is not None:
            loss = torch.zeros((), device=loss_device)
            for adv_feat, clean_feat in zip(adv_features, targets):
                if not torch.is_tensor(adv_feat) or not torch.is_tensor(clean_feat):
                    continue
                if clean_feat.device != adv_feat.device:
                    clean_feat = clean_feat.to(adv_feat.device)
                layer_loss = F.cosine_similarity(clean_feat, adv_feat, dim=1).mean()
                loss = loss + layer_loss.to(loss_device)
        else:
            loss = torch.zeros((), device=loss_device)
            for feat in adv_features:
                if not torch.is_tensor(feat):
                    continue
                norm = torch.norm(feat, p=2)
                layer_loss = -torch.log(norm + self.eps)
                loss = loss + layer_loss.to(loss_device)
        return loss


class FeatureExtractor:
    def __init__(self, model, n_last_layers=10, layer_types=(nn.Conv2d,), exclude_last_layers=0):
        self.activations = []
        self.hooks = []

        layers = [
            m for m in model.modules()
            if isinstance(m, layer_types)
        ]
        if not layers:
            layers = [
                m for m in model.modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))
            ]

        if exclude_last_layers < 0:
            raise ValueError('exclude_last_layers must be non-negative.')
        selectable_layers = layers[:-exclude_last_layers] if exclude_last_layers else layers
        selected_layers = selectable_layers[-n_last_layers:] if n_last_layers else selectable_layers
        if not selected_layers:
            raise RuntimeError(
                f'FeatureExtractor found {len(layers)} candidate layers, but selected none after '
                f'excluding the last {exclude_last_layers}. Use a smaller exclude_last_layers, '
                'a positive n_last_layers value, or a model with supported layers.'
            )

        for layer in selected_layers:
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