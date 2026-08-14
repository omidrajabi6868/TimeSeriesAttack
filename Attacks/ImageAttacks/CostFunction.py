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

    @staticmethod
    def _align_feature_pair(clean_feat, adv_feat):
        """Flatten feature tensors and align only an uneven batch dimension."""
        if clean_feat.ndim > 2:
            clean_feat = clean_feat.flatten(start_dim=1)
        if adv_feat.ndim > 2:
            adv_feat = adv_feat.flatten(start_dim=1)
        if clean_feat.ndim != 2 or adv_feat.ndim != 2:
            raise RuntimeError('Feature-based objective expects 2D feature activations after flattening.')
        if clean_feat.shape[1] != adv_feat.shape[1]:
            raise RuntimeError(
                'Feature-based objective received incompatible feature dimensions: '
                f'{clean_feat.shape[1]} and {adv_feat.shape[1]}.'
            )
        batch_size = min(clean_feat.shape[0], adv_feat.shape[0])
        if batch_size <= 0:
            raise RuntimeError('Feature-based objective received empty activations.')
        return clean_feat[:batch_size], adv_feat[:batch_size]

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
                clean_feat, adv_feat = self._align_feature_pair(clean_feat, adv_feat)
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
        self._activation_records = []
        self.hooks = []
        self.output_device = self._infer_output_device(model)

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

        for layer_idx, layer in enumerate(selected_layers):
            self.hooks.append(
                layer.register_forward_hook(self._make_hook(layer_idx))
            )

    @staticmethod
    def _infer_output_device(model):
        if isinstance(model, nn.DataParallel) and model.device_ids:
            return torch.device('cuda', model.device_ids[0])
        try:
            return next(model.parameters()).device
        except StopIteration:
            return None

    @property
    def activations(self):
        grouped = []
        for layer_idx in sorted({idx for idx, _ in self._activation_records}):
            layer_outputs = [
                self._activation_to_output_device(out)
                for idx, out in self._activation_records
                if idx == layer_idx and torch.is_tensor(out)
            ]
            if not layer_outputs:
                continue
            grouped.append(torch.cat(layer_outputs, dim=0) if len(layer_outputs) > 1 else layer_outputs[0])
        return grouped

    def _activation_to_output_device(self, activation):
        if self.output_device is None or activation.device == self.output_device:
            return activation
        return activation.to(self.output_device)

    def _make_hook(self, layer_idx):
        def hook(module, inp, out):
            self._activation_records.append((layer_idx, out))
        return hook

    def clear(self):
        self._activation_records.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()