from contextlib import contextmanager

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
                layer_loss = torch.log(norm + self.eps)
                loss = loss + layer_loss.to(loss_device)
        return loss

class PSPUAPObjective(AdversarialObjective):
    """
    PSP-UAP feature-activation objective.

    Faithful to the official PSP-UAP implementation.

    Without reweighting:

        L = - sum_l log(
                ||phi_l(x)||_2^2 / 2 + eps
            )

    With PSP reweighting:

        KL_i =
            KL(
                p_semantic_i || p_delta_i
            )

        w_i = 1 / sqrt(KL_i)

        phi'_l,i = w_i * phi_l,i

        L = - sum_l log(
                ||phi'_l||_2^2 / 2 + eps
            )

    Important:
        The official implementation applies the PSP weights
        to the convolutional activations when p_active=True.

    Parameters
    ----------
    feature_extractor:
        FeatureExtractor instance used to obtain hooked
        convolutional activations.

    eps:
        Numerical stability constant.

    p_active:
        Corresponds to the official --p_active option.

        If True:
            only the first p_rate fraction of convolutional
            layers are used and activations are ReLU-truncated.

        If False:
            all selected convolutional layers are used without
            ReLU truncation.

    p_rate:
        Fraction of convolutional layers used when p_active=True.

        IMPORTANT:
            The official code takes the FIRST p_rate fraction
            of the convolutional layers, not the last fraction.

    re_weight:
        Enable PSP KL-based reweighting.

    temperature:
        KL temperature, corresponding to args.temper.

    detach_weights:
        Detach PSP weights before multiplying activations.
        This matches the official implementation.

    maximize_activations:
        Select the sign of the feature objective. False returns the positive
        log-activation score so gradient descent minimizes activations; True
        returns its negative so gradient descent maximizes activations.
    """

    def __init__(
        self,
        feature_extractor=None,
        eps=1e-9,
        p_active=False,
        p_rate=1.0,
        re_weight=False,
        temperature=1.0,
        detach_weights=True,
        maximize_activations=False,
    ):
        super().__init__()

        self.feature_extractor = feature_extractor

        self.eps = eps

        self.p_active = p_active
        self.p_rate = p_rate

        self.re_weight = re_weight
        self.temperature = temperature
        self.detach_weights = detach_weights
        # Keep activation minimization as this project's intentional default,
        # while making the experiment direction explicit and reproducible.
        self.maximize_activations = bool(maximize_activations)

    # =========================================================
    # KL divergence used by official PSP-UAP implementation
    # =========================================================

    @staticmethod
    def kd_loss(
        student,
        teacher,
        temperature,
    ):
        """
        Official PSP-UAP KD/KL loss.

        The official implementation is:

            log_s = log_softmax(student / T)
            soft_t = softmax(teacher / T)

            loss = kl_div(
                log_s,
                soft_t,
                reduction='none'
            )

            loss *= T^2

            loss = loss.sum(
                dim=-1,
                keepdim=True
            )

        In PSP-UAP:

            student = output_delta
            teacher = output_semantic

        Therefore:

            KL(
                semantic || delta
            )

        is computed.
        """

        # Handle torchvision InceptionOutputs without
        # introducing a hard dependency here.
        if hasattr(teacher, "logits"):
            teacher = teacher.logits

        if hasattr(student, "logits"):
            student = student.logits

        if student.shape != teacher.shape:
            raise RuntimeError(
                "PSPUAPObjective received incompatible "
                "student and teacher logits: "
                f"{student.shape} vs {teacher.shape}"
            )

        # The reference implementation operates on multi-class logits.  A
        # single sigmoid/BCE logit would otherwise make log_softmax identically
        # zero and, consequently, every KL weight infinite.  Represent the
        # same Bernoulli model as two logits so PSP reweighting also works for
        # binary classifiers used by this project.
        if student.ndim == 1:
            student = student.unsqueeze(-1)
            teacher = teacher.unsqueeze(-1)
        if student.shape[-1] == 1:
            student = torch.cat((torch.zeros_like(student), student), dim=-1)
            teacher = torch.cat((torch.zeros_like(teacher), teacher), dim=-1)

        T = float(temperature)

        if T <= 0:
            raise ValueError(
                "temperature must be > 0."
            )

        log_student = F.log_softmax(
            student / T,
            dim=-1,
        )

        soft_teacher = F.softmax(
            teacher / T,
            dim=-1,
        )

        loss = F.kl_div(
            log_student,
            soft_teacher,
            reduction="none",
        )

        loss = loss * (T ** 2)

        loss = loss.sum(
            dim=-1,
            keepdim=True,
        )

        return loss

    # =========================================================
    # PSP weights
    # =========================================================

    def compute_psp_weights(
        self,
        semantic_logits,
        delta_logits,
    ):
        """
        Compute the exact PSP-UAP reweighting:

            KL = KL(
                semantic || delta
            )

            weight = 1 / KL

            weight = sqrt(weight)

        Therefore:

            weight = 1 / sqrt(KL)

        The official implementation subsequently detaches
        these weights before applying them to activations.
        """

        kl = self.kd_loss(
            student=delta_logits,
            teacher=semantic_logits,
            temperature=self.temperature,
        )

        # Official:
        #
        # weights = 1 / weights
        #
        weights = 1.0 / torch.clamp(
            kl,
            min=self.eps,
        )

        # Official:
        #
        # weights = torch.sqrt(weights)
        #
        weights = torch.sqrt(
            weights
        )

        if self.detach_weights:
            weights = weights.detach()

        return weights

    # =========================================================
    # Determine number of layers
    # =========================================================

    def _select_features(
        self,
        features,
    ):
        """
        Reproduce the official p_active/p_rate behavior.

        Official code:

            truncate = int(
                len(activations) * args.p_rate
            )

            for i in range(truncate):
                ...

        Therefore the FIRST layers are selected.
        """

        if not self.p_active:
            return features

        truncate = int(
            len(features) * self.p_rate
        )

        if (
            truncate <= 0
            and self.p_rate != 0.0
        ):
            truncate += 1

        return features[:truncate]

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        outputs=None,
        targets=None,
        semantic_logits=None,
        delta_logits=None,
        weights=None,
    ):
        """
        Parameters
        ----------
        outputs:
            Adversarial hooked feature activations.

            Expected:

                list[Tensor]

            where each Tensor is typically:

                [B, C, H, W]

        targets:
            Kept for compatibility with the generic objective
            interface. Not used by PSP-UAP.

        semantic_logits:
            Model output for the clean semantic prior.

            Required when:

                re_weight=True
                and weights is not supplied.

        delta_logits:
            Model output for the adversarial semantic sample.

            Required when:

                re_weight=True
                and weights is not supplied.

        weights:
            Optional precomputed PSP weights.

            If supplied, semantic_logits and delta_logits are
            not required.

            Shape should correspond to the activation batch:

                [B]
                [B, 1]

        Returns
        -------
        loss:
            Scalar PSP-UAP activation-maximization loss.
        """

        # -----------------------------------------------------
        # Obtain activations
        # -----------------------------------------------------

        features = outputs

        if (
            features is None
            and self.feature_extractor is not None
        ):
            features = self.feature_extractor.activations

        if features is None:
            raise RuntimeError(
                "PSPUAPObjective did not receive feature "
                "activations."
            )

        if torch.is_tensor(features):
            features = [features]

        if len(features) == 0:
            raise RuntimeError(
                "PSPUAPObjective received no feature "
                "activations."
            )

        # -----------------------------------------------------
        # Select layers
        # -----------------------------------------------------

        features = self._select_features(
            features
        )

        if len(features) == 0:
            raise RuntimeError(
                "PSPUAPObjective selected zero feature "
                "layers. Check p_rate."
            )

        # -----------------------------------------------------
        # Compute PSP weights
        # -----------------------------------------------------

        if self.re_weight:

            if weights is None:

                if (
                    semantic_logits is None
                    or delta_logits is None
                ):
                    raise RuntimeError(
                        "PSPUAPObjective requires either "
                        "`weights` or both `semantic_logits` "
                        "and `delta_logits` when re_weight=True."
                    )

                weights = self.compute_psp_weights(
                    semantic_logits=semantic_logits,
                    delta_logits=delta_logits,
                )

            if not torch.is_tensor(weights):
                raise TypeError(
                    "PSP weights must be a torch.Tensor."
                )

            # Official code:
            #
            # weights = weights.squeeze(-1)
            #
            if (
                weights.ndim > 1
                and weights.shape[-1] == 1
            ):
                weights = weights.squeeze(-1)

            if self.detach_weights:
                weights = weights.detach()

        # -----------------------------------------------------
        # Activation loss
        # -----------------------------------------------------

        loss = None

        for feat in features:

            if not torch.is_tensor(feat):
                continue

            # =================================================
            # p_active=True
            # =================================================

            if self.p_active:

                # Official implementation:
                #
                # activation =
                #     torch.where(
                #         activation > 0,
                #         activation,
                #         0
                #     )
                #
                feat = torch.where(
                    feat > 0,
                    feat,
                    torch.zeros_like(feat),
                )

                # -------------------------------------------------
                # Flatten each sample independently.
                #
                # [B, C, H, W]
                # ->
                # [B, C*H*W]
                # -------------------------------------------------

                batch_size = feat.shape[0]

                activation = feat.reshape(
                    batch_size,
                    -1,
                )

                # =================================================
                # PSP reweighting
                # =================================================

                if self.re_weight:

                    if weights.shape[0] != batch_size:
                        raise RuntimeError(
                            "PSP weight batch dimension does not "
                            "match activation batch dimension: "
                            f"{weights.shape[0]} vs {batch_size}."
                        )

                    # [B] -> [B, 1]
                    sample_weights = (
                        weights
                        .to(
                            device=activation.device,
                            dtype=activation.dtype,
                        )
                        .reshape(
                            batch_size,
                            1,
                        )
                    )

                    # -------------------------------------------------
                    # THIS IS THE IMPORTANT MULTIPLICATION.
                    #
                    # Official:
                    #
                    # weighted_activations =
                    #     ac_tensor * weights.detach().view(B, 1)
                    # -------------------------------------------------

                    activation = (
                        activation *
                        sample_weights
                    )

                # -------------------------------------------------
                # Official activation objective
                # -------------------------------------------------

                activation = activation.reshape(-1)

                layer_loss = torch.log(
                    torch.sum(
                        torch.square(
                            activation
                        )
                    )
                    / 2.0
                    + self.eps
                )

            # =================================================
            # p_active=False
            # =================================================

            else:

                # IMPORTANT:
                #
                # The official implementation does NOT apply
                # ReLU truncation here.
                #
                # It also does not apply PSP sample weights
                # in this branch.
                #
                # This reproduces l2_layer_loss_weight() exactly.
                #

                activation = feat

                layer_loss = torch.log(
                    torch.sum(
                        torch.square(
                            activation
                        )
                    )
                    / 2.0
                    + self.eps
                )

            # -------------------------------------------------
            # Select the activation direction used by gradient descent.
            # -------------------------------------------------

            signed_layer_loss = (
                -layer_loss if self.maximize_activations else layer_loss
            )
            if loss is None:
                loss = signed_layer_loss
            else:
                loss = loss + signed_layer_loss

        if loss is None:
            raise RuntimeError(
                "PSPUAPObjective found no tensor "
                "activations."
            )

        return loss

class FeatureExtractor:
    def __init__(self, model, n_last_layers=10, layer_types=(nn.Conv2d,), exclude_last_layers=0):
        self._activation_records = []
        self.hooks = []
        self.capture_enabled = True
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
            if self.capture_enabled:
                self._activation_records.append((layer_idx, out))
        return hook

    def clear(self):
        self._activation_records.clear()

    @contextmanager
    def suspend_capture(self):
        """Temporarily disable hooks without removing them.

        PSP-UAP hooks every convolutional layer while optimizing the UAP.  The
        hooks are not needed for ASR evaluation and retaining those feature
        maps can use far more memory than the model forward itself.
        """
        was_enabled = self.capture_enabled
        self.clear()
        self.capture_enabled = False
        try:
            yield
        finally:
            self.clear()
            self.capture_enabled = was_enabled

    def remove(self):
        for h in self.hooks:
            h.remove()
