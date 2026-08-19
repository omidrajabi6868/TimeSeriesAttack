import random
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


import math
import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class PSPTransformSampler:
    """
    PSP-UAP pseudo-semantic prior sampler + input transformation.

    Faithful to the released PSP-UAP implementation:

        Chanhui Lee et al.
        "Data-free Universal Adversarial Perturbation
         with Pseudo-semantic Prior"
        CVPR 2025.

    Pipeline:

        random prior z
              |
              v
        example_prior = z + delta
              |
              v
        repeat num_copies
              |
              v
        random crop + resize
              |
              v
        semantic_prior
              |
              +---------> + delta
              |                 |
              |                 v
              |          semantic_delta
              |
              +-----------------------+
                                      |
                            input transformation
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                  semantic_prior            semantic_delta

    IMPORTANT:
        The same transformation parameters are applied
        to the clean and adversarial versions.

    This class intentionally combines the prior generation
    and transformation functionality. A separate PSPSampler
    is NOT required.
    """

    def __init__(
        self,
        image_size,
        prior="gauss",
        prior_std=127.0,
        prior_batch=1,
        num_copies=10,

        # Official random crop used by PSP-UAP.
        crop_scale=(0.08, 1.0),
        crop_ratio=(3.0 / 4.0, 4.0 / 3.0),

        # Input transformation.
        input_transform=True,

        # Official transformation parameters.
        angle=6.0,
        scale_t_low=0.8,
        scale_t_high=4.0,

        # Number of blocks for shuffle.
        shuffle_block=2,

        device=None,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.height, self.width = image_size

        self.prior = prior
        self.prior_std = prior_std
        self.prior_batch = prior_batch
        self.num_copies = num_copies

        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

        self.input_transform = input_transform

        self.angle = angle
        self.scale_t_low = scale_t_low
        self.scale_t_high = scale_t_high

        self.shuffle_block = shuffle_block

        self.device = device
        self.interpolation = interpolation

    # =========================================================
    # Gaussian prior
    # =========================================================

    def sample_gaussian_prior(
        self,
        batch_size,
        device,
        dtype,
    ):
        """
        Reproduce get_gauss_prior() / make_some_noise_gauss():

            mean = [127.5, 127.5, 127.5]
            std  = [s, s+10, s+20]

        clipped to [0, 255], then converted to [0, 1].
        """

        std = float(self.prior_std)

        mean = torch.tensor(
            [127.5, 127.5, 127.5],
            device=device,
            dtype=dtype,
        ).view(1, 3, 1, 1)

        channel_std = torch.tensor(
            [std, std + 10.0, std + 20.0],
            device=device,
            dtype=dtype,
        ).view(1, 3, 1, 1)

        noise = torch.randn(
            batch_size,
            3,
            self.height,
            self.width,
            device=device,
            dtype=dtype,
        )

        noise = mean + noise * channel_std

        noise = noise.clamp(
            0.0,
            255.0,
        )

        return noise / 255.0

    # =========================================================
    # Jigsaw prior
    # =========================================================

    def sample_jigsaw_prior(
        self,
        delta,
        batch_size,
        num_blocks=5,
    ):
        """
        Generate the jigsaw-style prior used by the PSP-UAP
        prior generation.

        The official implementation constructs the jigsaw
        pattern from the current delta.
        """

        if delta.ndim == 4:
            base = delta[0]
        else:
            base = delta

        C, H, W = base.shape

        width_lengths = self._get_partition_lengths(
            W,
            num_blocks,
            device=base.device,
        )

        height_lengths = self._get_partition_lengths(
            H,
            num_blocks,
            device=base.device,
        )

        width_perm = torch.randperm(
            num_blocks,
            device=base.device,
        )

        height_perm = torch.randperm(
            num_blocks,
            device=base.device,
        )

        width_splits = torch.split(
            base,
            width_lengths,
            dim=2,
        )

        shuffled_columns = []

        for wi in width_perm:

            column = width_splits[wi]

            height_splits = torch.split(
                column,
                height_lengths,
                dim=1,
            )

            shuffled_rows = [
                height_splits[hi]
                for hi in height_perm
            ]

            shuffled_column = torch.cat(
                shuffled_rows,
                dim=1,
            )

            shuffled_columns.append(
                shuffled_column
            )

        shuffled = torch.cat(
            shuffled_columns,
            dim=2,
        )

        return (
            shuffled
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
            .clone()
        )

    @staticmethod
    def _get_partition_lengths(
        length,
        num_blocks,
        device,
    ):
        """
        Positive random partition lengths whose sum is
        exactly `length`.
        """

        if num_blocks <= 1:
            return (int(length),)

        random_values = torch.rand(
            num_blocks,
            device=device,
        )

        lengths = torch.round(
            random_values
            / random_values.sum()
            * length
        ).long()

        lengths = torch.clamp(
            lengths,
            min=1,
        )

        difference = (
            int(length)
            - int(lengths.sum())
        )

        max_idx = torch.argmax(
            lengths
        )

        lengths[max_idx] += difference

        return tuple(
            lengths.tolist()
        )

    # =========================================================
    # Random crop
    # =========================================================

    def random_crop(
        self,
        image,
    ):
        """
        Same random crop strategy as semantic_aug.py.
        """

        C, H, W = image.shape

        image_area = H * W

        for _ in range(10):

            target_scale = random.uniform(
                self.crop_scale[0],
                self.crop_scale[1],
            )

            target_ratio = random.uniform(
                self.crop_ratio[0],
                self.crop_ratio[1],
            )

            target_area = (
                image_area * target_scale
            )

            target_width = int(
                round(
                    math.sqrt(
                        target_area
                        * target_ratio
                    )
                )
            )

            target_height = int(
                round(
                    math.sqrt(
                        target_area
                        / target_ratio
                    )
                )
            )

            if (
                target_height <= H
                and target_width <= W
                and target_height > 0
                and target_width > 0
            ):

                top = random.randint(
                    0,
                    H - target_height,
                )

                left = random.randint(
                    0,
                    W - target_width,
                )

                crop = image[
                    :,
                    top:top + target_height,
                    left:left + target_width,
                ]

                return crop

        # -----------------------------------------------------
        # Official fallback
        # -----------------------------------------------------

        in_ratio = float(W) / float(H)

        if in_ratio < min(self.crop_ratio):

            target_width = W

            target_height = int(
                round(
                    W / min(self.crop_ratio)
                )
            )

        elif in_ratio > max(self.crop_ratio):

            target_height = H

            target_width = int(
                round(
                    H * max(self.crop_ratio)
                )
            )

        else:

            target_height = H
            target_width = W

        target_height = min(
            target_height,
            H,
        )

        target_width = min(
            target_width,
            W,
        )

        top = (
            H - target_height
        ) // 2

        left = (
            W - target_width
        ) // 2

        return image[
            :,
            top:top + target_height,
            left:left + target_width,
        ]

    # =========================================================
    # Random crop + resize
    # =========================================================

    def random_crop_and_resize(
        self,
        X,
    ):
        """
        Generate one independently cropped/resized semantic
        sample for every input sample.

        This corresponds to:

            random_crop_and_resize(
                semantic_priors[j].unsqueeze(0),
                scale_crop=(0.08, 1),
                ratio_crop=(3./4., 4./3.)
            )

        in the official training loop.
        """

        B, C, H, W = X.shape

        output = []

        for i in range(B):

            crop = self.random_crop(
                X[i]
            )

            resized = transforms.Resize(
                (H, W)
            )(
                crop.unsqueeze(0)
            ).squeeze(0)

            output.append(
                resized
            )

        return torch.stack(
            output,
            dim=0,
        )

    # =========================================================
    # Truncated normal
    # =========================================================

    @staticmethod
    def truncated_normal(
        mean=0.0,
        std=1.0,
        low=-2.0,
        high=2.0,
    ):
        """
        Same rejection-sampling implementation as the
        official semantic_aug.py.
        """

        while True:

            value = torch.normal(
                mean=float(mean),
                std=float(std),
                size=(1,),
            )

            value = float(
                value.item()
            )

            if low <= value <= high:
                return value

    # =========================================================
    # Rotation
    # =========================================================

    def rotate_pair(
        self,
        delta,
        semantic_kd,
    ):
        """
        Faithful to rotate_fill_prior().

        A single sampled angle is applied to both images.

        Empty pixels created by rotation are replaced by
        the corresponding original pixels.
        """

        angle = self.truncated_normal(
            mean=0.0,
            std=self.angle,
            low=-self.angle,
            high=self.angle,
        )

        original_delta = delta.clone()
        original_kd = semantic_kd.clone()

        rotated_delta = TF.rotate(
            delta,
            angle=angle,
            fill=0,
            interpolation=self.interpolation,
        )

        mask_delta = (
            rotated_delta == 0
        ).float()

        final_delta = (
            rotated_delta
            * (1.0 - mask_delta)
            + original_delta
            * mask_delta
        )

        rotated_kd = TF.rotate(
            semantic_kd,
            angle=angle,
            fill=0,
            interpolation=self.interpolation,
        )

        mask_kd = (
            rotated_kd == 0
        ).float()

        final_kd = (
            rotated_kd
            * (1.0 - mask_kd)
            + original_kd
            * mask_kd
        )

        return (
            final_delta,
            final_kd,
        )

    # =========================================================
    # Scaling
    # =========================================================

    def scaling_pair(
        self,
        delta,
        semantic_kd,
    ):
        """
        Faithful to scaling_transform().

            ratio ~ Uniform(0.8, 4.0)

        Both branches receive exactly the same ratio.
        """

        ratio = torch.empty(
            1,
            device=delta.device,
        ).uniform_(
            self.scale_t_low,
            self.scale_t_high,
        ).item()

        final_delta = (
            delta * ratio
        )

        final_kd = (
            semantic_kd * ratio
        )

        return (
            final_delta,
            final_kd,
        )

    # =========================================================
    # Shuffle
    # =========================================================

    def shuffle_pair(
        self,
        delta,
        semantic_kd,
    ):
        """
        Faithful to shuffle_only().

        A single spatial permutation is generated and applied
        identically to delta and semantic_kd.
        """

        C, H, W = delta.shape

        num_blocks = self.shuffle_block

        width_lengths = (
            self._get_partition_lengths(
                W,
                num_blocks,
                device=delta.device,
            )
        )

        height_lengths = (
            self._get_partition_lengths(
                H,
                num_blocks,
                device=delta.device,
            )
        )

        width_perm = torch.randperm(
            num_blocks,
            device=delta.device,
        )

        height_perm = torch.randperm(
            num_blocks,
            device=delta.device,
        )

        final_delta = self._shuffle_single(
            delta,
            width_lengths,
            height_lengths,
            width_perm,
            height_perm,
        )

        final_kd = self._shuffle_single(
            semantic_kd,
            width_lengths,
            height_lengths,
            width_perm,
            height_perm,
        )

        return (
            final_delta,
            final_kd,
        )

    @staticmethod
    def _shuffle_single(
        X,
        width_lengths,
        height_lengths,
        width_perm,
        height_perm,
    ):
        """
        Apply an already sampled block permutation.
        """

        width_splits = torch.split(
            X,
            width_lengths,
            dim=2,
        )

        shuffled_strips = []

        for wi in width_perm:

            strip = width_splits[wi]

            height_splits = torch.split(
                strip,
                height_lengths,
                dim=1,
            )

            shuffled_height_sections = [
                height_splits[hi]
                for hi in height_perm
            ]

            shuffled_strip = torch.cat(
                shuffled_height_sections,
                dim=1,
            )

            shuffled_strips.append(
                shuffled_strip
            )

        return torch.cat(
            shuffled_strips,
            dim=2,
        )

    # =========================================================
    # Input transformation
    # =========================================================

    def apply_random_transform(
        self,
        semantic_delta,
        semantic_prior,
    ):
        """
        Select exactly one transformation with probability 1/3
        each, as in the official implementation.

        Returns:
            transformed_delta,
            transformed_prior,
            transform_name
        """

        probability = torch.rand(
            1,
            device=semantic_delta.device,
        ).item()

        if probability < 1.0 / 3.0:

            transformed_delta, transformed_prior = (
                self.rotate_pair(
                    semantic_delta,
                    semantic_prior,
                )
            )

            name = "rotation"

        elif probability < 2.0 / 3.0:

            transformed_delta, transformed_prior = (
                self.scaling_pair(
                    semantic_delta,
                    semantic_prior,
                )
            )

            name = "scaling"

        else:

            transformed_delta, transformed_prior = (
                self.shuffle_pair(
                    semantic_delta,
                    semantic_prior,
                )
            )

            name = "shuffle"

        return (
            transformed_delta,
            transformed_prior,
            name,
        )

    # =========================================================
    # Main sampling function
    # =========================================================

    def sample(
        self,
        delta,
        batch_size=None,
        num_copies=None,
        jigsaw_blocks=None,
        input_transform=None,
        return_transform=False,
    ):
        """
        Generate the complete PSP-UAP training samples.

        Parameters
        ----------
        delta:
            Current universal perturbation.

            Shape:
                [3, H, W]
            or
                [1, 3, H, W]

        batch_size:
            Kept for compatibility.

            In PSP-UAP, the number of generated semantic
            samples is controlled by num_copies * prior_batch.

        num_copies:
            Number of semantic copies.

        jigsaw_blocks:
            Optional override for the jigsaw prior.

        input_transform:
            Override self.input_transform.

        return_transform:
            Return the transform names.

        Returns
        -------
        semantic_prior:
            Clean semantic samples.

        semantic_delta:
            Adversarial semantic samples.

        optionally:
            transform_names
        """

        if delta.ndim == 3:
            delta = delta.unsqueeze(0)

        if delta.ndim != 4:
            raise ValueError(
                "delta must have shape "
                "[3,H,W] or [1,3,H,W]."
            )

        device = delta.device
        dtype = delta.dtype

        if num_copies is None:
            num_copies = self.num_copies

        if input_transform is None:
            input_transform = self.input_transform

        # =====================================================
        # 1. Generate random prior
        # =====================================================

        if self.prior == "gauss":

            # The official implementation's random batch
            # represents one prior batch.
            prior_batch = self.prior_batch

            random_batch = (
                self.sample_gaussian_prior(
                    batch_size=prior_batch,
                    device=device,
                    dtype=dtype,
                )
            )

        elif self.prior == "jigsaw":

            blocks = (
                jigsaw_blocks
                if jigsaw_blocks is not None
                else 5
            )

            random_batch = (
                self.sample_jigsaw_prior(
                    delta=delta,
                    batch_size=self.prior_batch,
                    num_blocks=blocks,
                )
            )

        else:

            raise ValueError(
                f"Unsupported prior '{self.prior}'. "
                "Use 'gauss' or 'jigsaw'."
            )

        # =====================================================
        # 2. Construct pseudo-semantic prior
        #
        # Official:
        #
        # example_prior = delta + random_batch
        #
        # =====================================================

        if random_batch.shape[0] == 1:

            example_prior = (
                delta
                + random_batch
            )

        else:

            example_prior = (
                delta.expand(
                    random_batch.shape[0],
                    -1,
                    -1,
                    -1,
                )
                + random_batch
            )

        # =====================================================
        # 3. Repeat for semantic copies
        #
        # Official:
        #
        # semantic_priors =
        #     example_prior.repeat(num_copise, ...)
        #
        # =====================================================

        semantic_priors = (
            example_prior.repeat(
                num_copies,
                1,
                1,
                1,
            )
        )

        # =====================================================
        # 4. Random crop + resize EACH semantic sample
        #
        # IMPORTANT:
        #
        # The crop is independently sampled for every copy.
        #
        # =====================================================

        semantic_priors = (
            self.random_crop_and_resize(
                semantic_priors
            )
        )

        # =====================================================
        # 5. Construct adversarial semantic samples
        #
        # OFFICIAL:
        #
        # semantic_delta =
        #     semantic_priors.detach()
        #     + delta.repeat(...)
        #
        # =====================================================

        delta_repeated = (
            delta.repeat(
                semantic_priors.shape[0],
                1,
                1,
                1,
            )
        )

        semantic_delta = (
            semantic_priors.detach()
            + delta_repeated
        )

        # =====================================================
        # 6. Clean branch used for KL
        #
        # OFFICIAL:
        #
        # semantic_priors_kd =
        #     semantic_priors.clone().detach()
        #
        # =====================================================

        semantic_priors_kd = (
            semantic_priors
            .clone()
            .detach()
        )

        transform_names = []

        # =====================================================
        # 7. Input transformation
        # =====================================================

        if input_transform:

            transformed_delta = []
            transformed_prior = []

            for j in range(
                semantic_delta.shape[0]
            ):

                (
                    delta_j,
                    prior_j,
                    name,
                ) = self.apply_random_transform(
                    semantic_delta[j],
                    semantic_priors_kd[j],
                )

                transformed_delta.append(
                    delta_j
                )

                transformed_prior.append(
                    prior_j
                )

                transform_names.append(
                    name
                )

            semantic_delta = torch.stack(
                transformed_delta,
                dim=0,
            )

            semantic_priors_kd = torch.stack(
                transformed_prior,
                dim=0,
            )

        # =====================================================
        # 8. Return
        # =====================================================

        if return_transform:

            return (
                semantic_priors_kd,
                semantic_delta,
                transform_names,
            )

        return (
            semantic_priors_kd,
            semantic_delta,
        )
