
import random
import torchvision.transforms as transforms

class TransformSampler:
    def __init__(
        self,
        height,
        width,
        p_brightness=0.5,
        p_contrast=0.5,
        p_affine=0.8,
    ):
        self.height = height
        self.width = width

        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_affine = p_affine

    def sample(self, n):

        sampled = []

        for _ in range(n):

            ops = []

            # -------------------------------
            # Brightness
            # -------------------------------
            if random.random() < self.p_brightness:
                ops.append(
                    transforms.ColorJitter(
                        brightness=0.2
                    )
                )

            # -------------------------------
            # Contrast
            # -------------------------------
            if random.random() < self.p_contrast:
                ops.append(
                    transforms.ColorJitter(
                        contrast=0.2
                    )
                )

            # -------------------------------
            # Rotation
            # Scaling
            # Translation
            # Shearing
            # -------------------------------
            if random.random() < self.p_affine:
                ops.append(
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                        shear=2,
                    )
                )

            # -------------------------------
            # Identity
            # -------------------------------
            if len(ops) == 0:
                ops.append(transforms.Lambda(lambda x: x))

            # -------------------------------
            # Keep input size fixed
            # -------------------------------
            ops.append(
                transforms.Resize(
                    (self.height, self.width)
                )
            )

            sampled.append(
                transforms.Compose(ops)
            )

        return sampled