import torch

class FourierFilter:
    def __init__(self, mode="high_pass", bandwidth=60):
        self.mode = mode
        self.bandwidth = bandwidth

    def __call__(self, perturbation):
        """
        perturbation: [C, H, W] or [B, C, H, W]
        """

        if perturbation.dim() == 3:
            perturbation = perturbation.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, C, H, W = perturbation.shape

        spectrum = torch.fft.fft2(
            perturbation,
            dim=(-2, -1)
        )

        spectrum = torch.fft.fftshift(
            spectrum,
            dim=(-2, -1)
        )

        yy, xx = torch.meshgrid(
            torch.arange(H, device=perturbation.device),
            torch.arange(W, device=perturbation.device),
            indexing="ij"
        )

        cy = H // 2
        cx = W // 2

        distance = torch.sqrt(
            (yy - cy) ** 2 +
            (xx - cx) ** 2
        )

        if self.mode == "high_pass":
            mask = distance >= self.bandwidth

        elif self.mode == "low_pass":
            mask = distance <= self.bandwidth

        else:
            raise ValueError(
                f"Unknown Fourier filter: {self.mode}"
            )

        mask = mask.to(spectrum.dtype)

        spectrum = spectrum * mask

        spectrum = torch.fft.ifftshift(
            spectrum,
            dim=(-2, -1)
        )

        filtered = torch.fft.ifft2(
            spectrum,
            dim=(-2, -1)
        ).real

        if squeeze:
            filtered = filtered.squeeze(0)

        return filtered