import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

class BitDepthReduction(nn.Module):
    """Reduces color depth to `bit_depth` bits (1 to 8 bits)."""
    def __init__(self, bit_depth: int = 1):
        super().__init__()
        assert 1 <= bit_depth <= 8, "Bit depth must be between 1 and 8."
        self.max_val = float((1 << bit_depth) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * self.max_val) / self.max_val


class MedianSmoothing(nn.Module):
    """Local spatial smoothing via sliding window median filter."""
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        # FIXED: Asymmetric padding preserves exact (H, W) dimensions for even kernel sizes like 2x2
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size // 2
        pad_top = (kernel_size - 1) // 2
        pad_bottom = kernel_size // 2
        self.padding = (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_padded = F.pad(x, self.padding, mode='reflect')
        patches = F.unfold(x_padded, kernel_size=self.kernel_size)
        patches = patches.view(b, c, self.kernel_size * self.kernel_size, h, w)
        medians, _ = torch.median(patches, dim=2)
        return medians


class NonLocalMeansSmoothing(nn.Module):
    """Non-local spatial smoothing over similar patches using OpenCV."""
    def __init__(self, search_window: int = 11, patch_size: int = 3, filter_strength: float = 4.0):
        super().__init__()
        self.search_window = search_window
        self.patch_size = patch_size
        self.filter_strength = filter_strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        b, c, h, w = x_np.shape
        out_np = np.empty_like(x_np)

        for i in range(b):
            img = x_np[i]
            if c == 1:  # Grayscale
                img_uint8 = (img[0] * 255.0).astype(np.uint8)
                denoised = cv2.fastNlMeansDenoising(
                    img_uint8, None, self.filter_strength, self.patch_size, self.search_window
                )
                out_np[i, 0] = denoised.astype(np.float32) / 255.0
            else:  # RGB
                img_uint8 = (np.transpose(img, (1, 2, 0)) * 255.0).astype(np.uint8)
                denoised = cv2.fastNlMeansDenoisingColored(
                    img_uint8, None, self.filter_strength, self.filter_strength,
                    self.patch_size, self.search_window
                )
                out_np[i] = np.transpose(denoised.astype(np.float32) / 255.0, (2, 0, 1))

        return torch.from_numpy(out_np).to(x.device)


class JointFeatureSqueezingDetector(nn.Module):
    """Calculates L1 softmax distance across multiple squeezers."""
    def __init__(self, model: nn.Module, squeezers: list, threshold: float = 0.10):
        super().__init__()
        self.model = model
        self.squeezers = nn.ModuleList(squeezers)
        self.threshold = threshold

    def forward(self, x: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            logits_orig = self.model(x)
            prob_orig = F.sigmoid(logits_orig)

            max_l1_dist = torch.zeros(x.size(0), device=x.device)

            for squeezer in self.squeezers:
                x_sq = squeezer(x)
                prob_sq = F.sigmoid(self.model(x_sq))
                l1_dist = torch.norm(prob_orig - prob_sq, p=1, dim=1)
                max_l1_dist = torch.max(max_l1_dist, l1_dist)

            predictions = (prob_orig > 0).float().view(-1)
            return {
                "predictions": predictions,
                "is_adversarial": max_l1_dist > self.threshold,
                "max_distance": max_l1_dist
            }

