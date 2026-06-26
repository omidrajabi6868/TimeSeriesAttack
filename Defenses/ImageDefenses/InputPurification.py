import math

import torch
import torch.nn.functional as F


class FeatureDistillation(torch.nn.Module):
    """
    JPEG-style feature-distillation preprocessor from
    "Feature Distillation: DNN-Oriented JPEG Compression Against Adversarial Examples".

    The defense keeps the JPEG pipeline structure (level shift, block DCT,
    quantize/dequantize, IDCT) and replaces the JPEG quantization table with a
    DNN-oriented table: coefficients with high benign-data DCT variation are
    treated as accuracy-sensitive and receive very light quantization, while the
    remaining coefficients use a defensive JPEG table controlled by ``quality``.
    """

    def __init__(self, std_map, block=8, quality=50.0, preserve_ratio=0.5, preserved_quant_step=1.0):
        super().__init__()
        if block != 8:
            raise ValueError("FeatureDistillation currently supports the JPEG 8x8 block size only.")
        if not 0.0 < float(quality) <= 100.0:
            raise ValueError("quality must be in (0, 100].")
        if not 0.0 <= float(preserve_ratio) <= 1.0:
            raise ValueError("preserve_ratio must be in [0, 1].")

        self.block = block
        self.quality = float(quality)
        self.preserve_ratio = float(preserve_ratio)
        self.preserved_quant_step = float(preserved_quant_step)

        std_map = std_map.detach().float()
        if std_map.shape != (block, block):
            raise ValueError(f"std_map must have shape ({block}, {block}); got {tuple(std_map.shape)}.")

        self.register_buffer("std_map", std_map)
        self.register_buffer("jpeg_table", self._build_jpeg_table())
        self.register_buffer("defensive_table", self._quality_scaled_table(self.quality))
        self.register_buffer("accuracy_sensitive_mask", self._build_accuracy_sensitive_mask(std_map))
        self.register_buffer("quantization_table", self._build_fd_table())

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW tensor; got shape {tuple(x.shape)}.")

        orig_h, orig_w = x.shape[-2:]
        n = self.block
        pad_h = (n - orig_h % n) % n
        pad_w = (n - orig_w % n) % n
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        b, c, h, w = x.shape
        dct_mat = self.build_dct_matrix(x.device, n, dtype=x.dtype)
        q = self.quantization_table.to(device=x.device, dtype=x.dtype)

        # Standard JPEG level shift before DCT. Inputs are expected in [0, 1].
        pixels = x.mul(255.0).sub(128.0)
        blocks = pixels.unfold(2, n, n).unfold(3, n, n).contiguous().view(-1, n, n)

        freq = self.dct2(blocks, dct_mat)
        freq = torch.round(freq / q) * q
        rec = self.idct2(freq, dct_mat)

        rec = rec.view(b, c, h // n, w // n, n, n)
        rec = rec.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        rec = rec.add(128.0).div(255.0).clamp(0.0, 1.0)
        return rec[..., :orig_h, :orig_w]

    def _build_accuracy_sensitive_mask(self, std_map):
        """Select high-variation benign DCT coefficients as DNN-important bands."""
        if self.preserve_ratio == 0.0:
            return torch.zeros_like(std_map, dtype=torch.bool)
        if self.preserve_ratio == 1.0:
            return torch.ones_like(std_map, dtype=torch.bool)

        flat = std_map.reshape(-1)
        keep = max(1, int(round(flat.numel() * self.preserve_ratio)))
        threshold = torch.topk(flat, keep, largest=True).values.min()
        return std_map >= threshold

    def _build_fd_table(self):
        preserved = torch.full_like(self.defensive_table, self.preserved_quant_step)
        return torch.where(self.accuracy_sensitive_mask, preserved, self.defensive_table).clamp_min(1.0)

    def _quality_scaled_table(self, quality):
        base = self._build_jpeg_table()
        if quality < 50.0:
            scale = 5000.0 / quality
        else:
            scale = 200.0 - 2.0 * quality
        return torch.floor((base * scale + 50.0) / 100.0).clamp(1.0, 255.0)

    @staticmethod
    def build_dct_matrix(device, block, dtype=torch.float32):
        dct = torch.zeros((block, block), device=device, dtype=dtype)
        for k in range(block):
            for n in range(block):
                if k == 0:
                    dct[k, n] = 1.0 / math.sqrt(block)
                else:
                    dct[k, n] = math.sqrt(2.0 / block) * math.cos(math.pi * (2 * n + 1) * k / (2 * block))
        return dct

    @staticmethod
    def dct2(x, dct):
        return dct @ x @ dct.t()

    @staticmethod
    def idct2(x, dct):
        return dct.t() @ x @ dct

    @staticmethod
    def _build_jpeg_table():
        return torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 36, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ], dtype=torch.float32)
