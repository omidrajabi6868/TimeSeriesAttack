"""DiffPure-style diffusion training and adversarial purification utilities.

This module intentionally keeps the implementation self-contained and
PyTorch-native so it can be trained on the project dataset before being used as
an input-purification defense.  The purification follows the core DiffPure
recipe: diffuse an input to a small timestep, then solve the learned reverse
process back to a clean image before classification.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        exponent = -torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device) / max(half - 1, 1)
        emb = timesteps.float().unsqueeze(1) * exponent.exp().unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_group_count(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_valid_group_count(out_channels, groups), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """Compact unconditional DDPM noise-prediction U-Net for image tensors."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        c = int(base_channels)
        self.init = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(c, c, time_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResidualBlock(c * 2, c * 2, time_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResidualBlock(c * 4, c * 4, time_dim)
        self.mid2 = ResidualBlock(c * 4, c * 4, time_dim)
        self.upsample2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(c * 4, c * 2, time_dim)
        self.upsample1 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(c * 2, c, time_dim)
        self.out_norm = nn.GroupNorm(_valid_group_count(c), c)
        self.out = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        pad_h = (4 - orig_size[0] % 4) % 4
        pad_w = (4 - orig_size[1] % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        t = self.time_mlp(timesteps)
        x1 = self.down1(self.init(x), t)
        x2 = self.down2(self.downsample1(x1), t)
        h = self.mid2(self.mid1(self.downsample2(x2), t), t)
        h = self.upsample2(h)
        h = self.up2(torch.cat([h, x2], dim=1), t)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, x1], dim=1), t)
        h = self.out(F.silu(self.out_norm(h)))
        return h[..., :orig_size[0], :orig_size[1]]


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        betas = torch.linspace(float(beta_start), float(beta_end), int(timesteps), dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))

    @property
    def timesteps(self) -> int:
        return int(self.betas.numel())

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        return values.gather(0, timesteps).view(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * x_start + self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * noise


class DiffusionPurifier(nn.Module):
    """Trainable DDPM and DiffPure-style purifier for inputs in [0, 1]."""

    def __init__(self, model: Optional[nn.Module] = None, image_channels: int = 3, base_channels: int = 64, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.model = model if model is not None else DiffusionUNet(image_channels, base_channels)
        self.schedule = DiffusionSchedule(timesteps, beta_start, beta_end)
        self.image_channels = int(image_channels)
        self.base_channels = int(base_channels)

    def training_loss(self, images: torch.Tensor) -> torch.Tensor:
        x0 = images.mul(2.0).sub(1.0)
        t = torch.randint(0, self.schedule.timesteps, (x0.shape[0],), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.schedule.q_sample(x0, t, noise)
        pred_noise = self.model(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def purify(self, images: torch.Tensor, diffusion_step: int = 100, reverse_steps: Optional[int] = None, stochastic: bool = True) -> torch.Tensor:
        """Diffuse to ``diffusion_step`` and reverse-denoise back to image space.

        ``reverse_steps`` can subsample the reverse trajectory for speed.  Leaving
        it unset uses every reverse step from ``diffusion_step`` to zero, which is
        the most faithful DDPM approximation of DiffPure's reverse process.
        """
        was_training = self.training
        self.eval()
        max_t = self.schedule.timesteps - 1
        t_start = int(max(0, min(diffusion_step, max_t)))
        x = images.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        t = torch.full((x.shape[0],), t_start, device=x.device, dtype=torch.long)
        x = self.schedule.q_sample(x, t, torch.randn_like(x))

        if reverse_steps is None or reverse_steps >= t_start + 1:
            step_sequence = range(t_start, -1, -1)
        else:
            num_points = max(2, int(reverse_steps))
            points = torch.linspace(t_start, 0, num_points, device=x.device).round().long().unique_consecutive()
            if int(points[-1].item()) != 0:
                points = torch.cat([points, torch.zeros(1, device=x.device, dtype=torch.long)])
            step_sequence = points.tolist()

        for step in step_sequence:
            step_t = torch.full((x.shape[0],), int(step), device=x.device, dtype=torch.long)
            beta_t = self.schedule._extract(self.schedule.betas, step_t, x.shape)
            sqrt_one_minus = self.schedule._extract(self.schedule.sqrt_one_minus_alpha_cumprod, step_t, x.shape)
            sqrt_recip_alpha = self.schedule._extract(self.schedule.sqrt_recip_alphas, step_t, x.shape)
            model_mean = sqrt_recip_alpha * (x - beta_t * self.model(x, step_t) / sqrt_one_minus)
            if step > 0:
                var = self.schedule._extract(self.schedule.posterior_variance, step_t, x.shape)
                noise = torch.randn_like(x) if stochastic else torch.zeros_like(x)
                x = model_mean + torch.sqrt(var.clamp_min(1e-20)) * noise
            else:
                x = model_mean
        if was_training:
            self.train()
        return x.add(1.0).div(2.0).clamp(0.0, 1.0)

    def predict(self, classifier: nn.Module, inputs: torch.Tensor, batch_size: int = 32, diffusion_step: int = 100, reverse_steps: Optional[int] = None, stochastic: bool = True, output_device: Optional[torch.device] = None) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if output_device is None:
            output_device = inputs.device
        preds = []
        classifier.eval()
        self.eval()
        with torch.no_grad():
            for start in range(0, inputs.shape[0], batch_size):
                end = min(start + batch_size, inputs.shape[0])
                purified = self.purify(inputs[start:end], diffusion_step=diffusion_step, reverse_steps=reverse_steps, stochastic=stochastic)
                preds.append((classifier(purified) > 0).float().view(-1).cpu())
        return torch.cat(preds, dim=0).to(output_device)

    def checkpoint_dict(self, extra: Optional[Dict] = None) -> Dict:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "timesteps": self.schedule.timesteps,
            "beta_start": float(self.schedule.betas[0].item()),
            "beta_end": float(self.schedule.betas[-1].item()),
        }
        if extra:
            payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        purifier = cls(
            image_channels=int(checkpoint.get("image_channels", 3)),
            base_channels=int(checkpoint.get("base_channels", 64)),
            timesteps=int(checkpoint.get("timesteps", 1000)),
            beta_start=float(checkpoint.get("beta_start", 1e-4)),
            beta_end=float(checkpoint.get("beta_end", 2e-2)),
        )
        purifier.model.load_state_dict(checkpoint["model_state_dict"])
        return purifier

    def save_checkpoint(self, checkpoint_path, extra: Optional[Dict] = None) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(extra), path)
