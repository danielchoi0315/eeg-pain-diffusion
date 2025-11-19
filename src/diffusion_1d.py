"""
diffusion_1d.py

1D Gaussian diffusion wrapper around EEGUNetDeep.

- Correct forward diffusion:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
- Cosine or linear beta schedule
- Training (epsilon-prediction) forward()
- Sampling utilities (p_sample, p_sample_loop)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(
    timesteps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Create beta schedule.

    - "linear": classic DDPM linear schedule
    - "cosine": cosine schedule (Nichol & Dhariwal 2021)
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(min=1e-8, max=0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


@dataclass
class DiffusionConfig:
    timesteps: int = 200
    schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02


class GaussianDiffusion1D(nn.Module):
    """
    Gaussian diffusion process around a denoising model (epsilon predictor).
    """

    def __init__(self, model: nn.Module, config: DiffusionConfig, device: torch.device):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device

        betas = make_beta_schedule(
            timesteps=config.timesteps,
            schedule=config.schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        self.register_buffer("betas", betas.to(device))

        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat(
                [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0
            ),
        )

        # ---- Correct forward diffusion ----
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )

        # For sampling
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "sqrt_recipm1_alphas",
            torch.sqrt(1.0 / alphas - 1.0),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diffuse x_start to x_t using q(x_t | x_0).

        Returns:
            x_t, noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_t = sqrt_ab * x_start + sqrt_one_minus_ab * noise
        return x_t, noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training step:
        - sample t ~ Uniform(0, T-1)
        - sample epsilon
        - get x_t
        - predict epsilon, minimize MSE
        """
        b = x.size(0)
        device = x.device

        t = torch.randint(
            low=0,
            high=self.config.timesteps,
            size=(b,),
            device=device,
            dtype=torch.long,
        )
        x_t, noise = self.q_sample(x, t)
        pred_noise = self.model(x_t, t)

        if pred_noise.shape != noise.shape:
            min_len = min(pred_noise.shape[-1], noise.shape[-1])
            pred_noise = pred_noise[..., :min_len]
            noise = noise[..., :min_len]

        return F.mse_loss(pred_noise, noise)

    # ---------------- Sampling utilities ---------------- #

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step: p_theta(x_{t-1} | x_t)
        """
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1)

        eps_theta = self.model(x, t)

        if eps_theta.shape != x.shape:
            min_len = min(eps_theta.shape[-1], x.shape[-1])
            eps_theta = eps_theta[..., :min_len]
            x = x[..., :min_len]

        # DDPM mean
        model_mean = sqrt_recip_alpha * (x - betas_t / sqrt_one_minus_ab * eps_theta)

        # t = 0 → deterministic
        nonzero_mask = (t != 0).float().view(-1, 1, 1)
        noise = torch.randn_like(x)
        sigma_t = torch.sqrt(betas_t)
        x_prev = model_mean + nonzero_mask * sigma_t * noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Generate samples starting from pure noise.
        shape: (B, C, T)
        """
        device = self.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.config.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x
