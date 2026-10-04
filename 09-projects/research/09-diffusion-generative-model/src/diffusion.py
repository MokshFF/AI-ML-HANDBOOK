"""Denoising Diffusion Probabilistic Model (DDPM) Implementation."""
import math
import torch
import torch.nn as nn
from typing import Tuple

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class GaussianDiffusion:
    def __init__(self, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        return x_t, noise

class MiniUNetDenoiser(nn.Module):
    def __init__(self, in_channels: int = 1, time_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, 16)
        self.conv2 = nn.Conv2d(16, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = torch.relu(self.conv1(x))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return self.conv2(h)

if __name__ == "__main__":
    diff = GaussianDiffusion(timesteps=100)
    x0 = torch.randn(4, 1, 16, 16)
    t = torch.randint(0, 100, (4,))
    xt, noise = diff.q_sample(x0, t)
    
    model = MiniUNetDenoiser(in_channels=1)
    pred_noise = model(xt, t)
    print(f"Sampled xt shape: {xt.shape} | Predicted noise shape: {pred_noise.shape}")
