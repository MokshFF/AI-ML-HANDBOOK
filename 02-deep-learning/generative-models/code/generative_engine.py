"""
Generative Deep Learning Models in PyTorch.
Implements:
1. Vanilla Autoencoder (AE).
2. Variational Autoencoder (VAE) with reparameterization trick and ELBO loss.
3. Generative Adversarial Network (GAN) with Generator and Discriminator.
4. Denoising Diffusion Probabilistic Model (DDPM) noise schedule and forward sampling.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


# ============================================================================
# 1. Vanilla Autoencoder
# ============================================================================

class VanillaAutoencoder(nn.Module):
    """
    Deterministic Autoencoder with dimensional bottleneck.
    """
    def __init__(self, in_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 12),
            nn.ReLU(),
            nn.Linear(12, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, in_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# ============================================================================
# 2. Variational Autoencoder (VAE)
# ============================================================================

class VariationalAutoencoder(nn.Module):
    """
    Probabilistic VAE (Kingma & Welling, 2013).
    Encoder predicts mean mu and log-variance log(sigma^2).
    Reparameterization: z = mu + sigma * epsilon.
    """
    def __init__(self, in_dim: int = 16, hidden_dim: int = 12, latent_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: enables backpropagation through stochastic nodes."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def elbo_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        ELBO = Reconstruction Loss (MSE) + KL Divergence against standard normal prior N(0, I)
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        # Analytical KL divergence for Gaussian prior
        kl_div = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total_loss = recon_loss + kl_div
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_div": kl_div}


# ============================================================================
# 3. Generative Adversarial Network (GAN)
# ============================================================================

class GANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, out_dim),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GANDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 4. Diffusion Fundamentals (DDPM Noise Schedule)
# ============================================================================

class DDPMNoiseSchedule:
    """
    Linear beta noise schedule for Denoising Diffusion Probabilistic Models (Ho et al., 2020).
    Enables direct forward diffusion sampling to arbitrary timestep t without iterative looping.
    """
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        # Cumulative product alpha_bar = prod_{s=1}^t alpha_s
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion step:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, *([1] * (x_start.ndim - 1)))
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, *([1] * (x_start.ndim - 1)))

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
