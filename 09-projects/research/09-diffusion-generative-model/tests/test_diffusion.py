"""Tests for Diffusion Generative Pipeline."""
import torch
from diffusion import GaussianDiffusion, SinusoidalTimeEmbedding, MiniUNetDenoiser

def test_sinusoidal_embedding():
    emb_module = SinusoidalTimeEmbedding(dim=32)
    t = torch.tensor([0, 50, 100])
    out = emb_module(t)
    assert out.shape == (3, 32)
    assert not torch.isnan(out).any()

def test_gaussian_diffusion_forward():
    diff = GaussianDiffusion(timesteps=50)
    x0 = torch.zeros(2, 1, 8, 8)
    t = torch.tensor([49, 49])
    xt, noise = diff.q_sample(x0, t)
    # At t=T, variance should be close to 1.0 (pure Gaussian noise)
    assert xt.shape == x0.shape
    assert xt.std() > 0.5

def test_denoiser_network():
    model = MiniUNetDenoiser(in_channels=1)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([5, 12])
    pred = model(x, t)
    assert pred.shape == x.shape
