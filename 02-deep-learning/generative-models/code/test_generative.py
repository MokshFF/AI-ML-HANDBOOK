import pytest
import torch
from generative_engine import (
    VanillaAutoencoder,
    VariationalAutoencoder,
    GANGenerator,
    GANDiscriminator,
    DDPMNoiseSchedule
)


def test_vanilla_autoencoder():
    ae = VanillaAutoencoder(in_dim=16, latent_dim=4)
    x = torch.randn(8, 16)
    recon, latent = ae(x)
    assert recon.shape == (8, 16)
    assert latent.shape == (8, 4)


def test_variational_autoencoder():
    torch.manual_seed(42)
    vae = VariationalAutoencoder(in_dim=16, hidden_dim=12, latent_dim=4)
    x = torch.randn(8, 16)
    
    recon, mu, logvar = vae(x)
    assert recon.shape == (8, 16)
    assert mu.shape == (8, 4)
    assert logvar.shape == (8, 4)

    losses = vae.elbo_loss(recon, x, mu, logvar)
    assert "loss" in losses and "recon_loss" in losses and "kl_div" in losses
    assert losses["kl_div"].item() >= 0.0

    # Test backpropagation through stochastic node
    losses["loss"].backward()
    for p in vae.parameters():
        assert p.grad is not None


def test_gan_generator_and_discriminator():
    latent_dim = 16
    data_dim = 8
    gen = GANGenerator(latent_dim=latent_dim, out_dim=data_dim)
    disc = GANDiscriminator(in_dim=data_dim)

    z = torch.randn(4, latent_dim)
    fake_data = gen(z)
    assert fake_data.shape == (4, data_dim)

    d_out_fake = disc(fake_data)
    assert d_out_fake.shape == (4, 1)
    assert ((d_out_fake >= 0.0) & (d_out_fake <= 1.0)).all()

    real_data = torch.randn(4, data_dim)
    d_out_real = disc(real_data)
    assert d_out_real.shape == (4, 1)


def test_ddpm_noise_schedule():
    schedule = DDPMNoiseSchedule(timesteps=50, beta_start=1e-4, beta_end=0.02)
    assert schedule.alphas_cumprod.shape == (50,)
    # Alpha cumprod should be strictly monotonically decreasing
    diffs = torch.diff(schedule.alphas_cumprod)
    assert (diffs < 0).all()

    x_0 = torch.ones(2, 4)
    # Timestep 0: x_t should be very close to x_0
    t_0 = torch.tensor([0, 0])
    x_t0, noise0 = schedule.q_sample(x_0, t_0)
    assert x_t0.shape == (2, 4)

    # Timestep 49: alpha_cumprod is small, variance dominated by noise
    t_49 = torch.tensor([49, 49])
    x_t49, noise49 = schedule.q_sample(x_0, t_49)
    assert x_t49.shape == (2, 4)
