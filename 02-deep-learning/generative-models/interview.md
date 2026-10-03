# Generative Deep Learning - Technical Interview Preparation

A curated question bank covering ELBO derivations, the Reparameterization Trick, GAN minimax game theory, mode collapse, and Diffusion mechanics.

---

## 1. Probabilistic Latent Variable Models & VAEs

### Q1: Derive the Evidence Lower Bound (ELBO) and explain why the marginal log-likelihood $\log p(x)$ cannot be directly optimized.
- **Intractability of the Marginal**:
  In a latent variable model, the marginal data likelihood is:
  $$p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$$
  Because $z$ is continuous and the mapping $p_\theta(x \mid z)$ is parameterized by a non-linear neural network, this integral cannot be computed analytically. Evaluating it numerically via Monte Carlo requires sampling exponentially many $z$ values, most of which have near-zero probability of generating $x$.
- **ELBO Derivation**:
  Introduce a variational recognition distribution $q_\phi(z \mid x)$ and rewrite $\log p_\theta(x)$:
  $$\log p_\theta(x) = \int q_\phi(z \mid x) \log p_\theta(x) dz = \int q_\phi(z \mid x) \log \left( \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right) dz$$
  $$= \int q_\phi(z \mid x) \log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} dz + \int q_\phi(z \mid x) \log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} dz$$
  $$= \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) + \mathbb{D}_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x))$$
  Because the Kullback-Leibler divergence is strictly non-negative ($\mathbb{D}_{\text{KL}} \ge 0$):
  $$\log p_\theta(x) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi; x)$$
  Maximizing $\mathcal{L}_{\text{ELBO}}$ simultaneously pushes up the evidence lower bound and minimizes the KL divergence between our approximation $q_\phi(z \mid x)$ and the true intractable posterior $p_\theta(z \mid x)$.

---

### Q2: What is the Reparameterization Trick, and why is it essential for training VAEs?
- **The Problem with Direct Stochastic Sampling**:
  To compute the reconstruction loss $\mathbb{E}_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)]$, we must sample $z \sim q_\phi(z \mid x)$.
  If we sample directly, $z$ is the output of a stochastic node. During backpropagation, upstream gradients $\frac{\partial \mathcal{L}}{\partial z}$ cannot pass backward through a random sampling operation to compute $\frac{\partial \mathcal{L}}{\partial \phi}$ (where $\phi$ are encoder weights predicting $\mu$ and $\sigma$).
- **The Reparameterization Solution**:
  We reformulate the stochastic variable as an affine deterministic transformation of an auxiliary noise source $\epsilon \sim \mathcal{N}(0, I)$:
  $$z = g_\phi(x, \epsilon) = \mu(x) + \sigma(x) \odot \epsilon$$
  Now:
  1. The random sampling operation is externalized and has no learnable parameters.
  2. $z$ is fully differentiable with respect to $\mu$ and $\sigma$:
     $$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$
  This allows standard backpropagation to update the encoder weights $\phi$ seamlessly.

---

## 2. Adversarial Game Theory & Diffusion

### Q3: Derive the optimal discriminator $D^*(x)$ in standard GANs and prove that the virtual objective minimizes Jensen-Shannon Divergence.
- **Discriminator Optimization**:
  The minimax value function is:
  $$V(D, G) = \int \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$
  To find the optimal $D$ for a fixed $G$, differentiate the integrand $f(y) = a \log y + b \log(1 - y)$ with respect to $y = D(x)$:
  $$f'(y) = \frac{a}{y} - \frac{b}{1 - y} = 0 \implies a(1 - y) = by \implies y^* = \frac{a}{a + b}$$
  Substituting $a = p_{\text{data}}(x)$ and $b = p_g(x)$:
  $$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$
- **Virtual Objective**:
  Plugging $D^*$ back into $V(D^*, G)$:
  $$V(D^*, G) = \int p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} dx + \int p_g(x) \log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)} dx$$
  $$= -\log 4 + \mathbb{D}_{\text{KL}}\left(p_{\text{data}} \parallel \frac{p_{\text{data}} + p_g}{2}\right) + \mathbb{D}_{\text{KL}}\left(p_g \parallel \frac{p_{\text{data}} + p_g}{2}\right)$$
  $$= -\log 4 + 2 \cdot \mathbb{D}_{\text{JS}}(p_{\text{data}} \parallel p_g)$$
  Thus, training the generator against an optimal discriminator minimizes the **Jensen-Shannon Divergence** between the true data distribution and the model distribution.

---

### Q4: How do Denoising Diffusion Probabilistic Models (DDPM) achieve closed-form jumping to timestep $t$?
- **Iterative Forward Transition**:
  $$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}$$
  Let $\alpha_t = 1 - \beta_t$. Expanding across timesteps:
  $$x_t = \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2}) + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$
  $$= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$
- **Sum of Independent Gaussians**:
  The sum of two independent Gaussians $\mathcal{N}(0, \sigma_1^2 I)$ and $\mathcal{N}(0, \sigma_2^2 I)$ is $\mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)I)$.
  The combined variance is $\alpha_t (1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}$.
  By induction across $t$ steps, with $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:
  $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}, \quad \bar{\epsilon} \sim \mathcal{N}(0, I)$$
  This closed-form property allows training batches to sample random timesteps $t \sim \mathcal{U}(1, T)$ independently, training the network in parallel without simulating prior steps.

---

## 3. Whiteboard Coding Drills

### Q5: Write the Reparameterization Trick and Gaussian ELBO loss in PyTorch.
```python
import torch
import torch.nn.functional as F

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # std = exp(0.5 * logvar)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_elbo(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    # Analytical KL divergence for N(0, I) prior
    kl_div = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + kl_div
    return total_loss, recon_loss, kl_div
```
