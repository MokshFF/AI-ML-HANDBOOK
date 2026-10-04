# Seminal Research Papers: Generative AI & Alignment

Foundational probabilistic generative models, diffusion mechanics, and direct preference alignment.

---

## 1. Denoising Diffusion Probabilistic Models (DDPM)
- **Title**: Denoising Diffusion Probabilistic Models
- **Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Year**: 2020
- **Link**: https://arxiv.org/abs/2006.11239
- **Problem**: GANs suffer from training instability and mode collapse; VAEs produce blurry samples.
- **Main Idea**: Formulate generation as reversing a Markovian forward process that gradually corrupts data with Gaussian noise.
- **Key Contribution**: Simplified the variational lower bound (ELBO) objective into a weighted mean squared error predicting added noise $\epsilon$.
- **Important Architecture/Math**:
  $$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \; t)\|^2 \right]$$
- **Why It Matters**: The core foundation for modern visual generative AI (Stable Diffusion, Midjourney, Sora).
- **Prerequisites**: Markov chains, variational autoencoders, Langevin dynamics.
- **Suggested Follow-up Papers**: *Score-Based Generative Modeling through Stochastic Differential Equations* (Song et al., 2020); *High-Resolution Image Synthesis with Latent Diffusion Models* (Rombach et al., 2022).

---

## 2. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)
- **Title**: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **Authors**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year**: 2023
- **Link**: https://arxiv.org/abs/2305.18290
- **Problem**: Reinforcement Learning from Human Feedback (RLHF) via PPO is notoriously unstable, complex, and memory-intensive (requires 4 concurrent models).
- **Main Idea**: Mathematically derive an exact closed-form equivalence between the optimal policy and the reward function, enabling direct optimization on pairwise preferences via cross-entropy.
- **Key Contribution**: Eliminated the separate reward model and reinforcement learning loop while achieving equal or superior alignment stability.
- **Important Architecture/Math**:
  $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
- **Why It Matters**: Replaced RLHF-PPO across industry labs for fine-tuning open foundation models.
- **Prerequisites**: Bradley-Terry preference model, KL divergence, RLHF fundamentals.
- **Suggested Follow-up Papers**: *KTO: Model Alignment as Prospect Theoretic Optimization* (Ethayarajh et al., 2024); *Direct Nash Optimization* (Rosset et al., 2024).
