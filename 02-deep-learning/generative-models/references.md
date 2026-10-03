# Generative Deep Learning - Curated References & Bibliography

A curated collection of foundational research papers, landmark monographs, and official documentation on generative deep learning.

---

## 1. Landmark Research Papers

### 1.1 Variational Inference & VAEs
- **Auto-Encoding Variational Bayes (VAE)**
  - *Authors*: Diederik P. Kingma, Max Welling (ICLR, 2014).
  - *Significance*: Formulated the stochastic gradient variational Bayes (SGVB) estimator, Evidence Lower Bound (ELBO), and reparameterization trick.
- **Stochastic Backpropagation and Approximate Inference in Deep Generative Models**
  - *Authors*: Danilo Jimenez Rezende, Shakir Mohamed, Daan Wierstra (ICML, 2014).
  - *Significance*: Independently formulated amortized variational inference in deep networks.
- **Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**
  - *Authors*: Irina Higgins, Loic Matthey, Arka Pal, et al. (ICLR, 2017).
  - *Significance*: Introduced adjustable hyperparameter $\beta$ on the KL penalty to discover disentangled latent factors.

### 1.2 Adversarial Frameworks (GANs)
- **Generative Adversarial Nets**
  - *Authors*: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (NeurIPS, 2014).
  - *Significance*: Introduced the two-player minimax game framework for implicit generative modeling.
- **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)**
  - *Authors*: Alec Radford, Luke Metz, Soumith Chintala (ICLR, 2016).
  - *Significance*: Established architectural guidelines (strided convs, batchnorm, leaky relu) stabilizing adversarial training.
- **Wasserstein GAN (WGAN)**
  - *Authors*: Martin Arjovsky, Soumith Chintala, Léon Bottou (ICML, 2017).
  - *Significance*: Replaced JS divergence with Earth Mover's Distance, solving vanishing gradients.
- **Improved Training of Wasserstein GANs (WGAN-GP)**
  - *Authors*: Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (NeurIPS, 2017).
  - *Significance*: Replaced weight clipping with gradient penalty, achieving unprecedented stability.

### 1.3 Diffusion & Score-Based Models
- **Deep Unsupervised Learning using Nonequilibrium Thermodynamics**
  - *Authors*: Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli (ICML, 2015).
  - *Significance*: Original physics-inspired formulation of diffusion probabilistic models.
- **Denoising Diffusion Probabilistic Models (DDPM)**
  - *Authors*: Jonathan Ho, Ajay Jain, Pieter Abbeel (NeurIPS, 2020).
  - *Significance*: Modern formulation proving equivalence between diffusion models and denoising score matching, sparking modern generative visual AI (Stable Diffusion, Midjourney, DALL-E).
- **High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)**
  - *Authors*: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (CVPR, 2022).
  - *Significance*: Applied diffusion in the compressed latent space of a pretrained autoencoder, drastically cutting computational demands.

---

## 2. Textbooks & Monographs

- **Generative Deep Learning (2nd Edition)** by David Foster (O'Reilly, 2023).
- **Deep Learning (Chapter 20: Deep Generative Models)** by Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press).
- **What are Diffusion Models?** by Lilian Weng (Canonical technical overview blog post).
