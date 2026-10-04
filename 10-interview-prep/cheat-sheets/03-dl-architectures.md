# Cheat Sheet: Deep Learning Architectures

| Architecture | Signature Innovation | Primary Domain | Strengths | Trade-Offs |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet** | Additive residual skip connection ($x + F(x)$) | Vision | Solves vanishing gradient; enables 1000+ layers | Additive features may limit representation capacity |
| **ConvNeXt** | Modernized 7x7 depthwise separable convs | Vision | Matches Swin Transformer speed and accuracy | Lacks global cross-attention context |
| **Vision Transformer (ViT)** | Patch projection ($16 \times 16$) + self-attention | Vision | Global receptive field from layer 1 | Lacks inductive bias; requires massive pretraining |
| **LSTM / GRU** | Gated cell memory states ($\{f_t, i_t, o_t\}$) | Sequences | Mitigates vanishing gradient in temporal series | Sequential processing; cannot parallelize across time |
| **Transformer** | Scaled Dot-Product Multi-Head Attention | Language, Vision, Audio | Fully parallelizable; global attention | $\mathcal{O}(N^2)$ memory & compute scaling |
| **U-Net** | Encoder-decoder with skip concatenation | Segmentation, Diffusion | High-resolution spatial detail preservation | High memory consumption during training |
| **DDPM (Diffusion)** | Learned reverse Gaussian denoising process | Image/Audio Gen | High sample fidelity; stable training | Slow multi-step iterative sampling ($T=50-1000$) |
