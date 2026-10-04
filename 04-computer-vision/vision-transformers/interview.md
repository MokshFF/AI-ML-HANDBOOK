# Vision Transformers & Image Embeddings - Technical Interview Preparation

A curated question bank covering ViT patch embedding mechanics, inductive bias trade-offs, handling variable image resolutions, and visual metric learning.

---

## 1. Architectural & Theoretical Foundations

### Q1: Why do Vision Transformers underperform CNNs when trained from scratch on small datasets (e.g. standard ImageNet-1k without heavy augmentation), yet outperform CNNs when pre-trained on massive datasets (JFT-300M)?
- **Inductive Biases**:
  - CNNs have built-in hard-coded inductive priors: **translation equivariance** ($f(g(x)) = g(f(x))$) and **local spatial locality** (neighboring pixels are correlated; distant pixels are initially independent). Because of these priors, CNNs learn effective filters from modest amounts of training data.
  - Transformers possess virtually no visual inductive bias. Self-attention allows any patch to attend to any other patch globally from layer 1. The model must *learn* 2D spatial topology, grid geometry, and local edge detectors entirely from training data.
- **The Scaling Law**:
  On small datasets, ViT easily overfits because its hypothesis space is vast. On giant datasets (e.g., hundreds of millions of images), the lack of restrictive convolutional constraints allows ViT to learn richer, more flexible global visual representations that surpass CNN capacity.

---

### Q2: How can a Vision Transformer handle images with higher or variable resolution during inference compared to its pre-training resolution?
- **The Resolution Mismatch Problem**:
  If a model is pre-trained with patch size $P=16$ on $224 \times 224$ images, it has $14 \times 14 = 196$ position embeddings. At inference with $384 \times 384$ images, there are $24 \times 24 = 576$ patches. The fixed position embedding matrix dimension no longer matches.
- **Bicubic Positional Interpolation**:
  Dosovitskiy et al. introduced 2D bicubic interpolation:
  1. Omit the `[CLS]` token position embedding.
  2. Reshape the remaining $196$ position embedding vectors from $(196, D) \to (14, 14, D)$.
  3. Perform continuous 2D bicubic interpolation to upsample the grid to $(24, 24, D)$.
  4. Flatten back to $(576, D)$ and re-attach the `[CLS]` embedding.
  This allows fine-tuning and inference at higher resolutions with minimal fine-tuning.

---

### Q3: What is the architectural difference between CRNN and TrOCR for text recognition?
- **CRNN (CNN + RNN + CTC)**:
  Uses CNN layers to extract high-level feature maps, collapses the vertical dimension to form a 1D feature sequence, passes it through a Bidirectional LSTM, and optimizes the transcription alignment via Connectionist Temporal Classification (CTC) loss without requiring pre-segmented character labels.
- **TrOCR (Transformer OCR)**:
  Completely eliminates CNNs and RNNs. It utilizes a pre-trained Vision Transformer (e.g. DeiT or BEiT) as the image encoder, generating patch sequence representations, coupled to a pre-trained causal text decoder (e.g., RoBERTa/BART), generating text autoregressively with standard cross-entropy loss.

---

## 2. Whiteboard Coding Drills

### Q4: Implement Patch Embedding in PyTorch using a 2D convolution and sequence flattening.
```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)                  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
```
