# Attention & Transformers - Technical Interview Preparation

A curated question bank covering attention scaling derivations, computational complexity, Pre-LN vs. Post-LN dynamics, RoPE mechanics, and Vision Transformers.

---

## 1. Mathematical Derivations & Operational Complexity

### Q1: Derive the variance of the dot product $q^T k$ and explain why scaling by $\sqrt{d_k}$ is mathematically necessary.
- **Derivation**:
  Let $q, k \in \mathbb{R}^{d_k}$ where each element $q_i, k_i$ are independent random variables with mean 0 and variance 1 ($\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$, $\text{Var}(q_i) = \text{Var}(k_i) = 1$).
  The dot product is $z = q^T k = \sum_{i=1}^{d_k} q_i k_i$.
  1. **Mean**:
     $$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$
  2. **Variance**:
     Since the terms are independent:
     $$\text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left( \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 \right) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = \sum_{i=1}^{d_k} 1 \cdot 1 = d_k$$
- **Softmax Saturation Effect**:
  As $d_k$ grows (e.g. $d_k = 128$), the standard deviation of raw dot products is $\sqrt{d_k} \approx 11.3$.
  When inputs to the softmax have differences on the order of 10-20, the softmax function outputs values extremely close to 1 for the largest element and 0 for others.
  In this saturated regime, the derivative of softmax $\frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)$ vanishes to zero!
  Dividing by $\sqrt{d_k}$ scales $\text{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(q^T k) = 1.0$, guaranteeing stable gradient flow regardless of head dimension.

---

### Q2: What is the exact time and memory complexity of Self-Attention, and why is long context challenging?
- **Analysis for Sequence Length $N$ and Model Dimension $D$**:
  1. **Linear Projections**:
     $Q = X W_Q$, $K = X W_K$, $V = X W_V$: $3 \times (N \times D) \times (D \times D) \implies \mathcal{O}(N D^2)$.
  2. **Attention Scores**:
     $Q K^T$: $(N \times D) \times (D \times N) \implies \mathcal{O}(N^2 D)$ operations.
     Memory required to store the attention matrix: $\mathcal{O}(N^2)$ per head, $\mathcal{O}(h N^2)$ total.
  3. **Context Weighting**:
     $\text{Softmax}(A) V$: $(N \times N) \times (N \times D) \implies \mathcal{O}(N^2 D)$ operations.
  4. **Output Projection**:
     $O W_O$: $(N \times D) \times (D \times D) \implies \mathcal{O}(N D^2)$.
- **Total Complexity**:
  $$\text{FLOPs} = \mathcal{O}(N D^2 + N^2 D)$$
  $$\text{Activation Memory} = \mathcal{O}(N^2 \cdot h + N \cdot D)$$
- **Long Context Bottleneck**:
  When $N \gg D$ (e.g. $N = 32\text{k}$ or $128\text{k}$ tokens, $D = 4096$), the quadratic $\mathcal{O}(N^2)$ term dominates both compute and GPU VRAM. This has driven the adoption of FlashAttention (tiled memory hierarchy computation without materializing $N \times N$ matrices in HBM) and sparse linear attention alternatives.

---

## 2. Architectural Choices & Paradigms

### Q3: Why did the LLM industry transition from Post-LayerNorm to Pre-LayerNorm?
- **Post-LN**:
  $$x_{l+1} = \text{LayerNorm}(x_l + \mathcal{F}(x_l))$$
  Gradients traversing backwards must differentiate through the LayerNorm normalization factor $\sigma_l$ at every single layer. For deep architectures ($L > 30$), the expected gradient magnitude decays exponentially with depth, causing the model to train extremely slowly or diverge completely unless a very long warmup learning rate schedule is used.
- **Pre-LN**:
  $$x_{l+1} = x_l + \mathcal{F}(\text{LayerNorm}(x_l))$$
  The identity path $x_l$ is unnormalized and uninterrupted. Expanding across $L$ layers:
  $$x_L = x_0 + \sum_{l=0}^{L-1} \mathcal{F}_l(\text{LayerNorm}(x_l))$$
  $$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{l=0}^{L-1} \frac{\partial \mathcal{F}_l}{\partial x_0} \right)$$
  The direct $I$ connection guarantees gradient signals propagate back to input embeddings with zero attenuation, enabling stable training without fragile warmup schedules.

---

### Q4: Why do Vision Transformers (ViT) require much more training data than CNNs to achieve competitive performance?
- **Inductive Biases**:
  - **Convolutional Networks** have two hard-coded inductive biases:
    1. *Translational Equivariance*: A feature detector learned at one pixel position applies identically across the entire image.
    2. *Locality*: Nearby pixels are strongly correlated; distant pixels are initially independent.
    These assumptions are structurally built into the convolutional kernel, allowing CNNs to learn robust visual representations from small datasets (e.g., ImageNet-1k with 1.2M images).
  - **Vision Transformers (ViT)** have **no locality bias**: every patch can attend to every other patch at layer 1. The model must *learn* from scratch that nearby pixels are related.
- **Consequence of Data Scale**:
  - On small datasets (ImageNet-1k without pretraining), ViT underperforms ResNet because it lacks these visual priors.
  - On massive datasets (JFT-300M, LAION-5B), ViT significantly outperforms CNNs because its unconstrained capacity allows it to discover richer, global multi-modal patterns that rigid convolutional weight sharing cannot express.

---

## 3. Whiteboard Coding Drills

### Q5: Implement Multi-Head Attention in PyTorch using efficient batched tensor reshaping (no loops over heads).
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape
        # Compute Q, K, V in one single linear projection: (B, S, 3 * D)
        qkv = self.qkv_proj(x)
        # Reshape to (3, B, num_heads, S, d_k)
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        # Context aggregation: (B, num_heads, S, d_k)
        context = torch.matmul(attn_weights, v)
        # Concatenate heads: (B, S, D)
        context = context.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(context)
```
