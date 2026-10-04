# Vision Transformers (ViT), Image Embeddings & OCR Architectures

A comprehensive technical guide to applying the Transformer architecture directly to 2D computer vision, patch projections, inductive bias differences between CNNs and Transformers, metric learning image embeddings, and sequence recognition in Optical Character Recognition (OCR).

---

## 1. Vision Transformer (ViT) Foundations (Dosovitskiy et al., 2020)

Prior to ViT, computer vision was dominated by convolutional operators. The Vision Transformer demonstrated that pure multi-head self-attention applied directly to sequences of non-overlapping image patches achieves or exceeds state-of-the-art accuracy when scaled.

### 1.1 Patch Linear Projection
To handle 2D image $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$, ViT slices it into a grid of non-overlapping patches $\mathbf{x}_p \in \mathbb{R}^{P \times P \times C}$:
$$N = \frac{H \cdot W}{P^2} = \text{sequence length}$$
Each patch is flattened into a 1D vector of dimension $P^2 C$ and linearly projected to model dimension $D$:
$$\mathbf{x}_p^{(i)} \mathbf{W}_e, \quad \mathbf{W}_e \in \mathbb{R}^{(P^2 C) \times D}$$

### 1.2 The Complete Token Sequence
Similar to BERT, a learnable class token $\mathbf{x}_{\text{cls}} \in \mathbb{R}^D$ is prepended, and 1D learnable position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are added:
$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \, \mathbf{x}_p^{(1)}\mathbf{W}_e; \, \dots; \, \mathbf{x}_p^{(N)}\mathbf{W}_e] + \mathbf{E}_{\text{pos}}$$

```
Image (C x H x W)
      │
  [Patch Extraction & Projection]  Conv2d (stride=P, kernel=P)
      │
 Sequence of N Patch Vectors (N x D)
      │
 [Prepend [CLS] + Add Pos Embeddings]
      │
 ┌────▼──────────────────────────────┐
 │   Transformer Encoder Block x L   │
 │   - LayerNorm (Pre-LN)            │
 │   - Multi-Head Self-Attention     │
 │   - Residual Connection           │
 │   - LayerNorm (Pre-LN)            │
 │   - MLP (GELU)                    │
 │   - Residual Connection           │
 └────┬──────────────────────────────┘
      │
  [CLS] Token Output (z_L^0 in R^D) ──> Metric Embedding / MLP Classification Head
```

---

## 2. Inductive Bias: CNNs vs. Vision Transformers

| Dimension | Convolutional Neural Networks (CNNs) | Vision Transformers (ViTs) |
| :--- | :--- | :--- |
| **Inductive Bias** | High: built-in translation equivariance and local spatial locality. | Low: learns spatial relationships and geometry entirely from scratch via attention. |
| **Data Efficiency** | High sample efficiency on small datasets (e.g. CIFAR-10). | Requires large-scale pretraining (e.g. JFT-300M, ImageNet-22k) or aggressive data augmentation. |
| **Receptive Field** | Grows linearly with depth ($O(L \cdot k)$). | Global ($O(1)$) across all image patches from layer 1. |
| **Computational Scaling** | $O(H \cdot W \cdot C^2)$ per convolution. | $O(N^2 \cdot D) = O\left(\frac{H^2 W^2}{P^4} \cdot D\right)$ quadratic in patches. |

---

## 3. Image Embeddings & Metric Learning

Image embeddings map raw pixels into a dense latent metric space $\mathbb{R}^D$ where geometric distance reflects visual or semantic similarity:
- **Unit Hypersphere Projection**:
  $$\mathbf{e} = \frac{\mathbf{h}}{\|\mathbf{h}\|_2} \in \mathbb{S}^{D-1}$$
- **Pairwise Cosine Similarity**:
  $$\text{sim}(\mathbf{e}_1, \mathbf{e}_2) = \mathbf{e}_1^\top \mathbf{e}_2 = \cos(\theta)$$
- **Applications**: Reverse image search, visual deduplication, few-shot classification, and visual retrieval (CLIP / vector databases).

---

## 4. Optical Character Recognition (OCR) Pipelines

OCR extracts transcribed digital text from images containing typography or handwriting.
- **CRNN (Shi et al., 2015)**:
  $$\text{Image} \xrightarrow{\text{CNN}} \text{Feature Sequence} \xrightarrow{\text{BiLSTM}} \text{Frame Logits} \xrightarrow{\text{CTC Loss}} \text{Text}$$
- **TrOCR (Li et al., 2021)**:
  Modern encoder-decoder Transformer: a pre-trained Vision Transformer acts as the visual encoder, feeding a causal text Transformer decoder that autoregressively generates BPE text tokens.

---

## 5. Implementation Blueprint

- [`code/vit_engine.py`](code/vit_engine.py): Pure PyTorch implementations of `PatchEmbedding`, `TransformerEncoderBlock`, `VisionTransformerMini`, `compute_image_similarity`, and `extract_l2_normalized_embedding`.
- [`code/test_vision_transformers.py`](code/test_vision_transformers.py): Unit test suite verifying patch projection dimensions, `[CLS]` token pooling, and cosine metric properties.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab demonstrating 2D-to-1D patch embeddings, ViT forward passes, and image embedding similarity comparisons.
