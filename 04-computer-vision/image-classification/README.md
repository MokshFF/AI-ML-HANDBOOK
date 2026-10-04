# Image Classification & Transfer Learning

A comprehensive guide to deep convolutional image classification, residual representations, feature extraction, linear probing vs. fine-tuning, and multi-class ranking metrics.

---

## 1. Deep Convolutional Classification Framework

Image classification maps an input image tensor $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$ to a categorical probability distribution over $K$ discrete classes:
$$\mathbf{z} = f_\theta(\mathbf{X}), \quad \hat{\mathbf{y}} = \text{softmax}(\mathbf{z}) \in \Delta^{K-1}$$

### Modern Architectural Decomposition
1. **Stem**: Initial convolution ($7 \times 7$ or $3 \times 3$ with stride) + BatchNorm + Non-linearity, projecting high-resolution raw pixels into low-resolution, high-channel space.
2. **Body (Residual Stages)**: Successive stages of residual blocks ($F(\mathbf{x}) + \mathbf{x}$) that downsample spatial resolution while doubling channel capacity.
3. **Global Average Pooling (GAP)**: Replaces parameter-heavy fully connected layers by averaging each 2D feature map into a single scalar:
   $$\mathbf{h}_c = \frac{1}{H' W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \mathbf{F}_{c, i, j}$$
   This grants translational invariance and drastically reduces parameter count.
4. **Classification Head**: A single linear transformation $\mathbf{z} = \mathbf{W} \mathbf{h} + \mathbf{b}$ where $\mathbf{W} \in \mathbb{R}^{K \times d}$.

```
Input (3, H, W)
      │
   [Stem]  Conv2d + BN + ReLU
      │
 [Stage 1] Residual Blocks (Channels: C)
      │
 [Stage 2] Downsampling Stride=2 (Channels: 2C)
      │
 [Stage 3] Downsampling Stride=2 (Channels: 4C)
      │
   [GAP]   Global Average Pooling (Channels: 4C) ──> Dense Embedding Vector h in R^d
      │
  [Linear] Classification Head (W in R^(K x d))
      │
 Output Logits z in R^K ──> Softmax Probabilities
```

---

## 2. Transfer Learning Paradigms

Training deep networks from scratch on small target domains causes severe overfitting. Transfer learning leverages representations learned on large-scale datasets (e.g., ImageNet-1k, ImageNet-22k).

### Linear Probing vs. Fine-Tuning
| Property | Linear Probing | Full Fine-Tuning |
| :--- | :--- | :--- |
| **Trainable Weights** | Head only ($\mathbf{W}_{\text{head}}, \mathbf{b}_{\text{head}}$); backbone frozen | All backbone + head weights |
| **Computational Cost** | Very low; feature vectors can be cached | Full forward + backward pass through deep network |
| **Risk of Overfitting** | Minimal (convex optimization) | High on small target datasets |
| **Representational Adaptation** | Fixed generic visual representations | Adapts low/mid-level filters to target domain |
| **Optimal Learning Rate** | Standard ($10^{-2} - 10^{-3}$) | Small / Discriminative ($10^{-5} - 10^{-4}$) |

---

## 3. Evaluation Metrics: Top-1 vs. Top-$k$ Accuracy

In large-scale multi-class classification with fine-grained or semantically overlapping categories (e.g. ImageNet 1,000 classes), Top-1 accuracy can be overly punitive.
- **Top-1 Accuracy**:
  $$\text{Acc}_1 = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i = \arg\max_k z_{i, k})$$
- **Top-$k$ Accuracy**:
  $$\text{Acc}_k = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in \text{top } k \text{ indices of } \mathbf{z}_i)$$

---

## 4. Implementation Blueprint

- [`code/classifier.py`](code/classifier.py): PyTorch implementations of `ResidualBlockMini`, `ImageClassifier` with GAP and feature extraction, `TransferLearningWrapper` (freezing/unfreezing logic), and `compute_accuracy_topk`.
- [`code/test_image_classification.py`](code/test_image_classification.py): Pytest unit tests for forward shapes, feature vector dimensions, gradient freezing, and top-$k$ logic.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab exploring representation extraction, transfer learning steps, and accuracy metrics.
