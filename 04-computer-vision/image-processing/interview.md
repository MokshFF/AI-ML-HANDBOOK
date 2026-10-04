# Image Processing & Augmentation - Technical Interview Preparation

A curated question bank covering digital image representations, spatial filtering, mathematical morphology, and modern data augmentation techniques.

---

## 1. Core Mathematical & Conceptual Questions

### Q1: Why are Sobel filters mathematically preferred over simple central finite differences for edge detection?
- **Central Difference Formula**:
  $$\frac{\partial I}{\partial x} \approx \frac{I(x+1, y) - I(x-1, y)}{2}$$
- **Limitations of Simple Difference**:
  In real-world images, sensor noise produces high-frequency pixel variations. A simple finite difference amplifies this high-frequency noise, creating spurious false-positive edges.
- **Sobel's Separable Design**:
  The 3x3 Sobel operator is mathematically separable into a smoothing filter along one axis and a central difference derivative along the orthogonal axis:
  $$\mathbf{S}_x = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} * \begin{bmatrix} -1 & 0 & +1 \end{bmatrix}$$
  The column vector $[1, 2, 1]^\top$ performs triangular Gaussian-like smoothing in the vertical direction, suppressing high-frequency noise while computing the horizontal spatial derivative.

---

### Q2: What is the Vicinal Risk Minimization (VRM) principle, and how does Mixup implement it?
- **Empirical Risk Minimization (ERM)**:
  Standard training minimizes empirical loss over discrete delta distributions centered strictly on training samples:
  $$P_\delta(x, y) = \frac{1}{N} \sum_{i=1}^N \delta(x = x_i, y = y_i)$$
  This encourages neural networks to memorize samples and output over-confident predictions near training points, leading to fragility under slight adversarial perturbations.
- **Vicinal Risk Minimization (VRM)**:
  VRM replaces discrete delta points with continuous probability distributions in the neighborhood (vicinity) of each sample. Mixup defines the vicinity as the linear interpolation line between any two randomly drawn pairs:
  $$\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda) \mathbf{x}_j, \quad \tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda) \mathbf{y}_j$$
  This enforces a linear inductive bias: predictions between two classes change smoothly rather than exhibiting abrupt decision boundaries.

---

### Q3: Why does CutMix outperform Cutout and Mixup on classification and object localization benchmarks?
1. **No Blank Pixel Information Loss (vs. Cutout)**:
   Cutout drops entire regions to zero, wasting model processing capacity on completely non-informative synthetic black pixels. CutMix replaces dropped regions with informative pixels from another image, preserving $100\%$ pixel utilization.
2. **Natural Image Statistics (vs. Mixup)**:
   Mixup produces unnatural, translucent "ghost" images with overlapping edges that do not exist in the natural visual world. CutMix preserves locally coherent natural texture, geometry, and sharp boundary statistics.
3. **Implicit Localization Signal**:
   Because labels are proportional to area, the model learns not only which class is present, but where features are localized across the visual field.

---

## 2. Whiteboard Coding Drills

### Q4: Implement vectorized CutMix augmentation in PyTorch for an image batch.
```python
import numpy as np
import torch

def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Args:
        x: (B, C, H, W) image batch tensor
        y: (B, num_classes) one-hot label tensor
    """
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.shape
    index = torch.randperm(batch_size)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x_mixed = x.clone()
    x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exact box area ratio
    actual_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
    y_mixed = actual_lam * y + (1.0 - actual_lam) * y[index]

    return x_mixed, y_mixed
```
