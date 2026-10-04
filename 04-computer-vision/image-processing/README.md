# Image Processing, Spatial Filtering & Modern Augmentations

A rigorous guide to image representations, 2D spatial convolution filtering (Gaussian blur, Sobel edge detectors), and advanced data augmentation techniques (Cutout, Mixup, CutMix) for deep vision models.

---

## 1. Digital Image Representation & Channel Normalization

A digital color image is represented as a 3D rank-3 tensor $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ (or in PyTorch batch format $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$), where $H$ is height, $W$ is width, and $C \in \{1, 3\}$ corresponds to grayscale or RGB channels.

### Channel-Wise Standardization
To ensure numerical stability and consistent gradient scale across features, images are normalized using population channel statistics:
$$\mathbf{X}_{c,\text{norm}} = \frac{\mathbf{X}_c / 255.0 - \mu_c}{\sigma_c}$$
For standard ImageNet pre-trained backbones:
- $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$
- $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$

---

## 2. 2D Spatial Convolutions & Filtering

The discrete 2D convolution of an image $I$ with a kernel $K \in \mathbb{R}^{(2k+1) \times (2k+1)}$ is given by:
$$(I * K)(i, j) = \sum_{u=-k}^k \sum_{v=-k}^k I(i - u, j - v) K(u, v)$$

### 2.1 Gaussian Smoothing (Low-Pass Filter)
Smooths high-frequency noise using an isotropic 2D Gaussian distribution:
$$K(u, v) = \frac{1}{2\pi \sigma^2} \exp\left( -\frac{u^2 + v^2}{2\sigma^2} \right)$$
Normalized such that $\sum_{u, v} K(u, v) = 1$.

### 2.2 Sobel Edge Detection (Spatial Derivative Approximation)
Sobel operators estimate directional intensity gradients using separable discrete difference kernels:
$$\mathbf{S}_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad \mathbf{S}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$
- **Horizontal & Vertical Gradients**: $G_x = I * \mathbf{S}_x, \; G_y = I * \mathbf{S}_y$
- **Gradient Magnitude**: $G = \sqrt{G_x^2 + G_y^2}$
- **Gradient Orientation**: $\theta = \arctan2(G_y, G_x)$

---

## 3. Modern Regularization via Data Augmentation

Data augmentation acts as an implicit data-dependent prior that regularizes over-parameterized neural networks.

| Technique | Mathematical Formulation | Mechanism / Intuition |
| :--- | :--- | :--- |
| **Cutout** (DeVries & Taylor, 2017) | Set square region $R$ to zero: $\mathbf{x}_{\text{aug}} = \mathbf{x} \odot \mathbf{M}$ | Prevents CNNs from over-relying on a single dominant local visual cue (e.g. eyes). |
| **Mixup** (Zhang et al., 2017) | $\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda) \mathbf{x}_j$<br>$\tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda) \mathbf{y}_j$<br>$\lambda \sim \text{Beta}(\alpha, \alpha)$ | Enforces linear behavior in between training examples, smoothing decision boundaries. |
| **CutMix** (Yun et al., 2019) | $\tilde{\mathbf{x}} = \mathbf{M} \odot \mathbf{x}_i + (\mathbf{1} - \mathbf{M}) \odot \mathbf{x}_j$<br>$\tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda) \mathbf{y}_j$<br>$\lambda = 1 - \frac{\text{Area}(R)}{H \times W}$ | Combines spatial localization benefit of Cutout with label mixing of Mixup without blank pixel artifacts. |

---

## 4. Implementation Blueprint

- [`code/transforms_filters.py`](code/transforms_filters.py): Pure Python/NumPy implementations of `normalize_image`, `gaussian_kernel`, `apply_conv2d`, `sobel_edge_detector`, `cutout`, `mixup`, and `cutmix`.
- [`code/test_image_processing.py`](code/test_image_processing.py): Pytest unit test suite verifying filter shapes, normalization ranges, and label interpolation.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab exploring image transformations and visual augmentations.
