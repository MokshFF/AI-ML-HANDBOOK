"""
Computer Vision Image Processing: Filtering, Convolution, and Augmentation.

Implements from scratch using NumPy and PyTorch:
- normalize_image: Channel-wise mean/std normalization.
- gaussian_kernel & apply_conv2d: 2D spatial filtering.
- sobel_edge_detector: Horizontal/Vertical gradient filters and gradient magnitude.
- cutout: Random rectangular region masking.
- mixup: Linear convex combination of image pairs and targets.
- cutmix: Spatial rectangular patch pasting with area-proportional label blending.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def normalize_image(image: np.ndarray, mean: Tuple[float, ...] = (0.485, 0.456, 0.406), std: Tuple[float, ...] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalizes an image array of shape (H, W, C) with values in [0, 255] or [0, 1]
    using channel-wise mean and standard deviation.
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    return (img - mean_arr) / std_arr


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Generates a normalized 2D Gaussian filter kernel of dimensions (size, size).
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer.")

    radius = size // 2
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def apply_conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies 2D spatial convolution on a single-channel image (H, W) or multi-channel image (H, W, C).
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    if image.ndim == 2:
        h, w = image.shape
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        output = np.zeros((h, w), dtype=np.float32)
        # Flip kernel for formal mathematical convolution
        k_flipped = np.flip(kernel)
        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_h, j:j + k_w]
                output[i, j] = np.sum(region * k_flipped)
        return output
    elif image.ndim == 3:
        h, w, c = image.shape
        channels = [apply_conv2d(image[:, :, ch], kernel) for ch in range(c)]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError("Image must be 2D (H, W) or 3D (H, W, C).")


def sobel_edge_detector(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies 3x3 Sobel filters to compute horizontal gradient G_x, vertical gradient G_y,
    and gradient magnitude G = sqrt(G_x^2 + G_y^2).
    """
    gray = image
    if image.ndim == 3:
        # Standard luminance conversion: 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    gx = apply_conv2d(gray, sobel_x)
    gy = apply_conv2d(gray, sobel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return gx, gy, magnitude


def cutout(image: np.ndarray, mask_size: int = 16) -> np.ndarray:
    """
    Applies Cutout data augmentation: randomly erases a square region of size mask_size x mask_size.
    """
    out = image.copy()
    h, w = out.shape[:2]

    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)

    half = mask_size // 2
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)

    out[y1:y2, x1:x2] = 0
    return out


def mixup(
    img1: torch.Tensor,
    img2: torch.Tensor,
    label1: torch.Tensor,
    label2: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup data augmentation (Zhang et al., 2017).
    x_mix = lambda * x1 + (1 - lambda) * x2
    y_mix = lambda * y1 + (1 - lambda) * y2
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    mixed_x = lam * img1 + (1.0 - lam) * img2
    mixed_y = lam * label1 + (1.0 - lam) * label2
    return mixed_x, mixed_y


def cutmix(
    img1: torch.Tensor,
    img2: torch.Tensor,
    label1: torch.Tensor,
    label2: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix data augmentation (Yun et al., 2019).
    Pastes a random patch from img2 onto img1.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    _, h, w = img1.shape  # Expects (C, H, W)

    # Box dimensions proportional to sqrt(1 - lambda)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_img = img1.clone()
    mixed_img[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]

    # Adjusted lambda based on actual patch area
    area_ratio = float((bbx2 - bbx1) * (bby2 - bby1)) / (h * w)
    adjusted_lam = 1.0 - area_ratio
    mixed_label = adjusted_lam * label1 + (1.0 - adjusted_lam) * label2

    return mixed_img, mixed_label
