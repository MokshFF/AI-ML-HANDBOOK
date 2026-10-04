"""
Unit tests for Computer Vision Image Processing & Augmentation.
"""

import numpy as np
import pytest
import torch
from transforms_filters import (
    normalize_image,
    gaussian_kernel,
    apply_conv2d,
    sobel_edge_detector,
    cutout,
    mixup,
    cutmix,
)


def test_normalize_image():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    norm = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert norm.shape == (10, 10, 3)
    # (1.0 - 0.5) / 0.5 = 1.0
    assert np.allclose(norm, 1.0)


def test_gaussian_and_conv2d():
    kernel = gaussian_kernel(size=3, sigma=1.0)
    assert kernel.shape == (3, 3)
    assert np.isclose(kernel.sum(), 1.0)

    # Test filtering on a uniform image
    uniform_img = np.ones((8, 8), dtype=np.float32)
    filtered = apply_conv2d(uniform_img, kernel)
    # Inner region should remain 1.0
    assert np.allclose(filtered[2:-2, 2:-2], 1.0, atol=1e-5)


def test_sobel_edge_detector():
    # Vertical edge step
    step_img = np.zeros((10, 10), dtype=np.float32)
    step_img[:, 5:] = 10.0

    gx, gy, mag = sobel_edge_detector(step_img)
    assert gx.shape == (10, 10)
    assert gy.shape == (10, 10)
    assert mag.shape == (10, 10)
    # Strong horizontal gradient at column 5
    assert mag[:, 5].mean() > mag[:, 1].mean()


def test_cutout():
    np.random.seed(42)
    img = np.ones((32, 32, 3), dtype=np.float32)
    out = cutout(img, mask_size=8)
    assert out.shape == (32, 32, 3)
    assert (out == 0).sum() > 0


def test_mixup_and_cutmix():
    torch.manual_seed(42)
    img1 = torch.ones((3, 16, 16))
    img2 = torch.zeros((3, 16, 16))
    y1 = torch.tensor([1.0, 0.0])
    y2 = torch.tensor([0.0, 1.0])

    # Mixup
    mix_x, mix_y = mixup(img1, img2, y1, y2, alpha=0.5)
    assert mix_x.shape == (3, 16, 16)
    assert mix_y.shape == (2,)
    assert torch.isclose(mix_y.sum(), torch.tensor(1.0))

    # Cutmix
    cm_x, cm_y = cutmix(img1, img2, y1, y2, alpha=1.0)
    assert cm_x.shape == (3, 16, 16)
    assert cm_y.shape == (2,)
    assert torch.isclose(cm_y.sum(), torch.tensor(1.0))
