"""
Convolutional Neural Networks (CNN) from scratch and in PyTorch.
Implements:
1. Analytical 2D Convolution and Pooling operations.
2. Receptive Field Tracker.
3. LeNet-5, VGG Block, Residual Block (ResNet), and Inverted Residual MBConv (EfficientNet).
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


# ============================================================================
# 1. Convolution Operations from Scratch
# ============================================================================

def conv2d_forward_scratch(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    2D convolution forward pass from scratch without high-level libraries.
    X: (N, C_in, H, W)
    W: (C_out, C_in, K_h, K_w)
    b: (C_out,)
    stride: step size
    padding: zero-padding applied to both spatial dimensions
    Returns: output (N, C_out, H_out, W_out)
    """
    N, C_in, H_in, W_in = X.shape
    C_out, _, K_h, K_w = W.shape
    
    # Calculate output dimensions
    H_out = (H_in - K_h + 2 * padding) // stride + 1
    W_out = (W_in - K_w + 2 * padding) // stride + 1
    
    # Apply zero-padding
    if padding > 0:
        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0
        )
    else:
        X_padded = X
        
    out = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)
    
    # Slide kernel over spatial dimensions
    for n in range(N):
        for c_out in range(C_out):
            kernel = W[c_out]  # (C_in, K_h, K_w)
            bias = b[c_out]
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + K_h
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + K_w
                    window = X_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, c_out, i, j] = np.sum(window * kernel) + bias
                    
    return out


def maxpool2d_forward_scratch(
    X: np.ndarray,
    pool_size: int = 2,
    stride: int = 2
) -> np.ndarray:
    """
    Max pooling 2D forward pass from scratch.
    X: (N, C, H, W)
    Returns: (N, C, H_out, W_out)
    """
    N, C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    out = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + pool_size
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + pool_size
                    out[n, c, i, j] = np.max(X[n, c, h_start:h_end, w_start:w_end])
    return out


# ============================================================================
# 2. Receptive Field Calculator
# ============================================================================

def compute_receptive_field(layers: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    """
    Computes effective receptive field (RF) and feature jump across sequential layers.
    Each layer dict contains {'k': kernel_size, 's': stride, 'p': padding}.
    Formula:
      RF_0 = 1, Jump_0 = 1
      RF_l = RF_{l-1} + (k_l - 1) * Jump_{l-1}
      Jump_l = Jump_{l-1} * s_l
    """
    rf = 1
    jump = 1
    history = []
    
    for idx, l in enumerate(layers):
        k = l.get("k", 1)
        s = l.get("s", 1)
        rf = rf + (k - 1) * jump
        jump = jump * s
        history.append({
            "layer_idx": idx + 1,
            "kernel": k,
            "stride": s,
            "receptive_field": rf,
            "jump": jump
        })
    return history


# ============================================================================
# 3. Modern PyTorch CNN Architectures
# ============================================================================

class LeNet5(nn.Module):
    """
    Classic LeNet-5 Architecture (LeCun et al., 1998)
    Designed for 32x32 (or 28x28 padded) single-channel images.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),  # 32x32 -> 32x32
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                          # 32x32 -> 16x16
            nn.Conv2d(6, 16, kernel_size=5, stride=1),                      # 16x16 -> 12x12
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)                           # 12x12 -> 6x6
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat_flat = torch.flatten(feat, 1)
        return self.classifier(feat_flat)


class ResidualBlock(nn.Module):
    """
    He et al. (2015) ResNet BasicBlock with identity / projection shortcut.
    F(x) + x resolves the degradation problem in deep networks.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class MiniResNet(nn.Module):
    """
    Compact ResNet demonstrating staged residual downsampling.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ResidualBlock(16, 16, stride=1)
        self.stage2 = ResidualBlock(16, 32, stride=2)
        self.stage3 = ResidualBlock(32, 64, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block (MBConv) from MobileNetV2 / EfficientNet.
    Features: Expansion (1x1) -> Depthwise Conv (3x3) -> Squeeze-and-Excitation (SE) -> Projection (1x1).
    """
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4, stride: int = 1, se_ratio: float = 0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers: List[nn.Module] = []
        # 1. 1x1 Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # 2. 3x3 Depthwise Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        # 3. Squeeze-and-Excitation (SE) Block
        se_dim = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_dim, hidden_dim, 1),
            nn.Sigmoid()
        )

        # 4. 1x1 Linear Pointwise Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat * self.se(feat)
        out = self.project(feat)
        if self.use_residual:
            return out + x
        return out
