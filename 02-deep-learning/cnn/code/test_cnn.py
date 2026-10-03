import pytest
import numpy as np
import torch
import torch.nn.functional as F
from cnn_engine import (
    conv2d_forward_scratch,
    maxpool2d_forward_scratch,
    compute_receptive_field,
    LeNet5,
    ResidualBlock,
    MiniResNet,
    MBConvBlock
)


def test_conv2d_scratch_vs_pytorch():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Dimensions: Batch=2, InChannels=3, H=8, W=8
    X_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    # Kernels: OutChannels=4, InChannels=3, Kh=3, Kw=3
    W_np = np.random.randn(4, 3, 3, 3).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)
    
    stride = 2
    padding = 1
    
    # Scratch forward
    out_scratch = conv2d_forward_scratch(X_np, W_np, b_np, stride=stride, padding=padding)
    
    # PyTorch reference
    X_pt = torch.tensor(X_np)
    W_pt = torch.tensor(W_np)
    b_pt = torch.tensor(b_np)
    out_pt = F.conv2d(X_pt, W_pt, b_pt, stride=stride, padding=padding).numpy()
    
    assert out_scratch.shape == out_pt.shape
    assert np.allclose(out_scratch, out_pt, atol=1e-5)


def test_maxpool2d_scratch_vs_pytorch():
    X_np = np.random.randn(2, 2, 6, 6).astype(np.float32)
    out_scratch = maxpool2d_forward_scratch(X_np, pool_size=2, stride=2)
    
    X_pt = torch.tensor(X_np)
    out_pt = F.max_pool2d(X_pt, kernel_size=2, stride=2).numpy()
    
    assert out_scratch.shape == (2, 2, 3, 3)
    assert np.allclose(out_scratch, out_pt, atol=1e-5)


def test_receptive_field_formula():
    # Stack of three 3x3 convs with stride 1
    layers = [{'k': 3, 's': 1}, {'k': 3, 's': 1}, {'k': 3, 's': 1}]
    history = compute_receptive_field(layers)
    assert history[0]['receptive_field'] == 3
    assert history[1]['receptive_field'] == 5
    assert history[2]['receptive_field'] == 7

    # Layer with stride 2
    layers_strided = [{'k': 3, 's': 2}, {'k': 3, 's': 1}]
    hist_strided = compute_receptive_field(layers_strided)
    assert hist_strided[0]['receptive_field'] == 3
    # RF2 = 3 + (3 - 1) * 2 = 7
    assert hist_strided[1]['receptive_field'] == 7


def test_lenet5_forward():
    model = LeNet5(in_channels=1, num_classes=10)
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_resnet_and_residual_block():
    block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
    x = torch.randn(2, 16, 14, 14)
    out = block(x)
    assert out.shape == (2, 16, 14, 14)

    # Downsampling residual block
    block_down = ResidualBlock(in_channels=16, out_channels=32, stride=2)
    out_down = block_down(x)
    assert out_down.shape == (2, 32, 7, 7)

    # MiniResNet full network
    mini_resnet = MiniResNet(in_channels=3, num_classes=5)
    x_img = torch.randn(2, 3, 32, 32)
    logits = mini_resnet(x_img)
    assert logits.shape == (2, 5)


def test_mbconv_block():
    mb = MBConvBlock(in_channels=16, out_channels=16, expand_ratio=4, stride=1, se_ratio=0.25)
    x = torch.randn(2, 16, 28, 28)
    out = mb(x)
    assert out.shape == (2, 16, 28, 28)
