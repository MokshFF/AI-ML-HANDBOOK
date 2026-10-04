"""
Unit tests for Semantic Segmentation Engine: U-Net, Dice Loss, and mIoU.
"""

import pytest
import torch
from segmentation_engine import (
    DoubleConv,
    UNetMini,
    DiceLoss,
    compute_mean_iou,
)


def test_double_conv():
    conv = DoubleConv(in_channels=4, out_channels=8)
    x = torch.randn(2, 4, 16, 16)
    out = conv(x)
    assert out.shape == (2, 8, 16, 16)


def test_unet_mini_forward():
    torch.manual_seed(42)
    model = UNetMini(in_channels=3, num_classes=3, base_features=8)
    # Spatial dimensions must be divisible by 8 (3 downsamplings)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 3, 32, 32)


def test_dice_loss():
    loss_fn = DiceLoss()
    # High confidence correct prediction -> Dice Loss should be close to 0
    logits = torch.tensor([
        [[[10.0, -10.0], [-10.0, 10.0]], [[-10.0, 10.0], [10.0, -10.0]]]
    ])  # (B=1, C=2, H=2, W=2)
    targets = torch.tensor([[[0, 1], [1, 0]]])  # Matches argmax exactly

    loss = loss_fn(logits, targets)
    assert 0.0 <= loss.item() < 0.1


def test_compute_mean_iou():
    preds = torch.tensor([
        [[0, 0], [1, 1]]
    ])
    targets = torch.tensor([
        [[0, 1], [1, 1]]
    ])
    # Class 0: pred has (0,0), target has (0,0). intersection=1, union=2 -> IoU=0.5
    # Class 1: pred has (1,0),(1,1), target has (0,1),(1,0),(1,1). intersection=2, union=3 -> IoU=2/3
    miou, class_ious = compute_mean_iou(preds, targets, num_classes=2)
    assert pytest.approx(class_ious[0].item(), 0.01) == 0.5
    assert pytest.approx(class_ious[1].item(), 0.01) == 0.6667
    assert pytest.approx(miou, 0.01) == (0.5 + 2/3) / 2
