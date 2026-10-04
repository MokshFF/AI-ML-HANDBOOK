"""
Semantic Segmentation Engine: Mini U-Net, Dice Loss, and Mean IoU Evaluation.

Implements from scratch using pure PyTorch:
- DoubleConv: Fundamental 2-layer convolution building block with BatchNorm.
- UNetMini: Classic encoder-decoder architecture with contracting path, expanding path, and skip connections.
- DiceLoss & CombinedLoss: Soft Dice loss and combined BCE/Dice loss for segmentation imbalance.
- compute_mean_iou: Pixel-level confusion matrix and per-class / mean Intersection over Union (mIoU).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetMini(nn.Module):
    """
    Lightweight U-Net architecture for educational semantic segmentation.
    Includes contracting path (encoder), bottleneck, expanding path (decoder), and skip connections.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_features: int = 16):
        super().__init__()
        f = base_features

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))

        # Expanding Path (Decoder) + Skip Connections
        self.up1 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 4, f * 2)

        self.up3 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 2, f)

        # Output head: 1x1 conv projecting to num_classes
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)         # (B, f, H, W)
        x2 = self.down1(x1)      # (B, 2f, H/2, W/2)
        x3 = self.down2(x2)      # (B, 4f, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(x3)  # (B, 8f, H/8, W/8)

        # Decoder with skip connection concatenation
        d1 = self.up1(b)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv_up3(d3)

        logits = self.outc(d3)
        return logits


class DiceLoss(nn.Module):
    """
    Soft Dice Loss:
    L_dice = 1 - (2 * sum(p * g) + eps) / (sum(p^2) + sum(g^2) + eps)
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw unnormalized scores
            targets: (B, H, W) integer ground-truth class labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets to (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs * probs + targets_one_hot * targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Average loss across classes
        dice_loss = 1.0 - torch.mean(dice_score)
        return dice_loss


def compute_mean_iou(pred_masks: torch.Tensor, target_masks: torch.Tensor, num_classes: int) -> Tuple[float, torch.Tensor]:
    """
    Computes per-class and mean Intersection over Union (mIoU).
    Args:
        pred_masks: (B, H, W) predicted class indices
        target_masks: (B, H, W) ground-truth class indices
    Returns:
        mean_iou: Mean IoU across all valid classes
        class_ious: Tensor of length num_classes
    """
    class_ious = []
    for c in range(num_classes):
        pred_c = (pred_masks == c)
        target_c = (target_masks == c)

        intersection = (pred_c & target_c).sum().float().item()
        union = (pred_c | target_c).sum().float().item()

        if union == 0:
            # Class not present in ground truth or predictions; omit or set to 1.0
            class_ious.append(float("nan"))
        else:
            class_ious.append(intersection / union)

    valid_ious = [iou for iou in class_ious if not torch.isnan(torch.tensor(iou))]
    mean_iou = sum(valid_ious) / max(1, len(valid_ious))
    return mean_iou, torch.tensor(class_ious)
