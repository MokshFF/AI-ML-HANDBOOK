"""
Computer Vision Image Classification & Transfer Learning.

Implements:
- ResidualBlockMini: Residual convolution block with identity shortcut.
- ImageClassifier: Lightweight ResNet-style ConvNet with feature extractor and classification head.
- TransferLearningWrapper: Linear probing and fine-tuning manager with layer freezing/unfreezing.
- compute_accuracy_topk: Top-1 and Top-k accuracy metrics.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockMini(nn.Module):
    """
    Standard 2-layer residual block with batch normalization and skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ImageClassifier(nn.Module):
    """
    ResNet-style convolutional image classifier with global average pooling.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_dim: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ResidualBlockMini(base_dim, base_dim, stride=1)
        self.stage2 = ResidualBlockMini(base_dim, base_dim * 2, stride=2)
        self.stage3 = ResidualBlockMini(base_dim * 2, base_dim * 4, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = base_dim * 4
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled feature embedding vector before the classification layer.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.classifier(features)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return {"logits": logits, "features": features, "loss": loss}


class TransferLearningWrapper(nn.Module):
    """
    Wraps an existing backbone for transfer learning:
    supports backbone freezing (linear probe) and selective unfreezing (fine-tuning).
    """
    def __init__(self, backbone: ImageClassifier, target_classes: int):
        super().__init__()
        self.backbone = backbone
        # Replace classifier head with target classes
        self.new_head = nn.Linear(backbone.embedding_dim, target_classes)

    def freeze_backbone(self):
        """
        Freezes all backbone feature extractor layers (linear probing mode).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.new_head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreezes all layers for full fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.backbone.forward_features(x)
        logits = self.new_head(features)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return {"logits": logits, "features": features, "loss": loss}


def compute_accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Computes top-k accuracy percentages for the specified values of k.
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return res
