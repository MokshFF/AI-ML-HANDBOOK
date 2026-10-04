"""
Audio Classification & Speaker Verification Engine.

Implements from scratch using pure PyTorch:
- SpectrogramCNNClassifier: 2D Convolutional neural network processing log-Mel spectrograms.
- SpeakerEmbeddingModel: Extracts L2-normalized d-vectors / x-vectors for speaker verification.
- compute_triplet_loss: Margin-based metric learning for audio embeddings.
- evaluate_verification: Computes False Acceptance Rate (FAR) and False Rejection Rate (FRR).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramConvBlock(nn.Module):
    """
    2D Convolution block with BatchNorm, ReLU, and optional MaxPool.
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramCNNClassifier(nn.Module):
    """
    2D CNN for acoustic scene / audio event classification.
    Accepts log-Mel spectrograms of shape (B, 1, num_mels, time_frames).
    """
    def __init__(self, num_classes: int = 5, base_dim: int = 16, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = SpectrogramConvBlock(1, base_dim, pool=True)
        self.conv2 = SpectrogramConvBlock(base_dim, base_dim * 2, pool=True)
        self.conv3 = SpectrogramConvBlock(base_dim * 2, embedding_dim, pool=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embedding_dim, num_classes)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled latent representation vector (d-vector).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        emb = self.extract_embedding(x)
        logits = self.fc(emb)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return {"logits": logits, "embedding": emb, "loss": loss}


class SpeakerEmbeddingModel(nn.Module):
    """
    Extracts L2-normalized speaker representations on the unit sphere for verification.
    """
    def __init__(self, backbone: SpectrogramCNNClassifier):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone.extract_embedding(x)
        return F.normalize(emb, p=2, dim=-1)


def compute_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Computes Triplet Margin Loss:
    L(a, p, n) = max(0, ||a - p||_2 - ||a - n||_2 + margin)
    """
    dist_pos = torch.norm(anchor - positive, p=2, dim=-1)
    dist_neg = torch.norm(anchor - negative, p=2, dim=-1)
    losses = F.relu(dist_pos - dist_neg + margin)
    return torch.mean(losses)


def evaluate_verification(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates binary speaker verification decisions against ground-truth pairs (1 = same speaker, 0 = different).
    Computes False Acceptance Rate (FAR), False Rejection Rate (FRR), and Overall Accuracy.
    """
    preds = (scores >= threshold).long()
    same_speaker = (labels == 1)
    diff_speaker = (labels == 0)

    # False Rejection: Same speaker classified as different (score < threshold)
    num_same = same_speaker.sum().item()
    false_rejects = ((preds == 0) & same_speaker).sum().item()
    frr = false_rejects / max(1, num_same)

    # False Acceptance: Different speaker classified as same (score >= threshold)
    num_diff = diff_speaker.sum().item()
    false_accepts = ((preds == 1) & diff_speaker).sum().item()
    far = false_accepts / max(1, num_diff)

    accuracy = (preds == labels).float().mean().item()
    return {"FAR": far, "FRR": frr, "accuracy": accuracy}
