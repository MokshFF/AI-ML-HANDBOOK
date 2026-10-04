"""
Unit tests for Image Classification, Transfer Learning, and Evaluation Metrics.
"""

import pytest
import torch
from classifier import (
    ResidualBlockMini,
    ImageClassifier,
    TransferLearningWrapper,
    compute_accuracy_topk,
)


def test_residual_block_shapes():
    x = torch.randn(2, 16, 28, 28)
    # Identity shortcut
    block_same = ResidualBlockMini(16, 16, stride=1)
    out_same = block_same(x)
    assert out_same.shape == (2, 16, 28, 28)

    # Downsampling shortcut
    block_down = ResidualBlockMini(16, 32, stride=2)
    out_down = block_down(x)
    assert out_down.shape == (2, 32, 14, 14)


def test_image_classifier_forward_and_features():
    torch.manual_seed(42)
    model = ImageClassifier(in_channels=3, num_classes=5, base_dim=16)
    x = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 2, 4, 1])

    out = model(x, targets=targets)
    assert out["logits"].shape == (4, 5)
    assert out["features"].shape == (4, 16 * 4)
    assert out["loss"] is not None
    assert out["loss"].item() > 0


def test_transfer_learning_freezing_and_unfreezing():
    torch.manual_seed(42)
    backbone = ImageClassifier(in_channels=3, num_classes=10, base_dim=16)
    tl_model = TransferLearningWrapper(backbone, target_classes=2)

    # Freeze backbone (linear probe)
    tl_model.freeze_backbone()
    for param in backbone.parameters():
        assert not param.requires_grad
    for param in tl_model.new_head.parameters():
        assert param.requires_grad

    # Forward pass and backward step
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    out = tl_model(x, targets=y)
    out["loss"].backward()

    # Head parameters must have gradients, backbone must not
    assert tl_model.new_head.weight.grad is not None
    assert backbone.stem[0].weight.grad is None

    # Unfreeze
    tl_model.unfreeze_backbone()
    for param in backbone.parameters():
        assert param.requires_grad


def test_topk_accuracy():
    # Model predictions: batch of 3 samples, 4 classes
    logits = torch.tensor([
        [10.0, 2.0, 1.0, 0.0],  # Pred class 0
        [1.0, 8.0, 3.0, 0.0],   # Pred class 1
        [0.0, 2.0, 5.0, 6.0],   # Pred class 3 (class 2 is 2nd highest)
    ])
    # Targets: [0, 1, 2] -> 2 correct top-1, 3 correct top-2
    targets = torch.tensor([0, 1, 2])

    top1, top2 = compute_accuracy_topk(logits, targets, topk=(1, 2))
    assert pytest.approx(top1, 0.1) == 66.67
    assert pytest.approx(top2, 0.1) == 100.0
