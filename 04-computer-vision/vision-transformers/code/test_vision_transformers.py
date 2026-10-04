"""
Unit tests for Vision Transformer (ViT) & Image Embedding Engine.
"""

import pytest
import torch
from vit_engine import (
    PatchEmbedding,
    VisionTransformerMini,
    compute_image_similarity,
    extract_l2_normalized_embedding,
)


def test_patch_embedding_shapes():
    patch_embed = PatchEmbedding(in_channels=3, patch_size=4, embed_dim=32)
    # Image: (B=2, C=3, H=16, W=16) -> (16/4)^2 = 16 patches
    x = torch.randn(2, 3, 16, 16)
    out = patch_embed(x)
    assert out.shape == (2, 16, 32)


def test_vision_transformer_forward_and_loss():
    torch.manual_seed(42)
    model = VisionTransformerMini(
        img_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=5,
        embed_dim=32,
        depth=2,
        num_heads=4,
    )
    x = torch.randn(3, 3, 16, 16)
    targets = torch.tensor([0, 2, 4])

    out = model(x, targets=targets)
    assert out["logits"].shape == (3, 5)
    assert out["cls_embedding"].shape == (3, 32)
    assert out["patch_tokens"].shape == (3, 16, 32)
    assert out["loss"] is not None
    assert out["loss"].item() > 0


def test_image_similarity_and_retrieval():
    # Construct identical vectors and orthogonal vectors
    emb1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    emb2 = torch.tensor([[1.0, 0.0], [0.0, -1.0]])

    sim = compute_image_similarity(emb1, emb2)
    assert sim.shape == (2, 2)
    # emb1[0] and emb2[0] are identical -> similarity 1.0
    assert pytest.approx(sim[0, 0].item(), 1e-4) == 1.0
    # emb1[0] and emb2[1] are orthogonal -> similarity 0.0
    assert pytest.approx(sim[0, 1].item(), 1e-4) == 0.0
    # emb1[1] and emb2[1] are opposite -> similarity -1.0
    assert pytest.approx(sim[1, 1].item(), 1e-4) == -1.0


def test_extract_l2_normalized_embedding():
    model = VisionTransformerMini(img_size=16, patch_size=4, in_channels=3, num_classes=3, embed_dim=16, depth=1)
    img = torch.randn(1, 3, 16, 16)
    emb = extract_l2_normalized_embedding(model, img)
    assert emb.shape == (1, 16)
    # Norm should be exactly 1.0
    assert pytest.approx(torch.norm(emb, p=2, dim=-1).item(), 1e-4) == 1.0
