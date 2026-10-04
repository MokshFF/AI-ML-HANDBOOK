"""
Vision Transformer (ViT) & Image Embedding Engine.

Implements from scratch using pure PyTorch:
- PatchEmbedding: Splits 2D images into patches and linearly projects to embedding dimension.
- VisionTransformerMini: Complete ViT architecture with [CLS] token, learnable position embeddings,
  Transformer encoder layers, and classification head.
- compute_image_similarity: Cosine similarity and Euclidean distance for image embeddings.
- extract_image_embeddings: Utility to compute L2-normalized image representations.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Slices image into non-overlapping patches of size (P x P) and projects into embedding dimension D.
    Implemented efficiently using Conv2d with kernel_size = stride = patch_size.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 4, embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        # Flatten spatial dimensions into token sequence: (B, D, N) -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Block for Vision Transformers.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerMini(nn.Module):
    """
    Lightweight Vision Transformer (ViT) model.
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")

        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 1. Patch projection
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)

        # 2. [CLS] token and Position Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # 3. Stack of Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. MLP Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize [CLS] token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts representations:
        Returns:
            cls_embedding: (B, embed_dim) - global pooled visual representation
            patch_tokens: (B, num_patches, embed_dim) - local patch tokens
        """
        b = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, N+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_embedding = x[:, 0]
        patch_tokens = x[:, 1:]
        return cls_embedding, patch_tokens

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cls_embedding, patch_tokens = self.forward_features(x)
        logits = self.head(cls_embedding)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return {
            "logits": logits,
            "cls_embedding": cls_embedding,
            "patch_tokens": patch_tokens,
            "loss": loss,
        }


def compute_image_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise cosine similarity between two sets of image embeddings.
    Args:
        emb1: (N, D)
        emb2: (M, D)
    Returns:
        (N, M) matrix of cosine similarities in [-1, 1].
    """
    norm1 = F.normalize(emb1, p=2, dim=-1)
    norm2 = F.normalize(emb2, p=2, dim=-1)
    return torch.matmul(norm1, norm2.t())


def extract_l2_normalized_embedding(model: VisionTransformerMini, image: torch.Tensor) -> torch.Tensor:
    """
    Extracts L2-normalized unit sphere embedding for retrieval or vector search.
    """
    model.eval()
    with torch.no_grad():
        cls_emb, _ = model.forward_features(image)
        normalized = F.normalize(cls_emb, p=2, dim=-1)
    return normalized
