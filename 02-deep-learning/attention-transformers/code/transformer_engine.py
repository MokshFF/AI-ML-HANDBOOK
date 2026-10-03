"""
Attention & Transformer Architecture from scratch in PyTorch.
Implements:
1. Scaled Dot-Product Attention with optional causal/future masking.
2. Multi-Head Attention (MHA).
3. Sinusoidal Positional Encoding.
4. Pre-LN Transformer Block (Self-Attention + MLP).
5. Mini Autoregressive Transformer LM (GPT-style).
6. Vision Transformer (ViT) Patch Embedding.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# 1. Scaled Dot-Product Attention
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + Mask) @ V
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k, v: (batch_size, num_heads, seq_len, head_dim)
        mask: boolean or additive mask broadcastable to (batch_size, num_heads, seq_len, seq_len)
        Returns: (output, attention_weights)
        """
        d_k = q.size(-1)
        # Scaled dot-product scores: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Mask out forbidden positions with large negative value
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


# ============================================================================
# 2. Multi-Head Attention (MHA)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention projects Q, K, V into h subspaces:
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) @ W_O
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = q.size(0)

        # 1. Linear projections and reshape to (B, H, S, d_k)
        q_proj = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        out, _ = self.attention(q_proj, k_proj, v_proj, mask=mask)

        # 3. Concatenate heads and project output: (B, S, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)


# ============================================================================
# 3. Positional Encodings
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Vaswani et al. (2017) Sinusoidal Positional Encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding up to sequence length of x."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ============================================================================
# 4. Pre-LN Transformer Block & GPT-Style Language Model
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block (standard in GPT, LLaMA, Modern Transformers).
    x = x + MHA(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    """
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x), mask=mask)
        # Pre-LN MLP
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    """
    Autoregressive Decoder-Only Transformer (GPT Architecture).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017)
        self.lm_head.weight = self.token_embed.weight

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper triangular mask to prevent attending to future tokens."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) token IDs
        Returns: logits (batch_size, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        causal_mask = self.generate_causal_mask(seq_len, x.device)

        h = self.dropout(self.pos_embed(self.token_embed(x) * math.sqrt(self.d_model)))
        for block in self.blocks:
            h = block(h, mask=causal_mask)
            
        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits


# ============================================================================
# 5. Vision Transformer (ViT) Patch Embedding
# ============================================================================

class ViTPatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and linearly projects each patch to d_model:
    (B, C, H, W) -> (B, NumPatches, d_model)
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, d_model: int = 64):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 2D conv with kernel_size = stride = patch_size implements patch projection
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: (B, num_patches, d_model)
        """
        patches = self.proj(x)  # (B, d_model, H/P, W/P)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        return patches
