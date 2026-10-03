import pytest
import torch
from transformer_engine import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    MiniTransformerLM,
    ViTPatchEmbedding
)


def test_scaled_dot_product_attention_and_causal_mask():
    attn = ScaledDotProductAttention(dropout=0.0)
    # Batch=1, Heads=1, SeqLen=3, HeadDim=4
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    
    # Causal lower-triangular mask
    mask = torch.tril(torch.ones(3, 3)).unsqueeze(0).unsqueeze(0)
    out, weights = attn(q, k, v, mask=mask)

    assert out.shape == (1, 1, 3, 4)
    assert weights.shape == (1, 1, 3, 3)
    # Weights along last dimension must sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 3), atol=1e-5)
    # Position 0 must not attend to positions 1 or 2 (strictly upper triangle = 0)
    assert torch.isclose(weights[0, 0, 0, 1], torch.tensor(0.0))
    assert torch.isclose(weights[0, 0, 0, 2], torch.tensor(0.0))


def test_multi_head_attention():
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 6, 32)
    out = mha(x, x, x)
    assert out.shape == (2, 6, 32)


def test_sinusoidal_positional_encoding():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=100)
    x = torch.zeros(2, 10, 16)
    out = pe(x)
    assert out.shape == (2, 10, 16)
    # Verify non-zero positional offsets
    assert not torch.allclose(out, torch.zeros_like(out))


def test_transformer_block():
    block = TransformerBlock(d_model=32, num_heads=4, mlp_ratio=2, dropout=0.0)
    x = torch.randn(3, 8, 32)
    out = block(x)
    assert out.shape == (3, 8, 32)


def test_mini_transformer_lm():
    model = MiniTransformerLM(
        vocab_size=120,
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_len=64,
        dropout=0.1
    )
    tokens = torch.randint(0, 120, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, 120)

    # Autoregressive loss computation check
    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.randint(0, 120, (2, 12))
    loss = criterion(logits.view(-1, 120), targets.view(-1))
    loss.backward()
    assert loss.item() > 0.0


def test_vit_patch_embedding():
    vit_patch = ViTPatchEmbedding(img_size=32, patch_size=4, in_channels=3, d_model=48)
    img = torch.randn(2, 3, 32, 32)
    patches = vit_patch(img)
    # Num patches = (32 / 4)^2 = 8 * 8 = 64
    assert patches.shape == (2, 64, 48)
