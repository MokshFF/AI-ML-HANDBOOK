import pytest
import math
import torch
import torch.nn as nn
from training_toolkit import (
    ScratchBatchNorm1d,
    ScratchLayerNorm,
    get_cosine_warmup_lr,
    clip_grad_norm_scratch,
    EarlyStoppingHandler
)


def test_scratch_batchnorm1d():
    torch.manual_seed(42)
    bn = ScratchBatchNorm1d(num_features=4)
    x = torch.randn(20, 4) * 5.0 + 3.0  # Mean ~3, Std ~5

    # 1. Training mode
    bn.train()
    out = bn(x)
    assert out.shape == (20, 4)
    # Output should have mean ~0 and std ~1 across batch
    assert torch.allclose(out.mean(dim=0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(out.var(dim=0, unbiased=False), torch.ones(4), atol=1e-2)

    # 2. Evaluation mode
    bn.eval()
    x_test = torch.randn(5, 4)
    out_eval = bn(x_test)
    assert out_eval.shape == (5, 4)


def test_scratch_layernorm():
    ln = ScratchLayerNorm(normalized_shape=8)
    x = torch.randn(4, 8) * 10.0 + 2.0
    out = ln(x)
    assert out.shape == (4, 8)
    # Each sample should have mean ~0 and variance ~1 across feature dimension
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4), atol=1e-4)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(4), atol=1e-2)


def test_cosine_warmup_lr():
    base_lr = 1e-3
    min_lr = 1e-5
    warmup_steps = 10
    total_steps = 100

    # Step 0: lr = 0
    assert get_cosine_warmup_lr(0, total_steps, warmup_steps, base_lr, min_lr) == 0.0
    # Step 5: halfway through warmup
    assert math.isclose(get_cosine_warmup_lr(5, total_steps, warmup_steps, base_lr, min_lr), base_lr * 0.5)
    # Step 10: warmup complete
    assert math.isclose(get_cosine_warmup_lr(10, total_steps, warmup_steps, base_lr, min_lr), base_lr)
    # Step 100+: min_lr
    assert math.isclose(get_cosine_warmup_lr(100, total_steps, warmup_steps, base_lr, min_lr), min_lr)


def test_gradient_clipping():
    p1 = nn.Parameter(torch.tensor([3.0, 4.0]))  # Norm = 5.0
    p1.grad = torch.tensor([3.0, 4.0])

    max_norm = 1.0
    orig_norm = clip_grad_norm_scratch([p1], max_norm=max_norm)
    assert math.isclose(orig_norm, 5.0, rel_tol=1e-4)

    # Scaled grad norm should be equal to max_norm = 1.0
    new_norm = math.sqrt(p1.grad.pow(2).sum().item())
    assert math.isclose(new_norm, 1.0, rel_tol=1e-4)
    # Direction should be preserved: [3/5, 4/5] = [0.6, 0.8]
    assert torch.allclose(p1.grad, torch.tensor([0.6, 0.8]), atol=1e-4)


def test_early_stopping():
    handler = EarlyStoppingHandler(patience=3, min_delta=0.01)
    
    # Improving losses: should not stop
    assert not handler.step(1.0)
    assert not handler.step(0.9)
    assert not handler.step(0.8)

    # Stagnant losses: count towards patience
    assert not handler.step(0.805)  # counter = 1
    assert not handler.step(0.801)  # counter = 2
    assert handler.step(0.802)      # counter = 3 -> early stop triggered!
    assert handler.early_stop is True
