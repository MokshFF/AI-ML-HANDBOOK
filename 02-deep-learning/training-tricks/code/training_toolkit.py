"""
Deep Learning Training Toolkit: Regularization, Normalization, Schedulers & Diagnostics.
Implements:
1. BatchNorm1d from scratch with running statistics and train/eval toggles.
2. LayerNorm from scratch.
3. Cosine Annealing with Warmup learning rate calculator.
4. Gradient norm clipping from scratch.
5. Early Stopping callback handler.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ============================================================================
# 1. Normalization Layers from Scratch
# ============================================================================

class ScratchBatchNorm1d(nn.Module):
    """
    Batch Normalization 1D (Ioffe & Szegedy, 2015).
    Normalizes across the batch dimension (N).
    Maintains running mean and variance for test-time inference.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable affine parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Non-learnable running statistics
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, num_features)
        """
        if self.training:
            # Batch mean and unbiased variance
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Exponential moving average update
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            # Bessel's correction factor for sample variance
            n = x.size(0)
            unbiased_var = (n / max(n - 1, 1)) * batch_var
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * unbiased_var.detach()

            # Normalize using batch statistics
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Normalize using running statistics
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * x_norm + self.beta


class ScratchLayerNorm(nn.Module):
    """
    Layer Normalization (Ba, Kiros & Hinton, 2016).
    Normalizes across feature dimensions independently for each sample.
    Batch-size independent; identical behavior during training and evaluation.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., normalized_shape)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ============================================================================
# 2. Learning Rate Schedule with Warmup
# ============================================================================

def get_cosine_warmup_lr(
    current_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 1e-6
) -> float:
    """
    Computes learning rate for linear warmup followed by cosine decay:
    - Step < warmup_steps: linear increase from 0 to base_lr.
    - Step >= warmup_steps: cosine decay from base_lr down to min_lr.
    """
    if current_step < warmup_steps:
        # Linear warmup
        return base_lr * float(current_step) / float(max(1, warmup_steps))
    elif current_step >= total_steps:
        return min_lr
    else:
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay


# ============================================================================
# 3. Gradient Norm Clipping from Scratch
# ============================================================================

def clip_grad_norm_scratch(parameters: List[nn.Parameter], max_norm: float) -> float:
    """
    Clips gradient norms of an iterable of parameters:
    total_norm = sqrt( sum_i ||grad_i||_2^2 )
    grad_i = grad_i * (max_norm / total_norm) if total_norm > max_norm
    Returns: total_norm (float)
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return 0.0

    total_norm_sq = sum(p.grad.detach().pow(2).sum().item() for p in params_with_grad)
    total_norm = math.sqrt(total_norm_sq)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)

    return total_norm


# ============================================================================
# 4. Early Stopping Handler
# ============================================================================

class EarlyStoppingHandler:
    """
    Monitors validation loss; signals early stopping when loss stops improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Returns True if training should stop, else False.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
