"""
Deep Learning Fundamentals: Forward propagation, backpropagation,
activations, loss functions, weight initializations, and optimizers from scratch.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional


# ============================================================================
# 1. Custom Computational Graph & Autograd Layer (NumPy)
# ============================================================================

class DenseLayer:
    """
    Fully connected dense layer: Y = X @ W + b
    Implements forward and backward passes analytically.
    """
    def __init__(self, in_features: int, out_features: int, init_method: str = "he"):
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight Initialization
        if init_method == "he":
            # He / Kaiming normal: std = sqrt(2 / in_features)
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        elif init_method == "xavier":
            # Xavier / Glorot normal: std = sqrt(2 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
        else:
            self.W = np.random.randn(in_features, out_features) * 0.01
            
        self.b = np.zeros((1, out_features))
        
        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.X_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_cache = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backpropagation through linear layer:
        dL/dX = grad_output @ W.T
        dL/dW = X.T @ grad_output
        dL/db = sum(grad_output, axis=0)
        """
        assert self.X_cache is not None, "Forward must be called before backward"
        self.grad_W = np.dot(self.X_cache.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)


# ============================================================================
# 2. Activation Functions with Analytical Gradients
# ============================================================================

class ReLU:
    def __init__(self):
        self.cache: Optional[np.ndarray] = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.cache = Z
        return np.maximum(0, Z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache is not None
        return grad_output * (self.cache > 0).astype(np.float64)


class Sigmoid:
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        self.out = np.where(
            Z >= 0,
            1.0 / (1.0 + np.exp(-Z)),
            np.exp(Z) / (1.0 + np.exp(Z))
        )
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.out is not None
        return grad_output * self.out * (1.0 - self.out)


class SoftmaxCrossEntropyLoss:
    """
    Combined Softmax + Cross-Entropy loss for numerical stability:
    L = - (1/N) * sum_i log( p_{y_i} )
    dL/dZ = (P - Y) / N
    """
    def __init__(self):
        self.probs: Optional[np.ndarray] = None
        self.y_onehot: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        # Subtract max for numerical stability (prevent exp overflow)
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shift_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        N = logits.shape[0]
        n_classes = logits.shape[1]
        
        # Convert y to one-hot if needed
        if y.ndim == 1:
            self.y_onehot = np.zeros((N, n_classes))
            self.y_onehot[np.arange(N), y] = 1.0
        else:
            self.y_onehot = y
            
        eps = 1e-15
        loss = -np.sum(self.y_onehot * np.log(self.probs + eps)) / N
        return float(loss)

    def backward(self) -> np.ndarray:
        assert self.probs is not None and self.y_onehot is not None
        N = self.probs.shape[0]
        return (self.probs - self.y_onehot) / N


# ============================================================================
# 3. Optimizers Implemented from Scratch
# ============================================================================

class ScratchOptimizer:
    def __init__(self, layers: List[DenseLayer]):
        self.layers = layers

    def zero_grad(self):
        for layer in self.layers:
            layer.grad_W.fill(0.0)
            layer.grad_b.fill(0.0)

    def step(self):
        raise NotImplementedError


class SGDMomentum(ScratchOptimizer):
    """
    SGD with Polyak Momentum:
    v = beta * v + lr * grad
    theta = theta - v
    """
    def __init__(self, layers: List[DenseLayer], lr: float = 0.01, momentum: float = 0.9):
        super().__init__(layers)
        self.lr = lr
        self.momentum = momentum
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            self.v_W[i] = self.momentum * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.momentum * self.v_b[i] + self.lr * layer.grad_b
            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]


class RMSprop(ScratchOptimizer):
    """
    RMSProp: divides learning rate by exponentially decaying average of squared gradients:
    s = gamma * s + (1 - gamma) * grad^2
    theta = theta - lr * grad / (sqrt(s) + eps)
    """
    def __init__(self, layers: List[DenseLayer], lr: float = 0.001, alpha: float = 0.99, eps: float = 1e-8):
        super().__init__(layers)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.s_W = [np.zeros_like(l.W) for l in self.layers]
        self.s_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            self.s_W[i] = self.alpha * self.s_W[i] + (1.0 - self.alpha) * (layer.grad_W ** 2)
            self.s_b[i] = self.alpha * self.s_b[i] + (1.0 - self.alpha) * (layer.grad_b ** 2)
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.eps)


class AdamW(ScratchOptimizer):
    """
    AdamW (Adaptive Moment Estimation with Decoupled Weight Decay):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    theta = theta - lr * weight_decay * theta - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(
        self,
        layers: List[DenseLayer],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(layers)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m_W = [np.zeros_like(l.W) for l in self.layers]
        self.m_b = [np.zeros_like(l.b) for l in self.layers]
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            # 1st moment
            self.m_W[i] = self.beta1 * self.m_W[i] + (1.0 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * layer.grad_b
            
            # 2nd moment
            self.v_W[i] = self.beta2 * self.v_W[i] + (1.0 - self.beta2) * (layer.grad_W ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (layer.grad_b ** 2)
            
            # Bias correction
            m_W_hat = self.m_W[i] / (1.0 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1.0 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1.0 - self.beta2 ** self.t)
            
            # Decoupled weight decay
            layer.W -= self.lr * self.weight_decay * layer.W
            
            # Parameter update
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


# ============================================================================
# 4. PyTorch Reference Architecture
# ============================================================================

class ReferencePyTorchMLP(nn.Module):
    """
    Standard PyTorch Multi-Layer Perceptron (MLP) for classification/regression.
    """
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
