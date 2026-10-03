"""
Calculus and Optimization for ML - Computational Engine
Implements numerical differentiation, micro-autograd computational graph,
Jacobian/Hessian computation, convexity testing, and gradient descent optimizers.
"""

from __future__ import annotations
import math
from typing import Callable
import numpy as np


class Value:
    """
    Micro-autograd scalar node building a dynamic computational DAG.
    Supports reverse-mode automatic differentiation.
    """
    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Value | float) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __mul__(self, other: Value | float) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: Value | float) -> "Value":
        return self + (-other)

    def __rsub__(self, other: float) -> "Value":
        return Value(other) - self

    def __pow__(self, exponent: int | float) -> "Value":
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Runs topological sort and executes chain rule backpropagation."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Computes derivative via central difference formula: O(h^2) error."""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Computes gradient vector grad(f)(x) in R^N using central difference."""
    grad = np.zeros_like(x, dtype=np.float64)
    x_mut = x.astype(np.float64).copy()
    for i in range(len(x)):
        old_val = x_mut[i]
        x_mut[i] = old_val + h
        f_plus = f(x_mut)
        x_mut[i] = old_val - h
        f_minus = f(x_mut)
        x_mut[i] = old_val
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def numerical_jacobian(F: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Computes Jacobian matrix J in R^{M x N} for vector-valued function F: R^N -> R^M.
    J_{ij} = d F_i / d x_j
    """
    x_mut = x.astype(np.float64).copy()
    y0 = F(x)
    M = len(y0)
    N = len(x)
    J = np.zeros((M, N), dtype=np.float64)

    for j in range(N):
        old_val = x_mut[j]
        x_mut[j] = old_val + h
        y_plus = F(x_mut)
        x_mut[j] = old_val - h
        y_minus = F(x_mut)
        x_mut[j] = old_val
        J[:, j] = (y_plus - y_minus) / (2 * h)
    return J


def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """
    Computes Hessian matrix H in R^{N x N} of second partial derivatives.
    H_{ij} = d^2 f / (d x_i d x_j)
    """
    N = len(x)
    H = np.zeros((N, N), dtype=np.float64)
    x_mut = x.astype(np.float64).copy()

    for i in range(N):
        for j in range(N):
            if i == j:
                # 1D second derivative formula
                old_i = x_mut[i]
                x_mut[i] = old_i + h
                f_plus = f(x_mut)
                x_mut[i] = old_i - h
                f_minus = f(x_mut)
                x_mut[i] = old_i
                f_0 = f(x_mut)
                H[i, i] = (f_plus - 2 * f_0 + f_minus) / (h ** 2)
            else:
                # Cross partial derivative
                old_i = x_mut[i]
                old_j = x_mut[j]

                x_mut[i] = old_i + h
                x_mut[j] = old_j + h
                f_pp = f(x_mut)

                x_mut[i] = old_i + h
                x_mut[j] = old_j - h
                f_pm = f(x_mut)

                x_mut[i] = old_i - h
                x_mut[j] = old_j + h
                f_mp = f(x_mut)

                x_mut[i] = old_i - h
                x_mut[j] = old_j - h
                f_mm = f(x_mut)

                x_mut[i] = old_i
                x_mut[j] = old_j

                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * (h ** 2))
    return H


def check_convexity_at_point(H: np.ndarray, tol: float = 1e-6) -> bool:
    """Returns True if Hessian H is positive semi-definite (all eigenvalues >= -tol)."""
    eigenvalues = np.linalg.eigvalsh(H)
    return bool(np.all(eigenvalues >= -tol))


def gradient_descent(
    f: Callable[[np.ndarray], float],
    init_x: np.ndarray,
    lr: float = 0.01,
    num_steps: int = 100,
    tol: float = 1e-6
) -> tuple[np.ndarray, list[float]]:
    """Standard gradient descent with trackable loss trajectory."""
    x = init_x.astype(np.float64).copy()
    history = [float(f(x))]

    for _ in range(num_steps):
        grad = numerical_gradient(f, x)
        if np.linalg.norm(grad) < tol:
            break
        x -= lr * grad
        history.append(float(f(x)))

    return x, history


def adam_optimizer(
    f: Callable[[np.ndarray], float],
    init_x: np.ndarray,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    num_steps: int = 100,
    tol: float = 1e-6
) -> tuple[np.ndarray, list[float]]:
    """Adam (Adaptive Moment Estimation) optimizer from scratch."""
    x = init_x.astype(np.float64).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [float(f(x))]

    for t in range(1, num_steps + 1):
        grad = numerical_gradient(f, x)
        if np.linalg.norm(grad) < tol:
            break

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)

        m_hat = m / (1.0 - (beta1 ** t))
        v_hat = v / (1.0 - (beta2 ** t))

        x -= (lr / (np.sqrt(v_hat) + eps)) * m_hat
        history.append(float(f(x)))

    return x, history
