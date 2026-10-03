"""
Tests for Calculus and Optimization module.
"""

import numpy as np
import pytest
from calculus_engine import (
    Value,
    numerical_derivative,
    numerical_gradient,
    numerical_jacobian,
    numerical_hessian,
    check_convexity_at_point,
    gradient_descent,
    adam_optimizer,
)


def test_micro_autograd_basic():
    # f(x, y) = (x * y) + (x ** 2)
    # df/dx = y + 2x
    # df/dy = x
    x = Value(3.0)
    y = Value(4.0)
    z = (x * y) + (x ** 2)
    z.backward()

    assert np.isclose(z.data, 12.0 + 9.0)
    assert np.isclose(x.grad, 4.0 + 2.0 * 3.0)  # 10.0
    assert np.isclose(y.grad, 3.0)


def test_micro_autograd_relu():
    x1 = Value(-2.0)
    y1 = x1.relu()
    y1.backward()
    assert np.isclose(y1.data, 0.0)
    assert np.isclose(x1.grad, 0.0)

    x2 = Value(3.0)
    y2 = x2.relu()
    y2.backward()
    assert np.isclose(y2.data, 3.0)
    assert np.isclose(x2.grad, 1.0)


def test_numerical_derivative():
    # f(x) = x^3 - 4x + 1 -> f'(x) = 3x^2 - 4
    f = lambda x: x ** 3 - 4 * x + 1
    x = 2.0
    exact = 3 * (2.0 ** 2) - 4  # 8.0
    approx = numerical_derivative(f, x)
    assert np.isclose(approx, exact, atol=1e-5)


def test_numerical_gradient():
    # f(x, y) = x^2 + 3y^2 -> grad = [2x, 6y]
    f = lambda v: v[0] ** 2 + 3 * (v[1] ** 2)
    v = np.array([2.0, -1.0])
    exact_grad = np.array([4.0, -6.0])
    approx_grad = numerical_gradient(f, v)
    assert np.allclose(approx_grad, exact_grad, atol=1e-4)


def test_numerical_jacobian():
    # F(x, y) = [x^2 + y, 3*x*y]
    # J = [[2x, 1], [3y, 3x]]
    def F(v):
        return np.array([v[0] ** 2 + v[1], 3 * v[0] * v[1]])

    v = np.array([2.0, 3.0])
    exact_J = np.array([[4.0, 1.0], [9.0, 6.0]])
    approx_J = numerical_jacobian(F, v)
    assert np.allclose(approx_J, exact_J, atol=1e-4)


def test_numerical_hessian_and_convexity():
    # f(x, y) = x^2 + y^2 (strictly convex everywhere)
    f_convex = lambda v: v[0] ** 2 + v[1] ** 2
    v = np.array([1.0, 2.0])
    exact_H = np.array([[2.0, 0.0], [0.0, 2.0]])
    approx_H = numerical_hessian(f_convex, v)
    assert np.allclose(approx_H, exact_H, atol=1e-3)
    assert check_convexity_at_point(approx_H) is True

    # Saddle point: f(x, y) = x^2 - y^2 (H has eigenvalues 2 and -2)
    f_saddle = lambda v: v[0] ** 2 - v[1] ** 2
    H_saddle = numerical_hessian(f_saddle, v)
    assert check_convexity_at_point(H_saddle) is False


def test_optimizers_convergence():
    # Minimize quadratic bowl: f(x, y) = (x - 3)^2 + 2*(y + 1)^2
    f = lambda v: (v[0] - 3.0) ** 2 + 2.0 * (v[1] + 1.0) ** 2
    init_pt = np.array([0.0, 0.0])

    # GD
    opt_gd, hist_gd = gradient_descent(f, init_pt, lr=0.1, num_steps=100)
    assert np.allclose(opt_gd, np.array([3.0, -1.0]), atol=1e-2)
    assert hist_gd[-1] < hist_gd[0]

    # Adam
    opt_adam, hist_adam = adam_optimizer(f, init_pt, lr=0.1, num_steps=150)
    assert np.allclose(opt_adam, np.array([3.0, -1.0]), atol=1e-2)
    assert hist_adam[-1] < hist_adam[0]
