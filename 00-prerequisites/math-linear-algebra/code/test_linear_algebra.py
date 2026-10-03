"""
Tests for Linear Algebra core algorithms from scratch.
"""

import numpy as np
import pytest
from linear_algebra_core import (
    vector_dot_product,
    cosine_similarity,
    matrix_multiply,
    vector_norm,
    frobenius_norm,
    project_vector,
    gram_schmidt,
    determinant_2x2_or_3x3,
    power_iteration,
    PCAScratch,
)


def test_dot_product_and_cosine():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, -5.0, 6.0])

    dot_scratch = vector_dot_product(u, v)
    dot_np = np.dot(u, v)
    assert np.isclose(dot_scratch, dot_np)

    cos_sim = cosine_similarity(u, u)
    assert np.isclose(cos_sim, 1.0)


def test_matrix_multiply():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    B = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # 2x3

    C_scratch = matrix_multiply(A, B)
    C_np = A @ B
    assert np.allclose(C_scratch, C_np)


def test_norms():
    v = np.array([3.0, -4.0])
    assert np.isclose(vector_norm(v, p=1), 7.0)
    assert np.isclose(vector_norm(v, p=2), 5.0)
    assert np.isclose(vector_norm(v, p="inf"), 4.0)

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(frobenius_norm(A), np.linalg.norm(A, "fro"))


def test_vector_projection():
    v = np.array([3.0, 4.0])
    u = np.array([1.0, 0.0])  # x-axis
    p = project_vector(v, u)
    assert np.allclose(p, np.array([3.0, 0.0]))


def test_gram_schmidt():
    V = np.array([
        [1.0, 1.0],
        [1.0, 0.0]
    ])
    Q = gram_schmidt(V)
    # Check orthogonality: Q^T Q == I
    I_approx = Q.T @ Q
    assert np.allclose(I_approx, np.eye(2), atol=1e-7)


def test_determinant():
    A2 = np.array([[4.0, 7.0], [2.0, 6.0]])
    assert np.isclose(determinant_2x2_or_3x3(A2), np.linalg.det(A2))

    A3 = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    assert np.isclose(determinant_2x2_or_3x3(A3), np.linalg.det(A3))


def test_power_iteration():
    # Symmetric positive definite matrix
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    true_eigenvalues, true_eigenvectors = np.linalg.eigh(A)
    max_idx = np.argmax(true_eigenvalues)
    expected_val = true_eigenvalues[max_idx]

    val, vec = power_iteration(A)
    assert np.isclose(val, expected_val, atol=1e-3)


def test_pca_scratch():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    pca = PCAScratch(n_components=2)
    X_proj = pca.fit_transform(X)

    assert X_proj.shape == (100, 2)
    assert len(pca.explained_variance_) == 2
    assert np.sum(pca.explained_variance_ratio_) <= 1.0

    # Reconstruction test
    X_rec = pca.inverse_transform(X_proj)
    assert X_rec.shape == (100, 5)
