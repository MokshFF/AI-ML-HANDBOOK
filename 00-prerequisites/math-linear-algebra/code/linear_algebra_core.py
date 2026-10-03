"""
Linear Algebra for ML - Core Algorithms from Scratch
Implements fundamental matrix operations, vector norms, projections,
Gram-Schmidt orthogonalization, power iteration, SVD, and PCA.
"""

from __future__ import annotations
import numpy as np


def vector_dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """Computes inner product u^T v with shape validation."""
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError(f"Vectors must be 1D, got shapes {u.shape} and {v.shape}")
    if len(u) != len(v):
        raise ValueError(f"Vector dimension mismatch: {len(u)} vs {len(v)}")
    return float(np.sum(u * v))


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    """Computes cos(theta) = (u . v) / (||u||_2 * ||v||_2)."""
    norm_u = np.sqrt(vector_dot_product(u, u))
    norm_v = np.sqrt(vector_dot_product(v, v))
    return vector_dot_product(u, v) / (norm_u * norm_v + eps)


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = A @ B from definition.
    A: shape (M, K), B: shape (K, N) -> C: shape (M, N)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices.")
    M, K_a = A.shape
    K_b, N = B.shape
    if K_a != K_b:
        raise ValueError(f"Incompatible inner dimensions: {K_a} vs {K_b}")

    C = np.zeros((M, N), dtype=np.float64)
    # Using row-vectorized dot product for efficiency while demonstrating mechanics
    for i in range(M):
        for j in range(N):
            C[i, j] = np.dot(A[i, :], B[:, j])
    return C


def vector_norm(v: np.ndarray, p: int | float | str = 2) -> float:
    """
    Computes Lp norm of a vector:
    ||v||_p = (sum |v_i|^p)^(1/p)
    """
    if p == 1:
        return float(np.sum(np.abs(v)))
    elif p == 2:
        return float(np.sqrt(np.sum(v ** 2)))
    elif p == np.inf or p == "inf":
        return float(np.max(np.abs(v)))
    else:
        return float(np.sum(np.abs(v) ** p) ** (1.0 / p))


def frobenius_norm(A: np.ndarray) -> float:
    """Computes Frobenius norm ||A||_F = sqrt(sum_i sum_j a_ij^2) = sqrt(Tr(A^T A))."""
    return float(np.sqrt(np.sum(A ** 2)))


def project_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Projects vector v onto vector u:
    proj_u(v) = ((v . u) / (u . u)) * u
    """
    denom = vector_dot_product(u, u)
    if np.isclose(denom, 0.0):
        raise ValueError("Cannot project onto zero vector.")
    scalar = vector_dot_product(v, u) / denom
    return scalar * u


def gram_schmidt(V: np.ndarray) -> np.ndarray:
    """
    Orthonormalizes column vectors of matrix V using the Gram-Schmidt process.
    V: shape (n_features, n_vectors)
    Returns: Q of shape (n_features, n_vectors) with orthonormal columns.
    """
    n_features, k = V.shape
    Q = np.zeros((n_features, k), dtype=np.float64)

    for j in range(k):
        v = V[:, j].astype(np.float64)
        for i in range(j):
            q_i = Q[:, i]
            v -= np.dot(v, q_i) * q_i
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            raise ValueError(f"Linearly dependent vector encountered at column {j}.")
        Q[:, j] = v / norm_v

    return Q


def determinant_2x2_or_3x3(A: np.ndarray) -> float:
    """Computes determinant for 2x2 or 3x3 matrices analytically."""
    if A.shape == (2, 2):
        return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    elif A.shape == (3, 3):
        return float(
            A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
        )
    else:
        raise ValueError("Analytical method limited to 2x2 or 3x3; use LU decomposition for larger.")


def power_iteration(A: np.ndarray, num_iterations: int = 100, tol: float = 1e-8) -> tuple[float, np.ndarray]:
    """
    Computes the dominant eigenvalue and its corresponding eigenvector via Power Iteration.
    b_{k+1} = (A b_k) / ||A b_k||
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    
    n = A.shape[0]
    b = np.random.randn(n)
    b = b / np.linalg.norm(b)

    eigenvalue = 0.0
    for _ in range(num_iterations):
        b_next = np.dot(A, b)
        norm = np.linalg.norm(b_next)
        if norm < 1e-12:
            return 0.0, b
        b_next = b_next / norm
        
        # Rayleigh quotient: lambda = (b^T A b) / (b^T b)
        new_eigenvalue = float(np.dot(b_next, np.dot(A, b_next)))
        if np.abs(new_eigenvalue - eigenvalue) < tol:
            break
        b = b_next
        eigenvalue = new_eigenvalue

    return eigenvalue, b


class PCAScratch:
    """
    Principal Component Analysis (PCA) implemented via SVD of centered data.
    X_centered = U * Sigma * V^T
    Principal components are rows of V^T (columns of V).
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        X_centered = X - self.mean_

        # SVD: X_centered = U * S * Vt
        # Covariance C = (1 / (N - 1)) * X_centered^T * X_centered
        # Using np.linalg.svd on X_centered
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Principal directions are rows of Vt
        self.components_ = Vt[: self.n_components]

        # Explained variance: lambda_i = S_i^2 / (N - 1)
        eigenvalues = (S ** 2) / (n_samples - 1)
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / (total_variance + 1e-9)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA is not fitted yet.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA is not fitted yet.")
        return np.dot(X_proj, self.components_) + self.mean_
