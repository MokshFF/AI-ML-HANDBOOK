"""
Supervised Learning - Linear Models from Scratch
Implements OLS Linear Regression, Ridge, Lasso (Coordinate Descent),
Elastic Net, and Logistic Regression with L2 Regularization.
"""

from __future__ import annotations
import numpy as np


class LinearRegressionScratch:
    """
    Ordinary Least Squares (OLS) Linear Regression.
    Supports both closed-form Normal Equations and Gradient Descent.
    
    Objective: min_w ||y - X w||_2^2
    Normal equation: w = (X^T X)^{-1} X^T y
    """
    def __init__(self, solver: str = "normal", lr: float = 0.01, epochs: int = 1000):
        self.solver = solver
        self.lr = lr
        self.epochs = epochs
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1).astype(np.float64)

        if self.solver == "normal":
            # Add bias column of ones: X_ext = [1, X]
            X_ext = np.hstack([np.ones((n_samples, 1)), X.astype(np.float64)])
            # (X^T X)^{-1} X^T y
            try:
                theta = np.linalg.inv(X_ext.T @ X_ext) @ X_ext.T @ y
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(X_ext.T @ X_ext) @ X_ext.T @ y
            self.bias_ = float(theta[0, 0])
            self.weights_ = theta[1:].ravel()
        elif self.solver == "gd":
            self.weights_ = np.zeros(n_features, dtype=np.float64)
            self.bias_ = 0.0
            for _ in range(self.epochs):
                y_pred = X @ self.weights_ + self.bias_
                error = y_pred.reshape(-1, 1) - y
                dw = (2.0 / n_samples) * (X.T @ error).ravel()
                db = (2.0 / n_samples) * np.sum(error)
                self.weights_ -= self.lr * dw
                self.bias_ -= self.lr * db
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        return X @ self.weights_ + self.bias_


class RidgeRegressionScratch:
    """
    Ridge Regression (L2 Regularized OLS).
    
    Objective: min_w ||y - X w||_2^2 + alpha * ||w||_2^2
    Normal equation: w = (X^T X + alpha * I)^{-1} X^T y (excluding bias from penalty)
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionScratch":
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1).astype(np.float64)

        # Center X and y so bias can be computed independently
        x_mean = np.mean(X, axis=0, keepdims=True)
        y_mean = np.mean(y)
        X_centered = X - x_mean
        y_centered = y - y_mean

        # Regularized normal equation
        I = np.eye(n_features)
        A = X_centered.T @ X_centered + self.alpha * I
        self.weights_ = (np.linalg.inv(A) @ X_centered.T @ y_centered).ravel()
        self.bias_ = float(y_mean - np.dot(x_mean.ravel(), self.weights_))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        return X @ self.weights_ + self.bias_


class LassoRegressionScratch:
    """
    Lasso Regression (L1 Regularized OLS) via Coordinate Descent.
    
    Objective: min_w (1 / (2*n)) * ||y - X w||_2^2 + alpha * ||w||_1
    Uses the soft-thresholding operator S(z, gamma) = sign(z) * max(|z| - gamma, 0).
    """
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-5):
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressionScratch":
        n_samples, n_features = X.shape
        X = X.astype(np.float64)
        y = y.astype(np.float64).ravel()

        x_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - x_mean
        y_centered = y - y_mean

        w = np.zeros(n_features, dtype=np.float64)
        norm_sq = np.sum(X_centered ** 2, axis=0)

        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(n_features):
                if norm_sq[j] < 1e-12:
                    continue
                # Partial residual excluding feature j
                y_pred_j = X_centered @ w - X_centered[:, j] * w[j]
                r_j = y_centered - y_pred_j
                rho_j = float(np.dot(X_centered[:, j], r_j))
                # Soft-threshold update
                w[j] = self._soft_threshold(rho_j, self.alpha * n_samples) / norm_sq[j]

            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.weights_ = w
        self.bias_ = float(y_mean - np.dot(x_mean, self.weights_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        return X @ self.weights_ + self.bias_


class LogisticRegressionScratch:
    """
    Binary Logistic Regression with optional L2 regularization.
    
    Model: P(y=1 | x) = sigma(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))
    Loss: Binary Cross-Entropy (Log-Loss) + (lambda / (2*n)) * ||w||_2^2
    """
    def __init__(self, lr: float = 0.1, epochs: int = 1000, l2_reg: float = 0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        n_samples, n_features = X.shape
        X = X.astype(np.float64)
        y = y.astype(np.float64).ravel()

        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0

        for _ in range(self.epochs):
            z = X @ self.weights_ + self.bias_
            y_hat = self._sigmoid(z)
            error = y_hat - y

            # Gradients with L2 regularization on weights (not bias)
            dw = (1.0 / n_samples) * (X.T @ error) + (self.l2_reg / n_samples) * self.weights_
            db = (1.0 / n_samples) * np.sum(error)

            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        z = X @ self.weights_ + self.bias_
        p1 = self._sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
