"""
Python for ML - Data Processing & Numerical Manipulation
Production-ready implementations for array vectorization, categorical encoding,
feature scaling, and Pandas memory optimization.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def loop_vs_vectorized_sum(arr: np.ndarray) -> tuple[float, float, float]:
    """
    Demonstrates time disparity between native Python loop and NumPy vectorization.
    Returns: (loop_result, vectorized_result, speedup_ratio)
    """
    import time

    # Python loop
    t0 = time.perf_counter()
    total_loop = 0.0
    for val in arr.flat:
        total_loop += float(val)
    t_loop = time.perf_counter() - t0

    # Vectorized NumPy sum
    t1 = time.perf_counter()
    total_vec = float(np.sum(arr))
    t_vec = time.perf_counter() - t1

    speedup = t_loop / (t_vec + 1e-9)
    return total_loop, total_vec, speedup


class StandardScalerScratch:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    z = (x - u) / s
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerScratch instance is not fitted yet.")
        return (X - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerScratch instance is not fitted yet.")
        return X_scaled * (self.std_ + self.eps) + self.mean_


class MinMaxScalerScratch:
    """
    Transforms features by scaling each feature to a given range [0, 1].
    x_scaled = (x - min) / (max - min)
    """
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8):
        self.feature_range = feature_range
        self.eps = eps
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScalerScratch":
        self.data_min_ = np.min(X, axis=0, keepdims=True)
        self.data_max_ = np.max(X, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("MinMaxScalerScratch is not fitted.")
        norm = (X - self.data_min_) / (self.data_max_ - self.data_min_ + self.eps)
        scale = self.feature_range[1] - self.feature_range[0]
        return norm * scale + self.feature_range[0]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def one_hot_encode(categories: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """
    Vectorized One-Hot Encoding in pure NumPy.
    
    Args:
        categories: 1D integer class array, shape (N,).
        num_classes: Optional total number of classes. If None, inferred as max(categories) + 1.

    Returns:
        2D binary matrix of shape (N, num_classes).
    """
    if categories.ndim != 1:
        raise ValueError("Categories array must be 1-dimensional.")
    k = num_classes if num_classes is not None else int(np.max(categories)) + 1
    return np.eye(k, dtype=np.float32)[categories]


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric types in a Pandas DataFrame to minimize memory footprint.
    
    Useful in data preprocessing when handling large tabular datasets.
    """
    optimized = df.copy()
    for col in optimized.columns:
        col_type = optimized[col].dtype
        if col_type == object:
            num_unique = optimized[col].nunique()
            num_total = len(optimized[col])
            if num_total > 0 and (num_unique / num_total) <= 0.5:
                optimized[col] = optimized[col].astype("category")
        elif np.issubdtype(col_type, np.integer):
            c_min = optimized[col].min()
            c_max = optimized[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                optimized[col] = optimized[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                optimized[col] = optimized[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                optimized[col] = optimized[col].astype(np.int32)
        elif np.issubdtype(col_type, np.floating):
            optimized[col] = optimized[col].astype(np.float32)
    return optimized
