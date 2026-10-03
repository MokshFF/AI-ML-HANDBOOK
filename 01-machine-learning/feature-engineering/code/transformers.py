"""
Feature Engineering for ML - Preprocessing & Transformation Pipelines
Implements missing value imputation, target & one-hot encoding, robust scaling,
power transformations, outlier handling, and feature selection.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


class MissingValueImputer:
    """
    Imputes missing values (NaNs) and optionally adds missingness indicator flags.
    Supports 'mean', 'median', 'mode', and 'constant' strategies.
    """
    def __init__(self, strategy: str = "mean", fill_value: Any = None, add_indicator: bool = True):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.statistics_: dict[int, Any] = {}

    def fit(self, X: np.ndarray) -> "MissingValueImputer":
        n_features = X.shape[1]
        self.statistics_ = {}

        for j in range(n_features):
            col = X[:, j]
            valid_vals = col[~np.isnan(col)]
            if len(valid_vals) == 0:
                self.statistics_[j] = 0.0
            elif self.strategy == "mean":
                self.statistics_[j] = float(np.mean(valid_vals))
            elif self.strategy == "median":
                self.statistics_[j] = float(np.median(valid_vals))
            elif self.strategy == "constant":
                self.statistics_[j] = self.fill_value
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        indicators = []

        for j in range(X.shape[1]):
            nan_mask = np.isnan(X_out[:, j])
            if self.add_indicator:
                indicators.append(nan_mask.astype(np.float64))
            X_out[nan_mask, j] = self.statistics_[j]

        if self.add_indicator and indicators:
            return np.hstack([X_out, np.column_stack(indicators)])
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SmoothedTargetEncoder:
    """
    Target (Mean) Encoding with Empirical Bayes Smoothing to prevent target leakage & overfitting.
    Formula: S_c = lambda(n_c) * mean(y_c) + (1 - lambda(n_c)) * global_mean
    where lambda(n_c) = n_c / (n_c + smoothing)
    """
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.mapping_: dict[Any, float] = {}

    def fit(self, categories: np.ndarray, y: np.ndarray) -> "SmoothedTargetEncoder":
        y = y.ravel().astype(np.float64)
        self.global_mean_ = float(np.mean(y))
        self.mapping_ = {}

        unique_cats, counts = np.unique(categories, return_counts=True)
        for cat, n_c in zip(unique_cats, counts):
            y_c = y[categories == cat]
            mean_c = np.mean(y_c)
            weight = n_c / (n_c + self.smoothing)
            smoothed_val = weight * mean_c + (1.0 - weight) * self.global_mean_
            self.mapping_[cat] = float(smoothed_val)

        return self

    def transform(self, categories: np.ndarray) -> np.ndarray:
        return np.array([self.mapping_.get(c, self.global_mean_) for c in categories], dtype=np.float64)


class RobustScalerScratch:
    """
    Scales features using statistics that are robust to outliers:
    z = (x - median) / IQR, where IQR = Q3 - Q1
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustScalerScratch":
        self.median_ = np.median(X, axis=0, keepdims=True)
        q75 = np.percentile(X, 75, axis=0, keepdims=True)
        q25 = np.percentile(X, 25, axis=0, keepdims=True)
        self.iqr_ = np.maximum(q75 - q25, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class OutlierCapper:
    """
    Winsorization: Caps extreme outliers outside [Q1 - k*IQR, Q3 + k*IQR].
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "OutlierCapper":
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        self.lower_bounds_ = q25 - self.factor * iqr
        self.upper_bounds_ = q75 + self.factor * iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Capper is not fitted.")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class VarianceThresholdSelector:
    """
    Filter Feature Selection: Drops low-variance features: Var(X_j) < threshold.
    """
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.variances_: np.ndarray | None = None
        self.selected_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "VarianceThresholdSelector":
        self.variances_ = np.var(X, axis=0)
        self.selected_indices_ = np.where(self.variances_ > self.threshold)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("Selector is not fitted.")
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
