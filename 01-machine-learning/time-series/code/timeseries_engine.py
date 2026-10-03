"""
Time Series for ML - Forecasting, Stationarity & Feature Engineering
Implements temporal cross-validation, differencing for stationarity,
autoregressive AR(p) models, lag/rolling feature extraction, and time-series metrics.
"""

from __future__ import annotations
from typing import Generator
import numpy as np


class TimeSeriesSplitter:
    """
    Temporal Cross-Validation (Walk-Forward / Expanding Window).
    Ensures training strictly precedes validation to prevent lookahead leakage.
    """
    def __init__(self, n_splits: int = 4, test_size: int = 12):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, n_samples: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        for i in range(self.n_splits):
            test_end = n_samples - i * self.test_size
            test_start = test_end - self.test_size
            if test_start <= 0:
                break
            train_idx = np.arange(0, test_start)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx


def difference_series(series: np.ndarray, lag: int = 1) -> np.ndarray:
    """Computes differenced series y'_t = y_t - y_{t-lag} to achieve stationarity."""
    return series[lag:] - series[:-lag]


def invert_difference(diff_series: np.ndarray, initial_val: float) -> np.ndarray:
    """Reconstructs original series from differenced values: y_t = y_0 + cumsum(y')."""
    return np.r_[initial_val, initial_val + np.cumsum(diff_series)]


class AutoregressiveModelScratch:
    """
    Autoregressive AR(p) model fit via Ordinary Least Squares on lag matrix.
    Model: y_t = c + sum_{i=1}^p phi_i y_{t-i} + epsilon_t
    """
    def __init__(self, p: int = 2):
        self.p = p
        self.intercept_: float = 0.0
        self.phi_: np.ndarray | None = None
        self.last_observations_: np.ndarray | None = None

    def fit(self, series: np.ndarray) -> "AutoregressiveModelScratch":
        y = series.ravel().astype(np.float64)
        n = len(y)
        if n <= self.p:
            raise ValueError(f"Series length {n} must be greater than lag order p={self.p}.")

        # Construct design matrix of lags: X has shape (n - p, p)
        X = np.zeros((n - self.p, self.p), dtype=np.float64)
        for i in range(self.p):
            X[:, i] = y[self.p - 1 - i : n - 1 - i]
        y_target = y[self.p :]

        # OLS fit: [1, X]
        X_ext = np.hstack([np.ones((len(X), 1)), X])
        theta = np.linalg.pinv(X_ext.T @ X_ext) @ X_ext.T @ y_target

        self.intercept_ = float(theta[0])
        self.phi_ = theta[1:]
        self.last_observations_ = y[-self.p :]
        return self

    def forecast(self, steps: int = 1) -> np.ndarray:
        """Autoregressive multi-step recursive forecasting."""
        if self.phi_ is None or self.last_observations_ is None:
            raise RuntimeError("Model is not fitted.")

        history = list(self.last_observations_)
        forecasts = []

        for _ in range(steps):
            lags = np.array(history[-self.p :][::-1])
            y_next = self.intercept_ + np.dot(self.phi_, lags)
            forecasts.append(float(y_next))
            history.append(float(y_next))

        return np.array(forecasts)


def extract_timeseries_features(series: np.ndarray, max_lags: int = 3, rolling_windows: list[int] = [3, 7]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts lag features and rolling window averages for tabular machine learning.
    Returns: (X_features, y_targets)
    """
    y = series.ravel().astype(np.float64)
    n = len(y)
    start_idx = max(max_lags, max(rolling_windows))

    feature_rows = []
    target_vals = []

    for t in range(start_idx, n):
        row = []
        # Lag features
        for l in range(1, max_lags + 1):
            row.append(y[t - l])
        # Rolling statistics
        for w in rolling_windows:
            window_slice = y[t - w : t]
            row.append(float(np.mean(window_slice)))
            row.append(float(np.std(window_slice)))

        feature_rows.append(row)
        target_vals.append(y[t])

    return np.array(feature_rows), np.array(target_vals)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) in [0, 100%].
    sMAPE = (100% / N) * sum (2 * |y_t - y_hat| / (|y_t| + |y_hat|))
    """
    y_t = y_true.ravel().astype(np.float64)
    y_p = y_pred.ravel().astype(np.float64)
    denom = np.abs(y_t) + np.abs(y_p) + 1e-12
    return float(np.mean(200.0 * np.abs(y_t - y_p) / denom))
