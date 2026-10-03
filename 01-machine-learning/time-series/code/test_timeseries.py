"""
Tests for Time Series forecasting and feature extraction.
"""

import numpy as np
import pytest
from timeseries_engine import (
    TimeSeriesSplitter,
    difference_series,
    invert_difference,
    AutoregressiveModelScratch,
    extract_timeseries_features,
    calculate_smape,
)


def test_timeseries_splitter():
    splitter = TimeSeriesSplitter(n_splits=3, test_size=10)
    splits = list(splitter.split(n_samples=100))
    assert len(splits) == 3

    for tr_idx, te_idx in splits:
        assert len(te_idx) == 10
        # Crucial test: Training indices strictly precede test indices
        assert np.max(tr_idx) < np.min(te_idx)


def test_differencing_and_inversion():
    series = np.array([10.0, 12.0, 15.0, 19.0, 24.0])
    diff = difference_series(series, lag=1)
    expected_diff = np.array([2.0, 3.0, 4.0, 5.0])
    assert np.allclose(diff, expected_diff)

    inverted = invert_difference(diff, initial_val=series[0])
    assert np.allclose(inverted, series)


def test_autoregressive_model_scratch():
    # Linear trend + autoregressive signal
    np.random.seed(42)
    t = np.arange(100)
    signal = 2.0 + 0.5 * np.sin(0.2 * t) + np.random.normal(0, 0.1, size=100)

    ar = AutoregressiveModelScratch(p=2).fit(signal[:80])
    forecast = ar.forecast(steps=5)

    assert len(forecast) == 5
    assert not np.isnan(forecast).any()
    # Forecasts should be in realistic range
    assert np.all(forecast > 0.0)


def test_extract_timeseries_features():
    series = np.arange(20, dtype=float)
    X, y = extract_timeseries_features(series, max_lags=2, rolling_windows=[3])

    # start_idx is max(2, 3) = 3 -> 20 - 3 = 17 rows
    assert X.shape[0] == 17
    assert len(y) == 17
    # Feature columns: 2 lags + 1 rolling mean + 1 rolling std = 4
    assert X.shape[1] == 4


def test_calculate_smape():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([105.0, 195.0, 310.0])
    smape = calculate_smape(y_true, y_pred)
    assert 0.0 <= smape <= 10.0  # Close predictions
