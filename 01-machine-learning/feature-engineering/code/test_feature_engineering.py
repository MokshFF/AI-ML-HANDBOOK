"""
Tests for Feature Engineering transformers.
"""

import numpy as np
import pytest
from transformers import (
    MissingValueImputer,
    SmoothedTargetEncoder,
    RobustScalerScratch,
    OutlierCapper,
    VarianceThresholdSelector,
)


def test_missing_value_imputer():
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [np.nan, 6.0],
        [5.0, 8.0]
    ])
    imputer = MissingValueImputer(strategy="mean", add_indicator=True)
    X_imputed = imputer.fit_transform(X)

    # Output shape: 2 original features + 2 indicator columns = 4
    assert X_imputed.shape == (4, 4)
    # Check mean imputation on col 0: (1+3+5)/3 = 3.0
    assert np.isclose(X_imputed[2, 0], 3.0)
    # Check indicator on row 2, col 0 was missing
    assert X_imputed[2, 2] == 1.0


def test_smoothed_target_encoder():
    cats = np.array(["A", "A", "B", "B", "B", "C"])
    y = np.array([1, 1, 0, 0, 0, 1])

    encoder = SmoothedTargetEncoder(smoothing=5.0).fit(cats, y)
    encoded = encoder.transform(cats)

    assert len(encoded) == 6
    # Category A (mean=1.0, count=2): smoothed towards global mean 0.5
    assert encoded[0] < 1.0
    assert encoded[0] > 0.5
    # Category B (mean=0.0, count=3): smoothed towards global mean 0.5
    assert encoded[2] > 0.0
    assert encoded[2] < 0.5


def test_robust_scaler():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])  # Extreme outlier 100
    scaler = RobustScalerScratch().fit(X)
    X_scaled = scaler.transform(X)

    # Median is 3.0, IQR is 4 - 2 = 2.0
    assert np.isclose(scaler.median_[0, 0], 3.0)
    assert np.isclose(scaler.iqr_[0, 0], 2.0)
    assert np.isclose(X_scaled[2, 0], 0.0)  # Median scaled to 0


def test_outlier_capper():
    X = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 50.0, -20.0]).reshape(-1, 1)
    capper = OutlierCapper(factor=1.5).fit(X)
    X_capped = capper.transform(X)

    assert np.max(X_capped) < 50.0
    assert np.min(X_capped) > -20.0


def test_variance_threshold_selector():
    # Col 0: zero variance, Col 1: normal variance, Col 2: high variance
    X = np.array([
        [1.0, 2.0, 10.0],
        [1.0, 3.0, 20.0],
        [1.0, 2.5, 30.0],
        [1.0, 3.5, 40.0]
    ])
    selector = VarianceThresholdSelector(threshold=0.0).fit(X)
    X_sel = selector.transform(X)

    assert X_sel.shape == (4, 2)
    assert 0 not in selector.selected_indices_
