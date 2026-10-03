"""
Tests for Python for ML modules.
"""

import numpy as np
import pandas as pd
import pytest
from core_patterns import MLDataset, DatasetConfig, batch_generator, Timer, memoize_transformation
from data_processing import (
    loop_vs_vectorized_sum,
    StandardScalerScratch,
    MinMaxScalerScratch,
    one_hot_encode,
    optimize_dataframe_memory,
)


def test_timer():
    with Timer("Test Timer") as t:
        arr = np.sum(np.ones(10000))
    assert t.elapsed > 0


def test_ml_dataset_validation():
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, size=50)

    ds = MLDataset(X, y)
    assert len(ds) == 50
    feat, target = ds[0]
    assert feat.shape == (4,)

    # Mismatch length
    with pytest.raises(ValueError):
        MLDataset(X[:10], y)

    # NaN detection
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        MLDataset(X_nan, y)


def test_batch_generator():
    X = np.random.randn(25, 4)
    y = np.random.randint(0, 2, size=25)
    ds = MLDataset(X, y)

    batches = list(batch_generator(ds, batch_size=10, shuffle=False, drop_last=False))
    assert len(batches) == 3
    assert len(batches[0][0]) == 10
    assert len(batches[1][0]) == 10
    assert len(batches[2][0]) == 5

    # With drop_last
    batches_drop = list(batch_generator(ds, batch_size=10, shuffle=False, drop_last=True))
    assert len(batches_drop) == 2


def test_memoize_transformation():
    call_count = 0

    @memoize_transformation
    def square_arr(a: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return a ** 2

    arr = np.array([1, 2, 3])
    _ = square_arr(arr)
    _ = square_arr(arr)
    assert call_count == 1


def test_loop_vs_vectorized():
    arr = np.random.randn(5000)
    loop_res, vec_res, speedup = loop_vs_vectorized_sum(arr)
    assert np.isclose(loop_res, vec_res)
    assert speedup > 0


def test_standard_scaler():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScalerScratch()
    X_norm = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-6)
    assert np.allclose(np.std(X_norm, axis=0), 1.0, atol=1e-6)

    X_rec = scaler.inverse_transform(X_norm)
    assert np.allclose(X, X_rec)


def test_min_max_scaler():
    X = np.array([[10.0], [20.0], [30.0]])
    scaler = MinMaxScalerScratch(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(X_scaled[0, 0], 0.0)
    assert np.isclose(X_scaled[-1, 0], 1.0)


def test_one_hot_encode():
    cats = np.array([0, 1, 2, 0])
    encoded = one_hot_encode(cats, num_classes=3)
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float32)
    assert np.allclose(encoded, expected)


def test_optimize_dataframe_memory():
    df = pd.DataFrame({
        "ints": [1, 2, 3, 4],
        "floats": [1.1, 2.2, 3.3, 4.4],
        "cats": ["apple", "banana", "apple", "banana"]
    })
    opt_df = optimize_dataframe_memory(df)
    assert opt_df["ints"].dtype == np.int8
    assert opt_df["floats"].dtype == np.float32
    assert opt_df["cats"].dtype.name == "category"
