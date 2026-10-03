"""
Tests for Model Evaluation metrics and cross-validation.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    brier_score_loss,
)
from metrics import (
    confusion_matrix_scratch,
    classification_metrics_scratch,
    roc_curve_and_auc_scratch,
    regression_metrics_scratch,
    brier_score_scratch,
    stratified_kfold_scratch,
)


def test_confusion_matrix_and_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])

    tp, fp, tn, fn = confusion_matrix_scratch(y_true, y_pred)
    assert tp == 3
    assert fp == 1
    assert tn == 3
    assert fn == 1

    metrics = classification_metrics_scratch(y_true, y_pred)
    assert np.isclose(metrics["accuracy"], accuracy_score(y_true, y_pred))
    assert np.isclose(metrics["precision"], precision_score(y_true, y_pred))
    assert np.isclose(metrics["recall"], recall_score(y_true, y_pred))
    assert np.isclose(metrics["f1"], f1_score(y_true, y_pred))


def test_roc_auc_scratch():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.4, 0.1, 0.3, 0.2])

    fpr, tpr, auc_scratch = roc_curve_and_auc_scratch(y_true, y_scores)
    auc_sk = roc_auc_score(y_true, y_scores)
    assert np.isclose(auc_scratch, auc_sk, atol=1e-2)
    assert len(fpr) == len(tpr)


def test_regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    res = regression_metrics_scratch(y_true, y_pred)
    assert np.isclose(res["mse"], mean_squared_error(y_true, y_pred))
    assert np.isclose(res["r2"], r2_score(y_true, y_pred))


def test_brier_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_probs = np.array([0.8, 0.2, 0.9, 0.6, 0.1])

    brier_scratch = brier_score_scratch(y_true, y_probs)
    brier_sk = brier_score_loss(y_true, y_probs)
    assert np.isclose(brier_scratch, brier_sk)


def test_stratified_kfold_scratch():
    # 70 negatives, 30 positives (30% positive prevalence)
    y = np.array([0] * 70 + [1] * 30)
    splits = list(stratified_kfold_scratch(y, n_splits=5, shuffle=True, seed=42))

    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) + len(val_idx) == 100
        # Positive prevalence in val_idx should be exactly 30% (6 / 20)
        pos_ratio = np.mean(y[val_idx])
        assert np.isclose(pos_ratio, 0.30, atol=0.05)
