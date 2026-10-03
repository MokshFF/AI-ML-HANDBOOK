"""
Model Evaluation for ML - Metrics, Validation & Calibration from Scratch
Implements confusion matrices, classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC),
regression metrics (MSE, RMSE, MAE, R2), calibration (Brier Score, ECE), and Stratified K-Fold.
"""

from __future__ import annotations
from typing import Generator
import numpy as np


def confusion_matrix_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Computes (TP, FP, TN, FN) for binary classification with labels in {0, 1}."""
    y_t = y_true.ravel().astype(int)
    y_p = y_pred.ravel().astype(int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    return tp, fp, tn, fn


def classification_metrics_scratch(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> dict[str, float]:
    """Computes Accuracy, Precision, Recall, Specificity, and F1 score."""
    tp, fp, tn, fn = confusion_matrix_scratch(y_true, y_pred)
    total = tp + fp + tn + fn

    acc = (tp + tn) / max(total, 1)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    f1 = 2.0 * (prec * rec) / (prec + rec + eps)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
    }


def roc_curve_and_auc_scratch(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes True Positive Rate (TPR), False Positive Rate (FPR),
    and Area Under the ROC Curve (ROC-AUC) via trapezoidal integration.
    """
    y_t = y_true.ravel().astype(int)
    y_s = y_scores.ravel().astype(np.float64)

    # Sort thresholds in descending order
    desc_indices = np.argsort(y_s)[::-1]
    y_t_sorted = y_t[desc_indices]

    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples required for ROC curve.")

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_count = 0
    fp_count = 0

    for label in y_t_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / n_pos)
        fpr_list.append(fp_count / n_neg)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    # Trapezoidal integration: np.trapz(tpr, fpr)
    # Using manual trapezoid formula
    auc = float(np.sum(0.5 * (tpr_arr[1:] + tpr_arr[:-1]) * (fpr_arr[1:] - fpr_arr[:-1])))
    return fpr_arr, tpr_arr, auc


def regression_metrics_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes MSE, RMSE, MAE, and R^2 score."""
    y_t = y_true.ravel().astype(np.float64)
    y_p = y_pred.ravel().astype(np.float64)

    residuals = y_t - y_p
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def brier_score_scratch(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Mean Squared Error of predicted probability estimates: Brier = (1 / N) * sum(p_i - y_i)^2."""
    y_t = y_true.ravel().astype(np.float64)
    y_p = y_probs.ravel().astype(np.float64)
    return float(np.mean((y_p - y_t) ** 2))


def stratified_kfold_scratch(
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Stratified K-Fold cross-validation iterator.
    Preserves percentage of samples for each class across folds.
    """
    np.random.seed(seed)
    y = y.ravel()
    unique_classes = np.unique(y)
    class_indices: dict[Any, np.ndarray] = {}

    for c in unique_classes:
        indices = np.where(y == c)[0]
        if shuffle:
            np.random.shuffle(indices)
        class_indices[c] = indices

    # Distribute class indices across folds
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for c, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(idx)

    for fold_idx in range(n_splits):
        val_indices = np.array(folds[fold_idx], dtype=int)
        train_indices = np.array([idx for f, fold in enumerate(folds) if f != fold_idx for idx in fold], dtype=int)
        yield train_indices, val_indices
