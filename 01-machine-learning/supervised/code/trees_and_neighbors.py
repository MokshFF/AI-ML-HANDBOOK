"""
Supervised Learning - Trees, Neighbors, Naive Bayes & Support Vector Machines
Implements KNN, Gaussian Naive Bayes, CART Decision Tree, and Linear SVM from scratch.
"""

from __future__ import annotations
from typing import Any
import numpy as np


class KNNClassifierScratch:
    """
    k-Nearest Neighbors (k-NN) Classifier.
    Non-parametric lazy learner using Euclidean distance metric.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierScratch":
        self.X_train_ = X.astype(np.float64)
        self.y_train_ = y.ravel()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Model is not fitted.")
        X = X.astype(np.float64)
        predictions = []

        for x in X:
            # Vectorized Euclidean distance to all training samples
            distances = np.linalg.norm(self.X_train_ - x, axis=1)
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train_[k_indices]
            # Majority vote
            vals, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(vals[np.argmax(counts)])

        return np.array(predictions)


class GaussianNaiveBayesScratch:
    """
    Gaussian Naive Bayes Classifier.
    Assumes features are conditionally independent given class label:
    P(x_1, ..., x_D | y=c) = prod_j N(x_j; mu_cj, sigma_cj^2)
    """
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_priors_: dict[Any, float] = {}
        self.means_: dict[Any, np.ndarray] = {}
        self.variances_: dict[Any, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesScratch":
        X = X.astype(np.float64)
        y = y.ravel()
        self.classes_ = np.unique(y)
        n_samples = len(y)

        for c in self.classes_:
            X_c = X[y == c]
            self.class_priors_[c] = len(X_c) / n_samples
            self.means_[c] = np.mean(X_c, axis=0)
            self.variances_[c] = np.var(X_c, axis=0) + self.var_smoothing

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")
        X = X.astype(np.float64)
        predictions = []

        for x in X:
            posteriors = {}
            for c in self.classes_:
                prior_log = np.log(self.class_priors_[c])
                mu = self.means_[c]
                var = self.variances_[c]
                # Log likelihood of Gaussian: -0.5*log(2*pi*var) - ((x - mu)^2 / (2*var))
                log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(((x - mu) ** 2) / var)
                posteriors[c] = prior_log + log_prob
            # Select class with max log posterior
            best_class = max(posteriors, key=posteriors.get)  # type: ignore
            predictions.append(best_class)

        return np.array(predictions)


class TreeNode:
    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: float | None = None,
        left: "TreeNode | None" = None,
        right: "TreeNode | None" = None,
        value: Any | None = None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Binary Classification Tree using the CART algorithm with Gini Impurity.
    Gini(S) = 1 - sum_{c} p_c^2
    """
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: TreeNode | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        parent_gini = self._gini(y)
        best_gain = -1.0
        best_feat, best_thresh = None, None

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for t in thresholds:
                left_mask = X[:, feat_idx] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                w_left = np.sum(left_mask) / n_samples
                w_right = np.sum(right_mask) / n_samples
                gini_split = w_left * self._gini(y[left_mask]) + w_right * self._gini(y[right_mask])
                gain = parent_gini - gini_split

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = float(t)

        return best_feat, best_thresh

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]

        # Stopping criteria: pure leaf, max depth reached, or cannot split
        if len(unique_classes) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return TreeNode(value=majority_class)

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None or best_thresh is None:
            return TreeNode(value=majority_class)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_idx=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifierScratch":
        self.root = self._build_tree(X.astype(np.float64), y.ravel())
        return self

    def _predict_row(self, node: TreeNode, x: np.ndarray) -> Any:
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:  # type: ignore
            return self._predict_row(node.left, x)  # type: ignore
        return self._predict_row(node.right, x)  # type: ignore

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model is not fitted.")
        return np.array([self._predict_row(self.root, x) for x in X])


class LinearSVMScratch:
    """
    Linear Support Vector Machine via Subgradient Descent on Soft-Margin Hinge Loss.
    
    Objective: min_w (1/2) * ||w||_2^2 + C * sum max(0, 1 - y_i (w^T x_i + b))
    Labels y must be in {-1, +1}.
    """
    def __init__(self, C: float = 1.0, lr: float = 0.001, epochs: int = 1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        n_samples, n_features = X.shape
        X = X.astype(np.float64)
        # Convert {0, 1} to {-1, +1} if needed
        y_svm = np.where(y.ravel() <= 0, -1.0, 1.0)

        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                margin = y_svm[i] * (np.dot(X[i], self.weights_) + self.bias_)
                if margin < 1.0:
                    # Violated margin: subgradient includes hinge loss
                    self.weights_ -= self.lr * (self.weights_ - self.C * y_svm[i] * X[i])
                    self.bias_ -= self.lr * (-self.C * y_svm[i])
                else:
                    # Satisfied margin: subgradient only regularizer
                    self.weights_ -= self.lr * self.weights_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        decision = X @ self.weights_ + self.bias_
        return np.where(decision >= 0, 1, 0)
