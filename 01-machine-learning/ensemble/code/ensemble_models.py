"""
Ensemble Methods for ML - Algorithms from Scratch
Implements Bagging, Random Forest, AdaBoost, Gradient Boosting,
Voting Classifiers, and Stacking Generalization.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaggingClassifierScratch:
    """
    Bootstrap Aggregating (Bagging) Classifier.
    Trains independent estimators on bootstrap samples (sampling with replacement)
    and computes Out-Of-Bag (OOB) error.
    """
    def __init__(self, n_estimators: int = 10, max_samples: float = 1.0, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.estimators: list[DecisionTreeClassifier] = []
        self.oob_indices: list[np.ndarray] = []
        self.oob_score_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifierScratch":
        np.random.seed(self.seed)
        n_samples = len(X)
        sample_size = int(self.max_samples * n_samples)

        self.estimators = []
        self.oob_indices = []
        oob_predictions: dict[int, list[Any]] = {i: [] for i in range(n_samples)}

        for _ in range(self.n_estimators):
            boot_idx = np.random.choice(n_samples, size=sample_size, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(boot_idx))

            clf = DecisionTreeClassifier(max_depth=5, random_state=self.seed)
            clf.fit(X[boot_idx], y[boot_idx])
            self.estimators.append(clf)
            self.oob_indices.append(oob_idx)

            for idx in oob_idx:
                pred = clf.predict(X[idx : idx + 1])[0]
                oob_predictions[idx].append(pred)

        # Calculate OOB accuracy
        correct = 0
        evaluated = 0
        for i, preds in oob_predictions.items():
            if len(preds) > 0:
                vals, counts = np.unique(preds, return_counts=True)
                maj = vals[np.argmax(counts)]
                if maj == y[i]:
                    correct += 1
                evaluated += 1

        self.oob_score_ = correct / max(evaluated, 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        all_preds = np.array([clf.predict(X) for clf in self.estimators])  # Shape (n_estimators, N)
        final_preds = []
        for j in range(X.shape[0]):
            vals, counts = np.unique(all_preds[:, j], return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds)


class AdaBoostClassifierScratch:
    """
    AdaBoost (Adaptive Boosting) for Binary Classification (labels in {-1, +1}).
    Uses decision stumps (depth=1 trees) as weak learners.
    Minimizes exponential loss: L(y, f(x)) = exp(-y * f(x)).
    """
    def __init__(self, n_estimators: int = 50, lr: float = 1.0):
        self.n_estimators = n_estimators
        self.lr = lr
        self.stumps: list[DecisionTreeClassifier] = []
        self.alphas: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifierScratch":
        n_samples = len(X)
        y_signed = np.where(y.ravel() <= 0, -1.0, 1.0)
        # Uniform sample weights
        w = np.ones(n_samples) / n_samples

        self.stumps = []
        self.alphas = []

        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y_signed, sample_weight=w)
            pred = stump.predict(X)

            # Weighted error rate
            err = np.sum(w[pred != y_signed])
            if err >= 0.5:
                # Invert or break if stump is no better than random guess
                break
            err = max(err, 1e-10)

            # Estimator weight alpha_m
            alpha = self.lr * 0.5 * np.log((1.0 - err) / err)

            # Weight update: w_i * exp(-alpha * y_i * h_m(x_i))
            w *= np.exp(-alpha * y_signed * pred)
            w /= np.sum(w)  # Normalize

            self.stumps.append(stump)
            self.alphas.append(alpha)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        stump_preds = np.array([alpha * stump.predict(X) for alpha, stump in zip(self.alphas, self.stumps)])
        total_pred = np.sum(stump_preds, axis=0)
        return np.where(total_pred >= 0, 1, 0)


class GradientBoostingRegressorScratch:
    """
    Gradient Tree Boosting for Regression (Squared Error Loss).
    Performs functional gradient descent in function space.
    Pseudo-residuals for MSE: r_im = y_i - F_{m-1}(x_i).
    """
    def __init__(self, n_estimators: int = 50, lr: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.initial_pred_: float = 0.0
        self.trees: list[DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressorScratch":
        y = y.astype(np.float64).ravel()
        self.initial_pred_ = float(np.mean(y))
        y_pred = np.full_like(y, self.initial_pred_)

        self.trees = []
        for _ in range(self.n_estimators):
            # Compute negative gradient (residuals)
            residuals = y - y_pred

            # Fit weak learner to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update predictions with shrinkage
            y_pred += self.lr * tree.predict(X)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(len(X), self.initial_pred_)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred


class StackingClassifierScratch:
    """
    Stacked Generalization (Stacking).
    Combines predictions of heterogeneous base models using an out-of-fold meta-learner.
    """
    def __init__(self, base_models: list[Any], meta_model: Any, n_splits: int = 3):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.fitted_base_models_: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingClassifierScratch":
        n_samples = len(X)
        k = len(self.base_models)
        meta_features = np.zeros((n_samples, k))

        # K-fold splits
        indices = np.random.permutation(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0

        for fold_idx in range(self.n_splits):
            val_idx = indices[current : current + fold_sizes[fold_idx]]
            train_idx = np.setdiff1d(indices, val_idx)
            current += fold_sizes[fold_idx]

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            for m_idx, model in enumerate(self.base_models):
                clone = type(model)(**model.get_params())
                clone.fit(X_tr, y_tr)
                meta_features[val_idx, m_idx] = clone.predict(X_val)

        # Fit final meta-model on out-of-fold features
        self.meta_model.fit(meta_features, y)

        # Fit all base models on full training data
        self.fitted_base_models_ = []
        for model in self.base_models:
            m = type(model)(**model.get_params())
            m.fit(X, y)
            self.fitted_base_models_.append(m)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        test_meta = np.column_stack([m.predict(X) for m in self.fitted_base_models_])
        return self.meta_model.predict(test_meta)
