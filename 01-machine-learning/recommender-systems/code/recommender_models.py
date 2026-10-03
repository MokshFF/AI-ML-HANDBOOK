"""
Recommender Systems for ML - Algorithms & Evaluation from Scratch
Implements Collaborative Filtering, Matrix Factorization (FunkSVD), Content-Based Filtering,
and Ranking Evaluation Metrics (Precision@K, Recall@K, MAP@K, NDCG@K, MRR).
"""

from __future__ import annotations
import numpy as np


class MatrixFactorizationSVD:
    """
    Regularized Matrix Factorization with User and Item Biases (FunkSVD).
    Model: r_hat_{ui} = mu + b_u + b_i + p_u^T q_i
    Objective: min sum (r_{ui} - r_hat_{ui})^2 + lambda * (b_u^2 + b_i^2 + ||p_u||^2 + ||q_i||^2)
    """
    def __init__(self, n_factors: int = 10, lr: float = 0.01, reg: float = 0.02, epochs: int = 50, seed: int = 42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed
        self.global_mean_: float = 0.0
        self.user_biases_: np.ndarray | None = None
        self.item_biases_: np.ndarray | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None

    def fit(self, R: np.ndarray) -> "MatrixFactorizationSVD":
        """R: User-Item rating matrix of shape (n_users, n_items). NaNs represent unobserved ratings."""
        np.random.seed(self.seed)
        n_users, n_items = R.shape

        observed_mask = ~np.isnan(R)
        self.global_mean_ = float(np.mean(R[observed_mask]))

        self.user_biases_ = np.zeros(n_users, dtype=np.float64)
        self.item_biases_ = np.zeros(n_items, dtype=np.float64)
        self.user_factors_ = np.random.normal(0, 0.1, size=(n_users, self.n_factors))
        self.item_factors_ = np.random.normal(0, 0.1, size=(n_items, self.n_factors))

        users, items = np.where(observed_mask)

        for _ in range(self.epochs):
            for u, i in zip(users, items):
                rating = R[u, i]
                pred = (
                    self.global_mean_
                    + self.user_biases_[u]
                    + self.item_biases_[i]
                    + np.dot(self.user_factors_[u], self.item_factors_[i])
                )
                err = rating - pred

                # Update biases
                self.user_biases_[u] += self.lr * (err - self.reg * self.user_biases_[u])
                self.item_biases_[i] += self.lr * (err - self.reg * self.item_biases_[i])

                # Update latent factors
                p_u_old = self.user_factors_[u].copy()
                self.user_factors_[u] += self.lr * (err * self.item_factors_[i] - self.reg * self.user_factors_[u])
                self.item_factors_[i] += self.lr * (err * p_u_old - self.reg * self.item_factors_[i])

        return self

    def predict(self, u: int, i: int) -> float:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("Model is not fitted.")
        pred = (
            self.global_mean_
            + self.user_biases_[u]
            + self.item_biases_[i]
            + np.dot(self.user_factors_[u], self.item_factors_[i])
        )
        return float(np.clip(pred, 1.0, 5.0))

    def predict_all(self) -> np.ndarray:
        return (
            self.global_mean_
            + self.user_biases_[:, np.newaxis]
            + self.item_biases_[np.newaxis, :]
            + (self.user_factors_ @ self.item_factors_.T)
        )


class ContentBasedRecommender:
    """
    Content-Based Filtering using Item Feature Profiles and User Taste Vectors.
    """
    def __init__(self):
        self.item_profiles_: np.ndarray | None = None
        self.user_profiles_: np.ndarray | None = None

    def fit(self, item_features: np.ndarray, user_item_interactions: np.ndarray) -> "ContentBasedRecommender":
        """
        item_features: (n_items, n_features)
        user_item_interactions: (n_users, n_items) binary or interaction weights
        """
        self.item_profiles_ = item_features.astype(np.float64)
        # User profile is interaction-weighted average of item feature vectors
        interactions = np.nan_to_num(user_item_interactions, nan=0.0)
        user_weights = interactions.astype(np.float64)
        row_sums = user_weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.user_profiles_ = (user_weights @ self.item_profiles_) / row_sums
        return self

    def recommend(self, user_idx: int, top_k: int = 5) -> list[int]:
        user_vec = self.user_profiles_[user_idx]
        scores = self.item_profiles_ @ user_vec
        return list(np.argsort(scores)[::-1][:top_k])


# ---------------------------------------------------------
# Ranking & Retrieval Metrics
# ---------------------------------------------------------

def precision_at_k(actual: list[int] | set[int], predicted: list[int], k: int) -> float:
    """Fraction of recommended items in top-k that are relevant."""
    pred_k = predicted[:k]
    if len(pred_k) == 0:
        return 0.0
    relevant_retrieved = len(set(pred_k) & set(actual))
    return float(relevant_retrieved / k)


def recall_at_k(actual: list[int] | set[int], predicted: list[int], k: int) -> float:
    """Fraction of relevant items that are successfully retrieved in top-k."""
    if len(actual) == 0:
        return 0.0
    pred_k = predicted[:k]
    relevant_retrieved = len(set(pred_k) & set(actual))
    return float(relevant_retrieved / len(actual))


def ndcg_at_k(actual: list[int] | set[int], predicted: list[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    actual_set = set(actual)
    pred_k = predicted[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k, 1):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 1)

    # Ideal DCG (all relevant items ranked at the top)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mean_reciprocal_rank(actual: list[int] | set[int], predicted: list[int]) -> float:
    """MRR: 1 / rank of the first relevant item in predicted list."""
    actual_set = set(actual)
    for rank, item in enumerate(predicted, 1):
        if item in actual_set:
            return 1.0 / rank
    return 0.0
