"""Practical ML Coding Interview Implementations."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any

# 1. NumPy: Stable Softmax & Vectorized Pairwise Distance
def numerically_stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes softmax stably by subtracting the maximum value along the axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def pairwise_euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise squared Euclidean distances between arrays of shape (N, D) and (M, D).
    Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a * b^T.
    """
    a2 = np.sum(a ** 2, axis=1, keepdims=True)  # (N, 1)
    b2 = np.sum(b ** 2, axis=1, keepdims=True)  # (M, 1)
    d2 = a2 + b2.T - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(0.0, d2))

# 2. Pandas: Point-in-Time As-Of Join (Preventing Data Leakage)
def point_in_time_feature_join(events_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the most recent feature record prior to the event timestamp to prevent leakage.
    events_df: columns ['entity_id', 'event_time', ...]
    features_df: columns ['entity_id', 'feature_time', 'feature_value']
    """
    events_sorted = events_df.sort_values("event_time")
    features_sorted = features_df.sort_values("feature_time")
    merged = pd.merge_asof(
        events_sorted,
        features_sorted,
        left_on="event_time",
        right_on="feature_time",
        by="entity_id",
        direction="backward"
    )
    return merged

# 3. Scikit-Learn: Custom Outlier Clipping Transformer
class OutlierClipperTransformer:
    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: np.ndarray, y: Any = None):
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

# 4. PyTorch: Multi-Head Self-Attention from Scratch
class MultiHeadAttentionFromScratch(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b, seq_len, _ = x.shape
        # Project and reshape to (B, NumHeads, SeqLen, d_k)
        q = self.w_q(x).view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(out)

# 5. Core ML Algorithms: K-Means from Scratch
class VectorizedKMeans:
    def __init__(self, k: int = 3, max_iter: int = 50, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None

    def fit(self, X: np.ndarray):
        np.random.seed(self.seed)
        n_samples = X.shape[0]
        # Random initial centroids
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iter):
            # Compute distance to each centroid (N, K)
            dists = pairwise_euclidean_distance(X, self.centroids)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for c in range(self.k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = X[np.random.choice(n_samples)]
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = pairwise_euclidean_distance(X, self.centroids)
        return np.argmin(dists, axis=1)

# 6. Evaluation Metrics: ROC-AUC from Scratch
def calculate_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates ROC-AUC using Mann-Whitney U test formula."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank all scores
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = np.sum(ranks[pos_mask])
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

# 7. Embeddings & Vector Search: Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(ranking_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Combines multiple ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for r_list in ranking_lists:
        for rank, doc_id in enumerate(r_list, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# 8. LLM Applications: Token Bucket Rate Limiter
class TokenBucketLimiter:
    def __init__(self, capacity: int = 10, refill_per_sec: float = 5.0):
        self.capacity = capacity
        self.refill = refill_per_sec
        self.tokens = capacity
        self.last_ts = 0.0

    def allow(self, current_ts: float) -> bool:
        if self.last_ts > 0.0:
            elapsed = current_ts - self.last_ts
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill)
        self.last_ts = current_ts

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False
