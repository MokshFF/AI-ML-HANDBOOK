"""Tests for Practical Coding Solutions."""
import numpy as np
import pandas as pd
import torch
from coding_solutions import (
    numerically_stable_softmax,
    pairwise_euclidean_distance,
    point_in_time_feature_join,
    OutlierClipperTransformer,
    MultiHeadAttentionFromScratch,
    VectorizedKMeans,
    calculate_roc_auc,
    reciprocal_rank_fusion,
    TokenBucketLimiter
)

def test_stable_softmax():
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    probs = numerically_stable_softmax(logits)
    assert not np.isnan(probs).any()
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]

def test_pairwise_distance():
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0]])
    dist = pairwise_euclidean_distance(a, b)
    assert np.isclose(dist[0, 0], 0.0)
    assert np.isclose(dist[1, 0], 5.0)

def test_asof_join():
    events = pd.DataFrame({
        "entity_id": ["u1", "u1"],
        "event_time": [100, 250]
    })
    features = pd.DataFrame({
        "entity_id": ["u1", "u1"],
        "feature_time": [80, 200],
        "risk_score": [0.2, 0.8]
    })
    merged = point_in_time_feature_join(events, features)
    assert merged.loc[0, "risk_score"] == 0.2
    assert merged.loc[1, "risk_score"] == 0.8

def test_outlier_clipper():
    X = np.array([[-100.0], [1.0], [2.0], [3.0], [500.0]])
    clipper = OutlierClipperTransformer(lower_percentile=10, upper_percentile=90)
    clipped = clipper.fit_transform(X)
    assert clipped.min() > -100.0
    assert clipped.max() < 500.0

def test_multi_head_attention():
    mha = MultiHeadAttentionFromScratch(d_model=32, num_heads=4)
    x = torch.randn(2, 8, 32)
    out = mha(x)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()

def test_kmeans():
    X = np.vstack([
        np.random.normal(0, 0.2, (20, 2)),
        np.random.normal(5, 0.2, (20, 2))
    ])
    km = VectorizedKMeans(k=2, seed=42)
    km.fit(X)
    preds = km.predict(X)
    assert len(set(preds)) == 2

def test_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_score)
    assert 0.5 < auc <= 1.0

def test_rrf():
    list1 = ["docA", "docB", "docC"]
    list2 = ["docB", "docA", "docD"]
    fused = reciprocal_rank_fusion([list1, list2])
    # docA and docB appear in top 2 in both lists
    top_docs = [d for d, _ in fused[:2]]
    assert "docA" in top_docs and "docB" in top_docs

def test_rate_limiter():
    limiter = TokenBucketLimiter(capacity=2, refill_per_sec=1.0)
    assert limiter.allow(current_ts=1.0) is True
    assert limiter.allow(current_ts=1.0) is True
    assert limiter.allow(current_ts=1.0) is False
    assert limiter.allow(current_ts=2.1) is True
