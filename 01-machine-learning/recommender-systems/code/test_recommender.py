"""
Tests for Recommender Systems models and ranking metrics.
"""

import numpy as np
import pytest
from recommender_models import (
    MatrixFactorizationSVD,
    ContentBasedRecommender,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
)


def test_matrix_factorization_svd():
    # 5 users x 4 items rating matrix (1-5 scale) with some missing entries (NaN)
    R = np.array([
        [5.0, 3.0, np.nan, 1.0],
        [4.0, np.nan, np.nan, 1.0],
        [1.0, 1.0, 5.0, 4.0],
        [np.nan, 1.0, 4.0, 5.0],
        [5.0, 4.0, 1.0, 2.0]
    ])

    mf = MatrixFactorizationSVD(n_factors=3, epochs=60, lr=0.02, seed=42).fit(R)
    pred_0_0 = mf.predict(0, 0)
    assert 3.5 <= pred_0_0 <= 5.5

    # Check that missing prediction is reasonable
    pred_missing = mf.predict(0, 2)
    assert 1.0 <= pred_missing <= 5.0

    all_preds = mf.predict_all()
    assert all_preds.shape == (5, 4)


def test_content_based_recommender():
    # 4 items x 3 features (Action, Romance, SciFi)
    item_features = np.array([
        [1.0, 0.0, 1.0],  # Item 0: SciFi Action
        [0.0, 1.0, 0.0],  # Item 1: Romance
        [1.0, 0.0, 0.0],  # Item 2: Action
        [0.0, 1.0, 0.1]   # Item 3: Romance Drama
    ])
    # User 0 likes Item 0 and 2 (Action/SciFi fan)
    interactions = np.array([
        [5.0, 0.0, 4.0, 0.0]
    ])

    cb = ContentBasedRecommender().fit(item_features, interactions)
    recs = cb.recommend(user_idx=0, top_k=2)

    # Should recommend Action/SciFi items first
    assert recs[0] in [0, 2]
    assert recs[1] in [0, 2]


def test_ranking_metrics():
    actual = [1, 3, 5]
    predicted = [1, 2, 3, 4, 5]

    # Precision@3: [1, 2, 3] -> hits are 1 and 3 -> 2/3
    assert np.isclose(precision_at_k(actual, predicted, k=3), 2.0 / 3.0)

    # Recall@3: 2 hits out of 3 actual -> 2/3
    assert np.isclose(recall_at_k(actual, predicted, k=3), 2.0 / 3.0)

    # NDCG@3: DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 = 2.1309
    ndcg = ndcg_at_k(actual, predicted, k=3)
    assert 0.65 <= ndcg <= 0.75

    # MRR: First hit at rank 1 -> MRR = 1/1 = 1.0
    assert np.isclose(mean_reciprocal_rank(actual, predicted), 1.0)

    # First hit at rank 2
    assert np.isclose(mean_reciprocal_rank(actual, [9, 3, 1]), 0.5)
