import pytest
from case_study_sim import (
    CatalogItem,
    RecommendationSystemEngine,
    Transaction,
    FraudDetectionEngine,
    HybridSearchEngine
)


def test_recommendation_system():
    catalog = [
        CatalogItem("movie_1", ["action", "sci-fi"], 0.8, [1.0, 0.0]),
        CatalogItem("movie_2", ["romance", "comedy"], 0.6, [0.0, 1.0]),
        CatalogItem("movie_3", ["action", "thriller"], 0.9, [0.9, 0.1]),
    ]
    rec_engine = RecommendationSystemEngine(catalog)

    # User likes action
    candidates = rec_engine.candidate_generation(["action"], top_k=2)
    assert len(candidates) == 2
    assert "action" in candidates[0].tags

    # Rank with user embedding preferring action [1.0, 0.0]
    ranked = rec_engine.rank_candidates([1.0, 0.0], candidates)
    assert len(ranked) == 2
    assert {ranked[0][0].item_id, ranked[1][0].item_id} == {"movie_1", "movie_3"}


def test_fraud_detection_engine():
    engine = FraudDetectionEngine()

    # Normal transaction -> APPROVE
    t_normal = Transaction("tx1", "u1", amount=45.0, country="US", is_foreign=False, velocity_1h=1)
    res_normal = engine.evaluate_transaction(t_normal)
    assert res_normal["action"] == "APPROVE"
    assert res_normal["risk_score"] < 0.80

    # High velocity suspicious transaction -> CHALLENGE
    t_suspicious = Transaction("tx2", "u2", amount=850.0, country="US", is_foreign=True, velocity_1h=8)
    res_susp = engine.evaluate_transaction(t_suspicious)
    assert res_susp["action"] in {"CHALLENGE", "BLOCK"}

    # Hard rule block -> High-risk country
    t_blocked = Transaction("tx3", "u3", amount=10.0, country="XX", is_foreign=True, velocity_1h=1)
    res_blocked = engine.evaluate_transaction(t_blocked)
    assert res_blocked["action"] == "BLOCK"
    assert res_blocked["risk_score"] == 1.0


def test_hybrid_search():
    docs = {
        "d1": "deep learning transformer attention models",
        "d2": "classical decision tree random forest algorithms",
        "d3": "deep neural network training optimization"
    }
    embs = {
        "d1": [1.0, 0.8],
        "d2": [0.0, 0.1],
        "d3": [0.9, 0.7]
    }
    searcher = HybridSearchEngine(docs, embs)

    # Query matching d1 semantically and lexically
    results = searcher.search(["transformer"], [1.0, 0.8], alpha=0.5, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "d1"
