"""Tests for Two-Stage Recommender."""
from recommender import TwoStageRecommender, generate_mock_interactions

def test_recommender_pipeline():
    records = generate_mock_interactions(50, 30, 500)
    rec = TwoStageRecommender(n_users=50, n_items=30, embed_dim=8)
    rec.fit(records, epochs=10)
    
    recs = rec.recommend(0, top_k=5)
    assert len(recs) == 5
    assert all(isinstance(idx, int) for idx, _ in recs)
    
    test_dict = {0: [recs[0][0]]}
    ndcg = rec.evaluate_ndcg(test_dict, k=5)
    assert ndcg > 0.0
