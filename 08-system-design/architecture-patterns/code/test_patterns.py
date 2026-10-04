import pytest
from patterns import (
    CandidateItem,
    CandidateRetriever,
    FineRanker,
    DiversityReranker,
    AsyncInferenceQueue
)


def test_multi_stage_cascade():
    catalog = [
        CandidateItem(item_id="i1", category="tech", popularity_score=0.9, relevance_features=[1.0, 0.8]),
        CandidateItem(item_id="i2", category="tech", popularity_score=0.8, relevance_features=[0.9, 0.7]),
        CandidateItem(item_id="i3", category="tech", popularity_score=0.7, relevance_features=[0.8, 0.6]),
        CandidateItem(item_id="i4", category="news", popularity_score=0.95, relevance_features=[0.5, 0.5]),
        CandidateItem(item_id="i5", category="sports", popularity_score=0.6, relevance_features=[0.1, 0.1]),
    ]

    # Stage 1: Retrieve items in {"tech", "news"}
    retriever = CandidateRetriever(catalog)
    candidates = retriever.retrieve(preferred_categories={"tech", "news"}, top_k=4)
    assert len(candidates) == 4
    assert all(c.category in {"tech", "news"} for c in candidates)

    # Stage 2: Fine rank with user preferences
    ranker = FineRanker()
    user_features = [1.0, 0.9]
    scored = ranker.score_candidates(user_features, candidates)
    assert len(scored) == 4
    assert scored[0][1] >= scored[1][1]  # sorted descending

    # Stage 3: Diversity reranker (max 2 per category, top 3)
    final_items = DiversityReranker.rerank(scored, max_per_category=2, final_k=3)
    assert len(final_items) == 3
    tech_count = sum(1 for item, _ in final_items if item.category == "tech")
    assert tech_count <= 2


def test_async_inference_queue():
    def mock_worker(payload):
        return {"processed_length": len(payload.get("text", ""))}

    q = AsyncInferenceQueue(worker_fn=mock_worker)

    t1 = q.submit_task("task_001", {"text": "hello world"})
    assert t1.status == "PENDING"
    assert len(q.queue) == 1

    processed_task = q.process_next()
    assert processed_task.status == "COMPLETED"
    assert processed_task.result["processed_length"] == 11
    assert len(q.queue) == 0
