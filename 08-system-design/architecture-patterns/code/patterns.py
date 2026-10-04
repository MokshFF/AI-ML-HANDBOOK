"""
ML System Design Architecture Patterns:
1. Multi-Stage Cascade Ranking Pattern (Candidate Retrieval -> Fine Ranking -> Diversity Reranking).
2. Asynchronous Event-Driven Inference Queue with Decoupled Workers.
3. Feature Cache with TTL and Fallback Defaults.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


# --------------------------------------------------------------------------- #
# 1. Multi-Stage Cascade Ranking Pattern
# --------------------------------------------------------------------------- #
@dataclass
class CandidateItem:
    item_id: str
    category: str
    popularity_score: float
    relevance_features: List[float] = field(default_factory=list)


class CandidateRetriever:
    """Stage 1: Fast, lightweight retrieval from millions of items down to hundreds."""
    def __init__(self, catalog: Sequence[CandidateItem]):
        self.catalog = list(catalog)

    def retrieve(self, preferred_categories: Set[str], top_k: int = 50) -> List[CandidateItem]:
        # Filter by category and rank by coarse score
        filtered = [item for item in self.catalog if item.category in preferred_categories]
        filtered.sort(key=lambda x: x.popularity_score, reverse=True)
        return filtered[:top_k]


class FineRanker:
    """Stage 2: High-capacity scoring model evaluating user-item feature interactions."""
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights or [0.6, 0.4]

    def score_candidates(self, user_features: List[float], candidates: Sequence[CandidateItem]) -> List[Tuple[CandidateItem, float]]:
        scored = []
        for item in candidates:
            # Simulated model score = dot product of user & item features + popularity
            interaction = sum(u * i for u, i in zip(user_features, item.relevance_features)) if item.relevance_features else 0.0
            score = 0.7 * interaction + 0.3 * item.popularity_score
            scored.append((item, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class DiversityReranker:
    """Stage 3: Business rules, deduplication, and category diversity interleaving."""
    @staticmethod
    def rerank(scored_items: Sequence[Tuple[CandidateItem, float]], max_per_category: int = 2, final_k: int = 5) -> List[Tuple[CandidateItem, float]]:
        selected = []
        cat_counts: Dict[str, int] = {}
        for item, score in scored_items:
            c = item.category
            if cat_counts.get(c, 0) < max_per_category:
                selected.append((item, score))
                cat_counts[c] = cat_counts.get(c, 0) + 1
            if len(selected) >= final_k:
                break
        return selected


# --------------------------------------------------------------------------- #
# 2. Asynchronous Event-Driven Inference Pattern
# --------------------------------------------------------------------------- #
@dataclass
class InferenceTask:
    task_id: str
    payload: Dict[str, Any]
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED
    result: Optional[Any] = None
    created_at: float = field(default_factory=time.time)


class AsyncInferenceQueue:
    """Decouples client ingestion from heavy model compute via a message queue pattern."""
    def __init__(self, worker_fn: Callable[[Dict[str, Any]], Any]):
        self.worker_fn = worker_fn
        self.tasks: Dict[str, InferenceTask] = {}
        self.queue: List[str] = []

    def submit_task(self, task_id: str, payload: Dict[str, Any]) -> InferenceTask:
        task = InferenceTask(task_id=task_id, payload=payload)
        self.tasks[task_id] = task
        self.queue.append(task_id)
        return task

    def process_next(self) -> Optional[InferenceTask]:
        if not self.queue:
            return None
        tid = self.queue.pop(0)
        task = self.tasks[tid]
        task.status = "PROCESSING"
        try:
            task.result = self.worker_fn(task.payload)
            task.status = "COMPLETED"
        except Exception as e:
            task.status = "FAILED"
            task.result = str(e)
        return task

    def get_task_status(self, task_id: str) -> Optional[InferenceTask]:
        return self.tasks.get(task_id)
