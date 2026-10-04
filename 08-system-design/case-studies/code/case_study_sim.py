"""
Executable Case Study Simulations:
1. End-to-end Multi-Stage Recommendation Engine (Collaborative Filtering Retrieval + Gradient Boosted Scoring).
2. Real-Time Transaction Fraud Detection Engine (Rule-based heuristics + ML score combination).
3. Hybrid Search Ranker (BM25 lexical + dense semantic embedding fusion).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# 1. Recommendation System Simulator
# --------------------------------------------------------------------------- #
@dataclass
class CatalogItem:
    item_id: str
    tags: List[str]
    popularity: float
    embedding: List[float]


class RecommendationSystemEngine:
    """Simulates a multi-stage recommendation system: Retrieval -> Scoring -> Post-processing."""
    def __init__(self, catalog: List[CatalogItem]):
        self.catalog = catalog

    def candidate_generation(self, user_liked_tags: List[str], top_k: int = 10) -> List[CatalogItem]:
        # Fast tag overlap retrieval
        scored = []
        user_tags_set = set(user_liked_tags)
        for item in self.catalog:
            overlap = len(user_tags_set & set(item.tags))
            score = overlap * 2.0 + item.popularity
            scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:top_k]]

    def rank_candidates(self, user_embedding: List[float], candidates: List[CatalogItem]) -> List[Tuple[CatalogItem, float]]:
        # Fine ranking using cosine similarity + popularity
        ranked = []
        for item in candidates:
            dot = sum(u * v for u, v in zip(user_embedding, item.embedding))
            norm_u = math.sqrt(sum(u * u for u in user_embedding)) or 1.0
            norm_v = math.sqrt(sum(v * v for v in item.embedding)) or 1.0
            cos_sim = dot / (norm_u * norm_v)
            final_score = 0.8 * cos_sim + 0.2 * item.popularity
            ranked.append((item, round(final_score, 4)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# --------------------------------------------------------------------------- #
# 2. Real-Time Fraud Detection Engine
# --------------------------------------------------------------------------- #
@dataclass
class Transaction:
    txn_id: str
    user_id: str
    amount: float
    country: str
    is_foreign: bool
    velocity_1h: int  # count of transactions in last 1 hour


class FraudDetectionEngine:
    """Combines deterministic hard rules with an ML risk scoring model."""
    def __init__(self, high_risk_countries: Optional[set] = None):
        self.high_risk_countries = high_risk_countries or {"XX", "ZZ"}

    def evaluate_transaction(self, txn: Transaction) -> Dict[str, Any]:
        # 1. Hard Rule Checks (Immediate Rejection)
        if txn.country in self.high_risk_countries:
            return {"action": "BLOCK", "reason": "High-risk jurisdiction", "risk_score": 1.0}
        if txn.amount > 10000.0 and txn.velocity_1h > 5:
            return {"action": "BLOCK", "reason": "Extreme velocity and amount", "risk_score": 0.98}

        # 2. ML Risk Scoring Model (Simulated Logistic Model)
        # Features: normalized amount, velocity, foreign flag
        w_amt, w_vel, w_foreign, bias = 0.001, 0.15, 0.4, -1.5
        z = w_amt * txn.amount + w_vel * txn.velocity_1h + (w_foreign if txn.is_foreign else 0.0) + bias
        risk_score = 1.0 / (1.0 + math.exp(-z))
        risk_score = round(risk_score, 4)

        if risk_score >= 0.90:
            action = "BLOCK"
        elif risk_score >= 0.70:
            action = "CHALLENGE"  # Trigger 2FA / OTP verification
        else:
            action = "APPROVE"

        return {"action": action, "risk_score": risk_score, "reason": "ML scoring threshold"}


# --------------------------------------------------------------------------- #
# 3. Hybrid Search Engine Simulator
# --------------------------------------------------------------------------- #
class HybridSearchEngine:
    """Combines BM25 lexical keyword matching with dense semantic embeddings."""
    def __init__(self, documents: Dict[str, str], doc_embeddings: Dict[str, List[float]]):
        self.documents = documents
        self.doc_embeddings = doc_embeddings

    def search(self, query_terms: List[str], query_vector: List[float], alpha: float = 0.5, top_k: int = 3) -> List[Tuple[str, float]]:
        scores = {}
        for doc_id, text in self.documents.items():
            # Lexical term frequency
            words = text.lower().split()
            lex_score = sum(words.count(term.lower()) for term in query_terms) / (len(words) or 1)

            # Semantic cosine similarity
            emb = self.doc_embeddings.get(doc_id, [0.0] * len(query_vector))
            dot = sum(q * d for q, d in zip(query_vector, emb))
            norm_q = math.sqrt(sum(q * q for q in query_vector)) or 1.0
            norm_d = math.sqrt(sum(d * d for d in emb)) or 1.0
            sem_score = max(0.0, dot / (norm_q * norm_d))

            hybrid = alpha * sem_score + (1.0 - alpha) * lex_score
            scores[doc_id] = round(hybrid, 4)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
