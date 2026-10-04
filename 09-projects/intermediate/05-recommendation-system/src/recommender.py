"""Two-Stage Collaborative Filtering & Ranking Recommender."""
import numpy as np
from typing import List, Dict, Tuple

class TwoStageRecommender:
    def __init__(self, n_users: int = 200, n_items: int = 100, embed_dim: int = 16):
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        np.random.seed(42)
        self.user_embeds = np.random.normal(0, 0.1, (n_users, embed_dim))
        self.item_embeds = np.random.normal(0, 0.1, (n_items, embed_dim))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

    def fit(self, interactions: List[Tuple[int, int, float]], lr: float = 0.05, epochs: int = 25):
        for _ in range(epochs):
            for u, i, r in interactions:
                pred = (
                    np.dot(self.user_embeds[u], self.item_embeds[i]) 
                    + self.user_bias[u] 
                    + self.item_bias[i]
                )
                err = r - pred
                
                # Update biases
                self.user_bias[u] += lr * (err - 0.02 * self.user_bias[u])
                self.item_bias[i] += lr * (err - 0.02 * self.item_bias[i])
                
                # Update embeddings
                u_emb = self.user_embeds[u].copy()
                self.user_embeds[u] += lr * (err * self.item_embeds[i] - 0.02 * self.user_embeds[u])
                self.item_embeds[i] += lr * (err * u_emb - 0.02 * self.item_embeds[i])

    def recommend(self, user_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        scores = (
            self.item_embeds @ self.user_embeds[user_id] 
            + self.user_bias[user_id] 
            + self.item_bias
        )
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]

    def evaluate_ndcg(self, test_interactions: Dict[int, List[int]], k: int = 5) -> float:
        ndcg_list = []
        for u, true_items in test_interactions.items():
            if not true_items:
                continue
            recs = [item_id for item_id, _ in self.recommend(u, top_k=k)]
            dcg = 0.0
            for rank, item_id in enumerate(recs):
                if item_id in true_items:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(true_items), k)))
            ndcg_list.append(dcg / (idcg + 1e-8))
        return float(np.mean(ndcg_list))

def generate_mock_interactions(n_users: int = 100, n_items: int = 50, n_records: int = 1000):
    np.random.seed(42)
    interactions = []
    for _ in range(n_records):
        u = np.random.randint(0, n_users)
        i = np.random.randint(0, n_items)
        r = float(np.random.choice([1.0, 0.0], p=[0.7, 0.3]))
        interactions.append((u, i, r))
    return interactions

if __name__ == "__main__":
    records = generate_mock_interactions()
    train_records = records[:800]
    
    rec = TwoStageRecommender(n_users=100, n_items=50)
    rec.fit(train_records)
    user_0_recs = rec.recommend(0, top_k=5)
    print("User 0 Top 5 Recommendations:", user_0_recs)
