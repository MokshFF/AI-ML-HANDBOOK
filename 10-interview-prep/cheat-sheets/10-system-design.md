# Cheat Sheet: ML System Design Quick Reference

| Design Dimension | Options & Rules of Thumb | Latency / Scale Budget |
| :--- | :--- | :--- |
| **Online vs Offline Inference** | Online: User-dependent features, dynamic context. Offline: Batch precompute and store in KV. | Online: $< 50\text{ ms}$. Offline: Batch daily/hourly. |
| **Multi-Stage Cascade** | Stage 1 (Retrieval): 10M $\to$ 500 items via ANN. Stage 2 (Ranking): 500 $\to$ 50. Stage 3 (Rerank): 50 $\to$ 10. | Stage 1: $< 10\text{ ms}$. Stage 2: $< 25\text{ ms}$. Stage 3: $< 5\text{ ms}$. |
| **Storage Topology** | Hot (Redis / RAM): In-memory feature vectors. Warm (PostgreSQL/Mongo): User metadata. Cold (S3): Event logs. | Hot: $< 2\text{ ms}$. Warm: $< 10\text{ ms}$. Cold: Batch queries. |
| **Caching Tiers** | L1: Application in-memory cache. L2: Distributed Redis cache. Semantic: Embedding cosine similarity cache. | L1: $< 100\, \mu\text{s}$. L2: $< 2\text{ ms}$. Semantic: $< 10\text{ ms}$. |
| **Availability & Fallbacks**| Circuit breaker: If model service P99 $> 40\text{ ms}$, fallback to popular items or cached recommendations. | Zero downtime; fail open with graceful degradation. |
