# Machine Learning System Design Interview Blueprints

End-to-end architectural frameworks, scale estimations, multi-stage retrieval, low-latency serving, and failure modes.

---

## 1. System Design Blueprint Framework
Every Senior/Staff ML System Design interview should follow this 7-step sequence:
1. **Clarify Requirements & Constraints** (Functional, Non-Functional, Scale, Latency SLA)
2. **High-Level Architectural Topology** (Offline training, nearline streaming, online inference)
3. **Data Pipeline & Feature Engineering** (Batch ETL, Kafka streaming, Feature Store point-in-time joins)
4. **Model Selection & Objective Function** (Two-Tower, GBDT, Transformer, Loss formulations)
5. **Serving Infrastructure & Latency Optimization** (Micro-batching, TensorRT, Caching tiers)
6. **Evaluation, Telemetry & Continuous Monitoring** (Online A/B testing, offline NDCG/AUC, drift)
7. **Failure Modes & Trade-Offs** (Graceful degradation, fallback policies, cold starts)

---

## 2. Classic Problem Deep Dives

### Q1: Design a Real-Time Recommendation Feed at 100,000 QPS
- **Tags**: `System Design` | `Practical`
- **Short Answer**: Implement a multi-stage funnel cascade: Stage 1 Candidate Retrieval (Two-Tower ANN vector search over 10M items down to 500 candidates in $< 10\text{ ms}$), Stage 2 Heavy Ranking (DLRM / DeepFM scoring 500 candidates down to 50 in $< 25\text{ ms}$), Stage 3 Diversity & Business Reranking (Maximal Marginal Relevance, deduplication, sponsored insertion in $< 5\text{ ms}$).
- **Architecture**:
  ```mermaid
  flowchart LR
      A[Client App Request] --> B[API Gateway / Load Balancer]
      B --> C[Stage 1: Two-Tower ANN Retrieval Top-500]
      C --> D[Stage 2: DLRM Fine Ranker Top-50]
      D --> E[Stage 3: MMR Diversity Reranker Top-10]
      E --> B
  ```
- **Scale Assumptions**:
  - $100,000\text{ QPS}$ peak.
  - $10,000,000$ active catalog items.
  - P99 Latency SLA: $< 50\text{ ms}$.
- **Key Trade-Off**: Two-Tower retrieval is fast because item embeddings can be precomputed offline and indexed in Milvus/HNSW, but it cannot model fine-grained dynamic feature cross-interactions (User Age $\times$ Item Category). The heavy ranker cross-network models these interactions, but is computationally restricted to $\le 500$ candidates.
- **Follow-Up Questions**:
  1. *How do you prevent popularity bias from monopolizing the user's feed?*
  2. *How do you handle real-time negative feedback (e.g. user skips video within 1 second)?*

---

### Q2: Design a Real-Time Payment Fraud Detection System at 50,000 TPS
- **Tags**: `System Design` | `Gotcha` | `Practical`
- **Short Answer**: Hybrid architecture combining a deterministic rules engine (sanctioned lists, velocity limits in $< 3\text{ ms}$) with compiled tree ensembles (Treelite GBDT scoring in $< 10\text{ ms}$) and an online Redis feature store providing sub-millisecond aggregations (e.g. `transactions_last_10m`).
- **Scale Assumptions**:
  - $50,000\text{ TPS}$, P99 latency $< 30\text{ ms}$.
  - Target False Positive Rate (FPR) $< 0.1\%$ to protect legitimate customer checkout conversion.
- **Failure Mode & Fallback**:
  - If feature store or model service times out ($> 25\text{ ms}$), fail-open with transaction approval flagged for post-settlement asynchronous human review, rather than blocking legitimate payment cardholders at the register.
