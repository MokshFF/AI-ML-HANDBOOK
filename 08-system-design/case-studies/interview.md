# ML System Design Interview Framework & Questions

### 1. The 45-Minute ML System Design Interview Framework
When answering ML system design questions, follow this structured pacing:
- **Phase 1: Scope & Clarify Requirements (5 mins)**:
  - Clarify business goals, positive/negative engagement signals, inputs/outputs.
  - Establish non-functional requirements (latency SLA, throughput QPS, availability).
  - Calculate back-of-the-envelope scale (DAU, QPS, storage footprint).
- **Phase 2: High-Level Architecture (10 mins)**:
  - Draw end-to-end dataflow diagram: Ingestion -> Training -> Serving -> Client.
  - Distinguish offline training pipelines from online real-time inference loops.
- **Phase 3: Deep Dive into Core ML Components (15 mins)**:
  - Feature engineering & Feature Store (Online vs Offline, point-in-time joins).
  - Model architecture choices (e.g., Two-Tower vs Cross-Encoder, GBDT vs DNN).
  - Loss functions and evaluation metrics (offline AUC/nDCG vs online CTR/conversion).
- **Phase 4: Serving, Scaling & Trade-Offs (10 mins)**:
  - Multi-stage cascades (Candidate Retrieval -> Fine Ranking -> Diversity Reranking).
  - Caching strategies, model quantization, and autoscaling.
  - Failure modes, cold-start handling, and privacy/compliance.
- **Phase 5: Wrap-up & Monitoring (5 mins)**:
  - Data drift and concept drift detection (KS-test, PSI).
  - Telemetry (Prometheus, Grafana, A/B testing).

---

### Q1: In a recommendation system, why do we use a Two-Tower model for retrieval rather than a single unified model?
**Answer:**
1. **Mathematical Decoupling**: A Two-Tower model computes $u = f_{\text{user}}(X_{\text{user}})$ and $v = g_{\text{item}}(X_{\text{item}})$, and the final score is the inner product $\langle u, v \rangle$.
2. **Offline Precomputation**: Because the item tower does not depend on the user, embeddings for all 10 million items in the catalog can be precomputed and indexed in a vector database (e.g., Milvus, HNSW) offline.
3. **Sub-Linear Search**: At query time, the system only needs to evaluate the user tower once ($5\text{ ms}$), and then executes Approximate Nearest Neighbor (ANN) search to find the top 500 items in $O(\log N)$ time ($\approx 8\text{ ms}$).
4. In contrast, a unified model (e.g. cross-encoder) concatenates user and item features $[X_u, X_i]$, requiring 10 million forward passes per request, which is computationally impossible within real-time latency budgets.

---

### Q2: How do you design an ML system to handle extreme class imbalance in real-time fraud detection (e.g. 0.05% positive rate)?
**Answer:**
1. **Loss Function Calibration**: Use Focal Loss or weighted cross-entropy:
   $$\mathcal{L} = -w_1 \cdot y \log(p) - w_0 \cdot (1 - y) \log(1 - p)$$
   where $w_1 \gg w_0$ heavily penalizes false negatives (missed fraud).
2. **Evaluation Metric Selection**: Never use accuracy or ROC-AUC (which are inflated by the overwhelming negative class). Use **Precision-Recall AUC (PR-AUC)**, Precision@Top-$K$, and False Positive Rate at $95\%$ Recall.
3. **Subsampling / Downsampling**: Downsample the majority negative class (e.g. keep all fraud cases and 10% of legitimate transactions) and mathematically adjust the predicted probabilities using the Bayes odds ratio adjustment:
   $$p_{\text{calibrated}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{\beta}}$$
   where $\beta$ is the sampling ratio.
4. **Ensemble Architecture**: Combine an aggressive high-recall GBDT model with deterministic velocity rules and a 2FA challenge gate.
