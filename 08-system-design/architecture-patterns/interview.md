# ML Architecture Patterns Interview Questions & Answers

### Q1: Why do modern recommendation and search systems use a multi-stage cascade rather than a single end-to-end model?
**Answer:**
1. **Computational Infeasibility**: A catalog may contain 100 million items. Evaluating a 50-layer deep neural network with 100 features for every user against 100 million items requires $10^{10}$ forward passes per query, which is physically impossible within a 50 ms latency window.
2. **Efficiency via Hierarchy**:
   - *Retrieval Stage*: Uses fast approximate nearest neighbor (ANN) search over precomputed item embeddings in $O(\log N)$ time, eliminating 99.99% of irrelevant items.
   - *Ranking Stage*: Spends heavy GPU compute only on the top 500-1000 plausible candidates using expressive user-item interaction cross-features.
   - *Reranking Stage*: Enforces non-differentiable business rules (diversity, fairness, ad insertion) that cannot be cleanly expressed inside gradient descent loss functions.

---

### Q2: What is the difference between Lambda Architecture and Kappa Architecture in ML systems?
**Answer:**
- **Lambda Architecture**:
  - Maintains two separate paths: a **Speed Layer** (stream processing, e.g. Flink/Kafka for low-latency real-time features) and a **Batch Layer** (Hadoop/Spark for accurate historical batch processing).
  - Merges views at query time.
  - *Drawback*: Requires maintaining dual codebases for the same business logic, leading to subtle logic drift between online and offline features.
- **Kappa Architecture**:
  - Eliminates the batch processing system entirely; treats all data as an append-only immutable event stream (Kafka).
  - Uses a single stream-processing engine (e.g., Apache Flink) for both real-time event processing and historical re-processing (by rewinding stream offsets).
  - *Advantage*: Guarantees identical feature definitions for online serving and offline model training.

---

### Q3: When should you choose Asynchronous Event-Driven inference over Synchronous RPC?
**Answer:**
- **Use Asynchronous Event-Driven Inference when**:
  - Processing takes longer than typical HTTP timeout budgets ($> 2 - 5\text{ seconds}$), such as batch document OCR, video object tracking, or complex multi-agent execution.
  - Traffic exhibits sudden bursty spikes that would overwhelm synchronous servers; message queues (Kafka, SQS) buffer requests gracefully.
  - Clients do not need immediate responses (e.g. background fraud auditing, weekly report generation).
- **Use Synchronous RPC when**:
  - A human user is waiting directly on the response to render a UI (e.g., search autocomplete, real-time checkout fraud authorization).
