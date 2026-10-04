# Data Versioning & Feature Stores Interview Questions & Answers

### Q1: What is "Point-in-Time Correctness" in Feature Stores, and what happens if you violate it?
**Answer:**
**Definition**:
Point-in-time correctness (also called time-travel or AS-OF join) guarantees that when generating training datasets from historical observations, each observation at timestamp $T$ is joined only with feature values that were computed and available at or before $T$ ($t_{\text{feature}} \le T$).

**Consequences of Violation (Data Leakage)**:
If you simply join on `user_id` using the current feature snapshot:
- A user who committed fraud at 2:00 PM had 0 chargebacks at that moment.
- By 5:00 PM, the system flagged them and updated `chargeback_count = 1`.
- If you train on the 5:00 PM state, the model sees `chargeback_count = 1` and easily predicts fraud.
- In production, when the next transaction arrives, `chargeback_count` will be 0, causing the model to completely fail to detect fraud in real time.

---

### Q2: How does DVC differ from Git LFS (Large File Storage)?
**Answer:**
- **Git LFS**:
  - Tight coupling with Git server infrastructure (requires Git LFS support on GitHub/GitLab).
  - Harder to customize remote backends (e.g. multi-cloud buckets, SSH storage).
  - Lacks native ML pipeline awareness.
- **DVC (Data Version Control)**:
  - Completely decoupled: Git stores only tiny human-readable `.dvc` files; binary data can be sent to arbitrary storage (S3, GCS, Azure Blob, Google Drive, SFTP, NFS).
  - **Pipeline DAGs**: `dvc.yaml` tracks dependencies between data preparation scripts, raw data, and trained models, skipping execution if inputs have not changed.
  - Native integration with metrics, plots, and experiment tracking.

---

### Q3: Why is a dual-storage (Online vs Offline) architecture necessary for Feature Stores?
**Answer:**
Online and offline feature serving have incompatible access patterns:
1. **Online Serving Requirements**:
   - Access pattern: Single-row point lookups by Entity ID (e.g., `user_id="12345"`).
   - Latency: Strictly sub-10 ms (P99).
   - Throughput: High QPS (thousands of requests per second).
   - Optimal storage: In-memory Key-Value store (Redis, DynamoDB, Cassandra).
2. **Offline Training Requirements**:
   - Access pattern: Massive scans, multi-table AS-OF joins across millions of rows and hundreds of columns.
   - Latency: Minutes to hours acceptable.
   - Scale: Terabytes to Petabytes.
   - Optimal storage: Columnar data lakes / warehouses (Parquet, Delta Lake, Snowflake, BigQuery).
A Feature Store bridges both by providing a single feature definition with an automated ingestion sync from offline to online.
