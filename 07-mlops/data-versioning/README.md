# Data & Feature Versioning: DVC & Feature Stores

Comprehensive guide and implementation of modern dataset and feature engineering infrastructure, covering Data Version Control (DVC), Feature Stores (Feast architecture), Point-in-Time correctness, and data lineage DAGs.

---

## 1. System Architecture

```
                                  +---------------------------------+
                                  |    Raw Data Stream / Lakehouse  |
                                  +----------------+----------------+
                                                   |
                             +---------------------+---------------------+
                             |                                           |
                             v                                           v
               [DVC Content-Addressed Store]                 [Feature Transformation Pipeline]
               - SHA-256 data fingerprinting                             |
               - .dvc metadata in Git                                    v
               - Payload in Cloud Object Storage             +-----------------------+
                                                             |     Feature Store     |
                                                             +-----------+-----------+
                                                                         |
                                           +-----------------------------+-----------------------------+
                                           |                                                           |
                                           v                                                           v
                             [Online Store (Redis / DynamoDB)]                     [Offline Store (Parquet / Snowflake)]
                             - Sub-10ms point lookups                              - Historical event logs
                             - Serves real-time inference                          - Point-in-Time (AS-OF) join for training
```

---

## 2. Core Concepts

### 2.1 DVC (Data Version Control)
- **Problem**: Git cannot handle 100 GB+ datasets without catastrophic performance degradation.
- **Solution**:
  - DVC replaces large files with small text pointer files (`.dvc`) containing SHA-256 content hashes.
  - The pointers are committed into Git branches alongside code.
  - Actual binary datasets are synchronized to remote storage (`dvc push` / `dvc pull` to S3, GCS, or Azure Blob).
  - Ensures deterministic reproducibility: any historical Git commit points to the exact dataset version used to train that model.

### 2.2 Feature Stores (The Feast Pattern)
A Feature Store standardizes feature definitions across the entire machine learning lifecycle:
1. **Online Feature Store**:
   - Ultra-low latency key-value storage (Redis, DynamoDB, Bigtable).
   - Serves the latest feature vector for an entity ($O(1)$ lookup) during live online inference.
2. **Offline Feature Store**:
   - High-throughput analytical storage (BigQuery, Snowflake, Delta Lake, Parquet).
   - Stores the complete append-only historical log of feature values over time.

### 2.3 Point-in-Time Correctness (AS-OF Join)
- **Target Leakage / Lookahead Bias**: If training data uses feature values computed after the label event occurred, the model learns an artificial relationship that will not exist in production.
- **AS-OF Join Mechanism**:
  For an observation occurring at timestamp $T$, the feature store performs a backward time-travel join to retrieve the most recent feature record timestamped $t \le T$.

---

## 3. Directory Structure

```
07-mlops/data-versioning/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── data_registry.py
    └── test_data_registry.py
```
