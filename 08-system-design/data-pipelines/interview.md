# Data Pipelines for ML Interview Questions & Answers

### Q1: How do you handle out-of-order and late-arriving events in streaming ML pipelines?
**Answer:**
1. **Event Time vs Ingestion Time**: Always compute window aggregations based on **Event Time** (the timestamp generated on the client device when the action occurred) rather than Ingestion Time (when the server received it).
2. **Watermarking (Apache Flink pattern)**: A watermark is a temporal threshold that specifies how late an event can be before the window is finalized. For example, a watermark of $t - 10\text{ seconds}$ tells the engine: "Assume all events timestamped earlier than $t - 10\text{s}$ have arrived."
3. **Late-Arrival Policy**:
   - Minor delays: Allowed to update the active window state before watermark closes.
   - Extremely late arrivals (past watermark): Emitted to a side-output (Dead-Letter / Late Arrival Topic) and incorporated during daily batch reconciliation runs.

---

### Q2: What is Change Data Capture (CDC), and why is it preferred over periodic batch polling?
**Answer:**
- **Periodic Batch Polling**: Runs queries like `SELECT * FROM users WHERE updated_at > :last_poll`.
  - *Drawbacks*: Puts heavy read load on production relational databases; polling frequency creates latency; cannot capture records that were inserted and deleted between poll intervals.
- **Change Data Capture (CDC - e.g. Debezium)**:
  - Taps directly into the database transaction write-ahead log (WAL / binlog) at the storage engine level.
  - *Benefits*: Captures every INSERT, UPDATE, and DELETE operation with near-zero overhead on production DB; streams changes to Kafka in sub-second latency for immediate online feature store synchronization.

---

### Q3: How do you backfill a Feature Store when introducing a new feature definition?
**Answer:**
1. **Code Implementation**: Write the feature transformation logic in a unified framework (e.g. PySpark or SQL).
2. **Deterministic Partitioning**: Split the historical range (e.g., past 2 years of raw event logs) into non-overlapping temporal chunks (e.g., daily or 4-hour intervals).
3. **Parallel Re-computation**: Launch distributed workers (Spark/Ray cluster) to process chunks independently and write feature values into historical offline Parquet tables.
4. **Validation & Sync**: Run statistical parity checks between the backfilled historical values and newly streaming real-time values before activating the feature for model training.
