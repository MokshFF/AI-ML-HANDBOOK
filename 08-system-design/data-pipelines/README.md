# Data Pipelines for Machine Learning

Comprehensive guide and implementation of production data engineering patterns for ML, covering stream processing, sliding/tumbling windows, idempotency, dead-letter queues, and backfill execution.

---

## 1. End-to-End Stream Architecture

```
                                [Client Events / CDC]
                                          |
                                          v
                              +-----------------------+
                              |   Kafka Event Stream  |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Apache Flink / Spark  |
                              |   Stream Aggregator   |
                              +-----+-----------+-----+
                                    |           |
            (Malformed Payload)     |           | (Valid Stream)
                     v              |           v
            +------------------+    |    +----------------------+
            | Dead-Letter      |    |    | Idempotent           |
            | Queue (DLQ)      |    |    | Deduplicator         |
            +------------------+    |    +----------+-----------+
                                    |               |
                                    v               v
                             [Window Aggregations (1m, 5m, 1h)]
                                    |
                                    v
                             [Online Feature Store (Redis)]
```

---

## 2. Core Concepts

### 2.1 Windowing Semantics
- **Tumbling Windows**: Fixed-size, non-overlapping temporal intervals (e.g., 0:00-0:05, 0:05-0:10). Every event belongs to exactly one window.
- **Sliding / Hopping Windows**: Fixed-size, overlapping intervals (e.g., a 10-minute window computed every 1 minute). Events belong to multiple overlapping windows.
- **Session Windows**: Dynamic windows bounded by inactivity gaps (e.g., a user browsing session that closes after 30 minutes of idle time).

### 2.2 Idempotence & At-Least-Once Delivery
Message brokers (Kafka, RabbitMQ, SQS) typically guarantee **at-least-once** delivery. Network timeouts during consumer acknowledgments can lead to identical events being delivered multiple times:
- Without deduplication, cumulative metrics (e.g., `user_click_count_24h`) become artificially inflated.
- **Idempotency Key**: Attaching a unique hash or UUID to each event allows consumers to check a fast in-memory Bloom filter or Redis set to drop duplicates.

### 2.3 Dead-Letter Queues (DLQ)
When processing millions of events per hour, upstream schema mutations or corrupted payloads will inevitably cause parsing exceptions:
- Dropping corrupted records silently leads to silent data degradation.
- Halting the pipeline causes message accumulation and violates latency SLAs.
- **The DLQ Pattern**: The consumer catches parsing exceptions, sends the failed raw payload with full stack trace metadata to a specialized DLQ topic, and continues processing the stream.

---

## 3. Directory Structure

```
08-system-design/data-pipelines/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── pipeline_core.py
    └── test_pipeline_core.py
```
