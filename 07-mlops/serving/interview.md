# ML Model Serving Interview Questions & Answers

### Q1: Why use FastAPI over Flask for production machine learning serving?
**Answer:**
1. **Asynchronous Concurrency (`async`/`await`)**: FastAPI runs on ASGI (Uvicorn), non-blockingly handling thousands of concurrent I/O connections (e.g. database feature lookups or external API calls) on a single thread.
2. **Strict Schema Validation**: Built on Pydantic, FastAPI parses and validates incoming JSON payloads against strict Python type annotations, rejecting malformed feature matrices before they reach the model.
3. **High Performance**: Outperforms traditional WSGI frameworks like Flask/Django by $2\times - 3\times$ in raw request throughput.
4. **Auto-Generated OpenAPI / Swagger Docs**: Facilitates immediate contract sharing between ML engineers and frontend/backend teams.

---

### Q2: What is the difference between Kubernetes Liveness and Readiness probes, and why are both essential for ML services?
**Answer:**
- **Liveness Probe**:
  - *Purpose*: Checks if the container process is alive and not deadlocked.
  - *Action on failure*: Kubernetes kills and restarts the container.
- **Readiness Probe**:
  - *Purpose*: Checks if the container is ready to accept live traffic.
  - *Action on failure*: Kubernetes temporarily isolates the container from the Service load balancer without killing it.
- **Critical ML Scenario**:
  Loading a 10 GB model into GPU memory takes 30-60 seconds. During this warm-up time:
  - If only a liveness probe exists and fails, Kubernetes enters an infinite restart crash-loop.
  - With a readiness probe, Kubernetes waits until the model weights are loaded into memory and verified before sending the first user request, ensuring zero dropped requests during rolling updates.

---

### Q3: How does Dynamic Micro-Batching achieve higher throughput without exceeding latency SLAs?
**Answer:**
In modern accelerators (GPUs/TPUs), computing a batch of 8 or 16 inputs takes almost the same time as computing a batch of 1 due to high degree of tensor parallelism.
**Dynamic Micro-Batching**:
1. Incoming requests enter an in-memory queue.
2. The batcher dispatches the batch when either:
   - Size limit: Number of queued requests reaches `max_batch_size` (e.g. 16).
   - Time limit: The oldest request has waited `max_wait_ms` (e.g. 10 ms).
3. The model processes the batch in one forward pass and dispatches predictions back to their respective async caller contexts.
4. This yields massive throughput gains (e.g., $5\times$ higher QPS) while bounding the maximum latency delay by `max_wait_ms`.
