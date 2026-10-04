# Production Machine Learning Serving

Comprehensive guide and implementation of high-performance model serving architectures, covering FastAPI, containerization, Kubernetes concepts, dynamic micro-batching, and deployment strategies.

---

## 1. Serving Architecture & Request Lifecycle

```
                                [Client / API Gateway]
                                          |
                                          | (HTTPS / gRPC)
                                          v
                              +-----------------------+
                              | Kubernetes Ingress /  |
                              |   Load Balancer       |
                              +-----------+-----------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
     [Pod 1: FastAPI Worker]                             [Pod 2: FastAPI Worker]
     +-----------------------------------+               +-----------------------------------+
     |  /healthz & /ready Probes         |               |  /healthz & /ready Probes         |
     |  Pydantic Schema Validation       |               |  Pydantic Schema Validation       |
     |  +-----------------------------+  |               |  +-----------------------------+  |
     |  | Dynamic Micro-Batch Queue   |  |               |  | Dynamic Micro-Batch Queue   |  |
     |  +--------------+--------------+  |               |  +--------------+--------------+  |
     |                 |                 |               |                 |                 |
     |                 v                 |               |                 v                 |
     |    [TensorRT / ONNX Runtime]      |               |    [TensorRT / ONNX Runtime]      |
     |    Optimized Model Weights        |               |    Optimized Model Weights        |
     +-----------------------------------+               +-----------------------------------+
```

---

## 2. Core Concepts

### 2.1 Serving Paradigms
| Paradigm | Latency Target | Throughput | Use Cases |
|---|---|---|---|
| **Real-Time (Synchronous REST/gRPC)** | $10 - 100\text{ ms}$ | Moderate | Fraud detection, autocomplete, real-time bidding |
| **Near-Real-Time (Streaming / Queue)** | $100\text{ ms} - 5\text{ s}$ | High | Content moderation, notification delivery |
| **Batch Inference (Offline ETL)** | Minutes to Hours | Massive ($10^7+$ rows) | Weekly churn scores, email campaign targeting |

### 2.2 Dynamic Micro-Batching
- **The Dilemma**: Single-request inference under-utilizes GPU/CPU matrix multiplication engines (arithmetic intensity is low). However, waiting for large batches introduces unacceptable latency.
- **The Solution**: A background queue aggregates incoming concurrent requests over a micro-window (e.g., $10\text{ ms}$) or until `max_batch_size` is reached. The combined batch runs in a single forward pass, providing $3\times - 8\times$ higher throughput at minimal latency cost.

### 2.3 Kubernetes Lifecycle & Health Probes
- **Liveness Probe (`/healthz`)**: Verifies the process is responsive. If it fails, Kubernetes restarts the container.
- **Readiness Probe (`/ready`)**: Verifies the model weights are loaded and warmed up. Traffic is routed to the Pod only after readiness succeeds, eliminating 502/503 errors during rolling deployments.
- **Horizontal Pod Autoscaler (HPA)**: Dynamically scales replica count based on CPU utilization or custom latency/concurrency metrics.

---

## 3. Directory Structure

```
07-mlops/serving/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── serving_engine.py
    └── test_serving_engine.py
```
