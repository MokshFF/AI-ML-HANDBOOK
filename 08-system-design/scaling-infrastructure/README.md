# Scaling Machine Learning Infrastructure

Comprehensive guide and implementation of distributed machine learning infrastructure, covering cluster autoscaling, distributed tensor and pipeline parallelism, ZeRO memory optimization, and multi-tier caching architectures.

---

## 1. Distributed Serving & Training Topologies

```
+-------------------------------------------------------------------------------+
|                       Distributed Model Sharding Schemes                      |
+-------------------------------------------------------------------------------+
| 1. Tensor Parallelism (TP - Megatron-LM)                                      |
|    Shards individual weight matrices (Q, K, V projections and MLP layers)     |
|    across intra-node GPUs connected via high-speed NVLink (>600 GB/s).        |
+-------------------------------------------------------------------------------+
| 2. Pipeline Parallelism (PP)                                                  |
|    Partitions layers sequentially across inter-node GPUs (Layers 1-16 on      |
|    Node 1, Layers 17-32 on Node 2) connected via 400 Gbps InfiniBand.         |
+-------------------------------------------------------------------------------+
| 3. DeepSpeed ZeRO (Zero Redundancy Optimizer)                                 |
|    - ZeRO-1: Shards Optimizer states (4x memory reduction)                    |
|    - ZeRO-2: Shards Optimizer states + Gradients (8x memory reduction)        |
|    - ZeRO-3: Shards Optimizer states + Gradients + Model Weights              |
+-------------------------------------------------------------------------------+
```

---

## 2. Core Concepts

### 2.1 Concurrency-Based Autoscaling (KEDA)
- Traditional Kubernetes HPA scales on CPU/Memory usage. However, ML models saturated with batch requests may keep GPUs at 100% compute even when requests are queuing up and latency SLA is violated.
- **KEDA (Kubernetes Event-driven Autoscaling)**: Scales based on **Queue Depth** or **Active In-Flight Concurrency**.
- **Anti-Flapping (Thrashing) Protection**:
  - Scale-up must be responsive (e.g. 15s cooldown).
  - Scale-down must be conservative (e.g. 120s cooldown) to prevent destroying pods during brief traffic lulls.

### 2.2 Sizing Foundation Models in VRAM
For a model with $N$ parameters in precision $P$ (bytes):
$$\text{Memory}_{\text{weights}} = N \times P$$
- For FP16 ($P=2$), a 70B parameter model requires $140\text{ GB}$ just for static weights.
- Adding KV cache buffers, CUDA kernels, and activation memory requires an additional $20\% - 30\%$ overhead.
- Therefore, a 70B model requires at least $2\times 80\text{ GB}$ (or $4\times 80\text{ GB}$) GPUs under Tensor Parallelism ($\text{TP}=2$ or $\text{TP}=4$) to run inference.

---

## 3. Directory Structure

```
08-system-design/scaling-infrastructure/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── cluster_sim.py
    └── test_cluster_sim.py
```
