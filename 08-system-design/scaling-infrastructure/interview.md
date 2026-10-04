# Scaling Infrastructure Interview Questions & Answers

### Q1: What is the difference between Tensor Parallelism and Pipeline Parallelism?
**Answer:**
- **Tensor Parallelism (TP)**:
  - *Mechanism*: Splits individual weight matrices across GPUs (e.g. Column-parallel linear layer in MLP followed by Row-parallel layer).
  - *Communication*: Requires `All-Reduce` collective operations at every Transformer layer.
  - *Hardware requirement*: Extremely high communication bandwidth; must run within a single node over NVLink ($> 600 - 900\text{ GB/s}$).
- **Pipeline Parallelism (PP)**:
  - *Mechanism*: Splits consecutive layers of the network across different machines (e.g. GPU 0 runs layers 1-16; GPU 1 runs layers 17-32).
  - *Communication*: Point-to-point communication only at layer boundaries (sending hidden activations forward and gradients backward).
  - *Hardware requirement*: Runs across distinct physical servers connected via standard 100-400 Gbps Ethernet or InfiniBand.
  - *Challenge*: Introduces "pipeline bubble" idle time, mitigated by micro-batching (1F1B schedule).

---

### Q2: Why is CPU-based autoscaling often inadequate for ML inference clusters, and what metric should be used instead?
**Answer:**
- **Inadequacy of CPU/GPU Utilization**:
  GPU Tensor Cores can register 99% utilization whether serving 1 request or 100 queued requests. If autoscaling relies solely on GPU utilization, it will fail to scale up when traffic doubles, causing queue backlog and severe latency spikes.
- **Optimal Metrics**:
  1. **Average Concurrency per Replica**: Current in-flight requests divided by target capacity (e.g., maintain $\le 10$ concurrent requests per pod).
  2. **P99 Inference Latency**: Trigger scale-up when latency approaches SLA thresholds.
  3. **Queue Depth**: Monitor RabbitMQ / Kafka / Redis queue backlog.

---

### Q3: How does DeepSpeed ZeRO-3 eliminate memory redundancy without the communication overhead of model parallelism?
**Answer:**
In standard Distributed Data Parallel (DDP), every GPU holds an identical copy of:
1. Model parameters
2. Gradients
3. Optimizer states (FP32 master weights, momentum, variance in Adam)
**ZeRO-3 (Zero Redundancy Optimizer)**:
- Shards parameters, gradients, and optimizer states evenly across all data-parallel GPUs ($N$ ranks).
- During forward propagation, each layer initiates an `All-Gather` to dynamically collect the weights from peers just in time to compute the layer, and immediately frees them from memory afterwards.
- Reduces memory footprint by $N$-fold, enabling a cluster of 64 modest GPUs to train a 100B+ model without complex model refactoring.
