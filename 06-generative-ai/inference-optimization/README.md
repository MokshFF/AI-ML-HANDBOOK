# Large Language Model Inference Optimization

Comprehensive guide and implementation of modern LLM serving optimizations, covering the Roofline Model, KV Cache architectures (MHA/GQA/MQA), PagedAttention, Continuous Batching, Speculative Decoding, and INT8/INT4 Quantization (AWQ/GPTQ).

---

## 1. The Serving Performance Landscape

Serving LLMs presents two distinct execution phases with radically different hardware profiles:

| Phase | Bound By | Mechanism | Critical Metric |
|---|---|---|---|
| **Prefill Phase** | **Compute Bound** | Processes input prompt tokens concurrently in parallel matrix multiplications | **Time to First Token (TTFT)** |
| **Decode Phase** | **Memory Bandwidth Bound** | Generates tokens sequentially one-by-one; streams all model weights for each token | **Inter-Token Latency (ITL)** / Tokens/sec |

```
[Prefill Phase (Prompt)] -------------> [Decode Phase (Generation Step 1, 2, ... N)]
Compute Bound (High arithmetic intensity) Memory Bandwidth Bound (Low arithmetic intensity)
TFLOPS matter most                      Memory Bandwidth (TB/s) matters most
```

---

## 2. Key Optimization Technologies

### 2.1 KV Cache & Attention Architectures
Autoregressive decoding saves previously computed Key and Value tensors to avoid recalculating past context:
- **Footprint**:
  $$\text{Memory}_{\text{KV}} = 2 \times B \times L \times N_{\text{kv}} \times d_{\text{head}} \times P$$
- **Evolution**:
  - **Multi-Head Attention (MHA)**: $N_{\text{kv}} = N_{\text{query}}$ (e.g., 32 heads). Massive memory footprint, limiting batch size.
  - **Grouped-Query Attention (GQA)**: $N_{\text{kv}} = \frac{N_{\text{query}}}{G}$ (e.g., 8 heads for 32 query heads). Reduces KV cache footprint by $4\times - 8\times$ with negligible quality loss (standard in LLaMA-2/3, Mistral).
  - **Multi-Query Attention (MQA)**: $N_{\text{kv}} = 1$. Maximum compression ($32\times$), slight degradation in complex reasoning.

### 2.2 PagedAttention & Continuous Batching
Introduced by Kwon et al. (vLLM, 2023):
- **Traditional Static Batching**: Allocates contiguous memory buffers sized for maximum possible sequence length. Results in 60-80% memory waste due to internal/external fragmentation.
- **Continuous / Iteration-Level Batching**: Evaluates the batch at each token decode step rather than waiting for an entire sequence to finish. Completed requests leave immediately and new requests join the running batch.
- **PagedAttention**: Inspired by OS virtual memory paging. Partitions KV cache into non-contiguous physical blocks (e.g., 16 tokens/block). A block table maps logical sequence tokens to physical memory slots, achieving near 100% memory utilization.

### 2.3 Speculative Decoding
- **Motivation**: Target model decode is slow because it reads billions of weights from memory for a single token.
- **Mechanism**:
  1. A small, ultra-fast draft model (e.g., 1B parameter draft for a 70B target) autoregressively drafts $K$ candidate tokens.
  2. The large target model evaluates all $K$ tokens in a single parallel verification forward pass.
  3. Rejection sampling accepts valid tokens with probability $\min\left(1, \frac{P_{\text{target}}(x)}{P_{\text{draft}}(x)}\right)$.
  4. Yields a $2\times - 3\times$ latency speedup with **provably mathematically identical output distribution**.

### 2.4 Post-Training Quantization (PTQ)
- **INT8 Quantization (SmoothQuant)**: Migrates activation outliers into weights so matrix multiplies can execute on fast INT8 Tensor Cores.
- **Weight-Only INT4 (AWQ / GPTQ)**:
  - Compresses 16-bit weights to 4 bits (e.g., 70B model goes from 140 GB to ~35 GB, fitting on a single 80 GB GPU).
  - **AWQ (Activation-aware Weight Quantization)**: Identifies the 1% most salient weight channels by observing activation magnitudes and preserves them in higher precision or scales them to protect accuracy.

---

## 3. Directory Structure

```
06-generative-ai/inference-optimization/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── inference_engine.py
    └── test_inference.py
```
