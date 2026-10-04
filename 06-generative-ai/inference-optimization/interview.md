# LLM Inference Optimization Interview Questions & Answers

### Q1: Why is LLM generation during the decode phase memory-bandwidth bound rather than compute bound?
**Answer:**
During the decode phase, the model generates exactly one token per sequence per forward step.
To compute that single token:
1. Every weight parameter in the neural network must be loaded from High Bandwidth Memory (HBM) into on-chip cache (SRAM) and Tensor Core registers.
2. For a 70B model in FP16 (140 GB of weights), generating 1 token requires streaming 140 GB of data through memory.
3. On an NVIDIA A100 GPU with ~2,000 GB/s memory bandwidth, the maximum theoretical speed at batch size 1 is:
   $$\text{Speed} = \frac{2,000\text{ GB/s}}{140\text{ GB}} \approx 14.3\text{ tokens/second}$$
The GPU Tensor Cores (capable of 312 TFLOPS) sit idle for most of the clock cycles waiting for memory transfers.
Increasing batch size groups multiple tokens into the same weight read, boosting arithmetic intensity and transitioning execution toward compute bound.

---

### Q2: What problem does PagedAttention solve, and how does it work?
**Answer:**
**The Problem**:
In conventional LLM serving systems, GPU memory must be allocated contiguously for the worst-case maximum sequence length (e.g., 4096 tokens). Because actual request lengths vary unpredictably:
- **Internal Fragmentation**: Reserved slots for tokens that were never generated remain empty.
- **External Fragmentation**: Memory gaps between requests cannot fit new incoming requests.
- Over 60-80% of GPU memory was wasted, severely restricting maximum concurrency.

**How PagedAttention Works (vLLM)**:
1. Translates the classic OS virtual memory paging concept to the KV cache.
2. Divides the KV cache into fixed-size physical blocks (e.g., 16 tokens per block).
3. The server maintains a **Block Table** for each request that maps contiguous logical sequence indices to arbitrary non-contiguous physical blocks in GPU VRAM.
4. Eliminates memory fragmentation entirely, achieving >96% memory utilization and boosting serving throughput by $2\times - 4\times$.

---

### Q3: Explain how Speculative Decoding guarantees the exact same output distribution as the target model.
**Answer:**
Speculative decoding uses **rejection sampling**:
1. Draft model generates candidate token $x \sim q(x)$.
2. Target model computes its probability $p(x)$ under the full context.
3. The candidate token $x$ is accepted with probability:
   $$\alpha = \min\left(1, \frac{p(x)}{q(x)}\right)$$
4. If $x$ is accepted, it is kept. If $x$ is rejected, a replacement token is sampled from the adjusted residual distribution:
   $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))}$$
Mathematically, the composite probability of accepting or sampling from the residual evaluates exactly to $p(x)$. Thus, the final output distribution is **lossless** and identical to running the large target model alone.

---

### Q4: What is the difference between TTFT and ITL, and how do you optimize each?
**Answer:**
- **TTFT (Time to First Token)**:
  - Measures latency from when the user submits the prompt until the first token appears on screen.
  - Dominated by the **prefill phase** (processing the full prompt).
  - *Optimizations*: FlashAttention, prompt caching (reusing KV cache across shared prefixes), chunked prefill, tensor parallelism.
- **ITL (Inter-Token Latency)**:
  - Measures the time elapsed between each subsequent generated token (streaming speed).
  - Dominated by the **decode phase** (memory bandwidth bound).
  - *Optimizations*: Weight quantization (INT4/INT8 to cut bytes transferred), Grouped-Query Attention (GQA), speculative decoding, tensor parallelism.

---

### Q5: What is the difference between AWQ and standard round-to-nearest (RTN) quantization?
**Answer:**
- **Round-to-Nearest (RTN)** uniformly rounds all weights to the nearest discrete quantization bin. However, LLMs have a small fraction (~0.1% to 1%) of outlier activation channels with magnitudes $100\times$ larger than normal channels. RTN truncates or distorts these critical weights, causing severe perplexity degradation at 4-bit precision.
- **AWQ (Activation-aware Weight Quantization)**:
  1. Observes activation distributions on a small calibration set.
  2. Identifies the most salient weights (those multiplied by high-magnitude activations).
  3. Instead of keeping them in FP16 (which creates mixed-precision overhead), AWQ protects them by finding per-channel scale factors that minimize quantization error specifically on the salient features:
     $$W' = W \cdot s, \quad X' = X \cdot s^{-1}$$
  4. Keeps all weights in uniform 4-bit format while preserving perplexity close to FP16.
