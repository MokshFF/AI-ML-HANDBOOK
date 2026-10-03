# Deep Learning Training Dynamics - Technical Interview Preparation

A curated question bank covering normalization mechanics, mixed precision arithmetic, learning rate schedules, and distributed training paradigms.

---

## 1. Normalization Mechanics & Pitfalls

### Q1: How does Batch Normalization behave differently during training vs. evaluation? What catastrophic bug occurs if `model.eval()` is omitted?
- **Training Mode (`model.train()`)**:
  - Computes the mean $\mu_{\mathcal{B}}$ and variance $\sigma_{\mathcal{B}}^2$ directly from the current mini-batch:
    $$\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
  - Updates running exponential moving average statistics:
    $$\mu_{\text{run}} \leftarrow (1 - \alpha) \mu_{\text{run}} + \alpha \mu_{\mathcal{B}}$$
    $$\sigma_{\text{run}}^2 \leftarrow (1 - \alpha) \sigma_{\text{run}}^2 + \alpha \sigma_{\mathcal{B}}^2$$
- **Evaluation Mode (`model.eval()`)**:
  - Uses the frozen running statistics $\mu_{\text{run}}$ and $\sigma_{\text{run}}^2$ computed during training.
  - Normalization is deterministic and independent across individual samples.
- **The Catastrophic Omission Bug**:
  If a model is evaluated without calling `model.eval()`:
  1. *Batch Size 1 Failure*: If running inference on a single sample ($m=1$), sample variance $\sigma_{\mathcal{B}}^2 = 0$. Division by $\sqrt{0 + \epsilon}$ causes numerical instability or zeroes out all activations, resulting in random nonsensical predictions.
  2. *Data Contamination at Test Time*: A test sample's prediction will depend on which other test samples happen to share its batch, violating independent inference.

---

### Q2: Why is Batch Normalization standard in CNNs, while Layer Normalization is universally used in Transformers?
- **Why BatchNorm Excels in CNNs**:
  In computer vision, images have fixed spatial resolutions ($H \times W$). Computing statistics across the batch dimension $(N, H, W)$ preserves channel-specific feature distributions while keeping activation distributions bounded across large feature maps.
- **Why BatchNorm Fails in Transformers**:
  1. *Variable Sequence Lengths*: Text sequences have variable lengths padded with zeros. A batch mean across sequence positions includes padding tokens, corrupting real token statistics.
  2. *Small Batch Constraints*: Large language models (LLMs) often train with batch size 1 to 4 per GPU due to VRAM limits. At $N \le 4$, BatchNorm's empirical mean and variance estimates have high statistical noise, destabilizing training.
- **Why LayerNorm Excels in Transformers**:
  LayerNorm computes mean and variance across the hidden embedding dimension $D$ of a single token:
  $$\mu = \frac{1}{D} \sum_{k=1}^D x_{tk}, \quad \sigma^2 = \frac{1}{D} \sum_{k=1}^D (x_{tk} - \mu)^2$$
  It is completely independent of batch size and sequence length, making it invariant to padding tokens and ideal for autoregressive generation.

---

## 2. Mixed Precision & Distributed Systems

### Q3: Explain the difference between FP16 and BF16 in Mixed-Precision training. Why does FP16 require dynamic loss scaling while BF16 does not?
- **Bit Allocation**:
  - **FP32**: 1 sign bit, 8 exponent bits, 23 mantissa bits (range: $\sim 10^{-38}$ to $10^{38}$).
  - **FP16**: 1 sign bit, 5 exponent bits, 10 mantissa bits (range: $\sim 6 \times 10^{-5}$ to $65,504$).
  - **BF16 (Bfloat16)**: 1 sign bit, 8 exponent bits, 7 mantissa bits (range: $\sim 10^{-38}$ to $10^{38}$).
- **The Underflow Problem in FP16**:
  In deep networks, backpropagated gradients often have magnitudes smaller than $6 \times 10^{-5}$ (e.g., $10^{-7}$). In FP16, these gradients underflow directly to zero, halting weight updates.
  - *Fix (Dynamic Loss Scaling)*: Multiply the loss by a large factor $S$ (e.g., $2^{15}$) before backward propagation to shift small gradients into FP16's representable range, then divide gradients by $S$ before the optimizer step.
- **Why BF16 Eliminates Loss Scaling**:
  BF16 preserves all 8 exponent bits from FP32, giving it the exact same dynamic range as full single-precision floating point. Gradients do not underflow, completely eliminating the need for loss scaling and dramatically simplifying training stability.

---

### Q4: Explain the 3 stages of ZeRO (Zero Redundancy Optimizer) in distributed training.
For standard Adam with $M$ parameters in FP16/FP32 mixed precision:
- Parameters (FP16): $2M$ bytes
- Gradients (FP16): $2M$ bytes
- Optimizer states (FP32 master weights, 1st moment, 2nd moment): $4M + 4M + 4M = 12M$ bytes
- Total static memory: **$16M$ bytes** (e.g., a 70B model requires $1.12\text{ Terabytes}$ of static VRAM before activations!).

- **ZeRO Stage 1 (Optimizer State Partitioning)**:
  - Shards optimizer states ($12M$) across $N_{\text{GPUs}}$ workers.
  - Memory per GPU: $2M + 2M + \frac{12M}{N_{\text{GPUs}}}$. Up to **$4\times$ memory reduction** with zero communication overhead.
- **ZeRO Stage 2 (Gradient Partitioning)**:
  - Shards gradients ($2M$) alongside optimizer states.
  - Memory per GPU: $2M + \frac{2M + 12M}{N_{\text{GPUs}}}$. Up to **$8\times$ memory reduction**.
- **ZeRO Stage 3 / FSDP (Parameter Partitioning)**:
  - Shards model parameters ($2M$) across all workers.
  - Memory per GPU: $\frac{16M}{N_{\text{GPUs}}}$. Linear memory scaling with GPU count, enabling training of massive foundation models.

---

## 3. Whiteboard Coding Drills

### Q5: Implement gradient norm clipping from scratch in PyTorch.
```python
import math
import torch
import torch.nn as nn

def clip_grad_norm(parameters, max_norm: float):
    # Filter parameters that have gradients
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    # Compute global L2 norm across all parameters
    total_norm_sq = sum(p.grad.data.norm(2).item() ** 2 for p in params)
    total_norm = math.sqrt(total_norm_sq)

    # Scale gradients if total norm exceeds threshold
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.data.mul_(clip_coef)

    return total_norm
```
