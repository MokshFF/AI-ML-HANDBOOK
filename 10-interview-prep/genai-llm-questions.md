# Generative AI & Large Language Models Interview Questions

Modern foundation models, KV caching, parameter-efficient fine-tuning, preference alignment, and agent architectures.

---

## 1. Intermediate Questions

### Q1: KV Cache: Computational & Memory Footprint Analysis
- **Tags**: `Mathematical` | `System Design` | `Practical`
- **Short Answer**: In autoregressive generation, computing attention for token $t$ requires Keys and Values from all preceding tokens $1 \dots t-1$. Without caching, past KV states are recomputed at every step (quadratic compute $\mathcal{O}(N^2)$). Storing past KV activations in GPU memory reduces inference step complexity to linear $\mathcal{O}(N)$, at the expense of massive GPU VRAM consumption.
- **Detailed Explanation**:
  For an autoregressive Transformer:
  $$\text{KV Cache Size per Token} = 2 \times (\text{Layers}) \times (\text{KV Heads}) \times (\text{Head Dimension}) \times (\text{Bytes per Parameter})$$
  - The factor of 2 accounts for Keys and Values.
  - In FP16 / BF16, bytes per parameter = 2.
  - In LLaMA-2-70B: 80 layers, 8 KV heads (Grouped-Query Attention), head dimension 128:
    $$\text{Memory per token} = 2 \times 80 \times 8 \times 128 \times 2 \text{ bytes} = 327,680 \text{ bytes} \approx 320 \text{ KB per token}$$
  - For a single request with context length $N = 4,096$:
    $$4,096 \times 320 \text{ KB} = 1.31 \text{ GB of VRAM per request!}$$
  - At a concurrency of 64 concurrent streams:
    $$64 \times 1.31 \text{ GB} = 83.84 \text{ GB of VRAM}$$
    The KV cache alone exceeds the memory of an 80GB A100 GPU before even loading the model weights!
- **Mitigations**:
  1. **Grouped-Query Attention (GQA)**: Shares 1 KV head across $G$ query heads (e.g. $G=8$ reduces KV cache by $8\times$).
  2. **KV Cache Quantization**: Quantizing FP16 KV states to FP8 or INT4 saves $2-4\times$ memory.
  3. **PagedAttention (vLLM)**: Allocates KV cache in virtual memory blocks, eliminating internal/external memory fragmentation.
- **Example**: Moving from Multi-Head Attention (MHA) to GQA in 70B models allowed serving $8\times$ larger concurrent batch sizes on standard DGX nodes without out-of-memory (OOM) crashes.
- **Common Misconception**: Believing the prefill (prompt processing) phase is memory-bandwidth bound. Prefill processes all prompt tokens in parallel and is compute-bound. The autoregressive decoding phase processes 1 token at a time and is strictly memory-bandwidth bound.
- **Follow-Up Questions**:
  1. *How does speculative decoding achieve $2-3\times$ speedup on memory-bound decoding?*
  2. *What is chunked prefill, and how does it prevent inter-token latency spikes?*

---

### Q2: Parameter-Efficient Fine-Tuning: LoRA & QLoRA Mechanics
- **Tags**: `Mathematical` | `Conceptual` | `Practical`
- **Short Answer**: LoRA (Low-Rank Adaptation) freezes the pre-trained base model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable low-rank decomposition matrices $\Delta W = B \cdot A$ where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$. QLoRA quantizes $W_0$ to 4-bit NormalFloat (NF4), double quantizes constants, and pages optimizer states to CPU to allow fine-tuning 70B models on single consumer GPUs.
- **Detailed Explanation**:
  $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$
  - $A$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$, and $B$ is initialized to 0, ensuring $\Delta W = 0$ at training start (no initial perturbation).
  - $\alpha$ is a scaling hyperparameter; scaling by $\frac{\alpha}{r}$ stabilizes tuning when experimenting with rank $r$.
  - **Parameter Reduction**: For $d = 4096, k = 4096$, full weights contain $16.7\text{M}$ parameters. A LoRA rank $r=8$ contains only $(4096 \times 8) + (8 \times 4096) = 65,536$ parameters (a **$99.6\%$ reduction**!).
  - **Inference Efficiency**: In production, $\Delta W$ can be folded back into base weights: $W_{\text{deployed}} = W_0 + \frac{\alpha}{r} B A$, introducing **zero additional latency overhead**.
- **Example**: Fine-tuning an entire 70B model in FP16 requires $> 800\text{ GB}$ of VRAM (weights + gradients + Adam optimizer states). With QLoRA (NF4 base weights + LoRA adapters on attention & MLP), it runs within a single 48GB A6000 GPU.
- **Common Misconception**: Believing higher rank $r$ always improves downstream performance. For narrow instruction-following tasks, $r=8$ or $r=16$ performs as well as $r=256$, with higher ranks prone to catastrophic forgetting of base pre-training knowledge.
- **Follow-Up Questions**:
  1. *Why is NormalFloat4 (NF4) information-theoretically optimal for normally distributed pre-trained weights?*
  2. *How does DORA (Weight-Decomposed Low-Rank Adaptation) decouple directional and magnitude updates?*

---

## 2. Advanced Questions

### Q3: Alignment Methodologies: RLHF vs DPO (Direct Preference Optimization)
- **Tags**: `Mathematical` | `Conceptual` | `Gotcha`
- **Short Answer**: RLHF trains an explicit Reward Model on pairwise human preferences $(y_w \succ y_l)$, then uses PPO reinforcement learning to optimize the policy LLM with a KL penalty against reference model drift. DPO mathematically solves for the optimal policy directly from preference pairs in closed form, eliminating the reward model and unstable PPO actor-critic training.
- **Detailed Explanation**:
  **RLHF Objective**:
  $$\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta}[R(x, y)] - \beta D_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$
  Requires running 4 models concurrently in VRAM: Actor (active policy), Critic (value head), Reference Model (frozen), and Reward Model. Highly prone to PPO hyperparameter instability and reward hacking.
  **DPO Formulation**:
  The authors of DPO demonstrated that the optimal reward function satisfies:
  $$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
  Substituting into the Bradley-Terry preference probability $P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$ yields the closed-form DPO loss:
  $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
  DPO treats the language model *itself* as the implicit reward model.
- **Example**: Migrating an enterprise chatbot alignment pipeline from RLHF-PPO to DPO reduced training cluster infrastructure costs by $65\%$ and eliminated policy collapse incidents.
- **Common Misconception**: Assuming DPO cannot overfit. Because DPO optimizes relative log-probabilities directly, low $\beta$ values ($< 0.05$) can drive the policy to degrade prompt continuation coherence while driving preference loss down.
- **Follow-Up Questions**:
  1. *How does KTO (Kahneman-Tversky Optimization) align models using un-paired binary signals (thumb up / thumb down)?*
  2. *What is length bias in DPO, and how does Length-Normalized DPO mitigate verbosity inflation?*
