# Deep Learning Interview Questions & Deep Dives

Core neural network mechanics, backpropagation dynamics, normalization layers, optimization landscapes, and scaling behavior.

---

## 1. Beginner Questions

### Q1: Vanishing & Exploding Gradients: Causes & Mitigations
- **Tags**: `Conceptual` | `Mathematical` | `Gotcha`
- **Short Answer**: In deep networks, the chain rule computes parameter gradients through repeated matrix multiplications $\prod_{l=1}^L W_l \cdot \sigma'(z_l)$. If weight eigenvalues or activation derivatives are $< 1$ (e.g. Sigmoid $\le 0.25$), gradients shrink exponentially toward 0. If $> 1$, gradients explode toward $\infty$.
- **Detailed Explanation**:
  $$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial a_L} \prod_{l=2}^L \left( W_l^T \operatorname{diag}(\sigma'(z_{l-1})) \right) \frac{\partial z_1}{\partial W_1}$$
  - **Sigmoid activation**: Maximum derivative is $\sigma'(0) = 0.25$. Over 10 layers, gradient magnitude shrinks by at least $0.25^{10} \approx 10^{-6}$.
  - **Mitigations**:
    1. **Non-saturating activations**: ReLU ($\sigma'(z) = 1$ for $z > 0$), LeakyReLU, GELU, Swish.
    2. **Principled Initialization**: He/Kaiming initialization (scales weights by $\sqrt{2/n_{in}}$ for ReLU) and Xavier/Glorot (scales by $\sqrt{2/(n_{in} + n_{out})}$).
    3. **Residual Connections**: Creates gradient highways: $\frac{\partial (x + F(x))}{\partial x} = I + \frac{\partial F(x)}{\partial x}$, ensuring gradients propagate unimpeded even if $\frac{\partial F(x)}{\partial x} \approx 0$.
    4. **Normalization**: LayerNorm, BatchNorm.
    5. **Gradient Clipping**: Truncates norm $\|g\| > c$.
- **Example**: Training a 20-layer standard MLP with Sigmoid and normal random initialization results in zero loss reduction after 100 epochs because the first 5 layers experience numerical underflow in gradients. Switching to He-initialized ReLU allows immediate convergence.
- **Common Misconception**: Believing ReLU completely eliminates vanishing gradients. In the negative regime ($z < 0$), the derivative is strictly 0 ("Dying ReLU"), causing permanently dead neurons if biased negatively.
- **Follow-Up Questions**:
  1. *How does Parametric ReLU (PReLU) or ELU address the dying ReLU problem?*
  2. *What is gradient penalty in WGAN-GP, and how does it prevent gradient explosion?*

---

## 2. Intermediate Questions

### Q2: Normalization Taxonomy: BatchNorm vs LayerNorm vs RMSNorm
- **Tags**: `Conceptual` | `Mathematical` | `Practical`
- **Short Answer**: BatchNorm normalizes across the batch dimension per feature channel (sensitive to batch size; fails on sequences). LayerNorm normalizes across feature dimensions per single sample (batch-size independent; standard in NLP). RMSNorm simplifies LayerNorm by enforcing scale invariance without centering by mean, accelerating execution by $10-20\%$.
- **Detailed Explanation**:
  Let tensor shape be $(B, T, C)$ (Batch, Time/Sequence, Channel/Embedding):
  - **Batch Normalization**: Computes $\mu, \sigma$ across $(B, T)$ for each channel $c \in [1, C]$.
    - *Pro*: Stabilizes deep CNN activations.
    - *Con*: Training-inference mismatch; requires running average buffers; collapses when $B < 8$.
  - **Layer Normalization**: Computes $\mu, \sigma$ across channel dimension $C$ for each $(b, t)$ independently:
    $$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
    - *Pro*: Completely independent across batch samples and sequence lengths.
  - **Root Mean Square Normalization (RMSNorm)**: Discards mean centering entirely:
    $$\text{RMS}(x) = \sqrt{\frac{1}{C} \sum_{i=1}^C x_i^2}, \quad y = \frac{x}{\text{RMS}(x) + \epsilon} \odot \gamma$$
    - *Why it works*: Studies show the scaling property provides $99\%$ of LayerNorm's regularization benefit, and omitting mean calculation saves kernel memory bandwidth in modern LLMs (LLaMA, Mistral).
- **Example**: In a distributed LLM training run with batch size 1 per GPU, BatchNorm would divide by zero or yield degenerate zero variance. LayerNorm or RMSNorm executes identically on batch size 1.
- **Common Misconception**: Believing LayerNorm has a train/eval mode switch like BatchNorm. LayerNorm computes exact statistics per sample dynamically during both training and evaluation; it stores no running history.
- **Follow-Up Questions**:
  1. *Why does Pre-LN allow training deeper Transformers without warmup compared to Post-LN?*
  2. *What is Group Normalization, and why is it preferred in high-resolution computer vision detection?*

---

### Q3: Adam vs AdamW: Weight Decay vs L2 Regularization
- **Tags**: `Mathematical` | `Debugging` | `Gotcha`
- **Short Answer**: In standard Adam with L2 regularization, the weight penalty $\lambda w$ is added directly to the gradient before computing moving averages $m_t$ and $v_t$, meaning weights with large historical gradients receive less regularization. AdamW decouples weight decay, subtracting $\lambda w$ directly from the parameter update, restoring proper proportional decay.
- **Detailed Explanation**:
  In standard SGD, L2 regularization and weight decay are mathematically identical:
  $$g_t = \nabla f(w) + \lambda w \implies w_{t+1} = w_t - \eta g_t = (1 - \eta \lambda) w_t - \eta \nabla f(w)$$
  In Adam with L2 regularization:
  $$g_t = \nabla f(w) + \lambda w, \quad v_t = \beta_2 v_t + (1 - \beta_2) g_t^2, \quad w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t$$
  Here, the effective weight decay rate is $\frac{\eta}{\sqrt{v_t}} \lambda$. Features that receive frequent, large gradients have large $v_t$, so their weights are decayed *less* than inactive features!
  In **AdamW (Decoupled Weight Decay)**:
  $$w_{t+1} = w_t - \eta \lambda w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t$$
  Every parameter is decayed strictly proportional to its magnitude, independent of adaptive gradient variance.
- **Example**: In Transformer pre-training, switching from standard Adam + L2 to AdamW recovers $15-20\%$ generalization improvement and enables much higher learning rates.
- **Common Misconception**: Passing `weight_decay=0.01` to `torch.optim.Adam` thinking it implements AdamW. `torch.optim.Adam` implements classical L2 regularization inside the gradient; you must explicitly import `torch.optim.AdamW`.
- **Follow-Up Questions**:
  1. *Why should bias terms and LayerNorm gain/bias parameters be exempted from weight decay?*
  2. *How does the Lion optimizer compare to AdamW in memory usage?*

---

## 3. Advanced Questions

### Q4: Multi-Head Attention: Computational Complexity & Memory Footprints
- **Tags**: `Mathematical` | `System Design` | `Coding`
- **Short Answer**: Multi-Head Attention computes $\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$. Calculating the attention matrix requires $\mathcal{O}(N^2 \cdot d)$ compute and $\mathcal{O}(N^2)$ memory for sequence length $N$. FlashAttention solves memory bottlenecks by computing attention in SRAM tiles without materializing the $N \times N$ matrix in high-bandwidth GPU memory (HBM).
- **Detailed Explanation**:
  Let $Q, K, V \in \mathbb{R}^{N \times d}$:
  1. $S = Q K^T \in \mathbb{R}^{N \times N}$ requires $2 N^2 d$ FLOPs and materializes $N^2$ elements in GPU HBM.
  2. $P = \operatorname{Softmax}(S / \sqrt{d_k}) \in \mathbb{R}^{N \times N}$ requires $3 N^2$ operations.
  3. $O = P V \in \mathbb{R}^{N \times d}$ requires $2 N^2 d$ FLOPs.
  At sequence length $N = 32,768$ in FP16 (2 bytes):
  $$N^2 \times 2 \text{ bytes} = (32,768)^2 \times 2 \approx 2.14 \text{ GB per attention head per layer!}$$
  For 32 layers with 32 heads, this would require $> 2.1 \text{ TB}$ of activation memory alone, crashing any GPU.
- **FlashAttention Solution**:
  Uses online softmax accumulation to compute $P$ and $O$ incrementally in fast GPU SRAM (20 TB/s bandwidth) in small blocks ($128 \times 128$), writing only the final output $O \in \mathbb{R}^{N \times d}$ back to slow HBM (2 TB/s bandwidth).
- **Example**: In a production LLaMA-70B model, FlashAttention-2 reduces training time by $3.5\times$ and reduces activation memory from quadratic $\mathcal{O}(N^2)$ to linear $\mathcal{O}(N)$.
- **Common Misconception**: Believing FlashAttention alters the mathematical outputs of attention. It is mathematically identical to standard attention up to floating-point numerical precision; it is purely an IO-aware hardware execution rearrangement.
- **Follow-Up Questions**:
  1. *What is the difference between Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)?*
  2. *Why does KV cache grow linearly with context length during inference, and how does PagedAttention alleviate fragmentation?*
