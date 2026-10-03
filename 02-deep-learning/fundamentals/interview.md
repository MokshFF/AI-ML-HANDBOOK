# Deep Learning Fundamentals - Technical Interview Preparation

A curated question bank covering mathematical derivations, activation dynamics, weight initialization, and first-order optimization algorithms.

---

## 1. Architectural & Theoretical Foundations

### Q1: Why does initializing all weights to zero completely prevent multi-layer neural networks from learning?
- **Symmetry Breaking**:
  If all weights in layer $l$ are initialized to zero (or any identical constant), every neuron in that layer receives the exact same linear combination of inputs:
  $$z_j^{[l]} = \sum_i W_{ji}^{[l]} a_i^{[l-1]} + b_j^{[l]} = b_j^{[l]}$$
  Applying activation $\sigma$ produces identical forward activations across all hidden units: $a_1^{[l]} = a_2^{[l]} = \dots = a_k^{[l]}$.
- **Backward Symmetry**:
  During backpropagation, the upstream gradients $\delta_j^{[l]} = \frac{\partial L}{\partial z_j^{[l]}}$ will be identical for all neurons in the layer. Consequently, the weight gradient updates $\frac{\partial L}{\partial W_{ji}^{[l]}} = a_i^{[l-1]} \delta_j^{[l]}$ are identical across all neurons.
  Even after thousands of gradient descent steps, all neurons will compute the exact same feature. The effective representational capacity of the entire layer collapses to that of a single neuron.
- **Biases Exception**:
  Biases can safely be initialized to zero because non-zero, randomized weights break the symmetry of inputs across neurons.

---

### Q2: Derive the variance preservation formula for He/Kaiming initialization with ReLU activations.
- **Forward Pass Variance**:
  For linear transformation $z = \sum_{i=1}^{d_{\text{in}}} w_i x_i$, assuming $w_i$ and $x_i$ are independent with zero mean:
  $$\text{Var}(z) = d_{\text{in}} \text{Var}(w) \mathbb{E}[x^2]$$
- **Effect of ReLU**:
  Let $x = \max(0, z_{\text{prev}})$. If $z_{\text{prev}}$ is symmetric around 0 with variance $\sigma_{\text{prev}}^2$:
  $$\mathbb{E}[x^2] = \int_{0}^{\infty} z^2 p(z) dz = \frac{1}{2} \text{Var}(z_{\text{prev}})$$
  Substituting back into the variance formula:
  $$\text{Var}(z) = d_{\text{in}} \text{Var}(w) \cdot \frac{1}{2} \text{Var}(z_{\text{prev}})$$
- **Preserving Signal Scale**:
  To ensure $\text{Var}(z) = \text{Var}(z_{\text{prev}})$ across arbitrary network depth, we require:
  $$\frac{1}{2} d_{\text{in}} \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{2}{d_{\text{in}}}$$
  Hence, weights are sampled from $\mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right)$. In contrast, Xavier initialization uses $\frac{1}{d_{\text{in}}}$, which causes signals to decay exponentially by a factor of $0.5^L$ across $L$ ReLU layers.

---

### Q3: What is the "Dying ReLU" problem, what causes it, and how is it resolved?
- **Root Cause**:
  The gradient of ReLU is 0 for all negative inputs: $\frac{d}{dz}\max(0, z) = 0$ when $z < 0$. If a neuron's weights receive an excessively large gradient update, the weights may adjust such that $w^T x + b < 0$ for all samples in the training distribution.
  Once this occurs:
  1. The output activation is permanently $0$.
  2. The gradient flowing through the neuron is permanently $0$.
  3. The optimizer can never update the weights again. The neuron is functionally "dead".
- **Diagnostic Symptoms**:
  A significant percentage (e.g., $>40\%$) of hidden units in intermediate layers outputting zero across entire mini-batches; sudden plateau in validation loss.
- **Mitigation Strategies**:
  1. Reduce the learning rate to prevent oversized parameter updates.
  2. Use **Leaky ReLU** ($\max(\alpha z, z)$ with $\alpha = 0.01$) or **PReLU** (parametric $\alpha$ learned via gradient descent).
  3. Use smooth, non-monotonic activations such as **GELU** or **SiLU/Swish**, which retain small non-zero gradients in negative regimes.

---

## 2. Optimization & Algorithmic Mechanics

### Q4: Why is $L_2$ Regularization fundamentally different from Weight Decay in Adam (leading to AdamW)?
- **In Standard SGD**:
  $$\nabla (L + \frac{\lambda}{2}\|\theta\|^2) = g_t + \lambda \theta$$
  $$\theta_{t+1} = \theta_t - \eta (g_t + \lambda \theta_t) = (1 - \eta \lambda) \theta_t - \eta g_t$$
  $L_2$ regularization is mathematically identical to weight decay.
- **In Adaptive Optimizers (Adam)**:
  Adam scales updates inversely by the historical second moment $\sqrt{v_t}$:
  $$\Delta \theta_t \propto \frac{g_t + \lambda \theta_t}{\sqrt{v_t} + \epsilon}$$
  - For weights with **frequently large gradients** ($v_t \gg 1$), the regularizer $\lambda \theta_t$ is divided by a large denominator, causing **under-regularization**.
  - For weights with **infrequent/sparse gradients** ($v_t \ll 1$), the regularizer is divided by a small denominator, causing **over-regularization**.
- **The AdamW Solution (Loshchilov & Hutter, 2019)**:
  Decouple weight decay completely from gradient moments:
  $$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
  This ensures every parameter decays at rate proportional to its current magnitude, restoring the true intended regularization effect.

---

### Q5: Why is bias correction necessary in Adam, and what would happen without it?
- **Exponential Moving Average Dynamics**:
  Adam initializes moment accumulators to zero: $m_0 = 0, v_0 = 0$.
  Expanding $m_t$ recursively:
  $$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$
  Taking the expectation under the assumption that gradients $g_i$ come from a stationary distribution with mean $\mathbb{E}[g]$:
  $$\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i\right] = \mathbb{E}[g] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g] (1 - \beta_1^t)$$
- **Consequence of Omitting Bias Correction**:
  At $t=1$ with $\beta_2 = 0.999$, $1 - \beta_2^1 = 0.001$. Without dividing by $(1 - \beta_2^t)$, $v_1$ is underestimated by a factor of 1000!
  Because $v_t$ appears in the denominator $\sqrt{v_t}$, this causes the initial update steps to explode violently:
  $$\Delta \theta_1 \approx \frac{\eta \cdot 0.1 g_1}{\sqrt{0.001 g_1^2}} = \frac{0.1}{\sqrt{0.001}} \eta \approx 3.16 \cdot \eta \cdot \text{sign}(g_1)$$
  Bias correction $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ guarantees that $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$ and $\mathbb{E}[\hat{v}_t] = \mathbb{E}[g^2]$ right from step $t=1$.

---

## 3. Whiteboard Coding Drills

### Q6: Write a numerically stable Softmax + Cross-Entropy loss forward and backward pass in pure NumPy.
```python
import numpy as np

def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray):
    """
    logits: (N, C) unnormalized raw scores
    targets: (N,) integer class indices
    Returns: loss (float), d_logits (N, C)
    """
    N = logits.shape[0]
    # Shift logits by max for numerical overflow protection
    shift_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shift_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Cross-entropy loss with epsilon clipping
    eps = 1e-15
    correct_log_probs = np.log(probs[np.arange(N), targets] + eps)
    loss = -np.mean(correct_log_probs)
    
    # Analytical gradient: (P - Y) / N
    d_logits = probs.copy()
    d_logits[np.arange(N), targets] -= 1.0
    d_logits /= N
    
    return float(loss), d_logits
```
