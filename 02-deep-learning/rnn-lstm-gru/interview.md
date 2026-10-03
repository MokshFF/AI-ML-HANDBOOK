# Recurrent Neural Networks - Technical Interview Preparation

A curated question bank covering BPTT derivations, vanishing gradient proofs, LSTM/GRU gating equations, and bidirectional architectures.

---

## 1. Sequence Dynamics & Gradient Proofs

### Q1: Mathematically derive why plain RNNs suffer from vanishing and exploding gradients during Backpropagation Through Time (BPTT).
- **Mathematical Derivation**:
  Consider unrolling an RNN across $T$ timesteps with state $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$.
  For a loss $L_T$ computed at the final timestep, its gradient with respect to hidden state $h_1$ is given by the chain rule:
  $$\frac{\partial L_T}{\partial h_1} = \frac{\partial L_T}{\partial h_T} \prod_{k=2}^T \frac{\partial h_k}{\partial h_{k-1}}$$
  Each Jacobian matrix factor is:
  $$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - h_k^2) \cdot W_{hh}^T$$
- **Eigenvalue Decomposition**:
  Assuming $W_{hh}$ has eigendecomposition $W_{hh} = Q \Lambda Q^{-1}$:
  $$(W_{hh}^T)^{T-1} = Q^{-T} \Lambda^{T-1} Q^T$$
  Because $|\text{diag}(1 - h_k^2)| \le 1$ everywhere:
  1. **If all eigenvalues $|\lambda_i| < 1$**: The product decays exponentially as $(\lambda_{\max})^{T-1} \to 0$. Gradients for tokens early in the sequence vanish to zero, rendering long-term dependencies unlearnable.
  2. **If any eigenvalue $|\lambda_i| > 1$**: The product explodes exponentially as $(\lambda_{\max})^{T-1} \to \infty$, causing weight divergence and `NaN` losses.

---

### Q2: Why is the LSTM Forget Gate bias initialized to +1.0 (the Jozefowicz trick)?
- **Mathematical Rationale**:
  The forget gate activation is $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$.
  If bias $b_f$ is initialized to 0, then at initialization when weights are near zero, $\sigma(0) = 0.5$.
  - This means that at the start of training, the cell state retains only $50\%$ of its prior memory at each timestep: $C_t \approx 0.5 C_{t-1}$.
  - Over a sequence of length 20, memory decays to $(0.5)^{20} \approx 10^{-6}$, effectively re-introducing the vanishing gradient problem before the network has learned anything!
- **Setting $b_f = 1.0$ (or $+2.0$)**:
  $\sigma(1.0) \approx 0.73$ and $\sigma(2.0) \approx 0.88$.
  This ensures that by default, the network **remembers everything** at the beginning of training, and only learns to forget information when explicitly driven by loss gradients.

---

## 2. Gating Mechanics & Architecture Trade-offs

### Q3: How does the Gated Recurrent Unit (GRU) differ from an LSTM, and when should you choose one over the other?
- **Key Architectural Differences**:
  1. **Merged Memory**: GRU merges the cell state $C_t$ and hidden state $h_t$ into a single state $h_t$.
  2. **Coupled Gates**: GRU does not have a separate forget and input gate; the update gate $z_t$ performs both simultaneously:
     $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
  3. **Parameter Count**:
     - LSTM: $4 \times (d \cdot h + h^2 + h)$
     - GRU: $3 \times (d \cdot h + h^2 + h)$ ($\mathbf{25\% \text{ reduction in parameters}}$).
- **When to Choose**:
  - **Choose GRU**: When computational efficiency, training latency, or limited GPU memory is critical, or on small datasets where LSTM is prone to overfitting.
  - **Choose LSTM**: When modeling long, complex sequences with distinct short-term and long-term memory requirements, or where empirical benchmarks show GRU underfitting.

---

### Q4: Why can Bidirectional RNNs be used for sequence classification or translation encoders, but CANNOT be used for autoregressive language modeling?
- **Causality Constraint**:
  - In autoregressive generation (e.g., GPT next-token prediction), the task is to predict $x_{t+1}$ conditioned strictly on past tokens: $P(x_{t+1} \mid x_1, \dots, x_t)$.
  - A Bidirectional model computes $\overleftarrow{h}_t$ by processing the sequence backwards from $x_T$ down to $x_t$.
  - If a bidirectional network were used for autoregressive training, the backward pass would directly see the target token $x_{t+1}$ during the forward pass, creating catastrophic **future label leakage**. The model would simply memorize copying the target without learning meaningful representations.
- **Valid Use Cases for BiRNNs**:
  - Non-causal tasks where the complete sequence is known beforehand: Named Entity Recognition (NER), Sentiment Classification, Part-of-Speech Tagging, and Seq2Seq Encoders.

---

## 3. Whiteboard Coding Drills

### Q5: Write a clean PyTorch implementation of an LSTM Cell from raw tensor operations.
```python
import torch

class CustomLSTMCell:
    def __init__(self, in_features: int, hidden_dim: int):
        # Combined weight for [f, i, c, o] gates
        self.W_ih = torch.randn(4 * hidden_dim, in_features) * 0.05
        self.W_hh = torch.randn(4 * hidden_dim, hidden_dim) * 0.05
        self.bias = torch.zeros(4 * hidden_dim)
        # Forget gate bias trick
        self.bias[:hidden_dim] = 1.0

    def __call__(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        # Compute all 4 gates in a single matrix multiply
        gates = x @ self.W_ih.T + h_prev @ self.W_hh.T + self.bias
        f, i, c_cand, o = gates.chunk(4, dim=-1)
        
        f_t = torch.sigmoid(f)
        i_t = torch.sigmoid(i)
        c_cand_t = torch.tanh(c_cand)
        o_t = torch.sigmoid(o)
        
        c_t = f_t * c_prev + i_t * c_cand_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
```
