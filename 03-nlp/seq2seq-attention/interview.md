# Seq2Seq with Attention - Technical Interview Preparation

A curated question bank covering sequence-to-sequence architectures, attention mechanisms (Bahdanau vs. Luong), teacher forcing, exposure bias, and evaluation metrics (BLEU, ROUGE).

---

## 1. Core Architectural Concepts

### Q1: What was the "information bottleneck" of vanilla Seq2Seq models, and how does the attention mechanism eliminate it?
- **The Information Bottleneck**:
  In original Seq2Seq models (Sutskever et al., 2014; Cho et al., 2014), the entire variable-length source sequence $\mathbf{x} = (x_1, \dots, x_{T_x})$ is compressed into a single, fixed-size hidden vector $\mathbf{h}_{T_x}$ (the "thought vector" or context vector $\mathbf{c}$). As sequence length increases beyond 20–30 tokens, compressing all semantic, syntactic, and morphological information into a single vector of 512 or 1024 dimensions leads to severe information loss. The model suffers catastrophic forgetting of early tokens.
- **The Attention Solution**:
  Instead of compressing the input into one static vector, the encoder outputs a sequence of hidden state annotations $(\mathbf{h}_1, \dots, \mathbf{h}_{T_x})$. At *each* decoding step $t$, the decoder generates dynamic attention weights $\alpha_{t, i}$ indicating how relevant each source token $i$ is to producing the current target token $y_t$. The context vector $\mathbf{c}_t = \sum_{i=1}^{T_x} \alpha_{t, i} \mathbf{h}_i$ becomes a dynamic, step-specific weighted sum, effectively creating a direct shortcut path from every encoder state to the decoder.

---

### Q2: Compare Bahdanau (Additive) Attention vs. Luong (Multiplicative) Attention in detail.
- **Formulation Comparison**:
  | Feature | Bahdanau (Additive) Attention (2014) | Luong (Multiplicative) Attention (2015) |
  | :--- | :--- | :--- |
  | **Score Function** | $\mathbf{v}_a^\top \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_i)$ | Dot: $\mathbf{s}_t^\top \mathbf{h}_i$, General: $\mathbf{s}_t^\top \mathbf{W}_a \mathbf{h}_i$, Concat: $\mathbf{v}_a^\top \tanh(\mathbf{W}_a [\mathbf{s}_t; \mathbf{h}_i])$ |
  | **Timing / State** | Uses decoder previous state $\mathbf{s}_{t-1}$ to compute $\mathbf{c}_t$ *before* generating $\mathbf{s}_t$ | Uses current state $\mathbf{s}_t$ (after RNN transition) to compute $\mathbf{c}_t$ |
  | **Output Layer** | $[\mathbf{s}_t; \mathbf{c}_t; \mathbf{y}_{t-1}] \to \text{linear}$ | $\tilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t; \mathbf{s}_t]) \to \text{linear}$ |
  | **Computational Complexity** | Slower; involves matrix addition inside non-linearity | Faster; matrix multiplication is highly optimized on GPUs |
  | **Receptive Field** | Global soft alignment | Global or Local (Gaussian window around aligned position) |

---

### Q3: What is "Teacher Forcing", what is "Exposure Bias", and how can exposure bias be mitigated?
- **Teacher Forcing**:
  During training, instead of feeding the decoder's own predicted token $\hat{y}_{t-1}$ as input to step $t$, the ground-truth token $y_{t-1}^*$ is fed. This stabilizes and accelerates training, preventing early errors from cascading throughout the entire sequence.
- **Exposure Bias**:
  During inference, ground-truth tokens are unavailable; the decoder must consume its own potentially erroneous autoregressive predictions $\hat{y}_{t-1}$. Because the model was only ever exposed to perfect prefixes during training, a single mistake at step $t$ shifts the input distribution into unseen territory, leading to compounding hallucination or premature termination.
- **Mitigation Techniques**:
  1. **Scheduled Sampling (Bengio et al., 2015)**: Anneal the probability of using ground-truth tokens vs. model predictions from $1.0 \to \epsilon$ over training steps.
  2. **Beam Search**: Explore top-$K$ hypotheses simultaneously at inference time instead of greedy single-token sampling.
  3. **Reinforcement Learning / Policy Gradient (MIXER / Actor-Critic)**: Optimize sequence-level metrics directly (e.g., BLEU, ROUGE) via REINFORCE, training the model directly in inference decoding mode.

---

## 2. Evaluation Metrics

### Q4: How is BLEU score calculated, and why is the Brevity Penalty necessary?
- **Modified N-gram Precision**:
  Standard precision can be gamed by repeating a frequent valid word (e.g., predicting "the the the the" would get $100\%$ precision against a reference containing "the"). BLEU clips matching n-gram counts to the maximum frequency with which that n-gram appears in any reference sentence:
  $$p_n = \frac{\sum_{C \in \{\text{Candidates}\}} \sum_{n\text{-gram} \in C} \text{Count}_{\text{clip}}(n\text{-gram})}{\sum_{C \in \{\text{Candidates}\}} \sum_{n\text{-gram} \in C} \text{Count}(n\text{-gram})}$$
- **Brevity Penalty (BP)**:
  Without a penalty, very short translations could achieve artificially high precision (e.g., generating only 1 single correct word achieves $p_1 = 1.0$). If candidate length $c$ is shorter than reference length $r$:
  $$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \le r \end{cases}$$
- **Final BLEU**:
  $$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \log p_n \right)$$

---

## 3. Whiteboard Coding Drills

### Q5: Implement Luong Dot and General Attention scoring in vectorized PyTorch.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim: int, score_type: str = "dot"):
        super().__init__()
        self.score_type = score_type
        if score_type == "general":
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        # query: (B, 1, H) - decoder hidden state at step t
        # keys:  (B, S, H) - all encoder states
        if self.score_type == "dot":
            scores = torch.bmm(query, keys.transpose(1, 2))  # (B, 1, S)
        elif self.score_type == "general":
            projected = self.linear(keys)                      # (B, S, H)
            scores = torch.bmm(query, projected.transpose(1, 2)) # (B, 1, S)
        else:
            raise ValueError(f"Unknown score type {self.score_type}")

        weights = F.softmax(scores, dim=-1)                   # (B, 1, S)
        context = torch.bmm(weights, keys)                    # (B, 1, H)
        return context, weights
```
