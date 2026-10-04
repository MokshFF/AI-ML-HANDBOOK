# Language Modeling - Technical Interview Preparation

A curated question bank covering autoregressive language modeling, perplexity derivation, sampling algorithms, and exposure bias.

---

## 1. Core Mathematical & Conceptual Foundations

### Q1: Prove that Perplexity is mathematically equivalent to the exponential of cross-entropy loss.
- **Derivation**:
  1. Let sequence $\mathbf{w} = (w_1, \dots, w_N)$ have empirical distribution $P$ and model distribution $Q$.
  2. The cross-entropy loss under natural logarithm base $e$ is:
     $$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^N \log Q(w_i \mid w_{<i}) = - \frac{1}{N} \log \prod_{i=1}^N Q(w_i \mid w_{<i}) = - \frac{1}{N} \log Q(\mathbf{w})$$
  3. Perplexity is defined as the inverse geometric mean of sequence probability:
     $$\text{PPL}(\mathbf{w}) = Q(\mathbf{w})^{-1/N} = \left( \exp(\log Q(\mathbf{w})) \right)^{-1/N} = \exp\left( - \frac{1}{N} \log Q(\mathbf{w}) \right) = \exp(\mathcal{L}_{\text{CE}})$$
  4. If base-2 logarithm is used, $\text{PPL} = 2^{\mathcal{H}(P, Q)}$.

---

### Q2: Why does Top-$p$ (Nucleus) sampling outperform fixed Top-$k$ sampling across variable generation contexts?
- **Failure Mode of Fixed Top-$k$**:
  - In high-certainty contexts (e.g., `"The capital of France is"`), the model puts $99\%$ probability on `"Paris"`. A fixed $k=50$ still samples from 49 unlikely tokens, risking hallucination.
  - In flat, high-entropy contexts (e.g., opening a novel), there may be hundreds of plausible continuations. A small $k=10$ artificially truncates viable, creative candidates.
- **Why Top-$p$ Solves This**:
  Nucleus sampling dynamically expands and contracts the candidate pool based on the cumulative probability distribution. When the distribution is peaked, the nucleus contains 1–2 tokens; when flat, it expands to include dozens, matching the true entropy of the prediction context.

---

### Q3: What is "Weight Tying" between input embeddings and output projection, and why is it beneficial?
- **Definition**:
  Setting $\mathbf{W}_{\text{out}} = \mathbf{W}_{\text{in}}^\top$, sharing weights between the token embedding lookup matrix and the final linear vocabulary projection layer (Press & Wolf, 2016).
- **Benefits**:
  1. *Parameter Savings*: For a vocabulary of 50,000 and hidden dimension 4,096, a single embedding matrix takes $50,000 \times 4,096 \approx 204\text{M}$ parameters. Tying eliminates 204M redundant parameters.
  2. *Regularization & Representation Alignment*: It forces word input vectors and output classification hyperplanes to reside in the exact same vector space, preventing over-parameterization and improving generalization.

---

## 2. Whiteboard Coding Drills

### Q4: Implement Top-$p$ (Nucleus) sampling with temperature filtering from scratch in PyTorch.
```python
import torch
import torch.nn.functional as F

def nucleus_sample(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Samples a token from logits using Nucleus (Top-p) filtering and temperature scaling.
    """
    scaled = logits / max(temperature, 1e-5)
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens exceeding cumulative probability top_p
    mask_to_remove = cumulative_probs > top_p
    # Keep at least the first token
    mask_to_remove[..., 1:] = mask_to_remove[..., :-1].clone()
    mask_to_remove[..., 0] = False

    indices_to_remove = sorted_indices[mask_to_remove]
    scaled[indices_to_remove] = float('-inf')

    probs = F.softmax(scaled, dim=-1)
    sampled_id = torch.multinomial(probs, num_samples=1)
    return int(sampled_id.item())
```
