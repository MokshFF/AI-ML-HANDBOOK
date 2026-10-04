# Language Modeling: N-Gram Smoothing, Causal Transformers & Decoding Strategies

A rigorous foundation of autoregressive language modeling spanning classical statistical n-gram formulations with Laplace smoothing, modern Pre-LN Causal Transformers (GPT architecture), intrinsic perplexity evaluation, and stochastic decoding strategies (Temperature, Top-$k$, and Nucleus / Top-$p$ sampling).

---

## 1. Classical Statistical Language Modeling

A language model computes the joint probability of a sequence of words $\mathbf{w} = (w_1, w_2, \dots, w_N)$ via the chain rule of probability:
$$P(w_1, \dots, w_N) = \prod_{i=1}^N P(w_i \mid w_1, \dots, w_{i-1})$$

### The Markov Assumption & N-Grams
Under an $(n-1)$-order Markov assumption, the next token depends only on the preceding $n-1$ tokens:
$$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-n+1}^{i-1})$$

### Laplace & Add-$\alpha$ (Lidstone) Smoothing
Raw Maximum Likelihood Estimation assigns zero probability to unseen n-grams, causing infinite perplexity. Add-$\alpha$ smoothing redistributes probability mass:
$$P_{\alpha}(w_i \mid w_{i-n+1}^{i-1}) = \frac{C(w_{i-n+1}^i) + \alpha}{C(w_{i-n+1}^{i-1}) + \alpha |V|}$$
where $|V|$ is vocabulary size, and $\alpha \in (0, 1]$ (Laplace smoothing sets $\alpha = 1$).

---

## 2. Neural Causal Language Models (GPT Architecture)

Modern large language models replace discrete counts with deep causal Transformer decoders.
- **Pre-LN Architecture**: LayerNorm is placed on the input paths before multi-head attention and feedforward blocks, stabilizing optimization for deep networks.
- **Causal Masking**: An upper-triangular mask of $-\infty$ is added to raw attention logits before softmax, guaranteeing that position $i$ only attends to positions $j \le i$:
  $$\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
  $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$
- **Weight Tying**: The output projection head shares the exact same parameter matrix as the input token embedding layer ($\mathbf{W}_{\text{head}} = \mathbf{W}_{\text{emb}}^\top$), drastically reducing parameter count and improving generalization.

```
Input Tokens:     [t_1]        [t_2]        [t_3]        [t_4]
                    │            │            │            │
             Token + Pos Embedding
                    │            │            │            │
             ┌──────▼────────────▼────────────▼────────────▼──────┐
             │            Causal Transformer Blocks               │
             │      (Lower-Triangular Masked Attention)          │
             └──────┬────────────┬────────────┬────────────┬──────┘
                    │            │            │            │
Logits (t+1):     [logits_2]   [logits_3]   [logits_4]   [logits_5]
Target:             t_2          t_3          t_4          t_5
```

---

## 3. Intrinsic Evaluation: Perplexity

Perplexity ($\text{PPL}$) measures the model's uncertainty when predicting test text. Mathematically, it is the exponential of the average negative log-likelihood (cross-entropy loss):
$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid w_{<i})$$
$$\text{PPL} = \exp(\text{Loss}) = 2^{\mathcal{H}(P, \hat{P})}$$
- **Interpretation**: A model with $\text{PPL} = 50$ is as uncertain as uniformly guessing among 50 equally probable words at each step. Lower is better.

---

## 4. Decoding & Sampling Strategies

Given next-token unnormalized logits $\mathbf{z} \in \mathbb{R}^{|V|}$:

1. **Greedy Decoding**:
   $$\hat{w} = \arg\max_i z_i$$
   Prone to repetitive loops and degenerate generation.
2. **Temperature Scaling ($T$)**:
   $$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   - $T \to 0$: Approaches greedy decoding (peaked, deterministic).
   - $T > 1$: Flattens distribution, increasing diversity.
3. **Top-$k$ Sampling (Fan et al., 2018)**:
   Restricts sampling to the $k$ most probable tokens, zeroing out the remaining long tail:
   $$V^{(k)} = \text{top } k \text{ tokens}; \quad p_i' = \frac{p_i}{\sum_{j \in V^{(k)}} p_j}$$
4. **Nucleus / Top-$p$ Sampling (Holtzman et al., 2019)**:
   Dynamically adjusts candidate pool size by choosing the smallest subset $V^{(p)}$ whose cumulative probability mass exceeds $p$:
   $$\sum_{i \in V^{(p)}} p_i \ge p$$
   Adapts to varying model certainty (few candidates when confident, many when uncertain).

---

## 5. Implementation Blueprint

- [`code/lm_engine.py`](code/lm_engine.py): Contains `NgramLanguageModel`, `MiniCausalLM`, `compute_perplexity`, `sample_next_token`, and `generate_sequence`.
- [`code/test_language_modeling.py`](code/test_language_modeling.py): Unit tests verifying scoring, causality, and sampling.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab comparing n-gram vs neural causal perplexity and generation behaviors.
