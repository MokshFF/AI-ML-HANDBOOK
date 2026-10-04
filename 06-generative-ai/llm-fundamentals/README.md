# LLM Fundamentals

> **Last reviewed:** 2026-10. Architectures and best practices in this field change quickly; verify details against the papers and docs in [`references.md`](references.md).

## Learning objectives
After this module you can explain and implement: BPE tokenization, token embeddings, positional encoding, a decoder-only Transformer, next-token-prediction pretraining, decoding (greedy, temperature, top-k, top-p), the KV cache, context-window management, and scaling-law reasoning.

## 1. Tokenization and BPE
LLMs operate on **tokens**, not characters or words. **Byte-Pair Encoding** (Sennrich et al., 2015) starts from characters (or bytes) and repeatedly merges the most frequent adjacent pair, recording the merges in order. Encoding replays merges in learned order, so unseen words fall back to smaller pieces; byte-level variants guarantee no unknown token. Tokenization affects cost (tokens are billed/limited), multilingual fairness (some languages need more tokens per word), and arithmetic/spelling behaviour.

## 2. Embeddings and positional information
Token ids index an embedding matrix $E\in\mathbb{R}^{V\times d}$. Self-attention is permutation-equivariant, so order must be injected:
$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d}\big),\qquad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d}\big)$$
Modern models commonly use **rotary** (RoPE) or relative/bias schemes (e.g. ALiBi) instead of absolute sinusoids; the `code/` module implements the classic sinusoidal form for clarity.

## 3. Decoder-only Transformer
Each block applies (pre-)LayerNorm, causal multi-head self-attention
$\text{softmax}(QK^\top/\sqrt{d_k}+M)V$ with a causal mask $M$, and an MLP, each with a residual connection. Output logits come from a final projection, often **weight-tied** to the embedding. Variants that reduce KV memory include multi-query and grouped-query attention.

## 4. Pretraining objective
Next-token prediction minimises the cross-entropy
$$\mathcal{L}=-\frac{1}{T}\sum_{t}\log p_\theta(x_t\mid x_{<t}),\qquad \text{PPL}=e^{\mathcal{L}}$$
over a very large corpus. Post-training (instruction tuning, preference optimisation) then adapts the base model; see [`../fine-tuning/`](../fine-tuning/) and [`../safety-alignment/`](../safety-alignment/).

## 5. Scaling laws
Loss tends to fall as a power law in parameters $N$ and data $D$ (Kaplan et al., 2020). Hoffmann et al. (2022) fit $L(N,D)=E+A/N^\alpha+B/D^\beta$ and, with compute $C\approx 6ND$, found that parameters and tokens should grow together; the often-quoted "~20 tokens per parameter" comes from their IsoFLOP/envelope analyses, while the parametric fit used in `compute_optimal` implies a larger, compute-dependent ratio (the notebook prints 50-120). That spread is itself a lesson: treat such constants as **empirical and setup-dependent**. Modern practice often trains smaller models on far more tokens because inference cost matters, not only training compute.

## 6. Inference, decoding and the KV cache
Generation is autoregressive. **Greedy** picks the argmax; **temperature** $T$ rescales logits $z/T$; **top-k** keeps the $k$ highest-probability tokens; **top-p (nucleus)** keeps the smallest set whose mass reaches $p$ (Holtzman et al., 2019). Without a cache every step would recompute keys/values for the full prefix; the **KV cache** stores them so each step processes only the new token. The cache grows linearly with sequence length, which is why long contexts are memory-bound (see [`../inference-optimization/`](../inference-optimization/)).

## 7. Context windows
The context window is the maximum tokens (prompt + output) a model can attend to. Longer windows cost more compute/memory, and models can under-use information in the middle of long inputs ("lost in the middle", Liu et al., 2023). Manage context with truncation that protects the system prefix, summarisation, or retrieval ([`../rag/`](../rag/)).

## Common mistakes
- Treating temperature 0 as "always correct" rather than "deterministic-ish" (hardware nondeterminism can remain).
- Applying top-p before temperature or forgetting to renormalise.
- Assuming tokens = words when budgeting context or cost.
- Quoting scaling-law constants as universal.

## Code and notebook
- [`code/llm_core.py`](code/llm_core.py): BPE, sinusoidal PE, `TinyGPT` with KV cache, `filter_logits`/`generate`, `fit_to_context`, scaling-law helpers.
- [`code/test_llm_core.py`](code/test_llm_core.py): 8 tests, including cache-vs-recompute equivalence and nucleus correctness.
- [`notebook.ipynb`](notebook.ipynb): runnable walkthrough.
