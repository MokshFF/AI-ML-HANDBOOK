# Natural Language Processing Interview Questions

Tokenization mechanics, word representations, sequence architectures, decoding strategies, and linguistic evaluation.

---

## 1. Beginner Questions

### Q1: Subword Tokenization: Byte-Pair Encoding (BPE) vs WordPiece vs Unigram
- **Tags**: `Conceptual` | `Coding` | `Practical`
- **Short Answer**: Subword tokenization handles Out-Of-Vocabulary (OOV) words by decomposing rare words into frequent character n-grams. BPE iteratively merges the most frequent adjacent byte/character pairs. WordPiece merges pairs that maximize the training corpus language model likelihood. Unigram starts with a huge seed vocabulary and iteratively prunes candidates based on likelihood loss.
- **Detailed Explanation**:
  - **Character tokenization**: Small vocab ($< 300$), but sequences become $5\times$ longer, exploding quadratic attention memory.
  - **Word tokenization**: Huge vocab ($> 500,000$), massive embedding tables, and fails completely on unseen words or misspellings (`"unhapppppppy"` $\to \text{[UNK]}$).
  - **Byte-Pair Encoding (BPE)**:
    1. Initialize vocabulary with all base characters/bytes.
    2. Count frequency of all adjacent symbol pairs in corpus.
    3. Merge the single most frequent pair $(c_1, c_2) \to c_{12}$.
    4. Repeat until reaching desired target vocabulary size (e.g. 32,000 in LLaMA or 100,000 in GPT-4).
- **Example**: The word `"electrochemical"` may not exist in training text, but BPE segments it into `["electro", "##chemical"]` or `["elec", "tro", "chem", "ical"]`, allowing the network to compose semantic representations from familiar stems.
- **Common Misconception**: Assuming BPE whitespace handling is identical across models. GPT-2/GPT-4 uses byte-level BPE with a leading space marker `Ġ`, meaning `" apple"` and `"apple"` have different token IDs.
- **Follow-Up Questions**:
  1. *Why does Byte-Fallback ensure a zero-OOV rate in modern tokenizers?*
  2. *How does tokenization bias arithmetic and programming code performance in LLMs?*

---

## 2. Intermediate Questions

### Q2: LLM Decoding Strategies: Greedy vs Beam Search vs Temperature & Top-p (Nucleus)
- **Tags**: `Conceptual` | `Mathematical` | `Gotcha`
- **Short Answer**: Greedy decoding picks the single argmax token at each step (deterministic, prone to repetitive loops). Beam search tracks top-B cumulative likelihood trajectories (great for translation/summarization, poor for open-ended creative dialogue). Temperature scales logit entropy, while Top-p (Nucleus) sampling truncates candidate tokens to the smallest cumulative probability mass $\ge p$.
- **Detailed Explanation**:
  Given logits $z_i$:
  1. **Temperature Scaling**:
     $$P(y_t = w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
     - $T \to 0$: Collapses distribution into argmax (greedy).
     - $T = 1$: True model probability.
     - $T > 1$: Flattens distribution toward uniform (random, creative, higher hallucination risk).
  2. **Top-k Sampling**: Retains only the top $k$ highest-probability tokens and renormalizes.
  3. **Top-p (Nucleus) Sampling**: Selects smallest set $V^{(p)}$ such that $\sum_{w \in V^{(p)}} P(w) \ge p$. Dynamically expands candidate pool when model is uncertain, and narrows it to 1-2 tokens when model is highly confident.
- **Example**: In code generation, high predictability is required: set $T = 0.2, \text{top\_p} = 0.95$. In creative storytelling, set $T = 0.8, \text{top\_p} = 0.9$.
- **Common Misconception**: Thinking Beam Search is always superior to sampling for LLMs. For open-ended generation, beam search tends to output unnatural, repetitive, generic sequences ("the cat was on the mat on the mat").
- **Follow-Up Questions**:
  1. *What causes degeneration and repetition in autoregressive language models?*
  2. *How does Min-P sampling improve over Top-P on dynamic logit tail distributions?*
