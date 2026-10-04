# Word Embeddings - Technical Interview Preparation

A curated question bank covering Word2Vec derivations, negative sampling dynamics, GloVe matrix factorization, and polysemy trade-offs.

---

## 1. Word2Vec & Negative Sampling Mechanics

### Q1: Why is standard Softmax intractable for Word2Vec on large vocabularies, and how does Negative Sampling reduce complexity?
- **Intractability of Full Softmax**:
  The probability of predicting context word $w_O$ given center word $w_I$ via standard softmax is:
  $$P(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w=1}^V \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})}$$
  The denominator requires summing over all $V$ words in the vocabulary. With $V = 100,000$ and embedding dimension $D = 300$, computing the partition function and its gradient requires $3 \times 10^7$ floating-point operations *per training token*. Over a billion-word corpus, this is computationally prohibitive.
- **Negative Sampling Solution (SGNS)**:
  Instead of normalizing over the entire dictionary, SGNS re-frames the task as distinguishing the true positive context word from $K$ randomly sampled noise words (negative samples).
  The loss for a single word pair $(w_I, w_O)$ with $K$ noise words $\{w_1, \dots, w_K\}$ is:
  $$\mathcal{L} = -\log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{k=1}^K \log \sigma(-\mathbf{v}'_{w_k} \cdot \mathbf{v}_{w_I})$$
  - The computational complexity drops from $\mathcal{O}(V \cdot D)$ to $\mathcal{O}(K \cdot D)$.
  - With $K \approx 5$ to $20$, training speed increases by several orders of magnitude while preserving high embedding quality.

---

### Q2: Why is the negative sampling distribution raised to the power of 0.75 ($U(w)^{3/4}$)?
- **Mathematical Rationale**:
  Let $U(w)$ be the unigram frequency distribution of words in the corpus:
  $$P_n(w) = \frac{U(w)^{3/4}}{\sum_{w'} U(w')^{3/4}}$$
- **Effect on Probabilities**:
  Consider three words with raw corpus frequencies:
  - Frequent word ("the"): $U(w) = 0.01 \implies (0.01)^{0.75} \approx 0.0316$
  - Rare word ("platypus"): $U(w) = 0.000001 \implies (10^{-6})^{0.75} = 10^{-4.5} \approx 0.0000316$
  Taking the $0.75$ power dampens the probability of extremely common words (which would otherwise dominate $>90\%$ of negative samples without providing informative signal), while significantly boosting the relative sampling probability of rare domain terms by over **30x**, giving the model adequate negative signal across the full vocabulary.

---

## 2. Model Comparisons & Contextualization Limits

### Q3: Compare Word2Vec and GloVe: What are the fundamental conceptual and mathematical differences?
- **Word2Vec**:
  - *Method*: Predictive, online streaming.
  - *Mechanism*: Slides a local window across text; optimizes parameters using local stochastic gradient descent.
  - *Limitation*: Fails to explicitly utilize vast statistical information contained in global co-occurrence frequencies across the whole corpus.
- **GloVe**:
  - *Method*: Count-based matrix factorization.
  - *Mechanism*: Scans the entire corpus once to build a global sparse co-occurrence matrix $X \in \mathbb{R}^{V \times V}$. Then fits a log-bilinear weighted least squares model $\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \approx \log X_{ij}$.
  - *Advantage*: Explicitly optimizes global statistics while downweighting noisy or rare co-occurrences via weighting function $f(X_{ij})$.

---

### Q4: What is the fundamental limitation of static word embeddings (Word2Vec, GloVe, fastText) that led to the emergence of contextualized models (BERT, ELMo)?
- **The Polysemy Bottleneck**:
  Static embeddings assign a **single, fixed vector** $\mathbf{v}_w \in \mathbb{R}^D$ to each word in the vocabulary, regardless of its sentence context.
  Consider the word "apple":
  1. *"He took a bite of the crisp red **apple**."* (Fruit)
  2. *"**Apple** announced record quarterly earnings on Wall Street."* (Technology Corporation)
  In Word2Vec, the vector $\mathbf{v}_{\text{apple}}$ is forced to occupy an artificial linear compromise between technology terms (Microsoft, Google) and agricultural produce (orange, banana).
- **The Contextualized Solution**:
  Models like BERT and Transformers dynamically compute contextualized token representations $h_t = f(w_1, \dots, w_t, \dots, w_N)$ via multi-head self-attention. The representation of "apple" in sentence 1 is close to "pear", while in sentence 2 it is close to "Nvidia".

---

## 3. Whiteboard Coding Drills

### Q5: Write a function to find the top-K most similar words to a target vector using cosine similarity.
```python
import numpy as np

def get_top_k_similar(target_vec: np.ndarray, embed_matrix: np.ndarray, top_k: int = 5):
    """
    target_vec: (D,) query vector
    embed_matrix: (V, D) word embedding matrix
    Returns: top_indices (list of int), top_scores (list of float)
    """
    # Normalize rows of embed_matrix and target_vec
    norms = np.linalg.norm(embed_matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embed_normalized = embed_matrix / norms
    
    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-12)
    
    # Vectorized cosine similarity: (V,)
    similarities = np.dot(embed_normalized, target_norm)
    
    # Sort descending
    top_indices = np.argsort(-similarities)[:top_k]
    top_scores = similarities[top_indices]
    
    return top_indices.tolist(), top_scores.tolist()
```
