# Text Preprocessing - Technical Interview Preparation

A curated question bank covering tokenization algorithms, subword modeling (BPE), stemming vs. lemmatization, and TF-IDF mathematics.

---

## 1. Tokenization & Morphological Analysis

### Q1: How does Byte-Pair Encoding (BPE) construct its subword vocabulary, and why does it solve the Out-Of-Vocabulary (OOV) problem?
- **Vocabulary Construction Algorithm**:
  1. *Initialization*: Tokenize raw training text into characters. Add a special end-of-word symbol (e.g., `</w>`). The base vocabulary consists of all individual characters (e.g., ASCII/Unicode bytes).
  2. *Frequency Counting*: Count the frequencies of all adjacent character/token pairs across the entire corpus.
  3. *Iterative Merging*: Identify the single most frequent adjacent pair (e.g., `'e'` and `'s'`) and merge it into a single new vocabulary token (`'es'`).
  4. *Repeat*: Repeat for $K$ merge iterations until the target vocabulary size (e.g., 32,000 or 50,000) is reached.
- **Why It Eliminates the OOV Problem**:
  Because the base vocabulary retains all individual characters and bytes (especially Byte-level BPE used in GPT-2/LLaMA), any completely unseen word, typo, or foreign term can always be decomposed down to its constituent subwords or individual characters. The model never encounters an unknown `[UNK]` token.

---

### Q2: Compare Stemming and Lemmatization: What are the practical trade-offs in search engines and NLP pipelines?
- **Algorithmic Differences**:
  - **Stemming**: Applies heuristic string rules (like Porter or Snowball) that slice off common grammatical prefixes and suffixes (`-ing`, `-ed`, `-s`, `-tion`). It is context-free, ignores part-of-speech, and often produces non-words (`university` $\to$ `univers`).
  - **Lemmatization**: Uses complete morphological dictionaries (like WordNet) and requires Part-of-Speech (POS) tagging to resolve words to their true canonical dictionary headword (lemma).
- **Practical Trade-offs**:
  - *Speed & Scale*: Stemming is orders of magnitude faster ($O(1)$ string slicing) and consumes near-zero memory, making it historically popular for high-throughput search engine indexing (Elasticsearch, Lucene).
  - *Precision*: Lemmatization prevents erroneous conflations (e.g., stemming might conflate `organization` and `organ` into `organ`), which is vital for clinical NLP, legal document analysis, and translation.

---

## 2. Statistical Vector Space Models

### Q3: Explain the mathematical rationale behind TF-IDF smoothing and L2 normalization.
- **Formula**:
  $$\text{IDF}(t) = \log\left( \frac{1 + N}{1 + \text{df}(t)} \right) + 1$$
- **Why Smooth the Denominator ($1 + \text{df}$)**:
  Prevents zero-division errors if a test document contains a term not observed in the training corpus ($\text{df}(t) = 0$).
- **Why Smooth the Numerator ($1 + N$)**:
  Ensures that terms appearing in *every* document ($\text{df}(t) = N$) receive a strictly positive IDF rather than $\log(1) = 0$, preventing their frequency information from being completely discarded.
- **Why Add $+1$ to the Logarithm**:
  Ensures non-zero IDF weights for all terms, preserving baseline term frequency signals.
- **Why L2 Normalization**:
  Long documents naturally contain higher raw word counts than short documents simply because of author verbosity. Without $L_2$ normalization ($\frac{\mathbf{v}}{\|\mathbf{v}\|_2}$), long documents would have artificially massive dot-product similarities. $L_2$ normalization converts the vector into directional unit coordinates on the hypersphere, ensuring cosine similarity measures **topical focus** rather than document length.

---

### Q4: What are the fundamental limitations of Bag-of-Words and TF-IDF representations?
1. **Sparsity & High Dimensionality**: Vocabularies contain 50,000+ words, resulting in vectors that are $>99.9\%$ sparse, consuming high memory.
2. **Total Loss of Syntax & Word Order**: "Dog bites man" and "Man bites dog" yield identical Bag-of-Words vectors despite conveying opposite meanings.
3. **Orthogonality & Zero Semantic Similarity**: Distinct synonyms like "automobile" and "car" have completely disjoint one-hot indices. In TF-IDF space, their inner product is $0$, completely failing to recognize their semantic equivalence. This limitation motivated dense, continuous **word embeddings** (Word2Vec, GloVe).

---

## 3. Whiteboard Coding Drills

### Q5: Write a pure NumPy implementation of TF-IDF transform with L2 normalization.
```python
import numpy as np

def compute_tfidf(doc_term_matrix: np.ndarray, doc_frequencies: np.ndarray, N: int):
    """
    doc_term_matrix: (num_docs, vocab_size) raw term counts
    doc_frequencies: (vocab_size,) number of docs containing each term
    N: total number of docs
    Returns: normalized_tfidf (num_docs, vocab_size)
    """
    # Smooth IDF
    idf = np.log((1.0 + N) / (1.0 + doc_frequencies)) + 1.0
    
    # TF * IDF
    tfidf = doc_term_matrix.astype(np.float64) * idf
    
    # L2 row normalization
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return tfidf / norms
```
