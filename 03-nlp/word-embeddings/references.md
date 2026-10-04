# Word Embeddings - Curated References & Bibliography

A curated collection of foundational publications, landmark research papers, and technical documentation on distributed word representations.

---

## 1. Landmark Research Papers

### 1.1 Word2Vec Foundations
- **Efficient Estimation of Word Representations in Vector Space (CBOW & Skip-Gram)**
  - *Authors*: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (ICLR, 2013).
  - *Significance*: Introduced Continuous Bag-of-Words and Skip-Gram neural network architectures.
- **Distributed Representations of Words and Phrases and their Compositionality (Negative Sampling)**
  - *Authors*: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S. Corrado, Jeffrey Dean (NeurIPS, 2013).
  - *Significance*: Formulated Negative Sampling (SGNS), frequent word subsampling, and demonstrated linear semantic analogy arithmetic.

### 1.2 Global Co-occurrence & Subword Extensions
- **GloVe: Global Vectors for Word Representation**
  - *Authors*: Jeffrey Pennington, Richard Socher, Christopher D. Manning (EMNLP, 2014).
  - *Significance*: Unified local context prediction and global matrix factorization under a log-bilinear weighted least squares model.
- **Enriching Word Vectors with Subword Information (fastText)**
  - *Authors*: Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov (TACL, 2017).
  - *Significance*: Incorporated character n-grams into Skip-Gram, enabling robust out-of-vocabulary representation and morphological transfer.

### 1.3 Theoretical Analyses of Word Embeddings
- **Neural Word Embedding as Implicit Matrix Factorization**
  - *Authors*: Omer Levy, Yoav Goldberg (NeurIPS, 2014).
  - *Significance*: Proved that Skip-Gram with Negative Sampling is mathematically equivalent to factorizing a shifted Pointwise Mutual Information (PMI) matrix.

---

## 2. Textbooks & Monographs

- **Speech and Language Processing (3rd ed., Chapter 6: Vector Semantics and Embeddings)** by Daniel Jurafsky and James H. Martin (Stanford University).
- **Gensim Python Library Documentation**: [radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
- **GloVe Project Repository**: [nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
