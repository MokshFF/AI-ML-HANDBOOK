# Transformer Models & Downstream Heads - References & Further Reading

Seminal research papers, benchmark datasets, and documentation for Transformer encoder architectures and task adaptation.

---

## 1. Seminal Research Papers

- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018 / 2019)
  - *Authors*: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
  - *Paper*: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
  - *Contribution*: Introduced Masked Language Modeling (MLM), Next Sentence Prediction (NSP), and fine-tuning paradigms across 11 NLP benchmarks.

- **RoBERTa: A Robustly Optimized BERT Pretraining Approach** (2019)
  - *Authors*: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
  - *Paper*: [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
  - *Contribution*: Demonstrated that BERT was significantly undertrained; removing NSP and training with larger mini-batches and dynamic masking substantially improves performance.

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** (2019)
  - *Authors*: Nils Reimers, Iryna Gurevych
  - *Paper*: [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
  - *Contribution*: Evaluated pooling strategies on top of BERT representations for semantic search and bi-encoder architectures.

- **SQuAD: 100,000+ Questions for Machine Comprehension of Text** (2016)
  - *Authors*: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang
  - *Paper*: [arXiv:1606.05250](https://arxiv.org/abs/1606.05250)
  - *Contribution*: Established the canonical extractive question answering benchmark.

---

## 2. Textbooks & Guides

- **Speech and Language Processing (3rd ed. draft)**
  - *Authors*: Dan Jurafsky, James H. Martin
  - *Chapter*: Chapter 11: Contextual Embeddings and Transformers.
- **Hugging Face Transformers Documentation**
  - *URL*: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
