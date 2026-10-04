# Language Modeling - References & Further Reading

Foundational citations and seminal papers on statistical language modeling, causal autoregressive transformers, and generation decoding strategies.

---

## 1. Seminal Research Papers

- **Improving Language Understanding by Generative Pre-Training (GPT-1)** (2018)
  - *Authors*: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI)
  - *Contribution*: Demonstrated that generative pre-training on unlabeled text followed by discriminative fine-tuning achieves state-of-the-art across broad NLP benchmarks.

- **Language Models are Unsupervised Multitask Learners (GPT-2)** (2019)
  - *Authors*: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI)
  - *Contribution*: Demonstrated zero-shot task transfer capabilities through scaling up autoregressive causal models.

- **The Curious Case of Neural Text Degeneration (Nucleus Sampling)** (2019 / 2020)
  - *Authors*: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
  - *Paper*: [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
  - *Contribution*: Identified that maximization-based decoding (beam search/greedy) causes repetition and unnatural text; introduced Nucleus (Top-$p$) sampling.

- **Using the Output Embedding to Improve Language Models (Weight Tying)** (2016 / 2017)
  - *Authors*: Ofir Press, Lior Wolf
  - *Paper*: [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
  - *Contribution*: Proved tying input word embeddings to softmax output projection matrix reduces perplexity and cuts parameters in half.

- **Hierarchical Neural Story Generation (Top-$k$ Sampling)** (2018)
  - *Authors*: Angela Fan, Mike Lewis, Yann Dauphin
  - *Paper*: [arXiv:1805.04833](https://arxiv.org/abs/1805.04833)
  - *Contribution*: Introduced Top-$k$ sampling to prevent sampling from the improbable tail of the distribution.

---

## 2. Textbooks & Survey Articles

- **Speech and Language Processing (3rd ed. draft)**
  - *Authors*: Dan Jurafsky, James H. Martin
  - *Chapters*: Chapter 3 (N-gram Language Models), Chapter 9 (RNNs & Language Models), Chapter 10 (Transformers & Pretraining).
  - *URL*: [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
