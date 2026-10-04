# Seq2Seq with Attention - References & Further Reading

Curated citations, foundational research papers, and seminal works that introduced sequence-to-sequence modeling, attention mechanisms, and translation evaluation metrics.

---

## 1. Seminal Research Papers

- **Sequence to Sequence Learning with Neural Networks** (2014)
  - *Authors*: Ilya Sutskever, Oriol Vinyals, Quoc V. Le
  - *Paper*: [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
  - *Impact*: Introduced multi-layer LSTM sequence-to-sequence architectures and demonstrated that reversing the source sentence dramatically improves translation quality.

- **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation** (2014)
  - *Authors*: Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
  - *Paper*: [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
  - *Impact*: Introduced the Gated Recurrent Unit (GRU) and the formal encoder-decoder paradigm.

- **Neural Machine Translation by Jointly Learning to Align and Translate** (2014 / 2015)
  - *Authors*: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
  - *Paper*: [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
  - *Impact*: Introduced the additive attention mechanism, breaking the fixed-length vector bottleneck and paving the way for the Transformer.

- **Effective Approaches to Attention-based Neural Machine Translation** (2015)
  - *Authors*: Minh-Thang Luong, Hieu Pham, Christopher D. Manning
  - *Paper*: [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
  - *Impact*: Introduced multiplicative/dot-product attention, local attention windows, and input-feeding architectures.

- **BLEU: A Method for Automatic Evaluation of Machine Translation** (2002)
  - *Authors*: Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu
  - *Paper*: [ACL Anthology: P02-1040](https://aclanthology.org/P02-1040/)
  - *Impact*: Established modified n-gram precision with brevity penalty as the standard automatic evaluation metric for machine translation.

---

## 2. Textbooks & Survey Articles

- **Speech and Language Processing (3rd ed. draft)**
  - *Authors*: Dan Jurafsky, James H. Martin
  - *Chapters*: Chapter 10 (RNNs & LSTMs), Chapter 11 (Encoder-Decoder Models, Attention & Contextual Embeddings).
  - *URL*: [Stanford NLP Book](https://web.stanford.edu/~jurafsky/slp3/)

- **Deep Learning**
  - *Authors*: Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press)
  - *Chapter*: Chapter 10: Sequence Modeling: Recurrent and Recursive Nets.

---

## 3. Recommended Code Tutorials & Guides

- **PyTorch Official Tutorials**: *Translation with a Sequence to Sequence Network and Attention* (Sean Robertson).
- **Harvard NLP**: *The Annotated Encoder-Decoder* (OpenNMT technical report).
