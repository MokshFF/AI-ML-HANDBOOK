# Recurrent Neural Networks - Curated References & Bibliography

A curated collection of landmark research papers, seminal monographs, and official documentation on sequential deep learning.

---

## 1. Landmark Research Papers

### 1.1 Foundations & Early Recurrent Models
- **Finding Structure in Time (Elman RNN)**
  - *Author*: Jeffrey L. Elman (Cognitive Science, 1990).
  - *Significance*: Established the canonical Elman simple recurrent network architecture with recurrent context units.
- **Learning Long-Term Dependencies with Gradient Descent is Difficult**
  - *Authors*: Yoshua Bengio, Patrice Simard, Paolo Frasconi (IEEE Transactions on Neural Networks, 1994).
  - *Significance*: The mathematical proof formalizing vanishing and exploding gradients in recurrent networks.

### 1.2 Gated Architectures & Solutions
- **Long Short-Term Memory (LSTM)**
  - *Authors*: Sepp Hochreiter, Jürgen Schmidhuber (Neural Computation, 1997).
  - *Significance*: Introduced the constant error carousel and gating mechanism, revolutionizing sequential modeling.
- **Learning to Forget: Continual Prediction with LSTM (Forget Gate)**
  - *Authors*: Felix A. Gers, Jürgen Schmidhuber, Fred Cummins (Neural Computation, 2000).
  - *Significance*: Added the adaptive forget gate $f_t$ to the original LSTM cell.
- **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (GRU)**
  - *Authors*: Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, et al. (EMNLP, 2014).
  - *Significance*: Introduced the Gated Recurrent Unit (GRU) and the Seq2Seq encoder-decoder architecture.
- **An Empirical Exploration of Recurrent Network Architectures (Forget Gate Bias Trick)**
  - *Authors*: Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever (ICML, 2015).
  - *Significance*: Evaluated thousands of recurrent architectures; proved initializing forget gate bias to 1-2 matches or beats exotic variants.

### 1.3 Bidirectional & Sequence-to-Sequence Modeling
- **Bidirectional Recurrent Neural Networks**
  - *Authors*: Mike Schuster, Kuldip K. Paliwal (IEEE Transactions on Signal Processing, 1997).
  - *Significance*: Formulated simultaneous forward and backward sequence processing for complete context capture.
- **Sequence to Sequence Learning with Neural Networks**
  - *Authors*: Ilya Sutskever, Oriol Vinyals, Quoc V. Le (NeurIPS, 2014).
  - *Significance*: Multi-layer deep LSTM mapping arbitrary sequence lengths to arbitrary target sequence lengths.

---

## 2. Textbooks & Guides

- **Deep Learning (Chapter 10: Sequence Modeling)** by Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press).
- **Understanding LSTM Networks** by Christopher Olah (Canonical visual blog post).
- **PyTorch Recurrent Layers (`torch.nn.RNN`, `torch.nn.LSTM`, `torch.nn.GRU`)**: [pytorch.org/docs/stable/nn.html#recurrent-layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
