# Automatic Speech Recognition (ASR) - References & Further Reading

Seminal research papers, benchmark corpora, and milestone architectures in automatic speech recognition.

---

## 1. Seminal Research Papers

- **Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks (CTC)** (2006)
  - *Authors*: Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber
  - *Conference*: ICML 2006
  - *Contribution*: Introduced CTC loss and forward-backward dynamic programming for unaligned sequence transcription.

- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** (2020)
  - *Authors*: Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli
  - *Paper*: [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
  - *Contribution*: Pioneered self-supervised masked contrastive pre-training from raw audio waveforms.

- **Conformer: Convolution-augmented Transformer for Speech Recognition** (2020)
  - *Authors*: Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang
  - *Paper*: [arXiv:2005.08100](https://arxiv.org/abs/2005.08100)
  - *Contribution*: Combined self-attention with depthwise convolution, establishing the leading acoustic modeling backbone.

- **Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)** (2022 / 2023)
  - *Authors*: Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI)
  - *Paper*: [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
  - *Contribution*: Demonstrated zero-shot robust multilingual ASR and translation via 680,000 hours of weakly supervised audio-to-text pre-training.

---

## 2. Textbooks & Corpora

- **Speech and Language Processing (3rd ed. draft)**
  - *Authors*: Dan Jurafsky, James H. Martin
  - *Chapter*: Chapter 16: Automatic Speech Recognition and Text-to-Speech.
- **LibriSpeech: An ASR Corpus Based on Public Domain Audio Books**
  - *Authors*: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur (ICASSP 2015).
