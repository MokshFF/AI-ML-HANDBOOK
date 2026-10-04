# Speech Synthesis (TTS) - References & Further Reading

Seminal research papers, milestone vocoders, and benchmark publications in neural speech synthesis.

---

## 1. Seminal Research Papers

- **FastSpeech: Fast, Robust and Controllable Text to Speech** (2019)
  - *Authors*: Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu
  - *Paper*: [arXiv:1905.09263](https://arxiv.org/abs/1905.09263)
  - *Contribution*: Introduced non-autoregressive feedforward Transformer TTS with length regulation and duration prediction.

- **FastSpeech 2: Fast and High-Quality End-to-End Text to Speech** (2020 / 2022)
  - *Authors*: Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu
  - *Paper*: [arXiv:2006.04558](https://arxiv.org/abs/2006.04558)
  - *Contribution*: Directly incorporated pitch and energy predictors, removing the teacher-student distillation requirement.

- **Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Tacotron 2)** (2017 / 2018)
  - *Authors*: Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, R. J. Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu
  - *Paper*: [arXiv:1712.05884](https://arxiv.org/abs/1712.05884)
  - *Contribution*: Established the canonical two-stage neural TTS paradigm (Seq2Seq acoustic model + WaveNet vocoder).

- **HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis** (2020)
  - *Authors*: Jungil Kong, Jaehyeon Kim, Jaekyoung Bae
  - *Paper*: [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)
  - *Contribution*: Introduced multi-period and multi-scale discriminators for real-time, high-fidelity neural vocoding.

- **Signal Estimation from Modified Short-Time Fourier Transform (Griffin-Lim Algorithm)** (1984)
  - *Authors*: Daniel Griffin, Jae Lim
  - *Journal*: IEEE Transactions on Acoustics, Speech, and Signal Processing, 32(2), 236-243.
  - *Contribution*: The mathematical foundation of iterative phase retrieval from magnitude spectrograms.

---

## 2. Textbooks & Authoritative Surveys

- **Neural Approaches to Conversational AI**
  - *Authors*: Jianfeng Gao, Michel Galley, Lihong Li
  - *Publisher*: Foundations and Trends in Information Retrieval
- **Coqui TTS Open-Source Library**
  - *URL*: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
