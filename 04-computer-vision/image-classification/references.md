# Image Classification & Transfer Learning - References & Further Reading

Seminal research papers, benchmark datasets, and documentation for deep convolutional classifiers and transfer learning.

---

## 1. Seminal Research Papers

- **Deep Residual Learning for Image Recognition (ResNet)** (2015 / 2016)
  - *Authors*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  - *Paper*: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
  - *Contribution*: Introduced identity shortcut connections that enable training networks of 100+ layers without vanishing gradients.

- **Network In Network (Global Average Pooling)** (2013 / 2014)
  - *Authors*: Min Lin, Qiang Chen, Shuicheng Yan
  - *Paper*: [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
  - *Contribution*: Introduced $1 \times 1$ convolutions and Global Average Pooling to replace dense FC layers.

- **How transferable are features in deep neural networks?** (2014)
  - *Authors*: Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson
  - *Paper*: [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
  - *Contribution*: Quantified how feature generalizability transitions to specificity across layers in deep CNNs.

- **Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution** (2022)
  - *Authors*: Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang
  - *Paper*: [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
  - *Contribution*: Analyzed trade-offs between linear probing and fine-tuning, demonstrating that LP-FT (linear probing followed by fine-tuning) yields superior OOD robustness.

---

## 2. Textbooks & Survey Articles

- **Deep Learning**
  - *Authors*: Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press)
  - *Chapter*: Chapter 9: Convolutional Networks.
- **PyTorch Vision Model Zoo Documentation**
  - *URL*: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
