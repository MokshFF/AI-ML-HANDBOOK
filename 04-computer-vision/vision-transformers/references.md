# Vision Transformers & Image Embeddings - References & Further Reading

Seminal research papers, benchmark evaluations, and foundational publications on Transformer architectures applied to vision and OCR.

---

## 1. Seminal Research Papers

- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)** (2020 / 2021)
  - *Authors*: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
  - *Paper*: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
  - *Contribution*: Demonstrated that applying standard Transformer encoder directly to image patches attains state-of-the-art on ImageNet when trained on large datasets.

- **Training data-efficient image transformers & distillation through attention (DeiT)** (2020 / 2021)
  - *Authors*: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou
  - *Paper*: [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
  - *Contribution*: Introduced distillation token and training recipes allowing ViTs to train competitively on ImageNet-1k without massive proprietary pre-training corpora.

- **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** (2021)
  - *Authors*: Alec Radford et al. (OpenAI)
  - *Paper*: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
  - *Contribution*: Scaled contrastive vision-language representation learning using dual encoders, creating generalizable zero-shot image embeddings.

- **An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (CRNN)** (2015 / 2017)
  - *Authors*: Baoguang Shi, Xiang Bai, Cong Yao
  - *Paper*: [arXiv:1507.05717](https://arxiv.org/abs/1507.05717)
  - *Contribution*: Established the canonical CRNN architecture with CTC alignment for Optical Character Recognition.

- **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models** (2021)
  - *Authors*: Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei
  - *Paper*: [arXiv:2109.10282](https://arxiv.org/abs/2109.10282)
  - *Contribution*: Unified text recognition with pure encoder-decoder Transformer architectures.

---

## 2. Textbooks & Survey Articles

- **Transformers in Vision: A Survey** (2021)
  - *Authors*: Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, Mubarak Shah
  - *Paper*: [arXiv:2101.01169](https://arxiv.org/abs/2101.01169)
- **Hugging Face ViT Model Documentation**
  - *URL*: [https://huggingface.co/docs/transformers/model_doc/vit](https://huggingface.co/docs/transformers/model_doc/vit)
