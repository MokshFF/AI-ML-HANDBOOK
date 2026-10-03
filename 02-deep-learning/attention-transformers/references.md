# Attention & Transformers - Curated References & Bibliography

A curated collection of foundational research papers, landmark monographs, and official documentation on Attention and Transformer architectures.

---

## 1. Landmark Research Papers

### 1.1 The Attention Revolution
- **Neural Machine Translation by Jointly Learning to Align and Translate (Additive Attention)**
  - *Authors*: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (ICLR, 2015).
  - *Significance*: Introduced the first soft attention mechanism to overcome the fixed-length vector bottleneck in Seq2Seq models.
- **Attention Is All You Need**
  - *Authors*: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (NeurIPS, 2017).
  - *Significance*: The milestone breakthrough eliminating recurrence and convolutions entirely, establishing the Transformer architecture.

### 1.2 Paradigm Specializations (BERT, GPT, T5)
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
  - *Authors*: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (NAACL, 2019).
  - *Significance*: Popularized masked language modeling (MLM) and bidirectional representation learning.
- **Improving Language Understanding by Generative Pre-Training (GPT-1)**
  - *Authors*: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI, 2018).
  - *Significance*: Demonstrated that generative autoregressive pretraining followed by discriminative fine-tuning achieves superior general NLP transfer.
- **Language Models are Few-Shot Learners (GPT-3)**
  - *Authors*: Tom B. Brown, Benjamin Mann, Nick Ryder, et al. (NeurIPS, 2020).
  - *Significance*: Demonstrated emergent in-context learning capabilities in 175B parameter decoder-only models.
- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)**
  - *Authors*: Colin Raffel, Noam Shazeer, Adam Roberts, et al. (JMLR, 2020).
  - *Significance*: Unified all NLP tasks under a sequence-to-sequence text-to-text formulation.

### 1.3 Structural & Positional Innovations
- **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)**
  - *Authors*: Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu (arXiv, 2021).
  - *Significance*: Formulated Rotary Position Embedding (RoPE), standard in modern frontier LLMs.
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
  - *Authors*: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (NeurIPS, 2022).
  - *Significance*: Re-engineered self-attention with GPU SRAM tiling and online softmax, reducing memory from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ without approximation.
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)**
  - *Authors*: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. (ICLR, 2021).
  - *Significance*: Successfully applied pure Transformers to visual recognition, inspiring Vision Transformers.

---

## 2. Textbooks & Engineering Documentation

- **The Illustrated Transformer** by Jay Alammar (Canonical visual guide).
- **Hugging Face Transformers Documentation**: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **PyTorch Transformer Modules (`torch.nn.Transformer`, `torch.nn.MultiheadAttention`)**: [pytorch.org/docs/stable/nn.html#transformer-layers](https://pytorch.org/docs/stable/nn.html#transformer-layers)
