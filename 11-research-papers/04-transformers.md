# Seminal Research Papers: Transformer Mechanics

Core architectural breakthroughs in self-attention, relative position embeddings, and hardware-accelerated kernels.

---

## 1. Attention Is All You Need
- **Title**: Attention Is All You Need
- **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- **Year**: 2017
- **Link**: https://arxiv.org/abs/1706.03762
- **Problem**: Recurrent and convolutional sequence models suffer from sequential execution bottlenecks and inability to model long-range token dependencies directly.
- **Main Idea**: Discard recurrence and convolutions entirely; base sequence transduction solely on multi-head scaled dot-product attention.
- **Key Contribution**: Proposed the encoder-decoder Transformer architecture with Multi-Head Attention and Sinusoidal Positional Encodings.
- **Important Architecture/Math**:
  $$\operatorname{Attention}(Q, K, V) = \operatorname{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V, \quad \operatorname{MHA}(Q, K, V) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
- **Why It Matters**: The foundational architecture for almost all modern language models (GPT, BERT, LLaMA), vision foundation models, and speech systems.
- **Prerequisites**: Matrix multiplication, softmax, sequence modeling concepts.
- **Suggested Follow-up Papers**: *FlashAttention* (Dao et al., 2022); *RoFormer* (Su et al., 2021).

---

## 2. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Title**: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- **Year**: 2022
- **Link**: https://arxiv.org/abs/2205.14135
- **Problem**: Self-attention memory and compute scale quadratically $\mathcal{O}(N^2)$ with sequence length, bottlenecked by high-bandwidth memory (HBM) read/writes.
- **Main Idea**: Tiled execution of exact attention entirely within fast GPU on-chip SRAM using online softmax scaling, never materializing the $N \times N$ matrix in slow HBM.
- **Key Contribution**: Reduced attention memory footprint from $\mathcal{O}(N^2)$ to linear $\mathcal{O}(N)$ while delivering $2-4\times$ wall-clock training acceleration.
- **Important Architecture/Math**:
  $$\text{Online Softmax Accumulation: } m_{\text{new}} = \max(m_{\text{prev}}, x), \quad d_{\text{new}} = d_{\text{prev}} e^{m_{\text{prev}} - m_{\text{new}}} + e^{x - m_{\text{new}}}$$
- **Why It Matters**: Enabled training and inference of modern foundation models with $32\text{k}$ to $1\text{M}+$ token context windows.
- **Prerequisites**: GPU memory hierarchy (HBM vs SRAM), kernel fusion, standard self-attention.
- **Suggested Follow-up Papers**: *FlashAttention-2: Faster Attention with Better Work Partitioning* (Dao, 2023).
