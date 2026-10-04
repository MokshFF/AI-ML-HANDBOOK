# Seminal Research Papers: Natural Language Processing

Milestones in self-supervised representations, masked language modeling, few-shot prompting, and scaling laws.

---

## 1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Title**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Year**: 2018
- **Link**: https://arxiv.org/abs/1810.04805
- **Problem**: Unidirectional language models (e.g. OpenAI GPT) only look at left-to-right context, limiting performance on token-level and sequence classification tasks.
- **Main Idea**: Train a deep bidirectional Transformer encoder using a Masked Language Model (MLM) objective and Next Sentence Prediction (NSP).
- **Key Contribution**: Proved that task-agnostic bidirectional pre-training followed by simple task-specific fine-tuning sets state-of-the-art results across 11 NLP tasks.
- **Important Architecture/Math**:
  $$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{Masked}} \log P(x_i \mid \tilde{X}; \theta)$$
- **Why It Matters**: Revolutionized applied NLP from 2018-2022; established pre-training + fine-tuning as the dominant paradigm.
- **Prerequisites**: Transformer encoder, WordPiece tokenization, cross-entropy loss.
- **Suggested Follow-up Papers**: *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (Liu et al., 2019); *DeBERTa* (He et al., 2020).

---

## 2. Training Compute-Optimal Large Language Models (Chinchilla)
- **Title**: Training Compute-Optimal Large Language Models
- **Authors**: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al.
- **Year**: 2022
- **Link**: https://arxiv.org/abs/2203.15556
- **Problem**: Existing foundation models (e.g. GPT-3 175B, Gopher 280B) were undertrained relative to their parameter size due to flawed power-law scaling assumptions.
- **Main Idea**: Rigorously analyze loss across $>400$ training runs; parameters and tokens should scale equally: for every doubling of model parameters, training tokens must also double.
- **Key Contribution**: Established the Chinchilla scaling law: $D \approx 20 \times P$ (20 tokens per parameter). Demonstrated that a 70B model trained on 1.4T tokens outperformed a 280B model.
- **Important Architecture/Math**:
  $$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}, \quad \alpha \approx 0.34, \; \beta \approx 0.28$$
- **Why It Matters**: Redefined open-source and proprietary LLM training budgets, directly inspiring the LLaMA family and modern compact frontier models.
- **Prerequisites**: Kaplan scaling laws, FLOP computation budgets, power-law regressions.
- **Suggested Follow-up Papers**: *Scaling Laws for Neural Language Models* (Kaplan et al., 2020); *LLaMA: Open and Efficient Foundation Language Models* (Touvron et al., 2023).
