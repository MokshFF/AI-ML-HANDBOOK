# Machine Learning & AI Interview Preparation (`10-interview-prep`)

## Overview
Comprehensive, production-grade interview preparation guide covering classical machine learning, deep learning, NLP, foundation models, system design, practical coding challenges, and behavioral leadership frameworks.

---

## 1. Technical Screening & Theory Compendiums
Every question in these compendiums is divided into seniority tiers (**Beginner**, **Intermediate**, **Advanced**, **Expert**), tagged by domain nature (`Conceptual`, `Mathematical`, `Coding`, `Debugging`, `System Design`, `Practical`, `Gotcha`), and structured with:
- **Short answer**
- **Detailed explanation**
- **Concrete example**
- **Common misconception**
- **Follow-up probing questions**

| Resource | Scope & Key Topics |
| :--- | :--- |
| [**Machine Learning Questions**](./ml-questions.md) | Bias-variance tradeoff, L1/L2 regularization, GBDTs vs Random Forest, ROC-AUC vs PR-AUC, target leakage, SVM duality, and probability calibration. |
| [**Deep Learning Questions**](./deep-learning-questions.md) | Vanishing/exploding gradients, Kaiming/Xavier initialization, BatchNorm vs LayerNorm vs RMSNorm, Adam vs AdamW, Multi-Head Attention, and FlashAttention. |
| [**Natural Language Processing Questions**](./nlp-questions.md) | Byte-Pair Encoding (BPE), subword tokenization, Word2Vec, BERT vs GPT, decoding strategies (Greedy, Beam Search, Temperature, Top-P, Min-P), and NLP metrics. |
| [**Generative AI & LLM Questions**](./genai-llm-questions.md) | KV cache memory calculations, PagedAttention, LoRA & QLoRA mathematics, RLHF vs DPO preference alignment, and agent state machines. |
| [**System Design Questions**](./system-design-questions.md) | End-to-end 7-step system design interview blueprint, recommendation feeds at 100k QPS, real-time fraud detection at 50k TPS, and multi-tier caching. |
| [**Behavioral & Leadership Guide**](./behavioral.md) | STAR framework responses for ML engineering trade-offs (accuracy vs latency), resolving silent production data drift incidents, and stakeholder management. |

---

## 2. Interactive Topics & Hands-on Modules

| Directory | Topic | Scope |
| :--- | :--- | :--- |
| [`coding-questions/`](./coding-questions/) | **Practical Coding Solutions** | Vectorized NumPy operations, Pandas as-of joins, custom Scikit-Learn transformers, PyTorch attention from scratch, K-Means, and rate limiters. |
| [`cheat-sheets/`](./cheat-sheets/) | **10 High-Yield Cheat Sheets** | Dense, quick-reference tables of formulas, hyperparameter behaviors, architectures, optimization rules, RAG, LLMOps, and system design tradeoffs. |

---

## 3. High-Yield Cheat Sheets Quick Links
Inside [`cheat-sheets/`](./cheat-sheets/):
1. [`01-ml-algorithms.md`](./cheat-sheets/01-ml-algorithms.md) - Classical ML algorithms, objectives, assumptions, and failure modes
2. [`02-ml-metrics.md`](./cheat-sheets/02-ml-metrics.md) - Classification, regression, and ranking metrics
3. [`03-dl-architectures.md`](./cheat-sheets/03-dl-architectures.md) - ResNet, ConvNeXt, ViT, LSTM, Transformer, U-Net, Diffusion
4. [`04-optimization.md`](./cheat-sheets/04-optimization.md) - SGD, AdamW, Cosine Annealing, Gradient Clipping, Mixed Precision
5. [`05-nlp.md`](./cheat-sheets/05-nlp.md) - Tokenization, TF-IDF, Word2Vec, BERT, BLEU, ROUGE, Perplexity
6. [`06-transformers.md`](./cheat-sheets/06-transformers.md) - MHA, GQA, RoPE, RMSNorm, SwiGLU, FlashAttention
7. [`07-rag.md`](./cheat-sheets/07-rag.md) - Chunking, HNSW, Hybrid search, RRF, Cross-encoder rerankers, RAGAS
8. [`08-llms.md`](./cheat-sheets/08-llms.md) - Scaling laws, LoRA, QLoRA, DPO, vLLM, Speculative decoding
9. [`09-mlops.md`](./cheat-sheets/09-mlops.md) - DVC, Feast, FastAPI, Triton, Prometheus, PSI drift monitoring
10. [`10-system-design.md`](./cheat-sheets/10-system-design.md) - Scale estimation rules of thumb, two-stage cascades, caching tiers

---

## 4. Standard Directory Schema
Every topic directory in this module follows our standard five-component structure:
- `README.md` - Module introduction, learning objectives, and concept matrix
- `notebook.ipynb` - Reproducible, runnable interactive notebook
- `code/` - Clean, modular Python scripts and helper utilities
- `interview.md` - Technical screening questions, edge cases, and design discussions
- `references.md` - Research papers, textbooks, and documentation

## Prerequisites
Before beginning this module, review:
- Foundational math and coding prerequisites in [`../00-prerequisites/`](../00-prerequisites/)
- The end-to-end learning pathways defined in [`../ROADMAP.md`](../ROADMAP.md)
