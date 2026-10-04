# Seminal Research Papers: Retrieval-Augmented Generation (RAG)

Architectures for grounding LLMs on external parametric and non-parametric memory stores.

---

## 1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Title**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Authors**: Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, et al.
- **Year**: 2020
- **Link**: https://arxiv.org/abs/2005.11401
- **Problem**: Pre-trained language models store factual knowledge implicitly in weights, making it difficult to update knowledge, cite sources, or eliminate hallucinations.
- **Main Idea**: Combine pre-trained parametric sequence-to-sequence memory (BART) with non-parametric corpus retrieval (DPR via Wikipedia index).
- **Key Contribution**: Formalized end-to-end differentiable RAG architectures for sequence-level and token-level marginalization.
- **Important Architecture/Math**:
  $$P_{\text{RAG-Sequence}}(y \mid x) = \sum_{z \in \text{top-K}} P_\eta(z \mid x) \prod_{i=1}^N P_\theta(y_i \mid x, z, y_{1:i-1})$$
- **Why It Matters**: Created the RAG industry paradigm that grounds enterprise AI workflows today.
- **Prerequisites**: Dense passage retrieval (DPR), sequence-to-sequence models, maximum inner product search.
- **Suggested Follow-up Papers**: *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)* (Gao et al., 2022); *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (Es et al., 2023).
