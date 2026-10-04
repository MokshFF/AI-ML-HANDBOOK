# Cheat Sheet: Retrieval-Augmented Generation (RAG)

| Stage | Key Approaches | Best Practices | Critical Metrics |
| :--- | :--- | :--- | :--- |
| **Chunking** | Fixed-size (256-512 tokens), semantic, parent-document | $10-20\%$ token overlap to preserve boundaries | Chunk coherence |
| **Embedding Model** | Dense dual-encoders (e.g. BGE, E5, OpenAI text-embed-3) | Normalize vectors to unit length; match domain | Cosine similarity distribution |
| **Vector Index** | HNSW (Hierarchical Navigable Small World), IVF-PQ | HNSW for latency ($< 5\text{ ms}$); IVF-PQ for memory scale | Recall@K, QPS throughput |
| **Hybrid Search** | BM25 lexical + Dense vector ANN | Reciprocal Rank Fusion (RRF): $\sum \frac{1}{60 + r_i}$ | Mitigates keyword blindspots |
| **Reranking** | Cross-Encoder (e.g. BGE-Reranker, Cohere Rerank) | Rerank top 50-100 candidates down to top 5-10 | MRR@10, NDCG@10 |
| **Evaluation (RAGAS)**| Context Relevance, Faithfulness, Answer Relevance | LLM-as-a-judge on golden evaluation test set | Faithfulness $\ge 0.95$ |
