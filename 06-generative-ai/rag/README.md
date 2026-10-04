# Retrieval-Augmented Generation (RAG)

> **Last reviewed:** 2026-10. Vector databases, embedding models and framework APIs evolve quickly; the concepts and failure modes below are the stable part. The code is dependency-light and vendor-neutral.

## Learning objectives
Build and evaluate a RAG pipeline: ingestion and parsing, chunking, embeddings, vector search with metadata filters, BM25 and hybrid retrieval, reranking, context construction, grounded generation with citations, evaluation, and advanced architectures.

## Pipeline

```
documents -> parse -> chunk (+metadata) -> embed -> index
query -> (rewrite) -> retrieve [dense + BM25 + filters] -> fuse (RRF) -> rerank -> build context
      -> generate with citations -> validate citations/groundedness -> answer
```

RAG (Lewis et al., 2020) gives a model access to external, updatable knowledge without retraining, and enables attribution. It does **not** eliminate hallucination: retrieval can miss, and the generator can ignore or misread context.

## 1. Document ingestion and parsing
Quality here caps everything downstream. Extract clean text from HTML/PDF/Office/markdown; preserve structure (headings, tables, lists, page numbers) as **metadata**; deduplicate; record source, version, timestamp and access-control labels. Scanned documents need OCR ([`../../04-computer-vision/`](../../04-computer-vision/)). `parse_markdown` keeps the heading path (`Billing > Refunds`) for every chunk.

## 2. Chunking
Chunks are the unit of retrieval. Too large: diluted embeddings, wasted context. Too small: lost context. Strategies: fixed windows with overlap (`chunk_words`), sentence-packing (`chunk_sentences`), structure-aware splitting by headings/sections, and **parent-document retrieval** (index small chunks, return the larger parent: `expand_to_parents`). Tune size/overlap on your own retrieval metrics, not folklore.

## 3. Embeddings and vector databases
A bi-encoder maps text to vectors; relevance ~ cosine similarity (Karpukhin et al., 2020). Choose models by benchmark (e.g. MTEB) *and* your domain data; embed queries and documents with a compatible model (some need instruction prefixes); store the model name/version with the index and re-embed when it changes. Production vector stores add approximate nearest-neighbour indexes (e.g. HNSW, IVF), filtering, replication and persistence; `VectorStore` here is exact brute-force search to show the semantics. The shipped `HashingEmbedder` is **lexical**, not semantic, purely so the demo is offline.

## 4. Retrieval: filters, hybrid search, reranking
- **Metadata filtering** (tenant, date, language, document type, ACL) must be applied **at retrieval time** so unauthorised content never reaches the prompt.
- **Hybrid search** combines dense (semantic) and sparse (BM25: exact terms, IDs, rare names) rankings. **Reciprocal Rank Fusion** $\text{RRF}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$ needs no score calibration.
- **Reranking**: retrieve a wide candidate pool cheaply, then rescore with a more expensive cross-encoder/late-interaction model (e.g. ColBERT-style) or an LLM. **MMR** reduces redundancy: $\arg\max_i \lambda\,\text{sim}(q,d_i)-(1-\lambda)\max_{j\in S}\text{sim}(d_i,d_j)$.

## 5. Context construction, generation, citations
Select chunks within a token budget; number them; label sources; put the most relevant evidence where the model uses it best (long contexts can under-use the middle, Liu et al., 2023). Instruct the model to answer **only** from the context, to cite `[n]`, and to abstain when unsupported. Treat retrieved text as **untrusted data** (indirect prompt injection; see [`../safety-alignment/`](../safety-alignment/)). Validate citations in code (`invalid_citations`) and check that cited chunks actually support the claims.

## 6. Evaluating RAG
Evaluate retrieval and generation separately, then end to end:
- **Retrieval**: Recall@k, MRR, nDCG against labelled relevant chunks.
- **Generation**: groundedness/faithfulness (are claims supported by context?), answer relevance, citation precision/recall, abstention quality.
- Frameworks such as RAGAS popularise reference-free metrics; they rely on LLM judges, so calibrate against human labels ([`../evaluation/`](../evaluation/)).

## 7. Advanced architectures (concepts)
Query rewriting / multi-query with fusion (`multi_query_search`), HyDE (Gao et al., 2022), parent-child and hierarchical indexes, graph-structured retrieval, agentic/iterative retrieval where the model decides when and what to search (Self-RAG, Asai et al., 2023), multimodal RAG over page images ([`../multimodal/`](../multimodal/)), and caching. Each adds latency, cost and failure modes: add complexity only when evaluation shows the simple pipeline is the bottleneck.

## Common mistakes
- Evaluating only the final answer, never retrieval.
- Chunking by arbitrary size, ignoring document structure.
- Applying access filters after generation, or not at all.
- Changing the embedding model without re-indexing.
- Assuming citations imply correctness.
- Using top-k = 20 with no reranking and flooding the context.

## Code and notebook
- [`code/rag_pipeline.py`](code/rag_pipeline.py): ingestion, chunking, embedder, `VectorStore`, `BM25`, RRF, MMR, rerank, parent expansion, context/citations, groundedness proxy, retrieval metrics.
- [`code/test_rag_pipeline.py`](code/test_rag_pipeline.py): 12 tests.
- [`notebook.ipynb`](notebook.ipynb): end-to-end offline demo with retrieval evaluation.
