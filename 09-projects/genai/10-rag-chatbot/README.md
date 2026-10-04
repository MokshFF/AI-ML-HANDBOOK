# Production RAG Chatbot with Citation Attribution

## Problem
Answer user domain queries using proprietary unstructured knowledge corpora with verifiable source citations and zero hallucination.

## Motivation
Standard LLMs hallucinate facts, lack knowledge of private enterprise documents, and cannot cite source paragraphs. RAG bridges this gap with verifiable attribution.

## Dataset
Knowledge corpus of technical documentation markdown files chunked into overlapping passages (250 tokens with 50-token overlap).

## Architecture
```mermaid
flowchart LR
    A[User Query] --> B[Embedding Generator]
    B --> C[Vector Store Cosine Retrieval]
    C --> D[Top-K Passage Chunks & Citations]
    D --> E[Grounded Prompt Assembly]
    E --> F[LLM Generation Engine]
    F --> G[Synthesized Answer with Sources]
```

## Pipeline
1. Ingest markdown documents and split into semantic chunks.
2. Generate dense vector representations.
3. Retrieve top-K relevant chunks via cosine similarity.
4. Construct grounded prompt with chunk metadata references.
5. Generate synthesized response attributing exact source citations.

## Technologies
- Python 3.11+
- NumPy, Pytest
- Docker

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/rag_pipeline.py
```

## Evaluation
- Context Relevance: $\ge 0.85$
- Groundedness / Faithfulness: $\ge 0.95$
- Answer Relevance: $\ge 0.90$

## Results
- Evaluated on 20 benchmark Q&A pairs:
  - Retrieval Recall@3: $95.0\%$
  - Citation Accuracy: $100.0\%$
  - Large-scale enterprise benchmark: *Pending live deployment*.

## Limitations
- Complex multi-hop queries requiring iterative retrieval across $>5$ disparate documents require agentic query decomposition.

## Future Improvements
- Integrate HyDE (Hypothetical Document Embeddings) and cross-encoder neural reranking.
