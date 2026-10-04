# RAG - Interview Questions

### 1. RAG vs. fine-tuning: how do you choose?
RAG for fresh, large, changing, attributable knowledge and access control; fine-tuning for style, format, skills or latency/cost reduction. They combine: fine-tune for behaviour, retrieve for facts. See [`../fine-tuning/`](../fine-tuning/).

### 2. How do you choose chunk size and overlap?
Treat them as hyperparameters: sweep on labelled queries measuring Recall@k and answer quality; respect document structure; consider parent-document retrieval. Overlap reduces boundary loss at index-size cost.

### 3. Why hybrid search? Explain Reciprocal Rank Fusion.
Dense retrieval captures paraphrase; BM25 nails exact tokens (IDs, names, codes). RRF sums $1/(k+\text{rank})$ across rankings, avoiding score-scale calibration and being robust to one failing retriever.

### 4. Bi-encoder vs. cross-encoder?
Bi-encoders embed independently (fast, indexable, less precise); cross-encoders score query-document pairs jointly (precise, too slow for the whole corpus). Use bi-encoder recall then cross-encoder rerank.

### 5. How do you enforce permissions in RAG?
Store ACL metadata; filter during retrieval (pre-filter in the vector query); never rely on the prompt to hide documents; log what was retrieved; test with cross-tenant queries.

### 6. How would you evaluate a RAG system?
Separate retrieval metrics (Recall@k, MRR, nDCG) from generation metrics (faithfulness, relevance, citation accuracy, abstention); build a golden set from real queries; use LLM judges only after validating against human labels; monitor in production.

### 7. The answer is wrong though the right document exists. Debug it.
Check ingestion/parsing, chunk contained the evidence, retrieval rank, filters, reranking, context truncation/ordering, prompt instructions, then generation faithfulness. Log every stage's output.

### 8. What is indirect prompt injection in RAG?
Retrieved content containing instructions that hijack the model. Mitigate with data/instruction separation, least-privilege tools, output validation, and source trust tiers. No single defence is sufficient.

### 9. How do you handle questions the corpus can't answer?
Instruct abstention, set a retrieval-score threshold, evaluate abstention explicitly, and return "no relevant sources" rather than forcing an answer.

### Coding drill
Implement `reciprocal_rank_fusion` and prove with a test that a document ranked 1st in two lists beats one ranked 1st in only one.
