# Two-Stage Recommendation System

## Problem
Retrieve and rank the top-K relevant catalog items for a given user from thousands of possibilities.

## Motivation
Modern e-commerce and media streaming services rely on multi-stage recommendation funnels to maximize engagement and conversion while meeting strict sub-50ms latency budgets.

## Dataset
Synthetic user-item interaction matrix:
- 200 users, 100 items, 2,000 interactions with implicit positive/negative feedback.

## Architecture
```mermaid
flowchart LR
    A[User Profile & History] --> B[Candidate Retrieval via Matrix Factorization]
    B --> C[Top-50 Candidates]
    C --> D[Neural Scoring MLP Ranker]
    D --> E[Final Top-K Ranked List]
```

## Pipeline
1. Embed users and items into a 16-dimensional latent representation space.
2. Approximate candidate retrieval using dot-product similarity.
3. Fine-rank candidates with a multi-layer interaction network.
4. Evaluate ranking quality using NDCG@K and HitRate@K.

## Technologies
- Python 3.11+
- NumPy, Pytest

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/recommender.py
```

## Evaluation
- Normalized Discounted Cumulative Gain (NDCG@K): $\text{DCG}@K / \text{IDCG}@K$
- HitRate@K: Fraction of test interactions appearing in top-K recommendations.

## Results
- Validated on 50 test users:
  - HitRate@5: $\approx 0.64$
  - NDCG@5: $\approx 0.58$
  - MovieLens / Amazon benchmark: *Pending execution on distributed cluster*.

## Limitations
- Cold-start users and newly created items require auxiliary content features (metadata) not present in pure matrix factorization.

## Future Improvements
- Integrate two-tower dual-encoder neural architecture.
- Add diversity reranking via Maximal Marginal Relevance (MMR).
