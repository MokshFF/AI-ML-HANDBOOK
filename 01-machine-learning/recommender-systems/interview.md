# Recommender Systems: Technical Interview Question Bank

Technical screening questions, ranking metric derivations, two-tower system designs, and cold-start architectures.

---

## 1. Collaborative Filtering & Matrix Factorization

### Q1: Why can't standard Truncated SVD from linear algebra ($R = U \Sigma V^T$) be applied directly to user-item rating matrices?
- **Answer Outline**:
  - Standard SVD is mathematically defined only on **fully complete, dense matrices**.
  - In a recommender system, the rating matrix $R$ is $> 99\%$ unobserved (missing entries).
  - Imputing missing entries with zero introduces massive distortion: an unrated item is treated as an actively hated item (rating 0), biasing latent singular vectors completely toward zero.
  - **FunkSVD Solution**: Formulates matrix factorization as an optimization problem defined **strictly over the set of observed entries**:
    $$\min_{P, Q} \sum_{(u, i) \in \text{Observed}} (r_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda (\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)$$
    optimizing latent embeddings via SGD without touching or zero-imputing unobserved values.

### Q2: How does Implicit Feedback differ from Explicit Feedback, and how does the ALS (Alternating Least Squares) algorithm handle it?
- **Answer Outline**:
  - **Difference**: Explicit feedback (1-5 star ratings) contains clear positive and negative labels. Implicit feedback (clicks, watch time, purchases) only indicates positive interactions; unobserved entries are a mixture of negative interest and lack of exposure.
  - **Hu, Koren, Volinsky Formulation (Implicit ALS)**:
    - Binarizes interaction: $p_{ui} = 1$ if $r_{ui} > 0$, else $0$.
    - Defines confidence: $c_{ui} = 1 + \alpha r_{ui}$ (higher confidence for multiple views).
    - Objective: $\min_{P, Q} \sum_{u, i} c_{ui} (p_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda (\|P\|_F^2 + \|Q\|_F^2)$.
    - Evaluates all $M \times N$ pairs, but because $c_{ui} = 1$ for unobserved pairs, the loss can be solved in closed form alternately across $P$ and $Q$ in $\mathcal{O}(M k^3 + N k^3)$ time, enabling massive parallel distributed training.

---

## 2. Ranking Metrics: NDCG & MAP

### Q3: Derive the Discounted Cumulative Gain (DCG) metric and explain why logarithmic discounting is applied.
- **Answer Outline**:
  - **Cumulative Gain**: $\text{CG} = \sum_{i=1}^K \text{rel}_i$. It ignores the position of relevant items (recommending a relevant item at position 10 produces the same CG as position 1).
  - **Logarithmic Discounting**: Human attention drops rapidly when scrolling through a feed. Dividing relevance by $\log_2(i + 1)$ imposes a smooth concave discount factor:
    $$\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
    - Position 1 discount: $\log_2(2) = 1.0$ (no discount).
    - Position 2 discount: $\log_2(3) \approx 1.58$.
    - Position 4 discount: $\log_2(5) \approx 2.32$.
  - Normalizing by Ideal DCG ($\text{IDCG}$, computed by sorting items in perfect descending relevance) ensures $\text{NDCG} \in [0, 1]$, making ranking performance comparable across users with different numbers of relevant items.

---

## 3. Production Architecture: Two-Tower Systems

### Q4: Describe the industrial Two-Tower (Dual-Encoder) architecture used in candidate generation at scale.
- **Answer Outline**:
  - **Problem**: Modern catalogs have $10^8$ items. Evaluating a heavy deep neural network on $10^8$ items per request within a 20 ms SLA is physically impossible.
  - **Two-Tower Architecture**:
    1. **User Tower**: Deep neural network encoding user history, device, demographics into vector $\mathbf{u} \in \mathbb{R}^d$.
    2. **Item Tower**: Deep neural network encoding item features, video tags, text descriptions into vector $\mathbf{v} \in \mathbb{R}^d$.
    3. **Similarity**: Cosine similarity / dot product $\langle \mathbf{u}, \mathbf{v} \rangle$.
  - **Offline Precomputation & Serving**:
    - Item embeddings $\mathbf{v}_i$ are pre-computed offline and indexed in an Approximate Nearest Neighbor vector database (HNSW / ScaNN / Milvus).
    - At runtime, only the User Tower executes dynamically to produce $\mathbf{u}$; a sub-millisecond MIPS (Maximum Inner Product Search) query retrieves the top 1,000 candidates for downstream ranking.

---

## 4. Coding Drill: Vectorized NDCG@K Calculation

### Task
Implement a function computing NDCG@K for a list of binary relevance values.

```python
import numpy as np

def compute_ndcg_at_k(relevance: list[int], k: int) -> float:
    rel = np.array(relevance[:k], dtype=float)
    if len(rel) == 0:
        return 0.0

    # DCG
    discounts = np.log2(np.arange(2, len(rel) + 2))
    dcg = np.sum((2.0 ** rel - 1.0) / discounts)

    # Ideal DCG
    ideal_rel = np.sort(rel)[::-1]
    idcg = np.sum((2.0 ** ideal_rel - 1.0) / discounts)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)
```
