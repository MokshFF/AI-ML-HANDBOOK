# Data Structures for Machine Learning: Interview Question Bank

Systems algorithms, streaming analytics, spatial indexing, and computational graph architecture interview questions.

---

## 1. System Performance & Memory Layout

### Q1: Why do machine learning systems utilize contiguous multidimensional arrays rather than linked lists or pointer arrays?
- **Answer Outline**:
  1. **Spatial Cache Locality**: Reading an element in a contiguous array loads the surrounding 64-byte CPU cache line into L1 cache, making sequential access virtually instantaneous. Linked lists scatter nodes randomly across heap memory, incurring cache misses on almost every dereference.
  2. **SIMD Vectorization**: CPUs and GPUs feature vector registers capable of performing identical arithmetic across 8 to 64 contiguous numbers simultaneously (AVX, NEON, CUDA warps). Scattered pointer structures cannot be vectorized.
  3. **Zero Pointer Overhead**: A 64-bit float in a contiguous buffer requires exactly 8 bytes. A linked node requires the data (8 bytes) plus forward/backward pointers (16 bytes) plus allocator headers (16 bytes), bloating memory footprint by 400-500%.

### Q2: What is the "Hashing Trick" (Feature Hashing) in large-scale machine learning, and what are its trade-offs?
- **Answer Outline**:
  - In massive-scale linear models (e.g., ad click prediction with billions of rare categorical features), maintaining an explicit string-to-index dictionary requires tens of gigabytes of RAM.
  - The **Hashing Trick** applies a fast hash function $h(\text{feature}) \pmod M$ to map arbitrary strings directly into an array of fixed size $M$ without storing keys.
  - **Trade-offs**:
    - **Pros**: Constant memory footprint, stateless streaming tokenization, parallelizable without locking.
    - **Cons**: Irreversible (cannot recover feature name), hash collisions introduce minor feature degradation (though mitigated by signed hash trick $\xi(x) \in \{-1, +1\}$).

---

## 2. Spatial Indexing & Nearest Neighbor Search

### Q3: Why do KD-Trees perform poorly for high-dimensional vector search (e.g., 768-dim embeddings in LLM RAG)?
- **Answer Outline**:
  - The **Curse of Dimensionality**: To prune a branch, the query hyper-sphere must not intersect the partitioning hyperplane. In high dimensions, hyper-spheres have virtually all their volume near the surface, causing the query radius to intersect almost all partitioning hyperplanes across the tree.
  - As dimension $D > 20$, the number of leaves that must be inspected approaches $2^D$, degrading KD-Tree lookup complexity from $\mathcal{O}(\log N)$ to $\mathcal{O}(N)$ brute-force search.
  - **Modern Alternative**: Approximate Nearest Neighbor (ANN) graphs such as HNSW (Hierarchical Navigable Small World) or Inverted File with Product Quantization (IVF-PQ).

---

## 3. Computational Graphs & Scheduling

### Q4: How does an autograd engine determine the execution order for backpropagation in a neural network?
- **Answer Outline**:
  - Forward operations record their tensor dependencies to construct a Directed Acyclic Graph (DAG) where nodes represent operations/tensors and directed edges represent data flow.
  - For backpropagation, the engine performs a **Topological Sort** (using DFS or Kahn's Algorithm) starting from the scalar loss node.
  - Inverting the topological order ensures that all gradients flowing into a node from downstream consumers are fully accumulated before that node computes its own upstream gradient.

---

## 4. Coding Drill: Top-$K$ Frequent Embeddings via Min-Heap

### Task
Implement an algorithm to extract the top-$K$ highest-scoring items from an un-indexed stream of $N$ items with memory complexity bounded by $\mathcal{O}(K)$ and time complexity $\mathcal{O}(N \log K)$.

```python
import heapq
from typing import Iterator

def top_k_streaming(stream: Iterator[tuple[float, str]], k: int) -> list[tuple[float, str]]:
    # Min-heap maintains smallest of the top-k at root
    min_heap: list[tuple[float, str]] = []
    for score, item_id in stream:
        if len(min_heap) < k:
            heapq.heappush(min_heap, (score, item_id))
        elif score > min_heap[0][0]:
            heapq.heapreplace(min_heap, (score, item_id))
    return sorted(min_heap, key=lambda x: x[0], reverse=True)
```
