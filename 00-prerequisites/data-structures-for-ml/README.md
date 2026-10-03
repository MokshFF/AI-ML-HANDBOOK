# Data Structures for Machine Learning: Algorithms, Complexity & Systems Design

An engineering-focused guide to classical data structures, computational complexity, memory hierarchy, cache locality, and practical applications across machine learning pipelines and deep learning runtimes.

---

## 1. Complexity Analysis & The Memory Hierarchy

### 1.1 Asymptotic Notations
- **Big-$\mathcal{O}$ ($f(n) = \mathcal{O}(g(n))$)**: Asymptotic upper bound ($|f(n)| \le c \cdot |g(n)|$ for $n \ge n_0$). Characterizes worst-case resource consumption.
- **Big-$\Omega$ ($f(n) = \Omega(g(n))$)**: Asymptotic lower bound.
- **Big-$\Theta$ ($f(n) = \Theta(g(n))$)**: Tight bound ($\mathcal{O}$ and $\Omega$ match).
- **Amortized Analysis**: Average time per operation over a worst-case sequence of operations (e.g., dynamic array resizing).

### 1.2 The Hardware Reality: Cache Hierarchy vs. Big-O
In modern hardware architectures, memory latency dominates CPU throughput:
- **L1 Cache**: ~1 ns (4 clock cycles)
- **L2 Cache**: ~4 ns (14 clock cycles)
- **L3 Cache**: ~10 ns (40-60 clock cycles)
- **Main DRAM**: ~60-100 ns (200-300 clock cycles)

An algorithm with theoretically $\mathcal{O}(N)$ pointer traversals (linked lists) is frequently $10\times - 50\times$ slower in practice than an algorithm operating on a contiguous $\mathcal{O}(N)$ array because contiguous memory leverages CPU cache line prefetching (64 bytes per transfer) and SIMD vectorization.

---

## 2. Core Data Structures in Machine Learning

### 2.1 Arrays (Contiguous Memory & Tensors)
- **Formal Structure**: A contiguous sequence of homogeneous elements indexed by an integer offset:
  $$\text{Address}(A[i]) = \text{Base} + i \cdot \text{sizeof}(\text{type})$$
- **Time Complexity**: Access $\mathcal{O}(1)$, Search $\mathcal{O}(N)$, Appends (amortized) $\mathcal{O}(1)$.
- **ML Applications**:
  - Tensors in PyTorch/NumPy (NDArrays).
  - Dense weights, activation vectors, and gradient accumulators.
  - Column-oriented feature stores.

### 2.2 Hash Maps (Dictionary / Hash Tables)
- **Formal Structure**: Maps keys to values via a deterministic hash function $h(k) \pmod M$. Collisions are handled via separate chaining (linked buckets) or open addressing (linear/quadratic probing).
- **Time Complexity**: Average Lookups/Inserts $\mathcal{O}(1)$, Worst-case (all collisions) $\mathcal{O}(N)$.
- **Rehashing**: Triggered when load factor $\alpha = \frac{N}{M}$ exceeds threshold (typically $0.75$).
- **ML Applications**:
  - **Categorical Embedding Lookup**: Mapping sparse string or integer token IDs to dense latent vectors.
  - **Feature Hashing (Hashing Trick)**: Compressing massive vocabulary spaces (e.g., in click-through rate models) into a bounded hash table ($h(x) \pmod K$) without storing dictionary keys.

### 2.3 Stacks & Queues
- **Stack (LIFO - Last In, First Out)**:
  - $\mathcal{O}(1)$ push and pop.
  - **ML Application**: Backtracking in depth-first tree search (decision tree pruning, Monte Carlo Tree Search).
- **Queue (FIFO - First In, First Out)**:
  - $\mathcal{O}(1)$ enqueue and dequeue.
  - **Circular Ring Buffer**: Fixed-size queue backed by an array with head and tail modular pointers.
  - **ML Application**: Experience Replay Buffers in Reinforcement Learning (DQN), real-time sliding window telemetry in streaming anomaly detection.

### 2.4 Trees & Spatial Indices
- **Binary Decision Trees**: Hierarchical axis-aligned partitioning of feature space based on impurity metrics (Gini, Entropy, MSE).
- **KD-Tree (k-Dimensional Tree)**:
  - Spatial binary tree that recursively splits points across cycling feature dimensions at the median.
  - **Complexity**: Construction $\mathcal{O}(N \log N)$, Nearest Neighbor query $\mathcal{O}(\log N)$ in low dimensions ($D < 20$).
  - **Curse of Dimensionality**: When $D > 20$, KD-Trees degrade to $\mathcal{O}(N)$ brute-force search.
- **Trie (Prefix Tree)**:
  - Tree structure storing token sequences along root-to-leaf paths.
  - **ML Application**: Subword tokenization (Byte-Pair Encoding, WordPiece) and autocomplete language modeling.

### 2.5 Heaps / Priority Queues
- **Formal Structure**: Complete binary tree satisfying the heap property: for min-heap, $\text{parent} \le \text{children}$.
- **Time Complexity**: Insert $\mathcal{O}(\log K)$, Extract Min/Max $\mathcal{O}(\log K)$, Peek $\mathcal{O}(1)$.
- **ML Applications**:
  - **Top-$K$ Candidate Selection**: In search engines and recommendation two-tower retrieval, finding the top $K$ items among $N$ candidates takes $\mathcal{O}(N \log K)$ using a min-heap of size $K$, vastly superior to $\mathcal{O}(N \log N)$ full sorting.
  - **Beam Search Decoding**: In autoregressive LLM text generation, maintaining the top $B$ most likely sequence candidates at each generation step.

### 2.6 Graphs & Directed Acyclic Graphs (DAGs)
- **Representations**: Adjacency Matrix ($\mathcal{O}(V^2)$ space) vs. Adjacency List ($\mathcal{O}(V + E)$ space).
- **Topological Sorting (Kahn's Algorithm)**: Computes a linear ordering of vertices such that for every directed edge $(u, v)$, $u$ precedes $v$.
- **ML Applications**:
  - **Neural Network Computational Graphs**: Autograd engines (PyTorch, TensorFlow, ONNX) model operator dependencies as DAGs to schedule execution order and memory deallocation.
  - **Data Lineage & Feature Pipelines**: Airflow, Prefect, and Kubeflow pipelines coordinate distributed ETL jobs as DAGs.
  - **Graph Neural Networks (GNNs)**: Molecular property prediction, social network analysis, and knowledge graphs.

---

## 3. Comparative Complexity Matrix

| Data Structure | Access (Avg / Worst) | Search (Avg / Worst) | Insert (Avg / Worst) | Delete (Avg / Worst) | Primary ML Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Array** | $\mathcal{O}(1) / \mathcal{O}(1)$ | $\mathcal{O}(N) / \mathcal{O}(N)$ | $\mathcal{O}(N) / \mathcal{O}(N)$ | $\mathcal{O}(N) / \mathcal{O}(N)$ | Dense Tensors, Feature Vectors |
| **Hash Map** | N/A | $\mathcal{O}(1) / \mathcal{O}(N)$ | $\mathcal{O}(1) / \mathcal{O}(N)$ | $\mathcal{O}(1) / \mathcal{O}(N)$ | Token Vocabularies, Feature Stores |
| **Ring Buffer** | $\mathcal{O}(1) / \mathcal{O}(1)$ | $\mathcal{O}(N) / \mathcal{O}(N)$ | $\mathcal{O}(1) / \mathcal{O}(1)$ | $\mathcal{O}(1) / \mathcal{O}(1)$ | RL Replay Buffers, Sliding Windows |
| **Binary Heap**| N/A | $\mathcal{O}(N) / \mathcal{O}(N)$ | $\mathcal{O}(\log K) / \mathcal{O}(\log K)$ | $\mathcal{O}(\log K) / \mathcal{O}(\log K)$ | Top-K Retrieval, Beam Search |
| **KD-Tree** | N/A | $\mathcal{O}(\log N) / \mathcal{O}(N)$ | $\mathcal{O}(\log N) / \mathcal{O}(N)$ | $\mathcal{O}(\log N) / \mathcal{O}(N)$ | Spatial k-NN Search |
| **DAG** | N/A | $\mathcal{O}(V + E) / \mathcal{O}(V + E)$ | $\mathcal{O}(1) / \mathcal{O}(1)$ | $\mathcal{O}(V + E) / \mathcal{O}(V + E)$ | Computational Autograd Graphs |

---

## 4. Common Architectural Mistakes

1. **Hash Table Rehashing Latency Spikes**: In low-latency model inference servers (< 10 ms SLA), a sudden in-memory hash table rehash can block the serving thread for hundreds of milliseconds. Mitigate by pre-allocating hash capacity with `dict.fromkeys()` or custom fixed-size buckets.
2. **Using Trees in High Dimensions**: Applying KD-Trees or Ball-Trees to 768-dimensional transformer embeddings; spatial trees deteriorate to brute-force search when $D > 20$. Modern vector search requires Approximate Nearest Neighbor (ANN) structures like HNSW (Hierarchical Navigable Small World graphs) or IVF-PQ (Inverted File with Product Quantization).
3. **LinkedLists for Numerical Sequences**: Implementing tensor operations over linked structures creates catastrophic memory fragmentation and defeats CPU SIMD vector units.

---

## 5. Topic Module Resources
- Interactive Lab: [`notebook.ipynb`](./notebook.ipynb)
- Source Implementation: [`code/structures.py`](./code/structures.py)
- Pytest Suite: [`code/test_structures.py`](./code/test_structures.py)
- Interview Drill: [`interview.md`](./interview.md)
- Curated Citations: [`references.md`](./references.md)
