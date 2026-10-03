# Graph Neural Networks - Technical Interview Preparation

A curated question bank covering graph message passing mechanics, symmetric normalization derivations, inductive scalability, over-smoothing, and attention coefficients.

---

## 1. Non-Euclidean Geometry & Equivariance

### Q1: Why do standard Convolutional Neural Networks and MLPs fail on graph-structured data?
- **Failure of Standard CNNs**:
  1. *Lack of Shift Invariance & Coordinate Grid*: Images have fixed grid structures with standard up/down/left/right directions. In graphs, there is no spatial grid or canonical ordering of neighbors.
  2. *Variable Degree*: Pixels in a 2D image always have exactly 8 adjacent neighbors. Graph nodes can have 1 neighbor or 100,000 neighbors. A fixed-size convolutional kernel $W \in \mathbb{R}^{K \times K}$ cannot be applied.
- **Failure of Standard MLPs**:
  Concatenating all node features into a giant vector and feeding it to an MLP violates **permutation equivariance**:
  If the arbitrary numbering of nodes is permuted ($1 \to 3, 3 \to 1$), the adjacency matrix undergoes $P A P^T$ and features become $P X$.
  An MLP will produce completely different output embeddings simply because the arbitrary node indexing changed!
- **Permutation Invariance vs. Equivariance**:
  - *Graph-Level Property (Invariance)*: Predicting whole-graph properties (e.g., molecule toxicity):
    $$f(P A P^T, P X) = f(A, X)$$
  - *Node-Level Property (Equivariance)*: Predicting individual node roles:
    $$f(P A P^T, P X) = P f(A, X)$$
  GNN message passing operations ($\sum, \text{mean}, \max$) are structurally guaranteed to be permutation equivariant.

---

### Q2: Why is the adjacency matrix in GCNs normalized symmetrically as $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ rather than by random walk normalization $\tilde{D}^{-1} \tilde{A}$?
- **Random Walk Normalization ($\tilde{D}^{-1} \tilde{A}$)**:
  - Takes the direct average of neighbor features: $( \tilde{D}^{-1} \tilde{A} X )_i = \frac{1}{\tilde{d}_i} \sum_{j \in \tilde{\mathcal{N}}(i)} X_j$.
  - While it prevents feature scale explosion for node $i$, $\tilde{D}^{-1} \tilde{A}$ is **asymmetric** ($(\tilde{D}^{-1} \tilde{A})^T \neq \tilde{D}^{-1} \tilde{A}$).
- **Symmetric Normalization ($\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$)**:
  - Weight of edge $(i, j)$ is $\frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}}$.
  - *Symmetry*: The matrix is symmetric, ensuring all eigenvalues are real and bounded within $[-1, 1]$.
  - *Spectral Connection*: Symmetrically normalized adjacency directly corresponds to the normalized symmetric graph Laplacian $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$, maintaining direct mathematical fidelity to first-order Chebyshev spectral graph convolutions.
  - *Hub Protection*: If high-degree node $j$ connects to low-degree node $i$, node $j$'s feature influence on node $i$ is scaled down by $\sqrt{\tilde{d}_j}$, preventing dominant hub nodes from completely overwhelming low-degree nodes.

---

## 2. Scalability, Attention & Over-Smoothing

### Q3: What is the Over-Smoothing problem in deep GNNs, and why do most GNNs only have 2 to 4 layers?
- **The Phenomenon**:
  Unlike CNNs where networks with 100+ layers (ResNet) excel, stacking $>4$ GCN layers causes dramatic performance degradation. Node representations become virtually identical across the entire graph.
- **Mathematical Cause**:
  Repeated multiplication by $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ is a discrete diffusion process acting as a low-pass filter over graph frequencies.
  By the Perron-Frobenius theorem, as $L \to \infty$, the power iteration converges to the stationary distribution (the dominant eigenvector of the graph Laplacian):
  $$\lim_{L \to \infty} h_i^{(L)} = c \cdot \sqrt{\text{deg}(i)}$$
  All feature information from $X$ is completely lost; every node's embedding becomes proportional only to its degree.
- **Remedies**:
  1. **Shallow Depth**: Most practical GNN architectures use only 2-3 layers.
  2. **Jumping Knowledge (JK-Net)**: Connects all intermediate layer representations directly to the final prediction layer via concatenation or max-pooling.
  3. **DropEdge**: Randomly drops graph edges during training to reduce over-smoothing message propagation.
  4. **APPNP**: Decouples neural feature transformation from propagation via Personalized PageRank.

---

### Q4: How does GraphSAGE enable mini-batch training on massive, billion-node industrial graphs?
- **The Full-Batch Problem in GCN**:
  In a standard GCN, computing the layer 2 representation of node $v$ requires all 1-hop neighbors of $v$. Computing their representations requires all 2-hop neighbors.
  In scale-free real-world networks (social networks, web graphs), the neighborhood grows exponentially ($d^L$), causing "neighbor explosion". To update even 1 node, a 3-layer GCN may require expanding into millions of nodes, forcing full-graph batch training.
- **The GraphSAGE Solution**:
  1. **Fixed-Size Neighborhood Sampling**:
     Instead of aggregating all neighbors $\mathcal{N}(v)$, GraphSAGE samples a fixed number of neighbors uniformly at random at each depth:
     $$S_1 = 25 \quad (\text{1-hop}), \quad S_2 = 10 \quad (\text{2-hop})$$
  2. **Bounded Compute**:
     For a batch of size $B$, the maximum number of nodes involved in the computation graph is strictly bounded:
     $$\text{Nodes}_{\text{batch}} \le B \times (1 + S_1 + S_1 \cdot S_2)$$
     For $B=512$, $S_1=25, S_2=10$, the graph fits comfortably in GPU memory, enabling standard mini-batch SGD on arbitrary graph sizes.

---

## 3. Whiteboard Coding Drills

### Q5: Implement a vectorized Graph Convolutional Layer in PyTorch.
```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.W)

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        # A_tilde = A + I
        N = adj.size(0)
        adj_tilde = adj + torch.eye(N, device=adj.device)
        deg = torch.sum(adj_tilde, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ adj_tilde @ D_inv_sqrt

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # H @ W -> A_norm @ (H @ W) + b
        hw = torch.matmul(x, self.W)
        out = torch.matmul(adj_norm, hw) + self.bias
        return out
```
