# Graph Neural Networks - Curated References & Bibliography

A curated collection of foundational research papers, landmark monographs, and official libraries for Graph Neural Networks.

---

## 1. Landmark Research Papers

### 1.1 Spectral & Spatial Message Passing Foundations
- **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**
  - *Authors*: Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst (NeurIPS, 2016).
  - *Significance*: Formulated Chebyshev polynomial spectral filters on graph Laplacians.
- **Semi-Supervised Classification with Graph Convolutional Networks (GCN)**
  - *Authors*: Thomas N. Kipf, Max Welling (ICLR, 2017).
  - *Significance*: Derived the first-order localized spectral approximation, establishing modern GCNs.
- **Neural Message Passing for Quantum Chemistry (MPNN Framework)**
  - *Authors*: Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl (ICML, 2017).
  - *Significance*: Unified spatial graph deep learning under the Message-Aggregate-Update abstraction.

### 1.2 Scalability & Attention Mechanisms
- **Inductive Representation Learning on Large Graphs (GraphSAGE)**
  - *Authors*: William L. Hamilton, Rex Ying, Jure Leskovec (NeurIPS, 2017).
  - *Significance*: Introduced neighborhood sampling, enabling inductive mini-batch training on massive web-scale graphs.
- **Graph Attention Networks (GAT)**
  - *Authors*: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (ICLR, 2018).
  - *Significance*: Applied self-attention to graph neighborhoods, enabling anisotropic edge weighting.
- **How Powerful are Graph Neural Networks? (GIN)**
  - *Authors*: Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka (ICLR, 2019).
  - *Significance*: Proved the theoretical equivalence between message passing GNNs and the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test.

### 1.3 Over-Smoothing & Deep Architectures
- **Deeper Insights into Graph Convolutional Networks: An Analysis of Over-Smoothing**
  - *Authors*: Qimai Li, Zhichao Han, Xiao-Ming Wu (AAAI, 2018).
  - *Significance*: Mathematical analysis proving GCN performs Laplacian smoothing leading to embedding collapse.
- **Representation Learning on Graphs with Jumping Knowledge Networks**
  - *Authors*: Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka (ICML, 2018).
  - *Significance*: Flexible multi-layer skip connections mitigating over-smoothing.

---

## 2. Textbooks & Engineering Frameworks

- **Graph Representation Learning** by William L. Hamilton (Synthesis Lectures on Artificial Intelligence and Machine Learning, Morgan & Claypool, 2020).
- **PyTorch Geometric (PyG)**: Canonical high-performance library for graph deep learning. [pyg.org](https://pyg.org/)
- **Deep Graph Library (DGL)**: Framework-agnostic scalable graph neural network library. [dgl.ai](https://www.dgl.ai/)
