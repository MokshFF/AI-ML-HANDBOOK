# Unsupervised Learning: Technical Interview Question Bank

Technical screening questions, mathematical derivations, clustering metrics, and manifold learning trade-offs.

---

## 1. Clustering Algorithms & Geometry

### Q1: How do you determine the optimal number of clusters $K$ in K-Means, and what are the limitations of the Elbow Method?
- **Answer Outline**:
  1. **Elbow Method**: Plots Inertia (WCSS) vs. $K$. Looks for the point of diminishing marginal returns (inflection "elbow").
     - *Limitation*: Highly subjective; real-world continuous data often exhibits smooth exponential decay without a distinct elbow.
  2. **Silhouette Coefficient**: Measures how well an object is separated from neighboring clusters:
     $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$
     where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance. Higher average silhouette indicates better defined clusters.
  3. **Gap Statistic**: Compares $\log(W_k)$ against expected log within-cluster dispersion under a null reference uniform distribution.
  4. **Downstream Task Performance**: If clustering feeds a downstream recommendation or classification pipeline, select $K$ maximizing the downstream target metric.

### Q2: Why does K-Means fail on elongated or concentric clusters (such as concentric circles), while DBSCAN succeeds?
- **Answer Outline**:
  - **K-Means Assumption**: K-Means defines clusters by Euclidean distance to a single central point $\boldsymbol{\mu}_k$, inducing convex polyhedral Voronoi cells. It intrinsically assumes clusters are spherical, isotropic, and roughly equal in variance and volume.
  - **Concentric Circles**: A concentric ring has its geometric centroid located in empty space near the center of the inner ring. K-Means slices across both rings because it cannot represent non-convex topologies.
  - **DBSCAN Success**: DBSCAN defines clusters as continuous connected components of high density ($|N_\epsilon(p)| \ge \text{MinPts}$). It traverses chains of density-reachable core points regardless of spatial curvature.

---

## 2. Dimensionality Reduction & Manifold Learning

### Q3: What is the "Crowding Problem" in dimensionality reduction, and how does t-SNE resolve it?
- **Answer Outline**:
  - In high-dimensional spaces $\mathbb{R}^D$, volume grows exponentially with dimension: the volume of a sphere of radius $r$ is proportional to $r^D$. Consequently, there is vastly more volume in the boundary "shell" of high-dimensional space than near the center, allowing many distinct clusters to be equidistant from one another.
  - When mapping into a 2D plane $\mathbb{R}^2$, the available area at distance $r$ scales only as $2\pi r$. If we used a standard Gaussian distribution in low dimensions, moderately distant points in high dimensions would experience strong attractive forces, crowding onto the center of the map.
  - **t-SNE Solution**: t-SNE replaces the Gaussian distribution in the low-dimensional embedding with a heavy-tailed **Student-t distribution with 1 degree of freedom (Cauchy distribution)**:
    $$q_{ij} \propto \frac{1}{1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2}$$
  - The heavy tails mean that low-dimensional points must be placed much further apart to produce the same small probability $q_{ij}$, naturally pushing distinct clusters away from each other and eliminating crowding.

### Q4: Why can't t-SNE be used directly as a feature preprocessing step for downstream inference on test data?
- **Answer Outline**:
  - t-SNE is a **transductive non-parametric** algorithm. It does not learn an explicit mathematical transformation function $f(\mathbf{x}): \mathbb{R}^D \to \mathbb{R}^2$.
  - It optimizes the low-dimensional coordinates $\mathbf{y}_i$ directly via gradient descent on the entire training batch.
  - When an unseen test point $\mathbf{x}_{\text{test}}$ arrives, there is no matrix or formula to project it without re-running the entire gradient descent optimization over the combined dataset.
  - **Alternative**: PCA, Autoencoders, or UMAP (which learns a reusable parametric simplicial set representation).

---

## 3. Anomaly Detection & Density Estimation

### Q5: How does an Isolation Forest detect anomalies, and why is it faster than distance-based outlier detectors (like LOF or k-NN)?
- **Answer Outline**:
  - **Mechanism**: Outliers are structurally isolated far from dense clusters. When an Isolation Tree randomly picks a feature and a random split value, an outlier is partitioned off into a leaf node near the root of the tree with very few recursive cuts. Normal points require many splits to isolate.
  - **Time Complexity**:
    - Distance-based detectors (k-NN distance, Local Outlier Factor) require pairwise distance matrices, taking $\mathcal{O}(N^2 D)$ time.
    - Isolation Forest constructs $T$ trees from small subsamples of size $\psi$ (typically $\psi = 256$), capping tree depth at $\approx \log_2(\psi) = 8$. Training complexity is strictly $\mathcal{O}(T \cdot \psi \log \psi)$—linear in dataset size $N$ during scoring, making it capable of processing millions of records in seconds.

---

## 4. Coding Drill: Vectorized K-Means Assignment & Update

### Task
Implement the core expectation and maximization steps of K-Means in vectorized NumPy.

```python
import numpy as np

def kmeans_step(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Pairwise Euclidean distances: (N, K)
    dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    k = len(centroids)
    new_centroids = np.zeros_like(centroids)
    for c_idx in range(k):
        members = X[labels == c_idx]
        new_centroids[c_idx] = np.mean(members, axis=0) if len(members) > 0 else centroids[c_idx]

    return labels, new_centroids
```
