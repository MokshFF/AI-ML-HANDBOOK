# Linear Algebra for Machine Learning: Foundations & Computational Mechanics

Linear algebra provides the mathematical language and operational tools that underpin modern machine learning, deep learning architectures, and dimensionality reduction techniques.

---

## 1. Scalars, Vectors, Matrices, and Tensors

### 1.1 Intuition
In computing, numbers rarely exist in isolation. A single metric (such as loss) is a scalar. A collection of features describing an entity (e.g., house price, square footage, bedrooms) forms a vector. A dataset of multiple entities forms a matrix. Multi-channel audio, images, or batches of token sequences form tensors.

### 1.2 Formal Definitions & Mathematical Representation
- **Scalar** ($\alpha \in \mathbb{R}$): A single real value (Rank-0 tensor).
- **Vector** ($\mathbf{x} \in \mathbb{R}^D$): An ordered $D$-tuple representing a point or directional displacement in $D$-dimensional space:
  $$\mathbf{x} = \begin{bmatrix} x_1 & x_2 & \dots & x_D \end{bmatrix}^T$$
- **Matrix** ($A \in \mathbb{R}^{M \times N}$): A 2D grid of numbers with $M$ rows and $N$ columns representing a linear map from $\mathbb{R}^N \to \mathbb{R}^M$.
- **Tensor** ($\mathcal{T} \in \mathbb{R}^{D_1 \times D_2 \times \dots \times D_K}$): A multidimensional array of order $K$ generalizing scalars, vectors, and matrices.

```python
import numpy as np

scalar = np.float32(3.14)                  # Rank 0
vector = np.array([1.0, 2.0, 3.0])         # Rank 1: (3,)
matrix = np.ones((64, 128))                # Rank 2: (64, 128)
tensor = np.zeros((32, 3, 224, 224))       # Rank 4: Batch x Channel x Height x Width
```

### 1.3 Common Mistakes
- **1D Array vs. Column Vector**: In NumPy, `arr.shape == (D,)` is neither a row nor a column vector. It does not behave like an $(M, 1)$ or $(1, N)$ matrix during outer products unless reshaped or expanded with `None` (`arr[:, np.newaxis]`).

---

## 2. Vector Operations, Dot Products & Geometry

### 2.1 Intuition
The dot product measures the degree of directional alignment between two vectors, scaled by their lengths. It is the fundamental building block of dense layers, convolutions, and self-attention mechanisms.

### 2.2 Formal Definition & Mathematics
For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$:
$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^D u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$

From the Cauchy-Schwarz inequality:
$$|\mathbf{u}^T \mathbf{v}| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$$
Equality holds if and only if $\mathbf{u}$ and $\mathbf{v}$ are linearly dependent (collinear).

### 2.3 Cosine Similarity
$$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \cos(\theta) \in [-1, 1]$$

```python
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))
```

---

## 3. Matrix Multiplication & Transpose

### 3.1 Intuition
Matrix multiplication represents the composition of linear transformations. Computing $C = AB$ means applying transformation $B$ first, followed by transformation $A$.

### 3.2 Formal Definition & Complexity
Given $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, the product $C = AB \in \mathbb{R}^{M \times N}$ has elements:
$$C_{ij} = \sum_{k=1}^K A_{ik} B_{kj}$$
Standard matrix multiplication requires $\mathcal{O}(MKN)$ operations.

### 3.3 Transpose Properties
The transpose $A^T$ reflects elements across the main diagonal: $(A^T)_{ij} = A_{ji}$.
- $(AB)^T = B^T A^T$ (reverse order law)
- A matrix is **symmetric** if $A = A^T$, and **skew-symmetric** if $A = -A^T$.

---

## 4. Vector and Matrix Norms

### 4.1 Intuition
Norms measure the "size" or magnitude of vectors and matrices, forming the basis of distance metrics and regularization penalties ($L_1$ Lasso, $L_2$ Ridge / weight decay).

### 4.2 Vector $L_p$ Norms
$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^D |x_i|^p \right)^{1/p}$$
- **$L_1$ Norm (Manhattan)**: $\|\mathbf{x}\|_1 = \sum |x_i|$ (promotes sparsity in optimization).
- **$L_2$ Norm (Euclidean)**: $\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T \mathbf{x}}$ (standard smooth distance).
- **$L_\infty$ Norm (Chebyshev)**: $\|\mathbf{x}\|_\infty = \max_i |x_i|$.

### 4.3 Matrix Frobenius Norm
$$\|A\|_F = \sqrt{\sum_{i=1}^M \sum_{j=1}^N A_{ij}^2} = \sqrt{\text{Tr}(A^T A)}$$

---

## 5. Matrix Inverses, Rank & Determinants

### 5.1 Intuition & Invertibility
A square matrix $A \in \mathbb{R}^{N \times N}$ is invertible if there exists a matrix $A^{-1}$ such that:
$$A A^{-1} = A^{-1} A = I_N$$
Inversion undoes the transformation applied by $A$. If $A$ flattens space into a lower dimension, the transformation loses information and cannot be inverted.

### 5.2 Determinant ($\det(A)$)
- **Geometric Meaning**: The signed factor by which area (2D) or volume ($N$-D) is scaled under the transformation $A$.
- If $\det(A) = 0$, the transformation collapses volume to zero; hence, $A$ is singular (non-invertible).
- Properties: $\det(AB) = \det(A)\det(B)$, $\det(A^{-1}) = \frac{1}{\det(A)}$, $\det(A^T) = \det(A)$.

### 5.3 Matrix Rank
- **Row/Column Rank**: The maximum number of linearly independent rows (or columns) in $A$.
- A matrix $A \in \mathbb{R}^{M \times N}$ has **full rank** if $\text{rank}(A) = \min(M, N)$.
- **Rank-Nullity Theorem**: For linear map $A: \mathbb{R}^N \to \mathbb{R}^M$:
  $$\text{rank}(A) + \dim(\ker(A)) = N$$

---

## 6. Orthogonality, Projections & Gram-Schmidt

### 6.1 Orthogonality
Vectors $\mathbf{u}, \mathbf{v}$ are **orthogonal** if $\mathbf{u}^T \mathbf{v} = 0$. They are **orthonormal** if additionally $\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$.
A square matrix $Q$ is **orthogonal** if:
$$Q^T Q = Q Q^T = I \implies Q^{-1} = Q^T$$
Orthogonal matrices preserve Euclidean distance and angles: $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$.

### 6.2 Vector Projection
The projection of vector $\mathbf{v}$ onto the subspace spanned by non-zero vector $\mathbf{u}$ is:
$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v}^T \mathbf{u}}{\mathbf{u}^T \mathbf{u}} \mathbf{u}$$

---

## 7. Eigenvalues, Eigenvectors & Spectral Decomposition

### 7.1 Intuition
When a linear transformation acts on most vectors, it rotates and scales them. **Eigenvectors** are privileged directions that undergo pure scaling without rotation.

### 7.2 Formal Definition
For $A \in \mathbb{R}^{N \times N}$, non-zero $\mathbf{v} \in \mathbb{C}^N$ is an eigenvector with eigenvalue $\lambda \in \mathbb{C}$ if:
$$A \mathbf{v} = \lambda \mathbf{v} \iff (A - \lambda I) \mathbf{v} = \mathbf{0}$$
Non-trivial solutions exist if and only if the characteristic equation holds:
$$\det(A - \lambda I) = 0$$

### 7.3 Spectral Theorem for Symmetric Matrices
If $A \in \mathbb{R}^{N \times N}$ is real and symmetric ($A = A^T$), then:
1. All eigenvalues $\lambda_i$ are real.
2. Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.
3. $A$ has an orthogonal diagonalization:
   $$A = Q \Lambda Q^T = \sum_{i=1}^N \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

---

## 8. Singular Value Decomposition (SVD)

### 8.1 Intuition
Eigendecomposition applies strictly to square matrices. **SVD** generalizes eigendecomposition to any rectangular matrix $A \in \mathbb{R}^{M \times N}$. It reveals the fundamental geometry of every linear map: a rotation, followed by scaling along orthogonal axes, followed by a second rotation.

### 8.2 Mathematical Formulation
$$A = U \Sigma V^T$$
Where:
- $U \in \mathbb{R}^{M \times M}$ is orthogonal ($U^T U = I_M$); columns $\mathbf{u}_i$ are **left singular vectors** (eigenvectors of $A A^T$).
- $\Sigma \in \mathbb{R}^{M \times N}$ is diagonal with non-negative entries $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ (**singular values**, $\sigma_i = \sqrt{\lambda_i(A^T A)}$).
- $V \in \mathbb{R}^{N \times N}$ is orthogonal ($V^T V = I_N$); columns $\mathbf{v}_i$ are **right singular vectors** (eigenvectors of $A^T A$).

### 8.3 Eckart-Young-Mirsky Low-Rank Approximation
The optimal rank-$k$ approximation ($k < r$) minimizing Frobenius error $\|A - A_k\|_F$ is obtained by truncating SVD to the top $k$ singular values:
$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

---

## 9. The Deep Connection Between SVD and PCA

### 9.1 Intuition
Principal Component Analysis (PCA) seeks orthogonal directions of maximum variance in a dataset.

### 9.2 Derivation from SVD
Let $X \in \mathbb{R}^{N \times D}$ be a dataset of $N$ observations with zero empirical mean ($\sum_{i=1}^N X_{i, :} = \mathbf{0}$).
The sample covariance matrix is:
$$C = \frac{1}{N - 1} X^T X \in \mathbb{R}^{D \times D}$$

If we compute the SVD of the centered data matrix $X$:
$$X = U \Sigma V^T$$
Then the covariance matrix becomes:
$$C = \frac{1}{N - 1} (U \Sigma V^T)^T (U \Sigma V^T) = \frac{1}{N - 1} V \Sigma^T U^T U \Sigma V^T = V \left( \frac{\Sigma^2}{N - 1} \right) V^T$$

**Key Insight**:
1. The right singular vectors $V$ of centered data $X$ are the exact eigenvectors of the sample covariance matrix $C$.
2. The singular values $\sigma_i$ directly yield the explained variances: $\lambda_i = \frac{\sigma_i^2}{N - 1}$.
3. Computing PCA via SVD on $X$ avoids forming $X^T X$ explicitly, preventing quadratic memory costs and numerical precision loss.

---

## 10. Topic Module Resources
- Interactive Lab: [`notebook.ipynb`](./notebook.ipynb)
- Source Implementation: [`code/linear_algebra_core.py`](./code/linear_algebra_core.py)
- Pytest Suite: [`code/test_linear_algebra.py`](./code/test_linear_algebra.py)
- Interview Drill: [`interview.md`](./interview.md)
- Curated Citations: [`references.md`](./references.md)
