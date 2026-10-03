# Linear Algebra for Machine Learning: Interview Question Bank

Technical screening questions, mathematical derivations, and engineering trade-offs covering matrix factorizations, rank deficiency, eigenvalues, and SVD.

---

## 1. Fundamentals & Invertibility

### Q1: Under what conditions is a square matrix $A \in \mathbb{R}^{N \times N}$ invertible?
- **Answer Outline**:
  The following statements are mathematically equivalent (The Invertible Matrix Theorem):
  1. $A$ has an inverse $A^{-1}$ such that $A A^{-1} = I$.
  2. $\det(A) \neq 0$.
  3. $\text{rank}(A) = N$ (Full row and column rank).
  4. The null space is trivial: $\ker(A) = \{\mathbf{0}\}$ (only the zero vector maps to zero).
  5. The columns of $A$ form a linearly independent basis for $\mathbb{R}^N$.
  6. None of the eigenvalues of $A$ are zero ($\lambda_i \neq 0, \forall i$).
  7. The condition number $\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} < \infty$.

### Q2: What is the condition number of a matrix, and why is it critical in numerical optimization?
- **Answer Outline**:
  - The condition number is defined as $\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$.
  - It measures the sensitivity of the solution of linear system $A\mathbf{x} = \mathbf{b}$ to perturbations in $\mathbf{b}$ or $A$.
  - In gradient descent, an ill-conditioned Hessian ($\kappa(H) \gg 1$) creates steep elongated ravines, causing gradient oscillations and slow convergence.

---

## 2. Spectral Theory & SVD

### Q3: Why is PCA implemented via SVD rather than eigendecomposition of the covariance matrix?
- **Answer Outline**:
  1. **Numerical Stability**: Forming $C = \frac{1}{N-1}X^T X$ squares the condition number: $\kappa(X^T X) = (\kappa(X))^2$. SVD operates directly on $X$, preserving floating-point precision.
  2. **Memory Efficiency**: When the number of features $D$ is massive (e.g., $D = 100,000$ in genomics or NLP), storing $C \in \mathbb{R}^{D \times D}$ requires gigabytes, whereas thin SVD truncates computation to $k$ components directly.

### Q4: Explain the geometric interpretation of Singular Value Decomposition ($A = U \Sigma V^T$).
- **Answer Outline**:
  - Any real linear transformation acts on the unit sphere by:
    1. Rotating the space via orthonormal basis $V^T$.
    2. Stretching along coordinate axes by singular values $\sigma_1, \dots, \sigma_r$ into a hyper-ellipse.
    3. Rotating the resulting hyper-ellipse into the output space via orthonormal basis $U$.

---

## 3. Regularization & Norms

### Q5: Geometrically, why does $L_1$ regularization produce sparse weights while $L_2$ does not?
- **Answer Outline**:
  - The optimization problem can be viewed as minimizing loss subject to a constraint $\|\mathbf{w}\|_p \le C$.
  - The $L_2$ ball is a smooth hypersphere; level contours of the loss function touch the smooth boundary at points where all components are non-zero.
  - The $L_1$ ball is a cross-polytope (rhombus in 2D) with sharp corners aligned along coordinate axes. Loss contours are statistically much more likely to intersect the constraint set at these axis vertices, forcing non-essential weights strictly to zero.

---

## 4. Coding Drill: Power Iteration Algorithm

### Task
Implement the power iteration method to find the largest eigenvalue and its associated eigenvector of a real symmetric positive semi-definite matrix.

```python
import numpy as np

def power_iteration(A: np.ndarray, num_iters: int = 100, tol: float = 1e-7) -> tuple[float, np.ndarray]:
    n = A.shape[0]
    # Initialize random unit vector
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    eigenval = 0.0
    for _ in range(num_iters):
        v_next = np.dot(A, v)
        norm = np.linalg.norm(v_next)
        if norm < 1e-12:
            return 0.0, v
        v_next /= norm

        # Rayleigh quotient
        new_eigenval = float(np.dot(v_next, np.dot(A, v_next)))
        if abs(new_eigenval - eigenval) < tol:
            break
        v = v_next
        eigenval = new_eigenval

    return eigenval, v
```
