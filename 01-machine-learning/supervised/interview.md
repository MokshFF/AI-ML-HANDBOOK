# Supervised Learning: Technical Interview Question Bank

Technical screening questions, mathematical derivations, optimization mechanics, and trade-off analyses across classical supervised algorithms.

---

## 1. Linear & Regularized Models

### Q1: Derive the closed-form Normal Equation for Ridge Regression and explain why adding $\lambda I$ guarantees invertibility.
- **Answer Outline**:
  - Ridge objective: $\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2 = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w}) + \lambda \mathbf{w}^T \mathbf{w}$.
  - Expand: $\mathbf{y}^T \mathbf{y} - 2\mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w} + \lambda \mathbf{w}^T \mathbf{w}$.
  - Compute gradient with respect to $\mathbf{w}$: $\nabla_{\mathbf{w}} \mathcal{L} = -2X^T \mathbf{y} + 2X^T X \mathbf{w} + 2\lambda \mathbf{w}$.
  - Set to zero: $(X^T X + \lambda I)\mathbf{w} = X^T \mathbf{y} \implies \mathbf{w}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$.
  - **Invertibility**: $X^T X$ is positive semi-definite (eigenvalues $\mu_i \ge 0$). Adding $\lambda I$ shifts all eigenvalues by $\lambda > 0$, making all eigenvalues strictly positive ($\mu_i + \lambda \ge \lambda > 0$). Hence, $\det(X^T X + \lambda I) = \prod (\mu_i + \lambda) > 0$, ensuring the matrix is strictly non-singular and invertible.

### Q2: Why does Lasso yield exact zero coefficients (sparsity) while Ridge only shrinks them asymptotically?
- **Answer Outline**:
  - The objective can be written as minimizing $\|\mathbf{y} - X\mathbf{w}\|_2^2$ subject to a constraint $\|\mathbf{w}\|_p \le C$.
  - For Ridge ($p=2$), the constraint region is a smooth hypersphere. Contours of the quadratic loss function touch the smooth boundary at points where weights are non-zero.
  - For Lasso ($p=1$), the constraint region is a diamond/cross-polytope with sharp corners centered exactly on the coordinate axes. Elliptical contours of the loss function have high probability of intersecting these corners where one or more coordinates $w_j = 0$.
  - In subgradient / coordinate descent, the soft-thresholding operator $S(\rho_j, \lambda) = \text{sign}(\rho_j) \max(|\rho_j| - \lambda, 0)$ sets $w_j = 0$ whenever $|\rho_j| \le \lambda$.

---

## 2. Classification & Non-Linear Boundaries

### Q3: Why can't we use Mean Squared Error (MSE) directly as the loss function for Logistic Regression?
- **Answer Outline**:
  - Substituting the non-linear sigmoid $\hat{y} = \sigma(\mathbf{w}^T \mathbf{x})$ into squared error yields $\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \sum (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))^2$.
  - This loss function is **non-convex** with respect to $\mathbf{w}$ because the second derivative contains products involving $\sigma'(z)$, producing flat plateaus and local minima where gradient descent gets stuck.
  - The derivative of MSE with respect to weights includes $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. When a prediction is completely wrong (e.g., $y=1$ but $\sigma(z) \approx 0$), $\sigma'(z) \to 0$, causing vanishing gradients precisely when the error is largest!
  - Binary Cross-Entropy (Log-Loss) cancels the sigmoid saturation term in the gradient, ensuring convex optimization and steep gradients for large errors.

### Q4: Explain the Kernel Trick in Support Vector Machines and state Mercer's Condition.
- **Answer Outline**:
  - The dual formulation of SVM depends only on dot products between training vectors: $W(\boldsymbol{\alpha}) = \sum \alpha_i - \frac{1}{2} \sum \sum \alpha_i \alpha_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j \rangle$.
  - To learn non-linear boundaries, we map inputs into a higher-dimensional space $\phi(\mathbf{x})$. Computing $\langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$ directly can be computationally prohibitive or infinite-dimensional.
  - A **Kernel** $K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$ computes this inner product directly in low-dimensional input coordinates in $\mathcal{O}(D)$ time.
  - **Mercer's Theorem**: A function $K(\mathbf{x}, \mathbf{x}')$ is a valid kernel if and only if its Gram matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is symmetric and positive semi-definite for any finite set of points.

---

## 3. Coding Drill: Coordinate Descent for Lasso

### Task
Implement a standalone function performing coordinate descent updates for Lasso regression.

```python
import numpy as np

def lasso_coordinate_descent(X: np.ndarray, y: np.ndarray, alpha: float, max_iter: int = 1000, tol: float = 1e-5) -> np.ndarray:
    n, d = X.shape
    w = np.zeros(d)
    norm_sq = np.sum(X ** 2, axis=0)

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(d):
            if norm_sq[j] < 1e-12:
                continue
            # Residual without feature j
            r_j = y - (X @ w - X[:, j] * w[j])
            rho_j = float(np.dot(X[:, j], r_j))
            # Soft thresholding
            if rho_j > alpha * n:
                w[j] = (rho_j - alpha * n) / norm_sq[j]
            elif rho_j < -alpha * n:
                w[j] = (rho_j + alpha * n) / norm_sq[j]
            else:
                w[j] = 0.0

        if np.max(np.abs(w - w_old)) < tol:
            break
    return w
```
