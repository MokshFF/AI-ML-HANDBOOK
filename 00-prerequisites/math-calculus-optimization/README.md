# Calculus & Optimization for Machine Learning: Analytical Foundations & Algorithmic Mechanics

A comprehensive guide to multivariable differential calculus, computational graph automatic differentiation, curvature analysis via Hessians, and first/second-order optimization mechanics in machine learning.

---

## 1. Derivatives, Partial Derivatives & Gradients

### 1.1 Intuition
In machine learning, loss functions measure prediction error. The **derivative** tells us how the loss changes when we tweak an individual weight. When models have millions or billions of parameters, the **gradient** vector compiles all these individual rates of change into a single directional compass pointing toward the steepest ascent.

### 1.2 Formal Definitions & Mathematics

#### Scalar Derivative
For $f: \mathbb{R} \to \mathbb{R}$:
$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

#### Partial Derivatives
For a scalar field of multiple variables $f: \mathbb{R}^N \to \mathbb{R}$, the partial derivative with respect to $x_i$ treats all other variables $x_{j \neq i}$ as constant:
$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_N) - f(x_1, \dots, x_i, \dots, x_N)}{h}$$

#### The Gradient Vector ($\nabla f$)
The gradient collects all partial derivatives into an $N$-dimensional vector:
$$\nabla f(\mathbf{x}) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_N} \right]^T \in \mathbb{R}^N$$

**Directional Derivative Theorem**:
The rate of change of $f$ in the direction of unit vector $\mathbf{v}$ ($\|\mathbf{v}\|_2 = 1$) is:
$$D_{\mathbf{v}} f(\mathbf{x}) = \nabla f(\mathbf{x})^T \mathbf{v} = \|\nabla f(\mathbf{x})\|_2 \cos(\theta)$$
- Maximal increase occurs when $\mathbf{v}$ is parallel to $\nabla f$ ($\theta = 0 \implies \cos(\theta) = 1$).
- Maximal decrease occurs in direction $-\nabla f$ ($\theta = \pi \implies \cos(\theta) = -1$).
- Orthogonal directions ($\nabla f^T \mathbf{v} = 0$) experience zero instantaneous change (tangent to level curves).

```python
import numpy as np

# Numerical gradient computation via central differences
def compute_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad
```

---

## 2. The Chain Rule & Automatic Differentiation

### 2.1 Intuition
Deep neural networks are composite functions: $\hat{\mathbf{y}} = f_L(f_{L-1}(\dots f_1(\mathbf{x}; \mathbf{W}_1)\dots; \mathbf{W}_L))$. Computing how the final loss varies with early layers requires propagating derivatives backwards through this composite chain.

### 2.2 Multivariate Chain Rule
If $z = f(y_1, y_2, \dots, y_M)$ and each $y_j = g_j(x_1, \dots, x_N)$, then:
$$\frac{\partial z}{\partial x_i} = \sum_{j=1}^M \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

In vector notation with intermediate vector $\mathbf{y} \in \mathbb{R}^M$:
$$\nabla_{\mathbf{x}} z = J_{\mathbf{y}}(\mathbf{x})^T \nabla_{\mathbf{y}} z$$
where $J_{\mathbf{y}}(\mathbf{x})$ is the Jacobian matrix.

### 2.3 Reverse-Mode vs. Forward-Mode Autodiff
- **Forward-Mode**: Propagates tangents alongside forward execution. Complexity is $\mathcal{O}(N)$ passes for $N$ inputs. Ideal when $N \ll M$.
- **Reverse-Mode (Backpropagation)**: Computes adjoints backwards from scalar loss ($M = 1$) to all $N$ parameters in a **single backward pass** ($\mathcal{O}(1)$ relative to forward cost). This makes training neural networks with billions of parameters computationally tractable.

---

## 3. The Jacobian & The Hessian

### 3.1 The Jacobian Matrix ($J$)
For a vector-valued function $\mathbf{F}: \mathbb{R}^N \to \mathbb{R}^M$ mapping $\mathbf{x} \mapsto [f_1(\mathbf{x}), \dots, f_M(\mathbf{x})]^T$:
$$J \in \mathbb{R}^{M \times N}, \quad J_{ij} = \frac{\partial f_i}{\partial x_j}$$
The linear approximation around $\mathbf{x}_0$ is:
$$\mathbf{F}(\mathbf{x}) \approx \mathbf{F}(\mathbf{x}_0) + J(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)$$

### 3.2 The Hessian Matrix ($H$)
For scalar function $f: \mathbb{R}^N \to \mathbb{R}$, the Hessian contains all second-order partial derivatives:
$$H \in \mathbb{R}^{N \times N}, \quad H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$
By Schwarz's Theorem (Clairaut's Theorem), if second partial derivatives are continuous, $H$ is symmetric ($H = H^T$).

### 3.3 Second-Order Taylor Expansion & Curvature
$$f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T H(\mathbf{x}) \Delta \mathbf{x}$$

At a stationary point where $\nabla f(\mathbf{x}^*) = \mathbf{0}$:
- If $H$ is **Positive Definite** ($\lambda_i > 0, \forall i$): $\mathbf{x}^*$ is a **Local Minimum**.
- If $H$ is **Negative Definite** ($\lambda_i < 0, \forall i$): $\mathbf{x}^*$ is a **Local Maximum**.
- If $H$ is **Indefinite** (eigenvalues of mixed signs): $\mathbf{x}^*$ is a **Saddle Point**.

---

## 4. Convexity: Theory & Implications

### 4.1 Intuition
A convex optimization landscape is bowl-shaped: any local minimum is guaranteed to be a global minimum. Non-convex functions (such as deep neural network loss surfaces) possess complex landscapes with saddle points, local minima, and flat plateaus.

### 4.2 Formal Definition
A set $C \subseteq \mathbb{R}^N$ is convex if $\forall \mathbf{x}, \mathbf{y} \in C$ and $\alpha \in [0, 1]$:
$$\alpha \mathbf{x} + (1 - \alpha) \mathbf{y} \in C$$

A function $f: C \to \mathbb{R}$ is convex if $\forall \mathbf{x}, \mathbf{y} \in C, \alpha \in [0, 1]$:
$$f(\alpha \mathbf{x} + (1 - \alpha) \mathbf{y}) \le \alpha f(\mathbf{x}) + (1 - \alpha) f(\mathbf{y})$$

### 4.3 Second-Order Condition for Convexity
A twice-differentiable function $f$ is convex on a convex domain if and only if its Hessian is **positive semi-definite** everywhere:
$$H(\mathbf{x}) \succeq 0 \iff \mathbf{v}^T H(\mathbf{x}) \mathbf{v} \ge 0, \quad \forall \mathbf{v} \in \mathbb{R}^N$$

---

## 5. Optimization Algorithms: From Gradient Descent to Adam

### 5.1 Vanilla Gradient Descent
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$
Where $\eta > 0$ is the learning rate.
- **Learning Rate Dilemma**:
  - $\eta > \frac{2}{\lambda_{\max}(H)}$ causes explosive divergence.
  - $\eta \ll \frac{1}{\lambda_{\max}(H)}$ results in prohibitively slow convergence.

### 5.2 Momentum
Accumulates an exponentially decaying velocity vector $\mathbf{v}_t$ to dampen oscillations in high-curvature directions while accelerating along consistent ravines:
$$\mathbf{v}_{t} = \beta \mathbf{v}_{t-1} + (1 - \beta) \nabla L(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_t$$

### 5.3 Adam (Adaptive Moment Estimation)
Maintains running estimates of both the first moment (mean gradient) and second uncentered moment (uncentered variance of gradient), with bias correction for initial steps:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{First Moment})$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{Second Moment})$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad (\text{Bias Correction})$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 6. Common Mistakes & Failure Modes

1. **Vanishing / Exploding Gradients**: Repeated multiplication of Jacobians across deep layers ($J_L \dots J_1$). If singular values are $< 1$, gradients vanish exponentially; if $> 1$, gradients explode.
2. **Confusing Saddle Points with Local Minima**: In high-dimensional optimization ($D > 10^6$), critical points with $\nabla L \approx \mathbf{0}$ are almost always saddle points rather than local minima, because the probability that all $10^6$ eigenvalues of $H$ are simultaneously positive is negligible ($2^{-10^6}$).
3. **Evaluating Gradients on Unscaled Features**: Unscaled features create ill-conditioned, elongated elliptical contours ($\kappa(H) \gg 1$), causing gradient descent to oscillate perpendicular to the descent path.

---

## 7. Topic Module Resources
- Interactive Lab: [`notebook.ipynb`](./notebook.ipynb)
- Source Implementation: [`code/calculus_engine.py`](./code/calculus_engine.py)
- Pytest Suite: [`code/test_calculus.py`](./code/test_calculus.py)
- Interview Drill: [`interview.md`](./interview.md)
- Curated Citations: [`references.md`](./references.md)
