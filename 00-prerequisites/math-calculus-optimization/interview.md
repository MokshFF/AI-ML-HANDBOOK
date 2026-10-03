# Calculus & Optimization for Machine Learning: Interview Question Bank

Technical screening questions, mathematical derivations, and engineering trade-offs covering automatic differentiation, gradient descent dynamics, Hessians, and non-convex landscapes.

---

## 1. Automatic Differentiation & The Chain Rule

### Q1: Why is reverse-mode automatic differentiation (backpropagation) favored over forward-mode in deep learning?
- **Answer Outline**:
  - In deep learning, the loss function maps from $N$ parameters to $1$ scalar value: $L: \mathbb{R}^N \to \mathbb{R}$, where $N \sim 10^7 - 10^{11}$.
  - **Forward-mode autodiff** computes directional derivatives $\frac{\partial \mathbf{y}}{\partial x_i}$ by propagating vector-Jacobian products from input to output. Evaluating the gradient with respect to all $N$ parameters requires $N$ separate forward passes: $\mathcal{O}(N)$ compute.
  - **Reverse-mode autodiff** seeds the scalar output with $\frac{\partial L}{\partial L} = 1.0$ and pulls gradients backward through vector-Jacobian products in a single pass. Computing the full gradient vector $\nabla_{\mathbf{w}} L$ takes only $\sim 2-3\times$ the computational cost of a single forward pass: $\mathcal{O}(1)$ relative to $N$.

### Q2: What causes vanishing and exploding gradients in deep neural networks, and what are the standard architectural solutions?
- **Answer Outline**:
  - By the chain rule: $\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial L}{\partial \mathbf{h}_L} \prod_{l=1}^{L-1} \frac{\partial \mathbf{h}_{l+1}}{\partial \mathbf{h}_l} = \frac{\partial L}{\partial \mathbf{h}_L} \prod_{l=1}^{L-1} (\mathbf{W}_{l+1}^T \text{diag}(\sigma'(\mathbf{z}_l)))$.
  - If the spectral norm of transition matrices is $< 1$ (or saturating activations like sigmoid where $\sigma' \le 0.25$), the product decays exponentially toward zero: **vanishing gradient**.
  - If the spectral norm is $> 1$, the product grows exponentially toward infinity: **exploding gradient**.
  - **Remedies**:
    1. Residual skip connections ($\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l) \implies \frac{\partial \mathbf{h}_{l+1}}{\partial \mathbf{h}_l} = I + \frac{\partial F}{\partial \mathbf{h}_l}$ ensuring an identity gradient highway).
    2. Non-saturating activations (ReLU, GELU).
    3. Proper weight initialization (He / Xavier).
    4. Normalization layers (LayerNorm, BatchNorm).
    5. Gradient clipping (for exploding gradients in RNNs/LLMs).

---

## 2. Hessians, Curvature, and Second-Order Methods

### Q3: Why don't we train large deep neural networks using Newton's method ($\mathbf{w}_{t+1} = \mathbf{w}_t - H^{-1} \nabla L$)?
- **Answer Outline**:
  1. **Memory Complexity**: For $N = 10^9$ parameters, storing the Hessian matrix $H \in \mathbb{R}^{N \times N}$ requires $10^{18}$ floats $\approx 4 \times 10^9$ GB of RAM, which is completely intractable.
  2. **Computational Complexity**: Inverting an $N \times N$ matrix requires $\mathcal{O}(N^3)$ operations.
  3. **Attraction to Saddle Points**: Newton's method jumps toward critical points where $\nabla L = \mathbf{0}$ regardless of whether they are minima or maxima. If $H$ has negative eigenvalues (common in deep learning), Newton steps directly toward saddle points.

### Q4: In high-dimensional optimization, why are saddle points vastly more prevalent than local minima?
- **Answer Outline**:
  - For a critical point to be a local minimum, all $N$ eigenvalues of the Hessian must be strictly positive ($\lambda_i > 0$).
  - Assuming random sign distribution of eigenvalues around stationary points, the probability of encountering a true local minimum scales as $2^{-N}$.
  - For $N = 10^6$, this probability is effectively zero. Almost all critical points where $\nabla L \approx \mathbf{0}$ are saddle points with both positive and negative curvature directions.

---

## 3. First-Order Optimizers: Momentum & Adam

### Q5: How does the Adam optimizer mitigate pathological curvature, and what does the bias correction step do?
- **Answer Outline**:
  - **Second-moment scaling**: Dividing by $\sqrt{v_t} + \epsilon$ rescales the gradient by coordinate-wise historical variance. Along high-curvature directions where gradients fluctuate violently, $v_t$ is large, dampening step sizes; along flat plateaus, $v_t$ is small, amplifying steps.
  - **Bias Correction**: Initializing $m_0 = \mathbf{0}, v_0 = \mathbf{0}$ biases the exponential moving averages toward zero, especially when $\beta_2 = 0.999$. Dividing by $1 - \beta^t$ scales up early estimates so that $\mathbb{E}[\hat{m}_t] \approx \mathbb{E}[g_t]$ from step $t = 1$.

---

## 4. Coding Drill: Vectorized Gradient Descent with Momentum

### Task
Implement a Python function performing mini-batch gradient descent with Polyak momentum.

```python
import numpy as np
from typing import Callable

def momentum_gradient_descent(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    init_params: np.ndarray,
    lr: float = 1e-3,
    beta: float = 0.9,
    steps: int = 100
) -> np.ndarray:
    w = init_params.copy()
    v = np.zeros_like(w)
    for _ in range(steps):
        g = grad_fn(w)
        v = beta * v + (1.0 - beta) * g
        w -= lr * v
    return w
```
