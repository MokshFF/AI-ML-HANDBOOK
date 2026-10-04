# Cheat Sheet: Optimization & Regularization

| Technique | Mathematical Update | Key Benefit | Hyperparameter Guidelines |
| :--- | :--- | :--- | :--- |
| **SGD + Momentum** | $v_t = \beta v_{t-1} + g_t; \; w_t = w_{t-1} - \eta v_t$ | Dampens oscillations in high curvature valleys | $\beta = 0.9$ |
| **RMSprop** | $s_t = \gamma s_{t-1} + (1-\gamma) g_t^2; \; w_t = w_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t$ | Per-parameter adaptive learning rates | $\gamma = 0.99, \epsilon = 10^{-8}$ |
| **Adam** | Combines 1st ($m_t$) and 2nd ($v_t$) moment estimates | Fast initial convergence on sparse gradients | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ |
| **AdamW** | Decouples weight decay from adaptive gradient scale | Restores true weight decay; better generalization | $\text{weight\_decay} \in [0.01, 0.1]$ |
| **Cosine Annealing** | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$ | Smooth decay preventing sharp saddle trapping | Warmup for first $5\%$ of steps |
| **Gradient Clipping** | $g \leftarrow g \cdot \min(1, \frac{c}{\|g\|})$ | Prevents gradient explosion in RNNs/LLMs | Threshold $c \in [0.5, 1.0]$ |
| **Mixed Precision (AMP)** | FP16/BF16 matrix math with FP32 master weights | $2-3\times$ compute speedup; $50\%$ memory savings | Use BF16 on modern GPUs (no loss scaling required) |
