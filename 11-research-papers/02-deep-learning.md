# Seminal Research Papers: Deep Learning Foundations

Seminal architectures, optimization breakthroughs, and normalization mechanisms that enabled training ultra-deep neural networks.

---

## 1. Deep Residual Learning for Image Recognition
- **Title**: Deep Residual Learning for Image Recognition
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year**: 2015
- **Link**: https://arxiv.org/abs/1512.03385
- **Problem**: Degradation problem: As network depth increases, accuracy saturates and degrades rapidly due to optimization vanishing gradient bottlenecks.
- **Main Idea**: Reformulate layers as learning residual functions with reference to layer inputs, rather than unreferenced functions.
- **Key Contribution**: Introduced identity shortcut connections that pass inputs directly across convolutional blocks without extra parameters.
- **Important Architecture/Math**:
  $$y = \mathcal{F}(x, \{W_i\}) + x, \quad \frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + I \right)$$
- **Why It Matters**: Enabled training networks with over 150 to 1,000 layers, winning ImageNet 2015 and establishing the universal backbone for modern deep learning.
- **Prerequisites**: Convolutional neural networks, backpropagation, matrix calculus.
- **Suggested Follow-up Papers**: *Identity Mappings in Deep Residual Networks* (He et al., 2016); *Densely Connected Convolutional Networks* (Huang et al., 2017).

---

## 2. Decoupled Weight Decay Regularization (AdamW)
- **Title**: Decoupled Weight Decay Regularization
- **Authors**: Ilya Loshchilov, Frank Hutter
- **Year**: 2017
- **Link**: https://arxiv.org/abs/1711.05101
- **Problem**: L2 regularization in adaptive gradient methods (Adam) does not function as true weight decay, leading to suboptimal generalization.
- **Main Idea**: Decouple weight decay from gradient updates by subtracting the decay term directly from the parameter update step.
- **Key Contribution**: Restored the proportional decay rate across all parameters regardless of adaptive historical gradient variance.
- **Important Architecture/Math**:
  $$\theta_{t+1} = \theta_t - \eta_t \lambda \theta_t - \frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
- **Why It Matters**: Standard optimizer for virtually all modern Transformer, LLM, and diffusion model pretraining pipelines.
- **Prerequisites**: Stochastic gradient descent, Adam optimizer mechanics, L2 regularization.
- **Suggested Follow-up Papers**: *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2014); *Symbolic Discovery of Optimization Algorithms* (Chen et al., Lion, 2023).
