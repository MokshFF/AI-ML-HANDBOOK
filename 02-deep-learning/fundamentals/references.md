# Deep Learning Fundamentals - Curated References & Bibliography

A curated collection of foundational textbooks, landmark research papers, and technical documentation on computational graphs, activation dynamics, and first-order optimization.

---

## 1. Authoritative Textbooks & Monographs

- **Deep Learning**
  - *Authors*: Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016).
  - *Significance*: The foundational reference on deep feedforward networks, numerical computation, backpropagation, and optimization algorithms.
  - *URL*: [deeplearningbook.org](https://www.deeplearningbook.org/)

- **Neural Networks and Deep Learning**
  - *Author*: Michael Nielsen (Determination Press, 2015).
  - *Significance*: Clear, intuitive visual derivation of the 4 fundamental backpropagation equations.
  - *URL*: [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/)

---

## 2. Landmark Research Papers

### 2.1 Backpropagation & Computational Graphs
- **Learning Representations by Back-propagating Errors**
  - *Authors*: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams (Nature, 1986).
  - *Significance*: Seminal paper demonstrating that backpropagation learns useful internal representations.
- **Approximation by Superpositions of a Sigmoidal Function**
  - *Author*: George Cybenko (Mathematics of Control, Signals and Systems, 1989).
  - *Significance*: Mathematical proof of the Universal Approximation Theorem for feedforward networks.

### 2.2 Weight Initialization & Activations
- **Understanding the Difficulty of Training Deep Feedforward Neural Networks (Xavier Initialization)**
  - *Authors*: Xavier Glorot, Yoshua Bengio (AISTATS, 2010).
  - *Significance*: Demonstrated variance propagation across layers and derived Glorot/Xavier initialization.
- **Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet (He Initialization)**
  - *Authors*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (ICCV, 2015).
  - *Significance*: Addressed variance halving in ReLU units, deriving Kaiming/He initialization and PReLU.
- **Gaussian Error Linear Units (GELUs)**
  - *Authors*: Dan Hendrycks, Kevin Gimpel (arXiv, 2016).
  - *Significance*: Introduced the probabilistic gating GELU activation now standard in modern Transformers.
- **Searching for Activation Functions (Swish / SiLU)**
  - *Authors*: Prajit Ramachandran, Barret Zoph, Quoc V. Le (Google Brain, 2017).
  - *Significance*: Automated neural architecture search finding $x \cdot \sigma(\beta x)$, outperforming ReLU on deep models.

### 2.3 First-Order Optimization
- **Adam: A Method for Stochastic Optimization**
  - *Authors*: Diederik P. Kingma, Jimmy Ba (ICLR, 2015).
  - *Significance*: Introduced first and second moment tracking with bias correction.
- **Decoupled Weight Decay Regularization (AdamW)**
  - *Authors*: Ilya Loshchilov, Frank Hutter (ICLR, 2019).
  - *Significance*: Proved $L_2$ regularization $\neq$ weight decay in adaptive methods; established the AdamW standard.
- **On the Convergence of Adam and Beyond (AMSGrad)**
  - *Authors*: Sashank J. Reddi, Satyen Kale, Sanjiv Kumar (ICLR, 2018).
  - *Significance*: Identified convergence vulnerabilities in Adam when gradients are sparse or non-stationary.

---

## 3. Toolkits & Frameworks

- **PyTorch Core Documentation (`torch.nn`, `torch.optim`)**: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)
- **PyTorch Autograd Engine Mechanics**: [pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)
