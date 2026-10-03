# Deep Learning Training Dynamics - Curated References & Bibliography

A curated collection of foundational research papers, landmark monographs, and official documentation on regularization, normalization, scheduling, and distributed scale.

---

## 1. Landmark Research Papers

### 1.1 Normalization & Optimization Dynamics
- **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**
  - *Authors*: Sergey Ioffe, Christian Szegedy (ICML, 2015).
  - *Significance*: Introduced Batch Normalization, standardizing deep visual network training.
- **How Does Batch Normalization Help Optimization?**
  - *Authors*: Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry (NeurIPS, 2018).
  - *Significance*: Demonstrated that BatchNorm smooths the loss landscape rather than reducing internal covariate shift.
- **Layer Normalization**
  - *Authors*: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (arXiv, 2016).
  - *Significance*: Formulated Layer Normalization, universally adopted in modern Transformer and sequential architectures.
- **Group Normalization**
  - *Authors*: Yuxin Wu, Kaiming He (ECCV, 2018).
  - *Significance*: Normalization independent of batch size for visual perception tasks.

### 1.2 Regularization & Data Augmentation
- **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**
  - *Authors*: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (JMLR, 2014).
  - *Significance*: Canonical paper on stochastic neuron deactivation as an ensemble regularizer.
- **mixup: Beyond Empirical Risk Minimization**
  - *Authors*: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz (ICLR, 2018).
  - *Significance*: Linear convex interpolation of input samples and target vectors.
- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**
  - *Authors*: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, et al. (ICCV, 2019).
  - *Significance*: Spatial patch cutting and pasting with proportional label mixing.

### 1.3 Learning Rate Schedules & Distributed Scalability
- **SGDR: Stochastic Gradient Descent with Warm Restarts (Cosine Annealing)**
  - *Authors*: Ilya Loshchilov, Frank Hutter (ICLR, 2017).
  - *Significance*: Introduced cosine learning rate schedules with periodic warm restarts.
- **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour**
  - *Authors*: Priya Goyal, Piotr Dollár, Ross Girshick, et al. (arXiv, 2017).
  - *Significance*: Established linear scaling rule and gradual warmup for massive batch training.
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**
  - *Authors*: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxian He (SC, 2020).
  - *Significance*: Introduced memory sharding of optimizer states, gradients, and parameters in DeepSpeed and PyTorch FSDP.

---

## 2. Textbooks & Production Frameworks

- **PyTorch Distributed Overview (`torch.distributed`)**: [pytorch.org/tutorials/beginner/dist_overview.html](https://pytorch.org/tutorials/beginner/dist_overview.html)
- **DeepSpeed Documentation (Microsoft)**: [deepspeed.ai](https://www.deepspeed.ai/)
- **PyTorch FSDP (Fully Sharded Data Parallel)**: [pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
