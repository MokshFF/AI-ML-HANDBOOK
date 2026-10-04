# Semantic Segmentation - References & Further Reading

Seminal research papers, architectures, and benchmarks in dense pixel-level image segmentation.

---

## 1. Seminal Research Papers

- **U-Net: Convolutional Networks for Biomedical Image Segmentation** (2015)
  - *Authors*: Olaf Ronneberger, Philipp Fischer, Thomas Brox
  - *Paper*: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
  - *Contribution*: Introduced symmetrical contracting and expanding paths connected by high-resolution skip connections.

- **Fully Convolutional Networks for Semantic Segmentation (FCN)** (2014 / 2015)
  - *Authors*: Jonathan Long, Evan Shelhamer, Trevor Darrell
  - *Paper*: [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
  - *Contribution*: Adapted classification ConvNets into fully convolutional dense spatial predictors using in-network upsampling.

- **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs** (2016 / 2017)
  - *Authors*: Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
  - *Paper*: [arXiv:1606.00915](https://arxiv.org/abs/1606.00915)
  - *Contribution*: Introduced atrous (dilated) convolutions and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context without losing spatial resolution.

- **V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation (Dice Loss)** (2016)
  - *Authors*: Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi
  - *Paper*: [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
  - *Contribution*: Formulated the continuous differentiable Soft Dice loss function for 3D and 2D segmentation with extreme class imbalance.

---

## 2. Textbooks & Survey Articles

- **Computer Vision: Algorithms and Applications (2nd ed.)**
  - *Author*: Richard Szeliski (Springer)
  - *Chapter*: Chapter 7: Deep Learning & Dense Prediction.
- **Cityscapes Dataset: Semantic Understanding of Urban Street Scenes**
  - *Authors*: Marius Cordts et al.
  - *URL*: [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
