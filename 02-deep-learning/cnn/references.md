# Convolutional Neural Networks - Curated References & Bibliography

A curated bibliography of seminal papers and landmark milestones in Convolutional Neural Networks.

---

## 1. Landmark Research Papers

### 1.1 Foundations & Early Milestones
- **Gradient-Based Learning Applied to Document Recognition (LeNet-5)**
  - *Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner (Proceedings of the IEEE, 1998).
  - *Significance*: Established modern ConvNet paradigm (convolution, sub-sampling, weight sharing).
- **ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)**
  - *Authors*: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (NeurIPS, 2012).
  - *Significance*: Sparked the deep learning revolution by winning ImageNet 2012 via GPU-accelerated CNNs with ReLU and Dropout.

### 1.2 Depth & Structural Regularity
- **Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)**
  - *Authors*: Karen Simonyan, Andrew Zisserman (ICLR, 2015).
  - *Significance*: Standardized $3 \times 3$ convolutional stacks and demonstrated the critical importance of visual representation depth.
- **Going Deeper with Convolutions (GoogLeNet / Inception)**
  - *Authors*: Christian Szegedy, Wei Liu, Yangqing Jia, et al. (CVPR, 2015).
  - *Significance*: Introduced multi-scale Inception modules with $1 \times 1$ bottleneck convolutions.
- **Deep Residual Learning for Image Recognition (ResNet)**
  - *Authors*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (CVPR, 2016).
  - *Significance*: Introduced identity skip connections, enabling networks beyond 100+ layers and winning ImageNet 2015.

### 1.3 Efficiency, Mobile Backbones & Neural Architecture Search
- **MobileNetV2: Inverted Residuals and Linear Bottlenecks**
  - *Authors*: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (CVPR, 2018).
  - *Significance*: Established inverted residuals with linear bottlenecks and depthwise separable convolutions for edge devices.
- **Squeeze-and-Excitation Networks (SENet)**
  - *Authors*: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu (CVPR, 2018).
  - *Significance*: Introduced dynamic channel-wise feature recalibration via SE blocks.
- **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**
  - *Authors*: Mingxing Tan, Quoc V. Le (ICML, 2019).
  - *Significance*: Formulated compound scaling balancing network depth, width, and image resolution.

---

## 2. Textbooks & Engineering Toolkits

- **Deep Learning for Computer Vision** by Rajalingappaa Shanmugamani (O'Reilly).
- **Computer Vision: Algorithms and Applications (2nd ed.)** by Richard Szeliski (Springer).
- **Torchvision Models Documentation**: [pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
- **Timm (PyTorch Image Models)**: Canonical open-source vision backbone repository by Ross Wightman. [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
