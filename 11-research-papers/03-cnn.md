# Seminal Research Papers: Convolutional Neural Networks

Milestones in spatial feature extraction, receptive fields, and multi-scale visual modeling.

---

## 1. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
- **Title**: ImageNet Classification with Deep Convolutional Neural Networks
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **Year**: 2012
- **Link**: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
- **Problem**: High-resolution image classification across 1,000 categories without hand-crafted SIFT/HOG descriptors.
- **Main Idea**: Train a deep 8-layer convolutional neural network end-to-end on GPUs using non-saturating non-linearities and aggressive regularization.
- **Key Contribution**: Popularized ReLU activations, Dropout, GPU-accelerated convolution, and data augmentation.
- **Important Architecture/Math**:
  $$\text{ReLU}(z) = \max(0, z), \quad \text{Overlapping Max Pooling with stride } s < z$$
- **Why It Matters**: Catalyzed the modern AI spring by winning the ImageNet 2012 challenge with an unprecedented $15.3\%$ top-5 error rate ($10.8\%$ better than runner-up).
- **Prerequisites**: Convolution operation, stochastic gradient descent, GPU hardware basics.
- **Suggested Follow-up Papers**: *Very Deep Convolutional Networks for Large-Scale Image Recognition* (Simonyan & Zisserman, VGG, 2014); *Going Deeper with Convolutions* (Szegedy et al., GoogLeNet, 2015).

---

## 2. A ConvNet for the 2020s (ConvNeXt)
- **Title**: A ConvNet for the 2020s
- **Authors**: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
- **Year**: 2022
- **Link**: https://arxiv.org/abs/2201.03545
- **Problem**: Vision Transformers (ViT) were outperforming pure ConvNets, raising the question of whether CNN architectures had hit intrinsic limits.
- **Main Idea**: Systematically modernize a standard ResNet using Vision Transformer architectural design choices (patchify stem, 7x7 depthwise convs, inverted bottleneck, GELU, fewer normalizations).
- **Key Contribution**: Proved pure ConvNets can match or exceed Swin Transformer accuracy while maintaining superior inference simplicity and throughput.
- **Important Architecture/Math**:
  $$\text{Depthwise 7x7 Conv} \to \text{LayerNorm} \to \text{Linear}(4C) \to \text{GELU} \to \text{Linear}(C)$$
- **Why It Matters**: Re-established ConvNets as state-of-the-art vision backbones for downstream detection and segmentation tasks.
- **Prerequisites**: ResNet, Swin Transformer, depthwise separable convolutions.
- **Suggested Follow-up Papers**: *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders* (Woo et al., 2023).
