# Seminal Research Papers: Computer Vision

Milestones in vision transformers, real-time object detection, and universal segmentation.

---

## 1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)
- **Title**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **Authors**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, et al.
- **Year**: 2020
- **Link**: https://arxiv.org/abs/2010.11929
- **Problem**: Reliance on convolutional inductive bias (spatial locality and translation invariance) limits vision architectures from leveraging massive data scaling.
- **Main Idea**: Flatten 2D image into a sequence of $16 \times 16$ pixel patches, project them linearly into 1D token embeddings, and feed into a standard Transformer encoder.
- **Key Contribution**: Demonstrated that pure Transformers without convolutional layers achieve state-of-the-art vision benchmarks when pre-trained on large datasets (JFT-300M).
- **Important Architecture/Math**:
  $$z_0 = [x_{\text{class}}; \; x_p^1 E; \; x_p^2 E; \dots; \; x_p^N E] + E_{\text{pos}}, \quad E \in \mathbb{R}^{(P^2 C) \times D}$$
- **Why It Matters**: Unified computer vision and natural language processing onto a single core Transformer architecture.
- **Prerequisites**: Transformer encoder, 2D convolution, transfer learning.
- **Suggested Follow-up Papers**: *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al., 2021); *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2021).

---

## 2. Segment Anything (SAM)
- **Title**: Segment Anything
- **Authors**: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, et al.
- **Year**: 2023
- **Link**: https://arxiv.org/abs/2304.02643
- **Problem**: Existing image segmentation models were specialized for narrow label categories and lacked zero-shot promptability.
- **Main Idea**: Build a promptable segmentation model (points, bounding boxes, free-form text) trained on an unprecedented dataset (SA-1B: 1 billion masks on 11 million images).
- **Key Contribution**: Decoupled heavy ViT image encoder from a lightweight sub-50ms prompt decoder running in real-time in web browsers.
- **Important Architecture/Math**:
  $$\text{Mask Decoder: Two-way cross-attention between prompt tokens and image embedding}$$
- **Why It Matters**: Created the first true foundation model for computer vision segmentation.
- **Prerequisites**: ViT backbone, semantic segmentation, cross-attention.
- **Suggested Follow-up Papers**: *SAM 2: Segment Anything in Images and Videos* (Ravi et al., 2024).
