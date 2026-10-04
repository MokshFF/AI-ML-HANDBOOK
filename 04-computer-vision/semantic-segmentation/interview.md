# Semantic Segmentation - Technical Interview Preparation

A curated question bank covering U-Net architectures, transposed convolutions vs. bilinear upsampling, checkerboard artifacts, Dice Loss derivatives, and boundary refinement.

---

## 1. Architectural & Theoretical Foundations

### Q1: Compare Transposed Convolution (Deconvolution) vs. Bilinear Upsampling + Conv2d. Why do transposed convolutions often cause "checkerboard artifacts"?
- **Transposed Convolution Mechanics**:
  Transposed convolution places filter weights into an expanded grid with stride $s$, learning upsampling parameters.
- **Checkerboard Artifacts**:
  When kernel size $k$ is not evenly divisible by stride $s$ (e.g. $k=3, s=2$), adjacent filter strides produce uneven overlap in output pixel updates. Some pixels receive contributions from two filter locations while adjacent pixels receive contributions from only one. This creates a high-frequency grid-like "checkerboard" texture artifact in reconstructed feature maps.
- **Modern Solution**:
  Replace `ConvTranspose2d` with parameter-free **Bilinear / Nearest-Neighbor Upsampling** followed by a standard $3 \times 3$ `Conv2d`. This guarantees uniform spatial interpolation with zero checkerboard patterns.

---

### Q2: Why is Cross-Entropy Loss inadequate for medical image segmentation, and how does Soft Dice Loss resolve the issue?
- **The Issue with Cross-Entropy**:
  Cross-entropy treats every single pixel independently. If an organ or lesion covers $100$ pixels in a $1000 \times 1000$ image ($0.01\%$ foreground), a trivial model predicting "all background" achieves $99.99\%$ pixel accuracy and near-zero cross-entropy loss despite failing completely on the clinical task.
- **Why Soft Dice Loss Solves It**:
  Soft Dice loss measures global set overlap rather than independent pixel classifications:
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |\mathbf{P} \cap \mathbf{G}|}{|\mathbf{P}| + |\mathbf{G}|}$$
  Because the denominator scales with the size of the true foreground plus predicted foreground rather than total image area, small target regions are weighted proportionally to their relative overlap rather than drowned out by background mass.

---

### Q3: What is the difference between Semantic, Instance, and Panoptic Segmentation?
- **Semantic Segmentation**: Predicts class labels per pixel. No distinction is made between separate object instances (e.g., five overlapping cars are merged into a single mask).
- **Instance Segmentation**: Detects and delineates individual countable objects ("things"). Background pixels (e.g., roads, grass, buildings) are ignored.
- **Panoptic Segmentation**: Complete scene understanding. Every pixel is assigned a semantic class and every countable object is assigned an instance ID.

---

## 2. Whiteboard Coding Drills

### Q4: Implement a differentiable Soft Dice Loss module in PyTorch for multi-class segmentation.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W) integer class indices
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs * probs + targets_one_hot * targets_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)
```
