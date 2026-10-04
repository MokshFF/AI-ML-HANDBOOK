# Image Classification & Transfer Learning - Technical Interview Preparation

A curated question bank covering convolutional backbones, global average pooling, linear probing vs fine-tuning, catastrophic forgetting, and top-$k$ evaluation.

---

## 1. Architectural & Optimization Concepts

### Q1: Why did Global Average Pooling (GAP) replace large Fully Connected (FC) layers in modern ConvNets?
- **Historical Problem (VGG/AlexNet)**:
  In VGG-16, the convolutional backbone outputs feature maps of size $7 \times 7 \times 512 = 25,088$. Flattening into two dense layers of 4,096 units requires:
  $$25,088 \times 4,096 + 4,096 \times 4,096 \approx 102.7\text{M} + 16.7\text{M} \approx 119.4\text{M parameters}$$
  Over $85\%$ of VGG's total 138M parameters were concentrated solely in the final dense classification head, causing massive memory usage and severe overfitting.
- **The GAP Innovation (Network in Network / ResNet)**:
  Global Average Pooling averages each feature map across the entire spatial grid $(H \times W \to 1 \times 1)$, producing a $1 \times C$ vector.
  1. *Zero Parameters*: GAP requires zero learned parameters, completely eliminating overfitting in the transition from conv feature maps to class logits.
  2. *Scale & Resolution Invariance*: GAP enables the network to accept variable-sized input images during inference without reshaping weights.

---

### Q2: What is "Catastrophic Forgetting" during transfer learning fine-tuning, and how can it be prevented?
- **Catastrophic Forgetting**:
  When a pre-trained backbone is fine-tuned with a randomly initialized new head using a high uniform learning rate, the huge initial random gradient updates from the uncalibrated head violently disrupt the pre-trained weights in the backbone, destroying valuable general visual representations.
- **Prevention Techniques**:
  1. **Two-Stage Training (Warmup / Linear Probing First)**: Freeze the backbone for $N$ epochs and train *only* the new head until loss stabilizes. Then unfreeze the backbone for gentle fine-tuning.
  2. **Discriminative Layer Learning Rates**: Apply exponentially smaller learning rates to earlier layers (e.g. $\eta_{\text{stem}} = 10^{-6}$, $\eta_{\text{mid}} = 10^{-5}$, $\eta_{\text{head}} = 10^{-3}$).
  3. **Weight Decay / L2 Regularization towards Pre-trained Weights** ($L_2$-SP): Penalize distance from pre-trained initialization: $\frac{\alpha}{2} \|\mathbf{w} - \mathbf{w}_0\|_2^2$.

---

### Q3: When is Linear Probing preferred over Full Fine-Tuning?
- **Data Size**: When target dataset has very few examples (e.g., $< 1,000$ images total). Fine-tuning millions of backbone weights risks extreme overfitting.
- **Domain Similarity**: When the target domain is visually close to the pre-training domain (e.g., natural photography vs natural photography).
- **Compute Constraints**: Linear probing only requires computing backbone embeddings once, caching them on disk, and solving a quick linear/logistic regression.

---

## 2. Whiteboard Coding Drills

### Q4: Implement a PyTorch transfer learning module with layer freezing and custom classifier replacement.
```python
import torch
import torch.nn as nn

class TransferModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
```
