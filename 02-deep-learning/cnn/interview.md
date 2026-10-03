# Convolutional Neural Networks - Technical Interview Preparation

A curated question bank covering spatial operations, receptive fields, residual gradient dynamics, and modern CNN architectural patterns.

---

## 1. Spatial Geometry & Operational Mechanics

### Q1: State the general formula for the output dimensions of a 2D convolution and explain the purpose of Dilation.
- **Output Formula**:
  For an input of height $H$, kernel size $K$, padding $P$, stride $S$, and dilation rate $d$:
  $$H_{\text{out}} = \left\lfloor \frac{H + 2P - d(K - 1) - 1}{S} \right\rfloor + 1$$
- **Dilated (Atrous) Convolutions**:
  - Dilation inserts $d - 1$ spaces between kernel elements without introducing additional learnable parameters.
  - An effective kernel size becomes $K' = d(K - 1) + 1$. For example, a $3 \times 3$ kernel with dilation $d=2$ covers a $5 \times 5$ spatial area while maintaining only 9 parameters.
  - **Use Case**: Semantic segmentation (e.g., DeepLab) and dense audio modeling (WaveNet), where expanding the receptive field without downsampling preserves fine-grained spatial resolution.

---

### Q2: What is the computational savings of Depthwise Separable Convolutions over standard 2D convolutions?
- **Standard Convolution**:
  Input: $(H, W, D_{\text{in}})$, Output: $(H, W, D_{\text{out}})$, Kernel: $D_k \times D_k$.
  $$\text{FLOPs}_{\text{standard}} = H \cdot W \cdot D_{\text{in}} \cdot D_{\text{out}} \cdot D_k^2$$
- **Depthwise Separable Convolution (MobileNet / EfficientNet)**:
  Decomposed into two sequential stages:
  1. **Depthwise Conv**: Applies a single $D_k \times D_k$ filter per input channel ($\text{groups} = D_{\text{in}}$):
     $$\text{FLOPs}_{\text{depthwise}} = H \cdot W \cdot D_{\text{in}} \cdot D_k^2$$
  2. **Pointwise Conv**: Applies $1 \times 1$ convolutions across channels to project from $D_{\text{in}} \to D_{\text{out}}$:
     $$\text{FLOPs}_{\text{pointwise}} = H \cdot W \cdot D_{\text{in}} \cdot D_{\text{out}}$$
- **Ratio of Computation**:
  $$\text{Ratio} = \frac{H \cdot W \cdot D_{\text{in}} \cdot D_k^2 + H \cdot W \cdot D_{\text{in}} \cdot D_{\text{out}}}{H \cdot W \cdot D_{\text{in}} \cdot D_{\text{out}} \cdot D_k^2} = \frac{1}{D_{\text{out}}} + \frac{1}{D_k^2}$$
  For standard $3 \times 3$ kernels ($D_k = 3$) with $D_{\text{out}} \ge 64$:
  $$\text{Ratio} \approx \frac{1}{9} \approx \mathbf{11\% \text{ of the original computation (an 8-9x speedup!)}}$$

---

## 2. Architectural Breakthroughs & Residual Dynamics

### Q3: What is the "Degradation Problem" in deep CNNs, and how does ResNet solve it mathematically?
- **The Degradation Problem**:
  Prior to ResNet, stacking more layers beyond $\sim 20$ caused training error to *increase* (not overfitting, since training error worsened).
  In theory, a deeper model should have a solution space that includes the shallower model by setting added layers to the identity mapping $f(x) = x$. However, multiple non-linear layers ($W_2 \cdot \sigma(W_1 x + b_1) + b_2$) struggle to learn the identity function via standard gradient descent.
- **The Residual Shortcut**:
  ResNet reformulates the layer to learn residual mapping $\mathcal{F}(x) = \mathcal{H}(x) - x$, outputting:
  $$\mathcal{H}(x) = \mathcal{F}(x) + x$$
  If identity is the optimal mapping, the optimizer easily drives weights toward zero ($\mathcal{F}(x) \to 0$).
- **Gradient Highway Proof**:
  Consider recursion across $L$ residual units: $x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}_i(x_i)$.
  During backpropagation:
  $$\frac{\partial \mathcal{E}}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_L} \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}_i(x_i) \right)$$
  The term $\frac{\partial \mathcal{E}}{\partial x_L} \cdot I$ provides an uninterrupted gradient channel directly from the loss back to any arbitrary shallow layer $x_l$, completely preventing vanishing gradients even in 1000-layer networks.

---

### Q4: Why did Global Average Pooling (GAP) replace Flattening + Dense Layers in modern architectures?
- **Parameter Explosion**:
  In AlexNet and VGG-16, the transition from final feature maps ($7 \times 7 \times 512$) to fully connected layers ($4096$) required:
  $$7 \times 7 \times 512 \times 4096 \approx 102.7\text{ Million Parameters!}$$
  Over $80\%$ of all parameters in VGG-16 were concentrated in this single flattening transition, making it extremely prone to severe overfitting.
- **GAP Benefits**:
  1. Computes the spatial average of each channel ($H \times W \to 1 \times 1$).
  2. Reduces parameters from millions to zero for the pooling step.
  3. Enforces direct correspondence between feature map channels and semantic category confidence maps (interpretability).
  4. Allows input images of arbitrary spatial dimensions during test-time inference.

---

## 3. Whiteboard Coding Drills

### Q5: Write a PyTorch implementation of a ResNet Residual Block with optional projection shortcut.
```python
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Identity shortcut or 1x1 projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)
```
