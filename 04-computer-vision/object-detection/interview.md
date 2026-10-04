# Object Detection - Technical Interview Preparation

A curated question bank covering two-stage vs. single-stage detectors, anchor box mechanics, RoI Align vs. RoI Pooling, NMS variants (Soft-NMS), and mAP evaluation.

---

## 1. Architectural & Theoretical Foundations

### Q1: Compare Two-Stage detectors (Faster R-CNN) and Single-Stage detectors (YOLO / RetinaNet). What are the fundamental trade-offs?
- **Speed vs. Accuracy Trade-off**:
  - *Two-Stage (Faster R-CNN)*: Decouples candidate proposal generation (RPN) from classification/refinement. Filtered proposals drastically reduce foreground-background class imbalance (e.g., from 100,000:1 to 300:1). Historically achieves higher localization precision and mAP on small objects, but runs slower ($5 - 20$ FPS).
  - *Single-Stage (YOLO/SSD)*: Formulates detection as a direct dense regression and classification problem over a unified feature grid. High inference throughput ($30 - 150+$ FPS), making it standard for real-time edge robotics and video surveillance.
- **The Extreme Class Imbalance Problem in Single-Stage Detectors**:
  Single-stage detectors evaluate tens of thousands of candidate locations per image, where $>99\%$ are easy background negative examples. RetinaNet introduced **Focal Loss** ($\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$) to dynamically down-weight easy background gradients, closing the accuracy gap with two-stage models.

---

### Q2: What was the flaw in RoI Pooling, and how did RoI Align (He et al., Mask R-CNN) resolve it?
- **RoI Pooling Quantization Errors**:
  RoI Pooling rounds floating-point proposal coordinates to integer grid coordinates twice:
  1. Quantizing proposal box $[x, y, w, h]$ to the feature map stride (e.g. dividing by 16 and rounding to integer).
  2. Dividing the region into $k \times k$ bins (e.g. $7 \times 7$) and rounding bin boundaries.
  This introduces spatial misalignment of up to 1 entire feature cell ($\approx 16$ pixels in original image space). While harmless for coarse classification, it severely degrades bounding box regression and pixel-accurate mask prediction.
- **RoI Align Solution**:
  RoI Align avoids all quantization. It samples 4 regular sampling points inside each continuous bin and calculates feature values using **bilinear interpolation** from adjacent feature grid cells, preserving exact sub-pixel spatial alignment.

---

### Q3: What is the failure mode of standard greedy NMS in crowded scenes, and how does Soft-NMS address it?
- **Standard NMS Failure Mode**:
  If two real objects of the same class are positioned close together (e.g., two people standing side by side with high bounding box overlap $\text{IoU} > 0.5$), standard greedy NMS completely eliminates the bounding box of the second person, causing a false negative.
- **Soft-NMS (Bodla et al., 2017)**:
  Instead of abruptly zeroing the score of overlapping boxes, Soft-NMS decays their confidence score continuously as a Gaussian or linear function of IoU:
  $$s_i \leftarrow s_i \exp\left( - \frac{\text{IoU}(M, b_i)^2}{\sigma} \right)$$
  Detections with high overlap and strong visual evidence survive with attenuated scores rather than being permanently discarded.

---

## 2. Whiteboard Coding Drills

### Q4: Implement vectorized Intersection-over-Union (IoU) between box sets $A$ and $B$ in PyTorch.
```python
import torch

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Args:
        boxes1: (N, 4) in [x1, y1, x2, y2]
        boxes2: (M, 4) in [x1, y1, x2, y2]
    Returns:
        (N, M) pairwise IoU tensor
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Intersection coordinates
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h

    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-7)
```
