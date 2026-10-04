# Semantic Segmentation: U-Net, Transposed Convolutions & Dice Loss

A rigorous mathematical and architectural guide to dense pixel-level visual prediction, comparing semantic, instance, and panoptic segmentation, detailing U-Net encoder-decoder skip connections, transposed convolution upsampling, and addressing severe spatial imbalance using Soft Dice Loss.

---

## 1. Pixel-Level Segmentation Paradigms

1. **Semantic Segmentation**: Assigns every pixel $(i, j)$ in an image to a discrete class category $c \in \{1, \dots, C\}$ (e.g., road, sky, car, person). It treats multiple instances of the same class as an indistinguishable collective mass.
2. **Instance Segmentation**: Detects individual object instances and delineates a distinct pixel mask for each (e.g. distinguishing Person A from Person B).
3. **Panoptic Segmentation**: Unifies semantic and instance segmentation: assigns every pixel both a semantic label ("stuff", e.g. grass, sky) and a unique instance identity ("things", e.g. specific pedestrians).

```
Semantic:  [Sky] [Sky] [Road] [Car] [Car] [Car] [Road]  (All cars share label 'Car')
Instance:  [  ]  [   ] [    ] [Car 1]   [Car 2] [    ]  (Only distinct foreground objects)
Panoptic:  [Sky:1]     [Road:1] [Car:1] [Car:2] [Road:1] (Unified stuff + things)
```

---

## 2. The U-Net Architecture (Ronneberger et al., 2015)

Standard classification backbones downsample feature maps to extract invariant high-level semantic features, destroying precise spatial boundary coordinates. U-Net resolves this via a symmetrical encoder-decoder topology with **long skip connections**.

```
Input Image (C_in, H, W)
      │
  [DoubleConv] ─────────────────────────── Skip Connection ──────────────────────────┐
      │ MaxPool 2x2                                                                 │
  [DoubleConv] ───────────────── Skip Connection ───────────────┐                   │
      │ MaxPool 2x2                                             │                   │
  [DoubleConv] ──────── Skip Connection ────┐                   │                   │
      │ MaxPool 2x2                         │                   │                   │
  [Bottleneck]                              │                   │                   │
      │ ConvTranspose2d 2x2                 │                   │                   │
  [Concat + DoubleConv] <───────────────────┘                   │                   │
      │ ConvTranspose2d 2x2                                     │                   │
  [Concat + DoubleConv] <───────────────────────────────────────┘                   │
      │ ConvTranspose2d 2x2                                                         │
  [Concat + DoubleConv] <───────────────────────────────────────────────────────────┘
      │
  [Conv2d 1x1] ──> Output Mask Logits (Num_Classes, H, W)
```

### Why Skip Connections are Vital
- The contracting path (encoder) extracts deep semantic abstractions ("what" is in the image) while reducing spatial resolution.
- The expanding path (decoder) restores spatial dimensions using transposed convolutions or bilinear upsampling.
- Concatenating early encoder feature maps directly with decoder layers restores fine-grained pixel localization ("where" edges and boundaries are) that was lost during pooling.

---

## 3. Loss Functions: Cross-Entropy vs. Soft Dice Loss

In clinical and remote sensing imagery, target lesions or objects often occupy $< 1\%$ of total pixels. Pixel-wise Cross-Entropy loss is dominated by trivial background pixels, causing models to predict all-background masks.

### Soft Dice Loss Formulation
The Sørensen-Dice coefficient measures region overlap. Soft Dice Loss relaxes discrete sets into continuous probability maps:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_i g_i + \epsilon}{\sum_{i=1}^N p_i^2 + \sum_{i=1}^N g_i^2 + \epsilon}$$
where $p_i \in [0, 1]$ is the predicted softmax probability and $g_i \in \{0, 1\}$ is the ground-truth binary label.
- **Combined Loss**: Often trained using $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{Dice}}$ to balance smooth gradient flow with boundary overlap.

---

## 4. Evaluation: Mean Intersection over Union (mIoU)

For class $c$, let $n_{ij}$ denote the number of pixels of class $i$ predicted as class $j$:
$$\text{IoU}_c = \frac{n_{cc}}{\sum_j n_{cj} + \sum_i n_{ic} - n_{cc}} = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$
$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \text{IoU}_c$$

---

## 5. Implementation Blueprint

- [`code/segmentation_engine.py`](code/segmentation_engine.py): Pure PyTorch implementations of `DoubleConv`, `UNetMini`, `DiceLoss`, and `compute_mean_iou`.
- [`code/test_semantic_segmentation.py`](code/test_semantic_segmentation.py): Unit tests for architecture shapes, Dice Loss convergence, and confusion-matrix IoU arithmetic.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab exploring U-Net forward/backward passes, loss comparison, and mask evaluations.
