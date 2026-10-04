# Object Detection: Geometry, Anchor Boxes, NMS & YOLO Grid Heads

A comprehensive mathematical and implementation guide to two-stage and single-stage object detection architectures, bounding box coordinate conversions, Intersection over Union (IoU), Non-Maximum Suppression (NMS), and Mean Average Precision (mAP) evaluation.

---

## 1. Object Detection Problem Formulation

Unlike image classification, object detection simultaneously answers **what** is present and **where** it is located. For an input image $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$, the model predicts an arbitrary number of object instances:
$$\mathcal{Y} = \{(\mathbf{b}_k, c_k, s_k)\}_{k=1}^K$$
where:
- $\mathbf{b}_k = (x_{\min}, y_{\min}, x_{\max}, y_{\max})$ or $(c_x, c_y, w, h)$ is the spatial bounding box.
- $c_k \in \{1, \dots, C\}$ is the categorical class label.
- $s_k \in [0, 1]$ is the objectness confidence score.

---

## 2. Core Geometric Foundations

### 2.1 Intersection over Union (IoU)
Measures the geometric overlap between a predicted box $B_{\text{pred}}$ and a ground-truth box $B_{\text{gt}}$:
$$\text{IoU}(B_{\text{pred}}, B_{\text{gt}}) = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}}) + \text{Area}(B_{\text{gt}}) - \text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}$$
- $\text{IoU} \in [0, 1]$; $\text{IoU} \ge 0.5$ is conventionally considered a True Positive match in Pascal VOC; COCO evaluates across thresholds $[0.50 : 0.05 : 0.95]$.

### 2.2 Non-Maximum Suppression (NMS)
Dense detector heads generate hundreds of candidate boxes per object. NMS iteratively filters duplicates:
1. Discard all candidate boxes with confidence score below threshold $\tau_{\text{score}}$ (e.g. $0.05$).
2. Sort remaining candidate boxes in descending order of confidence score.
3. Select highest-scoring box $B^*$ and append to final detections list.
4. Calculate $\text{IoU}(B^*, B_j)$ for all remaining candidates. If $\text{IoU} > \tau_{\text{iou}}$ (e.g. $0.5$), remove $B_j$.
5. Repeat steps 3–4 until candidate list is exhausted.

---

## 3. Detection Paradigms: Two-Stage vs. Single-Stage

```
Two-Stage (Faster R-CNN):
Image ──> Backbone ──> Feature Map ──> Region Proposal Net (RPN) ──> RoI Align / Pooling ──> Fast R-CNN Head (Class + Reg)

Single-Stage (YOLO / RetinaNet):
Image ──> Backbone ──> Multi-Scale Features (FPN) ──> Dense Grid / Anchor Head ──> NMS Post-Processing
```

### 3.1 Two-Stage: Faster R-CNN
- **Stage 1 (RPN)**: Dense sliding-window network predicting candidate object proposals ("regions of interest") using multi-scale anchor boxes.
- **RoI Align**: Extracts fixed-size feature vectors ($7 \times 7$) from proposals using bilinear interpolation (avoiding quantization errors of RoI Pooling).
- **Stage 2**: Dedicated MLP heads refine box coordinates and predict fine-grained class probabilities.

### 3.2 Single-Stage: YOLO (You Only Look Once)
- Divides the feature map into an $S \times S$ spatial grid.
- If an object center falls into cell $(c_x, c_y)$, that cell is responsible for detecting the object.
- **Grid Offset Parameterization**:
  $$b_x = \frac{\sigma(t_x) + c_x}{S}, \quad b_y = \frac{\sigma(t_y) + c_y}{S}$$
  $$b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}$$
  where $p_w, p_h$ are anchor priors.

---

## 4. Evaluation: Mean Average Precision (mAP)

For each class $c$:
1. Rank detections by descending confidence.
2. Match each prediction to the highest-IoU unassigned ground-truth box. If $\text{IoU} \ge \tau$, mark as True Positive (TP); otherwise False Positive (FP).
3. Compute cumulative Precision $P(r)$ and Recall $r$.
4. **Average Precision (AP)**: Compute area under interpolated precision-recall curve:
   $$\text{AP} = \int_0^1 p_{\text{interp}}(r) \, dr, \quad p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$
5. **mAP**: Mean of AP across all $C$ categories.

---

## 5. Implementation Blueprint

- [`code/detection_engine.py`](code/detection_engine.py): Vectorized PyTorch implementations of `box_cxcywh_to_xyxy`, `compute_iou`, `non_maximum_suppression`, `MiniYOLOHead`, and `compute_average_precision`.
- [`code/test_object_detection.py`](code/test_object_detection.py): Unit tests for geometry, overlapping suppression, grid decoding, and AP integration.
- [`notebook.ipynb`](notebook.ipynb): Interactive lab exploring bounding box operations, NMS filtering, and YOLO grid outputs.
