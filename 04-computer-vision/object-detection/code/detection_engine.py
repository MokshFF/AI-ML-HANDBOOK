"""
Object Detection Mechanics: Coordinate Conversions, IoU, NMS, and YOLO Grid Head.

Implements from scratch using pure PyTorch and NumPy:
- box_cxcywh_to_xyxy & box_xyxy_to_cxcywh: Coordinate transformations.
- compute_iou: Pairwise Intersection over Union matrix.
- non_maximum_suppression: Greedy post-processing filtering overlapping candidate boxes.
- MiniYOLOHead: Dense single-stage detection head with grid decoding.
- compute_average_precision: Interpolated AP under Precision-Recall curves.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from center-size format (cx, cy, w, h)
    to corner format (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from corner format (x1, y1, x2, y2)
    to center-size format (cx, cy, w, h).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Intersection over Union (IoU) between two sets of boxes (xyxy format).
    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        (N, M) matrix of IoU values in [0, 1].
    """
    # Area of each box: (x2 - x1) * (y2 - y1)
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

    # Union = Area1 + Area2 - Intersection
    union = area1[:, None] + area2 - intersection
    iou = intersection / union.clamp(min=1e-7)
    return iou


def non_maximum_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Greedy Non-Maximum Suppression (NMS).
    Filters redundant bounding boxes whose IoU with a higher-scoring candidate exceeds iou_threshold.
    Returns indices of retained boxes.
    """
    # Filter out low-confidence candidates
    valid_mask = scores >= score_threshold
    if not valid_mask.any():
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    indices = torch.where(valid_mask)[0]
    boxes = boxes[indices]
    scores = scores[indices]

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        current_idx = order[0]
        keep.append(indices[current_idx].item())

        if order.numel() == 1:
            break

        # Compute IoU of remaining boxes with the highest scoring box
        current_box = boxes[current_idx].unsqueeze(0)
        remaining_boxes = boxes[order[1:]]
        ious = compute_iou(current_box, remaining_boxes).squeeze(0)

        # Retain only boxes whose IoU is below the threshold
        below_thresh = torch.where(ious <= iou_threshold)[0]
        # order[1:] shifted indices
        order = order[below_thresh + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class MiniYOLOHead(nn.Module):
    """
    Lightweight Single-Stage YOLO-style Detection Head.
    Divides feature map into S x S grid cells; predicts (tx, ty, tw, th), objectness, and class logits.
    """
    def __init__(self, in_channels: int = 64, grid_size: int = 4, num_anchors: int = 1, num_classes: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Output per anchor: 4 (box coords) + 1 (objectness) + num_classes
        self.out_per_anchor = 5 + num_classes
        self.conv = nn.Conv2d(in_channels, num_anchors * self.out_per_anchor, kernel_size=1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, C_in, S, S)
        Returns:
            decoded_boxes: (B, num_boxes, 4) in normalized [0, 1] xyxy format
            objectness: (B, num_boxes) in [0, 1]
            class_probs: (B, num_boxes, num_classes)
        """
        b, _, s, _ = features.shape
        raw = self.conv(features)  # (B, A * (5+C), S, S)
        raw = raw.view(b, self.num_anchors, self.out_per_anchor, s, s)
        raw = raw.permute(0, 3, 4, 1, 2).contiguous()  # (B, S, S, A, 5+C)

        tx = raw[..., 0]
        ty = raw[..., 1]
        tw = raw[..., 2]
        th = raw[..., 3]
        obj_logits = raw[..., 4]
        class_logits = raw[..., 5:]

        # Grid offsets
        device = features.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(s, device=device),
            torch.arange(s, device=device),
            indexing="ij"
        )
        grid_x = grid_x.view(1, s, s, 1).expand(b, s, s, self.num_anchors)
        grid_y = grid_y.view(1, s, s, 1).expand(b, s, s, self.num_anchors)

        # YOLO parameterization: cx = (sigmoid(tx) + grid_x) / S, cy = (sigmoid(ty) + grid_y) / S
        cx = (torch.sigmoid(tx) + grid_x) / float(s)
        cy = (torch.sigmoid(ty) + grid_y) / float(s)
        w = torch.sigmoid(tw)  # Normalized box width
        h = torch.sigmoid(th)  # Normalized box height

        cxcywh = torch.stack([cx, cy, w, h], dim=-1).view(b, -1, 4)
        xyxy = box_cxcywh_to_xyxy(cxcywh)

        objectness = torch.sigmoid(obj_logits).view(b, -1)
        class_probs = F.softmax(class_logits, dim=-1).view(b, -1, self.num_classes)

        return {
            "boxes_xyxy": xyxy,
            "objectness": objectness,
            "class_probs": class_probs,
        }


def compute_average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using continuous all-point trapezoidal/step area integration.
    """
    # Prepend 0 and append 1 for recall, prepend 1 and append 0 for precision
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area under curve where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)
