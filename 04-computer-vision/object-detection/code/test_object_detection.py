"""
Unit tests for Object Detection Engine: IoU, NMS, YOLO Grid Head, and AP.
"""

import numpy as np
import pytest
import torch
from detection_engine import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    compute_iou,
    non_maximum_suppression,
    MiniYOLOHead,
    compute_average_precision,
)


def test_box_conversions():
    # Box: center at (5, 5), width 4, height 6 -> x1=3, y1=2, x2=7, y2=8
    cxcywh = torch.tensor([[5.0, 5.0, 4.0, 6.0]])
    xyxy = box_cxcywh_to_xyxy(cxcywh)
    assert torch.allclose(xyxy, torch.tensor([[3.0, 2.0, 7.0, 8.0]]))

    # Invert back
    restored = box_xyxy_to_cxcywh(xyxy)
    assert torch.allclose(restored, cxcywh)


def test_compute_iou():
    # Identical boxes: IoU should be 1.0
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = compute_iou(b1, b2)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0))

    # Disjoint boxes: IoU should be 0.0
    b3 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    iou_disjoint = compute_iou(b1, b3)
    assert torch.isclose(iou_disjoint[0, 0], torch.tensor(0.0))

    # 50% overlap box: width 10, height 10, shift x by 5
    # intersection: 5 * 10 = 50, union: 100 + 100 - 50 = 150 -> IoU = 50/150 = 1/3
    b4 = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    iou_partial = compute_iou(b1, b4)
    assert torch.isclose(iou_partial[0, 0], torch.tensor(1.0 / 3.0), atol=1e-4)


def test_non_maximum_suppression():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],  # Box 0: High score
        [1.0, 1.0, 10.0, 10.0],  # Box 1: Overlaps heavily with Box 0
        [50.0, 50.0, 60.0, 60.0], # Box 2: Distinct location
    ])
    scores = torch.tensor([0.9, 0.75, 0.85])

    keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.1)
    # Box 0 and Box 2 should be kept; Box 1 suppressed
    assert keep.tolist() == [0, 2]


def test_mini_yolo_head_forward():
    head = MiniYOLOHead(in_channels=16, grid_size=4, num_anchors=2, num_classes=3)
    features = torch.randn(2, 16, 4, 4)

    out = head(features)
    num_boxes = 4 * 4 * 2  # 32 boxes per image

    assert out["boxes_xyxy"].shape == (2, num_boxes, 4)
    assert out["objectness"].shape == (2, num_boxes)
    assert out["class_probs"].shape == (2, num_boxes, 3)

    # All probabilities and objectness in [0, 1]
    assert (out["objectness"] >= 0.0).all() and (out["objectness"] <= 1.0).all()
    assert torch.allclose(out["class_probs"].sum(dim=-1), torch.tensor(1.0))


def test_average_precision():
    recalls = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 0.9, 0.85, 0.7, 0.6, 0.5])

    ap = compute_average_precision(precisions, recalls)
    assert 0.0 < ap <= 1.0
