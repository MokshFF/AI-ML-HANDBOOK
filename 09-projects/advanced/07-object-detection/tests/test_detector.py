"""Tests for Object Detector and NMS."""
import torch
from detector import compute_iou, non_max_suppression, LightweightDetector

def test_iou_calculation():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    ious = compute_iou(b1, b2)
    assert torch.isclose(ious[0, 0], torch.tensor(1.0))
    assert 0.14 < ious[0, 1].item() < 0.15

def test_nms_filtering():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0]
    ])
    scores = torch.tensor([0.9, 0.85, 0.8])
    kept = non_max_suppression(boxes, scores, iou_thresh=0.5)
    assert kept == [0, 2]

def test_detector_forward():
    model = LightweightDetector(n_classes=3, n_anchors=4)
    imgs = torch.randn(2, 3, 32, 32)
    cls_logits, bbox_deltas = model(imgs)
    assert cls_logits.shape == (2, 4, 3)
    assert bbox_deltas.shape == (2, 4, 4)
