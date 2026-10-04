"""Anchor-Based Object Detector and NMS Engine."""
import torch
import torch.nn as nn
from typing import List, Tuple

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise IoU between boxes: [x1, y1, x2, y2]."""
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area
    return inter_area / (union_area + 1e-8)

def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> List[int]:
    """Greedy Non-Maximum Suppression."""
    order = scores.argsort(descending=True)
    keep = []
    
    while len(order) > 0:
        idx = order[0].item()
        keep.append(idx)
        if len(order) == 1:
            break
            
        ious = compute_iou(boxes[idx].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        mask = ious <= iou_thresh
        order = order[1:][mask]
        
    return keep

class LightweightDetector(nn.Module):
    def __init__(self, n_classes: int = 3, n_anchors: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.cls_head = nn.Linear(32 * 4 * 4, n_anchors * n_classes)
        self.reg_head = nn.Linear(32 * 4 * 4, n_anchors * 4)
        self.n_anchors = n_anchors
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x).flatten(1)
        cls_logits = self.cls_head(feat).view(-1, self.n_anchors, self.n_classes)
        bbox_deltas = self.reg_head(feat).view(-1, self.n_anchors, 4)
        return cls_logits, bbox_deltas

if __name__ == "__main__":
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 49.0, 52.0],  # Overlap with first box
        [100.0, 100.0, 150.0, 150.0]
    ])
    scores = torch.tensor([0.92, 0.88, 0.95])
    kept = non_max_suppression(boxes, scores, iou_thresh=0.5)
    print("Boxes kept after NMS:", kept)
