# Object Detection - References & Further Reading

Seminal research papers, milestones, and benchmark suites in modern object detection.

---

## 1. Seminal Research Papers

- **You Only Look Once: Unified, Real-Time Object Detection (YOLO)** (2015 / 2016)
  - *Authors*: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
  - *Paper*: [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
  - *Contribution*: Formulated object detection as a single regression problem from image pixels to bounding box coordinates and class probabilities.

- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** (2015)
  - *Authors*: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
  - *Paper*: [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
  - *Contribution*: Introduced Region Proposal Networks (RPN) sharing full-image convolutional features with the detection network.

- **Mask R-CNN (RoI Align)** (2017)
  - *Authors*: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
  - *Paper*: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
  - *Contribution*: Introduced RoI Align to eliminate quantization errors, extending Faster R-CNN to pixel-level instance segmentation.

- **Focal Loss for Dense Object Detection (RetinaNet)** (2017 / 2018)
  - *Authors*: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
  - *Paper*: [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
  - *Contribution*: Solved extreme foreground-background class imbalance in dense one-stage detectors using Focal Loss.

- **Soft-NMS -- Improving Object Detection With One Line of Code** (2017)
  - *Authors*: Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S. Davis
  - *Paper*: [arXiv:1704.04503](https://arxiv.org/abs/1704.04503)
  - *Contribution*: Decayed detection scores as a continuous function of overlap rather than hard zeroing.

---

## 2. Benchmark Suites & Datasets

- **Microsoft COCO: Common Objects in Context**
  - *Authors*: Tsung-Yi Lin et al.
  - *URL*: [https://cocodataset.org/](https://cocodataset.org/)
- **The Pascal Visual Object Classes Challenge (VOC)**
  - *Authors*: Mark Everingham et al.
  - *URL*: [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)
