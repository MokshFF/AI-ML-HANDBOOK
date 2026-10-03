# Image Classification - Interview Preparation & Question Bank

This document outlines high-frequency technical, conceptual, and system-design questions related to **Image Classification**.

---

## 1. Conceptual & Theoretical Foundations

### Q1: What are the fundamental principles and assumptions underlying Image Classification?
- **Key Discussion Points**:
  - Primary problem formulation and mathematical objectives.
  - Assumptions made regarding data distribution, feature independence, or linearity.
  - Failure modes when underlying assumptions are violated in real-world scenarios.

### Q2: How does Image Classification compare to alternative paradigms or legacy approaches?
- **Key Discussion Points**:
  - Computational complexity (time and space during training vs. inference).
  - Sample efficiency and data volume requirements.
  - Interpretability vs. expressive capacity trade-offs.

---

## 2. Practical Engineering & Troubleshooting

### Q3: What are the most common failure modes and diagnostic strategies?
- **Common Symptoms**:
  - Divergent loss curves, vanishing/exploding gradients, or stagnant metric improvement.
  - High variance (overfitting) vs. high bias (underfitting).
  - Train-serve skew, distribution shift, or data leakage.
- **Diagnostic Playbook**:
  - Baseline testing on minimal synthetic data (sanity check capacity to overfit 1 batch).
  - Gradient clipping, learning rate warmup, and normalization checks.
  - Feature attribution and ablation analysis.

---

## 3. Production & Scalability Considerations

### Q4: How would you design and deploy this in a latency-critical production pipeline?
- **Key Dimensions**:
  - Batching strategies vs. streaming/real-time inference constraints.
  - Quantization, pruning, distillation, and hardware target (CPU vs. GPU vs. Edge).
  - Telemetry: SLA metrics (p95/p99 latency), drift monitoring, and fallbacks.

---

## 4. Coding & Whiteboard Drills
- Implement the core mechanism from scratch in pure Python / NumPy without high-level abstractions.
- Vectorize key operations to avoid explicit Python loops.
- Handle edge cases: zero division, non-invertible matrices, extreme outliers, or missing tokens.
