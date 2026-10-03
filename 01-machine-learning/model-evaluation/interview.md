# Model Evaluation: Technical Interview Question Bank

Technical screening questions, evaluation dilemmas, metric trade-offs, and probability calibration diagnostics.

---

## 1. Classification Metrics & Thresholding

### Q1: In a credit card fraud detection system with 0.1% fraud prevalence, your model achieves 99.8% accuracy and an ROC-AUC of 0.95. Why might the model still be unusable in production?
- **Answer Outline**:
  - **The Imbalance Illusion**: A naive model predicting "Not Fraud" for every transaction achieves 99.9% accuracy. Reporting 99.8% accuracy is actually worse than the naive baseline.
  - **ROC-AUC Insensitivity**: The False Positive Rate $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ has a massive denominator of $\approx 99.9\%$ legitimate transactions ($\text{TN}$). Even 5,000 false positive fraud alerts per day produces a tiny FPR $\approx 0.005$, keeping the ROC curve artificially elevated.
  - **Production Reality (Precision-Recall)**: Those 5,000 false alarms will overwhelm the human fraud operations team. If the model only detected 50 true frauds alongside those 5,000 false alarms, the **Precision is only $\approx 1\%$**! Inspecting **PR-AUC** and tuning threshold for high precision is required.

### Q2: What is the probabilistic interpretation of ROC-AUC?
- **Answer Outline**:
  - The Area Under the Receiver Operating Characteristic Curve (ROC-AUC) equals the probability that a randomly drawn positive instance $(\mathbf{x}^+)$ receives a higher model score than a randomly drawn negative instance $(\mathbf{x}^-)$:
    $$\text{ROC-AUC} = P(\hat{s}(\mathbf{x}^+) > \hat{s}(\mathbf{x}^-))$$
  - Mathematically equivalent to the normalized Wilcoxon-Mann-Whitney rank-sum statistic:
    $$\text{AUC} = \frac{1}{N^+ N^-} \sum_{i: y_i=1} \sum_{j: y_j=0} \mathbb{I}(\hat{s}_i > \hat{s}_j)$$
  - It is completely threshold-independent and invariant to monotonic score transformations (e.g., scaling probabilities does not change AUC).

---

## 2. Validation Strategies & Diagnostics

### Q3: When does standard K-Fold cross-validation produce completely invalid, over-optimistic performance estimates?
- **Answer Outline**:
  1. **Time-Series / Temporal Data**: Random shuffling allows future data into the training fold to predict historical validation folds (lookahead bias). Use `TimeSeriesSplit` (Walk-Forward).
  2. **Grouped / Clustered Data**: If multiple MRI scans come from the same patient, random K-Fold puts some scans from patient A in train and other scans from patient A in test. The model memorizes patient anatomy rather than disease patterns. Use `GroupKFold`.
  3. **High-Cardinality Target Encoding / Scaling Before Splitting**: Preprocessing on the full dataset leaks validation fold distribution statistics into the training fold. Wrap all transforms in an isolated `Pipeline`.

### Q4: Why are modern deep neural networks often poorly calibrated despite having near-perfect accuracy, and how do you calibrate them?
- **Answer Outline**:
  - **Cause (Over-confidence)**: Deep networks have immense parameter capacity trained with cross-entropy loss to drive logits to extreme values ($z \to \pm \infty$) to reduce loss to zero. Weight decay and unconstrained depth cause model probabilities to cluster around $0.0$ and $1.0$, even on ambiguous edge cases.
  - **Calibration Diagnosis**: Reliability diagram (plotting predicted probability bins against actual fraction of positives) and Expected Calibration Error (ECE).
  - **Correction (Platt Scaling / Temperature Scaling)**:
    - Temperature scaling introduces a single learnable scalar parameter $T > 0$ on the validation set logits:
      $$\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
    - $T > 1$ softens probability distributions without altering the rank ordering of predictions (preserving accuracy and ROC-AUC while fixing calibration).

---

## 3. Coding Drill: Vectorized ROC-AUC Calculation via Rank Sum

### Task
Implement ROC-AUC calculation from scratch using the Wilcoxon-Mann-Whitney rank sum formulation in $\mathcal{O}(N \log N)$ time.

```python
import numpy as np

def fast_roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_t = y_true.ravel().astype(bool)
    y_s = y_scores.ravel()

    n_pos = int(np.sum(y_t))
    n_neg = len(y_t) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples required.")

    # Rank all scores (1-indexed)
    ranked_indices = np.argsort(y_s)
    ranks = np.empty_like(ranked_indices, dtype=float)
    ranks[ranked_indices] = np.arange(1, len(y_s) + 1)

    # Sum ranks of positive instances
    sum_ranks_pos = np.sum(ranks[y_t])

    # Mann-Whitney U statistic: U = R_pos - (n_pos * (n_pos + 1)) / 2
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))
```
