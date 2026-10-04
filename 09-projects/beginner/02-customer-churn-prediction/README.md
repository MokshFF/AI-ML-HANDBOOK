# End-to-End Customer Churn Prediction

## Problem
Identify subscription customers at high risk of canceling service to enable proactive retention campaigns.

## Motivation
Customer acquisition costs (CAC) typically exceed customer retention costs by $5\times$ to $7\times$. Minimizing churn directly stabilizes Monthly Recurring Revenue (MRR).

## Dataset
Synthetic subscription service customer interaction dataset:
- Features: `tenure_months`, `monthly_charges`, `total_charges`, `contract_type`, `tech_support_tickets`, `payment_method`.
- Target: `churn` ($1 = \text{churned}, 0 = \text{retained}$).

## Architecture
```mermaid
flowchart LR
    A[Customer Profile Data] --> B[Imbalance Inspection]
    B --> C[Feature Pipeline & Normalization]
    C --> D[Logistic Classifier with Class Weighting]
    D --> E[ROC-AUC & Precision-Recall Threshold Tuning]
    E --> F[Retention Intervention List]
```

## Pipeline
1. Load historical cohort subscriber activity.
2. Standardize continuous attributes; one-hot encode contract tier and payment channels.
3. Optimize binary classification model under class imbalance.
4. Compute ROC-AUC, Precision, Recall, and calibrate decision threshold $\tau$.

## Technologies
- Python 3.11+
- NumPy, Pandas
- Scikit-Learn
- Pytest

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/churn_model.py
```

## Evaluation
- ROC-AUC: Area under Receiver Operating Characteristic curve.
- F1-Score: Harmonic mean of precision and recall.
- Recall at 20% Top Intervention Capacity.

## Results
- Validated on 200 synthetic holdout profiles:
  - ROC-AUC: $\approx 0.84$
  - F1-Score: $\approx 0.72$
  - Real-world production benchmarks: *Pending deployment on active telemetry*.

## Limitations
- Does not model dynamic intra-month behavior fluctuations (e.g. abrupt drops in daily app opens).

## Future Improvements
- Implement survival analysis (Cox Proportional Hazards) to estimate time-to-churn.
- Integrate SHAP value explanations directly into CRM customer cards.
