# Cheat Sheet: Machine Learning Evaluation Metrics

Mathematical definitions, optimal use cases, and gotchas for ML metrics.

| Metric | Formula | Best For | Vulnerability / Pitfall |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{Total}}$ | Perfectly balanced classes | Deceptive on imbalanced classes ($99\%$ negative) |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | High cost of false alarms (Spam, loan approval) | Ignores false negatives completely |
| **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | High cost of missing positives (Cancer, fraud) | Maximized trivially by predicting positive always |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Balance of precision and recall | Treats FP and FN costs as symmetric |
| **ROC-AUC** | $\int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$ | Global rank-ordering capability | Overly optimistic under extreme class imbalance |
| **PR-AUC** | $\int_0^1 \text{Precision}(r) \, dr$ | Severe class imbalance ($< 1\%$ positive) | Baseline is positive class prevalence, not 0.5 |
| **Log Loss (Cross-Entropy)** | $-\frac{1}{N} \sum [y_i \log p_i + (1-y_i) \log(1-p_i)]$ | Calibrated probability estimation | Heavily penalizes confident wrong predictions |
| **MAE** | $\frac{1}{N} \sum \|y_i - \hat{y}_i\|$ | Robust regression with outliers | Non-differentiable at 0 |
| **RMSE** | $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$ | Standard regression penalizing large errors | Sensitive to extreme outliers |
| **MAPE** | $\frac{100\%}{N} \sum \left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Business/forecasting percentage error | Undefined when $y_i = 0$; asymmetric bias |
| **NDCG@K** | $\frac{\text{DCG}@K}{\text{IDCG}@K} = \frac{\sum \frac{2^{rel_i}-1}{\log_2(i+1)}}{\text{Ideal DCG}}$ | Search and recommendation ranking quality | Sensitive to relevance grade scale definitions |
| **mAP@IoU** | Mean AP across categories at specified IoU threshold | Object detection localization & classification | Sensitive to NMS suppression threshold |
