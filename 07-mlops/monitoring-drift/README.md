# Production Model Monitoring & Drift Detection

Comprehensive guide and implementation of machine learning observability, covering Data Drift (Covariate Shift), Concept Drift, Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI), and Prometheus telemetry.

---

## 1. Model Degradation Taxonomy

```
+-------------------------------------------------------------------------------+
|                            Model Degradation Types                            |
+---------------------------------------+---------------------------------------+
| 1. Data Drift (Covariate Shift)       | 2. Concept Drift                      |
|    Shift in Input Distribution P(X)   |    Shift in Relationship P(Y | X)     |
|    - User demographic changes         |    - Competitor pricing changes       |
|    - Sensor calibration decay         |    - Macroeconomic shifts             |
|    - Schema / preprocessing bugs      |    - Fraud ring behavior evolution    |
+---------------------------------------+---------------------------------------+
| 3. Label Drift (Prior Probability)    | 4. Software & Pipeline Bugs           |
|    Shift in Target Distribution P(Y)  |    - Upstream missing values          |
|    - Disease outbreak increasing      |    - Feature store schema mismatch    |
|      positive diagnosis rates         |    - Timezone / unit conversion errors|
+---------------------------------------+---------------------------------------+
```

---

## 2. Statistical Testing Methodologies

### 2.1 Kolmogorov-Smirnov (KS) Two-Sample Test
- **Hypothesis**: $H_0$: Reference sample $X_{\text{ref}}$ and Production sample $X_{\text{cur}}$ come from the same continuous distribution.
- **Statistic**: Maximum vertical difference between empirical cumulative distribution functions (ECDFs):
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{cur}}(x)|$$
- **Significance**: If $p < 0.05$, reject $H_0$ and conclude that input distribution has drifted.

### 2.2 Population Stability Index (PSI)
Used in risk modeling and financial credit scoring to quantify distribution changes:
$$\text{PSI} = \sum_{i=1}^B \left(P_{\text{cur}, i} - P_{\text{ref}, i}\right) \times \ln\left(\frac{P_{\text{cur}, i}}{P_{\text{ref}, i}}\right)$$
- **Standard Thresholds**:
  - $\text{PSI} < 0.10$: Minor shift; no action required.
  - $0.10 \le \text{PSI} < 0.25$: Moderate shift; trigger automated warning / queue for review.
  - $\text{PSI} \ge 0.25$: Significant shift; trigger automated retraining and rollback verification.

### 2.3 Earth Mover's Distance (Wasserstein-1)
Measures the minimum work (mass $\times$ distance) required to transform one probability distribution into another:
$$W_1(P, Q) = \int_{-\infty}^\infty |F_P(x) - F_Q(x)| \, dx$$
Unlike KL divergence, Wasserstein distance is symmetric, always finite, and provides a continuous metric even for disjoint distributions.

---

## 3. Directory Structure

```
07-mlops/monitoring-drift/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── drift_detector.py
    └── test_drift_detector.py
```
