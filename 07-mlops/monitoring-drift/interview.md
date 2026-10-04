# Model Monitoring & Drift Detection Interview Questions & Answers

### Q1: What is the fundamental difference between Data Drift and Concept Drift?
**Answer:**
Let $X$ denote features and $Y$ denote labels:
- **Data Drift (Covariate Shift)**:
  - $P(X)$ changes while $P(Y | X)$ remains constant.
  - *Example*: An autonomous vehicle camera enters heavy rain. The input pixels $P(X)$ change radically due to raindrops, but the physical definition of a stop sign $P(Y | X)$ is unchanged.
  - *Detection*: Immediate; can be computed on unlabeled inference traffic using KS-tests or PSI.
- **Concept Drift**:
  - $P(Y | X)$ changes (the underlying relationship between features and target changes).
  - *Example*: During high inflation, a borrower with a 720 credit score and $50k income becomes significantly more likely to default than historically. The input profile $X$ looks identical, but the outcome probability $Y$ has changed.
  - *Detection*: Delayed; requires waiting for ground truth labels (e.g. 30-90 days for loan default outcomes).

---

### Q2: Why is the Kolmogorov-Smirnov test well-suited for numerical features, and what are its limitations?
**Answer:**
- **Strengths**:
  - Non-parametric: Makes zero assumptions about underlying distribution (Gaussian, Exponential, Bimodal, etc.).
  - Scale-invariant: Invariant to monotonic transformations of the coordinates.
  - Sensitive to shifts in shape, spread, and median simultaneously.
- **Limitations**:
  - Extreme sample size sensitivity: With millions of production requests ($N > 100,000$), trivial statistical noise yields $p < 0.0001$.
  - Continuous data only: Not valid for categorical or discrete features (Chi-square test or Jensen-Shannon divergence should be used instead).
  - Practical solution: Combine KS p-values with Population Stability Index (PSI) or Wasserstein distance to measure effect size.

---

### Q3: How do you design an alert triaging strategy to prevent "alert fatigue" in MLOps monitoring?
**Answer:**
1. **Tiered Severity Levels**:
   - *P3 (Info / Warning)*: $0.10 \le \text{PSI} < 0.25$ on non-critical features. Logged to dashboard; no pager alert.
   - *P1 (Critical / Action)*: $\text{PSI} \ge 0.25$ on top 3 most important feature columns OR rolling prediction accuracy drop $> 10\%$. Pagers on-call ML engineer.
2. **Feature Attribution Weighting**: Do not alert on drift in low-importance features (e.g. features with $< 1\%$ Shapley/gain importance).
3. **Sliding Window Persistence**: Require drift to persist across multiple consecutive evaluation batches (e.g. 3 consecutive hours) to avoid alerting on temporary diurnal traffic spikes.
4. **Automated Remediation**: Trigger an automated shadow retraining pipeline; if the retrained candidate passes offline validation, present it for one-click deployment.
