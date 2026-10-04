# MLOps CI/CD Interview Questions & Answers

### Q1: How does CI/CD for Machine Learning differ from traditional Software CI/CD?
**Answer:**
Traditional CI/CD tests only **Code**. In Machine Learning, production outcomes depend on three independent axes:
1. **Code**: Model definition, data transformation, inference serving scripts.
2. **Data**: Evolving distributions, missing values, schema shifts, and volume changes.
3. **Model Artifact**: Non-deterministic weights, non-linear performance dynamics, and latency profiles.
**Key Differences in ML CI/CD**:
- Continuous Integration must validate both code and incoming data quality (schema checks, null checks).
- Testing is statistical rather than purely boolean (evaluating ROC-AUC, F1, slice parity).
- Requires **Continuous Training (CT)**: Automated retraining triggers activated by data drift or schedule, not just code commits.

---

### Q2: What is "Shadow Deployment", and how does it compare to "Canary Deployment"?
**Answer:**
- **Shadow Deployment**:
  - The incoming live production request is duplicated: one copy goes to the active production model (Champion) and one copy goes asynchronously to the new model (Challenger).
  - Only the Champion's prediction is returned to the user.
  - The Challenger's predictions and latency are logged for silent evaluation against real production traffic with zero risk of user-facing failure.
- **Canary Deployment**:
  - A small fraction of real users (e.g., 5%) are routed directly to the new model and receive its predictions.
  - Performance, business metrics, and error rates are monitored in real time.
  - If metrics remain healthy, traffic is incrementally shifted (5% -> 25% -> 100%); otherwise, traffic is instantly rolled back.

---

### Q3: Why is Slice Testing critical before promoting a model to production?
**Answer:**
Aggregate metrics like overall accuracy or RMSE can be deceptive:
1. **Underrepresented Classes & Demographics**: In a dataset where 95% of users are on desktop and 5% on mobile, a model that drastically degrades on mobile can still show an aggregate accuracy increase.
2. **High-Value Subgroups**: In e-commerce or B2B SaaS, enterprise tier customers or high-volume power users represent critical revenue despite being small in number.
3. **Fairness & Bias**: Automated slice testing enforces that model false positive/negative rates remain balanced across protected demographic attributes.
