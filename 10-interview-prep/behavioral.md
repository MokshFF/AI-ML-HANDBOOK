# Machine Learning Behavioral & Leadership Questions

Situational frameworks, engineering trade-offs, stakeholder communication, and post-mortem incident leadership using the STAR (Situation, Task, Action, Result) methodology.

---

## 1. Navigating Modeling Trade-Offs (Accuracy vs Latency)

### Question: "Describe a situation where a technically superior model could not be deployed to production. How did you resolve the conflict?"
- **STAR Response Framework**:
  - **Situation**: Our research team developed a high-performing ensemble (3-way blend of DeBERTa-v3, RoBERTa, and CatBoost) that improved customer support ticket categorization macro-F1 from $0.82$ to $0.89$.
  - **Task**: The production SLA required end-to-end inference in $< 40\text{ ms}$ at 1,200 QPS. The ensemble took $165\text{ ms}$ and required 12 additional GPU nodes, blowing our quarterly cloud budget by $\$140,000$.
  - **Action**: Rather than insisting on the ensemble, I proposed a Knowledge Distillation strategy. We used the heavy ensemble as an offline teacher model to label 2,000,000 unlabeled historical tickets, and trained a compact 4-layer DistilBERT student model. We then quantized the student to INT8 using ONNX Runtime.
  - **Result**: The distilled model achieved $0.875$ F1 (retaining $80\%$ of the ensemble's gain) with an inference latency of $12\text{ ms}$ on standard CPU instances, running at $10\%$ of the GPU cost budget and meeting all SLA requirements.

---

## 2. Production Incidents & Post-Mortem Leadership

### Question: "Tell me about a time an ML model failed or degraded silently in production. How did you diagnose and remediate it?"
- **STAR Response Framework**:
  - **Situation**: Following a major e-commerce marketing campaign, conversion rates dropped by $14\%$ over 48 hours without any HTTP 500 server errors or pipeline crashes.
  - **Task**: Identify the root cause of the silent revenue drop and establish preventive safeguards.
  - **Action**: I inspected the input feature telemetry and discovered that a frontend refactor had renamed the checkout event payload key from `item_sku` to `product_id`. The data pipeline defaulted the missing feature to `-1`. Our GBDT recommendation model interpreted all items as unknown cold-start products, serving generic non-personalized recommendations.
    I immediately rolled back the client feature mapping, added automated Great Expectations schema validation at ingestion to reject unexpected null/default rates, and deployed automated Population Stability Index (PSI) drift alerts to Slack.
  - **Result**: Revenue recovered immediately upon rollback, and the new data contract test prevented 3 similar silent schema-breaking regressions over the subsequent year.
