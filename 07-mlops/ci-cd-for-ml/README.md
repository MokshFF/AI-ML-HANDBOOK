# CI/CD for Machine Learning (Continuous Training & Deployment)

Comprehensive guide and implementation of modern MLOps Continuous Integration and Continuous Deployment (CI/CD) pipelines, covering data validation gates, champion-challenger model qualification, slice testing, and GitHub Actions automation.

---

## 1. MLOps CI/CD Lifecycle & Continuous Training (CT)

```
                                  [Code & Data Push]
                                          |
                                          v
                         +---------------------------------+
                         |         GitHub Actions          |
                         +----------------+----------------+
                                          |
                                          v
                        [Phase 1: Code Linting & Unit Tests]
                        - flake8, ruff, black
                        - pytest unit tests on feature logic
                                          | (Pass)
                                          v
                        [Phase 2: Data Quality & Schema Gate]
                        - Great Expectations / Pandera
                        - Null checks, range checks, type checks
                                          | (Pass)
                                          v
                        [Phase 3: Model Qualification Gate]
                        - Champion vs Challenger comparison
                        - Subgroup / slice regression tests
                        - Latency SLA verification
                                          | (Pass)
                                          v
                        [Phase 4: Build & Container Packaging]
                        - Docker multi-stage build
                        - Push image to registry
                                          |
                                          v
                        [Phase 5: Canary / Shadow Deployment]
                        - Deploy 5% traffic to Kubernetes Pod
                        - Automated rollback if error rate spikes
```

---

## 2. Core Concepts

### 2.1 The Three Levels of MLOps (Google Cloud Architecture)
1. **Level 0 (Manual Process)**: Data science workflows run manually in exploratory notebooks; models deployed as artifacts manually.
2. **Level 1 (ML Pipeline Automation / CT)**: Pipeline stages (ingestion, validation, training, evaluation) run as automated workflows triggered by data drift or new labels.
3. **Level 2 (Full CI/CD Automation)**: Source code changes in Git trigger automated testing, end-to-end pipeline execution, model qualification gates, and zero-downtime serving updates.

### 2.2 Model Qualification & Slice Testing
A model with higher average accuracy can still introduce critical regressions:
- **Simpson's Paradox**: An aggregate gain across the entire population can hide a sharp loss on an underrepresented demographic slice or critical platform (e.g., iOS vs. Android).
- **Slice Gates**: Mandate that performance on designated critical slices cannot drop by more than a predefined delta (e.g. $\le 2\%$) regardless of overall performance gains.

---

## 3. Directory Structure

```
07-mlops/ci-cd-for-ml/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── ml_pipeline_ci.py
    └── test_ml_pipeline_ci.py
```
