# MLOps Experiment Tracking & Model Registry

Comprehensive guide and implementation of machine learning experiment tracking and model registry governance, covering MLflow, Weights & Biases (W&B) concepts, run provenance, and production model lifecycle management.

---

## 1. System Architecture

```
+-------------------------------------------------------------------------------+
|                       MLOps Experiment & Registry System                      |
|                                                                               |
|  [Data Scientist / CI Worker]                                                 |
|          |                                                                    |
|          | 1. Start Run & Log Hyperparameters                                 |
|          | 2. Stream Metrics per Step (Loss, AUC, F1)                         |
|          | 3. Save Model Weights & Artifacts                                  |
|          v                                                                    |
|  +-------------------------------------+    +------------------------------+  |
|  |       MLflow / W&B Tracking         |    |        Artifact Store        |  |
|  |  (Parameters, Metrics, Metadata)   |    |      (S3, GCS, Blob, OCI)    |  |
|  +------------------+------------------+    +--------------+---------------+  |
|                     |                                      |                  |
|                     | 4. Select Best Candidate             |                  |
|                     v                                      |                  |
|  +-------------------------------------+                   |                  |
|  |            Model Registry           |<------------------+                  |
|  |  - Versioning (v1, v2, v3...)       |                                      |
|  |  - Lifecycle Stages:                |                                      |
|  |    [None] -> [Staging] ->           |                                      |
|  |    [Production] -> [Archived]       |                                      |
|  +------------------+------------------+                                      |
|                     |                                                         |
|                     | 5. Continuous Deployment                                |
|                     v                                                         |
|         [Production Inference Fleet]                                          |
+-------------------------------------------------------------------------------+
```

---

## 2. Core Concepts

### 2.1 Experiment Tracking (MLflow vs Weights & Biases)
- **Parameters**: Static configuration values defined prior to training (learning rate, batch size, model architecture, feature sets).
- **Metrics**: Scalar values updated dynamically over training steps or epochs (training loss, validation accuracy, gradient norms).
- **Artifacts**: Heavy binary files stored in object storage (model weights, ONNX exports, confusion matrix plots, dataset sample slices).
- **Tags & Metadata**: Git commit SHA, docker container image digest, user email, compute environment (GPU type, CUDA version).

### 2.2 Model Registry & Governance
The Model Registry acts as the bridge between model exploration and production deployment:
- **Unique Identification**: Every model has a canonical registered name (e.g., `fraud_detection_xgboost`) and monotonically increasing integer version numbers (`1`, `2`, `...`).
- **Stage Lifecycle**:
  - `None`: Newly registered model undergoing offline validation.
  - `Staging`: Deployed to shadow or canary environments for latency and integration testing.
  - `Production`: Actively serving production traffic. Promoted models automatically trigger deprecation/archival of previous active versions.
  - `Archived`: Inactive historical versions retained for auditability, compliance, and rollback capabilities.

---

## 3. Directory Structure

```
07-mlops/experiment-tracking/
├── README.md
├── notebook.ipynb
├── interview.md
├── references.md
└── code/
    ├── tracker.py
    └── test_tracker.py
```
